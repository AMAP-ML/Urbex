import time
import ray
import torch
from tqdm import tqdm
import os
import socket
import subprocess
import time
import signal
import atexit
import requests
from codetiming import Timer
import torch.distributed as dist
from roll.utils.logging import get_logger
from roll.utils.checkpoint_manager import download_model
from roll.configs.worker_config import WorkerConfig
from roll.platforms import current_platform
from collections import defaultdict
from typing import Any

logger = get_logger()


class RtpScheduler:
    
    actor_infer: WorkerConfig
    
    @classmethod
    def start_rtp_server(cls, worker_config: WorkerConfig):
        cls.actor_infer = worker_config
        model_path = str(download_model(cls.actor_infer.model_args.model_name_or_path))
        tokenizer_path = model_path
        devices = cls.actor_infer.device_mapping
        world_size = len(devices)
        
        def _get(key: str, default: Any) -> str:
            val = cls.actor_infer.strategy_args.strategy_config.get(key, default)
            if val is None:
                raise ValueError(f"{key} must be set in strategy_config of rtp")
            return str(val)

        def _prepare_env():
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices)),
                    "MODEL_TYPE": _get("model_type", None),
                    "CHECKPOINT_PATH": model_path,
                    "TOKENIZER_PATH": tokenizer_path,
                    "CONCURRENCY_LIMIT": _get("concurrency_limit", 1024),
                    "MAX_CONTEXT_BATCH_SIZE": _get("max_context_batch_size", 6),
                    "NOT_USE_DEFAULT_STREAM": _get("not_use_default_stream", True),
                    "TP_SIZE": _get("tp_size", 1),
                    "DP_SIZE": _get("dp_size", 1),
                    "WORLD_SIZE": _get("world_size", world_size),
                    "EP_SIZE": _get("ep_size", 1),
                    "START_PORT": _get("start_port", None),
                    "USE_RPC_MODEL": _get("use_rpc_model", 1),
                    "MAX_SEQ_LEN": _get("max_seq_len", 4096),
                    "ENABLE_FMHA": _get("enable_fmha", "on"),
                    "WARM_UP": _get("warm_up", 0),
                    "LOG_LEVEL": _get("log_level", "INFO"),
                    "NCCL_DISABLE_ABORT": _get("nccl_disable_abort", 1),
                    "FT_DISABLE_CUSTOM_AR": _get("ft_disable_custom_ar", 1),
                    "CUDA_LAUNCH_BLOCKING": _get("cuda_launch_blocking", 1),
                    "DEVICE_RESERVE_MEMORY_BYTES": _get("device_reserve_memory_bytes", -40960000000),
                    "RESERVER_RUNTIME_MEM_MB": _get("reserver_runtime_mem_mb", 4096),
                    "ENABLE_COMM_OVERLAP": _get("enable_comm_overlap", 0),
                }
            )
            return env

        def _wait_for_port(host="localhost", port=26000, timeout=60):
            start = time.time()
            while True:
                try:
                    with socket.create_connection((host, port), timeout=1):
                        return
                except (OSError, socket.timeout):
                    if time.time() - start > timeout:
                        raise RuntimeError(f"RTP server did not listen on {port} within {timeout}s")
                    time.sleep(1)

        env = _prepare_env()
        cmd = ["/opt/conda310/bin/python3", "-m", "rtp_llm.start_server"]
        _rtp_process = subprocess.Popen(cmd, env=env, cwd="/root/project/RTP-LLM/",stdout=open('/tmp/rtp.log','ab'), stderr=open('/tmp/rtp.log','ab'))
        
        def _kill_rtp():
            nonlocal _rtp_process
            if _rtp_process is None:
                return
            try:
                if _rtp_process.poll() is None:
                    _rtp_process.kill()
                    _rtp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _rtp_process.kill()
                _rtp_process.wait()
            _rtp_process = None

        def _signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, will shut RTP down when process exits")
            signal.default_int_handler(signum, frame)
        
        atexit.register(_kill_rtp)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _signal_handler)
            
        rest_port, update_weight_port = int(env["START_PORT"]), int(env["START_PORT"])+6
        cls.actor_infer.strategy_args.strategy_config["rtp_server_rpc_url"] = f"127.0.0.1:{rest_port+1}"
        rtp_weight_memory_rest_url = cls.actor_infer.strategy_args.strategy_config["rtp_weight_memory_rest_url"] = f"http://127.0.0.1:{rest_port+6}"

        _wait_for_port(port=rest_port, timeout=300)
        _wait_for_port(port=update_weight_port, timeout=300)
        logger.info("--- Sending a request to the server to clear the KV cache ---")
        while True:
            try:
                url = f"{rtp_weight_memory_rest_url}/pause"
                response = requests.post(url)
                response.raise_for_status()
                logger.info(f"Response content of the pause request:{response.text}")
                url = f"{rtp_weight_memory_rest_url}/detach_physical_memory"
                clear_response = requests.post(url)
                logger.info(f"Response status code for clearing the KV cache:{clear_response.status_code}")
                logger.info(f"Clear the response content of the KV cache:{clear_response.json()}")
                break
            except requests.exceptions.RequestException as e:
                logger.info(f"Pause request error: {e}")
            except requests.exceptions.JSONDecodeError:
                logger.info(f"The response content of clearing the KV cache is not in valid JSON format:{clear_response.text}")
            except Exception as e:
                logger.info(e)
    
    @classmethod
    def pipeline_model_update(cls, actor_train, actor_infer):
        actor_infer.load_states(blocking=True)
        refs = [
            w.start_model_update.remote(
                model_update_url=cls.actor_infer.strategy_args.strategy_config["rtp_weight_memory_rest_url"]
            )
            for w in actor_train.workers
        ]
        data = ray.get(refs)
        actor_infer.offload_states(blocking=True)
        return data
    
    @staticmethod
    def train_model_update(url, model, models_unwrapped, client):
        broadcast_time_cost = 0
        kv_cache = defaultdict(dict)

        with Timer("model_update_total") as timer_total:
            update_url = f"{url}/update_weight"
            with Timer("broadcast") as timer_broadcast:
                for key, value in tqdm(
                        model.all_gather_weights_as_hf_inflight(models=models_unwrapped),
                        desc="Updating Weights"
                    ):
                    if client is not None:
                        if "k_proj" in key or "q_proj" in key:
                            kv_cache[key] = value
                            continue
                        elif "v_proj" in key:
                            q_key = key.replace("v_proj", "q_proj")
                            k_key = key.replace("v_proj", "k_proj")
                            value = torch.cat([
                                kv_cache[q_key],
                                kv_cache[k_key],
                                value,
                            ], dim=0)
                            if len(value.shape) >= 2:
                                value = torch.transpose(value, 0, 1).contiguous()
                            key = key.replace("v_proj", "qkv_proj")
                            kv_cache.clear() 
                        if "mlp" in key and "weight" in key and len(value.shape) >= 2:
                            value = torch.transpose(value, 0, 1).contiguous()
                        if "o_proj" in key and len(value.shape) >= 2:
                            value = torch.transpose(value, 0, 1).contiguous()
                        logger.info(f"weight key: {key}, dtype: {value.dtype}, shape: {tuple(value.shape)}")
                        try:
                            client.send(update_url, name=key, t=value.half().cpu(), method="shm")
                        except Exception as e:
                            logger.warning(f"Ignored weight update failure for key '{key}': {e}")
                        del value
                        current_platform.empty_cache()
                dist.barrier()
            broadcast_time_cost += timer_broadcast.last
        metrics = {
            "all_gather": timer_total.last - broadcast_time_cost,
            "broadcast": broadcast_time_cost,
        }
        return metrics