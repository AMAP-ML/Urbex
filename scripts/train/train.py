"""Urbex GRPO training entrypoint.

Thin wrapper around ROLL's AgenticPipeline: compose the Hydra config, bring up
Ray, run. Invoked by scripts/train/run_train.sh.
"""
import argparse

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
from roll.utils.str_utils import print_pipeline_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="../configs", help="Config dir, relative to this file")
    parser.add_argument("--config_name", default="grpo", help="Config file name without extension")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. max_steps=100")
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="urbex_train", version_base=None)
    cfg = compose(config_name=args.config_name, overrides=args.overrides)
    config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()
    print_pipeline_config(config)
    AgenticPipeline(pipeline_config=config).run()


if __name__ == "__main__":
    main()
