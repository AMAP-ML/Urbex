#!/usr/bin/env python3
"""Aggregate per-run eval logs into per-dataset mean / std / min / max.

Inputs (one sweep_root, multiple datasets, multiple runs per dataset):
    ${SWEEP_ROOT}/<dataset>/eval/hit_k_run_<NN>.log
    ${SWEEP_ROOT}/<dataset>/eval/3d_metrics_run_<NN>.log

Outputs:
    ${OUTPUT_DIR}/summary.json   machine-readable, full per-run + aggregate
    ${OUTPUT_DIR}/summary.md     Markdown tables
    ${OUTPUT_DIR}/summary.txt    same content printed to stdout

Usage:
    python scripts/eval/aggregate.py \\
        --sweep_root ./output/eval_20260101_120000 \\
        --output    ./output/eval_20260101_120000/summary \\
        --num_runs 5 \\
        --datasets cityrefer_ND cityrefer_NO stpls3d
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List


# --- regex patterns ---------------------------------------------------------

# eval_3d_metrics.py prints:
#     IoU_BEV (XY):           mean=0.1234  median=0.1234  max=0.5678
#     Acc@BEV-IoU>=0.25:      502/502 = 100.00%
#     Acc@dist_2d<=5m:        X/Y = Z%
_RE_3D_STAT = re.compile(
    r"^\s+(?P<name>[\w()_\-\s]+?):\s+mean=(?P<mean>[\d.eE+\-]+)\s+"
    r"median=(?P<median>[\d.eE+\-]+)\s+max=(?P<mx>[\d.eE+\-]+)",
    re.MULTILINE,
)
_RE_3D_ACC = re.compile(
    r"^\s+Acc@(?P<cond>[\w>=<\.\-+]+):\s+\d+/\d+\s+=\s+(?P<pct>[\d.]+)%",
    re.MULTILINE,
)
_RE_3D_EVALUATED = re.compile(r"evaluated:\s+(?P<n>\d+)")
_RE_3D_NO_BBOX = re.compile(r"no world_bbox \(skip\):\s+(?P<n>\d+)")

# eval_hit_k.py prints:
#     Hit@1: 100/502 = 19.92%
#     Evaluated:  502 samples (denominator includes no-answer attempts)
_RE_HIT_K = re.compile(r"Hit@(?P<k>\d+):\s+\d+/\d+\s+=\s+(?P<pct>[\d.]+)%")
_RE_HIT_EVALUATED = re.compile(r"Evaluated:\s+(?P<n>\d+) samples")
_RE_HIT_NO_ANSWER = re.compile(r"no_answer:\s+(?P<n>\d+)")


# --- parsers ----------------------------------------------------------------


def parse_eval_3d_metrics(log_path: Path) -> Dict[str, float]:
    """Pull metric values from a single eval_3d_metrics.py stdout log."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")

    out: Dict[str, float] = {}

    for m in _RE_3D_STAT.finditer(text):
        out[m.group("name").strip()] = float(m.group("mean"))

    for m in _RE_3D_ACC.finditer(text):
        out[f"Acc@{m.group('cond')}"] = float(m.group("pct"))

    m = _RE_3D_EVALUATED.search(text)
    if m:
        out["evaluated"] = float(m.group("n"))
    m = _RE_3D_NO_BBOX.search(text)
    if m:
        out["no_world_bbox"] = float(m.group("n"))

    return out


def parse_eval_hit_k(log_path: Path) -> Dict[str, float]:
    """Pull Hit@k values + counts from a single eval_hit_k.py stdout log."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")

    out: Dict[str, float] = {}
    for m in _RE_HIT_K.finditer(text):
        out[f"Hit@{m.group('k')}"] = float(m.group("pct"))

    m = _RE_HIT_EVALUATED.search(text)
    if m:
        out["evaluated"] = float(m.group("n"))
    m = _RE_HIT_NO_ANSWER.search(text)
    if m:
        out["no_answer"] = float(m.group("n"))

    return out


# --- aggregation ------------------------------------------------------------


def aggregate(per_run: List[Dict[str, float]]) -> Dict[str, Dict[str, object]]:
    """Compute mean / std / min / max / n over a list of per-run metric dicts.

    A metric is included in the output only if it appears in ≥1 run.
    std uses sample standard deviation (ddof=1); falls back to 0.0 for n=1.
    """
    if not per_run:
        return {}

    all_keys: set = set()
    for d in per_run:
        all_keys.update(d.keys())

    out: Dict[str, Dict[str, object]] = {}
    for key in sorted(all_keys):
        vals = [d[key] for d in per_run if key in d]
        if not vals:
            continue
        out[key] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
            "values": vals,
        }
    return out


# --- main -------------------------------------------------------------------


def build_markdown(summary: dict) -> str:
    lines: List[str] = ["# Sweep summary", ""]
    lines.append(f"- sweep root : `{summary['sweep_root']}`")
    lines.append(f"- num runs   : {summary['num_runs']}")
    lines.append("")

    for ds, info in summary["datasets"].items():
        if "error" in info:
            lines.append(f"## {ds}  —  {info['error']}")
            lines.append("")
            continue

        lines.append(f"## {ds}")
        lines.append("")
        lines.append(
            f"- rollouts succeeded : {info['hit_k']['n_runs']} / {summary['num_runs']}"
        )

        for section_name, section_key, fmt_pct in (
            ("eval_hit_k", "hit_k", "pct"),
            ("eval_3d_metrics", "3d_metrics", "value"),
        ):
            agg = info[section_key]["aggregate"]
            if not agg:
                lines.append("")
                lines.append(f"### {section_name}")
                lines.append("_no valid runs_")
                continue

            lines.append("")
            lines.append(f"### {section_name}")
            lines.append("")
            lines.append("| metric | mean | std | min | max | n | per-run |")
            lines.append("|---|---:|---:|---:|---:|---:|---|")
            for k, v in agg.items():
                vals = v["values"]
                if fmt_pct == "pct":
                    vals_str = ", ".join(f"{x:.2f}" for x in vals)
                else:
                    vals_str = ", ".join(f"{x:.4f}" for x in vals)
                if k == "evaluated" or k == "no_answer" or k == "no_world_bbox":
                    lines.append(
                        f"| {k} | {v['mean']:.1f} | {v['std']:.2f} | "
                        f"{v['min']:.0f} | {v['max']:.0f} | {v['n']} | {vals_str} |"
                    )
                elif fmt_pct == "pct":
                    lines.append(
                        f"| {k} | {v['mean']:.2f}% | {v['std']:.2f} | "
                        f"{v['min']:.2f} | {v['max']:.2f} | {v['n']} | {vals_str} |"
                    )
                else:
                    lines.append(
                        f"| {k} | {v['mean']:.4f} | {v['std']:.4f} | "
                        f"{v['min']:.4f} | {v['max']:.4f} | {v['n']} | {vals_str} |"
                    )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--sweep_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cityrefer_ND", "cityrefer_NO", "stpls3d"],
    )
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "sweep_root": str(sweep_root),
        "num_runs": args.num_runs,
        "datasets": {},
    }

    for ds in args.datasets:
        eval_dir = sweep_root / ds / "eval"
        if not eval_dir.exists():
            summary["datasets"][ds] = {
                "error": f"no eval dir at {eval_dir} (rollouts may have all failed)"
            }
            continue

        hit_k_per_run: List[Dict[str, float]] = []
        d3_per_run: List[Dict[str, float]] = []
        per_run_meta: List[Dict[str, object]] = []

        for run_idx in range(1, args.num_runs + 1):
            run_str = f"{run_idx:02d}"
            hk_log = eval_dir / f"hit_k_run_{run_str}.log"
            d3_log = eval_dir / f"3d_metrics_run_{run_str}.log"

            hk = parse_eval_hit_k(hk_log)
            d3 = parse_eval_3d_metrics(d3_log)
            hit_k_per_run.append(hk)
            d3_per_run.append(d3)
            per_run_meta.append(
                {
                    "run": run_idx,
                    "hit_k_log": str(hk_log),
                    "hit_k_present": bool(hk),
                    "3d_metrics_log": str(d3_log),
                    "3d_metrics_present": bool(d3),
                }
            )

        valid_hk = [m for m in hit_k_per_run if m]
        valid_d3 = [m for m in d3_per_run if m]

        summary["datasets"][ds] = {
            "per_run": per_run_meta,
            "hit_k": {
                "per_run_raw": hit_k_per_run,
                "aggregate": aggregate(valid_hk),
                "n_runs": len(valid_hk),
            },
            "3d_metrics": {
                "per_run_raw": d3_per_run,
                "aggregate": aggregate(valid_d3),
                "n_runs": len(valid_d3),
            },
        }

    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    txt_path = output_dir / "summary.txt"

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    md_text = build_markdown(summary)
    md_path.write_text(md_text, encoding="utf-8")
    txt_path.write_text(md_text, encoding="utf-8")

    print(md_text)
    print(f"[aggregate] json : {json_path}")
    print(f"[aggregate] md   : {md_path}")
    print(f"[aggregate] txt  : {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())