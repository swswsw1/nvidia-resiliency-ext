"""Post-hoc analyzer for trigger-driven FR dumps.

The trainer's cheap-stats trigger writes one file per (rank, trigger-step):
    `_dump_{rank}_step{step:06d}.json`
Each file is ALREADY block-sliced — its `entries` list is exactly one
default_pg-bracketed block on that rank. So this analyzer is just
"glob, group by step, hand to analyze_quiet" — no merging, no slicing.

Usage:
    python fr_analyze_dumps.py /path/to/trace_dir
    python fr_analyze_dumps.py /path/to/trace_dir --k 2.0 --floor-ms 10.0

Output:
    `verdicts.jsonl` in the trace dir, one line per trigger-step:
        {"step": N, "ranks": [...], "result": {...}}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from typing import Dict

# Import the analyzer in-process.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fr_straggler_analyzer as analyzer  # noqa: E402


DUMP_RE = re.compile(r"^_dump_(\d+)_step(\d{6})\.json$")
STAGING_SUBDIR = ".dumps_staging"


def discover_dumps(trace_dir: str) -> Dict[int, Dict[int, str]]:
    """Return {step: {rank: path}} for every trigger-driven dump file."""
    out: Dict[int, Dict[int, str]] = defaultdict(dict)
    for name in os.listdir(trace_dir):
        m = DUMP_RE.match(name)
        if not m:
            continue
        rank = int(m.group(1))
        step = int(m.group(2))
        out[step][rank] = os.path.join(trace_dir, name)
    return out


def stage_step(trace_dir: str, step: int, rank_files: Dict[int, str], staging_root: str) -> str:
    """Materialize per-rank files for one step into a flat directory the
    analyzer can read with its `_dump_*.json` glob."""
    step_dir = os.path.join(staging_root, f"step_{step:06d}")
    os.makedirs(step_dir, exist_ok=True)
    for rank, src in rank_files.items():
        dst = os.path.join(step_dir, f"_dump_{rank}.json")
        if os.path.exists(dst):
            continue
        # Copy (not symlink) so analyzer's reads are immune to stale paths.
        shutil.copy2(src, dst)
    src_cfg = os.path.join(trace_dir, "run_config.log")
    if os.path.exists(src_cfg):
        cfg_dst = os.path.join(step_dir, "run_config.log")
        if not os.path.exists(cfg_dst):
            shutil.copy2(src_cfg, cfg_dst)
    return step_dir


def main():
    parser = argparse.ArgumentParser(description="Post-hoc analyzer for trigger-driven FR dumps")
    parser.add_argument("trace_dir", help="Directory the trainer wrote dumps to")
    parser.add_argument("--k", type=float, default=2.0,
                        help="Detector k (stdev/MAD multiplier)")
    parser.add_argument("--floor-ms", type=float, default=10.0,
                        help="Detector floor (ms)")
    args = parser.parse_args()

    dumps = discover_dumps(args.trace_dir)
    if not dumps:
        print(f"[fr_analyze_dumps] no _dump_*_step*.json files in {args.trace_dir}")
        return 0

    staging_root = os.path.join(args.trace_dir, STAGING_SUBDIR)
    os.makedirs(staging_root, exist_ok=True)
    verdicts_path = os.path.join(args.trace_dir, "verdicts.jsonl")
    open(verdicts_path, "w").close()  # truncate per run

    print(f"[fr_analyze_dumps] {len(dumps)} trigger step(s) in {args.trace_dir}")
    print(f"[fr_analyze_dumps] verdicts -> {verdicts_path}")

    for step in sorted(dumps):
        rank_files = dumps[step]
        step_dir = stage_step(args.trace_dir, step, rank_files, staging_root)
        result = analyzer.analyze_quiet(step_dir, k=args.k, floor_ms=args.floor_ms)
        # analyze_quiet returns a set; not JSON-able. Sort for stable output.
        result["head_straggler_ranks"] = sorted(result.get("head_straggler_ranks") or [])

        verdict = {
            "step": step,
            "ranks": sorted(rank_files.keys()),
            "result": result,
        }
        with open(verdicts_path, "a") as f:
            f.write(json.dumps(verdict) + "\n")
            f.flush()
            os.fsync(f.fileno())

        print(
            f"  step={step:>6}  ranks={sorted(rank_files)}  "
            f"windows={result.get('n_windows')}  origins={result.get('n_origins')}  "
            f"head_ranks={result['head_straggler_ranks']}  hit={result.get('hit')}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
