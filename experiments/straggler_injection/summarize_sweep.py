"""Summarize online-monitor sweep results into a Rank x Delay table per type.

Reads the final verdict from each trace dir's `verdicts.jsonl` along with
ground truth from `run_config.log`, then prints a table per injection type
matching the format of the prior offline sweep.

A cell is:
  Full Match = heads == {injected_rank}
  Partial    = injected_rank in heads but heads != {injected_rank}
  Miss       = injected_rank not in heads

Usage:
  python3 summarize_sweep.py <traces_dir> [--since 20260422_140000]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


TIMESTAMP_RE = re.compile(r"^(\d{8}_\d{6})_(none|host|kernel)$")


def parse_run_config(path: str) -> Optional[dict]:
    cfg = {}
    try:
        with open(path) as f:
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                cfg[k.strip()] = v.strip()
    except FileNotFoundError:
        return None
    return cfg


def last_final_verdict(path: str) -> Optional[dict]:
    """Return the last `final: true` verdict from a verdicts.jsonl (or
    just the last verdict if no final was emitted)."""
    try:
        with open(path) as f:
            lines = [line for line in f if line.strip()]
    except FileNotFoundError:
        return None
    if not lines:
        return None
    final = None
    last = None
    for line in lines:
        try:
            v = json.loads(line)
        except json.JSONDecodeError:
            continue
        last = v
        if v.get("final"):
            final = v
    return final or last


def classify(heads: List[str], inject_rank: Optional[int]) -> str:
    if inject_rank is None:
        return "-"
    inject_s = str(inject_rank)
    if not heads:
        return "Miss"
    heads_set = set(heads)
    if heads_set == {inject_s}:
        return "Full"
    if inject_s in heads_set:
        return "Partial"
    return "Miss"


def collect(traces_root: str, since: Optional[str]) -> List[dict]:
    rows = []
    for name in sorted(os.listdir(traces_root)):
        m = TIMESTAMP_RE.match(name)
        if not m:
            continue
        ts, inject_type = m.group(1), m.group(2)
        if since and ts < since:
            continue
        if inject_type == "none":
            continue  # baselines not in the rank x delay grid
        trace_dir = os.path.join(traces_root, name)
        cfg = parse_run_config(os.path.join(trace_dir, "run_config.log"))
        if not cfg:
            continue
        verdict = last_final_verdict(os.path.join(trace_dir, "verdicts.jsonl"))
        if not verdict:
            rows.append({
                "dir": name, "type": inject_type,
                "rank": int(cfg.get("inject_rank", -1)),
                "delay": float(cfg.get("inject_delay_ms", 0)),
                "heads": None, "hit": None, "status": "NO_VERDICT",
            })
            continue
        result = verdict.get("result", {})
        heads = result.get("head_straggler_ranks", []) or []
        rows.append({
            "dir": name,
            "type": inject_type,
            "rank": int(cfg.get("inject_rank", -1)),
            "delay": float(cfg.get("inject_delay_ms", 0)),
            "heads": heads,
            "hit": bool(result.get("hit")),
            "status": classify(heads, int(cfg.get("inject_rank", -1))),
        })
    return rows


def build_table(rows: List[dict], inject_type: str) -> Tuple[List[float], Dict[int, Dict[float, str]]]:
    subset = [r for r in rows if r["type"] == inject_type]
    delays = sorted({r["delay"] for r in subset})
    ranks = sorted({r["rank"] for r in subset})
    table: Dict[int, Dict[float, str]] = defaultdict(dict)
    for r in subset:
        # If multiple runs exist for the same cell, keep the most recent.
        table[r["rank"]][r["delay"]] = r["status"]
    return delays, dict(sorted(table.items()))


def print_table(title: str, delays: List[float], table: Dict[int, Dict[float, str]]):
    if not table:
        print(f"\n=== {title} ===  (no runs)\n")
        return
    print(f"\n=== {title} ===")
    header = "Rank \\ Delay(ms)".ljust(18) + " ".join(f"{d:>7.0f}" for d in delays)
    print(header)
    print("-" * len(header))
    for rank, by_delay in table.items():
        cells = [f"{by_delay.get(d, '-'):>7}" for d in delays]
        print(f"Rank {rank:<12}" + " ".join(cells))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traces_dir", help="Path to the traces/ directory containing per-run subdirs")
    p.add_argument("--since", help="Only include runs with timestamp >= this (YYYYMMDD_HHMMSS)")
    args = p.parse_args()

    rows = collect(args.traces_dir, args.since)
    if not rows:
        print("No injection runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Total injection runs: {len(rows)}")
    no_verdict = [r for r in rows if r["status"] == "NO_VERDICT"]
    if no_verdict:
        print(f"  {len(no_verdict)} with no verdict (monitor didn't run or crashed):")
        for r in no_verdict[:5]:
            print(f"    - {r['dir']}")

    for t in ("host", "kernel"):
        delays, table = build_table(rows, t)
        print_table(f"{t.title()} Injection", delays, table)

    # Overall summary
    total = len(rows) - len(no_verdict)
    full = sum(1 for r in rows if r["status"] == "Full")
    partial = sum(1 for r in rows if r["status"] == "Partial")
    miss = sum(1 for r in rows if r["status"] == "Miss")
    print(f"\nSummary: Full={full} Partial={partial} Miss={miss} (of {total} with verdicts)")


if __name__ == "__main__":
    main()
