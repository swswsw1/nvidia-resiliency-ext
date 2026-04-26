"""Online straggler monitor daemon.

Watches a trace directory while training is live. As each rank writes
periodic FR dump chunks (`_dump_{rank}_{chunk:04d}.json`), the daemon
merges them per rank into a staging view, invokes `analyze_quiet` from
fr_straggler_analyzer, and appends structured verdicts to
`verdicts.jsonl` inside the trace dir.

Exits on quiescence: if no new chunks appear for `--quiescence-s`,
assumes training is done, emits one final verdict, and exits.

Launched out-of-process so training is never blocked on analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Dict, Set, Tuple

# Import the existing analyzer in-process — lower latency than subprocessing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fr_straggler_analyzer as analyzer  # noqa: E402


CHUNK_RE = re.compile(r"^_dump_(\d+)_iter(\d{4})\.json$")
STAGING_SUBDIR = ".monitor_view"


def discover_chunks(trace_dir: str) -> Dict[int, Dict[int, str]]:
    """Return {rank: {chunk_idx: path}} for every chunk file present."""
    out: Dict[int, Dict[int, str]] = {}
    for name in os.listdir(trace_dir):
        m = CHUNK_RE.match(name)
        if not m:
            continue
        rank = int(m.group(1))
        chunk_idx = int(m.group(2))
        out.setdefault(rank, {})[chunk_idx] = os.path.join(trace_dir, name)
    return out


def merge_rank_chunks(chunk_paths) -> dict:
    """Merge chunks for one rank. Each chunk is a full FR buffer snapshot;
    entries overlap across chunks. Merger keeps the LATEST version of each
    entry, keyed by (pg_id, collective_seq_id, p2p_seq_id). Rationale: GPU
    timing fields are populated asynchronously by the NCCL watchdog, so the
    most recent snapshot of a given entry has the most complete timing.

    Each merged entry is tagged with `_iter_num` = the training iteration
    whose dump first produced this entry (parsed from the chunk filename).

    Returns a trace dict structurally identical to a legacy `_dump_{rank}.json`.
    pg_config / pg_status are taken from the latest chunk.
    """
    chunk_paths_sorted = sorted(chunk_paths.items())  # by chunk idx (time)
    # Keep earliest iter_num seen (entry first captured at that iter).
    latest: Dict[Tuple, dict] = {}
    earliest_iter: Dict[Tuple, int] = {}
    pg_config = {}
    pg_status = {}
    iter_re = re.compile(r"_iter(\d+)\.json$")
    for _, path in chunk_paths_sorted:
        m = iter_re.search(path)
        chunk_iter = int(m.group(1)) if m else -1
        with open(path) as f:
            d = json.load(f)
        pg_config = d.get("pg_config", pg_config) or pg_config
        pg_status = d.get("pg_status", pg_status) or pg_status
        for e in d.get("entries", []):
            key = (e.get("pg_id"), e.get("collective_seq_id"), e.get("p2p_seq_id", -1))
            latest[key] = e   # last-write-wins on content (for GPU timing maturation)
            if key not in earliest_iter or chunk_iter < earliest_iter[key]:
                earliest_iter[key] = chunk_iter
    for key, e in latest.items():
        e["_iter_num"] = earliest_iter[key]
    merged_entries = sorted(
        latest.values(),
        key=lambda e: (e.get("collective_seq_id", 0), e.get("p2p_seq_id", -1)),
    )
    return {
        "pg_config": pg_config,
        "pg_status": pg_status,
        "entries": merged_entries,
    }


def stage_view(trace_dir: str, chunks_by_rank: Dict[int, Dict[int, str]]) -> str:
    """Materialize merged per-rank traces in `trace_dir/.monitor_view/`
    where `analyze_quiet` can read them. Returns the staging path."""
    staging = os.path.join(trace_dir, STAGING_SUBDIR)
    os.makedirs(staging, exist_ok=True)
    for rank, chunk_paths in chunks_by_rank.items():
        merged = merge_rank_chunks(chunk_paths)
        out_path = os.path.join(staging, f"_dump_{rank}.json")
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(merged, f)
        os.replace(tmp, out_path)
    # Mirror run_config.log so analyze_quiet can read ground truth.
    src_cfg = os.path.join(trace_dir, "run_config.log")
    if os.path.exists(src_cfg):
        shutil.copy2(src_cfg, os.path.join(staging, "run_config.log"))
    return staging


def coverage_summary(chunks_by_rank: Dict[int, Dict[int, str]]) -> dict:
    return {
        "ranks_seen": sorted(chunks_by_rank.keys()),
        "chunks_per_rank": {r: len(ch) for r, ch in chunks_by_rank.items()},
        "total_chunks": sum(len(ch) for ch in chunks_by_rank.values()),
    }


def run(trace_dir: str, world_size: int, poll_s: float, analyze_every_s: float,
        quiescence_s: float, k: float, floor_ms: float) -> int:
    verdicts_path = os.path.join(trace_dir, "verdicts.jsonl")
    print(f"[monitor] watching {trace_dir}", flush=True)
    print(f"[monitor] world_size={world_size} poll={poll_s}s analyze_every={analyze_every_s}s "
          f"quiescence={quiescence_s}s k={k} floor_ms={floor_ms}", flush=True)
    print(f"[monitor] verdicts -> {verdicts_path}", flush=True)

    last_total_chunks = 0
    last_analysis_ts = 0.0
    last_new_chunk_ts = time.time()
    verdict_idx = 0
    exit_code = 0

    while True:
        try:
            chunks = discover_chunks(trace_dir)
        except FileNotFoundError:
            # Trace dir may not exist yet at daemon start — wait.
            time.sleep(poll_s)
            continue

        total_chunks = sum(len(c) for c in chunks.values())
        now = time.time()

        if total_chunks > last_total_chunks:
            last_new_chunk_ts = now
            last_total_chunks = total_chunks

        full_coverage = len(chunks) >= world_size
        due_for_analysis = full_coverage and (now - last_analysis_ts) >= analyze_every_s
        quiescent = (now - last_new_chunk_ts) >= quiescence_s and total_chunks > 0

        if due_for_analysis or quiescent:
            try:
                staging = stage_view(trace_dir, chunks)
                result = analyzer.analyze_quiet(staging, k=k, floor_ms=floor_ms)
                # analyze_quiet returns a set for head_straggler_ranks — not JSON-able.
                result["head_straggler_ranks"] = sorted(result.get("head_straggler_ranks") or [])
                verdict = {
                    "idx": verdict_idx,
                    "ts": now,
                    "final": bool(quiescent),
                    "coverage": coverage_summary(chunks),
                    "result": result,
                }
                with open(verdicts_path, "a") as f:
                    f.write(json.dumps(verdict) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                print(
                    f"[monitor] verdict #{verdict_idx} "
                    f"{'(FINAL) ' if quiescent else ''}"
                    f"total_chunks={total_chunks} "
                    f"windows={result.get('n_windows')} "
                    f"heads={result.get('n_heads')} "
                    f"head_ranks={result.get('head_straggler_ranks')} "
                    f"hit={result.get('hit')}",
                    flush=True,
                )
                verdict_idx += 1
                last_analysis_ts = now
            except Exception as e:
                # Don't crash on partial data / early windows.
                print(f"[monitor] analysis error (continuing): {e}", flush=True)
                last_analysis_ts = now  # avoid tight retry loop

        if quiescent:
            print(f"[monitor] quiescent for {quiescence_s}s, exiting", flush=True)
            break

        time.sleep(poll_s)

    return exit_code


def parse_args():
    p = argparse.ArgumentParser(description="Online FR straggler monitor")
    p.add_argument("trace_dir", help="Directory that the trainer is writing chunks into")
    p.add_argument("--world-size", type=int, default=8,
                   help="Number of ranks to wait for before running an analysis")
    p.add_argument("--poll-s", type=float, default=0.5,
                   help="Filesystem poll interval in seconds")
    p.add_argument("--analyze-every-s", type=float, default=5.0,
                   help="Minimum seconds between two analysis invocations")
    p.add_argument("--quiescence-s", type=float, default=15.0,
                   help="No-new-chunk timeout that signals end of training")
    p.add_argument("--k", type=float, default=2.0, help="Analyzer k parameter")
    p.add_argument("--floor-ms", type=float, default=10.0, help="Analyzer floor_ms parameter")
    return p.parse_args()


def main():
    args = parse_args()
    sys.exit(run(
        trace_dir=args.trace_dir,
        world_size=args.world_size,
        poll_s=args.poll_s,
        analyze_every_s=args.analyze_every_s,
        quiescence_s=args.quiescence_s,
        k=args.k,
        floor_ms=args.floor_ms,
    ))


if __name__ == "__main__":
    main()
