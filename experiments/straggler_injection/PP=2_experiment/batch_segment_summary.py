"""
Sweep all batch traces and produce a per-rank segment-winner summary table.

For each trace:
  - Partition into default_pg-bounded segments
  - Per segment: rank flagged in most windows = "winner"
  - Count per-rank segment wins
  - Identify rank with max wins
  - Compare to ground truth
"""

import glob
import os
import sys
from collections import defaultdict

from fr_straggler_analyzer import (
    apply_p2p_duration_detection,
    compute_window_stats,
    group_collectives_by_windows,
    load_ground_truth,
    load_trace_dir,
    partition_reset_bounded_segments,
)


def analyze_trace(trace_dir: str) -> dict:
    collectives_by_file, _, _ = load_trace_dir(trace_dir)
    windows, order = group_collectives_by_windows(collectives_by_file)
    ws_map = {k: compute_window_stats(k, c, k=3.0, floor_ms=20.0) for k, c in windows.items()}
    apply_p2p_duration_detection(ws_map, windows, k=3.0, floor_ms=20.0)
    world_ranks = sorted(collectives_by_file.keys(), key=int)
    segments = partition_reset_bounded_segments(ws_map, order, set(world_ranks))

    winner_counts = defaultdict(int)
    for seg in segments:
        flag_count = defaultdict(int)
        for key in seg.keys:
            ws = ws_map.get(key)
            if ws is None:
                continue
            for r in ws.straggler_ranks:
                flag_count[r] += 1
        if not flag_count:
            continue
        max_flags = max(flag_count.values())
        for r, c in flag_count.items():
            if c == max_flags:
                winner_counts[r] += 1

    total_segments = len(segments)
    if winner_counts:
        max_wins = max(winner_counts.values())
        identified = [r for r, c in winner_counts.items() if c == max_wins]
    else:
        max_wins, identified = 0, []

    return {
        "world_ranks": world_ranks,
        "total_segments": total_segments,
        "winner_counts": dict(winner_counts),
        "identified": identified,
        "max_wins": max_wins,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: batch_segment_summary.py <traces_root>")
        sys.exit(1)
    root = sys.argv[1]
    trace_dirs = sorted(glob.glob(os.path.join(root, "20260512_*")))
    if not trace_dirs:
        print(f"No traces found in {root}")
        sys.exit(1)

    rows = []
    for td in trace_dirs:
        # Suppress per-trace stdout noise
        import io, contextlib
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                res = analyze_trace(td)
        except Exception as e:
            print(f"  ERROR on {os.path.basename(td)}: {e}", file=sys.stderr)
            continue
        gt = load_ground_truth(td)
        rows.append({
            "trace": os.path.basename(td),
            "inject_type": gt.get("inject_type", "?") if gt else "?",
            "inject_rank": str(gt.get("inject_rank", "?")) if gt else "?",
            "inject_delay_ms": gt.get("inject_delay_ms", "?") if gt else "?",
            **res,
        })

    if not rows:
        print("No rows produced.")
        return

    world_ranks = rows[0]["world_ranks"]

    header = (
        f"{'type':<7} | {'inj_r':>5} | {'delay':>5} | {'segs':>4} | "
        + " | ".join(f"r{r}" for r in world_ranks)
        + f" | {'ident':>7} | {'match':>5}"
    )
    print(header)
    print("-" * len(header))

    correct = 0
    partial = 0
    wrong = 0
    for row in rows:
        wc = row["winner_counts"]
        ident = ",".join(sorted(row["identified"], key=int)) if row["identified"] else "-"
        if row["inject_type"] == "none":
            match = "—"
        else:
            inj = row["inject_rank"]
            if row["identified"] == [inj]:
                match = "OK"
                correct += 1
            elif inj in row["identified"]:
                match = "PART"
                partial += 1
            else:
                match = "MISS"
                wrong += 1
        cells = " | ".join(f"{wc.get(r, 0):>2}" for r in world_ranks)
        print(
            f"{row['inject_type']:<7} | "
            f"{row['inject_rank']:>5} | "
            f"{str(row['inject_delay_ms']):>5} | "
            f"{row['total_segments']:>4} | "
            f"{cells} | "
            f"{ident:>7} | "
            f"{match:>5}"
        )

    total = correct + partial + wrong
    if total:
        print()
        print(f"Summary: {correct}/{total} exact match, "
              f"{partial}/{total} partial (inj rank tied), "
              f"{wrong}/{total} miss")


if __name__ == "__main__":
    main()
