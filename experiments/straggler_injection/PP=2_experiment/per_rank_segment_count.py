"""
Per-rank straggler count, segment-based.

Logic (as requested):
  Default_pg windows partition the trace into segments (one per training iter).
  Inside each segment, identify "the slowest rank" — the rank flagged
  in the most windows within that segment.
  Then count per rank: in how many segments was it the slowest?
  The rank with the max count is the identified straggler.
"""

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


def main():
    if len(sys.argv) != 2:
        print("Usage: per_rank_segment_count.py <trace_dir>")
        sys.exit(1)
    trace_dir = sys.argv[1]

    collectives_by_file, _, _ = load_trace_dir(trace_dir)
    windows, collectives_to_order = group_collectives_by_windows(collectives_by_file)

    all_window_stats = {}
    for key, colls in windows.items():
        all_window_stats[key] = compute_window_stats(key, colls, k=3.0, floor_ms=20.0)
    apply_p2p_duration_detection(all_window_stats, windows, k=3.0, floor_ms=20.0)

    world_ranks = sorted(collectives_by_file.keys(), key=int)
    segments = partition_reset_bounded_segments(
        all_window_stats, collectives_to_order, set(world_ranks)
    )

    # Per-segment: rank -> count of windows in this segment where rank is flagged
    segment_winners = []          # list of (segment_idx, winners, vote_breakdown)
    overall_winner_count = defaultdict(int)

    for segment in segments:
        flag_count = defaultdict(int)
        for key in segment.keys:
            ws = all_window_stats.get(key)
            if ws is None:
                continue
            for rank in ws.straggler_ranks:
                flag_count[rank] += 1

        if not flag_count:
            segment_winners.append((segment.index, [], dict(flag_count)))
            continue

        max_flags = max(flag_count.values())
        winners = [r for r, c in flag_count.items() if c == max_flags]
        segment_winners.append((segment.index, winners, dict(flag_count)))
        for w in winners:
            overall_winner_count[w] += 1

    print(f"Trace: {trace_dir}")
    gt = load_ground_truth(trace_dir)
    if gt:
        print(f"Ground truth: inject_type={gt.get('inject_type')}, "
              f"inject_rank={gt.get('inject_rank')}, "
              f"inject_delay_ms={gt.get('inject_delay_ms')}")
    print(f"Total segments (= default_pg-bounded iter blocks): {len(segments)}")
    print()

    print("=== Per-segment slowest rank (flagged in most windows) ===")
    for seg_idx, winners, votes in segment_winners:
        if not winners:
            print(f"  segment {seg_idx:3d}: no straggler flagged")
        else:
            vote_str = ", ".join(f"r{r}={c}" for r, c in sorted(votes.items(), key=lambda x: -x[1]))
            winner_str = ",".join(sorted(winners, key=int))
            print(f"  segment {seg_idx:3d}: winner=rank({winner_str})  [{vote_str}]")

    print()
    print("=== Final per-rank segment win count ===")
    print(f"{'rank':>4} | {'segments won':>13}")
    print(f"{'-'*4}-+-{'-'*13}")
    for r in world_ranks:
        print(f"{r:>4} | {overall_winner_count.get(r, 0):>13}")

    if overall_winner_count:
        max_wins = max(overall_winner_count.values())
        final_winners = [r for r, c in overall_winner_count.items() if c == max_wins]
        print()
        print(f"Identified straggler(s): rank(s) "
              f"{', '.join(sorted(final_winners, key=int))} "
              f"(won {max_wins}/{len(segments)} segments)")
    else:
        print("\nNo rank flagged in any segment.")


if __name__ == "__main__":
    main()
