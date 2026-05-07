"""
FR Straggler Analyzer — standalone offline analysis of FR traces from straggler injection experiments.

Data flow:
  Load trace dir → Parse entries (completed only) → Build collectives_by_file
    → group_collectives_by_windows()
    → Per-window per-rank stats (all 3 timing signals)
    → Identify straggler rank(s) per window (max deviation from median)
    → Build PG overlap graph (PGs sharing ranks get edges)
    → Graph traversal to find root-cause PG (earliest straggler in causal chain)
    → Print results + ground truth comparison

Usage:
  python fr_straggler_analyzer.py /path/to/trace_dir
  python fr_straggler_analyzer.py /path/to/trace_dir -v
  python fr_straggler_analyzer.py /path/to/trace_dir --pg TENSOR
"""

import argparse
import glob
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Collective:
    """A single collective operation from an FR dump entry."""
    record_id: int
    file_id: str  # rank id (extracted from filename)
    collective_seq_id: int
    p2p_seq_id: int
    pg_id: int  # c10d handle (local to rank)
    op_id: int
    profiling_name: str
    state: str
    time_created_ns: int
    time_discovered_started_ns: Optional[int]
    time_discovered_completed_ns: Optional[int]
    duration_ms: Optional[float]
    process_group: List[str]  # [megatron_id, pg_desc]
    input_sizes: List[List[int]]
    output_sizes: List[List[int]]
    input_dtypes: List[str]
    output_dtypes: List[str]
    is_p2p: bool


# ---------------------------------------------------------------------------
# 1. Load trace directory
# ---------------------------------------------------------------------------

def load_trace_dir(trace_dir: str) -> Tuple[Dict[str, List[Collective]], Dict[str, dict], Dict[str, dict]]:
    """
    Load all rank dumps from a trace directory.

    Returns:
        collectives_by_file: rank_id -> list of Collective (completed only)
        pg_configs: megatron_id -> {desc, ranks}
        pg_status: rank_id -> {c10d_handle -> status}
    """
    collectives_by_file: Dict[str, List[Collective]] = {}
    pg_configs: Dict[str, dict] = {}
    pg_status: Dict[str, dict] = {}

    json_files = sorted(glob.glob(os.path.join(trace_dir, "_dump_*.json")))
    if not json_files:
        raise ValueError(f"No _dump_*.json files found in {trace_dir}")

    for filepath in json_files:
        file_id = Path(filepath).stem
        rank_id = file_id.split("_")[-1]

        with open(filepath, "r") as f:
            data = json.load(f)

        # Extract pg_config
        if "pg_config" in data:
            for group_id, mapping in data["pg_config"].items():
                ranks_str = mapping.get("ranks", "").strip("[]").split(",")
                ranks = set()
                for r in ranks_str:
                    r = r.strip()
                    if r:
                        try:
                            ranks.add(int(r))
                        except ValueError:
                            pass
                if group_id not in pg_configs:
                    pg_configs[group_id] = {"desc": mapping.get("desc", ""), "ranks": ranks}
                else:
                    pg_configs[group_id]["ranks"] |= ranks

        # Extract pg_status
        if "pg_status" in data:
            pg_status[rank_id] = data["pg_status"]

        # Extract collectives — completed entries only
        collectives = []
        for entry in data.get("entries", []):
            if "collective_seq_id" not in entry:
                continue
            if entry.get("state") != "completed":
                continue
            p2p_seq_id = entry.get("p2p_seq_id", -1)
            collectives.append(Collective(
                record_id=entry.get("record_id", -1),
                file_id=rank_id,
                collective_seq_id=entry["collective_seq_id"],
                p2p_seq_id=p2p_seq_id,
                pg_id=entry["pg_id"],
                op_id=entry.get("op_id", -1),
                profiling_name=entry.get("profiling_name", ""),
                state=entry["state"],
                time_created_ns=entry["time_created_ns"],
                time_discovered_started_ns=entry.get("time_discovered_started_ns"),
                time_discovered_completed_ns=entry.get("time_discovered_completed_ns"),
                duration_ms=entry.get("duration_ms"),
                process_group=entry["process_group"],
                input_sizes=entry.get("input_sizes", []),
                output_sizes=entry.get("output_sizes", []),
                input_dtypes=entry.get("input_dtypes", []),
                output_dtypes=entry.get("output_dtypes", []),
                is_p2p=p2p_seq_id > 0 or entry.get("is_p2p", False),
            ))

        collectives_by_file[rank_id] = collectives

    logger.info(f"Loaded {len(json_files)} rank dumps from {trace_dir}")
    for rank_id, colls in sorted(collectives_by_file.items(), key=lambda x: int(x[0])):
        logger.info(f"  rank {rank_id}: {len(colls)} completed entries")

    return collectives_by_file, pg_configs, pg_status


# ---------------------------------------------------------------------------
# 2. Windowing — two-pass architecture
#
# Pass 1 (row compaction): Per-rank, compress consecutive same-megatron_id
#         entries into windows. P2P entries are singletons (no merging).
# Pass 2 (column ordering): Group windows by (megatron_id, window_id),
#         sort by min(time_created_ns), assign global_idx.
#
# No majority voting. Timestamp-only ordering.
# ---------------------------------------------------------------------------

@dataclass
class Window:
    """A contiguous run of same-PG entries from one rank."""
    megatron_id: str
    pg_desc: str
    window_id: int  # occurrence count for collectives, p2p_seq_id for P2P
    entries: List[Collective]
    is_p2p: bool
    rank_id: str

    @property
    def min_time_created_ns(self) -> int:
        return min(e.time_created_ns for e in self.entries)

    @property
    def max_time_created_ns(self) -> int:
        return max(e.time_created_ns for e in self.entries)

    @property
    def participating_ranks(self) -> Set[str]:
        """For collectives, this is set after cross-rank grouping. For P2P, derived from entries."""
        return set(e.file_id for e in self.entries)


def _parse_p2p_participants(profiling_name: str, pg_config: Optional[dict]) -> Set[int]:
    """
    Parse P2P participants from profiling_name like "send 0->3" or "recv 2<-3".
    Returns global rank IDs if pg_config is available, otherwise local indices.
    """
    import re
    # Match patterns like "send 0->3" or "recv 2<-3"
    match = re.search(r'(\d+)\s*[<>-]+\s*(\d+)', profiling_name)
    if not match:
        return set()
    local_src, local_dst = int(match.group(1)), int(match.group(2))
    # For now, return the local indices. In practice, we'd map through pg_config.
    # The Window participating_ranks will be populated after cross-rank grouping.
    return {local_src, local_dst}


def _assign_windows_per_rank(
    collectives_by_file: Dict[str, List[Collective]],
) -> Dict[str, List[Window]]:
    """
    Pass 1: Per-rank window assignment (row compaction).

    For each rank:
    - Walk entries in order (already time-sorted by time_created_ns)
    - Collectives: merge consecutive same-megatron_id into windows.
      window_id = occurrence count of this megatron_id as a distinct run.
    - P2P: each entry is its own singleton window.
      window_id = p2p_seq_id (globally unique per operation).

    Returns:
        Dict[rank_id, List[Window]] — windows per rank, in chronological order.
    """
    windows_by_rank: Dict[str, List[Window]] = {}

    for rank_id, entries in collectives_by_file.items():
        windows: List[Window] = []
        pg_occurrence_count: Dict[str, int] = defaultdict(int)  # megatron_id -> count

        i = 0
        while i < len(entries):
            entry = entries[i]
            megatron_id = entry.process_group[0]
            pg_desc = entry.process_group[1]
            is_p2p = entry.is_p2p

            if is_p2p:
                # P2P: singleton window, window_id = p2p_seq_id
                windows.append(Window(
                    megatron_id=megatron_id,
                    pg_desc=pg_desc,
                    window_id=entry.p2p_seq_id,
                    entries=[entry],
                    is_p2p=True,
                    rank_id=rank_id,
                ))
                i += 1
            else:
                # Collective: merge consecutive same-megatron_id entries
                window_entries = [entry]
                j = i + 1
                while j < len(entries):
                    next_entry = entries[j]
                    next_megatron_id = next_entry.process_group[0]
                    next_is_p2p = next_entry.is_p2p
                    if next_is_p2p or next_megatron_id != megatron_id:
                        break
                    window_entries.append(next_entry)
                    j += 1

                # window_id = occurrence count of this megatron_id
                window_id = pg_occurrence_count[megatron_id]
                pg_occurrence_count[megatron_id] += 1

                windows.append(Window(
                    megatron_id=megatron_id,
                    pg_desc=pg_desc,
                    window_id=window_id,
                    entries=window_entries,
                    is_p2p=False,
                    rank_id=rank_id,
                ))
                i = j

        windows_by_rank[rank_id] = windows

    return windows_by_rank


def _order_windows_globally(
    windows_by_rank: Dict[str, List[Window]],
) -> Tuple[Dict[Tuple[str, str, int], List[Collective]], Dict[Tuple[str, str, int], int]]:
    """
    Pass 2: Cross-rank window ordering (column compaction).

    1. Collect all windows from all ranks.
    2. Group by (megatron_id, window_id) — same key for collectives and P2P.
    3. Sort groups by min(time_created_ns).
    4. Assign global_idx = position in sorted order.

    Returns:
        grouped_windows: (megatron_id, pg_desc, window_id) -> List[Collective]
        collectives_to_order: (megatron_id, pg_desc, window_id) -> global_idx
    """
    # Step 1: Collect all windows and group by (megatron_id, window_id)
    # Use (megatron_id, window_id) as grouping key; pg_desc is carried for output
    from collections import defaultdict

    GroupKey = Tuple[str, int]  # (megatron_id, window_id)
    groups: Dict[GroupKey, List[Window]] = defaultdict(list)

    for rank_id, windows in windows_by_rank.items():
        for w in windows:
            key = (w.megatron_id, w.window_id)
            groups[key].append(w)

    # Step 2: For each group, compute ordering timestamp and pg_desc.
    # Collectives use earliest enqueue. For P2P in Megatron's pipeline
    # schedule, the receiver posts recv first;
    # Latest enqueue therefore selects the sender/data-ready side. 
    group_info: List[Tuple[GroupKey, int, str, List[Window]]] = []
    for key, window_list in groups.items():
        is_p2p = any(w.is_p2p for w in window_list)
        order_ts = (
            max(w.max_time_created_ns for w in window_list)
            if is_p2p
            else min(w.min_time_created_ns for w in window_list)
        )
        # pg_desc should be consistent across windows in the group
        pg_desc = window_list[0].pg_desc
        group_info.append((key, order_ts, pg_desc, window_list))

    # Step 3: Sort by ordering timestamp
    group_info.sort(key=lambda x: x[1])

    # Step 4: Build output structures
    grouped_windows: Dict[Tuple[str, str, int], List[Collective]] = {}
    collectives_to_order: Dict[Tuple[str, str, int], int] = {}

    for global_idx, (key, min_ts, pg_desc, window_list) in enumerate(group_info):
        megatron_id, window_id = key
        output_key = (megatron_id, pg_desc, window_id)

        # Flatten entries from all windows in this group
        all_entries: List[Collective] = []
        for w in window_list:
            all_entries.extend(w.entries)

        grouped_windows[output_key] = all_entries
        collectives_to_order[output_key] = global_idx

    return grouped_windows, collectives_to_order


def group_collectives_by_windows(
    collectives_by_file: Dict[str, List[Collective]],
) -> Tuple[Dict[Tuple[str, str, int], List[Collective]], Dict[Tuple[str, str, int], int]]:
    """
    Two-pass windowing: group collectives by PG and temporal phase.

    Pass 1: Per-rank row compaction (consecutive same-PG → window)
    Pass 2: Cross-rank column ordering (by timestamp → global_idx)

    Returns:
        grouped_windows: (megatron_id, pg_desc, window_id) -> List[Collective]
        collectives_to_order: (megatron_id, pg_desc, window_id) -> global_idx
    """
    # Pass 1: Per-rank window assignment
    windows_by_rank = _assign_windows_per_rank(collectives_by_file)

    # Log pass 1 results
    total_windows = sum(len(ws) for ws in windows_by_rank.values())
    logger.info(f"\nPass 1 (row compaction): {total_windows} windows across {len(windows_by_rank)} ranks")
    for rank_id in sorted(windows_by_rank.keys(), key=int):
        windows = windows_by_rank[rank_id]
        collective_wins = sum(1 for w in windows if not w.is_p2p)
        p2p_wins = sum(1 for w in windows if w.is_p2p)
        logger.debug(f"  rank {rank_id}: {len(windows)} windows ({collective_wins} collective, {p2p_wins} p2p)")

    # Pass 2: Cross-rank ordering
    grouped_windows, collectives_to_order = _order_windows_globally(windows_by_rank)

    logger.info(f"Pass 2 (column ordering): {len(grouped_windows)} unique window groups")

    return grouped_windows, collectives_to_order


# ---------------------------------------------------------------------------
# 3. Per-window statistics
# ---------------------------------------------------------------------------

# WindowKey identifies one grouped window across ranks:
#   (megatron_id, pg_desc, window_idx)
# For collectives, window_idx is the per-PG occurrence count. 
# For P2P, window_idx is the p2p_seq_id for one directional transfer.
WindowKey = Tuple[str, str, int]

@dataclass
class RankWindowStats:
    """Timing stats for one rank within one window."""
    rank_id: str
    n_entries: int
    mean_created_offset_ms: float   # mean(time_created - window_min_created) in ms
    mean_started_offset_ms: float   # mean(time_started - window_min_started) in ms
    mean_duration_ms: float         # mean(duration_ms) from cudaEventElapsedTime


@dataclass
class WindowStats:
    """Stats for one window across all participating ranks."""
    key: Tuple[str, str, int]  # (megatron_id, pg_desc, window_idx)
    rank_stats: Dict[str, RankWindowStats]
    median_created_offset_ms: float
    median_started_offset_ms: float
    median_duration_ms: float
    straggler_ranks: Set[str]  # ranks flagged as stragglers
    straggler_reasons: Dict[str, List[str]]  # rank -> signal(s) that flagged it

    @property
    def straggler_signal(self) -> str:
        """Compact display string for summary tables."""
        parts = []
        for rank_id in sorted(self.straggler_reasons.keys(), key=int):
            reasons = "; ".join(self.straggler_reasons[rank_id])
            parts.append(f"r{rank_id}: {reasons}")
        return " | ".join(parts)


def get_window_ranks(ws: WindowStats) -> Set[str]:
    """Observed participating ranks for a window."""
    return set(ws.rank_stats.keys())


def compute_window_stats(
    window_key: Tuple[str, str, int],
    collectives: List[Collective],
    k: float = 3.0,
    floor_ms: float = 20.0,
) -> WindowStats:
    """
    Compute per-rank timing stats for a single window and identify stragglers.

    Straggler identification: single-worst-outlier with gap-to-runner-up.
    Flags the rank with largest offset only if its gap to the runner-up exceeds
    a threshold based on k * stdev(non-worst offsets) or floor_ms.
    """
    import math

    # Group collectives by rank
    by_rank: Dict[str, List[Collective]] = defaultdict(list)
    for c in collectives:
        by_rank[c.file_id].append(c)

    if not by_rank:
        return WindowStats(
            key=window_key, rank_stats={},
            median_created_offset_ms=0, median_started_offset_ms=0,
            median_duration_ms=0, straggler_ranks=set(), straggler_reasons={},
        )

    # Compute per-rank means for each signal
    rank_stats: Dict[str, RankWindowStats] = {}

    # For relative offsets, find the window-wide min timestamp per signal
    all_created = [c.time_created_ns for c in collectives]
    min_created = min(all_created)

    all_started = [c.time_discovered_started_ns for c in collectives
                   if c.time_discovered_started_ns is not None]
    min_started = min(all_started) if all_started else 0

    for rank_id, rank_colls in by_rank.items():
        # time_created offsets (relative to window min for readability)
        created_offsets = [(c.time_created_ns - min_created) / 1e6 for c in rank_colls]
        mean_created = sum(created_offsets) / len(created_offsets)

        # time_started offsets
        started_offsets = []
        for c in rank_colls:
            if c.time_discovered_started_ns is not None:
                started_offsets.append((c.time_discovered_started_ns - min_started) / 1e6)
        mean_started = sum(started_offsets) / len(started_offsets) if started_offsets else 0.0

        # duration_ms: CUDA-event-measured elapsed time from FR.
        duration_values = []
        for c in rank_colls:
            if c.duration_ms is not None:
                duration_values.append(c.duration_ms)
        mean_duration = sum(duration_values) / len(duration_values) if duration_values else 0.0

        rank_stats[rank_id] = RankWindowStats(
            rank_id=rank_id,
            n_entries=len(rank_colls),
            mean_created_offset_ms=mean_created,
            mean_started_offset_ms=mean_started,
            mean_duration_ms=mean_duration,
        )

    # Compute medians across ranks (kept for reporting, not used in flagging)
    created_values = sorted(rs.mean_created_offset_ms for rs in rank_stats.values())
    started_values = sorted(rs.mean_started_offset_ms for rs in rank_stats.values())
    duration_values = sorted(rs.mean_duration_ms for rs in rank_stats.values())

    def median(vals):
        n = len(vals)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2

    med_created = median(created_values)
    med_started = median(started_values)
    med_duration = median(duration_values)

    # --- Single-worst-outlier with gap-to-runner-up ---
    straggler_ranks = set()
    straggler_reasons: Dict[str, List[str]] = defaultdict(list)
    is_p2p_window = any(c.is_p2p for c in collectives)
    skip_p2p_warmup = is_p2p_window and window_key[2] <= 2

    if len(rank_stats) >= 2 and not skip_p2p_warmup:
        # time_created: the late CPU enqueue side is the candidate straggler.
        sorted_by_created = sorted(
            rank_stats.items(),
            key=lambda x: x[1].mean_created_offset_ms,
            reverse=True,
        )
        worst_rank, worst_rs = sorted_by_created[0]
        worst_offset = worst_rs.mean_created_offset_ms
        runner_up_offset = sorted_by_created[1][1].mean_created_offset_ms
        gap = worst_offset - runner_up_offset

        # Compute gap threshold
        if len(sorted_by_created) <= 2:
            gap_threshold = floor_ms
        else:
            # stdev of non-worst offsets
            non_worst_offsets = [rs.mean_created_offset_ms for _, rs in sorted_by_created[1:]]
            mean_non_worst = sum(non_worst_offsets) / len(non_worst_offsets)
            variance = sum((x - mean_non_worst) ** 2 for x in non_worst_offsets) / len(non_worst_offsets)
            stdev = math.sqrt(variance)
            gap_threshold = max(k * stdev, floor_ms)

        if gap > gap_threshold:
            straggler_ranks.add(worst_rank)
            straggler_reasons[worst_rank].append(
                f"time_created gap={gap:.1f}ms (thresh={gap_threshold:.1f}ms)"
            )

        if not is_p2p_window:
            # Collective duration_ms: the late rank joins last, so its CUDA-event
            # duration is shortest. Use the same gap-to-runner-up test as tc_gap.
            sorted_by_duration = sorted(
                (
                    (rank_id, rs)
                    for rank_id, rs in rank_stats.items()
                    if rs.mean_duration_ms > 0
                ),
                key=lambda x: x[1].mean_duration_ms,
            )
            if len(sorted_by_duration) >= 2:
                shortest_rank, shortest_rs = sorted_by_duration[0]
                shortest = shortest_rs.mean_duration_ms
                runner_up = sorted_by_duration[1][1].mean_duration_ms
                duration_gap = runner_up - shortest

                if len(sorted_by_duration) <= 2:
                    duration_threshold = floor_ms
                else:
                    non_shortest = [rs.mean_duration_ms for _, rs in sorted_by_duration[1:]]
                    mean_non_shortest = sum(non_shortest) / len(non_shortest)
                    variance = sum((x - mean_non_shortest) ** 2 for x in non_shortest) / len(non_shortest)
                    stdev = math.sqrt(variance)
                    duration_threshold = max(k * stdev, floor_ms)

                if duration_gap > duration_threshold:
                    straggler_ranks.add(shortest_rank)
                    straggler_reasons[shortest_rank].append(
                        f"duration_ms short gap={duration_gap:.1f}ms "
                        f"(thresh={duration_threshold:.1f}ms)"
                    )

    return WindowStats(
        key=window_key,
        rank_stats=rank_stats,
        median_created_offset_ms=med_created,
        median_started_offset_ms=med_started,
        median_duration_ms=med_duration,
        straggler_ranks=straggler_ranks,
        straggler_reasons=dict(straggler_reasons),
    )


def apply_p2p_duration_detection(
    all_window_stats: Dict[WindowKey, WindowStats],
    grouped_windows: Dict[WindowKey, List[Collective]],
    k: float = 3.0,
    floor_ms: float = 20.0,
) -> None:
    """
    First-pass P2P duration detector.

    P2P has only two coalesced entries per directional transfer, so there is
    no meaningful within-window rank distribution. Instead, compare the max
    duration_ms of each P2P window against the low-duration baseline for the
    same PG across time. In flagged windows, attribute the straggler to the
    rank with max(time_created_ns), which is the sender/data-ready side for
    high-duration P2P windows in the pp2_default_pg traces.
    """
    import math

    samples_by_pg: Dict[Tuple[str, str], List[Tuple[WindowKey, float, str]]] = defaultdict(list)

    for key, collectives in grouped_windows.items():
        if key[2] <= 2:
            continue
        if not any(c.is_p2p for c in collectives):
            continue
        if len(collectives) != 2:
            continue

        duration_entries = [c for c in collectives if c.duration_ms is not None]
        if len(duration_entries) != 2:
            continue

        signal_ms = max(c.duration_ms or 0.0 for c in duration_entries)
        sender_rank = max(collectives, key=lambda c: c.time_created_ns).file_id
        megatron_id, pg_desc, _ = key
        samples_by_pg[(megatron_id, pg_desc)].append((key, signal_ms, sender_rank))

    def median(vals: List[float]) -> float:
        n = len(vals)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2

    for samples in samples_by_pg.values():
        if len(samples) < 3:
            continue

        values = sorted(signal_ms for _, signal_ms, _ in samples)
        baseline_values = values[:max(1, len(values) // 2)]
        baseline_ms = median(baseline_values)

        if len(baseline_values) <= 1:
            threshold_ms = floor_ms
        else:
            mean_base = sum(baseline_values) / len(baseline_values)
            variance = sum((x - mean_base) ** 2 for x in baseline_values) / len(baseline_values)
            threshold_ms = max(k * math.sqrt(variance), floor_ms)

        for key, signal_ms, sender_rank in samples:
            delta_ms = signal_ms - baseline_ms
            if delta_ms <= threshold_ms:
                continue
            ws = all_window_stats[key]
            ws.straggler_ranks.add(sender_rank)
            ws.straggler_reasons.setdefault(sender_rank, []).append(
                f"p2p recv duration_ms={signal_ms:.1f}ms baseline={baseline_ms:.1f}ms "
                f"delta={delta_ms:.1f}ms (thresh={threshold_ms:.1f}ms)"
            )


# ---------------------------------------------------------------------------
# 4. Graph traversal for cascade attribution
#
#    Builds an overlap graph of straggler-flagged windows and finds HEADs
#    (cascade roots) via dynamic programming on global_idx ordering.
# ---------------------------------------------------------------------------

@dataclass
class CascadeResult:
    """Result of cascade graph traversal."""
    heads: Set[WindowKey]  # Windows with no straggler-predecessor
    predecessors: Dict[WindowKey, List[WindowKey]]  # All predecessors per node
    successors: Dict[WindowKey, List[WindowKey]]  # All successors per node
    longest_path_length: Dict[WindowKey, int]  # For choosing chain in non-verbose mode
    best_predecessor: Dict[WindowKey, Optional[WindowKey]]  # For longest chain reconstruction


@dataclass
class Segment:
    """Reset-bounded contiguous slice of globally ordered windows."""
    index: int
    keys: List[WindowKey]
    segment_type: str  # "complete" or "open_tail"
    terminal_reset: Optional[WindowKey]


def build_cascade_graph(
    window_stats: Dict[WindowKey, WindowStats],
    collectives_to_order: Dict[WindowKey, int],
) -> CascadeResult:
    """
    Build overlap graph of straggler windows and identify HEADs via dynamic programming.

    Nodes = windows with non-empty straggler_ranks.
    Edges = windows sharing any participating rank, directed by global_idx (lower → higher).
    HEADs = nodes with no straggler-predecessor (empty predecessors list).
    """
    # Filter out init windows (win≤1): rank 0 has startup coordination overhead
    # that creates false straggler signals. Real stragglers appear in steady-state.
    straggler_windows = {
        k: ws for k, ws in window_stats.items()
        if ws.straggler_ranks and k[2] >= 2  # k[2] is window_idx
    }

    if not straggler_windows:
        return CascadeResult(
            heads=set(),
            predecessors={},
            successors={},
            longest_path_length={},
            best_predecessor={},
        )

    # Map window keys → integer node IDs for efficient graph ops
    keys = sorted(straggler_windows.keys(), key=lambda k: collectives_to_order.get(k, 0))
    key_to_node = {k: i for i, k in enumerate(keys)}
    node_to_key = {i: k for k, i in key_to_node.items()}

    # Participating ranks per window (full membership, not just stragglers)
    node_ranks: Dict[int, Set[str]] = {
        key_to_node[k]: get_window_ranks(ws) for k, ws in straggler_windows.items()
    }

    # Build undirected adjacency (directionality applied at traversal time)
    node_ids = list(node_to_key.keys())
    neighbors: Dict[int, Set[int]] = defaultdict(set)
    for n1 in node_ids:
        for n2 in node_ids:
            if n1 != n2 and node_ranks[n1] & node_ranks[n2]:
                neighbors[n1].add(n2)
                neighbors[n2].add(n1)

    # Dynamic programming: compute predecessors and longest path for each node
    node_order = {n: collectives_to_order.get(node_to_key[n], 0) for n in node_ids}
    sorted_nodes = sorted(node_ids, key=lambda n: node_order[n])

    # Track ALL predecessors (not just best one)
    predecessors: Dict[int, List[int]] = {n: [] for n in node_ids}
    # Track longest path length and best predecessor for chain reconstruction
    longest_len: Dict[int, int] = {n: 1 for n in node_ids}
    best_pred: Dict[int, Optional[int]] = {n: None for n in node_ids}

    for node in sorted_nodes:
        node_gidx = node_order[node]
        for nb in neighbors[node]:
            nb_gidx = node_order[nb]
            if nb_gidx > node_gidx:
                # node → nb is a valid directed edge (earlier → later)
                predecessors[nb].append(node)
                if longest_len[node] + 1 > longest_len[nb]:
                    longest_len[nb] = longest_len[node] + 1
                    best_pred[nb] = node

    # Build successor map by inverting predecessors
    successors: Dict[int, List[int]] = {n: [] for n in node_ids}
    for n, preds in predecessors.items():
        for p in preds:
            successors[p].append(n)

    # HEADs = nodes with empty predecessors list
    heads = {node_to_key[n] for n in node_ids if not predecessors[n]}

    # Convert back to window keys
    return CascadeResult(
        heads=heads,
        predecessors={node_to_key[n]: [node_to_key[p] for p in preds]
                      for n, preds in predecessors.items()},
        successors={node_to_key[n]: [node_to_key[s] for s in succs]
                    for n, succs in successors.items()},
        longest_path_length={node_to_key[n]: length for n, length in longest_len.items()},
        best_predecessor={node_to_key[n]: (node_to_key[p] if p is not None else None)
                          for n, p in best_pred.items()},
    )


# ---------------------------------------------------------------------------
# 4.5 Reset-bounded segmentation
# ---------------------------------------------------------------------------

def partition_reset_bounded_segments(
    all_window_stats: Dict[WindowKey, WindowStats],
    collectives_to_order: Dict[WindowKey, int],
    world_ranks: Set[str],
) -> List[Segment]:
    """
    Partition ordered windows into reset-bounded segments.

    A reset window is any window observed on exactly all world ranks. Complete
    segments end at a reset window, inclusive. Any trailing windows after the
    last reset become an open_tail segment.
    """
    ordered_keys = sorted(all_window_stats.keys(), key=lambda k: collectives_to_order.get(k, 0))
    segments: List[Segment] = []
    current_keys: List[WindowKey] = []

    for key in ordered_keys:
        current_keys.append(key)
        if get_window_ranks(all_window_stats[key]) == world_ranks:
            segments.append(Segment(
                index=len(segments),
                keys=current_keys,
                segment_type="complete",
                terminal_reset=key,
            ))
            current_keys = []

    if current_keys:
        segments.append(Segment(
            index=len(segments),
            keys=current_keys,
            segment_type="open_tail",
            terminal_reset=None,
        ))

    return segments


def filter_maps_to_segment(
    all_window_stats: Dict[WindowKey, WindowStats],
    collectives_to_order: Dict[WindowKey, int],
    segment: Segment,
) -> Tuple[Dict[WindowKey, WindowStats], Dict[WindowKey, int]]:
    """Filter window stats and ordering maps to one segment."""
    return (
        {k: all_window_stats[k] for k in segment.keys},
        {k: collectives_to_order[k] for k in segment.keys},
    )


def get_longest_chain_from_head(
    head: WindowKey,
    cascade: CascadeResult,
) -> List[WindowKey]:
    """Walk forward from HEAD following longest downstream path."""
    chain = [head]
    current = head
    while cascade.successors.get(current):
        # Pick successor with longest remaining path
        succs = cascade.successors[current]
        best = max(succs, key=lambda s: cascade.longest_path_length.get(s, 0))
        chain.append(best)
        current = best
    return chain


def get_full_dag_from_head(
    head: WindowKey,
    cascade: CascadeResult,
) -> List[Tuple[WindowKey, int]]:
    """
    BFS from HEAD, returning (window_key, depth) pairs for tree rendering.
    Each node appears once at its first-encountered depth.
    """
    result: List[Tuple[WindowKey, int]] = []
    visited: Set[WindowKey] = set()
    queue: List[Tuple[WindowKey, int]] = [(head, 0)]

    while queue:
        node, depth = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        result.append((node, depth))
        for succ in sorted(cascade.successors.get(node, []),
                           key=lambda s: cascade.longest_path_length.get(s, 0),
                           reverse=True):
            if succ not in visited:
                queue.append((succ, depth + 1))

    return result


# ---------------------------------------------------------------------------
# 5. Output formatting
# ---------------------------------------------------------------------------

def load_ground_truth(trace_dir: str) -> Optional[dict]:
    """Load run_config.log if present."""
    config_path = os.path.join(trace_dir, "run_config.log")
    if not os.path.exists(config_path):
        return None
    gt = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                gt[key.strip()] = val.strip()
    return gt


def print_summary(
    all_window_stats: Dict[WindowKey, WindowStats],
    collectives_to_order: Dict[WindowKey, int],
    cascade: CascadeResult,
    pg_filter: Optional[str] = None,
    verbose_cascade: bool = False,
    title: str = "Straggler Analysis Summary",
):
    """Print summary table of windows with stragglers, plus cascade chains from HEADs."""
    logger.info(f"\n=== {title} ===\n")

    # Header
    logger.info(
        f"{'GIdx':>4} | {'PG Desc':<40} | {'Win':>3} | {'Ranks':>20} | {'Straggler':>8} | {'Signal':<40}"
    )
    logger.info("-" * 130)

    straggler_window_count = 0
    total_windows = 0

    for key in sorted(all_window_stats.keys(), key=lambda k: collectives_to_order.get(k, 0)):
        ws = all_window_stats[key]
        megatron_id, pg_desc, window_idx = key

        if pg_filter and pg_filter.upper() not in pg_desc.upper():
            continue

        total_windows += 1
        gidx = collectives_to_order.get(key, -1)
        participating_ranks = sorted(int(r) for r in ws.rank_stats.keys())
        ranks_str = ",".join(str(r) for r in participating_ranks)

        if ws.straggler_ranks:
            straggler_window_count += 1
            straggler_str = ",".join(sorted(ws.straggler_ranks, key=int))
            signal_str = ws.straggler_signal
        else:
            straggler_str = "-"
            signal_str = ""

        logger.info(
            f"{gidx:>4} | {pg_desc:<40} | {window_idx:>3} | {ranks_str:>20} | {straggler_str:>8} | {signal_str:<40}"
        )

    logger.info(f"\nWindows with stragglers: {straggler_window_count}/{total_windows}")

    # Cascade chains from HEADs
    if cascade.heads:
        logger.info("\n=== Cascade Chains (from HEADs) ===\n")
        sorted_heads = sorted(cascade.heads, key=lambda k: collectives_to_order.get(k, 0))

        for head_idx, head in enumerate(sorted_heads):
            head_ws = all_window_stats[head]
            head_gidx = collectives_to_order.get(head, -1)
            head_ranks = ",".join(sorted(head_ws.straggler_ranks, key=int))
            megatron_id, pg_desc, window_idx = head

            logger.info(f"  HEAD {head_idx}: [{head_gidx}] {pg_desc} win={window_idx} (straggler rank(s): {head_ranks})")

            if verbose_cascade:
                # Full DAG rendering with indentation
                dag = get_full_dag_from_head(head, cascade)
                for node, depth in dag[1:]:  # Skip head (already printed)
                    ws = all_window_stats[node]
                    gidx = collectives_to_order.get(node, -1)
                    _, node_pg_desc, node_win_idx = node
                    straggler_str = ",".join(sorted(ws.straggler_ranks, key=int))
                    indent = "    " + "  " * depth
                    logger.info(f"{indent}→ [{gidx}] {node_pg_desc} win={node_win_idx} straggler={straggler_str}")
            else:
                # Longest chain only
                chain = get_longest_chain_from_head(head, cascade)
                for node in chain[1:]:  # Skip head (already printed)
                    ws = all_window_stats[node]
                    gidx = collectives_to_order.get(node, -1)
                    _, node_pg_desc, node_win_idx = node
                    straggler_str = ",".join(sorted(ws.straggler_ranks, key=int))
                    logger.info(f"      → [{gidx}] {node_pg_desc} win={node_win_idx} straggler={straggler_str}")

            logger.info("")  # Blank line between HEADs
    else:
        logger.info("\nNo HEADs found (no straggler windows).")


def print_combined_segmented_summary(
    all_window_stats: Dict[WindowKey, WindowStats],
    collectives_to_order: Dict[WindowKey, int],
    segments: List[Segment],
    pg_filter: Optional[str] = None,
):
    """Print one global window table with inline reset-bounded segment separators."""
    logger.info("\n=== Global Window Table (Segment-Bounded) ===\n")
    logger.info(
        f"{'GIdx':>4} | {'Seg':>5} | {'Type':<10} | {'PG Desc':<40} | {'Win':>3} | {'Ranks':>20} | {'Straggler':>8} | {'Signal':<40}"
    )
    logger.info("-" * 150)

    total_windows = 0
    straggler_window_count = 0

    for segment in segments:
        start_gidx = collectives_to_order.get(segment.keys[0], -1) if segment.keys else -1
        end_gidx = collectives_to_order.get(segment.keys[-1], -1) if segment.keys else -1
        if segment.segment_type == "open_tail":
            label = f"Segment {segment.index} open_tail / lower confidence, gidx {start_gidx}..{end_gidx}"
        else:
            label = f"Segment {segment.index} complete, gidx {start_gidx}..{end_gidx}"
        logger.info(f"--- {label} ---")

        for key in sorted(segment.keys, key=lambda k: collectives_to_order.get(k, 0)):
            ws = all_window_stats[key]
            _, pg_desc, window_idx = key

            if pg_filter and pg_filter.upper() not in pg_desc.upper():
                continue

            total_windows += 1
            gidx = collectives_to_order.get(key, -1)
            ranks_str = ",".join(str(r) for r in sorted(int(r) for r in get_window_ranks(ws)))

            if ws.straggler_ranks:
                straggler_window_count += 1
                straggler_str = ",".join(sorted(ws.straggler_ranks, key=int))
                signal_str = ws.straggler_signal
            else:
                straggler_str = "-"
                signal_str = ""

            logger.info(
                f"{gidx:>4} | {segment.index:>5} | {segment.segment_type:<10} | "
                f"{pg_desc:<40} | {window_idx:>3} | {ranks_str:>20} | "
                f"{straggler_str:>8} | {signal_str:<40}"
            )

    logger.info(f"\nWindows with stragglers: {straggler_window_count}/{total_windows}")


def print_segment_slices(
    segments: List[Segment],
    all_window_stats: Dict[WindowKey, WindowStats],
    collectives_to_order: Dict[WindowKey, int],
):
    """Print where reset-bounded segments slice the global window stream."""
    logger.info("\n=== Segment Slices ===\n")

    for segment in segments:
        if not segment.keys:
            continue

        start_gidx = collectives_to_order.get(segment.keys[0], -1)
        end_gidx = collectives_to_order.get(segment.keys[-1], -1)
        straggler_windows = sum(1 for k in segment.keys if all_window_stats[k].straggler_ranks)

        if segment.terminal_reset is None:
            terminal = "no terminal reset"
            label = f"Segment {segment.index} ({segment.segment_type} / lower confidence)"
        else:
            reset_gidx = collectives_to_order.get(segment.terminal_reset, -1)
            _, reset_pg_desc, reset_win = segment.terminal_reset
            terminal = f"terminal reset=[{reset_gidx}] {reset_pg_desc} win={reset_win}"
            label = f"Segment {segment.index} ({segment.segment_type})"

        logger.info(
            f"{label}: gidx {start_gidx}..{end_gidx}, {len(segment.keys)} windows, "
            f"{straggler_windows} straggler windows, {terminal}"
        )


def print_segment_cascade_heads(
    segment_cascades: List[Tuple[Segment, Dict[WindowKey, WindowStats], Dict[WindowKey, int], CascadeResult]],
):
    """Print compact per-segment cascade roots."""
    logger.info("\n=== Segment Cascade Heads ===\n")

    for segment, segment_stats, segment_order, cascade in segment_cascades:
        label = f"Segment {segment.index} ({segment.segment_type}"
        if segment.segment_type == "open_tail":
            label += " / lower confidence"
        label += ")"

        if not cascade.heads:
            logger.info(f"{label}: no HEADs")
            continue

        logger.info(f"{label}: {len(cascade.heads)} HEAD(s)")
        for head_idx, head in enumerate(sorted(cascade.heads, key=lambda k: segment_order.get(k, 0))):
            head_ws = segment_stats[head]
            head_gidx = segment_order.get(head, -1)
            _, pg_desc, window_idx = head
            head_ranks = ",".join(sorted(head_ws.straggler_ranks, key=int))
            logger.info(
                f"  HEAD {head_idx}: [{head_gidx}] {pg_desc} win={window_idx} "
                f"(straggler rank(s): {head_ranks})"
            )


def print_detailed(
    all_window_stats: Dict[Tuple[str, str, int], WindowStats],
    collectives_to_order: Dict[Tuple[str, str, int], int],
    pg_filter: Optional[str] = None,
):
    """Print detailed per-rank breakdown for each window."""
    for key in sorted(all_window_stats.keys(), key=lambda k: collectives_to_order.get(k, 0)):
        ws = all_window_stats[key]
        megatron_id, pg_desc, window_idx = key

        if pg_filter and pg_filter.upper() not in pg_desc.upper():
            continue

        straggler_marker = " [HAS STRAGGLER]" if ws.straggler_ranks else ""
        logger.info(f"\n--- {pg_desc} (megatron_id={megatron_id}, window={window_idx}){straggler_marker} ---")
        logger.info(
            f"  {'Rank':>6} | {'N':>4} | {'created_offset':>16} | {'started_offset':>16} | {'duration_ms':>12} | Reason"
        )

        for rank_id in sorted(ws.rank_stats.keys(), key=int):
            rs = ws.rank_stats[rank_id]
            marker = " <-- STRAGGLER" if rank_id in ws.straggler_ranks else ""
            reason = "; ".join(ws.straggler_reasons.get(rank_id, []))
            logger.info(
                f"  {rank_id:>6} | {rs.n_entries:>4} | "
                f"{rs.mean_created_offset_ms:>13.3f}ms | "
                f"{rs.mean_started_offset_ms:>13.3f}ms | "
                f"{rs.mean_duration_ms:>9.3f}ms | {reason}{marker}"
            )

        logger.info(
            f"  {'median':>6} | {'':>4} | "
            f"{ws.median_created_offset_ms:>13.3f}ms | "
            f"{ws.median_started_offset_ms:>13.3f}ms | "
            f"{ws.median_duration_ms:>9.3f}ms |"
        )


def print_ground_truth_comparison(
    all_window_stats: Dict[Tuple[str, str, int], WindowStats],
    ground_truth: Optional[dict],
):
    """Compare detected stragglers against ground truth."""
    if not ground_truth:
        logger.info("\nNo run_config.log found — skipping ground truth comparison.")
        return

    logger.info("\n=== Ground Truth Comparison ===\n")
    inject_type = ground_truth.get("inject_type", "unknown")
    inject_rank = ground_truth.get("inject_rank", "unknown")
    inject_delay = ground_truth.get("inject_delay_ms", "unknown")
    logger.info(f"  Ground truth: inject_type={inject_type}, inject_rank={inject_rank}, inject_delay_ms={inject_delay}")

    if inject_type == "none":
        # Check that no rank stands out
        flagged_windows = sum(1 for ws in all_window_stats.values() if ws.straggler_ranks)
        total = len(all_window_stats)
        logger.info(f"  Expected: no stragglers")
        logger.info(f"  Detected: {flagged_windows}/{total} windows flagged")
        return

    # Count how many windows flagged the injected rank
    correct = 0
    wrong = 0
    missed = 0
    total_with_inject_rank = 0

    for ws in all_window_stats.values():
        # Only count windows where the injected rank participates
        if inject_rank in ws.rank_stats:
            total_with_inject_rank += 1
            if inject_rank in ws.straggler_ranks:
                correct += 1
            elif ws.straggler_ranks:
                wrong += 1
            else:
                missed += 1

    logger.info(f"  Windows where rank {inject_rank} participates: {total_with_inject_rank}")
    logger.info(f"  Correctly flagged rank {inject_rank}: {correct}/{total_with_inject_rank}")
    logger.info(f"  Missed (no straggler flagged): {missed}/{total_with_inject_rank}")
    logger.info(f"  Wrong rank flagged: {wrong}/{total_with_inject_rank}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(trace_dir: str, verbose: bool = False, pg_filter: Optional[str] = None,
            k: float = 3.0, floor_ms: float = 20.0):
    """Run the full analysis pipeline."""

    # 1. Load
    collectives_by_file, pg_configs, pg_status = load_trace_dir(trace_dir)

    # 2. Window (two-pass: row compaction → column ordering)
    # Returns both grouped windows and their global ordering 
    windows, collectives_to_order = group_collectives_by_windows(collectives_by_file)
    logger.info(f"\nWindowing produced {len(windows)} unique window groups")

    # 3. Per-window stats + straggler identification
    logger.info(f"\nComputing per-window statistics (k={k}, floor_ms={floor_ms})...")
    all_window_stats: Dict[Tuple[str, str, int], WindowStats] = {}
    for key, colls in windows.items():
        all_window_stats[key] = compute_window_stats(key, colls, k=k, floor_ms=floor_ms)
    apply_p2p_duration_detection(all_window_stats, windows, k=k, floor_ms=floor_ms)
    straggler_count = sum(1 for ws in all_window_stats.values() if ws.straggler_ranks)
    logger.info(f"  {len(all_window_stats)} windows analyzed, {straggler_count} with stragglers")

    # 4. Reset-bounded segmentation, then build cascade graph per segment
    world_ranks = set(collectives_by_file.keys())
    segments = partition_reset_bounded_segments(all_window_stats, collectives_to_order, world_ranks)
    reset_count = sum(1 for s in segments if s.terminal_reset is not None)
    logger.info(
        f"\nReset-bounded segmentation: {len(segments)} segments "
        f"({reset_count} reset windows, world_ranks={sorted(world_ranks, key=int)})"
    )

    segment_cascades: List[Tuple[Segment, Dict[WindowKey, WindowStats], Dict[WindowKey, int], CascadeResult]] = []
    for segment in segments:
        segment_stats, segment_order = filter_maps_to_segment(
            all_window_stats, collectives_to_order, segment,
        )
        cascade = build_cascade_graph(segment_stats, segment_order)
        segment_cascades.append((segment, segment_stats, segment_order, cascade))

        if segment.terminal_reset is None:
            logger.info(
                f"  segment {segment.index}: {segment.segment_type} / lower confidence, "
                f"{len(segment.keys)} windows, {len(cascade.heads)} HEADs"
            )
        else:
            reset_gidx = collectives_to_order.get(segment.terminal_reset, -1)
            _, reset_pg_desc, reset_window_idx = segment.terminal_reset
            logger.info(
                f"  segment {segment.index}: {segment.segment_type}, "
                f"{len(segment.keys)} windows, {len(cascade.heads)} HEADs, "
                f"terminal reset=[{reset_gidx}] {reset_pg_desc} win={reset_window_idx}"
            )

    # 5. Output
    if verbose:
        print_detailed(all_window_stats, collectives_to_order, pg_filter)

    print_segment_slices(segments, all_window_stats, collectives_to_order)
    print_segment_cascade_heads(segment_cascades)
    print_combined_segmented_summary(all_window_stats, collectives_to_order, segments, pg_filter)

    # 6. Ground truth
    ground_truth = load_ground_truth(trace_dir)
    print_ground_truth_comparison(all_window_stats, ground_truth)


def analyze_quiet(trace_dir: str, k: float, floor_ms: float) -> dict:
    """Run analysis without logging, return summary stats for grid sweep."""
    collectives_by_file, pg_configs, pg_status = load_trace_dir(trace_dir)
    windows, collectives_to_order = group_collectives_by_windows(collectives_by_file)

    all_window_stats: Dict[Tuple[str, str, int], WindowStats] = {}
    for key, colls in windows.items():
        all_window_stats[key] = compute_window_stats(key, colls, k=k, floor_ms=floor_ms)
    apply_p2p_duration_detection(all_window_stats, windows, k=k, floor_ms=floor_ms)

    cascade = build_cascade_graph(all_window_stats, collectives_to_order)

    ground_truth = load_ground_truth(trace_dir)
    inject_type = ground_truth.get("inject_type", "none")
    inject_rank = ground_truth.get("inject_rank")

    straggler_count = sum(1 for ws in all_window_stats.values() if ws.straggler_ranks)
    head_straggler_ranks = set()
    for head_key in cascade.heads:
        ws = all_window_stats.get(head_key)
        if ws:
            head_straggler_ranks.update(ws.straggler_ranks)

    hit = inject_rank is not None and str(inject_rank) in head_straggler_ranks

    return {
        "inject_type": inject_type,
        "inject_rank": inject_rank,
        "n_windows": len(windows),
        "n_straggler_windows": straggler_count,
        "n_heads": len(cascade.heads),
        "head_straggler_ranks": head_straggler_ranks,
        "hit": hit,
    }


def grid_sweep(trace_dirs: List[str]):
    """Run grid sweep over k and floor_ms values."""
    k_values = [1.5, 2.0, 3.0]
    floor_values = [5.0, 10.0, 20.0]

    print("\n" + "=" * 80)
    print("GRID SWEEP: Single-worst-outlier with gap-to-runner-up")
    print("=" * 80)

    # Categorize traces
    baselines = []
    injections = []
    for td in trace_dirs:
        gt = load_ground_truth(td)
        if gt.get("inject_type") == "none":
            baselines.append(td)
        else:
            injections.append((td, gt))

    print(f"\nTraces: {len(baselines)} baseline, {len(injections)} injection")
    for td, gt in injections:
        print(f"  - {os.path.basename(td)}: {gt.get('inject_type')} rank {gt.get('inject_rank')} @ {gt.get('inject_delay_ms')}ms")

    # Run grid
    results = []
    for k in k_values:
        for floor_ms in floor_values:
            row = {"k": k, "floor_ms": floor_ms}

            # Baseline HEADs
            baseline_heads = []
            for td in baselines:
                r = analyze_quiet(td, k, floor_ms)
                baseline_heads.append(r["n_heads"])
            row["baseline_heads"] = baseline_heads

            # Injection hits
            injection_hits = []
            for td, gt in injections:
                r = analyze_quiet(td, k, floor_ms)
                injection_hits.append({
                    "name": os.path.basename(td),
                    "inject_rank": gt.get("inject_rank"),
                    "hit": r["hit"],
                    "n_heads": r["n_heads"],
                    "head_ranks": r["head_straggler_ranks"],
                })
            row["injection_hits"] = injection_hits

            results.append(row)

    # Print results table
    print("\n" + "-" * 80)
    print(f"{'k':>4} | {'floor':>5} | {'baseline #HEADs':>15} | injection HEADs contain injected rank?")
    print("-" * 80)

    passing_cells = []
    for row in results:
        k, floor_ms = row["k"], row["floor_ms"]
        bh = row["baseline_heads"]
        baseline_str = ",".join(str(h) for h in bh) if len(bh) <= 3 else f"max={max(bh)}"

        hits = row["injection_hits"]
        hit_strs = []
        all_hit = True
        for h in hits:
            mark = "✓" if h["hit"] else "✗"
            hit_strs.append(f"r{h['inject_rank']}:{mark}")
            if not h["hit"]:
                all_hit = False

        # Pass criteria: baseline ≤1 HEAD each, all injections hit
        baseline_pass = all(h <= 1 for h in bh)
        if baseline_pass and all_hit:
            passing_cells.append((k, floor_ms))

        status = "PASS" if (baseline_pass and all_hit) else ""
        print(f"{k:>4.1f} | {floor_ms:>5.0f} | {baseline_str:>15} | {' '.join(hit_strs)} {status}")

    print("-" * 80)

    if passing_cells:
        # Pick most conservative (largest k, then largest floor_ms)
        best = max(passing_cells, key=lambda x: (x[0], x[1]))
        print(f"\n✓ BEST CELL: k={best[0]}, floor_ms={best[1]} (most conservative passing)")
    else:
        print("\n✗ NO PASSING CELL — review results above")

    return results


def main():
    parser = argparse.ArgumentParser(description="FR Straggler Analyzer")
    parser.add_argument("trace_dir", nargs="?", help="Path to trace directory containing _dump_*.json files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed per-rank per-window output")
    parser.add_argument("--pg", default=None, help="Filter output to PGs matching this substring")
    parser.add_argument("--k", type=float, default=3.0,
                        help="Stdev multiplier for gap threshold (default: 3.0)")
    parser.add_argument("--floor-ms", type=float, default=20.0,
                        help="Minimum gap threshold in ms (default: 20.0)")
    parser.add_argument("--grid-sweep", nargs="+", metavar="TRACE_DIR",
                        help="Run grid sweep on multiple trace directories")
    args = parser.parse_args()

    if args.grid_sweep:
        grid_sweep(args.grid_sweep)
        return

    if not args.trace_dir:
        parser.error("trace_dir is required unless --grid-sweep is specified")

    # Auto-save output to {trace_dir}/analysis.log alongside run_config.log
    log_path = os.path.join(args.trace_dir, "analysis.log")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Auto-saving output to {log_path}")

    analyze(args.trace_dir, verbose=args.verbose, pg_filter=args.pg, k=args.k, floor_ms=args.floor_ms)


if __name__ == "__main__":
    main()
