"""
FR Straggler Analyzer — standalone offline analysis of FR traces from straggler injection experiments.

Designed to consume one default_pg-bracketed BLOCK at a time (produced by the
monitor's block slicer). Within a block, the terminal default_pg is the natural
anchor — host/kernel straggler signals propagate to it, so we pick up flagged
ranks there and walk backward through earlier flagged-for-this-rank windows
until we hit a strict transition (next earlier rank-containing window not
flagged for that rank). That window is the origin.

Data flow:
  Load trace dir → Parse entries (completed only) → Build collectives_by_file
    → group_collectives_by_windows()
    → Per-window per-rank stats (host = late time_created, kernel = short duration_ms)
    → find_anchor(): terminal default_pg in scope, its flagged ranks
    → backward_walk(): per-rank backward walk to first-appearance window
    → Origins sorted by gidx
    → Print results + ground truth comparison

Usage:
  python fr_straggler_analyzer.py /path/to/block_dir
  python fr_straggler_analyzer.py /path/to/block_dir -v
  python fr_straggler_analyzer.py /path/to/block_dir --pg TENSOR
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
    duration_ms: Optional[float]    # true kernel duration via cudaEventElapsedTime
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
            collectives.append(Collective(
                record_id=entry.get("record_id", -1),
                file_id=rank_id,
                collective_seq_id=entry["collective_seq_id"],
                p2p_seq_id=entry.get("p2p_seq_id", -1),
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
                is_p2p=entry.get("is_p2p", False),
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
            is_p2p = entry.p2p_seq_id > 0

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
                    next_is_p2p = next_entry.p2p_seq_id > 0
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

    # Step 2: For each group, compute min timestamp and pg_desc
    group_info: List[Tuple[GroupKey, int, str, List[Window]]] = []
    for key, window_list in groups.items():
        min_ts = min(w.min_time_created_ns for w in window_list)
        # pg_desc should be consistent across windows in the group
        pg_desc = window_list[0].pg_desc
        group_info.append((key, min_ts, pg_desc, window_list))

    # Step 3: Sort by min timestamp
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

@dataclass
class RankWindowStats:
    """Timing stats for one rank within one window."""
    rank_id: str
    n_entries: int
    mean_created_offset_ms: float   # mean(time_created - window_min_created) in ms
    mean_started_offset_ms: float   # mean(time_started - window_min_started) in ms
    mean_gpu_duration_us: float     # mean duration_ms * 1000 (cudaEventElapsedTime)


@dataclass
class WindowStats:
    """Stats for one window across all participating ranks."""
    key: Tuple[str, str, int]  # (megatron_id, pg_desc, window_idx)
    rank_stats: Dict[str, RankWindowStats]
    median_created_offset_ms: float
    median_started_offset_ms: float
    median_gpu_duration_us: float
    straggler_ranks: Set[str]  # ranks flagged as stragglers
    straggler_signal: str  # which signal flagged them


def compute_window_stats(
    window_key: Tuple[str, str, int],
    collectives: List[Collective],
    k: float = 3.0,
    floor_ms: float = 20.0,
) -> WindowStats:
    """
    Compute per-rank timing stats for a single window and identify stragglers.

    Two independent detection paths, both can fire on the same window:
      1. Host-side: gap-to-runner-up on time_created_ns. Catches host
         injections (sleep on the rank's CPU) where the rank enqueues late.
      2. Kernel-side: median+MAD deficit on duration_ms. Catches kernel
         injections. 

    NOTE: kernel-side detection uses within-window median+MAD, NOT a
    per-rank baseline. This is intentional — duration_ms on a waiting rank
    INFLATES during cascade while a rank's own duration may DROP relative
    to its baseline. Per-rank baseline would invert the signal and flag
    victims as roots.
    """
    # Group collectives by rank
    by_rank: Dict[str, List[Collective]] = defaultdict(list)
    for c in collectives:
        by_rank[c.file_id].append(c)

    if not by_rank:
        return WindowStats(
            key=window_key, rank_stats={},
            median_created_offset_ms=0, median_started_offset_ms=0,
            median_gpu_duration_us=0, straggler_ranks=set(), straggler_signal="",
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

        # gpu_duration — use duration_ms field (cudaEventElapsedTime), converted to µs.
        # Do NOT compute from time_discovered_completed_ns - time_discovered_started_ns;
        # those are watchdog poll timestamps, not kernel boundaries (see fr_concepts.md §14).
        gpu_durs = []
        for c in rank_colls:
            if c.duration_ms is not None:
                gpu_durs.append(c.duration_ms * 1000.0)   # ms -> µs for display compatibility
        mean_gpu_dur = sum(gpu_durs) / len(gpu_durs) if gpu_durs else 0.0

        rank_stats[rank_id] = RankWindowStats(
            rank_id=rank_id,
            n_entries=len(rank_colls),
            mean_created_offset_ms=mean_created,
            mean_started_offset_ms=mean_started,
            mean_gpu_duration_us=mean_gpu_dur,
        )

    # Compute medians across ranks (kept for reporting, not used in flagging)
    created_values = sorted(rs.mean_created_offset_ms for rs in rank_stats.values())
    started_values = sorted(rs.mean_started_offset_ms for rs in rank_stats.values())
    gpu_dur_values = sorted(rs.mean_gpu_duration_us for rs in rank_stats.values())

    def median(vals):
        n = len(vals)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2

    med_created = median(created_values)
    med_started = median(started_values)
    med_gpu_dur = median(gpu_dur_values)

    # --- Host-side detection on time_created_ns ---
    # Two regimes by window size:
    #   n >= 3: median + MAD. Flags every rank with excess above threshold.
    #     Robust against >1 stragglers per window and handles cascade
    #     situations where multiple ranks are off-median.
    #   n == 2: MAD is degenerate (MAD = range/2, so k*MAD always > excess
    #     for k>1 → never fires). Fall back to gap-to-runner-up vs floor.
    #     With n=2 the "gap" is just |a-b|, which we compare to floor_ms.
    straggler_ranks = set()
    straggler_signal = ""
    n = len(rank_stats)

    if n >= 3:
        created_offsets = [rs.mean_created_offset_ms for rs in rank_stats.values()]
        created_median = median(sorted(created_offsets))
        created_mad = median(sorted(abs(v - created_median) for v in created_offsets))
        mad_scale = 1.4826 * created_mad if created_mad > 0 else 0.0

        threshold = max(floor_ms, k * mad_scale)
        host_flags = []
        for rank_id, rs in rank_stats.items():
            excess = rs.mean_created_offset_ms - created_median
            if excess > threshold:
                straggler_ranks.add(rank_id)
                host_flags.append(f"r{rank_id}={rs.mean_created_offset_ms:.1f}ms (excess={excess:.1f}ms)")
        if host_flags:
            straggler_signal = (f"time_created median={created_median:.1f}ms "
                                f"thresh={threshold:.1f}ms (MAD={created_mad:.2f}): "
                                + ", ".join(host_flags))

    elif n == 2:
        sorted_by_created = sorted(
            rank_stats.items(),
            key=lambda x: x[1].mean_created_offset_ms,
            reverse=True,
        )
        worst_rank, worst_rs = sorted_by_created[0]
        gap = worst_rs.mean_created_offset_ms - sorted_by_created[1][1].mean_created_offset_ms
        if gap > floor_ms:
            straggler_ranks.add(worst_rank)
            straggler_signal = f"time_created gap={gap:.1f}ms (thresh={floor_ms:.1f}ms, n=2 floor-only)"

    # --- Kernel-side detection on duration_ms (cudaEventElapsedTime) ---
    # Within-window median+MAD comparison. Flags ranks whose duration is
    # abnormally SHORT vs the window's median — kernel straggler arrived
    # late and its peers' kernels then ran for less wall-clock time
    # because they finished and waited. The "late arriver" itself can have
    # an inflated duration; what we detect here is its peers' DEFICIT
    # against the median. Either way the window gets flagged, which is
    # what matters for PG-level origin identification.
    #
    # NOTE: ignores `floor_ms` (that's ms-scale for time_created; kernels
    # can be µs). Uses scale-aware floor below. Constants 50µs / 25% are
    # tuned empirically on TP=2 8-rank traces; may need re-tuning on
    # very large or very small collectives.
    gpu_means_us = [rs.mean_gpu_duration_us for rs in rank_stats.values()
                    if rs.mean_gpu_duration_us > 0]
    if len(gpu_means_us) >= 2:
        gpu_median_us = median(sorted(gpu_means_us))
        gpu_mad_us = median(sorted(abs(v - gpu_median_us) for v in gpu_means_us))
        # 1.4826 * MAD ≈ stdev for normally-distributed data.
        mad_scale_us = 1.4826 * gpu_mad_us if gpu_mad_us > 0 else 0.0
        # Scale-aware floor: absolute noise floor (50µs) + relative floor
        # (25% of window median). Absolute prevents flagging pure µs-scale
        # jitter; relative scales with collective size so the same
        # detector works on 100µs TP kernels and 100ms DP allreduces.
        absolute_noise_floor_us = 50.0
        relative_floor_us = 0.25 * gpu_median_us
        gpu_floor_us = max(absolute_noise_floor_us, relative_floor_us)
        for rank_id, rs in rank_stats.items():
            if rs.mean_gpu_duration_us <= 0:
                continue
            deficit_us = gpu_median_us - rs.mean_gpu_duration_us
            if mad_scale_us > 0:
                z_ok = deficit_us > k * mad_scale_us
            else:
                z_ok = True   # MAD=0 — trust the floor
            if z_ok and deficit_us > gpu_floor_us:
                straggler_ranks.add(rank_id)
                if mad_scale_us > 0:
                    extra = (f"gpu_dur={rs.mean_gpu_duration_us:.0f}µs vs "
                             f"median={gpu_median_us:.0f}µs "
                             f"(deficit={deficit_us:.0f}µs, "
                             f"{deficit_us / mad_scale_us:.1f}σ, "
                             f"floor={gpu_floor_us:.0f}µs)")
                else:
                    extra = (f"gpu_dur={rs.mean_gpu_duration_us:.0f}µs vs "
                             f"median={gpu_median_us:.0f}µs "
                             f"(deficit={deficit_us:.0f}µs, MAD=0, "
                             f"floor={gpu_floor_us:.0f}µs)")
                straggler_signal = (straggler_signal + "; " + extra) if straggler_signal else extra

    return WindowStats(
        key=window_key,
        rank_stats=rank_stats,
        median_created_offset_ms=med_created,
        median_started_offset_ms=med_started,
        median_gpu_duration_us=med_gpu_dur,
        straggler_ranks=straggler_ranks,
        straggler_signal=straggler_signal,
    )

WindowKey = Tuple[str, str, int]

# ---------------------------------------------------------------------------
# 5. Backward-walk attribution
#
# Per the block-based design: the terminal default_pg is the propagation sink
# where straggler signals converge. We pick its flagged ranks as the anchor
# and walk backward — for each flagged rank, step to the most recent earlier
# window where that rank participated and ALSO is flagged. Stop on strict
# transition (next earlier rank-containing window has rank NOT flagged): that
# transition marks where the rank's straggler signal first appeared in this
# block. The current window at that moment is the origin.
#
# Independent walks per flagged rank (no fusion). Co-fault traces benefit
# from this: each flagged rank's chain stands on its own.
# ---------------------------------------------------------------------------


@dataclass
class Origin:
    """Result of one rank's backward walk."""
    rank: str
    window_key: WindowKey
    walk_depth: int   # number of backward steps taken (0 = origin is anchor)


def find_anchor(
    window_stats: Dict[WindowKey, WindowStats],
    collectives_to_order: Dict[WindowKey, int],
) -> Optional[WindowKey]:
    """Return the highest-gidx default_pg window in scope, or None."""
    default_keys = [k for k in window_stats if k[1] == "default_pg"]
    if not default_keys:
        return None
    return max(default_keys, key=lambda k: collectives_to_order.get(k, 0))


def backward_walk(
    rank: str,
    current: WindowKey,
    window_stats: Dict[WindowKey, WindowStats],
    collectives_to_order: Dict[WindowKey, int],
) -> Tuple[WindowKey, int]:
    """
    Walk backward from `current` while `rank` stays flagged.

    Termination: strict transition. The most recent earlier window where
    `rank` participated decides — if `rank` is flagged there, step back; if
    not, return `current` (signal turned ON between that earlier window and
    `current`). If no such earlier window exists, return `current` (edge of
    scope).

    Returns (origin_window_key, walk_depth).
    """
    depth = 0
    while True:
        cur_gidx = collectives_to_order.get(current, 0)
        prev_key: Optional[WindowKey] = None
        prev_gidx = -1
        for k, ws in window_stats.items():
            if rank not in ws.rank_stats:
                continue
            g = collectives_to_order.get(k, 0)
            if g >= cur_gidx:
                continue
            if g > prev_gidx:
                prev_gidx = g
                prev_key = k
        if prev_key is None:
            return current, depth
        if rank not in window_stats[prev_key].straggler_ranks:
            return current, depth
        current = prev_key
        depth += 1


def find_origins(
    window_stats: Dict[WindowKey, WindowStats],
    collectives_to_order: Dict[WindowKey, int],
) -> List[Origin]:
    """
    For each rank flagged at the anchor, run a backward walk. Return all
    origins sorted by gidx ascending. Independent chains are preserved.
    """
    anchor = find_anchor(window_stats, collectives_to_order)
    if anchor is None:
        return []
    anchor_ws = window_stats[anchor]
    if not anchor_ws.straggler_ranks:
        return []
    origins: List[Origin] = []
    for rank in anchor_ws.straggler_ranks:
        origin_key, depth = backward_walk(rank, anchor, window_stats, collectives_to_order)
        origins.append(Origin(rank=rank, window_key=origin_key, walk_depth=depth))
    origins.sort(key=lambda o: collectives_to_order.get(o.window_key, 0))
    return origins


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
    origins: List[Origin],
    pg_filter: Optional[str] = None,
):
    """Print summary table of flagged windows, plus origins from backward walks."""
    logger.info("\n=== Straggler Analysis Summary ===\n")

    logger.info(
        f"{'GIdx':>4} | {'PG Desc':<40} | {'Win':>3} | {'Ranks':>20} | {'Straggler':>8} | {'Signal':<40}"
    )
    logger.info("-" * 130)

    straggler_window_count = 0
    total_windows = 0

    for key in sorted(all_window_stats.keys(), key=lambda k: collectives_to_order.get(k, 0)):
        ws = all_window_stats[key]
        _, pg_desc, window_idx = key

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

    if origins:
        logger.info("\n=== Origins (backward walks from terminal default_pg) ===\n")
        for o in origins:
            _, pg_desc, win_idx = o.window_key
            gidx = collectives_to_order.get(o.window_key, -1)
            logger.info(
                f"  rank {o.rank}: origin = [{gidx}] {pg_desc} win={win_idx} "
                f"(walked back {o.walk_depth} step{'s' if o.walk_depth != 1 else ''})"
            )
    else:
        logger.info("\nNo origins found (anchor has no flagged ranks).")


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
            f"  {'Rank':>6} | {'N':>4} | {'created_offset':>16} | {'started_offset':>16} | {'gpu_dur':>12} |"
        )

        for rank_id in sorted(ws.rank_stats.keys(), key=int):
            rs = ws.rank_stats[rank_id]
            marker = " <-- STRAGGLER" if rank_id in ws.straggler_ranks else ""
            logger.info(
                f"  {rank_id:>6} | {rs.n_entries:>4} | "
                f"{rs.mean_created_offset_ms:>13.3f}ms | "
                f"{rs.mean_started_offset_ms:>13.3f}ms | "
                f"{rs.mean_gpu_duration_us:>9.1f}us |{marker}"
            )

        logger.info(
            f"  {'median':>6} | {'':>4} | "
            f"{ws.median_created_offset_ms:>13.3f}ms | "
            f"{ws.median_started_offset_ms:>13.3f}ms | "
            f"{ws.median_gpu_duration_us:>9.1f}us |"
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

    collectives_by_file, _pg_configs, _pg_status = load_trace_dir(trace_dir)
    windows, collectives_to_order = group_collectives_by_windows(collectives_by_file)
    logger.info(f"\nWindowing produced {len(windows)} unique window groups:")
    for key in sorted(windows.keys(), key=lambda k: collectives_to_order.get(k, 0)):
        megatron_id, pg_desc, window_idx = key
        ranks = set(c.file_id for c in windows[key])
        gidx = collectives_to_order.get(key, -1)
        logger.info(f"  [gidx={gidx}] ({megatron_id}, {pg_desc}, win={window_idx}): {len(windows[key])} entries, ranks={sorted(ranks, key=int)}")

    logger.info(f"\nComputing per-window statistics (k={k}, floor_ms={floor_ms})...")
    all_window_stats: Dict[Tuple[str, str, int], WindowStats] = {}
    for key, colls in windows.items():
        all_window_stats[key] = compute_window_stats(key, colls, k=k, floor_ms=floor_ms)
    straggler_count = sum(1 for ws in all_window_stats.values() if ws.straggler_ranks)
    logger.info(f"  {len(all_window_stats)} windows analyzed, {straggler_count} with stragglers")

    logger.info("\nFinding origins via backward walk from terminal default_pg...")
    origins = find_origins(all_window_stats, collectives_to_order)
    logger.info(f"  {len(origins)} origin(s) found")

    if verbose:
        print_detailed(all_window_stats, collectives_to_order, pg_filter)

    print_summary(all_window_stats, collectives_to_order, origins, pg_filter)

    ground_truth = load_ground_truth(trace_dir)
    print_ground_truth_comparison(all_window_stats, ground_truth)


def analyze_quiet(trace_dir: str, k: float, floor_ms: float) -> dict:
    """Run analysis without logging, return summary stats. Used by the monitor."""
    collectives_by_file, _pg_configs, _pg_status = load_trace_dir(trace_dir)
    windows, collectives_to_order = group_collectives_by_windows(collectives_by_file)

    all_window_stats: Dict[Tuple[str, str, int], WindowStats] = {}
    for key, colls in windows.items():
        all_window_stats[key] = compute_window_stats(key, colls, k=k, floor_ms=floor_ms)

    origins = find_origins(all_window_stats, collectives_to_order)

    ground_truth = load_ground_truth(trace_dir) or {}
    inject_type = ground_truth.get("inject_type", "none")
    inject_rank = ground_truth.get("inject_rank")

    straggler_count = sum(1 for ws in all_window_stats.values() if ws.straggler_ranks)
    origin_ranks: Set[str] = {o.rank for o in origins}
    origin_payload = [
        {
            "rank": o.rank,
            "pg_desc": o.window_key[1],
            "window_idx": o.window_key[2],
            "gidx": collectives_to_order.get(o.window_key, -1),
            "walk_depth": o.walk_depth,
        }
        for o in origins
    ]

    hit = inject_rank is not None and str(inject_rank) in origin_ranks

    return {
        "inject_type": inject_type,
        "inject_rank": inject_rank,
        "n_windows": len(windows),
        "n_straggler_windows": straggler_count,
        "n_origins": len(origins),
        # Backwards-compat field name for monitor verdict consumers — same set,
        # now sourced from origins instead of cascade-graph HEADs.
        "head_straggler_ranks": origin_ranks,
        "n_heads": len(origins),
        "origins": origin_payload,
        "hit": hit,
    }


def main():
    parser = argparse.ArgumentParser(description="FR Straggler Analyzer (block-based)")
    parser.add_argument("trace_dir", help="Path to a block directory containing _dump_*.json files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed per-rank per-window output")
    parser.add_argument("--pg", default=None, help="Filter output to PGs matching this substring")
    parser.add_argument("--k", type=float, default=3.0,
                        help="Stdev multiplier for gap threshold (default: 3.0)")
    parser.add_argument("--floor-ms", type=float, default=20.0,
                        help="Minimum gap threshold in ms (default: 20.0)")
    args = parser.parse_args()

    log_path = os.path.join(args.trace_dir, "analysis.log")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Auto-saving output to {log_path}")

    analyze(args.trace_dir, verbose=args.verbose, pg_filter=args.pg, k=args.k, floor_ms=args.floor_ms)


if __name__ == "__main__":
    main()