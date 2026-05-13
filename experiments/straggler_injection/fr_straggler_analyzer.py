"""
FR Straggler Analyzer — standalone offline analysis of FR traces from straggler injection experiments.

Operates on a single block dir produced by the upstream block-slicer
(Block N = entries between default_pg terminal N-1 and default_pg terminal N).

Data flow:
  Load trace dir → Parse entries (completed only) → Build collectives_by_file
    → group_collectives_by_windows()
    → Per-window per-rank stats (host time_created + kernel duration_ms)
    → Within-window flagging (compute_window_stats):
        host: MAD on per-rank means for n≥3; gap-to-runner-up for n=2
        kernel: MAD-deficit on duration_ms with scale-aware floor for n≥3;
                gap-to-runner-up on duration for n=2
    → P2P duration MAD detection (apply_p2p_duration_detection) — augments
    → Build cascade DAG over flagged windows (rank-overlap edges, gidx-directed)
    → HEADs = roots; render forward chains
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
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
    process_group: List[str]  # [megatron_id, pg_desc]
    input_sizes: List[List[int]]
    output_sizes: List[List[int]]
    input_dtypes: List[str]
    output_dtypes: List[str]
    is_p2p: bool
    duration_ms: Optional[float] = None  # watchdog-populated kernel duration


# ---------------------------------------------------------------------------
# 1. Load trace directory
# ---------------------------------------------------------------------------

def load_trace_dir(trace_dir: str) -> Tuple[Dict[str, List[Collective]], Dict[str, dict], Dict[str, dict]]:
    """
    Load all rank dumps from a trace directory (or single block dir).

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
                process_group=entry["process_group"],
                input_sizes=entry.get("input_sizes", []),
                output_sizes=entry.get("output_sizes", []),
                input_dtypes=entry.get("input_dtypes", []),
                output_dtypes=entry.get("output_dtypes", []),
                is_p2p=entry.get("is_p2p", False),
                duration_ms=entry.get("duration_ms"),
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
        return set(e.file_id for e in self.entries)


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
    """
    windows_by_rank: Dict[str, List[Window]] = {}

    for rank_id, entries in collectives_by_file.items():
        windows: List[Window] = []
        pg_occurrence_count: Dict[str, int] = defaultdict(int)

        i = 0
        while i < len(entries):
            entry = entries[i]
            megatron_id = entry.process_group[0]
            pg_desc = entry.process_group[1]
            is_p2p = entry.p2p_seq_id > 0

            if is_p2p:
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
    """
    GroupKey = Tuple[str, int]  # (megatron_id, window_id)
    groups: Dict[GroupKey, List[Window]] = defaultdict(list)

    for rank_id, windows in windows_by_rank.items():
        for w in windows:
            key = (w.megatron_id, w.window_id)
            groups[key].append(w)

    group_info: List[Tuple[GroupKey, int, str, List[Window]]] = []
    for key, window_list in groups.items():
        min_ts = min(w.min_time_created_ns for w in window_list)
        pg_desc = window_list[0].pg_desc
        group_info.append((key, min_ts, pg_desc, window_list))

    group_info.sort(key=lambda x: x[1])

    grouped_windows: Dict[Tuple[str, str, int], List[Collective]] = {}
    collectives_to_order: Dict[Tuple[str, str, int], int] = {}

    for global_idx, (key, min_ts, pg_desc, window_list) in enumerate(group_info):
        megatron_id, window_id = key
        output_key = (megatron_id, pg_desc, window_id)

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
    """
    windows_by_rank = _assign_windows_per_rank(collectives_by_file)

    total_windows = sum(len(ws) for ws in windows_by_rank.values())
    logger.info(f"\nPass 1 (row compaction): {total_windows} windows across {len(windows_by_rank)} ranks")
    for rank_id in sorted(windows_by_rank.keys(), key=int):
        windows = windows_by_rank[rank_id]
        collective_wins = sum(1 for w in windows if not w.is_p2p)
        p2p_wins = sum(1 for w in windows if w.is_p2p)
        logger.debug(f"  rank {rank_id}: {len(windows)} windows ({collective_wins} collective, {p2p_wins} p2p)")

    grouped_windows, collectives_to_order = _order_windows_globally(windows_by_rank)

    logger.info(f"Pass 2 (column ordering): {len(grouped_windows)} unique window groups")

    return grouped_windows, collectives_to_order


# ---------------------------------------------------------------------------
# 3. Per-window statistics + MAD-based straggler flagging
# ---------------------------------------------------------------------------

@dataclass
class RankWindowStats:
    """Timing stats for one rank within one window."""
    rank_id: str
    n_entries: int
    mean_created_offset_ms: float
    mean_started_offset_ms: float
    mean_gpu_duration_us: float
    mean_duration_ms: Optional[float] = None  # mean of populated duration_ms (watchdog)


@dataclass
class WindowStats:
    """Stats for one window across all participating ranks."""
    key: Tuple[str, str, int]  # (megatron_id, pg_desc, window_idx)
    rank_stats: Dict[str, RankWindowStats]
    median_created_offset_ms: float
    median_started_offset_ms: float
    median_gpu_duration_us: float
    straggler_ranks: Set[str] = field(default_factory=set)
    straggler_signals: List[str] = field(default_factory=list)  # one string per detector that fired
    is_p2p: bool = False
    megatron_id: str = ""
    pg_desc: str = ""
    # Per-window skew stats (host-side, time_created). Used by within-window
    # flagging and the sibling-PG baseline second pass.
    #   gap_to_runner_up_ms: max - second-max of per-rank mean_created_offset_ms.
    #   late_rank: the rank with the largest mean_created_offset.
    gap_to_runner_up_ms: Optional[float] = None
    late_rank: Optional[str] = None

    @property
    def straggler_signal(self) -> str:
        """Joined signal string for output."""
        return " | ".join(self.straggler_signals) if self.straggler_signals else ""


def _median(vals: List[float]) -> float:
    n = len(vals)
    if n == 0:
        return 0.0
    s = sorted(vals)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _mad(vals: List[float], median: float) -> float:
    """Median absolute deviation."""
    if not vals:
        return 0.0
    deviations = [abs(v - median) for v in vals]
    return _median(deviations)


# Kernel-side detector: scale-aware floor for the duration_ms deficit.
#   Absolute floor (50µs) blocks pure µs-scale noise from flagging.
#   Relative floor (25% of window median duration) scales naturally with
#   collective size so the same detector works for fast TP kernels and
#   long DP allreduces.
KERNEL_FLOOR_ABS_US = 50.0
KERNEL_FLOOR_REL = 0.25


def compute_window_stats(
    window_key: Tuple[str, str, int],
    collectives: List[Collective],
    k: float = 3.0,
    floor_ms: float = 20.0,
) -> WindowStats:
    """
    Compute per-rank timing stats for a single window and identify stragglers via MAD.

    MAD rule: a rank is flagged if its time_created excess above the median exceeds
              max(floor_ms, k * MAD * 1.4826).
    """
    by_rank: Dict[str, List[Collective]] = defaultdict(list)
    for c in collectives:
        by_rank[c.file_id].append(c)

    megatron_id, pg_desc, window_idx = window_key
    is_p2p = any(c.p2p_seq_id > 0 for c in collectives)

    if not by_rank:
        return WindowStats(
            key=window_key, rank_stats={},
            median_created_offset_ms=0, median_started_offset_ms=0,
            median_gpu_duration_us=0,
            is_p2p=is_p2p, megatron_id=megatron_id, pg_desc=pg_desc,
        )

    # Per-rank means
    rank_stats: Dict[str, RankWindowStats] = {}

    all_created = [c.time_created_ns for c in collectives]
    min_created = min(all_created)

    all_started = [c.time_discovered_started_ns for c in collectives
                   if c.time_discovered_started_ns is not None]
    min_started = min(all_started) if all_started else 0

    for rank_id, rank_colls in by_rank.items():
        created_offsets = [(c.time_created_ns - min_created) / 1e6 for c in rank_colls]
        mean_created = sum(created_offsets) / len(created_offsets)

        started_offsets = [
            (c.time_discovered_started_ns - min_started) / 1e6
            for c in rank_colls if c.time_discovered_started_ns is not None
        ]
        mean_started = sum(started_offsets) / len(started_offsets) if started_offsets else 0.0

        # Kernel duration: use `duration_ms` (true cudaEventElapsedTime).
        # Do NOT derive from time_discovered_completed_ns - time_discovered_started_ns
        # — those are watchdog poll timestamps, not kernel boundaries
        # (see fr_concepts.md §14). Convert to µs for the kernel detector.
        durs_ms = [c.duration_ms for c in rank_colls if c.duration_ms is not None]
        mean_dur_ms = (sum(durs_ms) / len(durs_ms)) if durs_ms else None
        mean_gpu_dur = (mean_dur_ms * 1000.0) if mean_dur_ms is not None else 0.0

        rank_stats[rank_id] = RankWindowStats(
            rank_id=rank_id,
            n_entries=len(rank_colls),
            mean_created_offset_ms=mean_created,
            mean_started_offset_ms=mean_started,
            mean_gpu_duration_us=mean_gpu_dur,
            mean_duration_ms=mean_dur_ms,
        )

    # Medians (for reporting)
    created_values = [rs.mean_created_offset_ms for rs in rank_stats.values()]
    started_values = [rs.mean_started_offset_ms for rs in rank_stats.values()]
    gpu_dur_values = [rs.mean_gpu_duration_us for rs in rank_stats.values()]

    med_created = _median(created_values)
    med_started = _median(started_values)
    med_gpu_dur = _median(gpu_dur_values)

    # --- Host-side detection on time_created ---
    #
    # For n≥3 ranks: median + MAD on per-rank means.
    # For n==2 ranks: MAD is degenerate (k*MAD*1.4826 always > excess for k>1
    #   so the detector never fires regardless of how large the gap is).
    #   Fall back to gap-to-runner-up vs floor — with n=2 the runner-up IS
    #   the only other peer, so the gap is just |mean_a - mean_b|.
    straggler_ranks: Set[str] = set()
    straggler_signals: List[str] = []
    gap_to_runner_up_ms: Optional[float] = None
    late_rank: Optional[str] = None
    n = len(rank_stats)

    if n >= 2:
        sorted_by_created = sorted(
            rank_stats.items(),
            key=lambda x: x[1].mean_created_offset_ms,
            reverse=True,
        )
        late_rank = sorted_by_created[0][0]
        gap_to_runner_up_ms = (
            sorted_by_created[0][1].mean_created_offset_ms
            - sorted_by_created[1][1].mean_created_offset_ms
        )

    if n >= 3:
        mad_created = _mad(created_values, med_created)
        threshold = max(floor_ms, k * mad_created * 1.4826)

        flagged_here: List[Tuple[str, float]] = []
        for rank_id, rs in rank_stats.items():
            excess = rs.mean_created_offset_ms - med_created
            if excess > threshold:
                straggler_ranks.add(rank_id)
                flagged_here.append((rank_id, excess))

        if flagged_here:
            flagged_here.sort(key=lambda x: -x[1])
            top = flagged_here[0]
            straggler_signals.append(
                f"tc MAD excess={top[1]:.1f}ms thr={threshold:.1f}ms "
                f"(k={k}, floor={floor_ms:.0f})"
            )
    elif n == 2 and gap_to_runner_up_ms is not None and gap_to_runner_up_ms > floor_ms:
        straggler_ranks.add(late_rank)
        straggler_signals.append(
            f"tc gap={gap_to_runner_up_ms:.1f}ms thr={floor_ms:.0f}ms "
            f"(n=2 runner-up, late={late_rank})"
        )

    # --- Kernel-side detection on duration_ms (cudaEventElapsedTime) ---
    #
    # In a synchronous collective, the late-starting rank's comm kernel runs
    # SHORTEST because all ranks finish together (see fr_concepts.md §10/§14).
    # We flag ranks whose mean duration is abnormally short relative to peers.
    #
    # For n≥3 ranks: median + MAD on per-rank mean durations, plus a
    #   scale-aware floor (absolute 50µs OR 25% of median, whichever is larger).
    # For n==2 ranks: MAD degenerate same as host-side; fall back to
    #   gap-to-runner-up on duration (the smaller-duration rank is the
    #   straggler if the gap exceeds the scale-aware floor).
    gpu_means_us = [rs.mean_gpu_duration_us for rs in rank_stats.values()
                    if rs.mean_gpu_duration_us > 0]
    if len(gpu_means_us) >= 3:
        gpu_median_us = _median(gpu_means_us)
        gpu_mad_us = _mad(gpu_means_us, gpu_median_us)
        mad_scale_us = 1.4826 * gpu_mad_us if gpu_mad_us > 0 else 0.0
        gpu_floor_us = max(KERNEL_FLOOR_ABS_US, KERNEL_FLOOR_REL * gpu_median_us)

        kernel_flags: List[Tuple[str, float]] = []
        for rank_id, rs in rank_stats.items():
            if rs.mean_gpu_duration_us <= 0:
                continue
            deficit_us = gpu_median_us - rs.mean_gpu_duration_us
            z_ok = (deficit_us > k * mad_scale_us) if mad_scale_us > 0 else True
            if z_ok and deficit_us > gpu_floor_us:
                straggler_ranks.add(rank_id)
                kernel_flags.append((rank_id, deficit_us))

        if kernel_flags:
            kernel_flags.sort(key=lambda x: -x[1])
            top = kernel_flags[0]
            straggler_signals.append(
                f"dur deficit={top[1]:.0f}us median={gpu_median_us:.0f}us "
                f"floor={gpu_floor_us:.0f}us (k={k}, late={top[0]})"
            )
    elif len(gpu_means_us) == 2:
        dur_pairs = [
            (rid, rs.mean_gpu_duration_us) for rid, rs in rank_stats.items()
            if rs.mean_gpu_duration_us > 0
        ]
        dur_pairs.sort(key=lambda x: x[1])
        short_rank, short_dur_us = dur_pairs[0]
        long_dur_us = dur_pairs[1][1]
        gap_us = long_dur_us - short_dur_us
        gpu_floor_us = max(KERNEL_FLOOR_ABS_US, KERNEL_FLOOR_REL * long_dur_us)
        if gap_us > gpu_floor_us:
            straggler_ranks.add(short_rank)
            straggler_signals.append(
                f"dur gap={gap_us:.0f}us long={long_dur_us:.0f}us "
                f"floor={gpu_floor_us:.0f}us (n=2 runner-up, late={short_rank})"
            )

    return WindowStats(
        key=window_key,
        rank_stats=rank_stats,
        median_created_offset_ms=med_created,
        median_started_offset_ms=med_started,
        median_gpu_duration_us=med_gpu_dur,
        straggler_ranks=straggler_ranks,
        straggler_signals=straggler_signals,
        is_p2p=is_p2p,
        megatron_id=megatron_id,
        pg_desc=pg_desc,
        gap_to_runner_up_ms=gap_to_runner_up_ms,
        late_rank=late_rank,
    )


# ---------------------------------------------------------------------------
# 3b. P2P duration MAD detection — augments straggler_ranks on P2P windows
#
#  Baseline scope: same-PG history within the block first; fall back to
#  cross-PG pooling (same op type + same message size) if the same-PG sample
#  is too small.
#
#  Attribution: if any rank's duration_ms exceeds the baseline by the MAD
#  threshold, flag BOTH endpoints of the 2-rank P2P pair (honest attribution).
#  Consistent with the DAG choice to leave 2-rank P2P HEADs unresolved.
# ---------------------------------------------------------------------------

P2P_MIN_SAMPLES = 5
P2P_K = 3.0
P2P_FLOOR_MS = 1.0  # 1 ms minimum excess for kernel-side flagging


def _p2p_op_type(profiling_name: str) -> str:
    """Extract op type ('send' or 'recv') from a P2P profiling_name."""
    n = (profiling_name or "").lower()
    if "send" in n:
        return "send"
    if "recv" in n:
        return "recv"
    return "p2p"


def _p2p_msg_size(collectives: List[Collective]) -> int:
    """Compute total message size in elements (input + output) for grouping."""
    total = 0
    for c in collectives:
        for shape in c.input_sizes or []:
            prod = 1
            for d in shape:
                prod *= d
            total += prod
        for shape in c.output_sizes or []:
            prod = 1
            for d in shape:
                prod *= d
            total += prod
    return total


def apply_p2p_duration_detection(
    all_window_stats: Dict[Tuple[str, str, int], WindowStats],
    grouped_windows: Dict[Tuple[str, str, int], List[Collective]],
    collectives_to_order: Dict[Tuple[str, str, int], int],
    k: float = P2P_K,
    floor_ms: float = P2P_FLOOR_MS,
    min_samples: int = P2P_MIN_SAMPLES,
) -> None:
    """
    Second-pass detector: flag P2P windows whose duration_ms exceeds a MAD-based
    baseline. Mutates `all_window_stats` in place (adds to straggler_ranks and
    appends to straggler_signals).
    """
    # Index P2P windows by (megatron_id, op_type, msg_size) for baseline pooling.
    # 'history' is windows with strictly lower gidx.
    p2p_entries: List[Tuple[Tuple[str, str, int], int, str, str, int, List[Collective]]] = []
    # tuple: (key, gidx, megatron_id, op_type, msg_size, collectives)

    for key, colls in grouped_windows.items():
        ws = all_window_stats.get(key)
        if ws is None or not ws.is_p2p:
            continue
        megatron_id, pg_desc, window_idx = key
        op_type = _p2p_op_type(colls[0].profiling_name if colls else "")
        msg_size = _p2p_msg_size(colls)
        gidx = collectives_to_order.get(key, 0)
        p2p_entries.append((key, gidx, megatron_id, op_type, msg_size, colls))

    # Sort by gidx so "history" is well-defined
    p2p_entries.sort(key=lambda x: x[1])

    # Build per-(megatron_id, op_type, msg_size) and per-(op_type, msg_size) pools
    same_pg_pool: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    cross_pg_pool: Dict[Tuple[str, int], List[float]] = defaultdict(list)

    # We'll walk in gidx order, using only strictly-earlier entries as history.
    for idx, (key, gidx, mid, op, sz, colls) in enumerate(p2p_entries):
        ws = all_window_stats[key]

        # Gather this window's per-rank duration_ms (one or both endpoints)
        rank_durations: List[Tuple[str, float]] = []
        for c in colls:
            if c.duration_ms is not None:
                rank_durations.append((c.file_id, c.duration_ms))

        if not rank_durations:
            # No watchdog duration on either side — nothing to flag from this signal.
            # Still need to update pools for future windows (skip — no data).
            continue

        # Build baseline from history
        same_pg_hist = same_pg_pool.get((mid, op, sz), [])
        if len(same_pg_hist) >= min_samples:
            baseline = list(same_pg_hist)
            baseline_scope = f"same-PG n={len(baseline)}"
        else:
            cross_hist = cross_pg_pool.get((op, sz), [])
            if len(cross_hist) >= min_samples:
                baseline = list(cross_hist)
                baseline_scope = f"cross-PG n={len(baseline)}"
            else:
                # Not enough history yet — skip flagging but keep accumulating
                for _, d in rank_durations:
                    same_pg_pool[(mid, op, sz)].append(d)
                    cross_pg_pool[(op, sz)].append(d)
                continue

        # MAD threshold on baseline
        med = _median(baseline)
        mad = _mad(baseline, med)
        threshold = max(floor_ms, k * mad * 1.4826)

        max_excess = 0.0
        max_rank = None
        any_exceed = False
        for rank_id, d in rank_durations:
            excess = d - med
            if excess > threshold:
                any_exceed = True
                if excess > max_excess:
                    max_excess = excess
                    max_rank = rank_id

        if any_exceed:
            # Honest attribution: flag BOTH endpoints of the 2-rank P2P.
            # Endpoints = all participating ranks for this window.
            participants = set(ws.rank_stats.keys())
            ws.straggler_ranks |= participants
            ws.straggler_signals.append(
                f"p2p_dur MAD excess={max_excess:.1f}ms thr={threshold:.1f}ms "
                f"({baseline_scope}, max_rank={max_rank}, k={k}, floor={floor_ms})"
            )

        # Update pools AFTER scoring (so this window's values aren't in its own baseline)
        for _, d in rank_durations:
            same_pg_pool[(mid, op, sz)].append(d)
            cross_pg_pool[(op, sz)].append(d)


# ---------------------------------------------------------------------------
# 4. Cascade DAG construction
# ---------------------------------------------------------------------------

# Static warmup filter: skip windows whose `window_idx` falls inside the
# training-loop warmup phase. Empirically on PP=2 traces, win=0/1 carry
# init coordination noise and win=2 carries the pipeline-fill cold start
# (stage-1 ranks enqueue ~2s after stage-0 ranks while activations propagate
# through the pipe for the first time), which is ~40x larger than steady-state
# straggler signal and saturates the cascade HEAD ranking. Anything in
# `window_idx >= WARMUP_WINDOW_IDX` is treated as steady state.
#
# A dynamic version would track per-iteration wall-clock from the inter-
# default_pg gap and declare steady state once iter-time stabilizes — see
# discussion in design notes. This is the cheap static fallback.
WARMUP_WINDOW_IDX = 3

WindowKey = Tuple[str, str, int]


@dataclass
class CascadeResult:
    """Result of cascade graph traversal."""
    heads: Set[WindowKey]
    predecessors: Dict[WindowKey, List[WindowKey]]
    successors: Dict[WindowKey, List[WindowKey]]
    longest_path_length: Dict[WindowKey, int]
    best_predecessor: Dict[WindowKey, Optional[WindowKey]]


def build_cascade_graph(
    window_stats: Dict[WindowKey, WindowStats],
    collectives_to_order: Dict[WindowKey, int],
) -> CascadeResult:
    """
    Build overlap graph of straggler windows and identify HEADs via DP.

    Nodes = windows with non-empty straggler_ranks and `window_idx >=
            WARMUP_WINDOW_IDX` (skips init coordination + pipeline-fill cold
            start on non-P2P windows; P2P uses p2p_seq_id which is large, so
            the filter doesn't affect P2P).
    Edges = windows sharing any participating rank, directed by global_idx.
    HEADs = nodes with no predecessor (cascade roots).
    """
    straggler_windows = {
        k: ws for k, ws in window_stats.items()
        if ws.straggler_ranks and k[2] >= WARMUP_WINDOW_IDX
    }

    if not straggler_windows:
        return CascadeResult(
            heads=set(), predecessors={}, successors={},
            longest_path_length={}, best_predecessor={},
        )

    keys = sorted(straggler_windows.keys(), key=lambda k: collectives_to_order.get(k, 0))
    key_to_node = {k: i for i, k in enumerate(keys)}
    node_to_key = {i: k for k, i in key_to_node.items()}

    # Participating ranks per window (full membership)
    node_ranks: Dict[int, Set[str]] = {
        key_to_node[k]: set(ws.rank_stats.keys()) for k, ws in straggler_windows.items()
    }

    node_ids = list(node_to_key.keys())
    neighbors: Dict[int, Set[int]] = defaultdict(set)
    for n1 in node_ids:
        for n2 in node_ids:
            if n1 != n2 and node_ranks[n1] & node_ranks[n2]:
                neighbors[n1].add(n2)
                neighbors[n2].add(n1)

    node_order = {n: collectives_to_order.get(node_to_key[n], 0) for n in node_ids}
    sorted_nodes = sorted(node_ids, key=lambda n: node_order[n])

    predecessors: Dict[int, List[int]] = {n: [] for n in node_ids}
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

    successors: Dict[int, List[int]] = {n: [] for n in node_ids}
    for n, preds in predecessors.items():
        for p in preds:
            successors[p].append(n)

    heads = {node_to_key[n] for n in node_ids if not predecessors[n]}

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


def get_longest_chain_from_head(
    head: WindowKey, cascade: CascadeResult,
) -> List[WindowKey]:
    """Walk forward from HEAD following longest downstream path."""
    chain = [head]
    current = head
    while cascade.successors.get(current):
        succs = cascade.successors[current]
        best = max(succs, key=lambda s: cascade.longest_path_length.get(s, 0))
        chain.append(best)
        current = best
    return chain


def get_full_dag_from_head(
    head: WindowKey, cascade: CascadeResult,
) -> List[Tuple[WindowKey, int]]:
    """BFS from HEAD, returning (window_key, depth) pairs for tree rendering."""
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


def parse_inject_ranks(gt: dict) -> List[str]:
    """Return injected ranks as a list of stringified ints.

    Accepts both formats: `inject_ranks: 2,3` (multi) and `inject_rank: 3`
    (legacy single). Empty list if neither present or value is empty.
    """
    raw = gt.get("inject_ranks") or gt.get("inject_rank") or ""
    return [r.strip() for r in str(raw).split(",") if r.strip()]


def print_summary(
    all_window_stats: Dict[WindowKey, WindowStats],
    collectives_to_order: Dict[WindowKey, int],
    cascade: CascadeResult,
    pg_filter: Optional[str] = None,
    verbose_cascade: bool = False,
):
    """Print summary table + cascade chains from HEADs."""
    logger.info("\n=== Straggler Analysis Summary ===\n")
    logger.info(
        f"{'GIdx':>4} | {'PG Desc':<40} | {'Win':>3} | {'Ranks':>20} | "
        f"{'Straggler':>10} | {'Signal':<70}"
    )
    logger.info("-" * 160)

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
            f"{gidx:>4} | {pg_desc:<40} | {window_idx:>3} | {ranks_str:>20} | "
            f"{straggler_str:>10} | {signal_str:<70}"
        )

    logger.info(f"\nWindows with stragglers: {straggler_window_count}/{total_windows}")

    if cascade.heads:
        logger.info("\n=== Cascade Chains (from HEADs) ===\n")
        sorted_heads = sorted(cascade.heads, key=lambda k: collectives_to_order.get(k, 0))

        for head_idx, head in enumerate(sorted_heads):
            head_ws = all_window_stats[head]
            head_gidx = collectives_to_order.get(head, -1)
            head_ranks = ",".join(sorted(head_ws.straggler_ranks, key=int))
            megatron_id, pg_desc, window_idx = head

            logger.info(
                f"  HEAD {head_idx}: [{head_gidx}] {pg_desc} win={window_idx} "
                f"(straggler rank(s): {head_ranks})"
            )

            if verbose_cascade:
                dag = get_full_dag_from_head(head, cascade)
                for node, depth in dag[1:]:
                    ws = all_window_stats[node]
                    gidx = collectives_to_order.get(node, -1)
                    _, node_pg_desc, node_win_idx = node
                    straggler_str = ",".join(sorted(ws.straggler_ranks, key=int))
                    indent = "    " + "  " * depth
                    logger.info(f"{indent}→ [{gidx}] {node_pg_desc} win={node_win_idx} straggler={straggler_str}")
            else:
                chain = get_longest_chain_from_head(head, cascade)
                for node in chain[1:]:
                    ws = all_window_stats[node]
                    gidx = collectives_to_order.get(node, -1)
                    _, node_pg_desc, node_win_idx = node
                    straggler_str = ",".join(sorted(ws.straggler_ranks, key=int))
                    logger.info(f"      → [{gidx}] {node_pg_desc} win={node_win_idx} straggler={straggler_str}")

            logger.info("")
    else:
        logger.info("\nNo HEADs found (no straggler windows).")


def print_detailed(
    all_window_stats: Dict[Tuple[str, str, int], WindowStats],
    collectives_to_order: Dict[Tuple[str, str, int], int],
    pg_filter: Optional[str] = None,
):
    """Per-rank breakdown for each window."""
    for key in sorted(all_window_stats.keys(), key=lambda k: collectives_to_order.get(k, 0)):
        ws = all_window_stats[key]
        megatron_id, pg_desc, window_idx = key

        if pg_filter and pg_filter.upper() not in pg_desc.upper():
            continue

        straggler_marker = " [HAS STRAGGLER]" if ws.straggler_ranks else ""
        logger.info(f"\n--- {pg_desc} (megatron_id={megatron_id}, window={window_idx}){straggler_marker} ---")
        logger.info(
            f"  {'Rank':>6} | {'N':>4} | {'created_offset':>16} | {'started_offset':>16} | "
            f"{'gpu_dur':>12} | {'dur_ms':>10} |"
        )

        for rank_id in sorted(ws.rank_stats.keys(), key=int):
            rs = ws.rank_stats[rank_id]
            marker = " <-- STRAGGLER" if rank_id in ws.straggler_ranks else ""
            dur_str = f"{rs.mean_duration_ms:>7.2f}ms" if rs.mean_duration_ms is not None else f"{'-':>10}"
            logger.info(
                f"  {rank_id:>6} | {rs.n_entries:>4} | "
                f"{rs.mean_created_offset_ms:>13.3f}ms | "
                f"{rs.mean_started_offset_ms:>13.3f}ms | "
                f"{rs.mean_gpu_duration_us:>9.1f}us | "
                f"{dur_str:>10} |{marker}"
            )

        logger.info(
            f"  {'median':>6} | {'':>4} | "
            f"{ws.median_created_offset_ms:>13.3f}ms | "
            f"{ws.median_started_offset_ms:>13.3f}ms | "
            f"{ws.median_gpu_duration_us:>9.1f}us | "
            f"{'':>10} |"
        )
        if ws.straggler_signals:
            for sig in ws.straggler_signals:
                logger.info(f"  signal: {sig}")


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
    inject_ranks = parse_inject_ranks(ground_truth)
    inject_delay = ground_truth.get("inject_delay_ms", "unknown")
    inject_ranks_str = ",".join(inject_ranks) if inject_ranks else "unknown"
    logger.info(f"  Ground truth: inject_type={inject_type}, inject_ranks={inject_ranks_str}, inject_delay_ms={inject_delay}")

    if inject_type == "none":
        flagged_windows = sum(1 for ws in all_window_stats.values() if ws.straggler_ranks)
        total = len(all_window_stats)
        logger.info(f"  Expected: no stragglers")
        logger.info(f"  Detected: {flagged_windows}/{total} windows flagged")
        return

    for inject_rank in inject_ranks:
        correct = 0
        wrong = 0
        missed = 0
        total_with_inject_rank = 0

        for ws in all_window_stats.values():
            if inject_rank in ws.rank_stats:
                total_with_inject_rank += 1
                if inject_rank in ws.straggler_ranks:
                    correct += 1
                elif ws.straggler_ranks:
                    wrong += 1
                else:
                    missed += 1

        logger.info(f"  --- Rank {inject_rank} ---")
        logger.info(f"  Windows where rank {inject_rank} participates: {total_with_inject_rank}")
        logger.info(f"  Correctly flagged rank {inject_rank}: {correct}/{total_with_inject_rank}")
        logger.info(f"  Missed (no straggler flagged): {missed}/{total_with_inject_rank}")
        logger.info(f"  Wrong rank flagged: {wrong}/{total_with_inject_rank}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(
    trace_dir: str,
    verbose: bool = False,
    pg_filter: Optional[str] = None,
    k: float = 3.0,
    floor_ms: float = 20.0,
    p2p_k: float = P2P_K,
    p2p_floor_ms: float = P2P_FLOOR_MS,
):
    """Run the full analysis pipeline."""

    # 1. Load
    collectives_by_file, pg_configs, pg_status = load_trace_dir(trace_dir)

    # 2. Window
    windows, collectives_to_order = group_collectives_by_windows(collectives_by_file)
    logger.info(f"\nWindowing produced {len(windows)} unique window groups:")
    for key in sorted(windows.keys(), key=lambda k: collectives_to_order.get(k, 0)):
        megatron_id, pg_desc, window_idx = key
        ranks = set(c.file_id for c in windows[key])
        gidx = collectives_to_order.get(key, -1)
        logger.info(
            f"  [gidx={gidx}] ({megatron_id}, {pg_desc}, win={window_idx}): "
            f"{len(windows[key])} entries, ranks={sorted(ranks, key=int)}"
        )

    # 3. Per-window stats + MAD on time_created
    logger.info(f"\nComputing per-window statistics (k={k}, floor_ms={floor_ms})...")
    all_window_stats: Dict[Tuple[str, str, int], WindowStats] = {}
    for key, colls in windows.items():
        all_window_stats[key] = compute_window_stats(key, colls, k=k, floor_ms=floor_ms)
    tc_count = sum(1 for ws in all_window_stats.values() if ws.straggler_ranks)
    logger.info(f"  {len(all_window_stats)} windows analyzed, {tc_count} flagged by time_created MAD")

    # 3b. P2P duration MAD detection
    logger.info(f"\nApplying P2P duration detection (k={p2p_k}, floor_ms={p2p_floor_ms})...")
    apply_p2p_duration_detection(
        all_window_stats, windows, collectives_to_order,
        k=p2p_k, floor_ms=p2p_floor_ms,
    )
    total_flagged = sum(1 for ws in all_window_stats.values() if ws.straggler_ranks)
    logger.info(f"  {total_flagged} windows flagged after P2P duration pass (delta: +{total_flagged - tc_count})")

    # 4. Cascade DAG
    logger.info("\nBuilding cascade graph...")
    cascade = build_cascade_graph(all_window_stats, collectives_to_order)
    logger.info(f"  {len(cascade.heads)} HEADs found")

    # 5. Output
    if verbose:
        print_detailed(all_window_stats, collectives_to_order, pg_filter)

    print_summary(all_window_stats, collectives_to_order, cascade, pg_filter, verbose)

    # 6. Ground truth
    ground_truth = load_ground_truth(trace_dir)
    print_ground_truth_comparison(all_window_stats, ground_truth)


def test_synthetic_abc():
    """A → B → C, A has straggler, B/C are downstream cascade."""
    print("\n=== Synthetic A/B/C Test ===\n")

    key_a: WindowKey = ("mg_a", "PG_A", 3)
    key_b: WindowKey = ("mg_b", "PG_B", 3)
    key_c: WindowKey = ("mg_c", "PG_C", 3)

    window_stats: Dict[WindowKey, WindowStats] = {
        key_a: WindowStats(
            key=key_a,
            rank_stats={"0": None, "1": None},  # type: ignore
            median_created_offset_ms=0, median_started_offset_ms=0, median_gpu_duration_us=0,
            straggler_ranks={"0"}, straggler_signals=["test"],
        ),
        key_b: WindowStats(
            key=key_b,
            rank_stats={str(i): None for i in range(8)},  # type: ignore
            median_created_offset_ms=0, median_started_offset_ms=0, median_gpu_duration_us=0,
            straggler_ranks={"3"}, straggler_signals=["test"],
        ),
        key_c: WindowStats(
            key=key_c,
            rank_stats={"6": None, "7": None},  # type: ignore
            median_created_offset_ms=0, median_started_offset_ms=0, median_gpu_duration_us=0,
            straggler_ranks={"7"}, straggler_signals=["test"],
        ),
    }

    collectives_to_order: Dict[WindowKey, int] = {key_a: 0, key_b: 1, key_c: 2}

    cascade = build_cascade_graph(window_stats, collectives_to_order)

    print(f"HEADs: {cascade.heads}")
    assert cascade.heads == {key_a}, f"Expected HEAD={{A}}, got {cascade.heads}"

    print(f"\nPredecessors[A]: {cascade.predecessors[key_a]}")
    print(f"Predecessors[B]: {cascade.predecessors[key_b]}")
    print(f"Predecessors[C]: {cascade.predecessors[key_c]}")
    assert cascade.predecessors[key_a] == []
    assert cascade.predecessors[key_b] == [key_a]
    assert cascade.predecessors[key_c] == [key_b]

    chain = get_longest_chain_from_head(key_a, cascade)
    print(f"\nLongest chain from A: {chain}")
    assert chain == [key_a, key_b, key_c]

    print("\n✓ Synthetic A/B/C test PASSED\n")


def test_synthetic_branching():
    """A → B, A → C (branching from single HEAD)."""
    print("\n=== Synthetic Branching Test ===\n")

    key_a: WindowKey = ("mg_a", "PG_A", 3)
    key_b: WindowKey = ("mg_b", "PG_B", 3)
    key_c: WindowKey = ("mg_c", "PG_C", 3)

    window_stats: Dict[WindowKey, WindowStats] = {
        key_a: WindowStats(
            key=key_a,
            rank_stats={"0": None, "1": None, "2": None, "3": None},  # type: ignore
            median_created_offset_ms=0, median_started_offset_ms=0, median_gpu_duration_us=0,
            straggler_ranks={"0"}, straggler_signals=["test"],
        ),
        key_b: WindowStats(
            key=key_b,
            rank_stats={"0": None, "1": None},  # type: ignore
            median_created_offset_ms=0, median_started_offset_ms=0, median_gpu_duration_us=0,
            straggler_ranks={"1"}, straggler_signals=["test"],
        ),
        key_c: WindowStats(
            key=key_c,
            rank_stats={"2": None, "3": None},  # type: ignore
            median_created_offset_ms=0, median_started_offset_ms=0, median_gpu_duration_us=0,
            straggler_ranks={"3"}, straggler_signals=["test"],
        ),
    }

    collectives_to_order: Dict[WindowKey, int] = {key_a: 0, key_b: 1, key_c: 2}

    cascade = build_cascade_graph(window_stats, collectives_to_order)

    print(f"HEADs: {cascade.heads}")
    assert cascade.heads == {key_a}

    print(f"Successors[A]: {cascade.successors[key_a]}")
    assert set(cascade.successors[key_a]) == {key_b, key_c}

    dag = get_full_dag_from_head(key_a, cascade)
    print(f"Full DAG from A: {dag}")
    assert len(dag) == 3

    print("\n✓ Synthetic branching test PASSED\n")


def _make_collective(
    rank: str, seq_id: int, time_created_ms: float, pg_desc: str = "TP",
    megatron_id: str = "mg",
) -> Collective:
    """Test helper: minimal Collective constructor."""
    return Collective(
        record_id=0,
        file_id=rank,
        collective_seq_id=seq_id,
        p2p_seq_id=0,
        pg_id=0,
        op_id=0,
        profiling_name="nccl:all_reduce",
        state="completed",
        time_created_ns=int(time_created_ms * 1e6),
        time_discovered_started_ns=int(time_created_ms * 1e6) + 1000,
        time_discovered_completed_ns=int(time_created_ms * 1e6) + 2000,
        process_group=[megatron_id, pg_desc],
        input_sizes=[],
        output_sizes=[],
        input_dtypes=[],
        output_dtypes=[],
        is_p2p=False,
    )


def _make_collective_with_dur(
    rank: str, seq: int, time_created_ms: float, duration_ms: Optional[float],
    pg_desc: str = "TENSOR_MODEL_PARALLEL_GROUP", megatron_id: str = "mg",
) -> Collective:
    """Test helper: Collective with controllable duration_ms field."""
    c = _make_collective(rank, seq, time_created_ms, pg_desc=pg_desc, megatron_id=megatron_id)
    c.duration_ms = duration_ms
    return c


def test_host_gap_to_runner_up_n2():
    """2-rank TP window: rank 3 consistently 50ms behind rank 2 → flagged via
    gap-to-runner-up on per-rank means."""
    print("\n=== Test: host gap-to-runner-up (n=2) ===")
    key: WindowKey = ("28", "TENSOR_MODEL_PARALLEL_GROUP", 4)
    colls: List[Collective] = []
    for seq in range(5):
        base = 100.0 + seq * 10.0
        colls.append(_make_collective("2", seq, base))
        colls.append(_make_collective("3", seq, base + 50.0))
    ws = compute_window_stats(key, colls, k=3.0, floor_ms=20.0)
    print(f"  gap_to_runner_up_ms={ws.gap_to_runner_up_ms}")
    print(f"  late_rank={ws.late_rank}")
    print(f"  straggler_ranks={ws.straggler_ranks}")
    assert ws.late_rank == "3"
    assert abs(ws.gap_to_runner_up_ms - 50.0) < 0.1
    assert ws.straggler_ranks == {"3"}
    print("  PASSED")


def test_host_n2_below_floor():
    """2-rank window with 10ms gap (below 20ms floor) → recorded but not flagged."""
    print("\n=== Test: host n=2 below floor ===")
    key: WindowKey = ("28", "TENSOR_MODEL_PARALLEL_GROUP", 4)
    colls: List[Collective] = []
    for seq in range(5):
        base = 100.0 + seq * 10.0
        colls.append(_make_collective("2", seq, base))
        colls.append(_make_collective("3", seq, base + 10.0))
    ws = compute_window_stats(key, colls, k=3.0, floor_ms=20.0)
    print(f"  gap_to_runner_up_ms={ws.gap_to_runner_up_ms}")
    print(f"  straggler_ranks={ws.straggler_ranks}")
    assert abs(ws.gap_to_runner_up_ms - 10.0) < 0.1
    assert ws.late_rank == "3"
    assert ws.straggler_ranks == set()
    print("  PASSED")


def test_host_n2_single_sample():
    """2-rank, N=1 (one DP allreduce). Above floor: flag; below: don't."""
    print("\n=== Test: host n=2 single sample ===")
    key: WindowKey = ("3", "DATA_PARALLEL_GROUP_WITH_CP", 0)
    ws = compute_window_stats(key, [
        _make_collective("1", 0, 100.0),
        _make_collective("3", 0, 150.0),
    ], k=3.0, floor_ms=20.0)
    print(f"  N=1 gap=50ms: late={ws.late_rank} flagged={ws.straggler_ranks}")
    assert ws.late_rank == "3"
    assert ws.straggler_ranks == {"3"}

    ws = compute_window_stats(key, [
        _make_collective("1", 0, 100.0),
        _make_collective("3", 0, 105.0),
    ], k=3.0, floor_ms=20.0)
    print(f"  N=1 gap=5ms: late={ws.late_rank} flagged={ws.straggler_ranks}")
    assert ws.late_rank == "3"
    assert ws.straggler_ranks == set()
    print("  PASSED")


def test_kernel_deficit_n2():
    """2-rank window where rank 3's kernel runs much shorter than rank 2's
    (classic kernel-straggler signature). Expected: flag rank 3 via dur gap."""
    print("\n=== Test: kernel deficit (n=2) ===")
    key: WindowKey = ("28", "TENSOR_MODEL_PARALLEL_GROUP", 4)
    colls: List[Collective] = []
    for seq in range(5):
        base = 100.0 + seq * 10.0
        # Same time_created so host detector doesn't fire; durations differ
        colls.append(_make_collective_with_dur("2", seq, base, 10.0))   # 10ms kernel
        colls.append(_make_collective_with_dur("3", seq, base, 0.5))    # 0.5ms kernel
    ws = compute_window_stats(key, colls, k=3.0, floor_ms=20.0)
    print(f"  straggler_ranks={ws.straggler_ranks}")
    print(f"  signals={ws.straggler_signals}")
    assert ws.straggler_ranks == {"3"}
    assert any("dur gap" in s for s in ws.straggler_signals)
    print("  PASSED")


def test_kernel_no_fire_when_clean():
    """2-rank with similar durations (5ms vs 5.1ms) — should NOT flag."""
    print("\n=== Test: kernel clean (n=2, no fire) ===")
    key: WindowKey = ("28", "TENSOR_MODEL_PARALLEL_GROUP", 4)
    colls: List[Collective] = []
    for seq in range(5):
        base = 100.0 + seq * 10.0
        colls.append(_make_collective_with_dur("2", seq, base, 5.0))
        colls.append(_make_collective_with_dur("3", seq, base, 5.1))
    ws = compute_window_stats(key, colls, k=3.0, floor_ms=20.0)
    print(f"  straggler_ranks={ws.straggler_ranks}")
    assert ws.straggler_ranks == set()
    print("  PASSED")


def test_kernel_scale_aware_floor():
    """Scale-aware floor: relative (25%) dominates on large kernels. With
    medium durations (10ms vs 9ms = 10% gap = 1000µs), relative floor (25% of
    10ms = 2500µs) blocks the flag even though absolute is well above 50µs."""
    print("\n=== Test: kernel scale-aware floor blocks small relative ===")
    key: WindowKey = ("28", "TENSOR_MODEL_PARALLEL_GROUP", 4)
    colls: List[Collective] = []
    for seq in range(5):
        base = 100.0 + seq * 10.0
        colls.append(_make_collective_with_dur("2", seq, base, 10.0))
        colls.append(_make_collective_with_dur("3", seq, base, 9.0))   # only 10% short
    ws = compute_window_stats(key, colls, k=3.0, floor_ms=20.0)
    print(f"  straggler_ranks={ws.straggler_ranks}")
    assert ws.straggler_ranks == set(), "10% deficit should not fire — relative floor blocks"
    print("  PASSED")


def analyze_quiet(trace_dir: str, k: float, floor_ms: float) -> dict:
    """Run analysis without logging, return summary stats for grid sweep."""
    # Silence logger output during grid sweep
    prev_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        collectives_by_file, pg_configs, pg_status = load_trace_dir(trace_dir)
        windows, collectives_to_order = group_collectives_by_windows(collectives_by_file)

        all_window_stats: Dict[Tuple[str, str, int], WindowStats] = {}
        for key, colls in windows.items():
            all_window_stats[key] = compute_window_stats(key, colls, k=k, floor_ms=floor_ms)

        apply_p2p_duration_detection(all_window_stats, windows, collectives_to_order)

        cascade = build_cascade_graph(all_window_stats, collectives_to_order)
    finally:
        logger.setLevel(prev_level)

    ground_truth = load_ground_truth(trace_dir) or {}
    inject_type = ground_truth.get("inject_type", "none")
    inject_ranks = parse_inject_ranks(ground_truth)

    straggler_count = sum(1 for ws in all_window_stats.values() if ws.straggler_ranks)
    head_straggler_ranks = set()
    for head_key in cascade.heads:
        ws = all_window_stats.get(head_key)
        if ws:
            head_straggler_ranks.update(ws.straggler_ranks)

    injected_set = set(inject_ranks)
    hit = bool(injected_set) and injected_set.issubset(head_straggler_ranks)

    return {
        "inject_type": inject_type,
        "inject_ranks": inject_ranks,
        "n_windows": len(windows),
        "n_straggler_windows": straggler_count,
        "n_heads": len(cascade.heads),
        "head_straggler_ranks": head_straggler_ranks,
        "hit": hit,
    }


def grid_sweep(trace_dirs: List[str]):
    """Grid sweep over MAD k and floor_ms values."""
    k_values = [2.5, 3.0, 3.5]
    floor_values = [10.0, 20.0, 30.0]

    print("\n" + "=" * 80)
    print("GRID SWEEP: MAD on time_created (with P2P duration second pass)")
    print("=" * 80)

    baselines = []
    injections = []
    for td in trace_dirs:
        gt = load_ground_truth(td) or {}
        if gt.get("inject_type", "none") == "none":
            baselines.append(td)
        else:
            injections.append((td, gt))

    print(f"\nTraces: {len(baselines)} baseline, {len(injections)} injection")
    for td, gt in injections:
        ranks_str = ",".join(parse_inject_ranks(gt)) or "?"
        print(f"  - {os.path.basename(td)}: {gt.get('inject_type')} rank(s) {ranks_str} @ {gt.get('inject_delay_ms')}ms")

    results = []
    for k in k_values:
        for floor_ms in floor_values:
            row = {"k": k, "floor_ms": floor_ms}

            baseline_heads = []
            for td in baselines:
                r = analyze_quiet(td, k, floor_ms)
                baseline_heads.append(r["n_heads"])
            row["baseline_heads"] = baseline_heads

            injection_hits = []
            for td, gt in injections:
                r = analyze_quiet(td, k, floor_ms)
                injection_hits.append({
                    "name": os.path.basename(td),
                    "inject_ranks": parse_inject_ranks(gt),
                    "hit": r["hit"],
                    "n_heads": r["n_heads"],
                    "head_ranks": r["head_straggler_ranks"],
                })
            row["injection_hits"] = injection_hits
            results.append(row)

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
            ranks_label = ",".join(h["inject_ranks"]) or "?"
            hit_strs.append(f"r{ranks_label}:{mark}")
            if not h["hit"]:
                all_hit = False

        baseline_pass = all(h <= 1 for h in bh)
        if baseline_pass and all_hit:
            passing_cells.append((k, floor_ms))

        status = "PASS" if (baseline_pass and all_hit) else ""
        print(f"{k:>4.1f} | {floor_ms:>5.0f} | {baseline_str:>15} | {' '.join(hit_strs)} {status}")

    print("-" * 80)

    if passing_cells:
        best = max(passing_cells, key=lambda x: (x[0], x[1]))
        print(f"\n✓ BEST CELL: k={best[0]}, floor_ms={best[1]} (most conservative passing)")
    else:
        print("\n✗ NO PASSING CELL — review results above")

    return results


def main():
    parser = argparse.ArgumentParser(description="FR Straggler Analyzer (cascade DAG + MAD detection)")
    parser.add_argument("trace_dir", nargs="?", help="Path to block directory containing _dump_*.json files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed per-rank per-window output")
    parser.add_argument("--pg", default=None, help="Filter output to PGs matching this substring")
    parser.add_argument("--k", type=float, default=3.0,
                        help="MAD multiplier for time_created flagging (default: 3.0)")
    parser.add_argument("--floor-ms", type=float, default=20.0,
                        help="Minimum excess for time_created flagging in ms (default: 20.0)")
    parser.add_argument("--p2p-k", type=float, default=P2P_K,
                        help=f"MAD multiplier for P2P duration flagging (default: {P2P_K})")
    parser.add_argument("--p2p-floor-ms", type=float, default=P2P_FLOOR_MS,
                        help=f"Minimum excess for P2P duration flagging in ms (default: {P2P_FLOOR_MS})")
    parser.add_argument("--test", action="store_true", help="Run synthetic tests")
    parser.add_argument("--grid-sweep", nargs="+", metavar="TRACE_DIR",
                        help="Run grid sweep on multiple trace directories")
    args = parser.parse_args()

    if args.test:
        test_synthetic_abc()
        test_synthetic_branching()
        test_host_gap_to_runner_up_n2()
        test_host_n2_below_floor()
        test_host_n2_single_sample()
        test_kernel_deficit_n2()
        test_kernel_no_fire_when_clean()
        test_kernel_scale_aware_floor()
        return

    if args.grid_sweep:
        grid_sweep(args.grid_sweep)
        return

    if not args.trace_dir:
        parser.error("trace_dir is required unless --test or --grid-sweep is specified")

    log_path = os.path.join(args.trace_dir, "analysis.log")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Auto-saving output to {log_path}")

    analyze(
        args.trace_dir,
        verbose=args.verbose,
        pg_filter=args.pg,
        k=args.k,
        floor_ms=args.floor_ms,
        p2p_k=args.p2p_k,
        p2p_floor_ms=args.p2p_floor_ms,
    )


if __name__ == "__main__":
    main()