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
    mean_gpu_duration_us: float     # mean(completed - started) in µs


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
    threshold_ms: float = 5.0,
) -> WindowStats:
    """
    Compute per-rank timing stats for a single window and identify stragglers.

    Straggler identification: rank with largest deviation from median on time_created_ns,
    if deviation exceeds threshold_ms.
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
        # time_created offsets
        created_offsets = [(c.time_created_ns - min_created) / 1e6 for c in rank_colls]
        mean_created = sum(created_offsets) / len(created_offsets)

        # time_started offsets
        started_offsets = []
        for c in rank_colls:
            if c.time_discovered_started_ns is not None:
                started_offsets.append((c.time_discovered_started_ns - min_started) / 1e6)
        mean_started = sum(started_offsets) / len(started_offsets) if started_offsets else 0.0

        # gpu_duration
        gpu_durs = []
        for c in rank_colls:
            if c.time_discovered_started_ns is not None and c.time_discovered_completed_ns is not None:
                dur_us = (c.time_discovered_completed_ns - c.time_discovered_started_ns) / 1e3
                gpu_durs.append(dur_us)
        mean_gpu_dur = sum(gpu_durs) / len(gpu_durs) if gpu_durs else 0.0

        rank_stats[rank_id] = RankWindowStats(
            rank_id=rank_id,
            n_entries=len(rank_colls),
            mean_created_offset_ms=mean_created,
            mean_started_offset_ms=mean_started,
            mean_gpu_duration_us=mean_gpu_dur,
        )

    # Compute medians across ranks
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

    # Identify stragglers: rank with max deviation from median on time_created
    straggler_ranks = set()
    straggler_signal = ""

    if len(rank_stats) >= 2:
        # Check time_created first (host-side signal)
        max_dev_rank = None
        max_dev = 0.0
        for rank_id, rs in rank_stats.items():
            dev = rs.mean_created_offset_ms - med_created
            if dev > max_dev:
                max_dev = dev
                max_dev_rank = rank_id

        if max_dev_rank is not None and max_dev > threshold_ms:
            straggler_ranks.add(max_dev_rank)
            straggler_signal = f"time_created +{max_dev:.1f}ms"

        # Also check time_started
        max_started_dev_rank = None
        max_started_dev = 0.0
        for rank_id, rs in rank_stats.items():
            dev = rs.mean_started_offset_ms - med_started
            if dev > max_started_dev:
                max_started_dev = dev
                max_started_dev_rank = rank_id

        if max_started_dev_rank is not None and max_started_dev > threshold_ms:
            if max_started_dev_rank not in straggler_ranks:
                straggler_ranks.add(max_started_dev_rank)
                if straggler_signal:
                    straggler_signal += f"; time_started +{max_started_dev:.1f}ms (rank {max_started_dev_rank})"
                else:
                    straggler_signal = f"time_started +{max_started_dev:.1f}ms"

    return WindowStats(
        key=window_key,
        rank_stats=rank_stats,
        median_created_offset_ms=med_created,
        median_started_offset_ms=med_started,
        median_gpu_duration_us=med_gpu_dur,
        straggler_ranks=straggler_ranks,
        straggler_signal=straggler_signal,
    )


# ---------------------------------------------------------------------------
# 4. Group PGs by temporal order
#    Adapted from fr_attribution.py:group_pgs (lines 770-938)
#
#    Builds a PG overlap graph and finds longest paths via DFS with
#    monotonicity on global_idx (scheduling order). Head of each path
#    is the wavefront PG — its straggler ranks are the root cause.
# ---------------------------------------------------------------------------


def group_pgs_temporal(
    window_stats: Dict[Tuple[str, str, int], WindowStats],
    collectives_to_order: Dict[Tuple[str, str, int], int],
) -> Dict[int, List[Tuple[str, str, int]]]:
    """
    Group straggler PG windows by finding longest paths in the overlap graph.
    Adapted from fr_attribution.py:group_pgs.

    Nodes = PG windows with straggler ranks.
    Edges = PG windows sharing any participating rank.
    DFS with monotonicity: only traverse lower → higher scheduling order (global_idx).
    Longest paths are kept; subset paths are removed.

    Returns:
        grouped_pgs: group_id -> list of window keys in path order (head first)
    """
    straggler_windows = {
        k: ws for k, ws in window_stats.items() if ws.straggler_ranks
    }
    if not straggler_windows:
        return {}

    # Map window keys → integer node IDs
    keys = sorted(straggler_windows.keys(), key=lambda k: collectives_to_order.get(k, 0))
    key_to_node = {k: i for i, k in enumerate(keys)}
    node_to_key = {i: k for k, i in key_to_node.items()}

    # All participating ranks per window (edges based on full membership, not just stragglers)
    node_ranks: Dict[int, Set[str]] = {
        key_to_node[k]: set(ws.rank_stats.keys()) for k, ws in straggler_windows.items()
    }

    # Build adjacency: PG windows sharing any rank get an edge
    graph: Dict[int, Set[int]] = defaultdict(set)
    node_ids = list(node_to_key.keys())
    for n1 in node_ids:
        graph[n1].add(n1)
        for n2 in node_ids:
            if n1 != n2 and node_ranks[n1] & node_ranks[n2]:
                graph[n1].add(n2)

    # Use BFS-like longest path for DAG (topological order by global_idx)
    # Since we have monotonicity constraint, this is a DAG - use dynamic programming
    node_order_map = {n: collectives_to_order.get(node_to_key[n], 0) for n in node_ids}
    sorted_nodes = sorted(node_ids, key=lambda n: node_order_map[n])

    # dp[n] = (longest path length ending at n, predecessor)
    dp: Dict[int, Tuple[int, Optional[int]]] = {n: (1, None) for n in node_ids}

    for node in sorted_nodes:
        node_order = node_order_map[node]
        for nb in graph[node]:
            if nb != node and node_order_map[nb] > node_order:
                # nb comes after node in ordering
                if dp[node][0] + 1 > dp[nb][0]:
                    dp[nb] = (dp[node][0] + 1, node)

    # Reconstruct paths starting from nodes with no incoming edges used
    visited_global: Set[int] = set()
    all_paths: List[List[int]] = []

    # Find path heads: nodes that aren't predecessors of longer paths
    for start in sorted(node_ids, key=lambda n: len(node_ranks.get(n, set())), reverse=True):
        if start in visited_global:
            continue

        # Walk forward to find the end of this path
        current = start
        path = [current]
        visited_global.add(current)

        while True:
            node_order = node_order_map[current]
            # Find best next step
            best_next = None
            best_len = 0
            for nb in graph[current]:
                if nb != current and nb not in visited_global and node_order_map[nb] > node_order:
                    if dp[nb][0] > best_len:
                        best_len = dp[nb][0]
                        best_next = nb
            if best_next is None:
                break
            path.append(best_next)
            visited_global.add(best_next)
            current = best_next

        all_paths.append(path)

    # Remove subset paths
    unique: List[List[int]] = []
    for i, p1 in enumerate(all_paths):
        s1 = set(p1)
        is_sub = False
        for j, p2 in enumerate(all_paths):
            if i != j:
                s2 = set(p2)
                if s1 < s2:
                    is_sub = True
                    break
                if s1 == s2 and p1 not in unique and p2 not in unique:
                    unique.append(p1)
                    is_sub = True
                    break
        if not is_sub:
            unique.append(p1)

    return {gid: [node_to_key[n] for n in path] for gid, path in enumerate(unique)}


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
    all_window_stats: Dict[Tuple[str, str, int], WindowStats],
    collectives_to_order: Dict[Tuple[str, str, int], int],
    grouped_pgs: Dict[int, List[Tuple[str, str, int]]],
    pg_filter: Optional[str] = None,
):
    """Print summary table of windows with stragglers, plus grouped PG paths."""
    logger.info("\n=== Straggler Analysis Summary ===\n")

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

    # Grouped PG paths (temporal order)
    if grouped_pgs:
        logger.info("\n=== Grouped PG Paths (temporal order) ===\n")
        for group_id, path_keys in grouped_pgs.items():
            head_key = path_keys[0]
            head_ws = all_window_stats[head_key]
            head_ranks = ",".join(sorted(head_ws.straggler_ranks, key=int))
            logger.info(f"  Path {group_id} (head straggler rank(s): {head_ranks}):")
            for step, key in enumerate(path_keys):
                megatron_id, pg_desc, window_idx = key
                ws = all_window_stats[key]
                gidx = collectives_to_order.get(key, -1)
                straggler_str = ",".join(sorted(ws.straggler_ranks, key=int))
                marker = " <-- HEAD" if step == 0 else ""
                logger.info(
                    f"    [{gidx:>3}] {pg_desc:<40} win={window_idx} straggler={straggler_str}{marker}"
                )
    else:
        logger.info("\nNo grouped PG paths found.")


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

def analyze(trace_dir: str, verbose: bool = False, pg_filter: Optional[str] = None, threshold_ms: float = 5.0):
    """Run the full analysis pipeline."""

    # 1. Load
    collectives_by_file, pg_configs, pg_status = load_trace_dir(trace_dir)

    # 2. Window (two-pass: row compaction → column ordering)
    # Returns both grouped windows and their global ordering (no default_pg hack)
    windows, collectives_to_order = group_collectives_by_windows(collectives_by_file)
    logger.info(f"\nWindowing produced {len(windows)} unique window groups:")
    for key in sorted(windows.keys(), key=lambda k: collectives_to_order.get(k, 0)):
        megatron_id, pg_desc, window_idx = key
        ranks = set(c.file_id for c in windows[key])
        gidx = collectives_to_order.get(key, -1)
        logger.info(f"  [gidx={gidx}] ({megatron_id}, {pg_desc}, win={window_idx}): {len(windows[key])} entries, ranks={sorted(ranks, key=int)}")

    # 3. Per-window stats + straggler identification
    logger.info("\nComputing per-window statistics...")
    all_window_stats: Dict[Tuple[str, str, int], WindowStats] = {}
    for key, colls in windows.items():
        all_window_stats[key] = compute_window_stats(key, colls, threshold_ms=threshold_ms)
    straggler_count = sum(1 for ws in all_window_stats.values() if ws.straggler_ranks)
    logger.info(f"  {len(all_window_stats)} windows analyzed, {straggler_count} with stragglers")

    # 4. Group PGs by temporal order (graph traversal for root cause)
    logger.info("\nGrouping PGs by temporal order (graph traversal)...")
    grouped_pgs = group_pgs_temporal(all_window_stats, collectives_to_order)
    logger.info(f"  {len(grouped_pgs)} causal paths found")

    # 5. Output
    if verbose:
        print_detailed(all_window_stats, collectives_to_order, pg_filter)

    print_summary(all_window_stats, collectives_to_order, grouped_pgs, pg_filter)

    # 6. Ground truth
    ground_truth = load_ground_truth(trace_dir)
    print_ground_truth_comparison(all_window_stats, ground_truth)


def main():
    parser = argparse.ArgumentParser(description="FR Straggler Analyzer")
    parser.add_argument("trace_dir", help="Path to trace directory containing _dump_*.json files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed per-rank per-window output")
    parser.add_argument("--pg", default=None, help="Filter output to PGs matching this substring")
    parser.add_argument("--threshold-ms", type=float, default=5.0,
                        help="Straggler detection threshold in ms (default: 5.0)")
    args = parser.parse_args()

    # Auto-save output to {trace_dir}/analysis.log alongside run_config.log
    log_path = os.path.join(args.trace_dir, "analysis.log")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Auto-saving output to {log_path}")

    analyze(args.trace_dir, verbose=args.verbose, pg_filter=args.pg, threshold_ms=args.threshold_ms)


if __name__ == "__main__":
    main()
