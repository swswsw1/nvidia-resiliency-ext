"""Block slicer for FR trace dumps.

Shared module used by both the online monitor (post-hoc, slicing many chunks
across many blocks) and the trainer (in-process, extracting a single most-
recent block per dump trigger).

Block model
-----------
A block is the interval bounded by two consecutive default_pg occurrences,
half-open on the left and closed on the right: (prev_default_pg, this_default_pg].
Block 1 is (-inf, first_default_pg]. Block 0 (entries before any default_pg)
is intentionally NOT produced — pre-init noise without a terminal anchor.

`block_seq_id` (= the terminal default_pg's `collective_seq_id`) is the
canonical cross-rank block identifier. Collective seq_ids increment in
lockstep across all participants of a collective PG, so the i-th default_pg
barrier carries the same seq_id on every rank — usable as a stable key
even if rank 0's chunk index 7 corresponds to rank 3's chunk index 6, etc.

Public API
----------
- merge_rank_chunks(chunk_paths)              # file-based: combine on-disk chunks → one rank's view
- merge_rank_snapshots(snapshots)             # in-memory: combine parsed FR dicts → one rank's view
- slice_into_blocks(per_rank_merged)          # all blocks across whole trace
- select_last_complete_block(blocks)          # most recent block where every rank has the terminal
- extract_last_block_from_dump(per_rank_merged)   # convenience: slice + select last complete
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# default_pg recognition (PG-desc lookup, not c10d-handle)
# ---------------------------------------------------------------------------

def _is_default_pg_entry(entry: dict) -> bool:
    """True if this FR entry belongs to the default_pg (world) group.

    Identified by `process_group[1] == "default_pg"`. Using pg_desc rather
    than pg_id (c10d handle, which is rank-local) means cross-rank matching
    works without ID translation.
    """
    pg = entry.get("process_group", [])
    return len(pg) >= 2 and pg[1] == "default_pg"


# ---------------------------------------------------------------------------
# Per-rank chunk merger
# ---------------------------------------------------------------------------

_ITER_RE = re.compile(r"_iter(\d+)\.json$")


def _merge_iter_tagged_snapshots(
    iter_tagged_snapshots: List[Tuple[int, dict]],
) -> dict:
    """Shared core for both `merge_rank_chunks` (file-based) and
    `merge_rank_snapshots` (in-memory).

    Args:
        iter_tagged_snapshots: list of (iter_num, snapshot_dict) tuples in
            chronological order. `iter_num=-1` indicates "no iter info";
            entries from such snapshots will be tagged with `_iter_num=-1`.

    Returns the merged trace dict. See `merge_rank_chunks` for the contract.
    """
    latest: Dict[Tuple, dict] = {}
    earliest_iter: Dict[Tuple, int] = {}
    pg_config: dict = {}
    pg_status: dict = {}
    for iter_num, snapshot in iter_tagged_snapshots:
        pg_config = snapshot.get("pg_config", pg_config) or pg_config
        pg_status = snapshot.get("pg_status", pg_status) or pg_status
        for e in snapshot.get("entries", []):
            key = (e.get("pg_id"), e.get("collective_seq_id"), e.get("p2p_seq_id", -1))
            latest[key] = e   # last-write-wins on content (for GPU timing maturation)
            if key not in earliest_iter or iter_num < earliest_iter[key]:
                earliest_iter[key] = iter_num
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


def merge_rank_chunks(chunk_paths: Dict[int, str]) -> dict:
    """Merge multiple FR-buffer snapshot chunks for one rank into a single trace.

    Each chunk is a full FR ring-buffer snapshot, so entries overlap across
    chunks. The merger keeps the LATEST version of each entry, keyed by
    (pg_id, collective_seq_id, p2p_seq_id). Rationale: GPU timing fields
    are populated asynchronously by the NCCL watchdog, so the most recent
    snapshot of a given entry has the most complete timing.

    Each merged entry is tagged with `_iter_num` = the smallest training
    iteration index at which this entry was first seen (parsed from the
    chunk filename via `_iter(\\d+)\\.json$`). Useful for downstream
    consumers that want to know when an entry first appeared.

    File-I/O variant; for trainer-side in-process use see `merge_rank_snapshots`.

    Args:
        chunk_paths: {chunk_idx (int): file path (str)} for one rank's chunks.

    Returns:
        A dict structurally identical to a legacy `_dump_{rank}.json`:
            {
                "pg_config": {...},   # latest non-empty pg_config seen
                "pg_status": {...},   # latest non-empty pg_status seen
                "entries":   [...],   # deduplicated and seq-sorted
            }
    """
    chunk_paths_sorted = sorted(chunk_paths.items())  # by chunk idx (time)
    iter_tagged: List[Tuple[int, dict]] = []
    for _, path in chunk_paths_sorted:
        m = _ITER_RE.search(path)
        chunk_iter = int(m.group(1)) if m else -1
        with open(path) as f:
            iter_tagged.append((chunk_iter, json.load(f)))
    return _merge_iter_tagged_snapshots(iter_tagged)


def merge_rank_snapshots(
    snapshots: List[dict],
    iter_nums: Optional[List[int]] = None,
) -> dict:
    """In-memory variant of `merge_rank_chunks` for trainer-side use.

    Accepts already-parsed FR-buffer dicts (e.g. from
    `pickle.loads(torch._C._distributed_c10d._dump_nccl_trace(...))`),
    skipping file I/O entirely. Same merge semantics as `merge_rank_chunks`:
    last-write-wins on entry content keyed by
    (pg_id, collective_seq_id, p2p_seq_id), and `_iter_num` records the
    earliest iteration each entry was seen.

    Args:
        snapshots: list of FR-buffer dicts in chronological order (oldest
            first). Each dict has `entries`, `pg_config`, `pg_status`.
        iter_nums: optional parallel list of training iteration indices.
            If omitted, snapshots are tagged with `_iter_num = position in
            the list` (0-indexed).

    Returns the merged trace dict — see `merge_rank_chunks` for the contract.
    """
    if iter_nums is None:
        iter_nums = list(range(len(snapshots)))
    elif len(iter_nums) != len(snapshots):
        raise ValueError(
            f"iter_nums length ({len(iter_nums)}) must match snapshots length ({len(snapshots)})"
        )
    return _merge_iter_tagged_snapshots(list(zip(iter_nums, snapshots)))


# ---------------------------------------------------------------------------
# Block slicing
# ---------------------------------------------------------------------------

def slice_into_blocks(
    per_rank_merged: Dict[int, dict],
) -> List[dict]:
    """Slice merged per-rank traces into blocks bounded by default_pg occurrences.

    Block boundaries are aligned across ranks via the default_pg's
    `collective_seq_id` (which increments in lockstep on the same PG).
    Per rank, entries are included in block i if their `time_created_ns`
    falls in `(prev_default_pg_time_R, terminal_default_pg_time_R]`. A rank
    that has evicted the previous default_pg from its FR buffer gets a
    partial slice (everything up to the terminal); a rank that has evicted
    the terminal gets an empty slice. Both cases record warnings.

    Args:
        per_rank_merged: {rank_id (int): {"entries": [...], "pg_config": ...,
            "pg_status": ...}}. The structure produced by `merge_rank_chunks`.

    Returns:
        List of block dicts, sorted by `block_seq_id` ascending:
            {
                "block_id":     int,    # 1-indexed ordinal
                "block_seq_id": int,    # collective_seq_id of terminal default_pg
                "by_rank":      {rank: {"entries": [...], "pg_config": ...,
                                        "pg_status": ...}},
                "warnings":     [str, ...],
            }
        Block 0 (entries before any default_pg) is intentionally not produced.
    """
    # Discover all default_pg seq_ids seen on any rank.
    all_default_seqs = set()
    for rank, data in per_rank_merged.items():
        for e in data.get("entries", []):
            if _is_default_pg_entry(e):
                seq = e.get("collective_seq_id")
                if seq is not None:
                    all_default_seqs.add(seq)
    sorted_seqs = sorted(all_default_seqs)

    blocks: List[dict] = []
    for i, terminal_seq in enumerate(sorted_seqs):
        prev_seq: Optional[int] = sorted_seqs[i - 1] if i >= 1 else None
        block: dict = {
            "block_id": i + 1,
            "block_seq_id": terminal_seq,
            "by_rank": {},
            "warnings": [],
        }
        for rank, data in per_rank_merged.items():
            entries = data.get("entries", [])
            terminal_e = next(
                (e for e in entries
                 if _is_default_pg_entry(e) and e.get("collective_seq_id") == terminal_seq),
                None,
            )
            if terminal_e is None:
                # Rank evicted this default_pg or wasn't in the world group;
                # produce empty slice for this rank.
                block["by_rank"][rank] = {
                    "entries": [],
                    "pg_config": data.get("pg_config", {}),
                    "pg_status": data.get("pg_status", {}),
                }
                block["warnings"].append(
                    f"rank {rank} missing default_pg seq {terminal_seq}"
                )
                continue
            terminal_t = terminal_e["time_created_ns"]

            if prev_seq is None:
                start_t: int = -1   # -inf: include everything up to terminal
            else:
                prev_e = next(
                    (e for e in entries
                     if _is_default_pg_entry(e) and e.get("collective_seq_id") == prev_seq),
                    None,
                )
                if prev_e is None:
                    start_t = -1
                    block["warnings"].append(
                        f"rank {rank} missing prev default_pg seq {prev_seq} — partial block"
                    )
                else:
                    start_t = prev_e["time_created_ns"]

            block_entries = [
                e for e in entries
                if start_t < e.get("time_created_ns", 0) <= terminal_t
            ]
            block["by_rank"][rank] = {
                "entries": block_entries,
                "pg_config": data.get("pg_config", {}),
                "pg_status": data.get("pg_status", {}),
            }
        blocks.append(block)
    return blocks


# ---------------------------------------------------------------------------
# Selection helpers (trainer-side use)
# ---------------------------------------------------------------------------

def _block_is_complete(block: dict) -> bool:
    """True iff every rank in `by_rank` has a non-empty slice for this block.

    A complete block = the terminal default_pg was present on every rank.
    Partial blocks (where some rank evicted the terminal) are not complete.
    """
    for rank_data in block["by_rank"].values():
        if not rank_data.get("entries"):
            return False
    return True


def select_last_complete_block(blocks: List[dict]) -> Optional[dict]:
    """Return the most recent block whose terminal default_pg is present on
    every rank. Returns None if no complete block exists.

    Intended for trainer-side use: after dumping the FR buffer at a trigger
    point, the trainer wants to write *one* finalized block to disk — the
    latest one for which every rank's view is intact. Earlier complete
    blocks would already have been written by previous triggers; later
    blocks may still be in-flight.
    """
    for block in reversed(blocks):
        if _block_is_complete(block):
            return block
    return None


def extract_last_block_from_dump(
    per_rank_merged: Dict[int, dict],
) -> Optional[dict]:
    """Convenience for trainer-side: slice into blocks, return the last complete one.

    Equivalent to `select_last_complete_block(slice_into_blocks(per_rank_merged))`.
    """
    return select_last_complete_block(slice_into_blocks(per_rank_merged))


__all__ = [
    "merge_rank_chunks",
    "merge_rank_snapshots",
    "slice_into_blocks",
    "select_last_complete_block",
    "extract_last_block_from_dump",
]
