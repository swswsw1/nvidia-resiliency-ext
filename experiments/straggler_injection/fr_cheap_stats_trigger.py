"""Cheap-stats trigger for FR straggler detection.

Monkey-patches `dist.{all_reduce, barrier, broadcast}` for the default_pg
(world) group only, times each call with CUDA events without
`cuda.synchronize()`, and triggers a single FR ring-buffer dump per
training step when the same candidate-straggler rank persists across
multiple cross-rank evaluations.

Straggler-inversion (see fr_concepts.md §14)
--------------------------------------------
In a synchronous collective, the rank that arrived LAST has the SHORTEST
kernel wait time — its kernel finishes almost immediately because every
other rank is already at the barrier. So we identify the candidate
straggler as the rank with the smallest mean wait time within the window.

Pipeline
--------
For each timed call:
  start_evt.record() → original_call() → end_evt.record()
  Push (start_evt, end_evt) onto a deque.
On the NEXT timed call (and at every eval), drain completed events with
`event.query()` (non-blocking) and append `start.elapsed_time(end)` to
the rolling buffer. No `cuda.synchronize()`, no per-call CPU stalls.

When buffer is full and `call_count % check_freq == 0`:
  - all_gather local-mean wait time across ranks
  - argmin → candidate straggler
  - push into history deque (size = persistence)
  - if history is full AND unanimous → set dump_requested = True

Lifecycle
---------
    trigger = StragglerTrigger(window_size=20, check_freq=5, persistence=3,
                               log_path="cheap_stats.jsonl")
    trigger.patch_distributed()                  # BEFORE importing DDP
    # ... training loop runs ...
    if trigger.consume_dump_request():
        dump_one_block(rank, trace_dir, step)
    trigger.unpatch()                            # at shutdown (optional)

All ranks run their own StragglerTrigger; the in-evaluator all_gather
synchronizes their views, so trigger decisions match across ranks and
they all dump simultaneously.
"""

from __future__ import annotations

import inspect
import json
import os
import pickle
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_default_pg(group: Any) -> bool:
    """True if `group` refers to the world (default) PG. None is the implicit
    default used when callers omit the kwarg."""
    if group is None:
        return True
    try:
        return group is dist.group.WORLD
    except Exception:
        return False


# ---------------------------------------------------------------------------
# StragglerTrigger
# ---------------------------------------------------------------------------

class StragglerTrigger:
    """In-process cheap-stats trigger. One instance per rank."""

    def __init__(
        self,
        window_size: int = 20,
        check_freq: int = 5,
        persistence: int = 3,
        device_id: int = 0,
        log_path: Optional[str] = None,
        offset_ms: float = 1.0,
    ):
        if window_size < 1 or check_freq < 1 or persistence < 1:
            raise ValueError("window_size, check_freq, persistence must all be >= 1")
        if offset_ms <= 0:
            raise ValueError("offset_ms must be > 0")
        self._window_size = window_size
        self._check_freq = check_freq
        self._persistence = persistence
        self._device_id = device_id
        self._log_path = log_path
        # Option 3 — one-sided absolute threshold above median.
        # A rank is flagged when its mean wait exceeds `median + offset_ms`.
        # Why one-sided + absolute (not two-sided MAD):
        #   - MAD breaks in bimodal data (N/2 stragglers): median sits between
        #     the fast and slow clusters, every rank's deviation ≈ MAD, no rank
        #     stands out. Empirically observed with 2/4 host injection.
        #   - Two-sided flagging in bimodal data flags every rank (each is far
        #     from median in some direction) — no information.
        #   - At our probe point (end-of-iter dist.barrier), slow ranks have
        #     LONGER wait times (intra-iter collectives resync first; residual
        #     reflects GPU-stream-queue position). So one-sided above-median is
        #     the right sign here. §14's inversion does not hold at this probe.
        #   - The trigger only needs to fire; the cascade analyzer picks which
        #     rank is actually the root cause from the dumped block.
        self._offset_ms = offset_ms

        # Rolling state
        self._buffer: Deque[float] = deque(maxlen=window_size)
        self._pending_events: Deque[Tuple[torch.cuda.Event, torch.cuda.Event]] = deque()
        # Per-rank consecutive-flag counters. A rank's counter increments on
        # every eval where it's flagged as outlier (either direction) and
        # resets to 0 on any eval where it's NOT flagged. Trigger fires when
        # any rank's counter reaches persistence.
        self._flag_counters: Dict[int, int] = {}

        # Bookkeeping
        self._call_count = 0
        self._eval_count = 0
        self._dump_requested = False
        self._last_triggered_call_count = -1   # dedup: don't re-trigger on same step

        # Patching state
        self._patched = False
        self._originals: Dict[str, Callable] = {}

        # Lazy-init state (populated on first timed call)
        self._lazy_done = False
        self._rank: int = -1
        self._world_size: int = -1
        self._device: Optional[torch.device] = None
        # Dedicated process group for the evaluator's all_gather. Created at
        # lazy init via dist.new_group() so the all_gather DOESN'T land on
        # default_pg. If we used default_pg here, every eval would inject a
        # default_pg entry into the FR ring buffer, contaminating the
        # block boundaries that dump_one_block uses to slice.
        self._eval_group: Any = None

    # ----- Patching --------------------------------------------------------

    def patch_distributed(self) -> None:
        """Replace dist.{all_reduce, barrier, broadcast} with timing wrappers.
        Saves dist.all_gather (used by the evaluator) so we never recurse
        through the patched path even if all_gather itself is patched later.

        Call BEFORE importing DDP / framework code so wrappers are seen.
        """
        if self._patched:
            return
        # Save originals
        self._originals['all_reduce'] = dist.all_reduce
        self._originals['barrier'] = dist.barrier
        self._originals['broadcast'] = dist.broadcast
        self._originals['all_gather'] = dist.all_gather   # for evaluator use

        # Install wrappers
        dist.all_reduce = self._make_wrapper(dist.all_reduce, 'all_reduce')
        dist.barrier = self._make_wrapper(dist.barrier, 'barrier')
        dist.broadcast = self._make_wrapper(dist.broadcast, 'broadcast')
        self._patched = True

    def unpatch(self) -> None:
        """Restore originals. Idempotent."""
        if not self._patched:
            return
        for name in ('all_reduce', 'barrier', 'broadcast'):
            setattr(dist, name, self._originals[name])
        self._patched = False

    def _make_wrapper(self, original: Callable, name: str) -> Callable:
        sig = inspect.signature(original)

        def wrapper(*args, **kwargs):
            # Resolve `group` from positional or keyword args.
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                group = bound.arguments.get('group', None)
            except TypeError:
                group = kwargs.get('group', None)

            if not _is_default_pg(group):
                return original(*args, **kwargs)
            return self._timed_call(original, args, kwargs)

        wrapper.__wrapped__ = original
        wrapper.__name__ = name
        wrapper.__qualname__ = f"StragglerTrigger.wrapped_{name}"
        return wrapper

    # ----- Lazy init -------------------------------------------------------

    def _lazy_init(self) -> bool:
        """Populate rank/world/device + create dedicated eval PG on first
        timed call. Returns False if dist isn't initialized yet (wrapper
        falls through to plain pass-through for that call).

        The eval PG is created via dist.new_group() with all ranks but a
        distinct megatron_id from default_pg. Its entries appear in FR
        with the new group's desc, NOT "default_pg", so dump_one_block's
        block-boundary detection skips over them.
        """
        if self._lazy_done:
            return True
        if not (dist.is_available() and dist.is_initialized()):
            return False
        try:
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
            self._device = torch.device(f'cuda:{self._device_id}')
            # Create dedicated eval PG. dist.new_group() is COLLECTIVE — all
            # ranks must call it. Since _lazy_init runs on the first timed
            # call (which is itself a collective), all ranks reach this
            # point in the same iteration. NCCL backend matches default_pg.
            self._eval_group = dist.new_group(
                ranks=list(range(self._world_size)),
                backend='nccl',
            )
            self._lazy_done = True
            return True
        except Exception:
            return False

    # ----- Timed call ------------------------------------------------------

    def _timed_call(self, original: Callable, args: tuple, kwargs: dict):
        if not self._lazy_init():
            return original(*args, **kwargs)

        self._drain_pending()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        result = original(*args, **kwargs)
        end_evt.record()
        self._pending_events.append((start_evt, end_evt))
        self._call_count += 1

        # Re-drain in case the events we just enqueued already fired (small kernels).
        self._drain_pending()

        if (len(self._buffer) >= self._window_size
                and self._call_count % self._check_freq == 0
                and self._call_count != self._last_triggered_call_count):
            self._evaluate()

        return result

    def _drain_pending(self) -> int:
        """Pop completed events from the head of the queue. Non-blocking
        (`event.query()` returns immediately). Stops at the first un-completed
        event so we never reorder timings."""
        drained = 0
        while self._pending_events:
            start_evt, end_evt = self._pending_events[0]
            if not end_evt.query():
                break
            duration_ms = start_evt.elapsed_time(end_evt)
            self._buffer.append(duration_ms)
            self._pending_events.popleft()
            drained += 1
        return drained

    # ----- Cross-rank evaluation ------------------------------------------

    def _evaluate(self):
        """Two-sided MAD outlier detection + per-rank persistence counters.

        Replaces argmin+unanimity. Why:
          - argmin picks one rank per eval, which can't represent multi-rank
            stragglers and flaps under tiny float jitter (see 20260512 two-rank
            experiment: argmin alternated rank 0/1, silenced unanimity).
          - The empirical sign of "straggler ↔ wait time" is not stable —
            §14 of fr_concepts.md predicts shortest=straggler at a pure barrier
            probe, but at the end-of-iter barrier after fwd/bwd the slow ranks
            actually have *longer* wait times (intra-iter collectives resync
            then the residual is GPU-stream-queue-position).
        Two-sided MAD threshold handles both cases without committing to a sign:
        any rank far from the median (in either direction) is flagged.
        """
        local_mean = sum(self._buffer) / len(self._buffer)

        # all_gather one float per rank using the SAVED original (defensive
        # against future patching) AND the dedicated eval PG (so this call
        # doesn't land on default_pg and contaminate block boundaries).
        local_t = torch.tensor([local_mean], dtype=torch.float64, device=self._device)
        gathered = [torch.zeros_like(local_t) for _ in range(self._world_size)]
        try:
            self._originals['all_gather'](gathered, local_t, group=self._eval_group)
        except Exception as e:
            self._log({"eval_idx": self._eval_count, "skipped": True, "reason": str(e)})
            return
        wait_times = [t.item() for t in gathered]

        # Compute median across ranks; threshold = median + offset.
        sorted_w = sorted(wait_times)
        n = len(sorted_w)
        median = sorted_w[n // 2] if n % 2 else 0.5 * (sorted_w[n // 2 - 1] + sorted_w[n // 2])
        threshold = median + self._offset_ms

        # Flag every rank whose wait exceeds median + offset (one-sided slow).
        # For each rank: increment counter if flagged this eval, reset to 0
        # otherwise. Trigger fires when ANY rank's counter hits persistence.
        flagged_this_eval = [
            r for r, w in enumerate(wait_times) if w > threshold
        ]
        flagged_set = set(flagged_this_eval)
        for r in range(self._world_size):
            if r in flagged_set:
                self._flag_counters[r] = self._flag_counters.get(r, 0) + 1
            else:
                self._flag_counters[r] = 0

        self._eval_count += 1

        # Rank(s) whose persistence counter just reached the threshold.
        triggering_ranks = sorted(
            r for r, c in self._flag_counters.items() if c >= self._persistence
        )
        triggered = bool(triggering_ranks)
        if triggered:
            self._dump_requested = True
            self._last_triggered_call_count = self._call_count

        self._log({
            "eval_idx": self._eval_count,
            "call_count": self._call_count,
            "wait_times_ms": wait_times,
            "median_ms": median,
            "threshold_ms": threshold,
            "flagged_ranks": flagged_this_eval,
            "flag_counters": dict(sorted(self._flag_counters.items())),
            "triggering_ranks": triggering_ranks,
            "buffer_local_mean_ms": local_mean,
            "dump_triggered": triggered,
        })

    def _log(self, record: dict) -> None:
        # Only rank 0 writes (all_gather already broadcast cross-rank info).
        if self._log_path is None or self._rank != 0:
            return
        try:
            with open(self._log_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except OSError:
            pass

    # ----- Public access --------------------------------------------------

    def consume_dump_request(self) -> bool:
        """Single-shot read: returns True iff a dump was requested since the
        last call. Resets the flag on read."""
        v = self._dump_requested
        self._dump_requested = False
        return v

    def stats_summary(self) -> dict:
        """Snapshot of current state. For debugging / tests."""
        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "patched": self._patched,
            "lazy_done": self._lazy_done,
            "call_count": self._call_count,
            "eval_count": self._eval_count,
            "buffer_size": len(self._buffer),
            "buffer_window_size": self._window_size,
            "check_freq": self._check_freq,
            "persistence": self._persistence,
            "offset_ms": self._offset_ms,
            "flag_counters": dict(self._flag_counters),
            "pending_events": len(self._pending_events),
            "dump_requested": self._dump_requested,
        }


# ---------------------------------------------------------------------------
# Trigger-driven dump path
# ---------------------------------------------------------------------------

def dump_one_block(rank: int, trace_dir: str, step: int) -> Optional[str]:
    """Snapshot the current FR ring buffer, slice into blocks, write only
    the most recent complete block to disk. Per-rank — each rank dumps its
    own data at the same logical step (the trigger has already
    cross-rank-aligned via all_gather).

    Filename: `_dump_{rank}_step{step:06d}.json`. The payload is structured
    identically to the monitor's per-rank-block output (`pg_config`,
    `pg_status`, `entries`) plus two trigger metadata fields.

    Returns the output path, or None if no complete block was available
    in the buffer (e.g. trigger fired before any default_pg landed).
    """
    # Lazy import keeps module import cheap and dependency-free.
    from fr_block_slicer import (
        merge_rank_snapshots, slice_into_blocks, select_last_complete_block,
    )

    pickle_bytes = torch._C._distributed_c10d._dump_nccl_trace(  # type: ignore[attr-defined]
        includeCollectives=True,
        includeStackTraces=False,
        onlyActive=False,
    )
    snapshot = pickle.loads(pickle_bytes)
    merged = merge_rank_snapshots([snapshot])
    blocks = slice_into_blocks({rank: merged})
    last = select_last_complete_block(blocks)
    if last is None:
        return None

    rank_data = last["by_rank"][rank]
    payload = {
        "pg_config": rank_data["pg_config"],
        "pg_status": rank_data["pg_status"],
        "entries": rank_data["entries"],
        "_trigger_step": step,
        "_trigger_block_seq_id": last["block_seq_id"],
    }
    os.makedirs(trace_dir, exist_ok=True)
    out_path = os.path.join(trace_dir, f"_dump_{rank}_step{step:06d}.json")
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, out_path)
    return out_path


__all__ = ["StragglerTrigger", "dump_one_block"]
