# NOTES — future blockers

Properties of the current scope that are NOT bugs today but WILL become real at the
listed scale points. Track here so we don't rediscover them under pressure.

---

## 1. `merge_rank_chunks` is file-I/O bound — FIXED

**Resolved**: `merge_rank_snapshots(snapshots, iter_nums=None)` now lives in
`fr_block_slicer.py` alongside `merge_rank_chunks`. Both delegate to a shared
private core `_merge_iter_tagged_snapshots(list of (iter_num, dict))`. The
trainer-side trigger module can call `merge_rank_snapshots` with already-
parsed FR dicts and skip disk I/O entirely. Same merge semantics, same
`_iter_num` tagging rule.

---

## 2. "Complete block" = every rank has a non-empty slice

**Today**: `select_last_complete_block` requires every rank's slice for that block to
be non-empty. Strict by design. This works because:
- 8 ranks, FR ring buffer comfortably exceeds one iter's worth of entries.
- Trigger-driven dumps fire right after a `dist.barrier()`, so the triggering rank's
  view of the terminal default_pg is always fresh on every rank.

**Becomes relevant when**: scaling beyond ~100 ranks AND/OR running long enough that
some rank evicts older default_pg entries before the analyzer reads them. At that
scale, "complete" as currently defined will silently filter out blocks where 1 of N
ranks evicted, even when the other 99 had clean data.

**DO NOT relax now**. Strict is correct for the testbed. The relaxed form would be:
```python
def select_last_complete_block(blocks, min_rank_fraction=1.0):
    """Return latest block where ≥min_rank_fraction of ranks have non-empty slices."""
```
Add the threshold parameter only when there's a measured eviction problem, not
preemptively. The wrong threshold corrupts attribution silently.

---

## 3. Trigger persistence logic misses alternating stragglers and co-faults

**Today**: `fr_cheap_stats_trigger.StragglerTrigger._evaluate` does:
- argmin over per-rank mean wait times → exactly one straggler per eval
- Append to `history` deque, fire only if all entries unanimous

**Two failure modes this hides:**

a) **Alternating-straggler.** Two ranks both have meaningfully short wait times,
   close enough that argmin flips between them eval-to-eval:
   - Eval 1: argmin = r2, history = [2]
   - Eval 2: argmin = r3, history = [2, 3] → not unanimous, no fire
   - Eval 3: argmin = r2, history = [3, 2] (maxlen=2) → not unanimous
   - Eval 4: argmin = r3, history = [2, 3] → not unanimous
   - ...trigger NEVER fires, even though both ranks are real stragglers.

b) **Co-fault.** Two distinct stragglers fire simultaneously (e.g. inject r0 + r5).
   argmin picks ONE per eval (deterministically, whichever has smaller mean by a hair).
   The *other* straggler is invisible to the trigger. If the chosen one is not stable
   across evals — same problem as (a). Even when it is, the second straggler is
   silently dropped by the time the dump fires.

**Becomes mandatory when**: running multi-injection sweeps, or in production where
a slow rank and a slowly-failing rank appear together. The trainer already supports
`INJECT_RANKS=0,5` — the moment we exercise that, the trigger may not fire at all
on a real two-rank fault.

**Fix sketch (two coupled changes)**:

```python
# 1. Per-eval: median+MAD inversion → set of ranks (not single argmin)
median = sorted(wait_times)[len(wait_times) // 2]
mad = sorted([abs(w - median) for w in wait_times])[len(wait_times) // 2]
threshold = median - k * 1.4826 * mad   # below this = candidate straggler
flagged = {r for r, w in enumerate(wait_times) if w < threshold}

# 2. Persistence: sliding window of eval flag-sets, count-based threshold
recent_evals.append(flagged)              # deque(maxlen=M)
counts = Counter(r for fs in recent_evals for r in fs)
fired = [r for r, c in counts.items() if c >= N]
if fired:
    self._dump_requested = True
```

Defaults: `M=5, N=2` (rank flagged in 2 of last 5 evals → fire). Configurable
alongside the existing `window_size`, `check_freq`, `persistence` knobs.

**Why we're NOT fixing now**: the single-injection live pipeline works (7/7 hit,
100% accuracy on the host_r3 run). The fix touches both per-eval detection and
multi-eval persistence — two coupled changes. Worth doing under the pressure of
a real co-fault failure that gives us a concrete validation target, not against
synthesized data only.

**Trigger to fix**: first failed multi-injection run (e.g. `INJECT_RANKS=3,5`).
At that point, the broken case becomes a regression test, and the fix has a
real signal to optimize against.

---

## 4. `pg_id` is rank-local; never deduplicate across ranks at this layer

**Today**: `merge_rank_chunks` keys entries on `(pg_id, collective_seq_id, p2p_seq_id)`.
This works **only because the merger runs per-rank** — one rank's view at a time. The
keys are unique within that rank.

**The footgun**: `pg_id` is the c10d handle, which is **rank-local**. `pg_id=2` on
rank 0 might be the TP[0,1] PG, while `pg_id=2` on rank 7 is the TP[6,7] PG. They
are *different PGs that happen to share an integer handle*. Anyone who tries to
"optimize" by merging across ranks at this layer (e.g. building a global entry
table keyed on `pg_id`) will silently produce garbage.

**Cross-rank alignment uses different keys**:
- `process_group[0]` (megatron_id) — globally unique across the job
- `(megatron_id, collective_seq_id)` — identifies the same logical collective on
  every participating rank
- These are what `slice_into_blocks` uses to find the same default_pg barrier across
  ranks. See `fr_concepts.md` §2 for the full ID-system breakdown.

**Becomes a real problem when**: someone (future-us, future-collaborator) caches a
`pg_id → pg_desc` mapping at the global level instead of per-rank, or merges chunks
from multiple ranks into one entry list, or builds a multi-rank dedup index keyed
on `pg_id`. None of these are obvious mistakes — they all look like reasonable
optimizations. The damage is silent: entries get overwritten with semantically
unrelated data and the analyzer downstream sees a corrupted view that still
type-checks.

**Mitigation**: documented in `fr_block_slicer.py` module docstring. If we ever
cross-rank merge, switch the key to `(megatron_id, collective_seq_id, p2p_seq_id)`.
