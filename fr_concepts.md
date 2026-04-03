# FR Concepts Quick Reference

Built up from weeks of trace-walking, code-reading, and discussion with sbak.

## 1. What Windowing Is Really Doing

Windowing is establishing an order of **what ranks were logically in the same temporal phase doing the same thing**. A PG tells you who *can* communicate together, but not who *was actually doing the same thing at the same time*. Windowing reconstructs that temporal grouping by replaying all ranks' timelines.

Most of the time, the PG membership is sufficient — for TP and EP, all member ranks execute in lockstep within a microbatch, so "same PG" = "same thing at the same time." But with PP-induced microbatch skew (interleaved 1F1B), even ranks within the same PG may be processing different microbatches at different times. When PP isn't sufficient to tell you, windowing goes deeper: look at which ranks actually showed up together, not just who's in the PG roster. For all other parallelism types, the PG directly tells you all ranks in it are doing the same thing.

**The wavefront selection is a time-based ordering**: each active PG gets greedily consumed before moving on. The algorithm picks whichever PG instance the most ranks are currently pointing at, consumes those entries, then moves to the next. This produces a temporal order where each PG wavefront "happens before" the next.

**Why you can't just look at a single collective in isolation**: in real traces, different ranks have other PGs' collectives and p2p ops interleaved before and after. Single-record comparison doesn't tell you "which wave of which PG this belongs to." The window gives you the bucket within which cross-rank comparison is meaningful.

## 2. The Two ID Systems

**megatron_id** (= `process_group[0]` in entries, = key in `pg_config`):
- Framework-level, globally unique across the job
- megatron_id=35 always means TP with ranks [0,1], everywhere
- `(desc, ranks)` uniquely identifies a PG instance and is in bijection with megatron_id — Megatron would never create two PGs with the same desc and ranks, that would be a bug

**c10d_handle** (= `pg_id` in entries, = key in `pg_status`):
- Backend-level, local to each rank
- c10d_handle=5 means "the 5th PG this rank created"
- **c10d_handle determines what PG *type/slot* it is on a given rank (the 5th PG = TP), but not which *instance*** — different megatron_ids map to the same c10d_handle across ranks (rank 0's TP[0,1] = megatron_id 35 → c10d=5; rank 4's TP[4,5] = megatron_id 37 → c10d=5)
- Within a single rank, megatron_id and c10d_handle are in bijection. Across ranks, only megatron_id is reliable.

**The chord analogy**: A PG instance is a chord — say C major = ranks [0, 1]. c10d_handle tells you "this is a major chord" (the slot/type). megatron_id tells you "this is *C* major specifically" (the instance). All notes (ranks) in the chord must sound together.

## 3. The Three Levels of Counting

**collective_seq_id** — per (rank, PG instance), counts individual collectives.
"This is the 84920th collective on TP35."
- Effectively per-PG-instance because all members increment in lockstep — a collective is a joint operation, both sides enqueue or neither does. The trace confirms it: rank 0 and rank 1 have identical TP seq_ids.
- **The definitive proof it's per-PG**: if it were per-rank (global counter across all PGs), the ring buffer would read 1, 2, 3, 4... monotonically. Instead you see 29, 29, 194, 84920, 84921... — each PG has its own counter, jumping between wildly different ranges as the CPU switches PGs.
- For p2p: multiple entries share the same collective_seq_id (one logical PP communication = multiple p2p ops = one seq_id). `p2p_seq_id` distinguishes individual send/recv within one logical collective. **This is the special thing about PP entries — not that they share megatron_id (every entry on the same PG does), but that multiple entries share the same seq_id.**
- **Cannot be used for cross-rank alignment**: different PG instances diverge (TP[0,1] at ~39407 vs TP[4,5] at ~30835 because different PP stages do different amounts of TP work per step), and p2p participation causes further divergence.

**window_idx** — per PG instance, counts logical phases/iterations.
"This is TP35's 2nd appearance as a group in the schedule."
- Megatron creates PGs once at initialization (`torch.distributed.new_group`) and reuses them for the entire job lifetime. megatron_id=35 is the same Python object from step 1 to step 10000. Without window_idx, different iterations' entries all tagged megatron_id=35 would get lumped together.
- window_idx separates "TP[0,1] in iteration N" from "TP[0,1] in iteration N+1."
- It's like a higher-level seq_id: seq_id counts notes within a playing, window_idx counts how many times the chord has been played.
- **It's per-PG, not a global iteration counter** — TP35 could be at window 2 while PP39 is still at window 1. In practice, for a well-structured training loop, window indices roughly correspond to iteration boundaries, but the mechanism is per-PG.

**global idx** (from `collectives_to_order`, lines 195-204) — cross-PG temporal order.
"This group was the 7th thing that happened in the reconstructed global schedule."
- Sequential number assigned in wavefront consumption order
- Used in `group_pgs()` for the DFS monotonicity constraint (only traverse from lower to higher scheduling order)

**Hierarchy**: global idx (cross-PG order) > window_idx (within-PG iteration boundary) > seq_id (individual operations within one window).

## 4. Why Graph Traversal Works: The Independent Fault Model

For each PG/rank that's missing, only two possibilities exist:
- It's an **independent root cause** (broke on its own)
- It was **infected/blocked** by an earlier PG/rank that failed first

**The key insight**: we deduce all missing ranks specifically to **exclude all dependently-broken ranks/PGs** — i.e., all the ones that were infected by the cascade. What we're actually searching for is the **independent faulty rank** (at least in the FR/comm world).

If a rank is the *earliest* to go missing (head of a causal chain), it cannot have been infected by anything later. It either independently failed, or it independently failed AND infected others downstream.

**The practical assumption that makes this work** (from experience and the Meta paper): in real large-scale training, **typically only 1 independent root cause exists, which then infects potentially thousands of other ranks**. Under this assumption, the head PG's missing ranks from each path = the answer. If there are multiple paths, the true root cause is necessarily among the union of all heads' missing ranks.

**Important caveat**: this is a heuristic attribution, not a mathematical proof. The graph connects PGs that share any rank (maximally pessimistic edge construction — assumes any shared rank can transmit failure). The algorithm reports the earliest symptomatic PG as root cause by policy, which is justified by the single-root-cause assumption but not proven for all cases.

## 5. Window Splitting: Why and When

Two conditions trigger a new window for a PG:
1. **Same ranks come back** (`already_participated`): the PG reappears with the same ranks → new iteration
2. **≥2 new ranks appear** (`has_significant_new_ranks >= 2`): a batch of previously-unseen ranks joins → different microbatch/PP phase

**The ≥2 new ranks guard is fundamentally a PP guardrail.** For TP/EP, all member ranks execute in lockstep — you'd never see "new ranks" appearing (either all are there or none). The skew only arises when a PG spans multiple PP stages processing different microbatches at different times.

**What this edge case means**: even within the same PG (same megatron_id, same desc, same ranks list), because of interleaved 1F1B, different microbatches mean different ranks are doing different things at different times. So logically they're in different temporal phases, even though they're under the same PG umbrella. Window splitting catches this.

**Key implementation detail**: this isn't hardcoded with if/else to distinguish p2p vs collective. The `pgs_with_active_ranks_last_iter` state variable naturally does the right thing — PP groups get split into multiple windows because they "disappear and come back," while TP/EP groups stay in the active set continuously and don't trigger the gate.

## 6. The Two-Window Hierarchy for Straggler Detection

There are two completely different "window" concepts that must not be conflated:

**Runtime capture window** (N steps / T seconds): "From now on, for the next N training steps, enable FR+cudaEvent timing, then dump." This just selects which temporal segment of the training run to collect detailed traces for. It's the observation interval.

**Offline analysis window** (`group_collectives_by_windows`): Within the captured dump, partition collectives into `(PG_type, sub_group, window_idx)` buckets by wavefront replay. This aligns "which collectives are the same logical round" for cross-rank comparison.

Relationship: "N steps / T seconds is just cutting a segment from the full timeline to zoom into. Inside that segment, `group_collectives_by_windows` slices finer logical windows by PG/phase."

## 7. FR Dump Structure

Each rank produces one dump file (`_dump_N.json`):

**`pg_config`**: All PGs defined in this job (full set from init). Key = megatron_id. The "roster."

**`pg_status`**: Only PGs this rank has actually used (subset). Key = c10d_handle. Contains `last_enqueued_collective`, `last_completed_collective` — these persist even after ring buffer entries are evicted, filling the gap.

**`entries`**: Ring buffer (default 2000). Only the tail — recent entries. Both ID systems present: `pg_id` (c10d_handle) and `process_group[0]` (megatron_id).

**Participating ranks are derived, not stored**: FR dumps don't record who participated. Windowing cross-references all rank dumps, groups entries by (PG, window), and counts who showed up vs. who should have (from pg_config.ranks). This is one layer above FR — precise, not heuristic.

## 8. The Three Timestamps

| Timestamp | Source | What it marks | Always available? |
|-----------|--------|---------------|-------------------|
| `time_created_ns` | Host CPU `clock_gettime` | CPU created the collective record | Yes (default FR) |
| `time_discovered_started_ns` | cudaEvent poll by watchdog | GPU comm kernel began | Only with `TORCH_NCCL_ENABLE_TIMING` |
| `time_discovered_completed_ns` | cudaEvent poll by watchdog | GPU comm kernel finished | Only with `TORCH_NCCL_ENABLE_TIMING` |

Host-side analysis (who enqueued late) doesn't need cudaEvent. GPU-side analysis (comm kernel execution time) does.

Null values mean the watchdog hasn't polled yet, or timing wasn't enabled. sbak: "FR needs to enable timings only when measuring timings."

## 9. FR Overhead: Three Layers

**Layer 1 — Ring buffer metadata (always on, zero cost)**: Just bookkeeping in C10D's own data structures for timeout detection. One entry per collective call, fixed-size buffer.

**Layer 2 — CUDA events for completion (already in PyTorch, minimal)**: PyTorch records CUDA events on the NCCL stream for `work.wait()`. One lightweight timestamp marker per collective, not a kernel launch. This is what populates `state` (scheduled/started/completed).

**Layer 3 — Timing discovery (watchdog polls, opt-in via `TORCH_NCCL_ENABLE_TIMING`)**: Watchdog calls `cudaEventQuery` (non-blocking) and `cudaEventElapsedTime`. Populates `time_discovered_started/completed_ns`.

**Key contrast**: one all_reduce = one FR entry, regardless of how many NCCL kernels it generates. CUPTI intercepts every kernel launch (thousands per step). 1-2 orders of magnitude difference.

## 10. Why CUPTI Can't Do Distributed Straggler Detection

CUPTI records per-GPU kernel name + duration. No PG, no rank mapping, no collective identity. You can't correlate rank 0's `ncclDevKernel_AllReduce_Sum` with rank 1's — same kernel name doesn't mean same logical collective (could be different AllReduces on different PGs).

**The NCCL duration paradox**: CUPTI shows rank A's NCCL kernel = 3369μs, rank B's = 256μs. Rank A looks slow. But rank A may have *arrived first* and spent most of that time spin-waiting inside the kernel for others. Rank B arrived late but finished quickly. Without knowing entry time, duration alone is ambiguous. That's why NVRx filters NCCL kernels out of CUPTI scoring entirely.

FR directly matches: "rank 0 and rank 1 both called `nccl:_all_gather_base` on PG 35 with collective_seq_id=39407." Same operation, compare timestamps.

**The missing context is fatal for straggler detection**: straggler detection requires matching the same logical operation across ranks — "for this specific AllReduce on the TP group, which rank was late?" CUPTI fundamentally cannot answer this. FR can.

**The right analysis granularity is PG-level, not p2p-level.** P2Ps within one collective are independent (rank 0→1 doesn't depend on rank 0→2). Causality flows between PGs (via shared ranks and scheduling order), not between individual p2p operations. Graph traversal works at PG level: "is EP blocked because TP before it (sharing ranks) hasn't completed?"

## 11. Ring Buffer Limitation for Straggler Detection

Default 2000 entries — enough for fault attribution (snapshot at crash), NOT for straggler detection (need to observe patterns over multiple iterations). sbak: "It's set to 2000 which is enough for fault analysis. Not enough for straggler. And trace dump should happen by trainer."

Ring buffer tail (oldest surviving entry) is closest to the root cause in attribution. For straggler, extending the buffer or dumping periodically from the trainer is needed.

## 12. Why seq_id Cross-Rank Matching Fails

**The p2p problem — concrete from gpu_error_1st trace**: During the PP phase, ranks 0, 1, 2 all have collective_seq_id=26, but their p2p operations happen at completely different times (rank 0/2 at t≈0s, rank 1 at t≈3.5s) with different communication partners. Using "csid=26 = same round" lumps operations separated by 3.5s, causing false missing detection and window corruption.

**Why non-flat PGs break seq_id alignment**: sbak's EP/CP example — rank 0 participates in TP, EP, and CP. Rank 8 only in TP (not in EP or CP). Each PG has its own counter, and different ranks weave through different sets of PGs at different paces. TP seq=3 on rank 0 happens at a completely different point in the global schedule than TP seq=3 on rank 8.

The old approach (`default_pg_order` + cross-rank seq_id) solved PG type ordering but not the finer problem: within the same PG type, which entries across ranks correspond to the same logical round. Windowing solves both levels — it's not just a replacement for pg_order, it's a deeper layer.

## 13. Test Traces Reference

16 GPUs, TP=2, PP=4, DP=2. TP pairs: [0,1] [2,3] ... [14,15]. PP groups: [0,4,8,12] [1,5,9,13] [2,6,10,14] [3,7,11,15].

In p2p profiling_name (`recv 2<-3`): numbers are PG-internal local rank indices, NOT global rank IDs.

Why TP does three kinds of collectives: `_all_gather_base` (before ColumnParallelLinear — gather sharded weight), `_reduce_scatter_base` (after RowParallelLinear — reduce partial results), `broadcast` (unsharded params like LayerNorm). All on the same TP PG instance, sharing one seq_id counter.

| Trace | Missing Ranks | Source |
|-------|--------------|--------|
| gpu_error_1st | {12, 14} | fault_injection.log |
| gpu_error_2nd | {9, 14} | fault_injection.log |
| lock_gil_1st | {9, 14} | fault_injection.log |
| lock_gil_2nd | {10, 15} | fault_injection.log |
