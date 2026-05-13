# FR Concepts Quick Reference

  

  

Built up from weeks of trace-walking, code-reading, and discussion with sbak.

  

**Core framing**: Faults, stragglers, and hangs don't happen at kernel level — they originate and cascade through process groups. A straggler isn't "kernel X was slow," it's "rank R was late to PG Y's round." A fault isn't "kernel X died," it's "PG Y's round failed because rank R didn't show up." FR gives you PG identity; windowing reconstructs temporal alignment across ranks. Together they make cross-rank causal analysis possible. Everything below is in service of this.

  

## 1. What Windowing Is Really Doing

  

The fundamental question is: **which ranks were actually doing the same thing at the same time?** This is what windowing answers — not just who *can* communicate together, but who *was* doing so in the same logical phase of the schedule.

  

**PGs are the convenient proxy.** For most parallelism types (TP, EP), all PG members execute in lockstep within a microbatch, so "same PG" = "same thing at the same time" and PG membership alone is sufficient.

  

**PP is the tricky case.** With interleaved 1F1B, even ranks within the same PP PG may be processing different microbatches at different times. The PG roster tells you they *can* communicate, not that they *are* doing so right now. When PG membership isn't enough, windowing goes deeper: look at which ranks actually showed up together, not just who's in the roster.

  

**The wavefront selection is a schedule-order reconstruction**: the algorithm picks whichever PG instance the most ranks are currently pointing at (majority vote), consumes those entries, then advances to the next. This reconstructs the logical scheduling order — each PG wavefront "happens before" the next in the global schedule. (Not "time-based" in the sense of wall-clock timestamps — it's about replay order.)

  

**Why you can't just look at a single collective in isolation**: in real traces, different ranks have other PGs' collectives and p2p ops interleaved before and after. Single-record comparison doesn't tell you "which wave of which PG this belongs to." The window gives you the bucket within which cross-rank comparison is meaningful.

**Windowing flattens the seq_id layer.** A window groups all consecutive entries of the same PG until a different PG appears. If a rank's trace shows `T T T T D T T T T D`, the first four T's form one window, the next four T's form another — the D in between marks the boundary. This collapses many seq_ids into one window_id.

**Why this is the right granularity for straggler cascade** (pending empirical validation):

The core insight: cascade boundaries exist where PG membership changes. Within a window, all member ranks execute the same collectives together — if rank X is late to the first collective, it's late to all of them. Cascade within a window is instant and complete. But when PG switches (T → D), different ranks become involved, and the straggler can infect a new set. The cascade arrows point *between* windows.

Going finer (per-seq_id) adds noise and repetitive information — the same delay appears N times. Going one level above (grouping multiple windows of the same PG type) skips over the intermediate PGs when PG membership change, which is where cascade actually propagates.

**Windowing as 2D compaction.** The input is a matrix: rows = ranks, columns = FR entries (already time-ordered by `time_created_ns` within each rank). Pass 1 compacts each row: consecutive same-PG entries merge into one window, and singleton entries also become their own window — every entry gets a window_id. Pass 2 compacts across columns: order windows by timestamp, assign global_idx. The output is a 2D-compacted total ordering of windows that enables graph traversal for attribution and straggler analysis.

  

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

  

- Used in `group_pgs()` for the DFS monotonicity constraint (only traverse from lower to higher scheduling order) — this is the index that drives the graph traversal for both fault attribution and straggler PG ordering

  

**Hierarchy**: global idx (cross-PG order) > window_idx (within-PG iteration boundary) > seq_id (individual operations within one window).

  

  

## 4. Why Graph Traversal Works: The Independent Fault Model

  

  

For each PG/rank that's missing, only two possibilities exist:

  

- It's an **independent root cause** (broke on its own)

  

- It was **infected/blocked** by an earlier PG/rank that failed first

  

  

**The key insight**: we deduce all missing ranks specifically to **exclude all dependently-broken ranks/PGs** — i.e., all the ones that were infected by the cascade. What we're actually searching for is the **independent faulty rank** (at least in the FR/comm world).

  

  

If a rank is the *earliest* to go missing (head of a causal chain), it cannot have been infected by anything later. It either independently failed, or it independently failed AND infected others downstream.

  

  

**The practical assumption that makes this work** (from experience and the Meta paper): in real large-scale training, **typically only a couple of independent root causes exist, which then infect potentially thousands of other ranks**. The algorithm builds all causal chains and gathers their heads — the union of head PGs' missing ranks across all paths is the answer. This gets us close to the truth.

  

**Important caveat**: this is a heuristic attribution, not a mathematical proof. The graph connects PGs that share any rank (maximally pessimistic edge construction — assumes any shared rank can transmit failure). The algorithm reports the earliest symptomatic PG as root cause by policy, which is justified by the few-root-causes assumption but not proven for all cases.

  

  

## 5. Window Splitting: Why and When

  

  

Two conditions trigger a new window for a PG (both are fundamentally PP scenarios):

  

1. **Same ranks come back** (`already_participated`): the PG reappears with the same ranks → new iteration. This happens because Megatron reuses the same PG object (same megatron_id = same `(desc, ranks)`) across all training steps, so within the ring buffer you see the same group appearing repeatedly.

  

2. **≥2 new ranks appear** (`has_significant_new_ranks >= 2`): a batch of previously-unseen ranks joins → different microbatch/PP phase within the same PG.

  

  

**The ≥2 new ranks guard is fundamentally a PP guardrail.** For TP/EP, all member ranks execute in lockstep — you'd never see "new ranks" appearing (either all are there or none). The skew only arises when a PG spans multiple PP stages processing different microbatches at different times.

  

  

**What this edge case means**: even within the same PG (same megatron_id, same desc, same ranks list), because of interleaved 1F1B, different microbatches mean different ranks are doing different things at different times. ==So logically they're in different temporal phases, even though they're under the same PG umbrella. Window splitting catches this.==

  

  

**Key implementation detail**: this isn't hardcoded with if/else to distinguish p2p vs collective. The `pgs_with_active_ranks_last_iter` state variable naturally does the right thing — PP groups get split into multiple windows because they "disappear and come back," while TP/EP groups stay in the active set continuously and don't trigger the gate.

  

  

## 6. The Two-Window Hierarchy for Straggler Detection

  

  

There are two completely different "window" concepts that must not be conflated:

  

  

**Runtime capture window** (N steps / T seconds — TBD): "From now on, for the next N training steps, enable FR+cudaEvent timing, then dump." This just selects which temporal segment of the training run to collect detailed traces for. It's the observation interval. The exact trigger and window size are still under design.

  

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

| ------------------------------ | -------------------------- | --------------------------------- | ------------------------------------ |

| `time_created_ns` | Host CPU `clock_gettime` | CPU created the collective record | Yes (default FR) |

| `time_discovered_started_ns` | cudaEvent poll by watchdog | GPU comm kernel began | Only with `TORCH_NCCL_ENABLE_TIMING` |

| `time_discovered_completed_ns` | cudaEvent poll by watchdog | GPU comm kernel finished | Only with `TORCH_NCCL_ENABLE_TIMING` |

  

  

Host-side analysis (who enqueued late) doesn't need cudaEvent. GPU-side analysis (comm kernel execution time) does.

  

  

Null values mean the watchdog hasn't polled yet, or timing wasn't enabled. sbak: "FR needs to enable timings only when measuring timings." With `TORCH_NCCL_ENABLE_TIMING=1`, completed entries will have their timing fields populated; entries still in `scheduled` state may still show `null` because the events haven't fired yet.

  

**Where to go deeper**: `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp` — search for `ncclCommWatchdog` (watchdog entry), `WorkNCCL::checkAndSetException` (CUDA event polling), `TORCH_NCCL_ENABLE_TIMING`, `TORCH_NCCL_TRACE_BUFFER_SIZE`. Ring buffer struct: `torch/csrc/distributed/c10d/FlightRecorder.hpp`. Dump function: `torch/_C/_distributed_c10d._dump_nccl_trace()` (called by `trace_collector.py`; C++ impl searches `dumpNcclTrace` in ProcessGroupNCCL.cpp).

## 9. FR Overhead: Three Layers

  

  

**Layer 1 — Ring buffer metadata (always on, zero cost)**: The ring buffer entries themselves — per-collective bookkeeping in C10D's data structures for timeout detection. Records `profiling_name`, `pg_id`, `process_group`, `collective_seq_id`, `time_created_ns`, and `state`. Fixed-size buffer (default 2000 entries), always running.

  

**Layer 2 — CUDA events for completion (already in PyTorch, minimal)**: PyTorch records CUDA events on the NCCL stream for `work.wait()`. One lightweight timestamp marker per collective, not a kernel launch. This is what populates `state` (scheduled/started/completed).

  

**Layer 3 — Timing discovery (watchdog polls, opt-in via `TORCH_NCCL_ENABLE_TIMING`)**: Watchdog calls `cudaEventQuery` (non-blocking) and `cudaEventElapsedTime`. Populates `time_discovered_started/completed_ns`. This has real overhead — not free — which is why it's opt-in and why we only enable it during triggered capture windows.

  

**Key contrast**: one all_reduce = one FR entry, regardless of how many NCCL kernels it generates. CUPTI intercepts every kernel launch (thousands per step). 1-2 orders of magnitude difference.

  

  

## 10. Why PG-Level Granularity Is Fundamental (and Kernel-Level Is Not)

  

**PGs are the unit where causality lives.** A straggler isn't "kernel X was slow" — it's "rank R was late to PG Y's round W." A fault isn't "kernel X died" — it's "PG Y's round failed because rank R didn't show up." Faults, stragglers, and hangs all cascade through PGs via shared ranks and scheduling order. The cascade graph is built from PG overlap, not from kernel adjacency. If you analyze at kernel granularity, you're looking at surface symptoms with no structure to trace back to root cause.

  

**FR gives you that structure.** Every entry carries PG identity (megatron_id = which logical group), collective identity (profiling_name + seq_id), and rank membership. Windowing then reconstructs temporal alignment across ranks — "which collectives are the same logical round of this PG." This is what makes cross-rank comparison meaningful: not just "rank 0 and rank 1 both called AllReduce" but "rank 0 and rank 1 were in the same PG wavefront at the same logical point in the schedule."

  

  

**CUPTI operates below this level.** CUPTI records per-GPU kernel name + duration at the CUDA driver level — no PG, no rank mapping, no collective identity. You cannot correlate rank 0's `ncclDevKernel_AllReduce_Sum` with rank 1's: same kernel name doesn't mean same logical collective (could be different AllReduces on different PGs, interleaved arbitrarily). Even if you could match kernels across ranks, you'd have the wrong abstraction — kernels don't carry the logical grouping that makes causal reasoning possible.

  

  

## 11. Ring Buffer Limitation for Straggler Detection

  

  

Default 2000 entries — enough for fault attribution (snapshot at crash), NOT for straggler detection (need to observe patterns over multiple iterations). sbak: "It's set to 2000 which is enough for fault analysis. Not enough for straggler. And trace dump should happen by trainer."

  

  

Ring buffer tail (oldest surviving entry) is closest to the root cause in attribution. For straggler, extending the buffer or dumping periodically from the trainer is needed. Two broad approaches: (1) passive — always-on FR dump at fixed intervals; (2) active — anomaly-triggered capture (detect slow step, then enable Layer 3 timing for N steps and dump). The active approach is more efficient; the right design is TBD.

  

## 12. Why seq_id Cross-Rank Matching Fails

  

**The conceptual problem**: `collective_seq_id` is per-(rank, PG instance). Different ranks participate in different sets of PGs at different rates — rank 0 might be in TP, EP, and CP while rank 8 is only in TP. Each PG has its own counter, so the ranks' seq_ids diverge as they weave through their different PG sets at different paces. "TP seq=3 on rank 0" and "TP seq=3 on rank 8" refer to the same PG instance but happen at completely different points in the global schedule.

  

**Concrete example — non-flat PGs**: rank 0 participates in TP, EP, and CP. Rank 8 only in TP. By the time rank 0 is at TP seq=3, it has already gone through rounds of EP and CP that rank 8 never saw. The counters are structurally incomparable.

  

**Concrete example — p2p (gpu_error_1st trace)**: During the PP phase, ranks 0, 1, 2 all have collective_seq_id=26, but their p2p operations happen at completely different times (rank 0/2 at t≈0s, rank 1 at t≈3.5s) with different communication partners. Using "csid=26 = same round" lumps operations separated by 3.5s, causing false missing detection and window corruption.

  

The old approach (`default_pg_order` + cross-rank seq_id) was wrong for this: it solved PG type ordering but not the finer problem of which entries across ranks correspond to the same logical round. Windowing solves both levels — it's not just a replacement for pg_order, it's a fundamentally deeper approach.

  

## 13. Test Traces Reference

  

**Note**: All traces below are fault/hang injection traces — error injections where training fails or hangs. They are not straggler traces. Straggler injection traces exist separately and will be added later.

  

  

16 GPUs, TP=2, PP=4, DP=2. TP pairs: [0,1] [2,3] ... [14,15]. PP groups: [0,4,8,12] [1,5,9,13] [2,6,10,14] [3,7,11,15].

  

  

In p2p profiling_name (`recv 2<-3`): numbers are PG-internal local rank indices, NOT global rank IDs.

  

  

Why TP does three kinds of collectives: `_all_gather_base` (before ColumnParallelLinear — gather sharded weight), `_reduce_scatter_base` (after RowParallelLinear — reduce partial results), `broadcast` (unsharded params like LayerNorm). All on the same TP PG instance, sharing one seq_id counter.

  

  

| Trace | Missing Ranks | Source |

  

|-------|--------------|--------|

  

| gpu_error_1st | {12, 14} | fault_injection.log |

  

| gpu_error_2nd | {9, 14} | fault_injection.log |

  

| lock_gil_1st | {9, 14} | fault_injection.log |

  

| lock_gil_2nd | {10, 15} | fault_injection.log |

## 14. Why `time_discovered_*` Never Works for Kernel Straggler Detection

With `TORCH_NCCL_ENABLE_TIMING=1`, we get both the `time_discovered_*` polling timestamps AND `duration_ms`. These are coupled through the start event (`ncclStartEvent_`) — without the start event, neither exists; with it, you get both. There's no flag to get one without the other.

Initially we computed `discovered_completed - discovered_started` expecting it to reflect GPU kernel duration. Across 1.2M entries from our 240+ straggler injection traces, operations taking 6 seconds have the same ~200ns discovered diff as operations taking 10µs. This measures how fast the CPU can call `cudaEventQuery()` twice — the watchdog polls in a loop, and discovered timestamps record when polling *noticed* the event, not when the GPU event fired. A 2-second kernel finishing 1ns before one poll and 1ns after the next shows a 2ns discovered diff.

So what ARE discovered timestamps for? Timeout detection — compare `now - discovered_started > threshold` *before* completion. Wall-clock correlation across ranks. In-flight state tracking. The watchdog's original purpose is hang detection, not performance timing.

**For straggler detection**: `duration_ms` (via `cudaEventElapsedTime(start, end)`) is the accurate kernel timing. Host straggler → late `time_created_ns`. Kernel straggler → shortest `duration_ms`.

**Sources**: [PR #141737](https://github.com/pytorch/pytorch/pull/141737), `FlightRecorderDetail.hpp` (`update_state()` for polling), `ProcessGroupNCCL.cpp` (`getDuration()` for cudaEventElapsedTime).