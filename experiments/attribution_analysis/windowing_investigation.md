# Attribution Windowing Investigation V 4

Date: 2026-04-09, updated 2026-04-10

  

Code under test: fr_attribution.py, group_collectives_by_windows() (lines 339-456)

## Question

Does window splitting (window_idx++) actually trigger? Specifically, do PP groups that appear twice in a rank's trace (before and after TP/EP/TDPCP blocks) get separated into different windows?

## Traces tested

All 4 unit test traces in tests/attribution/unit/fr_traces/:

  

|Trace|Setup|Fault|Expected missing|
|---|---|---|---|
|gpu_error_1st|16 GPUs, TP=2, PP=4, DP=2, MoE|GPU_ERROR on ranks 12, 14|{12, 14}|
|gpu_error_2nd|16 GPUs, TP=4, PP=2, DP=2, MoE|GPU_ERROR on ranks 9, 14|{9, 14}|
|lock_gil_1st|16 GPUs, TP=4, PP=2, DP=2, MoE|GIL lock on ranks 9, 14|{9, 14}|
|lock_gil_2nd|16 GPUs, TP=2, PP=4, DP=2, MoE|GIL lock on ranks 10, 15|{10, 15}|

  
  

In gpu_error_1st gpu_error_2nd, the two PP stages are both scheduled.

|Trace|Two PP blocks?|All scheduled?|
|---|---|---|
|gpu_error_1st|Mixed — ranks 0,1 have two blocks; rank 4 has one|Yes|
|gpu_error_2nd|Yes — all ranks have two blocks|Yes|
|lock_gil_1st|No — PP entries are contiguous (one block)|Yes|
|lock_gil_2nd|No — PP entries are contiguous (one block)|Yes|

  

## Hypothesis why inside one window 2 PP stages are both scheduled

scheduled doesn't mean "hasn't executed on GPU." It means "the watchdog hasn't confirmed completion."

  

The state lifecycle is:

  

1. CPU enqueues operation → FR entry created as scheduled
    
2. GPU executes it on the NCCL stream
    
3. Watchdog thread polls the CUDA completion event → state changes to completed
    

  

Step 3 is what's broken in fault traces:

  

- GPU error: GPU is dead. CUDA events can't be queried. Watchdog can't update anything.
    
- GIL lock: Watchdog is a Python thread. GIL is locked. Watchdog can't run.
    

  

So the GPU may have finished PP phase 1 just fine. But nobody updated the FR entry. It stays scheduled forever.

  

That's why both PP phases are scheduled — not because they're both genuinely in-flight, but because the watchdog is dead and can't update anything.

  

## How to reproduce

  

cd /raid/wei23/wei/nvidia-resiliency-ext

  

python3 experiments/attribution_analysis/run_windowing_check.py tests/attribution/unit/fr_traces/gpu_error_1st

  

The script bypasses the NVRx package __init__.py (which requires the mcp module) by shimming the module imports. It loads all rank dump files, runs group_collectives_by_windows(), and prints (pg_id, pg_desc, window_idx) keys.

## Result: window splitting never triggers

Ran on all 4 test traces. Every trace produces the same result:

  

gpu_error_1st: NO PG has window_idx > 0. Everything is window 0.

  

gpu_error_2nd: NO PG has window_idx > 0. Everything is window 0.

  

lock_gil_1st: NO PG has window_idx > 0. Everything is window 0.

  

lock_gil_2nd: NO PG has window_idx > 0. Everything is window 0.

  

Full output for gpu_error_1st (PP groups):

  

pg= 43 PIPELINE_MODEL_PARALLEL_GROUP window=0 entries= 16 ranks=['0', '12', '4', '8']

  

pg= 46 PIPELINE_MODEL_PARALLEL_GROUP window=0 entries= 14 ranks=['1', '13', '5', '9']

  

pg= 49 PIPELINE_MODEL_PARALLEL_GROUP window=0 entries= 18 ranks=['10', '14', '2', '6']

  

pg= 52 PIPELINE_MODEL_PARALLEL_GROUP window=0 entries= 14 ranks=['11', '15', '3', '7']

  

Both PP blocks (before and after TP/EP/TDPCP) are lumped into one window.

### Why the outer gate is always False

if current_pg not in pgs_with_active_ranks_last_iter:    # outer gate

    if already_participated or (                          # inner check

        has_previous_participants and has_significant_new_ranks

    ):

        should_create_new_window = True

  
  
  

The inner check has the right idea: "have the same ranks come back?" or "are new ranks appearing?". But the outer gate prevents it from ever running.

#### How pgs_with_active_ranks_last_iter works

After each wavefront round consumes a PG, the code scans every rank's pointer and collects the set of PGs they currently point to. This set is stored as pgs_with_active_ranks_last_iter.

#### Why current_pg is always in that set

The wavefront selects current_pg by majority vote — it picks whichever PG the most rank pointers currently aim at. But pgs_with_active_ranks_last_iter was built by scanning those same pointers.

  

So the sequence is:

  

1. Previous round finishes. Scan all pointers. Store the set of PGs they point to as pgs_with_active_ranks_last_iter.
    
2. This round starts. Scan all pointers again. Pick the PG with the most votes. That's current_pg.
    

  

The pointers didn't change between step 1 and step 2.  current_pg has votes now, so it's already in pgs_with_active_ranks_last_iter.

  

==The gate checks current_pg not in pgs_with_active_ranks_last_iter. This is always False. The inner check never runs.==

#### example

Take gpu_error_1st, rank 0's sequence: 

Rank 0 (20): PP43 PP43 | TP35 EP71 EP71 TP35 EP71 TP35 TP35 TP35 | TDPCP55 | PP43 PP43 | TP35 TP35 TP35 TP35 | TCPG63 ETMPG91 | TP35

  

When the wavefront is about to select PP43 for the second time, rank 0's pointer aims at PP43, rank 4's pointer aims at PP43, rank 8's pointer aims at PP43, rank 12's pointer aims at PP43. Those are the 4 members of PP43. Their pointers advanced to their second PP43 entries when the previous PG (TDPCP55/57/58) was consumed. So PP43 specifically is in the snapshot pgs_with_active_ranks_last_iter

  

When the wavefront finishes consuming the first TP/EP block and is about to select PP43 for the second time, what's in pgs_with_active_ranks_last_iter? It was built by scanning all 16 rank pointers after the previous round. Ranks 0-3 already have their pointers on their second PP entries (PP43, PP46, PP49, PP52), because those are their next entries in line. So PP43 is in the set. The gate fails. No new window.

  

#### My Guess what the gate was trying to do

The intent seems to be: distinguish TP/EP fast alternation (don't split — same compute phase) from PP long absence (do split — new phase). The idea is that during many rounds of TP/EP consumption, PP would disappear from the set, and when PP comes back, the gate would notice it was gone.

  

But PP doesn't disappear from the set. It re-enters one round before being selected. When the last PG before PP (e.g., TDPCP55) is consumed, the ranks that were on TDPCP55 advance their pointers to their next entry — which is PP. So PP is in the snapshot before it's voted on.

  

The variable tracks "where do pointers aim right now," but what we'd need is something like "what PGs have already been consumed" — a history of past rounds, not a snapshot of current pointer targets. These are different things, and the current implementation always contains the majority vote winner by construction.

#### This is structural, not trace-specific

This holds for every PG type (PP, TP, EP), every topology, every trace. There is no scenario where the outer gate passes, because the majority vote winner is always in the pointer snapshot by definition.

## Why unit tests still pass

Across all 4 traces, no missing rank is ever detected through PP. PIPELINE_MODEL_PARALLEL_GROUP has empty missing ranks in every reference output. Detection comes entirely from other PG types:

  

|Trace|Missing ranks|Detected through|
|---|---|---|
|gpu_error_1st|{12, 14}|EP74, TP41, TP42, ETMPG94, TCPG69, TCPG70, TDPCP58|
|gpu_error_2nd|{9, 14}|EP72, TP37, TP38, ETMPG90, TCPG69, TCPG70, TDPCP64|
|lock_gil_1st|{9, 14}|EP72, TP37, TP38, TDPCP64|
|lock_gil_2nd|{10, 15}|EP73, EP74, TP40, TP42, ETMPG93, ETMPG94, TDPCP57, TDPCP58|

  

Window splitting  doesn't affect these tests because the detection signal comes from non-PP PGs where window splitting was never relevant.

  

The detection asks one question: did this rank show up at all, yes or no? It's binary: present or absent. It does NOT count how many entries each rank has. So even if you merge them, you get the right result.

  
  

## Why this matters for straggler detection

Stragglers are continuous, not binary — we care about how late a rank is, not just whether it's present. With every PG lumped into a single window (window 0), all temporal phases are merged. A rank that's 50ms late in one phase but fine in another gets its timing signal averaged across all phases, making per-phase attribution impossible.

  

This affects all PG types, not just PP. For example, TP35 appears multiple times in rank 0's trace (before and after EP blocks). With no window splitting, all TP35 appearances are in one bucket — we can't tell which TP phase had the straggler.

## Every reappearance = new window?

Why don't we just split all of them. Every time the wavefront selects a PG that was previously consumed, increment window_idx.

  

This produces more windows (TP/EP alternation creates many small windows instead of one big one)

  

- ![unchecked](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    The only cost is more windows to process, which is computationally cheap? But I want to check with you
    

  
  

## gpu_error_1st Example

Parallelism grid (PP stage = pipeline stage):

  

Stage 0: ranks 0, 1, 2, 3 TP pairs: [0,1], [2,3]

  

Stage 1: ranks 4, 5, 6, 7 TP pairs: [4,5], [6,7]

  

Stage 2: ranks 8, 9,10,11 TP pairs: [8,9], [10,11]

  

Stage 3: ranks 12,13,14,15 TP pairs: [12,13], [14,15]

  

PP groups: P 43=[0,4,8,12], P 46=[1,5,9,13], P 49=[2,6,10,14], P 52=[3,7,11,15]

  

Entry sequences per rank (PP appears twice for stages 0-2, separated by TP/EP/TDPCP):

  

Rank 0 (20): PP43 PP43 | TP35 EP71 EP71 TP35 EP71 TP35 TP35 TP35 | TDPCP55 | PP43 PP43 | TP35 TP35 TP35 TP35 | TCPG63 ETMPG91 | TP35

  

Rank 1 (20): PP46 PP46 | TP35 EP71 EP71 TP35 EP71 TP35 TP35 TP35 | TDPCP55 | PP46 PP46 | TP35 TP35 TP35 TP35 | TCPG63 ETMPG91 | TP35

  

Rank 2 (20): PP49 PP49 | TP36 EP71 EP71 TP36 EP71 TP36 TP36 TP36 | TDPCP55 | PP49 PP49 | TP36 TP36 TP36 TP36 | TCPG64 ETMPG91 | TP36

  

Rank 3 (20): PP52 PP52 | TP36 EP71 EP71 TP36 EP71 TP36 TP36 TP36 | TDPCP55 | PP52 PP52 | TP36 TP36 TP36 TP36 | TCPG64 ETMPG91 | TP36

  

Rank 4 ( 8): PP43 PP43 PP43 | TP37 TP37 | TCPG65 ETMPG92 | TP37

  

Rank 5 ( 8): PP46 PP46 PP46 | TP37 TP37 | TCPG65 ETMPG92 | TP37

  

Rank 6 ( 8): PP49 PP49 PP49 | TP38 TP38 | TCPG66 ETMPG92 | TP38

  

Rank 7 ( 8): PP52 PP52 PP52 | TP38 TP38 | TCPG66 ETMPG92 | TP38

  

Rank 8 (19): PP43 PP43 PP43 | TP39 EP73 EP73 TP39 EP73 TP39 TP39 TP39 | TDPCP57 | PP43 PP43 | TP39 TP39 | TCPG67 ETMPG93 | TP39

  

Rank 9 (19): PP46 PP46 PP46 | TP39 EP73 EP73 TP39 EP73 TP39 TP39 TP39 | TDPCP57 | PP46 PP46 | TP39 TP39 | TCPG67 ETMPG93 | TP39

  

Rank 10 (19): PP49 PP49 PP49 | TP40 EP73 EP73 TP40 EP73 TP40 TP40 TP40 | TDPCP57 | PP49 PP49 | TP40 TP40 | TCPG68 ETMPG93 | TP40

  

Rank 11 (19): PP52 PP52 PP52 | TP40 EP73 EP73 TP40 EP73 TP40 TP40 TP40 | TDPCP57 | PP52 PP52 | TP40 TP40 | TCPG68 ETMPG93 | TP40

  

Rank 12 (27): TP41 EP74 EP74 TP41 EP74 TP41 TP41 TP41 | TDPCP58 | PP43 PP43 | TP41 TP41 TP41 TP41 | TCPG69 ETMPG94 | TP41 | EP74 EP74 EP74 TP41 | TDPCP58 | PP43 PP43 | TP41 EP74

  

Rank 13 (16): EP74 TP41 EP74 TP41 TP41 TP41 | TDPCP58 | PP46 PP46 | TP41 TP41 TP41 TP41 | TCPG69 ETMPG94 | TP41

  

Rank 14 (43): TP42 EP74 EP74 TP42 EP74 TP42 TP42 TP42 | TDPCP58 | PP49 PP49 | TP42 TP42 TP42 TP42 | TCPG70 ETMPG94 | TP42 | EP74 EP74 EP74 TP42 | TDPCP58 | PP49 PP49 | TP42 EP74 EP74 TP42 EP74 TP42 TP42 TP42 | TDPCP58 | PP49 PP49 | TP42 TP42 TP42 TP42 | TCPG70 ETMPG94 | TP42

  

Rank 15 (15): EP74 EP74 TP42 TP42 TP42 | TDPCP58 | PP52 PP52 | TP42 TP42 TP42 TP42 | TCPG70 ETMPG94 | TP42

  
  
**