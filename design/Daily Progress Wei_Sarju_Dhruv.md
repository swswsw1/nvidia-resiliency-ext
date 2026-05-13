FR-Based Straggler Detection for NVRx
1. The Injection: Simulating a Straggler
What We Injected
We inject a 300ms CPU sleep on rank 2 inside the CUPTI detection section, followed by a dist.barrier() that forces all ranks to synchronize before the section ends:
with straggler.Detector.detection_section("fwd", profile_cuda=True):
    output = ddp_model(data)
    if local_rank == 2:
        sleep(0.3)
    dist.barrier()  # all ranks must wait for rank 2

Why This Is a Realistic Straggler
This injection mimics real-world straggler scenarios where something delays a rank between compute and communication:
Thermal throttling: GPU clock drops, compute takes longer, the rank reaches the next synchronization point late.
Data loader bottleneck: CPU blocks on disk I/O or network fetch, delaying the next batch.
OS interference: NIC soft-interrupts, systemctl daemon-reload, or garbage collection pauses steal CPU cycles from the training thread.
Host-side logging overhead: Excessive serialization consumes CPU time between operations.
In all these cases, one rank is slow to reach the synchronization point. The barrier (or any blocking collective like allreduce) forces all other ranks to wait, turning a single-rank delay into a cluster-wide slowdown.
Why the Barrier Matters
The dist.barrier() is critical to the injection. Without it, the sleep would only delay rank 2's local computation — other ranks could potentially continue. With the barrier, every rank must wait for rank 2 to finish sleeping before the detection section can exit. This means:
Rank 2 sleeps 300ms, then hits the barrier (instant pass — it's the last to arrive)
Ranks 0, 1, 3 finish the forward pass quickly, then block at the barrier for ~300ms waiting for rank 2
The total wall-clock time of the detection section is ~300ms for all ranks, not just rank 2.

2. The Old Approach: Why CUPTI Doesn't Catch It
How CUPTI Detection Works
The existing NVRx straggler detector uses CUPTI to profile GPU kernels within user-annotated detection_section blocks. It measures:
CPU elapsed time of the entire section (wall-clock start to end)
GPU kernel execution times for every CUDA kernel launched within the section
It then computes relative performance scores by comparing each rank's kernel median times against the fastest rank.
Why CUPTI Sees Nothing Wrong
The barrier equalizes section timing across all ranks.
Because the barrier forces synchronization inside the detection section, the section's CPU elapsed time is nearly identical on every rank:
Rank 0: section time ≈ 305ms (forward pass ~5ms + barrier wait ~300ms)
Rank 1: section time ≈ 305ms (forward pass ~5ms + barrier wait ~300ms)
Rank 2: section time ≈ 305ms (forward pass ~5ms + sleep 300ms + barrier ~0ms)
Rank 3: section time ≈ 305ms (forward pass ~5ms + barrier wait ~300ms)

All ranks report ~305ms. CUPTI sees uniform performance. The straggler is invisible.
GPU kernel times are identical across ranks.
The actual compute kernels (matrix multiplications, activations) in ddp_model(data) run at the same speed on all GPUs. The sleep is a CPU operation that produces no GPU kernels. CUPTI profiles GPU kernels and sees no difference between ranks.
NCCL kernels are explicitly filtered out.
Even if CUPTI captured the barrier's underlying NCCL kernel, the ReportGenerator._filter_out_nccl_kernels() method removes all NCCL kernels:
```
def _filter_out_nccl_kernels(self, kernel_summaries):
    # TODO: NCCL kernels are skipped due to huge differences in execution
    # time between ranks observed: E.g. ncclDevKernel_AllReduce_Sum
    # MED=256us on a "fast" rank VS MED=3369us on a "slow" rank
    return {k: v for k, v in kernel_summaries.items() if "ncclDev" not in k}
```

What CUPTI Reports
Rank 0 GPUs relative perf: {0: 0.96, 1: 0.90, 2: 0.97, 3: 0.90}
Rank 0 GPUs individual perf: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}


All ranks appear similar. Rank 2 (the actual straggler) shows 0.97 relative performance — it looks slightly better than average, not worse. The straggler is completely hidden.

3. The New Approach: Why FR-Based Detection Catches It

How It Works
The FR-based detector reads PyTorch's Flight Recorder ring buffer, which automatically records every NCCL collective (including the barrier) with timestamps. No code wrapping needed.
```
# No detection_section wrappers. One call per step.
straggler.FRDetector.initialize(check_interval=100)

for step, batch in enumerate(loader):
    output = model(batch)
    if local_rank == 2:
        sleep(0.3)
    dist.barrier()
    loss.backward()
    optimizer.step()
    result = straggler.FRDetector.step()
```

Every 100 steps, the detector:
Reads completed entries from the FR ring buffer in-memory
Gathers entries from all ranks via all_gather_object
Groups entries by process group and temporal phase using wavefront replay
Compares duration_ms across ranks within each (PG, window) to find outliers
Why FR Catches What CUPTI Misses
The FR records duration_ms for each NCCL collective — this is the CUDA-event-measured time for the operation, including any wait time inside the collective.
For the barrier (or any blocking collective):
Ranks 0, 1, 3 enter the barrier quickly, then wait ~300ms for rank 2. Their duration_ms ≈ 300ms.
Rank 2 enters the barrier last (after the sleep). Everyone else is already waiting. The barrier completes instantly. Its duration_ms ≈ 0.4ms.
rank 0: duration_ms ≈ 300ms  ← waited for rank 2
rank 1: duration_ms ≈ 300ms  ← waited for rank 2
rank 2: duration_ms ≈ 0.4ms  ← arrived last (THE STRAGGLER)
rank 3: duration_ms ≈ 300ms  ← waited for rank 2

The detector computes the median duration (~300ms) and checks each rank's ratio:
rank 2 ratio: 0.4 / 300 = 0.001

Since 0.001 < 0.10 (the threshold), rank 2 is flagged as the straggler.
Three reasons this works when CUPTI doesn't:
No manual instrumentation. The FR records the barrier automatically. The sleep delays rank 2's entry into the barrier, which directly affects duration_ms. The signal exists whether or not the user wrapped anything in a detection_section.


NCCL timing is the signal, not noise. Instead of filtering NCCL kernels out, the FR approach makes NCCL timing the primary detection signal. The "huge differences in execution time" that the CUPTI developers discarded are exactly the straggler signal — you just need to know that low duration = straggler in blocking collectives.


The barrier doesn't hide the straggler — it reveals it. In CUPTI, the barrier equalizes wall-clock time across ranks, hiding the straggler. In FR, the barrier creates a divergence in collective duration: the late-arriving rank gets a short duration, everyone else gets a long one. The barrier that masked the straggler in CUPTI is the mechanism that exposes it in FR.


Detection Output
[Step 99] Straggler: rank 2, PG 0 (default_pg), window 0, 1.0σ slow, cause: compute slowdown

The detector identifies:
Which rank: rank 2
Which process group: PG 0 (default_pg)
Which phase: window 0
Cause: compute slowdown (the delay was before the collective, not during network transfer)


```
straggler.FRDetector.initialize(check_interval=100)

for step, batch in enumerate(loader):
    output = model(batch)
    if local_rank == 2:
        sleep(0.3)
    dist.barrier()
    loss.backward()
    optimizer.step()
    result = straggler.FRDetector.step()
```
Every 100 steps, the detector reads completed entries from all ranks, groups them by process group and temporal phase via wavefront windowing, and computes three metrics per rank per (PG, window):
duration_ms — PyTorch's CUDA-event-measured collective time. Includes wait time for other ranks. In blocking collectives, the straggler has the lowest duration because it arrives last and everyone else was already waiting. This is the primary detection signal.
Queuing delay (time_discovered_started_ns - time_created_ns) — Time between CPU enqueue and GPU start. A large value relative to peers indicates host-side interference (OS jitter, data loader bottleneck, logging overhead). Requires TORCH_NCCL_ENABLE_TIMING=1.
Inter-collective gap (time_created_ns[i] - time_created_ns[i-1]) — Time between consecutive collective enqueues on the same rank. Reflects how long the rank spent in compute or host-side work between collectives. A large value relative to peers indicates slow compute (thermal throttling, heavy kernels) or CPU delays.
Detection logic. These metrics are compared across ranks within each (PG, window):
If rank_duration / median_duration < 0.10 → flag as straggler (median-ratio, robust with small N)
If (μ - rank_duration) / σ > k → flag as straggler (μ+kσ, effective with large N)
If current_group_mean > baseline × (1 + threshold) → flag uniform degradation (all ranks slowed together)
Result for the injection.
rank 0: duration_ms ≈ 300ms  (waited for rank 2)
rank 1: duration_ms ≈ 300ms  (waited for rank 2)
rank 2: duration_ms ≈ 0.4ms  (arrived last)
rank 3: duration_ms ≈ 300ms  (waited for rank 2)

median = 300ms
rank 2 ratio = 0.4 / 300 = 0.001 → below 0.10 threshold → straggler
Output:
[Step 99] Straggler: rank 2, PG 0 (default_pg), window 0, 1.0σ slow, cause: compute slowdown
The detector identifies which rank (rank 2), which process group (PG 0), which phase (window 0), and the cause classification (compute slowdown — the delay was before the collective, not during network transfer).


4. The Straggler Inversion Principle
This is the core insight that makes the FR approach work and explains why CUPTI was confused.
In a blocking collective, the measured duration encodes the wait time, not the transfer time.
For a healthy collective where all ranks arrive simultaneously:
All ranks: duration ≈ transfer_time (e.g., 2ms)

For a collective where rank 2 is 300ms late:
Rank 2:      duration ≈ transfer_time        (0.4ms — arrived last, no wait)
Ranks 0,1,3: duration ≈ wait_time + transfer  (300ms — waited for rank 2)

The straggler's duration is inversely correlated with the slowdown it causes. The worse the straggler, the shorter its collective duration appears.
This explains the CUPTI code comment: "ncclDevKernel_AllReduce_Sum MED=256us on a 'fast' rank VS MED=3369us on a 'slow' rank." The 256μs rank was the straggler — it arrived last to every allreduce, so its kernel was fast. The 3369μs rank was healthy — it arrived first and waited. CUPTI couldn't interpret this, so it filtered NCCL kernels out entirely.
The FR approach inverts the check: flag the rank with the lowest duration as the straggler.

5. Architecture
Module Structure
src/nvidia_resiliency_ext/straggler/
├── straggler.py              # FRDetector: orchestrates the 5-phase pipeline
├── fr_collector.py           # Reads FR ring buffer, extracts completed entries with timing
├── fr_windowed_analyzer.py   # Wavefront replay, groups by PG and phase, computes per-rank stats
├── fr_outlier_detector.py    # Median-ratio + μ+kσ detection, temporal baseline comparison
├── dist_utils.py             # Cross-rank communication helpers (unchanged)
├── statistics.py             # Statistic enum (unchanged)
└── __init__.py               # Exports FRDetector and related classes

Pipeline
Phase 1  Training runs. FR records every NCCL collective automatically.
   │
Phase 2  Every N steps, each rank reads its FR ring buffer in-memory.
   │     Filters for completed entries. Extracts timing fields.
   │     (No disk I/O — reading a data structure in process memory)
   │
Phase 3  Ranks exchange lightweight entry data via all_gather_object.
   │
Phase 4  Wavefront replay groups entries by (PG, phase).
   │     Separates PP phase 1 from PP phase 2.
   │     Computes per-rank timing stats within each window.
   │
Phase 5  Outlier detection:
         • Median-ratio check (primary, robust with small N):
           if rank_duration / median < 0.10 → straggler
         • μ+kσ check (secondary, effective with large N):
           if (μ - rank_duration) / σ > k → straggler
         • Temporal baseline comparison:
           if current_group_mean > baseline × 1.1 → uniform degradation

Prerequisites
PyTorch with NCCL backend
TORCH_NCCL_ENABLE_TIMING=1 (enables duration_ms and time_discovered_started_ns)
Usage
import os
os.environ["TORCH_NCCL_ENABLE_TIMING"] = "1"

from nvidia_resiliency_ext.attribution import straggler

straggler.FRDetector.initialize(check_interval=100, k=2.0)

for step, batch in enumerate(loader):
    output = model(batch)
    loss.backward()
    optimizer.step()

    result = straggler.FRDetector.step()
    if result is not None and rank == 0:
        print(result)

straggler.FRDetector.shutdown()


6. Comparison
Feature
CUPTI (old)
FR-based (new)
Instrumentation
Manual detection_section() wrappers
None — FR records automatically
NCCL handling
Filtered out (_filter_out_nccl_kernels)
Primary detection signal
Catches sleep+barrier straggler
No — barrier equalizes section timing
Yes — barrier creates duration inversion
Catches thermal throttling
Partial (compute kernels only)
Yes (via collective timing inversion)
Catches OS interference
No
Yes (delays CPU enqueue of collectives)
Process group awareness
None
Full (PG identity in FR entries)
Phase granularity
None
Per-window via wavefront replay
Overhead
CUPTI profiling cost
~0% (in-memory buffer read)
Dependencies
CUPTI C++ extension
Pure Python + PyTorch FR


7. Files
File
Purpose
fr_collector.py
FRCollector class. Reads FR ring buffer via _dump_nccl_trace(). Handles pickle deserialization. Deduplicates entries via record_id. Extracts CollectiveEntry objects with timing properties.
fr_windowed_analyzer.py
FRWindowedAnalyzer class. Wavefront replay algorithm adapted from fr_attribution.py. Groups completed entries into (pg_config_id, pg_description, window_idx) buckets. Computes RankTimingStats per rank per window.
fr_outlier_detector.py
FROutlierDetector class. Median-ratio detection for small N, μ+kσ for large N. Temporal baseline comparison for uniform degradation. Produces StragglerReport and DegradationReport.
straggler.py
FRDetector class. Orchestrates the 5-phase pipeline. Handles cross-rank gathering via all_gather_object. Class-method pattern (same as existing Detector).


8. Known Limitations and Future Work
Cannot explain why a rank is slow. The FR tells you which rank and which PG/phase, but not the root cause at the OS level. For deep root-cause analysis, complementary tools like SysOM-AI (eBPF-based cross-layer profiling) are needed.


Small rank counts weaken μ+kσ. With 4 ranks, a single outlier skews both μ and σ, producing severity ~1.73 (below k=2.0 threshold). The median-ratio fallback handles this, but more robust small-sample methods could improve sensitivity.


TORCH_NCCL_ENABLE_TIMING=1 is required for duration_ms and time_discovered_started_ns. Without it, only time_created_ns and time_discovered_completed_ns are available.


Ring buffer overwrites. The FR's 2000-entry buffer holds ~100 steps of history. Very long check intervals risk losing older entries.


Windowing with complex hybrid parallelism (interleaved pipeline schedules, context parallelism) needs further validation.


Integration with NVRx attribution pipeline. StragglerReport could feed into the existing attribution pipeline, triggering targeted health checks on the flagged rank.



