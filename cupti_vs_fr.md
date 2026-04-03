# Why FR Can Do Distributed Straggler Detection and CUPTI Cannot

## The Core Problem

Straggler detection requires **matching the same logical operation across ranks**: "for this specific AllReduce on the TP group, which rank was late?" This requires cross-rank correlation context.

## What CUPTI Records

CUPTI (`CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`) intercepts at the CUDA driver level. Per GPU, for every kernel launch, it records:
- Kernel name string (e.g., `ncclDevKernel_AllReduce_Sum_blk_512_1_1_grid_108_1_1`)
- Duration in microseconds
- Block/grid dimensions

That's it. **No PG, no rank mapping, no collective identity, no ordering context.**

Running CUPTI on all 16 GPUs gives 16 independent lists of kernel names and durations. You cannot correlate rank 0's `ncclDevKernel_AllReduce_Sum` with rank 1's — same kernel name doesn't mean same logical collective (could be different AllReduces on different PGs).

## What FR Records

Per collective call, FR records:
- Which rank (dump file)
- Which PG (`process_group[0]` = megatron_id, maps to specific rank set)
- Which logical collective (`profiling_name` + `collective_seq_id` within the PG)
- CPU enqueue time (`time_created_ns`)
- Optionally: GPU start/end (`time_discovered_started_ns`, `time_discovered_completed_ns`)

You can directly match: "rank 0 and rank 1 both called `nccl:_all_gather_base` on PG 35 (TENSOR_MODEL_PARALLEL_GROUP, ranks [0,1]) with collective_seq_id=39407." Same operation, compare their timestamps.

## Why NCCL Kernel Duration Is Misleading for Stragglers

CUPTI reports `ncclDevKernel_AllReduce_Sum: MED=256μs` on one rank vs `MED=3369μs` on another. Which is the straggler?

The rank with 3369μs might have **arrived first** and spent most of that time spin-waiting inside the NCCL kernel for others. The rank with 256μs might have **arrived late** but finished quickly because everyone else was already waiting.

```
Time:    0ms                              50ms    55ms
GPU 0:   [compute 10ms][---NCCL kernel: waiting...transfer---]  duration=45ms
GPU 2:   [======compute 50ms======][NCCL kernel: transfer]      duration=5ms
```

GPU 2 is the straggler (slow compute), but CUPTI shows it has the SHORTEST NCCL kernel duration. Without knowing when each rank entered the collective, the duration alone is ambiguous. This is why NVRx filters NCCL kernels out of CUPTI scoring.

## Overhead Comparison

| Aspect | CUPTI | FR |
|--------|-------|-----|
| Hooks at | CUDA driver level | PyTorch C10D collective dispatch |
| Records per step | Thousands (every kernel launch) | Tens to hundreds (one per collective call) |
| Overhead | 2-10% (commonly cited) | Negligible (ring buffer bookkeeping) |
| Must be enabled | Yes (explicit `cuptiActivityEnable`) | Layer 1+2 always on; Layer 3 needs `TORCH_NCCL_ENABLE_TIMING` |
| Cross-rank context | None | PG membership, collective identity, ordering |

## Summary

CUPTI = powerful single-GPU profiler, useless for distributed straggler attribution.
FR = lightweight distributed context, directly supports cross-rank collective matching.

The existing NVRx straggler module uses CUPTI for GPU kernel scoring (compute kernels only, NCCL filtered out) and CPU wall-clock for section-level detection. It can tell you "rank X was slow in section Y" but not which collective or which PG. FR-based detection fills this gap.
