**FR-Based Straggler Detection: Proposal and Open Questions**

*Author: Wei Shen |  March 2026*

This document proposes extending the Flight Recorder infrastructure in NVRx to detect and attribute communication stragglers during training. It covers: what the current straggler module does and cannot do, the two straggler types observable via FR, the proposed two-phase design, and unresolved gaps organized by topic.

 

# **1\. Current State of Straggler Detection**

Straggler detection in the existing NVRx straggler module — and in the broader landscape beyond NVRx — is fundamentally application-level: it wraps sections of the training loop and measures how long those sections take on each rank. The same is true of other common approaches (e.g. DCGM-based GPU utilization monitoring, custom heartbeat timers in training frameworks). These approaches can tell you that a rank is slow, but they operate at section or step granularity. They cannot tell you which collective within a step was the bottleneck, and they cannot tell you whether the slowdown originated on the host (CPU-side collective scheduling) or on the GPU (comm kernel execution). This document proposes using FR timestamps to fill that gap.

## **1.1 What the existing straggler module does**

The straggler/ module detects slow ranks at section granularity. The user wraps a training section with Detector.detection\_section(name), which records CPU wall-clock elapsed time per step on each rank. Periodically, ReportGenerator all-reduces per-rank summaries and computes two scores:

•   	**Relative score**: this rank's elapsed / fastest rank's elapsed. Threshold default 0.75 — below this is flagged as straggler.

•   	**Individual score**: this rank's elapsed / this rank's own historical best. Detects a rank that degraded relative to itself.

For GPU-side measurement, CuptiManager (a CUPTI C++ extension) records per-GPU-kernel durations when profiling is active. Reports fire at a configurable wall-clock interval (e.g. 5 minutes).

## **1.2 What this cannot do — and why it matters**

•   	**Cannot attribute a slow step to any specific collective or parallelism group**. All it can say is: rank X was slow in section Y. The deeper reason is that both section-timing and CUPTI-based approaches operate below the level where causality lives. A straggler isn't "kernel X was slow" — it's "rank R was late to PG Y's round W." Without PG identity (which communication group, which logical round), you have kernel symptoms with no structure to trace back to root cause. CUPTI records kernel name and duration at the CUDA driver level — no PG, no rank membership, no collective identity. Even comparing NCCL kernel durations across ranks is misleading: the straggler (late-arriving rank) has the *shortest* kernel duration because all ranks finish the synchronous collective at approximately the same wall-clock time, so the late-starting kernel runs for the least time. NVRx already knows this — it filters NCCL kernels out of CUPTI scoring entirely.

•   	**Cannot distinguish host-side slowdown from GPU-side slowdown**. A rank's collective scheduling may be late because its preceding compute took longer (host-side), or the comm kernel itself may be slow because of degraded GPU bandwidth or interconnect (GPU-side). These have different remediation paths: host-side points to compute bottlenecks or CPU scheduling issues; GPU-side points to hardware or topology issues. Without this distinction, the straggler report provides a rank name but no direction for debugging.

•   	**CUPTI profiling is always-on when enabled**, adding sustained overhead. There is no anomaly-triggered activation.

 

# **2\. FR Timestamps and the Two Straggler Types**

## **2.1 The three timestamps**

 

| Timestamp | How recorded | What it marks |
| :---- | :---- | :---- |
| time\_created\_ns | Host CPU, default FR (no extra config) | CPU creates the collective record — host has decided to issue this op |
| time\_discovered\_started\_ns | cudaEvent (TORCH\_NCCL\_ENABLE\_TIMING) | GPU comm kernel begins executing |
| time\_discovered\_completed\_ns | cudaEvent (TORCH\_NCCL\_ENABLE\_TIMING) | GPU comm kernel finishes |

 

From these, two intervals can be defined per rank per collective:

 

**Host-side scheduling delay**:

	host\_delay \= time\_discovered\_started\_ns − time\_created\_ns

Roughly corresponds to the time from when the host created the collective record to when the comm kernel actually began executing on the GPU. If a rank's host\_delay is significantly larger than peers in the same window, it is a host-side straggler candidate.

 

**GPU-side kernel duration**:

	gpu\_duration \= time\_discovered\_completed\_ns − time\_discovered\_started\_ns

The time the comm kernel actually ran on the GPU. If a rank's gpu\_duration is significantly smaller than peers in the same window, it is a GPU-side straggler candidate (see Section 2.2 for why smaller means straggler).

## **2.2 The two straggler types**

**Host-side straggler**: the rank's CPU issued the collective later than peers — its preceding compute (or other host-side work) ran long, delaying the enqueue. Observable as a late `time_created_ns` within a window, or an outlier `host_delay = time_discovered_started_ns − time_created_ns`. Detectable without cudaEvent timing since `time_created_ns` is recorded by default FR (Layer 1+2, always on).

**GPU-side straggler**: the rank's comm kernel starts late relative to peers — the GPU was still occupied with compute or there was interconnect delay. The detection signal is the combination of two facts: (1) `time_discovered_started_ns` is latest among peers — the comm kernel started last; (2) because the collective is synchronous (all ranks finish at approximately the same wall-clock time), the late-starting rank's comm kernel runs for the shortest duration. sbak: "late comer has shortest timing." Duration alone is ambiguous — a short `gpu_duration` could mean the rank was fast *or* that it arrived last. The late `time_discovered_started_ns` is the primary signal; short duration is the corroboration. Effective BW (bytes / duration) is artificially high for the same reason. Requires `TORCH_NCCL_ENABLE_TIMING` (Layer 3, non-trivial overhead, opt-in).

Whether host-side and GPU-side can co-occur — e.g. a rank whose preceding compute was slow (pushing `time_created_ns` late) AND whose comm kernel itself was slow — is an open question for attribution ordering. See Open Questions.

 

 

# **3\. Proposed Design**

## **3.1 Why windowing is needed**

The three timestamps exist per-collective, so in principle you could compare them across ranks for any single collective. The problem is cross-rank alignment: which collective on rank A corresponds to which on rank B?

`collective_seq_id` is per-(rank, PG instance) and cannot be used for cross-rank matching. The root reason: different ranks participate in different sets of PGs at different rates. Rank 0 might be in TP, EP, and CP; rank 8 only in TP. Each PG has its own counter, so by the time rank 0 reaches TP seq=3, it has already gone through EP and CP rounds that rank 8 never saw — the counters are structurally incomparable. P2P makes this worse: multiple send/recv entries within one logical PP communication share the same seq_id, so seq_id no longer counts individual operations cleanly.

`group_collectives_by_windows()` in `fr_attribution.py` solves this from first principles: replay all ranks' timelines simultaneously and at each step pick whichever PG instance the most ranks are currently pointing at (majority-vote wavefront). This reconstructs the global scheduling order without relying on any local counter. The result is `(process_group, sub_group, window_idx)` buckets — one per PG wavefront — where all entries in a bucket correspond to the same logical round of that PG across all participating ranks. This infrastructure is directly reusable for straggler detection: the same bucket that lets us check who was *missing* (fault attribution) also lets us compare *timestamps* (straggler attribution).

## **3.2 Two-window hierarchy**

 

| Runtime capture window | Coarse. Spans N detection\_section steps during which cudaEvent timing is on. Determined by the trigger logic. Answers: what time slice are we capturing? |
| :---- | :---- |
| **FR attribution window (PG/phase)** | Fine. Within the captured dump, group\_collectives\_by\_windows() partitions collectives into one bucket per PG wavefront. Determined offline by replay. Answers: which collectives are the same logical round? |

 

The runtime capture window is the outer boundary — the observation interval we collect. The FR attribution window is the inner structure — how we analyze what was captured. They are distinct and must not be conflated.

What counts as one "step" here: the same detection\_section unit the existing straggler Detector already uses — typically one logical training step (forward \+ backward \+ optimizer), whatever the user has wrapped. The runtime capture window of N steps means: collect FR for the next N calls to that same section. This reuses existing infrastructure and avoids redefining what a "step" is.

## **3.3 Phase 1 — Runtime trigger and capture**

The existing Detector.detection\_section() already measures CPU wall-clock elapsed per step. The elapsed here is the full end-to-end wall-clock time of the section: forward pass, backward pass, optimizer step, allreduce, and synchronization. This is the coarse trigger signal — it cannot distinguish host vs GPU cause, but it can detect that something is wrong.

The trigger logic:

•   	Maintain a moving average of per-step elapsed times (e.g. EMA or sliding window over last K steps). When current\_elapsed \> moving\_avg × (1 \+ threshold), trigger a capture.

•   	On trigger: enable cudaEvent timing for FR via TORCH\_NCCL\_ENABLE\_TIMING at runtime. sbak confirmed PyTorch supports this without restart. 

•   	Collect FR traces for N detection\_section steps (runtime capture window). N is TBD.

•   	After N steps: disable cudaEvent timing, dump the FR buffer (same format as existing attribution dumps), resume normal operation.

**Two detection modes, different overhead**:
- *Host-side-only mode*: uses `time_created_ns` from default FR (Layer 1, always on, zero overhead). Can run continuously or at a low trigger threshold. Detects late collective scheduling without any extra configuration.
- *Full mode (host + GPU-side)*: additionally enables `TORCH_NCCL_ENABLE_TIMING` (Layer 3) to populate `time_discovered_started/completed_ns`. Layer 3 has real overhead (watchdog polling `cudaEventQuery`/`cudaEventElapsedTime`) — it's opt-in and should only be active during the triggered capture window, not continuously.

The trigger fires on a slow step in either mode; the choice of which mode to activate depends on whether we suspect host-side or GPU-side cause, or we default to full mode on anomaly detection.

## **3.4 Phase 2 — Offline attribution**

Given the triggered dump:

•   	Load FR JSON dumps from all ranks. Run group\_collectives\_by\_windows() to partition into (PG, sub\_group, window\_idx) buckets.

•   	For each (PG, window), compute per-rank host\_delay and gpu\_duration.

•   	Within the same (PG, window), compare these metrics across all participating ranks:

◦   	**Host-side straggler**: rank whose `time_created_ns` is an outlier (late relative to peers in the same window), or whose `host_delay = time_discovered_started_ns − time_created_ns` is significantly larger than peers. The former measures when the CPU decided to issue the collective; the latter includes queuing delay and preceding GPU work. Which is the better primary metric is an open question.

◦   	**GPU-side straggler**: rank with the latest `time_discovered_started_ns` (late kernel start) AND the shortest `gpu_duration` (short kernel run because everyone else waited). Duration alone is insufficient — use both signals together. Effective BW (collective size / gpu_duration) is artificially high for the same rank and can serve as a normalized cross-collective comparator (sbak: "collect effective BW per process group and do the same as failure attribution").

•   	Report: straggler candidates per (PG, window), labeled host or GPU. This parallels hang attribution — instead of missing/present, we are flagging timing outliers within the same windowed bucket structure.

•   	Thresholds are empirical: run clean (no-straggler) traces, compute distributions of `Δ_created`, `host_delay`, `gpu_duration` across windows, derive ε from P95/P99 × safety multiplier. Classification: if `time_created_ns` or `host_delay` is anomalous → host-side; if both `time_created_ns` and `time_discovered_started_ns` are aligned but `gpu_duration` is anomalous → GPU-side; if both are off → co-occurring (see Open Questions).

## **3.5 Future: earliest-straggler PG (not immediate)**

Once basic per-window detection works, a natural extension: if rank R is a straggler in window W\_k, is there an earlier window where R first showed degradation? This mirrors hang attribution: walk back through the window graph to find the earliest symptomatic PG. Designing this is deferred.

 

# **4. Open Questions**

1. **Runtime metric:**  
     
   - Is it appropriate to use **whole‑step / whole‑section elapsed wall‑clock time** as the runtime trigger for enabling FR timing?  
   - Or should we be more specific (e.g., comm‑only elapsed) from the beginning?

   

2. **Exact meaning of FR fields:**  
     
   - Are we correctly interpreting `time_created_ns`, `time_discovered_started_ns`, `time_discovered_completed_ns` as:  
     - “host scheduling time,” “GPU kernel start,” and “GPU kernel end” respectively, for the purpose of straggler attribution?

   

3. **Window semantics:**  
   - “we can use the same window mechanism as failure attribution,”  
   - Is it reasonable to treat the runtime window (N steps / T seconds) as simply a **temporal selection** of the trace, and rely entirely on `group_collectives_by_windows` for **logical PG/phase alignment** inside that segment?

4. **Formalizing host vs GPU metrics:**  
     
   - For host‑side stragglers, which metric would you consider primary in practice:  
     - `time_created_ns` outliers,  
     - `time_discovered_started_ns - time_created_ns` outliers,  
     - or a combination?  
   - For GPU‑side stragglers, is the intended operational rule:  
     - late kernel start time **plus** shortest kernel duration / highest effective BW within the same PG/window?

   

5. **Assumptions and thresholds:**  
     
   - To what extent are we relying on the assumption that, in non‑straggler runs, `created`, `started`, and `completed` are “almost aligned” across ranks for a given PG/wavefront?  
   - Should we explicitly design **ε / threshold values** from empirical distributions observed in non‑straggler runs (e.g., P95/P99 of |Δcreated|, host delay, GPU duration), and treat outliers as stragglers?

   

6. **Ambiguous cases with both host and GPU slow:**  
     
   - In cases where a rank is an outlier in both host delay and GPU duration:  
     - Do we treat it as both host‑ and GPU‑side straggler?  
     - Or is there a preferred attribution ordering (e.g., first assign to host if host delay is outside baseline, and only assign to GPU if host is within baseline)?

---

# **5. Proposed Next Steps**

1. **Empirical baseline study:**  
     
   - On existing `_dump_*.json` traces where we believe there is no strong straggler:  
     - Compute distributions of:  
       - Inter‑rank differences in `time_created_ns` for a given PG/window,  
       - Host delay: `time_discovered_started_ns - time_created_ns`,  
       - GPU duration: `time_discovered_completed_ns - time_discovered_started_ns`,  
   - Use these to estimate realistic **baseline ranges and ε thresholds** for “normal” behavior.

   

2. **Prototype offline analyzer:**  
     
   - On top of `fr_attribution.py`, implement a **purely offline** prototype that:  
     - Reuses `group_collectives_by_windows` to align PG/windows,  
     - Computes host delay, GPU duration, effective BW per rank per window,  
     - Flags candidate host‑side and GPU‑side stragglers as outliers (using baseline statistics).  
   - Run this on both synthetic traces and real traces (with known issues) to see if the “late comer with shortest duration / highest BW” pattern emerges as expected.

   

3. **Refine runtime triggering strategy:**  
     
   - Based on empirical results, refine:  
     - Which runtime elapsed metric we should use (and how noisy it is),  
     - How large the runtime profiling window should be (N steps / T seconds) to reliably capture the relevant FR data without excessive overhead.

These next steps are ordered: the empirical baseline study unblocks the prototype, which unblocks the runtime trigger design. The offline prototype on existing fault traces (even without straggler injection) will establish whether the "late start + short duration" pattern is cleanly separable from noise before we commit to the full runtime integration.  
