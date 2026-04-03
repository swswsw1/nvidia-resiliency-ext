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

•   	**Cannot attribute a slow step to any specific collective or parallelism group**. All it can say is: rank X was slow in section Y. This means the operator still has to manually correlate the detection with logs, profiles, or other signals to find the actual bottleneck — slow section timing alone does not narrow down root cause.

•   	**Cannot distinguish host-side slowdown from GPU-side slowdown**. A rank's collective scheduling may be late because its preceding compute took longer (host-side), or the comm kernel itself may execute slowly because of degraded GPU bandwidth or interconnect (GPU-side). These have different remediation paths: host-side points to compute bottlenecks or CPU scheduling issues; GPU-side points to hardware or topology issues. Without this distinction, the straggler report provides a rank name but no direction for debugging.

•   	CUPTI profiling is always-on when enabled, adding sustained overhead. There is no anomaly-triggered activation.

 

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

**Host-side straggler**: collective scheduling is late — the rank's CPU issued the collective later than peers. Observable as a late time\_created\_ns within a window, or large host\_delay. Visible without cudaEvent timing since time\_created\_ns is recorded by default FR.

**GPU-side straggler**: the comm kernel execution is slow. The detection signal is counterintuitive: the straggler has the **smallest** gpu\_duration among peers in the same window. Why: in a synchronous collective (e.g. AllReduce), all ranks finish at approximately the same wall-clock time — the collective cannot complete until all ranks join. The rank whose comm kernel starts latest therefore runs for the shortest duration. sbak: "late comer has shortest timing." Requires cudaEvent timing.

Whether both straggler types can co-exist on the same rank simultaneously — e.g. a rank whose preceding compute is slow (host-side) AND whose comm kernel is slow (GPU-side) — is an open question in terms of how to attribute and report them jointly. See Gap C.

 

 

# **3\. Proposed Design**

## **3.1 Why windowing is needed**

The three timestamps exist per-collective, so in principle you could compare them across ranks for any single collective. The problem is cross-rank alignment: how do you know which collective on rank A corresponds to which on rank B? collective\_seq\_id is unreliable (especially with P2P operations).

The existing group\_collectives\_by\_windows() in fr\_attribution.py solves this by replaying all ranks' timelines simultaneously and grouping collectives into (process\_group, sub\_group, window\_idx) buckets — one bucket per PG wavefront. Within a window, collectives from all participating ranks correspond to the same logical round of that PG (e.g. one round of TP AllReduce, or one round of DP gradient reduction). This infrastructure is directly reusable.

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

Important split: host-side straggler detection does not require cudaEvent timing — time\_created\_ns is recorded by default FR. Only GPU-side straggler detection needs it. A lightweight host-side-only mode could run with a lower trigger threshold or even always-on, while the heavier cudaEvent mode is reserved for confirmed anomalies.

## **3.4 Phase 2 — Offline attribution**

Given the triggered dump:

•   	Load FR JSON dumps from all ranks. Run group\_collectives\_by\_windows() to partition into (PG, sub\_group, window\_idx) buckets.

•   	For each (PG, window), compute per-rank host\_delay and gpu\_duration.

•   	Within the same (PG, window), compare these metrics across all participating ranks:

◦   	**Host-side straggler**: outlier in cross-rank time\_created\_ns offset (or in host\_delay). Which metric is correct: Gap B.

◦   	**GPU-side straggler**: outlier by having the smallest gpu\_duration (= latest kernel start). Whether to normalize by collective size (effective BW): Gap C.

•   	Report: straggler candidates per (PG, window), labeled host or GPU.

sbak's framing: "collect effective BW per process group and do the same as failure attribution." The PG-level attribution parallels the existing hang attribution — instead of missing/present, we are looking at timing outliers.

## **3.5 Future: earliest-straggler PG (not immediate)**

Once basic per-window detection works, a natural extension: if rank R is a straggler in window W\_k, is there an earlier window where R first showed degradation? This mirrors hang attribution: walk back through the window graph to find the earliest symptomatic PG. Designing this is deferred.

 

# **4\. Questions**

# **Group 1: What do the timestamps actually measure?**

These questions are about the semantics of the three Flight Recorder timestamps: time\_created\_ns, time\_discovered\_started\_ns, time\_discovered\_completed\_ns.  
 

## **Q1. What happens between time\_created\_ns and time\_discovered\_started\_ns?**

My current understanding is that this window contains whatever computation the GPU was still running before it could begin executing the communication kernel. But I want to confirm: is this interval purely computation kernels, or does it also include framework overhead, queueing delay, or other host-side scheduling latency? In other words, what does this gap represent in practice?  
 

## **Q2. What can we observe without Flight Recorder? What is the baseline?**

Without FR (or without GPU-side timing enabled), I believe we can observe:  
•   	Host-side: step/section wall-clock time, heartbeat timeouts, host stalls  
•   	We cannot observe: when the GPU actually started or finished executing a comm kernel  
Specifically: even with FR enabled, if cudaEvent timing (or equivalent) is not turned on, do we lose time\_discovered\_started\_ns and time\_discovered\_completed\_ns? And if so, what exactly provides those GPU-side timestamps — CUPTI, cudaEvent callbacks, or something else in the NCCL/PyTorch stack?  
   
   
 

# **Group 2: Are the assumptions behind the detection logic valid?**

These questions are about the implicit assumptions we need to make in order to classify stragglers as host-side vs. GPU-side.  
 

## **Q3. How strong is the assumption that time\_created\_ns is nearly identical across ranks?**

The detection logic seems to rely on: if two ranks run the same script, their CPUs will enqueue the same collective at almost the same wall-clock time, so time\_created\_ns should differ by only a small epsilon. My concern is: how strong is this assumption in practice? Specifically:  
•   	Under normal (non-straggler) conditions, what is the typical spread of time\_created\_ns across ranks for the same collective? Nanoseconds?   
•   	If one rank has a host-side slowdown that shifts its entire timeline by some delta (e.g., a slow compute kernel earlier in the step), will time\_created\_ns time\_discovered\_started\_ns, and time\_discovered\_completed\_ns all shift by approximately the same delta? Or can they shift independently?  
 

## **Q4. Can host-side and GPU-side stragglers co-occur, and how do we attribute them?**

Could both exist simultaneously — a rank whose compute was slow (host-side) may also have a GPU that executes comm slowly (GPU-side). In that case:  
•   	Is the right approach to report both scores independently, or to establish a priority order (e.g., attribute to host-side first, then check GPU-side on the residual)?  
•   	Is there a case where a host-side straggler **masks** a GPU-side straggler, making it look like the GPU-side duration is normal when it actually wasn't?  
 

## **Q5. What does "straggler finishes faster" actually mean?**

sbak mentioned that a straggler finishes the comm kernel faster. I believe the correct interpretation is:  
•   	All ranks finish the collective at approximately the same wall-clock time (because the collective is synchronous).  
•   	A straggler starts **later** (because earlier compute, for instance, was slow), so its completed \- started duration is **shorter**.  
If so, then the GPU-side straggler metric is not "whose completed \- started is largest" but rather "whose time\_discovered\_started\_ns is latest relative to the group"? Or do we need both?  
 

## **Q6. Do comm kernels from different ranks truly start and end at the same time?**

For a synchronous collective like AllReduce: I assume all ranks' comm kernels must finish at approximately the same time by definition of the collective. But do they also start at the same time?   
For an ideal case, does this mean time\_discovered\_started\_ns and time\_discovered\_completed\_ns are approximately the same across participating ranks?  
   
   
 

# **Group 3: How do we set the detection threshold (epsilon)?**

These questions are about the practical challenge of choosing thresholds that separate real stragglers from noise.  
 

## **Q7. What is a principled way to set epsilon for host-side vs. GPU-side detection?**

The naive approach is: compare `time_created_ns` across ranks; if one differs by more than epsilon, it's a host-side straggler. But if epsilon is too small, normal jitter triggers false positives; if epsilon is too large, a genuine host-side straggler that shifts `time_discovered_started_ns` by the same delta gets misclassified as GPU-side.

I understand the general direction: run a clean (no-straggler) experiment, collect `Δ_created(i,j) = |created_i − created_j|`, `Δ_host(i) = started_i − created_i`, and `Δ_gpu(i) = completed_i − started_i` across many windows and runs, compute the distribution (mean, median, P95, P99), and derive epsilon from something like P99 × some safety multiplier. Once we have that baseline, the classification logic would look like: if `created` or `Δ_host` is anomalously large for one rank → host-side straggler; if both `created` and `started` are aligned but `Δ_gpu` is anomalous → GPU-side straggler; if multiple signals are off → both co-occurring.

## **Q8. Can we distinguish host-side from GPU-side straggler purely from the three timestamps?**

Given only time\_created\_ns, time\_discovered\_started\_ns, time\_discovered\_completed\_ns for each rank, can we always unambiguously determine whether a slowdown is host-side or GPU-side? My concern is the following scenario:  
•   	Rank A has a host-side slowdown: its time\_created\_ns is 5ms later than Rank B's.  
•   	As a result, Rank A's time\_discovered\_started\_ns is also \~5ms later.  
•   	Rank A's completed \- started appears normal (or even shorter, per Q5).  
In this case, the 5ms offset in time\_discovered\_started\_ns is entirely explained by the host-side delay. But if we set epsilon for host-side detection too conservatively (say, epsilon \= 2ms), we might miss the host-side signal and misread the started offset as a GPU-side anomaly. Is there a canonical way to handle this ambiguity?  
 

 

### Draft Doc: Questions on FR‑Based Straggler Attribution (with Our Current Thinking)

#### Background

We would like to use PyTorch Flight Recorder (FR) timestamps, together with optional NCCL/CUDA timing, to attribute **host‑side** and **GPU‑side** stragglers, and to integrate this into the existing FR‑based failure attribution pipeline (`fr_attribution.py`). This document summarizes:

- Our current working mental model (what we think is happening).  
- The concrete questions we would like to clarify.  
- The next steps we see for validating and implementing this.

---

### 1\. Our Current Working Model

##### 1.1 Runtime “elapsed” and trigger

- At runtime, we imagine tracking a **coarse‑grained elapsed time per training step or section**:  
  - For example, a full training step or a `Detector.detection_section("train_step")`‑like scope that covers compute \+ comm \+ optimizer.  
- We maintain a **moving average** (and possibly variance) of this step/section elapsed time.  
- When the current elapsed time significantly exceeds the moving average (e.g., `current > moving_avg + k·std`), we **trigger a “profiling window”**:  
  - For the next **N steps** or **T seconds**, we enable FR/NCCL timing to collect more detailed traces.  
  - After the window closes, we disable timing to avoid overhead.

This runtime window is purely a **“where to zoom in”** mechanism; it does not change how we interpret FR internally.

##### 1.2 FR timestamps and what they mean

For each collective on each rank, we assume roughly the following:

- **Host‑side:**  
  - `time_created_ns` (or analogous field) ≈ when the collective is created/scheduled on the host.  
  - `time_discovered_started_ns - time_created_ns` ≈ host‑side / pre‑kernel delay (including runtime overhead \+ queued compute before comm can start).  
- **GPU‑side:**  
  - `time_discovered_started_ns` ≈ when the comm kernel actually starts executing on GPU.  
  - `time_discovered_completed_ns` ≈ when the comm kernel finishes on GPU.  
  - `time_discovered_completed_ns - time_discovered_started_ns` ≈ **GPU comm duration** for that collective.

We understand that:

- Host‑side scheduling time (`time_created_ns`) is available in FR by default.  
- To get accurate GPU‑side start/end, we may need `TORCH_NCCL_ENABLE_TIMING` or equivalent CUDA event timing.

##### 1.3 Two kinds of “windows”

We distinguish two window concepts:

1. **Runtime collection window** (N steps or T seconds):  
     
   - Decides **which temporal segment of the training run** we collect detailed FR traces for, after detecting that the run has slowed down.

   

2. **Offline analysis window** (PG/phase‑based, as in `group_collectives_by_windows`):  
     
   - Given a set of FR dumps, groups collectives by `(process_group, subgroup, window_idx)` so that we align **“wavefronts” of a given PG** across ranks.

Our current assumption is:

- The **runtime window** crops the global timeline into a smaller segment we care about.  
- Inside that cropped segment, we still use the existing **PG/phase windowing** (`group_collectives_by_windows`) to align collectives across ranks before comparing timings.

##### 1.4 Host‑side vs GPU‑side stragglers

Our current interpretation of the HPC intuition (“late comer has shortest timing / highest BW”) is:

- **Host‑side straggler:**  
    
  - Within the same PG and the same wavefront/window:  
    - A rank whose **scheduling / host delay** is a clear outlier:  
      - Either `time_created_ns` itself is much later than peers, or  
      - `time_discovered_started_ns - time_created_ns` is much larger than peers.  
  - This can be detected **without GPU timing**, using FR’s default scheduling timestamps.


- **GPU‑side straggler:**  
    
  - Within the same PG and the same wavefront/window:  
    - The “late comer” rank starts its comm kernel significantly later than others:  
      `time_discovered_started_ns` is much larger than peers.  
    - Because all ranks must finish the collective together, this rank’s **kernel duration** is actually **shortest**:  
      `time_discovered_completed_ns - time_discovered_started_ns` is smaller than peers.  
    - Thus, its **effective BW** (bytes / duration) appears artificially **high**.  
  - This requires **GPU start/end timing** (NCCL/CUDA timing).

In both cases, we do not assume timestamps are exactly equal in non‑straggler runs; rather, we expect them to follow a relatively **tight baseline distribution**, and we treat anomalies as **statistical outliers**.

##### 1.5 Integration with existing FR attribution

Our current understanding of “straggler can be added easily” / “not a separate one” is:

- We can **reuse** the existing FR ingestion and matching logic:  
  - `process_file`, `collectives_by_file`, `group_collectives_by_windows`, and the cross‑rank PG/window matching in `analyze_matches`.  
- We would **add metrics** per `(PG, window, rank)`:  
  - Host delay (`created`, `started - created`),  
  - GPU duration (`completed - started`),  
  - Effective BW (size / duration).  
- Straggler attribution then runs as an additional view on top of the same PG/window alignment used for failure attribution, with different metrics and thresholds.

---

Based on this working model, the key questions we would like to ask are:

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

### 3\. Proposed Next Steps

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

We would appreciate feedback on whether our current mental model matches your intent, and which parts of the above plan you would adjust or simplify before we invest in baseline studies and prototyping.  
