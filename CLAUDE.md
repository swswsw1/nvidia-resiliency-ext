# CLAUDE.md — Wei's NVRx Work

## Who I Am

Wei Shen, PhD student at UW CSE. Advised by Ratul Mahajan and Arvind Krishnamurthy. Main codebase collaborator: Seonmyeong Bak (sbak) at NVIDIA, whose manager Amar oversees NVRx. Ratul and sbak's new manager (Phanishayee) know each other from MSR.

## The Project: FR-Based Straggler Detection

I am building straggler detection for distributed GPU training using PyTorch Flight Recorder (FR) traces, within the NVRx codebase. This is my PhD project contribution.

### What it is
Use FR timestamps to detect and attribute communication stragglers at per-collective, per-PG granularity — distinguishing host-side vs GPU-side stragglers. Reuses the windowing infrastructure from sbak's existing fault attribution code.

### Why it matters
The existing CUPTI-based straggler module (and all similar approaches) can only say "rank X was slow in section Y." It cannot tell you which collective was the bottleneck, which PG it belonged to, or whether the slowdown was host-side or GPU-side. FR-based detection fills this gap. See `docs/cupti_vs_fr.md` for the full argument.

### Two straggler types
- **Host-side**: late `time_created_ns` — rank's CPU scheduled the collective late. Detectable with default FR (no extra config).
- **GPU-side**: shortest `gpu_duration` (`completed - started`) — counterintuitive: in synchronous collectives, the late-arriving rank's kernel runs shortest because all ranks finish together. Requires `TORCH_NCCL_ENABLE_TIMING`.

### Current status: proposal stage, preparing experiments
- Proposal written: `docs/fr_straggler_design.md` (shared with sbak and Ratul)
- No straggler detection code yet
- Need to produce experimental results to convince Ratul → Ratul talks to Amar about collaboration/cluster access
- sbak said NVIDIA "will make their impl soon" — there is time pressure
- sbak also said: "You don't need to beat existing NVRx. Simply showcase how well your proposed impl works."

### Key design decisions still open
- **Ring buffer size**: FR buffer is 2000 entries — enough for fault attribution (snapshot at crash), NOT enough for straggler detection (need to observe patterns over time). sbak: "trace dump should happen by trainer." Runtime capture window design must address this.
- **When to start capturing**: The trigger mechanism. Current idea: detect elapsed time anomaly at section level, then enable FR timing for N steps. But need a global barrier or sync point as starting reference.
- **Graph traversal for stragglers**: sbak confirmed the same graph logic applies — different ranks slow in different collectives, trace back through PG overlap graph to find root straggler PG. "If there are multiple active PGs, each forms a separate path."

### Foundation: windowing + attribution knowledge
I learned straggler detection's infrastructure by studying sbak's fault attribution code (`fr_attribution.py`). The key shared mechanism is `group_collectives_by_windows()` — it replays all ranks' timelines and groups collectives into `(PG_type, sub_group, window_idx)` buckets via majority-vote wavefront selection. This is what enables cross-rank matching of "the same logical collective" without relying on seq_id alignment (which breaks with p2p). For straggler detection, the same windowing gives us the buckets within which to compare timing across ranks.

See `docs/fr_concepts.md` for detailed reference on FR dump structure, ID systems, windowing mechanism, and timestamps.

### Immediate TODOs (what Ratul wants)
1. **Run straggler injection experiment** — inject slowness (host-side and kernel-side), show FR-based approach catches it. sbak confirmed CUPTI-based detector misses comm kernel stragglers (it filters NCCL kernels out).
2. **Showcase the method works** — not a comparison paper, just demonstrate detection capability with data.
3. **Deepen the proposal doc** — address ring buffer limitation, trigger mechanism, graph traversal for stragglers.

### Cluster access situation
- dgx01: 1 node, primary dev server
- Tillicum/Klone (UW): checkpoint partition only, can get 2 nodes of H200, trying to get 4
- AI2 (Dirk): would require setting up a formal straggler project to use their cluster
- NVIDIA cluster access: depends on Ratul convincing Amar

## Key Files

### Straggler-related (my focus)
- `src/nvidia_resiliency_ext/attribution/straggler/` — existing CUPTI-based module (the baseline)
  - `straggler.py` — `Detector` class, `detection_section` API
  - `reporting.py` — statistical scoring, all-gather across ranks
  - `cupti_src/` — C++ CUPTI extension
- `src/nvidia_resiliency_ext/attribution/trace_analyzer/fr_attribution.py` — windowing + attribution pipeline (sbak's code, foundation I build on)
- `tests/attribution/unit/fr_traces/` — test traces: gpu_error_1st, gpu_error_2nd, lock_gil_1st, lock_gil_2nd
- `tests/attribution/unit/` — attribution unit tests
- `tests/straggler/unit/` — straggler unit tests

### My docs
- `docs/fr_straggler_design.md` — straggler detection proposal (actively updating)
- `docs/fr_concepts.md` — FR concept quick reference
- `docs/cupti_vs_fr.md` — why FR works for distributed straggler detection and CUPTI doesn't

## Build and Test

```bash
# Install from source (skip CUPTI if no GPU)
STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1 pip install .

# Run attribution tests
pytest -s -vvv ./tests/attribution/unit/

# Run straggler tests (CPU-only subset)
pytest -s -vvv tests/straggler/unit/ -k "test_all_gather_object_calls_num or test_fail_if_not_initialized"

# Format / lint (black==24.10.0, isort==5.13.2 profile="black", ruff==0.6.9, line length 100)
black . && isort . && ruff check .
```

## Environment

- Primary dev: `dgx01` at `/raid/wei23/wei/nvidia-resiliency-ext/`
- Claude Code on dgx01 via SyFI relay (`cayenne.cs.washington.edu:3456`)
- Mac local + Cursor Remote SSH to dgx01

## Terminology Quick Reference

| Term | Meaning |
|------|---------|
| FR | Flight Recorder — PyTorch's per-rank ring buffer of collective metadata |
| PG | Process Group — a subset of ranks that communicate together |
| megatron_id | Framework-level PG identifier (key in `pg_config`) |
| c10d_handle / pg_id | Backend PG handle (key in `pg_status`). Same handle on different ranks = same PG slot, NOT same instance |
| collective_seq_id | Per-PG, per-rank local counter. NOT usable for cross-rank alignment |
| Window | `(PG_type, sub_group, window_idx)` — one round of a PG across ranks, from wavefront replay |
| Wavefront | PG type most ranks are currently at (majority vote) |
| sub_group | Specific PG instance (e.g., TP[0,1] vs TP[2,3]) |
| host_delay | `time_discovered_started_ns - time_created_ns` |
| gpu_duration | `time_discovered_completed_ns - time_discovered_started_ns` |

For deeper reference: `docs/fr_concepts.md`

## Code Quality

- All commits GPG-signed, format: `#<Issue Number> - <Commit Title>`
- PR prefix `[WIP]` while under review
- Contributions through personal forks
