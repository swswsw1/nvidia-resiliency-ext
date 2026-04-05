# CLAUDE.md — Wei's NVRx Work

## Who I Am

Wei Shen, PhD student at UW CSE. Advised by Ratul Mahajan and Arvind Krishnamurthy. Main codebase collaborator: Seonmyeong Bak (sbak) at NVIDIA, 

## The Project: FR-Based Straggler Detection

I am building straggler detection for distributed GPU training using PyTorch Flight Recorder (FR) traces, within the NVRx codebase. This is my PhD project contribution.

### What it is
Use FR timestamps to detect and attribute communication stragglers at per-collective, per-PG granularity — distinguishing host-side vs GPU-side stragglers. Reuses the windowing infrastructure from sbak's existing fault attribution code.

### Why it matters
The existing CUPTI-based straggler module (and all similar approaches) can only say "rank X was slow in section Y." It cannot tell you which collective was the bottleneck, which PG it belonged to, or whether the slowdown was host-side or GPU-side. FR-based detection fills this gap. See Section 10 of `design/fr_concepts.md` for the full argument.

### Two straggler types
- **Host-side**: late `time_created_ns` — rank's CPU scheduled the collective late. Detectable with default FR (no extra config).
- **GPU-side**: shortest `gpu_duration` (`completed - started`) — counterintuitive: in synchronous collectives, the late-arriving rank's kernel runs shortest because all ranks finish together. Requires `TORCH_NCCL_ENABLE_TIMING`.

### Current status: proposal stage, preparing experiments
- Proposal written: `design/fr_straggler_design.md` (shared with sbak and Ratul)
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

See `design/fr_concepts.md` for detailed reference on FR dump structure, ID systems, windowing mechanism, and timestamps.

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
- `design/fr_straggler_design.md` — straggler detection proposal (actively updating)
- `design/fr_concepts.md` — FR concept quick reference

@design/fr_concepts.md
@design/fr_straggler_design.md

## NVRx Architecture

Source lives in `src/nvidia_resiliency_ext/`. Key modules:

- **`fault_tolerance/`** — In-job restart without reallocating SLURM nodes. `ft_launcher` CLI, gRPC rank monitoring, section-based timeout model (`begin_section`/`end_section`).
- **`inprocess/`** — Detect failures and restart within a single process. Wraps training function with `Wrapper`/`Compose`; background monitors detect hangs; dynamic rank reassignment on restart.
- **`checkpointing/`** — `async_ckpt/` (background saves) and `local/` (fast local-storage saves).
- **`attribution/`** — `straggler/` (CUPTI-based, the baseline I'm extending), `trace_analyzer/` (FR attribution pipeline — sbak's code), LLM log analysis pipeline, MCP server.
- **`ptl_resiliency/`** — PyTorch Lightning callbacks wrapping the above.
- **`shared_utils/`** — Health checks (NVML/gRPC), structured logging, centralized log collection, GPU memory logging.
- **`services/`** — Standalone services at repo root (not inside `src/`): `nvrx_attrsvc/` (FastAPI LLM log analysis), `nvrx_smonsvc/` (SLURM job monitor).

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1` | Skip CUPTI C++ build (use when no GPU/CUDA) |
| `NVRX_LOG_DEBUG=1` | Enable debug logging |
| `CUDA_PATH` | Override CUDA installation path (auto-detected otherwise) |
| `TORCH_NCCL_ENABLE_TIMING=1` | Enable FR GPU-side timestamps (Layer 3, opt-in overhead) |

## Build and Test

```bash
# Install from source (skip CUPTI if no GPU)
STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1 pip install .

# Run tests by module
pytest -s -vvv ./tests/attribution/unit/
pytest -s -vvv ./tests/straggler/unit/
pytest -s -vvv ./tests/fault_tolerance/unit/
pytest -s -vvv ./tests/inprocess/
pytest -s -vvv ./tests/checkpointing/unit/
pytest -s -vvv ./tests/ptl_resiliency/unit/

# Straggler CPU-only subset (no CUPTI needed)
pytest -s -vvv tests/straggler/unit/ -k "test_all_gather_object_calls_num or test_fail_if_not_initialized"

# Notes:
# - Functional tests (func/) require multi-GPU/SLURM — not run in standard CI
# - MKL_SERVICE_FORCE_INTEL=1 may be needed for MKL threading issues
# - Install wheel before CI: pip install ./dist/nvidia_resiliency_ext-*-cp${PY_VER}-*.whl

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

For deeper reference: `design/fr_concepts.md`

## Code Quality

- All commits GPG-signed (`git commit -S`), format: `#<Issue Number> - <Commit Title>` (imperative mood)
- PR prefix `[WIP]` while under review
- Contributions through personal forks; do not push directly to upstream
- All changes must begin with a tracked issue, approved by NVRx engineers before code review starts
- New components require an accompanying README and at least one test
- Build log must be clean (no warnings or errors) before submitting a PR
