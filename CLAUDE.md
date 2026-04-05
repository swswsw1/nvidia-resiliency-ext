# CLAUDE.md — Wei's NVRx Work

## Who I Am

Wei Shen, PhD student at UW CSE. Advised by Ratul Mahajan and Arvind Krishnamurthy. Main codebase collaborator: Seonmyeong Bak (sbak) at NVIDIA. Mentoring undergrad Sarju.

## The Project: FR-Based Straggler Detection

I am building straggler detection for distributed GPU training using PyTorch Flight Recorder (FR) traces, within the NVRx codebase. This is my PhD project contribution.

### What it is
Use FR timestamps to detect communication stragglers at per-collective, per-PG granularity and **attribute them to the root cause rank/PG** — tracing observed slowdowns back to the earliest straggler in the causal chain via PG overlap graph traversal. Distinguishes host-side (late `time_created_ns`) vs GPU-side (shortest `gpu_duration`) straggler types within each window. Reuses windowing and graph traversal from sbak's fault attribution code. See `fr_concepts.md` §1 (windowing), §4 (graph traversal), §10 (why PG granularity).

### Why it matters
The existing CUPTI-based straggler module (and all similar approaches) can only say "rank X was slow in section Y." It cannot tell you which collective was the bottleneck, which PG it belonged to, whether the slowdown was host-side or GPU-side, or — critically — **which rank caused the slowdown first** vs which ranks were just downstream victims. FR-based detection with graph traversal fills all of these gaps. See `fr_concepts.md` §10 for the full argument.

### Two straggler types
Both require comparing the relevant signal across ranks within the same window — not a single-entry check.

- **Host CPU-side**: late `time_created_ns` — rank's CPU scheduled the collective late. Detectable with default FR (no extra config).
- **Kernel GPU-side**: shortest `gpu_duration` (`completed - started`) — counterintuitive: in synchronous collectives, the late-arriving rank's kernel runs shortest because all ranks finish together. Requires `TORCH_NCCL_ENABLE_TIMING`.

For the full detection design, see `design/fr_straggler_design.md` (initial proposal draft, shared with sbak and Ratul).

### Current status: traces collected, analysis next
- Straggler injection experiments completed on Cayenne (B200, 8 GPUs): baseline, host-side, kernel-side traces collected with FR timing enabled. See experiment doc for details.
- No straggler detection/analysis code yet — traces need to be analyzed to verify the expected signals.
- Need experimental results to convince Ratul → Ratul talks to Amar about collaboration/cluster access.
- sbak said NVIDIA "will make their impl soon" — there is time pressure.
- sbak: "You don't need to beat existing NVRx. Simply showcase how well your proposed impl works."

### TODOs
1. ~~**Run straggler injection experiment**~~ — traces collected (baseline, host, kernel).
2. **Verify injection signals in traces** ← *in progress* — confirm that host-side and kernel-side injections produce the expected FR timestamp patterns before moving to the full analyzer.
3. **Build offline analyzer** — verify FR signals in traces, demonstrate detection capability.
4. **Deepen the proposal doc** — showcase detection capability with data.
5. *(Deferred)* Ring buffer size + trigger mechanism design — sbak: "trace dump should happen by trainer."
6. *(Deferred)* Graph traversal for stragglers — sbak confirmed same logic as fault attribution applies. "If there are multiple active PGs, each forms a separate path."

### Cluster access
- **Cayenne** (8× B200): primary. Shared — check `nvidia-smi` before running. Requires `NCCL_NVLS_ENABLE=0`.
- **Coriander** (8× H200): secondary. No NVLS issue. Also shared.
- **Tillicum/Klone** (UW): checkpoint partition, 2 nodes of H200, working on getting 4.

## Key Files

### Straggler-related (my focus)
- `src/nvidia_resiliency_ext/attribution/straggler/` — existing CUPTI-based module (the baseline)
  - `straggler.py` — `Detector` class, `detection_section` API
  - `reporting.py` — statistical scoring, all-gather across ranks
- `src/nvidia_resiliency_ext/attribution/trace_analyzer/fr_attribution.py` — windowing + attribution pipeline (sbak's code, foundation I build on)
- `tests/attribution/unit/fr_traces/` — existing fault injection traces (gpu_error, lock_gil variants)

### My docs
- `design/fr_straggler_design.md` — initial proposal draft (see fr_concepts.md for current reference)
- `design/fr_concepts.md` — FR deep reference (timestamps, windowing, PG ID systems)
- `experiments/straggler_injection_experiment_doc.md` — experiment status, implementation, problems encountered

## How We Document Experiments

The experiment doc (`experiments/straggler_injection_experiment_doc.md`) is the source of truth for what we tried and why. When hitting blockers or pivoting approach during experiments:
- **Always update the "Problems encountered and solutions" section** in the experiment doc before changing code. Each entry should describe: what was tried, what happened, and the fix/workaround.
- Code can be updated cleanly — don't carry investigation history in code comments.
- There are separate "Problems encountered" sections for environment setup vs implementation. Put blockers in the right one.

@design/fr_concepts.md
@experiments/straggler_injection_experiment_doc.md

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
| `TORCH_NCCL_TRACE_BUFFER_SIZE=N` | Set FR ring buffer size (default 0/disabled in nv25.04 — must set explicitly) |
| `NCCL_NVLS_ENABLE=0` | Disable NVLS on B200 (workaround for NCCL 2.26.3 hang on Cayenne) |

## Code Quality

- All commits GPG-signed (`git commit -S`), format: `#<Issue Number> - <Commit Title>` (imperative mood)
- PR prefix `[WIP]` while under review
- Contributions through personal forks; do not push directly to upstream
- All changes must begin with a tracked issue, approved by NVRx engineers before code review starts
- New components require an accompanying README and at least one test
- Build log must be clean (no warnings or errors) before submitting a PR