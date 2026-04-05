## Environment Setup

### Overview

Container `wei_straggler_exp` runs on `nvcr.io/nvidia/pytorch:25.04-py3` with Megatron-LM and NVRx installed from source. The setup has been tested on two clusters; 
adapt the volume mount and UID to your environment.


**Problems encountered and solutions:**
1. **NGC 24.12 doesn't support B200**: PyTorch 2.6.0 only has sm_70–sm_90. Switched to 25.04.
2. **UID mapping error in 25.04**: `getpwuid(): uid not found: 1007` broke transformer_engine import.
   - Fix: `echo 'wei23:x:1007:1007::/home/wei23:/bin/bash' >> /etc/passwd`
3. **pip constraint conflicts**: NGC pins `packaging==23.2`, `protobuf==4.24.4` via `/etc/pip/constraint.txt`.
   - Fix: `unset PIP_CONSTRAINT` before pip install commands.
4. **Megatron-LM install pulls newer torch**: `pip install -e .` upgrades to torch 2.11+cu130 (incompatible with driver).
   - Fix: `pip install -e . --no-deps --no-build-isolation`
5. **NVRx build needs poetry + grpcio-tools**:
   - Fix: `pip install poetry-core 'poetry-dynamic-versioning[backend]' grpcio-tools` then install NVRx.
6. **Git safe directories**: Container can't access mounted repos.
   - Fix: `git config --global --add safe.directory /workspace/nvidia-resiliency-ext` (and same for Megatron-LM)
7. **NCCL hangs with 3+ GPUs (Docker device access)**: Without `--privileged`, Docker doesn't expose NVLink/NVSwitch fabric devices. NCCL falls back to Socket transport, which hangs on barrier for 3+ GPUs. 2 GPUs work because they can use direct CUDA P2P without the fabric.
   - Fix: `--gpus all --privileged --ipc=host --ulimit memlock=-1` on `docker run`
   - Diagnosis: `NCCL_DEBUG=INFO` showed `NET/IB : No device found` and `Using network Socket` without `--privileged`; with `--privileged`, IB devices visible (`mlx5_0-3`)
8. **NCCL NVLS hang on B200**: Even with `--privileged`, NCCL 2.26.3 hangs during NVLS (NVLink SHARP / NVSwitch multicast) initialization on B200. `NCCL_DEBUG=INFO` shows channels/trees/proxy setup but never reaches "Init COMPLETE". `MNNVL healthMask 0x80` suggests fabric manager state isn't fully exposed.
   - Fix: `NCCL_NVLS_ENABLE=0` to disable NVLS. Falls back to standard NVLink P2P + ring/tree algorithms.
   - Verified: 8-GPU `dist.barrier()` completes in <5s with this env var.
   - Performance impact: Negligible for our small model experiment. NVLS is an optimization for large collectives.
   - reference to https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable

### Quick Start (Generic)

```bash
# 1. Create container
#    - Replace WORKSPACE_PATH with your local path containing nvidia-resiliency-ext/
#    - --privileged --ipc=host are critical: without them, Docker doesn't expose
#      NVLink/NVSwitch fabric devices and NCCL falls back to Socket transport,
#      which hangs on barrier for 3+ GPUs.
docker run -dit --name wei_straggler_exp \
  --gpus all --privileged --ipc=host --ulimit memlock=-1 \
  -v WORKSPACE_PATH:/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:25.04-py3 bash

# 2. Fix UID mapping (NGC 25.04 doesn't create host user → breaks transformer_engine import)
#    Replace YOUR_UID/YOUR_GID with output of `id -u` / `id -g` on the host
docker exec wei_straggler_exp bash -c \
  "echo 'YOUR_USER:x:YOUR_UID:YOUR_GID::/home/YOUR_USER:/bin/bash' >> /etc/passwd"

# 3. Git safe directories (container can't access mounted repos otherwise)
docker exec wei_straggler_exp bash -c "
  git config --global --add safe.directory /workspace/nvidia-resiliency-ext
  git config --global --add safe.directory /workspace/Megatron-LM
"

# 4. Clone Megatron-LM (if not already present)
docker exec wei_straggler_exp bash -c \
  "cd /workspace && git clone https://github.com/NVIDIA/Megatron-LM.git"

# 5. Install Megatron (--no-deps prevents torch upgrade to incompatible version)
docker exec wei_straggler_exp bash -c "
  unset PIP_CONSTRAINT &&
  cd /workspace/Megatron-LM && pip install -e . --no-deps --no-build-isolation
"

# 6. Install NVRx build deps + NVRx (skip CUPTI C++ build if no need)
docker exec wei_straggler_exp bash -c "
  unset PIP_CONSTRAINT &&
  pip install poetry-core 'poetry-dynamic-versioning[backend]' grpcio-tools &&
  cd /workspace/nvidia-resiliency-ext &&
  STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1 pip install -e . --no-build-isolation
"
```

### Verification

```bash
docker exec wei_straggler_exp python -c "
import torch; print(f'torch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, {torch.cuda.get_device_name(0)}')
import megatron; print('megatron: OK')
import nvidia_resiliency_ext; print('nvidia_resiliency_ext: OK')
"
```

Then test 8-GPU NCCL barrier:
```bash
docker exec wei_straggler_exp bash -c "
cat > /tmp/test_nccl.py << 'PYEOF'
import torch, torch.distributed as dist
dist.init_process_group('nccl')
dist.barrier()
if dist.get_rank() == 0: print('8-GPU NCCL barrier: OK')
dist.destroy_process_group()
PYEOF
torchrun --nproc_per_node=8 /tmp/test_nccl.py
"
```

### Cluster-Specific Notes

**Note**: Both clusters are shared. Check `nvidia-smi` before running experiments to ensure GPUs are available.

#### Cayenne (B200, 8x NVIDIA B200 sm_100)

- **Workspace mount**: `-v /raid/wei23/wei:/workspace`
- **UID**: 1007 (`echo 'wei23:x:1007:1007::/home/wei23:/bin/bash' >> /etc/passwd`)
- **NCCL NVLS hang (B200-specific)**: Even with `--privileged`, NCCL 2.26.3 hangs during NVLS (NVLink SHARP / NVSwitch multicast) initialization on B200. `NCCL_DEBUG=INFO` shows channels/trees/proxy setup but never reaches "Init COMPLETE". `MNNVL healthMask 0x80` suggests fabric manager state isn't fully exposed.
  - Fix: `NCCL_NVLS_ENABLE=0` to disable NVLS. Falls back to standard NVLink P2P + ring/tree algorithms.
  - Performance impact: Negligible for our small model experiment. NVLS is an optimization for large collectives.
  - Ref: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable
- **Verified**: PyTorch 2.7.0a0+nv25.04, TE 2.2.0, Flash Attention 2.7.3, Apex 0.1
- **Verified**: 8-GPU barrier OK (with `NCCL_NVLS_ENABLE=0`), Megatron example TP=2 on 2 GPUs OK

#### Coriander (H200, 8x NVIDIA H200 sm_90a) 

- **Workspace mount**: `-v /m-coriander/coriander/wei23:/workspace`
- **UID**: 1011 (`echo 'wei23:x:1011:1011::/home/wei23:/bin/bash' >> /etc/passwd`)
- **No NVLS issue**: H200 NCCL works without `NCCL_NVLS_ENABLE=0`. 8-GPU barrier completes fine out of the box.
- **Verified (2026-04-05)**: PyTorch 2.7.0a0+nv25.04, megatron-core 0.18.0rc0, NVRx 0.0.0.dev784
- **Verified**: 8-GPU NCCL barrier OK (no NVLS workaround needed), all imports OK

### Common Gotchas (Both Clusters)

1. **NGC 24.12 doesn't support B200**: PyTorch 2.6.0 only has sm_70–sm_90. Use 25.04.
2. **pip constraint conflicts**: NGC pins `packaging==23.2`, `protobuf==4.24.4` via `/etc/pip/constraint.txt`. Fix: `unset PIP_CONSTRAINT` before pip install commands.
3. **Megatron-LM install pulls newer torch**: `pip install -e .` upgrades to torch 2.11+cu130 (incompatible with driver). Fix: `--no-deps --no-build-isolation`.
4. **NCCL hangs with 3+ GPUs without `--privileged`**: Docker doesn't expose NVLink/NVSwitch fabric devices. Diagnosis: `NCCL_DEBUG=INFO` shows `NET/IB : No device found` and `Using network Socket`.

---

## Implementation Plan

### Context

We need a Megatron training script with straggler injection to demonstrate that FR-based detection (with windowing) can identify host-side and kernel-side stragglers. This is a POC/MVP for Monday's meeting.


### Problems encountered and solutions

1. **PP=2 hangs on Cayenne B200 (2026-04-05)**: Originally planned TP=2, PP=2, DP=2. The 8-GPU run gets past NCCL init (version 2.26.3 printed), dataset creation, and model setup, but never prints a single iteration. `nvidia-smi` shows GPUs 4–7 (PP stage 1) at 100% utilization spinning, GPUs 0–3 (PP stage 0) at 0% — PP scheduling deadlock where stage 1 waits for stage 0 activations that never arrive. Reproduced consistently (12+ minutes, then reproduced with `timeout 120`). `NCCL_NVLS_ENABLE=0` was already set. Also tried adding `device_id` to `init_process_group()` for the "device used by this process is currently unknown" warnings — no effect. Root cause is likely in Megatron's PP forward/backward schedule (`get_forward_backward_func`) interacting with NCCL 2.26.3 on B200, not basic NCCL connectivity (barrier works fine).
   - Fix: Dropped PP, switched to TP=2, PP=1, DP=4. Still gives us TP collectives (allgather, reduce_scatter) and DP collectives (allreduce for grad sync) — sufficient for the FR straggler detection POC. Can revisit PP after POC if needed.

2. **Empty FR traces — ring buffer not populated (2026-04-05)**: All three experiment runs completed successfully (30 iterations each) but `_dump_*.json` files were 85 bytes — `pg_config: {}`, `pg_status: {}`, `entries: []`. `TORCH_NCCL_ENABLE_TIMING=1` was set in the launch script but `TORCH_NCCL_TRACE_BUFFER_SIZE` was not. In PyTorch 2.7.0a0+nv25.04, the FR ring buffer defaults to 0 (disabled) unless explicitly set.
   - Fix: Added `TORCH_NCCL_TRACE_BUFFER_SIZE=10000` to `run.sh`. Verified with a single-GPU allreduce test that both env vars together produce populated traces with `pg_config`, `pg_status`, entries with `time_created_ns` and `time_discovered_started_ns`.
   - Note: The default ring buffer size is 2000, which sbak says is enough for fault analysis but not for straggler detection in real training runs. For our toy 30-iteration experiment on one node, 10000 is more than sufficient. For real multi-node runs with thousands of iterations, the buffer would overflow and evict early entries — either a larger buffer or periodic dumping from the trainer would be needed (see fr_concepts.md §11).


### What we're building

One script: `nvidia-resiliency-ext/experiments/straggler_injection/run_straggler_exp.py`

Based on Megatron-LM's `examples/run_simple_mcore_train_loop.py`, extended with:
- TP=2, PP=1, DP=4 on 8 GPUs (see "Problems encountered" for why PP was dropped)
- Configurable straggler injection (host-side, kernel-side, or none)
- FR trace dumping with `TORCH_NCCL_ENABLE_TIMING=1`

Plus a launch script: `nvidia-resiliency-ext/experiments/straggler_injection/run.sh`

### Key changes from the base example

1. **Parallelism**: `initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)`
   - DP=4 is implicit (8 GPUs / TP=2 / PP=1 = DP=4)
   - `num_layers=4`
   - `num_microbatches=1` (no PP pipeline to fill)
   - `hidden_size=256` so compute is non-trivial and collectives are meaningful

2. **Injection via CLI args**:
   ```
   --inject-type {none, host, kernel}
   --inject-rank 3
   --inject-delay-ms 50
   ```
   Injection happens every iteration. Clean baseline is a separate run with `--inject-type none`.

3. **Host-side injection**: `time.sleep(delay_s)` at the start of the training iteration on the target rank, before compute begins. Simulates slow dataloader / OS preemption.

4. **Kernel-side injection**: `torch.cuda._sleep(cycles)` on the target rank's CUDA stream, submitted before the forward pass (no CPU sync). GPU is busy, CPU continues normally. Simulates preceding compute taking longer.

5. **FR trace dumping**: At the end of training, every rank calls:
   ```python
   trace = torch._C._distributed_c10d._dump_nccl_trace(
       includeCollectives=True, includeStackTraces=False, onlyActive=False
   )
   ```
   Writes `_dump_{rank}.json` to output directory. Uses `onlyActive=False` to capture completed ops.

6. **Iterations**: ~30 total. Injection on every iteration. Clean baseline is a separate run with `--inject-type none`.

### Launch script (`run.sh`)

```bash
./run.sh none           # clean baseline
./run.sh host 3 50      # host-side straggler on rank 3, 50ms delay
./run.sh kernel 3 50    # kernel-side straggler on rank 3, 50ms delay
```

Key env vars set by the script:
- `TORCH_NCCL_TRACE_BUFFER_SIZE=10000` — enables the FR ring buffer (default is 0/disabled in nv25.04)
- `TORCH_NCCL_ENABLE_TIMING=1` — populates `time_discovered_started/completed_ns` via cudaEvent
- `NCCL_NVLS_ENABLE=0` — B200 NVLS workaround (see environment problems #8)

### Trace output

Traces are stored under `experiments/straggler_injection/traces/`, one timestamped directory per run (e.g. `20260405_135545_none/`). Each contains `_dump_{0..7}.json` (one per rank) and a `run_config.log` with the run parameters. Directories are never overwritten — each run gets its own timestamp.

### The three experiment types

- **none** (baseline) — no injection. Clean run that gives us the "normal" timing distributions to compare against.
- **host** (host-side straggler) — `time.sleep(50ms)` delays the CPU on the target rank before each iteration. Rank 3's `time_created_ns` should be visibly late relative to peers in the same PG/window, because the host scheduled the collective later.
- **kernel** (kernel-side straggler) — `torch.cuda._sleep()` delays the GPU stream on the target rank, but CPU returns immediately. Rank 3's `time_created_ns` stays normal (CPU isn't blocked), but `time_discovered_started_ns` should be late (GPU was busy). The straggler rank's `gpu_duration` should be *shorter* than peers — because in synchronous collectives, all ranks finish at approximately the same wall-clock time, so the late-starting rank runs the comm kernel for less time.

### Where injection hooks go in the training loop

```python
for iteration in range(num_iterations):
    # === HOST-SIDE INJECTION POINT ===
    # time.sleep() here → delays CPU → late time_created_ns for all
    # subsequent collectives this iteration
    if inject_type == "host" and rank == inject_rank:
        time.sleep(delay_s)

    # === KERNEL-SIDE INJECTION POINT ===
    # torch.cuda._sleep() here → delays GPU stream → late time_discovered_started_ns
    # for subsequent comm kernels, but CPU proceeds normally → normal time_created_ns
    if inject_type == "kernel" and rank == inject_rank:
        torch.cuda._sleep(gpu_delay_cycles)

    optim.zero_grad()
    losses = forward_backward_func(...)
    finalize_model_grads([model])
    optim.step()
```

### GPU sleep cycle calibration

`torch.cuda._sleep(N)` takes clock cycles. On B200 (~2.1 GHz boost):
- 50ms ≈ 105M cycles → `int(50e-3 * 2.1e9)` = `105_000_000`
- We'll calibrate empirically in the first run by measuring actual delay.

### Files to create

| File | Purpose |
|------|---------|
| `experiments/straggler_injection/run_straggler_exp.py` | Main training + injection + FR dump script |
| `experiments/straggler_injection/run.sh` | Launch wrapper with configurable injection params |

### Files to reference (read-only)

| File | What we reuse |
|------|--------------|
| `Megatron-LM/examples/run_simple_mcore_train_loop.py` | Base training loop structure |
| `src/.../trace_analyzer/trace_collector.py` | FR dump pattern (`_dump_nccl_trace` call) |
| `src/.../inprocess/tools/inject_fault.py` | `torch.cuda._sleep()` pattern (line 207) |
| `src/.../trace_analyzer/fr_attribution.py` | `group_collectives_by_windows()` for later analysis |

### Documentation

- Environment setup is documented at the top of this plan file (one-time, not repeated)
- Each experiment run will be logged in `experiments/straggler_injection/EXPERIMENT_LOG.md`:
  - Command used
  - Config (inject type, rank, delay, iterations)
  - Key results (timestamps, signals observed)
  - Any issues encountered
- No duplication between setup docs and run docs


### Verification

1. **Baseline run**: All 8 ranks complete 30 iterations, FR traces dumped with timing fields populated (not null)
2. **Host-side run**: Rank 3's `time_created_ns` is visibly later than other ranks in injection iterations
3. **Kernel-side run**: Rank 3's `time_created_ns` is normal but `time_discovered_started_ns` is late, `gpu_duration` is shorter than peers
4. **Sanity check**: Run `group_collectives_by_windows()` on the traces to verify windowing produces meaningful buckets with multiple PG types (TP, DP)
