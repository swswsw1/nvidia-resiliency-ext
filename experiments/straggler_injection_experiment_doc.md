## Environment Setup

### Overview

Container `wei_straggler_exp` runs on `nvcr.io/nvidia/pytorch:25.04-py3` with Megatron-LM and NVRx installed from source. The setup has been tested on two clusters; adapt the volume mount and UID to your environment.

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

### What we're building

One script: `nvidia-resiliency-ext/experiments/straggler_injection/run_straggler_exp.py`

Based on Megatron-LM's `examples/run_simple_mcore_train_loop.py`, extended with:
- TP=2, PP=2, DP=2 on 8 GPUs (all parallelisms active)
- Configurable straggler injection (host-side, kernel-side, or none)
- FR trace dumping with `TORCH_NCCL_ENABLE_TIMING=1`

Plus a launch script: `nvidia-resiliency-ext/experiments/straggler_injection/run.sh`

### Key changes from the base example

1. **Parallelism**: `initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)`
   - DP=2 is implicit (8 GPUs / TP=2 / PP=2 = DP=2)
   - `num_layers=4` (must be divisible by PP=2 → 2 layers per stage)
   - `num_microbatches=4` (must be ≥ PP=2)
   - `hidden_size` increased from 12 to something larger (e.g., 256) so compute is non-trivial and collectives are meaningful

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
#!/bin/bash
INJECT_TYPE=${1:-none}   # none, host, kernel
INJECT_RANK=${2:-3}
INJECT_DELAY_MS=${3:-50}
OUTPUT_DIR="./traces_${INJECT_TYPE}"

docker exec wei_straggler_exp bash -c "
  cd /workspace/nvidia-resiliency-ext &&
  TORCH_NCCL_ENABLE_TIMING=1 \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  torchrun --nproc_per_node=8 \
    experiments/straggler_injection/run_straggler_exp.py \
    --inject-type $INJECT_TYPE \
    --inject-rank $INJECT_RANK \
    --inject-delay-ms $INJECT_DELAY_MS \
    --output-dir $OUTPUT_DIR
"
```

Run three times:
```bash
./run.sh none       # baseline
./run.sh host 3 50  # host-side straggler on rank 3
./run.sh kernel 3 50 # kernel-side straggler on rank 3
```

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
4. **Sanity check**: Run `group_collectives_by_windows()` on the traces to verify windowing produces meaningful buckets with multiple PG types (TP, PP, DP)
