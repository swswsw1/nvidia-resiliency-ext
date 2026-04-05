#!/bin/bash
# Launch straggler injection experiment inside the container.
#
# Usage:
#   ./run.sh none           # clean baseline
#   ./run.sh host 3 50      # host-side straggler on rank 3, 50ms delay
#   ./run.sh kernel 3 50    # kernel-side straggler on rank 3, 50ms delay

set -euo pipefail

INJECT_TYPE=${1:-none}
INJECT_RANK=${2:-3}
INJECT_DELAY_MS=${3:-50}
OUTPUT_DIR="/workspace/nvidia-resiliency-ext/experiments/straggler_injection/traces_${INJECT_TYPE}"

echo "=== Straggler Injection Experiment ==="
echo "Type: ${INJECT_TYPE}, Rank: ${INJECT_RANK}, Delay: ${INJECT_DELAY_MS}ms"
echo "Output: ${OUTPUT_DIR}"
echo "======================================="

docker exec wei_straggler_exp bash -c "
  cd /workspace/nvidia-resiliency-ext &&
  TORCH_NCCL_ENABLE_TIMING=1 \
  NCCL_NVLS_ENABLE=0 \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  torchrun --nproc_per_node=8 \
    experiments/straggler_injection/run_straggler_exp.py \
    --inject-type ${INJECT_TYPE} \
    --inject-rank ${INJECT_RANK} \
    --inject-delay-ms ${INJECT_DELAY_MS} \
    --output-dir ${OUTPUT_DIR}
"
