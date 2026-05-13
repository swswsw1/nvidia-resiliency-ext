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
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/workspace/nvidia-resiliency-ext/experiments/straggler_injection/traces/${TIMESTAMP}_${INJECT_TYPE}"

echo "=== Straggler Injection Experiment ==="
echo "Type: ${INJECT_TYPE}, Rank: ${INJECT_RANK}, Delay: ${INJECT_DELAY_MS}ms"
echo "Output: ${OUTPUT_DIR}"
echo "======================================="

docker exec wei_straggler_exp bash -c "
  mkdir -p ${OUTPUT_DIR} &&
  echo 'inject_type: ${INJECT_TYPE}
inject_rank: ${INJECT_RANK}
inject_delay_ms: ${INJECT_DELAY_MS}
timestamp: ${TIMESTAMP}
parallelism: TP=2, PP=1, DP=4
num_iterations: 30' > ${OUTPUT_DIR}/run_config.log &&
  cd /workspace/nvidia-resiliency-ext &&
  TORCH_NCCL_TRACE_BUFFER_SIZE=10000 \
  TORCH_NCCL_ENABLE_TIMING=1 \
  NCCL_NVLS_ENABLE=0 \
  torchrun --nproc_per_node=8 \
    experiments/straggler_injection/run_straggler_exp.py \
    --inject-type ${INJECT_TYPE} \
    --inject-rank ${INJECT_RANK} \
    --inject-delay-ms ${INJECT_DELAY_MS} \
    --output-dir ${OUTPUT_DIR}
"
