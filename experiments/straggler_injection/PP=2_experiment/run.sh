#!/bin/bash
set -euo pipefail

INJECT_TYPE=${1:-none}
INJECT_RANK=${2:-3}
INJECT_DELAY_MS=${3:-50}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/workspace/nvidia-resiliency-ext/experiments/straggler_injection/PP=2_experiment/traces/${TIMESTAMP}_${INJECT_TYPE}_rank${INJECT_RANK}_${INJECT_DELAY_MS}ms"

echo "=== Straggler Injection Experiment ==="
echo "Type: ${INJECT_TYPE}, Rank: ${INJECT_RANK}, Delay: ${INJECT_DELAY_MS}ms"
echo "Output: ${OUTPUT_DIR}"
echo "======================================="

docker exec kevin_straggler_exp bash -c "
  mkdir -p ${OUTPUT_DIR} &&
  echo 'inject_type: ${INJECT_TYPE}
inject_rank: ${INJECT_RANK}
inject_delay_ms: ${INJECT_DELAY_MS}
timestamp: ${TIMESTAMP}
parallelism: TP=2, PP=2, DP=2
num_iterations: 30' > ${OUTPUT_DIR}/run_config.log &&
  PYTHONPATH=/workspace/Megatron-LM \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  TORCH_NCCL_TRACE_BUFFER_SIZE=10000 \
  TORCH_NCCL_ENABLE_TIMING=1 \
  torchrun --nproc_per_node=8 \
    /workspace/nvidia-resiliency-ext/experiments/straggler_injection/PP=2_experiment/run_simple_pp2_train_loop.py \
    --inject-type ${INJECT_TYPE} \
    --inject-rank ${INJECT_RANK} \
    --inject-delay-ms ${INJECT_DELAY_MS} \
    --output-dir ${OUTPUT_DIR}
"
