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

docker exec sarju2_straggler_exp bash -c "
set -e
mkdir -p ${OUTPUT_DIR}
echo 'inject_type: ${INJECT_TYPE}
inject_rank: ${INJECT_RANK}
inject_delay_ms: ${INJECT_DELAY_MS}
timestamp: ${TIMESTAMP}
parallelism: TP=2, PP=1, DP=4
num_iterations: 30' > ${OUTPUT_DIR}/run_config.log
cd /workspace/nvidia-resiliency-ext

# Launch the online straggler monitor in background. It polls the trace dir
# as the trainer writes chunks, emits verdicts.jsonl, and self-exits on
# quiescence (no new chunks for --quiescence-s seconds).
python3 experiments/straggler_injection/fr_straggler_monitor.py \
    ${OUTPUT_DIR} --world-size 8 > ${OUTPUT_DIR}/monitor.log 2>&1 &
MONITOR_PID=\$!
echo \"[run.sh] monitor pid=\${MONITOR_PID}\"

PYTHONPATH=/workspace/Megatron-LM:/workspace/nvidia-resiliency-ext/src \
TORCH_NCCL_TRACE_BUFFER_SIZE=500 \
TORCH_NCCL_ENABLE_TIMING=1 \
NCCL_NVLS_ENABLE=0 \
torchrun --nproc_per_node=8 \
  experiments/straggler_injection/run_straggler_exp.py \
  --inject-type ${INJECT_TYPE} \
  --inject-rank ${INJECT_RANK} \
  --inject-delay-ms ${INJECT_DELAY_MS} \
  --output-dir ${OUTPUT_DIR}

# Wait for monitor to drain and exit on its own quiescence timer.
wait \${MONITOR_PID}
"
