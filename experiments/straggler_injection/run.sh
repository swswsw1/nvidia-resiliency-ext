#!/bin/bash
# Launch straggler injection experiment inside the container.
#
# Usage:
#   ./run.sh none           # clean baseline
#   ./run.sh host 3 50      # host-side straggler on rank 3, 50ms delay
#   ./run.sh kernel 3 50    # kernel-side straggler on rank 3, 50ms delay
#
# Env overrides:
#   GPUS=0,1,4,5            comma-separated CUDA_VISIBLE_DEVICES (default 0..7)
#   NPROC=4                 must match #GPUs (default 8)
#   TRIGGER=1               enable cheap-stats trigger (default 0)
#   FR_TRIG_WINDOW=10       trigger rolling window
#   FR_TRIG_CHECK_FREQ=3    trigger eval cadence
#   FR_TRIG_PERSISTENCE=2   trigger persistence count

set -euo pipefail

INJECT_TYPE=${1:-none}
INJECT_RANK=${2:-3}
INJECT_DELAY_MS=${3:-50}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
TRIGGER=${TRIGGER:-0}
FR_TRIG_WINDOW=${FR_TRIG_WINDOW:-10}
FR_TRIG_CHECK_FREQ=${FR_TRIG_CHECK_FREQ:-3}
FR_TRIG_PERSISTENCE=${FR_TRIG_PERSISTENCE:-2}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/workspace/nvidia-resiliency-ext/experiments/straggler_injection/traces/${TIMESTAMP}_${INJECT_TYPE}"

# DP = NPROC / TP / PP. With TP=2, PP=1: DP = NPROC / 2.
DP=$(( NPROC / 2 ))

echo "=== Straggler Injection Experiment ==="
echo "Type: ${INJECT_TYPE}, Rank: ${INJECT_RANK}, Delay: ${INJECT_DELAY_MS}ms"
echo "GPUs: ${GPUS} (nproc=${NPROC}, TP=2 PP=1 DP=${DP})"
echo "Trigger: ${TRIGGER} (window=${FR_TRIG_WINDOW} check=${FR_TRIG_CHECK_FREQ} persist=${FR_TRIG_PERSISTENCE})"
echo "Output: ${OUTPUT_DIR}"
echo "======================================="

docker exec sarju2_straggler_exp bash -c "
set -e
mkdir -p ${OUTPUT_DIR}
echo 'inject_type: ${INJECT_TYPE}
inject_rank: ${INJECT_RANK}
inject_delay_ms: ${INJECT_DELAY_MS}
timestamp: ${TIMESTAMP}
parallelism: TP=2, PP=1, DP=${DP}
gpus: ${GPUS}
trigger: ${TRIGGER}
num_iterations: 30' > ${OUTPUT_DIR}/run_config.log
cd /workspace/nvidia-resiliency-ext

CUDA_VISIBLE_DEVICES=${GPUS} \
PYTHONPATH=/workspace/Megatron-LM:/workspace/nvidia-resiliency-ext/src \
TORCH_NCCL_TRACE_BUFFER_SIZE=500 \
TORCH_NCCL_ENABLE_TIMING=1 \
NCCL_NVLS_ENABLE=0 \
FR_CHEAP_STATS_TRIGGER=${TRIGGER} \
FR_TRIG_WINDOW=${FR_TRIG_WINDOW} \
FR_TRIG_CHECK_FREQ=${FR_TRIG_CHECK_FREQ} \
FR_TRIG_PERSISTENCE=${FR_TRIG_PERSISTENCE} \
FR_TRIG_LOG=${OUTPUT_DIR}/cheap_stats.jsonl \
torchrun --nproc_per_node=${NPROC} \
  experiments/straggler_injection/run_straggler_exp.py \
  --inject-type ${INJECT_TYPE} \
  --inject-rank ${INJECT_RANK} \
  --inject-delay-ms ${INJECT_DELAY_MS} \
  --output-dir ${OUTPUT_DIR}

# Post-hoc analysis: each trigger fire produced N rank-files; group by
# step and run the analyzer per group. Writes verdicts.jsonl alongside
# the dumps.
python3 experiments/straggler_injection/fr_analyze_dumps.py ${OUTPUT_DIR}
"
