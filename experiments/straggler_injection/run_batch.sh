#!/bin/bash
# Batch runner: loops through ranks and delays, runs experiment + analyzer.
#
# Usage:
#   ./run_batch.sh              # run all combinations
#   ./run_batch.sh kernel       # only kernel injection
#   ./run_batch.sh host         # only host injection


# Config
RANKS=(3)
DELAYS=(50 )
INJECT_TYPES="host kernel"

SCRIPT_DIR="/workspace/nvidia-resiliency-ext/experiments/straggler_injection"
TRACE_BASE="${SCRIPT_DIR}/traces"
RESULTS_DIR="./batch_results"

mkdir -p ${RESULTS_DIR}

for TYPE in ${INJECT_TYPES}; do
  for RANK in "${RANKS[@]}"; do
    for DELAY in "${DELAYS[@]}"; do
      echo "=========================================="
      echo "Running: ${TYPE} rank=${RANK} delay=${DELAY}ms"
      echo "=========================================="

      # Run experiment
      bash run.sh ${TYPE} ${RANK} ${DELAY}

      # Find the latest trace directory
      LATEST_DIR=$(docker exec wei_straggler_exp bash -c "ls -td ${TRACE_BASE}/*_${TYPE} | head -1")

      # Run analyzer and save results
      OUTPUT_FILE="${RESULTS_DIR}/analysis_${TYPE}_rank${RANK}_${DELAY}ms.txt"
      docker exec wei_straggler_exp python3 ${SCRIPT_DIR}/fr_straggler_analyzer.py ${LATEST_DIR} -v 

    done
  done
done

echo "=========================================="
echo "All done! Results in ${RESULTS_DIR}/"
echo "=========================================="
