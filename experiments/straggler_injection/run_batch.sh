#!/bin/bash
# Batch runner: sweeps all inject types, ranks, and delays.
#
# Usage:
#   ./run_batch.sh              # all types (baseline + host + kernel)
#   ./run_batch.sh host         # only host injection
#   ./run_batch.sh kernel       # only kernel injection

INJECT_TYPES=${1:-"none host kernel"}
RANKS=(0 1 2 3 4 5 6 7)
DELAYS=(10 30 50 80 100)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_BASE="/workspace/nvidia-resiliency-ext/experiments/straggler_injection/traces"
RESULTS_DIR="${SCRIPT_DIR}/batch_results"

mkdir -p "${RESULTS_DIR}"

for TYPE in ${INJECT_TYPES}; do
  if [ "${TYPE}" = "none" ]; then
    # Baseline: single run, no rank/delay sweep needed
    echo "=========================================="
    echo "Running baseline (no injection)"
    echo "=========================================="
    bash "${SCRIPT_DIR}/run.sh" none 0 0

    LATEST_DIR=$(docker exec kevin_straggler_exp bash -c "ls -td ${TRACE_BASE}/*_none_* | head -1")
    OUTPUT_FILE="${RESULTS_DIR}/analysis_none_baseline.txt"
    docker exec kevin_straggler_exp python3 \
      /workspace/nvidia-resiliency-ext/experiments/straggler_injection/fr_straggler_analyzer.py \
      "${LATEST_DIR}" -v > "${OUTPUT_FILE}" 2>&1
    echo "  → ${OUTPUT_FILE}"
  else
    for RANK in "${RANKS[@]}"; do
      for DELAY in "${DELAYS[@]}"; do
        echo "=========================================="
        echo "Running: ${TYPE} rank=${RANK} delay=${DELAY}ms"
        echo "=========================================="

        bash "${SCRIPT_DIR}/run.sh" "${TYPE}" "${RANK}" "${DELAY}"

        LATEST_DIR=$(docker exec kevin_straggler_exp bash -c \
          "ls -td ${TRACE_BASE}/*_${TYPE}_rank${RANK}_${DELAY}ms | head -1")
        OUTPUT_FILE="${RESULTS_DIR}/analysis_${TYPE}_rank${RANK}_${DELAY}ms.txt"
        docker exec kevin_straggler_exp python3 \
          /workspace/nvidia-resiliency-ext/experiments/straggler_injection/fr_straggler_analyzer.py \
          "${LATEST_DIR}" -v > "${OUTPUT_FILE}" 2>&1
        echo "  → ${OUTPUT_FILE}"
      done
    done
  fi
done

echo "=========================================="
echo "All done. Results in ${RESULTS_DIR}/"
ls "${RESULTS_DIR}/"
echo "=========================================="
