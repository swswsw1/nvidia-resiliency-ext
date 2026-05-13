#!/bin/bash
# Batch runner: sweeps {inject type, rank, delay} combinations and runs the
# experiment + analyzer for each, so one invocation produces many trace dirs.
#
# Usage:
#   ./run_batch.sh                # all types in INJECT_TYPES
#   ./run_batch.sh host           # only host injection
#   ./run_batch.sh kernel         # only kernel injection
#   ./run_batch.sh "host kernel"  # explicit list

set -euo pipefail

# Sweep config — edit these to change what gets run.
RANKS=(0 1 2 3 4 5 6 7)
DELAYS=(10 30 50 80 100)
INJECT_TYPES="${1:-host kernel}"

CONTAINER="sarju2_straggler_exp"
SWEEP_TAG="sweep_$(date +%Y%m%d_%H%M%S)"

TOTAL=0
for TYPE in ${INJECT_TYPES}; do
  for _ in "${RANKS[@]}"; do for _ in "${DELAYS[@]}"; do TOTAL=$((TOTAL+1)); done; done
done
echo "Sweep tag: ${SWEEP_TAG}"
echo "Total runs: ${TOTAL}"

i=0
for TYPE in ${INJECT_TYPES}; do
  for RANK in "${RANKS[@]}"; do
    for DELAY in "${DELAYS[@]}"; do
      i=$((i+1))
      echo "=========================================="
      echo "[${i}/${TOTAL}] ${TYPE} rank=${RANK} delay=${DELAY}ms"
      echo "=========================================="
      bash run.sh "${TYPE}" "${RANK}" "${DELAY}"
    done
  done
done

echo "=========================================="
echo "Sweep ${SWEEP_TAG} complete."
echo "Each run wrote verdicts.jsonl in its trace dir."
echo "=========================================="
