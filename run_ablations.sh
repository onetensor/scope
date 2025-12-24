#!/usr/bin/env bash
set -euo pipefail

# 3-run ablation set:
# A) baseline (spectral disabled)
# B) bias-only (spectral enabled, pointer mask OFF)
# C) bias + pointer-mask (spectral enabled, pointer mask ON + schedule)

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_ITERATIONS="${NUM_ITERATIONS:-2000}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"  # 0 -> only at end
VAL_TOKENS="${VAL_TOKENS:-10485760}"
SCOPE_LOG_EVERY="${SCOPE_LOG_EVERY:-200}"
SEED="${SEED:-1337}"
SPECTRAL_POINTER_VAL_FORCE="${SPECTRAL_POINTER_VAL_FORCE:-0}"
SPECTRAL_POINTER_VAL_HALF_BLOCKS="${SPECTRAL_POINTER_VAL_HALF_BLOCKS:--1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_LOG_DIR="${LOG_DIR:-logs/ablations/${STAMP}}"
mkdir -p "${BASE_LOG_DIR}"

run_one() {
  local name="$1"
  local spectral_bias="$2"
  local use_pointer_mask="$3"
  local pointer_schedule="$4"
  shift 4
  echo "=== ${name} ==="
  env \
    RUN_NAME="${name}" \
    LOG_DIR="${BASE_LOG_DIR}" \
    SEED="${SEED}" \
    NUM_ITERATIONS="${NUM_ITERATIONS}" \
    VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
    VAL_TOKENS="${VAL_TOKENS}" \
    SCOPE_LOG_EVERY="${SCOPE_LOG_EVERY}" \
    SPECTRAL_BIAS="${spectral_bias}" \
    SPECTRAL_USE_POINTER_MASK="${use_pointer_mask}" \
    SPECTRAL_POINTER_SCHEDULE="${pointer_schedule}" \
    SPECTRAL_POINTER_VAL_FORCE="${SPECTRAL_POINTER_VAL_FORCE}" \
    SPECTRAL_POINTER_VAL_HALF_BLOCKS="${SPECTRAL_POINTER_VAL_HALF_BLOCKS}" \
    "$@"
}

torchrun_cmd=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" scoped_medium.py)

# A) Baseline
run_one "A_baseline" 0 0 0 "${torchrun_cmd[@]}"

# B) Bias-only
run_one "B_bias_only" 1 0 0 "${torchrun_cmd[@]}"

# C) Bias + pointer mask (scheduled)
run_one "C_bias_pointer" 1 1 1 "${torchrun_cmd[@]}"

echo "Logs written to: ${BASE_LOG_DIR}"
