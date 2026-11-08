#!/usr/bin/env bash
# Execute frozen V10 with the unchanged official A3 simulator.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SIM_REPO:?Set SIM_REPO to the cloned official Eter0109/a3_dual_arm_sim checkout}"
SIM_REPO="$(cd "$SIM_REPO" && pwd)"
EXPECTED_COMMIT=0729068ee1e75a562cfbc4f2c83ab9539cf75081
ACTUAL_COMMIT="$(git -C "$SIM_REPO" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "Official simulator commit mismatch: expected $EXPECTED_COMMIT; found $ACTUAL_COMMIT" >&2
  exit 2
fi
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$SIM_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
export LP_NUM_THREADS="${LP_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/v10_$(date +%Y%m%d_%H%M%S)}"
"$PYTHON" "$ROOT/evaluation/evaluate_policy.py" \
  --policy policy:make_policy_b --policy-code "$ROOT/frozen_policy_v10" \
  --output "$OUTPUT_DIR" --tasks 2x5 2x10 \
  --profiles basic medium advanced --episodes 20 --seed-start 24000 \
  --max-steps 500 --max-steps-2x10 3200 --source-column 1 \
  --success-hold-steps 5 --render-cameras "$@"
