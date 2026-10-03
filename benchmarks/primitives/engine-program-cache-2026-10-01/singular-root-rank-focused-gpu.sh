#!/usr/bin/env bash
set -u
cd /home/bresilla/data/code/OUSD/molla || exit 1
d=benchmarks/primitives/engine-program-cache-2026-10-01
export CARGO_BUILD_JOBS=6
export NVIDIA_VERSION
NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
timeout --kill-after=10s 1800s nix develop --impure -c make -j1 test-gpu \
    P=molla-solvers T=rigid_singular_forces GPU_FEATURES=gpu \
    A='gpu_locked_singular_d6_roots_keep_supported_spin_under_null_axis_loads --exact --nocapture --test-threads=1' \
    > "$d/singular-root-rank-focused-gpu.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$d/singular-root-rank-focused-gpu.exit"
exit "$rc"
