#!/usr/bin/env bash
set -u
cd /home/bresilla/data/code/OUSD/molla || exit 1
d=benchmarks/primitives/engine-program-cache-2026-10-01
export CARGO_BUILD_JOBS=6
export NVIDIA_VERSION
NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
overall=0
run() {
    local name="$1" rc
    shift
    timeout --kill-after=10s 1800s nix develop --impure -c make -j1 "$@" \
        > "$d/singular-root-rank-$name.log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$d/singular-root-rank-$name.exit"
    printf 'gate=%s exit=%s date=%s\n' "$name" "$rc" "$(date -Is)"
    if [ "$rc" -ne 0 ]; then overall="$rc"; fi
}
run full-cpu test APP_TARGET='-p molla-solvers --test rigid_singular_forces -- --nocapture'
run full-gpu test-gpu P=molla-solvers T=rigid_singular_forces GPU_FEATURES=gpu \
    A='--nocapture --test-threads=1'
run weak-inverse-gpu test-gpu P=molla-kernels T=featherstone_mass_inverse_gpu GPU_FEATURES=gpu \
    A='--nocapture --test-threads=1'
printf '%s\n' "$overall" > "$d/singular-root-rank-regressions.exit"
exit "$overall"
