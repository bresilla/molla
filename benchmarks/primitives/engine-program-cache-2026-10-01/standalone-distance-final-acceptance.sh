#!/usr/bin/env bash
set -u
cd /home/bresilla/data/code/OUSD/molla || exit 1
d=benchmarks/primitives/engine-program-cache-2026-10-01
prefix=standalone-distance-final
manifest="$d/$prefix-source-sha256.txt"
printf '%s\n' "$$" > "$d/$prefix-acceptance-runner.pid"
export CARGO_BUILD_JOBS=6
export NVIDIA_VERSION
NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
trap 'rc=$?; printf "%s\n" "$rc" > "$d/$prefix-acceptance.exit"' EXIT
run() {
    local name="$1" budget="$2" rc
    shift 2
    sha256sum -c "$manifest" > "$d/$prefix-$name-source-before.log" 2>&1 || exit 97
    printf 'gate_start=%s date=%s\n' "$name" "$(date -Is)"
    timeout --kill-after=10s "${budget}s" nix develop --impure -c make -j1 "$@" \
        > "$d/$prefix-$name.log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$d/$prefix-$name.exit"
    printf 'gate_end=%s exit=%s date=%s\n' "$name" "$rc" "$(date -Is)"
    sha256sum -c "$manifest" > "$d/$prefix-$name-source-after.log" 2>&1 || exit 97
    if [ "$rc" -ne 0 ]; then exit "$rc"; fi
}
run mixed-distance-gpu 1800 test-gpu P=molla-solvers T=featherstone_mixed_distance \
    GPU_FEATURES=gpu A='--nocapture --test-threads=1'
run cpu-primitives 7200 test-primitives
run check-all 1800 check-all
run clippy-full 3600 clippy-check
run test-all 10800 test-all TEST_ALL_ARGS=--no-fail-fast
run harden 14400 harden TEST_ALL_ARGS=--no-fail-fast
run examples-gpu 3600 example-primitives-gpu A=all
run gpu-primitives 43200 test-primitives-gpu
exit 0
