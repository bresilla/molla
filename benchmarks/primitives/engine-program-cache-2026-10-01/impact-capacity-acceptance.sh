#!/usr/bin/env bash
set -u
cd /home/bresilla/data/code/OUSD/molla || exit 1
d=benchmarks/primitives/engine-program-cache-2026-10-01
prefix=impact-capacity-acceptance
manifest="$d/impact-capacity-source-sha256.txt"
printf '%s\n' "$$" > "$d/$prefix-runner.pid"
awk '{print $22}' "/proc/$$/stat" > "$d/$prefix-runner.start_ticks"
export CARGO_BUILD_JOBS=6
export NVIDIA_VERSION
NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
trap 'rc=$?; printf "%s\n" "$rc" > "$d/$prefix.exit.tmp"; mv "$d/$prefix.exit.tmp" "$d/$prefix.exit"' EXIT
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
run static-shaders 3600 fmt-check check-all clippy-check check-primitives-shader-wiring
run remaining-production 43200 test-primitives-gpu \
    'PRIMITIVE_GPU_LANES=test-primitives-impact-position-gpu test-primitives-impact-partition-gpu test-primitives-restitution-gpu test-primitives-dispatch-metrics-gpu test-primitives-multidof-feedback-gpu test-primitives-loop-rows-gpu test-primitives-loop-resources-gpu test-primitives-contact-activation-gpu test-primitives-program-cache-gpu'
run examples 3600 example-primitives A=all example-primitives-gpu
run cpu-primitives 3600 test-primitives
run test-all 10800 test-all TEST_ALL_ARGS=--no-fail-fast
run harden 14400 harden TEST_ALL_ARGS=--no-fail-fast
run original-full-gpu 43200 test-primitives-gpu
exit 0
