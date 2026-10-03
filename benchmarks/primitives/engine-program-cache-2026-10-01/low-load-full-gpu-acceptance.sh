#!/usr/bin/env bash
set -u
cd /home/bresilla/data/code/OUSD/molla || exit 1
d=benchmarks/primitives/engine-program-cache-2026-10-01
prefix=low-load-full-gpu-acceptance
manifest="$d/quaternion-boundary-source-sha256.txt"
printf '%s\n' "$$" > "$d/$prefix-runner.pid"
awk '{print $22}' "/proc/$$/stat" > "$d/$prefix-runner.start_ticks"
export CARGO_BUILD_JOBS=6
export NVIDIA_VERSION
NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
trap 'rc=$?; printf "%s\n" "$rc" > "$d/$prefix.exit.tmp"; mv "$d/$prefix.exit.tmp" "$d/$prefix.exit"' EXIT
sha256sum -c "$manifest" > "$d/$prefix-source-before.log" 2>&1 || exit 97
{
    date -Is
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
    nvidia-smi pmon -c 1
} > "$d/$prefix-environment-before.log" 2>&1
printf 'gate_start=original-full-gpu date=%s\n' "$(date -Is)"
timeout --kill-after=10s 43200s nix develop --impure -c make -j1 test-primitives-gpu \
    > "$d/$prefix.log" 2>&1
rc=$?
printf 'gate_end=original-full-gpu exit=%s date=%s\n' "$rc" "$(date -Is)"
sha256sum -c "$manifest" > "$d/$prefix-source-after.log" 2>&1 || exit 97
{
    date -Is
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
    nvidia-smi pmon -c 1
} > "$d/$prefix-environment-after.log" 2>&1
exit "$rc"
