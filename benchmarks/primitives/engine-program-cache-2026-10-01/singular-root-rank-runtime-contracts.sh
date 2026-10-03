#!/usr/bin/env bash
set -u
cd /home/bresilla/data/code/OUSD/molla || exit 1
d=benchmarks/primitives/engine-program-cache-2026-10-01
export CARGO_BUILD_JOBS=6
export NVIDIA_VERSION
NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
sha256sum -c "$d/singular-root-rank-workspace-source-sha256.txt" \
    > "$d/singular-root-rank-runtime-contracts-source-before.log" 2>&1 || exit 97
printf 'gate_start=runtime-contracts date=%s\n' "$(date -Is)"
timeout --kill-after=10s 7200s nix develop --impure -c make -j1 test-primitives-runtime-gpu \
    > "$d/singular-root-rank-runtime-contracts.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$d/singular-root-rank-runtime-contracts.exit"
sha256sum -c "$d/singular-root-rank-workspace-source-sha256.txt" \
    > "$d/singular-root-rank-runtime-contracts-source-after.log" 2>&1
guard=$?
printf '%s\n' "$guard" > "$d/singular-root-rank-runtime-contracts-source-after.exit"
printf 'gate_end=runtime-contracts exit=%s source_guard=%s date=%s\n' "$rc" "$guard" "$(date -Is)"
if [ "$guard" -ne 0 ]; then exit 97; fi
exit "$rc"
