#!/usr/bin/env bash
set -eu
output=$(realpath -m -- "${1:?telemetry CSV path}")
case "$output" in */benchmarks/soil/*.csv) ;; *) echo "Not a soil telemetry CSV: $output" >&2; exit 2 ;; esac
for descriptor in /proc/[0-9]*/fd/1; do
    test "$(readlink "$descriptor" 2>/dev/null || true)" = "$output" || continue
    pid=${descriptor#/proc/}
    pid=${pid%%/*}
    test "$(cat "/proc/$pid/comm" 2>/dev/null || true)" = nvidia-smi || continue
    kill "$pid" 2>/dev/null || true
done
