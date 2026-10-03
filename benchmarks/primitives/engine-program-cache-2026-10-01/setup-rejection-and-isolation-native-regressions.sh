#!/usr/bin/env bash
set -u
cd /home/bresilla/data/code/OUSD/molla || exit 1
d=benchmarks/primitives/engine-program-cache-2026-10-01
export CARGO_BUILD_JOBS=6
export NVIDIA_VERSION
NVIDIA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
overall=0

run_gate() {
    local gate="$1" budget="$2" rc
    sha256sum -c "$d/setup-rejection-and-isolation-workspace-source-sha256.txt" \
        > "$d/setup-rejection-and-isolation-native-$gate-source-verification.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        printf 'source verification failed before %s\n' "$gate"
        printf '97\n' > "$d/setup-rejection-and-isolation-native-$gate.exit"
        printf '97\n' > "$d/setup-rejection-and-isolation-native-regressions.exit"
        exit 97
    fi
    printf 'gate_start=%s date=%s\n' "$gate" "$(date -Is)"
    timeout --kill-after=10s "${budget}s" nix develop --impure -c make -j1 "$gate" \
        TEST_ALL_ARGS=--no-fail-fast > "$d/setup-rejection-and-isolation-native-$gate.log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$d/setup-rejection-and-isolation-native-$gate.exit"
    printf 'gate_end=%s exit=%s date=%s\n' "$gate" "$rc" "$(date -Is)"
    if [ "$rc" -ne 0 ]; then overall="$rc"; fi
}

same_process() {
    [ -r "/proc/$1/stat" ] || return 1
    [ "$(awk '{print $22}' "/proc/$1/stat" 2>/dev/null)" = "$2" ] || return 1
    [ "$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null)" != Z ]
}

run_gate test-primitives 7200
run_gate check-all 1800
printf 'waiting_for_owned_original_gpu timeout_pid=%s make_pid=%s\n' "$1" "$3"
while same_process "$1" "$2" || same_process "$3" "$4"; do sleep 15; done
printf 'owned_original_gpu_processes_stopped date=%s exit_not_observed=true\n' "$(date -Is)"
run_gate test-all 10800
run_gate harden 14400
printf '%s\n' "$overall" > "$d/setup-rejection-and-isolation-native-regressions.exit"
exit "$overall"
