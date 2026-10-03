#!/usr/bin/env bash
set -u
cd /home/bresilla/data/code/OUSD/molla || exit 1
d=benchmarks/primitives/engine-program-cache-2026-10-01
prefix=lbvh-contention-remaining
manifest="$d/quaternion-boundary-source-sha256.txt"
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
run barrier 3600 test-gpu P=molla-solvers T=barrier_newton_gpu 'A=lbvh_ --nocapture --test-threads=1'
run normal 3600 test-gpu P=molla-solvers T=normal_contact_gpu 'A=--nocapture --test-threads=1'
run drift 3600 test-gpu P=molla-solvers T=featherstone_drift_gpu 'A=--nocapture --test-threads=1'
run contact-tail 21600 test-primitives-friction-gpu test-primitives-coupled-constraints-gpu \
    test-primitives-coupled-response-gpu test-primitives-mixed-contacts-gpu \
    test-primitives-contact-islands-gpu test-primitives-contact-response-gpu \
    test-primitives-contact-publication-gpu test-primitives-runtime-contacts-gpu \
    test-primitives-contact-reactions-gpu test-primitives-contact-breakage-gpu
run later-lanes 21600 test-primitives-gpu \
    'PRIMITIVE_GPU_LANES=test-primitives-impact-episodes-gpu test-primitives-impact-gpu test-primitives-impact-response-gpu test-primitives-impact-position-gpu test-primitives-impact-partition-gpu test-primitives-restitution-gpu test-primitives-dispatch-metrics-gpu test-primitives-multidof-feedback-gpu test-primitives-loop-rows-gpu test-primitives-loop-resources-gpu test-primitives-contact-activation-gpu test-primitives-program-cache-gpu'
exit 0
