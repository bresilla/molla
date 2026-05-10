SHELL := /bin/bash

PROJECT_NAME := $(shell sed -n '/^[[:space:]]*[^#\[[:space:]]/p' PROJECT | head -1 | tr -d '[:space:]')
PROJECT_VERSION := $(shell sed -n '/^[[:space:]]*[^#\[[:space:]]/p' PROJECT | sed -n '2p' | tr -d '[:space:]')
ifeq ($(PROJECT_NAME),)
    $(error Error: PROJECT file not found or invalid)
endif

CARGO := nixVulkan cargo
# DISPLAY=:1 is required so winit can find the host X server when
# cargo is invoked from contexts (nix shell, systemd user unit, etc.)
# that don't inherit the desktop session's DISPLAY env var. Only
# matters for targets that actually open a window (`run`).
# WAYLAND_DISPLAY= forces winit onto the X11 path: if both env vars
# are set, winit prefers Wayland and we want it pinned to X11.
RUN_ENV := DISPLAY=:1 WAYLAND_DISPLAY=

$(info ------------------------------------------)
$(info Project: $(PROJECT_NAME) v$(PROJECT_VERSION))
$(info ------------------------------------------)

.PHONY: build b compile c run r run-profile test t test-gpu test-featherstone-gpu test-featherstone-multidof-gpu test-autograd-featherstone-gpu test-xpbd-gpu test-raycast-gpu test-imu-gpu test-contact-gpu test-frame-transform-gpu test-tiled-camera-gpu test-actuator-gpu test-actuator-pipeline-gpu test-equality-gpu test-equality-joint-mimic-gpu test-particles-gpu test-hydroelastic-gpu test-mujoco-gpu test-sparse-grid-gpu test-mpm-gpu test-mpm-v3-gpu test-soft-vt-gpu test-soft-friction-gpu test-fem-gpu test-oracle bench-cpu bench-gpu bench-featherstone check fmt clean help h

build:
	@$(CARGO) build --workspace --all-targets

b: build

compile:
	@$(CARGO) clean
	@$(MAKE) build

c: compile

# Launches `molla_viewer` (the Bevy app). `BIN` overrides the bin name.
# `RELEASE=1` builds with optimisations (recommended — debug builds are
# noticeably laggy under Bevy).
RELEASE_FLAG := $(if $(RELEASE),--release,)
BIN := molla_viewer
run:
	@$(RUN_ENV) $(CARGO) run $(RELEASE_FLAG) -p $(BIN) -- $(ARGS)

# Run with Tracy instrumentation. Build with `make run-profile RELEASE=1`,
# then in another terminal launch the Tracy GUI (`nix-shell -p tracy &&
# tracy` or download from https://github.com/wolfpld/tracy/releases) and
# click "Connect". Every Bevy schedule + system + render pass appears
# on the timeline.
run-profile:
	@$(RUN_ENV) $(CARGO) run $(RELEASE_FLAG) -p $(BIN) --features profile -- $(ARGS)

r: run

test:
	@$(CARGO) test --workspace

t: test

# GPU tests are gated behind `--ignored` because the wgpu adapter is
# only reachable through nixVulkan. `make test-gpu` runs only the
# GPU-specific parity test (skips the Newton-oracle tests, which need a
# venv'd Python — those run via `make test-oracle`).
test-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test cpu_gpu_parity -- --ignored

test-featherstone-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test featherstone_gpu_parity -- --ignored

test-featherstone-multidof-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test featherstone_multidof_gpu -- --ignored

# Stage 23.1.1 — GPU adjoint dispatcher parity + closed-form perf check.
# The dispatcher currently routes through CPU closed-form; will become a
# real CPU-vs-GPU test when Stage 23.1.2 populates the WGSL kernel
# bodies. Run in `--release` so the perf check (closed-form vs FD
# speedup) reflects optimised code.
test-autograd-featherstone-gpu:
	@$(RUN_ENV) $(CARGO) test --release -p molla --features gpu --test autograd_featherstone_gpu -- --ignored

# Stage 23.1.2 — pure-analytic CPU adjoint, GPU multi-articulation
# closed-form Jacobian kernel parity, and rayon-parallel batch perf.
# All tests are `#[ignore]`-marked; run with --release for realistic
# perf numbers (the AnalyticQd vs ClosedForm bwd/fwd ratios + the
# parallel speedup measurements are sensitive to optimisation level).
test-autograd-featherstone-v3-gpu:
	@$(RUN_ENV) $(CARGO) test --release -p molla --features gpu --test autograd_featherstone_v3_gpu -- --ignored --nocapture

test-xpbd-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test xpbd_gpu_parity -- --ignored

test-raycast-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test sensor_raycast_gpu -- --ignored

test-imu-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test sensor_imu_gpu -- --ignored

test-contact-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test sensor_contact_gpu -- --ignored

test-frame-transform-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test sensor_frame_transform_gpu -- --ignored

test-tiled-camera-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test sensor_tiled_camera_gpu -- --ignored

test-actuator-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test actuator_pd_gpu -- --ignored

test-actuator-pipeline-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test actuator_pipeline_gpu -- --ignored --test-threads=1

test-equality-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test xpbd_equality_gpu -- --ignored --test-threads=1

# Stage 32 — joint-mimic equality constraint (`EqualityType::JointMimic`).
# CPU portion runs in the default `cargo test` lane; this lane runs the
# GPU parity tests against the WGSL `project_joint_mimic` kernel.
test-equality-joint-mimic-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test equality_joint_mimic -- --ignored --test-threads=1

test-particles-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test xpbd_particles_gpu -- --ignored --test-threads=1

# Stage 22.1 — hydroelastic contact (CPU + GPU; the CPU-only tests
# also live behind `#[ignore]` to keep the default `cargo test` lane
# fast for plain CPU unit tests).
test-hydroelastic-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test hydroelastic_gpu -- --ignored

# Stage 22 / 22.1 — MuJoCo wrapper parity tests. Requires libmujoco
# system-installed (the `mujoco-rust` crate searches `~/.local/mujoco`
# and the standard system paths). Without libmujoco the build fails
# at link time with a clear error from `mujoco-rust`.
#
# **Heads-up:** the molla dev `nixVulkan` shell does NOT carry
# libmujoco — bindgen needs `libstdc++.so.6` which Nix doesn't drop
# into the wgpu shell. A future contributor on a libmujoco-equipped
# CI runner (the `mj-build` Docker image, the upstream Drake
# packaging recipe, or a fresh Ubuntu 22.04 machine with the
# mujoco .deb installed) can run this lane unmodified. See PLAN.md
# Stage 22.1 for the CI gap rationale.
test-mujoco-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features mujoco --test mujoco_solver -- --ignored

# Stage 24e — sparse-grid (NanoVDB-equivalent) tests. CPU portion
# runs in the default lane via `cargo test -p molla --test sparse_grid`;
# the GPU lookup-parity + latency tests are `--ignored` because they
# need a real wgpu adapter.
test-sparse-grid-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test sparse_grid -- --ignored

# Stage 24d.1 — semi-implicit MPM + Snow rheology + two-way rigid
# coupling + calibrated angle-of-repose tests (see
# tests/mpm_v3_gpu.rs). The Stage 24d v1 explicit-only tests still
# live in tests/mpm_gpu.rs; both are GPU-only and `#[ignore]`.
test-mpm-v3-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test mpm_v3_gpu -- --ignored
test-mpm-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test mpm_gpu -- --ignored

# Stage 33.2 — GPU port of the soft-on-soft VT contact pipeline (LBVH
# refit + gather_vt + apply_vt_penalty + commit_dq, all on the device).
# Tests mirror the Stage 33 CPU integration tests in
# tests/soft_vt_contact.rs and assert: (a) the GPU/CPU paths agree
# while in free fall, and (b) the cube rests on the cloth instead of
# free-falling through it. All `--ignored` (real wgpu adapter needed).
test-soft-vt-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test soft_vt_contact_gpu -- --ignored

# Stage 33.2.1 — GPU Coulomb friction + restitution port. Mirrors the
# CPU `soft_friction.rs` integration tests on the GPU VBD backend +
# adds a SoftBrake-style stick test. All `--ignored` (real wgpu adapter
# needed).
test-soft-friction-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test soft_friction_gpu -- --ignored

# Stage 36.5 — GPU port of the FEM v1 framework (P1 nodal scalar).
# Three parity tests (assembly, matvec, full steady-state CG) + one
# 16³ smoke run that emits per-iter readback timing for the v1
# "fully on-device CG = Stage 36.5.1 follow-up" caveat. All
# `--ignored` (real wgpu adapter needed).
test-fem-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla --features gpu --test fem_gpu -- --ignored

test-oracle:
	@$(RUN_ENV) $(CARGO) test -p molla --test oracle -- --ignored

bench-cpu:
	@$(CARGO) bench -p molla --bench step_cpu

bench-gpu:
	@$(RUN_ENV) $(CARGO) bench -p molla --bench step_gpu --features gpu

bench-featherstone:
	@$(RUN_ENV) $(CARGO) bench -p molla --bench featherstone_cpu_vs_gpu --features gpu

check:
	@$(CARGO) check --workspace --all-targets

fmt:
	@$(CARGO) fmt --all

clean:
	@$(CARGO) clean

help:
	@echo
	@echo "Usage: make [target]"
	@echo
	@echo "Available targets:"
	@echo "  build        Build the workspace (all targets)"
	@echo "  compile      Clean and rebuild"
	@echo "  run          Launch molla_viewer (RELEASE=1 recommended; BIN=name overrides)"
	@echo "  run-profile  Launch with Tracy instrumentation; connect Tracy GUI separately"
	@echo "  test         Run the test suite"
	@echo "  test-gpu     Run GPU-only tests via nixVulkan (CPU/GPU parity, etc.)"
	@echo "  bench-cpu    cargo bench the CPU integrator"
	@echo "  bench-gpu    cargo bench the GPU integrator via nixVulkan"
	@echo "  check        cargo check the workspace"
	@echo "  fmt          Format the workspace"
	@echo "  clean        Remove Cargo build artifacts"
	@echo

h: help
