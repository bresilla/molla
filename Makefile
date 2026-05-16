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

.PHONY: build b compile c run r run-profile test t test-examples test-differentiable test-gpu test-featherstone-gpu test-featherstone-multidof-gpu test-featherstone-polish-gpu test-articulation-gpu test-autograd-featherstone-gpu test-autograd-featherstone-v3-gpu test-optim-gpu test-robust-loss-gpu test-splat-gpu test-splat-pose-gpu test-tile-primitives-gpu test-xpbd-gpu test-gjk-gpu test-mpr-gpu test-narrow-phase-gpu test-broadphase-lbvh-gpu test-contact-reduction-gpu test-raycast-gpu test-imu-gpu test-contact-gpu test-contact-friction-gpu test-frame-transform-gpu test-tiled-camera-gpu test-actuator-gpu test-actuator-pipeline-gpu test-actuator-neural-lstm-gpu test-equality-gpu test-equality-joint-mimic-gpu test-particles-gpu test-joint-muscle-gpu test-hydroelastic-gpu test-sparse-grid-gpu test-sparse-grid-v3-gpu test-sparse-grid-v4-gpu test-mpm-gpu test-mpm-v3-gpu test-mpm-two-way-gpu test-implicit-mpm-gpu test-vbd-cloth-gpu test-vbd-tets-gpu test-vbd-tets-v3-gpu test-vbd-tets-v4-gpu test-vbd-extended-gpu test-soft-vt-gpu test-soft-friction-gpu test-fem-gpu test-fem-elasticity-gpu test-fem-stokes-gpu test-fem-dg-swe-gpu test-fem-ns-gpu test-fem-ns-lazy-cg-gpu test-fem-fft-ns-gpu test-fem-resident-cg-gpu test-fem-resident-pcg-gpu test-fem-wave-gpu test-fem-apic-gpu test-fem-transient-gpu test-fft-gpu test-bsr-gpu test-spgemm-symbolic-gpu bench-cpu bench-gpu bench-featherstone check fmt clean help h

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

# CPU/headless example hardening. This checks the published Rust examples and
# the viewer's scene registry/build-only smoke tests without opening a window.
test-examples:
	@$(CARGO) run -p molla --example hello_world
	@$(CARGO) run -p molla --example diff_arm_reach
	@$(CARGO) test -p molla_viewer

test-differentiable:
	@$(CARGO) test -p molla-autograd
	@$(CARGO) test -p molla-fem --test fem_adjoint
	@$(CARGO) test -p molla-fem --test fem_diff_optim
	@$(CARGO) test -p molla-fem --test fem_shape_optim
	@$(CARGO) test -p molla-fem --test fem_darcy_optim
	@$(CARGO) test -p molla-sensors --test raycast_adjoint

# GPU tests are gated behind `--ignored` because the wgpu adapter is
# only reachable through nixVulkan. `make test-gpu` runs the canonical
# CPU/GPU parity lane.
test-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test cpu_gpu_parity -- --ignored

test-featherstone-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test featherstone_gpu_parity -- --ignored

test-featherstone-multidof-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test featherstone_multidof_gpu -- --ignored

test-featherstone-polish-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test featherstone_polish_gpu -- --ignored --test-threads=1 --nocapture

test-articulation-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test articulation_gpu -- --ignored --test-threads=1 --nocapture

# Stage 23.1.1 — GPU adjoint dispatcher parity + closed-form perf check.
# The dispatcher currently routes through CPU closed-form; will become a
# real CPU-vs-GPU test when Stage 23.1.2 populates the WGSL kernel
# bodies. Run in `--release` so the perf check (closed-form vs FD
# speedup) reflects optimised code.
test-autograd-featherstone-gpu:
	@$(RUN_ENV) $(CARGO) test --release -p molla-kernels --features gpu --test autograd_featherstone_gpu -- --ignored

# Stage 23.1.2 — pure-analytic CPU adjoint, GPU multi-articulation
# closed-form Jacobian kernel parity, and rayon-parallel batch perf.
# All tests are `#[ignore]`-marked; run with --release for realistic
# perf numbers (the AnalyticQd vs ClosedForm bwd/fwd ratios + the
# parallel speedup measurements are sensitive to optimisation level).
test-autograd-featherstone-v3-gpu:
	@$(RUN_ENV) $(CARGO) test --release -p molla-kernels --features gpu --test autograd_featherstone_v3_gpu -- --ignored --nocapture

test-optim-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test optim_gpu_parity -- --ignored --nocapture

test-robust-loss-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test robust_loss_gpu -- --ignored --nocapture

test-splat-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test splat_raytrace_forward -- --ignored --nocapture
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test splat_raytrace_backward -- --ignored --nocapture
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test splat_sh_forward -- --ignored --nocapture
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test splat_sh_backward -- --ignored --nocapture
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test splat_sh_band3 -- --ignored --nocapture

test-splat-pose-gpu:
	@$(RUN_ENV) $(CARGO) test --release -p molla-kernels --features gpu --test splat_pose_recovery -- --ignored --nocapture

test-tile-primitives-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-compute-primitives --features gpu --test tile_primitives_gpu -- --ignored --nocapture
	@$(RUN_ENV) $(CARGO) test -p molla-compute-primitives --features gpu --test radix_sort_gpu -- --ignored --nocapture
	@$(RUN_ENV) $(CARGO) test -p molla-compute-primitives --features gpu --test radix_sort_64_gpu -- --ignored --nocapture

test-xpbd-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test xpbd_gpu_parity -- --ignored

test-gjk-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-geometry --features gpu --test gjk_gpu -- --ignored --nocapture

test-mpr-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-geometry --features gpu --test mpr_gpu -- --ignored --nocapture

test-narrow-phase-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-geometry --features gpu --test gpu_narrow_phase -- --ignored --nocapture

test-broadphase-lbvh-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-geometry --features gpu --test broadphase_lbvh_gpu -- --ignored --test-threads=1 --nocapture

test-contact-reduction-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test contact_reduction_gpu -- --ignored --nocapture

test-raycast-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-sensors --features gpu --test sensor_raycast_gpu -- --ignored

test-imu-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-sensors --features gpu --test sensor_imu_gpu -- --ignored

test-contact-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-sensors --features gpu --test sensor_contact_gpu -- --ignored

test-contact-friction-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test contact_friction_modes_gpu -- --ignored --test-threads=1

test-frame-transform-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-sensors --features gpu --test sensor_frame_transform_gpu -- --ignored

test-tiled-camera-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-sensors --features gpu --test sensor_tiled_camera_gpu -- --ignored

test-actuator-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-actuators --features gpu --test actuator_pd_gpu -- --ignored

test-actuator-pipeline-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-actuators --features gpu --test actuator_pipeline_gpu -- --ignored --test-threads=1

test-actuator-neural-lstm-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-actuators --features gpu --test actuator_neural_lstm_gpu -- --ignored --test-threads=1

test-equality-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test xpbd_equality_gpu -- --ignored --test-threads=1

# Stage 32 — joint-mimic equality constraint (`EqualityType::JointMimic`).
# CPU portion runs in the default `cargo test` lane; this lane runs the
# GPU parity tests against the WGSL `project_joint_mimic` kernel.
test-equality-joint-mimic-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test equality_joint_mimic -- --ignored --test-threads=1

test-particles-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test xpbd_particles_gpu -- --ignored --test-threads=1

test-joint-muscle-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test joint_muscle_gpu -- --ignored --test-threads=1

# Stage 22.1 — hydroelastic contact (CPU + GPU; the CPU-only tests
# also live behind `#[ignore]` to keep the default `cargo test` lane
# fast for plain CPU unit tests).
test-hydroelastic-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test hydroelastic_gpu -- --ignored

# Stage 24e — sparse-grid (NanoVDB-equivalent) tests. CPU portion
# runs in the default lane via `cargo test -p molla --test sparse_grid`;
# the GPU lookup-parity + latency tests are `--ignored` because they
# need a real wgpu adapter.
test-sparse-grid-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test sparse_grid -- --ignored

test-sparse-grid-v3-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test sparse_grid_v3 -- --ignored --test-threads=1 --nocapture

test-sparse-grid-v4-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test sparse_grid_v4 -- --ignored --test-threads=1 --nocapture

# Stage 24d.1 — semi-implicit MPM + Snow rheology + two-way rigid
# coupling + calibrated angle-of-repose tests (see
# tests/mpm_v3_gpu.rs). The Stage 24d v1 explicit-only tests still
# live in tests/mpm_gpu.rs; both are GPU-only and `#[ignore]`.
test-mpm-v3-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test mpm_v3_gpu -- --ignored
test-mpm-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test mpm_gpu -- --ignored

test-mpm-two-way-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test mpm_v3_gpu mpm_two_way_rigid_coupling_wrench -- --ignored --nocapture

test-implicit-mpm-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test implicit_mpm_gpu -- --ignored --nocapture

test-vbd-cloth-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test vbd_cloth_gpu -- --ignored --nocapture

test-vbd-tets-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test vbd_tets_gpu -- --ignored --nocapture

test-vbd-tets-v3-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test vbd_tets_v3_gpu -- --ignored --nocapture

test-vbd-tets-v4-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test vbd_tets_v4_gpu -- --ignored --nocapture

test-vbd-extended-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test vbd_extended_gpu -- --ignored --nocapture

# Stage 33.2 — GPU port of the soft-on-soft VT contact pipeline (LBVH
# refit + gather_vt + apply_vt_penalty + commit_dq, all on the device).
# Tests mirror the Stage 33 CPU integration tests in
# tests/soft_vt_contact.rs and assert: (a) the GPU/CPU paths agree
# while in free fall, and (b) the cube rests on the cloth instead of
# free-falling through it. All `--ignored` (real wgpu adapter needed).
test-soft-vt-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test soft_vt_contact_gpu -- --ignored

# Stage 33.2.1 — GPU Coulomb friction + restitution port. Mirrors the
# CPU `soft_friction.rs` integration tests on the GPU VBD backend +
# adds a SoftBrake-style stick test. All `--ignored` (real wgpu adapter
# needed).
test-soft-friction-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-solvers --features gpu --test soft_friction_gpu -- --ignored

# Stage 36.5 — GPU port of the FEM v1 framework (P1 nodal scalar).
# Three parity tests (assembly, matvec, full steady-state CG) + one
# 16³ smoke run that emits per-iter readback timing for the v1
# "fully on-device CG = Stage 36.5.1 follow-up" caveat. All
# `--ignored` (real wgpu adapter needed).
test-fem-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-fem --features gpu --test fem_gpu -- --ignored

# Stage 41.1 — GPU linear elasticity assembly + resident-CG end-to-end.
# Runs the 4³ assembly parity, the 8×4×4 cantilever solve parity, and the
# 16³ perf telemetry (peak |u| ~ 6e-2, GPU vs CPU wall-clock). All
# `--ignored` (real wgpu adapter needed).
test-fem-elasticity-gpu:
	@$(RUN_ENV) $(CARGO) test --release -p molla-fem --features gpu --test fem_elasticity_gpu -- --ignored --nocapture

# Stage 41 v2 — GPU Taylor-Hood Stokes parity vs CPU on the 8×8 lid-
# driven cavity. `--ignored` (real wgpu adapter needed).
test-fem-stokes-gpu:
	@$(RUN_ENV) $(CARGO) test --release -p molla-fem --features gpu --test fem_stokes_gpu -- --ignored --nocapture

test-fem-dg-swe-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-fem --features gpu --test fem_dg_swe_gpu -- --ignored --nocapture

test-fem-ns-gpu:
	@$(RUN_ENV) $(CARGO) test --release -p molla-fem --features gpu --test fem_ns_gpu -- --ignored --skip gpu_ns_perf_16x16_re100_100steps --nocapture

test-fem-ns-lazy-cg-gpu:
	@$(RUN_ENV) $(CARGO) test --release -p molla-fem --features gpu --test fem_ns_gpu_lazy_cg -- --ignored --nocapture

# Tier 5.1 — FFT pressure-Poisson + fractional-step Navier-Stokes
# coupling. These tests use the GPU FFT kernel as the pressure solve
# backbone and compare against analytic pressure / Taylor-Green decay.
test-fem-fft-ns-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-fem --features gpu --test fft_ns_coupling -- --ignored --nocapture

# Tier 5 — GPU-resident FEM linear-solve kernels. Long scaling sweeps stay
# explicit/manual; this target runs the correctness/parity gates.
test-fem-resident-cg-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-fem --features gpu --test fem_gpu_resident_cg -- --ignored --skip fem_gpu_resident_cg_scaling_sweep --nocapture

test-fem-resident-pcg-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-fem --features gpu --test fem_gpu_resident_pcg -- --ignored --skip pcg_vs_cg_wall_time_32_cube --skip fem_gpu_resident_cg_vs_pcg_scaling_sweep --nocapture

test-fem-wave-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-fem --features gpu --test fem_wave_gpu -- --ignored --nocapture

test-fem-apic-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-fem --features gpu --test fem_apic_gpu -- --ignored --nocapture

test-fem-transient-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-fem --features gpu --test fem_transient_gpu -- --ignored --nocapture

# Tier 5.1 — standalone GPU FFT parity against rustfft for the spectral
# kernels used by pressure projection and stable-fluid style solvers.
test-fft-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test fft_gpu -- --ignored --nocapture

test-bsr-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test bsr_gpu_parity -- --ignored --test-threads=1 --nocapture

test-spgemm-symbolic-gpu:
	@$(RUN_ENV) $(CARGO) test -p molla-kernels --features gpu --test spgemm_symbolic_gpu -- --ignored --test-threads=1 --nocapture

bench-cpu:
	@$(CARGO) bench -p molla --bench step_cpu

bench-gpu:
	@$(RUN_ENV) $(CARGO) bench -p molla --bench step_gpu --features gpu

bench-featherstone:
	@$(RUN_ENV) $(CARGO) bench -p molla-solvers --bench featherstone_cpu_vs_gpu --features gpu

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
	@echo "  test-contact-friction-gpu"
	@echo "                Run GPU rolling/torsional contact-friction tests"
	@echo "  bench-cpu    cargo bench the CPU integrator"
	@echo "  bench-gpu    cargo bench the GPU integrator via nixVulkan"
	@echo "  check        cargo check the workspace"
	@echo "  fmt          Format the workspace"
	@echo "  clean        Remove Cargo build artifacts"
	@echo

h: help
