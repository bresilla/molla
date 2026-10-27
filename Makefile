SHELL := /bin/bash

ifneq ($(wildcard PROJECT),)
PROJECT_NAME := $(shell sed -n '/^[[:space:]]*[^#\[[:space:]]/p' PROJECT | head -1 | tr -d '[:space:]')
PROJECT_VERSION := $(shell sed -n '/^[[:space:]]*[^#\[[:space:]]/p' PROJECT | sed -n '2p' | tr -d '[:space:]')
else
PROJECT_NAME := molla
PROJECT_VERSION := $(shell sed -n 's/^version = "\([^"]*\)"/\1/p' crates/molla/Cargo.toml | head -1)
endif
ifeq ($(PROJECT_NAME),)
    $(error Error: PROJECT file not found or invalid)
endif

TOP_DIR := $(CURDIR)
DOCS_DIR ?= $(TOP_DIR)/docs
CARGO := cargo
BACKEND ?= x11
DISPLAY ?= :1
APP_PKG ?= molla_viewer
APP_TARGET := -p $(APP_PKG)
RUN_WITH ?= nixVulkan
BENCH_ARGS ?=
TEST_ALL_ARGS ?=
PRIMITIVE_BENCH_PROFILE ?= release
SENSOR_BENCH ?= bench_scene_queries
SENSOR_BENCH_PKG ?= molla-sensors
LATENCY_BENCH ?= bench_sensor_latency
NVIDIA_SMI ?= nvidia-smi
FMT_ARGS ?=
CLIPPY_TARGET ?= --workspace --all-targets --all-features
PRIMITIVE_CPU_BASELINES := joint_control runtime_motor_models runtime_motor_coupling runtime_friction_contract runtime_capsule_manifold runtime_loop_compliance runtime_loop_controls runtime_loop_contacts runtime_loop_articulated_contacts runtime_rigid_fixed_loops runtime_scene runtime_tree runtime_mass runtime_limit_reactions runtime_softness loop_forces featherstone_polish_cpu featherstone_articulation_cpu contact_friction_modes runtime_multidof_limits runtime_d6_axis_certification rigid_singular_forces primitives_benchmark_process primitives_benchmark_settings
PRIMITIVE_GPU_BASELINES := featherstone_polish_gpu featherstone_multidof_gpu featherstone_motor_caps_gpu runtime_motor_models runtime_motor_coupling runtime_friction_contract runtime_capsule_manifold runtime_loop_compliance runtime_loop_controls runtime_loop_contacts runtime_loop_articulated_contacts runtime_rigid_fixed_loops contact_friction_modes_gpu featherstone_scalar_equality_gpu rigid_singular_forces
PRIMITIVE_CPU_BASELINES += runtime_rigid_axis_loops
PRIMITIVE_CPU_BASELINES += primitives_benchmark_contract
PRIMITIVE_CPU_BASELINES += runtime_mixed_drives
PRIMITIVE_CPU_BASELINES += runtime_locked_drive_chains
PRIMITIVE_GPU_BASELINES += runtime_locked_drive_chains
PRIMITIVE_CPU_BASELINES += runtime_locked_drive_planes
PRIMITIVE_GPU_BASELINES += runtime_locked_drive_planes
PRIMITIVE_GPU_BASELINES += runtime_mixed_drives
PRIMITIVE_GPU_BASELINES += runtime_rigid_axis_loops
PRIMITIVE_CPU_BASELINES += runtime_rigid_loop_drives
PRIMITIVE_GPU_BASELINES += runtime_rigid_loop_drives
PRIMITIVE_CPU_BASELINES += runtime_acceleration_drives
PRIMITIVE_GPU_BASELINES += runtime_acceleration_drives
PRIMITIVE_CPU_BASELINES += runtime_ball_drives
PRIMITIVE_GPU_BASELINES += runtime_ball_drives
PRIMITIVE_CPU_BASELINES += runtime_loop_brakes
PRIMITIVE_GPU_BASELINES += runtime_loop_brakes
PRIMITIVE_CPU_BASELINES += runtime_contact_sleep
PRIMITIVE_GPU_BASELINES += runtime_contact_sleep
PRIMITIVE_CPU_BASELINES += runtime_d6_singular_bounds
PRIMITIVE_GPU_BASELINES += runtime_d6_singular_bounds
PRIMITIVE_CPU_BASELINES += runtime_velocity_coupling
PRIMITIVE_GPU_BASELINES += runtime_velocity_coupling
PRIMITIVE_CPU_BASELINES += runtime_d6_angular_locks
PRIMITIVE_GPU_BASELINES += runtime_d6_angular_locks
PRIMITIVE_CPU_BASELINES += featherstone_disabled_joints
PRIMITIVE_GPU_BASELINES += featherstone_disabled_joints
PRIMITIVE_CPU_BASELINES += featherstone_mixed_distance
PRIMITIVE_GPU_BASELINES += featherstone_mixed_distance
PRIMITIVE_CPU_BASELINES += runtime_d6_angular_mass
PRIMITIVE_GPU_BASELINES += runtime_d6_angular_mass
PRIMITIVE_CPU_BASELINES += runtime_d6_hinge_antipodes
PRIMITIVE_GPU_BASELINES += runtime_d6_hinge_antipodes
PRIMITIVE_CPU_BASELINES += standalone_limit_precision
PRIMITIVE_GPU_BASELINES += standalone_limit_precision
PRIMITIVE_CPU_BASELINES += standalone_ball_limits
PRIMITIVE_GPU_BASELINES += standalone_ball_limits
PRIMITIVE_CPU_BASELINES += standalone_redundant_limits
PRIMITIVE_GPU_BASELINES += standalone_redundant_limits
PRIMITIVE_CPU_BASELINES += runtime_constraint_readout
PRIMITIVE_GPU_BASELINES += runtime_constraint_readout
PRIMITIVE_CPU_BASELINES += analysis_response_contract
PRIMITIVE_GPU_BASELINES += articulated_response_gpu
PRIMITIVE_CPU_BASELINES += runtime_density_mass
PRIMITIVE_GPU_BASELINES += runtime_density_mass

PRIMITIVE_CPU_LANES := test-primitives-baseline test-primitives-distance-cpu test-primitives-cone-cpu \
	test-primitives-reactions-cpu test-primitives-kinematic-motion-cpu test-primitives-breakage-cpu \
	test-primitives-friction-cpu test-primitives-coupled-constraints-cpu test-primitives-mixed-contacts-cpu \
	test-primitives-impact-oracle test-primitives-articulated-impact-cpu test-primitives-restitution-cpu \
	test-primitives-contact-geometry-cpu test-primitives-impact-identity test-primitives-dispatch-metrics test-primitives-loop-rows-cpu
PRIMITIVE_GPU_LANES := test-primitives-baseline-gpu test-primitives-runtime-gpu \
	test-primitives-distance-gpu test-primitives-distance-scale-gpu test-primitives-cone-gpu test-primitives-reactions-gpu \
	test-primitives-breakage-gpu test-primitives-kinematic-motion-gpu test-primitives-pose-precision-gpu \
	test-primitives-contact-pipeline-gpu test-primitives-impact-episodes-gpu test-primitives-impact-gpu \
	test-primitives-impact-response-gpu test-primitives-impact-position-gpu \
	test-primitives-impact-partition-gpu test-primitives-restitution-gpu test-primitives-dispatch-metrics-gpu test-primitives-multidof-feedback-gpu test-primitives-loop-rows-gpu test-primitives-loop-resources-gpu
TYPE ?= patch
PRIMITIVE_GPU_LANES += test-primitives-contact-activation-gpu
PRIMITIVE_CPU_LANES += test-primitives-softness-activation
PRIMITIVE_CPU_LANES += test-primitives-construction-cpu
PRIMITIVE_CPU_LANES += test-primitives-program-cache-cpu
PRIMITIVE_GPU_LANES += test-primitives-program-cache-gpu
HAS_REL := $(shell command -v git-rel 2>/dev/null)

# GPU test lane (see `test-gpu`): package P + test file T, extra args A.
P ?= molla-solvers
T ?= cpu_gpu_parity
GPU_FEATURES ?= gpu
REL := $(if $(RELEASE),--release,)

$(info ------------------------------------------)
$(info Project: $(PROJECT_NAME) v$(PROJECT_VERSION))
$(info Display: $(BACKEND) backend)
$(info ------------------------------------------)

.PHONY: build b compile c run r test t test-gpu test-sensors test-sensor-resources test-viewer-sensors test-all check check-all fmt-files fmt-check clippy-check harden bench bench-sensors bench-sensors-latency sensor-bench-info clean docs release help h
.PHONY: test-primitives-baseline test-primitives-baseline-gpu test-primitives-runtime-gpu test-primitives-distance-cpu test-primitives-distance-gpu
.PHONY: test-primitives-cone-geometry-cpu test-primitives-cone-geometry-gpu
.PHONY: test-primitives-loop-rows-cpu test-primitives-loop-rows-gpu test-primitives-loop-resources-gpu
.PHONY: test-primitives-cone-cpu test-primitives-cone-gpu
.PHONY: test-primitives-reactions-cpu test-primitives-reactions-gpu
.PHONY: test-primitives-breakage-cpu test-primitives-breakage-gpu
.PHONY: test-primitives-friction-cpu test-primitives-friction-gpu test-primitives-contact-islands-gpu test-primitives-contact-response-gpu
.PHONY: test-primitives-contact-pipeline-gpu
.PHONY: test-primitives-contact-publication-gpu
.PHONY: test-primitives-runtime-contacts-gpu
.PHONY: test-primitives-contact-reactions-gpu
.PHONY: test-primitives-contact-breakage-gpu
.PHONY: test-primitives test-primitives-gpu test-primitives-contact-geometry-cpu test-primitives-impact-identity

build:
	@$(CARGO) build $(APP_TARGET)

b: build

compile:
	@$(CARGO) clean
	@$(MAKE) build

c: compile

run:
	@$(RUN_WITH) $(CARGO) run $(APP_TARGET)

r: run

test:
	@$(CARGO) test $(APP_TARGET)

.PHONY: test-soil-particle-render
.PHONY: render-soil-tractor-resident
render-soil-tractor-resident:
	@test -n '$(SOIL_RESIDENT_OUTPUT)' -a ! -e '$(SOIL_RESIDENT_OUTPUT)' || { echo 'Set a fresh SOIL_RESIDENT_OUTPUT directory'; exit 2; }
	@$(CARGO) build --release -p molla-solvers --features gpu,resource-counters,norm-tree-newton --example soil_tractor_resident
	@timeout --kill-after=5s 180s $(RUN_WITH) $(CARGO) run --release -p molla-solvers --features gpu,resource-counters,norm-tree-newton --example soil_tractor_resident -- '$(SOIL_RESIDENT_OUTPUT)'

test-soil-particle-render:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --test mpm_render_gpu --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --test mpm_render_gpu -- --ignored --nocapture --test-threads=1

t: test

test-primitives:
	@set -e; for lane in $(PRIMITIVE_CPU_LANES); do $(MAKE) -j1 $$lane; done

test-primitives-gpu:
	@set -e; for lane in $(PRIMITIVE_GPU_LANES); do $(MAKE) -j1 $$lane; done

.PHONY: test-primitives-program-cache-cpu test-primitives-program-cache-gpu
test-primitives-program-cache-cpu:
	@$(MAKE) test APP_TARGET='-p molla-compute --features gpu --lib'
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --lib featherstone::scratch_tests -- --nocapture'

test-primitives-program-cache-gpu:
	@$(RUN_WITH) $(CARGO) test -p molla-compute --features gpu --lib engine::tests::gpu_program_cache -- --ignored --nocapture --test-threads=1

test-primitives-contact-geometry-cpu:
	@$(MAKE) test APP_TARGET='-p molla-geometry --lib -- --nocapture'

test-primitives-impact-identity:
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu --lib impact_episodes_gpu::tests -- --nocapture'

# GPU tests need a real wgpu adapter (only via nixVulkan) and are
# #[ignore]-marked. Pick the package P and test file T; pass extra args
# via A and add RELEASE=1 for perf lanes:
#   make test-gpu                                  # CPU/GPU parity lane
#   make test-gpu P=molla-geometry T=gjk_gpu A=--nocapture
#   make test-gpu P=molla-fem T=fem_ns_gpu RELEASE=1 A='--skip gpu_ns_perf...'
test-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p $(P) --features $(GPU_FEATURES) --test $(T) -- --ignored $(A)

.PHONY: test-soil-fast test-soil-gpu bench-soil check-soil-results test-soil-wrench-replay test-soil-sparse-map test-soil-particle-order
.PHONY: test-soil-sort test-soil-feedback test-soil-command-batch
SOIL_BATCH_FILTER ?= mpm::command_batch::tests
.PHONY: test-soil-gather
.PHONY: test-soil-fixed-order-runtime
test-soil-fixed-order-runtime:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --test mpm_fixed_order_runtime -- --ignored --nocapture --test-threads=1
.PHONY: test-soil-checkpoint
.PHONY: test-soil-tiles
.PHONY: test-soil-checkpoint-async
.PHONY: test-soil-identity
.PHONY: test-soil-world test-soil-disk-world test-soil-window test-soil-window-integer test-soil-motion-guard
.PHONY: test-soil-transfer
.PHONY: test-soil-pins
.PHONY: test-soil-regions
.PHONY: test-soil-quiescence
.PHONY: test-soil-settling
.PHONY: test-soil-numerical-fault
.PHONY: settle-soil
.PHONY: test-soil-fusion
SOIL_FUSION_FILTER ?= mpm::fused_gather::tests
SOIL_FUSION_ARGS ?= --skip benchmark
test-soil-fusion:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_FUSION_FILTER) -- --ignored --nocapture --test-threads=1 $(SOIL_FUSION_ARGS)

.PHONY: test-soil-store test-soil-store-gpu test-soil-solids test-soil-rigid-device

SOIL_RIGID_DEVICE_FILTER ?= mpm::rigid_device::tests
.PHONY: test-soil-runtime test-soil-frame-backup test-soil-completion
SOIL_RUNTIME_FILTER ?= rigid_world::gpu::soil::tests
.PHONY: test-soil-contact-projection
.PHONY: test-soil-normal-projection
.PHONY: test-soil-floor-projection
.PHONY: test-soil-contact-geometry
.PHONY: test-soil-candidates
SOIL_CANDIDATES_FILTER ?= soil_candidates_gpu::tests
test-soil-candidates:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_CANDIDATES_FILTER) -- --ignored --nocapture --test-threads=1

SOIL_GEOMETRY_FILTER ?= soil_contact_geometry_gpu::tests
test-soil-contact-geometry:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_GEOMETRY_FILTER) -- --ignored --nocapture --test-threads=1

SOIL_FLOOR_FILTER ?= soil_floor_projection_gpu::tests
test-soil-floor-projection:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_FLOOR_FILTER) -- --ignored --nocapture --test-threads=1

SOIL_NORMAL_FILTER ?= soil_normal_projection_gpu::tests
.PHONY: test-soil-friction-disk
.PHONY: test-soil-wall-friction
SOIL_WALL_FRICTION_FILTER ?= soil_wall_friction_gpu::tests
.PHONY: test-soil-shared-wall-friction
SOIL_SHARED_WALL_FILTER ?= soil_shared_wall_friction_gpu::tests
.PHONY: test-soil-shared-wall-geometry
SOIL_SHARED_WALL_GEOMETRY_FILTER ?= soil_shared_wall_geometry_gpu::tests
test-soil-shared-wall-geometry:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_SHARED_WALL_GEOMETRY_FILTER) -- --ignored --nocapture --test-threads=1

test-soil-shared-wall-friction:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_SHARED_WALL_FILTER) -- --ignored --nocapture --test-threads=1

test-soil-wall-friction:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_WALL_FRICTION_FILTER) -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-tangential-projection
.PHONY: test-soil-friction-projection
.PHONY: test-soil-friction-geometry
test-soil-friction-geometry:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib soil_friction_geometry_gpu::tests -- --ignored --nocapture --test-threads=1

test-soil-friction-projection:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib soil_friction_projection_gpu::tests -- --ignored --nocapture --test-threads=1

test-soil-tangential-projection:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib soil_tangential_projection_gpu::tests -- --ignored --nocapture --test-threads=1

test-soil-friction-disk:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib soil_friction_disk_gpu::tests -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-contact-motion
test-soil-contact-motion:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib soil_contact_motion_gpu::tests -- --ignored --nocapture --test-threads=1

test-soil-normal-projection:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_NORMAL_FILTER) -- --ignored --nocapture --test-threads=1

SOIL_PROJECTION_FILTER ?= soil_contact_projection_gpu::tests
test-soil-contact-projection:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_PROJECTION_FILTER) -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-mass-coupling test-soil-ordered-coupling test-soil-projection-commands test-soil-canonical-sticking test-soil-native-submissions test-soil-profile-budget
test-soil-profile-budget:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib profile_query_budget
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib profile_query_pages_cover_nested_windows_and_reuse -- --ignored --nocapture --test-threads=1

test-soil-native-submissions:
	@test -n "$$MOLLA_NATIVE_WORLD_CAPTURE" -a ! -e "$$MOLLA_NATIVE_WORLD_CAPTURE" || { echo 'Set MOLLA_NATIVE_WORLD_CAPTURE to a fresh output directory'; exit 2; }
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,native-lazy-geometry-drift,soa-scratch-newton,row-parallel-newton-preparation,parallel-candidate-index,deferred-native-drift,batched-soil-projection --lib native_world_composed_submissions_exact --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,native-lazy-geometry-drift,soa-scratch-newton,row-parallel-newton-preparation,parallel-candidate-index,deferred-native-drift,batched-soil-projection --lib native_world_composed_submissions_exact -- --ignored --nocapture --test-threads=1

test-soil-canonical-sticking:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,batched-soil-projection,ordered-sticking-nodes --lib canonical_node_order_source_and_limits
	@set -e; for name in canonical_node_order_gpu projected_commands_order_and_parity projected_commands_late_fault_rollback all_fixed_grid_faces_project_machine_and_particles_with_external_impulse active_mass_coupling_fixed_faces active_mass_coupling_lifecycle; do \
	  timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,batched-soil-projection,ordered-sticking-nodes --lib $$name -- --ignored --nocapture --test-threads=1; \
	done

test-soil-projection-commands:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,batched-soil-projection --lib --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,batched-soil-projection --lib projected_commands_order_and_parity -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,batched-soil-projection --lib projected_commands_late_fault_rollback -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,batched-soil-projection --lib rigid_world::gpu::soil::projected_tests::projected_tractor_runtime_conserves_impulse_and_rolls_back -- --ignored --exact --nocapture --test-threads=1

.PHONY: debug-soil-construction
SOIL_CONSTRUCTION_TEST_BINARY ?=
SOIL_CONSTRUCTION_DEBUG_DIR ?=
debug-soil-construction:
	@test -n '$(SOIL_CONSTRUCTION_TEST_BINARY)' -a -x '$(SOIL_CONSTRUCTION_TEST_BINARY)' || { echo 'Set SOIL_CONSTRUCTION_TEST_BINARY to a precompiled kernel test executable'; exit 2; }
	@test -n '$(SOIL_CONSTRUCTION_DEBUG_DIR)' -a ! -e '$(SOIL_CONSTRUCTION_DEBUG_DIR)' || { echo 'Set SOIL_CONSTRUCTION_DEBUG_DIR to a new directory'; exit 2; }
	@mkdir -p '$(SOIL_CONSTRUCTION_DEBUG_DIR)'
	@sha256sum '$(SOIL_CONSTRUCTION_TEST_BINARY)' > '$(SOIL_CONSTRUCTION_DEBUG_DIR)/executable-sha256.txt'
	@$(RUN_WITH) timeout --foreground --kill-after=5s 45s gdb -nx -q -batch -ex 'set pagination off' -ex 'set confirm off' -ex 'set debuginfod enabled off' -ex 'set logging file $(SOIL_CONSTRUCTION_DEBUG_DIR)/backtrace.log' -ex 'set logging enabled on' -ex run -ex 'thread apply all bt 20' -ex 'info sharedlibrary' -ex quit --args '$(SOIL_CONSTRUCTION_TEST_BINARY)' featherstone::construction_tests::six_joint_phase_pipeline_construction --ignored --nocapture --test-threads=1

.PHONY: test-soil-pipeline-cache test-soil-pipeline-cache-engine test-soil-pipeline-cache-machine
.PHONY: test-soil-joint-types test-soil-joint-types-runtime
.PHONY: test-soil-joint-types-paired
test-soil-joint-types-paired:
	@test '$(SOIL_PAIRED_TYPES_ARM)' = reference -o '$(SOIL_PAIRED_TYPES_ARM)' = candidate || { echo 'Set SOIL_PAIRED_TYPES_ARM=reference or candidate'; exit 2; }
	@test -n '$(SOIL_PAIRED_TYPES_OUTPUT)' -a ! -e '$(SOIL_PAIRED_TYPES_OUTPUT)' || { echo 'Set a fresh SOIL_PAIRED_TYPES_OUTPUT file'; exit 2; }
	@$(CARGO) test --release -p molla-kernels --features gpu,resource-counters --lib paired_joint_type_profile_retains_low_rates_and_guards --no-run
	@SOIL_PAIRED_TYPES_ARM='$(SOIL_PAIRED_TYPES_ARM)' SOIL_PAIRED_TYPES_OUTPUT='$(SOIL_PAIRED_TYPES_OUTPUT)' timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-kernels --features gpu,resource-counters --lib paired_joint_type_profile_retains_low_rates_and_guards -- --ignored --nocapture --test-threads=1

test-soil-joint-types:
	@$(CARGO) test --release -p molla-kernels --features gpu,resource-counters --lib featherstone::joint_types:: --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-kernels --features gpu,resource-counters --lib featherstone::joint_types:: -- --include-ignored --nocapture --test-threads=1

test-soil-joint-types-runtime:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib joint_types_friction_tractor_runtime_conserves_impulse_and_rolls_back --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib joint_types_friction_tractor_runtime_conserves_impulse_and_rolls_back -- --ignored --nocapture --test-threads=1

test-soil-pipeline-cache:
	@$(MAKE) test-soil-pipeline-cache-engine
	@$(MAKE) test-soil-pipeline-cache-machine

test-soil-pipeline-cache-engine:
	@$(CARGO) test --release -p molla-compute --features gpu --lib pipeline_cache:: --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-compute --features gpu --lib pipeline_cache:: -- --include-ignored --nocapture --test-threads=1

test-soil-pipeline-cache-machine:
	@$(CARGO) test --release -p molla-kernels --features gpu,resource-counters --lib featherstone::pipeline_cache_tests:: --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-kernels --features gpu,resource-counters --lib featherstone::pipeline_cache_tests:: -- --ignored --nocapture --test-threads=1

test-soil-mass-coupling:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::mass_coupling::tests -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-active-mass-coupling
.PHONY: test-soil-active-unilateral
.PHONY: test-soil-active-kinematics
.PHONY: test-soil-runtime-grid-budget
.PHONY: test-soil-active-runtime
.PHONY: test-soil-active-runtime-joint-types
.PHONY: test-soil-compact-tractor-runtime
.PHONY: test-soil-tractor-grid-selection
test-soil-tractor-grid-selection:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib tractor_grid_storage -- --nocapture --test-threads=1
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --example soil_tractor_benchmark -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib tractor_grid_storage_factory -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-tractor-grid-support
test-soil-tractor-grid-support:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib tractor_grid_initial_support --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib tractor_grid_initial_support -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-preparation-fixed-width
.PHONY: test-soil-preparation-certificate-mask
test-soil-preparation-certificate-mask:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation -- --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation_certificate_mask -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-preparation-captured-certificates
.PHONY: test-soil-preparation-axes
test-soil-preparation-axes:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation -- --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation_axis_parallel -- --ignored --nocapture --test-threads=1

test-soil-preparation-captured-certificates:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation -- --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation_captured_certificates -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation_cached_faults -- --ignored --nocapture --test-threads=1

test-soil-preparation-fixed-width:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation -- --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation_fixed_width -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-preparation-private-response
test-soil-preparation-private-response:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation -- --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation_private_response -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-preparation-zero-terms
test-soil-preparation-zero-terms:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation -- --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation_zero_terms -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-preparation-sparse-residual
test-soil-preparation-sparse-residual:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation -- --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib newton_preparation_sparse_residual -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-sparse-residual-runtime
test-soil-sparse-residual-runtime:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton,zero-residual-newton-preparation --lib newton_preparation -- --nocapture --test-threads=1
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton,zero-residual-newton-preparation --example soil_tractor_benchmark -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton,zero-residual-newton-preparation --lib newton_preparation_sparse_residual -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton,zero-residual-newton-preparation --lib newton_preparation_gpu_faults -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton,zero-residual-newton-preparation --lib active_unilateral_newton_feedback -- --ignored --nocapture --test-threads=1

test-soil-compact-tractor-runtime:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib compact_joint_types_friction_tractor_runtime --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib compact_joint_types_friction_tractor_runtime -- --ignored --nocapture --test-threads=1

test-soil-active-runtime-joint-types:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_runtime_joint_types_attachment_transactions --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_runtime_joint_types_attachment_transactions -- --ignored --nocapture --test-threads=1

test-soil-active-runtime:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_runtime_ --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_runtime_direct_machine_fault_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_runtime_attachment_transactions -- --ignored --nocapture --test-threads=1

test-soil-runtime-grid-budget:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib projected_runtime_grid_capacity_budgets --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib projected_runtime_grid_capacity_budgets -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_mass_coupling_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_unilateral_coupling_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_unilateral_paired_feedback -- --ignored --nocapture --test-threads=1

test-soil-active-kinematics:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib kinematics_only --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib kinematics_only -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib device_bridge_rejects_stale_handles_foreign_state_and_host_override -- --ignored --nocapture --test-threads=1

test-soil-active-unilateral:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_unilateral --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_unilateral_coupling_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_unilateral_paired_feedback -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_unilateral_newton_feedback -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-indexed-candidates
test-soil-indexed-candidates:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib indexed_candidates_preserve_world_order_and_publication --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib indexed_candidates_preserve_world_order_and_publication -- --ignored --nocapture --test-threads=1

test-soil-active-mass-coupling:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_mass_coupling --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib active_mass_coupling -- --ignored --nocapture --test-threads=1

test-soil-ordered-coupling:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib mpm::mass_coupling::tests::ordered --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib mpm::mass_coupling::tests::ordered -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib projected_grid_drives_all_particles_and_reports_one_reaction -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib late_particle_fault_blocks_machine_publication_and_frame_rollback_recovers -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib mpm::frame_backup::tests -- --ignored --nocapture --test-threads=1
.PHONY: test-soil-tractor test-soil-tractor-gpu test-soil-tractor-replay test-soil-p2g-parity test-soil-native-atomics
SOIL_TRACTOR_FILTER ?= soil_tractor::tests
.PHONY: test-soil-admm-tractor
SOIL_ADMM_PREPARATION ?= cooperative
SOIL_ADMM_ARM ?= native
SOIL_ADMM_OUTPUT ?=
SOIL_ADMM_FEATURE_ARGS = --features gpu,resource-counters,protected-newton $(if $(filter cooperative,$(SOIL_ADMM_PREPARATION)),--features row-parallel-newton-preparation)
SOIL_ADMM_PREPARATION_LABEL = $(if $(filter cooperative,$(SOIL_ADMM_PREPARATION)),compensated-row-parallel-cooperative-factor,compensated-dense-terms)
test-soil-admm-tractor:
	@case '$(SOIL_ADMM_PREPARATION)' in dense|cooperative) ;; *) echo 'SOIL_ADMM_PREPARATION must be dense or cooperative' >&2; exit 1;; esac
	@case '$(SOIL_ADMM_ARM)' in native|featherstone) ;; *) echo 'SOIL_ADMM_ARM must be native or featherstone' >&2; exit 1;; esac
	@test -n '$(SOIL_ADMM_OUTPUT)' && test ! -e '$(SOIL_ADMM_OUTPUT)' || { echo 'SOIL_ADMM_OUTPUT must name a new directory' >&2; exit 1; }
	@$(CARGO) test --release -p molla-solvers $(SOIL_ADMM_FEATURE_ARGS) --lib admm_actual_tractor_50k_arm --no-run
	@mkdir -p "$$(dirname '$(SOIL_ADMM_OUTPUT)')"
	@MOLLA_ADMM50K_ARM='$(SOIL_ADMM_ARM)' MOLLA_ADMM50K_PREPARATION='$(SOIL_ADMM_PREPARATION_LABEL)' MOLLA_ADMM50K_OUTPUT='$(abspath $(SOIL_ADMM_OUTPUT))' timeout --kill-after=5s 180s $(RUN_WITH) $(CARGO) test --release -p molla-solvers $(SOIL_ADMM_FEATURE_ARGS) --lib admm_actual_tractor_50k_arm -- --ignored --nocapture --test-threads=1

test-soil-tractor:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib soil_tractor::tests -- --nocapture --test-threads=1

test-soil-tractor-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_TRACTOR_FILTER) -- --ignored --nocapture --test-threads=1

test-soil-tractor-replay:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::replay_tests::captured_tractor_boundary_replay -- --ignored --nocapture --test-threads=1

test-soil-p2g-parity:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::replay_tests::p2g_shared_accumulation_matches_global_scatter -- --ignored --nocapture --test-threads=1

test-soil-native-atomics:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::replay_tests::native_float_atomic_capability_probe -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-native-mpm
.PHONY: test-soil-native-active
.PHONY: test-soil-native-ordered-shell
.PHONY: test-soil-native-precontact-compensation
.PHONY: test-soil-shared-gather test-soil-shared-gather-fast
test-soil-shared-gather-fast:
	@$(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib gather_chunk_source_contract -- --nocapture
	@$(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib gather_replay_uniform_versions -- --nocapture

test-soil-shared-gather:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib gather_chunk_exact_grid_and_evolution --no-run
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --test mpm_shared_gather_runtime --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib gather_chunk_exact_grid_and_evolution -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --test mpm_shared_gather_runtime -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-compensated-wrench test-soil-compensated-wrench-fast
test-soil-compensated-wrench-fast:
	@$(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib retained_wrench_source_contract -- --nocapture

test-soil-compensated-wrench:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib retained_wrench --no-run
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --test mpm_compensated_wrench_runtime --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib retained_wrench -- --include-ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --test mpm_compensated_wrench_runtime -- --ignored --nocapture --test-threads=1

.PHONY: bench-soil-native-precontact
bench-soil-native-precontact:
	@test -n "$(SOIL_PRECONTACT_OUTPUT)" && test ! -e "$(SOIL_PRECONTACT_OUTPUT)"
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_precontact_whole_step_cost --no-run
	@SOIL_PRECONTACT_OUTPUT='$(SOIL_PRECONTACT_OUTPUT)' timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_precontact_whole_step_cost -- --ignored --nocapture --test-threads=1

test-soil-native-precontact-compensation:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_precontact_compensation_source_contract -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_precontact_compensated_active_parity -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_precontact_compensation_gpu_lifecycle -- --ignored --nocapture --test-threads=1

test-soil-native-ordered-shell:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_active_ordered_shell_parity --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_active_ordered_shell_parity -- --ignored --nocapture --test-threads=1

.PHONY: diagnose-soil-native-force
.PHONY: diagnose-soil-native-force-substitution
diagnose-soil-native-force-substitution:
	@test -n "$(SOIL_NATIVE_FORCE_OUTPUT)" && test ! -e "$(SOIL_NATIVE_FORCE_OUTPUT)" && test -f "$(SOIL_NATIVE_FORCE_INPUT)/failure.json"
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_force_terms_preserve_scalar_expressions -- --nocapture
	@MOLLA_NATIVE_FORCE_REPLAY='$(SOIL_NATIVE_FORCE_OUTPUT)' MOLLA_NATIVE_FORCE_INPUT='$(SOIL_NATIVE_FORCE_INPUT)' timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_rigid_grid_substitution_diagnostic -- --ignored --nocapture --test-threads=1

diagnose-soil-native-force:
	@test -n "$(SOIL_NATIVE_FORCE_OUTPUT)" && test ! -e "$(SOIL_NATIVE_FORCE_OUTPUT)"
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_force_terms_preserve_scalar_expressions -- --nocapture
	@MOLLA_NATIVE_FORCE_REPLAY='$(SOIL_NATIVE_FORCE_OUTPUT)' timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_rigid_identical_grid_diagnostic -- --ignored --nocapture --test-threads=1

test-soil-native-active:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_active --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_active -- --ignored --skip diagnostic --skip ordered_shell --nocapture --test-threads=1

test-soil-native-mpm:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::replay_tests::native_explicit -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-frame-profile
test-soil-frame-profile:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::replay_tests::frame_profile_window -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-machine-rounding
SOIL_MACHINE_ROUNDING_FILTER ?= featherstone::rounding_tests::local_position_rounding
test-soil-machine-rounding:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-kernels --features gpu --lib $(SOIL_MACHINE_ROUNDING_FILTER) -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-rounding-capture check-soil-rounding-capture
SOIL_ROUNDING_CAPTURE_REPORT ?= $(SOIL_ROUNDING_CAPTURE)/audit.json
test-soil-rounding-capture:
	@perl scripts/check-soil-rounding-capture.pl --self-test

check-soil-rounding-capture:
	@test -n "$(SOIL_ROUNDING_CAPTURE)" -a -n "$(SOIL_ROUNDING_CAPTURE_LOG)" || { echo "Set SOIL_ROUNDING_CAPTURE and SOIL_ROUNDING_CAPTURE_LOG" >&2; exit 2; }
	@perl scripts/check-soil-rounding-capture.pl "$(SOIL_ROUNDING_CAPTURE)" "$(SOIL_ROUNDING_CAPTURE_LOG)" > "$(SOIL_ROUNDING_CAPTURE_REPORT)"

test-soil-runtime:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_RUNTIME_FILTER) -- --ignored --nocapture --test-threads=1

test-soil-completion:
	@$(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib rigid_world::gpu::soil_completion::tests -- --nocapture --test-threads=1

test-soil-frame-backup:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::frame_backup::tests -- --ignored --nocapture --test-threads=1

test-soil-rigid-device:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_RIGID_DEVICE_FILTER) -- --ignored --nocapture --test-threads=1

test-soil-solids:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::prescribed_solids::tests -- --ignored --nocapture --test-threads=1
test-soil-store:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::tiles::store:: -- --nocapture --test-threads=1

.PHONY: test-soil-store-preflight
test-soil-store-preflight:
	@$(MAKE) test-soil-store
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters $(REL) --lib --no-run'
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib -- --ignored --nocapture --test-threads=1 canonical_store_gpu framed_store_gpu live_gpu_boundary_save_reload_and_resume_preserve_history_and_freshness quiescence_gpu_world_identity_and_pending_boundaries

.PHONY: test-soil-store-transition
test-soil-store-transition:
	@$(MAKE) test-soil-store
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters $(REL) --lib -- disk_selection_admission canonical_backend_dispatch --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters $(REL) --lib --no-run'
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib -- --ignored --nocapture --test-threads=1 disk_capacity_rejects disk_gpu_pins_join active_swept_pin_capacity_failure live_disk_capture_write_load disk_write_failure load_failure_after_durable_write canonical_live_budget_retry canonical_live_unloaded_corruption framed_disk_capacity_retry

.PHONY: test-soil-store-index
test-soil-store-index:
	@$(MAKE) test-soil-store
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters $(REL) --lib -- disk_selection_admission canonical_backend_dispatch --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters $(REL) --lib --no-run'
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib -- --ignored --nocapture --test-threads=1 disk_capacity_rejects disk_gpu_pins_join active_swept_pin_capacity_failure live_disk_capture_write_load disk_write_failure load_failure_after_durable_write canonical_live_budget_retry canonical_live_unloaded_corruption framed_disk_capacity_retry canonical_store_gpu framed_store_gpu live_gpu_boundary_save_reload_and_resume_preserve_history_and_freshness

.PHONY: test-soil-store-stream
test-soil-store-stream:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::tiles::store::stream_tests -- --nocapture --test-threads=1

.PHONY: test-soil-store-index-pages
test-soil-store-index-pages:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::tiles::store::index:: -- --nocapture --test-threads=1

.PHONY: test-soil-store-incremental
test-soil-store-incremental:
	@$(MAKE) test-soil-store
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters $(REL) --lib --no-run'
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib -- --ignored --nocapture --test-threads=1 incremental_disk_capture_write_load_and_reopen incremental_canonical_live_overlap_transfer_and_reopen incremental_framed_disk_live_overlap_steps_and_reopen

test-soil-store-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::tiles::store::canonical_store::tests::canonical_store_gpu -- --ignored --nocapture --test-threads=1
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::tiles::store::tests::live_gpu_boundary -- --ignored --nocapture --test-threads=1
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::tiles::store::framed_store::tests::framed_store_gpu -- --ignored --nocapture --test-threads=1

test-soil-pins:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::world::pins::tests -- --ignored --nocapture --test-threads=1

test-soil-regions:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::world::regions::tests -- --ignored --nocapture --test-threads=1

test-soil-settling:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::world::settling::tests -- --ignored --nocapture --test-threads=1

test-soil-numerical-fault:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::numerical_fault_tests -- --ignored --nocapture --test-threads=1

SOIL_SETTLING_DIR ?= benchmarks/soil/settling-$(shell date +%Y%m%d-%H%M%S)
SOIL_SETTLING_RESUME ?=
settle-soil:
	@[[ '$(SOIL_BENCH_SECONDS)' =~ ^[1-9][0-9]*$$ && '$(SOIL_BENCH_SECONDS)' -le 30 ]] || { echo 'Quick execution deadline must be 1 through 30 seconds'; exit 2; }
	@test ! -e '$(SOIL_SETTLING_DIR)' || { echo 'Refusing to reuse settling output'; exit 2; }
	@$(MAKE) build APP_TARGET='-p molla-solvers --features gpu,resource-counters --example soil_settling --release'
	@mkdir -p '$(SOIL_SETTLING_DIR)'
	@git rev-parse HEAD > '$(SOIL_SETTLING_DIR)/revision.txt'
	@git status --short > '$(SOIL_SETTLING_DIR)/worktree.txt'
	@{ printf '%s\0' Makefile Cargo.toml Cargo.lock; find crates -type f \( -name '*.rs' -o -name '*.wgsl' -o -name Cargo.toml \) -print0; } | sort -z | xargs -0 sha256sum > '$(SOIL_SETTLING_DIR)/source-sha256.txt'
	@tar -czf '$(SOIL_SETTLING_DIR)/source.tar.gz' --null -T <({ printf '%s\0' Makefile Cargo.toml Cargo.lock; find crates -type f \( -name '*.rs' -o -name '*.wgsl' -o -name Cargo.toml \) -print0; } | sort -z)
	@cp target/release/examples/soil_settling '$(SOIL_SETTLING_DIR)/soil_settling'
	@sha256sum '$(SOIL_SETTLING_DIR)/soil_settling' > '$(SOIL_SETTLING_DIR)/executable-sha256.txt'
	@$(NVIDIA_SMI) --query-gpu=timestamp,name,driver_version,utilization.gpu,temperature.gpu --format=csv > '$(SOIL_SETTLING_DIR)/hardware-before.csv'
	@set -o pipefail; timeout --kill-after=5s $(SOIL_BENCH_SECONDS)s $(RUN_WITH) '$(SOIL_SETTLING_DIR)/soil_settling' --output '$(SOIL_SETTLING_DIR)/results' $(if $(strip $(SOIL_SETTLING_RESUME)),--resume-run '$(SOIL_SETTLING_RESUME)') 2>&1 | tee '$(SOIL_SETTLING_DIR)/run.log'; result=$$?; echo $$result > '$(SOIL_SETTLING_DIR)/run.exit'; \
	  $(NVIDIA_SMI) --query-gpu=timestamp,name,utilization.gpu,temperature.gpu --format=csv > '$(SOIL_SETTLING_DIR)/hardware-after.csv' || exit 2; \
	  sha256sum --check '$(SOIL_SETTLING_DIR)/source-sha256.txt' > '$(SOIL_SETTLING_DIR)/source-check.log'; source=$$?; echo $$source > '$(SOIL_SETTLING_DIR)/source.exit'; \
	  if [ $$source -ne 0 ]; then exit $$source; fi; exit $$result

SOIL_TRANSFER_FILTER ?= mpm::checkpoint::world::transfer::tests
SOIL_TRANSFER_ARGS ?= --skip benchmark
test-soil-transfer:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_TRANSFER_FILTER) -- --ignored --nocapture --test-threads=1 $(SOIL_TRANSFER_ARGS)

test-soil-world:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::world::tests -- --ignored --nocapture --test-threads=1

test-soil-disk-world:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::world::disk::tests -- --ignored --nocapture --test-threads=1

test-soil-window:
	@$(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --test mpm_grid_window_runtime --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --test mpm_grid_window_runtime grid_window_public_repeat_physical -- --ignored --nocapture --test-threads=1

test-soil-window-integer:
	@$(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --test mpm_integer_window_runtime --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --test mpm_integer_window_runtime integer_window_public_ -- --ignored --nocapture --test-threads=1

test-soil-motion-guard:
	@$(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib motion_guard_ --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib motion_guard_ -- --ignored --nocapture --test-threads=1

test-soil-quiescence:
	@$(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib quiescence_ --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib quiescence_gpu_ -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-quiescence-async
test-soil-quiescence-async:
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters $(REL) --lib -- --nocapture quiescence_async_ quiescence_layout_ quiescence_support_ quiescence_retained_support_ quiescence_region_support_'
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters $(REL) --lib quiescence_gpu_ --no-run'
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib -- --ignored --nocapture --test-threads=1 quiescence_gpu_async quiescence_gpu_streaks quiescence_gpu_support quiescence_gpu_retained_support quiescence_gpu_region_support quiescence_gpu_world_identity_and_pending_boundaries

.PHONY: test-soil-support-async
test-soil-support-async:
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters $(REL) --lib -- --nocapture quiescence_support_ quiescence_retained_support_ quiescence_region_support_'
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters $(REL) --lib quiescence_gpu_ --no-run'
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib -- --ignored --nocapture --test-threads=1 quiescence_gpu_support quiescence_gpu_retained_support quiescence_gpu_region_support quiescence_gpu_world_identity_and_pending_boundaries

test-soil-identity:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::soil_identity::tests -- --ignored --nocapture --test-threads=1

test-soil-checkpoint-async:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::asynchronous::tests -- --ignored --nocapture --test-threads=1

test-soil-tiles:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::tiles::tests -- --ignored --nocapture --test-threads=1
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::tiles::canonical::tests -- --ignored --nocapture --test-threads=1
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::tiles::framed::tests -- --ignored --nocapture --test-threads=1

test-soil-checkpoint:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::tests -- --ignored --nocapture --test-threads=1
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::recenter::tests -- --ignored --nocapture --test-threads=1
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::checkpoint::frame:: -- --ignored --nocapture --test-threads=1

SOIL_GATHER_FILTER ?= mpm::deterministic_p2g::tests
SOIL_GATHER_ARGS ?= --skip benchmark
.PHONY: test-soil-dense-gather
.PHONY: test-soil-ordered-tractor
SOIL_ORDERED_NEWTON_FEATURE ?= publication-products-newton
test-soil-ordered-tractor:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,$(SOIL_ORDERED_NEWTON_FEATURE) --lib ordered_tractor_full_trajectory_capture --no-run
	@timeout --kill-after=5s 180s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,$(SOIL_ORDERED_NEWTON_FEATURE) --lib ordered_tractor_full_trajectory_capture -- --ignored --nocapture --test-threads=1

test-soil-dense-gather:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib dense_gather_dispatch_and_shader_contract -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib dense_gather_native_lifecycle_and_schedules -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib dense_gather_native_two_dimensional_dispatch -- --ignored --nocapture --test-threads=1

test-soil-gather:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_GATHER_FILTER) -- --ignored --nocapture --test-threads=1 $(SOIL_GATHER_ARGS)

test-soil-command-batch:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib $(SOIL_BATCH_FILTER) -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-fem-feedback test-soil-fem-viewer
SOIL_FEM_FEEDBACK_ARGS ?= --skip benchmark
test-soil-fem-feedback:
	@$(MAKE) test-gpu P=molla-kernels T=soil_fem_feedback_gpu GPU_FEATURES='gpu,resource-counters' A='$(SOIL_FEM_FEEDBACK_ARGS) --nocapture --test-threads=1'

test-soil-fem-viewer:
	@$(RUN_WITH) $(CARGO) test -p molla_viewer --bin molla_viewer soil_retained_fem_bridge -- --ignored --nocapture --test-threads=1

test-soil-feedback:
	@$(MAKE) test-soil-fem-feedback
	@$(MAKE) test-soil-fem-viewer
	@$(MAKE) test-gpu P=molla-kernels T=soil_mpm_feedback_gpu GPU_FEATURES='gpu,resource-counters' A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=mpm_gpu GPU_FEATURES='gpu,resource-counters' A='mpm_retained_feedback --nocapture --test-threads=1'

test-soil-sort:
	@$(MAKE) test-gpu P=molla-compute-primitives T=radix_sort_gpu GPU_FEATURES='gpu,resource-counters' A='--skip benchmark --nocapture --test-threads=1'

test-soil-particle-order:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu,resource-counters --lib mpm::particle_order::tests -- --ignored --skip ordering_matched_inputs_and_trajectory_repeats --nocapture --test-threads=1

.PHONY: test-soil-order-replay
test-soil-order-replay:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib ordering_matched_inputs_and_trajectory_repeats --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib ordering_matched_inputs_and_trajectory_repeats -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-local-transfer
test-soil-local-transfer:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib local_transfer_ --no-run
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib local_transfer_source_contract -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib local_transfer_uniform_grid_affine_moment -- --ignored --nocapture --test-threads=1

.PHONY: report-soil-frontier test-soil-frontier
report-soil-frontier:
	@test -n '$(BASELINE)' -a -n '$(INPUT)' -a -n '$(SOIL_FRONTIER_OUTPUT)' || { echo 'BASELINE and INPUT must name data directories; SOIL_FRONTIER_OUTPUT must be fresh'; exit 2; }
	@$(CARGO) run --release -p molla-solvers --example soil_benchmark -- frontier --reference '$(BASELINE)' --input '$(INPUT)' --output '$(SOIL_FRONTIER_OUTPUT)'
test-soil-frontier:
	@$(CARGO) test --release -p molla-solvers --example soil_benchmark frontier::tests -- --nocapture

check-soil-results:
	@test -n '$(INPUT)' || { echo 'INPUT must name a soil result directory'; exit 2; }
	@$(CARGO) run --release -p molla-solvers --example soil_benchmark -- check --input '$(INPUT)' --output '$(if $(SOIL_CHECK_OUTPUT),$(SOIL_CHECK_OUTPUT),$(INPUT))' $(if $(BASELINE),--baseline '$(BASELINE)',) $(if $(INSTRUMENTATION_BASELINE),--instrumentation-baseline '$(INSTRUMENTATION_BASELINE)',)

test-soil-sparse-map:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu --lib mpm::active_tiles::tests::sparse_hash_collisions_retirement_and_overflow -- --exact --ignored --nocapture --test-threads=1

test-soil-wrench-replay:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features gpu --lib mpm::replay_tests::mpm_identical_grid_wrench_replay -- --exact --ignored --nocapture --test-threads=1

.PHONY: test-soil-newton-root
test-soil-newton-protected:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton --lib newton_ --no-run
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton --lib newton_ -- --nocapture
	@MOLLA_NEWTON_TEST_CONDITIONED=1 timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton --lib resident_newton_fischer_pairs_ -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton --lib newton_native_mpm_publication_and_feedback -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-newton-protected
.PHONY: test-soil-contact-profile
.PHONY: test-soil-newton-history
.PHONY: test-soil-newton-adjacent
.PHONY: test-soil-newton-wide
.PHONY: test-soil-newton-delta-cache
test-soil-newton-delta-cache:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,wide-newton --lib newton_delta_cache_ -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,wide-newton --lib newton_delta_cache_ -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-newton-cached
.PHONY: test-soil-newton-parallel-release
test-soil-newton-parallel-release:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,cached-newton --lib newton_parallel_release_ -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,cached-newton --lib newton_parallel_release_ -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-newton-parallel-core
test-soil-newton-parallel-core:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,cached-newton --lib newton_parallel_core_gpu_edges --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,cached-newton --lib newton_parallel_core_gpu_edges -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-newton-parallel-runtime
.PHONY: test-soil-newton-volatile
test-soil-newton-volatile:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,parallel-newton --lib newton_volatile_ -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,parallel-newton --lib newton_volatile_pairs_gpu -- --ignored --nocapture --test-threads=1

test-soil-newton-parallel-runtime:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,parallel-newton --lib newton_ -- --nocapture
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,parallel-newton --example soil_tractor_benchmark
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,parallel-newton --lib newton_failure_gpu_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,parallel-newton --lib newton_native_mpm_publication_and_feedback -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-newton-tiled-core
test-soil-newton-tiled-core:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,volatile-newton --lib newton_tiled_core_source_contract -- --nocapture
	@MOLLA_NEWTON_TILED_EDGES=1 timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,volatile-newton --lib newton_parallel_core_gpu_edges -- --ignored --nocapture --test-threads=1

.PHONY: inspect-soil-newton-pipeline
.PHONY: test-soil-newton-initialization
.PHONY: test-soil-newton-row-products
.PHONY: test-soil-newton-norm-tree
.PHONY: test-soil-newton-active-publication
.PHONY: test-soil-newton-zero-publication
.PHONY: test-soil-newton-publication-products
.PHONY: test-soil-newton-retained-direct
.PHONY: test-soil-newton-direct-quality
.PHONY: test-soil-newton-retained-direct-quality
test-soil-newton-retained-direct-quality:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_retained_direct_quality -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_retained_direct_quality_gpu_lifecycle -- --ignored --nocapture --test-threads=1

test-soil-newton-direct-quality:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_direct_quality -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_direct_quality_gpu_ratios -- --ignored --nocapture --test-threads=1

test-soil-newton-retained-direct:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_retained_direct -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_retained_direct_gpu_lifecycle -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-newton-publication-validation
test-soil-newton-publication-validation:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_publication_validation -- --nocapture
	@MOLLA_NEWTON_NORM_COMPARE=publication-validation timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_publication_validation -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-newton-publication-adaptive
test-soil-newton-publication-adaptive:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_publication_adaptive -- --nocapture
	@MOLLA_NEWTON_NORM_COMPARE=publication-adaptive timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_publication_adaptive_gpu_lifecycle -- --ignored --nocapture --test-threads=1

test-soil-newton-publication-products:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,active-publication-newton --lib newton_publication_products_preserve_order_and_scratch_bounds -- --nocapture
	@MOLLA_NEWTON_NORM_COMPARE=publication-products timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,active-publication-newton --lib newton_publication_products_gpu_lifecycle -- --ignored --nocapture --test-threads=1

test-soil-newton-zero-publication:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,active-publication-newton --lib newton_zero_publication_preserves_validation_and_first_axis -- --nocapture
	@MOLLA_NEWTON_NORM_COMPARE=publication-zero timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,active-publication-newton --lib newton_zero_publication_gpu_lifecycle -- --ignored --nocapture --test-threads=1

test-soil-newton-active-publication:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,norm-tree-newton --lib newton_active_publication_gpu_lifecycle --no-run
	@MOLLA_NEWTON_NORM_COMPARE=publication-active timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,norm-tree-newton --lib newton_active_publication_gpu_lifecycle -- --ignored --nocapture --test-threads=1

test-soil-newton-norm-tree:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,schur-newton --lib newton_norm_tree_gpu_edges --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,schur-newton --lib newton_norm_tree_gpu_edges -- --ignored --nocapture --test-threads=1

test-soil-newton-row-products:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,schur-newton --lib newton_row_products_gpu_edges --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,schur-newton --lib newton_row_products_gpu_edges -- --ignored --nocapture --test-threads=1

test-soil-newton-initialization:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,schur-newton --lib newton_loop_local_initialization_gpu_diagnostic --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,schur-newton --lib newton_loop_local_initialization_gpu_diagnostic -- --ignored --nocapture --test-threads=1

inspect-soil-newton-pipeline:
	@$(CARGO) build --release -p molla-solvers --features gpu,resource-counters,volatile-newton --example soil_pipeline_statistics
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) run --release -p molla-solvers --features gpu,resource-counters,volatile-newton --example soil_pipeline_statistics -- $(if $(SOIL_NEWTON_SPV),$(SOIL_NEWTON_SPV),--list)

.PHONY: test-soil-newton-volatile-runtime
.PHONY: test-soil-newton-schur-runtime
.PHONY: test-soil-newton-norm-runtime
.PHONY: test-soil-newton-active-publication-runtime
.PHONY: test-soil-newton-publication-products-runtime
.PHONY: test-soil-newton-retained-local-runtime
test-soil-newton-retained-local-runtime:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_ -- --nocapture
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib retained_conditioned_core -- --nocapture
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --example soil_tractor_benchmark
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib retained_conditioned_core_gpu_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_failure_gpu_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,retained-local-newton --lib newton_native_mpm_publication_and_feedback -- --ignored --nocapture --test-threads=1

test-soil-newton-publication-products-runtime:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,publication-products-newton --lib newton_ -- --nocapture
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,publication-products-newton --example soil_tractor_benchmark
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,publication-products-newton --lib newton_failure_gpu_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,publication-products-newton --lib newton_native_mpm_publication_and_feedback -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-newton-publication-shadow
.PHONY: test-soil-newton-publication-products-shadow
test-soil-newton-publication-products-shadow:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,publication-products-newton --lib newton_ -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,publication-products-newton --lib newton_shadow_comparator_gpu_lifecycle -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-dem-refit test-soil-dem-refit-gpu
test-soil-dem-refit:
	@$(CARGO) test --release -p molla-kernels --features gpu,resource-counters --lib --test soil_dem_topology_gpu --test soil_residency_gpu --no-run
	@$(CARGO) test --release -p molla-kernels --features gpu,resource-counters --lib dem_ordered_refit_source_contract -- --nocapture
	@$(CARGO) test --release -p molla-kernels --features gpu,resource-counters --lib dem_grouped_refit_source_and_tree_partition -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(MAKE) test-soil-dem-refit-gpu

test-soil-dem-refit-gpu:
	@set -e; output=$$(mktemp -d "$${SOIL_DEM_REFIT_OUTPUT_PARENT:-$${TMPDIR:-/tmp}}/molla-dem-refit.XXXXXX"); \
		echo "DEM exact-check evidence: $$output"; \
		MOLLA_DEM_GROUPED_OUTPUT="$$output/tree" $(CARGO) test --release -p molla-kernels --features gpu,resource-counters --lib dem_grouped_refit_gpu_tree_and_trajectory -- --ignored --nocapture --test-threads=1; \
		MOLLA_DEM_ORDERED_OUTPUT="$$output/trajectory" $(CARGO) test --release -p molla-kernels --features gpu,resource-counters --lib dem_ordered_refit_exact_trajectory_and_overflow -- --ignored --nocapture --test-threads=1
	@$(CARGO) test --release -p molla-kernels --features gpu,resource-counters --test soil_dem_topology_gpu -- --ignored --nocapture --test-threads=1
	@$(CARGO) test --release -p molla-kernels --features gpu,resource-counters --test soil_residency_gpu -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-native-fast-compensation
test-soil-native-fast-compensation:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib compensated_source_keeps_native_scatter_and_explicit_scope -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib native_fast_residual_matches_independent_f64_reference -- --ignored --nocapture --test-threads=1

test-soil-newton-publication-shadow:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,active-publication-newton --lib newton_ -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,active-publication-newton --lib newton_shadow_comparator_gpu_lifecycle -- --ignored --nocapture --test-threads=1

test-soil-newton-active-publication-runtime:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,active-publication-newton --lib newton_ -- --nocapture
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,active-publication-newton --example soil_tractor_benchmark
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,active-publication-newton --lib newton_failure_gpu_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,active-publication-newton --lib newton_native_mpm_publication_and_feedback -- --ignored --nocapture --test-threads=1

test-soil-newton-norm-runtime:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,norm-tree-newton --lib newton_ -- --nocapture
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,norm-tree-newton --example soil_tractor_benchmark
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,norm-tree-newton --lib newton_norm_tree_gpu_edges -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,norm-tree-newton --lib newton_failure_gpu_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,norm-tree-newton --lib newton_native_mpm_publication_and_feedback -- --ignored --nocapture --test-threads=1

test-soil-newton-schur-runtime:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,schur-newton --lib newton_ -- --nocapture
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,schur-newton --example soil_tractor_benchmark
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,schur-newton --lib newton_schur_inverse_gpu_edges -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,schur-newton --lib newton_failure_gpu_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,schur-newton --lib newton_native_mpm_publication_and_feedback -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-newton-schur-inverse-edges
test-soil-newton-schur-inverse-edges:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,volatile-newton --lib newton_schur_inverse --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,volatile-newton --lib newton_schur_inverse_gpu_edges -- --ignored --nocapture --test-threads=1

test-soil-newton-volatile-runtime:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,volatile-newton --lib newton_ -- --nocapture
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,volatile-newton --example soil_tractor_benchmark
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,volatile-newton --lib newton_parallel_core_gpu_edges -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,volatile-newton --lib newton_failure_gpu_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,volatile-newton --lib newton_native_mpm_publication_and_feedback -- --ignored --nocapture --test-threads=1

test-soil-newton-cached:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,cached-newton --lib newton_ -- --nocapture
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,cached-newton --example soil_tractor_benchmark
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,cached-newton --lib newton_failure_gpu_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,cached-newton --lib newton_native_mpm_publication_and_feedback -- --ignored --nocapture --test-threads=1

test-soil-newton-wide:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,wide-newton --lib newton_ -- --nocapture
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,wide-newton --example soil_tractor_benchmark
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,wide-newton --lib newton_failure_gpu_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,wide-newton --lib newton_native_mpm_publication_and_feedback -- --ignored --nocapture --test-threads=1

test-soil-newton-adjacent:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton,adjacent-newton --lib newton_ -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton,adjacent-newton --lib newton_history_gpu_ -- --ignored --nocapture --test-threads=1

test-soil-newton-history:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton,warm-newton --lib newton_ -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton,warm-newton --lib newton_history_gpu_keys_faults_and_reset -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton,warm-newton --lib newton_failure_gpu_lifecycle -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton,warm-newton --lib newton_native_mpm_publication_and_feedback -- --ignored --nocapture --test-threads=1

test-soil-contact-profile:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton --lib newton_contact_profile_contract -- --nocapture
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton --lib newton_native_mpm_profile_equivalence -- --ignored --nocapture --test-threads=1
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton --lib frame_profile_window -- --ignored --nocapture --test-threads=1

test-soil-newton-publication:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton --lib newton_publication --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters,protected-newton --lib newton_publication -- --ignored --nocapture --test-threads=1 --skip captured_input --skip capture_timing --skip trace_oracle

.PHONY: test-soil-newton-publication
test-soil-newton-root:
	@$(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib fischer_pairs_gpu_root_product --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features gpu,resource-counters --lib fischer_pairs_gpu_root_product -- --ignored --nocapture --test-threads=1

test-soil-fast:
	@case "$(PHASE)" in \
	  S00) $(MAKE) test APP_TARGET='-p molla-solvers --test soil_benchmark_contract -- --nocapture' && $(MAKE) test-soil-frontier ;; \
	  S01) $(MAKE) test APP_TARGET='-p molla-kernels --features gpu --test soil_material_gpu -- --nocapture' && $(MAKE) test APP_TARGET='-p molla_viewer --bin molla_viewer tire_testbench_soil_selection_tests -- --nocapture' ;; \
	  S02) $(MAKE) test APP_TARGET='-p molla-kernels --features gpu --lib soil_mpm_raster::tests -- --nocapture' ;; \
	  S03) $(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --test soil_mpm_observation_gpu --release -- --nocapture' && $(MAKE) test APP_TARGET='-p molla_viewer --bin molla_viewer tire_testbench_soil_selection_tests -- --nocapture' ;; \
	  S06) $(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters --lib mpm::fused_gather::tests --release -- --nocapture' ;; \
	  S07) $(MAKE) test APP_TARGET='-p molla-vehicle $(REL) --lib soil_state:: -- --nocapture' && $(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters $(REL) --lib mpm::checkpoint:: -- --nocapture' && $(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters $(REL) --lib mpm::soil_identity::tests -- --nocapture' ;; \
	  *) echo 'Implemented fast lanes: PHASE=S00 through S03, S06, S07 (focused increments, not full-plan acceptance)'; exit 2 ;; \
	esac

test-soil-gpu: $(if $(filter S03,$(PHASE)),test-soil-frame-loads-gpu test-soil-geometry-program-cache-gpu)
	@case "$(PHASE)" in \
	  S01) $(MAKE) test-gpu P=molla-kernels T=soil_material_gpu GPU_FEATURES='gpu,resource-counters' A='--nocapture --test-threads=1' ;; \
	  S02) $(MAKE) test-gpu P=molla-kernels T=soil_raster_gpu A='--skip raster_benchmark --nocapture --test-threads=1' ;; \
	  S03) $(MAKE) test-soil-particle-render && $(MAKE) test-soil-command-batch && $(MAKE) test-soil-feedback && $(MAKE) test-gpu P=molla-kernels T=soil_residency_gpu GPU_FEATURES='gpu,resource-counters' A='--skip benchmark --nocapture --test-threads=1' && $(MAKE) test-gpu P=molla-kernels T=soil_mpm_snapshot_gpu GPU_FEATURES='gpu,resource-counters' A='--nocapture --test-threads=1' && $(MAKE) test-gpu P=molla-kernels T=soil_mpm_observation_gpu GPU_FEATURES='gpu,resource-counters' A='--nocapture --test-threads=1' && $(MAKE) test-gpu P=molla-solvers T=mpm_gpu GPU_FEATURES='gpu,resource-counters' A='mpm_terrain_only_bridge --skip benchmark --nocapture --test-threads=1' ;; \
	  S04) $(MAKE) test-soil-dem-refit ;; \
	  S05) $(MAKE) test-gpu P=molla-solvers T=mpm_active_tiles_gpu GPU_FEATURES='gpu,resource-counters' A='--skip benchmark --skip mpm_profile --skip mpm_transfer --nocapture --test-threads=1' && $(MAKE) test-gpu P=molla-solvers T=mpm_sparse_grid_gpu GPU_FEATURES='gpu,resource-counters' A='--nocapture --test-threads=1' && $(MAKE) test-soil-sparse-map && $(MAKE) test-soil-particle-order && $(MAKE) test-soil-native-active ;; \
	  S06) $(MAKE) test-gpu P=molla-solvers T=mpm_active_tiles_gpu GPU_FEATURES='gpu,resource-counters' A='mpm_profile --skip benchmark --nocapture --test-threads=1' && $(MAKE) test-gpu P=molla-solvers T=mpm_active_tiles_gpu GPU_FEATURES='gpu,resource-counters' A='mpm_transfer --nocapture --test-threads=1' && $(MAKE) test-soil-sort && $(MAKE) test-soil-gather && $(MAKE) test-soil-fusion ;; \
	  S07) $(MAKE) test-soil-checkpoint && $(MAKE) test-soil-tiles && $(MAKE) test-soil-checkpoint-async && $(MAKE) test-soil-identity && $(MAKE) test-soil-world && $(MAKE) test-soil-disk-world && $(MAKE) test-soil-window && $(MAKE) test-soil-window-integer && $(MAKE) test-soil-transfer && $(MAKE) test-soil-pins && $(MAKE) test-soil-regions && $(MAKE) test-soil-motion-guard && $(MAKE) test-soil-quiescence && $(MAKE) test-soil-settling && $(MAKE) test-soil-numerical-fault && $(MAKE) test-soil-store-gpu && $(MAKE) test-soil-store-incremental ;; \
	  *) echo 'Implemented GPU lanes: PHASE=S01 through S07 (focused increments, not full-plan acceptance)'; exit 2 ;; \
	esac

SOIL_RUN_DIR ?= benchmarks/soil/$(shell date -u +%Y%m%dT%H%M%S)
SOIL_BENCH_SECONDS ?= 30
SOIL_TRACTOR_BENCH_SECONDS ?= 300
SOIL_TRACTOR_BENCH_TIMELINE ?= 1
SOIL_TRACTOR_BENCH_FRAMES ?= 30
SOIL_TRACTOR_BENCH_WARMUP ?= 3
SOIL_TRACTOR_BENCH_PROFILE ?= 0
SOIL_TRACTOR_BENCH_PREPARATION_PROFILE ?= 0
SOIL_TRACTOR_BENCH_LOADS ?= 0
SOIL_TRACTOR_BENCH_CONTACT_MOTION ?= 0
SOIL_TRACTOR_BENCH_COUPLING_NODES ?= 0
SOIL_TRACTOR_BENCH_UNILATERAL_NODES ?= 0
SOIL_TRACTOR_BENCH_UNILATERAL_MARGIN ?= 0.05
SOIL_TRACTOR_BENCH_UNILATERAL_ITERATIONS ?= 64
SOIL_TRACTOR_BENCH_FRICTION_MU ?=
SOIL_TRACTOR_BENCH_FRICTION_WALLS ?=
SOIL_TRACTOR_FRICTION_ARGS = $(if $(strip $(SOIL_TRACTOR_BENCH_FRICTION_MU)),--friction-mu '$(SOIL_TRACTOR_BENCH_FRICTION_MU)') $(if $(strip $(SOIL_TRACTOR_BENCH_FRICTION_WALLS)),--friction-walls '$(SOIL_TRACTOR_BENCH_FRICTION_WALLS)') --newton-soil '$(SOIL_TRACTOR_BENCH_NEWTON)'
SOIL_TRACTOR_UNILATERAL_ARGS = $(if $(filter-out 0,$(SOIL_TRACTOR_BENCH_UNILATERAL_NODES)),--unilateral-nodes '$(SOIL_TRACTOR_BENCH_UNILATERAL_NODES)' --unilateral-margin '$(SOIL_TRACTOR_BENCH_UNILATERAL_MARGIN)' --unilateral-iterations '$(SOIL_TRACTOR_BENCH_UNILATERAL_ITERATIONS)')
SOIL_TRACTOR_BENCH_ROUNDING ?= storage
SOIL_TRACTOR_BENCH_BOUNDARY_SCAN ?= full
SOIL_TRACTOR_BENCH_CAPTURE_EVERY ?= 0
SOIL_TRACTOR_BENCH_EXECUTION ?= generic
SOIL_TRACTOR_BENCH_GRID ?= dense
SOIL_TRACTOR_BENCH_TRANSFER ?= atomic
SOIL_TRACTOR_BENCH_FINAL_STATE ?= 0
SOIL_TRACTOR_BENCH_MOTION_BOUND ?= off
SOIL_TRACTOR_BENCH_MOTION_FUSION ?= 0
SOIL_TRACTOR_FIXED_ORDER_FEATURES ?= gpu,resource-counters,native-lazy-geometry-drift,parallel-indices-newton,strict-rounding-newton-preparation,parallel-candidate-index,deferred-native-drift
.PHONY: test-soil-tractor-fixed-order
test-soil-tractor-fixed-order:
	@$(CARGO) test --release -p molla-solvers --features $(SOIL_TRACTOR_FIXED_ORDER_FEATURES) --test soil_tractor_fixed_order --no-run
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test --release -p molla-solvers --features $(SOIL_TRACTOR_FIXED_ORDER_FEATURES) --test soil_tractor_fixed_order tractor_fixed_order_public_constructor_and_repeatability -- --ignored --nocapture --test-threads=1
SOIL_TRACTOR_BENCH_PAIRED_RATES ?= 0
SOIL_TRACTOR_BENCH_NATIVE_PREDICTION ?= 0
SOIL_TRACTOR_BENCH_NEWTON ?= 0
SOIL_REPLAY_RUN ?=
SOIL_REPLAY_OUTPUT ?=
SOIL_REPLAY_FRAME ?= final
.PHONY: render-soil-tractor test-soil-replay
SOIL_LIVE_OUTPUT ?=
SOIL_LIVE_FRAMES ?= 3
SOIL_LIVE_PARTICLES ?= 0
SOIL_LIVE_HEADLESS ?= 0
SOIL_LIVE_PROFILE ?= volatile
SOIL_LIVE_TRANSFER ?= atomic
SOIL_LIVE_PREDICTION ?= featherstone
SOIL_LIVE_PREPARATION ?= default
SOIL_LIVE_SUBMISSIONS ?= separate
SOIL_LIVE_STAGE_PROFILE ?= 0
SOIL_LIVE_DRIFT ?= full
SOIL_LIVE_GEOMETRY ?= eager
SOIL_LIVE_COMMA := ,
SOIL_LIVE_FEATURES = $(if $(filter soa,$(SOIL_LIVE_PROFILE)),soil-live-soa,soil-live)$(if $(filter cooperative,$(SOIL_LIVE_PREPARATION)),$(SOIL_LIVE_COMMA)soil-live-row-preparation)$(if $(filter composed,$(SOIL_LIVE_SUBMISSIONS)),$(SOIL_LIVE_COMMA)soil-live-batched-projection)$(if $(filter deferred,$(SOIL_LIVE_DRIFT)),$(SOIL_LIVE_COMMA)soil-live-deferred-drift)$(if $(filter native-lazy,$(SOIL_LIVE_GEOMETRY)),$(SOIL_LIVE_COMMA)soil-live-native-lazy-drift)
.PHONY: view-soil-tractor test-soil-live-viewer capture-soil-tractor-live
.PHONY: test-soil-surface test-soil-surface-gpu
.PHONY: test-soil-terrain-queries test-soil-terrain-queries-gpu
.PHONY: test-soil-brush-terrain test-soil-brush-terrain-gpu
.PHONY: test-soil-prepared-chassis test-soil-prepared-chassis-gpu
.PHONY: test-soil-prepared-fem test-soil-prepared-fem-gpu
test-soil-prepared-fem:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --lib prepared_vehicle_fem_byte_contract -- --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --test tire_fem_gpu --no-run'

test-soil-prepared-fem-gpu: test-soil-prepared-fem
	@$(RUN_WITH) timeout --kill-after=5s 90s $(CARGO) test -p molla-kernels --features gpu,resource-counters --release --test tire_fem_gpu tire_fem_resident_vehicle_fem_atlas_owner_steps_gpu -- --ignored --nocapture --test-threads=1

test-soil-prepared-chassis:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --lib prepared_chassis_step_budget_and_uniform_contract -- --nocapture'
	@$(MAKE) test-soil-brush-terrain

test-soil-prepared-chassis-gpu: test-soil-prepared-chassis
	@$(RUN_WITH) timeout --kill-after=5s 90s $(CARGO) test -p molla-kernels --features gpu,resource-counters --release --test tire_vehicle_gpu resident_terrain_ -- --ignored --nocapture --test-threads=1

test-soil-brush-terrain:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --lib prepared_brush_terrain_buffer_contract -- --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --test tire_vehicle_gpu --no-run'

test-soil-brush-terrain-gpu: test-soil-brush-terrain
	@$(RUN_WITH) timeout --kill-after=5s 90s $(CARGO) test -p molla-kernels --features gpu,resource-counters --release --test tire_vehicle_gpu resident_terrain_ -- --ignored --nocapture --test-threads=1

test-soil-terrain-queries:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --lib soil_terrain_query_layout_and_source -- --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --test soil_terrain_queries_gpu --no-run'

test-soil-terrain-queries-gpu: test-soil-terrain-queries
	@$(RUN_WITH) timeout --kill-after=5s 90s $(CARGO) test -p molla-kernels --features gpu,resource-counters --release --test soil_terrain_queries_gpu -- --ignored --nocapture --test-threads=1

test-soil-surface:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --lib soil_surface -- --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --test soil_surface_gpu --no-run'
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters --release --test mpm_surface_gpu --no-run'

test-soil-surface-gpu: test-soil-surface
	@$(RUN_WITH) timeout --kill-after=5s 90s $(CARGO) test -p molla-kernels --features gpu,resource-counters --release --test soil_surface_gpu -- --ignored --nocapture --test-threads=1
	@$(RUN_WITH) timeout --kill-after=5s 90s $(CARGO) test -p molla-solvers --features gpu,resource-counters --release --test mpm_surface_gpu -- --ignored --nocapture --test-threads=1

.PHONY: check-soil-live-profile
check-soil-live-profile:
	@case '$(SOIL_LIVE_PROFILE)' in volatile|soa) ;; *) echo 'SOIL_LIVE_PROFILE must be volatile or soa'; exit 2 ;; esac
	@case '$(SOIL_LIVE_TRANSFER)' in atomic|fixed-order|grid-reuse) ;; *) echo 'SOIL_LIVE_TRANSFER must be atomic, fixed-order or grid-reuse'; exit 2 ;; esac
	@case '$(SOIL_LIVE_PREDICTION)' in featherstone|native) ;; *) echo 'SOIL_LIVE_PREDICTION must be featherstone or native'; exit 2 ;; esac
	@case '$(SOIL_LIVE_PREPARATION)' in default|cooperative) ;; *) echo 'SOIL_LIVE_PREPARATION must be default or cooperative'; exit 2 ;; esac
	@case '$(SOIL_LIVE_SUBMISSIONS)' in separate|composed) ;; *) echo 'SOIL_LIVE_SUBMISSIONS must be separate or composed'; exit 2 ;; esac
	@case '$(SOIL_LIVE_STAGE_PROFILE)' in 0|1) ;; *) echo 'SOIL_LIVE_STAGE_PROFILE must be 0 or 1'; exit 2 ;; esac
	@case '$(SOIL_LIVE_DRIFT)' in full|deferred) ;; *) echo 'SOIL_LIVE_DRIFT must be full or deferred'; exit 2 ;; esac
	@case '$(SOIL_LIVE_GEOMETRY)' in eager|native-lazy) ;; *) echo 'SOIL_LIVE_GEOMETRY must be eager or native-lazy'; exit 2 ;; esac
	@test '$(SOIL_LIVE_DRIFT)' != deferred -o '$(SOIL_LIVE_PREDICTION)' = native || { echo 'Deferred drift requires SOIL_LIVE_PREDICTION=native'; exit 2; }

test-soil-live-viewer: check-soil-live-profile
	@$(MAKE) test APP_TARGET='-p molla_viewer --features $(SOIL_LIVE_FEATURES) --bin soil_tractor_viewer --release -- --nocapture'

.PHONY: test-soil-frame-loads-gpu
.PHONY: test-soil-machine-profile test-soil-machine-profile-gpu
test-soil-machine-profile:
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters --release --lib rigid_world::gpu::soil::profile::tests -- --nocapture'

test-soil-machine-profile-gpu:
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters --release --lib --no-run'
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu,resource-counters --release --lib rigid_world::gpu::soil::profile::tests::machine_profile_retains_slots_and_reuses_without_allocations -- --exact --ignored --nocapture --test-threads=1

.PHONY: test-soil-geometry-program-cache-gpu
test-soil-geometry-program-cache-gpu:
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters --release --lib geometry_program_cache_lifecycle --no-run'
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu,resource-counters --release --lib geometry_program_cache_lifecycle -- --ignored --nocapture --test-threads=1

test-soil-frame-loads-gpu:
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters --release --lib rigid_world::gpu::soil::loads:: --no-run'
	@timeout --kill-after=5s 90s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu,resource-counters --release --lib rigid_world::gpu::soil::loads:: -- --ignored --nocapture --test-threads=1

.PHONY: test-soil-live-idle-surface-gpu
test-soil-live-idle-surface-gpu: check-soil-live-profile
	@$(MAKE) test APP_TARGET='-p molla_viewer --features $(SOIL_LIVE_FEATURES) --bin soil_tractor_viewer --release --no-run'
	@timeout --kill-after=5s 180s $(RUN_WITH) $(CARGO) test -p molla_viewer --features $(SOIL_LIVE_FEATURES) --bin soil_tractor_viewer --release live::tests::idle_surface_worker_gpu -- --ignored --exact --nocapture --test-threads=1

view-soil-tractor: check-soil-live-profile
	@test '$(SOIL_LIVE_STAGE_PROFILE)' = 0 || { echo 'Stage profiling requires capture-soil-tractor-live'; exit 2; }
	@[[ '$(SOIL_LIVE_PARTICLES)' == 0 || '$(SOIL_LIVE_PARTICLES)' == 1 ]] || { echo 'SOIL_LIVE_PARTICLES must be 0 or 1'; exit 2; }
	@$(MAKE) run APP_TARGET='-p molla_viewer --features $(SOIL_LIVE_FEATURES) --bin soil_tractor_viewer --release -- --expect-profile $(SOIL_LIVE_PROFILE) --expect-preparation $(SOIL_LIVE_PREPARATION) --expect-submissions $(SOIL_LIVE_SUBMISSIONS) --expect-geometry $(SOIL_LIVE_GEOMETRY) --expect-drift $(SOIL_LIVE_DRIFT) --soil-transfer $(SOIL_LIVE_TRANSFER) --machine-prediction $(SOIL_LIVE_PREDICTION) $(if $(filter 1,$(SOIL_LIVE_PARTICLES)),--particles)'

capture-soil-tractor-live: check-soil-live-profile
	@[[ '$(SOIL_LIVE_PARTICLES)' == 0 || '$(SOIL_LIVE_PARTICLES)' == 1 ]] || { echo 'SOIL_LIVE_PARTICLES must be 0 or 1'; exit 2; }
	@[[ '$(SOIL_LIVE_HEADLESS)' == 0 || '$(SOIL_LIVE_HEADLESS)' == 1 ]] || { echo 'SOIL_LIVE_HEADLESS must be 0 or 1'; exit 2; }
	@test -n '$(SOIL_LIVE_OUTPUT)' -a ! -e '$(SOIL_LIVE_OUTPUT)' || { echo 'Set a fresh SOIL_LIVE_OUTPUT directory'; exit 2; }
	@[[ '$(SOIL_LIVE_FRAMES)' =~ ^[1-9][0-9]*$$ && '$(SOIL_LIVE_FRAMES)' -le 330 ]] || { echo 'Live capture frames must be 1..330'; exit 2; }
	@$(MAKE) build APP_TARGET='-p molla_viewer --features $(SOIL_LIVE_FEATURES) --bin soil_tractor_viewer --release'
	@mkdir -p '$(SOIL_LIVE_OUTPUT)'
	@git rev-parse HEAD > '$(SOIL_LIVE_OUTPUT)/revision.txt'
	@git status --short > '$(SOIL_LIVE_OUTPUT)/worktree.txt'
	@{ printf '%s\0' Makefile Cargo.toml Cargo.lock scripts/stop-soil-telemetry.sh; find crates bin/molla_viewer -type f \( -name '*.rs' -o -name '*.wgsl' -o -name 'Cargo.toml' \) -print0; } | sort -z | xargs -0 sha256sum > '$(SOIL_LIVE_OUTPUT)/source-sha256.txt'
	@tar -czf '$(SOIL_LIVE_OUTPUT)/source.tar.gz' --null -T <({ printf '%s\0' Makefile Cargo.toml Cargo.lock scripts/stop-soil-telemetry.sh; find crates bin/molla_viewer -type f \( -name '*.rs' -o -name '*.wgsl' -o -name 'Cargo.toml' \) -print0; } | sort -z)
	@sha256sum target/release/soil_tractor_viewer > '$(SOIL_LIVE_OUTPUT)/executable-sha256.txt'
	@set -o pipefail; \
	  $(NVIDIA_SMI) --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv -lms 500 > '$(SOIL_LIVE_OUTPUT)/hardware-timeline.csv' 2>&1 & monitor=$$!; \
	  trap 'bash scripts/stop-soil-telemetry.sh "$(SOIL_LIVE_OUTPUT)/hardware-timeline.csv"; kill "$$monitor" 2>/dev/null || true; wait "$$monitor" 2>/dev/null || true' EXIT; \
	  timeout --kill-after=5s 180s $(RUN_WITH) target/release/soil_tractor_viewer --expect-profile '$(SOIL_LIVE_PROFILE)' --expect-preparation '$(SOIL_LIVE_PREPARATION)' --expect-submissions '$(SOIL_LIVE_SUBMISSIONS)' --expect-geometry '$(SOIL_LIVE_GEOMETRY)' --expect-drift '$(SOIL_LIVE_DRIFT)' --soil-transfer '$(SOIL_LIVE_TRANSFER)' --machine-prediction '$(SOIL_LIVE_PREDICTION)' --capture-dir '$(SOIL_LIVE_OUTPUT)/capture' --frames '$(SOIL_LIVE_FRAMES)' $(if $(filter 1,$(SOIL_LIVE_PARTICLES)),--particles) $(if $(filter 1,$(SOIL_LIVE_HEADLESS)),--headless) $(if $(filter 1,$(SOIL_LIVE_STAGE_PROFILE)),--profile-stages) > '$(SOIL_LIVE_OUTPUT)/run.log' 2>&1; result=$$?; \
	  echo $$result > '$(SOIL_LIVE_OUTPUT)/run.exit'; \
	  sha256sum -c '$(SOIL_LIVE_OUTPUT)/source-sha256.txt' > '$(SOIL_LIVE_OUTPUT)/source-check.log'; source_result=$$?; echo $$source_result > '$(SOIL_LIVE_OUTPUT)/source-check.exit'; \
	  if [[ $$result == 0 ]]; then result=$$source_result; fi; exit $$result
	@test -s '$(SOIL_LIVE_OUTPUT)/capture/committed-frame.png'
	@grep -q '"event":"capture_complete"' '$(SOIL_LIVE_OUTPUT)/capture/frames.jsonl'

test-soil-replay:
	@$(MAKE) test APP_TARGET='-p molla_viewer --example soil_tractor_replay --release -- --nocapture'

render-soil-tractor:
	@test -n '$(SOIL_REPLAY_RUN)' -a -n '$(SOIL_REPLAY_OUTPUT)' || { echo 'Set SOIL_REPLAY_RUN and a new SOIL_REPLAY_OUTPUT directory'; exit 2; }
	@test ! -e '$(SOIL_REPLAY_OUTPUT)' || { echo 'Refusing to overwrite replay output'; exit 2; }
	@$(MAKE) build APP_TARGET='-p molla_viewer --example soil_tractor_replay --release'
	@timeout --kill-after=5s 120s $(RUN_WITH) target/release/examples/soil_tractor_replay '$(SOIL_REPLAY_RUN)' '$(SOIL_REPLAY_OUTPUT)' '$(SOIL_REPLAY_FRAME)'
	@test -s '$(SOIL_REPLAY_OUTPUT)/snapshot.png'

.PHONY: bench-soil-tractor
.PHONY: test-soil-machine-lazy-anchors
test-soil-machine-lazy-anchors:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib lazy_body_anchors_source_contract -- --nocapture'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu --release --lib lazy_body_anchors_match_exact_state_and_f64_kinematics -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-machine-lazy-anchors-cost
test-soil-machine-lazy-anchors-cost:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib lazy_body_anchors --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu --release --lib lazy_body_anchors_paired_tractor_drift_cost -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-lazy-drift-runtime
test-soil-lazy-drift-runtime:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features lazy-paired-drift --release --lib lazy_body_anchors -- --nocapture'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features lazy-paired-drift --release --lib paired_lazy_drift_public_runtime_preserves_rates_and_guards -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-paired-native-sum
test-soil-paired-native-sum:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib paired_native_sum --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu --release --lib paired_native_sum_matches_exact_state_and_f64_kinematics -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-paired-native-sum-cost
test-soil-paired-native-sum-cost:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib paired_native_sum_tractor_drift_cost --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu --release --lib paired_native_sum_tractor_drift_cost -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-paired-native-position-cost
test-soil-paired-native-position-cost:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib paired_native_position_sum_tractor_drift_cost --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu --release --lib paired_native_position_sum_tractor_drift_cost -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-paired-native-geometry-cost
test-soil-paired-native-geometry-cost:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib paired_native_geometry_sum_tractor_drift_cost --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu --release --lib paired_native_geometry_sum_tractor_drift_cost -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-native-geometry-oracle
test-soil-native-geometry-oracle:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib paired_native_geometry_matches_exact_state_and_f64_kinematics --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu --release --lib paired_native_geometry_matches_exact_state_and_f64_kinematics -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-native-geometry-runtime
test-soil-native-geometry-runtime:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features native-geometry-drift --release --lib native_geometry -- --nocapture'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features native-geometry-drift --release --lib paired_native_geometry_rejects_missing_device_feature -- --ignored --nocapture --test-threads=1'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features native-geometry-drift --release --lib paired_native_geometry_public_runtime_preserves_rates_and_guards -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-native-geometry-acceleration
test-soil-native-geometry-acceleration:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib paired_native_geometry_tractor_acceleration_cost --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu --release --lib paired_native_geometry_tractor_acceleration_cost -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-rigorous-native-geometry-acceleration
test-soil-rigorous-native-geometry-acceleration:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib paired_rigorous_native_geometry_tractor_acceleration_cost --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu --release --lib paired_rigorous_native_geometry_tractor_acceleration_cost -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-switched-native-geometry-acceleration
test-soil-switched-native-geometry-acceleration:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib paired_switched_native_geometry_tractor_acceleration_cost --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu --release --lib paired_switched_native_geometry_tractor_acceleration_cost -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-acceleration-oracle
test-soil-acceleration-oracle:
	@test -n "$$MOLLA_ACCELERATION_ORACLE_CAPTURE" || { echo 'Set absolute frozen capture directory'; exit 2; }
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib acceleration_mass_oracle_solve_contract -- --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib acceleration_capture_obeys_f64_mass_response -- --ignored --nocapture'

.PHONY: test-soil-acceleration-balance
test-soil-acceleration-balance:
	@test -n "$$MOLLA_ACCELERATION_ORACLE_CAPTURE" || { echo 'Set absolute frozen capture directory'; exit 2; }
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib acceleration_balance_oracle_contract -- --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib acceleration_capture_obeys_absolute_f64_balance -- --ignored --nocapture'

.PHONY: test-soil-acceleration-operator
test-soil-acceleration-operator:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib paired_zero_bias_native_geometry_acceleration_operator --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu --release --lib paired_zero_bias_native_geometry_acceleration_operator -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-acceleration-operator-oracle
test-soil-acceleration-operator-oracle:
	@test -n "$$MOLLA_ACCELERATION_ORACLE_CAPTURE" || { echo 'Set absolute frozen capture directory'; exit 2; }
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --release --lib acceleration_capture_obeys_f64_full_operator -- --ignored --nocapture'

.PHONY: test-soil-mass-predictor
test-soil-mass-predictor:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --lib retained_mass_predictor_gpu_contract --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --lib retained_mass_predictor_gpu_contract -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-bias-chain
test-soil-bias-chain:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --lib retained_bias_mass_chain_gpu_contract --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --lib retained_bias_mass_chain_gpu_contract -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-precise-operator-chain
test-soil-precise-operator-chain:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --lib retained_precise_operator_chain_gpu_contract --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-kernels --features gpu,resource-counters --release --lib retained_precise_operator_chain_gpu_contract -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-generalized-prediction
test-soil-generalized-prediction:
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters --release --lib generalized_prediction_gpu_physical_contract --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-solvers --features gpu,resource-counters --release --lib generalized_prediction_gpu_physical_contract -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-native-trajectory-replay
test-soil-native-trajectory-replay:
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters --release --lib native_prediction_actual_trajectory_replay --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-solvers --features gpu,resource-counters --release --lib native_prediction_actual_trajectory_replay -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-native-prediction-drift
test-soil-native-prediction-drift:
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters,native-geometry-drift --release --lib native_prediction_drift_gpu_contract --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-solvers --features gpu,resource-counters,native-geometry-drift --release --lib native_prediction_drift_gpu_contract -- --ignored --nocapture --test-threads=1'

.PHONY: test-soil-native-world
test-soil-native-world:
	@$(MAKE) test APP_TARGET='-p molla-solvers --features gpu,resource-counters,native-geometry-drift,retained-local-newton,zero-residual-newton-preparation --release --lib native_prediction_world_shares_mass_and_rolls_back --no-run'
	@$(MAKE) test CARGO='timeout --kill-after=5s 90s $(RUN_WITH) cargo' APP_TARGET='-p molla-solvers --features gpu,resource-counters,native-geometry-drift,retained-local-newton,zero-residual-newton-preparation --release --lib native_prediction_world_shares_mass_and_rolls_back -- --ignored --nocapture --test-threads=1'

SOIL_TRACTOR_BENCH_FEATURES ?= resource-counters
bench-soil-tractor:
	@test ! -e '$(SOIL_RUN_DIR)' || { echo 'Refusing to reuse an existing benchmark result directory'; exit 2; }
	@test '$(SOIL_TRACTOR_BENCH_TIMELINE)' = 0 -o '$(SOIL_TRACTOR_BENCH_TIMELINE)' = 1 || { echo 'SOIL_TRACTOR_BENCH_TIMELINE must be 0 or 1'; exit 2; }
	@[[ '$(SOIL_TRACTOR_BENCH_SECONDS)' =~ ^[1-9][0-9]*$$ && '$(SOIL_TRACTOR_BENCH_SECONDS)' -le 600 ]] || { echo 'Tractor diagnostic deadline must be 1 through 600 seconds'; exit 2; }
	@$(MAKE) build APP_TARGET='-p molla-solvers --features $(SOIL_TRACTOR_BENCH_FEATURES) --example soil_tractor_benchmark --release'
	@mkdir -p '$(SOIL_RUN_DIR)'
	@git rev-parse HEAD > '$(SOIL_RUN_DIR)/revision.txt'
	@git status --short > '$(SOIL_RUN_DIR)/worktree.txt'
	@{ printf '%s\0' Makefile Cargo.toml Cargo.lock scripts/stop-soil-telemetry.sh; find crates -type f \( -name '*.rs' -o -name '*.wgsl' -o -name 'Cargo.toml' \) -print0; } | sort -z | xargs -0 sha256sum > '$(SOIL_RUN_DIR)/source-sha256.txt'
	@tar -czf '$(SOIL_RUN_DIR)/source.tar.gz' --null -T <({ printf '%s\0' Makefile Cargo.toml Cargo.lock scripts/stop-soil-telemetry.sh; find crates -type f \( -name '*.rs' -o -name '*.wgsl' -o -name 'Cargo.toml' \) -print0; } | sort -z)
	@sha256sum target/release/examples/soil_tractor_benchmark > '$(SOIL_RUN_DIR)/executable-sha256.txt'
	@$(NVIDIA_SMI) --query-gpu=name,driver_version,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv > '$(SOIL_RUN_DIR)/hardware-before.csv'
	@set -o pipefail; \
	  monitor=; printf '%s\n' 'timeline=$(SOIL_TRACTOR_BENCH_TIMELINE)' > '$(SOIL_RUN_DIR)/telemetry-mode.txt'; \
	  if [[ '$(SOIL_TRACTOR_BENCH_TIMELINE)' == 1 ]]; then $(NVIDIA_SMI) --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv -lms 500 > '$(SOIL_RUN_DIR)/hardware-timeline.csv' 2>&1 & monitor=$$!; fi; \
	  trap 'if [[ -n "$$monitor" ]]; then bash scripts/stop-soil-telemetry.sh "$(SOIL_RUN_DIR)/hardware-timeline.csv"; kill "$$monitor" 2>/dev/null || true; wait "$$monitor" 2>/dev/null || true; fi' EXIT; \
	  timeout --kill-after=5s $(SOIL_TRACTOR_BENCH_SECONDS)s $(RUN_WITH) target/release/examples/soil_tractor_benchmark --output '$(SOIL_RUN_DIR)/data' --frames '$(SOIL_TRACTOR_BENCH_FRAMES)' --warmup '$(SOIL_TRACTOR_BENCH_WARMUP)' --profile-mpm '$(SOIL_TRACTOR_BENCH_PROFILE)' --profile-preparation '$(SOIL_TRACTOR_BENCH_PREPARATION_PROFILE)' --frame-loads '$(SOIL_TRACTOR_BENCH_LOADS)' --contact-motion '$(SOIL_TRACTOR_BENCH_CONTACT_MOTION)' --mass-coupling-nodes '$(SOIL_TRACTOR_BENCH_COUPLING_NODES)' --machine-rounding '$(SOIL_TRACTOR_BENCH_ROUNDING)' --machine-execution '$(SOIL_TRACTOR_BENCH_EXECUTION)' --paired-machine-rates '$(SOIL_TRACTOR_BENCH_PAIRED_RATES)' --native-machine-prediction '$(SOIL_TRACTOR_BENCH_NATIVE_PREDICTION)' --boundary-scan '$(SOIL_TRACTOR_BENCH_BOUNDARY_SCAN)' --capture-every '$(SOIL_TRACTOR_BENCH_CAPTURE_EVERY)' --soil-grid '$(SOIL_TRACTOR_BENCH_GRID)' --soil-transfer '$(SOIL_TRACTOR_BENCH_TRANSFER)' --final-soil-state '$(SOIL_TRACTOR_BENCH_FINAL_STATE)' --soil-motion-bound '$(SOIL_TRACTOR_BENCH_MOTION_BOUND)' --soil-motion-fusion '$(SOIL_TRACTOR_BENCH_MOTION_FUSION)' $(SOIL_TRACTOR_UNILATERAL_ARGS) $(SOIL_TRACTOR_FRICTION_ARGS) 2>&1 | tee '$(SOIL_RUN_DIR)/run.log'; result=$$?; \
	  echo $$result > '$(SOIL_RUN_DIR)/run.exit'; \
	  sha256sum -c '$(SOIL_RUN_DIR)/source-sha256.txt' > '$(SOIL_RUN_DIR)/source-check.log'; source_result=$$?; echo $$source_result > '$(SOIL_RUN_DIR)/source-check.exit'; \
	  $(NVIDIA_SMI) --query-gpu=name,driver_version,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv > '$(SOIL_RUN_DIR)/hardware-after.csv'; \
	  if [[ $$result == 0 ]]; then result=$$source_result; fi; exit $$result
SOIL_BENCH_PACKAGE = $(if $(filter radix,$(CASE)),molla-compute-primitives,$(if $(filter mpm-bridge active-domain mpm-profile,$(CASE)),molla-solvers,molla-kernels))
SOIL_BENCH_TEST = $(if $(filter radix,$(CASE)),radix_sort_gpu,$(if $(filter active-domain mpm-profile,$(CASE)),mpm_active_tiles_gpu,$(if $(filter mpm-bridge,$(CASE)),mpm_gpu,$(if $(filter dem,$(CASE)),soil_residency_gpu,soil_raster_gpu))))
SOIL_BENCH_FILTER = $(if $(filter radix,$(CASE)),parallel_retained_sort_paired_benchmark,$(if $(filter mpm-profile,$(CASE)),mpm_profile_stage_benchmark,$(if $(filter active-domain,$(CASE)),mpm_active_domain_benchmark,$(if $(filter mpm-bridge,$(CASE)),mpm_terrain_only_bridge_benchmark,$(if $(filter dem,$(CASE)),soil_dem_resident_benchmark,raster_benchmark)))))
SOIL_BENCH_FEATURES = $(if $(filter radix active-domain mpm-profile,$(CASE)),gpu$(comma)resource-counters,gpu)
comma := ,
SOIL_BENCH_LOG = $(CASE)
SOIL_SMOKE_BACKEND ?= mpm
SOIL_SMOKE_COUNTERS ?= 1
SOIL_SMOKE_FEATURES = $(if $(filter 1,$(SOIL_SMOKE_COUNTERS)),resource-counters,gpu)
ifeq ($(CASE)/$(MODE),smoke/baseline)
bench-soil:
	@case '$(SOIL_SMOKE_BACKEND)' in mpm|dem) ;; *) echo 'SOIL_SMOKE_BACKEND must be mpm or dem'; exit 2 ;; esac
	@case '$(SOIL_SMOKE_COUNTERS)' in 0|1) ;; *) echo 'SOIL_SMOKE_COUNTERS must be 0 or 1'; exit 2 ;; esac
	@[[ '$(SOIL_BENCH_SECONDS)' =~ ^[1-9][0-9]*$$ && '$(SOIL_BENCH_SECONDS)' -le 30 ]] || { echo 'Quick execution deadline must be 1 through 30 seconds'; exit 2; }
	@test ! -e '$(SOIL_RUN_DIR)' || { echo 'Refusing to reuse an existing benchmark result directory'; exit 2; }
	@$(MAKE) build APP_TARGET='-p molla-solvers --features $(SOIL_SMOKE_FEATURES) --example soil_benchmark --release'
	@mkdir -p '$(SOIL_RUN_DIR)'
	@git rev-parse HEAD > '$(SOIL_RUN_DIR)/revision.txt'
	@git status --short > '$(SOIL_RUN_DIR)/worktree.txt'
	@{ printf '%s\0' Makefile Cargo.toml Cargo.lock scripts/stop-soil-telemetry.sh; find crates -type f \( -name '*.rs' -o -name '*.wgsl' -o -name Cargo.toml \) -print0; } | sort -z | xargs -0 sha256sum > '$(SOIL_RUN_DIR)/source-sha256.txt'
	@tar -czf '$(SOIL_RUN_DIR)/source.tar.gz' --null -T <({ printf '%s\0' Makefile Cargo.toml Cargo.lock scripts/stop-soil-telemetry.sh; find crates -type f \( -name '*.rs' -o -name '*.wgsl' -o -name Cargo.toml \) -print0; } | sort -z)
	@sha256sum target/release/examples/soil_benchmark > '$(SOIL_RUN_DIR)/executable-sha256.txt'
	@cp target/release/examples/soil_benchmark '$(SOIL_RUN_DIR)/soil_benchmark'
	@$(NVIDIA_SMI) --query-gpu=name,driver_version,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv > '$(SOIL_RUN_DIR)/hardware-before.csv'
	@set -o pipefail; \
	  $(NVIDIA_SMI) --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv -lms 500 > '$(SOIL_RUN_DIR)/hardware-timeline.csv' 2>&1 & monitor=$$!; \
	  trap 'bash scripts/stop-soil-telemetry.sh "$(SOIL_RUN_DIR)/hardware-timeline.csv"; kill "$$monitor" 2>/dev/null || true; wait "$$monitor" 2>/dev/null || true' EXIT; \
	  timeout --kill-after=5s $(SOIL_BENCH_SECONDS)s $(RUN_WITH) $(CARGO) run --release -p molla-solvers --features $(SOIL_SMOKE_FEATURES) --example soil_benchmark -- smoke --backend '$(SOIL_SMOKE_BACKEND)' 2>&1 | tee '$(SOIL_RUN_DIR)/smoke.log'; result=$$?; echo $$result > '$(SOIL_RUN_DIR)/smoke.exit'; \
	  sha256sum -c '$(SOIL_RUN_DIR)/source-sha256.txt' > '$(SOIL_RUN_DIR)/source-check.log'; echo $$? > '$(SOIL_RUN_DIR)/source-check.exit'; \
	  $(NVIDIA_SMI) --query-gpu=name,driver_version,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv > '$(SOIL_RUN_DIR)/hardware-after.csv'; \
	  $(MAKE) check-soil-results INPUT='$(SOIL_RUN_DIR)' > '$(SOIL_RUN_DIR)/checker.log' 2>&1; checked=$$?; echo $$checked > '$(SOIL_RUN_DIR)/checker.exit'; \
	  if [[ $$result == 0 ]]; then result=$$checked; fi; exit $$result
else
bench-soil:
	@case "$(CASE)/$(MODE)" in raster/compare|dem/compare|mpm-bridge/compare|active-domain/compare|radix/compare|mpm-profile/diagnostic) ;; *) echo 'Baseline: CASE=smoke MODE=baseline; comparisons: CASE=raster, dem, mpm-bridge, active-domain or radix MODE=compare; profiling: CASE=mpm-profile MODE=diagnostic'; exit 2 ;; esac
	@[[ '$(SOIL_BENCH_SECONDS)' =~ ^[1-9][0-9]*$$ && '$(SOIL_BENCH_SECONDS)' -le 30 ]] || { echo 'Quick execution deadline must be 1 through 30 seconds'; exit 2; }
	@test ! -e '$(SOIL_RUN_DIR)' || { echo 'Refusing to reuse an existing benchmark result directory'; exit 2; }
	@$(MAKE) test APP_TARGET='-p $(SOIL_BENCH_PACKAGE) --features $(SOIL_BENCH_FEATURES) --test $(SOIL_BENCH_TEST) --release --no-run'
	@if [[ '$(CASE)' == active-domain ]]; then $(MAKE) build APP_TARGET='-p molla-solvers --example soil_benchmark --release'; fi
	@mkdir -p '$(SOIL_RUN_DIR)'
	@git rev-parse HEAD > '$(SOIL_RUN_DIR)/revision.txt'
	@sha256sum Makefile Cargo.lock crates/molla-kernels/Cargo.toml crates/molla-kernels/src/soil_{particle,mpm,dem}_raster.rs crates/molla-kernels/src/soil_dem_resident.rs crates/molla-kernels/src/tire_{mpm,dem}.rs crates/molla-kernels/shaders/soil_{raster_bins,mpm_raster,dem_raster}.wgsl crates/molla-kernels/shaders/tire_dem.wgsl crates/molla-kernels/tests/soil_{raster,residency,dem_topology}_gpu.rs crates/molla-compute-primitives/src/radix_sort.rs crates/molla-compute-primitives/shaders/radix_sort.wgsl > '$(SOIL_RUN_DIR)/source-sha256.txt'
	@sha256sum crates/molla-kernels/src/soil_mpm_snapshot.rs crates/molla-kernels/shaders/soil_mpm_summary.wgsl crates/molla-kernels/tests/soil_mpm_snapshot_gpu.rs crates/molla-solvers/Cargo.toml crates/molla-solvers/src/mpm.rs crates/molla-solvers/shaders/mpm.wgsl crates/molla-solvers/tests/mpm_gpu.rs bin/molla_viewer/src/examples/122_tire_realistic.rs >> '$(SOIL_RUN_DIR)/source-sha256.txt'
	@sha256sum crates/molla-solvers/src/mpm_active_tiles.rs crates/molla-solvers/shaders/mpm_active_tiles.wgsl crates/molla-solvers/tests/mpm_active_tiles_gpu.rs >> '$(SOIL_RUN_DIR)/source-sha256.txt'
	@sha256sum crates/molla-solvers/tests/mpm_sparse_grid_gpu.rs crates/molla-solvers/src/mpm_replay_tests.rs >> '$(SOIL_RUN_DIR)/source-sha256.txt'
	@sha256sum crates/molla-solvers/src/mpm_tile_map_tests.rs >> '$(SOIL_RUN_DIR)/source-sha256.txt'
	@sha256sum crates/molla-solvers/src/mpm_profile.rs >> '$(SOIL_RUN_DIR)/source-sha256.txt'
	@sha256sum crates/molla-solvers/src/mpm_particle_order.rs crates/molla-solvers/src/mpm_particle_order_tests.rs crates/molla-solvers/shaders/mpm_particle_order.wgsl >> '$(SOIL_RUN_DIR)/source-sha256.txt'
	@sha256sum crates/molla-compute-primitives/Cargo.toml crates/molla-compute-primitives/shaders/radix_sort_parallel.wgsl crates/molla-compute-primitives/tests/radix_sort_gpu.rs >> '$(SOIL_RUN_DIR)/source-sha256.txt'
	@sha256sum crates/molla-kernels/src/soil_gpu_identity.rs crates/molla-kernels/src/soil_mpm_summary.rs crates/molla-kernels/src/soil_mpm_observation.rs crates/molla-kernels/tests/soil_mpm_observation_gpu.rs >> '$(SOIL_RUN_DIR)/source-sha256.txt'
	@sha256sum crates/molla-solvers/examples/soil_benchmark.rs crates/molla-solvers/examples/soil_support/results.rs crates/molla-solvers/tests/soil_benchmark_contract.rs >> '$(SOIL_RUN_DIR)/source-sha256.txt'
	@$(NVIDIA_SMI) --query-gpu=name,driver_version,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv > '$(SOIL_RUN_DIR)/hardware-before.csv'
	@set -o pipefail; timeout --kill-after=5s $(SOIL_BENCH_SECONDS)s $(MAKE) test-gpu P=$(SOIL_BENCH_PACKAGE) T=$(SOIL_BENCH_TEST) GPU_FEATURES='$(SOIL_BENCH_FEATURES)' RELEASE=1 A='$(SOIL_BENCH_FILTER) --nocapture --test-threads=1' 2>&1 | tee '$(SOIL_RUN_DIR)/$(SOIL_BENCH_LOG).log'; result=$$?; echo $$result > '$(SOIL_RUN_DIR)/$(SOIL_BENCH_LOG).exit'; $(NVIDIA_SMI) --query-gpu=name,driver_version,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv > '$(SOIL_RUN_DIR)/hardware-after.csv'; \
	  if [[ '$(CASE)' == active-domain ]]; then \
	    $(MAKE) check-soil-results INPUT='$(SOIL_RUN_DIR)' > '$(SOIL_RUN_DIR)/checker.log' 2>&1; checked=$$?; echo $$checked > '$(SOIL_RUN_DIR)/checker.exit'; \
	    if [[ $$result == 0 ]]; then result=$$checked; fi; \
	  fi; exit $$result
endif

test-primitives-baseline:
	@$(MAKE) test APP_TARGET='-p molla-solvers $(foreach lane,$(PRIMITIVE_CPU_BASELINES),--test $(lane))'

.PHONY: test-primitives-softness-activation
test-primitives-softness-activation:
	@$(MAKE) test APP_TARGET='-p molla-sim --test runtime_softness_activation'

.PHONY: test-primitives-distance-scale-gpu
test-primitives-distance-scale-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib rigid_world::gpu::scale_tests -- --ignored --nocapture --test-threads=1

PRIMITIVE_DISTANCE_HZ ?= 60
.PHONY: test-primitives-distance-ten-second-cpu test-primitives-distance-ten-second-gpu
test-primitives-distance-ten-second-cpu:
	@PRIMITIVE_DISTANCE_HZ=$(PRIMITIVE_DISTANCE_HZ) timeout --kill-after=10s 120s $(CARGO) test $(REL) -p molla-solvers --test runtime_distance_contract cpu_distance_length_mass_scales_and_analytic_reactions -- --exact --nocapture

test-primitives-distance-ten-second-gpu:
	@PRIMITIVE_DISTANCE_HZ=$(PRIMITIVE_DISTANCE_HZ) timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --test runtime_distance_contract gpu_distance_length_mass_scales_and_analytic_reactions -- --exact --ignored --nocapture --test-threads=1

test-primitives-baseline-gpu:
	@set -e; for lane in $(PRIMITIVE_GPU_BASELINES); do \
		$(MAKE) test-gpu P=molla-solvers T=$$lane A='--nocapture --test-threads=1'; \
	done

test-primitives-runtime-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=runtime_primitives_contract_gpu GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'

test-primitives-loop-rows-gpu:
	@$(MAKE) test-gpu P=molla-kernels T=compliant_loop_rows_gpu A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-kernels T=quaternion_fk_gpu A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-kernels T=featherstone_mass_inverse_gpu A='--nocapture --test-threads=1'

test-primitives-loop-resources-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=runtime_loop_compliance GPU_FEATURES=resource-counters A='--nocapture --test-threads=1 --exact gpu_compliant_loop_numeric_edits_keep_device_resources_retained'

test-primitives-loop-rows-cpu:
	@$(MAKE) test APP_TARGET='-p molla-kernels --features gpu --lib compliant_loop_rows::tests -- --nocapture'

test-primitives-distance-cpu:
	@$(MAKE) test APP_TARGET='-p molla-solvers --test runtime_distance --test runtime_distance_sleep --test runtime_distance_contract --test runtime_com_roots -- --nocapture'

test-primitives-distance-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=runtime_distance_gpu GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=runtime_distance_sleep GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=runtime_distance_contract GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=runtime_com_roots GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'

test-primitives-cone-geometry-cpu:
	@$(MAKE) test APP_TARGET='-p molla-sim --test spherical_cone --test runtime_cone_configuration -- --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-solvers --test runtime_cone_capability -- --nocapture'

test-primitives-cone-geometry-gpu:
	@$(MAKE) test-gpu P=molla-kernels T=cone_geometry_gpu A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=runtime_cone_capability A='--nocapture --test-threads=1'

test-primitives-cone-cpu: test-primitives-cone-geometry-cpu
	@$(MAKE) test APP_TARGET='-p molla-solvers --test runtime_cone_limits --test runtime_ball_loop --test cone_precision -- --nocapture'

test-primitives-cone-gpu: test-primitives-cone-geometry-gpu
	@$(MAKE) test-gpu P=molla-solvers T=runtime_cone_limits A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=runtime_ball_loop GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'

PRIMITIVE_CONE_HZ ?= 60
.PHONY: test-primitives-cone-ten-second-cpu test-primitives-cone-ten-second-gpu
test-primitives-cone-ten-second-cpu:
	@PRIMITIVE_CONE_HZ=$(PRIMITIVE_CONE_HZ) timeout --kill-after=10s 120s $(CARGO) test $(REL) -p molla-solvers --test runtime_cone_limits cpu_cone_pendulum_is_bounded_for_ten_seconds -- --exact --nocapture

test-primitives-cone-ten-second-gpu:
	@PRIMITIVE_CONE_HZ=$(PRIMITIVE_CONE_HZ) timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --test runtime_cone_limits gpu_cone_pendulum_is_bounded_for_ten_seconds -- --exact --ignored --nocapture --test-threads=1

test-primitives-reactions-cpu:
	@$(MAKE) test APP_TARGET='-p molla-solvers --test reaction_wrenches --test reaction_boundary_acceleration --test runtime_joint_reactions --test runtime_body_control_wrenches --test runtime_kinematic_reactions --test rnea_inverse_dynamics --test runtime_tree -- --nocapture'

.PHONY: test-primitives-kinematic-motion-cpu
test-primitives-kinematic-motion-cpu:
	@$(MAKE) test APP_TARGET='-p molla-solvers --test runtime_kinematic_reactions --test runtime_kinematic_constraints -- --nocapture'

.PHONY: test-primitives-kinematic-motion-gpu
.PHONY: test-primitives-pose-precision-gpu
test-primitives-pose-precision-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=kinematic_motion_gpu GPU_FEATURES=resource-counters A='zero_coordinate --nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=runtime_pose_precision_gpu A='--nocapture --test-threads=1'

test-primitives-kinematic-motion-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=kinematic_motion_gpu GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=runtime_kinematic_reactions A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=runtime_kinematic_constraints A='--nocapture --test-threads=1'

.PHONY: test-primitives-restitution-cpu test-primitives-impact-oracle test-primitives-articulated-impact-cpu
test-primitives-articulated-impact-cpu:
	@$(MAKE) test APP_TARGET='-p molla-solvers --test articulated_impact -- --nocapture'

test-primitives-restitution-cpu:
	@$(MAKE) test APP_TARGET='-p molla-solvers --test runtime_restitution -- --nocapture'

.PHONY: test-primitives-restitution-gpu
test-primitives-restitution-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=runtime_restitution_gpu A='--nocapture --test-threads=1'

test-primitives-impact-oracle:
	@$(MAKE) test APP_TARGET='-p molla-solvers --lib impact_constraints::tests -- --nocapture'

.PHONY: test-primitives-impact-gpu
test-primitives-impact-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib impact_constraints_gpu::tests -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-impact-response-gpu
test-primitives-impact-response-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib contact_response_gpu::impact_tests -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-impact-position-gpu
test-primitives-impact-position-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib contact_response_gpu::position_tests -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-impact-partition-gpu
test-primitives-impact-partition-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib contact_response_gpu::partition_tests -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-impact-episodes-gpu
test-primitives-impact-episodes-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib impact_episodes_gpu::tests -- --ignored --nocapture --test-threads=1

test-primitives-friction-cpu:
	@$(MAKE) test APP_TARGET='-p molla-geometry --test contact_materials -- --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-solvers --lib contact_friction -- --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-solvers --test runtime_friction --test featherstone_contacts --test track_forces -- --nocapture'

test-primitives-friction-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib coulomb_disk_gpu::tests -- --ignored --nocapture --test-threads=1

PRIMITIVE_FRICTION_HZ ?= 60
PRIMITIVE_FRICTION_TIMEOUT ?= $(if $(filter 240,$(PRIMITIVE_FRICTION_HZ)),240,120)
.PHONY: test-primitives-friction-ten-second-cpu test-primitives-friction-ten-second-gpu
test-primitives-friction-ten-second-cpu:
	@test "$(PRIMITIVE_FRICTION_TIMEOUT)" -gt 0
	@PRIMITIVE_FRICTION_HZ=$(PRIMITIVE_FRICTION_HZ) timeout --kill-after=10s "$(PRIMITIVE_FRICTION_TIMEOUT)" $(CARGO) test $(REL) -p molla-solvers --test runtime_friction_contract loaded_static_friction_holds_ten_seconds_at_three_step_sizes -- --exact --nocapture

test-primitives-friction-ten-second-gpu:
	@test "$(PRIMITIVE_FRICTION_TIMEOUT)" -gt 0
	@echo "static hold hz=$(PRIMITIVE_FRICTION_HZ) wall_budget_seconds=$(PRIMITIVE_FRICTION_TIMEOUT)"
	@PRIMITIVE_FRICTION_HZ=$(PRIMITIVE_FRICTION_HZ) timeout --kill-after=10s "$(PRIMITIVE_FRICTION_TIMEOUT)" $(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --test runtime_friction_contract gpu_loaded_static_friction_holds_ten_seconds_at_three_step_sizes -- --exact --ignored --nocapture --test-threads=1

.PHONY: test-primitives-coupled-constraints-gpu
test-primitives-coupled-constraints-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib coupled_constraint_gpu::tests -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-coupled-constraints-cpu
test-primitives-coupled-constraints-cpu:
	@$(MAKE) test APP_TARGET='-p molla-solvers --lib coupled_constraints -- --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-solvers --lib constraint_allocation -- --nocapture'

.PHONY: test-primitives-coupled-response-gpu
test-primitives-coupled-response-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib contact_response_gpu::coupled_tests -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-contact-activation-gpu
test-primitives-contact-activation-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib rigid_contact_activation_gpu::tests -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-mixed-contacts-gpu
test-primitives-mixed-contacts-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=runtime_mixed_contacts_gpu A='--nocapture --test-threads=1'

.PHONY: test-primitives-mixed-contacts-cpu
test-primitives-mixed-contacts-cpu:
	@$(MAKE) test APP_TARGET='-p molla-solvers --test runtime_mixed_contacts_cpu -- --nocapture'

.PHONY: test-primitives-mixed-drives-cpu test-primitives-mixed-drives-gpu
test-primitives-mixed-drives-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_mixed_drives --test runtime_locked_drive_chains -- --nocapture

test-primitives-mixed-drives-gpu:
	@timeout --kill-after=10s 180s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_mixed_drives -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-locked-drives-gpu
test-primitives-locked-drives-gpu:
	@timeout --kill-after=10s 180s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_locked_drive_chains -- --ignored --nocapture --test-threads=1

test-primitives-contact-islands-gpu:
	@$(MAKE) test-gpu P=molla-compute-primitives T=radix_sort_gpu A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=contact_islands_gpu A='--nocapture --test-threads=1'

test-primitives-contact-response-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib contact_response_gpu::tests -- --ignored --nocapture --test-threads=1

test-primitives-contact-publication-gpu:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib contact_publication_gpu::tests -- --ignored --nocapture --test-threads=1

test-primitives-runtime-contacts-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=runtime_contacts_gpu A='--nocapture --test-threads=1'

test-primitives-contact-reactions-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=runtime_contact_reactions_gpu A='--nocapture --test-threads=1'
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features $(GPU_FEATURES) --lib rigid_world::gpu::tests -- --ignored --nocapture --test-threads=1

test-primitives-contact-breakage-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=runtime_contact_breakage_gpu A='--nocapture --test-threads=1'

.PHONY: test-primitives-dispatch-metrics test-primitives-dispatch-metrics-gpu
test-primitives-dispatch-metrics:
	@$(MAKE) test APP_TARGET='-p molla-core --features gpu-metrics --lib'

test-primitives-dispatch-metrics-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=runtime_dispatch_metrics_gpu GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'

.PHONY: test-primitives-multidof-feedback-gpu
test-primitives-multidof-feedback-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=runtime_multidof_feedback_gpu A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=runtime_multidof_limits A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=runtime_d6_axis_certification A='--nocapture --test-threads=1'

test-primitives-contact-pipeline-gpu:
	@$(MAKE) test-gpu P=molla-geometry T=lbvh_resident_gpu A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-geometry T=shape_bounds_gpu A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-geometry T=narrow_resident_gpu A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-geometry T=gjk_gpu A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-geometry T=gpu_narrow_phase A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-geometry T=broadphase_lbvh_gpu A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=barrier_newton_gpu A='lbvh_ --nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=normal_contact_gpu A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=featherstone_drift_gpu A='--nocapture --test-threads=1'
	@$(MAKE) test-primitives-friction-gpu
	@$(MAKE) test-primitives-coupled-constraints-gpu
	@$(MAKE) test-primitives-coupled-response-gpu
	@$(MAKE) test-primitives-mixed-contacts-gpu
	@$(MAKE) test-primitives-contact-islands-gpu
	@$(MAKE) test-primitives-contact-response-gpu
	@$(MAKE) test-primitives-contact-publication-gpu
	@$(MAKE) test-primitives-runtime-contacts-gpu
	@$(MAKE) test-primitives-contact-reactions-gpu
	@$(MAKE) test-primitives-contact-breakage-gpu

test-primitives-breakage-cpu:
	@$(MAKE) test APP_TARGET='-p molla-sim --test runtime_break_configuration -- --nocapture'
	@$(MAKE) test APP_TARGET='-p molla-solvers --test runtime_break_detachment --test runtime_joint_breakage -- --nocapture'

test-primitives-breakage-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=runtime_break_detachment GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=runtime_joint_breakage_gpu GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'

test-primitives-reactions-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=runtime_body_control_wrenches A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=reaction_wrenches_gpu GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'
	@$(MAKE) test-primitives-boundary-reactions-gpu
	@$(MAKE) test-gpu P=molla-solvers T=runtime_reactions_gpu GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=runtime_kinematic_reactions A='--nocapture --test-threads=1'
	@$(MAKE) test-gpu P=molla-solvers T=articulation_gpu A='gpu_rnea_ --nocapture --test-threads=1'

.PHONY: test-primitives-boundary-reactions-gpu
test-primitives-boundary-reactions-gpu:
	@$(MAKE) test-gpu P=molla-solvers T=reaction_boundary_acceleration GPU_FEATURES=resource-counters A='--nocapture --test-threads=1'

test-sensors:
	@$(MAKE) test APP_TARGET='-p molla-compute --features gpu --lib'
	@$(MAKE) test APP_TARGET='-p molla-geometry --no-default-features --lib mesh_bvh::tests'
	@$(MAKE) test APP_TARGET='-p molla-geometry --features gpu --lib validation_tests'
	@$(MAKE) test APP_TARGET='-p molla-sensors --no-default-features'
	@$(MAKE) test APP_TARGET='-p molla-sensors --features gpu --lib'
	@$(MAKE) test APP_TARGET='-p molla-sensors --features gpu --test sensor_allocation_gpu'
	@$(MAKE) test APP_TARGET='-p molla-sensors --features gpu --test sensor_large_coordinates_gpu'
	@$(MAKE) test APP_TARGET='-p molla --no-default-features --test sensor_api'
	@$(MAKE) test APP_TARGET='-p molla --features gpu --test sensor_api'
	@$(MAKE) test-gpu P=molla T=sensor_physics_gpu A='--test-threads=1'
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-sensors --features gpu --tests -- --ignored --test-threads=1
	@$(MAKE) test-sensor-resources

# Warm-frame GPU object retention and heap-allocation attribution (wgpu counters).
test-sensor-resources:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-sensors --features resource-counters --test sensor_resource_retention_gpu -- --ignored --test-threads=1 $(A)

test-viewer-sensors:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla_viewer --bin molla_viewer -- sensor_ --include-ignored --test-threads=1 $(A)

test-all:
	@$(RUN_WITH) $(CARGO) test --workspace --all-targets $(TEST_ALL_ARGS)

check:
	@$(CARGO) check $(APP_TARGET)

check-all:
	@$(CARGO) check --workspace --all-targets

fmt-files:
	@test -n "$(FILES)" || { echo 'Set FILES to the Rust files to format'; exit 1; }
	@rustfmt --edition 2024 $(FMT_ARGS) $(FILES)

fmt-check:
	@$(CARGO) fmt --all -- --check

clippy-check:
	@$(CARGO) clippy $(CLIPPY_TARGET) --keep-going -- -D warnings

.PHONY: check-primitives-shader-wiring
check-primitives-shader-wiring:
	@$(CARGO) run -p molla-solvers --features gpu --example check_primitives_shaders

.PHONY: check-primitives-gpu-wiring
check-primitives-gpu-wiring:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) run -p molla-solvers --features gpu --example prepare_primitives_gpu

.PHONY: check-primitives-gpu-step-wiring
check-primitives-gpu-step-wiring:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) run -p molla-solvers --features gpu --example prepare_primitives_gpu -- --step

.PHONY: test-primitives-singular-wiring-cpu test-primitives-singular-wiring-gpu test-primitives-chart-wiring-cpu test-primitives-chart-wiring-gpu test-primitives-velocity-wiring-cpu test-primitives-velocity-wiring-gpu
test-primitives-singular-wiring-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_singular_bounds cpu_active_phase_interval_at_both_xyz_poles -- --exact --nocapture

test-primitives-singular-wiring-gpu:
	@set -e; for pole in negative positive; do $(MAKE) -j1 test-primitives-singular-pole-gpu PRIMITIVE_XYZ_POLE=$$pole; done

PRIMITIVE_XYZ_POLE ?= positive
.PHONY: test-primitives-singular-pole-cpu test-primitives-singular-pole-gpu
test-primitives-singular-pole-cpu:
	@PRIMITIVE_XYZ_POLE=$(PRIMITIVE_XYZ_POLE) timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_singular_bounds cpu_active_phase_interval_at_both_xyz_poles -- --exact --nocapture

test-primitives-singular-pole-gpu:
	@PRIMITIVE_XYZ_POLE=$(PRIMITIVE_XYZ_POLE) timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_singular_bounds gpu_active_phase_interval_matches_cpu_at_both_xyz_poles -- --exact --ignored --nocapture

test-primitives-chart-wiring-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_singular_bounds cpu_free_d6_motion_crosses_both_xyz_poles_without_false_stops -- --exact --nocapture

test-primitives-chart-wiring-gpu:
	@set -e; for pole in negative positive; do $(MAKE) -j1 test-primitives-chart-pole-gpu PRIMITIVE_XYZ_POLE=$$pole; done

.PHONY: test-primitives-chart-pole-cpu test-primitives-chart-pole-gpu
test-primitives-chart-pole-cpu:
	@PRIMITIVE_XYZ_POLE=$(PRIMITIVE_XYZ_POLE) timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_singular_bounds cpu_free_d6_motion_crosses_both_xyz_poles_without_false_stops -- --exact --nocapture

test-primitives-chart-pole-gpu:
	@PRIMITIVE_XYZ_POLE=$(PRIMITIVE_XYZ_POLE) timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_singular_bounds gpu_free_d6_motion_crosses_both_xyz_poles_without_false_stops -- --exact --ignored --nocapture

.PHONY: test-primitives-angular-lock-wiring-cpu test-primitives-angular-lock-wiring-gpu
test-primitives-angular-lock-wiring-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_angular_locks --test featherstone_mixed_distance -- --nocapture

test-primitives-angular-lock-wiring-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_angular_locks --test featherstone_mixed_distance -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-disabled-wiring-cpu test-primitives-disabled-wiring-gpu
.PHONY: test-primitives-angular-mass-wiring-cpu test-primitives-angular-mass-wiring-gpu
test-primitives-angular-mass-wiring-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_angular_mass -- --nocapture

test-primitives-angular-mass-wiring-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_angular_mass -- --ignored --nocapture --test-threads=1 --skip bounded_hinges --skip regular_points --skip phase_intervals

.PHONY: test-primitives-angular-bound-wiring-cpu test-primitives-angular-bound-wiring-gpu
test-primitives-angular-bound-wiring-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_angular_mass bounded_hinges -- --nocapture

test-primitives-angular-bound-wiring-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_angular_mass bounded_hinges -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-angular-regular-wiring-cpu test-primitives-angular-regular-wiring-gpu
.PHONY: test-primitives-angular-regular-standalone-cpu test-primitives-angular-regular-standalone-gpu
.PHONY: test-primitives-d6-activation-cpu test-primitives-d6-interval-standalone-cpu test-primitives-d6-interval-standalone-gpu test-primitives-d6-interval-runtime-cpu test-primitives-d6-interval-runtime-gpu
test-primitives-d6-activation-cpu:
	@$(CARGO) test -p molla-sim --lib joint_control::coupled_stop_tests -- --nocapture

test-primitives-d6-interval-standalone-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_angular_mass cpu_standalone_regular_intervals_and_caps -- --nocapture

test-primitives-d6-interval-standalone-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_angular_mass gpu_standalone_regular_intervals_and_caps -- --ignored --nocapture --test-threads=1

test-primitives-d6-interval-runtime-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_angular_mass cpu_runtime_regular_intervals_and_caps -- --nocapture

test-primitives-d6-interval-runtime-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_angular_mass gpu_runtime_regular_intervals_and_caps -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-angular-phase-standalone-cpu test-primitives-angular-phase-standalone-gpu
test-primitives-angular-phase-standalone-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_angular_mass cpu_standalone_phase_intervals -- --nocapture

test-primitives-angular-phase-standalone-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_angular_mass gpu_standalone_phase_intervals -- --ignored --nocapture --test-threads=1

test-primitives-angular-regular-standalone-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_angular_mass cpu_standalone_regular_points -- --nocapture

test-primitives-angular-regular-standalone-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_angular_mass gpu_standalone_regular_points -- --ignored --nocapture --test-threads=1

test-primitives-angular-regular-wiring-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_angular_mass regular_points -- --nocapture

test-primitives-angular-regular-wiring-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_angular_mass regular_points -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-angular-narrow-wiring-cpu test-primitives-angular-narrow-wiring-gpu
test-primitives-angular-narrow-wiring-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_angular_mass narrow_intervals -- --nocapture

test-primitives-angular-narrow-wiring-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_angular_mass narrow_intervals -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-limit-reference-cpu test-primitives-angular-narrow-standalone-cpu test-primitives-angular-narrow-standalone-gpu
.PHONY: test-primitives-scalar-precision-cpu test-primitives-scalar-precision-gpu
.PHONY: test-primitives-d6-point-publication-cpu test-primitives-d6-point-publication-gpu
.PHONY: test-primitives-pitch-pole-cpu test-primitives-pitch-pole-gpu
.PHONY: test-primitives-radial-precision-cpu test-primitives-radial-precision-gpu
.PHONY: test-primitives-distance-activation-cpu
test-primitives-distance-activation-cpu:
	@$(CARGO) test -p molla-sim --lib joint_control::coupled_stop_tests::distance_activation -- --nocapture

.PHONY: test-primitives-ball-point-publication-cpu test-primitives-ball-point-publication-gpu
.PHONY: test-primitives-ball-angular-cpu test-primitives-ball-angular-gpu
.PHONY: test-primitives-redundant-limits-cpu test-primitives-redundant-limits-gpu
.PHONY: test-primitives-redundant-raw-gpu
test-primitives-redundant-raw-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test standalone_redundant_limits gpu_redundant_ -- --ignored --nocapture --test-threads=1
.PHONY: test-primitives-redundant-runtime-cpu test-primitives-redundant-runtime-gpu
test-primitives-redundant-runtime-cpu:
	@$(CARGO) test -p molla-solvers --lib distance_constraints::tests::equality_gap_resolves_arithmetic_noise_without_erasing_motion -- --exact --nocapture
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test standalone_redundant_limits cpu_runtime_ -- --nocapture --test-threads=1

test-primitives-redundant-runtime-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test standalone_redundant_limits gpu_runtime_ -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-redundant-runtime-mobile-gpu
.PHONY: test-primitives-redundant-runtime-hold-gpu
.PHONY: test-primitives-redundant-runtime-rollback-gpu
test-primitives-redundant-runtime-rollback-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test standalone_redundant_limits gpu_runtime_incompatible_radial_point_and_zero_caps_fail_without_advancing -- --exact --ignored --nocapture --test-threads=1

test-primitives-redundant-runtime-hold-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test standalone_redundant_limits gpu_runtime_redundant_points_and_zero_caps_hold_under_load -- --exact --ignored --nocapture --test-threads=1

test-primitives-redundant-runtime-mobile-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test standalone_redundant_limits gpu_runtime_redundant_points_and_caps_preserve_permitted_physical_motion -- --exact --ignored --nocapture --test-threads=1

.PHONY: test-primitives-radial-cap-stages-gpu
.PHONY: test-primitives-frame-reference-cpu
.PHONY: test-primitives-frame-reference-gpu
.PHONY: test-primitives-all-axis-sustained-cpu test-primitives-all-axis-sustained-gpu
.PHONY: test-primitives-boundary-radial-cpu test-primitives-boundary-radial-gpu
test-primitives-boundary-radial-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_boundary_radial_caps -- --nocapture

test-primitives-boundary-radial-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_boundary_radial_caps -- --ignored --nocapture --test-threads=1
test-primitives-all-axis-sustained-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_axis_certification -- --nocapture

test-primitives-all-axis-sustained-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_axis_certification -- --ignored --nocapture --test-threads=1
test-primitives-frame-reference-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --lib paired_fk_chains_retain_frames_after_model_upload_edits -- --ignored --nocapture --test-threads=1
test-primitives-frame-reference-cpu:
	@$(CARGO) test -p molla-sim --features gpu --lib joint_frame_reference -- --nocapture

test-primitives-radial-cap-stages-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --lib radial_point_zero_cap_first_step_stages -- --ignored --nocapture --test-threads=1

test-primitives-redundant-limits-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test standalone_redundant_limits -- --nocapture

test-primitives-redundant-limits-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test standalone_redundant_limits -- --ignored --nocapture --test-threads=1

test-primitives-ball-angular-cpu:
	@$(CARGO) test -p molla-sim --lib joint_control::coupled_stop_tests::ball_activation -- --nocapture
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test standalone_ball_limits -- --nocapture

test-primitives-ball-angular-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test standalone_ball_limits -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-quaternion-motion-cpu test-primitives-quaternion-motion-gpu
test-primitives-quaternion-motion-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test standalone_limit_precision cpu_quaternion_kinds -- --nocapture

test-primitives-quaternion-motion-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test standalone_limit_precision gpu_quaternion_kinds -- --ignored --nocapture --test-threads=1

test-primitives-ball-point-publication-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test standalone_limit_precision cpu_ball_points_publish -- --nocapture

test-primitives-ball-point-publication-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test standalone_limit_precision gpu_ball_points_publish -- --ignored --nocapture --test-threads=1

test-primitives-radial-precision-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test featherstone_mixed_distance -- --nocapture

test-primitives-radial-precision-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test featherstone_mixed_distance -- --ignored --nocapture --test-threads=1

test-primitives-pitch-pole-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_angular_mass near_pole_pitch -- --nocapture

test-primitives-pitch-pole-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_angular_mass near_pole_pitch -- --ignored --nocapture --test-threads=1

test-primitives-d6-point-publication-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test standalone_limit_precision cpu_d6_points_publish -- --nocapture

test-primitives-d6-point-publication-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test standalone_limit_precision gpu_d6_points_publish -- --ignored --nocapture --test-threads=1

test-primitives-scalar-precision-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test standalone_limit_precision -- --nocapture

test-primitives-scalar-precision-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test standalone_limit_precision -- --ignored --nocapture --test-threads=1

test-primitives-limit-reference-cpu:
	@$(CARGO) test -p molla-sim --lib joint_limit_reference::tests -- --nocapture
	@$(CARGO) test -p molla-sim --features gpu --lib model::tests::joint_limit_references -- --nocapture

test-primitives-angular-narrow-standalone-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_angular_mass gpu_standalone_narrow_hinges -- --ignored --nocapture --test-threads=1

test-primitives-angular-narrow-standalone-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_angular_mass cpu_standalone_narrow_hinges -- --nocapture

.PHONY: test-primitives-angular-phase-wiring-cpu test-primitives-angular-phase-wiring-gpu
test-primitives-angular-phase-wiring-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_angular_mass phase_intervals -- --nocapture

test-primitives-angular-phase-wiring-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_angular_mass phase_intervals -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-hinge-antipode-wiring-cpu test-primitives-hinge-antipode-wiring-gpu
test-primitives-hinge-antipode-wiring-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --lib joint_angles::tests::finite_pair_twist -- --nocapture
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_hinge_antipodes -- --nocapture

test-primitives-hinge-antipode-wiring-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_hinge_antipodes -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-hinge-antipode-standalone-cpu test-primitives-hinge-antipode-standalone-gpu
test-primitives-hinge-antipode-standalone-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_d6_hinge_antipodes cpu_standalone -- --nocapture

test-primitives-hinge-antipode-standalone-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_hinge_antipodes gpu_standalone -- --ignored --nocapture --test-threads=1

.PHONY: test-primitives-hinge-antipode-compensated-gpu
test-primitives-hinge-antipode-compensated-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_d6_hinge_antipodes gpu_standalone_compensated -- --ignored --nocapture --test-threads=1

test-primitives-disabled-wiring-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test featherstone_disabled_joints -- --nocapture

test-primitives-disabled-wiring-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test featherstone_disabled_joints -- --ignored --nocapture --test-threads=1

test-primitives-velocity-wiring-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_velocity_coupling cpu_strongly_coupled_velocity_drives_match_bounded_implicit_oracle -- --exact --nocapture

test-primitives-velocity-wiring-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_velocity_coupling gpu_strongly_coupled_velocity_drives_match_bounded_implicit_oracle -- --exact --ignored --nocapture

.PHONY: test-primitives-velocity-sustained-cpu test-primitives-velocity-sustained-gpu
test-primitives-velocity-sustained-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_velocity_coupling cpu_coupled_velocity_drives_preserve_one_second_motion_and_control_edits -- --exact --nocapture

test-primitives-velocity-sustained-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_velocity_coupling gpu_coupled_velocity_drives_preserve_one_second_motion_and_control_edits -- --exact --ignored --nocapture

.PHONY: test-primitives-velocity-raw-gpu
test-primitives-velocity-raw-gpu:
	@timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_velocity_coupling gpu_raw_coupled_roots_work_without_coordinate_compensation -- --exact --ignored --nocapture

PRIMITIVE_MOTOR_FEEDBACK ?= stable
.PHONY: test-primitives-motor-coupling-cpu test-primitives-motor-coupling-gpu test-primitives-motor-feedback-gpu
.PHONY: test-primitives-locked-drive-planes-cpu test-primitives-locked-drive-planes-gpu
test-primitives-locked-drive-planes-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_locked_drive_planes -- --nocapture

test-primitives-locked-drive-planes-gpu:
	@timeout --kill-after=10s 180s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_locked_drive_planes -- --ignored --nocapture --test-threads=1

test-primitives-motor-coupling-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test runtime_motor_coupling -- --nocapture

test-primitives-motor-coupling-gpu:
	@set -e; for feedback in explicit stable; do $(MAKE) -j1 test-primitives-motor-feedback-gpu PRIMITIVE_MOTOR_FEEDBACK=$$feedback; done

test-primitives-motor-feedback-gpu:
	@PRIMITIVE_MOTOR_FEEDBACK=$(PRIMITIVE_MOTOR_FEEDBACK) timeout --kill-after=10s 120s $(RUN_WITH) $(CARGO) test -p molla-solvers --features gpu --test runtime_motor_coupling gpu_coupled_motor_models_match_two_body_inverse_mass -- --exact --ignored --nocapture

harden:
	@git diff --check
	@$(MAKE) fmt-check
	@$(CARGO) check --workspace --no-default-features
	@$(CARGO) test --workspace --no-default-features $(TEST_ALL_ARGS)
	@$(MAKE) clippy-check
	@$(RUN_WITH) $(CARGO) test --workspace --all-targets --all-features $(TEST_ALL_ARGS)

bench:
	@$(RUN_WITH) $(CARGO) bench $(BENCH_ARGS)

.PHONY: bench-primitives test-primitives-benchmark-smoke
.PHONY: test-primitives-construction-cpu
test-primitives-construction-cpu:
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-sim --test runtime_builder --test runtime -- --nocapture
	@timeout --kill-after=10s 120s $(CARGO) test -p molla-solvers --test primitives_benchmark_fixtures -- --nocapture

bench-primitives:
	@$(RUN_WITH) $(CARGO) bench -p molla-solvers --features resource-counters --bench runtime_primitives --profile $(PRIMITIVE_BENCH_PROFILE) -- $(A)

test-primitives-benchmark-smoke:
	@$(RUN_WITH) $(CARGO) test $(REL) -p molla-solvers --features resource-counters --bench runtime_primitives -- --smoke $(A)

.PHONY: example-primitives example-primitives-gpu
example-primitives:
	@$(RUN_WITH) $(CARGO) run -p molla-solvers --example runtime_primitives -- $(A)

example-primitives-gpu:
	@$(RUN_WITH) $(CARGO) run -p molla-solvers --features gpu --example runtime_primitives -- --gpu $(A)

bench-sensors:
	@$(RUN_WITH) $(CARGO) bench -p $(SENSOR_BENCH_PKG) --features gpu --bench $(SENSOR_BENCH) -- $(A)

sensor-bench-info:
	@$(CARGO) --version
	@rustc --version
	@uname -srmo
	@git rev-parse HEAD
	@printf 'Dirty/untracked entries: '; git status --porcelain | wc -l
	@printf 'Crate source/manifests fingerprint: '; find crates -type f \( -name '*.rs' -o -name '*.wgsl' -o -name Cargo.toml \) -print0 | sort -z | xargs -0 sha256sum | sha256sum
	@sha256sum Cargo.toml Cargo.lock Makefile flake.nix flake.lock
	@if test -r /proc/driver/nvidia/version; then cat /proc/driver/nvidia/version; fi
	@printf 'NVIDIA telemetry command: %s\n' '$(NVIDIA_SMI)'
	@if command -v nvidia-smi >/dev/null; then $(NVIDIA_SMI) --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total --format=csv; fi

bench-sensors-latency: sensor-bench-info
	@$(MAKE) bench-sensors SENSOR_BENCH=$(LATENCY_BENCH) A='$(A)'

docs:
	@command -v mdbook >/dev/null 2>&1 || { echo "mdbook is not installed. Please install it first."; exit 1; }
	@mdbook build $(TOP_DIR)/book --dest-dir $(DOCS_DIR)

release:
	@if [ -z "$(HAS_REL)" ]; then \
		echo "git-rel is not installed. Please install it first."; \
		exit 1; \
	fi
	@if [ -z "$(TYPE)" ]; then \
		echo "Release type not specified. Use 'make release TYPE=[patch|minor|major|m.m.p]'"; \
		exit 1; \
	fi
	@git rel $(TYPE)

clean:
	@$(CARGO) clean

help:
	@echo
	@echo "Usage: make [target]"
	@echo
	@echo "Available targets:"
	@echo "  build        Build $(APP_PKG)"
	@echo "  compile      Clean and rebuild"
	@echo "  run          Run $(APP_PKG) ($(BACKEND) backend, $(RUN_WITH) wrapper)"
	@echo "  test         Test $(APP_PKG)"
	@echo "  test-gpu     Run a GPU test lane via $(RUN_WITH) (P= T= A= RELEASE=)"
	@echo "  test-primitives-baseline Run the existing rigid primitive CPU regressions"
	@echo "  test-primitives-baseline-gpu Run existing GPU primitives and scalar equality regressions"
	@echo "  test-primitives-runtime-gpu Run live GPU runtime and retained-resource regressions"
	@echo "  test-primitives-distance-scale-gpu Check 1000 articulated bodies, stages and one-second parity"
	@echo "  test-primitives-distance-ten-second-cpu/gpu 40 scale cases for 10s (PRIMITIVE_DISTANCE_HZ=60/120/240)"
	@echo "  test-primitives-cone-ten-second-cpu/gpu Active impacts and energy for 10s (PRIMITIVE_CONE_HZ=60/120/240)"
	@echo "  test-primitives-friction-ten-second-cpu/gpu Four loaded static-hold cases for 10s (PRIMITIVE_FRICTION_HZ=60/120/240; wall budget 120/120/240s)"
	@echo "  example-primitives Run public CPU primitive examples (A=case)"
	@echo "  example-primitives-gpu Run public GPU primitive examples (A=case)"
	@echo "  test-sensors Run sensor host, feature and explicit GPU regression lanes"
	@echo "  test-viewer-sensors Run headless viewer sensor lifecycle tests on GPU (RELEASE= A=)"
	@echo "  test-all     Run the full workspace all-target test suite"
	@echo "  check        Check $(APP_PKG)"
	@echo "  check-all    Check the full workspace all-target suite"
	@echo "  fmt-files    Format explicit Rust paths (FILES= FMT_ARGS=--check)"
	@echo "  fmt-check    Check formatting across the workspace"
	@echo "  clippy-check Run strict all-target/all-feature workspace clippy"
	@echo "  check-primitives-shader-wiring Parse/validate primitive WGSL on CPU"
	@echo "  check-primitives-gpu-wiring    Prepare GPU worlds, no physics frames (120s limit)"
	@echo "  check-primitives-gpu-step-wiring Compare three one-frame CPU/GPU fixtures (120s limit)"
	@echo "  test-primitives-singular-wiring-cpu/gpu Check both active XYZ pole rows (GPU 120s per original pole)"
	@echo "  test-primitives-singular-pole-cpu/gpu Isolated original pole (PRIMITIVE_XYZ_POLE=positive/negative; 120s)"
	@echo "  test-primitives-chart-wiring-cpu/gpu Cross both XYZ poles and stop at pitch bounds over 1s (GPU 120s per pole)"
	@echo "  test-primitives-chart-pole-cpu/gpu One-second original pole trajectory (PRIMITIVE_XYZ_POLE=positive/negative; 120s)"
	@echo "  test-primitives-angular-lock-wiring-cpu/gpu Mixed Euler locks and physical tangent rates (120s limit each)"
	@echo "  test-primitives-angular-mass-wiring-cpu/gpu Full-inertia angular locks and reactions (120s limit each)"
	@echo "  test-primitives-angular-bound-wiring-cpu/gpu Finite hinge stops and small-step reactions (120s limit each)"
	@echo "  test-primitives-angular-regular-wiring-cpu/gpu Principal/alternate XYZ point reactions (120s limit each)"
	@echo "  test-primitives-angular-regular-standalone-cpu/gpu Raw principal/alternate XYZ acceleration oracle (120s limit each)"
	@echo "  test-primitives-d6-activation-cpu Structural points versus coupled bounds/caps"
	@echo "  test-primitives-d6-interval-standalone-cpu/gpu All-axis narrow regular interval/cap response (120s limit each)"
	@echo "  test-primitives-d6-interval-runtime-cpu/gpu Public runtime interval/cap reaction oracle (120s limit each)"
	@echo "  test-primitives-angular-phase-standalone-cpu/gpu Raw narrow singular-phase acceleration oracle (120s limit each)"
	@echo "  test-primitives-angular-narrow-wiring-cpu/gpu Sub-f32 interval metadata and reactions (120s limit each)"
	@echo "  test-primitives-limit-reference-cpu Authored bound references and refresh/resize checks"
	@echo "  test-primitives-angular-narrow-standalone-cpu/gpu Raw sub-f32 hinge acceleration oracle (120s limit each)"
	@echo "  test-primitives-scalar-precision-cpu/gpu Raw sub-f32 prismatic/revolute interval oracle (120s limit each)"
	@echo "  test-primitives-d6-point-publication-cpu/gpu Authored fixed D6 state and bound-edit residues (120s limit each)"
	@echo "  test-primitives-pitch-pole-cpu/gpu Raw/runtime narrow pitch bounds near both poles (120s limit each)"
	@echo "  test-primitives-radial-precision-cpu/gpu Coupled radial bounds, retained motion and independent inertia (120s limit each)"
	@echo "  test-primitives-distance-activation-cpu Authored radial/cap coupled-row activation"
	@echo "  test-primitives-ball-point-publication-cpu/gpu Authored fixed Ball rotations and bound-edit residues (120s limit each)"
	@echo "  test-primitives-ball-angular-cpu/gpu Raw/runtime Ball partial-limit/cap response and reactions (120s limit each)"
	@echo "  test-primitives-redundant-limits-cpu/gpu Loaded redundant Ball/D6/radial points and zero caps (120s limit each)"
	@echo "  test-primitives-redundant-raw-gpu Raw loaded hold and permitted-motion regressions only (120s limit)"
	@echo "  test-primitives-redundant-runtime-cpu/gpu Public redundant points/caps, permitted motion and reactions (120s limit each)"
	@echo "  test-primitives-redundant-runtime-mobile-gpu Public permitted-motion redundant rows (120s limit)"
	@echo "  test-primitives-redundant-runtime-hold-gpu Public sustained redundant rows and reaction gates (120s limit)"
	@echo "  test-primitives-redundant-runtime-rollback-gpu Incompatible radial/cap rejection without partial state (120s limit)"
	@echo "  test-primitives-radial-cap-stages-gpu Isolated runtime FK refresh and radial/cap device regression (120s limit)"
	@echo "  test-primitives-frame-reference-cpu Authored frame tails, mutation, dimensions and zero-joint metadata"
	@echo "  test-primitives-frame-reference-gpu Paired mixed/fixed chains and authored frame upload edits (120s limit)"
	@echo "  test-primitives-all-axis-sustained-cpu/gpu 144 all-axis modes, COM/tensor mass sweep, 1-second motion (120s limit)"
	@echo "  test-primitives-boundary-radial-cpu/gpu Fixed/kinematic boundary poses, radial/Cartesian caps and independent motion (120s limit)"
	@echo "  test-primitives-quaternion-motion-cpu/gpu Retained Ball/Distance/Free/D6 motion without reseeding (120s limit each)"
	@echo "  test-primitives-angular-phase-wiring-cpu/gpu Singular phase/wrapped-pole reactions (120s limit each)"
	@echo "  test-primitives-hinge-antipode-wiring-cpu/gpu Finite hinge cut-locus recovery (120s limit each)"
	@echo "  test-primitives-hinge-antipode-standalone-cpu/gpu Raw finite hinge mass response (120s limit each)"
	@echo "  test-primitives-hinge-antipode-compensated-gpu Raw retained-coordinate hinge response (120s limit)"
	@echo "  test-primitives-disabled-wiring-cpu/gpu Disabled relative motion and child-load transmission (120s limit each)"
	@echo "  test-primitives-velocity-wiring-cpu/gpu Check strongly coupled velocity drives (120s limit each)"
	@echo "  test-primitives-velocity-sustained-cpu/gpu One-second Ball/D6/Free coupled drives and live control edits (120s)"
	@echo "  test-primitives-velocity-raw-gpu Raw coupled roots with coordinate compensation disabled (120s)"
	@echo "  test-primitives-motor-coupling-cpu/gpu Loaded mixed-model hinge/slider chains (GPU 120s per feedback mode)"
	@echo "  test-primitives-motor-feedback-gpu One original feedback mode (PRIMITIVE_MOTOR_FEEDBACK=explicit/stable)"
	@echo "  test-primitives-mixed-drives-cpu/gpu Simultaneous linear/angular drives, active friction, COM/tensor and edits (CPU120s/GPU180s)"
	@echo "  test-primitives-locked-drives-gpu 48 coupled Ball/D6 partial-lock chains, all axes/frames and CPU parity (180s)"
	@echo "  test-primitives-locked-drive-planes-cpu/gpu 24 noncoaxial two-free-axis chains, initial mobility oracle and hot controls (GPU180s)"
	@echo "  harden       Run diff whitespace check + fmt/check + strict clippy + all-feature tests"
	@echo "  bench        Run benchmarks (BENCH_ARGS=)"
	@echo "  bench-primitives Run the live CPU/GPU primitive scale matrix (A=)"
	@echo "  test-primitives-benchmark-smoke Run all primitive fixtures on CPU/GPU at smoke scale"
	@echo "  bench-sensors Run GPU sensor benchmarks (SENSOR_BENCH=bench_scene_queries A=)"
	@echo "  bench-sensors-latency Record per-frame latency matrix with build/device metadata"
	@echo "  sensor-bench-info Print source fingerprint, toolchain and device metadata"
	@echo "  docs         Build documentation with mdbook (DOCS_DIR=)"
	@echo "  release      Create a new release (TYPE=patch|minor|major|m.m.p)"
	@echo "  clean        Remove Cargo build artifacts"
	@echo
	@echo "Examples:"
	@echo "  make run"
	@echo "  make run APP_PKG=other        # run a different package"
	@echo "  make run BACKEND=x11          # force X11 / XWayland (.envrc auto-detects)"
	@echo "  make run RUN_WITH=nixGL       # OpenGL wrapper instead of Vulkan"
	@echo "  make run RUN_WITH=            # no wrapper (native run)"
	@echo "  make test-gpu P=molla-geometry T=gjk_gpu A=--nocapture"
	@echo "  make bench-sensors A=scene_queries_1024_rays/hierarchy.*warm"
	@echo "  make bench-sensors SENSOR_BENCH=bench_sensor_lidar A=--test"
	@echo

h: help
