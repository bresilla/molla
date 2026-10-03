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
