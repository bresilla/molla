//! GPU parity checklist + approximation ledger.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.
//! `tire_runtime_production_gate_report` stays in `mod.rs` for now
//! because it pulls in ~30 scattered `tire_*_report` /
//! `tetra_*_report` functions; once those reports move out it can
//! join this module.

use super::internal::csv_token;

/// CPU/GPU implementation state for one high-frequency tire feature.
///
/// This is a non-calibration checklist: it does not prove that a tire matches
/// measured data, but it prevents us from accidentally hiding production
/// tire work in a CPU-only path after the GPU-first rule was agreed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TireGpuParityStatus {
    CpuAndGpu,
    CpuOnlyDebug,
    PlannedGpu,
}

impl TireGpuParityStatus {
    pub const fn label(self) -> &'static str {
        match self {
            Self::CpuAndGpu => "cpu_and_gpu",
            Self::CpuOnlyDebug => "cpu_only_debug",
            Self::PlannedGpu => "planned_gpu",
        }
    }

    pub const fn has_gpu_path(self) -> bool {
        matches!(self, Self::CpuAndGpu)
    }

    pub const fn is_cpu_only_debug(self) -> bool {
        matches!(self, Self::CpuOnlyDebug)
    }
}

/// One row in the tire GPU parity checklist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TireGpuParityItem {
    pub feature_id: &'static str,
    pub status: TireGpuParityStatus,
    pub cpu_path: &'static str,
    pub gpu_path: &'static str,
    pub proof: &'static str,
    pub reason: &'static str,
    pub follow_up: &'static str,
}

impl TireGpuParityItem {
    pub const fn has_required_gpu_accounting(self) -> bool {
        match self.status {
            TireGpuParityStatus::CpuAndGpu => !self.gpu_path.is_empty() && !self.proof.is_empty(),
            TireGpuParityStatus::CpuOnlyDebug | TireGpuParityStatus::PlannedGpu => {
                !self.reason.is_empty() && !self.follow_up.is_empty()
            }
        }
    }

    pub fn to_text_line(self) -> String {
        format!(
            "{} status={} cpu={} gpu={} proof={} reason={} follow_up={}",
            self.feature_id,
            self.status.label(),
            csv_token(self.cpu_path),
            csv_token(self.gpu_path),
            csv_token(self.proof),
            csv_token(self.reason),
            csv_token(self.follow_up)
        )
    }
}

/// Current GPU parity checklist for the reusable real-tire stack.
///
/// Keep this list near the reusable tire API instead of in the viewer so
/// scene labels, rig reports, and tests all agree on which tire features are
/// GPU-backed, debug-only, or still planned.
pub fn tire_gpu_parity_checklist() -> Vec<TireGpuParityItem> {
    vec![
        TireGpuParityItem {
            feature_id: "brush_combined_slip",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "evaluate_brush_tire / evaluate_brush_tire_oriented",
            gpu_path: "molla-kernels::TireBrushGpuKernel / tire_brush.wgsl",
            proof: "make test-gpu P=molla-kernels T=tire_brush_gpu A=--nocapture plus CPU structural_lateral_bias conicity regression and shader source guard",
            reason: "",
            follow_up: "",
        },
        TireGpuParityItem {
            feature_id: "brush_camber_pneumatic_trail",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "RealWheelOrientation + evaluate_brush_tire_oriented camber pressure bias, camber thrust, contact-centre/resultant, pneumatic trail, and aligning moment telemetry",
            gpu_path: "tire_brush.wgsl extra.w camber_angle_rad with diag3 camber/contact-center/pneumatic-trail output",
            proof: "camber_thrust_flips_sign_and_reports_pneumatic_trail, tire_rig_camber_sweep_flips_lateral_force_sign, tire_rig_camber_sweep_bounds_moment_trail_and_sliding_saturation, and ignored hardware parity lane `make test-gpu P=molla-kernels T=tire_brush_gpu A='tire_brush_gpu_matches_cpu_camber_extension --nocapture'`",
            reason: "",
            follow_up: "",
        },
        TireGpuParityItem {
            feature_id: "brush_heightfield_material_normal",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "Terrain::friction_at / normal / water_film_depth in brush cells",
            gpu_path: "tire_brush.wgsl heightfield normal and friction-map sampling",
            proof: "tire_brush_gpu heightfield normal/world-force and friction-map parity",
            reason: "",
            follow_up: "",
        },
        TireGpuParityItem {
            feature_id: "terrain_surface_velocity_map",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "HeightfieldTerrain::with_surface_velocity_map sampled through Terrain::surface_velocity before brush/FEM contact",
            gpu_path: "TireFemContactFeedbackKernel / tire_fem_contact_feedback.wgsl optional spatial surface_velocity_map sampled per tread node",
            proof: "molla-vehicle heightfield_surface_velocity_map report plus tire_fem_gpu spatial surface-velocity-map feedback tests and standard-runner replay snapshot velocity export",
            reason: "",
            follow_up: "add direct GPU brush heightfield surface-velocity-map parity if brush-only terrain velocity maps become a hot path outside FEM feedback",
        },
        TireGpuParityItem {
            feature_id: "tetra_fem_node_step",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "TetraTireFem::step / step_on_terrain",
            gpu_path: "molla-kernels::TireFemGpuKernel / tire_fem.wgsl",
            proof: "make test-gpu P=molla-kernels T=tire_fem_gpu A=--nocapture",
            reason: "",
            follow_up: "",
        },
        TireGpuParityItem {
            feature_id: "tetra_fem_persistent_batch",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "Vec<TetraTireFem> vehicle/viewer stepping",
            gpu_path: "molla-kernels::TireFemGpuBatch persistent multi-tire API plus TireFemGpuAtlasBatch resident fused-atlas owner / tire_fem_atlas.wgsl",
            proof: "tire_fem_gpu_persistent_batch_steps_multiple_tires, tire_fem_batch_dispatch_plan_accounts_for_fused_atlas_scaling, tire_fem_batch_atlas_buffers_rewrite_topology_indices_for_fused_dispatch source guards, resident atlas node/state/hub-row reseed APIs, fused hub_reduce_atlas / step_resident_atlas_hubs path, and ignored tire_fem_batch_atlas_contract_dispatch_routes_packed_rows_on_gpu resident atlas hardware lane",
            reason: "",
            follow_up: "extend the Scene 120 combined resident-owner adoption into the richer live vehicle scenes after deeper live vehicle validation on an available wgpu adapter",
        },
        TireGpuParityItem {
            feature_id: "tetra_fem_hub_reduction",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "TetraTireFem bead reaction / hub state",
            gpu_path: "tire_fem_hub.wgsl persistent hub reduction plus tire_fem_atlas.wgsl hub_reduce_atlas resident fused-atlas hub reduction and reduced-contact world-force external load handoff; tire_fem_hub_params_with_brush_outputs maps GPU brush per-wheel world-force rows into FEM hub external-force params for resident atlas hub stepping; update_resident_atlas_hub_params_from_brush_output / tire_fem_atlas.wgsl::hub_params_from_brush_outputs update resident atlas hub-param rows directly from resident brush output buffers; update_resident_atlas_hub_params_from_contact_reduction_output / tire_fem_atlas.wgsl::hub_params_from_contact_reduction_outputs update resident atlas hub-param rows directly from packed FEM contact-reduction world-force rows; step_resident_atlas_hubs_with_resident_params, step_resident_atlas_hubs_from_brush_output, and step_resident_atlas_hubs_from_contact_reduction_output integrate those GPU-assembled rows without host overwrite; step_resident_vehicle_fem_hubs_from_brush_owner sequences resident vehicle brush output into resident FEM hub stepping; step_resident_vehicle_fem_atlas_from_brush_owner schedules a coherent resident FEM node + vehicle/brush + hub substep without CPU brush staging; TireResidentVehicleFemAtlasOwner plus create_resident_vehicle_fem_atlas_owner / step_resident_vehicle_fem_atlas_owner bundle the resident vehicle/chassis/brush owner and resident FEM atlas behind one long-lived runtime owner API, and Scene 120 now defaults to that combined owner with raw resident-atlas/per-tire persistent fallbacks; CPU Vehicle yaw equation consumes tire Mz and TireVehicleYawGpuKernel owns the first GPU yaw integration lane",
            proof: "tire_fem_gpu persistent-buffer hub tests, ignored resident atlas hardware lane exercising step_resident_atlas_hubs, vehicle::tests::tire_aligning_moment_couples_into_chassis_yaw, reduced-world-wrench hub-force adapter/shader guard, brush-output-to-hub adapter test, ignored tire_fem_resident_atlas_hub_params_from_brush_output_gpu resident writer hardware lane, ignored tire_fem_resident_atlas_hubs_step_from_brush_output_gpu resident writer+hub integration hardware lane, ignored tire_fem_resident_atlas_hubs_step_from_contact_reduction_output_gpu packed contact-reduction writer+hub integration hardware lane, ignored tire_fem_resident_vehicle_fem_atlas_substep_sequences_nodes_brush_and_hubs_gpu coherent substep hardware lane, ignored tire_fem_resident_vehicle_fem_atlas_owner_steps_gpu combined resident owner hardware lane, and tire_vehicle_gpu CPU/GPU yaw parity lane",
            reason: "",
            follow_up: "extend the Scene 120 combined resident-owner adoption into full live vehicle scenes with position, velocity, suspension and wheel assemblies all owned by resident GPU state",
        },
        TireGpuParityItem {
            feature_id: "vehicle_chassis_yaw_coupling",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "Vehicle::step sums off_x * Fz - off_z * Fx plus tire aligning moment Mz into chassis yaw",
            gpu_path: "TireVehicleYawGpuKernel / tire_vehicle_yaw.wgsl sums per-wheel tire force xyz + Mz and integrates yaw/yaw-rate on GPU",
            proof: "vehicle::tests::tire_aligning_moment_couples_into_chassis_yaw plus tire_vehicle_gpu reference/source guard and ignored hardware parity lane (`make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_yaw_gpu_matches_reference --nocapture'`)",
            reason: "",
            follow_up: "promote the planar GPU chassis-state kernel into a resident full vehicle/chassis owner once suspension and wheel state are GPU-owned",
        },
        TireGpuParityItem {
            feature_id: "vehicle_chassis_planar_coupling",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "Vehicle::step integrates world X/Z tire forces into chassis velocity/position and yaw moment into yaw/yaw-rate",
            gpu_path: "TireVehicleYawGpuKernel::dispatch_planar_blocking / tire_vehicle_yaw.wgsl integrate_planar sums per-wheel tire force xyz + Mz and integrates planar position, velocity, yaw, and yaw-rate on GPU; TireBrushGpuKernel::dispatch_input_buffer_resident plus dispatch_planar_from_brush_output_blocking / integrate_planar_from_brush_outputs consumes GPU-resident BrushWheelOutput rows directly; dispatch_planar_from_contact_reduction_output_blocking / integrate_planar_from_contact_reduction_outputs consumes packed GPU FEM contact-reduction rows directly, using row-2 world_force and row-3 world_moment.y; tire_vehicle_planar_from_feedback_output_reference and step_planar_resident_from_feedback_output / integrate_planar_state_from_feedback_force_rows consume resident FEM contact-feedback force rows directly by summing each tire's atlas node-force range; TireVehiclePlanarResidentState / integrate_planar_state_from_brush_outputs, integrate_planar_state_from_contact_reduction_outputs, and integrate_planar_state_from_feedback_force_rows keep planar chassis position/velocity/yaw rows resident with ping-pong GPU state buffers; TireVehicleResidentChassisBrushOwner sequences resident wheel spin, suspension, brush-row generation, resident brush output, and resident planar chassis integration without staging brush output through CPU; step_resident_chassis_brush_owner_from_planar_state / build_brush_rows_from_spin_suspension_planar_state feeds resident planar X/Z/yaw and velocity state back into the brush-row builder while vertical chassis state remains externally supplied; step_resident_chassis_brush_owner_from_resident_state / build_brush_rows_from_spin_suspension_chassis_state feeds both resident planar and resident vertical chassis state into brush-row generation",
            proof: "tire_vehicle_planar_reference_integrates_force_translation_and_yaw, tire_vehicle_planar_from_brush_reference_uses_world_force_and_aligning_moment, tire_vehicle_planar_from_contact_reduction_reference_uses_world_force_and_moment, and tire_vehicle_planar_from_feedback_reference_uses_atlas_force_rows source guards plus ignored hardware parity lanes (`make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_planar_gpu_matches_reference --nocapture'`, `make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_planar_contact_reduction_gpu_matches_reference --nocapture'`, `make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_resident_brush_output_feeds_planar_gpu_without_staging --nocapture'`, `make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_planar_resident_state_ping_pongs_from_brush_output --nocapture'`, `make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_planar_resident_state_ping_pongs_from_contact_reduction_output --nocapture'`, `make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_planar_resident_state_ping_pongs_from_feedback_output --nocapture'`, `make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_resident_chassis_brush_owner_sequences_brush_and_planar_state --nocapture'`, `make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_resident_chassis_brush_owner_feeds_planar_state_back_to_brush_rows --nocapture'`, and `make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_resident_chassis_brush_owner_uses_resident_planar_and_vertical_state --nocapture'`)",
            reason: "",
            follow_up: "wire the resident planar chassis-state owner into the full resident vehicle runtime and FEM tire assembly",
        },
        TireGpuParityItem {
            feature_id: "vehicle_wheel_spin_coupling",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "Vehicle::step updates each wheel spin from drive_torque - brake_torque*tanh(spin) - tire_fx*wheel_radius",
            gpu_path: "TireVehicleYawGpuKernel::dispatch_wheel_spin_blocking / tire_vehicle_yaw.wgsl integrate_wheel_spin updates per-wheel spin, acceleration, and net torque on GPU; TireVehicleWheelSpinResidentBatch / integrate_wheel_spin_state keeps spin/radius/inertia rows resident with ping-pong GPU state buffers; dispatch_brush_rows_from_spin_resident feeds that resident spin buffer into GPU brush rows without staging spin back through CPU",
            proof: "tire_vehicle_wheel_spin_reference_matches_vehicle_equation and tire_vehicle_wheel_spin_state_reference_preserves_resident_state_layout source guards plus ignored hardware parity lanes (`make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_wheel_spin_gpu_matches_reference --nocapture'` and `make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_wheel_spin_resident_batch_ping_pongs_state --nocapture'`)",
            reason: "",
            follow_up: "connect the resident wheel-spin state buffer to FEM tire outputs and the planar chassis owner without CPU staging",
        },
        TireGpuParityItem {
            feature_id: "vehicle_wheel_contact_kinematics",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "Vehicle::step rotates chassis corner offsets, applies yaw-rate velocity, rotates by yaw+steer, and computes v_long/v_lat/slip_ratio/slip_angle",
            gpu_path: "TireVehicleYawGpuKernel::dispatch_wheel_kinematics_blocking / tire_vehicle_yaw.wgsl compute_wheel_kinematics computes per-wheel contact-frame basis and slip inputs on GPU; dispatch_wheel_kinematics_from_spin_resident_blocking / compute_wheel_kinematics_from_spin_state consumes TireVehicleWheelSpinResidentBatch directly without staging spin rows through CPU",
            proof: "tire_vehicle_wheel_kinematics_reference_matches_vehicle_contact_frame and tire_vehicle_wheel_kinematics_from_spin_reference_uses_resident_spin_state source guards plus ignored hardware parity lanes (`make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_wheel_kinematics_gpu_matches_reference --nocapture'` and `make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_resident_spin_state_feeds_kinematics_gpu_without_staging --nocapture'`)",
            reason: "",
            follow_up: "feed these resident GPU slip/frame rows directly into the GPU brush/FEM tire kernels from a resident vehicle owner",
        },
        TireGpuParityItem {
            feature_id: "vehicle_suspension_normal_load",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "Vehicle::step computes suspension compression/travel/rate and normal load from chassis height, wheel radius, terrain height, spring stiffness, and damping",
            gpu_path: "TireVehicleYawGpuKernel::dispatch_suspension_blocking / tire_vehicle_yaw.wgsl compute_suspension computes per-wheel travel, travel rate, and normal load on GPU; TireVehicleSuspensionResidentBatch keeps geometry/parameter/output rows resident and dispatch_brush_rows_from_spin_suspension_resident feeds resident suspension travel directly into GPU brush rows; TireVehicleVerticalResidentState / integrate_vertical_state_from_suspension keeps chassis y/vy resident and compute_suspension_from_vertical_state feeds that resident vertical state back into suspension normal-load rows",
            proof: "tire_vehicle_suspension_reference_matches_vehicle_spring_damper, tire_vehicle_vertical_reference_integrates_suspension_load_against_gravity, and tire_vehicle_brush_rows_from_spin_suspension_reference_uses_resident_travel source guards plus ignored hardware parity lanes (`make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_suspension_gpu_matches_reference --nocapture'`, `make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_vertical_resident_state_ping_pongs_from_suspension_rows --nocapture'`, and `make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_resident_spin_and_suspension_feed_brush_gpu_without_staging --nocapture'`)",
            reason: "",
            follow_up: "feed resident suspension/normal-load rows into FEM tire force kernels and the full resident vehicle owner",
        },
        TireGpuParityItem {
            feature_id: "vehicle_to_brush_gpu_handoff",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "Vehicle/brush setup builds BrushWheelInput from chassis pose/velocity, wheel mounts, steering/spin, suspension travel, tire params, material rows, and terrain height",
            gpu_path: "TireVehicleYawGpuKernel::dispatch_brush_rows_blocking / dispatch_brush_rows_resident / dispatch_brush_rows_from_spin_resident / dispatch_brush_rows_from_spin_suspension_resident / TireVehicleResidentBrushOwner / tire_vehicle_yaw.wgsl build_brush_rows, build_brush_rows_from_spin_state, and build_brush_rows_from_spin_and_suspension_state emit BrushWheelInput-compatible seven-vec4 rows that TireBrushGpuKernel::dispatch_input_buffer_blocking or dispatch_input_buffer_resident can consume without CPU staging; TireBrushGpuKernel::readback_resident_output provides the narrow viewer telemetry seam; Scene 121 now uses the live TireVehicleYawGpuKernel → TireBrushGpuKernel resident vehicle-to-brush row handoff as its primary GPU force path after transient relaxation, with direct GPU brush and CPU fallback paths retained for parity",
            proof: "tire_vehicle_brush_rows_reference_builds_gpu_brush_inputs, tire_vehicle_brush_rows_from_spin_reference_uses_resident_spin_state, and tire_vehicle_brush_rows_from_spin_suspension_reference_uses_resident_travel source guards, tire_vehicle_brush_rows_gpu_matches_reference row parity lane, Scene 121 source guard `tire_traction_scene_wires_resident_vehicle_to_brush_handoff`, resident vehicle-to-brush hardware chain (`make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_brush_rows_resident_feed_brush_gpu_matches_staged_rows --nocapture'`), resident spin-to-brush hardware chain (`make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_resident_spin_state_feeds_brush_gpu_without_staging --nocapture'`), resident spin+suspension-to-brush hardware chain (`make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_resident_spin_and_suspension_feed_brush_gpu_without_staging --nocapture'`), and resident brush-owner hardware chain (`make test-gpu P=molla-kernels T=tire_vehicle_gpu A='tire_vehicle_resident_brush_owner_sequences_spin_suspension_and_brush --nocapture'`)",
            reason: "",
            follow_up: "move the same primary resident vehicle-to-brush pattern into broader resident chassis-state runtime paths and extend the direct handoff pattern to FEM tire assemblies",
        },
        TireGpuParityItem {
            feature_id: "legacy_vehicle_real_brush_adapter",
            status: TireGpuParityStatus::CpuOnlyDebug,
            cpu_path: "RealBrushTireModel implements the legacy TireModel trait by adapting slip/load/friction into evaluate_brush_tire on a synthetic flat patch; TireModelLab and Scene::VehicleOffroad use it for CPU viewer/runtime bridge coverage",
            gpu_path: "",
            proof: "real_brush_tire_model_adapts_native_brush_contact_to_vehicle_trait plus viewer source guards tire_model_lab_includes_native_real_brush_curve and vehicle_offroad_scene_uses_real_brush_tire_adapter, including VehicleOffroad truth-label/control-category coverage that marks the adapter as CPU-only legacy bridge coverage",
            reason: "the legacy TireModel trait does not carry full wheel pose, terrain maps, suspension rows, or resident GPU buffers; it is an adoption bridge so older scenes use the native brush law rather than the old simple tire",
            follow_up: "replace this CPU bridge in remaining vehicle scenes with resident vehicle-to-brush/FEM owners once those scenes own planar, vertical, wheel-spin, suspension, brush, and FEM state on GPU",
        },
        TireGpuParityItem {
            feature_id: "tetra_fem_xpbd_structural",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "TetraTireFem XPBD volume/bead/tread-belt/sidewall edge projection",
            gpu_path: "tire_fem.wgsl gather-style XPBD volume/bead/structural projection",
            proof: "tire_fem_gpu XPBD volume/bead/structural parity tests",
            reason: "",
            follow_up: "",
        },
        TireGpuParityItem {
            feature_id: "tetra_fem_sidewall_dihedral_xpbd",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "TetraTireFem sidewall quad dihedral XPBD projection",
            gpu_path: "tire_fem.wgsl gather-style sidewall dihedral XPBD projection",
            proof: "molla-vehicle tetra_sidewall_dihedral unit test plus tire_fem_gpu_sidewall_dihedral_projection_reduces_fold_angle",
            reason: "",
            follow_up: "replace or validate the dihedral projection against a laminated sidewall constraint if needed",
        },
        TireGpuParityItem {
            feature_id: "tetra_fem_sidewall_laminate_coupling_report",
            status: TireGpuParityStatus::CpuOnlyDebug,
            cpu_path: "TetraSidewallLaminateCouplingReport CPU fixture gates crossed bias-ply energy balance, radial/axial helper-ply activation, folded ply energy, and XPBD fold reduction",
            gpu_path: "",
            proof: "molla-vehicle sidewall_laminate_coupling_report_exercises_ply_energy_and_fold_projection plus standard non-calibration runner snapshot sidewall_laminate_coupling and viewer truth-label panel source guard",
            reason: "this is an executable diagnostic/fixture over the compact tetra ply stack; the force-producing four-family fibre stress and dihedral XPBD projection already have CPU/GPU paths, but the diagnostic reduction itself is not a production GPU kernel",
            follow_up: "turn the sidewall laminate coupling fixture into a force-producing laminated shell/carcass constraint, then add GPU parity for that constraint instead of only the CPU diagnostic report",
        },
        TireGpuParityItem {
            feature_id: "tetra_fem_sidewall_ply_strain_xpbd",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "project_sidewall_ply_strain_constraints applies a tensile |F*a0| XPBD envelope to sidewall tetra ply families and TetraSidewallPlyStrainXpbdReport gates the effect",
            gpu_path: "TireFemGpuParams::xpbd_sidewall_ply_stretch_limit / xpbd_sidewall_ply_compliance plus tire_fem.wgsl and tire_fem_atlas.wgsl Stage (f4) gather projection over incident sidewall ply families",
            proof: "molla-vehicle sidewall_ply_strain_xpbd_report_reduces_tensile_ply_overstretch plus standard non-calibration runner snapshot sidewall_ply_strain_xpbd, viewer truth-label panel source guard, molla-kernels tire_fem_gpu_shaders_include_sidewall_ply_strain_xpbd_guardrail, and ignored hardware lane `make test-gpu P=molla-kernels T=tire_fem_gpu A='tire_fem_gpu_sidewall_ply_strain_projection_reduces_overstretch --ignored --nocapture'`",
            reason: "",
            follow_up: "replace the compact ply-strain guardrail with a full laminated sidewall shell/carcass constraint once the shell model lands",
        },
        TireGpuParityItem {
            feature_id: "tetra_fem_semi_implicit_stiffness",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "TetraTireFemParams::semi_implicit_stiffness_scale CPU velocity denominator",
            gpu_path: "TireFemGpuParams::semi_implicit_stiffness_scale / tire_fem.wgsl denominator",
            proof: "molla-vehicle tetra_semi_implicit... plus tire_fem_gpu_semi_implicit_stiffness_damps_velocity_spike",
            reason: "",
            follow_up: "",
        },
        TireGpuParityItem {
            feature_id: "tetra_fem_torque_api",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "TetraTireFem::set_drive_torque_n_m / set_brake_torque_n_m / hub_omega / hub_angle / hub_is_dynamic / set_hub_dynamic / set_hub_position / set_hub_velocity / last_hub_reaction_xyz integrated through step_inner — drive/brake torque accelerates hub_omega, spin couples to tread bristle slip, and contact reaction integrates hub_v_xz when the hub is dynamic so the wheel physically rolls forward",
            gpu_path: "molla-kernels TireFemGpuParams carries hub_omega/drive_torque_n_m/brake_torque_n_m, TireFemTorqueGpuContract pins the uniform-field names + brake soft-sign shoulder + hub-mass clamp, tire_fem.wgsl reads the new phys5 row and adds the spin contribution `v_spin = ω·r_y / −ω·r_x` to the XPBD contact tangential motion with the same sign convention as the CPU step_inner. TireFemTorqueGpuStatus auto-advances to LiveValidated now that the WGSL phys5 row + spin-contact code are bundled and the three-leg CPU/GPU parity lane (node FEM, dynamic hub, contact-driven rolling with sub-cm hub agreement) passes on hardware",
            proof: "molla-vehicle tetra_tire_fem_torque_api_default_state_is_at_rest, tetra_tire_fem_torque_setters_clamp_nonfinite_and_negative_brake, tetra_tire_fem_drive_torque_spins_the_hub, tetra_tire_fem_brake_torque_decays_spin_toward_zero, tetra_tire_fem_spin_creates_backward_tread_slip_at_contact, tetra_tire_fem_dynamic_hub_default_is_kinematic, tetra_tire_fem_drive_torque_rolls_dynamic_hub_forward, tetra_tire_fem_kinematic_hub_ignores_contact_reaction, tetra_tire_fem_chassis_handoff_setters_pin_pose_and_velocity, tetra_tire_fem_zero_torque_default_state_does_not_disturb_existing_scenes; molla-kernels tire_fem_torque_contract source guards plus the ignored hardware lane `make test-gpu P=molla-kernels T=tire_fem_gpu A='tire_fem_gpu_spin_pushes_grounded_tread_backward_in_contact --nocapture'` which dispatches tire_fem.wgsl twice (omega=0 vs omega=80) and asserts the spin push appears in −X on the contact node; viewer scene 123 (TerrainLab) drives the FEM tire with this API end-to-end",
            reason: "PLAN_TOURQUE Stages A-F land the torque/spin DOF + dynamic-hub integration on the CPU; Stage G parts 1+2+3 mirror the spin contribution in tire_fem.wgsl and validate it on real hardware so the GPU contact path can be torque-driven without leaving the explicit CPU integrator out of sync. True GPU-side dynamic-hub XZ integration (atomic contact-reaction accumulation) is a further refinement; today the host integrates the hub via TetraTireFem and feeds hub_omega forward to the GPU each frame",
            follow_up: "PLAN_REALTIRE Stream C G4 closed the CPU-vs-GPU full-step parity lane: tire_fem_cpu_vs_gpu_rolling_contact_hub_x_parity drives both backends with identical torque inputs through step_persistent + contact feedback + hub kernel and confirms sub-cm hub trajectory agreement. The wrench hand-off is also on-device now: hub_reduce_with_tread_wrench in tire_fem_hub.wgsl reduces the contact-feedback force rows inside the hub dispatch (step_hub_persistent_with_tread_forces), validated against the host-summed path by tire_fem_gpu_hub_tread_wrench_reduction_matches_host_sum. Remaining enhancement: feed the feedback kernel its inputs from the persistent mesh buffers so the whole substep loop runs without any host readback",
        },
        TireGpuParityItem {
            feature_id: "tetra_fem_newton_solver_debug",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "TetraFemDiagonalNewtonReport, TetraFemElementNewtonReport, TetraFemPatchNewtonReport, TetraFemTinyMeshNewtonReport, TetraFemMatrixFreeNewtonScaleReport, TetraFemMatrixFreeNewtonSolveReport, TetraFemMatrixFreeNewtonPreconditionerReport, TetraFemMatrixFreeNewtonStepReport, TetraTireFem::step_on_terrain_matrix_free_newton_debug, shared matrix_free_newton_correction helper, and the TetraTireFemSolverMode::MatrixFreeNewtonDebug opt-in plus TetraTireFemParams::high_pressure_implicit default-flipped preset routing step_on_terrain through the matrix-free implicit timestep",
            gpu_path: "tire_fem_matrix_free_newton_gpu_contract / TireFemMatrixFreeNewtonGpuContract host-side dispatch contract, tire_fem_matrix_free_newton.wgsl WGSL kernel exposing residual_eval/jv_probe/diagonal_extract/apply_correction entry points, TireFemMatrixFreeNewtonGpuKernel that compiles the shader and dispatches all four entry points end-to-end with readback, and TireFemMatrixFreeNewtonGpuKernel::run_newton_outer_blocking host-orchestrated GMRES + Jacobi + line search outer iteration that drives the GPU primitives (status: Live, parity hardware lanes for both residual oracle and outer-iteration residual reduction)",
            proof: "molla-vehicle matrix-free Newton report tests, TetraTireFemSolverMode dispatch test, tetra_tire_fem_matrix_free_newton_debug_runs_across_named_presets scene-coverage validation across realtime_viewer/offroad_soft/high_pressure_reference presets, tetra_tire_fem_params_high_pressure_implicit_defaults_to_matrix_free_newton_mode default-flip preset test, standard non-calibration runner snapshots fem_diagonal_newton_solver, fem_element_newton_solver, fem_patch_newton_solver, fem_tiny_mesh_newton_solver, fem_matrix_free_newton_scale, fem_matrix_free_newton_solve, fem_matrix_free_newton_preconditioner, fem_matrix_free_newton_step, fem_matrix_free_newton_timestep, plus molla-kernels tire_fem_matrix_free_newton_contract source guards and ignored hardware lanes `make test-gpu P=molla-kernels T=tire_fem_matrix_free_newton_gpu A='tire_fem_matrix_free_newton_gpu_residual_matches_cpu_oracle --nocapture'` and `…A='tire_fem_matrix_free_newton_gpu_outer_iteration_reduces_residual --nocapture'`",
            reason: "the matrix-free Newton implicit timestep is now wired across CPU/GPU: the CPU helper, callable debug timestep, opt-in solver mode preset, default-flipped high-pressure preset, WGSL kernel, host-side dispatch wrapper, and host-orchestrated outer iteration all run and are exercised by source-guard and hardware lanes — and the live tire stepper can be opted in via TetraTireFemParams::high_pressure_implicit",
            follow_up: "switch the GPU persistent FEM batch substep so high_pressure_implicit tires run the live GPU dispatch directly, then sweep that lane through viewer scenes 109/120/121/122 and the vehicle scene for live FEM regression envelopes before flipping the realtime_viewer / offroad_soft presets to MatrixFreeNewtonDebug by default",
        },
        TireGpuParityItem {
            feature_id: "tetra_fem_contact_diagnostics",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "TetraTireFem::evaluate_tread_contact",
            gpu_path: "TireFemContactReductionKernel / tire_fem_contact_reduce.wgsl",
            proof: "tire_fem_gpu contact-reduction heightfield/SDF/material tests plus ignored large triangle-BVH hardware dispatch stress lane (`make test-gpu P=molla-kernels T=tire_fem_gpu A='tire_fem_gpu_contact_reduction_large_triangle_bvh_dispatch_stress --nocapture'`)",
            reason: "",
            follow_up: "",
        },
        TireGpuParityItem {
            feature_id: "tetra_fem_contact_feedback",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "TetraTireFem::apply_tread_contact_wrench",
            gpu_path: "TireFemContactFeedbackKernel / tire_fem_contact_feedback.wgsl plus no-readback GPU-resident feedback buffers, a centralized readback_resident_feedback_output viewer/telemetry seam, optional spatial terrain surface-velocity-map sampling, and tire_fem_atlas.wgsl::hub_params_from_feedback_force_rows for resident feedback-force-to-atlas-hub coupling",
            proof: "tire_fem_gpu contact-feedback heightfield/SDF/moving-ground/spatial-velocity-map/deformable-sinkage tests plus resident no-readback/readback-seam, binding-contract source guards, and tire_fem_feedback_force_rows_feed_gpu_hub_external_force_params; Scene 122 uses the resident FEM-feedback dispatch when MPM or Bekker terrain buffers are resident",
            reason: "",
            follow_up: "use the resident feedback-force-to-hub handoff in broader live vehicle/FEM owners, and make deformable terrain owners write resident velocity maps wherever the terrain surface flow is non-uniform",
        },
        TireGpuParityItem {
            feature_id: "thermal_wear_condition",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "TireThermalLayers / TireWearState / TireConditionReport",
            gpu_path: "TireThermalGpuKernel / TireThermalGpuBatch / tire_thermal.wgsl batched four-node thermal and wear-depth step plus GPU-ready rig replay artifact builder",
            proof: "molla-vehicle real_tire thermal/wear unit tests plus molla-kernels tire_thermal reference/shader/replay-artifact tests and ignored hardware one-shot/persistent parity lanes",
            reason: "",
            follow_up: "",
        },
        TireGpuParityItem {
            feature_id: "layered_transient_relaxation",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "TireLayeredTransientState / evaluate_brush_tire_layered_transient",
            gpu_path: "TireTransientGpuKernel / TireTransientGpuBatch / tire_transient.wgsl batched tread/belt/carcass relaxation plus transient-to-FEM-feedback kinematic adapters",
            proof: "molla-vehicle layered_transient unit tests plus molla-kernels tire_transient reference/shader tests, tire_fem_gpu transient-to-feedback adapter test, ignored hardware one-shot/persistent parity lanes, scene-121 GPU transient-to-brush wiring, and scene-122 GPU transient-to-FEM-feedback wiring/readouts",
            reason: "",
            follow_up: "extend transient-owned wheel-state stepping from the single-tire bench into broader GPU vehicle/chassis ownership",
        },
        TireGpuParityItem {
            feature_id: "bekker_soil_feedback",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "DeformableTerrain::apply_tire_contact / tire_soil_rut_matrix / TireSoilNormalShearCouplingReport",
            gpu_path: "TireBekkerGpuKernel / tire_bekker.wgsl plus TireBekkerFemTerrainOwner::apply_contacts_gpu_resident / TireBekkerGpuResidentTerrain feeding TireFemContactFeedbackKernel resident GPU terrain, plus TireFemContactReductionKernel / TireFemContactFeedbackKernel sinkage-map sampling",
            proof: "molla-vehicle deformable soil rut tests plus soil_normal_shear_coupling standard-runner report, tire_bekker_gpu writeback/reference and resident FEM-feedback terrain source guards, tire_fem_gpu sinkage sampling tests, and Scene 122 source guard for resident Bekker-terrain FEM feedback",
            reason: "",
            follow_up: "keep replacing diagnostic host terrain export with resident GPU terrain consumption in broader offroad scenes, then extend the same ownership pattern to DEM/MPM tire-specific coupling",
        },
        TireGpuParityItem {
            feature_id: "dem_granular_tire_feedback",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "GranularBed::apply_tire_contact / TireDemFeedback plus TireDemSparseLbvh Morton-sorted sparse-bed reference traversal, tire_dem_raster_surface_velocity_map top-layer DEM velocity export, and TireDemFemTerrainOwner owned height/sinkage/velocity terrain export",
            gpu_path: "TireDemGpuKernel / tire_dem.wgsl localized tire-cell impulse transfer plus dense, host-grid, unbounded sorted-grid CPU oracle, unbounded GPU cell-key/radix/range resident grid step, old GPU-built fixed-capacity API compatibility routed through the unbounded path, sparse LBVH node/index traversal for sparse beds, tire_dem_build_sparse_lbvh_morton_order_gpu_radix GPU Morton-key/radix sorted-order builder stage, tire_dem_build_sparse_lbvh_leaf_rows_gpu_radix GPU sorted leaf-range/AABB row builder stage, tire_dem_build_sparse_lbvh_tree_rows_gpu_radix GPU padded complete-binary internal-node tree row builder stage for arbitrary leaf counts, TireDemGpuKernel::build_resident_sparse_lbvh_tree_rows_gpu_radix no-readback resident GPU tree builder for arbitrary leaf counts, sparse-LBVH dispatch sizing plan, TireDemResidentState + update_resident_state + dispatch_lbvh_step_resident_state + dispatch_lbvh_step_resident + readback_resident_output no-readback/persistent-state seam, TireDemResidentLbvh + upload_resident_lbvh + upload_resident_lbvh_tree_rows + update_resident_lbvh + dispatch_lbvh_step_resident_state_with_resident_lbvh explicit resident topology-buffer seam, dispatch_resident_terrain_raster / TireDemResidentTerrainOutput resident GPU height/sinkage/surface_velocity_map export seam, TireDemResidentTerrainOutput::fem_contact_feedback_gpu_terrain direct GPU FEM-terrain handoff, TireDemResidentFemTerrainOwner reusable resident state+terrain owner with reseed_grain_state reset bridge, no-readback GPU LBVH topology build from cached Morton bounds, and explicit CPU mirror readback seam, TireDemFemTerrainOwner DEM surface_velocity_map rows plus GPU blocking sparse-LBVH step wrapper, and tire_fem_contact_terrain_with_dem_surface_velocity attaching those rows to FEM terrain contact",
            proof: "molla-vehicle granular DEM tire-contact tests plus tire_dem_gpu reference/GPU impulse, dense-step, host-grid-step, unbounded sorted-grid dense-cell retention, sparse-LBVH Morton ordering/reference parity, ignored GPU sparse-LBVH Morton-order radix builder lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_gpu_sparse_lbvh_morton_order_matches_cpu_oracle --nocapture'`), ignored GPU sparse-LBVH sorted leaf-row builder lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_gpu_sparse_lbvh_leaf_rows_match_cpu_sorted_leaf_chunks --nocapture'`), ignored GPU sparse-LBVH complete-binary tree-row builder lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_gpu_sparse_lbvh_tree_rows_match_cpu_complete_binary_oracle --nocapture'`), ignored GPU sparse-LBVH padded non-power-of-two tree-row builder lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_gpu_sparse_lbvh_tree_rows_pad_non_power_of_two_leaves --nocapture'`), ignored GPU-built resident sparse-LBVH dispatch lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_gpu_built_lbvh_tree_rows_feed_resident_sparse_step --nocapture'`), ignored no-readback resident GPU sparse-LBVH builder+dispatch lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_resident_gpu_lbvh_builder_feeds_sparse_step_without_tree_readback --nocapture'`), 14x12x5 large sparse-LBVH stress/reference parity with dispatch-plan sizing, ignored large no-readback resident sparse-LBVH builder stress lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_resident_gpu_lbvh_builder_large_sparse_bed_matches_reference --nocapture'`), DEM top-layer surface-velocity-map export test plus TireDemFemTerrainOwner FEM-terrain height/sinkage/velocity bridge and owned sparse-LBVH stepping assertions, Scene 119 DEM→FEM terrain bridge telemetry plus optional TireDemResidentFemTerrainOwner resident GPU stepping/source guard and resident-reseed telemetry source guard, Scene 122 live TireDemFemTerrainOwner FEM feedback hookup/source guard plus selectable GPU resident sparse-LBVH DEM step and resident-reseed telemetry source guard, Scene 122 source guard for TireDemResidentFemTerrainOwner / dispatch_resident_terrain_raster / fem_contact_feedback_gpu_terrain resident DEM-to-FEM terrain handoff, ignored GPU resident sparse-LBVH no-readback/readback-seam parity lane, ignored resident-state ping-pong sparse-LBVH lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_resident_state_sparse_lbvh_step_ping_pongs_without_reupload --nocapture'`), ignored resident terrain-raster lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_resident_terrain_raster_matches_cpu_export_without_grain_readback --nocapture'`), ignored reusable resident owner lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_resident_fem_terrain_owner_steps_and_keeps_gpu_terrain --nocapture'`), ignored resident owner reseed lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_resident_fem_terrain_owner_reseed_refreshes_gpu_state_and_bounds --nocapture'`), ignored owner-level TireDemFemTerrainOwner GPU sparse-LBVH parity lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_fem_terrain_owner_gpu_sparse_lbvh_step_matches_reference --nocapture'`), ignored large sparse-LBVH GPU hardware dispatch stress lane (`make test-gpu P=molla-kernels T=tire_dem_gpu A='tire_dem_gpu_large_sparse_lbvh_dispatch_stress_matches_reference --nocapture'`), GPU-radix sorted-grid/range parity, unbounded GPU-radix resident-grid-step parity, and old-API compatibility parity tests",
            reason: "",
            follow_up: "feed DEM surface_velocity_map rows directly into multi-scene FEM tire contact owners, then keep scaling the sparse-LBVH hardware stress as grain counts grow",
        },
        TireGpuParityItem {
            feature_id: "mpm_particle_soil_feedback",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "tire_mpm_reference_feedback material-point tire pressure/shear/compaction reference plus MpmSimulation::apply_tire_contacts_cpu_feedback_with",
            gpu_path: "TireMpmGpuKernel / tire_mpm.wgsl localized material-point tire feedback plus GPU MPM compaction/material-velocity-to-terrain raster, TireFemContactFeedbackKernel direct GPU height/sinkage/surface-velocity terrain consumption through dispatch_resident_with_gpu_terrain, TireFemContactReductionKernel GPU-terrain diagnostics, and MpmSimulation::apply_tire_contacts_gpu_resident_with in-place persistent-buffer composition bridge",
            proof: "tire_mpm_gpu reference/raster tests including surface-velocity-map parity plus ignored tire_mpm/mpm_gpu/tire_fem_gpu GPU-terrain parity lanes and Scene 122 source guard for resident MPM-terrain FEM feedback",
            reason: "",
            follow_up: "extend the Scene::TireRealistic MPM bridge into broader offroad/granular viewer scenes and keep replacing diagnostic host terrain export with direct GPU-terrain consumption where possible",
        },
        TireGpuParityItem {
            feature_id: "triangle_surface_contact",
            status: TireGpuParityStatus::CpuAndGpu,
            cpu_path: "TriangleMeshTerrain height query, per-triangle friction, closest-triangle signed distance, reusable CPU triangle BVH traversal, CPU multi-triangle contact manifold queries, and TetraTireFem tread-node manifold-radius projection",
            gpu_path: "TireFemContactReductionKernel / TireFemContactFeedbackKernel finite-triangle sampling with weighted multi-triangle manifold accumulation, per-triangle broadphase spheres, GPU-consumable triangle BVH atlas/traversal, and TireFemContactFeedbackPatchAccounting",
            proof: "tire_fem_gpu contact-reduction/contact-feedback triangle-mesh patch tests plus tire_fem_contact_triangle_exports_gpu_broadphase_sphere, tire_fem_contact_triangle_exports_gpu_bvh_atlas_rows, tire_fem_contact_triangle_bvh_large_atlas_stress_keeps_packed_rows_valid, ignored tire_fem_gpu_contact_reduction_large_triangle_bvh_dispatch_stress hardware lane, tire_fem_contact_shaders_use_weighted_triangle_manifold_sampling, and tire_fem_contact_feedback_output_reports_patch_force_and_bristle_accounting",
            reason: "",
            follow_up: "add deeper seam visualization if patch accounting shows remaining loss of detail, and keep scaling GPU triangle-atlas stress as terrain complexity grows",
        },
    ]
}

/// Text form of the GPU parity checklist for logs / rig reports.
pub fn tire_gpu_parity_text_report() -> String {
    tire_gpu_parity_checklist()
        .into_iter()
        .map(TireGpuParityItem::to_text_line)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Severity/category for a known tire-model approximation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TireApproximationStatus {
    RealtimeApproximation,
    PlannedPhysicsUpgrade,
    CpuOnlyDebug,
}

impl TireApproximationStatus {
    pub const fn label(self) -> &'static str {
        match self {
            Self::RealtimeApproximation => "realtime_approximation",
            Self::PlannedPhysicsUpgrade => "planned_physics_upgrade",
            Self::CpuOnlyDebug => "cpu_only_debug",
        }
    }
}

/// One blunt row in the known-approximation ledger.
///
/// This is separate from GPU parity: it says what the tire model still
/// approximates even when a CPU/GPU implementation exists. Keep it reusable so
/// viewer truth labels, rig reports, and docs do not accidentally oversell
/// scene 120/121/122 as experimentally validated tires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TireKnownApproximation {
    pub item_id: &'static str,
    pub status: TireApproximationStatus,
    pub current_model: &'static str,
    pub consequence: &'static str,
    pub next_step: &'static str,
    pub gpu_status: TireGpuParityStatus,
}

impl TireKnownApproximation {
    pub const fn has_required_accounting(self) -> bool {
        !self.item_id.is_empty()
            && !self.current_model.is_empty()
            && !self.consequence.is_empty()
            && !self.next_step.is_empty()
    }

    pub fn to_text_line(self) -> String {
        format!(
            "{} status={} gpu_status={} current={} consequence={} next_step={}",
            self.item_id,
            self.status.label(),
            self.gpu_status.label(),
            csv_token(self.current_model),
            csv_token(self.consequence),
            csv_token(self.next_step)
        )
    }
}

/// Known approximations in the current real-tire stack.
///
/// These are not failures; they are the explicit "do not overclaim" list for
/// productionization. Any new approximation added to the high-frequency tire
/// path should get a row here with a consequence and next step.
pub fn tire_known_approximation_ledger() -> Vec<TireKnownApproximation> {
    vec![
        TireKnownApproximation {
            item_id: "explicit_fem_elasticity",
            status: TireApproximationStatus::PlannedPhysicsUpgrade,
            current_model: "explicit compressible Neo-Hookean tetra update plus XPBD volume/bead/structural guards, an optional CPU/GPU diagonal semi-implicit stiffness denominator, and CPU/debug Newton residual gates from scalar mode through tiny actual tire mesh",
            consequence: "stable realtime demos still need small substeps; high-pressure presets have a first stiff-rubber stabilizer and Newton validation fixtures, but production tire stepping is not yet a full implicit mesh solve",
            next_step: "promote the tiny-mesh Newton residual/Jacobian into production-size tire mesh stepping and add GPU sparse or matrix-free parity before calling full implicit FEM complete",
            gpu_status: TireGpuParityStatus::CpuAndGpu,
        },
        TireKnownApproximation {
            item_id: "newton_solver_cpu_debug_only",
            status: TireApproximationStatus::CpuOnlyDebug,
            current_model: "TetraFemDiagonalNewtonReport, TetraFemElementNewtonReport, TetraFemPatchNewtonReport, TetraFemTinyMeshNewtonReport, TetraFemMatrixFreeNewtonScaleReport, TetraFemMatrixFreeNewtonSolveReport, TetraFemMatrixFreeNewtonPreconditionerReport, TetraFemMatrixFreeNewtonStepReport, TetraTireFem::step_on_terrain_matrix_free_newton_debug, shared matrix_free_newton_correction helper, TetraTireFemSolverMode opt-in solver preset, TetraTireFemParams::high_pressure_implicit default-flipped preset, molla-kernels tire_fem_matrix_free_newton_gpu_contract, tire_fem_matrix_free_newton.wgsl WGSL kernel, TireFemMatrixFreeNewtonGpuKernel live GPU dispatch wrapper, and TireFemMatrixFreeNewtonGpuKernel::run_newton_outer_blocking host-orchestrated GMRES+Jacobi+line-search outer iteration on top of GPU primitives solve reduced, element, patch, smallest-mesh dense, larger-mesh matrix-free residual/Jv, larger-mesh matrix-free GMRES correction, matrix-free diagonal-preconditioner, report-level line-searched implicit-step, callable debug timestep, opt-in solver preset, default-flipped preset, host-side GPU dispatch contract, live WGSL dispatch path, and a host-orchestrated GPU Newton outer iteration",
            consequence: "the matrix-free Newton implicit timestep now spans CPU + GPU end-to-end: per-DoF residual/Jv/diagonal/correction on GPU, host-orchestrated GMRES + Jacobi + line search outer iteration, opt-in production solver preset on TetraTireFemParams, default-flipped TetraTireFemParams::high_pressure_implicit, hardware lanes that confirm residual oracle parity and outer-iteration residual reduction — but viewer/vehicle scenes still default the realtime_viewer / offroad_soft tires to explicit/semi-implicit because the GPU persistent FEM batch substep has not been rewired to call run_newton_outer_blocking each substep",
            next_step: "switch the GPU persistent FEM batch substep so high_pressure_implicit tires call run_newton_outer_blocking directly each substep, validate live FEM scenes 109/120/121/122 and the vehicle scene against the explicit envelopes, then promote TetraTireFemSolverMode::MatrixFreeNewtonDebug into the default solver preset for realtime_viewer / offroad_soft tires before treating full implicit FEM as production",
            gpu_status: TireGpuParityStatus::CpuAndGpu,
        },
        TireKnownApproximation {
            item_id: "tetra_fem_torque_api_cpu_only",
            // `status` still reads CpuOnlyDebug because the matrix-free
            // Newton debug timestep does not yet integrate the spin
            // DOF; the standard step_on_terrain path however is fully
            // CPU+GPU live (see `gpu_status` below).
            status: TireApproximationStatus::CpuOnlyDebug,
            current_model: "TetraTireFem carries a spin DOF (hub_angle, hub_omega), drive/brake torque inputs (set_drive_torque_n_m, set_brake_torque_n_m), spin-coupled tread bristle slip, and a dynamic-hub XZ integrator that advances the hub from contact reaction when set_hub_dynamic is called. Chassis hand-off goes through set_hub_position / set_hub_velocity / last_hub_reaction_xyz. Stage G parts 1+2+3 mirror the spin contribution in tire_fem.wgsl via the phys5 uniform row and the XPBD-contact spin term; TireFemTorqueGpuStatus auto-advances to LiveValidated; the ignored hardware lanes tire_fem_gpu_spin_pushes_grounded_tread_backward_in_contact and tire_fem_cpu_vs_gpu_rolling_contact_hub_x_parity validate the wiring and the rolling trajectory",
            consequence: "torque-driven rolling on the CPU path is live and used by viewer scene 123 (TerrainLab); the GPU step reads omega and applies the spin-coupled contact slip with the same sign convention. The matrix-free Newton debug timestep does not yet integrate the spin DOF, and true GPU-side dynamic-hub XZ integration (atomic contact-reaction accumulation across threads) is a further refinement — the host integrates the hub via TetraTireFem and forwards hub_omega to the GPU each frame",
            next_step: "follow-up enhancement: add per-workgroup atomic accumulation of the contact reaction so the GPU dispatch can integrate hub_v_xz / hub[0]/[2] internally, add a CPU-vs-GPU full-step parity lane that drives both with identical torque inputs and confirms hub trajectory matches within tolerance, and mirror the dynamic-hub integration in the matrix-free Newton debug timestep",
            gpu_status: TireGpuParityStatus::CpuAndGpu,
        },
        TireKnownApproximation {
            item_id: "coarse_tetra_resolution",
            status: TireApproximationStatus::RealtimeApproximation,
            current_model: "viewer presets use low circumferential/radial/width resolution for interactive stepping; persistent multi-tire GPU batch ownership, resident fused-atlas ownership, resident state reseeding, deterministic FEM resolution sweeps, and mean/peak tread-belt stretch diagnostics now exist",
            consequence: "contact pressure and sidewall wrinkles are qualitative, not high-resolution tread-block detail; topology/volume scaling is now checked before raising default density",
            next_step: "use resolution-sweep telemetry, resident-atlas hardware parity/timing, and viewer runtime adoption before increasing default mesh density or adding tread-block detail",
            gpu_status: TireGpuParityStatus::CpuAndGpu,
        },
        TireKnownApproximation {
            item_id: "sidewall_bend_proxy",
            status: TireApproximationStatus::PlannedPhysicsUpgrade,
            current_model: "XPBD edge/hinge-distance proxy plus compact four-family explicit ply stack, sidewall dihedral diagnostics, sidewall ply stretch diagnostics, runner-level sidewall laminate audit/sanity gates for crossed bias/radial/axial helper plies, CPU/GPU dihedral-angle XPBD projection with a standard-runner fold-reduction effect gate, CPU sidewall_laminate_coupling ply-energy fixture gates, and CPU/GPU tensile ply-strain XPBD guardrails",
            consequence: "sidewall folding has a fold-angle constraint and the explicit laminate directions/stretch/energy response are audited in the standard runner; the sanity/coupling/effect/ply-strain gates catch folded, over-stretched, or uncoupled sidewalls with otherwise valid ply layout, but a true force-producing laminated shell/multi-ply carcass solve is still missing",
            next_step: "replace or validate the XPBD fold/ply projections against a full laminated sidewall constraint using the laminate sanity, ply-stretch, ply-energy coupling, ply-strain, and dihedral fold-reduction reports as the regression baseline",
            gpu_status: TireGpuParityStatus::CpuAndGpu,
        },
        TireKnownApproximation {
            item_id: "cavity_bead_sealing",
            status: TireApproximationStatus::PlannedPhysicsUpgrade,
            current_model: "polytropic cavity pressure with named/audited cavity_carcass_coupling preset parameter, simplified bead/rim mount, XPBD bead projection, bead mount error telemetry in FEM diagnostics/solver snapshots, decomposed bead-seat radial/axial/tangential drift diagnostics, preset-governance coverage, non-calibration cavity/bead sanity, and a cavity-coupling effect gate",
            consequence: "inflation affects load/shape, pressure scaling, pressure-to-carcass coupling and bead drift now have broad internal gates with drift direction visible; the named coupling proxy is swept for zero/nominal/high effect, but bead sealing and rim contact are still simplified rather than a validated rim-seat contact solve",
            next_step: "replace the audited coupling proxy with validated bead/rim contact and stronger structural pressure support using the cavity/bead sanity report, cavity-coupling effect report, bead-seat drift decomposition, and bead-error telemetry as regression signals",
            gpu_status: TireGpuParityStatus::CpuAndGpu,
        },
        TireKnownApproximation {
            item_id: "thermal_wear_curves",
            status: TireApproximationStatus::RealtimeApproximation,
            current_model: "deterministic hand-tuned layered thermal/wear multipliers with CPU reference, GPU batched four-node thermal plus wear-depth stepping, reusable GPU-ready rig replay artifacts, and non-calibration thermal condition sanity reports",
            consequence: "hot/cold/abused labels and thermal sanity reports are state-derived regression signals, not rubber-chemistry calibration",
            next_step: "keep presets governed and only add fitted thermal curves or deeper rubber-chemistry validation if calibration is explicitly in scope",
            gpu_status: TireGpuParityStatus::CpuAndGpu,
        },
        TireKnownApproximation {
            item_id: "legacy_vehicle_real_brush_adapter",
            status: TireApproximationStatus::CpuOnlyDebug,
            current_model: "RealBrushTireModel adapts legacy TireModel calls from slip_ratio, slip_angle, normal_load, and friction into evaluate_brush_tire on a synthetic flat contact patch with nominal speed for older TireModelLab and VehicleOffroad scenes",
            consequence: "legacy viewer paths now use native brush forces instead of the old simple tire, but they still lack full wheel pose, terrain maps, resident GPU buffers, FEM carcass state, suspension rows, and terrain feedback",
            next_step: "replace the bridge with resident vehicle-to-brush/FEM owners that keep chassis, wheel, suspension, terrain, brush, and FEM state on the GPU",
            gpu_status: TireGpuParityStatus::CpuOnlyDebug,
        },
        TireKnownApproximation {
            item_id: "bekker_only_soil",
            status: TireApproximationStatus::PlannedPhysicsUpgrade,
            current_model: "CPU Bekker heightfield compaction/excavation from tire contact cells, GPU Bekker height/sinkage writeback with FEM terrain ownership, CPU/GPU DEM tire-cell impulse transfer, dense DEM contact stepping, host-built cell-grid GPU DEM stepping, an unbounded sorted-grid CPU oracle plus GPU cell-key/radix/range resident grid step that keeps every grain in dense cells, old GPU-built fixed-capacity API compatibility routed through the unbounded DEM path, a Morton-sorted sparse LBVH CPU reference plus GPU sparse-LBVH traversal path and dispatch sizing plan for sparse DEM beds, CPU/GPU MPM material-point tire pressure/shear/compaction feedback, an in-place persistent-buffer MpmSimulation bridge that writes GPU tire feedback directly into the full particle/grid MPM solver, a GPU MPM compaction/material-velocity-to-height/sinkage/surface-velocity raster for FEM terrain buffers, and FEM contact feedback/reduction paths that consume those GPU terrain buffers directly",
            consequence: "soil ruts, rolling resistance, initial GPU DEM impulses, GPU cell-grid DEM stepping, tire-specific MPM particle feedback, MPM solver composition, GPU MPM-to-FEM terrain buffer export including spatial surface velocity for feedback, direct FEM contact feedback/reduction consumption, an unbounded GPU-radix DEM step, sparse-LBVH CPU/GPU path, 14x12x5 sparse-LBVH CPU stress/reference parity plus dispatch sizing, a large sparse-LBVH GPU hardware dispatch stress lane, Scene::TireRealistic MPM viewer/offroad ownership, and a Scene::SoilCompaction sparse-LBVH DEM owner bridge exist; full multi-scene offroad ownership and higher-grain GPU scaling remain missing",
            next_step: "extend the Scene::TireRealistic MPM bridge and Scene::SoilCompaction DEM owner into broader offroad/granular viewer scenes, then keep increasing sparse-LBVH GPU stress sizes as ownership expands",
            gpu_status: TireGpuParityStatus::CpuAndGpu,
        },
        TireKnownApproximation {
            item_id: "triangle_surface_simplification",
            status: TireApproximationStatus::RealtimeApproximation,
            current_model: "CPU triangle BVH over per-triangle AABBs for height, closest-surface, and multi-triangle contact-manifold queries; CPU TetraTireFem tread nodes consume manifold-radius projection; GPU finite-triangle closest-point sampling has weighted multi-triangle manifold accumulation, per-triangle horizontal broadphase spheres, a GPU-consumable BVH atlas/traversal path, and reusable FEM feedback patch force/bristle accounting",
            consequence: "static mesh plates/edges work for diagnostic meshes, seam/curb contacts can be represented as multiple CPU patches, CPU tread nodes can project from manifold radius before point penetration, and large CPU/GPU triangle terrain queries avoid scanning every triangle; GPU FEM contact now blends nearby patch normals/friction, exposes force/bristle summary accounting, and has an ignored large triangle-BVH hardware dispatch stress lane",
            next_step: "add deeper seam visualization only if patch accounting shows remaining loss of detail, and keep scaling triangle-atlas hardware stress as mesh terrain grows",
            gpu_status: TireGpuParityStatus::CpuAndGpu,
        },
    ]
}

/// Text form of the known-approximation ledger for viewer labels / reports.
pub fn tire_known_approximation_text_report() -> String {
    tire_known_approximation_ledger()
        .into_iter()
        .map(TireKnownApproximation::to_text_line)
        .collect::<Vec<_>>()
        .join("\n")
}
