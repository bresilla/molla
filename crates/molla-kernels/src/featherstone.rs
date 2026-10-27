//! Rust dispatch wrapper around `shaders/featherstone.wgsl`.
//!
//! Per-articulation parallelism — one workgroup invocation per chain.
//! For molla's typical scenes (one robot per scene) this is single-
//! threaded, but the architecture matches Newton's Featherstone-Warp
//! shape and scales naturally to RL "many robots in parallel" once
//! the articulation grouping is set up.

use molla_core::{Error, Result};
use molla_sim::{Contacts, Control, Model, State};
use molla_storage::{Buffer, BufferElement};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    dt: f32,
    joint_damping: f32,
    joint_count: u32,
    articulation_count: u32,
    /// Stage 17.1: 1 if the WGSL kernel should use the rigorous
    /// dense-Cholesky D solve (v.s. v1 sequential rank-1).
    rigorous_d_solve: u32,
    /// Stage 17.1: 1 if Distance joints should use velocity-level
    /// Lagrange-multiplier projection instead of the v1 hard-clamp.
    rigorous_distance: u32,
    drift_only: u32,
    stable_motor_feedback: u32,
    external_scalar_bounds: u32,
    compensated_coordinates: u32,
    control_body_wrenches: u32,
    paired_rates: u32,
}

const WORKGROUP_SIZE: u32 = 64;

fn articulation_scratch_capacity(offsets: &[u32]) -> u32 {
    offsets
        .windows(2)
        .map(|range| range[1].saturating_sub(range[0]))
        .max()
        .unwrap_or(0)
        .clamp(1, 32)
}

fn mixed_stop_capacity(model: &Model) -> Result<usize> {
    let starts = model.articulation_start.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let kinds = model.joint_type.host()?;
    let enabled = model.joint_enabled.host()?;
    let lower = model.joint_limit_lower.host()?;
    let upper = model.joint_limit_upper.host()?;
    let caps = model.joint_velocity_limit.host()?;
    let mut maximum = 1;
    for art in 0..model.articulation_count.max(1) {
        let range = if model.articulation_count == 0 {
            0..model.joint_count
        } else {
            starts[art] as usize..starts[art + 1] as usize
        };
        let coupled = |j: usize| {
            let d = offsets[j] as usize;
            enabled[j] != 0
                && ((kinds[j] == molla_sim::JointType::Ball as u32
                    && molla_sim::joint_control::ball_coupled_stops(
                        &lower[d..d + 3],
                        &upper[d..d + 3],
                        &caps[d..d + 3],
                    ))
                    || (kinds[j] == molla_sim::JointType::D6 as u32
                        && molla_sim::joint_control::d6_coupled_stops(
                            &lower[d..d + 6],
                            &upper[d..d + 6],
                            &caps[d..d + 6],
                        ))
                    || (kinds[j] == molla_sim::JointType::Distance as u32
                        && molla_sim::joint_control::distance_coupled_stops(
                            &lower[d..d + 6],
                            &upper[d..d + 6],
                            &caps[d..d + 6],
                        )))
        };
        if !range.clone().any(coupled) {
            continue;
        }
        let mut rows = 0;
        for j in range {
            if enabled[j] == 0 {
                continue;
            }
            let d = offsets[j] as usize;
            let locked = |k: usize| {
                lower[d + k].is_finite()
                    && upper[d + k].is_finite()
                    && (upper[d + k] - lower[d + k]).abs() < 1e-12
            };
            let kind = kinds[j];
            let axes = if kind == molla_sim::JointType::D6 as u32 {
                3
            } else if kind <= 1 {
                1
            } else {
                0
            };
            rows += (0..axes)
                .filter(|&k| !locked(k) && (lower[d + k].is_finite() || upper[d + k].is_finite()))
                .count();
            if kind == molla_sim::JointType::D6 as u32
                && !(3..6).all(locked)
                && (3..6).any(|k| lower[d + k].is_finite() || upper[d + k].is_finite())
            {
                rows += 4;
            }
            if kind == molla_sim::JointType::Ball as u32
                && !(0..3).all(locked)
                && (0..3).any(|k| lower[d + k].is_finite() || upper[d + k].is_finite())
            {
                rows += 4;
            }
            if kind == molla_sim::JointType::Distance as u32
                && (lower[d].is_finite() || upper[d].is_finite())
            {
                rows += if upper[d] == 0. { 3 } else { 1 };
            }
            rows += (0..offsets[j + 1] as usize - d)
                .filter(|&k| {
                    let structural = (kind <= 1 && locked(k))
                        || (kind == molla_sim::JointType::Ball as u32 && (0..3).all(locked))
                        || (kind == molla_sim::JointType::D6 as u32
                            && locked(k)
                            && (k < 3 || (3..6).all(locked)));
                    !structural && caps[d + k].is_finite()
                })
                .count();
        }
        maximum = maximum.max(rows);
    }
    Ok(maximum)
}

fn validate_compliance(model: &Model) -> Result<()> {
    let records = model.joint_compliance.host()?;
    if records.len() != model.joint_dof_total {
        return Err(Error::Gpu("invalid GPU joint compliance length".into()));
    }
    let offsets = model.joint_dof_offset.host()?;
    let kinds = model.joint_type.host()?;
    if offsets.len() != kinds.len() + 1
        || offsets.first() != Some(&0)
        || offsets.last().copied().map(|v| v as usize) != Some(records.len())
        || offsets.windows(2).any(|range| range[0] > range[1])
    {
        return Err(Error::Gpu("invalid GPU joint compliance offsets".into()));
    }
    for (joint, range) in offsets.windows(2).enumerate() {
        let active = records[range[0] as usize..range[1] as usize]
            .iter()
            .any(|record| record.enabled != 0);
        if active && (kinds[joint] != molla_sim::JointType::D6 as u32 || range[1] - range[0] != 6) {
            return Err(Error::Gpu(
                "joint compliance requires a six-DOF D6 joint".into(),
            ));
        }
    }
    for c in records.iter().filter(|c| c.enabled != 0) {
        c.validate()?;
        let packed = c.to_gpu();
        let omega = std::f32::consts::TAU * packed.natural_frequency;
        let stiffness = omega * omega;
        let damping = 2.0 * packed.damping_ratio * omega;
        if [
            packed.natural_frequency,
            packed.damping_ratio,
            stiffness,
            damping,
        ]
        .into_iter()
        .any(|value| !value.is_finite())
            || (c.natural_frequency > 0.0 && stiffness == 0.0)
            || [(c.lower, packed.lower), (c.upper, packed.upper)]
                .into_iter()
                .any(|(authored, value)| {
                    authored.is_finite()
                        && (!value.is_finite() || (authored != 0.0 && value == 0.0))
                })
            || (c.lower != c.upper && packed.lower == packed.upper)
        {
            return Err(Error::Gpu(
                "joint compliance exceeds GPU parameter precision".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "featherstone_rounding_tests.rs"]
mod rounding_tests;

#[cfg(test)]
#[path = "featherstone_no_contraction_tests.rs"]
mod no_contraction_tests;

#[cfg(test)]
#[path = "featherstone_zero_rotation_tests.rs"]
mod zero_rotation_tests;

#[cfg(test)]
#[path = "featherstone_body_rates_tests.rs"]
mod body_rates_tests;

#[cfg(test)]
#[path = "featherstone_acceleration_oracle.rs"]
mod acceleration_oracle;

#[path = "featherstone_lazy_anchors.rs"]
mod lazy_anchors;

#[path = "featherstone_drift_backend.rs"]
mod drift_backend;
use drift_backend::DriftBackend;

#[cfg(test)]
#[path = "featherstone_construction_tests.rs"]
mod construction_tests;

#[cfg(test)]
#[path = "featherstone_external_prediction_tests.rs"]
mod external_prediction_tests;
#[cfg(test)]
#[path = "featherstone_pipeline_cache_tests.rs"]
mod pipeline_cache_tests;
#[path = "featherstone_rates.rs"]
mod rates;
#[cfg(test)]
#[path = "featherstone_rates_tests.rs"]
mod rates_tests;

struct FeatherstoneBindings {
    buffers: [wgpu::Buffer; 55],
    groups: [wgpu::BindGroup; 5],
}

impl FeatherstoneBindings {
    fn matches(&self, buffers: &[&wgpu::Buffer; 55]) -> bool {
        self.buffers.iter().zip(buffers).all(|(a, b)| a == *b)
    }

    fn new(
        device: &wgpu::Device,
        layouts: [&wgpu::BindGroupLayout; 5],
        buffers: [&wgpu::Buffer; 55],
    ) -> Self {
        let ranges = [0..18, 18..31, 31..41, 41..50, 50..55];
        let groups = std::array::from_fn(|i| {
            let entries: Vec<_> = buffers[ranges[i].clone()]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::featherstone retained bindings"),
                layout: layouts[i],
                entries: &entries,
            })
        });
        Self {
            buffers: buffers.map(Clone::clone),
            groups,
        }
    }
}

struct CompensatedState {
    coordinates: Option<wgpu::Buffer>,
    poses: wgpu::Buffer,
    coordinate_error: wgpu::Buffer,
    position_error: wgpu::Buffer,
}

#[path = "featherstone_joint_types.rs"]
mod joint_types;

/// GPU Featherstone-style dynamics kernel.
pub struct FeatherstoneKernel {
    drift_backend: DriftBackend,
    specialized_joint_types: Option<std::sync::Arc<[u32]>>,
    split_acceleration: bool,
    external_prediction_only: bool,
    pipeline: wgpu::ComputePipeline,
    execution: FeatherstoneExecution,
    dynamics_pipeline: wgpu::ComputePipeline,
    dynamics_workgroup_size: u32,
    unrestricted_drift_pipeline: wgpu::ComputePipeline,
    #[cfg(test)]
    force_generic_drift: bool,
    #[cfg(test)]
    force_generic_dynamics: bool,
    layout_model: wgpu::BindGroupLayout,
    layout_state: wgpu::BindGroupLayout,
    layout_control: wgpu::BindGroupLayout,
    layout_arts: wgpu::BindGroupLayout,
    /// Bind group 5 — per-joint cable spring/damper + per-DOF muscle
    /// torque accumulator (Stage 39E + 42 v3).
    layout_cable_muscle: wgpu::BindGroupLayout,
    /// Articulation CSR — built from `model.articulation_start` at
    /// kernel construction. Length = `articulation_count + 1`.
    art_offsets: wgpu::Buffer,
    stop_status: wgpu::Buffer,
    active_articulations: wgpu::Buffer,
    velocity_delta: wgpu::Buffer,
    active_upload: Vec<u32>,
    /// Zero-filled fallback buffer used when the caller doesn't pass
    /// a custom per-DOF muscle torque buffer. Size = `dof_total *
    /// sizeof(f32)`, allocated once at kernel construction.
    zero_dof_buf: wgpu::Buffer,
    empty_state: [wgpu::Buffer; 2],
    bindings: [Option<FeatherstoneBindings>; 6],
    next_binding: usize,
    uniform_buffer: wgpu::Buffer,
    uniform_group: wgpu::BindGroup,
    bind_groups_created: u64,
    articulation_count: u32,
    joint_count: u32,
    dof_total: u32,
    stable_motor_feedback: bool,
    external_scalar_bounds: bool,
    mixed_stop_rows: usize,
    coordinate_count: usize,
    body_count: usize,
    coordinate_states: Vec<CompensatedState>,
    coordinate_fallback: wgpu::Buffer,
    position_fallback: wgpu::Buffer,
    rate_states: Vec<rates::RateState>,
    paired_rate_program: bool,
    paired_delta: Option<wgpu::Buffer>,
    rate_fallback: [wgpu::Buffer; 2],
    body_velocity_fallback: wgpu::Buffer,
    dispatch_timestamps: Option<(wgpu::QuerySet, [Option<u32>; 2])>,
}

struct FeatherstoneProgram {
    specialized_joint_types: Option<std::sync::Arc<[u32]>>,
    split_acceleration: bool,
    external_prediction_only: bool,
    paired_rates: bool,
    pipeline: wgpu::ComputePipeline,
    execution: FeatherstoneExecution,
    dynamics_pipeline: wgpu::ComputePipeline,
    dynamics_workgroup_size: u32,
    unrestricted_drift_pipeline: wgpu::ComputePipeline,
    drift_backend: DriftBackend,
    layout_model: wgpu::BindGroupLayout,
    layout_state: wgpu::BindGroupLayout,
    layout_control: wgpu::BindGroupLayout,
    layout_arts: wgpu::BindGroupLayout,
    layout_cable_muscle: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FeatherstoneRounding {
    #[default]
    Storage,
    /// Shared rounding barriers with persistent coordinate writeback.
    ExperimentalShared,
}

impl FeatherstoneRounding {
    pub fn label(self) -> &'static str {
        match self {
            Self::Storage => "portable-storage-rounding",
            Self::ExperimentalShared => "experimental-shared-scratch-with-coordinate-writeback",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FeatherstonePasses {
    #[cfg(test)]
    PhaseOnly(u32),
    #[cfg(test)]
    UnrestrictedScalarOnly,
    #[cfg(test)]
    LoopHintOnly(bool),
    #[cfg(test)]
    TractorJointKindsOnly,
    Both,
    BothSharedDynamics,
    BothConstraintScratch,
    BothSingleLaneDynamics,
    #[cfg(test)]
    GenericAndDynamics,
    #[cfg(test)]
    GenericAndSingleLaneDynamics,
    #[cfg(test)]
    GenericAndNoContractionDynamics,
    #[cfg(test)]
    GenericAndIntermediateRounding,
    #[cfg(test)]
    GenericAndResidualRounding,
    #[cfg(test)]
    GenericAndPassthroughControl,
    #[cfg(test)]
    GenericAndSharedDynamics,
    #[cfg(test)]
    GenericAndConstraintScratch,
    #[cfg(test)]
    GenericOnly,
    PairedUnrestricted,
    PairedExternalPrediction,
    #[cfg(test)]
    GenericWithoutZeroRotation,
    #[cfg(test)]
    UnrestrictedDriftWithoutZeroRotation,
    #[cfg(test)]
    GenericWithPositionProduct,
    #[cfg(test)]
    UnrestrictedDriftWithPositionProduct,
    #[cfg(test)]
    UnrestrictedDriftOnly,
    #[cfg(test)]
    UnrestrictedDriftLazyAnchors,
    #[cfg(test)]
    UnrestrictedDriftNativePosition,
    #[cfg(test)]
    UnrestrictedDriftNativeGeometry,
    #[cfg(test)]
    UnrestrictedDriftNativeGeometryLazyAnchors,
    #[cfg(test)]
    PairedAccelerationOnly,
    #[cfg(test)]
    PairedAccelerationNativeGeometry,
    #[cfg(test)]
    PairedRigorousAccelerationOnly,
    #[cfg(test)]
    PairedRigorousAccelerationNativeGeometry,
    #[cfg(test)]
    PairedRigorousAccelerationSwitchedGeometry,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FeatherstoneExecution {
    #[default]
    Generic,
    ExperimentalWorkgroupScratch,
    ExperimentalConstraintScratch,
    ExperimentalSingleLane,
    /// Specialize 1–32 model-wide joint types; type/count edits require reconstruction.
    /// Enabled flags, limits, compliance and physical state remain dynamic.
    ExperimentalJointTypes,
}

impl FeatherstoneExecution {
    pub fn label(self) -> &'static str {
        match self {
            Self::Generic => "runtime-mode-generic",
            Self::ExperimentalWorkgroupScratch => "experimental-workgroup-dynamics-scratch",
            Self::ExperimentalConstraintScratch => "experimental-workgroup-constraint-scratch",
            Self::ExperimentalSingleLane => "experimental-single-lane-private-dynamics",
            Self::ExperimentalJointTypes => "experimental-model-joint-types",
        }
    }
}

fn validate_execution(
    execution: FeatherstoneExecution,
    joints: u32,
    articulations: usize,
    limits: &wgpu::Limits,
) -> Result<()> {
    if !matches!(
        execution,
        FeatherstoneExecution::Generic | FeatherstoneExecution::ExperimentalJointTypes
    ) && (joints > 32
        || scratch_bytes(execution, joints) > limits.max_compute_workgroup_storage_size
        || articulations > limits.max_compute_workgroups_per_dimension as usize)
    {
        return Err(Error::Gpu(
            "workgroup dynamics scratch exceeds device limits".into(),
        ));
    }
    Ok(())
}

fn scratch_bytes(execution: FeatherstoneExecution, joints: u32) -> u32 {
    if execution == FeatherstoneExecution::ExperimentalSingleLane {
        return 256;
    }
    let base = 272 * joints + 3 * (24 * joints).div_ceil(16) * 16 + 288;
    if execution == FeatherstoneExecution::ExperimentalConstraintScratch {
        base + 528 * joints + (4 * joints).div_ceil(16) * 16
    } else {
        base
    }
}

impl FeatherstonePasses {
    #[cfg(test)]
    fn uses_passthrough(self) -> bool {
        matches!(
            self,
            Self::GenericAndNoContractionDynamics
                | Self::GenericAndPassthroughControl
                | Self::GenericAndIntermediateRounding
                | Self::GenericAndResidualRounding
        )
    }
    fn single_lane(self) -> bool {
        match self {
            Self::BothSingleLaneDynamics => true,
            #[cfg(test)]
            Self::GenericAndSingleLaneDynamics => true,
            _ => false,
        }
    }
    fn shared_dynamics(self) -> bool {
        match self {
            Self::BothSharedDynamics | Self::BothConstraintScratch => true,
            #[cfg(test)]
            Self::GenericAndSharedDynamics => true,
            #[cfg(test)]
            Self::GenericAndConstraintScratch => true,
            _ => false,
        }
    }

    fn dynamics_source(self, source: &str) -> Option<String> {
        if self == Self::PairedUnrestricted {
            return Some(rates::split_acceleration_source(source, true));
        }
        if self.single_lane() {
            assert_eq!(source.matches("@compute @workgroup_size(64)").count(), 1);
            return Some(source.replace(
                "@compute @workgroup_size(64)",
                "@compute @workgroup_size(1)",
            ));
        }
        if self.constraint_scratch() {
            return Some(constraint_scratch_source(source));
        }
        if self.shared_dynamics() {
            return Some(shared_dynamics_source(source));
        }
        #[cfg(test)]
        if self == Self::GenericAndPassthroughControl {
            return Some(no_contraction_tests::control_source(source));
        }
        #[cfg(test)]
        if matches!(
            self,
            Self::GenericAndNoContractionDynamics
                | Self::GenericAndIntermediateRounding
                | Self::GenericAndResidualRounding
        ) {
            return Some(no_contraction_tests::candidate_source(source));
        }
        #[cfg(test)]
        if self == Self::GenericAndDynamics {
            return Some(dynamics_source(source));
        }
        None
    }

    fn constraint_scratch(self) -> bool {
        match self {
            Self::BothConstraintScratch => true,
            #[cfg(test)]
            Self::GenericAndConstraintScratch => true,
            _ => false,
        }
    }

    fn dynamics_width(self) -> u32 {
        if self.single_lane() {
            return 1;
        }
        if self.shared_dynamics() {
            1
        } else {
            WORKGROUP_SIZE
        }
    }
}

impl FeatherstoneProgram {
    fn new(device: &wgpu::Device, joint_capacity: u32, mixed_stop_rows: usize) -> Self {
        Self::with_rounding(
            device,
            joint_capacity,
            mixed_stop_rows,
            FeatherstoneRounding::Storage,
        )
    }

    fn with_rounding(
        device: &wgpu::Device,
        joint_capacity: u32,
        mixed_stop_rows: usize,
        rounding: FeatherstoneRounding,
    ) -> Self {
        Self::with_options(
            device,
            joint_capacity,
            mixed_stop_rows,
            rounding,
            FeatherstoneExecution::Generic,
            false,
            false,
            None,
            None,
        )
    }

    fn with_options(
        device: &wgpu::Device,
        joint_capacity: u32,
        mixed_stop_rows: usize,
        rounding: FeatherstoneRounding,
        execution: FeatherstoneExecution,
        paired_rates: bool,
        unrestricted: bool,
        cache: Option<&wgpu::PipelineCache>,
        specialized_joint_types: Option<&[u32]>,
    ) -> Self {
        let shared = rounding == FeatherstoneRounding::ExperimentalShared;
        assert!(!unrestricted || paired_rates);
        Self::with_rate_precision_cached(
            device,
            joint_capacity,
            mixed_stop_rows,
            shared,
            false,
            shared,
            if unrestricted {
                FeatherstonePasses::PairedUnrestricted
            } else {
                match execution {
                    FeatherstoneExecution::Generic
                    | FeatherstoneExecution::ExperimentalJointTypes => FeatherstonePasses::Both,
                    FeatherstoneExecution::ExperimentalWorkgroupScratch => {
                        FeatherstonePasses::BothSharedDynamics
                    }
                    FeatherstoneExecution::ExperimentalConstraintScratch => {
                        FeatherstonePasses::BothConstraintScratch
                    }
                    FeatherstoneExecution::ExperimentalSingleLane => {
                        FeatherstonePasses::BothSingleLaneDynamics
                    }
                }
            },
            paired_rates,
            cache,
            specialized_joint_types,
        )
    }

    #[cfg(test)]
    fn with_precision(
        device: &wgpu::Device,
        joint_capacity: u32,
        mixed_stop_rows: usize,
        local: bool,
        native_sum: bool,
        local_coordinates: bool,
        passes: FeatherstonePasses,
    ) -> Self {
        Self::with_rate_precision(
            device,
            joint_capacity,
            mixed_stop_rows,
            local,
            native_sum,
            local_coordinates,
            passes,
            false,
        )
    }

    #[cfg(test)]
    fn with_rate_precision(
        device: &wgpu::Device,
        joint_capacity: u32,
        mixed_stop_rows: usize,
        local: bool,
        native_sum: bool,
        local_coordinates: bool,
        passes: FeatherstonePasses,
        paired_rates: bool,
    ) -> Self {
        Self::with_rate_precision_cached(
            device,
            joint_capacity,
            mixed_stop_rows,
            local,
            native_sum,
            local_coordinates,
            passes,
            paired_rates,
            None,
            None,
        )
    }

    fn with_rate_precision_cached(
        device: &wgpu::Device,
        joint_capacity: u32,
        mixed_stop_rows: usize,
        local: bool,
        native_sum: bool,
        local_coordinates: bool,
        passes: FeatherstonePasses,
        paired_rates: bool,
        cache: Option<&wgpu::PipelineCache>,
        specialized_joint_types: Option<&[u32]>,
    ) -> Self {
        let construction_trace =
            std::env::var_os("MOLLA_GPU_CONSTRUCTION_TRACE").map(|_| std::time::Instant::now());
        let trace = |stage: &str| {
            if let Some(start) = construction_trace {
                eprintln!(
                    "GPU construction featherstone joints={joint_capacity} mixed_rows={mixed_stop_rows} paired={paired_rates} t={:.6}s {stage}",
                    start.elapsed().as_secs_f64()
                );
            }
        };
        trace("begin");
        assert!(!native_sum || device.features().contains(wgpu::Features::SHADER_F64));
        #[cfg(test)]
        if passes.constraint_scratch() {
            let bytes = scratch_bytes(
                FeatherstoneExecution::ExperimentalConstraintScratch,
                joint_capacity,
            );
            assert!(bytes <= device.limits().max_compute_workgroup_storage_size);
        }
        let source = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            include_str!("../shaders/featherstone.wgsl"),
            include_str!("../shaders/featherstone_stops.wgsl"),
            include_str!("../shaders/featherstone_feedback.wgsl"),
            include_str!("../shaders/featherstone_mass_inverse.wgsl"),
            include_str!("../shaders/quaternion_precision.wgsl"),
            include_str!("../shaders/joint_euler_chart.wgsl"),
            include_str!("../shaders/euler_bound_precision.wgsl"),
            include_str!("../shaders/featherstone_mixed_stops.wgsl"),
            include_str!("../shaders/cone_geometry.wgsl"),
            include_str!("../shaders/featherstone_rates.wgsl"),
            include_str!("../shaders/featherstone_body_rates.wgsl")
        )
        .replace(
            "const MAX_JOINTS_PER_ART: u32 = 32u;",
            &format!("const MAX_JOINTS_PER_ART: u32 = {joint_capacity}u;"),
        );
        let source = source.replace(
            "const MAX_MIXED_STOP_ROWS: u32 = 1u;",
            &format!("const MAX_MIXED_STOP_ROWS: u32 = {mixed_stop_rows}u;"),
        );
        let source = if paired_rates {
            rates::shader_source(&source)
        } else {
            source.replace("uniforms.paired_rates", "0u")
        };
        #[cfg(test)]
        let source = if matches!(
            passes,
            FeatherstonePasses::GenericWithoutZeroRotation
                | FeatherstonePasses::UnrestrictedDriftWithoutZeroRotation
        ) {
            zero_rotation_tests::reference_source(&source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if matches!(
            passes,
            FeatherstonePasses::GenericWithPositionProduct
                | FeatherstonePasses::UnrestrictedDriftWithPositionProduct
        ) {
            zero_rotation_tests::xyz_candidate_source(&source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if matches!(
            passes,
            FeatherstonePasses::UnrestrictedDriftLazyAnchors
                | FeatherstonePasses::UnrestrictedDriftNativeGeometryLazyAnchors
        ) {
            lazy_anchors::source(&source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if native_sum {
            native_sum_test_source(&source, paired_rates, passes)
        } else {
            source
        };
        let source = if local_coordinates {
            assert_eq!(source.matches("fn coordinate_round(").count(), 1);
            format!(
                "{}\n{}",
                source.replace("fn coordinate_round(", "fn storage_coordinate_round("),
                include_str!("../shaders/featherstone_shared_coordinate.wgsl")
            )
        } else {
            source
        };
        #[cfg(test)]
        let source = if matches!(
            passes,
            FeatherstonePasses::UnrestrictedDriftOnly
                | FeatherstonePasses::UnrestrictedDriftLazyAnchors
                | FeatherstonePasses::UnrestrictedDriftNativePosition
                | FeatherstonePasses::UnrestrictedDriftNativeGeometry
                | FeatherstonePasses::UnrestrictedDriftNativeGeometryLazyAnchors
                | FeatherstonePasses::UnrestrictedDriftWithoutZeroRotation
                | FeatherstonePasses::UnrestrictedDriftWithPositionProduct
        ) {
            unrestricted_drift_source(&source)
        } else {
            source
        };
        let source = match specialized_joint_types {
            Some(kinds) => joint_types::source(&source, kinds)
                .expect("validated model joint types and shader load syntax"),
            None => source,
        };
        let primary_source = if passes == FeatherstonePasses::PairedExternalPrediction {
            rates::external_prediction_source(&source)
        } else if passes == FeatherstonePasses::PairedUnrestricted {
            rates::split_acceleration_source(&source, false)
        } else {
            source.clone()
        };
        #[cfg(test)]
        let primary_source = match passes {
            FeatherstonePasses::PairedRigorousAccelerationSwitchedGeometry => {
                switched_rigorous_acceleration_source(&primary_source)
            }
            FeatherstonePasses::PairedRigorousAccelerationOnly
            | FeatherstonePasses::PairedRigorousAccelerationNativeGeometry => {
                fixed_rigorous_acceleration_source(&primary_source)
            }
            FeatherstonePasses::PairedAccelerationOnly
            | FeatherstonePasses::PairedAccelerationNativeGeometry => {
                rates::split_acceleration_source(&primary_source, true)
            }
            FeatherstonePasses::TractorJointKindsOnly => {
                construction_tests::joint_kind_source(&primary_source, &[4, 1, 1, 1, 1, 1]).unwrap()
            }
            FeatherstonePasses::PhaseOnly(phase) => {
                construction_tests::phase_source(&primary_source, phase)
            }
            FeatherstonePasses::UnrestrictedScalarOnly => {
                rates::unrestricted_source(&primary_source)
            }
            _ => primary_source,
        };
        trace("modules begin");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::featherstone"),
            source: wgpu::ShaderSource::Wgsl(
                rates::dispatch_source(&primary_source, paired_rates).into(),
            ),
        });
        #[cfg(test)]
        let shader = if let FeatherstonePasses::LoopHintOnly(hint) = passes {
            construction_tests::loop_hint_shader(device, &primary_source, hint)
        } else {
            shader
        };
        let dynamics_shader = passes.dynamics_source(&source).map(|candidate| {
            let candidate = rates::dispatch_source(&candidate, paired_rates);
            #[cfg(test)]
            if passes == FeatherstonePasses::GenericAndResidualRounding {
                return no_contraction_tests::residual_shader(
                    device,
                    &candidate,
                    "featherstone_step",
                );
            }
            #[cfg(test)]
            if passes == FeatherstonePasses::GenericAndIntermediateRounding {
                return no_contraction_tests::intermediate_shader(
                    device,
                    &candidate,
                    "featherstone_step",
                );
            }
            #[cfg(test)]
            if passes == FeatherstonePasses::GenericAndPassthroughControl {
                return no_contraction_tests::control_shader(
                    device,
                    &candidate,
                    "featherstone_step",
                );
            }
            #[cfg(test)]
            if passes == FeatherstonePasses::GenericAndNoContractionDynamics {
                return no_contraction_tests::shader(device, &candidate, "featherstone_step");
            }
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("molla::featherstone dynamics"),
                source: wgpu::ShaderSource::Wgsl(candidate.into()),
            })
        });

        trace("modules ready");
        let layout_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone model"),
            entries: &storage_entries(18, true),
        });
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone state"),
            entries: &state_entries(),
        });
        let layout_control = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone control"),
            entries: &storage_entries(10, true),
        });
        let layout_arts = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone arts"),
            entries: &[
                storage_entries(1, true)[0],
                storage_entries(2, false)[1],
                storage_entries(3, true)[2],
                storage_entries(4, false)[3],
                storage_entries(5, false)[4],
                storage_entries(6, false)[5],
                storage_entries(7, false)[6],
                storage_entries(8, false)[7],
                storage_entries(9, false)[8],
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        // Bind group 5 — 4 cable per-joint buffers + 1 per-DOF muscle
        // torque accumulator. All read-only.
        let layout_cable_muscle =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::featherstone cable+muscle"),
                entries: &storage_entries(5, true),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::featherstone layout"),
            bind_group_layouts: &[
                Some(&layout_model),
                Some(&layout_state),
                Some(&layout_control),
                Some(&layout_arts),
                Some(&layout_uniforms),
                Some(&layout_cable_muscle),
            ],
            immediate_size: 0,
        });

        #[cfg(test)]
        let construction = (passes.uses_passthrough()
            || passes == FeatherstonePasses::PairedUnrestricted)
            .then(|| {
                eprintln!("selective pipeline: generic construction begin");
                std::time::Instant::now()
            });
        let constants = [("local_position_rounding", f64::from(local))];
        let constants: &[(&str, f64)] = &constants;
        #[cfg(test)]
        let constants = if matches!(passes, FeatherstonePasses::LoopHintOnly(_)) {
            &[][..]
        } else {
            constants
        };
        trace("generic pipeline begin");
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::featherstone"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("featherstone_step"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache,
        });

        trace("generic pipeline ready");
        #[cfg(test)]
        if let Some(start) = construction {
            eprintln!(
                "selective pipeline: generic ready after {:.6} seconds",
                start.elapsed().as_secs_f64()
            );
        }
        trace("unrestricted drift begin");
        let drift_backend = if matches!(
            passes,
            FeatherstonePasses::Both
                | FeatherstonePasses::BothSharedDynamics
                | FeatherstonePasses::BothConstraintScratch
                | FeatherstonePasses::BothSingleLaneDynamics
                | FeatherstonePasses::PairedUnrestricted
                | FeatherstonePasses::PairedExternalPrediction
        ) {
            DriftBackend::select(paired_rates, local, native_sum, local_coordinates)
        } else {
            DriftBackend::Eager
        };
        drift_backend
            .validate_features(device.features())
            .expect("validated paired drift device features");
        let unrestricted_drift_pipeline = if matches!(
            passes,
            FeatherstonePasses::Both
                | FeatherstonePasses::BothSharedDynamics
                | FeatherstonePasses::BothConstraintScratch
                | FeatherstonePasses::BothSingleLaneDynamics
        ) || (drift_backend != DriftBackend::Eager
            && matches!(
                passes,
                FeatherstonePasses::PairedUnrestricted
                    | FeatherstonePasses::PairedExternalPrediction
            )) {
            let drift_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("molla::featherstone unrestricted drift"),
                source: wgpu::ShaderSource::Wgsl(
                    rates::dispatch_source(
                        &drift_backend.source(&source, paired_rates),
                        paired_rates,
                    )
                    .into(),
                ),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::featherstone unrestricted drift"),
                layout: Some(&pipeline_layout),
                module: &drift_shader,
                entry_point: Some("featherstone_step"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("local_position_rounding", f64::from(local))],
                    ..Default::default()
                },
                cache,
            })
        } else {
            pipeline.clone()
        };

        trace("unrestricted drift ready");
        let dynamics_pipeline = if let Some(shader) = dynamics_shader {
            trace("dynamics pipeline begin");
            #[cfg(test)]
            let dynamics_construction = construction.map(|_| {
                eprintln!("selective pipeline: candidate construction begin");
                std::time::Instant::now()
            });
            let constants = [("local_position_rounding", f64::from(local))];
            let constants: &[(&str, f64)] = &constants;
            #[cfg(test)]
            let constants = if passes.uses_passthrough() {
                &[][..]
            } else {
                constants
            };
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::featherstone dynamics"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("featherstone_step"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    ..Default::default()
                },
                cache,
            });
            #[cfg(test)]
            if let Some(start) = dynamics_construction {
                eprintln!(
                    "selective pipeline: candidate ready after {:.6} seconds",
                    start.elapsed().as_secs_f64()
                );
            }
            pipeline
        } else {
            pipeline.clone()
        };

        trace("program ready");
        Self {
            specialized_joint_types: specialized_joint_types.map(std::sync::Arc::from),
            split_acceleration: passes == FeatherstonePasses::PairedUnrestricted,
            external_prediction_only: passes == FeatherstonePasses::PairedExternalPrediction,
            paired_rates,
            pipeline,
            execution: if specialized_joint_types.is_some() {
                FeatherstoneExecution::ExperimentalJointTypes
            } else if passes.single_lane() {
                FeatherstoneExecution::ExperimentalSingleLane
            } else if passes.constraint_scratch() {
                FeatherstoneExecution::ExperimentalConstraintScratch
            } else if passes.shared_dynamics() {
                FeatherstoneExecution::ExperimentalWorkgroupScratch
            } else {
                FeatherstoneExecution::Generic
            },
            dynamics_pipeline,
            dynamics_workgroup_size: if paired_rates {
                1
            } else {
                passes.dynamics_width()
            },
            unrestricted_drift_pipeline,
            drift_backend,
            layout_model,
            layout_state,
            layout_control,
            layout_arts,
            layout_cable_muscle,
            layout_uniforms,
        }
    }
}

#[cfg(test)]
fn dynamics_source(source: &str) -> String {
    assert!(source.contains("uniforms.drift_only"));
    source.replace("uniforms.drift_only", "0u")
}

fn shared_dynamics_source(source: &str) -> String {
    let source = hoist_scratch(
        source,
        &[
            "v_lin",
            "v_ang",
            "c_lin",
            "c_ang",
            "i_a",
            "p_lin",
            "p_ang",
            "a_lin",
            "a_ang",
            "d_inv_arr",
            "u_tau_arr",
            "dof_active_arr",
            "root_applied",
        ],
    );
    assert_eq!(source.matches("@compute @workgroup_size(64)").count(), 1);
    source.replace(
        "@compute @workgroup_size(64)",
        "@compute @workgroup_size(1)",
    )
}

fn hoist_scratch(source: &str, names: &[&str]) -> String {
    let mut source = source.to_owned();
    let mut declarations = String::new();
    for name in names {
        let prefix = format!("    var {name}:");
        let matches: Vec<_> = source
            .lines()
            .filter(|line| line.starts_with(&prefix))
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "missing unique scratch declaration: {name}"
        );
        let declaration = matches[0].to_owned();
        declarations.push_str(
            &declaration
                .trim_start()
                .replacen("var ", "var<workgroup> ", 1),
        );
        declarations.push('\n');
        source = source.replace(&format!("{declaration}\n"), "");
    }
    format!("{declarations}\n{source}")
}

fn constraint_scratch_source(source: &str) -> String {
    let mut source = hoist_scratch(
        &shared_dynamics_source(source),
        &[
            "n_dof_arr",
            "s_lin",
            "s_ang",
            "u_lin",
            "u_ang",
            "d_dense_inv",
        ],
    );
    for (parameter, kind, global) in [
        ("counts", "StopCounts", "n_dof_arr"),
        ("sl", "StopVectors", "s_lin"),
        ("sa", "StopVectors", "s_ang"),
        ("ul", "StopVectors", "u_lin"),
        ("ua", "StopVectors", "u_ang"),
        ("inverse", "StopBlocks", "d_dense_inv"),
    ] {
        for gap in ["", " "] {
            source = source.replace(
                &format!("{parameter}: ptr<function,{gap}{kind}>"),
                &format!("{parameter}: u32"),
            );
        }
        source = source
            .replace(&format!("(*{parameter})"), global)
            .replace(&format!("&{global}"), "0u");
    }
    source
}

#[cfg(test)]
fn native_sum_test_source(source: &str, paired_rates: bool, passes: FeatherstonePasses) -> String {
    assert_eq!(source.matches("fn position_sum(").count(), 1);
    assert_eq!(source.matches("fn coordinate_sum(").count(), 1);
    let source = format!(
        "{}\n{}\n{}",
        source
            .replace("fn position_sum(", "fn storage_position_sum(")
            .replace("fn coordinate_sum(", "fn storage_coordinate_sum("),
        include_str!("../shaders/featherstone_native_sum.wgsl"),
        "fn position_sum(child: u32, a: f32, b: f32) -> vec2<f32> { let high=position_round(child,a+b); return native_two_sum_with_high(a,b,high); }\nfn coordinate_sum(coord: u32, a: f32, b: f32) -> vec2<f32> { let high=coordinate_round(coord,a+b); let result=native_two_sum_with_high(a,b,high); atomicStore(&coordinate_error[coord],bitcast<u32>(result.y)); return result; }"
    );
    if paired_rates {
        let original =
            "let high=position_round(child,a+b); return native_two_sum_with_high(a,b,high);";
        assert_eq!(source.matches(original).count(), 1);
        let source = source.replace(original,"let high=position_round(child,a+b); let result=native_two_sum_with_high(a,b,high); return vec2<f32>(high,position_round(child,result.y));");
        if matches!(
            passes,
            FeatherstonePasses::UnrestrictedDriftNativePosition
                | FeatherstonePasses::UnrestrictedDriftNativeGeometry
                | FeatherstonePasses::UnrestrictedDriftNativeGeometryLazyAnchors
                | FeatherstonePasses::PairedAccelerationNativeGeometry
                | FeatherstonePasses::PairedRigorousAccelerationNativeGeometry
                | FeatherstonePasses::PairedRigorousAccelerationSwitchedGeometry
        ) {
            let source = source
                .replace("fn coordinate_sum(", "fn native_coordinate_sum(")
                .replace("fn storage_coordinate_sum(", "fn coordinate_sum(");
            if matches!(
                passes,
                FeatherstonePasses::UnrestrictedDriftNativeGeometry
                    | FeatherstonePasses::UnrestrictedDriftNativeGeometryLazyAnchors
                    | FeatherstonePasses::PairedAccelerationNativeGeometry
                    | FeatherstonePasses::PairedRigorousAccelerationNativeGeometry
                    | FeatherstonePasses::PairedRigorousAccelerationSwitchedGeometry
            ) {
                let original =
                    "fn position_sum(child: u32, a: f32, b: f32) -> vec2<f32> { let high=";
                assert_eq!(source.matches(original).count(), 1);
                source.replace(original,"fn position_sum(child: u32, a: f32, b: f32) -> vec2<f32> { if !fk_pair_rounding { return storage_position_sum(child,a,b); } let high=")
            } else {
                source
            }
        } else {
            source
        }
    } else {
        source
    }
}

#[cfg(test)]
fn fixed_rigorous_acceleration_source(source: &str) -> String {
    let source = rates::split_acceleration_source(source, true);
    assert_eq!(source.matches("uniforms.rigorous_d_solve").count(), 1);
    source.replace("uniforms.rigorous_d_solve", "1u")
}

#[cfg(test)]
fn switched_rigorous_acceleration_source(source: &str) -> String {
    let source = fixed_rigorous_acceleration_source(source);
    assert_eq!(source.matches("uniforms.joint_damping").count(), 3);
    let source = source.replace("uniforms.joint_damping", "0.0");
    let guard = "if !fk_pair_rounding { return storage_position_sum(child,a,b); }";
    assert_eq!(source.matches(guard).count(), 1);
    source.replace(guard, "if uniforms.joint_damping == 0.0 || !fk_pair_rounding { return storage_position_sum(child,a,b); }")
}

fn unrestricted_drift_source(source: &str) -> String {
    let stop_gate = "has_stops = has_stops && uniforms.external_scalar_bounds == 0u && !position_only && !refresh_only;";
    assert_eq!(source.matches(stop_gate).count(), 1);
    source
        .replace("uniforms.drift_only", "1u")
        .replace(stop_gate, "has_stops = false; mixed_stops = false;")
}

#[cfg(test)]
fn unrestricted_drift_program_source(source: &str, paired: bool, lazy: bool) -> String {
    let source = unrestricted_drift_source(source);
    if paired && lazy {
        lazy_anchors::source(&source)
    } else {
        source
    }
}

#[cfg(test)]
fn lazy_drift_enabled(
    paired: bool,
    local: bool,
    native_sum: bool,
    local_coordinates: bool,
) -> bool {
    DriftBackend::select(paired, local, native_sum, local_coordinates) == DriftBackend::Lazy
}

fn unrestricted_drift(model: &Model) -> Result<bool> {
    Ok(model
        .joint_limit_lower
        .host()?
        .iter()
        .all(|v| v.is_infinite())
        && model
            .joint_limit_upper
            .host()?
            .iter()
            .all(|v| v.is_infinite())
        && model
            .joint_velocity_limit
            .host()?
            .iter()
            .all(|v| v.is_infinite()))
}

fn kernel_configuration(model: &Model) -> (Vec<u32>, usize) {
    let joint_count = model.joint_count as u32;
    let art_offsets = if model.articulation_count == 0 {
        vec![0, joint_count]
    } else {
        model
            .articulation_start
            .host()
            .map(|starts| starts.to_vec())
            .unwrap_or_else(|_| vec![0, joint_count])
    };
    let joint_capacity = articulation_scratch_capacity(&art_offsets);
    let mixed_rows = mixed_stop_capacity(model)
        .unwrap_or(13 * joint_capacity as usize)
        .min(13 * joint_capacity as usize)
        .max(1);
    (art_offsets, mixed_rows)
}

impl FeatherstoneKernel {
    /// Retained unrestricted-drift pipeline identity, not the last dispatched phase.
    pub fn unrestricted_drift_backend(&self) -> &'static str {
        self.drift_backend.label()
    }
    /// Build selection for storage-rounded paired unrestricted drift.
    pub const fn paired_drift_build_variant() -> &'static str {
        DriftBackend::compiled().build_label()
    }

    /// Device features required by the configured storage-rounded paired drift backend.
    pub fn paired_drift_required_features() -> wgpu::Features {
        DriftBackend::compiled().required_features()
    }
    pub fn execution(&self) -> FeatherstoneExecution {
        self.execution
    }

    /// Set timestamp slots consumed by the next dispatch attempt, including refresh or drift.
    pub fn set_dispatch_timestamps(&mut self, timestamps: Option<(wgpu::QuerySet, [u32; 2])>) {
        self.dispatch_timestamps = timestamps.map(|(set, slots)| (set, slots.map(Some)));
    }

    /// Ends a timestamp interval begun by an earlier prediction pass.
    pub fn set_dispatch_end_timestamp(&mut self, query: wgpu::QuerySet, index: u32) {
        self.dispatch_timestamps = Some((query, [None, Some(index)]));
    }

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, model: &Model) -> Self {
        let (art_offsets, mixed_rows) = kernel_configuration(model);
        let program = FeatherstoneProgram::new(
            device,
            articulation_scratch_capacity(&art_offsets),
            mixed_rows,
        );
        Self::from_program(device, queue, model, art_offsets, mixed_rows, &program)
    }

    /// Reuse immutable shader/layout programs within an engine; buffers remain kernel-local.
    pub fn new_cached(engine: &molla_compute::Engine, model: &Model) -> Result<Self> {
        Self::new_cached_with_rounding(engine, model, FeatherstoneRounding::Storage)
    }

    pub fn new_cached_with_rounding(
        engine: &molla_compute::Engine,
        model: &Model,
        rounding: FeatherstoneRounding,
    ) -> Result<Self> {
        Self::new_cached_with_options(engine, model, rounding, FeatherstoneExecution::Generic)
    }

    pub fn new_cached_with_options(
        engine: &molla_compute::Engine,
        model: &Model,
        rounding: FeatherstoneRounding,
        execution: FeatherstoneExecution,
    ) -> Result<Self> {
        Self::new_cached_with_rate_mode(engine, model, rounding, execution, false)
    }

    /// Construct an experimental paired-rate program; prepare its state set before stepping.
    pub fn new_cached_with_paired_rates(
        engine: &molla_compute::Engine,
        model: &Model,
        rounding: FeatherstoneRounding,
        execution: FeatherstoneExecution,
    ) -> Result<Self> {
        Self::new_cached_with_rate_mode(engine, model, rounding, execution, true)
    }

    /// Constructs unrestricted paired drift/refresh/displacement without an ABA program.
    /// Ordinary force-prediction dispatch is rejected; rates must be supplied externally.
    pub fn new_cached_for_external_prediction(
        engine: &molla_compute::Engine,
        model: &Model,
    ) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("external prediction requires GPU".into()))?;
        let queue = engine.queue().unwrap();
        if !rates::unrestricted_model(model)? {
            return Err(Error::Gpu(
                "external prediction requires unrestricted paired drift".into(),
            ));
        }
        DriftBackend::select(true, false, false, false).validate_features(device.features())?;
        let (offsets, rows) = kernel_configuration(model);
        let capacity = articulation_scratch_capacity(&offsets);
        if offsets.len().saturating_sub(1)
            > device.limits().max_compute_workgroups_per_dimension as usize
        {
            return Err(Error::Gpu(
                "external prediction articulation dispatch limit".into(),
            ));
        }
        let build = |device: &wgpu::Device| {
            Ok(FeatherstoneProgram::with_rate_precision_cached(
                device,
                capacity,
                rows,
                false,
                false,
                false,
                FeatherstonePasses::PairedExternalPrediction,
                true,
                engine.pipeline_cache(),
                None,
            ))
        };
        #[cfg(not(target_arch = "wasm32"))]
        let program = engine.cached_gpu_program(
            &format!(
                "external-prediction;capacity={capacity};rows={rows};drift={}",
                Self::paired_drift_build_variant()
            ),
            build,
        )?;
        #[cfg(target_arch = "wasm32")]
        let program = build(device)?;
        Ok(Self::from_program(
            device, queue, model, offsets, rows, &program,
        ))
    }

    fn new_cached_with_rate_mode(
        engine: &molla_compute::Engine,
        model: &Model,
        rounding: FeatherstoneRounding,
        execution: FeatherstoneExecution,
        paired_rates: bool,
    ) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("engine has no GPU device".into()))?;
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("engine has no GPU queue".into()))?;
        let shared = rounding == FeatherstoneRounding::ExperimentalShared;
        DriftBackend::select(paired_rates, shared, false, shared)
            .validate_features(device.features())?;
        let (art_offsets, mixed_rows) = kernel_configuration(model);
        let joint_capacity = articulation_scratch_capacity(&art_offsets);
        let specialized_joint_types = if execution == FeatherstoneExecution::ExperimentalJointTypes
        {
            let kinds = model.joint_type.host()?;
            joint_types::validate_kinds(kinds)?;
            joint_types::validate_model(model, Some(kinds))?;
            Some(kinds)
        } else {
            None
        };
        let unrestricted = paired_rates
            && rounding == FeatherstoneRounding::Storage
            && matches!(
                execution,
                FeatherstoneExecution::Generic | FeatherstoneExecution::ExperimentalJointTypes
            )
            && rates::unrestricted_model(model)?;
        if paired_rates
            && art_offsets.len().saturating_sub(1)
                > device.limits().max_compute_workgroups_per_dimension as usize
        {
            return Err(Error::Gpu(
                "paired articulation dispatch exceeds device limits".into(),
            ));
        }
        validate_execution(
            execution,
            joint_capacity,
            art_offsets.len().saturating_sub(1),
            &device.limits(),
        )?;
        #[cfg(not(target_arch = "wasm32"))]
        let variant = format!(
            "joints={joint_capacity};mixed_rows={mixed_rows};rounding={};execution={};paired_rates={paired_rates};unrestricted={unrestricted};joint_types={specialized_joint_types:?};paired_drift={}",
            rounding.label(),
            execution.label(),
            Self::paired_drift_build_variant()
        );
        #[cfg(not(target_arch = "wasm32"))]
        let program = engine.cached_gpu_program(&variant, |device| {
            Ok(FeatherstoneProgram::with_options(
                device,
                joint_capacity,
                mixed_rows,
                rounding,
                execution,
                paired_rates,
                unrestricted,
                engine.pipeline_cache(),
                specialized_joint_types,
            ))
        })?;
        #[cfg(target_arch = "wasm32")]
        let program = FeatherstoneProgram::with_options(
            device,
            joint_capacity,
            mixed_rows,
            rounding,
            execution,
            paired_rates,
            unrestricted,
            engine.pipeline_cache(),
            specialized_joint_types,
        );
        Ok(Self::from_program(
            device,
            queue,
            model,
            art_offsets,
            mixed_rows,
            &program,
        ))
    }

    fn from_program(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        art_offsets_data: Vec<u32>,
        mixed_stop_rows: usize,
        program: &FeatherstoneProgram,
    ) -> Self {
        let joint_count = model.joint_count as u32;
        let pipeline = program.pipeline.clone();
        let layout_model = program.layout_model.clone();
        let layout_state = program.layout_state.clone();
        let layout_control = program.layout_control.clone();
        let layout_arts = program.layout_arts.clone();
        let layout_cable_muscle = program.layout_cable_muscle.clone();

        let articulation_count = (art_offsets_data.len().saturating_sub(1)) as u32;
        let art_offsets = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::featherstone art_offsets"),
                size: (art_offsets_data.len() * core::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            queue,
            &art_offsets,
            0,
            bytemuck::cast_slice(&art_offsets_data)
        );
        let active_upload = vec![1u32; articulation_count.max(1) as usize];
        let active_articulations = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla active articulations"),
                size: active_upload.len() as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            queue,
            &active_articulations,
            0,
            bytemuck::cast_slice(&active_upload),
        );
        let stop_status = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::featherstone stop status"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        let dof_total = model.joint_dof_total.max(1) as u32;
        let velocity_delta = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla generalized velocity increments"),
                size: dof_total as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let zero_dof_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::featherstone zero_dof"),
                size: ((dof_total as u64) * (core::mem::size_of::<f32>() as u64)).max(80),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        // Zero-fill once — never written again.
        let zeros: Vec<u8> = vec![0u8; (dof_total as usize) * core::mem::size_of::<f32>()];
        molla_core::gpu_write_buffer!(queue, &zero_dof_buf, 0, &zeros);

        let uniform_buffer = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::featherstone uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::featherstone uniforms bg"),
            layout: &program.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            specialized_joint_types: program.specialized_joint_types.clone(),
            split_acceleration: program.split_acceleration,
            external_prediction_only: program.external_prediction_only,
            execution: program.execution,
            unrestricted_drift_pipeline: program.unrestricted_drift_pipeline.clone(),
            drift_backend: program.drift_backend,
            dynamics_pipeline: program.dynamics_pipeline.clone(),
            dynamics_workgroup_size: program.dynamics_workgroup_size,
            #[cfg(test)]
            force_generic_drift: false,
            #[cfg(test)]
            force_generic_dynamics: false,
            layout_model,
            layout_state,
            layout_control,
            layout_arts,
            layout_cable_muscle,
            art_offsets,
            stop_status,
            active_articulations,
            velocity_delta,
            active_upload,
            zero_dof_buf,
            bindings: std::array::from_fn(|_| None),
            next_binding: 0,
            uniform_buffer,
            uniform_group,
            bind_groups_created: 1,
            empty_state: std::array::from_fn(|_| {
                molla_core::gpu_create_buffer!(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some("molla::featherstone empty state"),
                        size: 4,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }
                )
            }),
            articulation_count,
            joint_count,
            dof_total,
            stable_motor_feedback: false,
            external_scalar_bounds: false,
            mixed_stop_rows,
            coordinate_count: model.joint_coord_total,
            body_count: model.body_count,
            coordinate_states: Vec::new(),
            rate_states: Vec::new(),
            paired_rate_program: program.paired_rates,
            paired_delta: None,
            body_velocity_fallback: molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla disabled body velocity compensation"),
                    size: 32,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }
            ),
            rate_fallback: std::array::from_fn(|_| {
                molla_core::gpu_create_buffer!(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some("molla disabled paired rates"),
                        size: 8,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }
                )
            }),
            dispatch_timestamps: None,
            coordinate_fallback: molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla disabled coordinate compensation"),
                    size: model.joint_coord_total.max(1) as u64 * 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }
            ),
            position_fallback: molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla disabled position compensation"),
                    size: model.body_count.max(1) as u64 * 16,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }
            ),
        }
    }

    /// Sticky failures: scalar solve bit 5 and nonfinite position displacement bit 31.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.stop_status
    }

    pub fn bind_groups_created(&self) -> u64 {
        self.bind_groups_created
    }

    /// Prepare zeroed coordinate and FK translation residues for a fixed state set.
    /// Reset after coordinate/pose edits or foreign writes; rebuild after topology changes.
    /// Empty state lists disable compensation. Zero-coordinate bodies retain pose residues.
    /// Host-authored low parts can be initialized with `seed_coordinate_compensation`.
    pub fn prepare_coordinate_compensation(
        &mut self,
        device: &wgpu::Device,
        states: &[&State],
    ) -> Result<()> {
        let mut prepared: Vec<CompensatedState> = Vec::with_capacity(states.len());
        for state in states {
            if state.joint_q.len() != self.coordinate_count
                || self.body_count == 0
                || state.body_q.len() != self.body_count
            {
                return Err(Error::Solver(
                    "invalid compensated coordinate state size".into(),
                ));
            }
            let coordinates = if self.coordinate_count == 0 {
                None
            } else {
                Some(state.joint_q.device_buffer_current()?.clone())
            };
            let poses = state.body_q.device_buffer_current()?;
            if prepared.iter().any(|s| {
                (coordinates.is_some() && s.coordinates == coordinates) || &s.poses == poses
            }) {
                return Err(Error::Solver(
                    "duplicate compensated coordinate state".into(),
                ));
            }
            let error = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla coordinate rounding residues"),
                    size: self.coordinate_count.max(1) as u64 * 4,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            );
            let position_error = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla body position rounding residues"),
                    size: self.body_count.max(1) as u64 * 16,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            );
            prepared.push(CompensatedState {
                coordinates,
                poses: poses.clone(),
                coordinate_error: error,
                position_error,
            });
        }
        self.coordinate_states = prepared;
        self.bindings = std::array::from_fn(|_| None);
        Ok(())
    }

    /// Seed prepared residues from current uploaded host coordinates and translations.
    pub fn seed_coordinate_compensation(&self, queue: &wgpu::Queue, state: &State) -> Result<()> {
        let prepared = self.compensated_state(state)?;
        let coordinates: Vec<f32> = state
            .joint_q
            .host()?
            .iter()
            .map(|&value| (value - value as f32 as f64) as f32)
            .collect();
        let positions: Vec<[f32; 4]> = state
            .body_q
            .host()?
            .iter()
            .map(|pose| {
                let p = pose.position;
                [
                    (p.x - p.x as f32 as f64) as f32,
                    (p.y - p.y as f32 as f64) as f32,
                    (p.z - p.z as f32 as f64) as f32,
                    0.,
                ]
            })
            .collect();
        if coordinates.iter().any(|v| !v.is_finite())
            || positions.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(Error::Solver("nonfinite host coordinate residue".into()));
        }
        if !coordinates.is_empty() {
            queue.write_buffer(
                &prepared.coordinate_error,
                0,
                bytemuck::cast_slice(&coordinates),
            );
        }
        queue.write_buffer(
            &prepared.position_error,
            0,
            bytemuck::cast_slice(&positions),
        );
        Ok(())
    }

    /// Clear coordinate and body translation residues at an explicit edit boundary.
    pub fn encode_reset_coordinate_compensation(&self, encoder: &mut wgpu::CommandEncoder) {
        for state in &self.coordinate_states {
            encoder.clear_buffer(&state.coordinate_error, 0, None);
            encoder.clear_buffer(&state.position_error, 0, None);
        }
    }

    /// Per-coordinate f32 residues; checkpoint and restore together with coordinates.
    pub fn coordinate_compensation_buffer(&self, state: &State) -> Result<&wgpu::Buffer> {
        Ok(&self.compensated_state(state)?.coordinate_error)
    }

    /// Per-body xyz translation residues in vec4<f32>; checkpoint with body poses.
    pub fn position_compensation_buffer(&self, state: &State) -> Result<&wgpu::Buffer> {
        Ok(&self.compensated_state(state)?.position_error)
    }

    fn compensated_state(&self, state: &State) -> Result<&CompensatedState> {
        let q = if self.coordinate_count == 0 {
            None
        } else {
            Some(state.joint_q.device_buffer_current()?)
        };
        let poses = state.body_q.device_buffer_current()?;
        self.coordinate_states
            .iter()
            .find(|s| s.coordinates.as_ref() == q && &s.poses == poses)
            .ok_or_else(|| Error::Solver("unprepared compensated coordinate state".into()))
    }

    /// Select whole articulations; inactive state is copied without dynamics or drift.
    pub fn active_articulations_buffer(&self) -> &wgpu::Buffer {
        &self.active_articulations
    }

    pub fn set_active_articulations(&mut self, queue: &wgpu::Queue, active: &[bool]) -> Result<()> {
        if active.len() != self.articulation_count as usize
            && !(self.joint_count == 0 && active.is_empty())
        {
            return Err(Error::Solver("invalid GPU articulation selection".into()));
        }
        for (target, &active) in self.active_upload.iter_mut().zip(active) {
            *target = u32::from(active);
        }
        molla_core::gpu_write_buffer!(
            queue,
            &self.active_articulations,
            0,
            bytemuck::cast_slice(&self.active_upload),
        );
        Ok(())
    }

    /// Enable effective-inertia stabilization of joint feedback drives.
    pub fn set_stable_motor_feedback(&mut self, enabled: bool) {
        self.stable_motor_feedback = enabled;
    }

    /// Delegate nonlocked scalar position and velocity bounds to a coupled solver.
    pub fn set_external_scalar_bounds(&mut self, enabled: bool) {
        self.external_scalar_bounds = enabled;
    }

    pub fn set_status_buffer(&mut self, buffer: &wgpu::Buffer) -> Result<()> {
        if buffer.size() < 4 || !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(Error::Solver("invalid Featherstone status buffer".into()));
        }
        self.stop_status = buffer.clone();
        Ok(())
    }

    /// Borrow the cable+muscle bind-group layout so external callers
    /// (e.g. an integration test) can build their own custom bind
    /// group. The dispatch path uses a private path.
    pub fn cable_muscle_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout_cable_muscle
    }

    /// Cached joint DoF total for sizing externally-supplied muscle
    /// torque buffers.
    pub fn dof_total(&self) -> u32 {
        self.dof_total
    }

    /// Run one Featherstone step on the GPU. Caller must have
    /// uploaded the model + state + control beforehand.
    ///
    /// `muscle_torque` (optional) is a per-DOF buffer of `f32` of
    /// length `dof_total` containing the pre-computed Hill-type muscle
    /// torque sum for every DoF (zero on non-muscle DoFs). Pass
    /// `None` to use the kernel's internal zero buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
        joint_damping: f32,
        dt: f64,
        rigorous_d_solve: bool,
        rigorous_distance: bool,
        state_out: &mut State,
        muscle_torque: Option<&wgpu::Buffer>,
    ) -> Result<()> {
        if contacts.count != 0 {
            return Err(Error::Solver(
                "GPU Featherstone contact response is not implemented".into(),
            ));
        }
        self.dispatch_internal(
            device,
            queue,
            model,
            state_in,
            control,
            joint_damping,
            dt,
            rigorous_d_solve,
            rigorous_distance,
            state_out,
            muscle_torque,
            false,
            None,
            false,
        )
    }

    /// Per-DOF predictor velocity changes retained across drift-only dispatches.
    pub fn velocity_delta_buffer(&self) -> &wgpu::Buffer {
        &self.velocity_delta
    }

    /// Integrate input coordinates with output velocities and evaluate FK without applying forces.
    #[allow(clippy::too_many_arguments)]
    pub fn drift(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        control: &Control,
        dt: f64,
        rigorous_distance: bool,
        state_out: &mut State,
    ) -> Result<()> {
        self.dispatch_internal(
            device,
            queue,
            model,
            state_in,
            control,
            0.0,
            dt,
            false,
            rigorous_distance,
            state_out,
            None,
            true,
            None,
            false,
        )
    }

    /// Apply generalized coordinate displacements and rebuild FK without advancing time.
    /// Physical joint rates and prescribed body motion are preserved.
    /// Supply one f32 per DOF with bounds solved upstream; velocity caps are not applied.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_position_displacement(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        control: &Control,
        displacement: &wgpu::Buffer,
        state_out: &mut State,
    ) -> Result<()> {
        if displacement.size() < self.dof_total.max(1) as u64 * 4
            || !displacement.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(Error::Solver("invalid position displacement buffer".into()));
        }
        self.dispatch_internal(
            device,
            queue,
            model,
            state_in,
            control,
            0.0,
            1.0,
            false,
            false,
            state_out,
            None,
            true,
            Some(displacement),
            false,
        )
    }

    /// Rebuild body poses and velocities from coordinates without advancing time or forces.
    pub fn refresh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        control: &Control,
        state_out: &mut State,
    ) -> Result<()> {
        self.dispatch_internal(
            device, queue, model, state_in, control, 0.0, 0.0, false, false, state_out, None, true,
            None, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_internal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        control: &Control,
        joint_damping: f32,
        dt: f64,
        rigorous_d_solve: bool,
        rigorous_distance: bool,
        state_out: &mut State,
        muscle_torque: Option<&wgpu::Buffer>,
        drift_only: bool,
        displacement: Option<&wgpu::Buffer>,
        refresh_only: bool,
    ) -> Result<()> {
        let timestamps = self.dispatch_timestamps.take();
        if self.external_prediction_only && !drift_only {
            return Err(Error::Solver("external prediction program cannot dispatch ABA; use prepared prediction and drift".into()));
        }
        joint_types::validate_model(model, self.specialized_joint_types.as_deref())?;
        if (self.split_acceleration || self.external_prediction_only)
            && !rates::unrestricted_model(model)?
        {
            return Err(Error::Solver(
                "split paired prediction requires an unrestricted model".into(),
            ));
        }
        if model.joint_count == 0 {
            return Ok(());
        }
        if !self.external_scalar_bounds && mixed_stop_capacity(model)? > self.mixed_stop_rows {
            return Err(Error::Gpu(
                "mixed joint stops exceed retained row capacity; recreate the kernel".into(),
            ));
        }
        validate_compliance(model)?;
        let reversed = model.joint_reversed.host()?;
        if reversed.len() != model.joint_count || reversed.iter().any(|&value| value > 1) {
            return Err(Error::Gpu("invalid GPU joint traversal flags".into()));
        }
        if !control.body_external_f.is_empty() && control.body_external_f.len() != self.body_count {
            return Err(Error::Gpu(
                "control body wrench count does not match bodies".into(),
            ));
        }
        let compensation = if self.coordinate_states.is_empty() {
            None
        } else {
            let find = |state: &State| -> Result<&CompensatedState> {
                let q = if self.coordinate_count == 0 {
                    None
                } else {
                    Some(state.joint_q.device_buffer_current()?)
                };
                let poses = state.body_q.device_buffer_current()?;
                self.coordinate_states
                    .iter()
                    .find(|s| s.coordinates.as_ref() == q && &s.poses == poses)
                    .ok_or_else(|| Error::Solver("unprepared compensated coordinate state".into()))
            };
            Some((find(state_in)?, find(state_out)?))
        };
        let rates = if self.rate_states.is_empty() {
            None
        } else {
            if compensation.is_none() {
                return Err(Error::Solver(
                    "paired rates require coordinate compensation".into(),
                ));
            }
            let find = |state: &State| -> Result<&rates::RateState> {
                let mirror = state.joint_qd.device_buffer_current()?;
                let body = state.body_qd.device_buffer_current()?;
                self.rate_states
                    .iter()
                    .find(|s| &s.mirror == mirror && &s.body_mirror == body)
                    .ok_or_else(|| Error::Solver("unprepared paired generalized rate state".into()))
            };
            Some((find(state_in)?, find(state_out)?))
        };
        if self.stable_motor_feedback && !drift_only && !rigorous_d_solve {
            return Err(Error::Solver(
                "GPU stable feedback requires rigorous dynamics".into(),
            ));
        }

        // Copy state_in's body_q + joint coords into state_out so the
        // kernel mutates state_out's device buffers in place — same
        // pattern as the XPBD GPU dispatch. body_f is also propagated
        // so substepping callers see the same external wrench across
        // every substep (until they overwrite it).
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::featherstone encoder"),
        });
        if !drift_only {
            if let Some(delta) = &self.paired_delta {
                encoder.clear_buffer(delta, 0, None);
            }
        }
        copy_buffer(&mut encoder, &state_in.body_q, &state_out.body_q)?;
        copy_buffer(&mut encoder, &state_in.body_qd, &state_out.body_qd)?;
        if let Some((input, output)) = rates {
            encoder.copy_buffer_to_buffer(
                &input.body_low,
                0,
                &output.body_low,
                0,
                self.body_count as u64 * 32,
            );
        }
        copy_buffer(&mut encoder, &state_in.joint_q, &state_out.joint_q)?;
        if let Some((input, output)) = compensation {
            if self.coordinate_count != 0 {
                encoder.copy_buffer_to_buffer(
                    &input.coordinate_error,
                    0,
                    &output.coordinate_error,
                    0,
                    self.coordinate_count as u64 * 4,
                );
            }
            encoder.copy_buffer_to_buffer(
                &input.position_error,
                0,
                &output.position_error,
                0,
                self.body_count as u64 * 16,
            );
        }
        if !drift_only || displacement.is_some() || refresh_only {
            copy_buffer(&mut encoder, &state_in.joint_qd, &state_out.joint_qd)?;
            if let Some((input, output)) = rates {
                encoder.copy_buffer_to_buffer(
                    &input.pairs,
                    0,
                    &output.pairs,
                    0,
                    self.dof_total as u64 * 8,
                );
            }
        }
        copy_buffer(&mut encoder, &state_in.body_f, &state_out.body_f)?;

        let compensated_coordinates = u32::from(compensation.is_some());
        let buffers = [
            binding_buffer(&model.joint_type, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_parent, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_child, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_x_p, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_x_c, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_axis, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_coord_offset, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_dof_offset, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_limit_lower, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_limit_upper, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_friction, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_effort_limit, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_velocity_limit, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_compliance, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_reversed, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_motion_flags, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_limit_references, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_frame_references, &self.zero_dof_buf)?,
            binding_buffer(&model.body_mass, &self.zero_dof_buf)?,
            binding_buffer(&model.body_inertia, &self.zero_dof_buf)?,
            binding_buffer(&model.body_com, &self.zero_dof_buf)?,
            binding_buffer(&model.body_world, &self.zero_dof_buf)?,
            binding_buffer(&model.gravity, &self.zero_dof_buf)?,
            binding_buffer(&state_out.body_q, &self.zero_dof_buf)?,
            binding_buffer(&state_out.joint_q, &self.empty_state[0])?,
            binding_buffer(&state_out.joint_qd, &self.empty_state[1])?,
            binding_buffer(&state_in.body_f, &self.zero_dof_buf)?,
            binding_buffer(&state_out.body_qd, &self.zero_dof_buf)?,
            binding_buffer(&model.body_flags, &self.zero_dof_buf)?,
            binding_buffer(&model.body_kinematic_acceleration, &self.zero_dof_buf)?,
            binding_buffer(&model.body_inv_inertia, &self.zero_dof_buf)?,
            binding_buffer(&control.joint_force, &self.zero_dof_buf)?,
            binding_buffer(&control.joint_target_mode, &self.zero_dof_buf)?,
            binding_buffer(&control.joint_target_pos, &self.zero_dof_buf)?,
            binding_buffer(&control.joint_target_vel, &self.zero_dof_buf)?,
            binding_buffer(&control.joint_target_kp, &self.zero_dof_buf)?,
            binding_buffer(&control.joint_target_kd, &self.zero_dof_buf)?,
            binding_buffer(&control.joint_motor_max_force, &self.zero_dof_buf)?,
            binding_buffer(&control.body_external_f, &self.zero_dof_buf)?,
            binding_buffer(&control.joint_motor_model, &self.zero_dof_buf)?,
            binding_buffer(&control.joint_damping, &self.zero_dof_buf)?,
            &self.art_offsets,
            &self.stop_status,
            &self.active_articulations,
            displacement.unwrap_or(&self.velocity_delta),
            compensation
                .map(|(_, output)| &output.coordinate_error)
                .unwrap_or(&self.coordinate_fallback),
            compensation
                .map(|(_, output)| &output.position_error)
                .unwrap_or(&self.position_fallback),
            rates
                .map(|(_, output)| &output.pairs)
                .unwrap_or(&self.rate_fallback[0]),
            self.paired_delta.as_ref().unwrap_or(&self.rate_fallback[1]),
            rates
                .map(|(_, output)| &output.body_low)
                .unwrap_or(&self.body_velocity_fallback),
            binding_buffer(&model.joint_cable_stiffness_linear, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_cable_stiffness_bend, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_cable_damping_linear, &self.zero_dof_buf)?,
            binding_buffer(&model.joint_cable_damping_bend, &self.zero_dof_buf)?,
            muscle_torque.unwrap_or(&self.zero_dof_buf),
        ];
        let slot = if let Some(slot) = self
            .bindings
            .iter()
            .position(|entry| entry.as_ref().is_some_and(|entry| entry.matches(&buffers)))
        {
            slot
        } else {
            let slot = self.next_binding;
            self.next_binding = (slot + 1) % self.bindings.len();
            self.bind_groups_created += 5;
            self.bindings[slot] = Some(FeatherstoneBindings::new(
                device,
                [
                    &self.layout_model,
                    &self.layout_state,
                    &self.layout_control,
                    &self.layout_arts,
                    &self.layout_cable_muscle,
                ],
                buffers,
            ));
            slot
        };
        let bindings = self.bindings[slot].as_ref().unwrap();

        let uniforms = Uniforms {
            dt: dt as f32,
            joint_damping,
            joint_count: self.joint_count,
            articulation_count: self.articulation_count,
            rigorous_d_solve: u32::from(rigorous_d_solve),
            rigorous_distance: u32::from(rigorous_distance),
            drift_only: if refresh_only {
                3
            } else if displacement.is_some() {
                2
            } else {
                u32::from(drift_only)
            },
            stable_motor_feedback: u32::from(self.stable_motor_feedback),
            external_scalar_bounds: u32::from(self.external_scalar_bounds),
            compensated_coordinates,
            control_body_wrenches: u32::from(!control.body_external_f.is_empty()),
            paired_rates: u32::from(rates.is_some()),
        };
        molla_core::gpu_write_buffer!(
            queue,
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms)
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::featherstone pass"),
                timestamp_writes: timestamps.as_ref().map(|(query_set, indices)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: indices[0],
                        end_of_pass_write_index: indices[1],
                    }
                }),
            });
            let specialized = uniforms.drift_only == 1 && unrestricted_drift(model)?;
            #[cfg(test)]
            let specialized = specialized && !self.force_generic_drift;
            let pipeline = if specialized {
                &self.unrestricted_drift_pipeline
            } else {
                &self.pipeline
            };
            let dynamics = uniforms.drift_only == 0;
            #[cfg(test)]
            let dynamics = dynamics && !self.force_generic_dynamics;
            let pipeline = if dynamics {
                &self.dynamics_pipeline
            } else {
                pipeline
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bindings.groups[0], &[]);
            pass.set_bind_group(1, &bindings.groups[1], &[]);
            pass.set_bind_group(2, &bindings.groups[2], &[]);
            pass.set_bind_group(3, &bindings.groups[3], &[]);
            pass.set_bind_group(4, &self.uniform_group, &[]);
            pass.set_bind_group(5, &bindings.groups[4], &[]);
            let workgroup_size = if dynamics {
                self.dynamics_workgroup_size
            } else if self.paired_rate_program {
                1
            } else {
                WORKGROUP_SIZE
            };
            let groups = self.articulation_count.div_ceil(workgroup_size);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
            if dynamics && self.split_acceleration {
                pass.set_pipeline(&self.pipeline);
                pass.dispatch_workgroups(groups.max(1), 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
        }
        queue.submit(std::iter::once(encoder.finish()));

        state_out.body_q.mark_device_dirty();
        state_out.body_qd.mark_device_dirty();
        state_out.joint_q.mark_device_dirty();
        state_out.joint_qd.mark_device_dirty();
        Ok(())
    }
}

fn binding_buffer<'a, T: BufferElement>(
    buffer: &'a Buffer<T>,
    empty: &'a wgpu::Buffer,
) -> Result<&'a wgpu::Buffer> {
    let buf = if buffer.is_empty() {
        Some(empty)
    } else {
        buffer.device_buffer()
    };
    buf.ok_or_else(|| {
        Error::Gpu("featherstone binding: no device mirror — call upload_to_device first".into())
    })
}

fn storage_entries(count: u32, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

fn state_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let read = wgpu::BufferBindingType::Storage { read_only: true };
    let rw = wgpu::BufferBindingType::Storage { read_only: false };
    (0..13u32)
        .map(|i| {
            // 0..4: read-only model body data; 5..7: rw state;
            // 8: read-only body_f (external wrenches at COM, world).
            let ty = if i < 5 || i == 8 || i >= 10 { read } else { rw };
            wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        })
        .collect()
}

/// GPU-to-GPU buffer copy used to seed `state_out` from `state_in`
/// before the kernel mutates the output buffers in place.
fn copy_buffer<T: BufferElement>(
    encoder: &mut wgpu::CommandEncoder,
    src: &Buffer<T>,
    dst: &Buffer<T>,
) -> Result<()> {
    if src.is_empty() && dst.is_empty() {
        return Ok(());
    }
    let s = src
        .device_buffer()
        .ok_or_else(|| Error::Gpu("copy_buffer: src has no device mirror".into()))?;
    let d = dst
        .device_buffer()
        .ok_or_else(|| Error::Gpu("copy_buffer: dst has no device mirror".into()))?;
    let size = s.size().min(d.size());
    encoder.copy_buffer_to_buffer(s, 0, d, 0, size);
    Ok(())
}

#[cfg(test)]
mod scratch_tests {
    use super::{articulation_scratch_capacity, kernel_configuration};

    #[test]
    fn scratch_capacity_tracks_largest_articulation_not_total_scene() {
        assert_eq!(articulation_scratch_capacity(&[]), 1);
        assert_eq!(articulation_scratch_capacity(&[0, 0]), 1);
        assert_eq!(
            articulation_scratch_capacity(&(0..=1000).collect::<Vec<_>>()),
            1
        );
        assert_eq!(articulation_scratch_capacity(&[0, 1, 8, 10]), 7);
        assert_eq!(articulation_scratch_capacity(&[0, 32, 33]), 32);
        assert_eq!(articulation_scratch_capacity(&[0, 33]), 32);
    }

    #[test]
    fn program_configuration_tracks_mixed_row_mutations() {
        use molla_core::BodyId;
        use molla_math::Transform;
        use molla_sim::{BodyParams, ModelBuilder};
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(BodyParams::default());
        builder
            .add_joint_ball(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let mut model = builder.build().unwrap();
        assert_eq!(kernel_configuration(&model), (vec![0, 1], 1));
        model.joint_limit_lower.host_mut().unwrap()[..2].fill(-0.25);
        model.joint_limit_upper.host_mut().unwrap()[..2].fill(0.25);
        assert_eq!(kernel_configuration(&model), (vec![0, 1], 4));
        model.joint_velocity_limit.host_mut().unwrap()[..2].fill(0.5);
        assert_eq!(kernel_configuration(&model), (vec![0, 1], 6));
        model.joint_enabled.host_mut().unwrap()[0] = 0;
        assert_eq!(kernel_configuration(&model), (vec![0, 1], 1));
    }
}
