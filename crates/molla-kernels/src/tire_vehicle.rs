//! GPU vehicle-level tire/chassis coupling.
//!
//! This is intentionally compact: it sums per-wheel tire forces and aligning
//! moments with the same planar wrench equations used by
//! `molla_vehicle::Vehicle`, then integrates chassis translation plus
//! yaw/yaw-rate on GPU. It is a first GPU chassis-state owner for tire
//! wrenches, not a full suspension/wheel-assembly solver.

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};

use crate::tire_brush::{
    BrushTerrain, BrushWheelInput, BrushWheelOutput, TireBrushGpuKernel, TireBrushResidentOutput,
};
use crate::tire_fem::{
    TireFemContactFeedbackOutput, TireFemContactFeedbackResidentOutput, TireFemGpuBatchAtlasLayout,
};

const WORKGROUP_SIZE: u32 = 64;

#[path = "tire_vehicle_prepared_step.rs"]
mod prepared_step;
pub use prepared_step::PreparedTireVehicleChassisStep;

#[derive(Clone, Debug)]
pub struct TireVehicleYawGpuInput {
    /// Wheel offsets from chassis COM in world/body planar coordinates.
    /// Uses x/z for yaw moment arm; y is ignored.
    pub wheel_offsets: Vec<[f32; 4]>,
    /// Tire force xyz plus self-aligning yaw moment in w.
    pub tire_wrenches: Vec<[f32; 4]>,
    pub yaw: f32,
    pub yaw_rate: f32,
    pub yaw_inertia: f32,
    pub yaw_damping: f32,
    pub dt: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TireVehicleYawGpuOutput {
    pub yaw: f32,
    pub yaw_rate: f32,
    pub yaw_moment: f32,
    pub finite: bool,
}

#[derive(Clone, Debug)]
pub struct TireVehiclePlanarGpuInput {
    /// Wheel offsets from chassis COM in world/body planar coordinates.
    /// Uses x/z for yaw moment arm; y is ignored.
    pub wheel_offsets: Vec<[f32; 4]>,
    /// Tire force xyz plus self-aligning yaw moment in w.
    pub tire_wrenches: Vec<[f32; 4]>,
    pub position_xz: [f32; 2],
    pub velocity_xz: [f32; 2],
    pub yaw: f32,
    pub yaw_rate: f32,
    pub mass: f32,
    pub yaw_inertia: f32,
    pub linear_damping: f32,
    pub yaw_damping: f32,
    pub dt: f32,
}

#[derive(Clone, Debug)]
pub struct TireVehiclePlanarFromBrushGpuInput {
    /// Wheel offsets from chassis COM in world/body planar coordinates.
    /// Uses x/z for yaw moment arm; y is ignored.
    pub wheel_offsets: Vec<[f32; 4]>,
    pub position_xz: [f32; 2],
    pub velocity_xz: [f32; 2],
    pub yaw: f32,
    pub yaw_rate: f32,
    pub mass: f32,
    pub yaw_inertia: f32,
    pub linear_damping: f32,
    pub yaw_damping: f32,
    pub dt: f32,
}

pub struct TireVehiclePlanarResidentState {
    state_buffers: [wgpu::Buffer; 2],
    wheel_offsets: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    wheel_count: usize,
    current_state: usize,
    mass: f32,
    yaw_inertia: f32,
    linear_damping: f32,
    yaw_damping: f32,
}

#[derive(Clone, Debug)]
pub struct TireVehicleVerticalGpuInput {
    pub y: f32,
    pub vy: f32,
    pub mass: f32,
    pub gravity: f32,
    pub vertical_damping: f32,
    pub dt: f32,
}

pub struct TireVehicleVerticalResidentState {
    state_buffers: [wgpu::Buffer; 2],
    uniforms: wgpu::Buffer,
    wheel_count: usize,
    current_state: usize,
    mass: f32,
    gravity: f32,
    vertical_damping: f32,
}

impl TireVehicleVerticalResidentState {
    pub const fn wheel_count(&self) -> usize {
        self.wheel_count
    }

    pub fn current_state_buffer(&self) -> &wgpu::Buffer {
        &self.state_buffers[self.current_state]
    }
}

impl TireVehiclePlanarResidentState {
    pub const fn wheel_count(&self) -> usize {
        self.wheel_count
    }

    pub fn current_state_buffer(&self) -> &wgpu::Buffer {
        &self.state_buffers[self.current_state]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TireVehiclePlanarGpuOutput {
    pub position_xz: [f32; 2],
    pub velocity_xz: [f32; 2],
    pub yaw: f32,
    pub yaw_rate: f32,
    pub force_xz: [f32; 2],
    pub yaw_moment: f32,
    pub finite: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TireVehicleVerticalGpuOutput {
    pub y: f32,
    pub vy: f32,
    pub force_y: f32,
    pub finite: bool,
}

#[derive(Clone, Debug)]
pub struct TireVehicleWheelSpinGpuInput {
    /// Per-wheel state rows: x=spin_rad_s, y=wheel_radius_m,
    /// z=spin_inertia_kg_m2, w=reserved.
    pub wheel_state: Vec<[f32; 4]>,
    /// Per-wheel command rows: x=drive_torque_nm, y=brake_torque_nm,
    /// z=longitudinal_tire_force_n, w=reserved.
    pub wheel_commands: Vec<[f32; 4]>,
    pub dt: f32,
}

#[derive(Clone, Debug, Default)]
pub struct TireVehicleWheelSpinGpuOutput {
    /// Per-wheel output rows: x=next_spin_rad_s, y=spin_accel_rad_s2,
    /// z=net_torque_nm, w=finite_flag.
    pub rows: Vec<[f32; 4]>,
    pub finite: bool,
}

#[derive(Clone, Debug, Default)]
pub struct TireVehicleWheelSpinStateGpuOutput {
    /// Per-wheel state-compatible rows: x=spin_rad_s, y=wheel_radius_m,
    /// z=spin_inertia_kg_m2, w=finite_flag.
    pub rows: Vec<[f32; 4]>,
    pub finite: bool,
}

pub struct TireVehicleWheelSpinResidentBatch {
    state_buffers: [wgpu::Buffer; 2],
    commands: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    wheel_count: usize,
    current_state: usize,
}

impl TireVehicleWheelSpinResidentBatch {
    pub const fn wheel_count(&self) -> usize {
        self.wheel_count
    }

    pub fn current_state_buffer(&self) -> &wgpu::Buffer {
        &self.state_buffers[self.current_state]
    }
}

#[derive(Clone, Debug)]
pub struct TireVehicleWheelKinematicsGpuInput {
    /// Per-wheel mount rows in chassis frame: x=attach_x_m, y=attach_y_m,
    /// z=attach_z_m, w=wheel_radius_m.
    pub wheel_mounts: Vec<[f32; 4]>,
    /// Per-wheel state rows: x=spin_rad_s, y=steer_rad, zw=reserved.
    pub wheel_state: Vec<[f32; 4]>,
    pub velocity_xz: [f32; 2],
    pub yaw: f32,
    pub yaw_rate: f32,
}

#[derive(Clone, Debug)]
pub struct TireVehicleWheelKinematicsFromSpinGpuInput {
    /// Per-wheel mount rows in chassis frame: x=attach_x_m, y=attach_y_m,
    /// z=attach_z_m, w=wheel_radius_fallback_m.
    pub wheel_mounts: Vec<[f32; 4]>,
    /// Per-wheel controls: x=steer_rad, yzw=reserved.
    pub wheel_controls: Vec<[f32; 4]>,
    pub velocity_xz: [f32; 2],
    pub yaw: f32,
    pub yaw_rate: f32,
}

#[derive(Clone, Debug, Default)]
pub struct TireVehicleWheelKinematicsGpuOutput {
    /// Row `i`: x=v_long_m_s, y=v_lat_m_s, z=slip_ratio, w=slip_angle_rad.
    /// Row `wheel_count + i`: x=long_x, y=long_z, z=lat_x, w=lat_z.
    pub rows: Vec<[f32; 4]>,
    pub finite: bool,
}

#[derive(Clone, Debug)]
pub struct TireVehicleSuspensionGpuInput {
    /// Per-wheel geometry rows: x=attach_y_m, y=wheel_radius_m,
    /// z=rest_length_m, w=max_compression_m.
    pub suspension_geometry: Vec<[f32; 4]>,
    /// Per-wheel parameter rows: x=stiffness_n_m, y=damping_ns_m,
    /// z=ground_height_m, w=reserved.
    pub suspension_params: Vec<[f32; 4]>,
    pub chassis_y: f32,
    pub chassis_vy: f32,
}

#[derive(Clone, Debug, Default)]
pub struct TireVehicleSuspensionGpuOutput {
    /// Per-wheel output rows: x=normal_load_n, y=travel_m,
    /// z=travel_rate_m_s, w=finite_flag.
    pub rows: Vec<[f32; 4]>,
    pub finite: bool,
}

pub struct TireVehicleSuspensionResidentBatch {
    geometry: wgpu::Buffer,
    params: wgpu::Buffer,
    rows: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    wheel_count: usize,
}

impl TireVehicleSuspensionResidentBatch {
    pub const fn wheel_count(&self) -> usize {
        self.wheel_count
    }

    pub fn rows_buffer(&self) -> &wgpu::Buffer {
        &self.rows
    }
}

pub struct TireVehicleResidentBrushOwner {
    pub spin: TireVehicleWheelSpinResidentBatch,
    pub suspension: TireVehicleSuspensionResidentBatch,
    pub brush_input: TireVehicleBrushRowsFromSpinGpuInput,
}

impl TireVehicleResidentBrushOwner {
    pub const fn wheel_count(&self) -> usize {
        self.spin.wheel_count()
    }
}

pub struct TireVehicleResidentChassisBrushOwner {
    pub brush: TireVehicleResidentBrushOwner,
    pub planar: TireVehiclePlanarResidentState,
    pub vertical: TireVehicleVerticalResidentState,
}

impl TireVehicleResidentChassisBrushOwner {
    pub const fn wheel_count(&self) -> usize {
        self.brush.wheel_count()
    }
}

#[derive(Clone, Debug)]
pub struct TireVehicleBrushRowsGpuInput {
    /// Chassis world pose: x,y,z,yaw.
    pub chassis_pose: [f32; 4],
    /// Chassis velocity: vx,vy,vz,yaw_rate.
    pub chassis_velocity: [f32; 4],
    /// Per-wheel mount rows in chassis frame: x=attach_x_m, y=attach_y_m,
    /// z=attach_z_m, w=wheel_radius_m.
    pub wheel_mounts: Vec<[f32; 4]>,
    /// Per-wheel state rows: x=spin_rad_s, y=steer_rad, z=width_m,
    /// w=pressure_pa.
    pub wheel_state: Vec<[f32; 4]>,
    /// Per-wheel width strip counts, 1..=16; zero selects three strips.
    pub width_segments: Vec<u32>,
    pub tire_rows: Vec<[f32; 4]>,
    pub material_rows: Vec<[f32; 4]>,
    pub extra_rows: Vec<[f32; 4]>,
    /// Per-wheel suspension/contact rows: x=travel_m, y=travel_rate_m_s,
    /// z=ground_height_m, w=finite_flag.
    pub suspension_rows: Vec<[f32; 4]>,
}

#[derive(Clone, Debug)]
pub struct TireVehicleBrushRowsFromSpinGpuInput {
    /// Chassis world pose: x,y,z,yaw.
    pub chassis_pose: [f32; 4],
    /// Chassis velocity: vx,vy,vz,yaw_rate.
    pub chassis_velocity: [f32; 4],
    /// Per-wheel resident-compatible spin state rows:
    /// x=spin_rad_s, y=wheel_radius_m, z=spin_inertia_kg_m2, w=finite_flag.
    pub spin_state_rows: Vec<[f32; 4]>,
    /// Per-wheel mount rows in chassis frame: x=attach_x_m, y=attach_y_m,
    /// z=attach_z_m, w=wheel_radius_fallback_m.
    pub wheel_mounts: Vec<[f32; 4]>,
    /// Per-wheel controls: x=steer_rad, y=width_m, z=pressure_pa,
    /// w=width_segments (1..=16; zero selects three strips).
    pub wheel_controls: Vec<[f32; 4]>,
    pub tire_rows: Vec<[f32; 4]>,
    pub material_rows: Vec<[f32; 4]>,
    pub extra_rows: Vec<[f32; 4]>,
    /// Per-wheel suspension/contact rows: x=travel_m, y=travel_rate_m_s,
    /// z=ground_height_m, w=finite_flag.
    pub suspension_rows: Vec<[f32; 4]>,
}

#[derive(Clone, Debug, Default)]
pub struct TireVehicleBrushRowsGpuOutput {
    pub wheels: Vec<BrushWheelInput>,
    pub finite: bool,
}

pub struct TireVehicleBrushRowsResidentOutput {
    pub buffer: wgpu::Buffer,
    pub wheel_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VehicleUniforms {
    counts: [u32; 4],
    pose: [f32; 4],
    velocity: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
}

pub struct TireVehicleYawGpuKernel {
    device: wgpu::Device,
    pipeline_yaw: wgpu::ComputePipeline,
    pipeline_planar: wgpu::ComputePipeline,
    pipeline_planar_from_brush: wgpu::ComputePipeline,
    pipeline_planar_state_from_brush: wgpu::ComputePipeline,
    pipeline_planar_from_contact_reduction: wgpu::ComputePipeline,
    pipeline_planar_state_from_contact_reduction: wgpu::ComputePipeline,
    pipeline_planar_state_from_feedback: wgpu::ComputePipeline,
    pipeline_vertical_state_from_suspension: wgpu::ComputePipeline,
    pipeline_wheel_spin: wgpu::ComputePipeline,
    pipeline_wheel_spin_state: wgpu::ComputePipeline,
    pipeline_wheel_kinematics: wgpu::ComputePipeline,
    pipeline_wheel_kinematics_from_spin: wgpu::ComputePipeline,
    pipeline_suspension: wgpu::ComputePipeline,
    pipeline_suspension_from_vertical: wgpu::ComputePipeline,
    pipeline_brush_rows: wgpu::ComputePipeline,
    pipeline_brush_rows_from_spin: wgpu::ComputePipeline,
    pipeline_brush_rows_from_spin_suspension: wgpu::ComputePipeline,
    pipeline_brush_rows_from_spin_suspension_planar: wgpu::ComputePipeline,
    pipeline_brush_rows_from_spin_suspension_chassis: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_io4: wgpu::BindGroupLayout,
    layout_io5: wgpu::BindGroupLayout,
    layout_io6: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl TireVehicleYawGpuInput {
    fn validate(&self) -> Result<usize> {
        let n = self.wheel_offsets.len();
        if n == 0 {
            return Err(Error::Solver(
                "tire_vehicle_yaw: at least one wheel is required".into(),
            ));
        }
        if self.tire_wrenches.len() != n {
            return Err(Error::Solver(
                "tire_vehicle_yaw: offsets/wrenches length mismatch".into(),
            ));
        }
        if n > WORKGROUP_SIZE as usize {
            return Err(Error::Solver(format!(
                "tire_vehicle_yaw: wheel count {n} exceeds one-workgroup limit {WORKGROUP_SIZE}"
            )));
        }
        if !self.yaw_inertia.is_finite() || self.yaw_inertia <= 0.0 {
            return Err(Error::Solver(
                "tire_vehicle_yaw: yaw inertia must be positive".into(),
            ));
        }
        if !self.dt.is_finite() || self.dt < 0.0 {
            return Err(Error::Solver(
                "tire_vehicle_yaw: dt must be finite and non-negative".into(),
            ));
        }
        Ok(n)
    }
}

impl TireVehiclePlanarGpuInput {
    fn validate(&self) -> Result<usize> {
        let n = validate_wheels(
            "tire_vehicle_planar",
            &self.wheel_offsets,
            &self.tire_wrenches,
        )?;
        if !self.mass.is_finite() || self.mass <= 0.0 {
            return Err(Error::Solver(
                "tire_vehicle_planar: mass must be positive".into(),
            ));
        }
        if !self.yaw_inertia.is_finite() || self.yaw_inertia <= 0.0 {
            return Err(Error::Solver(
                "tire_vehicle_planar: yaw inertia must be positive".into(),
            ));
        }
        if !self.dt.is_finite() || self.dt < 0.0 {
            return Err(Error::Solver(
                "tire_vehicle_planar: dt must be finite and non-negative".into(),
            ));
        }
        Ok(n)
    }
}

impl TireVehiclePlanarFromBrushGpuInput {
    fn validate(&self, brush_count: usize) -> Result<usize> {
        let n = self.wheel_offsets.len();
        if n == 0 {
            return Err(Error::Solver(
                "tire_vehicle_planar_from_brush: at least one wheel is required".into(),
            ));
        }
        if brush_count != n {
            return Err(Error::Solver(format!(
                "tire_vehicle_planar_from_brush: offset count {n} != brush output count {brush_count}"
            )));
        }
        if n > WORKGROUP_SIZE as usize {
            return Err(Error::Solver(format!(
                "tire_vehicle_planar_from_brush: wheel count {n} exceeds one-workgroup limit {WORKGROUP_SIZE}"
            )));
        }
        if !self.mass.is_finite() || self.mass <= 0.0 {
            return Err(Error::Solver(
                "tire_vehicle_planar_from_brush: mass must be positive".into(),
            ));
        }
        if !self.yaw_inertia.is_finite() || self.yaw_inertia <= 0.0 {
            return Err(Error::Solver(
                "tire_vehicle_planar_from_brush: yaw inertia must be positive".into(),
            ));
        }
        if !self.dt.is_finite() || self.dt < 0.0 {
            return Err(Error::Solver(
                "tire_vehicle_planar_from_brush: dt must be finite and non-negative".into(),
            ));
        }
        Ok(n)
    }
}

impl TireVehicleVerticalGpuInput {
    fn validate(&self, wheel_count: usize) -> Result<usize> {
        if wheel_count == 0 {
            return Err(Error::Solver(
                "tire_vehicle_vertical: at least one wheel is required".into(),
            ));
        }
        if wheel_count > WORKGROUP_SIZE as usize {
            return Err(Error::Solver(format!(
                "tire_vehicle_vertical: wheel count {wheel_count} exceeds one-workgroup limit {WORKGROUP_SIZE}"
            )));
        }
        if !self.mass.is_finite() || self.mass <= 0.0 {
            return Err(Error::Solver(
                "tire_vehicle_vertical: mass must be positive".into(),
            ));
        }
        if !self.dt.is_finite() || self.dt < 0.0 {
            return Err(Error::Solver(
                "tire_vehicle_vertical: dt must be finite and non-negative".into(),
            ));
        }
        Ok(wheel_count)
    }
}

impl TireVehicleWheelSpinGpuInput {
    fn validate(&self) -> Result<usize> {
        let n = self.wheel_state.len();
        if n == 0 {
            return Err(Error::Solver(
                "tire_vehicle_wheel_spin: at least one wheel is required".into(),
            ));
        }
        if self.wheel_commands.len() != n {
            return Err(Error::Solver(
                "tire_vehicle_wheel_spin: state/command length mismatch".into(),
            ));
        }
        if n > WORKGROUP_SIZE as usize {
            return Err(Error::Solver(format!(
                "tire_vehicle_wheel_spin: wheel count {n} exceeds one-workgroup limit {WORKGROUP_SIZE}"
            )));
        }
        if !self.dt.is_finite() || self.dt < 0.0 {
            return Err(Error::Solver(
                "tire_vehicle_wheel_spin: dt must be finite and non-negative".into(),
            ));
        }
        Ok(n)
    }
}

impl TireVehicleWheelKinematicsGpuInput {
    fn validate(&self) -> Result<usize> {
        let n = self.wheel_mounts.len();
        if n == 0 {
            return Err(Error::Solver(
                "tire_vehicle_wheel_kinematics: at least one wheel is required".into(),
            ));
        }
        if self.wheel_state.len() != n {
            return Err(Error::Solver(
                "tire_vehicle_wheel_kinematics: mount/state length mismatch".into(),
            ));
        }
        if n > WORKGROUP_SIZE as usize {
            return Err(Error::Solver(format!(
                "tire_vehicle_wheel_kinematics: wheel count {n} exceeds one-workgroup limit {WORKGROUP_SIZE}"
            )));
        }
        Ok(n)
    }
}

impl TireVehicleWheelKinematicsFromSpinGpuInput {
    fn validate(&self, spin_count: usize) -> Result<usize> {
        let n = self.wheel_mounts.len();
        if n == 0 {
            return Err(Error::Solver(
                "tire_vehicle_wheel_kinematics_from_spin: at least one wheel is required".into(),
            ));
        }
        if self.wheel_controls.len() != n {
            return Err(Error::Solver(
                "tire_vehicle_wheel_kinematics_from_spin: mount/control length mismatch".into(),
            ));
        }
        if spin_count != n {
            return Err(Error::Solver(format!(
                "tire_vehicle_wheel_kinematics_from_spin: input wheel count {n} != resident spin wheel count {spin_count}"
            )));
        }
        if n > WORKGROUP_SIZE as usize {
            return Err(Error::Solver(format!(
                "tire_vehicle_wheel_kinematics_from_spin: wheel count {n} exceeds one-workgroup limit {WORKGROUP_SIZE}"
            )));
        }
        Ok(n)
    }

    fn packed_rows(&self) -> Vec<[f32; 4]> {
        let n = self.wheel_mounts.len();
        let mut rows = Vec::with_capacity(n * 2);
        for i in 0..n {
            rows.push(self.wheel_mounts[i]);
            rows.push(self.wheel_controls[i]);
        }
        rows
    }
}

impl TireVehicleSuspensionGpuInput {
    fn validate(&self) -> Result<usize> {
        let n = self.suspension_geometry.len();
        if n == 0 {
            return Err(Error::Solver(
                "tire_vehicle_suspension: at least one wheel is required".into(),
            ));
        }
        if self.suspension_params.len() != n {
            return Err(Error::Solver(
                "tire_vehicle_suspension: geometry/params length mismatch".into(),
            ));
        }
        if n > WORKGROUP_SIZE as usize {
            return Err(Error::Solver(format!(
                "tire_vehicle_suspension: wheel count {n} exceeds one-workgroup limit {WORKGROUP_SIZE}"
            )));
        }
        Ok(n)
    }
}

impl TireVehicleBrushRowsGpuInput {
    fn validate(&self) -> Result<usize> {
        let n = self.wheel_mounts.len();
        if n == 0 {
            return Err(Error::Solver(
                "tire_vehicle_brush_rows: at least one wheel is required".into(),
            ));
        }
        if self.wheel_state.len() != n
            || self.width_segments.len() != n
            || self.tire_rows.len() != n
            || self.material_rows.len() != n
            || self.extra_rows.len() != n
            || self.suspension_rows.len() != n
        {
            return Err(Error::Solver(
                "tire_vehicle_brush_rows: per-wheel row length mismatch".into(),
            ));
        }
        for &width in &self.width_segments {
            crate::tire_brush::validate_width_segments(width as f32)?;
        }
        if n > WORKGROUP_SIZE as usize {
            return Err(Error::Solver(format!(
                "tire_vehicle_brush_rows: wheel count {n} exceeds one-workgroup limit {WORKGROUP_SIZE}"
            )));
        }
        Ok(n)
    }

    fn packed_rows(&self) -> Vec<[f32; 4]> {
        let n = self.wheel_mounts.len();
        let mut rows = Vec::with_capacity(n * 6);
        for i in 0..n {
            rows.push(self.wheel_state[i]);
            rows.push(self.tire_rows[i]);
            rows.push(self.material_rows[i]);
            rows.push(self.extra_rows[i]);
            rows.push(self.suspension_rows[i]);
            rows.push([self.width_segments[i] as f32, 0.0, 0.0, 0.0]);
        }
        rows
    }
}

impl TireVehicleBrushRowsFromSpinGpuInput {
    fn validate(&self) -> Result<usize> {
        let n = self.spin_state_rows.len();
        for controls in &self.wheel_controls {
            crate::tire_brush::validate_width_segments(controls[3])?;
        }
        if n == 0 {
            return Err(Error::Solver(
                "tire_vehicle_brush_rows_from_spin: at least one wheel is required".into(),
            ));
        }
        if self.wheel_mounts.len() != n
            || self.wheel_controls.len() != n
            || self.tire_rows.len() != n
            || self.material_rows.len() != n
            || self.extra_rows.len() != n
            || self.suspension_rows.len() != n
        {
            return Err(Error::Solver(
                "tire_vehicle_brush_rows_from_spin: per-wheel row length mismatch".into(),
            ));
        }
        if n > WORKGROUP_SIZE as usize {
            return Err(Error::Solver(format!(
                "tire_vehicle_brush_rows_from_spin: wheel count {n} exceeds one-workgroup limit {WORKGROUP_SIZE}"
            )));
        }
        Ok(n)
    }

    fn packed_rows(&self) -> Vec<[f32; 4]> {
        let n = self.spin_state_rows.len();
        let mut rows = Vec::with_capacity(n * 6);
        for i in 0..n {
            rows.push(self.wheel_mounts[i]);
            rows.push(self.wheel_controls[i]);
            rows.push(self.tire_rows[i]);
            rows.push(self.material_rows[i]);
            rows.push(self.extra_rows[i]);
            rows.push(self.suspension_rows[i]);
        }
        rows
    }
}

pub fn tire_vehicle_yaw_reference(
    input: &TireVehicleYawGpuInput,
) -> Result<TireVehicleYawGpuOutput> {
    let _ = input.validate()?;
    let mut yaw_moment = 0.0_f32;
    for (off, wrench) in input.wheel_offsets.iter().zip(input.tire_wrenches.iter()) {
        let fx = finite_or(wrench[0], 0.0);
        let fz = finite_or(wrench[2], 0.0);
        let mz = finite_or(wrench[3], 0.0);
        yaw_moment += finite_or(off[0], 0.0) * fz - finite_or(off[2], 0.0) * fx + mz;
    }
    let inertia = finite_positive_or(input.yaw_inertia, 1.0, 1.0e-6);
    let dt = finite_or(input.dt, 0.0).max(0.0);
    let damping = finite_or(input.yaw_damping, 0.0).max(0.0);
    let yaw_rate =
        (inertia * finite_or(input.yaw_rate, 0.0) + yaw_moment * dt) / (inertia + damping * dt);
    let yaw = finite_or(input.yaw, 0.0) + yaw_rate * dt;
    Ok(TireVehicleYawGpuOutput {
        yaw,
        yaw_rate,
        yaw_moment,
        finite: yaw.is_finite() && yaw_rate.is_finite() && yaw_moment.is_finite(),
    })
}

pub fn tire_vehicle_planar_reference(
    input: &TireVehiclePlanarGpuInput,
) -> Result<TireVehiclePlanarGpuOutput> {
    let _ = input.validate()?;
    let mut force_x = 0.0_f32;
    let mut force_z = 0.0_f32;
    let mut yaw_moment = 0.0_f32;
    for (off, wrench) in input.wheel_offsets.iter().zip(input.tire_wrenches.iter()) {
        let fx = finite_or(wrench[0], 0.0);
        let fz = finite_or(wrench[2], 0.0);
        let mz = finite_or(wrench[3], 0.0);
        force_x += fx;
        force_z += fz;
        yaw_moment += finite_or(off[0], 0.0) * fz - finite_or(off[2], 0.0) * fx + mz;
    }
    let mass = finite_positive_or(input.mass, 1.0, 1.0e-6);
    let inertia = finite_positive_or(input.yaw_inertia, 1.0, 1.0e-6);
    let dt = finite_or(input.dt, 0.0).max(0.0);
    let linear_damping = finite_or(input.linear_damping, 0.0).max(0.0);
    let yaw_damping = finite_or(input.yaw_damping, 0.0).max(0.0);
    let vx =
        (mass * finite_or(input.velocity_xz[0], 0.0) + force_x * dt) / (mass + linear_damping * dt);
    let vz =
        (mass * finite_or(input.velocity_xz[1], 0.0) + force_z * dt) / (mass + linear_damping * dt);
    let px = finite_or(input.position_xz[0], 0.0) + vx * dt;
    let pz = finite_or(input.position_xz[1], 0.0) + vz * dt;
    let yaw_rate =
        (inertia * finite_or(input.yaw_rate, 0.0) + yaw_moment * dt) / (inertia + yaw_damping * dt);
    let yaw = finite_or(input.yaw, 0.0) + yaw_rate * dt;
    Ok(TireVehiclePlanarGpuOutput {
        position_xz: [px, pz],
        velocity_xz: [vx, vz],
        yaw,
        yaw_rate,
        force_xz: [force_x, force_z],
        yaw_moment,
        finite: px.is_finite()
            && pz.is_finite()
            && vx.is_finite()
            && vz.is_finite()
            && yaw.is_finite()
            && yaw_rate.is_finite()
            && force_x.is_finite()
            && force_z.is_finite()
            && yaw_moment.is_finite(),
    })
}

pub fn tire_vehicle_planar_from_brush_reference(
    input: &TireVehiclePlanarFromBrushGpuInput,
    brush_outputs: &[BrushWheelOutput],
) -> Result<TireVehiclePlanarGpuOutput> {
    let _ = input.validate(brush_outputs.len())?;
    let mut force_x = 0.0_f32;
    let mut force_z = 0.0_f32;
    let mut yaw_moment = 0.0_f32;
    for (off, brush) in input.wheel_offsets.iter().zip(brush_outputs.iter()) {
        let fx = finite_or(brush.world_force[0], 0.0);
        let fz = finite_or(brush.world_force[2], 0.0);
        let mz = finite_or(brush.diag2[1], 0.0);
        force_x += fx;
        force_z += fz;
        yaw_moment += finite_or(off[0], 0.0) * fz - finite_or(off[2], 0.0) * fx + mz;
    }
    let mass = finite_positive_or(input.mass, 1.0, 1.0e-6);
    let inertia = finite_positive_or(input.yaw_inertia, 1.0, 1.0e-6);
    let dt = finite_or(input.dt, 0.0).max(0.0);
    let linear_damping = finite_or(input.linear_damping, 0.0).max(0.0);
    let yaw_damping = finite_or(input.yaw_damping, 0.0).max(0.0);
    let vx =
        (mass * finite_or(input.velocity_xz[0], 0.0) + force_x * dt) / (mass + linear_damping * dt);
    let vz =
        (mass * finite_or(input.velocity_xz[1], 0.0) + force_z * dt) / (mass + linear_damping * dt);
    let px = finite_or(input.position_xz[0], 0.0) + vx * dt;
    let pz = finite_or(input.position_xz[1], 0.0) + vz * dt;
    let yaw_rate =
        (inertia * finite_or(input.yaw_rate, 0.0) + yaw_moment * dt) / (inertia + yaw_damping * dt);
    let yaw = finite_or(input.yaw, 0.0) + yaw_rate * dt;
    Ok(TireVehiclePlanarGpuOutput {
        position_xz: [px, pz],
        velocity_xz: [vx, vz],
        yaw,
        yaw_rate,
        force_xz: [force_x, force_z],
        yaw_moment,
        finite: px.is_finite()
            && pz.is_finite()
            && vx.is_finite()
            && vz.is_finite()
            && yaw.is_finite()
            && yaw_rate.is_finite()
            && force_x.is_finite()
            && force_z.is_finite()
            && yaw_moment.is_finite(),
    })
}

/// CPU reference for planar chassis integration from packed
/// `TireFemContactReductionOutput` GPU rows.
///
/// Each tire owns four `vec4` rows: contact diagnostics, contact range,
/// `world_force.xyz + sinkage`, and `world_moment.xyz + water_film`. The
/// planar yaw equation uses the same force-arm term as the brush path plus the
/// reduced world moment's y/up component.
pub fn tire_vehicle_planar_from_contact_reduction_rows_reference(
    input: &TireVehiclePlanarFromBrushGpuInput,
    contact_reduction_rows: &[[f32; 4]],
) -> Result<TireVehiclePlanarGpuOutput> {
    let n = input.validate(contact_reduction_rows.len() / 4)?;
    if contact_reduction_rows.len() != n * 4 {
        return Err(Error::Solver(format!(
            "tire_vehicle_planar_from_contact_reduction: packed row length {} must be 4 * wheel count {}",
            contact_reduction_rows.len(),
            n
        )));
    }
    let mut force_x = 0.0_f32;
    let mut force_z = 0.0_f32;
    let mut yaw_moment = 0.0_f32;
    for (i, off) in input.wheel_offsets.iter().enumerate() {
        let force = contact_reduction_rows[i * 4 + 2];
        let moment = contact_reduction_rows[i * 4 + 3];
        let fx = finite_or(force[0], 0.0);
        let fz = finite_or(force[2], 0.0);
        let mz = finite_or(moment[1], 0.0);
        force_x += fx;
        force_z += fz;
        yaw_moment += finite_or(off[0], 0.0) * fz - finite_or(off[2], 0.0) * fx + mz;
    }
    let mass = finite_positive_or(input.mass, 1.0, 1.0e-6);
    let inertia = finite_positive_or(input.yaw_inertia, 1.0, 1.0e-6);
    let dt = finite_or(input.dt, 0.0).max(0.0);
    let linear_damping = finite_or(input.linear_damping, 0.0).max(0.0);
    let yaw_damping = finite_or(input.yaw_damping, 0.0).max(0.0);
    let vx =
        (mass * finite_or(input.velocity_xz[0], 0.0) + force_x * dt) / (mass + linear_damping * dt);
    let vz =
        (mass * finite_or(input.velocity_xz[1], 0.0) + force_z * dt) / (mass + linear_damping * dt);
    let px = finite_or(input.position_xz[0], 0.0) + vx * dt;
    let pz = finite_or(input.position_xz[1], 0.0) + vz * dt;
    let yaw_rate =
        (inertia * finite_or(input.yaw_rate, 0.0) + yaw_moment * dt) / (inertia + yaw_damping * dt);
    let yaw = finite_or(input.yaw, 0.0) + yaw_rate * dt;
    Ok(TireVehiclePlanarGpuOutput {
        position_xz: [px, pz],
        velocity_xz: [vx, vz],
        yaw,
        yaw_rate,
        force_xz: [force_x, force_z],
        yaw_moment,
        finite: px.is_finite()
            && pz.is_finite()
            && vx.is_finite()
            && vz.is_finite()
            && yaw.is_finite()
            && yaw_rate.is_finite()
            && force_x.is_finite()
            && force_z.is_finite()
            && yaw_moment.is_finite(),
    })
}

pub fn tire_vehicle_planar_from_feedback_output_reference(
    input: &TireVehiclePlanarFromBrushGpuInput,
    feedback: &TireFemContactFeedbackOutput,
    atlas_layout: &TireFemGpuBatchAtlasLayout,
) -> Result<TireVehiclePlanarGpuOutput> {
    let n = input.validate(atlas_layout.rows.len())?;
    if feedback.forces.len() != atlas_layout.plan.total_node_count {
        return Err(Error::Solver(format!(
            "tire_vehicle_planar_from_feedback: feedback force rows {} must match atlas node count {}",
            feedback.forces.len(),
            atlas_layout.plan.total_node_count
        )));
    }
    let mut brush_like = Vec::with_capacity(n);
    for row in &atlas_layout.rows {
        let end = row.node_offset.checked_add(row.node_count).ok_or_else(|| {
            Error::Solver("tire_vehicle_planar_from_feedback: atlas node range overflow".into())
        })?;
        if end > feedback.forces.len() {
            return Err(Error::Solver(format!(
                "tire_vehicle_planar_from_feedback: atlas row node range {}..{} exceeds feedback force rows {}",
                row.node_offset,
                end,
                feedback.forces.len()
            )));
        }
        let mut force = [0.0_f32; 3];
        for f in &feedback.forces[row.node_offset..end] {
            for axis in 0..3 {
                force[axis] += finite_or(f[axis], 0.0);
            }
        }
        brush_like.push(BrushWheelOutput {
            world_force: [force[0], force[1], force[2], 0.0],
            ..Default::default()
        });
    }
    tire_vehicle_planar_from_brush_reference(input, &brush_like)
}

pub fn tire_vehicle_wheel_spin_reference(
    input: &TireVehicleWheelSpinGpuInput,
) -> Result<TireVehicleWheelSpinGpuOutput> {
    let _ = input.validate()?;
    let dt = finite_or(input.dt, 0.0).max(0.0);
    let mut rows = Vec::with_capacity(input.wheel_state.len());
    let mut all_finite = true;
    for (state, command) in input.wheel_state.iter().zip(input.wheel_commands.iter()) {
        let spin = finite_or(state[0], 0.0);
        let radius = finite_positive_or(state[1], 0.0, 1.0e-6);
        let inertia = finite_positive_or(state[2], 1.0, 1.0e-6);
        let drive = finite_or(command[0], 0.0);
        let brake_torque = finite_or(command[1], 0.0).max(0.0);
        let tire_fx = finite_or(command[2], 0.0);
        let brake = brake_torque * spin.tanh();
        let net_torque = drive - brake - tire_fx * radius;
        let accel = net_torque / inertia;
        let next_spin = spin + accel * dt;
        let finite =
            next_spin.is_finite() && accel.is_finite() && net_torque.is_finite() && radius > 0.0;
        all_finite &= finite;
        rows.push([next_spin, accel, net_torque, if finite { 1.0 } else { 0.0 }]);
    }
    Ok(TireVehicleWheelSpinGpuOutput {
        rows,
        finite: all_finite,
    })
}

pub fn tire_vehicle_wheel_spin_state_reference(
    input: &TireVehicleWheelSpinGpuInput,
) -> Result<TireVehicleWheelSpinStateGpuOutput> {
    let diagnostic = tire_vehicle_wheel_spin_reference(input)?;
    let mut rows = Vec::with_capacity(input.wheel_state.len());
    let mut all_finite = diagnostic.finite;
    for (state, row) in input.wheel_state.iter().zip(diagnostic.rows.iter()) {
        let radius = finite_positive_or(state[1], 0.0, 1.0e-6);
        let inertia = finite_positive_or(state[2], 1.0, 1.0e-6);
        let finite = row[3] > 0.5 && radius > 0.0 && inertia > 0.0;
        all_finite &= finite;
        rows.push([row[0], radius, inertia, if finite { 1.0 } else { 0.0 }]);
    }
    Ok(TireVehicleWheelSpinStateGpuOutput {
        rows,
        finite: all_finite,
    })
}

pub fn tire_vehicle_wheel_kinematics_reference(
    input: &TireVehicleWheelKinematicsGpuInput,
) -> Result<TireVehicleWheelKinematicsGpuOutput> {
    let n = input.validate()?;
    let yaw = finite_or(input.yaw, 0.0);
    let yaw_rate = finite_or(input.yaw_rate, 0.0);
    let vx = finite_or(input.velocity_xz[0], 0.0);
    let vz = finite_or(input.velocity_xz[1], 0.0);
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let mut rows = vec![[0.0_f32; 4]; n * 2];
    let mut all_finite = true;
    for (i, (mount, state)) in input
        .wheel_mounts
        .iter()
        .zip(input.wheel_state.iter())
        .enumerate()
    {
        let attach_x = finite_or(mount[0], 0.0);
        let attach_z = finite_or(mount[2], 0.0);
        let radius = finite_positive_or(mount[3], 0.0, 1.0e-6);
        let spin = finite_or(state[0], 0.0);
        let steer = finite_or(state[1], 0.0);
        let off_x = attach_x * cy - attach_z * sy;
        let off_z = attach_x * sy + attach_z * cy;
        let corner_vx = vx - yaw_rate * off_z;
        let corner_vz = vz + yaw_rate * off_x;
        let heading = yaw + steer;
        let (sh, ch) = heading.sin_cos();
        let long_x = ch;
        let long_z = sh;
        let lat_x = -sh;
        let lat_z = ch;
        let v_long = corner_vx * long_x + corner_vz * long_z;
        let v_lat = corner_vx * lat_x + corner_vz * lat_z;
        let denom = v_long.abs().max(0.5);
        let slip_ratio = (spin * radius - v_long) / denom;
        let slip_angle = (v_lat / denom).atan();
        let finite = v_long.is_finite()
            && v_lat.is_finite()
            && slip_ratio.is_finite()
            && slip_angle.is_finite()
            && long_x.is_finite()
            && long_z.is_finite()
            && lat_x.is_finite()
            && lat_z.is_finite();
        all_finite &= finite;
        rows[i] = [v_long, v_lat, slip_ratio, slip_angle];
        rows[n + i] = [long_x, long_z, lat_x, lat_z];
    }
    Ok(TireVehicleWheelKinematicsGpuOutput {
        rows,
        finite: all_finite,
    })
}

pub fn tire_vehicle_wheel_kinematics_from_spin_reference(
    input: &TireVehicleWheelKinematicsFromSpinGpuInput,
    spin_state_rows: &[[f32; 4]],
) -> Result<TireVehicleWheelKinematicsGpuOutput> {
    let n = input.validate(spin_state_rows.len())?;
    let yaw = finite_or(input.yaw, 0.0);
    let yaw_rate = finite_or(input.yaw_rate, 0.0);
    let vx = finite_or(input.velocity_xz[0], 0.0);
    let vz = finite_or(input.velocity_xz[1], 0.0);
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let mut rows = vec![[0.0_f32; 4]; n * 2];
    let mut all_finite = true;
    for i in 0..n {
        let mount = input.wheel_mounts[i];
        let controls = input.wheel_controls[i];
        let spin_state = spin_state_rows[i];
        let attach_x = finite_or(mount[0], 0.0);
        let attach_z = finite_or(mount[2], 0.0);
        let radius = finite_positive_or(spin_state[1], mount[3].max(1.0e-6), 1.0e-6);
        let spin = finite_or(spin_state[0], 0.0);
        let steer = finite_or(controls[0], 0.0);
        let off_x = attach_x * cy - attach_z * sy;
        let off_z = attach_x * sy + attach_z * cy;
        let corner_vx = vx - yaw_rate * off_z;
        let corner_vz = vz + yaw_rate * off_x;
        let heading = yaw + steer;
        let (sh, ch) = heading.sin_cos();
        let long_x = ch;
        let long_z = sh;
        let lat_x = -sh;
        let lat_z = ch;
        let v_long = corner_vx * long_x + corner_vz * long_z;
        let v_lat = corner_vx * lat_x + corner_vz * lat_z;
        let denom = v_long.abs().max(0.5);
        let slip_ratio = (spin * radius - v_long) / denom;
        let slip_angle = (v_lat / denom).atan();
        let finite = v_long.is_finite()
            && v_lat.is_finite()
            && slip_ratio.is_finite()
            && slip_angle.is_finite()
            && long_x.is_finite()
            && long_z.is_finite()
            && lat_x.is_finite()
            && lat_z.is_finite();
        all_finite &= finite;
        rows[i] = [v_long, v_lat, slip_ratio, slip_angle];
        rows[n + i] = [long_x, long_z, lat_x, lat_z];
    }
    Ok(TireVehicleWheelKinematicsGpuOutput {
        rows,
        finite: all_finite,
    })
}

pub fn tire_vehicle_suspension_reference(
    input: &TireVehicleSuspensionGpuInput,
) -> Result<TireVehicleSuspensionGpuOutput> {
    let _ = input.validate()?;
    let chassis_y = finite_or(input.chassis_y, 0.0);
    let chassis_vy = finite_or(input.chassis_vy, 0.0);
    let mut rows = Vec::with_capacity(input.suspension_geometry.len());
    let mut all_finite = true;
    for (geo, params) in input
        .suspension_geometry
        .iter()
        .zip(input.suspension_params.iter())
    {
        let attach_y = finite_or(geo[0], 0.0);
        let radius = finite_positive_or(geo[1], 0.0, 1.0e-6);
        let rest_length = finite_positive_or(geo[2], 0.0, 0.0);
        let max_compression = finite_positive_or(geo[3], rest_length * 0.9, 0.0);
        let stiffness = finite_or(params[0], 0.0).max(0.0);
        let damping = finite_or(params[1], 0.0).max(0.0);
        let ground_height = finite_or(params[2], 0.0);
        let attach_world_y = chassis_y + attach_y;
        let hub_y = ground_height + radius;
        let compression = rest_length - (attach_world_y - hub_y);
        let travel = compression.clamp(0.0, max_compression);
        let travel_rate = -chassis_vy;
        let normal_load = (stiffness * travel + damping * travel_rate).max(0.0);
        let finite = normal_load.is_finite() && travel.is_finite() && travel_rate.is_finite();
        all_finite &= finite;
        rows.push([
            normal_load,
            travel,
            travel_rate,
            if finite { 1.0 } else { 0.0 },
        ]);
    }
    Ok(TireVehicleSuspensionGpuOutput {
        rows,
        finite: all_finite,
    })
}

pub fn tire_vehicle_vertical_from_suspension_reference(
    input: &TireVehicleVerticalGpuInput,
    suspension_rows: &[[f32; 4]],
) -> Result<TireVehicleVerticalGpuOutput> {
    let _ = input.validate(suspension_rows.len())?;
    let total_suspension: f32 = suspension_rows
        .iter()
        .map(|row| finite_or(row[0], 0.0).max(0.0))
        .sum();
    let mass = finite_positive_or(input.mass, 1.0, 1.0e-6);
    let gravity = finite_or(input.gravity, 9.81).max(0.0);
    let damping = finite_or(input.vertical_damping, 0.0).max(0.0);
    let dt = finite_or(input.dt, 0.0).max(0.0);
    let force_y = total_suspension - mass * gravity;
    let vy = (mass * finite_or(input.vy, 0.0) + force_y * dt) / (mass + damping * dt).max(1.0e-6);
    let y = finite_or(input.y, 0.0) + vy * dt;
    let finite = y.is_finite() && vy.is_finite() && force_y.is_finite();
    Ok(TireVehicleVerticalGpuOutput {
        y,
        vy,
        force_y,
        finite,
    })
}

pub fn tire_vehicle_brush_rows_reference(
    input: &TireVehicleBrushRowsGpuInput,
) -> Result<TireVehicleBrushRowsGpuOutput> {
    let _ = input.validate()?;
    let chassis_x = finite_or(input.chassis_pose[0], 0.0);
    let chassis_z = finite_or(input.chassis_pose[2], 0.0);
    let yaw = finite_or(input.chassis_pose[3], 0.0);
    let vx = finite_or(input.chassis_velocity[0], 0.0);
    let vy = finite_or(input.chassis_velocity[1], 0.0);
    let vz = finite_or(input.chassis_velocity[2], 0.0);
    let yaw_rate = finite_or(input.chassis_velocity[3], 0.0);
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let mut wheels = Vec::with_capacity(input.wheel_mounts.len());
    let mut all_finite = true;
    for i in 0..input.wheel_mounts.len() {
        let mount = input.wheel_mounts[i];
        let state = input.wheel_state[i];
        let tire = input.tire_rows[i];
        let material = input.material_rows[i];
        let extra = input.extra_rows[i];
        let suspension = input.suspension_rows[i];
        let attach_x = finite_or(mount[0], 0.0);
        let attach_z = finite_or(mount[2], 0.0);
        let radius = finite_positive_or(mount[3], 0.0, 1.0e-6);
        let off_x = attach_x * cy - attach_z * sy;
        let off_z = attach_x * sy + attach_z * cy;
        let heading = yaw + finite_or(state[1], 0.0);
        let (sh, ch) = heading.sin_cos();
        let travel = finite_or(suspension[0], 0.0).max(0.0);
        let ground_height = finite_or(suspension[2], 0.0);
        let row = BrushWheelInput {
            hub_pos: [
                chassis_x + off_x,
                ground_height + radius - travel,
                chassis_z + off_z,
                input.width_segments[i] as f32,
            ],
            hub_vel: [vx - yaw_rate * off_z, vy, vz + yaw_rate * off_x, 0.0],
            heading: [ch, 0.0, sh, 0.0],
            spin: [
                finite_or(state[0], 0.0),
                radius,
                finite_or(state[2], 0.0),
                finite_or(state[3], 0.0),
            ],
            tire,
            material,
            extra,
        };
        all_finite &= row
            .hub_pos
            .iter()
            .chain(row.hub_vel.iter())
            .chain(row.heading.iter())
            .chain(row.spin.iter())
            .chain(row.tire.iter())
            .chain(row.material.iter())
            .chain(row.extra.iter())
            .all(|v| v.is_finite());
        wheels.push(row);
    }
    Ok(TireVehicleBrushRowsGpuOutput {
        wheels,
        finite: all_finite,
    })
}

pub fn tire_vehicle_brush_rows_from_spin_reference(
    input: &TireVehicleBrushRowsFromSpinGpuInput,
) -> Result<TireVehicleBrushRowsGpuOutput> {
    let _ = input.validate()?;
    let chassis_x = finite_or(input.chassis_pose[0], 0.0);
    let chassis_z = finite_or(input.chassis_pose[2], 0.0);
    let yaw = finite_or(input.chassis_pose[3], 0.0);
    let vx = finite_or(input.chassis_velocity[0], 0.0);
    let vy = finite_or(input.chassis_velocity[1], 0.0);
    let vz = finite_or(input.chassis_velocity[2], 0.0);
    let yaw_rate = finite_or(input.chassis_velocity[3], 0.0);
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let mut wheels = Vec::with_capacity(input.spin_state_rows.len());
    let mut all_finite = true;
    for i in 0..input.spin_state_rows.len() {
        let spin_state = input.spin_state_rows[i];
        let mount = input.wheel_mounts[i];
        let controls = input.wheel_controls[i];
        let tire = input.tire_rows[i];
        let material = input.material_rows[i];
        let extra = input.extra_rows[i];
        let suspension = input.suspension_rows[i];
        let attach_x = finite_or(mount[0], 0.0);
        let attach_z = finite_or(mount[2], 0.0);
        let radius = finite_positive_or(spin_state[1], mount[3].max(1.0e-6), 1.0e-6);
        let off_x = attach_x * cy - attach_z * sy;
        let off_z = attach_x * sy + attach_z * cy;
        let heading = yaw + finite_or(controls[0], 0.0);
        let (sh, ch) = heading.sin_cos();
        let travel = finite_or(suspension[0], 0.0).max(0.0);
        let ground_height = finite_or(suspension[2], 0.0);
        let row = BrushWheelInput {
            hub_pos: [
                chassis_x + off_x,
                ground_height + radius - travel,
                chassis_z + off_z,
                controls[3],
            ],
            hub_vel: [vx - yaw_rate * off_z, vy, vz + yaw_rate * off_x, 0.0],
            heading: [ch, 0.0, sh, 0.0],
            spin: [
                finite_or(spin_state[0], 0.0),
                radius,
                finite_or(controls[1], 0.0),
                finite_or(controls[2], 0.0),
            ],
            tire,
            material,
            extra,
        };
        all_finite &= row
            .hub_pos
            .iter()
            .chain(row.hub_vel.iter())
            .chain(row.heading.iter())
            .chain(row.spin.iter())
            .chain(row.tire.iter())
            .chain(row.material.iter())
            .chain(row.extra.iter())
            .all(|v| v.is_finite());
        wheels.push(row);
    }
    Ok(TireVehicleBrushRowsGpuOutput {
        wheels,
        finite: all_finite,
    })
}

pub fn tire_vehicle_brush_rows_from_spin_suspension_reference(
    input: &TireVehicleBrushRowsFromSpinGpuInput,
    suspension_rows: &[[f32; 4]],
) -> Result<TireVehicleBrushRowsGpuOutput> {
    let n = input.validate()?;
    if suspension_rows.len() != n {
        return Err(Error::Solver(format!(
            "tire_vehicle_brush_rows_from_spin_suspension: suspension row count {} != wheel count {n}",
            suspension_rows.len()
        )));
    }
    let mut merged = input.clone();
    for (dst, src) in merged
        .suspension_rows
        .iter_mut()
        .zip(suspension_rows.iter())
    {
        // Resident suspension rows are x=normal load, y=travel, z=travel rate,
        // w=finite. Brush row construction still needs ground height from the
        // contact row, so preserve the caller's z=ground_height_m.
        *dst = [src[1], src[2], dst[2], src[3]];
    }
    tire_vehicle_brush_rows_from_spin_reference(&merged)
}

impl TireVehicleYawGpuKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_vehicle_yaw"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/tire_vehicle_yaw.wgsl").into(),
            ),
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_vehicle_yaw io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let layout_io4 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_vehicle_yaw io4"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, true),
            ],
        });
        let layout_io5 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_vehicle_yaw io5"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, true),
                storage_entry(4, true),
            ],
        });
        let layout_io6 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_vehicle_yaw io6"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_vehicle_yaw uniforms"),
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::tire_vehicle_yaw layout"),
            bind_group_layouts: &[Some(&layout_io), Some(&layout_uniforms)],
            immediate_size: 0,
        });
        let pipeline_layout_io4 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::tire_vehicle_yaw io4 layout"),
            bind_group_layouts: &[Some(&layout_io4), Some(&layout_uniforms)],
            immediate_size: 0,
        });
        let pipeline_layout_io5 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::tire_vehicle_yaw io5 layout"),
            bind_group_layouts: &[Some(&layout_io5), Some(&layout_uniforms)],
            immediate_size: 0,
        });
        let pipeline_layout_io6 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::tire_vehicle_yaw io6 layout"),
            bind_group_layouts: &[Some(&layout_io6), Some(&layout_uniforms)],
            immediate_size: 0,
        });
        let pipeline_yaw = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_vehicle_yaw pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("integrate_yaw"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_planar = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_vehicle_planar pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("integrate_planar"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_planar_from_brush =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_planar_from_brush pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("integrate_planar_from_brush_outputs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_planar_state_from_brush =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_planar_state_from_brush pipeline"),
                layout: Some(&pipeline_layout_io4),
                module: &shader,
                entry_point: Some("integrate_planar_state_from_brush_outputs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_planar_from_contact_reduction =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_planar_from_contact_reduction pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("integrate_planar_from_contact_reduction_outputs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_planar_state_from_contact_reduction =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_planar_state_from_contact_reduction pipeline"),
                layout: Some(&pipeline_layout_io4),
                module: &shader,
                entry_point: Some("integrate_planar_state_from_contact_reduction_outputs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_planar_state_from_feedback =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_planar_state_from_feedback pipeline"),
                layout: Some(&pipeline_layout_io6),
                module: &shader,
                entry_point: Some("integrate_planar_state_from_feedback_force_rows"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_vertical_state_from_suspension =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_vertical_state_from_suspension pipeline"),
                layout: Some(&pipeline_layout_io4),
                module: &shader,
                entry_point: Some("integrate_vertical_state_from_suspension"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_wheel_spin =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_wheel_spin pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("integrate_wheel_spin"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_wheel_spin_state =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_wheel_spin_state pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("integrate_wheel_spin_state"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_wheel_kinematics =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_wheel_kinematics pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("compute_wheel_kinematics"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_wheel_kinematics_from_spin =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_wheel_kinematics_from_spin pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("compute_wheel_kinematics_from_spin_state"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_suspension =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_suspension pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("compute_suspension"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_suspension_from_vertical =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_suspension_from_vertical pipeline"),
                layout: Some(&pipeline_layout_io4),
                module: &shader,
                entry_point: Some("compute_suspension_from_vertical_state"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_brush_rows =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_brush_rows pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("build_brush_rows"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_brush_rows_from_spin =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_brush_rows_from_spin pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("build_brush_rows_from_spin_state"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_brush_rows_from_spin_suspension =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension pipeline"),
                layout: Some(&pipeline_layout_io4),
                module: &shader,
                entry_point: Some("build_brush_rows_from_spin_and_suspension_state"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_brush_rows_from_spin_suspension_planar =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension_planar pipeline"),
                layout: Some(&pipeline_layout_io5),
                module: &shader,
                entry_point: Some("build_brush_rows_from_spin_suspension_planar_state"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_brush_rows_from_spin_suspension_chassis =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension_chassis pipeline"),
                layout: Some(&pipeline_layout_io6),
                module: &shader,
                entry_point: Some("build_brush_rows_from_spin_suspension_chassis_state"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self {
            device: device.clone(),
            pipeline_yaw,
            pipeline_planar,
            pipeline_planar_from_brush,
            pipeline_planar_state_from_brush,
            pipeline_planar_from_contact_reduction,
            pipeline_planar_state_from_contact_reduction,
            pipeline_planar_state_from_feedback,
            pipeline_vertical_state_from_suspension,
            pipeline_wheel_spin,
            pipeline_wheel_spin_state,
            pipeline_wheel_kinematics,
            pipeline_wheel_kinematics_from_spin,
            pipeline_suspension,
            pipeline_suspension_from_vertical,
            pipeline_brush_rows,
            pipeline_brush_rows_from_spin,
            pipeline_brush_rows_from_spin_suspension,
            pipeline_brush_rows_from_spin_suspension_planar,
            pipeline_brush_rows_from_spin_suspension_chassis,
            layout_io,
            layout_io4,
            layout_io5,
            layout_io6,
            layout_uniforms,
        }
    }

    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehicleYawGpuInput,
    ) -> Result<TireVehicleYawGpuOutput> {
        let n = input.validate()?;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let offsets = mk_buf("tire_vehicle_yaw offsets", vec4_bytes(n), usage_ro);
        let wrenches = mk_buf("tire_vehicle_yaw wrenches", vec4_bytes(n), usage_ro);
        let out = mk_buf("tire_vehicle_yaw out", vec4_bytes(1), usage_rw);
        let stage = mk_buf(
            "tire_vehicle_yaw stage",
            vec4_bytes(1),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let uniforms = mk_buf(
            "tire_vehicle_yaw uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&offsets, 0, bytemuck::cast_slice(&input.wheel_offsets));
        queue.write_buffer(&wrenches, 0, bytemuck::cast_slice(&input.tire_wrenches));
        queue.write_buffer(&out, 0, bytemuck::cast_slice(&[[0.0_f32; 4]]));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [0.0, 0.0, finite_or(input.yaw, 0.0), 0.0],
                velocity: [0.0, 0.0, finite_or(input.yaw_rate, 0.0), 0.0],
                params0: [
                    1.0,
                    finite_positive_or(input.yaw_inertia, 1.0, 1.0e-6),
                    finite_or(input.dt, 0.0).max(0.0),
                    0.0,
                ],
                params1: [finite_or(input.yaw_damping, 0.0).max(0.0), 0.0, 0.0, 0.0],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_yaw io bg"),
            layout: &self.layout_io,
            entries: &[entry(0, &offsets), entry(1, &wrenches), entry(2, &out)],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_yaw uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_yaw encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_yaw pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_yaw);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &stage, 0, vec4_bytes(1));
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(device, &stage, "tire_vehicle_yaw output")?;
        let row = rows.first().copied().unwrap_or([0.0; 4]);
        Ok(TireVehicleYawGpuOutput {
            yaw: row[0],
            yaw_rate: row[1],
            yaw_moment: row[2],
            finite: row[3] > 0.5,
        })
    }

    pub fn dispatch_planar_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehiclePlanarGpuInput,
    ) -> Result<TireVehiclePlanarGpuOutput> {
        let n = input.validate()?;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let offsets = mk_buf("tire_vehicle_planar offsets", vec4_bytes(n), usage_ro);
        let wrenches = mk_buf("tire_vehicle_planar wrenches", vec4_bytes(n), usage_ro);
        let out = mk_buf("tire_vehicle_planar out", vec4_bytes(3), usage_rw);
        let stage = mk_buf(
            "tire_vehicle_planar stage",
            vec4_bytes(3),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let uniforms = mk_buf(
            "tire_vehicle_planar uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&offsets, 0, bytemuck::cast_slice(&input.wheel_offsets));
        queue.write_buffer(&wrenches, 0, bytemuck::cast_slice(&input.tire_wrenches));
        queue.write_buffer(&out, 0, bytemuck::cast_slice(&[[0.0_f32; 4]; 3]));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [
                    finite_or(input.position_xz[0], 0.0),
                    finite_or(input.position_xz[1], 0.0),
                    finite_or(input.yaw, 0.0),
                    0.0,
                ],
                velocity: [
                    finite_or(input.velocity_xz[0], 0.0),
                    finite_or(input.velocity_xz[1], 0.0),
                    finite_or(input.yaw_rate, 0.0),
                    0.0,
                ],
                params0: [
                    finite_positive_or(input.mass, 1.0, 1.0e-6),
                    finite_positive_or(input.yaw_inertia, 1.0, 1.0e-6),
                    finite_or(input.dt, 0.0).max(0.0),
                    finite_or(input.linear_damping, 0.0).max(0.0),
                ],
                params1: [finite_or(input.yaw_damping, 0.0).max(0.0), 0.0, 0.0, 0.0],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_planar io bg"),
            layout: &self.layout_io,
            entries: &[entry(0, &offsets), entry(1, &wrenches), entry(2, &out)],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_planar uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_planar encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_planar pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_planar);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &stage, 0, vec4_bytes(3));
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(device, &stage, "tire_vehicle_planar output")?;
        let state = rows.first().copied().unwrap_or([0.0; 4]);
        let yaw = rows.get(1).copied().unwrap_or([0.0; 4]);
        let force = rows.get(2).copied().unwrap_or([0.0; 4]);
        Ok(TireVehiclePlanarGpuOutput {
            position_xz: [state[0], state[1]],
            velocity_xz: [state[2], state[3]],
            yaw: yaw[0],
            yaw_rate: yaw[1],
            yaw_moment: yaw[2],
            finite: yaw[3] > 0.5,
            force_xz: [force[0], force[1]],
        })
    }

    pub fn dispatch_planar_from_brush_output_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehiclePlanarFromBrushGpuInput,
        brush_output_buffer: &wgpu::Buffer,
        brush_output_count: usize,
    ) -> Result<TireVehiclePlanarGpuOutput> {
        let n = input.validate(brush_output_count)?;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let offsets = mk_buf(
            "tire_vehicle_planar_from_brush offsets",
            vec4_bytes(n),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let out = mk_buf(
            "tire_vehicle_planar_from_brush out",
            vec4_bytes(3),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let stage = mk_buf(
            "tire_vehicle_planar_from_brush stage",
            vec4_bytes(3),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let uniforms = mk_buf(
            "tire_vehicle_planar_from_brush uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&offsets, 0, bytemuck::cast_slice(&input.wheel_offsets));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [
                    finite_or(input.position_xz[0], 0.0),
                    finite_or(input.position_xz[1], 0.0),
                    finite_or(input.yaw, 0.0),
                    0.0,
                ],
                velocity: [
                    finite_or(input.velocity_xz[0], 0.0),
                    finite_or(input.velocity_xz[1], 0.0),
                    finite_or(input.yaw_rate, 0.0),
                    0.0,
                ],
                params0: [
                    finite_positive_or(input.mass, 1.0, 1.0e-6),
                    finite_positive_or(input.yaw_inertia, 1.0, 1.0e-6),
                    finite_or(input.dt, 0.0).max(0.0),
                    finite_or(input.linear_damping, 0.0).max(0.0),
                ],
                params1: [finite_or(input.yaw_damping, 0.0).max(0.0), 0.0, 0.0, 0.0],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_planar_from_brush io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &offsets),
                entry(1, brush_output_buffer),
                entry(2, &out),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_planar_from_brush uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_planar_from_brush encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_planar_from_brush pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_planar_from_brush);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &stage, 0, vec4_bytes(3));
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(device, &stage, "tire_vehicle_planar_from_brush output")?;
        let state = rows.first().copied().unwrap_or([0.0; 4]);
        let yaw = rows.get(1).copied().unwrap_or([0.0; 4]);
        let force = rows.get(2).copied().unwrap_or([0.0; 4]);
        Ok(TireVehiclePlanarGpuOutput {
            position_xz: [state[0], state[1]],
            velocity_xz: [state[2], state[3]],
            yaw: yaw[0],
            yaw_rate: yaw[1],
            yaw_moment: yaw[2],
            finite: yaw[3] > 0.5,
            force_xz: [force[0], force[1]],
        })
    }

    pub fn dispatch_planar_from_contact_reduction_output_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehiclePlanarFromBrushGpuInput,
        contact_reduction_output_buffer: &wgpu::Buffer,
        contact_reduction_output_count: usize,
    ) -> Result<TireVehiclePlanarGpuOutput> {
        let n = input.validate(contact_reduction_output_count)?;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let offsets = mk_buf(
            "tire_vehicle_planar_from_contact_reduction offsets",
            vec4_bytes(n),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let out = mk_buf(
            "tire_vehicle_planar_from_contact_reduction out",
            vec4_bytes(3),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let stage = mk_buf(
            "tire_vehicle_planar_from_contact_reduction stage",
            vec4_bytes(3),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let uniforms = mk_buf(
            "tire_vehicle_planar_from_contact_reduction uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&offsets, 0, bytemuck::cast_slice(&input.wheel_offsets));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [
                    finite_or(input.position_xz[0], 0.0),
                    finite_or(input.position_xz[1], 0.0),
                    finite_or(input.yaw, 0.0),
                    0.0,
                ],
                velocity: [
                    finite_or(input.velocity_xz[0], 0.0),
                    finite_or(input.velocity_xz[1], 0.0),
                    finite_or(input.yaw_rate, 0.0),
                    0.0,
                ],
                params0: [
                    finite_positive_or(input.mass, 1.0, 1.0e-6),
                    finite_positive_or(input.yaw_inertia, 1.0, 1.0e-6),
                    finite_or(input.dt, 0.0).max(0.0),
                    finite_or(input.linear_damping, 0.0).max(0.0),
                ],
                params1: [finite_or(input.yaw_damping, 0.0).max(0.0), 0.0, 0.0, 0.0],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_planar_from_contact_reduction io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &offsets),
                entry(1, contact_reduction_output_buffer),
                entry(2, &out),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_planar_from_contact_reduction uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_planar_from_contact_reduction encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_planar_from_contact_reduction pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_planar_from_contact_reduction);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &stage, 0, vec4_bytes(3));
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(
            device,
            &stage,
            "tire_vehicle_planar_from_contact_reduction output",
        )?;
        let state = rows.first().copied().unwrap_or([0.0; 4]);
        let yaw = rows.get(1).copied().unwrap_or([0.0; 4]);
        let force = rows.get(2).copied().unwrap_or([0.0; 4]);
        Ok(TireVehiclePlanarGpuOutput {
            position_xz: [state[0], state[1]],
            velocity_xz: [state[2], state[3]],
            yaw: yaw[0],
            yaw_rate: yaw[1],
            yaw_moment: yaw[2],
            finite: yaw[3] > 0.5,
            force_xz: [force[0], force[1]],
        })
    }

    pub fn create_planar_resident_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehiclePlanarFromBrushGpuInput,
        brush_output_count: usize,
    ) -> Result<TireVehiclePlanarResidentState> {
        let n = input.validate(brush_output_count)?;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let state_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let state_a = mk_buf(
            "tire_vehicle_planar resident state a",
            vec4_bytes(3),
            state_usage,
        );
        let state_b = mk_buf(
            "tire_vehicle_planar resident state b",
            vec4_bytes(3),
            state_usage,
        );
        let offsets = mk_buf(
            "tire_vehicle_planar resident offsets",
            vec4_bytes(n),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let uniforms = mk_buf(
            "tire_vehicle_planar resident uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let state_rows = [
            [
                finite_or(input.position_xz[0], 0.0),
                finite_or(input.position_xz[1], 0.0),
                finite_or(input.velocity_xz[0], 0.0),
                finite_or(input.velocity_xz[1], 0.0),
            ],
            [
                finite_or(input.yaw, 0.0),
                finite_or(input.yaw_rate, 0.0),
                0.0,
                1.0,
            ],
            [0.0, 0.0, 0.0, 0.0],
        ];
        queue.write_buffer(&state_a, 0, bytemuck::cast_slice(&state_rows));
        queue.write_buffer(&state_b, 0, bytemuck::cast_slice(&state_rows));
        queue.write_buffer(&offsets, 0, bytemuck::cast_slice(&input.wheel_offsets));
        Ok(TireVehiclePlanarResidentState {
            state_buffers: [state_a, state_b],
            wheel_offsets: offsets,
            uniforms,
            wheel_count: n,
            current_state: 0,
            mass: input.mass,
            yaw_inertia: input.yaw_inertia,
            linear_damping: input.linear_damping,
            yaw_damping: input.yaw_damping,
        })
    }

    pub fn step_planar_resident_from_brush_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &mut TireVehiclePlanarResidentState,
        brush_output_buffer: &wgpu::Buffer,
        brush_output_count: usize,
        dt: f32,
    ) -> Result<()> {
        if brush_output_count != state.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_planar resident: brush output count {brush_output_count} != wheel count {}",
                state.wheel_count
            )));
        }
        let next_state = 1 - state.current_state;
        queue.write_buffer(
            &state.uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [state.wheel_count as u32, 0, 0, 0],
                pose: [0.0; 4],
                velocity: [0.0; 4],
                params0: [
                    finite_positive_or(state.mass, 1.0, 1.0e-6),
                    finite_positive_or(state.yaw_inertia, 1.0, 1.0e-6),
                    finite_or(dt, 0.0).max(0.0),
                    finite_or(state.linear_damping, 0.0).max(0.0),
                ],
                params1: [finite_or(state.yaw_damping, 0.0).max(0.0), 0.0, 0.0, 0.0],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_planar resident io bg"),
            layout: &self.layout_io4,
            entries: &[
                entry(0, &state.wheel_offsets),
                entry(1, brush_output_buffer),
                entry(2, &state.state_buffers[next_state]),
                entry(3, &state.state_buffers[state.current_state]),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_planar resident uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &state.uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_planar resident encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_planar resident pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_planar_state_from_brush);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        state.current_state = next_state;
        Ok(())
    }

    pub fn step_planar_resident_from_contact_reduction_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &mut TireVehiclePlanarResidentState,
        contact_reduction_output_buffer: &wgpu::Buffer,
        contact_reduction_output_count: usize,
        dt: f32,
    ) -> Result<()> {
        if contact_reduction_output_count != state.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_planar resident: contact reduction output count {contact_reduction_output_count} != wheel count {}",
                state.wheel_count
            )));
        }
        let next_state = 1 - state.current_state;
        queue.write_buffer(
            &state.uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [state.wheel_count as u32, 0, 0, 0],
                pose: [0.0; 4],
                velocity: [0.0; 4],
                params0: [
                    finite_positive_or(state.mass, 1.0, 1.0e-6),
                    finite_positive_or(state.yaw_inertia, 1.0, 1.0e-6),
                    finite_or(dt, 0.0).max(0.0),
                    finite_or(state.linear_damping, 0.0).max(0.0),
                ],
                params1: [finite_or(state.yaw_damping, 0.0).max(0.0), 0.0, 0.0, 0.0],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_planar resident contact-reduction io bg"),
            layout: &self.layout_io4,
            entries: &[
                entry(0, &state.wheel_offsets),
                entry(1, contact_reduction_output_buffer),
                entry(2, &state.state_buffers[next_state]),
                entry(3, &state.state_buffers[state.current_state]),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_planar resident contact-reduction uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &state.uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_planar resident contact-reduction encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_planar resident contact-reduction pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_planar_state_from_contact_reduction);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        state.current_state = next_state;
        Ok(())
    }

    pub fn step_planar_resident_from_feedback_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &mut TireVehiclePlanarResidentState,
        feedback_output: &TireFemContactFeedbackResidentOutput,
        atlas_layout: &TireFemGpuBatchAtlasLayout,
        dt: f32,
    ) -> Result<()> {
        if atlas_layout.rows.len() != state.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_planar resident feedback: atlas tire count {} != wheel count {}",
                atlas_layout.rows.len(),
                state.wheel_count
            )));
        }
        if feedback_output.node_count != atlas_layout.plan.total_node_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_planar resident feedback: feedback node rows {} != atlas node count {}",
                feedback_output.node_count, atlas_layout.plan.total_node_count
            )));
        }
        let next_state = 1 - state.current_state;
        queue.write_buffer(
            &state.uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [state.wheel_count as u32, 0, 0, 0],
                pose: [0.0; 4],
                velocity: [0.0; 4],
                params0: [
                    finite_positive_or(state.mass, 1.0, 1.0e-6),
                    finite_positive_or(state.yaw_inertia, 1.0, 1.0e-6),
                    finite_or(dt, 0.0).max(0.0),
                    finite_or(state.linear_damping, 0.0).max(0.0),
                ],
                params1: [finite_or(state.yaw_damping, 0.0).max(0.0), 0.0, 0.0, 0.0],
            }),
        );
        let offset_rows = atlas_layout
            .rows
            .iter()
            .map(|row| {
                [
                    row.node_offset as f32,
                    row.node_count as f32,
                    row.tire_index as f32,
                    0.0,
                ]
            })
            .collect::<Vec<_>>();
        let offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_vehicle_planar resident feedback atlas offsets"),
            size: (offset_rows.len().max(1) * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !offset_rows.is_empty() {
            queue.write_buffer(&offsets_buf, 0, bytemuck::cast_slice(&offset_rows));
        }
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_planar resident feedback io bg"),
            layout: &self.layout_io6,
            entries: &[
                entry(0, &state.wheel_offsets),
                entry(1, &feedback_output.forces),
                entry(2, &state.state_buffers[next_state]),
                entry(3, &state.state_buffers[state.current_state]),
                entry(4, &state.state_buffers[state.current_state]),
                entry(5, &offsets_buf),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_planar resident feedback uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &state.uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_planar resident feedback encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_planar resident feedback pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_planar_state_from_feedback);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        state.current_state = next_state;
        Ok(())
    }

    pub fn readback_planar_resident_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &TireVehiclePlanarResidentState,
    ) -> Result<TireVehiclePlanarGpuOutput> {
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_vehicle_planar resident stage"),
            size: vec4_bytes(3),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_planar resident readback encoder"),
        });
        encoder.copy_buffer_to_buffer(state.current_state_buffer(), 0, &stage, 0, vec4_bytes(3));
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(device, &stage, "tire_vehicle_planar resident output")?;
        let state_row = rows.first().copied().unwrap_or([0.0; 4]);
        let yaw = rows.get(1).copied().unwrap_or([0.0; 4]);
        let force = rows.get(2).copied().unwrap_or([0.0; 4]);
        Ok(TireVehiclePlanarGpuOutput {
            position_xz: [state_row[0], state_row[1]],
            velocity_xz: [state_row[2], state_row[3]],
            yaw: yaw[0],
            yaw_rate: yaw[1],
            yaw_moment: yaw[2],
            finite: yaw[3] > 0.5,
            force_xz: [force[0], force[1]],
        })
    }

    pub fn create_vertical_resident_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehicleVerticalGpuInput,
        wheel_count: usize,
    ) -> Result<TireVehicleVerticalResidentState> {
        let n = input.validate(wheel_count)?;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let state_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let state_a = mk_buf(
            "tire_vehicle_vertical resident state a",
            vec4_bytes(1),
            state_usage,
        );
        let state_b = mk_buf(
            "tire_vehicle_vertical resident state b",
            vec4_bytes(1),
            state_usage,
        );
        let uniforms = mk_buf(
            "tire_vehicle_vertical resident uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let row = [[finite_or(input.y, 0.0), finite_or(input.vy, 0.0), 0.0, 1.0]];
        queue.write_buffer(&state_a, 0, bytemuck::cast_slice(&row));
        queue.write_buffer(&state_b, 0, bytemuck::cast_slice(&row));
        Ok(TireVehicleVerticalResidentState {
            state_buffers: [state_a, state_b],
            uniforms,
            wheel_count: n,
            current_state: 0,
            mass: input.mass,
            gravity: input.gravity,
            vertical_damping: input.vertical_damping,
        })
    }

    pub fn step_vertical_resident_from_suspension(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &mut TireVehicleVerticalResidentState,
        suspension_rows: &wgpu::Buffer,
        suspension_count: usize,
        dt: f32,
    ) -> Result<()> {
        if suspension_count != state.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_vertical resident: suspension count {suspension_count} != wheel count {}",
                state.wheel_count
            )));
        }
        let next_state = 1 - state.current_state;
        queue.write_buffer(
            &state.uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [state.wheel_count as u32, 0, 0, 0],
                pose: [0.0; 4],
                velocity: [0.0; 4],
                params0: [
                    finite_positive_or(state.mass, 1.0, 1.0e-6),
                    finite_or(state.gravity, 9.81).max(0.0),
                    finite_or(dt, 0.0).max(0.0),
                    finite_or(state.vertical_damping, 0.0).max(0.0),
                ],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_vertical resident io bg"),
            layout: &self.layout_io4,
            entries: &[
                entry(0, suspension_rows),
                entry(1, suspension_rows),
                entry(2, &state.state_buffers[next_state]),
                entry(3, &state.state_buffers[state.current_state]),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_vertical resident uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &state.uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_vertical resident encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_vertical resident pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_vertical_state_from_suspension);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        state.current_state = next_state;
        Ok(())
    }

    pub fn readback_vertical_resident_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &TireVehicleVerticalResidentState,
    ) -> Result<TireVehicleVerticalGpuOutput> {
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_vehicle_vertical resident stage"),
            size: vec4_bytes(1),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_vertical resident readback encoder"),
        });
        encoder.copy_buffer_to_buffer(state.current_state_buffer(), 0, &stage, 0, vec4_bytes(1));
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(device, &stage, "tire_vehicle_vertical resident output")?;
        let row = rows.first().copied().unwrap_or([0.0; 4]);
        Ok(TireVehicleVerticalGpuOutput {
            y: row[0],
            vy: row[1],
            force_y: row[2],
            finite: row[3] > 0.5,
        })
    }

    pub fn dispatch_wheel_spin_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehicleWheelSpinGpuInput,
    ) -> Result<TireVehicleWheelSpinGpuOutput> {
        let n = input.validate()?;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let state = mk_buf("tire_vehicle_wheel_spin state", vec4_bytes(n), usage_ro);
        let commands = mk_buf("tire_vehicle_wheel_spin commands", vec4_bytes(n), usage_ro);
        let out = mk_buf("tire_vehicle_wheel_spin out", vec4_bytes(n), usage_rw);
        let stage = mk_buf(
            "tire_vehicle_wheel_spin stage",
            vec4_bytes(n),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let uniforms = mk_buf(
            "tire_vehicle_wheel_spin uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&state, 0, bytemuck::cast_slice(&input.wheel_state));
        queue.write_buffer(&commands, 0, bytemuck::cast_slice(&input.wheel_commands));
        let zero_rows = vec![[0.0_f32; 4]; n];
        queue.write_buffer(&out, 0, bytemuck::cast_slice(&zero_rows));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [0.0; 4],
                velocity: [0.0; 4],
                params0: [1.0, 1.0, finite_or(input.dt, 0.0).max(0.0), 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_wheel_spin io bg"),
            layout: &self.layout_io,
            entries: &[entry(0, &state), entry(1, &commands), entry(2, &out)],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_wheel_spin uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_wheel_spin encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_wheel_spin pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_wheel_spin);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &stage, 0, vec4_bytes(n));
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(device, &stage, "tire_vehicle_wheel_spin output")?;
        let finite = rows.iter().all(|row| row[3] > 0.5);
        Ok(TireVehicleWheelSpinGpuOutput { rows, finite })
    }

    pub fn create_wheel_spin_resident_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehicleWheelSpinGpuInput,
    ) -> Result<TireVehicleWheelSpinResidentBatch> {
        let n = input.validate()?;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let state_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let state_a = mk_buf(
            "tire_vehicle_wheel_spin resident state a",
            vec4_bytes(n),
            state_usage,
        );
        let state_b = mk_buf(
            "tire_vehicle_wheel_spin resident state b",
            vec4_bytes(n),
            state_usage,
        );
        let commands = mk_buf(
            "tire_vehicle_wheel_spin resident commands",
            vec4_bytes(n),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let uniforms = mk_buf(
            "tire_vehicle_wheel_spin resident uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&state_a, 0, bytemuck::cast_slice(&input.wheel_state));
        queue.write_buffer(&state_b, 0, bytemuck::cast_slice(&input.wheel_state));
        queue.write_buffer(&commands, 0, bytemuck::cast_slice(&input.wheel_commands));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [0.0; 4],
                velocity: [0.0; 4],
                params0: [1.0, 1.0, finite_or(input.dt, 0.0).max(0.0), 0.0],
                params1: [0.0; 4],
            }),
        );
        Ok(TireVehicleWheelSpinResidentBatch {
            state_buffers: [state_a, state_b],
            commands,
            uniforms,
            wheel_count: n,
            current_state: 0,
        })
    }

    pub fn upload_wheel_spin_resident_commands(
        &self,
        queue: &wgpu::Queue,
        batch: &TireVehicleWheelSpinResidentBatch,
        wheel_commands: &[[f32; 4]],
    ) -> Result<()> {
        if wheel_commands.len() != batch.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_wheel_spin resident: command count {} != wheel count {}",
                wheel_commands.len(),
                batch.wheel_count
            )));
        }
        queue.write_buffer(&batch.commands, 0, bytemuck::cast_slice(wheel_commands));
        Ok(())
    }

    pub fn step_wheel_spin_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &mut TireVehicleWheelSpinResidentBatch,
        dt: f32,
    ) -> Result<()> {
        let next_state = 1 - batch.current_state;
        queue.write_buffer(
            &batch.uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [batch.wheel_count as u32, 0, 0, 0],
                pose: [0.0; 4],
                velocity: [0.0; 4],
                params0: [1.0, 1.0, finite_or(dt, 0.0).max(0.0), 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_wheel_spin resident io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &batch.state_buffers[batch.current_state]),
                entry(1, &batch.commands),
                entry(2, &batch.state_buffers[next_state]),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_wheel_spin resident uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &batch.uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_wheel_spin resident encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_wheel_spin resident pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_wheel_spin_state);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        batch.current_state = next_state;
        Ok(())
    }

    pub fn readback_wheel_spin_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireVehicleWheelSpinResidentBatch,
    ) -> Result<TireVehicleWheelSpinStateGpuOutput> {
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_vehicle_wheel_spin resident stage"),
            size: vec4_bytes(batch.wheel_count),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_wheel_spin resident readback encoder"),
        });
        encoder.copy_buffer_to_buffer(
            batch.current_state_buffer(),
            0,
            &stage,
            0,
            vec4_bytes(batch.wheel_count),
        );
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(device, &stage, "tire_vehicle_wheel_spin resident output")?;
        let finite = rows.iter().all(|row| row[3] > 0.5);
        Ok(TireVehicleWheelSpinStateGpuOutput { rows, finite })
    }

    pub fn dispatch_wheel_kinematics_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehicleWheelKinematicsGpuInput,
    ) -> Result<TireVehicleWheelKinematicsGpuOutput> {
        let n = input.validate()?;
        let out_rows = n * 2;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mounts = mk_buf(
            "tire_vehicle_wheel_kinematics mounts",
            vec4_bytes(n),
            usage_ro,
        );
        let state = mk_buf(
            "tire_vehicle_wheel_kinematics state",
            vec4_bytes(n),
            usage_ro,
        );
        let out = mk_buf(
            "tire_vehicle_wheel_kinematics out",
            vec4_bytes(out_rows),
            usage_rw,
        );
        let stage = mk_buf(
            "tire_vehicle_wheel_kinematics stage",
            vec4_bytes(out_rows),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let uniforms = mk_buf(
            "tire_vehicle_wheel_kinematics uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&mounts, 0, bytemuck::cast_slice(&input.wheel_mounts));
        queue.write_buffer(&state, 0, bytemuck::cast_slice(&input.wheel_state));
        let zero_rows = vec![[0.0_f32; 4]; out_rows];
        queue.write_buffer(&out, 0, bytemuck::cast_slice(&zero_rows));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [0.0, 0.0, finite_or(input.yaw, 0.0), 0.0],
                velocity: [
                    finite_or(input.velocity_xz[0], 0.0),
                    finite_or(input.velocity_xz[1], 0.0),
                    finite_or(input.yaw_rate, 0.0),
                    0.0,
                ],
                params0: [1.0, 1.0, 0.0, 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_wheel_kinematics io bg"),
            layout: &self.layout_io,
            entries: &[entry(0, &mounts), entry(1, &state), entry(2, &out)],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_wheel_kinematics uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_wheel_kinematics encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_wheel_kinematics pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_wheel_kinematics);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &stage, 0, vec4_bytes(out_rows));
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(device, &stage, "tire_vehicle_wheel_kinematics output")?;
        let finite = rows
            .iter()
            .take(n)
            .all(|row| row.iter().all(|v| v.is_finite()))
            && rows
                .iter()
                .skip(n)
                .all(|row| row.iter().all(|v| v.is_finite()));
        Ok(TireVehicleWheelKinematicsGpuOutput { rows, finite })
    }

    pub fn dispatch_wheel_kinematics_from_spin_resident_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spin_batch: &TireVehicleWheelSpinResidentBatch,
        input: &TireVehicleWheelKinematicsFromSpinGpuInput,
    ) -> Result<TireVehicleWheelKinematicsGpuOutput> {
        let n = input.validate(spin_batch.wheel_count)?;
        let packed = input.packed_rows();
        let out_rows = n * 2;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let packed_buf = mk_buf(
            "tire_vehicle_wheel_kinematics_from_spin packed",
            vec4_bytes(packed.len()),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let out = mk_buf(
            "tire_vehicle_wheel_kinematics_from_spin out",
            vec4_bytes(out_rows),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let stage = mk_buf(
            "tire_vehicle_wheel_kinematics_from_spin stage",
            vec4_bytes(out_rows),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let uniforms = mk_buf(
            "tire_vehicle_wheel_kinematics_from_spin uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&packed_buf, 0, bytemuck::cast_slice(&packed));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [0.0, 0.0, finite_or(input.yaw, 0.0), 0.0],
                velocity: [
                    finite_or(input.velocity_xz[0], 0.0),
                    finite_or(input.velocity_xz[1], 0.0),
                    finite_or(input.yaw_rate, 0.0),
                    0.0,
                ],
                params0: [1.0, 1.0, 0.0, 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_wheel_kinematics_from_spin io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, spin_batch.current_state_buffer()),
                entry(1, &packed_buf),
                entry(2, &out),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_wheel_kinematics_from_spin uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_wheel_kinematics_from_spin encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_wheel_kinematics_from_spin pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_wheel_kinematics_from_spin);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &stage, 0, vec4_bytes(out_rows));
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(
            device,
            &stage,
            "tire_vehicle_wheel_kinematics_from_spin output",
        )?;
        let finite = rows.iter().all(|row| row.iter().all(|v| v.is_finite()));
        Ok(TireVehicleWheelKinematicsGpuOutput { rows, finite })
    }

    pub fn dispatch_suspension_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehicleSuspensionGpuInput,
    ) -> Result<TireVehicleSuspensionGpuOutput> {
        let n = input.validate()?;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let geometry = mk_buf("tire_vehicle_suspension geometry", vec4_bytes(n), usage_ro);
        let params = mk_buf("tire_vehicle_suspension params", vec4_bytes(n), usage_ro);
        let out = mk_buf("tire_vehicle_suspension out", vec4_bytes(n), usage_rw);
        let stage = mk_buf(
            "tire_vehicle_suspension stage",
            vec4_bytes(n),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let uniforms = mk_buf(
            "tire_vehicle_suspension uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(
            &geometry,
            0,
            bytemuck::cast_slice(&input.suspension_geometry),
        );
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&input.suspension_params));
        let zero_rows = vec![[0.0_f32; 4]; n];
        queue.write_buffer(&out, 0, bytemuck::cast_slice(&zero_rows));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [finite_or(input.chassis_y, 0.0), 0.0, 0.0, 0.0],
                velocity: [finite_or(input.chassis_vy, 0.0), 0.0, 0.0, 0.0],
                params0: [1.0, 1.0, 0.0, 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_suspension io bg"),
            layout: &self.layout_io,
            entries: &[entry(0, &geometry), entry(1, &params), entry(2, &out)],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_suspension uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_suspension encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_suspension pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_suspension);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &stage, 0, vec4_bytes(n));
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(device, &stage, "tire_vehicle_suspension output")?;
        let finite = rows.iter().all(|row| row[3] > 0.5);
        Ok(TireVehicleSuspensionGpuOutput { rows, finite })
    }

    pub fn create_suspension_resident_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehicleSuspensionGpuInput,
    ) -> Result<TireVehicleSuspensionResidentBatch> {
        let n = input.validate()?;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let geometry = mk_buf(
            "tire_vehicle_suspension resident geometry",
            vec4_bytes(n),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let params = mk_buf(
            "tire_vehicle_suspension resident params",
            vec4_bytes(n),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let rows = mk_buf(
            "tire_vehicle_suspension resident rows",
            vec4_bytes(n),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let uniforms = mk_buf(
            "tire_vehicle_suspension resident uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(
            &geometry,
            0,
            bytemuck::cast_slice(&input.suspension_geometry),
        );
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&input.suspension_params));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [finite_or(input.chassis_y, 0.0), 0.0, 0.0, 0.0],
                velocity: [finite_or(input.chassis_vy, 0.0), 0.0, 0.0, 0.0],
                params0: [1.0, 1.0, 0.0, 0.0],
                params1: [0.0; 4],
            }),
        );
        Ok(TireVehicleSuspensionResidentBatch {
            geometry,
            params,
            rows,
            uniforms,
            wheel_count: n,
        })
    }

    pub fn upload_suspension_resident_params(
        &self,
        queue: &wgpu::Queue,
        batch: &TireVehicleSuspensionResidentBatch,
        suspension_params: &[[f32; 4]],
    ) -> Result<()> {
        if suspension_params.len() != batch.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_suspension resident: params count {} != wheel count {}",
                suspension_params.len(),
                batch.wheel_count
            )));
        }
        queue.write_buffer(&batch.params, 0, bytemuck::cast_slice(suspension_params));
        Ok(())
    }

    pub fn step_suspension_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireVehicleSuspensionResidentBatch,
        chassis_y: f32,
        chassis_vy: f32,
    ) -> Result<()> {
        queue.write_buffer(
            &batch.uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [batch.wheel_count as u32, 0, 0, 0],
                pose: [finite_or(chassis_y, 0.0), 0.0, 0.0, 0.0],
                velocity: [finite_or(chassis_vy, 0.0), 0.0, 0.0, 0.0],
                params0: [1.0, 1.0, 0.0, 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_suspension resident io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &batch.geometry),
                entry(1, &batch.params),
                entry(2, &batch.rows),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_suspension resident uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &batch.uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_suspension resident encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_suspension resident pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_suspension);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    pub fn step_suspension_resident_from_vertical_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireVehicleSuspensionResidentBatch,
        vertical: &TireVehicleVerticalResidentState,
    ) -> Result<()> {
        if batch.wheel_count != vertical.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_suspension resident vertical: suspension count {} != vertical wheel count {}",
                batch.wheel_count, vertical.wheel_count
            )));
        }
        queue.write_buffer(
            &batch.uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [batch.wheel_count as u32, 0, 0, 0],
                pose: [0.0; 4],
                velocity: [0.0; 4],
                params0: [1.0, 1.0, 0.0, 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_suspension resident vertical io bg"),
            layout: &self.layout_io4,
            entries: &[
                entry(0, &batch.geometry),
                entry(1, &batch.params),
                entry(2, &batch.rows),
                entry(3, vertical.current_state_buffer()),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_suspension resident vertical uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &batch.uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_suspension resident vertical encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_suspension resident vertical pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_suspension_from_vertical);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    pub fn readback_suspension_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireVehicleSuspensionResidentBatch,
    ) -> Result<TireVehicleSuspensionGpuOutput> {
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_vehicle_suspension resident stage"),
            size: vec4_bytes(batch.wheel_count),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_suspension resident readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&batch.rows, 0, &stage, 0, vec4_bytes(batch.wheel_count));
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(device, &stage, "tire_vehicle_suspension resident output")?;
        let finite = rows.iter().all(|row| row[3] > 0.5);
        Ok(TireVehicleSuspensionGpuOutput { rows, finite })
    }

    pub fn dispatch_brush_rows_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehicleBrushRowsGpuInput,
    ) -> Result<TireVehicleBrushRowsGpuOutput> {
        let n = input.validate()?;
        let packed = input.packed_rows();
        let out_rows = n * 7;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mounts = mk_buf("tire_vehicle_brush_rows mounts", vec4_bytes(n), usage_ro);
        let packed_buf = mk_buf(
            "tire_vehicle_brush_rows packed",
            vec4_bytes(packed.len()),
            usage_ro,
        );
        let out = mk_buf(
            "tire_vehicle_brush_rows out",
            vec4_bytes(out_rows),
            usage_rw,
        );
        let stage = mk_buf(
            "tire_vehicle_brush_rows stage",
            vec4_bytes(out_rows),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let uniforms = mk_buf(
            "tire_vehicle_brush_rows uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&mounts, 0, bytemuck::cast_slice(&input.wheel_mounts));
        queue.write_buffer(&packed_buf, 0, bytemuck::cast_slice(&packed));
        let zero_rows = vec![[0.0_f32; 4]; out_rows];
        queue.write_buffer(&out, 0, bytemuck::cast_slice(&zero_rows));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [
                    finite_or(input.chassis_pose[0], 0.0),
                    finite_or(input.chassis_pose[1], 0.0),
                    finite_or(input.chassis_pose[2], 0.0),
                    finite_or(input.chassis_pose[3], 0.0),
                ],
                velocity: [
                    finite_or(input.chassis_velocity[0], 0.0),
                    finite_or(input.chassis_velocity[1], 0.0),
                    finite_or(input.chassis_velocity[2], 0.0),
                    finite_or(input.chassis_velocity[3], 0.0),
                ],
                params0: [1.0, 1.0, 0.0, 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_brush_rows io bg"),
            layout: &self.layout_io,
            entries: &[entry(0, &mounts), entry(1, &packed_buf), entry(2, &out)],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_brush_rows uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_brush_rows encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_brush_rows pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_brush_rows);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &stage, 0, vec4_bytes(out_rows));
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(device, &stage, "tire_vehicle_brush_rows output")?;
        let mut wheels = Vec::with_capacity(n);
        let mut finite = true;
        for i in 0..n {
            let base = i * 7;
            let row = BrushWheelInput {
                hub_pos: rows[base],
                hub_vel: rows[base + 1],
                heading: rows[base + 2],
                spin: rows[base + 3],
                tire: rows[base + 4],
                material: rows[base + 5],
                extra: rows[base + 6],
            };
            finite &= rows[base..base + 7]
                .iter()
                .flat_map(|row| row.iter())
                .all(|v| v.is_finite());
            wheels.push(row);
        }
        Ok(TireVehicleBrushRowsGpuOutput { wheels, finite })
    }

    pub fn dispatch_brush_rows_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireVehicleBrushRowsGpuInput,
    ) -> Result<TireVehicleBrushRowsResidentOutput> {
        let n = input.validate()?;
        let packed = input.packed_rows();
        let out_rows = n * 7;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let mounts = mk_buf(
            "tire_vehicle_brush_rows resident mounts",
            vec4_bytes(n),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let packed_buf = mk_buf(
            "tire_vehicle_brush_rows resident packed",
            vec4_bytes(packed.len()),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let out = mk_buf(
            "tire_vehicle_brush_rows resident out",
            vec4_bytes(out_rows),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let uniforms = mk_buf(
            "tire_vehicle_brush_rows resident uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&mounts, 0, bytemuck::cast_slice(&input.wheel_mounts));
        queue.write_buffer(&packed_buf, 0, bytemuck::cast_slice(&packed));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [
                    finite_or(input.chassis_pose[0], 0.0),
                    finite_or(input.chassis_pose[1], 0.0),
                    finite_or(input.chassis_pose[2], 0.0),
                    finite_or(input.chassis_pose[3], 0.0),
                ],
                velocity: [
                    finite_or(input.chassis_velocity[0], 0.0),
                    finite_or(input.chassis_velocity[1], 0.0),
                    finite_or(input.chassis_velocity[2], 0.0),
                    finite_or(input.chassis_velocity[3], 0.0),
                ],
                params0: [1.0, 1.0, 0.0, 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_brush_rows resident io bg"),
            layout: &self.layout_io,
            entries: &[entry(0, &mounts), entry(1, &packed_buf), entry(2, &out)],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_brush_rows resident uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_brush_rows resident encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_brush_rows resident pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_brush_rows);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(TireVehicleBrushRowsResidentOutput {
            buffer: out,
            wheel_count: n,
        })
    }

    pub fn dispatch_brush_rows_from_spin_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spin_batch: &TireVehicleWheelSpinResidentBatch,
        input: &TireVehicleBrushRowsFromSpinGpuInput,
    ) -> Result<TireVehicleBrushRowsResidentOutput> {
        let n = input.validate()?;
        if n != spin_batch.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_brush_rows_from_spin: input wheel count {n} != resident spin wheel count {}",
                spin_batch.wheel_count
            )));
        }
        let packed = input.packed_rows();
        let out_rows = n * 7;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let packed_buf = mk_buf(
            "tire_vehicle_brush_rows_from_spin packed",
            vec4_bytes(packed.len()),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let out = mk_buf(
            "tire_vehicle_brush_rows_from_spin out",
            vec4_bytes(out_rows),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let uniforms = mk_buf(
            "tire_vehicle_brush_rows_from_spin uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&packed_buf, 0, bytemuck::cast_slice(&packed));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [
                    finite_or(input.chassis_pose[0], 0.0),
                    finite_or(input.chassis_pose[1], 0.0),
                    finite_or(input.chassis_pose[2], 0.0),
                    finite_or(input.chassis_pose[3], 0.0),
                ],
                velocity: [
                    finite_or(input.chassis_velocity[0], 0.0),
                    finite_or(input.chassis_velocity[1], 0.0),
                    finite_or(input.chassis_velocity[2], 0.0),
                    finite_or(input.chassis_velocity[3], 0.0),
                ],
                params0: [1.0, 1.0, 0.0, 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_brush_rows_from_spin io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, spin_batch.current_state_buffer()),
                entry(1, &packed_buf),
                entry(2, &out),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_brush_rows_from_spin uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_brush_rows_from_spin encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_brush_rows_from_spin pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_brush_rows_from_spin);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(TireVehicleBrushRowsResidentOutput {
            buffer: out,
            wheel_count: n,
        })
    }

    pub fn dispatch_brush_rows_from_spin_suspension_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spin_batch: &TireVehicleWheelSpinResidentBatch,
        suspension_batch: &TireVehicleSuspensionResidentBatch,
        input: &TireVehicleBrushRowsFromSpinGpuInput,
    ) -> Result<TireVehicleBrushRowsResidentOutput> {
        let n = input.validate()?;
        if n != spin_batch.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_brush_rows_from_spin_suspension: input wheel count {n} != resident spin wheel count {}",
                spin_batch.wheel_count
            )));
        }
        if n != suspension_batch.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_brush_rows_from_spin_suspension: input wheel count {n} != resident suspension wheel count {}",
                suspension_batch.wheel_count
            )));
        }
        let packed = input.packed_rows();
        let out_rows = n * 7;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let packed_buf = mk_buf(
            "tire_vehicle_brush_rows_from_spin_suspension packed",
            vec4_bytes(packed.len()),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let out = mk_buf(
            "tire_vehicle_brush_rows_from_spin_suspension out",
            vec4_bytes(out_rows),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let uniforms = mk_buf(
            "tire_vehicle_brush_rows_from_spin_suspension uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&packed_buf, 0, bytemuck::cast_slice(&packed));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [
                    finite_or(input.chassis_pose[0], 0.0),
                    finite_or(input.chassis_pose[1], 0.0),
                    finite_or(input.chassis_pose[2], 0.0),
                    finite_or(input.chassis_pose[3], 0.0),
                ],
                velocity: [
                    finite_or(input.chassis_velocity[0], 0.0),
                    finite_or(input.chassis_velocity[1], 0.0),
                    finite_or(input.chassis_velocity[2], 0.0),
                    finite_or(input.chassis_velocity[3], 0.0),
                ],
                params0: [1.0, 1.0, 0.0, 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension io bg"),
            layout: &self.layout_io4,
            entries: &[
                entry(0, spin_batch.current_state_buffer()),
                entry(1, &packed_buf),
                entry(2, &out),
                entry(3, suspension_batch.rows_buffer()),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_brush_rows_from_spin_suspension);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(TireVehicleBrushRowsResidentOutput {
            buffer: out,
            wheel_count: n,
        })
    }

    pub fn dispatch_brush_rows_from_spin_suspension_planar_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spin_batch: &TireVehicleWheelSpinResidentBatch,
        suspension_batch: &TireVehicleSuspensionResidentBatch,
        planar_state: &TireVehiclePlanarResidentState,
        input: &TireVehicleBrushRowsFromSpinGpuInput,
        chassis_y: f32,
        chassis_vy: f32,
    ) -> Result<TireVehicleBrushRowsResidentOutput> {
        let n = input.validate()?;
        if n != spin_batch.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_brush_rows_from_spin_suspension_planar: input wheel count {n} != resident spin wheel count {}",
                spin_batch.wheel_count
            )));
        }
        if n != suspension_batch.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_brush_rows_from_spin_suspension_planar: input wheel count {n} != resident suspension wheel count {}",
                suspension_batch.wheel_count
            )));
        }
        if n != planar_state.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_brush_rows_from_spin_suspension_planar: input wheel count {n} != resident planar wheel count {}",
                planar_state.wheel_count
            )));
        }
        let packed = input.packed_rows();
        let out_rows = n * 7;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let packed_buf = mk_buf(
            "tire_vehicle_brush_rows_from_spin_suspension_planar packed",
            vec4_bytes(packed.len()),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let out = mk_buf(
            "tire_vehicle_brush_rows_from_spin_suspension_planar out",
            vec4_bytes(out_rows),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let uniforms = mk_buf(
            "tire_vehicle_brush_rows_from_spin_suspension_planar uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&packed_buf, 0, bytemuck::cast_slice(&packed));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [0.0, finite_or(chassis_y, 0.0), 0.0, 0.0],
                velocity: [0.0, finite_or(chassis_vy, 0.0), 0.0, 0.0],
                params0: [1.0, 1.0, 0.0, 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension_planar io bg"),
            layout: &self.layout_io5,
            entries: &[
                entry(0, spin_batch.current_state_buffer()),
                entry(1, &packed_buf),
                entry(2, &out),
                entry(3, suspension_batch.rows_buffer()),
                entry(4, planar_state.current_state_buffer()),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension_planar uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension_planar encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension_planar pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_brush_rows_from_spin_suspension_planar);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(TireVehicleBrushRowsResidentOutput {
            buffer: out,
            wheel_count: n,
        })
    }

    pub fn dispatch_brush_rows_from_spin_suspension_chassis_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spin_batch: &TireVehicleWheelSpinResidentBatch,
        suspension_batch: &TireVehicleSuspensionResidentBatch,
        planar_state: &TireVehiclePlanarResidentState,
        vertical_state: &TireVehicleVerticalResidentState,
        input: &TireVehicleBrushRowsFromSpinGpuInput,
    ) -> Result<TireVehicleBrushRowsResidentOutput> {
        let n = input.validate()?;
        if n != spin_batch.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_brush_rows_from_spin_suspension_chassis: input wheel count {n} != resident spin wheel count {}",
                spin_batch.wheel_count
            )));
        }
        if n != suspension_batch.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_brush_rows_from_spin_suspension_chassis: input wheel count {n} != resident suspension wheel count {}",
                suspension_batch.wheel_count
            )));
        }
        if n != planar_state.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_brush_rows_from_spin_suspension_chassis: input wheel count {n} != resident planar wheel count {}",
                planar_state.wheel_count
            )));
        }
        if n != vertical_state.wheel_count {
            return Err(Error::Solver(format!(
                "tire_vehicle_brush_rows_from_spin_suspension_chassis: input wheel count {n} != resident vertical wheel count {}",
                vertical_state.wheel_count
            )));
        }
        let packed = input.packed_rows();
        let out_rows = n * 7;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let packed_buf = mk_buf(
            "tire_vehicle_brush_rows_from_spin_suspension_chassis packed",
            vec4_bytes(packed.len()),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let out = mk_buf(
            "tire_vehicle_brush_rows_from_spin_suspension_chassis out",
            vec4_bytes(out_rows),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let uniforms = mk_buf(
            "tire_vehicle_brush_rows_from_spin_suspension_chassis uniforms",
            core::mem::size_of::<VehicleUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&packed_buf, 0, bytemuck::cast_slice(&packed));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&VehicleUniforms {
                counts: [n as u32, 0, 0, 0],
                pose: [0.0; 4],
                velocity: [0.0; 4],
                params0: [1.0, 1.0, 0.0, 0.0],
                params1: [0.0; 4],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension_chassis io bg"),
            layout: &self.layout_io6,
            entries: &[
                entry(0, spin_batch.current_state_buffer()),
                entry(1, &packed_buf),
                entry(2, &out),
                entry(3, suspension_batch.rows_buffer()),
                entry(4, planar_state.current_state_buffer()),
                entry(5, vertical_state.current_state_buffer()),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension_chassis uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension_chassis encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_vehicle_brush_rows_from_spin_suspension_chassis pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_brush_rows_from_spin_suspension_chassis);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(TireVehicleBrushRowsResidentOutput {
            buffer: out,
            wheel_count: n,
        })
    }

    pub fn create_resident_brush_owner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spin_input: &TireVehicleWheelSpinGpuInput,
        suspension_input: &TireVehicleSuspensionGpuInput,
        brush_input: TireVehicleBrushRowsFromSpinGpuInput,
    ) -> Result<TireVehicleResidentBrushOwner> {
        let n = brush_input.validate()?;
        if spin_input.wheel_state.len() != n || suspension_input.suspension_geometry.len() != n {
            return Err(Error::Solver(format!(
                "tire_vehicle_resident_brush_owner: wheel count mismatch brush={n} spin={} suspension={}",
                spin_input.wheel_state.len(),
                suspension_input.suspension_geometry.len()
            )));
        }
        let spin = self.create_wheel_spin_resident_batch(device, queue, spin_input)?;
        let suspension = self.create_suspension_resident_batch(device, queue, suspension_input)?;
        Ok(TireVehicleResidentBrushOwner {
            spin,
            suspension,
            brush_input,
        })
    }

    pub fn upload_resident_brush_owner_wheel_commands(
        &self,
        queue: &wgpu::Queue,
        owner: &TireVehicleResidentBrushOwner,
        wheel_commands: &[[f32; 4]],
    ) -> Result<()> {
        self.upload_wheel_spin_resident_commands(queue, &owner.spin, wheel_commands)
    }

    pub fn upload_resident_brush_owner_suspension_params(
        &self,
        queue: &wgpu::Queue,
        owner: &TireVehicleResidentBrushOwner,
        suspension_params: &[[f32; 4]],
    ) -> Result<()> {
        self.upload_suspension_resident_params(queue, &owner.suspension, suspension_params)
    }

    pub fn step_resident_brush_owner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        owner: &mut TireVehicleResidentBrushOwner,
        dt: f32,
        chassis_pose: [f32; 4],
        chassis_velocity: [f32; 4],
        chassis_y: f32,
        chassis_vy: f32,
    ) -> Result<TireVehicleBrushRowsResidentOutput> {
        owner.brush_input.chassis_pose = chassis_pose;
        owner.brush_input.chassis_velocity = chassis_velocity;
        self.step_wheel_spin_resident(device, queue, &mut owner.spin, dt)?;
        self.step_suspension_resident(device, queue, &owner.suspension, chassis_y, chassis_vy)?;
        self.dispatch_brush_rows_from_spin_suspension_resident(
            device,
            queue,
            &owner.spin,
            &owner.suspension,
            &owner.brush_input,
        )
    }

    pub fn create_resident_chassis_brush_owner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spin_input: &TireVehicleWheelSpinGpuInput,
        suspension_input: &TireVehicleSuspensionGpuInput,
        brush_input: TireVehicleBrushRowsFromSpinGpuInput,
        planar_input: &TireVehiclePlanarFromBrushGpuInput,
    ) -> Result<TireVehicleResidentChassisBrushOwner> {
        let vertical_input = TireVehicleVerticalGpuInput {
            y: suspension_input.chassis_y,
            vy: suspension_input.chassis_vy,
            mass: planar_input.mass,
            gravity: 9.81,
            vertical_damping: 0.0,
            dt: planar_input.dt,
        };
        self.create_resident_chassis_brush_owner_with_vertical(
            device,
            queue,
            spin_input,
            suspension_input,
            brush_input,
            planar_input,
            &vertical_input,
        )
    }

    pub fn create_resident_chassis_brush_owner_with_vertical(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spin_input: &TireVehicleWheelSpinGpuInput,
        suspension_input: &TireVehicleSuspensionGpuInput,
        brush_input: TireVehicleBrushRowsFromSpinGpuInput,
        planar_input: &TireVehiclePlanarFromBrushGpuInput,
        vertical_input: &TireVehicleVerticalGpuInput,
    ) -> Result<TireVehicleResidentChassisBrushOwner> {
        let brush = self.create_resident_brush_owner(
            device,
            queue,
            spin_input,
            suspension_input,
            brush_input,
        )?;
        let planar =
            self.create_planar_resident_state(device, queue, planar_input, brush.wheel_count())?;
        let vertical = self.create_vertical_resident_state(
            device,
            queue,
            vertical_input,
            brush.wheel_count(),
        )?;
        Ok(TireVehicleResidentChassisBrushOwner {
            brush,
            planar,
            vertical,
        })
    }

    pub fn step_resident_chassis_brush_owner(
        &self,
        brush_kernel: &TireBrushGpuKernel,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        owner: &mut TireVehicleResidentChassisBrushOwner,
        dt: f32,
        chassis_pose: [f32; 4],
        chassis_velocity: [f32; 4],
        chassis_y: f32,
        chassis_vy: f32,
        terrain: Option<&BrushTerrain<'_>>,
    ) -> Result<TireBrushResidentOutput> {
        let brush_rows = self.step_resident_brush_owner(
            device,
            queue,
            &mut owner.brush,
            dt,
            chassis_pose,
            chassis_velocity,
            chassis_y,
            chassis_vy,
        )?;
        let brush_output = brush_kernel.dispatch_input_buffer_resident(
            device,
            queue,
            &brush_rows.buffer,
            brush_rows.wheel_count,
            terrain,
        )?;
        self.step_planar_resident_from_brush_output(
            device,
            queue,
            &mut owner.planar,
            &brush_output.buffer,
            brush_output.wheel_count,
            dt,
        )?;
        Ok(brush_output)
    }

    pub fn step_resident_chassis_brush_owner_from_planar_state(
        &self,
        brush_kernel: &TireBrushGpuKernel,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        owner: &mut TireVehicleResidentChassisBrushOwner,
        dt: f32,
        chassis_y: f32,
        chassis_vy: f32,
        terrain: Option<&BrushTerrain<'_>>,
    ) -> Result<TireBrushResidentOutput> {
        self.step_wheel_spin_resident(device, queue, &mut owner.brush.spin, dt)?;
        self.step_suspension_resident(
            device,
            queue,
            &owner.brush.suspension,
            chassis_y,
            chassis_vy,
        )?;
        let brush_rows = self.dispatch_brush_rows_from_spin_suspension_planar_resident(
            device,
            queue,
            &owner.brush.spin,
            &owner.brush.suspension,
            &owner.planar,
            &owner.brush.brush_input,
            chassis_y,
            chassis_vy,
        )?;
        let brush_output = brush_kernel.dispatch_input_buffer_resident(
            device,
            queue,
            &brush_rows.buffer,
            brush_rows.wheel_count,
            terrain,
        )?;
        self.step_planar_resident_from_brush_output(
            device,
            queue,
            &mut owner.planar,
            &brush_output.buffer,
            brush_output.wheel_count,
            dt,
        )?;
        Ok(brush_output)
    }

    /// Advances the resident chassis with retained brush scratch and GPU terrain.
    /// Owner buffers must belong to the engine. Other vehicle stages retain their
    /// existing allocation/submission behavior; this is not an atomic frame API.
    pub fn step_resident_chassis_brush_owner_with_gpu_terrain(
        &self,
        engine: &molla_compute::Engine,
        owner: &mut TireVehicleResidentChassisBrushOwner,
        dt: f32,
        brush: &crate::PreparedTireBrushTerrain,
    ) -> Result<TireBrushResidentOutput> {
        brush.validate_owner(engine, owner.wheel_count())?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Solver("GPU vehicle engine required".into()))?;
        crate::soil_gpu_identity::validate_soil_device(device, &self.device)?;
        let queue = engine.queue().unwrap();
        self.step_wheel_spin_resident(device, queue, &mut owner.brush.spin, dt)?;
        self.step_suspension_resident_from_vertical_state(
            device,
            queue,
            &owner.brush.suspension,
            &owner.vertical,
        )?;
        let rows = self.dispatch_brush_rows_from_spin_suspension_chassis_resident(
            device,
            queue,
            &owner.brush.spin,
            &owner.brush.suspension,
            &owner.planar,
            &owner.vertical,
            &owner.brush.brush_input,
        )?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        brush.encode(engine, &mut encoder, &rows.buffer)?;
        queue.submit([encoder.finish()]);
        let output = brush.output();
        self.step_planar_resident_from_brush_output(
            device,
            queue,
            &mut owner.planar,
            &output.buffer,
            output.wheel_count,
            dt,
        )?;
        self.step_vertical_resident_from_suspension(
            device,
            queue,
            &mut owner.vertical,
            owner.brush.suspension.rows_buffer(),
            owner.brush.suspension.wheel_count(),
            dt,
        )?;
        Ok(output)
    }

    pub fn step_resident_chassis_brush_owner_from_resident_state(
        &self,
        brush_kernel: &TireBrushGpuKernel,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        owner: &mut TireVehicleResidentChassisBrushOwner,
        dt: f32,
        terrain: Option<&BrushTerrain<'_>>,
    ) -> Result<TireBrushResidentOutput> {
        self.step_wheel_spin_resident(device, queue, &mut owner.brush.spin, dt)?;
        self.step_suspension_resident_from_vertical_state(
            device,
            queue,
            &owner.brush.suspension,
            &owner.vertical,
        )?;
        let brush_rows = self.dispatch_brush_rows_from_spin_suspension_chassis_resident(
            device,
            queue,
            &owner.brush.spin,
            &owner.brush.suspension,
            &owner.planar,
            &owner.vertical,
            &owner.brush.brush_input,
        )?;
        let brush_output = brush_kernel.dispatch_input_buffer_resident(
            device,
            queue,
            &brush_rows.buffer,
            brush_rows.wheel_count,
            terrain,
        )?;
        self.step_planar_resident_from_brush_output(
            device,
            queue,
            &mut owner.planar,
            &brush_output.buffer,
            brush_output.wheel_count,
            dt,
        )?;
        self.step_vertical_resident_from_suspension(
            device,
            queue,
            &mut owner.vertical,
            owner.brush.suspension.rows_buffer(),
            owner.brush.suspension.wheel_count(),
            dt,
        )?;
        Ok(brush_output)
    }
}

fn read_vec4(device: &wgpu::Device, stage: &wgpu::Buffer, label: &str) -> Result<Vec<[f32; 4]>> {
    let slice = stage.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    let label = label.to_owned();
    let map_label = label.clone();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        sender
            .send(r.map_err(|e| Error::Gpu(format!("{map_label} map: {e:?}"))))
            .ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("{label} poll: {e:?}")))?;
    receiver
        .recv()
        .map_err(|e| Error::Gpu(format!("{label} recv: {e:?}")))??;
    let data = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    stage.unmap();
    Ok(out)
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn validate_wheels(label: &str, offsets: &[[f32; 4]], wrenches: &[[f32; 4]]) -> Result<usize> {
    let n = offsets.len();
    if n == 0 {
        return Err(Error::Solver(format!(
            "{label}: at least one wheel is required"
        )));
    }
    if wrenches.len() != n {
        return Err(Error::Solver(format!(
            "{label}: offsets/wrenches length mismatch"
        )));
    }
    if n > WORKGROUP_SIZE as usize {
        return Err(Error::Solver(format!(
            "{label}: wheel count {n} exceeds one-workgroup limit {WORKGROUP_SIZE}"
        )));
    }
    Ok(n)
}

fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() { v } else { fallback }
}

fn finite_positive_or(v: f32, fallback: f32, floor: f32) -> f32 {
    if v.is_finite() && v >= floor {
        v
    } else {
        fallback.max(floor)
    }
}
