//! Retained compliant-loop coordinates, world COM Jacobians and feedback rows.

use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::{GpuTransform, Real, Transform};
use molla_sim::JointTargetMode;
use molla_sim::joint_control::{JointAxis, JointMotor, JointSoftness, MotorModel};

pub const COMPLIANT_LOOP_ROW_FAULT: u32 = 1 << 15;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompliantLoopRequest {
    pub parent_frame: GpuTransform,
    pub child_frame: GpuTransform,
    /// Parent (-1 world), child, axis, term (0 limit, 1 position, 2 rate, 3 effort).
    pub metadata: [i32; 4],
    /// Lower/upper position bounds, motor target position, motor force cap.
    pub range: [f32; 4],
    /// Stiffness, damping, target rate, effective-mass scaling flag.
    pub feedback: [f32; 4],
    pub parent_position_residue: [f32; 4],
    pub child_position_residue: [f32; 4],
    pub parent_rotation_residue: [f32; 4],
    pub child_rotation_residue: [f32; 4],
}

impl CompliantLoopRequest {
    fn frames(parent: i32, child: u32, frames: [Transform; 2], axis: JointAxis) -> Result<Self> {
        if parent < -1 || child > i32::MAX as u32 || parent == child as i32 {
            return Err(Error::Gpu("invalid compliant loop endpoints".into()));
        }
        let mut packed = [GpuTransform::from(Transform::IDENTITY); 2];
        let mut residues = [[0.0; 4]; 2];
        let mut rotation_residues = [[0.0; 4]; 2];
        for (i, frame) in frames.into_iter().enumerate() {
            if !frame.position.is_finite()
                || !frame.rotation.is_finite()
                || (frame.rotation.length_squared() - 1.0).abs() > 1e-6
            {
                return Err(Error::Gpu("invalid compliant loop frame".into()));
            }
            packed[i] = frame.into();
            for axis in 0..3 {
                if !packed[i].position[axis].is_finite() {
                    return Err(Error::Gpu("compliant loop frame exceeds f32 range".into()));
                }
                residues[i][axis] =
                    (frame.position[axis] - packed[i].position[axis] as Real) as f32;
            }
            for axis in 0..4 {
                rotation_residues[i][axis] =
                    (frame.rotation.to_array()[axis] - packed[i].rotation[axis] as Real) as f32;
            }
        }
        Ok(Self {
            parent_frame: packed[0],
            child_frame: packed[1],
            metadata: [parent, child as i32, axis as i32, 0],
            range: [0.0; 4],
            feedback: [0.0; 4],
            parent_position_residue: residues[0],
            child_position_residue: residues[1],
            parent_rotation_residue: rotation_residues[0],
            child_rotation_residue: rotation_residues[1],
        })
    }

    pub fn limit(
        parent: i32,
        child: u32,
        frames: [Transform; 2],
        axis: JointAxis,
        limits: [Real; 2],
        softness: JointSoftness,
    ) -> Result<Self> {
        softness.validate()?;
        if limits.iter().any(|v| v.is_nan())
            || limits[0] > limits[1]
            || limits[0] == Real::INFINITY
            || limits[1] == Real::NEG_INFINITY
            || limits
                .iter()
                .any(|v| v.is_finite() && !(*v as f32).is_finite())
            || (limits[0] != limits[1] && limits[0] as f32 == limits[1] as f32)
        {
            return Err(Error::Gpu("invalid compliant loop interval".into()));
        }
        let mut request = Self::frames(parent, child, frames, axis)?;
        let omega = std::f64::consts::TAU * softness.natural_frequency;
        request.range = [limits[0] as f32, limits[1] as f32, 0.0, f32::INFINITY];
        request.feedback = [
            (omega * omega) as f32,
            (2.0 * softness.damping_ratio * omega) as f32,
            0.0,
            1.0,
        ];
        if request.feedback.iter().any(|v| !v.is_finite())
            || (omega > 0.0 && request.feedback[0] == 0.0)
        {
            return Err(Error::Gpu(
                "compliant loop softness exceeds f32 range".into(),
            ));
        }
        Ok(request)
    }

    pub fn motor(
        parent: i32,
        child: u32,
        frames: [Transform; 2],
        axis: JointAxis,
        motor: JointMotor,
    ) -> Result<Self> {
        motor.validate()?;
        let mut request = Self::frames(parent, child, frames, axis)?;
        request.metadata[3] = match motor.mode {
            JointTargetMode::Position | JointTargetMode::PositionVelocity => 1,
            JointTargetMode::Effort => 3,
            _ => 2,
        };
        request.range = [
            0.0,
            0.0,
            motor.target_position as f32,
            motor.max_force as f32,
        ];
        let stiffness = match motor.mode {
            JointTargetMode::Position | JointTargetMode::PositionVelocity => motor.stiffness,
            _ => 0.0,
        };
        let damping = if matches!(motor.mode, JointTargetMode::None | JointTargetMode::Effort) {
            0.0
        } else {
            motor.damping
        };
        let target_rate = match motor.mode {
            JointTargetMode::Velocity | JointTargetMode::PositionVelocity => motor.target_velocity,
            _ => 0.0,
        };
        request.feedback = [
            stiffness as f32,
            damping as f32,
            target_rate as f32,
            u32::from(
                motor.model == MotorModel::Acceleration && motor.mode != JointTargetMode::Effort,
            ) as f32,
        ];
        if request.feedback.iter().any(|v| !v.is_finite())
            || !request.range[2].is_finite()
            || (motor.max_force.is_finite() && !request.range[3].is_finite())
            || (stiffness > 0.0 && request.feedback[0] == 0.0)
            || (damping > 0.0 && request.feedback[1] == 0.0)
        {
            return Err(Error::Gpu("compliant loop motor exceeds f32 range".into()));
        }
        Ok(request)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompliantLoopRow {
    /// Endpoints and axis; w is 0 inactive, 1 feedback or 2 prescribed effort.
    pub metadata: [i32; 4],
    /// Child COM linear Jacobian xyz; coordinate error w.
    pub child_linear: [f32; 4],
    /// Child COM angular Jacobian xyz; coordinate rate w.
    pub child_angular: [f32; 4],
    /// Parent COM linear Jacobian xyz; lower force bound w.
    pub parent_linear: [f32; 4],
    /// Parent COM angular Jacobian xyz; upper force bound w.
    pub parent_angular: [f32; 4],
    pub feedback: [f32; 4],
}

pub struct CompliantLoopBindings<'a> {
    pub requests: &'a wgpu::Buffer,
    pub poses: &'a wgpu::Buffer,
    pub velocities: &'a wgpu::Buffer,
    pub centers: &'a wgpu::Buffer,
    pub position_residues: &'a wgpu::Buffer,
    pub rows: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
    pub rounding: &'a wgpu::Buffer,
    pub rotations: &'a wgpu::Buffer,
}

pub struct CompliantLoopKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

pub struct CompliantLoopDispatch {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    count: u32,
}

impl CompliantLoopDispatch {
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}

impl CompliantLoopKernel {
    pub fn new(engine: &Engine) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("missing compliant loop GPU device".into()))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compliant loop rows"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../shaders/quaternion_precision.wgsl"),
                    include_str!("../shaders/compliant_loop_rows.wgsl")
                )
                .into(),
            ),
        });
        let entries: Vec<_> = (0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding < 5 || binding == 8,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compliant loop rows"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compliant loop rows"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compliant loop rows"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("evaluate_rows"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self { pipeline, layout })
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        bindings: CompliantLoopBindings<'_>,
        count: u32,
    ) -> Result<CompliantLoopDispatch> {
        let buffers = [
            bindings.requests,
            bindings.poses,
            bindings.velocities,
            bindings.centers,
            bindings.position_residues,
            bindings.rows,
            bindings.status,
            bindings.rounding,
            bindings.rotations,
        ];
        if count == 0
            || bindings.requests.size()
                != u64::from(count) * std::mem::size_of::<CompliantLoopRequest>() as u64
            || bindings.rows.size() < u64::from(count) * 96
            || bindings.poses.size() == 0
            || !bindings.poses.size().is_multiple_of(32)
            || bindings.velocities.size() != bindings.poses.size()
            || bindings.centers.size() != bindings.poses.size() / 2
            || bindings.position_residues.size() != bindings.centers.size()
            || bindings.rotations.size() != bindings.poses.size()
            || bindings.status.size() < 4
            || bindings.rounding.size() < u64::from(count) * 4
            || count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension
            || buffers.iter().any(|buffer| {
                !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                    || buffer.size() > device.limits().max_storage_buffer_binding_size
            })
        {
            return Err(Error::Gpu("invalid compliant loop row capacity".into()));
        }
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compliant loop rows"),
            layout: &self.layout,
            entries: &entries,
        });
        Ok(CompliantLoopDispatch {
            pipeline: self.pipeline.clone(),
            group,
            count,
        })
    }
}

const _: () = {
    assert!(std::mem::size_of::<CompliantLoopRequest>() == 176);
    assert!(std::mem::size_of::<CompliantLoopRow>() == 96);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_preserve_feedback_modes_and_reject_unrepresentable_values() {
        let frames = [Transform::IDENTITY; 2];
        let softness = JointSoftness {
            natural_frequency: 7.0,
            damping_ratio: 0.6,
        };
        let limit = |parent, child, frames, interval, softness| {
            CompliantLoopRequest::limit(parent, child, frames, JointAxis::LinY, interval, softness)
        };
        assert!(limit(-1, 0, frames, [0.0; 2], softness).is_ok());
        let rotation =
            molla_math::Quat::from_rotation_z(0.7) * molla_math::Quat::from_rotation_y(-0.4);
        let rotated = [Transform::new(molla_math::Vec3::ZERO, rotation); 2];
        let packed = limit(-1, 0, rotated, [0.0; 2], softness).unwrap();
        for (frame, residue) in [
            (packed.parent_frame, packed.parent_rotation_residue),
            (packed.child_frame, packed.child_rotation_residue),
        ] {
            for (axis, residual) in residue.iter().enumerate() {
                let extended = frame.rotation[axis] as Real + *residual as Real;
                assert!((extended - rotation.to_array()[axis]).abs() < 2e-15);
            }
        }
        assert!(limit(0, 0, frames, [0.0; 2], softness).is_err());
        assert!(limit(-2, 0, frames, [0.0; 2], softness).is_err());
        assert!(limit(-1, u32::MAX, frames, [0.0; 2], softness).is_err());
        assert!(limit(-1, 0, frames, [1.0, 1.0 + 1e-12], softness).is_err());
        assert!(limit(-1, 0, frames, [f64::MAX; 2], softness).is_err());
        assert!(limit(-1, 0, frames, [f64::NAN, 1.0], softness).is_err());
        assert!(limit(-1, 0, frames, [2.0, 1.0], softness).is_err());
        assert!(limit(-1, 0, frames, [f64::INFINITY; 2], softness).is_err());
        let mut invalid_frames = frames;
        invalid_frames[0].rotation = molla_math::Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
        assert!(limit(-1, 0, invalid_frames, [0.0; 2], softness).is_err());
        assert!(
            limit(
                -1,
                0,
                frames,
                [0.0; 2],
                JointSoftness {
                    natural_frequency: 1e-100,
                    ..softness
                }
            )
            .is_err()
        );
        let motor = JointMotor {
            mode: JointTargetMode::PositionVelocity,
            stiffness: 12.0,
            damping: 3.0,
            target_position: 0.2,
            target_velocity: 0.3,
            ..Default::default()
        };
        let pack = |motor| CompliantLoopRequest::motor(-1, 0, frames, JointAxis::AngX, motor);
        let request = pack(motor).unwrap();
        assert_eq!(request.metadata[3], 1);
        assert_eq!(request.feedback, [12.0, 3.0, 0.3, 0.0]);
        let request = pack(JointMotor {
            mode: JointTargetMode::Velocity,
            model: MotorModel::Acceleration,
            ..motor
        })
        .unwrap();
        assert_eq!(request.metadata[3], 2);
        assert_eq!(request.feedback, [0.0, 3.0, 0.3, 1.0]);
        let request = pack(JointMotor {
            mode: JointTargetMode::Effort,
            model: MotorModel::Acceleration,
            stiffness: 0.0,
            damping: 0.0,
            ..motor
        })
        .unwrap();
        assert_eq!(request.metadata[3], 3);
        assert_eq!(request.feedback, [0.0; 4]);
        assert!(
            pack(JointMotor {
                max_force: f64::MAX,
                ..motor
            })
            .is_err()
        );
        assert!(
            pack(JointMotor {
                stiffness: 1e-100,
                ..motor
            })
            .is_err()
        );
    }
}
