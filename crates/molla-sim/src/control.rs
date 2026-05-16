//! `Control` — user-supplied joint targets and applied wrenches.

use molla_math::{Real, SpatialVector};
use molla_storage::Buffer;

/// Per-step control inputs handed to the solver.
///
/// **Per-DOF actuation** (length `joint_dof_total`):
/// - [`Self::joint_target_mode`] selects what kind of drive to run
///   (see [`super::JointTargetMode`]).
/// - [`Self::joint_target_pos`] / [`Self::joint_target_vel`] are the
///   targets fed into the chosen drive.
/// - [`Self::joint_target_kp`] / [`Self::joint_target_kd`] are the PD
///   gains (per-DOF).
/// - [`Self::joint_force`] is added as a feed-forward effort.
#[derive(Debug)]
pub struct Control {
    /// Per-DOF actuator mode (encoded [`super::JointTargetMode`] as `u32`).
    pub joint_target_mode: Buffer<u32>,
    /// Per-DOF position target (rad or m). Used when mode is
    /// `Position` or `PositionVelocity`. For `Effort` mode this slot
    /// carries the direct effort.
    pub joint_target_pos: Buffer<Real>,
    /// Per-DOF velocity target (rad/s or m/s). Used when mode is
    /// `Velocity` or `PositionVelocity`.
    pub joint_target_vel: Buffer<Real>,
    /// Per-DOF position gain (N·m/rad or N/m).
    pub joint_target_kp: Buffer<Real>,
    /// Per-DOF velocity gain (N·m·s/rad or N·s/m).
    pub joint_target_kd: Buffer<Real>,
    /// Direct joint torque/force feed-forward (N·m or N), added to
    /// any drive output. Length = `joint_dof_total`.
    pub joint_force: Buffer<Real>,
    /// External wrenches per body, world frame at COM.
    pub body_external_f: Buffer<SpatialVector>,
}

impl Control {
    /// An empty control set (no actuation, no external forces).
    #[must_use]
    pub fn empty() -> Self {
        use molla_storage::BufferUsage::Storage;
        Self {
            joint_target_mode: Buffer::from_vec(Vec::new(), Storage),
            joint_target_pos: Buffer::from_vec(Vec::new(), Storage),
            joint_target_vel: Buffer::from_vec(Vec::new(), Storage),
            joint_target_kp: Buffer::from_vec(Vec::new(), Storage),
            joint_target_kd: Buffer::from_vec(Vec::new(), Storage),
            joint_force: Buffer::from_vec(Vec::new(), Storage),
            body_external_f: Buffer::from_vec(Vec::new(), Storage),
        }
    }

    /// Allocate a zero-initialised control set sized for `model`.
    /// All per-DOF slots are sized to `joint_dof_total`. Modes default
    /// to `JointTargetMode::None` (no drive).
    #[must_use]
    pub fn for_model(model: &super::Model) -> Self {
        use molla_storage::BufferUsage::Storage;
        let n_dof = model.joint_dof_total;
        Self {
            joint_target_mode: Buffer::from_vec(vec![0u32; n_dof], Storage),
            joint_target_pos: Buffer::from_vec(vec![0.0; n_dof], Storage),
            joint_target_vel: Buffer::from_vec(vec![0.0; n_dof], Storage),
            joint_target_kp: Buffer::from_vec(vec![0.0; n_dof], Storage),
            joint_target_kd: Buffer::from_vec(vec![0.0; n_dof], Storage),
            joint_force: Buffer::from_vec(vec![0.0; n_dof], Storage),
            body_external_f: Buffer::from_vec(vec![SpatialVector::ZERO; model.body_count], Storage),
        }
    }

    /// Allocate device buffers and upload host data.
    #[cfg(feature = "gpu")]
    pub fn upload_to_device(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> molla_core::Result<()> {
        macro_rules! up {
            ($field:ident) => {{
                self.$field.ensure_device(device)?;
                self.$field.upload(queue)?;
            }};
        }
        up!(joint_target_mode);
        up!(joint_target_pos);
        up!(joint_target_vel);
        up!(joint_target_kp);
        up!(joint_target_kd);
        up!(joint_force);
        up!(body_external_f);
        Ok(())
    }
}
