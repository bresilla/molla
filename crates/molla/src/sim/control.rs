//! `Control` — user-supplied joint targets and applied wrenches.

use crate::math::{Real, SpatialVector};
use crate::storage::Buffer;

/// Per-step control inputs handed to the solver.
#[derive(Debug)]
pub struct Control {
    /// Joint position or velocity targets, depending on actuator mode.
    pub joint_target: Buffer<Real>,
    /// Direct joint torque/force (N·m or N).
    pub joint_force: Buffer<Real>,
    /// External wrenches per body, world frame at COM.
    pub body_external_f: Buffer<SpatialVector>,
}

impl Control {
    /// An empty control set (no actuation, no external forces).
    #[must_use]
    pub fn empty() -> Self {
        use crate::storage::BufferUsage::Storage;
        Self {
            joint_target: Buffer::from_vec(Vec::new(), Storage),
            joint_force: Buffer::from_vec(Vec::new(), Storage),
            body_external_f: Buffer::from_vec(Vec::new(), Storage),
        }
    }

    /// Allocate a zero-initialised control set sized for `model`.
    #[must_use]
    pub fn for_model(model: &super::Model) -> Self {
        use crate::storage::BufferUsage::Storage;
        Self {
            joint_target: Buffer::from_vec(Vec::new(), Storage),
            joint_force: Buffer::from_vec(Vec::new(), Storage),
            body_external_f: Buffer::from_vec(
                vec![SpatialVector::ZERO; model.body_count],
                Storage,
            ),
        }
    }

    /// Allocate device buffers and upload host data.
    #[cfg(feature = "gpu")]
    pub fn upload_to_device(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> crate::core::Result<()> {
        macro_rules! up {
            ($field:ident) => {{
                self.$field.ensure_device(device)?;
                self.$field.upload(queue)?;
            }};
        }
        up!(joint_target);
        up!(joint_force);
        up!(body_external_f);
        Ok(())
    }
}
