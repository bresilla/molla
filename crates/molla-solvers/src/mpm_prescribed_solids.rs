//! Prescribed rigid volume boundaries for MPM grid coupling.

use super::*;

/// Local geometry centered at the body's center of mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MpmSolidShape {
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: [f32; 3],
    },
    /// Finite cylinder with its axle along local Y.
    Cylinder {
        radius: f32,
        half_width: f32,
    },
}

/// World-space prescribed pose and velocities in metres, seconds and radians.
/// Orientation is a unit quaternion in XYZW order. Each occupied interior grid
/// node sticks to `linear_velocity + angular_velocity × (node - center)`.
/// This is a sticking volume boundary, not a calibrated tire/friction model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpmPrescribedSolid {
    pub shape: MpmSolidShape,
    pub center: [f32; 3],
    pub orientation: [f32; 4],
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
}

impl MpmPrescribedSolid {
    pub(super) fn rows(&self) -> Result<([f32; 4], [[f32; 4]; 4])> {
        let bad = || Error::Solver("MPM prescribed solid: invalid geometry or pose".into());
        if self
            .center
            .iter()
            .chain(&self.orientation)
            .chain(&self.linear_velocity)
            .chain(&self.angular_velocity)
            .any(|x| !x.is_finite())
        {
            return Err(bad());
        }
        let norm = self.orientation.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > 1e-4 {
            return Err(bad());
        }
        let (dims, radius) = match self.shape {
            MpmSolidShape::Sphere { radius } => ([radius, radius, radius, 1.0], radius),
            MpmSolidShape::Box {
                half_extents: [x, y, z],
            } => ([x, y, z, 2.0], (x * x + y * y + z * z).sqrt()),
            MpmSolidShape::Cylinder { radius, half_width } => {
                ([radius, half_width, radius, 3.0], radius.hypot(half_width))
            }
        };
        if !radius.is_finite() || dims[..3].iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err(bad());
        }
        let v = self.linear_velocity;
        let w = self.angular_velocity;
        Ok((
            [self.center[0], self.center[1], self.center[2], radius],
            [
                [v[0], v[1], v[2], 0.0],
                [w[0], w[1], w[2], 0.0],
                self.orientation.map(|x| x / norm),
                dims,
            ],
        ))
    }
}

impl MpmSimulation {
    pub(super) fn validate_prescribed_step(&self, dt: f32) -> Result<()> {
        self.validate_native_mode()?;
        if self.prescribed_solids
            && (!matches!(self.cfg.time_step, MpmTimeStep::Explicit)
                || !dt.is_finite()
                || dt <= 0.0
                || !self.cfg.gravity_y.is_finite())
        {
            return Err(Error::Solver("MPM prescribed solids require explicit stepping, finite gravity and positive finite dt".into()));
        }
        Ok(())
    }

    /// Replace all prescribed solids for explicit stepping. Interior overlaps belong
    /// to the first solid only. Wrenches use the same order, world axes and each
    /// solid's center of mass; removing/reordering entries changes that mapping.
    /// The caller supplies current poses and consumes equal/opposite wrenches.
    /// No outside-shell force, artificial grid mass, rigid-ground or brush force
    /// is added by this route. Legacy sphere coupling is a separate replacement
    /// API. These uploads must occur between submitted batches, not while recording
    /// distinct poses into one unsubmitted encoder. Read reactions before replacing
    /// poses/order; batched stepping reports only the final substep's wrench.
    pub fn set_prescribed_solids(
        &mut self,
        queue: &wgpu::Queue,
        solids: &[MpmPrescribedSolid],
    ) -> Result<()> {
        if !solids.is_empty()
            && (!matches!(self.cfg.time_step, MpmTimeStep::Explicit)
                || !self.cfg.gravity_y.is_finite())
        {
            return Err(Error::Solver(
                "MPM prescribed solids require explicit stepping and finite gravity".into(),
            ));
        }
        if solids.len() > self.cfg.max_rigid as usize {
            return Err(Error::Solver(
                "MPM prescribed solid capacity exceeded".into(),
            ));
        }
        let rows: Vec<_> = solids
            .iter()
            .map(MpmPrescribedSolid::rows)
            .collect::<Result<_>>()?;
        let centers: Vec<_> = rows.iter().map(|row| row.0).collect();
        let motion: Vec<_> = rows.iter().flat_map(|row| row.1).collect();
        if !rows.is_empty() {
            molla_core::gpu_write_buffer!(
                queue,
                &self.rigid_center,
                0,
                bytemuck::cast_slice(&centers)
            );
            molla_core::gpu_write_buffer!(queue, &self.rigid_vel, 0, bytemuck::cast_slice(&motion));
        }
        if self.rigid_count != solids.len() as u32 || self.prescribed_solids != !solids.is_empty() {
            self.invalidate_prepared_steps();
        }
        self.rigid_count = solids.len() as u32;
        self.prescribed_solids = !solids.is_empty();
        self.rigid_epoch = Arc::new(());
        Ok(())
    }
}

#[cfg(test)]
#[path = "mpm_prescribed_solids_tests.rs"]
mod tests;
