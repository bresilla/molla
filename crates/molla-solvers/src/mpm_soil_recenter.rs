//! Lossless coordinate translations of detached soil and its grid.

use super::*;

pub(super) fn exact_sum(a: f64, b: f64) -> Result<f64> {
    let sum = a + b;
    let split = sum - a;
    let error = (a - (sum - split)) + (b - split);
    if !sum.is_finite() || error != 0.0 {
        return Err(invalid(
            "soil recenter requires an exact coordinate translation",
        ));
    }
    Ok(sum)
}

fn shifted(value: f32, delta: f64) -> Result<f32> {
    if delta == 0.0 {
        return Ok(value);
    }
    let exact = exact_sum(f64::from(value), delta)?;
    let rounded = exact as f32;
    if !rounded.is_finite()
        || f64::from(rounded) != exact
        || (rounded != 0.0 && !rounded.is_normal())
    {
        return Err(invalid("soil recenter would round or flush a coordinate"));
    }
    Ok(rounded)
}

pub(super) fn translation(cfg: MpmConfig, origin: [f32; 3]) -> Result<[f64; 3]> {
    if origin.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite soil recenter origin"));
    }
    let mut delta = [0.0; 3];
    for axis in 0..3 {
        delta[axis] = exact_sum(f64::from(origin[axis]), -f64::from(cfg.grid_origin[axis]))?;
        for cells in [2.0f32, cfg.grid_size as f32 - 3.0] {
            for (before, after) in [
                (
                    cfg.grid_origin[axis] + cells * cfg.cell_size,
                    origin[axis] + cells * cfg.cell_size,
                ),
                (
                    cells.mul_add(cfg.cell_size, cfg.grid_origin[axis]),
                    cells.mul_add(cfg.cell_size, origin[axis]),
                ),
            ] {
                if !before.is_finite()
                    || !after.is_finite()
                    || shifted(before, delta[axis])? != after
                {
                    return Err(invalid(
                        "soil recenter changes a grid position-clamp boundary",
                    ));
                }
            }
        }
    }
    Ok(delta)
}

impl MpmSoilCheckpoint {
    /// Moves physical grid bounds without changing particle coordinates or history.
    /// Requires an integral lattice shift, identical particle stencil fractions and
    /// transfer/Jacobi support inside both windows. Future motion needs its own halo.
    pub fn with_grid_window(&self, new_origin: [f32; 3]) -> Result<Self> {
        self.require_origin_only_grid()?;
        self.validate()?;
        let offsets = window_offset(self.cfg, new_origin)?;
        let iterations = match self.cfg.time_step {
            MpmTimeStep::Explicit => 0,
            MpmTimeStep::Implicit { jacobi_iters } => jacobi_iters,
        };
        let margin = 3.0 + f64::from(iterations);
        let upper = f64::from(self.cfg.grid_size) - margin;
        let inv_h = 1.0 / self.cfg.cell_size;
        for particle in self.stream(0).chunks_exact(4) {
            for axis in 0..3 {
                if offsets[axis] == 0.0 {
                    continue;
                }
                let position = f32::from_bits(particle[axis]);
                let old = (position - self.cfg.grid_origin[axis]) * inv_h;
                let new = (position - new_origin[axis]) * inv_h;
                let old_base = (old - 0.5).floor();
                let new_base = (new - 0.5).floor();
                if !old.is_finite()
                    || !new.is_finite()
                    || f64::from(old) - f64::from(new) != offsets[axis]
                    || old - old_base != new - new_base
                    || [old_base, new_base]
                        .into_iter()
                        .any(|base| f64::from(base) < margin || f64::from(base) + 2.0 >= upper)
                {
                    return Err(invalid(
                        "soil window changes stencil fractions or intersects boundary support",
                    ));
                }
            }
        }
        let mut result = self.clone();
        result.cfg.grid_origin = new_origin;
        Ok(result)
    }

    /// Translate particles and grid together, preserving every other state word.
    /// Rejects non-exact f32 translations and changed position-clamp boundaries.
    /// The caller must subtract `new_origin - old_origin` from its external world
    /// origin. World tiles, rigid bodies and external snapshots are not translated.
    /// This does not promise bitwise-identical trajectories after further stepping.
    pub fn recentered(&self, new_origin: [f32; 3]) -> Result<Self> {
        self.require_origin_only_grid()?;
        self.validate()?;
        let delta = translation(self.cfg, new_origin)?;
        let mut result = self.clone();
        for particle in result.words[..self.count as usize * 4].chunks_exact_mut(4) {
            for axis in 0..3 {
                let before = f32::from_bits(particle[axis]);
                let after = shifted(before, delta[axis])?;
                let old_relative = before - self.cfg.grid_origin[axis];
                let new_relative = after - new_origin[axis];
                if !old_relative.is_finite() || old_relative != new_relative {
                    return Err(invalid("soil recenter changes particle grid coordinates"));
                }
                particle[axis] = after.to_bits();
            }
        }
        result.cfg.grid_origin = new_origin;
        Ok(result)
    }
}

pub(super) fn window_offset(cfg: MpmConfig, origin: [f32; 3]) -> Result<[f64; 3]> {
    let delta = translation(cfg, origin)?;
    let mut offsets = [0.0; 3];
    for axis in 0..3 {
        offsets[axis] = delta[axis] / f64::from(cfg.cell_size);
        if !offsets[axis].is_finite()
            || offsets[axis].fract() != 0.0
            || offsets[axis].abs() > f64::from(cfg.grid_size)
            || offsets[axis] * f64::from(cfg.cell_size) != delta[axis]
        {
            return Err(invalid(
                "soil window requires an exact overlapping lattice shift",
            ));
        }
    }
    Ok(offsets)
}

impl MpmSimulation {
    pub(crate) fn move_grid_window_from_checkpoint(
        &mut self,
        checkpoint: &MpmSoilCheckpoint,
        origin: [f32; 3],
    ) -> Result<()> {
        if checkpoint.config() != self.cfg
            || checkpoint.particle_count() != self.particle_count
            || checkpoint.grid_window_cells() != self.grid_window_cells
            || self.soil_identity_snapshot().is_some()
            || self.motion_guard.is_some()
        {
            return Err(invalid(
                "coupled grid window requires current unowned soil without a swept guard",
            ));
        }
        let shifted = checkpoint.with_grid_window(origin)?;
        if shifted.cfg.grid_origin.map(f32::to_bits) != self.cfg.grid_origin.map(f32::to_bits) {
            self.cfg.grid_origin = shifted.cfg.grid_origin;
            self.active_grid_needs_reset = true;
            self.invalidate_prepared_steps();
            self.tire_raster_cache = None;
            self.tire_feedback_cache = None;
        }
        Ok(())
    }

    /// Blocking detached-soil recenter at a submitted step boundary, not a hot path.
    /// Captures current state, rejects lossy shifts before upload, and reuses solver
    /// buffers. Success invalidates prepared work and terrain/feedback caches.
    /// Returns the offset added to local positions; subtract it from the external
    /// world origin. Attached rigids, installed soil IDs and swept guards reject.
    /// Archive/world owners must coordinate their own future rebasing transaction.
    /// The queue must match this device. Device loss is not transactional rollback.
    pub fn recenter_soil_with(
        &mut self,
        queue: &wgpu::Queue,
        new_origin: [f32; 3],
    ) -> Result<[f64; 3]> {
        if self.rigid_count != 0
            || self.soil_identity_snapshot().is_some()
            || self.motion_guard.is_some()
        {
            return Err(invalid(
                "soil recenter requires detached, unowned soil without a swept guard",
            ));
        }
        let delta = translation(self.cfg, new_origin)?;
        let snapshot = self.checkpoint_soil_with(queue)?;
        self.validate_soil_restore(&snapshot)?;
        if new_origin.map(f32::to_bits) == self.cfg.grid_origin.map(f32::to_bits) {
            return Ok(delta);
        }
        let shifted = snapshot.recentered(new_origin)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MPM soil recenter"),
            });
        self.encode_soil_restore(&mut encoder, &shifted);
        queue.submit([encoder.finish()]);
        self.cfg.grid_origin = new_origin;
        self.finish_soil_restore(&shifted);
        Ok(delta)
    }
}

#[cfg(test)]
#[path = "mpm_soil_recenter_tests.rs"]
mod tests;
