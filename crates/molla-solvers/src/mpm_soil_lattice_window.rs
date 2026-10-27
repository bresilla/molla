//! Integer-cell soil windows against a fixed lattice anchor.

use super::*;

impl MpmSoilCheckpoint {
    pub(super) fn validate_lattice_window(&self) -> Result<()> {
        crate::soil_candidates_gpu::validate_node_window(
            self.cfg.grid_origin,
            self.cfg.cell_size,
            self.cfg.grid_size,
            self.grid_window_cells,
        )?;
        let inv_h = 1.0 / self.cfg.cell_size;
        if !inv_h.is_finite() {
            return Err(invalid("integer window requires finite inverse cell size"));
        }
        let limit = (1_i64 << 23) - 1;
        for particle in self.stream(0).chunks_exact(4) {
            for axis in 0..3 {
                let xp = (f32::from_bits(particle[axis]) - self.cfg.grid_origin[axis]) * inv_h;
                let base = (xp - 0.5).floor();
                if !base.is_finite()
                    || f64::from(base) < -limit as f64
                    || f64::from(base) + 2.0 > limit as f64
                {
                    return Err(invalid(
                        "particle stencil exceeds exact lattice index range",
                    ));
                }
                let local = base as i64 - i64::from(self.grid_window_cells[axis]);
                if local < 0 || local + 2 >= i64::from(self.cfg.grid_size) {
                    return Err(invalid(
                        "particle stencil is outside the integer grid window",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Selects absolute integer window cells relative to config().grid_origin.
    /// Preserves every particle word and the anchor. Moved axes require overlapping
    /// windows and interior transfer/Jacobi support in both; future motion is separate.
    pub fn with_grid_window_cells(&self, cells: [i32; 3]) -> Result<Self> {
        self.validate()?;
        self.validate_lattice_window()?;
        crate::soil_candidates_gpu::validate_node_window(
            self.cfg.grid_origin,
            self.cfg.cell_size,
            self.cfg.grid_size,
            cells,
        )?;
        if cells
            .iter()
            .zip(self.grid_window_cells)
            .any(|(&a, b)| (i64::from(a) - i64::from(b)).abs() > i64::from(self.cfg.grid_size))
        {
            return Err(invalid("integer grid move requires overlapping windows"));
        }
        let iterations = match self.cfg.time_step {
            MpmTimeStep::Explicit => 0,
            MpmTimeStep::Implicit { jacobi_iters } => jacobi_iters,
        };
        let margin = 3 + i64::from(iterations);
        let upper = i64::from(self.cfg.grid_size) - margin;
        for particle in self.stream(0).chunks_exact(4) {
            for axis in 0..3 {
                if cells[axis] == self.grid_window_cells[axis] {
                    continue;
                }
                let xp = (f32::from_bits(particle[axis]) - self.cfg.grid_origin[axis])
                    * (1.0 / self.cfg.cell_size);
                let global = (xp - 0.5).floor() as i64;
                if [self.grid_window_cells[axis], cells[axis]]
                    .into_iter()
                    .any(|offset| {
                        let base = global - i64::from(offset);
                        base < margin || base + 2 >= upper
                    })
                {
                    return Err(invalid("integer grid move intersects boundary support"));
                }
            }
        }
        let mut result = self.clone();
        result.grid_window_cells = cells;
        Ok(result)
    }
}

impl MpmSimulation {
    /// Absolute active-window cells relative to config().grid_origin.
    pub fn grid_window_cells(&self) -> [i32; 3] {
        self.grid_window_cells
    }

    pub(crate) fn move_grid_window_cells_from_checkpoint(
        &mut self,
        checkpoint: &MpmSoilCheckpoint,
        cells: [i32; 3],
    ) -> Result<()> {
        if checkpoint.config() != self.cfg
            || checkpoint.particle_count() != self.particle_count
            || checkpoint.grid_window_cells != self.grid_window_cells
            || self.soil_identity_snapshot().is_some()
            || self.motion_guard.is_some()
        {
            return Err(invalid(
                "integer window move requires current unowned soil without a swept guard",
            ));
        }
        let shifted = checkpoint.with_grid_window_cells(cells)?;
        if self.grid_window_cells != shifted.grid_window_cells {
            self.grid_window_cells = shifted.grid_window_cells;
            self.active_grid_needs_reset = true;
            self.invalidate_prepared_steps();
            self.tire_raster_cache = None;
            self.tire_feedback_cache = None;
        }
        Ok(())
    }

    /// Blocking bounded window movement for detached, unowned soil.
    /// Captures for admission without uploading particles; rigids and swept guards reject.
    /// The queue must belong to this simulation's device.
    pub fn move_soil_grid_window_cells_with(
        &mut self,
        queue: &wgpu::Queue,
        cells: [i32; 3],
        max_particles: u32,
    ) -> Result<()> {
        if self.rigid_count != 0
            || self.soil_identity_snapshot().is_some()
            || self.motion_guard.is_some()
            || self.particle_count > max_particles
        {
            return Err(invalid(
                "integer window move requires bounded detached unowned soil without a swept guard",
            ));
        }
        crate::soil_candidates_gpu::validate_node_window(
            self.cfg.grid_origin,
            self.cfg.cell_size,
            self.cfg.grid_size,
            cells,
        )?;
        let snapshot = self.checkpoint_soil_with(queue)?;
        self.move_grid_window_cells_from_checkpoint(&snapshot, cells)
    }
}

#[cfg(test)]
#[path = "mpm_soil_lattice_window_tests.rs"]
mod tests;
