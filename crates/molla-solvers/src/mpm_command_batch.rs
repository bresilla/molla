//! Immutable parameter records and transactional caller-owned MPM encoding.

use super::*;

#[cfg(test)]
#[path = "mpm_command_batch_tests.rs"]
mod tests;

/// Experimental fixed timestep/substep record for one simulation configuration epoch.
pub struct PreparedMpmStep {
    epoch: Arc<()>,
    source: wgpu::Buffer,
    copies: Vec<(wgpu::Buffer, u64, u64)>,
    material_ids: Option<(u64, u64)>,
    substeps: usize,
    jacobi_iters: u32,
}

/// Exclusively borrows the simulation until submission or cancellation.
/// Encoder and submission queue must belong to the simulation's device.
pub struct MpmCommandBatch<'a> {
    simulation: &'a mut MpmSimulation,
    encoder: Option<wgpu::CommandEncoder>,
    initial_reset: bool,
    initial_quiescence_reset: bool,
    applied_material_ids: bool,
    submitted: bool,
}

impl MpmSimulation {
    pub(super) fn invalidate_prepared_steps(&mut self) {
        self.step_epoch = Arc::new(());
        self.invalidate_quiescence();
    }

    /// Captures experimental numeric records without queue writes or submissions; prepare outside stepping.
    /// Reseeding, material edits, dispatch-mode changes and storage growth invalidate old records.
    pub fn prepare_step(&self, dt: f32, substeps: usize) -> Result<PreparedMpmStep> {
        self.validate_prescribed_step(dt)?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(Error::Solver(
                "MPM prepared timestep must be finite and positive".into(),
            ));
        }
        if self.use_particle_order
            && self
                .particle_order
                .as_ref()
                .is_none_or(|o| o.capacity < self.particle_count)
        {
            return Err(Error::Solver(
                "MPM particle count exceeds prepared ordering capacity".into(),
            ));
        }
        let groups = self.device.limits().max_compute_workgroups_per_dimension;
        if self.particle_count.div_ceil(WORKGROUP_SIZE) > groups
            || self.rigid_count.div_ceil(WORKGROUP_SIZE) > groups
            || (!self.use_active_grid
                && self.cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE) > groups)
        {
            return Err(Error::Solver(
                "MPM prepared dispatch exceeds device limits".into(),
            ));
        }
        let active = if self.use_active_grid {
            Some(self.active_tiles.parameter_words(
                &self.cfg,
                self.particle_count,
                self.rigid_count,
                self.grid_window_cells,
            )?)
        } else {
            None
        };
        let ordering = if self.use_particle_order {
            Some(self.particle_order.as_ref().unwrap().parameter_words(
                &self.cfg,
                self.particle_count,
                self.grid_window_cells,
            ))
        } else {
            None
        };
        let mut bytes = Vec::new();
        let mut copies = Vec::new();
        let mut append = |destination: &wgpu::Buffer, data: &[u8]| {
            let offset = bytes.len() as u64;
            bytes.extend_from_slice(data);
            copies.push((destination.clone(), offset, data.len() as u64));
        };
        append(
            &self.uniform_buf,
            bytemuck::bytes_of(&self.step_uniforms(dt)),
        );
        append(
            &self.material_table,
            bytemuck::cast_slice(&self.step_material_table()),
        );
        if let Some((params, bounds)) = self.motion_guard.as_ref().and_then(|g| g.parameter_copy())
        {
            append(params, bytemuck::cast_slice(bounds));
        }
        if let Some(words) = active {
            append(&self.active_tiles.uniform, bytemuck::cast_slice(&words));
        }
        if let Some(words) = ordering {
            append(
                &self.particle_order.as_ref().unwrap().params,
                bytemuck::cast_slice(&words),
            );
        }
        let material_ids = self
            .particle_material_dirty
            .as_ref()
            .filter(|ids| !ids.is_empty())
            .map(|ids| {
                let offset = bytes.len() as u64;
                let data = bytemuck::cast_slice(ids);
                bytes.extend_from_slice(data);
                (offset, data.len() as u64)
            });
        if bytes.len() as u64 > self.device.limits().max_buffer_size {
            return Err(Error::Solver(
                "MPM parameter record exceeds device limits".into(),
            ));
        }
        let source = molla_core::gpu_create_buffer!(
            self.device,
            &wgpu::BufferDescriptor {
                label: Some("MPM immutable batch parameters"),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            }
        );
        source
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(&bytes);
        source.unmap();
        let jacobi_iters = match self.cfg.time_step {
            MpmTimeStep::Explicit => 0,
            MpmTimeStep::Implicit { jacobi_iters } => jacobi_iters,
        };
        Ok(PreparedMpmStep {
            epoch: self.step_epoch.clone(),
            source,
            copies,
            material_ids,
            substeps,
            jacobi_iters,
        })
    }

    /// Takes a caller encoder; use the returned batch to encode and submit all work in order.
    /// Dropping the batch cancels its commands and preserves pending reset/material initialization.
    pub fn begin_command_batch(&mut self, encoder: wgpu::CommandEncoder) -> MpmCommandBatch<'_> {
        let initial_reset = self.active_grid_needs_reset;
        let initial_quiescence_reset = self.quiescence_needs_reset();
        MpmCommandBatch {
            simulation: self,
            encoder: Some(encoder),
            initial_reset,
            initial_quiescence_reset,
            applied_material_ids: false,
            submitted: false,
        }
    }

    /// GPU feedback/raster views; bindings capture the current count and must be rebuilt after reseeding.
    pub fn tire_particle_buffers(&self) -> [&wgpu::Buffer; 3] {
        [&self.p_pos, &self.p_vel, &self.tire_compaction_buf]
    }
}

impl MpmCommandBatch<'_> {
    /// Records a full solver batch without host transfers, resource creation, submission or waiting.
    pub fn encode_step(&mut self, record: &PreparedMpmStep) -> Result<()> {
        if !Arc::ptr_eq(&record.epoch, &self.simulation.step_epoch) {
            return Err(Error::Solver(
                "MPM prepared step belongs to another simulation or stale configuration".into(),
            ));
        }
        if self.simulation.particle_count == 0 || record.substeps == 0 {
            return Ok(());
        }
        let encoder = self.encoder.as_mut().unwrap();
        for (destination, offset, bytes) in &record.copies {
            encoder.copy_buffer_to_buffer(&record.source, *offset, destination, 0, *bytes);
        }
        if self.simulation.particle_material_dirty.is_some() && !self.applied_material_ids {
            if let Some((offset, bytes)) = record.material_ids {
                encoder.copy_buffer_to_buffer(
                    &record.source,
                    offset,
                    &self.simulation.p_material,
                    0,
                    bytes,
                );
            }
            self.applied_material_ids = true;
        }
        self.simulation
            .encode_step_work(encoder, record.substeps, record.jacobi_iters, false);
        Ok(())
    }

    /// Records contact producers, feedback, exports and consumers between solver steps.
    /// Do not replace or separately submit this encoder; submit through the batch.
    pub fn encoder_mut(&mut self) -> &mut wgpu::CommandEncoder {
        self.simulation.invalidate_quiescence();
        self.encoder.as_mut().unwrap()
    }

    /// Submits the complete encoder and commits pending host initialization state.
    pub fn submit(mut self, queue: &wgpu::Queue) -> wgpu::SubmissionIndex {
        let submission = queue.submit([self.encoder.take().unwrap().finish()]);
        if self.applied_material_ids {
            self.simulation.particle_material_dirty = None;
        }
        self.submitted = true;
        submission
    }
}

impl Drop for MpmCommandBatch<'_> {
    fn drop(&mut self) {
        if !self.submitted {
            self.simulation.active_grid_needs_reset = self.initial_reset;
            if self.initial_quiescence_reset {
                self.simulation.invalidate_quiescence();
            }
        }
    }
}
