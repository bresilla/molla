//! Stable global soil identities in retained GPU storage.

use super::*;
use molla_vehicle::{SoilMaterialId, SoilParticleId};
use wgpu::util::DeviceExt;

/// Immutable request-time global identities in physical particle-slot/material-row
/// order. The CPU mapping mirrors read-only device IDs; it is not particle physics.
#[derive(Clone, Debug)]
pub struct MpmSoilIdentitySnapshot {
    epoch: Arc<()>,
    particles: Arc<[SoilParticleId]>,
    materials: Arc<[SoilMaterialId]>,
}

impl MpmSoilIdentitySnapshot {
    pub(super) fn from_selection(selection: &MpmSoilSelection) -> Self {
        Self {
            epoch: Arc::new(()),
            particles: selection.particle_ids().into(),
            materials: selection.material_ids().into(),
        }
    }
    pub fn particle_ids(&self) -> &[SoilParticleId] {
        &self.particles
    }
    pub fn material_ids(&self) -> &[SoilMaterialId] {
        &self.materials
    }
}

/// Read-only consumer views; each ID is two u32 words, low then high, with 8-byte
/// stride (WGSL array<vec2<u32>>). Retained consumers must validate their snapshot
/// epoch after restore/reseed/material-table edits. Do not modify these buffers.
/// Only the counted prefixes are live; unused capacity may contain older IDs.
pub struct MpmSoilIdentityBuffers<'a> {
    pub particles: &'a wgpu::Buffer,
    pub materials: &'a wgpu::Buffer,
    pub particle_count: u32,
    pub material_count: u32,
}

pub(super) struct SoilIdentity {
    particles: wgpu::Buffer,
    materials: wgpu::Buffer,
    snapshot: Option<MpmSoilIdentitySnapshot>,
}

impl MpmSimulation {
    pub fn soil_identity_snapshot(&self) -> Option<&MpmSoilIdentitySnapshot> {
        self.soil_identity.as_ref()?.snapshot.as_ref()
    }

    pub fn soil_identity_buffers(&self) -> Option<MpmSoilIdentityBuffers<'_>> {
        let identity = self.soil_identity.as_ref()?;
        let snapshot = identity.snapshot.as_ref()?;
        Some(MpmSoilIdentityBuffers {
            particles: &identity.particles,
            materials: &identity.materials,
            particle_count: snapshot.particles.len() as u32,
            material_count: snapshot.materials.len() as u32,
        })
    }

    /// Compare mapping epochs; device health and physical freshness are separate checks.
    pub fn is_soil_identity_current(&self, snapshot: &MpmSoilIdentitySnapshot) -> bool {
        self.soil_identity_snapshot()
            .is_some_and(|live| Arc::ptr_eq(&live.epoch, &snapshot.epoch))
    }

    /// Logical retained ID bytes; zero until the first identified restore.
    pub fn soil_identity_buffer_bytes(&self) -> u64 {
        self.soil_identity
            .as_ref()
            .map_or(0, |ids| ids.particles.size() + ids.materials.size())
    }

    pub(super) fn invalidate_soil_identity(&mut self) {
        if let Some(identity) = &mut self.soil_identity {
            identity.snapshot = None;
        }
    }

    /// Restore particle/history streams and global IDs in one GPU submission.
    /// The first call allocates two retained ID buffers; later calls reuse them.
    /// Existing physical/configuration/capacity/rigid restore checks still apply.
    /// Plain checkpoint restores, reseeds and material-table replacement invalidate
    /// identities; stepping and index-only particle ordering preserve them.
    /// Device loss is not rollback, and this call does not grant exclusive world ownership.
    pub fn restore_soil_selection(
        &mut self,
        engine: &Engine,
        selection: &MpmSoilSelection,
    ) -> Result<()> {
        self.restore_soil_selection_origin(engine, selection, false)
    }

    pub(super) fn restore_soil_selection_origin(
        &mut self,
        engine: &Engine,
        selection: &MpmSoilSelection,
        changing_origin: bool,
    ) -> Result<()> {
        engine.check_device()?;
        molla_kernels::validate_soil_device(engine.device().unwrap(), &self.device)?;
        let checkpoint = selection.checkpoint();
        self.validate_soil_restore_origin(checkpoint, changing_origin)?;
        let lengths = [
            u64::from(self.cfg.max_particles.max(1)) * 8,
            MAX_MPM_MATERIALS as u64 * 8,
        ];
        let limits = self.device.limits();
        if lengths.iter().any(|&bytes| {
            bytes > limits.max_buffer_size || bytes > limits.max_storage_buffer_binding_size
        }) {
            return Err(Error::Solver(
                "MPM soil identity buffers exceed device limits".into(),
            ));
        }
        let snapshot = MpmSoilIdentitySnapshot::from_selection(selection);
        let words: Vec<u32> = snapshot
            .particles
            .iter()
            .map(|id| id.0)
            .chain(snapshot.materials.iter().map(|id| id.0))
            .flat_map(|id| [id as u32, (id >> 32) as u32])
            .collect();
        if words.len() as u64 * 4 > limits.max_buffer_size {
            return Err(Error::Solver(
                "MPM soil identity upload exceeds device limits".into(),
            ));
        }
        if self.soil_identity.is_none() {
            let allocate = |label, size| {
                molla_core::gpu_create_buffer!(
                    self.device,
                    &wgpu::BufferDescriptor {
                        label: Some(label),
                        size,
                        usage: wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }
                )
            };
            self.soil_identity = Some(SoilIdentity {
                particles: allocate("MPM global particle IDs", lengths[0]),
                materials: allocate("MPM global material IDs", lengths[1]),
                snapshot: None,
            });
        }
        let source = molla_core::gpu_buffer_init!(
            self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("MPM identified soil restore metadata"),
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::COPY_SRC,
            }
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MPM identified soil restore"),
            });
        self.encode_soil_restore(&mut encoder, checkpoint);
        let identity = self.soil_identity.as_ref().unwrap();
        let particle_bytes = snapshot.particles.len() as u64 * 8;
        if particle_bytes != 0 {
            encoder.copy_buffer_to_buffer(&source, 0, &identity.particles, 0, particle_bytes);
        }
        encoder.copy_buffer_to_buffer(
            &source,
            particle_bytes,
            &identity.materials,
            0,
            snapshot.materials.len() as u64 * 8,
        );
        engine.queue().unwrap().submit([encoder.finish()]);
        self.cfg.grid_origin = checkpoint.config().grid_origin;
        self.finish_identified_soil_restore(checkpoint, snapshot);
        Ok(())
    }

    pub(super) fn finish_identified_soil_restore(
        &mut self,
        checkpoint: &MpmSoilCheckpoint,
        snapshot: MpmSoilIdentitySnapshot,
    ) {
        self.finish_soil_restore(checkpoint);
        self.soil_identity.as_mut().unwrap().snapshot = Some(snapshot);
    }
}

#[cfg(test)]
#[path = "mpm_soil_identity_tests.rs"]
mod tests;
