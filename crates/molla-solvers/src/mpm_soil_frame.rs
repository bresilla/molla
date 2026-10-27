//! Owned detached-soil coordinate frames and versioned framed checkpoints.

use super::*;

const FRAME_MAGIC: &[u8; 8] = b"MOLLFRAM";
const FRAME_HEADER: usize = HEADER_BYTES + 24;

fn validate_origin(origin: [f64; 3]) -> Result<()> {
    if origin.iter().any(|value| !value.is_finite()) {
        return Err(invalid("nonfinite soil world-frame origin"));
    }
    Ok(())
}

pub(super) fn translated_origin(
    origin: [f64; 3],
    before: [f32; 3],
    after: [f32; 3],
) -> Result<[f64; 3]> {
    let mut result = origin;
    for axis in 0..3 {
        let delta = recenter::exact_sum(f64::from(before[axis]), -f64::from(after[axis]))?;
        if delta != 0.0 {
            result[axis] = recenter::exact_sum(origin[axis], delta)?;
        }
    }
    Ok(result)
}

/// Local particle/grid state and the f64 world offset added to local positions.
/// Keep the offset separate during arithmetic; summing it into absolute f32
/// positions discards precision. Tile IDs, machines and physical clocks are absent.
#[derive(Clone, Debug)]
pub struct MpmSoilFrameCheckpoint {
    pub(super) origin: [f64; 3],
    pub(super) soil: MpmSoilCheckpoint,
}

impl MpmSoilFrameCheckpoint {
    pub fn new(soil: MpmSoilCheckpoint, world_origin: [f64; 3]) -> Result<Self> {
        validate_origin(world_origin)?;
        soil.validate()?;
        Ok(Self {
            origin: world_origin,
            soil,
        })
    }

    pub fn world_origin(&self) -> [f64; 3] {
        self.origin
    }

    pub fn soil(&self) -> &MpmSoilCheckpoint {
        &self.soil
    }

    /// World-tile membership in particle-slot order using split-coordinate
    /// arithmetic. This does not change archive identity or live residency.
    pub fn particle_tiles(
        &self,
        layout: molla_vehicle::SoilTileLayout,
    ) -> Result<Vec<molla_vehicle::SoilTileCoord>> {
        self.soil
            .stream(0)
            .chunks_exact(4)
            .map(|particle| {
                let local = std::array::from_fn(|axis| f64::from(f32::from_bits(particle[axis])));
                layout
                    .tile_at_local(local, self.origin)
                    .map_err(|error| invalid(&error.to_string()))
            })
            .collect()
    }

    /// Evaluate momentum about a world-space point using exact frame subtraction.
    /// The returned origin is world-space; translation does not rotate vectors.
    pub fn momenta_about_world(&self, origin: [f64; 3]) -> Result<MpmSoilMomenta> {
        let mut local = [0.0; 3];
        for axis in 0..3 {
            local[axis] = recenter::exact_sum(origin[axis], -self.origin[axis])?;
        }
        let mut result = self.soil.momenta_about(local)?;
        result.origin = origin;
        Ok(result)
    }

    /// Evaluate energy relative to a world-space height without forming absolute
    /// particle positions. Unsupported elastic states remain explicitly unavailable.
    pub fn energy_at_world_height(&self, height: f64) -> Result<MpmSoilEnergy> {
        let local = recenter::exact_sum(height, -self.origin[1])?;
        let mut result = self.soil.energy_at_height(local)?;
        result.reference_height = height;
        Ok(result)
    }

    /// Translate local coordinates and compensate the world offset exactly.
    /// Rejects any coordinate or world-offset rounding before returning a snapshot.
    pub fn recentered(&self, grid_origin: [f32; 3]) -> Result<Self> {
        let origin = translated_origin(self.origin, self.soil.cfg.grid_origin, grid_origin)?;
        Self::new(self.soil.recentered(grid_origin)?, origin)
    }

    /// Versioned frame envelope with a digest covering offset and local checkpoint.
    /// Filesystem durability and replacement are caller-owned.
    pub fn to_bytes(&self) -> Vec<u8> {
        let soil = self.soil.to_bytes();
        let mut bytes = Vec::with_capacity(FRAME_HEADER + soil.len());
        bytes.extend_from_slice(FRAME_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&[0; 32]);
        for value in self.origin {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&soil);
        let digest = blake3::hash(&bytes[HEADER_BYTES..]);
        bytes[12..HEADER_BYTES].copy_from_slice(digest.as_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8], max_particles: u32) -> Result<Self> {
        if bytes.len() < FRAME_HEADER
            || bytes.len() as u64 > FRAME_HEADER as u64 + max_encoded_bytes(max_particles) + 12
            || &bytes[..8] != FRAME_MAGIC
            || bytes[8..12] != 1u32.to_le_bytes()
            || bytes[12..HEADER_BYTES] != *blake3::hash(&bytes[HEADER_BYTES..]).as_bytes()
        {
            return Err(invalid(
                "invalid soil frame format, digest or allocation limit",
            ));
        }
        let origin = std::array::from_fn(|axis| {
            let start = HEADER_BYTES + axis * 8;
            f64::from_le_bytes(bytes[start..start + 8].try_into().unwrap())
        });
        validate_origin(origin)?;
        Self::new(
            MpmSoilCheckpoint::from_bytes(&bytes[FRAME_HEADER..], max_particles)?,
            origin,
        )
    }
}

/// Exclusive detached solver and world-frame offset. No mutable solver escape is
/// exposed. Previously retained raw GPU handles must not mutate owned state.
/// Checkpoint/recenter are blocking boundary operations, not per-step work.
/// Identified restores retain global IDs; rigid coupling, residency leases,
/// streaming and automatic window motion are absent.
pub struct MpmSoilFrame {
    simulation: MpmSimulation,
    origin: [f64; 3],
}

impl MpmSoilFrame {
    pub fn new(simulation: MpmSimulation, world_origin: [f64; 3]) -> Result<Self> {
        validate_origin(world_origin)?;
        if simulation.rigid_count != 0
            || simulation.soil_identity_snapshot().is_some()
            || simulation.motion_guard.is_some()
        {
            return Err(invalid(
                "soil frame requires detached, unowned, unguarded soil",
            ));
        }
        Ok(Self {
            simulation,
            origin: world_origin,
        })
    }

    pub fn world_origin(&self) -> [f64; 3] {
        self.origin
    }

    pub fn simulation(&self) -> &MpmSimulation {
        &self.simulation
    }

    pub fn into_parts(self) -> (MpmSimulation, [f64; 3]) {
        (self.simulation, self.origin)
    }

    pub fn step_n_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
        substeps: usize,
    ) -> Result<()> {
        self.simulation.step_n_with(device, queue, dt, substeps)
    }

    pub fn checkpoint_with(&mut self, queue: &wgpu::Queue) -> Result<MpmSoilFrameCheckpoint> {
        MpmSoilFrameCheckpoint::new(self.simulation.checkpoint_soil_with(queue)?, self.origin)
    }

    /// Restore local physical/history state and global IDs together, then publish
    /// its world offset. The engine must match this frame's device. Changed grid
    /// origins are allowed; other physical settings and capacity checks still apply.
    /// First use allocates retained ID buffers; device loss is not rollback.
    pub fn restore_selection(
        &mut self,
        engine: &Engine,
        selection: &MpmSoilFrameSelection,
    ) -> Result<()> {
        validate_origin(selection.world_origin())?;
        self.simulation
            .restore_soil_selection_origin(engine, selection.local_selection(), true)?;
        self.origin = selection.world_origin();
        Ok(())
    }

    /// Blocking identified capture and host partition at the submitted boundary.
    /// Requires installed IDs; neither residency ownership nor durable I/O is
    /// granted. Queue/device pairing and external raw-handle writes are caller contracts.
    pub fn checkpoint_tiles_with(
        &mut self,
        queue: &wgpu::Queue,
        layout: molla_vehicle::SoilTileLayout,
    ) -> Result<MpmSoilFrameTileArchive> {
        let identity = self
            .simulation
            .soil_identity_snapshot()
            .cloned()
            .ok_or_else(|| invalid("framed tile capture requires installed soil identities"))?;
        self.checkpoint_with(queue)?.partition_tiles(
            layout,
            identity.particle_ids(),
            identity.material_ids(),
        )
    }

    /// Prevalidates world-offset precision before capturing or changing local state.
    /// Installed IDs, their buffers and mapping epoch remain unchanged.
    /// The queue must match this device; device loss is not transactional rollback.
    pub fn recenter_with(&mut self, queue: &wgpu::Queue, grid_origin: [f32; 3]) -> Result<()> {
        let origin = translated_origin(self.origin, self.simulation.cfg.grid_origin, grid_origin)?;
        if let Some(identity) = self.simulation.soil_identity_snapshot().cloned() {
            let current = self.simulation.checkpoint_soil_with(queue)?;
            let shifted = current.recentered(grid_origin)?;
            self.simulation
                .validate_soil_restore_origin(&shifted, true)?;
            if self.simulation.cfg.grid_origin.map(f32::to_bits) != grid_origin.map(f32::to_bits) {
                let mut encoder = self.simulation.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("MPM identified frame recenter"),
                    },
                );
                self.simulation.encode_soil_restore(&mut encoder, &shifted);
                queue.submit([encoder.finish()]);
                self.simulation.cfg.grid_origin = grid_origin;
                self.simulation
                    .finish_identified_soil_restore(&shifted, identity);
            }
        } else {
            self.simulation.recenter_soil_with(queue, grid_origin)?;
        }
        self.origin = origin;
        Ok(())
    }

    /// Restore framed soil into retained buffers, allowing a different local grid
    /// origin but no changed physical settings. Validation precedes upload; success
    /// replaces the world offset and invalidates prepared work/observation caches.
    /// This particle-only restore invalidates installed global identities.
    /// The queue must match this device; device loss is not transactional rollback.
    pub fn restore_with(
        &mut self,
        queue: &wgpu::Queue,
        checkpoint: &MpmSoilFrameCheckpoint,
    ) -> Result<()> {
        validate_origin(checkpoint.origin)?;
        self.simulation
            .validate_soil_restore_origin(&checkpoint.soil, true)?;
        let mut encoder =
            self.simulation
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("MPM framed soil restore"),
                });
        self.simulation
            .encode_soil_restore(&mut encoder, &checkpoint.soil);
        queue.submit([encoder.finish()]);
        self.simulation.cfg.grid_origin = checkpoint.soil.cfg.grid_origin;
        self.simulation.finish_soil_restore(&checkpoint.soil);
        self.origin = checkpoint.origin;
        Ok(())
    }
}

#[cfg(test)]
#[path = "mpm_soil_frame_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "mpm_soil_frame_identity_tests.rs"]
mod identity_tests;
