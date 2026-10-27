//! Lossless soil-particle checkpoints at submitted MPM step boundaries.

use super::*;
use wgpu::util::DeviceExt;

#[path = "mpm_soil_activity.rs"]
mod activity;
pub use activity::{MpmSoilActivity, MpmSoilActivityReport, MpmSoilTileActivity};

#[path = "mpm_soil_momenta.rs"]
mod momenta;
pub use momenta::MpmSoilMomenta;

#[path = "mpm_soil_energy.rs"]
mod energy;
pub use energy::MpmSoilEnergy;

#[path = "mpm_soil_recenter.rs"]
mod recenter;

#[path = "mpm_soil_lattice_window.rs"]
mod lattice_window;

#[path = "mpm_soil_frame.rs"]
mod frame;
pub use frame::{MpmSoilFrame, MpmSoilFrameCheckpoint};

#[path = "mpm_soil_tiles.rs"]
mod tiles;
pub use tiles::{
    MpmSoilCanonicalSelection, MpmSoilCanonicalStore, MpmSoilCanonicalStoreRead,
    MpmSoilCanonicalStoreReadCompletion, MpmSoilCanonicalStoreReadTask,
    MpmSoilCanonicalStoreWritebackCompletion, MpmSoilCanonicalStoreWritebackTask,
    MpmSoilCanonicalTileArchive, MpmSoilFrameSelection, MpmSoilFrameStore, MpmSoilFrameStoreRead,
    MpmSoilFrameStoreReadCompletion, MpmSoilFrameStoreReadTask,
    MpmSoilFrameStoreWritebackCompletion, MpmSoilFrameStoreWritebackTask, MpmSoilFrameTileArchive,
    MpmSoilSelection, MpmSoilStore, MpmSoilStoreLimits, MpmSoilStoreRead,
    MpmSoilStoreReadCompletion, MpmSoilStoreReadLimits, MpmSoilStoreReadStats,
    MpmSoilStoreReadTask, MpmSoilStoreStats, MpmSoilStoreStorageStats,
    MpmSoilStoreWritebackCompletion, MpmSoilStoreWritebackTask, MpmSoilTileArchive,
};

#[path = "mpm_soil_world.rs"]
mod world;
pub use world::{
    MpmSoilDiskWorld, MpmSoilDiskWorldStatus, MpmSoilPinConfig, MpmSoilRegionEnvelope,
    MpmSoilSettling, MpmSoilSettlingConfig, MpmSoilSettlingLimits, MpmSoilSettlingProgress,
    MpmSoilSettlingStatus, MpmSoilTransferPath, MpmSoilTransferStats, MpmSoilWorld,
    MpmSoilWorldStatus,
};

#[path = "mpm_checkpoint_async.rs"]
mod asynchronous;
pub use asynchronous::{
    MpmSoilCheckpointCompletion, MpmSoilCheckpointReadback, MpmSoilCheckpointTicket,
    MpmSoilReadbackStats,
};

#[cfg(test)]
#[path = "mpm_checkpoint_tests.rs"]
pub(super) mod tests;

#[cfg(test)]
#[path = "mpm_lattice_checkpoint_tests.rs"]
mod lattice_tests;

#[cfg(test)]
#[path = "mpm_lattice_window_tests.rs"]
mod lattice_window_tests;

const MAGIC: &[u8; 8] = b"MOLLSOIL";
const VERSION: u32 = 1;
const LATTICE_VERSION: u32 = 2;
const HEADER_BYTES: usize = 44;
const STRIDES: [usize; 9] = [4, 4, 12, 12, 1, 1, 1, 1, 4];
const WORDS_PER_PARTICLE: usize = 40;

fn max_encoded_bytes(max_particles: u32) -> u64 {
    HEADER_BYTES as u64
        + (23 + 4 * MAX_MPM_MATERIALS as u64 + u64::from(max_particles) * WORDS_PER_PARTICLE as u64)
            * 4
}

fn checkpoint_digest(version: u32, payload: &[u8]) -> blake3::Hash {
    let mut hash = blake3::Hasher::new();
    if version == LATTICE_VERSION {
        hash.update(&version.to_le_bytes());
    }
    hash.update(payload);
    hash.finalize()
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("MPM soil checkpoint: {message}"))
}

struct CheckpointMetadata {
    cfg: MpmConfig,
    grid_window_cells: [i32; 3],
    materials: Vec<MpmRheology>,
    count: u32,
    dirty: Option<Vec<u32>>,
    identity: Option<MpmSoilIdentitySnapshot>,
}

impl CheckpointMetadata {
    fn bytes(&self) -> u64 {
        u64::from(self.count) * WORDS_PER_PARTICLE as u64 * 4 + 16
    }

    fn finish(self, raw: &[u32]) -> Result<MpmSoilCheckpoint> {
        let count = self.count as usize;
        if raw.len() != count * WORDS_PER_PARTICLE + 4 || raw[count * WORDS_PER_PARTICLE + 2] != 0 {
            return Err(invalid(
                "invalid capture length or latched grid fault; incomplete state cannot be archived",
            ));
        }
        let mut words = raw[..count * WORDS_PER_PARTICLE].to_vec();
        if let Some(ids) = self.dirty {
            let start = STRIDES[..7].iter().sum::<usize>() * count;
            words[start..start + count].copy_from_slice(&ids);
        }
        let checkpoint = MpmSoilCheckpoint {
            cfg: self.cfg,
            grid_window_cells: self.grid_window_cells,
            materials: self.materials,
            count: self.count,
            words,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }
}

/// Lossless volumetric particle state, not a heightfield or whole-world save.
/// Stores position, velocity, APIC C, elastic F, reference volume, mass, log-Jp,
/// material IDs/table and tire compaction/impulse diagnostics in slot order.
/// Grid fields, rigid bodies/wrenches, clocks, world-tile IDs and scheduling
/// caches are excluded. Material IDs are indices into this checkpoint's table.
#[derive(Clone, Debug)]
pub struct MpmSoilCheckpoint {
    cfg: MpmConfig,
    grid_window_cells: [i32; 3],
    materials: Vec<MpmRheology>,
    count: u32,
    words: Vec<u32>,
}

impl MpmSoilCheckpoint {
    /// Integer window displacement from the lattice anchor in config().grid_origin.
    /// Encoding and live solver admission have separate validation.
    pub fn grid_window_cells(&self) -> [i32; 3] {
        self.grid_window_cells
    }

    fn require_origin_only_grid(&self) -> Result<()> {
        if self.grid_window_cells != [0; 3] {
            return Err(invalid(
                "integer-window checkpoint requires lattice-aware runtime or storage",
            ));
        }
        Ok(())
    }

    /// Original physical configuration and allocation capacities. Decoding does
    /// not allocate a solver or validate these capacities against a GPU device.
    /// grid_origin is the lattice anchor; grid_window_cells supplies its window offset.
    pub fn config(&self) -> MpmConfig {
        self.cfg
    }

    pub fn particle_count(&self) -> u32 {
        self.count
    }

    /// Empty means the configuration's single rheology is authoritative.
    pub fn materials(&self) -> &[MpmRheology] {
        &self.materials
    }

    pub fn particle_bytes(&self) -> usize {
        self.words.len() * 4
    }

    /// Versioned little-endian encoding with a BLAKE3 payload digest.
    /// The caller owns archive I/O, durable replacement and access control.
    pub fn to_bytes(&self) -> Vec<u8> {
        let version = if self.grid_window_cells == [0; 3] {
            VERSION
        } else {
            LATTICE_VERSION
        };
        let mut words = Vec::new();
        if version == LATTICE_VERSION {
            words.extend(self.grid_window_cells.map(|v| v as u32));
        }
        words.extend(config_words(self.cfg));
        words.push(self.materials.len() as u32);
        for &material in &self.materials {
            words.extend(rheology_words(material));
        }
        words.push(self.count);
        words.extend_from_slice(&self.words);
        let mut bytes = Vec::with_capacity(HEADER_BYTES + words.len() * 4);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&[0; 32]);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let digest = checkpoint_digest(version, &bytes[HEADER_BYTES..]);
        bytes[12..HEADER_BYTES].copy_from_slice(digest.as_bytes());
        bytes
    }

    /// Decode under an explicit particle-payload allocation limit; reject corrupt,
    /// non-finite, unsupported, oversized and trailing data before GPU work.
    pub fn from_bytes(bytes: &[u8], max_particles: u32) -> Result<Self> {
        if bytes.len() < HEADER_BYTES {
            return Err(invalid("truncated checkpoint header"));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if ![VERSION, LATTICE_VERSION].contains(&version)
            || bytes.len() as u64
                > max_encoded_bytes(max_particles) + if version == LATTICE_VERSION { 12 } else { 0 }
            || &bytes[..8] != MAGIC
            || !(bytes.len() - HEADER_BYTES).is_multiple_of(4)
            || bytes[12..HEADER_BYTES]
                != *checkpoint_digest(version, &bytes[HEADER_BYTES..]).as_bytes()
        {
            return Err(invalid("invalid format, digest or allocation limit"));
        }
        let words: Vec<u32> = bytes[HEADER_BYTES..]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut reader = Words(&words);
        let grid_window_cells = if version == LATTICE_VERSION {
            let value = [
                reader.next()? as i32,
                reader.next()? as i32,
                reader.next()? as i32,
            ];
            if value == [0; 3] {
                return Err(invalid("zero-offset checkpoints use version 1"));
            }
            value
        } else {
            [0; 3]
        };
        let cfg = reader.config()?;
        let rows = reader.next()? as usize;
        if rows > MAX_MPM_MATERIALS {
            return Err(invalid("material table exceeds capacity"));
        }
        let materials = (0..rows)
            .map(|_| reader.rheology())
            .collect::<Result<Vec<_>>>()?;
        let count = reader.next()?;
        if count > max_particles
            || reader.0.len() as u64 != u64::from(count) * WORDS_PER_PARTICLE as u64
        {
            return Err(invalid("particle count or payload length mismatch"));
        }
        let checkpoint = Self {
            cfg,
            grid_window_cells,
            materials,
            count,
            words: reader.0.to_vec(),
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    fn stream(&self, stream: usize) -> &[u32] {
        let offset = STRIDES[..stream].iter().sum::<usize>() * self.count as usize;
        &self.words[offset..offset + STRIDES[stream] * self.count as usize]
    }

    fn validate(&self) -> Result<()> {
        let c = self.cfg;
        if c.grid_size.checked_pow(3).filter(|&n| n > 0).is_none()
            || c.max_particles == 0
            || c.max_rigid == 0
            || self.count > c.max_particles
            || !c.cell_size.is_finite()
            || c.cell_size <= 0.0
            || c.grid_origin.iter().any(|v| !v.is_finite())
            || !c.gravity_y.is_finite()
            || !c.youngs.is_finite()
            || c.youngs < 0.0
            || !c.poisson.is_finite()
            || c.poisson <= -1.0
            || c.poisson >= 0.5
            || !c.friction_angle.is_finite()
            || !c.cohesion.is_finite()
            || c.cohesion < 0.0
            || !c.boundary_friction.is_finite()
            || !(0.0..=1.0).contains(&c.boundary_friction)
            || !c.particle_mass.is_finite()
            || c.particle_mass <= 0.0
            || !c.particle_volume.is_finite()
            || c.particle_volume <= 0.0
            || self.materials.len() > MAX_MPM_MATERIALS
            || self.words.len() as u64 != u64::from(self.count) * WORDS_PER_PARTICLE as u64
        {
            return Err(invalid("invalid configuration or state length"));
        }
        validate_rheology(c.rheology)?;
        for &material in &self.materials {
            validate_rheology(material)?;
        }
        for stream in 0..STRIDES.len() {
            if stream == 7 {
                let rows = self.materials.len().max(1) as u32;
                if self.stream(stream).iter().any(|&id| id >= rows) {
                    return Err(invalid("material ID outside authoritative table"));
                }
            } else if self.stream(stream).iter().any(|&w| {
                let v = f32::from_bits(w);
                !v.is_finite() || ((stream == 4 || stream == 5) && v <= 0.0)
            }) {
                return Err(invalid(
                    "non-finite history or nonpositive mass/reference volume",
                ));
            }
        }
        Ok(())
    }
}

impl MpmSimulation {
    fn checkpoint_metadata(&self) -> Result<CheckpointMetadata> {
        if self
            .particle_material_dirty
            .as_ref()
            .is_some_and(|ids| ids.len() != self.particle_count as usize)
        {
            return Err(invalid("pending material IDs do not match particle count"));
        }
        Ok(CheckpointMetadata {
            cfg: self.cfg,
            grid_window_cells: self.grid_window_cells,
            materials: self.material_rows.clone(),
            count: self.particle_count,
            dirty: self.particle_material_dirty.clone(),
            identity: self.soil_identity_snapshot().cloned(),
        })
    }

    fn encode_checkpoint_copy(&self, encoder: &mut wgpu::CommandEncoder, stage: &wgpu::Buffer) {
        let mut offset = 0;
        if self.particle_count != 0 {
            for (buffer, stride) in self.checkpoint_buffers().into_iter().zip(STRIDES) {
                let size = u64::from(self.particle_count) * stride as u64 * 4;
                encoder.copy_buffer_to_buffer(buffer, 0, stage, offset, size);
                offset += size;
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.active_tiles.list,
            self.active_tiles.metadata_offset,
            stage,
            offset,
            16,
        );
    }

    pub(super) fn checkpoint_buffers(&self) -> [&wgpu::Buffer; 9] {
        [
            &self.p_pos,
            &self.p_vel,
            &self.p_c,
            &self.p_f,
            &self.p_volume,
            &self.p_mass,
            &self.p_jp,
            &self.p_material,
            &self.tire_compaction_buf,
        ]
    }

    /// Explicit blocking archive operation; never call on the physics hot path.
    /// The queue must belong to this simulation's device. Pending material edits
    /// are captured without flushing or consuming them. Active-grid overflow
    /// rejects the save; a reset request does not hide an existing fault latch.
    pub fn checkpoint_soil_with(&mut self, queue: &wgpu::Queue) -> Result<MpmSoilCheckpoint> {
        let metadata = self.checkpoint_metadata()?;
        let bytes = metadata.bytes();
        if bytes > self.device.limits().max_buffer_size {
            return Err(invalid("archive staging exceeds device capacity"));
        }
        let stage = molla_core::gpu_create_buffer!(
            self.device,
            &wgpu::BufferDescriptor {
                label: Some("MPM soil checkpoint readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MPM soil checkpoint capture"),
            });
        self.encode_checkpoint_copy(&mut encoder, &stage);
        queue.submit([encoder.finish()]);
        let slice = stage.slice(..);
        let (send, recv) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("MPM checkpoint poll: {e}")))?;
        recv.recv()
            .map_err(|e| Error::Gpu(format!("MPM checkpoint callback: {e}")))?
            .map_err(|e| Error::Gpu(format!("MPM checkpoint map: {e}")))?;
        let data = molla_core::gpu_readback!(slice);
        let result = metadata.finish(bytemuck::cast_slice(&data));
        drop(data);
        stage.unmap();
        result
    }

    /// Restore into existing buffers with one submitted copy transaction.
    /// Requires matching physical configuration, sufficient particle/order capacity
    /// and no attached rigid bodies. Buffer identities remain stable; prepared step
    /// records are invalidated. The next step rebuilds grid scratch and its epoch.
    /// Installed global soil identities are invalidated by this particle-only restore.
    /// Rebuild external count-dependent consumers and reset observation generations
    /// after restoring; previously exported terrain snapshots are not refreshed.
    /// The queue must belong to this simulation's device. Device loss is not rollback.
    pub fn restore_soil_with(
        &mut self,
        queue: &wgpu::Queue,
        checkpoint: &MpmSoilCheckpoint,
    ) -> Result<()> {
        self.validate_soil_restore(checkpoint)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MPM soil checkpoint restore"),
            });
        self.encode_soil_restore(&mut encoder, checkpoint);
        queue.submit([encoder.finish()]);
        self.finish_soil_restore(checkpoint);
        Ok(())
    }

    pub(super) fn validate_soil_restore(&self, checkpoint: &MpmSoilCheckpoint) -> Result<()> {
        self.validate_soil_restore_origin(checkpoint, false)
    }

    pub(super) fn validate_soil_restore_origin(
        &self,
        checkpoint: &MpmSoilCheckpoint,
        changing_origin: bool,
    ) -> Result<()> {
        checkpoint.validate()?;
        if checkpoint.grid_window_cells != [0; 3] {
            checkpoint.validate_lattice_window()?;
        }
        let mut compatible = checkpoint.cfg;
        compatible.max_particles = self.cfg.max_particles;
        compatible.max_rigid = self.cfg.max_rigid;
        if changing_origin {
            compatible.grid_origin = self.cfg.grid_origin;
        }
        if config_words(compatible) != config_words(self.cfg)
            || self.rigid_count != 0
            || checkpoint.count > self.cfg.max_particles
            || (self.use_particle_order
                && self
                    .particle_order
                    .as_ref()
                    .is_none_or(|order| checkpoint.count > order.capacity))
        {
            return Err(invalid(
                "incompatible configuration, capacity or attached rigid bodies",
            ));
        }
        let size = checkpoint.particle_bytes() as u64;
        if size > self.device.limits().max_buffer_size {
            return Err(invalid("restore upload exceeds device capacity"));
        }
        Ok(())
    }

    pub(super) fn encode_soil_restore(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        checkpoint: &MpmSoilCheckpoint,
    ) {
        let size = checkpoint.particle_bytes() as u64;
        if size != 0 {
            let source = molla_core::gpu_buffer_init!(
                self.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("MPM soil checkpoint upload"),
                    contents: bytemuck::cast_slice(&checkpoint.words),
                    usage: wgpu::BufferUsages::COPY_SRC,
                }
            );
            let mut offset = 0;
            for (buffer, stride) in self.checkpoint_buffers().into_iter().zip(STRIDES) {
                let bytes = u64::from(checkpoint.count) * stride as u64 * 4;
                encoder.copy_buffer_to_buffer(&source, offset, buffer, 0, bytes);
                offset += bytes;
            }
        }
        encoder.clear_buffer(
            &self.active_tiles.list,
            self.active_tiles.metadata_offset,
            Some(16),
        );
    }

    pub(super) fn finish_soil_restore(&mut self, checkpoint: &MpmSoilCheckpoint) {
        self.invalidate_soil_identity();
        self.grid_window_cells = checkpoint.grid_window_cells;
        self.material_rows = checkpoint.materials.clone();
        self.particle_material_dirty = None;
        self.particle_count = checkpoint.count;
        self.active_grid_needs_reset = true;
        self.invalidate_prepared_steps();
        let vectors = |stream| {
            checkpoint
                .stream(stream)
                .chunks_exact(4)
                .map(|v| std::array::from_fn(|i| f32::from_bits(v[i])))
                .collect()
        };
        self.host_pos = vectors(0);
        self.host_vel = vectors(1);
        self.tire_compaction = vectors(8);
        self.tire_raster_cache = None;
        self.tire_feedback_cache = None;
    }
}

fn rheology_words(r: MpmRheology) -> [u32; 4] {
    match r {
        MpmRheology::Sand { friction_angle } => [0, friction_angle.to_bits(), 0, 0],
        MpmRheology::Snow {
            theta_c,
            theta_s,
            hardening,
        } => [1, theta_c.to_bits(), theta_s.to_bits(), hardening.to_bits()],
        MpmRheology::Viscous {
            velocity_damping,
            affine_damping,
        } => [2, velocity_damping.to_bits(), affine_damping.to_bits(), 0],
    }
}

fn validate_rheology(r: MpmRheology) -> Result<()> {
    let valid = match r {
        MpmRheology::Sand { friction_angle } => {
            friction_angle.is_finite()
                && (0.0..std::f32::consts::FRAC_PI_2).contains(&friction_angle)
        }
        MpmRheology::Snow {
            theta_c,
            theta_s,
            hardening,
        } => {
            theta_c.is_finite()
                && (0.0..1.0).contains(&theta_c)
                && theta_s.is_finite()
                && theta_s >= 0.0
                && hardening.is_finite()
                && hardening >= 0.0
        }
        MpmRheology::Viscous {
            velocity_damping,
            affine_damping,
        } => {
            velocity_damping.is_finite()
                && velocity_damping >= 0.0
                && affine_damping.is_finite()
                && affine_damping >= 0.0
        }
    };
    if valid {
        Ok(())
    } else {
        Err(invalid("invalid material parameters"))
    }
}

fn config_words(c: MpmConfig) -> Vec<u32> {
    let mut words = vec![c.grid_size, c.max_particles, c.max_rigid];
    words.extend(
        [
            c.cell_size,
            c.grid_origin[0],
            c.grid_origin[1],
            c.grid_origin[2],
            c.gravity_y,
            c.youngs,
            c.poisson,
            c.friction_angle,
            c.cohesion,
            c.boundary_friction,
            c.particle_mass,
            c.particle_volume,
        ]
        .map(f32::to_bits),
    );
    words.extend(rheology_words(c.rheology));
    words.extend(match c.time_step {
        MpmTimeStep::Explicit => [0, 0],
        MpmTimeStep::Implicit { jacobi_iters } => [1, jacobi_iters],
    });
    words
}

struct Words<'a>(&'a [u32]);
impl Words<'_> {
    fn next(&mut self) -> Result<u32> {
        let (&head, tail) = self
            .0
            .split_first()
            .ok_or_else(|| invalid("truncated metadata"))?;
        self.0 = tail;
        Ok(head)
    }

    fn float(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.next()?))
    }

    fn rheology(&mut self) -> Result<MpmRheology> {
        let kind = self.next()?;
        let a = self.float()?;
        let b = self.float()?;
        let c = self.float()?;
        match kind {
            0 if b.to_bits() == 0 && c.to_bits() == 0 => {
                Ok(MpmRheology::Sand { friction_angle: a })
            }
            1 => Ok(MpmRheology::Snow {
                theta_c: a,
                theta_s: b,
                hardening: c,
            }),
            2 if c.to_bits() == 0 => Ok(MpmRheology::Viscous {
                velocity_damping: a,
                affine_damping: b,
            }),
            _ => Err(invalid("unsupported or noncanonical rheology encoding")),
        }
    }

    fn config(&mut self) -> Result<MpmConfig> {
        let mut cfg = MpmConfig {
            grid_size: self.next()?,
            max_particles: self.next()?,
            max_rigid: self.next()?,
            cell_size: self.float()?,
            grid_origin: [self.float()?, self.float()?, self.float()?],
            gravity_y: self.float()?,
            youngs: self.float()?,
            poisson: self.float()?,
            friction_angle: self.float()?,
            cohesion: self.float()?,
            boundary_friction: self.float()?,
            particle_mass: self.float()?,
            particle_volume: self.float()?,
            rheology: self.rheology()?,
            time_step: MpmTimeStep::Explicit,
        };
        let kind = self.next()?;
        let jacobi_iters = self.next()?;
        cfg.time_step = match (kind, jacobi_iters) {
            (0, 0) => MpmTimeStep::Explicit,
            (1, _) => MpmTimeStep::Implicit { jacobi_iters },
            _ => return Err(invalid("unsupported time step encoding")),
        };
        Ok(cfg)
    }
}
