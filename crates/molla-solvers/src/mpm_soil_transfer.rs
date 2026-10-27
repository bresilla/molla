//! Retained device gather for identified world-boundary particle transfers.

use super::*;

const RECORD_WORDS: u64 = WORDS_PER_PARTICLE as u64 + 2;
const IMPORT: u32 = 1 << 31;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MpmSoilTransferPath {
    #[default]
    /// No particle payload is written; solver metadata may still change.
    Unchanged,
    FullUpload,
    DeviceGather,
}

/// Membership counts are global-ID overlap, independent of upload strategy.
/// Logical upload bytes exclude driver overhead and the preceding archive capture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MpmSoilTransferStats {
    pub path: MpmSoilTransferPath,
    pub retained_particles: u32,
    pub imported_particles: u32,
    pub removed_particles: u32,
    pub upload_bytes: u64,
    pub dispatches: u32,
}

pub(super) struct SoilTransfer {
    source: Arc<()>,
    previous: wgpu::Buffer,
    upload: wgpu::Buffer,
    output: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

struct TransferUpload {
    words: Vec<u32>,
    stats: MpmSoilTransferStats,
}

fn transfer_upload(
    previous: &MpmSoilIdentitySnapshot,
    selection: &MpmSoilSelection,
) -> Result<TransferUpload> {
    let old = previous.particle_ids();
    let new = selection.particle_ids();
    if old.len() >= IMPORT as usize
        || new.len() >= IMPORT as usize
        || old.windows(2).any(|p| p[0] >= p[1])
        || new.windows(2).any(|p| p[0] >= p[1])
        || previous.material_ids() != selection.material_ids()
    {
        return Err(invalid(
            "transfer requires sorted unique IDs and an unchanged material mapping",
        ));
    }
    let mut mappings = Vec::with_capacity(new.len() + 4);
    mappings.extend_from_slice(&[old.len() as u32, new.len() as u32, 0, 0]);
    let mut imports = Vec::new();
    let mut cursor = 0;
    for (destination, id) in new.iter().enumerate() {
        while cursor < old.len() && old[cursor] < *id {
            cursor += 1;
        }
        if old.get(cursor) == Some(id) {
            mappings.push(cursor as u32);
        } else {
            mappings.push(IMPORT | imports.len() as u32);
            imports.push(destination);
        }
    }
    let imported = imports.len() as u32;
    mappings[2] = imported;
    mappings.reserve(imports.len() * RECORD_WORDS as usize);
    for particle in imports {
        for (stream, stride) in STRIDES.into_iter().enumerate() {
            mappings.extend_from_slice(
                &selection.checkpoint().stream(stream)[particle * stride..(particle + 1) * stride],
            );
        }
        let id = new[particle].0;
        mappings.extend_from_slice(&[id as u32, (id >> 32) as u32]);
    }
    let retained = new.len() as u32 - imported;
    Ok(TransferUpload {
        stats: MpmSoilTransferStats {
            path: MpmSoilTransferPath::DeviceGather,
            retained_particles: retained,
            imported_particles: imported,
            removed_particles: old.len() as u32 - retained,
            upload_bytes: mappings.len() as u64 * 4,
            dispatches: u32::from(!new.is_empty()),
        },
        words: mappings,
    })
}

fn sizes(capacity: u32, limits: &wgpu::Limits) -> Result<[u64; 3]> {
    let capacity = u64::from(capacity.max(1));
    let particle_bytes = capacity * RECORD_WORDS * 4;
    let upload_bytes = 16 + capacity * (RECORD_WORDS + 1) * 4;
    let sizes = [particle_bytes, upload_bytes, particle_bytes];
    if capacity >= u64::from(IMPORT)
        || upload_bytes / 4 > u64::from(u32::MAX)
        || capacity.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
        || sizes
            .iter()
            .any(|&n| n > limits.max_buffer_size || n > limits.max_storage_buffer_binding_size)
    {
        return Err(invalid(
            "soil transfer capacity exceeds device/index limits",
        ));
    }
    Ok(sizes)
}

pub(super) fn unchanged(
    simulation: &MpmSimulation,
    selection: &MpmSoilSelection,
) -> Result<Option<MpmSoilTransferStats>> {
    if simulation.grid_window_cells != selection.checkpoint().grid_window_cells() {
        return Err(invalid("particle transfer cannot change the grid window"));
    }
    if simulation.particle_material_dirty.is_some()
        || simulation.material_rows != selection.checkpoint().materials
    {
        return Err(invalid("changed/pending transfer material table"));
    }
    let identity = simulation
        .soil_identity_snapshot()
        .ok_or_else(|| invalid("transfer requires live global IDs"))?;
    if identity.material_ids() != selection.material_ids() {
        return Err(invalid("transfer changes global material mapping"));
    }
    if identity.particle_ids() == selection.particle_ids() {
        simulation.validate_soil_restore(selection.checkpoint())?;
        Ok(Some(MpmSoilTransferStats {
            retained_particles: simulation.particle_count,
            ..Default::default()
        }))
    } else {
        Ok(None)
    }
}

pub(super) fn restore_default(
    engine: &Engine,
    simulation: &mut MpmSimulation,
    selection: &MpmSoilSelection,
) -> Result<MpmSoilTransferStats> {
    engine.check_device()?;
    molla_kernels::validate_soil_device(engine.device().unwrap(), &simulation.device)?;
    if let Some(stats) = unchanged(simulation, selection)? {
        return Ok(stats);
    }
    let old = simulation.soil_identity_snapshot().unwrap().particle_ids();
    let mut cursor = 0;
    let mut retained = 0;
    for id in selection.particle_ids() {
        while cursor < old.len() && old[cursor] < *id {
            cursor += 1;
        }
        retained += u32::from(old.get(cursor) == Some(id));
    }
    let new_count = selection.checkpoint().count;
    let stats = MpmSoilTransferStats {
        path: MpmSoilTransferPath::FullUpload,
        retained_particles: retained,
        imported_particles: new_count - retained,
        removed_particles: old.len() as u32 - retained,
        upload_bytes: selection.checkpoint().particle_bytes() as u64
            + u64::from(new_count) * 8
            + selection.material_ids().len() as u64 * 8,
        dispatches: 0,
    };
    simulation.restore_soil_selection(engine, selection)?;
    Ok(stats)
}

impl SoilTransfer {
    pub(super) fn required_bytes(simulation: &MpmSimulation) -> Result<u64> {
        Ok(
            sizes(simulation.cfg.max_particles, &simulation.device.limits())?
                .into_iter()
                .sum(),
        )
    }
    pub(super) fn new(engine: &Engine, simulation: &MpmSimulation) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().unwrap();
        molla_kernels::validate_soil_device(device, &simulation.device)?;
        let sizes = sizes(simulation.cfg.max_particles, &device.limits())?;
        let buffer = |label, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let input_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let previous = buffer("MPM soil transfer previous", sizes[0], input_usage);
        let upload = buffer(
            "MPM soil transfer imports and mapping",
            sizes[1],
            input_usage,
        );
        let output = buffer(
            "MPM soil transfer output",
            sizes[2],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MPM soil transfer"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mpm_soil_transfer.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MPM soil transfer"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MPM soil transfer"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                buffer_entry(0, &previous),
                buffer_entry(1, &upload),
                buffer_entry(2, &output),
            ],
        });
        Ok(Self {
            source: simulation.soil_owner.clone(),
            previous,
            upload,
            output,
            bindings,
            pipeline,
        })
    }

    pub(super) fn bytes(&self) -> u64 {
        self.previous.size() + self.upload.size() + self.output.size()
    }

    /// The caller owns a frozen captured boundary and staged conservative archive
    /// writeback. Retained histories come from the device, not the host selection.
    pub(super) fn apply(
        &self,
        engine: &Engine,
        simulation: &mut MpmSimulation,
        selection: &MpmSoilSelection,
    ) -> Result<MpmSoilTransferStats> {
        engine.check_device()?;
        molla_kernels::validate_soil_device(engine.device().unwrap(), &simulation.device)?;
        if !Arc::ptr_eq(&self.source, &simulation.soil_owner) {
            return Err(invalid("foreign transfer owner"));
        }
        if let Some(stats) = unchanged(simulation, selection)? {
            return Ok(stats);
        }
        simulation.validate_soil_restore(selection.checkpoint())?;
        let identity = simulation
            .soil_identity_snapshot()
            .ok_or_else(|| invalid("transfer requires live global IDs"))?;
        let upload = transfer_upload(identity, selection)?;
        let snapshot = MpmSoilIdentitySnapshot::from_selection(selection);
        let view = simulation.soil_identity_buffers().unwrap();
        let old_count = u64::from(simulation.particle_count);
        let new_count = u64::from(selection.checkpoint().count);
        let queue = engine.queue().unwrap();
        molla_core::gpu_write_buffer!(queue, &self.upload, 0, bytemuck::cast_slice(&upload.words));
        let mut encoder =
            engine
                .device()
                .unwrap()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("MPM soil device transfer"),
                });
        if old_count != 0 && upload.stats.retained_particles != 0 {
            let mut offset = 0;
            for (buffer, stride) in simulation.checkpoint_buffers().into_iter().zip(STRIDES) {
                let bytes = old_count * stride as u64 * 4;
                encoder.copy_buffer_to_buffer(buffer, 0, &self.previous, offset, bytes);
                offset += bytes;
            }
            encoder.copy_buffer_to_buffer(view.particles, 0, &self.previous, offset, old_count * 8);
        }
        if new_count != 0 {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MPM soil identity-preserving gather"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bindings, &[]);
                pass.dispatch_workgroups((new_count as u32).div_ceil(64), 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
            let mut offset = 0;
            for (buffer, stride) in simulation.checkpoint_buffers().into_iter().zip(STRIDES) {
                let bytes = new_count * stride as u64 * 4;
                encoder.copy_buffer_to_buffer(&self.output, offset, buffer, 0, bytes);
                offset += bytes;
            }
            encoder.copy_buffer_to_buffer(&self.output, offset, view.particles, 0, new_count * 8);
        }
        encoder.clear_buffer(
            &simulation.active_tiles.list,
            simulation.active_tiles.metadata_offset,
            Some(16),
        );
        queue.submit([encoder.finish()]);
        simulation.finish_identified_soil_restore(selection.checkpoint(), snapshot);
        Ok(upload.stats)
    }
}

#[cfg(test)]
#[path = "mpm_soil_transfer_tests.rs"]
mod tests;
