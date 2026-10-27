//! Prepared terrain-only MPM readback with optional compact particle statistics.

use crate::soil_mpm_summary::CompactionSummaryPass;
use crate::{TireMpmTerrainRasterGpuOutput, TireMpmTerrainRasterParams};
use molla_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TireMpmCompactionSummary {
    pub maximum: f32,
    pub active_particles: u32,
}

pub struct TireMpmTerrainSnapshot {
    pub terrain: TireMpmTerrainRasterGpuOutput,
    pub compaction: Option<TireMpmCompactionSummary>,
}

/// Fixed-source blocking terrain bridge; reprepare on source/count/dimension changes.
/// Device, queue and all buffers must belong to the same live GPU context.
pub struct PreparedTireMpmTerrainReadback {
    device: wgpu::Device,
    queue: wgpu::Queue,
    outputs: [wgpu::Buffer; 3],
    stage: wgpu::Buffer,
    summary: CompactionSummaryPass,
    params: TireMpmTerrainRasterParams,
    terrain_bytes: u64,
}

impl PreparedTireMpmTerrainReadback {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        outputs: [&wgpu::Buffer; 3],
        compaction: &wgpu::Buffer,
        params: TireMpmTerrainRasterParams,
        active_threshold: f32,
    ) -> Result<Self> {
        let cells = params.validate()? as u64;
        let terrain_bytes = cells
            .checked_mul(24)
            .ok_or_else(|| Error::Solver("MPM snapshot size overflow".into()))?;
        let limits = device.limits();
        if !active_threshold.is_finite()
            || active_threshold < 0.0
            || terrain_bytes + 16 > limits.max_buffer_size
            || params.particle_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err(Error::Solver(
                "MPM snapshot: invalid threshold or device capacity".into(),
            ));
        }
        for (buffer, bytes) in outputs.iter().zip([cells * 4, cells * 4, cells * 16]) {
            if buffer.size() < bytes || !buffer.usage().contains(wgpu::BufferUsages::COPY_SRC) {
                return Err(Error::Solver(
                    "MPM snapshot: terrain buffer lacks capacity/COPY_SRC".into(),
                ));
            }
        }
        let compaction_bytes = u64::from(params.particle_count.max(1)) * 16;
        if compaction.size() < compaction_bytes
            || compaction.size() > limits.max_storage_buffer_binding_size
            || !compaction.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(Error::Solver(
                "MPM snapshot: invalid compaction storage buffer".into(),
            ));
        }
        let allocate = |label, size, usage| {
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
        let stage = allocate(
            "MPM terrain-only staging",
            terrain_bytes + 16,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let summary = CompactionSummaryPass::new(
            device,
            queue,
            compaction,
            params.particle_count,
            active_threshold,
        )?;
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            outputs: outputs.map(Clone::clone),
            stage,
            summary,
            params,
            terrain_bytes,
        })
    }

    pub fn params(&self) -> TireMpmTerrainRasterParams {
        self.params
    }

    /// Copies raster maps and optionally 16 bytes of reduced particle statistics.
    /// Uses one submission/map/wait and retains staging/resources between calls.
    pub fn read_blocking(&mut self, collect_compaction: bool) -> Result<TireMpmTerrainSnapshot> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MPM terrain-only snapshot"),
            });
        if collect_compaction {
            self.summary.encode(&mut encoder);
            encoder.copy_buffer_to_buffer(
                &self.summary.output,
                0,
                &self.stage,
                self.terrain_bytes,
                16,
            );
        }
        let scalar_bytes = self.terrain_bytes / 6;
        for (source, (offset, size)) in self.outputs.iter().zip([
            (0, scalar_bytes),
            (scalar_bytes, scalar_bytes),
            (scalar_bytes * 2, scalar_bytes * 4),
        ]) {
            encoder.copy_buffer_to_buffer(source, 0, &self.stage, offset, size);
        }
        self.queue.submit([encoder.finish()]);
        let bytes = self.terrain_bytes + if collect_compaction { 16 } else { 0 };
        let slice = self.stage.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let mapped = (|| -> Result<()> {
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("MPM terrain snapshot poll: {e}")))?;
            rx.recv()
                .map_err(|e| Error::Gpu(format!("MPM terrain snapshot callback: {e}")))?
                .map_err(|e| Error::Gpu(format!("MPM terrain snapshot map: {e}")))
        })();
        if let Err(error) = mapped {
            self.stage.unmap();
            return Err(error);
        }
        let data = molla_core::gpu_readback!(slice);
        let scalar_bytes = scalar_bytes as usize;
        let terrain_bytes = self.terrain_bytes as usize;
        let terrain = TireMpmTerrainRasterGpuOutput {
            heights: bytemuck::cast_slice(&data[..scalar_bytes]).to_vec(),
            sinkage: bytemuck::cast_slice(&data[scalar_bytes..scalar_bytes * 2]).to_vec(),
            surface_velocity_map: bytemuck::cast_slice(&data[scalar_bytes * 2..terrain_bytes])
                .to_vec(),
        };
        let compaction = collect_compaction.then(|| {
            let words: &[u32] = bytemuck::cast_slice(&data[terrain_bytes..]);
            TireMpmCompactionSummary {
                maximum: f32::from_bits(words[0]),
                active_particles: words[1],
            }
        });
        drop(data);
        self.stage.unmap();
        Ok(TireMpmTerrainSnapshot {
            terrain,
            compaction,
        })
    }
}
