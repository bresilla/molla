use crate::soil_particle_raster::{ParticleRasterKernel, PreparedParticleRaster, RasterParams};
use molla_core::{Error, Result};
use std::time::{Duration, Instant};

#[path = "soil_surface_observation.rs"]
mod observation;
pub use observation::SoilSurfaceObservation;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoilSurfaceConfig {
    pub origin: [f32; 2],
    pub spacing: f32,
    pub nx: u32,
    pub nz: u32,
    pub floor_height: f32,
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("soil geometric surface: {message}"))
}

impl SoilSurfaceConfig {
    /// Half-open cell edges rounded once from f64 affine coordinates to stored f32.
    pub fn edges(self) -> Result<Vec<f32>> {
        self.cells()?;
        let mut edges = Vec::with_capacity((self.nx + self.nz + 2) as usize);
        for (origin, n) in self.origin.into_iter().zip([self.nx, self.nz]) {
            let mut previous = f32::NEG_INFINITY;
            for i in 0..=n {
                let edge = (f64::from(origin) + f64::from(i) * f64::from(self.spacing)) as f32;
                if !edge.is_finite() || edge <= previous {
                    return Err(invalid("cell edges are not distinguishable in f32"));
                }
                edges.push(edge);
                previous = edge;
            }
        }
        Ok(edges)
    }

    pub fn cells(self) -> Result<u32> {
        let cells = self
            .nx
            .checked_mul(self.nz)
            .filter(|&n| n > 0 && self.nx <= 16_777_216 && self.nz <= 16_777_216)
            .ok_or_else(|| invalid("invalid dimensions"))?;
        if !self.spacing.is_finite()
            || self.spacing < 1e-6
            || !self.floor_height.is_finite()
            || self.origin.iter().any(|v| !v.is_finite())
            || self
                .origin
                .into_iter()
                .zip([self.nx, self.nz])
                .any(|(origin, n)| {
                    let end = origin + n as f32 * self.spacing;
                    !end.is_finite() || end <= origin
                })
        {
            return Err(invalid("invalid finite raster bounds"));
        }
        Ok(cells)
    }

    /// Owned GPU buffers, including bins, scan uniforms, outputs and readback staging.
    pub fn required_bytes(self, particles: u32) -> Result<u64> {
        if particles > 16_777_216 {
            return Err(invalid("particle count exceeds exact f32 counters"));
        }
        let cells = self.cells()?;
        let mut len = cells;
        let mut bytes = u64::from(cells) * 48
            + u64::from(particles.max(1)) * 4
            + (u64::from(self.nx) + u64::from(self.nz) + 2) * 4;
        loop {
            bytes = bytes
                .checked_add(u64::from(len) * 12 + 64)
                .ok_or_else(|| invalid("size overflow"))?;
            if len <= 64 {
                break;
            }
            len = len.div_ceil(64);
        }
        Ok(bytes)
    }
}

#[derive(Clone, Debug)]
pub struct SoilSurfaceSnapshot {
    pub config: SoilSurfaceConfig,
    pub heights: Vec<f32>,
    pub counts: Vec<u32>,
    pub outside_particles: u32,
}

/// Particle-centre height envelope, not a compaction law or a collision boundary.
/// Out-of-domain points are counted separately; empty cells report the configured floor.
pub struct PreparedSoilSurface {
    device: wgpu::Device,
    queue: wgpu::Queue,
    source: wgpu::Buffer,
    count: u32,
    config: SoilSurfaceConfig,
    raster: PreparedParticleRaster,
    outputs: [wgpu::Buffer; 3],
    stage: wgpu::Buffer,
    bytes: u64,
    observations: observation::State,
}

impl PreparedSoilSurface {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &wgpu::Buffer,
        particles: u32,
        config: SoilSurfaceConfig,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        let bytes = config.required_bytes(particles)?;
        let cells = u64::from(config.cells()?);
        let limits = device.limits();
        if bytes > max_owned_bytes
            || cells * 24 > limits.max_buffer_size
            || cells * 16 > u64::from(limits.max_storage_buffer_binding_size)
            || positions.size() < u64::from(particles.max(1)) * 16
            || positions.size() > u64::from(limits.max_storage_buffer_binding_size)
            || !positions.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(invalid("budget or source/device capacity exceeded"));
        }
        let edge_bytes = (u64::from(config.nx) + u64::from(config.nz) + 2) * 4;
        if edge_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || edge_bytes > limits.max_buffer_size
        {
            return Err(invalid("edge table exceeds device limits"));
        }
        let edge_values = config.edges()?;
        let edges = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("soil surface exact cell boundaries"),
                size: edge_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true,
            }
        );
        edges
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&edge_values));
        edges.unmap();
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
        let outputs = [4, 4, 16].map(|size| {
            allocate(
                "soil surface output",
                cells * size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            )
        });
        let stage = allocate(
            "soil surface bounded staging",
            cells * 24,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let kernel =
            ParticleRasterKernel::new(device, include_str!("../shaders/soil_surface_raster.wgsl"));
        let raster = kernel.prepare(
            device,
            [
                positions,
                &edges,
                positions,
                &outputs[0],
                &outputs[1],
                &outputs[2],
            ],
            &RasterParams {
                particles,
                nx: config.nx,
                nz: config.nz,
                terrain: [
                    config.origin[0],
                    config.origin[1],
                    config.spacing,
                    config.floor_height,
                ],
                extra: [0.0; 4],
                attribute_bytes: edge_bytes,
            },
        )?;
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            source: positions.clone(),
            count: particles,
            config,
            raster,
            outputs,
            stage,
            bytes,
            observations: Default::default(),
        })
    }

    pub fn retained_bytes(&self) -> u64 {
        self.bytes
    }
    pub fn readback_bytes(&self) -> u64 {
        self.stage.size()
    }
    pub fn config(&self) -> SoilSurfaceConfig {
        self.config
    }
    pub fn matches(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::Buffer,
        count: u32,
    ) -> bool {
        &self.device == device
            && &self.queue == queue
            && &self.source == source
            && self.count == count
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.raster.encode(encoder);
    }
    pub fn encode_reference(&self, encoder: &mut wgpu::CommandEncoder) {
        self.raster.encode_reference(encoder);
    }

    /// Reads only the completed raster; callers encode export first on the same queue.
    pub fn read_blocking(&mut self, timeout: Duration) -> Result<SoilSurfaceSnapshot> {
        self.check_readback_idle()?;
        let deadline = Instant::now()
            .checked_add(timeout)
            .filter(|_| !timeout.is_zero())
            .ok_or_else(|| invalid("invalid readback deadline"))?;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let cells = u64::from(self.config.cells()?);
        for (buffer, (offset, size)) in self.outputs.iter().zip([
            (0, cells * 4),
            (cells * 4, cells * 4),
            (cells * 8, cells * 16),
        ]) {
            encoder.copy_buffer_to_buffer(buffer, 0, &self.stage, offset, size);
        }
        let submitted = self.queue.submit([encoder.finish()]);
        let slice = self.stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let remaining = || {
            deadline
                .checked_duration_since(Instant::now())
                .filter(|d| !d.is_zero())
                .ok_or_else(|| invalid("readback timed out"))
        };
        let mapped = (|| -> Result<()> {
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submitted),
                    timeout: Some(remaining()?),
                })
                .map_err(|e| Error::Gpu(e.to_string()))?;
            rx.recv_timeout(remaining()?)
                .map_err(|e| Error::Gpu(e.to_string()))?
                .map_err(|e| Error::Gpu(e.to_string()))?;
            Ok(())
        })();
        if let Err(error) = mapped {
            self.stage.unmap();
            return Err(error);
        }
        let data = molla_core::gpu_readback!(slice);
        let result = decode(self.config, self.count, bytemuck::cast_slice(&data));
        drop(data);
        self.stage.unmap();
        result
    }
}

fn decode(
    config: SoilSurfaceConfig,
    particles: u32,
    values: &[f32],
) -> Result<SoilSurfaceSnapshot> {
    let cells = config.cells()? as usize;
    if values.len() != cells * 6 || values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("invalid raster output"));
    }
    let counter = |v: f32| -> Result<u32> {
        if v < 0.0 || v > particles as f32 || v.fract() != 0.0 {
            return Err(invalid("invalid particle counter"));
        }
        Ok(v as u32)
    };
    let counts = values[cells..cells * 2]
        .iter()
        .map(|&v| counter(v))
        .collect::<Result<Vec<_>>>()?;
    let mut outside = 0u64;
    for row in values[cells * 2..].chunks_exact(4) {
        outside += u64::from(counter(row[0])?);
        if row[1] != 0.0 || row[2] != 0.0 || row[3] != 0.0 {
            return Err(invalid("invalid source positions or diagnostic fields"));
        }
    }
    if counts.iter().map(|&v| u64::from(v)).sum::<u64>() + outside != u64::from(particles)
        || values[..cells]
            .iter()
            .zip(&counts)
            .any(|(&height, &count)| {
                height < config.floor_height || (count == 0 && height != config.floor_height)
            })
    {
        return Err(invalid("surface accounting failed"));
    }
    Ok(SoilSurfaceSnapshot {
        config,
        heights: values[..cells].to_vec(),
        counts,
        outside_particles: outside as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> SoilSurfaceConfig {
        SoilSurfaceConfig {
            origin: [-1.0; 2],
            spacing: 0.125,
            nx: 2,
            nz: 1,
            floor_height: -0.5,
        }
    }
    #[test]
    fn surface_config_and_memory_are_checked() {
        assert_eq!(config().required_bytes(3).unwrap(), 216);
        assert!(
            SoilSurfaceConfig {
                origin: [1.0e8; 2],
                spacing: 0.125,
                nx: 1024,
                nz: 1,
                ..config()
            }
            .edges()
            .is_err()
        );
        for bad in [
            SoilSurfaceConfig { nx: 0, ..config() },
            SoilSurfaceConfig {
                spacing: f32::NAN,
                ..config()
            },
            SoilSurfaceConfig {
                origin: [f32::MAX; 2],
                ..config()
            },
        ] {
            assert!(bad.cells().is_err());
        }
        assert!(config().required_bytes(16_777_217).is_err());
    }
    #[test]
    fn surface_decode_enforces_exact_accounting_and_finite_data() {
        let v = [0.25, -0.5, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let s = decode(config(), 3, &v).unwrap();
        assert_eq!(s.counts, [2, 0]);
        assert_eq!(s.outside_particles, 1);
        for (at, value) in [(0, f32::NAN), (1, 0.0), (2, 1.5), (4, 0.0), (5, 1.0)] {
            let mut bad = v;
            bad[at] = value;
            assert!(decode(config(), 3, &bad).is_err());
        }
    }
}
