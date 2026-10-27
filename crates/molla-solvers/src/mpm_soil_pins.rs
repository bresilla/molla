//! Exact bounded GPU occupancy pins and integer-tile halo expansion.

use super::*;
use molla_vehicle::SoilTileLayout;
use std::sync::OnceLock;

/// Half-open signed archive-tile box, unique output capacity and cubic halo.
/// GPU mode requires zero layout origin and power-of-two edge in 2^-126..2^127.
/// Coordinates are limited to [-2^30, 2^30]; halo radius is at most four tiles.
/// All active particles contribute, not only moving or physically awake material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmSoilPinConfig {
    pub minimum: SoilTileCoord,
    pub dimensions: [u32; 3],
    pub capacity: u32,
    pub halo_tiles: u32,
}

fn parameters(
    layout: SoilTileLayout,
    config: MpmSoilPinConfig,
    max_particles: u32,
    limits: &wgpu::Limits,
    budget: u64,
) -> Result<([u32; 12], [u64; 5])> {
    let bits = layout.edge().to_bits();
    let exponent = ((bits >> 52) & 2047) as i32 - 1023;
    let cells = config
        .dimensions
        .iter()
        .try_fold(1_u64, |n, &d| n.checked_mul(u64::from(d)));
    let slots = config
        .capacity
        .checked_mul(2)
        .and_then(u32::checked_next_power_of_two);
    let coordinate_limit = 1_i64 << 30;
    if layout.origin() != [0.0; 3]
        || bits & ((1_u64 << 52) - 1) != 0
        || !(-126..=127).contains(&exponent)
        || config.capacity == 0
        || config.halo_tiles > 4
        || cells.is_none_or(|n| n == 0 || n >= u64::from(u32::MAX))
        || slots.is_none()
        || config.dimensions.iter().any(|&d| d == 0)
        || (0..3).any(|a| {
            config.minimum.0[a] < -coordinate_limit
                || config.minimum.0[a] > coordinate_limit
                || config.minimum.0[a] + i64::from(config.dimensions[a]) > coordinate_limit + 1
        })
        || u64::from(max_particles) * u64::from(2 * config.halo_tiles + 1).pow(3)
            > u64::from(u32::MAX)
    {
        return Err(invalid(
            "unsupported GPU pin layout, coordinate box or capacity",
        ));
    }
    let slots = slots.unwrap();
    let sizes = [
        48,
        u64::from(slots) * 4,
        u64::from(config.capacity) * 4,
        16,
        16 + u64::from(config.capacity) * 4,
    ];
    if sizes.iter().sum::<u64>() > budget
        || sizes.iter().any(|&n| n > limits.max_buffer_size)
        || sizes[1..4]
            .iter()
            .any(|&n| n > limits.max_storage_buffer_binding_size)
        || slots.div_ceil(64) > limits.max_compute_workgroups_per_dimension
        || max_particles.div_ceil(64) > limits.max_compute_workgroups_per_dimension
    {
        return Err(invalid("GPU pin buffers exceed budget or device limits"));
    }
    let words = [
        config.minimum.0[0] as i32 as u32,
        config.minimum.0[1] as i32 as u32,
        config.minimum.0[2] as i32 as u32,
        exponent as u32,
        config.dimensions[0],
        config.dimensions[1],
        config.dimensions[2],
        slots - 1,
        0,
        config.capacity,
        config.halo_tiles,
        0,
    ];
    Ok((words, sizes))
}

struct Request {
    revision: u64,
    count: u32,
    signal: Arc<OnceLock<std::result::Result<(), wgpu::BufferAsyncError>>>,
}

fn local_config(
    layout: SoilTileLayout,
    config: MpmSoilPinConfig,
    origin: Option<[f64; 3]>,
) -> Result<MpmSoilPinConfig> {
    let Some(origin) = origin else {
        return Ok(config);
    };
    let offset = layout
        .tile_at_local([0.0; 3], origin)
        .map_err(|e| invalid(&e.to_string()))?;
    let mut local = config;
    for axis in 0..3 {
        if layout.origin()[axis] != 0.0 || origin[axis] != offset.0[axis] as f64 * layout.edge() {
            return Err(invalid(
                "GPU framed pins require a tile-aligned frame and zero layout origin",
            ));
        }
        let end = config.minimum.0[axis]
            .checked_add(i64::from(config.dimensions[axis]))
            .ok_or_else(|| invalid("GPU framed pin domain overflow"))?;
        if config.minimum.0[axis].unsigned_abs() >= 1u64 << 52 || end.unsigned_abs() >= 1u64 << 52 {
            return Err(invalid("GPU framed pin domain exceeds world tile range"));
        }
        local.minimum.0[axis] = config.minimum.0[axis]
            .checked_sub(offset.0[axis])
            .ok_or_else(|| invalid("GPU framed pin offset overflow"))?;
    }
    Ok(local)
}

pub(super) struct PinSnapshot {
    pub revision: u64,
    pub tiles: Vec<SoilTileCoord>,
}

pub(super) struct SoilPins {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    health: Arc<molla_compute::DeviceHealth>,
    source: Arc<()>,
    config: MpmSoilPinConfig,
    params: [u32; 12],
    uniform: wgpu::Buffer,
    table: wgpu::Buffer,
    keys: wgpu::Buffer,
    metadata: wgpu::Buffer,
    staging: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    pipelines: [wgpu::ComputePipeline; 2],
    request: Option<Request>,
    failure: Option<String>,
}

impl SoilPins {
    pub(super) fn new(
        engine: &Engine,
        simulation: &MpmSimulation,
        layout: SoilTileLayout,
        config: MpmSoilPinConfig,
        budget: u64,
    ) -> Result<Self> {
        Self::new_in_frame(engine, simulation, layout, config, budget, None)
    }

    pub(super) fn new_in_frame(
        engine: &Engine,
        simulation: &MpmSimulation,
        layout: SoilTileLayout,
        config: MpmSoilPinConfig,
        budget: u64,
        origin: Option<[f64; 3]>,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().unwrap();
        molla_kernels::validate_soil_device(device, &simulation.device)?;
        let (params, sizes) = parameters(
            layout,
            local_config(layout, config, origin)?,
            simulation.cfg.max_particles,
            &device.limits(),
            budget,
        )?;
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
        let uniform = buffer(
            "soil pin parameters",
            sizes[0],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let table = buffer("soil pin hash", sizes[1], wgpu::BufferUsages::STORAGE);
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let keys = buffer("soil pin keys", sizes[2], storage);
        let metadata = buffer("soil pin status", sizes[3], storage);
        let staging = buffer(
            "soil pin readback",
            sizes[4],
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let entries: Vec<_> = (0..5)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding == 1,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil pins"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soil pins"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil pins"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mpm_soil_pins.wgsl").into()),
        });
        let pipelines = ["clear", "mark"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil pins"),
            layout: &layout,
            entries: &[
                buffer_entry(0, &uniform),
                buffer_entry(1, &simulation.p_pos),
                buffer_entry(2, &table),
                buffer_entry(3, &keys),
                buffer_entry(4, &metadata),
            ],
        });
        Ok(Self {
            device: device.clone(),
            queue: engine.queue().unwrap().clone(),
            health: engine.device_health().unwrap().clone(),
            source: simulation.soil_owner.clone(),
            config,
            params,
            uniform,
            table,
            keys,
            metadata,
            staging,
            bindings,
            pipelines,
            request: None,
            failure: None,
        })
    }

    pub(super) fn recentered_parameters(
        &self,
        layout: SoilTileLayout,
        simulation: &MpmSimulation,
        origin: [f64; 3],
    ) -> Result<[u32; 12]> {
        self.health.check()?;
        if self.failure.is_some() || self.request.is_some() {
            return Err(invalid("GPU pins must be healthy and idle before recenter"));
        }
        let (params, _) = parameters(
            layout,
            local_config(layout, self.config, Some(origin))?,
            simulation.cfg.max_particles,
            &self.device.limits(),
            self.bytes(),
        )?;
        Ok(params)
    }

    pub(super) fn commit_recenter(&mut self, params: [u32; 12]) {
        self.params = params;
    }

    pub(super) fn bytes(&self) -> u64 {
        [
            &self.uniform,
            &self.table,
            &self.keys,
            &self.metadata,
            &self.staging,
        ]
        .iter()
        .map(|b| b.size())
        .sum()
    }

    fn fail(&mut self, message: String) -> Error {
        self.request = None;
        self.failure = Some(message.clone());
        Error::Gpu(message)
    }

    fn check(&mut self) -> Result<()> {
        if let Some(message) = &self.failure {
            return Err(Error::Gpu(message.clone()));
        }
        if let Err(error) = self.health.check() {
            return Err(self.fail(error.to_string()));
        }
        Ok(())
    }

    pub(super) fn submit(&mut self, simulation: &MpmSimulation, revision: u64) -> Result<()> {
        self.check()?;
        if self.request.is_some() || !Arc::ptr_eq(&self.source, &simulation.soil_owner) {
            return Err(invalid("GPU pin capture busy or foreign source"));
        }
        self.params[8] = simulation.particle_count;
        molla_core::gpu_write_buffer!(
            self.queue,
            &self.uniform,
            0,
            bytemuck::cast_slice(&self.params)
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("soil pins"),
            });
        for (index, groups) in [
            (0, (self.params[7] + 1).div_ceil(64)),
            (1, simulation.particle_count.div_ceil(64)),
        ] {
            if groups == 0 {
                continue;
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("soil pins"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines[index]);
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        encoder.copy_buffer_to_buffer(&self.metadata, 0, &self.staging, 0, 16);
        encoder.copy_buffer_to_buffer(&self.keys, 0, &self.staging, 16, self.keys.size());
        self.queue.submit([encoder.finish()]);
        let signal = Arc::new(OnceLock::new());
        let callback = signal.clone();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = callback.set(result);
            });
        self.request = Some(Request {
            revision,
            count: simulation.particle_count,
            signal,
        });
        Ok(())
    }

    pub(super) fn poll(&mut self) -> Result<Option<PinSnapshot>> {
        self.check()?;
        if self.request.is_none() {
            return Ok(None);
        }
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Err(self.fail(error.to_string()));
        }
        self.check()?;
        let Some(result) = self.request.as_ref().unwrap().signal.get() else {
            return Ok(None);
        };
        if let Err(error) = result {
            return Err(self.fail(format!("GPU pin map: {error}")));
        }
        let request = self.request.take().unwrap();
        let slice = self.staging.slice(..);
        let mapped = molla_core::gpu_readback!(slice);
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let result = (|| {
            if words[1] != 0
                || words[0] > self.config.capacity
                || words[2] != request.count
                || words[3] != self.config.halo_tiles
            {
                return Err(invalid(&format!(
                    "incomplete GPU pins: count={} flags={} (domain=1, position=2, capacity=4)",
                    words[0], words[1]
                )));
            }
            let [nx, ny, nz] = self.config.dimensions;
            let mut tiles = Vec::with_capacity(words[0] as usize);
            for &key in &words[4..4 + words[0] as usize] {
                if u64::from(key) >= u64::from(nx) * u64::from(ny) * u64::from(nz) {
                    return Err(invalid("GPU pin key outside declared domain"));
                }
                let local = [key % nx, (key / nx) % ny, key / (nx * ny)];
                tiles.push(SoilTileCoord(std::array::from_fn(|a| {
                    self.config.minimum.0[a] + i64::from(local[a])
                })));
            }
            tiles.sort_unstable();
            if tiles.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(invalid("duplicate GPU pin key"));
            }
            Ok(PinSnapshot {
                revision: request.revision,
                tiles,
            })
        })();
        drop(mapped);
        self.staging.unmap();
        result.map(Some)
    }
}

#[cfg(test)]
#[path = "mpm_soil_pins_tests.rs"]
mod tests;
