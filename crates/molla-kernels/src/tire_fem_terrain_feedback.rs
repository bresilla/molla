//! Retained synchronous CPU-tire feedback over resident terrain maps.

use super::*;
use molla_compute::{DeviceHealth, Engine};
use std::sync::Arc;

fn invalid(message: &str) -> Error {
    Error::Solver(format!("prepared FEM terrain feedback: {message}"))
}

fn sources<'a>(terrain: &'a TireFemContactFeedbackGpuTerrain<'a>) -> [Option<&'a wgpu::Buffer>; 5] {
    [
        Some(terrain.heights),
        terrain.friction_map,
        terrain.water_film_map,
        terrain.sinkage_map,
        terrain.surface_velocity_map,
    ]
}

pub(super) fn terrain_valid(
    device: &wgpu::Device,
    terrain: &TireFemContactFeedbackGpuTerrain<'_>,
) -> Result<()> {
    let cells = u64::from(terrain.nx) * u64::from(terrain.nz);
    if cells == 0
        || cells > u64::from(u32::MAX)
        || !terrain.spacing.is_finite()
        || terrain.spacing <= 0.0
        || terrain
            .origin
            .iter()
            .chain(&terrain.surface_velocity)
            .any(|x| !x.is_finite())
        || !terrain.friction.is_finite()
        || !terrain.soil_rolling_resistance_scale.is_finite()
    {
        return Err(invalid("invalid terrain dimensions or scalar parameters"));
    }
    for (source, stride) in sources(terrain).into_iter().zip([4, 4, 4, 4, 16]) {
        if let Some(buffer) = source {
            if !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                || buffer.size() < cells * stride
                || buffer.size() > device.limits().max_storage_buffer_binding_size
            {
                return Err(invalid("terrain buffer lacks storage capacity"));
            }
        }
    }
    Ok(())
}

fn uniforms(
    input: &TireFemContactFeedbackInput,
    terrain: &TireFemContactFeedbackGpuTerrain<'_>,
) -> Result<ContactFeedbackUniforms> {
    let p = input.params;
    let flags = (terrain.friction_map.is_some() as u32)
        | ((terrain.water_film_map.is_some() as u32) << 1)
        | ((terrain.sinkage_map.is_some() as u32) << 2)
        | ((terrain.surface_velocity_map.is_some() as u32) << 3);
    let value = ContactFeedbackUniforms {
        counts: [
            input.positions.len() as u32,
            input.tread_ids.len() as u32,
            0,
            0,
        ],
        phys0: [p.dt, p.contact_band, p.ground_y, p.omega],
        hub: [p.hub[0], p.hub[1], p.hub[2], p.rolling_radius],
        hub_vel: [
            p.hub_vel[0],
            p.hub_vel[1],
            p.hub_vel[2],
            p.normal_load_per_tread,
        ],
        heading: [p.heading[0], p.heading[1], p.heading[2], p.area_per_tread],
        material0: [
            p.tread_shear_stiffness,
            p.mu_static,
            p.mu_dynamic,
            p.bristle_damping,
        ],
        material1: [p.cohesion, p.bristle_relax_rate, 0.0, 0.0],
        terrain_velocity: [
            terrain.surface_velocity[0],
            terrain.surface_velocity[1],
            terrain.surface_velocity[2],
            0.0,
        ],
        terrain_origin: [
            terrain.origin[0],
            terrain.origin[1],
            terrain.spacing,
            terrain.friction,
        ],
        terrain_dims: [terrain.nx, terrain.nz, 1, flags],
        soil0: [
            terrain.soil_rolling_resistance_scale.max(0.0),
            0.0,
            0.0,
            0.0,
        ],
    };
    if p.dt <= 0.0
        || [
            value.phys0,
            value.hub,
            value.hub_vel,
            value.heading,
            value.material0,
            value.material1,
        ]
        .iter()
        .flatten()
        .any(|v| !v.is_finite())
    {
        return Err(invalid(
            "nonfinite feedback parameter or nonpositive timestep",
        ));
    }
    Ok(value)
}

/// Fixed-capacity feedback buffers and one retained readback slot. Every call
/// uploads current CPU node/bristle state, consumes the current GPU terrain, and
/// returns current velocities/forces/bristles after one submit/map/wait. This is
/// a synchronous CPU consumer bridge, not a fully resident tire solver.
/// Terrain buffers must belong to the supplied engine; changing their identity
/// or optional-map presence requires preparation again. Scalar values can change.
pub struct PreparedTireFemTerrainFeedback {
    device: wgpu::Device,
    queue: wgpu::Queue,
    health: Arc<DeviceHealth>,
    kernel: Arc<()>,
    pipeline: wgpu::ComputePipeline,
    buffers: [wgpu::Buffer; 8],
    uniform: wgpu::Buffer,
    dummy: wgpu::Buffer,
    staging: wgpu::Buffer,
    bindings: [wgpu::BindGroup; 2],
    terrain: [Option<wgpu::Buffer>; 5],
    nodes: usize,
    treads: usize,
    seen: Vec<bool>,
    failed: bool,
}

impl PreparedTireFemTerrainFeedback {
    /// Allocate under a logical GPU byte budget; capacity and source validation
    /// finish before allocation. CPU vector/driver overhead is not in this budget.
    pub fn new(
        engine: &Engine,
        kernel: &TireFemContactFeedbackKernel,
        max_nodes: usize,
        max_treads: usize,
        terrain: &TireFemContactFeedbackGpuTerrain<'_>,
        max_bytes: u64,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| invalid("GPU engine required"))?;
        let queue = engine
            .queue()
            .ok_or_else(|| invalid("GPU queue required"))?;
        crate::soil_gpu_identity::validate_soil_device(device, &kernel.device)?;
        terrain_valid(device, terrain)?;
        if max_nodes == 0
            || max_nodes > u32::MAX as usize
            || max_treads > max_nodes
            || (max_nodes as u32).div_ceil(WORKGROUP_SIZE)
                > device.limits().max_compute_workgroups_per_dimension
        {
            return Err(invalid("node/tread capacity exceeds dispatch limits"));
        }
        let n = max_nodes as u64;
        let t = max_treads.max(1) as u64;
        let sizes = [16 * n, 16 * n, 4 * n, 4 * t, 16 * t, 16 * n, 16 * t, 16 * n];
        let stage_bytes = 32 * n + 16 * t;
        let uniform_bytes = core::mem::size_of::<ContactFeedbackUniforms>() as u64;
        let bytes = sizes.iter().sum::<u64>() + stage_bytes + uniform_bytes + 32;
        if bytes > max_bytes
            || stage_bytes > device.limits().max_buffer_size
            || sizes.iter().any(|&n| {
                n > device.limits().max_storage_buffer_binding_size
                    || n > device.limits().max_buffer_size
            })
        {
            return Err(invalid(
                "buffers exceed memory budget/device binding limits",
            ));
        }
        let allocate = |size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("retained FEM terrain feedback"),
                    size,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buffers = sizes.map(|size| allocate(size, storage));
        let dummy = allocate(32, wgpu::BufferUsages::STORAGE);
        let uniform = allocate(
            uniform_bytes,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let staging = allocate(
            stage_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let maps = sources(terrain);
        let mut entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| entry(i as u32, buffer))
            .collect();
        for (binding, source) in [
            (8, maps[0]),
            (9, maps[1]),
            (10, maps[2]),
            (11, None),
            (12, maps[3]),
            (13, maps[4]),
            (14, None),
        ] {
            entries.push(entry(binding, source.unwrap_or(&dummy)));
        }
        let bindings = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("retained FEM feedback IO"),
                layout: &kernel.layout_io,
                entries: &entries,
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("retained FEM feedback parameters"),
                layout: &kernel.layout_uniforms,
                entries: &[entry(0, &uniform)],
            }),
        ];
        Ok(Self {
            device: device.as_ref().clone(),
            queue: queue.as_ref().clone(),
            health: engine.device_health().unwrap().clone(),
            kernel: kernel.identity.clone(),
            pipeline: kernel.pipeline.clone(),
            buffers,
            uniform,
            dummy,
            staging,
            bindings,
            terrain: maps.map(|b| b.cloned()),
            nodes: max_nodes,
            treads: max_treads,
            seen: vec![false; max_nodes],
            failed: false,
        })
    }

    pub fn buffer_bytes(&self) -> u64 {
        self.buffers.iter().map(wgpu::Buffer::size).sum::<u64>()
            + self.uniform.size()
            + self.dummy.size()
            + self.staging.size()
    }

    fn same_sources(&self, terrain: &TireFemContactFeedbackGpuTerrain<'_>) -> bool {
        self.terrain
            .iter()
            .zip(sources(terrain))
            .all(|(a, b)| a.as_ref() == b)
    }

    /// Tests retained program/engine/source identities and available capacity.
    /// Per-call shape/content validation still occurs before any upload.
    pub fn matches(
        &self,
        engine: &Engine,
        kernel: &TireFemContactFeedbackKernel,
        nodes: usize,
        treads: usize,
        terrain: &TireFemContactFeedbackGpuTerrain<'_>,
    ) -> bool {
        !self.failed
            && engine
                .device_health()
                .is_some_and(|h| Arc::ptr_eq(h, &self.health))
            && Arc::ptr_eq(&self.kernel, &kernel.identity)
            && nodes <= self.nodes
            && treads <= self.treads
            && self.same_sources(terrain)
    }

    pub fn run_blocking(
        &mut self,
        input: &TireFemContactFeedbackInput,
        terrain: &TireFemContactFeedbackGpuTerrain<'_>,
    ) -> Result<TireFemContactFeedbackOutput> {
        self.health.check()?;
        if self.failed {
            return Err(Error::Gpu("prepared FEM feedback readback failed".into()));
        }
        let n = input.positions.len();
        let t = input.tread_ids.len();
        if n == 0
            || n > self.nodes
            || t > self.treads
            || input.velocities.len() != n
            || input.masses.len() != n
            || input.bristles.len() != t
            || input.terrain.is_some()
            || !self.same_sources(terrain)
        {
            return Err(invalid(
                "input shape/source changed or CPU terrain supplied",
            ));
        }
        terrain_valid(&self.device, terrain)?;
        let uniforms = uniforms(input, terrain)?;
        if input
            .positions
            .iter()
            .chain(&input.velocities)
            .chain(&input.bristles)
            .flatten()
            .any(|v| !v.is_finite())
            || input.masses.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(invalid("nonfinite node/bristle input or negative mass"));
        }
        self.seen.fill(false);
        for &id in &input.tread_ids {
            if id as usize >= n || self.seen[id as usize] {
                return Err(invalid("invalid or duplicate tread node"));
            }
            self.seen[id as usize] = true;
        }
        for (buffer, data) in self.buffers[..5].iter().zip([
            bytemuck::cast_slice(&input.positions),
            bytemuck::cast_slice(&input.velocities),
            bytemuck::cast_slice(&input.masses),
            bytemuck::cast_slice(&input.tread_ids),
            bytemuck::cast_slice(&input.bristles),
        ]) {
            if !data.is_empty() {
                molla_core::gpu_write_buffer!(self.queue, buffer, 0, data);
            }
        }
        molla_core::gpu_write_buffer!(self.queue, &self.uniform, 0, bytemuck::bytes_of(&uniforms));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings[0], &[]);
            pass.set_bind_group(1, &self.bindings[1], &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        let node_bytes = n as u64 * 16;
        let bristle_bytes = t as u64 * 16;
        encoder.copy_buffer_to_buffer(&self.buffers[5], 0, &self.staging, 0, node_bytes);
        encoder.copy_buffer_to_buffer(&self.buffers[7], 0, &self.staging, node_bytes, node_bytes);
        if t > 0 {
            encoder.copy_buffer_to_buffer(
                &self.buffers[6],
                0,
                &self.staging,
                2 * node_bytes,
                bristle_bytes,
            );
        }
        self.queue.submit([encoder.finish()]);
        let slice = self.staging.slice(..2 * node_bytes + bristle_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let mapped = (|| {
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(e.to_string()))?;
            self.health.check()?;
            rx.recv()
                .map_err(|e| Error::Gpu(e.to_string()))?
                .map_err(|e| Error::Gpu(e.to_string()))
        })();
        if let Err(error) = mapped {
            self.failed = true;
            return Err(error);
        }
        let view = molla_core::gpu_readback!(slice);
        let rows: &[[f32; 4]] = bytemuck::cast_slice(&view);
        let output = TireFemContactFeedbackOutput {
            velocities: rows[..n].to_vec(),
            forces: rows[n..2 * n].to_vec(),
            bristles: rows[2 * n..].to_vec(),
        };
        drop(view);
        self.staging.unmap();
        Ok(output)
    }
}
