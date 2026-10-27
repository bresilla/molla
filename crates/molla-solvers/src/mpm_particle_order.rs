//! Retained stable tile ordering for particle-to-grid dispatch.

use super::*;
use crate::kernels::{RadixSortDispatch, RadixSortKernel};

#[cfg(test)]
#[path = "mpm_particle_order_tests.rs"]
mod tests;

pub(super) struct ParticleOrder {
    pub capacity: u32,
    parallel: bool,
    pub(super) keys: wgpu::Buffer,
    identities: wgpu::Buffer,
    pub(super) params: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    sort: RadixSortDispatch,
}

impl ParticleOrder {
    fn new(
        sim: &MpmSimulation,
        queue: &wgpu::Queue,
        capacity: u32,
        parallel: bool,
    ) -> Result<Self> {
        let device = &sim.device;
        let bytes = u64::from(capacity) * 4;
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
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let keys = allocate("MPM particle tile keys", bytes, usage);
        let identities = allocate("MPM ordered particle identities", bytes, usage);
        let params = allocate(
            "MPM particle order parameters",
            48,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MPM particle tile keys"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mpm_particle_order.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MPM particle tile keys"),
            layout: None,
            module: &shader,
            entry_point: Some("tile_keys"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MPM particle order"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                buffer_entry(0, &params),
                buffer_entry(1, &sim.p_pos),
                buffer_entry(2, &keys),
                buffer_entry(3, &identities),
            ],
        });
        let sorter = RadixSortKernel::new(Arc::new(device.clone()), Arc::new(queue.clone()), 256)?;
        let sort = if parallel {
            sorter.prepare_parallel(&keys, &identities, capacity as usize)?
        } else {
            sorter.prepare(&keys, &identities, capacity as usize)?
        };
        Ok(Self {
            capacity,
            parallel,
            keys,
            identities,
            params,
            bindings,
            pipeline,
            sort,
        })
    }

    pub fn upload(&self, queue: &wgpu::Queue, cfg: &MpmConfig, count: u32, cells: [i32; 3]) {
        let words = self.parameter_words(cfg, count, cells);
        molla_core::gpu_write_buffer!(queue, &self.params, 0, bytemuck::cast_slice(&words));
    }

    pub(super) fn parameter_words(
        &self,
        cfg: &MpmConfig,
        count: u32,
        cells: [i32; 3],
    ) -> [u32; 12] {
        [
            count,
            self.capacity,
            cfg.grid_size,
            cfg.grid_size.div_ceil(4),
            cfg.grid_origin[0].to_bits(),
            cfg.grid_origin[1].to_bits(),
            cfg.grid_origin[2].to_bits(),
            (1.0 / cfg.cell_size).to_bits(),
            cells[0] as u32,
            cells[1] as u32,
            cells[2] as u32,
            0,
        ]
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        materials: &wgpu::Buffer,
        offset: u64,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MPM particle tile keys"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.dispatch_workgroups(self.capacity.div_ceil(64), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        self.sort.encode(encoder);
        encoder.copy_buffer_to_buffer(
            &self.identities,
            0,
            materials,
            offset,
            u64::from(self.capacity) * 4,
        );
    }

    fn storage_bytes(&self) -> u64 {
        self.keys.size()
            + 2 * self.identities.size()
            + self.params.size()
            + self.sort.scratch_bytes()
    }
}

impl MpmSimulation {
    /// Prepare fixed-capacity stable tile indexing outside stepping, preserving material history.
    /// The queue must belong to this simulation's device. Preparation does not enable ordering.
    pub fn prepare_particle_tile_ordering(
        &mut self,
        queue: &wgpu::Queue,
        capacity: u32,
    ) -> Result<()> {
        self.prepare_particle_tile_ordering_impl(queue, capacity, false)
    }

    /// Prepare the parallel stable radix variant without enabling particle ordering.
    /// The queue must belong to this simulation's device.
    pub fn prepare_parallel_particle_tile_ordering(
        &mut self,
        queue: &wgpu::Queue,
        capacity: u32,
    ) -> Result<()> {
        self.prepare_particle_tile_ordering_impl(queue, capacity, true)
    }

    fn prepare_particle_tile_ordering_impl(
        &mut self,
        queue: &wgpu::Queue,
        capacity: u32,
        parallel: bool,
    ) -> Result<()> {
        let limits = self.device.limits();
        let material_bytes = u64::from(self.cfg.max_particles.max(1)) * 4;
        let total = material_bytes + u64::from(capacity) * 4;
        if capacity == 0
            || capacity < self.particle_count
            || capacity > self.cfg.max_particles
            || capacity.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || total > limits.max_storage_buffer_binding_size
            || total > limits.max_buffer_size
        {
            return Err(Error::Solver(
                "MPM particle ordering capacity is invalid".into(),
            ));
        }
        if self
            .particle_order
            .as_ref()
            .is_some_and(|order| order.capacity == capacity && order.parallel == parallel)
        {
            return Ok(());
        }
        if self.use_deterministic_p2g {
            return Err(Error::Solver(
                "Disable deterministic P2G before replacing particle ordering".into(),
            ));
        }
        let order = ParticleOrder::new(self, queue, capacity, parallel)?;
        let materials = molla_core::gpu_create_buffer!(
            self.device,
            &wgpu::BufferDescriptor {
                label: Some("MPM material identities and particle order"),
                size: total,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MPM indexed particles"),
            layout: &self.layout_particles,
            entries: &[
                buffer_entry(0, &self.p_pos),
                buffer_entry(1, &self.p_vel),
                buffer_entry(2, &self.p_c),
                buffer_entry(3, &self.p_f),
                buffer_entry(4, &self.p_volume),
                buffer_entry(5, &self.p_mass),
                buffer_entry(6, &self.p_jp),
                buffer_entry(7, &materials),
                buffer_entry(8, &self.material_table),
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.p_material, 0, &materials, 0, material_bytes);
        queue.submit([encoder.finish()]);
        self.p_material = materials;
        self.bg_particles = bindings;
        self.particle_order = Some(order);
        self.deterministic_p2g = None;
        self.invalidate_prepared_steps();
        Ok(())
    }

    /// Enable prepared P2G indexing without moving particle, material, or history arrays.
    pub fn set_particle_tile_ordering(&mut self, enabled: bool) -> Result<()> {
        if !enabled && self.use_deterministic_p2g {
            return Err(Error::Solver(
                "Disable deterministic P2G before particle ordering".into(),
            ));
        }
        if enabled
            && !self
                .particle_order
                .as_ref()
                .is_some_and(|o| o.capacity >= self.particle_count)
        {
            return Err(Error::Solver(
                "MPM particle ordering must be prepared for the live count".into(),
            ));
        }
        self.use_particle_order = enabled;
        self.invalidate_prepared_steps();
        Ok(())
    }

    pub fn particle_tile_ordering_enabled(&self) -> bool {
        self.use_particle_order
    }

    /// Whether the retained particle index uses the parallel sort variant.
    pub fn particle_tile_ordering_uses_parallel_sort(&self) -> bool {
        self.particle_order
            .as_ref()
            .is_some_and(|order| order.parallel)
    }

    /// Retained ordering bytes, including material-tail indices, sort scratch and uniforms.
    pub fn particle_tile_ordering_storage_bytes(&self) -> u64 {
        self.particle_order
            .as_ref()
            .map_or(0, ParticleOrder::storage_bytes)
    }

    pub(super) fn encode_particle_order(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        profiled: bool,
        step: usize,
    ) {
        if self.use_particle_order {
            self.profile_begin_encoder(
                profiled,
                encoder,
                MpmProfileStage::ParticleOrdering,
                Some(step),
            );
            self.particle_order.as_ref().unwrap().encode(
                encoder,
                &self.p_material,
                u64::from(self.cfg.max_particles.max(1)) * 4,
            );
            self.profile_end_encoder(profiled, encoder);
        }
    }
}
