//! Bounded active-tile lists and GPU-generated grid dispatch dimensions.

use super::*;

impl MpmSimulation {
    pub(crate) fn grid_fault_metadata(&self) -> Option<(&wgpu::Buffer, u64)> {
        (!self.active_grid_needs_reset)
            .then_some((&self.active_tiles.list, self.active_tiles.metadata_offset))
    }
}

#[cfg(test)]
#[path = "mpm_tile_map_tests.rs"]
mod tests;

pub(super) struct ActiveTiles {
    pub list: wgpu::Buffer,
    pub indirect: wgpu::Buffer,
    pub(super) uniform: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    pipelines: [wgpu::ComputePipeline; 3],
    pub capacity: u32,
    pub map_capacity: u32,
    width: u32,
    max_groups: u32,
    pub(super) metadata_offset: u64,
    tile_ids_offset: u64,
}

impl ActiveTiles {
    pub fn new(
        device: &wgpu::Device,
        cfg: &MpmConfig,
        positions: &wgpu::Buffer,
        spheres: &wgpu::Buffer,
        storage_cells: u32,
        tile_capacity: Option<u32>,
    ) -> Result<Self> {
        let width = cfg.grid_size.div_ceil(4);
        let world_tiles = width
            .checked_pow(3)
            .ok_or_else(|| Error::Solver("MPM active tile count overflow".into()))?;
        let capacity = tile_capacity.unwrap_or(world_tiles);
        let sparse = tile_capacity.is_some();
        let map_capacity = if sparse {
            capacity
                .checked_mul(2)
                .and_then(u32::checked_next_power_of_two)
                .ok_or_else(|| Error::Solver("MPM tile hash capacity overflow".into()))?
        } else {
            world_tiles
        };
        let metadata_offset = u64::from(storage_cells) * 4;
        let tile_ids_offset =
            metadata_offset + 16 + u64::from(map_capacity) * if sparse { 8 } else { 4 };
        let map_words = u64::from(map_capacity) + u64::from(capacity);
        let bytes = metadata_offset + 16 + map_words * if sparse { 8 } else { 4 };
        let limits = device.limits();
        if capacity == 0
            || bytes > limits.max_storage_buffer_binding_size
            || bytes > limits.max_buffer_size
        {
            return Err(Error::Solver(
                "MPM active tile list exceeds device capacity".into(),
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
        let list = allocate(
            "MPM stiffness and active tile list",
            bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let indirect_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let indirect = allocate("MPM active tile dispatch", 16, indirect_usage);
        let uniform = allocate(
            "MPM active tile parameters",
            64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MPM active tile builder"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MPM active tile builder"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mpm_active_tiles.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MPM active tile builder"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ["mark_particles", "mark_rigids", "finalize_tiles"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("SPARSE_MAP", f64::from(sparse))],
                    ..Default::default()
                },
                cache: None,
            })
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MPM active tile builder"),
            layout: &layout,
            entries: &[
                buffer_entry(0, &uniform),
                buffer_entry(1, positions),
                buffer_entry(2, spheres),
                buffer_entry(3, &list),
                buffer_entry(4, &indirect),
            ],
        });
        Ok(Self {
            list,
            indirect,
            uniform,
            bindings,
            pipelines,
            capacity,
            map_capacity,
            width,
            max_groups: limits.max_compute_workgroups_per_dimension,
            metadata_offset,
            tile_ids_offset,
        })
    }

    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        cfg: &MpmConfig,
        particles: u32,
        rigids: u32,
        cells: [i32; 3],
    ) -> Result<()> {
        let words = self.parameter_words(cfg, particles, rigids, cells)?;
        molla_core::gpu_write_buffer!(queue, &self.uniform, 0, bytemuck::cast_slice(&words));
        Ok(())
    }

    pub(super) fn parameter_words(
        &self,
        cfg: &MpmConfig,
        particles: u32,
        rigids: u32,
        cells: [i32; 3],
    ) -> Result<[u32; 16]> {
        if particles.div_ceil(64) > self.max_groups || rigids > self.max_groups {
            return Err(Error::Solver(
                "MPM active tile builder dispatch exceeds device capacity".into(),
            ));
        }
        let words = [
            cfg.grid_size,
            self.width,
            self.capacity,
            self.max_groups,
            cfg.grid_origin[0].to_bits(),
            cfg.grid_origin[1].to_bits(),
            cfg.grid_origin[2].to_bits(),
            (1.0 / cfg.cell_size).to_bits(),
            particles,
            rigids,
            (self.metadata_offset / 4) as u32,
            self.map_capacity,
            cells[0] as u32,
            cells[1] as u32,
            cells[2] as u32,
            0,
        ];
        Ok(words)
    }

    pub fn encode_build(&self, encoder: &mut wgpu::CommandEncoder, particles: u32, rigids: u32) {
        encoder.clear_buffer(&self.list, self.metadata_offset, Some(4));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MPM active tile marking"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.bindings, &[]);
        if particles > 0 {
            pass.set_pipeline(&self.pipelines[0]);
            pass.dispatch_workgroups(particles.div_ceil(64), 1, 1);
        }
        if rigids > 0 {
            pass.set_pipeline(&self.pipelines[1]);
            pass.dispatch_workgroups(rigids, 1, 1);
        }
        pass.set_pipeline(&self.pipelines[2]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub fn stats(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<MpmActiveGridStats> {
        Ok(self.read_metadata(device, queue, false)?.0)
    }

    pub fn read_metadata(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        include_tiles: bool,
    ) -> Result<(MpmActiveGridStats, Vec<u32>)> {
        let tile_bytes = if include_tiles {
            u64::from(self.capacity) * 4
        } else {
            0
        };
        let stage = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("MPM active tile statistics"),
                size: 16 + tile_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.list, self.metadata_offset, &stage, 0, 16);
        if include_tiles {
            encoder.copy_buffer_to_buffer(&self.list, self.tile_ids_offset, &stage, 16, tile_bytes);
        }
        queue.submit([encoder.finish()]);
        let slice = stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(e.to_string()))?;
        rx.recv()
            .map_err(|e| Error::Gpu(e.to_string()))?
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let data = molla_core::gpu_readback!(slice);
        let words: &[u32] = bytemuck::cast_slice(&data);
        let stats = MpmActiveGridStats {
            active_tiles: words[0],
            tile_capacity: self.capacity,
            overflow: words[2] & 1 != 0,
            numerical_fault: words[2] & 2 != 0,
            motion_bound_exceeded: words[2] & 4 != 0,
            completed_substeps: words[3],
        };
        let ids = if include_tiles {
            words[4..4 + stats.active_tiles.min(self.capacity) as usize].to_vec()
        } else {
            Vec::new()
        };
        drop(data);
        stage.unmap();
        Ok((stats, ids))
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        count: None,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    }
}
