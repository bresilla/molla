use super::*;
use crate::kernels::{RadixSortDispatch, RadixSortKernel};

#[cfg(test)]
#[path = "soil_grid_index_tests.rs"]
mod tests;

#[derive(Clone, Copy)]
pub(crate) struct SoilCandidateGridIndex<'a> {
    pub metadata: &'a wgpu::Buffer,
    pub metadata_word: u32,
    pub tile_ids_word: u32,
    pub tile_capacity: u32,
    pub sparse: bool,
}

pub(super) struct GridIndex {
    pub worlds: wgpu::Buffer,
    pub physical: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    sort: RadixSortDispatch,
    capacity: u32,
    parallel: bool,
}

impl GridIndex {
    pub fn bytes(capacity: u32) -> u64 {
        Self::bytes_for(capacity, cfg!(feature = "parallel-candidate-index"))
    }

    fn bytes_for(capacity: u32, parallel: bool) -> u64 {
        16 * u64::from(capacity)
            + 1024 * u64::from(capacity).div_ceil(256)
            + 64
            + u64::from(parallel) * 1024
    }

    pub(super) fn backend(&self) -> &'static str {
        if self.parallel {
            "parallel-stable-radix"
        } else {
            "serial-stable-radix"
        }
    }

    pub fn new(
        engine: &Engine,
        config: SoilCandidateConfig,
        index: SoilCandidateGridIndex<'_>,
        status: &wgpu::Buffer,
    ) -> Result<Self> {
        Self::build(
            engine,
            config,
            index,
            status,
            cfg!(feature = "parallel-candidate-index"),
        )
    }

    fn build(
        engine: &Engine,
        config: SoilCandidateConfig,
        index: SoilCandidateGridIndex<'_>,
        status: &wgpu::Buffer,
        parallel: bool,
    ) -> Result<Self> {
        let device = engine.device().ok_or_else(|| invalid("GPU required"))?;
        let capacity = index
            .tile_capacity
            .checked_mul(64)
            .ok_or_else(|| invalid("index capacity overflow"))?;
        let bytes = u64::from(capacity) * 4;
        let limits = device.limits();
        let bins_bytes = 1024 * u64::from(capacity).div_ceil(256) + u64::from(parallel) * 1024;
        if capacity == 0
            || Self::bytes_for(capacity, parallel) > config.max_owned_bytes
            || capacity.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || bytes > limits.max_storage_buffer_binding_size
            || bytes > limits.max_buffer_size
            || bins_bytes > limits.max_storage_buffer_binding_size
            || bins_bytes > limits.max_buffer_size
            || (parallel
                && (limits.max_compute_workgroup_storage_size < 9216
                    || limits.max_compute_workgroups_per_dimension < 256))
            || u64::from(index.metadata_word) * 4 + 16 > index.metadata.size()
            || (u64::from(index.tile_ids_word) + u64::from(index.tile_capacity)) * 4
                > index.metadata.size()
            || !index.metadata.usage().contains(wgpu::BufferUsages::STORAGE)
            || index.metadata.size() > limits.max_storage_buffer_binding_size
        {
            return Err(invalid("active grid index capacity or metadata mismatch"));
        }
        let make = |label| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }
            )
        };
        let worlds = make("candidate world-cell identities");
        let physical = make("candidate physical-cell identities");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("active candidate grid index"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_grid_index.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("active candidate grid index"),
            layout: None,
            module: &shader,
            entry_point: Some("index_cells"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("GRID_SIZE", config.grid_size as f64),
                    ("CAPACITY", capacity as f64),
                    ("METADATA_WORD", index.metadata_word as f64),
                    ("TILE_IDS_WORD", index.tile_ids_word as f64),
                    ("SPARSE", f64::from(index.sparse)),
                ],
                ..Default::default()
            },
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("active candidate grid index"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[index.metadata, &worlds, &physical, status]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let sorter = RadixSortKernel::new(device.clone(), engine.queue().unwrap().clone(), 256)?;
        let sort = if parallel {
            sorter.prepare_parallel(&worlds, &physical, capacity as usize)?
        } else {
            sorter.prepare(&worlds, &physical, capacity as usize)?
        };
        assert_eq!(
            Self::bytes_for(capacity, parallel),
            2 * bytes + sort.scratch_bytes()
        );
        Ok(Self {
            worlds,
            physical,
            pipeline,
            group,
            sort,
            capacity,
            parallel,
        })
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(self.capacity.div_ceil(64), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        self.sort.encode(encoder);
    }
}
