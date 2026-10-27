//! Deterministic dense-grid contact candidates with a frozen-constraint certificate.

use crate::soil_rate_encoding::{RateEncoding, candidate_source};
use molla_compute::Engine;
use molla_core::{Error, Result};

#[path = "soil_grid_index_gpu.rs"]
mod grid_index;
pub(crate) use grid_index::SoilCandidateGridIndex;

pub const CANDIDATE_INPUT: u32 = 256;
pub const CANDIDATE_CAPACITY: u32 = 512;
pub const CANDIDATE_MULTIPLE_BODIES: u32 = 1024;
pub const CANDIDATE_COVERAGE: u32 = 2048;

pub struct SoilCandidateInputs<'a> {
    /// Dense cubic grid: free velocity XYZ and nonnegative mass W.
    pub grid: &'a wgpu::Buffer,
    pub centers: &'a wgpu::Buffer,
    pub shapes: &'a wgpu::Buffer,
    pub jacobians: &'a wgpu::Buffer,
    pub corrected_rates: &'a wgpu::Buffer,
    pub corrected_velocities: &'a wgpu::Buffer,
}

pub struct SoilCandidateOutputs<'a> {
    pub queries: &'a wgpu::Buffer,
    /// Sorted grid indices, with u32::MAX in inactive slots.
    pub node_ids: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct SoilCandidateConfig {
    pub origin: [f32; 3],
    pub cell_size: f32,
    pub grid_size: u32,
    pub bodies: u32,
    pub dofs: u32,
    pub capacity: u32,
    pub dt: f32,
    /// Nonnegative world-space candidate envelope, subject to post-solve coverage.
    pub search_distance: f32,
    pub max_owned_bytes: u64,
}

/// Encodes candidates, then certifies all frozen body/node normal constraints.
/// The caller clears shared status, encodes candidates, geometry and the floor
/// solve, then encodes certification/publication without changing input state.
/// Machine publication and frame rollback remain external. Multiple candidates
/// for one node fault instead of selecting an owner or duplicating its mass.
/// All buffers share the Engine device; writable bindings do not alias others.
pub struct ExperimentalSoilCandidatesGpu {
    grid_index: Option<grid_index::GridIndex>,
    pipelines: [wgpu::ComputePipeline; 7],
    group: wgpu::BindGroup,
    count: wgpu::Buffer,
    params: wgpu::Buffer,
    dt: f32,
    dispatches: [u32; 7],
    bytes: u64,
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("soil candidates: {message}"))
}

pub(crate) fn validate_node_window(
    origin: [f32; 3],
    h: f32,
    size: u32,
    cells: [i32; 3],
) -> Result<()> {
    let limit = (1_i64 << 23) - 1;
    if size == 0 || !h.is_finite() || h <= 0.0 {
        return Err(invalid("invalid node window spacing or extent"));
    }
    for axis in 0..3 {
        let lo = i64::from(cells[axis]);
        let hi = lo + i64::from(size) - 1;
        if lo < -limit
            || hi > limit
            || !origin[axis].is_finite()
            || [lo, hi]
                .into_iter()
                .any(|i| !(origin[axis] + i as f32 * h).is_finite())
        {
            return Err(invalid(
                "node window exceeds finite half-cell coordinate range",
            ));
        }
    }
    Ok(())
}

impl ExperimentalSoilCandidatesGpu {
    pub fn indexed_sort_backend(&self) -> Option<&'static str> {
        self.grid_index.as_ref().map(|index| index.backend())
    }
    pub(crate) fn grid_storage_bytes(cells: u32, indexed: bool) -> u64 {
        20 * u64::from(cells)
            + 4 * u64::from(cells).div_ceil(256)
            + if indexed {
                grid_index::GridIndex::bytes(cells)
            } else {
                0
            }
    }

    pub fn new(
        engine: &Engine,
        input: SoilCandidateInputs<'_>,
        output: SoilCandidateOutputs<'_>,
        config: SoilCandidateConfig,
    ) -> Result<Self> {
        Self::with_rate_encoding(engine, input, output, config, RateEncoding::Scalar)
    }

    pub(crate) fn with_rate_encoding(
        engine: &Engine,
        input: SoilCandidateInputs<'_>,
        output: SoilCandidateOutputs<'_>,
        config: SoilCandidateConfig,
        encoding: RateEncoding,
    ) -> Result<Self> {
        Self::with_grid_index(engine, input, output, config, encoding, None)
    }

    pub(crate) fn with_grid_index(
        engine: &Engine,
        input: SoilCandidateInputs<'_>,
        output: SoilCandidateOutputs<'_>,
        config: SoilCandidateConfig,
        encoding: RateEncoding,
        index: Option<SoilCandidateGridIndex<'_>>,
    ) -> Result<Self> {
        Self::with_grid_window(engine, input, output, config, encoding, index, [0; 3])
    }

    pub(crate) fn with_grid_window(
        engine: &Engine,
        input: SoilCandidateInputs<'_>,
        output: SoilCandidateOutputs<'_>,
        config: SoilCandidateConfig,
        encoding: RateEncoding,
        index: Option<SoilCandidateGridIndex<'_>>,
        cells: [i32; 3],
    ) -> Result<Self> {
        engine.check_device()?;
        validate_node_window(config.origin, config.cell_size, config.grid_size, cells)?;
        let device = engine.device().ok_or_else(|| invalid("GPU required"))?;
        if config.grid_size < 5
            || (index.is_none() && config.grid_size > 128)
            || config.grid_size.checked_pow(3).is_none()
            || !(1..=64).contains(&config.bodies)
            || !(1..=32).contains(&config.dofs)
            || !(1..=16384).contains(&config.capacity)
            || !config.dt.is_finite()
            || config.dt <= 0.0
            || !config.cell_size.is_finite()
            || config.cell_size <= 0.0
            || !config.search_distance.is_finite()
            || config.search_distance < 0.0
            || config.origin.iter().any(|v| {
                !v.is_finite() || !(*v + config.grid_size as f32 * config.cell_size).is_finite()
            })
        {
            return Err(invalid(
                "invalid grid, time, search or capacity configuration",
            ));
        }
        let total = if let Some(index) = index {
            index
                .tile_capacity
                .checked_mul(64)
                .filter(|n| *n > 0)
                .ok_or_else(|| invalid("index capacity overflow"))?
        } else {
            config.grid_size.pow(3)
        };
        let groups = total.div_ceil(256);
        let bytes = Self::grid_storage_bytes(total, index.is_some()) + 132;
        let limits = device.limits();
        if bytes > config.max_owned_bytes
            || 20 * u64::from(total) > limits.max_storage_buffer_binding_size
            || limits.max_storage_buffers_per_shader_stage < if index.is_some() { 14 } else { 12 }
            || limits.max_compute_workgroup_storage_size
                < if encoding == RateEncoding::Paired {
                    2048
                } else {
                    1024
                }
            || limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_size_x < 256
            || total.max(config.capacity).div_ceil(256)
                > limits.max_compute_workgroups_per_dimension
        {
            return Err(invalid("owned-buffer or device limit exceeded"));
        }
        let n = u64::from(config.dofs);
        let b = u64::from(config.bodies);
        let c = u64::from(config.capacity);
        for (buffer, required) in [
            input.grid,
            input.centers,
            input.shapes,
            input.jacobians,
            input.corrected_rates,
            input.corrected_velocities,
            output.queries,
            output.node_ids,
            output.status,
        ]
        .iter()
        .zip([
            16 * if index.is_some_and(|i| !i.sparse) {
                u64::from(config.grid_size).pow(3)
            } else {
                u64::from(total)
            },
            16 * b,
            64 * b,
            24 * b * n,
            encoding.bytes(n),
            16 * c,
            48 * c,
            4 * c,
            16,
        ]) {
            if buffer.size() < required
                || buffer.size() > limits.max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid("buffer capacity or usage mismatch"));
            }
        }
        let grid_index = index
            .map(|index| grid_index::GridIndex::new(engine, config, index, output.status))
            .transpose()?;
        let make = |label, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false
                }
            )
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let mapping = make(
            "soil grid candidate mapping",
            20 * u64::from(total),
            storage,
        );
        let offsets = make(
            "soil candidate block offsets",
            4 * u64::from(groups),
            storage,
        );
        let count = make(
            "soil candidate count",
            68,
            storage | wgpu::BufferUsages::COPY_SRC,
        );
        let params = make(
            "soil candidate parameters",
            64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let words = [
            config.origin[0].to_bits(),
            config.origin[1].to_bits(),
            config.origin[2].to_bits(),
            config.cell_size.to_bits(),
            config.grid_size,
            total,
            config.bodies,
            config.capacity,
            config.dt.to_bits(),
            config.search_distance.to_bits(),
            config.dofs,
            groups,
            cells[0] as u32,
            cells[1] as u32,
            cells[2] as u32,
            0,
        ];
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&words)
        );
        let source = include_str!("../shaders/soil_candidates.wgsl").replace(
            "// ANALYTIC_GEOMETRY",
            include_str!("../shaders/soil_contact_geometry_common.wgsl"),
        );
        let source = if grid_index.is_some() {
            source.replace("fn world_id(index:u32)->u32 { return index; }", "@group(0) @binding(13) var<storage,read> worlds:array<u32>;\n@group(0) @binding(14) var<storage,read> physical:array<u32>;\nfn world_id(index:u32)->u32 { return worlds[index]; }")
                .replace("fn physical_id(index:u32)->u32 { return index; }", "fn physical_id(index:u32)->u32 { return physical[index]; }")
        } else {
            source
        };
        let source = candidate_source(source, encoding);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("deterministic soil candidates"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let entries: Vec<_> = (0..if grid_index.is_some() { 15 } else { 13 })
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 12 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: (1..=5).contains(&binding) || binding >= 13,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil candidates"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soil candidates"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = [
            "reset",
            "validate",
            "classify",
            "scan_blocks",
            "scatter",
            "certify",
            "publish_grid",
        ]
        .map(|entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let mut buffers = vec![
            input.grid,
            input.centers,
            input.shapes,
            input.jacobians,
            input.corrected_rates,
            input.corrected_velocities,
            output.queries,
            output.node_ids,
            &mapping,
            &offsets,
            &count,
            output.status,
            &params,
        ];
        if let Some(index) = &grid_index {
            buffers.extend([&index.worlds, &index.physical]);
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil candidates"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            grid_index,
            pipelines,
            group,
            count,
            params,
            dt: config.dt,
            dispatches: [
                total.max(config.capacity).div_ceil(256),
                config.bodies.div_ceil(64),
                groups,
                1,
                groups,
                groups,
                groups,
            ],
            bytes,
        })
    }

    fn encode_range(&self, encoder: &mut wgpu::CommandEncoder, range: std::ops::Range<usize>) {
        for stage in range {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[stage]);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(self.dispatches[stage], 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
    }
    pub fn encode_candidates(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(index) = &self.grid_index {
            index.encode(encoder);
        }
        self.encode_range(encoder, 0..5);
    }
    /// Certifies every occupied node/body inequality before guarded grid publication.
    pub fn encode_certify_and_publish_grid(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_range(encoder, 5..7);
    }
    /// First u32 is the candidate count; the remaining 64 bytes record coverage failure.
    pub fn count_buffer(&self) -> &wgpu::Buffer {
        &self.count
    }
    pub(crate) fn parameters_buffer(&self) -> &wgpu::Buffer {
        &self.params
    }
    pub(crate) fn set_dt(&mut self, queue: &wgpu::Queue, dt: f32) -> Result<()> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(invalid("positive finite timestep required"));
        }
        if self.dt.to_bits() != dt.to_bits() {
            molla_core::gpu_write_buffer!(queue, &self.params, 32, bytemuck::bytes_of(&dt));
            self.dt = dt;
        }
        Ok(())
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
}

#[cfg(test)]
#[path = "soil_candidates_tests.rs"]
mod tests;
