//! Device-resident midpoint-friction impacts over supplied contact and scalar rows.

use molla_core::{Error, Result};

pub const IMPACT_FAULT_INPUT: u32 = 1 << 25;
pub const IMPACT_FAULT_NONFINITE: u32 = 1 << 26;
pub const IMPACT_FAULT_PSD: u32 = 1 << 27;
pub const IMPACT_FAULT_CONVERGENCE: u32 = 1 << 28;

#[derive(Clone, Copy, Debug)]
pub struct ImpactGpuConfig {
    pub island_capacity: u32,
    pub blocks_per_island: u32,
    pub max_iterations: u32,
    /// Stationarity tolerance beyond each normal impulse's half-ULP response.
    pub velocity_tolerance: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImpactGpuBlock {
    /// Contact coefficients/speed; intervals use lower/upper impulse bounds in xy and feedback diagonal in z.
    pub contact: [f32; 4],
    /// Kind (0 contact, 1 bilateral, 2 nonnegative, 3 interval), moving flag, two zeros.
    pub metadata: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImpactGpuSolution {
    /// Normal/tangent impulses for contacts; scalar impulse in x. Padding is zero.
    pub impulse: [f32; 4],
    /// Final physical row velocities, not midpoint friction-law velocities.
    pub velocity: [f32; 4],
    /// Dynamic coefficient flag, midpoint sticking flag, iterations, residual bits.
    pub diagnostics: [u32; 4],
}

/// Each block occupies three rows; scalar blocks require two zero padding rows.
/// Islands use fixed square Gram slots with stride `3 * blocks_per_island`.
/// All counts and inputs remain device-owned. Nonzero status invalidates all outputs.
/// Contact normal speeds must be nonpositive and match the supplied incoming speed.
/// Friction capacity comes from the solved normal impulse, without a predictor offset.
/// This stage does not assemble response, select episodes or integrate positions.
pub struct ResidentImpactConstraints {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    solutions: wgpu::Buffer,
    diagnostics: wgpu::Buffer,
    config: ImpactGpuConfig,
}

impl ResidentImpactConstraints {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        gram: &wgpu::Buffer,
        free: &wgpu::Buffer,
        blocks: &wgpu::Buffer,
        counts: &wgpu::Buffer,
        island_count: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ImpactGpuConfig,
    ) -> Result<Self> {
        let invalid = || Error::Gpu("invalid impact capacity or buffers".into());
        let rows = u64::from(config.blocks_per_island) * 3;
        let islands = u64::from(config.island_capacity);
        let matrix_words = rows.checked_mul(rows).ok_or_else(invalid)?;
        let scratch_stride = matrix_words.checked_add(4 * rows + 1).ok_or_else(invalid)?;
        let scratch_words = scratch_stride.checked_mul(islands).ok_or_else(invalid)?;
        let limits = device.limits();
        if rows == 0
            || islands == 0
            || config.max_iterations == 0
            || !config.velocity_tolerance.is_normal()
            || config.velocity_tolerance <= 0.
            || limits.max_storage_buffers_per_shader_stage < 9
            || config.island_capacity.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || scratch_words > u32::MAX as u64
            || scratch_words * 4 > limits.max_storage_buffer_binding_size
            || scratch_words * 4 > limits.max_buffer_size
        {
            return Err(invalid());
        }
        let slots = rows / 3 * islands;
        for (buffer, bytes) in [
            (gram, matrix_words * islands * 4),
            (free, rows * islands * 4),
            (blocks, slots * 32),
            (counts, islands * 4),
            (island_count, 4),
            (status, 4),
        ] {
            if buffer.size() < bytes
                || buffer.size() > limits.max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid());
            }
        }
        let solutions = storage(device, "impact solutions", slots * 48);
        let diagnostics = storage(device, "impact diagnostics", islands * 16);
        let scratch = storage(device, "impact scratch", scratch_words * 4);
        let uniform = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("impact parameters"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            }
        );
        molla_core::gpu_mapped_upload!(uniform.slice(..)).copy_from_slice(bytemuck::cast_slice(&[
            config.island_capacity,
            config.blocks_per_island,
            config.max_iterations,
            config.velocity_tolerance.to_bits(),
            scratch_stride as u32,
            0,
            0,
            0,
        ]));
        uniform.unmap();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("impact layout"),
            entries: &(0..10)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 9 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 5,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                })
                .collect::<Vec<_>>(),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("midpoint impact constraints"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/impact_constraints.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("impact pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("midpoint impact constraints"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("solve_impacts"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffers = [
            gram,
            free,
            blocks,
            counts,
            island_count,
            status,
            &scratch,
            &solutions,
            &diagnostics,
            &uniform,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("impact bindings"),
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
            pipeline,
            group,
            solutions,
            diagnostics,
            config,
        })
    }

    /// Record a retained solve; caller-owned status remains sticky until reset.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("midpoint impact constraints"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.config.island_capacity.div_ceil(64), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    pub fn solution_buffer(&self) -> &wgpu::Buffer {
        &self.solutions
    }
    /// Per-island fault, iteration count, physical residual bits and reserved zero.
    pub fn diagnostic_buffer(&self) -> &wgpu::Buffer {
        &self.diagnostics
    }
    pub fn resource_counts(&self) -> [usize; 3] {
        [4, 1, 1]
    }
}

fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    )
}

#[cfg(test)]
#[path = "impact_constraints_gpu_tests.rs"]
mod tests;
