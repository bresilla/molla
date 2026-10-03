//! Retained coupled Coulomb-disk minimization over device-counted contact islands.

use molla_core::{Error, Result};

pub const FRICTION_FAULT_INPUT: u32 = 1 << 16;
pub const FRICTION_FAULT_NONFINITE: u32 = 1 << 17;
pub const FRICTION_FAULT_PSD: u32 = 1 << 18;
pub const FRICTION_FAULT_CONVERGENCE: u32 = 1 << 19;

#[derive(Clone, Copy, Debug)]
pub struct CoulombDiskConfig {
    pub island_capacity: u32,
    pub contacts_per_island: u32,
    pub max_iterations: u32,
    /// Published-impulse KKT velocity limit; interval rows include feedback.
    pub velocity_tolerance: f32,
}

/// Static/dynamic impulse radii and pre-predictor tangential point velocity.
/// Interval scalar blocks encode `[lower, upper, response_diagonal, 0]` instead.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CoulombDiskInput {
    pub static_cap: f32,
    pub dynamic_cap: f32,
    pub initial_velocity: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CoulombDiskSolution {
    /// Disk high vector or scalar high/low impulse pair, selected by regime.
    /// Disk residues are stored after the primary records in the solution buffer.
    pub impulse: [f32; 2],
    /// Physical relative velocity without the feedback diagonal.
    pub slip: [f32; 2],
    pub applied_cap: f32,
    /// 0: inactive, 2: sticking, 3: sliding, 4: bilateral, 5: nonnegative, 6: interval.
    pub regime: u32,
    pub iterations: u32,
    /// Island maximum disk or scalar KKT velocity error, including feedback.
    pub residual: f32,
}

/// Minimizes 0.5*lambda^T*G*lambda + free^T*lambda over coupled circular disks.
/// Gram rows use stride `2*contacts_per_island`; islands occupy fixed square slots.
/// Interval metadata adds its feedback diagonal to the physical Gram during solving.
pub struct ResidentCoulombDisks {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    solution: wgpu::Buffer,
    diagnostics: wgpu::Buffer,
    config: CoulombDiskConfig,
}

impl ResidentCoulombDisks {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        gram: &wgpu::Buffer,
        free_velocity: &wgpu::Buffer,
        contacts: &wgpu::Buffer,
        contact_counts: &wgpu::Buffer,
        island_count: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: CoulombDiskConfig,
    ) -> Result<Self> {
        Self::with_block_kinds(
            device,
            gram,
            free_velocity,
            contacts,
            contact_counts,
            island_count,
            status,
            None,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_block_kinds(
        device: &wgpu::Device,
        gram: &wgpu::Buffer,
        free_velocity: &wgpu::Buffer,
        contacts: &wgpu::Buffer,
        contact_counts: &wgpu::Buffer,
        island_count: &wgpu::Buffer,
        status: &wgpu::Buffer,
        block_kinds: Option<&wgpu::Buffer>,
        config: CoulombDiskConfig,
    ) -> Result<Self> {
        let invalid = || Error::Gpu("invalid Coulomb-disk capacity or buffers".into());
        let limits = device.limits();
        let rows = u64::from(config.contacts_per_island) * 2;
        let islands = u64::from(config.island_capacity);
        let scratch_stride = rows
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(6 * rows))
            .ok_or_else(invalid)?;
        let scratch_words = scratch_stride.checked_mul(islands).ok_or_else(invalid)?;
        if rows == 0
            || islands == 0
            || config.max_iterations == 0
            || !config.velocity_tolerance.is_normal()
            || config.velocity_tolerance <= 0.0
            || config.island_capacity.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || limits.max_storage_buffers_per_shader_stage < 10
            || scratch_words > u32::MAX as u64
            || scratch_words * 4 > limits.max_storage_buffer_binding_size
        {
            return Err(invalid());
        }
        for (input, bytes) in [
            (gram, islands * rows * rows * 4),
            (free_velocity, islands * rows * 4),
            (contacts, islands * rows * 8),
            (contact_counts, islands * 4),
            (island_count, 4),
            (status, 4),
            (
                block_kinds.unwrap_or(contact_counts),
                if block_kinds.is_some() {
                    islands * rows * 2
                } else {
                    islands * 4
                },
            ),
        ] {
            if input.size() < bytes
                || input.size() > limits.max_storage_buffer_binding_size
                || !input.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid());
            }
        }
        let solution = storage(device, "Coulomb disk solutions", islands * rows * 20);
        let diagnostics = storage(device, "Coulomb disk diagnostics", islands * 16);
        let scratch = storage(device, "Coulomb disk scratch", scratch_words * 4);
        let uniform = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("Coulomb disk parameters"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            }
        );
        molla_core::gpu_mapped_upload!(uniform.slice(..)).copy_from_slice(bytemuck::cast_slice(&[
            config.island_capacity,
            config.contacts_per_island,
            config.max_iterations,
            config.velocity_tolerance.to_bits(),
            scratch_stride as u32,
            u32::from(block_kinds.is_some()),
            0,
            0,
        ]));
        uniform.unmap();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Coulomb disk layout"),
            entries: &(0..11)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 8 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 5 || binding == 10,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                })
                .collect::<Vec<_>>(),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coupled Coulomb disks"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/coulomb_disk.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Coulomb disk pipeline"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("coupled Coulomb disks"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("solve_disks"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffers = [
            gram,
            free_velocity,
            contacts,
            contact_counts,
            island_count,
            status,
            &scratch,
            &solution,
            &uniform,
            &diagnostics,
            block_kinds.unwrap_or(contact_counts),
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Coulomb disk bindings"),
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
            solution,
            diagnostics,
            config,
        })
    }

    /// Record one retained dispatch; input counts and all solve data stay on device.
    /// Status is caller-owned and sticky; trust every output only when status is zero.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("coupled Coulomb disks"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.config.island_capacity.div_ceil(64), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    /// Primary 32-byte records followed by two disk residue floats per configured slot.
    pub fn solution_buffer(&self) -> &wgpu::Buffer {
        &self.solution
    }

    /// Byte offset of two disk impulse residue floats per slot after primary records.
    /// Scalar slots have zero residues; their high/low impulse pair stays in the primary record.
    pub fn impulse_residue_offset(&self) -> u64 {
        u64::from(self.config.island_capacity) * u64::from(self.config.contacts_per_island) * 32
    }
    /// Per-island fault, iteration count, residual bits and phase; retained while status is nonzero.
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
#[path = "coulomb_disk_gpu_tests.rs"]
mod tests;
