//! Device merge of disjoint physical impact and persistent-friction responses.

use crate::contact_response_gpu::{ResidentContactResponse, ResidentImpactResponse};
use molla_compute::Engine;
use molla_core::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub struct ImpactMergeConfig {
    pub contact_rows: u32,
    pub scalar_rows: u32,
    pub dofs: u32,
}

/// Merges physical impulses and response increments; excludes predictor and position changes.
/// Encode after both source responses, then consume results only while status is zero.
/// Sources must share model, contact-row and scalar-row ordering.
pub struct ResidentImpactMerge {
    pipelines: [wgpu::ComputePipeline; 2],
    group: wgpu::BindGroup,
    impulses: wgpu::Buffer,
    delta: wgpu::Buffer,
    scalars: wgpu::Buffer,
    work: u32,
}

impl ResidentImpactMerge {
    pub fn new(
        engine: &Engine,
        impact: &ResidentImpactResponse,
        friction: &ResidentContactResponse,
        status: &wgpu::Buffer,
        config: ImpactMergeConfig,
    ) -> Result<Self> {
        let invalid = || Error::Gpu("invalid impact/friction merge buffers or dimensions".into());
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let total = config
            .contact_rows
            .checked_add(config.scalar_rows)
            .ok_or_else(invalid)?;
        let work = config
            .contact_rows
            .max(config.scalar_rows)
            .max(config.dofs)
            .max(1)
            .div_ceil(64);
        if config.contact_rows == 0
            || work > limits.max_compute_workgroups_per_dimension
            || limits.max_storage_buffers_per_shader_stage < 12
        {
            return Err(invalid());
        }
        let sources = [
            impact.world_impulse_buffer(),
            friction.world_impulse_buffer(),
            impact.velocity_delta_buffer(),
            friction.velocity_delta_buffer(),
            impact.scalar_solution_buffer(),
            friction.scalar_solution_buffer(),
            impact.islands().row_slots_buffer(),
            friction.islands().row_slots_buffer(),
        ];
        let bytes = [
            u64::from(config.contact_rows) * 16,
            u64::from(config.contact_rows) * 16,
            u64::from(config.dofs.max(1)) * 4,
            u64::from(config.dofs.max(1)) * 4,
            u64::from(config.scalar_rows.max(1)) * 32,
            u64::from(config.scalar_rows.max(1)) * 32,
            u64::from(total) * 8,
            u64::from(total) * 8,
        ];
        for (&source, bytes) in sources.iter().zip(bytes) {
            if source.size() != bytes
                || bytes > limits.max_storage_buffer_binding_size
                || !source.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid());
            }
        }
        if status.size() < 4
            || status.size() > limits.max_storage_buffer_binding_size
            || !status.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(invalid());
        }
        let output = |label, size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    mapped_at_creation: false,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                }
            )
        };
        let impulses = output("merged physical contact impulses", bytes[0]);
        let delta = output("merged physical response increments", bytes[2]);
        let scalars = output("merged physical scalar reactions", bytes[4]);
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("impact merge dimensions"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            }
        );
        molla_core::gpu_mapped_upload!(params.slice(..)).copy_from_slice(bytemuck::cast_slice(&[
            config.contact_rows,
            config.scalar_rows,
            config.dofs,
            0,
        ]));
        params.unmap();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("physical response merge"),
            entries: &(0..13)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 12 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 8,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                })
                .collect::<Vec<_>>(),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("physical response merge"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/impact_merge.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("physical response merge"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ["validate", "merge"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let buffers = [
            sources[0], sources[1], sources[2], sources[3], sources[4], sources[5], sources[6],
            sources[7], &impulses, &delta, &scalars, status, &params,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("physical response merge"),
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
            pipelines,
            group,
            impulses,
            delta,
            scalars,
            work,
        })
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for buffer in [&self.impulses, &self.delta, &self.scalars] {
            encoder.clear_buffer(buffer, 0, None);
        }
        for pipeline in &self.pipelines {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(self.work, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
    }

    pub fn world_impulse_buffer(&self) -> &wgpu::Buffer {
        &self.impulses
    }
    pub fn velocity_delta_buffer(&self) -> &wgpu::Buffer {
        &self.delta
    }
    pub fn scalar_solution_buffer(&self) -> &wgpu::Buffer {
        &self.scalars
    }
}
