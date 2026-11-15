//! Resident batches of deformation-gradient energies and tangents.

use molla_core::{Error, Result};
use molla_math::{
    Mat3,
    constitutive::{ElasticMaterial, MaterialResponse},
};
use std::num::NonZeroU64;

pub const OUTPUT_FLOATS: usize = 100;
pub const SHADER_SOURCE: &str = concat!(
    include_str!("../shaders/constitutive.wgsl"),
    "\n",
    include_str!("../shaders/constitutive_batch.wgsl")
);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConstitutiveInput {
    values: [f32; 16],
}

impl ConstitutiveInput {
    pub fn new(
        material: ElasticMaterial,
        mut deformation: Mat3,
        dimensions: usize,
        projected: bool,
    ) -> Result<Self> {
        material.validate(dimensions)?;
        if dimensions == 2 {
            deformation.z_axis = molla_math::Vec3::ZERO;
        }
        let mut values = [0.0; 16];
        values[..9].copy_from_slice(&deformation.to_cols_array().map(|v| v as f32));
        values[9] = material.mu as f32;
        values[10] = material.lambda as f32;
        values[11] = material.model as u32 as f32;
        values[12] = dimensions as f32;
        values[13] = u32::from(projected) as f32;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Solver(
                "constitutive: input is not representable as f32".into(),
            ));
        }
        Ok(Self { values })
    }
}

pub fn decode_response(row: &[f32]) -> Result<MaterialResponse> {
    if row.len() != OUTPUT_FLOATS || row[1] != 0.0 || row.iter().any(|v| !v.is_finite()) {
        return Err(Error::Solver("constitutive: invalid GPU response".into()));
    }
    let mut gradient = [0.0; 9];
    for i in 0..9 {
        gradient[i] = f64::from(row[4 + i]);
    }
    let mut hessian = [[0.0; 9]; 9];
    for i in 0..9 {
        for k in 0..9 {
            hessian[i][k] = f64::from(row[16 + 9 * i + k]);
        }
    }
    Ok(MaterialResponse {
        energy: f64::from(row[0]),
        gradient: Mat3::from_cols_array(&gradient),
        hessian,
    })
}

pub struct ConstitutiveKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

pub struct ConstitutiveBindings {
    bindings: wgpu::BindGroup,
    count: u32,
}

impl ConstitutiveKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("constitutive"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("constitutive.layout"),
            entries: &(0..2)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding == 0,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("constitutive.pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("constitutive.batch"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("material_batch"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    pub fn bind(
        &self,
        device: &wgpu::Device,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: u32,
    ) -> Result<ConstitutiveBindings> {
        let sizes = [
            u64::from(count) * 64,
            u64::from(count) * OUTPUT_FLOATS as u64 * 4,
        ];
        let limits = device.limits();
        if count == 0 || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension {
            return Err(Error::Solver("constitutive: invalid dispatch count".into()));
        }
        for (buffer, size) in [input, output].into_iter().zip(sizes) {
            if buffer.size() < size
                || size > u64::from(limits.max_storage_buffer_binding_size)
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(Error::Solver(
                    "constitutive: invalid buffer capacity or usage".into(),
                ));
            }
        }
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("constitutive.bindings"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: input,
                        offset: 0,
                        size: NonZeroU64::new(sizes[0]),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: output,
                        offset: 0,
                        size: NonZeroU64::new(sizes[1]),
                    }),
                },
            ],
        });
        Ok(ConstitutiveBindings { bindings, count })
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, bindings: &ConstitutiveBindings) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("constitutive.batch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings.bindings, &[]);
        pass.dispatch_workgroups(bindings.count.div_ceil(64), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}
