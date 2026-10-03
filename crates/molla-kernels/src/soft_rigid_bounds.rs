//! Device-updated scalar position and velocity bounds for the coupled contact solve.

use crate::soft_rigid_contact::{SoftRigidContactGpu, SoftRigidDofBound};
use molla_core::{Error, Result};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoftRigidScalarBound {
    /// Generalized DOF, coordinate index, reserved, reserved.
    pub indices: [u32; 4],
    /// Position lower/upper bounds, nonnegative speed limit, reserved.
    pub limits: [f32; 4],
}

pub struct SoftRigidBoundsKernel {
    descriptors: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    count: u32,
    coordinates: u32,
}

impl SoftRigidBoundsKernel {
    pub fn new(
        device: &wgpu::Device,
        bounds: &[SoftRigidScalarBound],
        dofs: u32,
        coordinates: u32,
    ) -> Result<(Self, Vec<SoftRigidContactGpu>)> {
        if bounds.is_empty()
            || bounds.len() > 64
            || bounds.iter().any(|b| {
                b.indices[0] >= dofs
                    || b.indices[1] >= coordinates
                    || b.limits.iter().any(|v| !v.is_finite())
                    || b.limits[0] >= b.limits[1]
                    || b.limits[2] < 0.0
                    || b.limits[2] > 1e30
            })
        {
            return Err(Error::Build("invalid coupled scalar bounds".into()));
        }
        let rows = bounds
            .iter()
            .flat_map(|b| {
                [1.0, -1.0].map(|sign| {
                    SoftRigidDofBound {
                        dof: b.indices[0],
                        sign,
                        speed: 0.0,
                    }
                    .row(dofs)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let descriptors = molla_core::gpu_buffer_init!(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("FEM scalar bounds"),
                contents: bytemuck::cast_slice(bounds),
                usage: wgpu::BufferUsages::STORAGE,
            }
        );
        let entries: Vec<_> = (0..5)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 4 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 2,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FEM bounds"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FEM bounds"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soft_rigid_bounds.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FEM bounds"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_bounds"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok((
            Self {
                descriptors,
                layout,
                pipeline,
                count: bounds.len() as u32,
                coordinates,
            },
            rows,
        ))
    }

    pub fn row_count(&self) -> u32 {
        2 * self.count
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        coordinates: &wgpu::Buffer,
        contacts: &wgpu::Buffer,
        status: &wgpu::Buffer,
        base: u32,
        dt: f32,
    ) -> Result<()> {
        if !dt.is_finite()
            || dt <= 0.0
            || coordinates.size() < u64::from(self.coordinates) * 4
            || contacts.size() < (u64::from(base) + u64::from(self.row_count())) * 80
            || status.size() < 4
        {
            return Err(Error::Build(
                "invalid coupled bound buffers or timestep".into(),
            ));
        }
        let params = molla_core::gpu_buffer_init!(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[dt.to_bits(), base, self.count, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            }
        );
        let buffers = [&self.descriptors, coordinates, contacts, status, &params];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
        Ok(())
    }
}
