//! Device-resident primitive contact generation for deformable surface nodes.

use molla_core::{Error, Result};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoftRigidShapeGpu {
    pub position: [f32; 4],
    pub rotation: [f32; 4],
    /// Ball radius; box half-extents; cylinder radius/half-height. Friction in w.
    pub data: [f32; 4],
    /// Kind (0 ball, 1 box, 2 cylinder), body index or u32::MAX, reserved, reserved.
    pub ids: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoftRigidContactGpu {
    pub normal_gap: [f32; 4],
    pub material: [f32; 4],
    /// Particle, body, shape, status (0 inactive, 1 contact, 2 invalid input).
    pub ids: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    nodes: u32,
    shapes: u32,
    particles: u32,
    bodies: u32,
    margin: f32,
    padding: [u32; 3],
}

pub struct SoftRigidContactBindings<'a> {
    pub positions: &'a wgpu::Buffer,
    pub radii: &'a wgpu::Buffer,
    pub surface: &'a wgpu::Buffer,
    pub shapes: &'a wgpu::Buffer,
    pub body_poses: &'a wgpu::Buffer,
    pub particle_count: u32,
    pub body_count: u32,
}

pub struct SoftRigidContactKernel {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    output: wgpu::Buffer,
    node_count: u32,
    shape_count: u32,
}

impl SoftRigidContactKernel {
    pub fn new(device: &wgpu::Device, node_count: u32, shape_count: u32) -> Result<Self> {
        let count = node_count
            .checked_mul(shape_count)
            .ok_or_else(|| Error::Build("FEM contact pair count overflow".into()))?;
        let bytes = count.max(1) as u64 * std::mem::size_of::<SoftRigidContactGpu>() as u64;
        if bytes > device.limits().max_storage_buffer_binding_size as u64
            || count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension
        {
            return Err(Error::Build("FEM contact pairs exceed GPU limits".into()));
        }
        let entries: Vec<_> = (0..7)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 6 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 5,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FEM primitive contact bindings"),
            entries: &entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FEM primitive contacts"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soft_rigid_contact.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FEM primitive contact layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FEM primitive contact gather"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("gather"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FEM contact rows"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Ok(Self {
            layout,
            pipeline,
            output,
            node_count,
            shape_count,
        })
    }

    pub fn rows(&self) -> &wgpu::Buffer {
        &self.output
    }
    pub fn row_count(&self) -> u32 {
        self.node_count * self.shape_count
    }

    /// Encode parallel gather from current device positions and rigid transforms without readback.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bindings: SoftRigidContactBindings<'_>,
        margin: f32,
    ) -> Result<()> {
        if !margin.is_finite() || margin < 0.0 {
            return Err(Error::Build("invalid FEM GPU contact margin".into()));
        }
        if self.row_count() == 0 {
            return Ok(());
        }
        let required = [
            (bindings.positions, bindings.particle_count as u64 * 16),
            (bindings.radii, bindings.particle_count as u64 * 4),
            (bindings.surface, self.node_count as u64 * 4),
            (bindings.shapes, self.shape_count as u64 * 64),
            (bindings.body_poses, bindings.body_count.max(1) as u64 * 32),
        ];
        if required.iter().any(|(buffer, size)| buffer.size() < *size) {
            return Err(Error::Build("undersized FEM GPU contact buffer".into()));
        }
        let params = Params {
            nodes: self.node_count,
            shapes: self.shape_count,
            particles: bindings.particle_count,
            bodies: bindings.body_count,
            margin,
            padding: [0; 3],
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FEM contact parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            bindings.positions,
            bindings.radii,
            bindings.surface,
            bindings.shapes,
            bindings.body_poses,
            &self.output,
            &uniform,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FEM contact gather"),
            layout: &self.layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FEM contact gather"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.row_count().div_ceil(64), 1, 1);
        Ok(())
    }
}
