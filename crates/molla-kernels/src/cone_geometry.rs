//! Batched geodesic cone geometry evaluation with retained GPU bindings.

use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::{Quat, Vec3};
use molla_sim::{joint_control::SphericalConeLimits, spherical_cone};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConeGpuRequest {
    /// Relative quaternion expressed in the orthonormal cone basis.
    pub rotation: [f32; 4],
    pub primary: [f32; 4],
    pub transverse0: [f32; 4],
    pub transverse1: [f32; 4],
    pub angles: [f32; 2],
    /// Bit 0 enables an extent; bit 1 identifies an authored pi extent.
    pub enabled: [u32; 2],
}

impl ConeGpuRequest {
    pub fn new(limits: SphericalConeLimits, axis: Vec3, relative: Quat) -> Result<Self> {
        limits.validate()?;
        let [p, b0, b1] = spherical_cone::basis(axis)?;
        if !relative.length().is_finite() || relative.length() <= 1e-12 {
            return Err(Error::Gpu("invalid cone orientation".into()));
        }
        let angles = [limits.angle0, limits.angle1];
        if let Some(scale) = angles
            .into_iter()
            .flatten()
            .filter(|&a| a > 0.0)
            .reduce(f64::min)
            && ((scale as f32).powi(2) < f32::MIN_POSITIVE
                || angles
                    .into_iter()
                    .flatten()
                    .filter(|&a| a > 0.0)
                    .any(|a| ((scale / a) as f32).powi(2) < f32::MIN_POSITIVE))
        {
            return Err(Error::Gpu("cone extents exceed f32 precision".into()));
        }
        let vector = |v: Vec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
        let rotation = relative.normalize();
        let imaginary = Vec3::new(rotation.x, rotation.y, rotation.z);
        Ok(Self {
            rotation: [
                imaginary.dot(p) as f32,
                imaginary.dot(b0) as f32,
                imaginary.dot(b1) as f32,
                rotation.w as f32,
            ],
            primary: vector(p),
            transverse0: vector(b0),
            transverse1: vector(b1),
            angles: angles.map(|a| a.unwrap_or(0.0) as f32),
            enabled: angles.map(|a| match a {
                None => 0,
                Some(a) if a == std::f64::consts::PI => 3,
                Some(_) => 1,
            }),
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConeGpuGeometry {
    /// Swing coordinates, primary-axis angle, and violation in radians.
    pub swing: [f32; 4],
    /// Parent-frame angular Jacobian xyz and signed row error w.
    pub rows: [[f32; 4]; 3],
    /// Row count, bilateral row mask, status (0 valid, 1 reserved, 2 invalid), reserved.
    pub meta: [u32; 4],
}

pub struct ConeGeometryKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

pub struct ConeGeometryDispatch {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    count: u32,
}

impl ConeGeometryDispatch {
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}

impl ConeGeometryKernel {
    pub fn new(engine: &Engine) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("missing cone GPU device".into()))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cone geometry"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../shaders/cone_geometry.wgsl"),
                    include_str!("../shaders/cone_evaluate.wgsl")
                )
                .into(),
            ),
        });
        let entries: Vec<_> = (0..2)
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
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cone geometry"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cone geometry"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cone geometry"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("evaluate_cones"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self { pipeline, layout })
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        inputs: &wgpu::Buffer,
        outputs: &wgpu::Buffer,
        count: u32,
    ) -> Result<ConeGeometryDispatch> {
        let input_bytes = count as u64 * std::mem::size_of::<ConeGpuRequest>() as u64;
        let output_bytes = count as u64 * std::mem::size_of::<ConeGpuGeometry>() as u64;
        if count == 0
            || inputs.size() != input_bytes
            || outputs.size() < output_bytes
            || count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension
        {
            return Err(Error::Gpu("invalid cone geometry capacity".into()));
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cone geometry"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inputs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: outputs.as_entire_binding(),
                },
            ],
        });
        Ok(ConeGeometryDispatch {
            pipeline: self.pipeline.clone(),
            group,
            count,
        })
    }
}
