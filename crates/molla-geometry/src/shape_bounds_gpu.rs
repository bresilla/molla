//! Retained world AABB generation from device body poses.

use molla_core::{Error, Result};
use molla_math::{Real, Transform};
use molla_sim::{Model, ShapeGeometry};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Bounds {
    lo: [f32; 4],
    hi: [f32; 4],
    data: [f32; 4],
    mode: [u32; 4],
}

/// Device world bounds with compile-time local geometry bounds.
/// Rebuild after geometry, shape transforms, membership or bound buffers change.
/// Generic finite geometry uses transformed local boxes; planes remain unbounded.
/// Nonzero authored dimensions and margin must be representable as normal f32.
pub struct ResidentShapeBounds {
    pipeline: wgpu::ComputePipeline,
    inputs: Vec<wgpu::BindGroup>,
    output: wgpu::BindGroup,
    count: u32,
}

impl ResidentShapeBounds {
    /// Bind live poses, a 32-byte-per-shape AABB output and a sticky status word.
    /// Status bits match `NARROW_FAULT_INDEX` and `NARROW_FAULT_NONFINITE`.
    /// Output includes the margin and roundoff padding from transform magnitudes.
    pub fn new(
        device: &wgpu::Device,
        model: &Model,
        poses: &[&wgpu::Buffer],
        output: &wgpu::Buffer,
        status: &wgpu::Buffer,
        margin: Real,
    ) -> Result<Self> {
        let count = u32::try_from(model.shape_count).map_err(|_| invalid())?;
        let limits = device.limits();
        if count == 0
            || poses.is_empty()
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || u64::from(count) * 64 > limits.max_storage_buffer_binding_size
            || !margin.is_finite()
            || margin < 0.0
        {
            return Err(invalid());
        }
        let margin = scalar(margin)?;
        let members = checked(model.shape_body.device_buffer(), u64::from(count) * 4)?;
        let transforms = checked(model.shape_transform.device_buffer(), u64::from(count) * 32)?;
        checked(Some(output), u64::from(count) * 32)?;
        checked(Some(status), 4)?;
        for pose in poses {
            checked(Some(pose), model.body_count.max(1) as u64 * 32)?;
        }
        let mut local = Vec::with_capacity(count as usize);
        for (i, geom) in model.shape_geometry.iter().enumerate() {
            let mut row = Bounds {
                lo: [0.0; 4],
                hi: [0.0; 4],
                data: [0.0; 4],
                mode: [0; 4],
            };
            let (mode, values) = match *geom {
                ShapeGeometry::Plane { .. } => (1, [0.0; 3]),
                ShapeGeometry::Sphere { radius } => (2, [radius, 0.0, 0.0]),
                ShapeGeometry::Capsule {
                    half_height,
                    radius,
                } => (3, [half_height, radius, 0.0]),
                ShapeGeometry::Cylinder {
                    half_height,
                    radius,
                }
                | ShapeGeometry::Cone {
                    half_height,
                    radius,
                } => (4, [half_height, radius, 0.0]),
                ShapeGeometry::RoundCylinder {
                    half_height,
                    radius,
                    border_radius,
                } => (4, [half_height, radius, border_radius]),
                ShapeGeometry::Ellipsoid { half_axes } => (5, half_axes.to_array()),
                _ => (0, [0.0; 3]),
            };
            row.mode[0] = mode;
            if values.iter().any(|v| *v < 0.0) {
                return Err(invalid());
            }
            for (out, value) in row.data.iter_mut().zip(values) {
                *out = scalar(value)?;
            }
            if mode == 0 {
                let aabb =
                    crate::shape_world::shape_aabb_at_pose(model, i, Transform::IDENTITY, 0.0);
                if !aabb.min.is_finite() || !aabb.max.is_finite() || !aabb.min.cmple(aabb.max).all()
                {
                    return Err(invalid());
                }
                for axis in 0..3 {
                    row.lo[axis] = scalar(aabb.min[axis])?;
                    row.hi[axis] = scalar(aabb.max[axis])?;
                }
            }
            local.push(row);
        }
        if local.len() != count as usize {
            return Err(invalid());
        }
        let local_buffer = initialized(
            device,
            "resident local bounds",
            bytemuck::cast_slice(&local),
            wgpu::BufferUsages::STORAGE,
        );
        let uniform = initialized(
            device,
            "resident bounds params",
            bytemuck::cast_slice(&[count, margin.to_bits(), 0, 0]),
            wgpu::BufferUsages::UNIFORM,
        );
        let inputs_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident bounds input"),
            entries: &(0..4)
                .map(|i| entry(i, wgpu::BufferBindingType::Storage { read_only: true }))
                .collect::<Vec<_>>(),
        });
        let output_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident bounds output"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(2, wgpu::BufferBindingType::Uniform),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident shape bounds"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shape_bounds.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resident shape bounds"),
            bind_group_layouts: &[Some(&inputs_layout), Some(&output_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resident shape bounds"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("shape_bounds"),
            compilation_options: Default::default(),
            cache: None,
        });
        let inputs = poses
            .iter()
            .map(|pose| {
                group(
                    device,
                    &inputs_layout,
                    &[&local_buffer, members, transforms, pose],
                )
            })
            .collect();
        let output = group(device, &output_layout, &[output, status, &uniform]);
        Ok(Self {
            pipeline,
            inputs,
            output,
            count,
        })
    }

    /// Append one pass without allocation, submission, upload, status reset or readback.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, pose: usize) -> Result<()> {
        let inputs = self.inputs.get(pose).ok_or_else(invalid)?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resident shape bounds"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, inputs, &[]);
        pass.set_bind_group(1, &self.output, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
        Ok(())
    }

    /// Retained buffers, bind groups and pipelines, excluding caller inputs/outputs.
    pub fn resource_counts(&self) -> [usize; 3] {
        [2, self.inputs.len() + 1, 1]
    }
}

fn invalid() -> Error {
    Error::Gpu("invalid or unrepresentable resident shape bounds input".into())
}
fn scalar(value: Real) -> Result<f32> {
    let f = value as f32;
    if !f.is_finite() || (value != 0.0 && !f.is_normal()) {
        Err(invalid())
    } else {
        Ok(f)
    }
}
fn checked(buffer: Option<&wgpu::Buffer>, size: u64) -> Result<&wgpu::Buffer> {
    buffer
        .filter(|b| b.size() >= size && b.usage().contains(wgpu::BufferUsages::STORAGE))
        .ok_or_else(invalid)
}
fn initialized(
    device: &wgpu::Device,
    label: &str,
    data: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let buffer = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage,
            mapped_at_creation: true,
        }
    );
    molla_core::gpu_mapped_upload!(buffer.slice(..)).copy_from_slice(data);
    buffer.unmap();
    buffer
}
fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resident shape bounds"),
        layout,
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    })
}
