//! GPU batched vertex-triangle CCD. Mirrors the CPU
//! [`vt_ccd`](crate::soft_ccd::vt_ccd): one thread per query returns the
//! earliest TOI in `[0, 1]`, or `None` for no hit.

use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_storage::{Buffer, BufferUsage};

/// One VT-CCD query: vertex and triangle positions at the start and end
/// of the timestep.
#[derive(Copy, Clone, Debug)]
pub struct VtCcdQuery {
    pub v_at_t0: Vec3,
    pub v_at_t1: Vec3,
    pub tri_at_t0: [Vec3; 3],
    pub tri_at_t1: [Vec3; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    counts: [u32; 4],
}

const WORKGROUP_SIZE: u32 = 64;

/// Reusable pipeline for the VT-CCD kernel.
pub struct SoftCcdKernel {
    pipeline: wgpu::ComputePipeline,
    layout_in: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl SoftCcdKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::soft_ccd"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/soft_ccd.wgsl").into()),
        });
        let layout_in = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::soft_ccd in"),
            entries: &storage_entries(3, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::soft_ccd out"),
            entries: &storage_entries(1, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::soft_ccd uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::soft_ccd layout"),
            bind_group_layouts: &[Some(&layout_in), Some(&layout_out), Some(&layout_uniforms)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::soft_ccd pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("vt_ccd"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_in,
            layout_out,
            layout_uniforms,
        }
    }
}

/// Run a batch of VT-CCD queries on the GPU. Entry `i` is the earliest
/// TOI for `queries[i]`, or `None` for no hit — matching
/// [`vt_ccd`](crate::soft_ccd::vt_ccd) within f32 tolerance.
pub fn vt_ccd_batch_gpu(
    engine: &molla_compute::Engine,
    queries: &[VtCcdQuery],
) -> Result<Vec<Option<Real>>> {
    if queries.is_empty() {
        return Ok(Vec::new());
    }
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("soft_ccd_gpu: engine has no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("soft_ccd_gpu: engine has no queue".into()))?
        .clone();

    let n = queries.len();
    let pad = |v: Vec3| {
        let g = v.as_vec3();
        [g.x, g.y, g.z, 0.0f32]
    };
    let mut verts: Vec<[f32; 4]> = Vec::with_capacity(n * 2);
    let mut t0: Vec<[f32; 4]> = Vec::with_capacity(n * 3);
    let mut t1: Vec<[f32; 4]> = Vec::with_capacity(n * 3);
    for query in queries {
        verts.push(pad(query.v_at_t0));
        verts.push(pad(query.v_at_t1));
        for p in query.tri_at_t0 {
            t0.push(pad(p));
        }
        for p in query.tri_at_t1 {
            t1.push(pad(p));
        }
    }
    let verts_buf = make_storage(
        &device,
        "molla::soft_ccd verts",
        bytemuck::cast_slice(&verts),
    );
    molla_core::gpu_write_buffer!(queue, &verts_buf, 0, bytemuck::cast_slice(&verts));
    let t0_buf = make_storage(&device, "molla::soft_ccd tri0", bytemuck::cast_slice(&t0));
    molla_core::gpu_write_buffer!(queue, &t0_buf, 0, bytemuck::cast_slice(&t0));
    let t1_buf = make_storage(&device, "molla::soft_ccd tri1", bytemuck::cast_slice(&t1));
    molla_core::gpu_write_buffer!(queue, &t1_buf, 0, bytemuck::cast_slice(&t1));

    let mut out = Buffer::from_vec(vec![-1.0 as Real; n], BufferUsage::StorageReadWrite);
    out.ensure_device(&device)?;
    out.upload(&queue)?;

    let kernel = SoftCcdKernel::new(&device);
    let uniforms = Uniforms {
        counts: [n as u32, 0, 0, 0],
    };
    dispatch(
        &device, &queue, &kernel, &verts_buf, &t0_buf, &t1_buf, &mut out, uniforms, n,
    )?;
    out.download(&device, &queue)?;

    Ok(out
        .host()?
        .iter()
        .map(|&toi| if toi < 0.0 { None } else { Some(toi) })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    kernel: &SoftCcdKernel,
    verts: &wgpu::Buffer,
    t0: &wgpu::Buffer,
    t1: &wgpu::Buffer,
    out: &mut Buffer<Real>,
    uniforms: Uniforms,
    n: usize,
) -> Result<()> {
    let bg_in = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::soft_ccd in bg"),
        layout: &kernel.layout_in,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: verts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: t0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: t1.as_entire_binding(),
            },
        ],
    });
    let out_buf = out
        .device_buffer()
        .ok_or_else(|| Error::Gpu("soft_ccd out: no device mirror".into()))?;
    let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::soft_ccd out bg"),
        layout: &kernel.layout_out,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });
    let uniform_buf = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("molla::soft_ccd uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    );
    molla_core::gpu_write_buffer!(queue, &uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::soft_ccd uniforms bg"),
        layout: &kernel.layout_uniforms,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let groups = (n as u32).div_ceil(WORKGROUP_SIZE).max(1);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::soft_ccd encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::soft_ccd pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bg_in, &[]);
        pass.set_bind_group(1, &bg_out, &[]);
        pass.set_bind_group(2, &bg_uniforms, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    out.mark_device_dirty();
    Ok(())
}

fn make_storage(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    let size = (data.len().max(16)) as u64;
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    )
}

fn storage_entries(count: u32, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}
