//! Rust dispatch wrapper around `shaders/jacobian.wgsl` — spatial body
//! Jacobian for one or more (articulation, target_body, point_local)
//! requests in a single dispatch.
//!
//! Output layout per request: `6 × N` row-major (`[linear; angular]` per
//! column), packed in a flat `f32` buffer with caller-provided offsets.

use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_sim::{Model, State};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct JacobianRequest {
    pub articulation: u32,
    pub target_body: u32,
    /// Offset (in `f32` elements) into the output buffer where this
    /// request's `6 × N` Jacobian column starts.
    pub output_offset: u32,
    /// Offset (in transforms) into `body_q` where this request's FK
    /// snapshot starts. Use 0 to bind the engine's primary state.
    pub body_q_offset: u32,
    pub point_local: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    request_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64;

pub struct JacobianKernel {
    pipeline: wgpu::ComputePipeline,
    layout_model: wgpu::BindGroupLayout,
    layout_state: wgpu::BindGroupLayout,
    layout_arts: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    art_offsets_buf: wgpu::Buffer,
    /// Cumulative DOF per articulation (length = articulation_count + 1).
    pub dof_offsets: Vec<u32>,
}

impl JacobianKernel {
    pub fn new(engine: &Engine, model: &Model) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("JacobianKernel: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("JacobianKernel: no GPU queue".into()))?
            .clone();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::jacobian"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/jacobian.wgsl").into()),
        });

        let layout_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::jacobian model"),
            entries: &storage_entries(6, true),
        });
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::jacobian state"),
            entries: &storage_entries(1, true),
        });
        let layout_arts = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::jacobian arts"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::jacobian uniforms"),
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
            label: Some("molla::jacobian layout"),
            bind_group_layouts: &[&layout_model, &layout_state, &layout_arts, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::jacobian pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("jacobian_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Articulation CSR.
        let joint_count = model.joint_count as u32;
        let art_offsets: Vec<u32> = if model.articulation_count == 0 {
            vec![0u32, joint_count]
        } else {
            model
                .articulation_start
                .host()
                .map(|s| s.to_vec())
                .unwrap_or_else(|_| vec![0u32, joint_count])
        };

        // Cumulative DOF offsets per articulation.
        let joint_dof_offset = model
            .joint_dof_offset
            .host()
            .map_err(|e| Error::Gpu(format!("jacobian: joint_dof_offset host: {e}")))?;
        let mut dof_offsets = Vec::with_capacity(art_offsets.len());
        let mut acc: u32 = 0;
        for a in 0..(art_offsets.len() - 1) {
            dof_offsets.push(acc);
            let j_b = art_offsets[a] as usize;
            let j_e = art_offsets[a + 1] as usize;
            acc = acc.saturating_add((joint_dof_offset[j_e] - joint_dof_offset[j_b]) as u32);
        }
        dof_offsets.push(acc);

        let art_offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::jacobian art_offsets"),
            size: (art_offsets.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&art_offsets_buf, 0, bytemuck::cast_slice(&art_offsets));

        Ok(Self {
            pipeline,
            layout_model,
            layout_state,
            layout_arts,
            layout_uniforms,
            art_offsets_buf,
            dof_offsets,
        })
    }

    /// DOF count for articulation `a`.
    pub fn dof_width(&self, a: usize) -> usize {
        (self.dof_offsets[a + 1] - self.dof_offsets[a]) as usize
    }

    /// Allocate an output buffer big enough for `total_floats` entries.
    pub fn create_output_buffer(&self, device: &wgpu::Device, total_floats: usize) -> wgpu::Buffer {
        let len = total_floats.max(1);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::jacobian out"),
            size: (len * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Dispatch a batch of Jacobian requests using `state.body_q` as the
    /// FK source.
    pub fn dispatch(
        &self,
        engine: &Engine,
        model: &Model,
        state: &State,
        requests: &[JacobianRequest],
        jacobian_out: &wgpu::Buffer,
    ) -> Result<()> {
        let body_q = state
            .body_q
            .device_buffer()
            .ok_or_else(|| Error::Gpu("jacobian: state.body_q has no device mirror".into()))?;
        self.dispatch_with_body_q(engine, model, body_q, requests, jacobian_out)
    }

    /// Dispatch a batch of Jacobian requests using a caller-supplied
    /// `body_q` buffer (typically a multi-seed concatenation of FK
    /// snapshots). Each request's `body_q_offset` indexes into that
    /// buffer in transform-sized strides.
    pub fn dispatch_with_body_q(
        &self,
        engine: &Engine,
        model: &Model,
        body_q: &wgpu::Buffer,
        requests: &[JacobianRequest],
        jacobian_out: &wgpu::Buffer,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("JacobianKernel::dispatch: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("JacobianKernel::dispatch: no queue".into()))?
            .clone();

        if requests.is_empty() {
            return Ok(());
        }

        let req_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::jacobian requests"),
            size: std::mem::size_of_val(requests) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&req_buf, 0, bytemuck::cast_slice(requests));

        let bg_model = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::jacobian model bg"),
            layout: &self.layout_model,
            entries: &[
                buffer_entry(0, model.joint_type.device_buffer())?,
                buffer_entry(1, model.joint_parent.device_buffer())?,
                buffer_entry(2, model.joint_child.device_buffer())?,
                buffer_entry(3, model.joint_x_p.device_buffer())?,
                buffer_entry(4, model.joint_axis.device_buffer())?,
                buffer_entry(5, model.joint_dof_offset.device_buffer())?,
            ],
        });

        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::jacobian state bg"),
            layout: &self.layout_state,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: body_q.as_entire_binding(),
            }],
        });

        let bg_arts = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::jacobian arts bg"),
            layout: &self.layout_arts,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.art_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: req_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: jacobian_out.as_entire_binding(),
                },
            ],
        });

        let uniforms = Uniforms {
            request_count: requests.len() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::jacobian uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::jacobian uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::jacobian encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::jacobian pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_model, &[]);
            pass.set_bind_group(1, &bg_state, &[]);
            pass.set_bind_group(2, &bg_arts, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            let groups = (requests.len() as u32).div_ceil(WORKGROUP_SIZE).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Copy the full output buffer to host as `Vec<f32>`.
    pub fn read_back(
        &self,
        engine: &Engine,
        jacobian_out: &wgpu::Buffer,
        len: usize,
    ) -> Result<Vec<f32>> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("JacobianKernel::read_back: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("JacobianKernel::read_back: no queue".into()))?
            .clone();
        let len = len.max(1);
        let byte_size = (len * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::jacobian staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::jacobian readback encoder"),
        });
        encoder.copy_buffer_to_buffer(jacobian_out, 0, &staging, 0, byte_size);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("jacobian poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("jacobian recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("jacobian map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        Ok(out)
    }
}

/// One-shot single-request Jacobian compute. Convenience wrapper.
pub fn jacobian_gpu(
    engine: &Engine,
    model: &Model,
    state: &State,
    articulation: usize,
    target_body: usize,
    point_local: Vec3,
    out_jacobian: &mut [Real],
) -> Result<()> {
    let kernel = JacobianKernel::new(engine, model)?;
    let dof = kernel.dof_width(articulation);
    if out_jacobian.len() != 6 * dof {
        return Err(Error::Build(format!(
            "jacobian_gpu: out len {} != 6 * dof_total {}",
            out_jacobian.len(),
            6 * dof
        )));
    }
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("jacobian_gpu: no device".into()))?
        .clone();
    let req = JacobianRequest {
        articulation: articulation as u32,
        target_body: target_body as u32,
        output_offset: 0,
        body_q_offset: 0,
        point_local: [
            point_local.x as f32,
            point_local.y as f32,
            point_local.z as f32,
            0.0,
        ],
    };
    let total = 6 * dof;
    let buf = kernel.create_output_buffer(&device, total);
    kernel.dispatch(engine, model, state, &[req], &buf)?;
    let host = kernel.read_back(engine, &buf, total)?;
    for (i, v) in out_jacobian.iter_mut().enumerate() {
        *v = host[i] as Real;
    }
    Ok(())
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf =
        buf.ok_or_else(|| Error::Gpu(format!("jacobian binding {binding}: no device mirror")))?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    })
}

fn storage_entries(count: u32, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count).map(|i| storage_entry(i, read_only)).collect()
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
