//! Rust dispatch wrapper around `shaders/rnea.wgsl` — joint-space
//! inverse dynamics (RNEA) per articulation.
//!
//! Mirrors `solvers/rnea.rs::inverse_dynamics`. Per-articulation
//! parallelism: one workgroup invocation per chain. Output: per-DOF
//! `tau` packed in CSR-style buffer (per-articulation offsets given by
//! the `dof_offsets` CSR). Input `qdd` follows the same layout.

use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::Real;
use molla_sim::{Model, State};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    articulation_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// GPU RNEA kernel. Holds the pipeline and cached articulation+DOF
/// CSR offsets so dispatch is the only per-call cost.
pub struct InverseDynamicsKernel {
    pipeline: wgpu::ComputePipeline,
    layout_model: wgpu::BindGroupLayout,
    layout_state: wgpu::BindGroupLayout,
    layout_arts: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    art_offsets_buf: wgpu::Buffer,
    dof_offsets_buf: wgpu::Buffer,
    dof_offsets_host: Vec<u32>,
    articulation_count: u32,
}

impl InverseDynamicsKernel {
    pub fn new(engine: &Engine, model: &Model) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("InverseDynamicsKernel: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("InverseDynamicsKernel: no GPU queue".into()))?
            .clone();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::rnea"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/rnea.wgsl").into()),
        });

        let layout_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::rnea model"),
            entries: &storage_entries(11, true),
        });
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::rnea state"),
            entries: &storage_entries(2, true),
        });
        let layout_arts = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::rnea arts"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::rnea uniforms"),
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
            label: Some("molla::rnea layout"),
            bind_group_layouts: &[
                Some(&layout_model),
                Some(&layout_state),
                Some(&layout_arts),
                Some(&layout_uniforms),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::rnea pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("rnea_step"),
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
        let articulation_count = (art_offsets.len().saturating_sub(1)) as u32;

        // Per-articulation DOF offsets (cumulative).
        let joint_dof_offset = model
            .joint_dof_offset
            .host()
            .map_err(|e| Error::Gpu(format!("rnea: joint_dof_offset host: {e}")))?;
        let mut dof_offsets_host = Vec::with_capacity(art_offsets.len());
        let mut acc: u32 = 0;
        for a in 0..(art_offsets.len() - 1) {
            dof_offsets_host.push(acc);
            let j_b = art_offsets[a] as usize;
            let j_e = art_offsets[a + 1] as usize;
            let n = (joint_dof_offset[j_e] - joint_dof_offset[j_b]) as u32;
            acc = acc.saturating_add(n);
        }
        dof_offsets_host.push(acc);

        let art_offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::rnea art_offsets"),
            size: (art_offsets.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&art_offsets_buf, 0, bytemuck::cast_slice(&art_offsets));

        let dof_offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::rnea dof_offsets"),
            size: (dof_offsets_host.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&dof_offsets_buf, 0, bytemuck::cast_slice(&dof_offsets_host));

        Ok(Self {
            pipeline,
            layout_model,
            layout_state,
            layout_arts,
            layout_uniforms,
            art_offsets_buf,
            dof_offsets_buf,
            dof_offsets_host,
            articulation_count,
        })
    }

    /// Total DOFs across all articulations.
    pub fn dof_total(&self) -> usize {
        *self.dof_offsets_host.last().unwrap_or(&0) as usize
    }

    /// Per-articulation DOF offset.
    pub fn dof_offset(&self, art: usize) -> usize {
        self.dof_offsets_host[art] as usize
    }

    /// Per-articulation DOF width.
    pub fn dof_width(&self, art: usize) -> usize {
        (self.dof_offsets_host[art + 1] - self.dof_offsets_host[art]) as usize
    }

    /// Allocate a `qdd` / `tau`-shaped scratch buffer.
    pub fn create_dof_buffer(&self, device: &wgpu::Device, label: &str) -> wgpu::Buffer {
        let len = self.dof_total().max(1);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Dispatch RNEA over every articulation.
    pub fn dispatch(
        &self,
        engine: &Engine,
        model: &Model,
        state: &State,
        qdd_in: &wgpu::Buffer,
        tau_out: &wgpu::Buffer,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("InverseDynamicsKernel::dispatch: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("InverseDynamicsKernel::dispatch: no queue".into()))?
            .clone();

        if self.articulation_count == 0 {
            return Ok(());
        }

        let bg_model = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::rnea model bg"),
            layout: &self.layout_model,
            entries: &[
                buffer_entry(0, model.joint_type.device_buffer())?,
                buffer_entry(1, model.joint_parent.device_buffer())?,
                buffer_entry(2, model.joint_child.device_buffer())?,
                buffer_entry(3, model.joint_x_p.device_buffer())?,
                buffer_entry(4, model.joint_axis.device_buffer())?,
                buffer_entry(5, model.joint_dof_offset.device_buffer())?,
                buffer_entry(6, model.body_mass.device_buffer())?,
                buffer_entry(7, model.body_inertia.device_buffer())?,
                buffer_entry(8, model.body_com.device_buffer())?,
                buffer_entry(9, model.body_world.device_buffer())?,
                buffer_entry(10, model.gravity.device_buffer())?,
            ],
        });

        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::rnea state bg"),
            layout: &self.layout_state,
            entries: &[
                buffer_entry(0, state.body_q.device_buffer())?,
                buffer_entry(1, state.joint_qd.device_buffer())?,
            ],
        });

        let bg_arts = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::rnea arts bg"),
            layout: &self.layout_arts,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.art_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.dof_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: qdd_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tau_out.as_entire_binding(),
                },
            ],
        });

        let uniforms = Uniforms {
            articulation_count: self.articulation_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::rnea uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::rnea uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::rnea encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::rnea pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_model, &[]);
            pass.set_bind_group(1, &bg_state, &[]);
            pass.set_bind_group(2, &bg_arts, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            let groups = self.articulation_count.div_ceil(WORKGROUP_SIZE).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Read the full multi-articulation `tau` buffer back to a host `Vec<f32>`.
    pub fn read_back(&self, engine: &Engine, tau_out: &wgpu::Buffer) -> Result<Vec<f32>> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("InverseDynamicsKernel::read_back: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("InverseDynamicsKernel::read_back: no queue".into()))?
            .clone();
        let len = self.dof_total().max(1);
        let byte_size = (len * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::rnea staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::rnea readback encoder"),
        });
        encoder.copy_buffer_to_buffer(tau_out, 0, &staging, 0, byte_size);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("rnea poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("rnea recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("rnea map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        Ok(out)
    }
}

/// One-shot RNEA: upload `qdd`, dispatch the kernel, read `tau` back.
/// Convenience wrapper for tests / single-articulation calls; production
/// code should hold the kernel + buffers across calls.
pub fn inverse_dynamics_gpu(
    engine: &Engine,
    model: &Model,
    state: &State,
    articulation: usize,
    qdd_in: &[Real],
    tau_out: &mut [Real],
) -> Result<()> {
    let kernel = InverseDynamicsKernel::new(engine, model)?;
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("inverse_dynamics_gpu: no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("inverse_dynamics_gpu: no queue".into()))?
        .clone();
    let dof_w = kernel.dof_width(articulation);
    if qdd_in.len() != dof_w || tau_out.len() != dof_w {
        return Err(Error::Build(format!(
            "inverse_dynamics_gpu: qdd/tau len {}/{} != articulation DOFs {dof_w}",
            qdd_in.len(),
            tau_out.len()
        )));
    }

    // Build a CSR-shaped qdd_in scratch buffer (zeros for other arts).
    let mut qdd_full = vec![0f32; kernel.dof_total().max(1)];
    let off = kernel.dof_offset(articulation);
    for (i, v) in qdd_in.iter().enumerate() {
        qdd_full[off + i] = *v as f32;
    }
    let qdd_buf = kernel.create_dof_buffer(&device, "molla::rnea qdd");
    let tau_buf = kernel.create_dof_buffer(&device, "molla::rnea tau");
    queue.write_buffer(&qdd_buf, 0, bytemuck::cast_slice(&qdd_full));
    kernel.dispatch(engine, model, state, &qdd_buf, &tau_buf)?;
    let tau_full = kernel.read_back(engine, &tau_buf)?;
    for (i, v) in tau_out.iter_mut().enumerate() {
        *v = tau_full[off + i] as Real;
    }
    Ok(())
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "rnea binding {binding}: no device mirror — call upload_to_device first"
        ))
    })?;
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
