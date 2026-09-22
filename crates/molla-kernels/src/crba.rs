//! Rust dispatch wrapper around `shaders/crba.wgsl` — joint-space mass
//! matrix `M(q)` per articulation, on the GPU.
//!
//! Mirrors `solvers/crba.rs::mass_matrix`. Per-articulation parallelism:
//! one workgroup invocation per chain. Output is a flat `f32` buffer of
//! `Σ N_a · N_a` entries (CSR-style, one block per articulation), where
//! `N_a` is articulation `a`'s generalized velocity count. The
//! per-articulation byte offsets are computed in
//! [`MassMatrixKernel::mass_offsets`].

use molla_compute::Engine;
use molla_core::{Error, Result};
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

/// GPU CRBA mass-matrix kernel. Holds the pipeline + cached
/// articulation CSR / mass-offset CSR for `model`.
pub struct MassMatrixKernel {
    pipeline: wgpu::ComputePipeline,
    layout_model: wgpu::BindGroupLayout,
    layout_arts: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    art_offsets_buf: wgpu::Buffer,
    mass_offsets_buf: wgpu::Buffer,
    /// Cumulative output offsets per articulation, length = `articulation_count + 1`.
    /// `mass_offsets_host[a + 1] - mass_offsets_host[a]` = `N_a²`.
    mass_offsets_host: Vec<u32>,
    articulation_count: u32,
}

impl MassMatrixKernel {
    pub fn new(engine: &Engine, model: &Model) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("MassMatrixKernel: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("MassMatrixKernel: no GPU queue".into()))?
            .clone();

        Self::new_on_device(&device, &queue, model)
    }

    /// Construct the mass-matrix kernel on the caller's device and queue.
    pub fn new_on_device(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
    ) -> Result<Self> {

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::crba"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../shaders/crba.wgsl"),
                    include_str!("../shaders/articulated_response_common.wgsl")
                )
                .into(),
            ),
        });

        let layout_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::crba model"),
            entries: &storage_entries(16, true),
        });
        let layout_arts = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::crba arts"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::crba uniforms"),
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
            label: Some("molla::crba layout"),
            bind_group_layouts: &[Some(&layout_model), Some(&layout_arts), Some(&layout_uniforms)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::crba pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("crba_step"),
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
        if art_offsets.windows(2).any(|r| r[1] - r[0] > 64)
            || model.joint_type.host()?.iter().any(|&k| k > 7)
        {
            return Err(Error::Gpu(
                "CRBA supports at most 64 joints per articulation and joint kinds 0..7".into(),
            ));
        }

        // Mass-matrix output offsets: each articulation owns `N²` floats.
        let joint_dof_offset = model
            .joint_dof_offset
            .host()
            .map_err(|e| Error::Gpu(format!("crba: joint_dof_offset host: {e}")))?;
        let mut mass_offsets_host = Vec::with_capacity(art_offsets.len());
        let mut acc: u32 = 0;
        for a in 0..(art_offsets.len() - 1) {
            mass_offsets_host.push(acc);
            let j_b = art_offsets[a] as usize;
            let j_e = art_offsets[a + 1] as usize;
            let n = (joint_dof_offset[j_e] - joint_dof_offset[j_b]) as u32;
            acc = n
                .checked_mul(n)
                .and_then(|tile| acc.checked_add(tile))
                .ok_or_else(|| Error::Gpu("CRBA output size overflow".into()))?;
        }
        mass_offsets_host.push(acc);

        let art_offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::crba art_offsets"),
            size: (art_offsets.len() * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&art_offsets_buf, 0, bytemuck::cast_slice(&art_offsets));

        let mass_offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::crba mass_offsets"),
            size: (mass_offsets_host.len() * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &mass_offsets_buf,
            0,
            bytemuck::cast_slice(&mass_offsets_host),
        );

        Ok(Self {
            pipeline,
            layout_model,
            layout_arts,
            layout_uniforms,
            art_offsets_buf,
            mass_offsets_buf,
            mass_offsets_host,
            articulation_count,
        })
    }

    /// Total floats needed in the output buffer (sum of `N_a²` over all articulations).
    pub fn output_len(&self) -> usize {
        *self.mass_offsets_host.last().unwrap_or(&0) as usize
    }

    /// Offset (in `f32` elements) of articulation `art`'s `M` block in
    /// the output buffer.
    pub fn mass_offset(&self, art: usize) -> u32 {
        self.mass_offsets_host[art]
    }

    /// Size of articulation `art`'s mass matrix as `(N, N²)`.
    pub fn mass_extent(&self, art: usize) -> (u32, u32) {
        let bytes = self.mass_offsets_host[art + 1] - self.mass_offsets_host[art];
        let n = (bytes as f64).sqrt() as u32;
        (n, bytes)
    }

    /// Allocate a wgpu output buffer sized for the full multi-articulation
    /// mass matrix output.
    pub fn create_output_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let len = self.output_len().max(1);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::crba mass_matrix"),
            size: (len * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Dispatch CRBA over every articulation. Caller's responsibility:
    /// `state.body_q` must reflect current `joint_q` (call `eval_fk_gpu`
    /// or upload host-side `body_q` first), and `mass_out` must be at
    /// least `output_len()` floats.
    pub fn dispatch(
        &self,
        engine: &Engine,
        model: &Model,
        state: &State,
        mass_out: &wgpu::Buffer,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("MassMatrixKernel::dispatch: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("MassMatrixKernel::dispatch: no queue".into()))?
            .clone();

        self.dispatch_on_device(&device, &queue, model, state, mass_out)
    }

    /// Dispatch on the device and queue used to construct this kernel.
    pub fn dispatch_on_device(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state: &State,
        mass_out: &wgpu::Buffer,
    ) -> Result<()> {

        if self.articulation_count == 0 {
            return Ok(());
        }
        if mass_out.size() < self.output_len().max(1) as u64 * 4 {
            return Err(Error::Gpu("CRBA output buffer is undersized".into()));
        }

        let bg_model = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::crba model bg"),
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
                buffer_entry(9, state.body_q.device_buffer())?,
                buffer_entry(10, model.joint_x_c.device_buffer())?,
                buffer_entry(11, model.joint_reversed.device_buffer())?,
                buffer_entry(12, model.joint_enabled.device_buffer())?,
                buffer_entry(13, model.body_flags.device_buffer())?,
                buffer_entry(14, model.joint_limit_lower.device_buffer())?,
                buffer_entry(15, model.joint_limit_upper.device_buffer())?,
            ],
        });

        let bg_arts = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::crba arts bg"),
            layout: &self.layout_arts,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.art_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.mass_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mass_out.as_entire_binding(),
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
            label: Some("molla::crba uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::crba uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::crba encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::crba pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_model, &[]);
            pass.set_bind_group(1, &bg_arts, &[]);
            pass.set_bind_group(2, &bg_uniforms, &[]);
            let groups = self.articulation_count.div_ceil(WORKGROUP_SIZE).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Read back the full multi-articulation mass matrix as a flat
    /// `Vec<f32>`. Used in tests; production code should bind the buffer
    /// to downstream kernels instead of copying through host.
    pub fn read_back(&self, engine: &Engine, mass_out: &wgpu::Buffer) -> Result<Vec<f32>> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("MassMatrixKernel::read_back: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("MassMatrixKernel::read_back: no queue".into()))?
            .clone();
        let len = self.output_len().max(1);
        let byte_size = (len * core::mem::size_of::<f32>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::crba staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::crba readback encoder"),
        });
        encoder.copy_buffer_to_buffer(mass_out, 0, &staging, 0, byte_size);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("crba poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("crba recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("crba map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        Ok(out)
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "crba binding {binding}: no device mirror — call upload_to_device first"
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
