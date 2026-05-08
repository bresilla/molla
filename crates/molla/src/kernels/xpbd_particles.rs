//! Rust dispatch wrapper for `shaders/xpbd_particles.wgsl` —
//! single-thread GPU port of the cloth/rope/spring particle step
//! (predict + distance constraints + strain limit + plane collision
//! + velocity recovery). Replaces the host-side
//! `apply_particles_post_gpu` shim.

use crate::core::{Error, Result};
use crate::math::{Real, Vec3};
use crate::sim::{Model, State};
use crate::storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    particle_count: u32,
    spring_count: u32,
    iterations: u32,
    dt: f32,
    linear_damping: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// Owns the pipeline + per-substep scratch buffers (λ accumulator,
/// Jacobi deltas + counts). Sizes are bound to a specific
/// `(particle_count, spring_count)` — rebuild the kernel if either
/// changes.
pub struct XpbdParticlesKernel {
    pipeline: wgpu::ComputePipeline,
    layout_state: wgpu::BindGroupLayout,
    layout_springs: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,

    /// Per-spring λ accumulator (size `spring_count.max(1)`).
    lambdas: Buffer<Real>,
    /// Per-particle Jacobi delta scratch (size `particle_count.max(1)`).
    deltas: Buffer<Vec3>,
    /// Per-particle Jacobi count scratch.
    counts: Buffer<u32>,

    particle_count: u32,
    spring_count: u32,
}

impl XpbdParticlesKernel {
    pub fn new(
        device: &wgpu::Device,
        particle_count: u32,
        spring_count: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::xpbd_particles"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/xpbd_particles.wgsl").into(),
            ),
        });

        // State group: 6 read-only inputs.
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_particles state"),
            entries: &storage_entries(6, true),
        });
        // Springs group: 5 read-only inputs.
        let layout_springs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_particles springs"),
            entries: &storage_entries(5, true),
        });
        // Output / scratch group: 5 read-write (q, qd, lambdas, deltas, counts).
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_particles out"),
            entries: &storage_entries(5, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_particles uniforms"),
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
            label: Some("molla::xpbd_particles layout"),
            bind_group_layouts: &[
                &layout_state,
                &layout_springs,
                &layout_out,
                &layout_uniforms,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::xpbd_particles::step_particles_kernel"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("step_particles_kernel"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Allocate scratch buffers. wgpu requires nonzero sizes, so
        // bump zero counts to 1 — the kernel guards against
        // out-of-range accesses regardless.
        let pn = particle_count.max(1) as usize;
        let sn = spring_count.max(1) as usize;
        let mut lambdas = Buffer::from_vec(vec![0.0_f64; sn], BufferUsage::StorageReadWrite);
        let mut deltas = Buffer::from_vec(vec![Vec3::ZERO; pn], BufferUsage::StorageReadWrite);
        let mut counts = Buffer::from_vec(vec![0u32; pn], BufferUsage::StorageReadWrite);
        // Eager device alloc so the first dispatch doesn't have to.
        let _ = lambdas.ensure_device(device);
        let _ = deltas.ensure_device(device);
        let _ = counts.ensure_device(device);

        Self {
            pipeline,
            layout_state,
            layout_springs,
            layout_out,
            layout_uniforms,
            lambdas,
            deltas,
            counts,
            particle_count,
            spring_count,
        }
    }

    /// Run one full particle substep on the GPU. Mirrors
    /// `apply_particles_post_gpu`'s signature for drop-in
    /// replacement.
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        state_out: &mut State,
        dt: Real,
        iterations: u32,
    ) -> Result<()> {
        if model.particle_count == 0 {
            return Ok(());
        }

        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_particles state bg"),
            layout: &self.layout_state,
            entries: &[
                buffer_entry(0, state_in.particle_q.device_buffer())?,
                buffer_entry(1, state_in.particle_qd.device_buffer())?,
                buffer_entry(2, model.particle_inv_mass.device_buffer())?,
                buffer_entry(3, model.particle_radius.device_buffer())?,
                buffer_entry(4, model.particle_world.device_buffer())?,
                buffer_entry(5, model.gravity.device_buffer())?,
            ],
        });
        let bg_springs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_particles springs bg"),
            layout: &self.layout_springs,
            entries: &[
                buffer_entry(0, model.spring_a.device_buffer())?,
                buffer_entry(1, model.spring_b.device_buffer())?,
                buffer_entry(2, model.spring_rest_length.device_buffer())?,
                buffer_entry(3, model.spring_compliance.device_buffer())?,
                buffer_entry(4, model.spring_max_stretch.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_particles out bg"),
            layout: &self.layout_out,
            entries: &[
                buffer_entry(0, state_out.particle_q.device_buffer())?,
                buffer_entry(1, state_out.particle_qd.device_buffer())?,
                buffer_entry(2, self.lambdas.device_buffer())?,
                buffer_entry(3, self.deltas.device_buffer())?,
                buffer_entry(4, self.counts.device_buffer())?,
            ],
        });

        let uniforms = Uniforms {
            particle_count: model.particle_count as u32,
            spring_count: model.spring_count as u32,
            iterations,
            dt: dt as f32,
            linear_damping: model.linear_damping as f32,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::xpbd_particles uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_particles uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::xpbd_particles encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::xpbd_particles::step_particles_kernel"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_state, &[]);
            pass.set_bind_group(1, &bg_springs, &[]);
            pass.set_bind_group(2, &bg_out, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        state_out.particle_q.mark_device_dirty();
        state_out.particle_qd.mark_device_dirty();
        Ok(())
    }

    /// Number of particles the kernel was built for.
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }
    /// Number of springs the kernel was built for.
    pub fn spring_count(&self) -> u32 {
        self.spring_count
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!("xpbd_particles binding {binding}: no device mirror"))
    })?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    })
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
