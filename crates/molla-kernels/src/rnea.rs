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
use molla_sim::{BodyFlags, JointType, Model, State};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    articulation_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64;

fn validate_model(model: &Model) -> Result<()> {
    let kinds = model.joint_type.host()?;
    if kinds.iter().any(|&kind| {
        ![
            JointType::Fixed as u32,
            JointType::Revolute as u32,
            JointType::Prismatic as u32,
        ]
        .contains(&kind)
    }) {
        return Err(Error::Gpu(
            "GPU RNEA currently supports only fixed, revolute and prismatic joints".into(),
        ));
    }
    if model.joint_reversed.host()?.iter().any(|&value| value != 0) {
        return Err(Error::Gpu(
            "GPU RNEA does not yet support reversed joints".into(),
        ));
    }
    let flags = model.body_flags.host()?;
    if model
        .joint_child
        .host()?
        .iter()
        .zip(kinds)
        .any(|(&child, &kind)| {
            flags[child as usize] & BodyFlags::KINEMATIC != 0
                || (flags[child as usize] & BodyFlags::FIXED != 0
                    && kind != JointType::Fixed as u32)
        })
    {
        return Err(Error::Gpu(
            "GPU RNEA does not yet support prescribed body motion".into(),
        ));
    }
    let lower = model.joint_limit_lower.host()?;
    let upper = model.joint_limit_upper.host()?;
    if lower
        .iter()
        .zip(upper)
        .any(|(&lo, &hi)| lo.abs() < 1e30 && hi.abs() < 1e30 && (hi - lo).abs() < 1e-12)
    {
        return Err(Error::Gpu(
            "GPU RNEA does not yet support scalar equality locks".into(),
        ));
    }
    let too_large = if model.articulation_count == 0 {
        model.joint_count > 64
    } else {
        model
            .articulation_start
            .host()?
            .windows(2)
            .any(|range| range[1] - range[0] > 64)
    };
    if too_large {
        return Err(Error::Gpu(
            "GPU RNEA supports at most 64 joints per articulation".into(),
        ));
    }
    Ok(())
}

/// GPU RNEA for fixed/hinge/slider chains of at most 64 joints per articulation.
/// Dispatch creates transient bindings; reversed joints and equality locks are unsupported.
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
        validate_model(model)?;
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

        let art_offsets_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::rnea art_offsets"),
                size: (art_offsets.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            queue,
            &art_offsets_buf,
            0,
            bytemuck::cast_slice(&art_offsets)
        );

        let dof_offsets_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::rnea dof_offsets"),
                size: (dof_offsets_host.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            queue,
            &dof_offsets_buf,
            0,
            bytemuck::cast_slice(&dof_offsets_host)
        );

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
        molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size: (len * core::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        )
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
        validate_model(model)?;
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
        let uniform_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::rnea uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(queue, &uniform_buf, 0, bytemuck::bytes_of(&uniforms));
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
            molla_core::gpu_metrics::record_dispatch();
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
        let staging = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::rnea staging"),
                size: byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
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
        let mapped = molla_core::gpu_readback!(slice);
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
    molla_core::gpu_write_buffer!(queue, &qdd_buf, 0, bytemuck::cast_slice(&qdd_full));
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

#[cfg(test)]
mod tests {
    use super::*;
    use molla_core::BodyId;
    use molla_math::{Transform, Vec3};
    use molla_sim::{BodyParams, ModelBuilder};

    #[test]
    fn rejects_unimplemented_multidof_rnea_instead_of_zeroing_its_width() {
        for kind in [
            JointType::Ball,
            JointType::Free,
            JointType::Distance,
            JointType::D6,
        ] {
            let mut builder = ModelBuilder::new();
            let body = builder.add_body(BodyParams::new());
            match kind {
                JointType::Ball => builder.add_joint_ball(
                    BodyId::NONE,
                    body,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                ),
                JointType::Free => builder.add_joint_free(
                    BodyId::NONE,
                    body,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                ),
                JointType::Distance => builder.add_joint_distance(
                    BodyId::NONE,
                    body,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                ),
                _ => builder.add_joint_d6(
                    BodyId::NONE,
                    body,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                ),
            }
            .unwrap();
            assert!(
                validate_model(&builder.build().unwrap())
                    .unwrap_err()
                    .to_string()
                    .contains("only fixed")
            );
        }
    }

    #[test]
    fn rejects_unimplemented_scalar_modes_and_preserves_supported_modes() {
        for reversed in [false, true] {
            for locked in [false, true] {
                let mut builder = ModelBuilder::new();
                let body = builder.add_body(BodyParams::new());
                let joint = builder
                    .add_joint_revolute(
                        BodyId::NONE,
                        body,
                        Transform::IDENTITY,
                        Transform::IDENTITY,
                        Vec3::X,
                    )
                    .unwrap();
                builder.set_joint_reversed(joint, reversed).unwrap();
                if locked {
                    builder.set_joint_limits(joint, &[0.0], &[0.0]).unwrap();
                }
                assert_eq!(
                    validate_model(&builder.build().unwrap()).is_ok(),
                    !reversed && !locked
                );
            }
        }
    }

    #[test]
    fn rejects_articulations_larger_than_shader_scratch() {
        for count in [64, 65] {
            let mut builder = ModelBuilder::new();
            builder.begin_articulation();
            let mut parent = BodyId::NONE;
            for _ in 0..count {
                let child = builder.add_body(BodyParams::new());
                builder
                    .add_joint_fixed(parent, child, Transform::IDENTITY, Transform::IDENTITY)
                    .unwrap();
                parent = child;
            }
            assert_eq!(
                validate_model(&builder.build().unwrap()).is_ok(),
                count == 64
            );
        }
    }

    #[test]
    fn rejects_prescribed_motion_not_represented_by_generalized_rates() {
        for flags in [BodyFlags::KINEMATIC, BodyFlags::FIXED] {
            for scalar in [false, true] {
                let mut builder = ModelBuilder::new();
                let body = builder.add_body(BodyParams {
                    flags,
                    ..Default::default()
                });
                if scalar {
                    builder
                        .add_joint_revolute(
                            BodyId::NONE,
                            body,
                            Transform::IDENTITY,
                            Transform::IDENTITY,
                            Vec3::X,
                        )
                        .unwrap();
                } else {
                    builder
                        .add_joint_fixed(
                            BodyId::NONE,
                            body,
                            Transform::IDENTITY,
                            Transform::IDENTITY,
                        )
                        .unwrap();
                }
                assert_eq!(
                    validate_model(&builder.build().unwrap()).is_ok(),
                    flags == BodyFlags::FIXED && !scalar
                );
            }
        }
    }
}
