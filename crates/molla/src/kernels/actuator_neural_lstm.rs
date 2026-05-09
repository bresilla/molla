//! Rust dispatch wrapper for `shaders/actuator_neural_lstm.wgsl`.
//!
//! Single-cell LSTM forward pass + linear output projection. The
//! hidden state (`h`) and cell state (`c`) live in caller-owned
//! storage buffers — same `read_write` discipline as Stage 17's
//! `body_f` propagation: the shader reads the previous-step values,
//! writes back the next-step values in place, and the host reuses the
//! same buffers across substeps so the recurrence carries forward.
//!
//! Reuses the MLP scatter pass (`actuator_neural_mlp_scatter.wgsl`)
//! verbatim — the output-to-DOF copy is identical and there's no
//! reason to fork the kernel.

use crate::core::{Error, Result};
use crate::storage::Buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ForwardUniforms {
    input_dim: u32,
    hidden_dim: u32,
    output_dim: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScatterUniforms {
    output_dim: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const SCATTER_WORKGROUP_SIZE: u32 = 64;

pub struct ActuatorNeuralLstmKernel {
    forward_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    forward_layout_io: wgpu::BindGroupLayout,
    forward_layout_gates_ifg: wgpu::BindGroupLayout,
    forward_layout_gates_oy: wgpu::BindGroupLayout,
    forward_layout_uniforms: wgpu::BindGroupLayout,
    scatter_layout_io: wgpu::BindGroupLayout,
    scatter_layout_uniforms: wgpu::BindGroupLayout,
}

impl ActuatorNeuralLstmKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let forward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::actuator_neural_lstm::forward"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/actuator_neural_lstm.wgsl").into(),
            ),
        });
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::actuator_neural_lstm::scatter"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/actuator_neural_mlp_scatter.wgsl").into(),
            ),
        });

        // Forward layouts.
        // group(0): inputs (read), outputs (rw), h (rw), c (rw).
        let forward_layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::actuator_neural_lstm forward io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });
        // group(1): w_i, u_i, b_i, w_f, u_f, b_f, w_g, u_g, b_g (9 read).
        let forward_layout_gates_ifg =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::actuator_neural_lstm forward gates ifg"),
                entries: &storage_entries(9, true),
            });
        // group(2): w_o, u_o, b_o, w_y, b_y (5 read).
        let forward_layout_gates_oy =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::actuator_neural_lstm forward gates oy"),
                entries: &storage_entries(5, true),
            });
        let forward_layout_uniforms =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::actuator_neural_lstm forward uniforms"),
                entries: &[uniform_entry(0)],
            });

        let forward_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::actuator_neural_lstm::forward layout"),
                bind_group_layouts: &[
                    &forward_layout_io,
                    &forward_layout_gates_ifg,
                    &forward_layout_gates_oy,
                    &forward_layout_uniforms,
                ],
                push_constant_ranges: &[],
            });

        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::actuator_neural_lstm::forward"),
            layout: Some(&forward_pipeline_layout),
            module: &forward_shader,
            entry_point: Some("forward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Scatter layouts (mirror MLP).
        let scatter_layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::actuator_neural_lstm scatter io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let scatter_layout_uniforms =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::actuator_neural_lstm scatter uniforms"),
                entries: &[uniform_entry(0)],
            });
        let scatter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::actuator_neural_lstm::scatter layout"),
                bind_group_layouts: &[&scatter_layout_io, &scatter_layout_uniforms],
                push_constant_ranges: &[],
            });
        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::actuator_neural_lstm::scatter"),
            layout: Some(&scatter_pipeline_layout),
            module: &scatter_shader,
            entry_point: Some("scatter"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            forward_pipeline,
            scatter_pipeline,
            forward_layout_io,
            forward_layout_gates_ifg,
            forward_layout_gates_oy,
            forward_layout_uniforms,
            scatter_layout_io,
            scatter_layout_uniforms,
        }
    }

    /// Run a single LSTM time-step. `h_state` and `c_state` are read
    /// AND written: the kernel consumes the prior-step values and
    /// stores the next-step values in place.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &Buffer<crate::math::Real>,
        outputs: &mut Buffer<crate::math::Real>,
        h_state: &mut Buffer<crate::math::Real>,
        c_state: &mut Buffer<crate::math::Real>,
        w_i: &Buffer<crate::math::Real>,
        u_i: &Buffer<crate::math::Real>,
        b_i: &Buffer<crate::math::Real>,
        w_f: &Buffer<crate::math::Real>,
        u_f: &Buffer<crate::math::Real>,
        b_f: &Buffer<crate::math::Real>,
        w_g: &Buffer<crate::math::Real>,
        u_g: &Buffer<crate::math::Real>,
        b_g: &Buffer<crate::math::Real>,
        w_o: &Buffer<crate::math::Real>,
        u_o: &Buffer<crate::math::Real>,
        b_o: &Buffer<crate::math::Real>,
        w_y: &Buffer<crate::math::Real>,
        b_y: &Buffer<crate::math::Real>,
        input_dim: u32,
        hidden_dim: u32,
        output_dim: u32,
    ) -> Result<()> {
        if input_dim == 0 || hidden_dim == 0 || output_dim == 0 {
            return Ok(());
        }

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_neural_lstm forward io bg"),
            layout: &self.forward_layout_io,
            entries: &[
                buffer_entry(0, inputs.device_buffer())?,
                buffer_entry(1, outputs.device_buffer())?,
                buffer_entry(2, h_state.device_buffer())?,
                buffer_entry(3, c_state.device_buffer())?,
            ],
        });
        let bg_ifg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_neural_lstm forward gates ifg bg"),
            layout: &self.forward_layout_gates_ifg,
            entries: &[
                buffer_entry(0, w_i.device_buffer())?,
                buffer_entry(1, u_i.device_buffer())?,
                buffer_entry(2, b_i.device_buffer())?,
                buffer_entry(3, w_f.device_buffer())?,
                buffer_entry(4, u_f.device_buffer())?,
                buffer_entry(5, b_f.device_buffer())?,
                buffer_entry(6, w_g.device_buffer())?,
                buffer_entry(7, u_g.device_buffer())?,
                buffer_entry(8, b_g.device_buffer())?,
            ],
        });
        let bg_oy = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_neural_lstm forward gates oy bg"),
            layout: &self.forward_layout_gates_oy,
            entries: &[
                buffer_entry(0, w_o.device_buffer())?,
                buffer_entry(1, u_o.device_buffer())?,
                buffer_entry(2, b_o.device_buffer())?,
                buffer_entry(3, w_y.device_buffer())?,
                buffer_entry(4, b_y.device_buffer())?,
            ],
        });

        let uniforms = ForwardUniforms {
            input_dim,
            hidden_dim,
            output_dim,
            _pad0: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::actuator_neural_lstm forward uniforms"),
            size: core::mem::size_of::<ForwardUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_neural_lstm forward uniforms bg"),
            layout: &self.forward_layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::actuator_neural_lstm::forward encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::actuator_neural_lstm::forward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_ifg, &[]);
            pass.set_bind_group(2, &bg_oy, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        outputs.mark_device_dirty();
        h_state.mark_device_dirty();
        c_state.mark_device_dirty();
        Ok(())
    }

    /// Scatter `outputs[i]` into `joint_force[dof_indices[i]]`.
    pub fn dispatch_scatter(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        outputs: &Buffer<crate::math::Real>,
        dof_indices: &Buffer<u32>,
        joint_force: &mut Buffer<crate::math::Real>,
        output_dim: u32,
    ) -> Result<()> {
        if output_dim == 0 {
            return Ok(());
        }

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_neural_lstm scatter io bg"),
            layout: &self.scatter_layout_io,
            entries: &[
                buffer_entry(0, outputs.device_buffer())?,
                buffer_entry(1, dof_indices.device_buffer())?,
                buffer_entry(2, joint_force.device_buffer())?,
            ],
        });

        let uniforms = ScatterUniforms {
            output_dim,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::actuator_neural_lstm scatter uniforms"),
            size: core::mem::size_of::<ScatterUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_neural_lstm scatter uniforms bg"),
            layout: &self.scatter_layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = output_dim.div_ceil(SCATTER_WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::actuator_neural_lstm::scatter encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::actuator_neural_lstm::scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        joint_force.mark_device_dirty();
        Ok(())
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "actuator_neural_lstm binding {binding}: no device mirror"
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

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
