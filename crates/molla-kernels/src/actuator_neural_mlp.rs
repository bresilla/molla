//! Rust dispatch wrappers for `shaders/actuator_neural_mlp.wgsl` +
//! `shaders/actuator_neural_mlp_scatter.wgsl`.
//!
//! The forward kernel runs a single workgroup of 64 threads per
//! network instance (single-batch on-device inference). The scatter
//! kernel is a one-thread-per-output pass that copies the network
//! outputs into the appropriate `Control.joint_force[dof]` slots so
//! the MLP can drop into the actuator pipeline alongside the PD/PID
//! variants.

use molla_core::{Error, Result};
use molla_storage::Buffer;

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

pub struct ActuatorNeuralMlpKernel {
    forward_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    forward_layout_io: wgpu::BindGroupLayout,
    forward_layout_weights: wgpu::BindGroupLayout,
    forward_layout_uniforms: wgpu::BindGroupLayout,
    scatter_layout_io: wgpu::BindGroupLayout,
    scatter_layout_uniforms: wgpu::BindGroupLayout,
}

impl ActuatorNeuralMlpKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let forward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::actuator_neural_mlp::forward"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/actuator_neural_mlp.wgsl").into(),
            ),
        });
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::actuator_neural_mlp::scatter"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/actuator_neural_mlp_scatter.wgsl").into(),
            ),
        });

        // Forward layouts.
        let forward_layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::actuator_neural_mlp forward io"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });
        let forward_layout_weights =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::actuator_neural_mlp forward weights"),
                entries: &storage_entries(6, true),
            });
        let forward_layout_uniforms =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::actuator_neural_mlp forward uniforms"),
                entries: &[uniform_entry(0)],
            });

        let forward_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::actuator_neural_mlp::forward layout"),
                bind_group_layouts: &[
                    Some(&forward_layout_io),
                    Some(&forward_layout_weights),
                    Some(&forward_layout_uniforms),
                ],
                immediate_size: 0,
            });

        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::actuator_neural_mlp::forward"),
            layout: Some(&forward_pipeline_layout),
            module: &forward_shader,
            entry_point: Some("forward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Scatter layouts.
        let scatter_layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::actuator_neural_mlp scatter io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let scatter_layout_uniforms =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::actuator_neural_mlp scatter uniforms"),
                entries: &[uniform_entry(0)],
            });
        let scatter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::actuator_neural_mlp::scatter layout"),
                bind_group_layouts: &[Some(&scatter_layout_io), Some(&scatter_layout_uniforms)],
                immediate_size: 0,
            });
        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::actuator_neural_mlp::scatter"),
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
            forward_layout_weights,
            forward_layout_uniforms,
            scatter_layout_io,
            scatter_layout_uniforms,
        }
    }

    /// Run the forward pass. Inputs must already be in the buffer
    /// device-side; the host writes the activation outputs into
    /// `outputs[0..output_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &Buffer<molla_math::Real>,
        outputs: &mut Buffer<molla_math::Real>,
        w1: &Buffer<molla_math::Real>,
        b1: &Buffer<molla_math::Real>,
        w2: &Buffer<molla_math::Real>,
        b2: &Buffer<molla_math::Real>,
        w3: &Buffer<molla_math::Real>,
        b3: &Buffer<molla_math::Real>,
        input_dim: u32,
        hidden_dim: u32,
        output_dim: u32,
    ) -> Result<()> {
        if input_dim == 0 || hidden_dim == 0 || output_dim == 0 {
            return Ok(());
        }

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_neural_mlp forward io bg"),
            layout: &self.forward_layout_io,
            entries: &[
                buffer_entry(0, inputs.device_buffer())?,
                buffer_entry(1, outputs.device_buffer())?,
            ],
        });
        let bg_weights = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_neural_mlp forward weights bg"),
            layout: &self.forward_layout_weights,
            entries: &[
                buffer_entry(0, w1.device_buffer())?,
                buffer_entry(1, b1.device_buffer())?,
                buffer_entry(2, w2.device_buffer())?,
                buffer_entry(3, b2.device_buffer())?,
                buffer_entry(4, w3.device_buffer())?,
                buffer_entry(5, b3.device_buffer())?,
            ],
        });

        let uniforms = ForwardUniforms {
            input_dim,
            hidden_dim,
            output_dim,
            _pad0: 0,
        };
        let uniform_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::actuator_neural_mlp forward uniforms"),
                size: core::mem::size_of::<ForwardUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(queue, &uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_neural_mlp forward uniforms bg"),
            layout: &self.forward_layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::actuator_neural_mlp::forward encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::actuator_neural_mlp::forward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_weights, &[]);
            pass.set_bind_group(2, &bg_uniforms, &[]);
            // Single-instance forward = exactly one workgroup.
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        outputs.mark_device_dirty();
        Ok(())
    }

    /// Scatter `outputs[i]` into `joint_force[dof_indices[i]]` for
    /// `i ∈ [0, output_dim)`.
    pub fn dispatch_scatter(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        outputs: &Buffer<molla_math::Real>,
        dof_indices: &Buffer<u32>,
        joint_force: &mut Buffer<molla_math::Real>,
        output_dim: u32,
    ) -> Result<()> {
        if output_dim == 0 {
            return Ok(());
        }

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_neural_mlp scatter io bg"),
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
        let uniform_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::actuator_neural_mlp scatter uniforms"),
                size: core::mem::size_of::<ScatterUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(queue, &uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_neural_mlp scatter uniforms bg"),
            layout: &self.scatter_layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = output_dim.div_ceil(SCATTER_WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::actuator_neural_mlp::scatter encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::actuator_neural_mlp::scatter"),
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
            "actuator_neural_mlp binding {binding}: no device mirror"
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
