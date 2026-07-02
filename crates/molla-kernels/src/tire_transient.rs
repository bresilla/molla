//! GPU layered tire transient relaxation.
//!
//! This batches the scalar tread/belt/carcass relaxation state that the CPU
//! `molla_vehicle::TireLayeredTransientState` uses before brush/FEM contact.

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default, PartialEq)]
pub struct TireTransientGpuInput {
    /// `(raw_kappa, raw_alpha_rad, ds_m, radius_m)`.
    pub command: [f32; 4],
    /// `(tread_long_m, tread_lat_m, tread_yaw_m, length_scale)`.
    pub tread_lengths: [f32; 4],
    /// `(belt_long_m, belt_lat_m, belt_yaw_m, reserved)`.
    pub belt_lengths: [f32; 4],
    /// `(carcass_long_m, carcass_lat_m, carcass_yaw_m, reserved)`.
    pub carcass_lengths: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default, PartialEq)]
pub struct TireTransientGpuState {
    /// `(kappa_eff, alpha_eff_rad, yaw_rad, reserved)`.
    pub tread: [f32; 4],
    pub belt: [f32; 4],
    pub carcass: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default, PartialEq)]
pub struct TireTransientGpuOutput {
    /// `(kappa_eff, alpha_eff_rad, tread_yaw_rad, finite_flag)`.
    pub effective: [f32; 4],
    /// `(longitudinal_deflection_m, lateral_deflection_m, 0, 0)`.
    pub deflection: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default, PartialEq)]
struct TireTransientUniforms {
    count: u32,
    _pad: [u32; 3],
}

pub struct TireTransientGpuKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

pub struct TireTransientGpuBatch {
    capacity: usize,
    input_buf: wgpu::Buffer,
    state_in_buf: wgpu::Buffer,
    state_out_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    stage_state: wgpu::Buffer,
    stage_out: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

impl TireTransientGpuKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_transient"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_transient.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_transient io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                uniform_entry(4),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::tire_transient layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_transient::transient_step"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("transient_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// Allocate reusable buffers for up to `capacity` transient wheel states.
    pub fn create_batch(
        &self,
        device: &wgpu::Device,
        capacity: usize,
    ) -> Result<TireTransientGpuBatch> {
        let capacity = capacity.max(1);
        let in_bytes = (capacity * core::mem::size_of::<TireTransientGpuInput>()) as u64;
        let state_bytes = (capacity * core::mem::size_of::<TireTransientGpuState>()) as u64;
        let out_bytes = (capacity * core::mem::size_of::<TireTransientGpuOutput>()) as u64;
        let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient persistent input"),
            size: in_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let state_in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient persistent state in"),
            size: state_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let state_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient persistent state out"),
            size: state_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient persistent output"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient persistent uniforms"),
            size: core::mem::size_of::<TireTransientUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stage_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient persistent stage state"),
            size: state_bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let stage_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient persistent stage output"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_transient persistent bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        Ok(TireTransientGpuBatch {
            capacity,
            input_buf,
            state_in_buf,
            state_out_buf,
            out_buf,
            uniform_buf,
            stage_state,
            stage_out,
            bg,
        })
    }

    pub fn dispatch_persistent_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireTransientGpuBatch,
        inputs: &[TireTransientGpuInput],
        states: &[TireTransientGpuState],
    ) -> Result<(Vec<TireTransientGpuState>, Vec<TireTransientGpuOutput>)> {
        validate_transient_inputs(inputs, states)?;
        let n = inputs.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if n > batch.capacity {
            return Err(Error::Solver(format!(
                "tire_transient: batch capacity {} is smaller than {n} wheels",
                batch.capacity
            )));
        }
        let uniforms = TireTransientUniforms {
            count: n as u32,
            _pad: [0; 3],
        };
        let state_bytes = (n * core::mem::size_of::<TireTransientGpuState>()) as u64;
        let out_bytes = (n * core::mem::size_of::<TireTransientGpuOutput>()) as u64;
        queue.write_buffer(&batch.input_buf, 0, bytemuck::cast_slice(inputs));
        queue.write_buffer(&batch.state_in_buf, 0, bytemuck::cast_slice(states));
        queue.write_buffer(&batch.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_transient persistent encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_transient persistent pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &batch.bg, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&batch.state_out_buf, 0, &batch.stage_state, 0, state_bytes);
        encoder.copy_buffer_to_buffer(&batch.out_buf, 0, &batch.stage_out, 0, out_bytes);
        queue.submit(std::iter::once(encoder.finish()));
        Ok((
            read_buffer(device, &batch.stage_state, state_bytes, "persistent_state")?,
            read_buffer(device, &batch.stage_out, out_bytes, "persistent_output")?,
        ))
    }

    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &[TireTransientGpuInput],
        states: &[TireTransientGpuState],
    ) -> Result<(Vec<TireTransientGpuState>, Vec<TireTransientGpuOutput>)> {
        validate_transient_inputs(inputs, states)?;
        let n = inputs.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let in_bytes = std::mem::size_of_val(inputs) as u64;
        let state_bytes = (n * core::mem::size_of::<TireTransientGpuState>()) as u64;
        let out_bytes = (n * core::mem::size_of::<TireTransientGpuOutput>()) as u64;
        let uniforms = TireTransientUniforms {
            count: n as u32,
            _pad: [0; 3],
        };
        let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient input"),
            size: in_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let state_in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient state in"),
            size: state_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let state_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient state out"),
            size: state_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient output"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient stage state"),
            size: state_bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let stage_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient stage output"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_transient uniforms"),
            size: core::mem::size_of::<TireTransientUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&input_buf, 0, bytemuck::cast_slice(inputs));
        queue.write_buffer(&state_in_buf, 0, bytemuck::cast_slice(states));
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_transient bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_transient encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_transient pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&state_out_buf, 0, &stage_state, 0, state_bytes);
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage_out, 0, out_bytes);
        queue.submit(std::iter::once(encoder.finish()));
        Ok((
            read_buffer(device, &stage_state, state_bytes, "state")?,
            read_buffer(device, &stage_out, out_bytes, "output")?,
        ))
    }
}

pub fn tire_transient_reference_step(
    input: &TireTransientGpuInput,
    state: &TireTransientGpuState,
) -> (TireTransientGpuState, TireTransientGpuOutput) {
    let raw_kappa = input.command[0].clamp(-8.0, 8.0);
    let raw_alpha = input.command[1].clamp(-1.2, 1.2);
    let ds = input.command[2].max(0.0);
    let radius = input.command[3].max(1.0e-6);
    let scale = input.tread_lengths[3].max(1.0e-4).clamp(0.05, 20.0);
    let tread = advance_layer(
        state.tread,
        raw_kappa,
        raw_alpha,
        ds,
        [
            input.tread_lengths[0] * scale * 0.85,
            input.tread_lengths[1] * scale * 0.85,
            input.tread_lengths[2] * scale * 0.85,
        ],
    );
    let belt = advance_layer(
        state.belt,
        raw_kappa,
        raw_alpha,
        ds,
        [
            input.belt_lengths[0] * scale,
            input.belt_lengths[1] * scale,
            input.belt_lengths[2] * scale,
        ],
    );
    let carcass = advance_layer(
        state.carcass,
        raw_kappa,
        raw_alpha,
        ds,
        [
            input.carcass_lengths[0] * scale * 1.15,
            input.carcass_lengths[1] * scale * 1.15,
            input.carcass_lengths[2] * scale * 1.15,
        ],
    );
    let kappa_eff = 0.58 * tread[0] + 0.30 * belt[0] + 0.12 * carcass[0];
    let alpha_eff = 0.55 * tread[1] + 0.32 * belt[1] + 0.13 * carcass[1];
    let yaw_eff = 0.45 * tread[2] + 0.35 * belt[2] + 0.20 * carcass[2];
    let next = TireTransientGpuState {
        tread,
        belt,
        carcass,
    };
    let output = TireTransientGpuOutput {
        effective: [kappa_eff, alpha_eff, yaw_eff, 1.0],
        deflection: [
            radius * kappa_eff,
            radius * alpha_eff.tan().clamp(-4.0, 4.0),
            0.0,
            0.0,
        ],
    };
    (next, output)
}

fn advance_layer(
    mut layer: [f32; 4],
    raw_kappa: f32,
    raw_alpha: f32,
    ds: f32,
    lengths: [f32; 3],
) -> [f32; 4] {
    let blend = |length: f32| (1.0 - (-ds.max(0.0) / length.max(1.0e-4)).exp()).clamp(0.0, 1.0);
    let bx = blend(lengths[0]);
    let by = blend(lengths[1]);
    let bt = blend(lengths[2]);
    layer[0] += (raw_kappa - layer[0]) * bx;
    layer[1] += (raw_alpha - layer[1]) * by;
    layer[2] += (raw_alpha - layer[2]) * bt;
    layer[3] = 0.0;
    layer
}

fn validate_transient_inputs(
    inputs: &[TireTransientGpuInput],
    states: &[TireTransientGpuState],
) -> Result<()> {
    if inputs.len() != states.len() {
        return Err(Error::Solver(format!(
            "tire_transient: input/state length mismatch {} != {}",
            inputs.len(),
            states.len()
        )));
    }
    Ok(())
}

fn read_buffer<T: Pod>(
    device: &wgpu::Device,
    stage: &wgpu::Buffer,
    bytes: u64,
    label: &str,
) -> Result<Vec<T>> {
    let slice = stage.slice(0..bytes);
    let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
    let label = label.to_owned();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        sender
            .send(r.map_err(|e| Error::Gpu(format!("tire_transient {label} readback: {e:?}"))))
            .ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("tire_transient poll: {e:?}")))?;
    receiver
        .recv()
        .map_err(|e| Error::Gpu(format!("tire_transient recv: {e:?}")))??;
    let data = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    stage.unmap();
    Ok(out)
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
