//! Rust dispatch wrapper for `shaders/imu.wgsl` — the GPU IMU sensor
//! kernel.
//!
//! Pattern mirrors [`super::RaycastKernel`]: one pipeline, four bind
//! groups (`body`, `sensor`, `out`, `uniforms`). The owner ([`super::super::sensors::imu::ImuSensor`])
//! supplies the per-sensor mount arrays + a previous-frame `body_qd`
//! buffer so the kernel can finite-difference accelerations.
//!
//! Reference: Newton's `compute_sensor_imu_kernel`.

use molla_core::{Error, Result};
use molla_sim::{Model, State};
use molla_storage::Buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    sensor_count: u32,
    dt: f32,
    _pad0: f32,
    _pad1: f32,
}

const WORKGROUP_SIZE: u32 = 64;

pub struct ImuKernel {
    pipeline: wgpu::ComputePipeline,
    layout_body: wgpu::BindGroupLayout,
    layout_sensor: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl ImuKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::imu"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/imu.wgsl").into()),
        });

        // Body group: body_q, body_qd, body_qd_prev, body_com, body_world, gravity.
        let layout_body = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::imu body"),
            entries: &storage_entries(6, true),
        });
        // Sensor group: sensor_body, sensor_local_pos, sensor_local_rot.
        let layout_sensor = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::imu sensor"),
            entries: &storage_entries(3, true),
        });
        // Output group: accel, gyro.
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::imu out"),
            entries: &storage_entries(2, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::imu uniforms"),
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
            label: Some("molla::imu layout"),
            bind_group_layouts: &[
                Some(&layout_body),
                Some(&layout_sensor),
                Some(&layout_out),
                Some(&layout_uniforms),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::imu::compute_imu"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_imu"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            layout_body,
            layout_sensor,
            layout_out,
            layout_uniforms,
        }
    }

    /// Run the IMU kernel for `sensor_count` sensors.
    ///
    /// `body_qd_prev` must be uploaded to the device before calling
    /// (sensor owner's responsibility) — it carries the previous
    /// step's `state.body_qd` so the kernel can finite-difference.
    /// After the dispatch, the sensor swaps the roles for next frame.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state: &State,
        body_qd_prev: &Buffer<molla_math::SpatialVector>,
        sensor_body: &Buffer<u32>,
        sensor_local_pos: &Buffer<molla_math::Vec3>,
        sensor_local_rot: &Buffer<molla_math::Quat>,
        accel: &mut Buffer<molla_math::Vec3>,
        gyro: &mut Buffer<molla_math::Vec3>,
        sensor_count: u32,
        dt: f32,
    ) -> Result<()> {
        if sensor_count == 0 {
            return Ok(());
        }

        let bg_body = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::imu body bg"),
            layout: &self.layout_body,
            entries: &[
                buffer_entry(0, state.body_q.device_buffer())?,
                buffer_entry(1, state.body_qd.device_buffer())?,
                buffer_entry(2, body_qd_prev.device_buffer())?,
                buffer_entry(3, model.body_com.device_buffer())?,
                buffer_entry(4, model.body_world.device_buffer())?,
                buffer_entry(5, model.gravity.device_buffer())?,
            ],
        });
        let bg_sensor = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::imu sensor bg"),
            layout: &self.layout_sensor,
            entries: &[
                buffer_entry(0, sensor_body.device_buffer())?,
                buffer_entry(1, sensor_local_pos.device_buffer())?,
                buffer_entry(2, sensor_local_rot.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::imu out bg"),
            layout: &self.layout_out,
            entries: &[
                buffer_entry(0, accel.device_buffer())?,
                buffer_entry(1, gyro.device_buffer())?,
            ],
        });

        let uniforms = Uniforms {
            sensor_count,
            dt,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::imu uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::imu uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = sensor_count.div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::imu encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::imu::compute_imu"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_body, &[]);
            pass.set_bind_group(1, &bg_sensor, &[]);
            pass.set_bind_group(2, &bg_out, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        accel.mark_device_dirty();
        gyro.mark_device_dirty();
        Ok(())
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| Error::Gpu(format!("imu binding {binding}: no device mirror")))?;
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
