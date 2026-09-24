//! Rust dispatch wrapper for `shaders/imu.wgsl` — the GPU IMU sensor
//! kernel.
//!
//! Pattern mirrors [`super::RaycastKernel`]: one pipeline, four bind
//! groups (`body`, `sensor`, `out`, `uniforms`). The owner ([`super::super::sensors::imu::ImuSensor`])
//! supplies the per-sensor mount arrays + a previous-frame `body_qd`
//! buffer so the kernel can finite-difference accelerations.
//!
//! Reference: Newton's `compute_sensor_imu_kernel`.

use crate::sensor_bindings::{SensorBindings, required_buffers};
use molla_core::{Error, Result};
use molla_sim::{Model, State};
use molla_storage::Buffer;

/// Retained state/history/mount/output bindings and one parameter buffer.
pub struct ImuBindings(SensorBindings<11, 3>);

impl ImuBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 11]) -> bool {
        self.0.matches(buffers)
    }
}

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

    /// Buffer order: body poses/velocities/history/COM/world/gravity, mount
    /// bodies/positions/rotations, accelerometer/gyro. All belong to `device`.
    pub fn prepare(&self, device: &wgpu::Device, buffers: [&wgpu::Buffer; 11]) -> ImuBindings {
        ImuBindings(SensorBindings::new(
            device,
            buffers,
            [
                (&self.layout_body, 0..6),
                (&self.layout_sensor, 6..9),
                (&self.layout_out, 9..11),
            ],
            &self.layout_uniforms,
            "molla::imu prepared",
        ))
    }

    /// Encode without submitting. Submit before rewriting these bindings' parameters;
    /// use separate bindings for differing parameters in the same pending batch.
    /// The caller validates active lengths/indices and marks outputs device-authoritative.
    pub fn encode_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &ImuBindings,
        sensor_count: u32,
        dt: f32,
    ) -> Result<()> {
        if !dt.is_normal() || dt <= 0.0 {
            return Err(Error::Solver(
                "imu: dt must be positive and normal in f32".into(),
            ));
        }
        if sensor_count == 0 {
            return Ok(());
        }
        let uniforms = Uniforms {
            sensor_count,
            dt,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        bindings.0.encode(
            queue,
            encoder,
            &self.pipeline,
            bytemuck::cast_ref(&uniforms),
            sensor_count.div_ceil(WORKGROUP_SIZE),
        );
        Ok(())
    }

    /// Run the IMU kernel for `sensor_count` sensors.
    ///
    /// `body_qd_prev` must be uploaded to the device before calling
    /// (sensor owner's responsibility) — it carries the previous
    /// step's `state.body_qd` so the kernel can finite-difference.
    /// The caller advances history after evaluation.
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
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::imu encoder"),
        });
        self.encode(
            device,
            queue,
            &mut encoder,
            model,
            state,
            body_qd_prev,
            sensor_body,
            sensor_local_pos,
            sensor_local_rot,
            accel,
            gyro,
            sensor_count,
            dt,
        )?;
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Encode IMU evaluation into the caller's command stream.
    /// Submit the encoder before consuming its device-authoritative outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
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
        if !dt.is_normal() || dt <= 0.0 {
            return Err(Error::Solver(
                "imu: dt must be positive and normal in f32".into(),
            ));
        }
        if sensor_count == 0 {
            return Ok(());
        }

        let bindings = self.prepare(
            device,
            required_buffers(
                [
                    state.body_q.device_buffer(),
                    state.body_qd.device_buffer(),
                    body_qd_prev.device_buffer(),
                    model.body_com.device_buffer(),
                    model.body_world.device_buffer(),
                    model.gravity.device_buffer(),
                    sensor_body.device_buffer(),
                    sensor_local_pos.device_buffer(),
                    sensor_local_rot.device_buffer(),
                    accel.device_buffer(),
                    gyro.device_buffer(),
                ],
                "imu",
            )?,
        );
        self.encode_prepared(queue, encoder, &bindings, sensor_count, dt)?;
        accel.mark_device_dirty();
        gyro.mark_device_dirty();
        Ok(())
    }
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
