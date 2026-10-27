//! Independent tangential maximum-dissipation blocks with fixed impulse caps.

use molla_compute::Engine;
use molla_core::{Error, Result};

pub const FRICTION_DISK_INPUT: u32 = 1;
pub const FRICTION_DISK_METRIC: u32 = 2;
pub const FRICTION_DISK_NONFINITE: u32 = 4;
pub const FRICTION_DISK_ITERATIONS: u32 = 8;
pub const FRICTION_DISK_RESIDUAL: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilFrictionDiskInput {
    /// Symmetric inverse effective mass [W00, W01, W11], in inverse kg.
    pub inverse_mass: [f32; 3],
    /// Fixed nonnegative tangential impulse bound, in N s.
    pub max_impulse: f32,
    /// Pre-impulse machine-minus-soil velocity in an orthonormal tangent frame, m/s.
    pub relative_velocity: [f32; 2],
    pub padding: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilFrictionDiskOutput {
    /// Impulse on the machine, in N s; the soil receives its negative.
    pub impulse: [f32; 2],
    pub relative_velocity: [f32; 2],
    /// Reduced quadratic kinetic-energy change, in joules.
    pub energy_change: f32,
    pub normalized_multiplier: f32,
    pub normalized_residual: f32,
    pub normalized_radius: f32,
    pub fault: u32,
    pub iterations: u32,
    /// 0: zero cap, 1: interior/sticking, 2: boundary/sliding.
    pub mode: u32,
    pub padding: u32,
}

/// Retained GPU solve of independent 2D metric disks. Only zero-fault rows
/// are consumable. Interacting contacts, normal updates, world reactions and
/// state publication are not performed by this component.
pub struct ExperimentalSoilFrictionDiskGpu {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    output: wgpu::Buffer,
    count: u32,
    bytes: u64,
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("soil friction disk: {message}"))
}

impl ExperimentalSoilFrictionDiskGpu {
    pub fn new(
        engine: &Engine,
        input: &wgpu::Buffer,
        count: u32,
        max_iterations: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(|| invalid("GPU required"))?;
        let limits = device.limits();
        let records = u64::from(count.max(1));
        let bytes = 48 * records + 16;
        if !(1..=64).contains(&max_iterations)
            || bytes > max_owned_bytes
            || input.size() < 32 * records
            || input.size() > limits.max_storage_buffer_binding_size
            || !input.usage().contains(wgpu::BufferUsages::STORAGE)
            || 48 * records > limits.max_storage_buffer_binding_size
            || 48 * records > limits.max_buffer_size
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_storage_buffers_per_shader_stage < 2
        {
            return Err(invalid(
                "capacity, usage, iteration or device limit exceeded",
            ));
        }
        let create = |label, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let output = create(
            "soil friction disk output",
            48 * records,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let params = create(
            "soil friction disk dimensions",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[count, max_iterations, 0, 0])
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil friction disk"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_friction_disk.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil friction disk"),
            layout: None,
            module: &module,
            entry_point: Some("solve"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil friction disk"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[input, &output, &params]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            pipeline,
            group,
            output,
            count,
            bytes,
        })
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        &self.output
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
}

#[cfg(test)]
#[path = "soil_friction_disk_tests.rs"]
mod tests;
