//! Independent finite-machine Coulomb blocks with simultaneous normal-only coordinate walls.

use molla_compute::Engine;
use molla_core::{Error, Result};

pub const WALL_FRICTION_INPUT: u32 = 1;
pub const WALL_FRICTION_METRIC: u32 = 2;
pub const WALL_FRICTION_NONFINITE: u32 = 4;
pub const WALL_FRICTION_LINE_SEARCH: u32 = 8;
pub const WALL_FRICTION_ITERATIONS: u32 = 16;
pub const WALL_FRICTION_RESIDUAL: u32 = 32;
pub const WALL_FRICTION_JACOBIAN: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilWallFrictionInput {
    /// Symmetric positive-definite world point inverse mass; row padding is zero.
    pub inverse_machine: [[f32; 4]; 3],
    pub machine_velocity_friction: [f32; 4],
    /// Free world velocity and strictly positive finite soil-node mass.
    pub soil_velocity_mass: [f32; 4],
    /// Unit body-to-soil normal and nonnegative gap divided by timestep.
    pub normal_gap: [f32; 4],
    /// Two bits per XYZ axis: 0 absent, 1 lower, 2 upper; remaining words zero.
    pub boundary: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilWallFrictionOutput {
    pub machine_impulse_normal: [f32; 4],
    pub soil_velocity_energy: [f32; 4],
    pub machine_velocity_residual: [f32; 4],
    pub wall_impulse: [f32; 4],
    /// Fault, Newton iterations, backtracks, active wall-axis bits.
    pub status: [u32; 4],
}

/// Batched local blocks; only zero-fault outputs are consumable.
/// Blocks sharing a machine require an external coupled solve, not independent application.
/// Coulomb solutions can be nonunique; no particular admissible branch is guaranteed.
pub struct ExperimentalSoilWallFrictionGpu {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    output: wgpu::Buffer,
    count: u32,
    bytes: u64,
}

impl ExperimentalSoilWallFrictionGpu {
    pub fn new(
        engine: &Engine,
        input: &wgpu::Buffer,
        count: u32,
        iterations: u32,
        budget: u64,
    ) -> Result<Self> {
        Self::with_source(
            engine,
            input,
            count,
            iterations,
            budget,
            &Self::shader_source(),
        )
    }

    pub(crate) fn shader_source() -> String {
        [
            include_str!("../shaders/soil_wall_friction_block.wgsl"),
            include_str!("../shaders/soil_wall_friction.wgsl"),
        ]
        .join("\n")
    }

    pub(crate) fn with_source(
        engine: &Engine,
        input: &wgpu::Buffer,
        count: u32,
        iterations: u32,
        budget: u64,
        source: &str,
    ) -> Result<Self> {
        let invalid =
            || Error::Solver("soil wall friction: invalid capacity, usage or device limit".into());
        engine.check_device()?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let records = u64::from(count.max(1));
        let bytes = 80 * records + 16;
        if count > 16384
            || !(1..=64).contains(&iterations)
            || bytes > budget
            || input.size() < 112 * records
            || input.size() > limits.max_storage_buffer_binding_size
            || !input.usage().contains(wgpu::BufferUsages::STORAGE)
            || 80 * records > limits.max_storage_buffer_binding_size
            || 80 * records > limits.max_buffer_size
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_storage_size < 256
            || limits.max_storage_buffers_per_shader_stage < 2
        {
            return Err(invalid());
        }
        let output = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("wall friction local outputs"),
                size: 80 * records,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("wall friction local dimensions"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[count, iterations, 0, 0])
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("joint wall friction block"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("joint wall friction block"),
            layout: None,
            module: &module,
            entry_point: Some("solve"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("joint wall friction inputs"),
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
#[path = "soil_wall_friction_tests.rs"]
mod tests;
