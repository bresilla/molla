//! Joint finite-mass normal and tangential contact without fixed-wall constraints.

use molla_compute::Engine;
use molla_core::{Error, Result};

pub const FRICTION_INPUT: u32 = 1;
pub const FRICTION_MASS: u32 = 2;
pub const FRICTION_NONFINITE: u32 = 4;
pub const FRICTION_LINE_SEARCH: u32 = 8;
pub const FRICTION_ITERATIONS: u32 = 16;
pub const FRICTION_JACOBIAN: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilFrictionContact {
    /// Finite soil-node mass; zero denotes an inactive slot.
    pub mass: f32,
    /// Nonnegative Coulomb coefficient; numerical input, not a calibrated preset.
    pub friction: f32,
    /// Unconstrained soil velocity in the orthonormal tangent frame, in m/s.
    pub free_velocity: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilFrictionNormal {
    pub free_velocity: f32,
    /// Nonnegative normal gap divided by timestep, in m/s.
    pub gap_speed: f32,
    pub padding: [f32; 2],
}

pub struct SoilFrictionInputs<'a> {
    /// Symmetric positive-definite generalized machine mass, row-major.
    pub mass: &'a wgpu::Buffer,
    pub free_rates: &'a wgpu::Buffer,
    /// Contact-major pairs of orthonormal tangent Jacobian coefficients per DOF.
    pub rows: &'a wgpu::Buffer,
    /// Distinct physical soil nodes, each with one two-dimensional tangent frame.
    pub contacts: &'a wgpu::Buffer,
    /// Contact-major normal Jacobian coefficients, one per machine DOF.
    pub normal_rows: &'a wgpu::Buffer,
    pub normals: &'a wgpu::Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct SoilFrictionCapacity {
    pub dofs: u32,
    pub contacts: u32,
    pub max_iterations: u32,
    pub max_owned_bytes: u64,
}

/// Joint normal/friction residual solve with a pivoted Newton Jacobian.
/// Only zero-status outputs are consumable. Fixed walls and runtime publication
/// are not supported. Callers clear sticky status before an independent solve.
pub struct ExperimentalSoilFrictionProjectionGpu {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    rates: wgpu::Buffer,
    outputs: wgpu::Buffer,
    normals: wgpu::Buffer,
    status: wgpu::Buffer,
    bytes: u64,
}

impl ExperimentalSoilFrictionProjectionGpu {
    pub fn new(
        engine: &Engine,
        input: SoilFrictionInputs<'_>,
        capacity: SoilFrictionCapacity,
    ) -> Result<Self> {
        Self::with_source(
            engine,
            input,
            capacity,
            include_str!("../shaders/soil_friction_projection.wgsl"),
        )
    }

    fn with_source(
        engine: &Engine,
        input: SoilFrictionInputs<'_>,
        capacity: SoilFrictionCapacity,
        source: &str,
    ) -> Result<Self> {
        let invalid = || {
            Error::Solver(
                "soil frictional projection: capacity, usage or device limit exceeded".into(),
            )
        };
        engine.check_device()?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let n = u64::from(capacity.dofs);
        let c = u64::from(capacity.contacts.max(1));
        let bytes = 4 * n + 48 * c + 32;
        if !(1..=32).contains(&capacity.dofs)
            || capacity.contacts > 16384
            || !(1..=64).contains(&capacity.max_iterations)
            || bytes > capacity.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 11
            || limits.max_compute_workgroup_storage_size < 9552
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
        {
            return Err(invalid());
        }
        for (buffer, required) in [
            input.mass,
            input.free_rates,
            input.rows,
            input.contacts,
            input.normal_rows,
            input.normals,
        ]
        .into_iter()
        .zip([4 * n * n, 4 * n, 8 * c * n, 16 * c, 4 * c * n, 16 * c])
        {
            if buffer.size() < required
                || buffer.size() > limits.max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid());
            }
        }
        if [4 * n, 16 * c].into_iter().any(|size| {
            size > limits.max_storage_buffer_binding_size || size > limits.max_buffer_size
        }) {
            return Err(invalid());
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
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let rates = create("frictional projected machine rates", 4 * n, usage);
        let outputs = create(
            "frictional machine impulses and soil velocities",
            16 * c,
            usage,
        );
        let derivatives = create("frictional local Hessians", 16 * c, usage);
        let normals = create("normal impulse velocity and cap", 16 * c, usage);
        let status = create("frictional sticky status", 16, usage);
        let params = create(
            "frictional dimensions",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[capacity.dofs, capacity.contacts, capacity.max_iterations, 0])
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shared-machine frictional projection"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shared-machine frictional projection"),
            layout: None,
            module: &module,
            entry_point: Some("solve"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffers = [
            input.mass,
            input.free_rates,
            input.rows,
            input.contacts,
            &rates,
            &outputs,
            &derivatives,
            &status,
            &params,
            input.normal_rows,
            input.normals,
            &normals,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared-machine frictional projection"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &buffers
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
            rates,
            outputs,
            normals,
            status,
            bytes,
        })
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        &self.rates
    }
    /// Contact-major [machine impulse x/y, corrected soil velocity x/y].
    pub fn contact_output_buffer(&self) -> &wgpu::Buffer {
        &self.outputs
    }
    /// Contact-major [soil normal impulse, corrected normal velocity, last cap, local normal slope].
    pub fn normal_output_buffer(&self) -> &wgpu::Buffer {
        &self.normals
    }
    /// [sticky fault bits, total iterations, backtracks, pivot exchanges].
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
}

#[cfg(test)]
#[path = "soil_friction_projection_tests.rs"]
mod tests;
