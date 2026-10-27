//! Shared-machine tangential impulse disks with externally fixed caps.

use molla_compute::Engine;
use molla_core::{Error, Result};

pub const TANGENTIAL_INPUT: u32 = 1;
pub const TANGENTIAL_MASS: u32 = 2;
pub const TANGENTIAL_NONFINITE: u32 = 4;
pub const TANGENTIAL_LINE_SEARCH: u32 = 8;
pub const TANGENTIAL_ITERATIONS: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilTangentialContact {
    /// Finite soil-node mass; zero denotes an inactive slot with zero cap.
    pub mass: f32,
    /// Fixed nonnegative tangential impulse bound, in N s.
    pub max_impulse: f32,
    /// Unconstrained soil velocity in the orthonormal tangent frame, in m/s.
    pub free_velocity: [f32; 2],
}

pub struct SoilTangentialInputs<'a> {
    /// Symmetric positive-definite generalized machine mass, row-major.
    pub mass: &'a wgpu::Buffer,
    pub free_rates: &'a wgpu::Buffer,
    /// Contact-major pairs of orthonormal tangent Jacobian coefficients per DOF.
    pub rows: &'a wgpu::Buffer,
    /// Distinct physical soil nodes, each with one two-dimensional tangent frame.
    pub contacts: &'a wgpu::Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct SoilTangentialCapacity {
    pub dofs: u32,
    pub contacts: u32,
    pub max_iterations: u32,
    pub max_owned_bytes: u64,
}

/// Coupled finite-mass tangent solve. Only zero-status outputs are consumable.
/// Caps remain fixed: normal contact, floor constraints and runtime publication
/// are not performed. Callers clear the sticky status before an independent solve.
pub struct ExperimentalSoilTangentialProjectionGpu {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    rates: wgpu::Buffer,
    outputs: wgpu::Buffer,
    status: wgpu::Buffer,
    bytes: u64,
}

impl ExperimentalSoilTangentialProjectionGpu {
    pub fn new(
        engine: &Engine,
        input: SoilTangentialInputs<'_>,
        capacity: SoilTangentialCapacity,
    ) -> Result<Self> {
        let invalid = || {
            Error::Solver(
                "soil tangential projection: capacity, usage or device limit exceeded".into(),
            )
        };
        engine.check_device()?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let n = u64::from(capacity.dofs);
        let c = u64::from(capacity.contacts.max(1));
        let bytes = 4 * n + 32 * c + 32;
        if !(1..=32).contains(&capacity.dofs)
            || capacity.contacts > 16384
            || !(1..=64).contains(&capacity.max_iterations)
            || bytes > capacity.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_compute_workgroup_storage_size < 9280
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
        {
            return Err(invalid());
        }
        for (buffer, required) in [input.mass, input.free_rates, input.rows, input.contacts]
            .into_iter()
            .zip([4 * n * n, 4 * n, 8 * c * n, 16 * c])
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
        let rates = create("tangential projected machine rates", 4 * n, usage);
        let outputs = create(
            "tangential machine impulses and soil velocities",
            16 * c,
            usage,
        );
        let derivatives = create("tangential local Hessians", 16 * c, usage);
        let status = create("tangential sticky status", 16, usage);
        let params = create(
            "tangential dimensions",
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
            label: Some("shared-machine tangential projection"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_tangential_projection.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shared-machine tangential projection"),
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
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared-machine tangential projection"),
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
    /// [sticky fault bits, Newton iterations, backtracks, zero].
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
}

#[cfg(test)]
#[path = "soil_tangential_projection_tests.rs"]
mod tests;
