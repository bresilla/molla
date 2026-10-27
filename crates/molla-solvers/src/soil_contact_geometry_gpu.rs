//! Resident analytic contact geometry for explicitly paired soil nodes and bodies.

use molla_compute::Engine;
use molla_core::{Error, Result};

pub const GEOMETRY_INPUT: u32 = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilContactQuery {
    pub position_mass: [f32; 4],
    /// Boundary-feasible free velocity and positive timestep.
    pub velocity_dt: [f32; 4],
    /// Body slot, fixed-axis wall mask, two zero words.
    pub body_boundary: [u32; 4],
}

pub struct SoilContactGeometryInputs<'a> {
    /// One world COM vec4 per body, matching prescribed-solid order.
    pub centers: &'a wgpu::Buffer,
    /// Four vec4s per body: linear velocity, angular velocity, XYZW quaternion, shape.
    pub shapes: &'a wgpu::Buffer,
    /// Six world-COM rows per body, linear XYZ then angular XYZ, row-major.
    pub jacobians: &'a wgpu::Buffer,
    /// One explicit body pairing per distinct physical node; zero mass is inactive.
    pub queries: &'a wgpu::Buffer,
}

pub struct SoilContactGeometryOutputs<'a> {
    /// SoilFloorNode records, one per query.
    pub nodes: &'a wgpu::Buffer,
    pub rows: &'a wgpu::Buffer,
    /// Shared 16-byte floor-projection status; the caller clears it before encoding.
    pub status: &'a wgpu::Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct SoilContactGeometryCapacity {
    pub dofs: u32,
    pub bodies: u32,
    pub queries: u32,
    pub max_owned_bytes: u64,
}

/// Writes analytic normals, nonnegative gap/dt allowances and generalized rows.
/// Signed gaps remain available in the geometry buffer. Interior nodes receive
/// zero gap allowance, without position repair or additional energy injection.
/// Candidate enumeration, unique node ownership and multi-body contact are external.
/// Buffers share the Engine device; writable outputs do not alias any other binding.
pub struct ExperimentalSoilContactGeometryGpu {
    program: std::sync::Arc<GeometryProgram>,
    group: wgpu::BindGroup,
    geometry: wgpu::Buffer,
    queries: u32,
    bytes: u64,
}

struct GeometryProgram {
    pipeline: wgpu::ComputePipeline,
}

fn geometry_program(engine: &Engine) -> Result<std::sync::Arc<GeometryProgram>> {
    let source = include_str!("../shaders/soil_contact_geometry.wgsl").replace(
        "// ANALYTIC_GEOMETRY",
        include_str!("../shaders/soil_contact_geometry_common.wgsl"),
    );
    let key = format!(
        "contact-geometry-v1;generate;wgsl={}",
        blake3::hash(source.as_bytes())
    );
    engine.cached_gpu_program(&key, |device| {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("analytic soil contact geometry"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil contact geometry"),
            layout: None,
            module: &shader,
            entry_point: Some("generate"),
            compilation_options: Default::default(),
            cache: engine.pipeline_cache(),
        });
        Ok(GeometryProgram { pipeline })
    })
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("soil contact geometry: {message}"))
}

impl ExperimentalSoilContactGeometryGpu {
    pub fn new(
        engine: &Engine,
        input: SoilContactGeometryInputs<'_>,
        output: SoilContactGeometryOutputs<'_>,
        capacity: SoilContactGeometryCapacity,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(|| invalid("GPU required"))?;
        let limits = device.limits();
        let n = u64::from(capacity.dofs);
        let b = u64::from(capacity.bodies);
        let count = u64::from(capacity.queries.max(1));
        let bytes = 16 * count + 16;
        if !(1..=32).contains(&capacity.dofs)
            || !(1..=64).contains(&capacity.bodies)
            || capacity.queries > 16384
            || bytes > capacity.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 8
            || capacity.queries.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || 16 * count > limits.max_storage_buffer_binding_size
        {
            return Err(invalid("capacity or device limit exceeded"));
        }
        let buffers = [
            input.centers,
            input.shapes,
            input.jacobians,
            input.queries,
            output.nodes,
            output.rows,
            output.status,
        ];
        for (buffer, required) in buffers.iter().zip([
            16 * b,
            64 * b,
            24 * b * n,
            48 * count,
            48 * count,
            4 * count * n,
            16,
        ]) {
            if buffer.size() < required
                || buffer.size() > limits.max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid("buffer capacity or usage mismatch"));
            }
        }
        let geometry = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("soil signed contact geometry"),
                size: 16 * count,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("soil contact geometry parameters"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[capacity.dofs, capacity.bodies, capacity.queries, 0u32])
        );
        let program = geometry_program(engine)?;
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil contact geometry"),
            layout: &program.pipeline.get_bind_group_layout(0),
            entries: &[
                input.centers,
                input.shapes,
                input.jacobians,
                input.queries,
                output.nodes,
                output.rows,
                &geometry,
                output.status,
                &params,
            ]
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        Ok(Self {
            program,
            group,
            geometry,
            queries: capacity.queries,
            bytes,
        })
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.queries == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.program.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.queries.div_ceil(64), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    /// World unit normal XYZ and signed Euclidean distance W.
    pub fn geometry_buffer(&self) -> &wgpu::Buffer {
        &self.geometry
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
}

#[cfg(test)]
#[path = "soil_contact_geometry_tests.rs"]
mod tests;
