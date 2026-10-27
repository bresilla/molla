//! Analytic body contact frames, joint friction and world-space soil reactions.

use crate::soil_contact_geometry_gpu::*;
use crate::soil_friction_projection_gpu::*;
use molla_compute::Engine;
use molla_core::{Error, Result};

pub const FRICTION_BOUNDARY_UNSUPPORTED: u32 = 64;

pub struct SoilFrictionGeometryInputs<'a> {
    /// Symmetric positive-definite generalized machine mass, row-major.
    pub mass: &'a wgpu::Buffer,
    pub free_rates: &'a wgpu::Buffer,
    pub centers: &'a wgpu::Buffer,
    pub shapes: &'a wgpu::Buffer,
    pub jacobians: &'a wgpu::Buffer,
    pub queries: &'a wgpu::Buffer,
    /// One explicit nonnegative Coulomb coefficient per body.
    pub coefficients: &'a wgpu::Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct SoilFrictionGeometryCapacity {
    pub dofs: u32,
    pub bodies: u32,
    pub contacts: u32,
    pub max_iterations: u32,
    pub max_owned_bytes: u64,
}

/// Encodes analytic geometry, contact-frame packing, joint solve and recovery.
/// Only zero-status outputs are consumable. Active fixed-wall nodes reject.
/// Unique node pairing, frame rollback and world publication remain external.
pub struct ExperimentalSoilFrictionGeometryGpu {
    geometry: ExperimentalSoilContactGeometryGpu,
    solver: ExperimentalSoilFrictionProjectionGpu,
    pipelines: [wgpu::ComputePipeline; 2],
    groups: [wgpu::BindGroup; 2],
    velocities: wgpu::Buffer,
    impulses: wgpu::Buffer,
    contacts: u32,
    bytes: u64,
}

fn invalid() -> Error {
    Error::Solver("soil friction geometry: capacity, usage or device limit exceeded".into())
}

impl ExperimentalSoilFrictionGeometryGpu {
    pub fn required_bytes(capacity: SoilFrictionGeometryCapacity) -> Result<u64> {
        if !(1..=32).contains(&capacity.dofs)
            || !(1..=64).contains(&capacity.bodies)
            || capacity.contacts > 16384
            || !(1..=64).contains(&capacity.max_iterations)
        {
            return Err(invalid());
        }
        let c = u64::from(capacity.contacts.max(1));
        let n = u64::from(capacity.dofs);
        Ok(176 * c + 12 * c * n + 4 * n + 64)
    }
    pub fn new(
        engine: &Engine,
        input: SoilFrictionGeometryInputs<'_>,
        capacity: SoilFrictionGeometryCapacity,
    ) -> Result<Self> {
        engine.check_device()?;
        let bytes = Self::required_bytes(capacity)?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let n = u64::from(capacity.dofs);
        let b = u64::from(capacity.bodies);
        let c = u64::from(capacity.contacts.max(1));
        if bytes > capacity.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 11
            || limits.max_compute_workgroup_storage_size < 9552
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_invocations_per_workgroup < 64
            || capacity.contacts.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || [48 * c, 8 * c * n, 4 * n].into_iter().any(|size| {
                size > limits.max_storage_buffer_binding_size || size > limits.max_buffer_size
            })
        {
            return Err(invalid());
        }
        for (buffer, required) in [
            input.mass,
            input.free_rates,
            input.centers,
            input.shapes,
            input.jacobians,
            input.queries,
            input.coefficients,
        ]
        .into_iter()
        .zip([4 * n * n, 4 * n, 16 * b, 64 * b, 24 * b * n, 48 * c, 4 * b])
        {
            if buffer.size() < required
                || buffer.size() > limits.max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid());
            }
        }
        let create = |label, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false
                }
            )
        };
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let nodes = create("friction analytic nodes", 48 * c, usage);
        let normal_rows = create("friction normalized normal rows", 4 * c * n, usage);
        let contacts = create("friction tangent contacts", 16 * c, usage);
        let normals = create("friction normal contacts", 16 * c, usage);
        let tangent_rows = create("friction tangent rows", 8 * c * n, usage);
        let velocities = create("friction world soil velocities", 16 * c, usage);
        let impulses = create("friction world soil impulses", 16 * c, usage);
        let solver = ExperimentalSoilFrictionProjectionGpu::new(
            engine,
            SoilFrictionInputs {
                mass: input.mass,
                free_rates: input.free_rates,
                rows: &tangent_rows,
                contacts: &contacts,
                normal_rows: &normal_rows,
                normals: &normals,
            },
            SoilFrictionCapacity {
                dofs: capacity.dofs,
                contacts: capacity.contacts,
                max_iterations: capacity.max_iterations,
                max_owned_bytes: 4 * n + 48 * c + 32,
            },
        )?;
        let geometry = ExperimentalSoilContactGeometryGpu::new(
            engine,
            SoilContactGeometryInputs {
                centers: input.centers,
                shapes: input.shapes,
                jacobians: input.jacobians,
                queries: input.queries,
            },
            SoilContactGeometryOutputs {
                nodes: &nodes,
                rows: &normal_rows,
                status: solver.status_buffer(),
            },
            SoilContactGeometryCapacity {
                dofs: capacity.dofs,
                bodies: capacity.bodies,
                queries: capacity.contacts,
                max_owned_bytes: 16 * c + 16,
            },
        )?;
        let params = create(
            "friction geometry dimensions",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[capacity.dofs, capacity.bodies, capacity.contacts, 0])
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("friction contact frame bridge"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_friction_geometry.wgsl").into(),
            ),
        });
        let pipelines = ["prepare", "recover"].map(|entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let buffers = [
            &nodes,
            input.queries,
            input.centers,
            input.jacobians,
            input.coefficients,
            &contacts,
            &normals,
            &tangent_rows,
            solver.contact_output_buffer(),
            solver.normal_output_buffer(),
            &velocities,
            &impulses,
            solver.status_buffer(),
            &params,
            &normal_rows,
        ];
        let groups = std::array::from_fn(|stage| {
            let bindings: &[usize] = if stage == 0 {
                &[0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14]
            } else {
                &[0, 8, 9, 10, 11, 12, 13]
            };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("friction geometry stage"),
                layout: &pipelines[stage].get_bind_group_layout(0),
                entries: &bindings
                    .iter()
                    .map(|&binding| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffers[binding].as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        });
        Ok(Self {
            geometry,
            solver,
            pipelines,
            groups,
            velocities,
            impulses,
            contacts: capacity.contacts,
            bytes,
        })
    }
    fn stage(&self, encoder: &mut wgpu::CommandEncoder, stage: usize) {
        if self.contacts == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines[stage]);
        pass.set_bind_group(0, &self.groups[stage], &[]);
        pass.dispatch_workgroups(self.contacts.div_ceil(64), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.geometry.encode(encoder);
        self.stage(encoder, 0);
        self.solver.encode(encoder);
        self.stage(encoder, 1);
    }
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        self.solver.rates_buffer()
    }
    pub fn velocities_buffer(&self) -> &wgpu::Buffer {
        &self.velocities
    }
    /// World contact normal XYZ and signed geometric distance W.
    pub fn geometry_buffer(&self) -> &wgpu::Buffer {
        self.geometry.geometry_buffer()
    }
    /// World impulse on soil in XYZ, scalar normal impulse in W.
    pub fn contact_impulses_buffer(&self) -> &wgpu::Buffer {
        &self.impulses
    }
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        self.solver.status_buffer()
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
}

#[cfg(test)]
#[path = "soil_friction_geometry_tests.rs"]
mod tests;
