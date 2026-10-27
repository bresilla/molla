//! Analytic geometry with shared-machine Coulomb contact and coordinate walls.

use crate::soil_contact_geometry_gpu::*;
pub use crate::soil_friction_geometry_gpu::SoilFrictionGeometryInputs as SoilSharedWallGeometryInputs;
use crate::soil_rate_encoding::RateEncoding;
use crate::soil_shared_wall_friction_gpu::*;
use molla_compute::Engine;
use molla_core::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub struct SoilSharedWallGeometryCapacity {
    pub dofs: u32,
    pub bodies: u32,
    pub contacts: u32,
    pub max_sweeps: u32,
    pub local_iterations: u32,
    pub max_owned_bytes: u64,
}

/// Encodes analytic geometry, shared wall/contact solve and world-space recovery.
/// Only zero-status outputs are consumable. Wall impulses exclude machine reaction.
/// Free query velocities satisfy the existing coordinate-wall direction contract.
/// Unique node pairing, frame rollback and world publication remain external.
pub struct ExperimentalSoilSharedWallGeometryGpu {
    geometry: ExperimentalSoilContactGeometryGpu,
    solver: ExperimentalSoilSharedWallFrictionGpu,
    pipelines: [wgpu::ComputePipeline; 2],
    groups: [wgpu::BindGroup; 2],
    velocities: wgpu::Buffer,
    impulses: wgpu::Buffer,
    wall_impulses: wgpu::Buffer,
    contacts: u32,
    bytes: u64,
}

fn invalid() -> Error {
    Error::Solver("shared wall friction geometry: capacity, usage or device limit exceeded".into())
}

impl ExperimentalSoilSharedWallGeometryGpu {
    fn history_stride() -> u64 {
        #[cfg(test)]
        if std::env::var_os("SOIL_WALL_TEST_SPLIT_PREPARE").is_some() && redistribution_selected() {
            return 352;
        }
        336
    }

    pub fn required_bytes(capacity: SoilSharedWallGeometryCapacity) -> Result<u64> {
        Self::required_bytes_for_rates(capacity, RateEncoding::Scalar)
    }
    pub(crate) fn required_bytes_for_rates(
        capacity: SoilSharedWallGeometryCapacity,
        encoding: RateEncoding,
    ) -> Result<u64> {
        if !(1..=32).contains(&capacity.dofs)
            || !(1..=64).contains(&capacity.bodies)
            || capacity.contacts > 16384
            || !(1..=256).contains(&capacity.max_sweeps)
            || !(1..=64).contains(&capacity.local_iterations)
        {
            return Err(invalid());
        }
        let c = u64::from(capacity.contacts.max(1));
        let n = u64::from(capacity.dofs);
        if encoding == RateEncoding::Paired {
            let solver =
                ExperimentalSoilPairedWallFrictionGpu::required_bytes(SoilSharedWallCapacity {
                    dofs: capacity.dofs,
                    contacts: capacity.contacts,
                    max_sweeps: capacity.max_sweeps,
                    local_iterations: capacity.local_iterations,
                    max_owned_bytes: u64::MAX,
                })?;
            return Ok(solver + (176 + 20 * n) * c + 32);
        }
        Ok((368 + Self::history_stride()) * c + 36 * c * n + 4 * n * n + encoding.bytes(n) + 480)
    }
    pub fn new(
        engine: &Engine,
        input: SoilSharedWallGeometryInputs<'_>,
        capacity: SoilSharedWallGeometryCapacity,
    ) -> Result<Self> {
        Self::with_rate_encoding(engine, input, capacity, RateEncoding::Scalar)
    }
    pub(crate) fn with_rate_encoding(
        engine: &Engine,
        input: SoilSharedWallGeometryInputs<'_>,
        capacity: SoilSharedWallGeometryCapacity,
        encoding: RateEncoding,
    ) -> Result<Self> {
        engine.check_device()?;
        let bytes = Self::required_bytes_for_rates(capacity, encoding)?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let n = u64::from(capacity.dofs);
        let b = u64::from(capacity.bodies);
        let c = u64::from(capacity.contacts.max(1));
        if bytes > capacity.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 11
            || limits.max_compute_workgroup_storage_size < SHARED_WALL_WORKGROUP_BYTES
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_invocations_per_workgroup < 64
            || capacity.contacts.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || [
                64 * c,
                16 * c * n,
                112 * c,
                80 * c,
                if encoding == RateEncoding::Paired {
                    ExperimentalSoilPairedWallFrictionGpu::history_bytes(capacity.contacts)
                } else {
                    Self::history_stride() * c
                },
                if encoding == RateEncoding::Paired {
                    ExperimentalSoilPairedWallFrictionGpu::status_bytes()
                } else {
                    432
                },
                4 * n * n,
                encoding.bytes(n),
            ]
            .into_iter()
            .any(|size| {
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
        .zip([
            4 * n * n,
            encoding.bytes(n),
            16 * b,
            64 * b,
            24 * b * n,
            48 * c,
            4 * b,
        ]) {
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
        let contacts = create("shared wall contacts", 64 * c, usage);
        let point_rows = create("shared wall point rows", 16 * c * n, usage);
        let velocities = create("shared wall soil velocities", 16 * c, usage);
        let impulses = create("shared wall contact impulses", 16 * c, usage);
        let wall_impulses = create("shared wall external impulses", 16 * c, usage);
        let solver_capacity = SoilSharedWallCapacity {
            dofs: capacity.dofs,
            contacts: capacity.contacts,
            max_sweeps: capacity.max_sweeps,
            local_iterations: capacity.local_iterations,
            max_owned_bytes: bytes,
        };
        let solver_input = SoilSharedWallInputs {
            mass: input.mass,
            free_rates: input.free_rates,
            point_rows: &point_rows,
            contacts: &contacts,
        };
        let solver = match encoding {
            RateEncoding::Scalar => {
                ExperimentalSoilSharedWallFrictionGpu::new(engine, solver_input, solver_capacity)?
            }
            RateEncoding::Paired => {
                ExperimentalSoilPairedWallFrictionGpu::new(engine, solver_input, solver_capacity)?
                    .into_inner()
            }
        };
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
                include_str!("../shaders/soil_shared_wall_geometry.wgsl").into(),
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
            &point_rows,
            solver.output_buffer(),
            &velocities,
            &impulses,
            &wall_impulses,
            solver.status_buffer(),
            &params,
        ];
        let groups = std::array::from_fn(|stage| {
            let bindings: &[usize] = if stage == 0 {
                &[0, 1, 2, 3, 4, 5, 6, 11, 12]
            } else {
                &[0, 7, 8, 9, 10, 11, 12]
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
            wall_impulses,
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
    /// External wall impulse on soil; excluded from the machine reaction.
    pub fn wall_impulses_buffer(&self) -> &wgpu::Buffer {
        &self.wall_impulses
    }
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        self.solver.status_buffer()
    }
    pub(crate) fn local_inputs_buffer(&self) -> &wgpu::Buffer {
        self.solver.conditional_inputs_buffer()
    }
    pub(crate) fn failure_solver(&self) -> &ExperimentalSoilSharedWallFrictionGpu {
        &self.solver
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
    #[cfg(test)]
    pub(crate) fn trace_stages(
        &mut self,
        engine: std::sync::Arc<Engine>,
        path: std::path::PathBuf,
    ) {
        self.solver.trace_stages(engine, path);
    }
    #[cfg(test)]
    pub(crate) fn capture_first_input(
        &mut self,
        engine: &Engine,
    ) -> crate::soil_shared_wall_friction_gpu::capture::SharedWallSnapshot {
        self.solver.capture_first_input(engine)
    }
    #[cfg(test)]
    pub(crate) fn wall_trace(&self) -> [wgpu::Buffer; 5] {
        [
            self.velocities.clone(),
            self.impulses.clone(),
            self.wall_impulses.clone(),
            self.solver.conditional_inputs_buffer().clone(),
            self.solver.status_buffer().clone(),
        ]
    }
}

#[cfg(test)]
#[path = "soil_shared_wall_geometry_tests.rs"]
mod tests;
