//! Analytic contact geometry, GPU preparation and guarded ADMM world-space recovery.

use crate::soil_admm_gpu::{ExperimentalSoilAdmmGpu, SoilAdmmCapacity, SoilAdmmInputs};
use crate::soil_contact_geometry_gpu::*;
pub use crate::soil_friction_geometry_gpu::SoilFrictionGeometryInputs as SoilAdmmGeometryInputs;
use crate::soil_newton_geometry_gpu::{GeometryProgram, geometry_program};
use crate::soil_newton_prepare_gpu::{ExperimentalSoilNewtonPrepareGpu, SoilNewtonPrepareInputs};
use molla_compute::Engine;
use molla_core::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub struct SoilAdmmGeometryCapacity {
    pub dofs: u32,
    pub bodies: u32,
    pub contacts: u32,
    pub iterations: u32,
    pub max_owned_bytes: u64,
}

/// Uses paired generalized rates and explicit unique node/body queries.
/// Only zero-common-status output is consumable; faults latch until reset.
pub struct ExperimentalSoilAdmmGeometryGpu {
    geometry: ExperimentalSoilContactGeometryGpu,
    preparation: ExperimentalSoilNewtonPrepareGpu,
    solver: ExperimentalSoilAdmmGpu,
    program: std::sync::Arc<GeometryProgram>,
    groups: [wgpu::BindGroup; 2],
    outputs: [wgpu::Buffer; 3],
    status: wgpu::Buffer,
    contacts: u32,
    bytes: u64,
}

fn invalid() -> Error {
    Error::Solver(
        "soil ADMM geometry: invalid capacity, buffer usage, budget or device limit".into(),
    )
}

fn solver_capacity(cap: SoilAdmmGeometryCapacity) -> SoilAdmmCapacity {
    SoilAdmmCapacity {
        dofs: cap.dofs,
        contacts: cap.contacts,
        iterations: cap.iterations,
        max_owned_bytes: u64::MAX,
    }
}

impl ExperimentalSoilAdmmGeometryGpu {
    pub const fn preparation_backend_label() -> &'static str {
        ExperimentalSoilNewtonPrepareGpu::backend_label()
    }

    pub fn required_bytes(cap: SoilAdmmGeometryCapacity) -> Result<u64> {
        if !(1..=64).contains(&cap.bodies) {
            return Err(invalid());
        }
        Ok(
            ExperimentalSoilAdmmGpu::required_bytes(solver_capacity(cap))?
                + ExperimentalSoilNewtonPrepareGpu::required_bytes(cap.dofs, cap.contacts)?
                + (176 + 20 * u64::from(cap.dofs)) * u64::from(cap.contacts.max(1))
                + 64,
        )
    }

    pub fn new(
        engine: &Engine,
        input: SoilAdmmGeometryInputs<'_>,
        cap: SoilAdmmGeometryCapacity,
    ) -> Result<Self> {
        let bytes = Self::required_bytes(cap)?;
        engine.check_device()?;
        ExperimentalSoilNewtonPrepareGpu::check_selected_execution(engine, cap.contacts)?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let n = u64::from(cap.dofs);
        let b = u64::from(cap.bodies);
        let c = u64::from(cap.contacts.max(1));
        if bytes > cap.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 12
            || limits.max_bindings_per_bind_group < 16
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_storage_size < 38400
            || cap.contacts.max(1).div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || [
                96 * c,
                80 * c,
                16 * c * n,
                4 * n * n + 16 * n + 128,
                8 * n,
                32,
            ]
            .iter()
            .any(|&s| s > limits.max_buffer_size || s > limits.max_storage_buffer_binding_size)
            || [
                input.mass,
                input.free_rates,
                input.centers,
                input.shapes,
                input.jacobians,
                input.queries,
                input.coefficients,
            ]
            .iter()
            .zip([4 * n * n, 8 * n, 16 * b, 64 * b, 24 * b * n, 48 * c, 4 * b])
            .any(|(buffer, size)| {
                buffer.size() < size
                    || buffer.size() > limits.max_storage_buffer_binding_size
                    || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            })
        {
            return Err(invalid());
        }
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
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
        let nodes = create("ADMM geometry nodes", 48 * c, usage);
        let normals = create("ADMM geometry normal rows", 4 * c * n, usage);
        let contacts = create("ADMM geometry contacts", 64 * c, usage);
        let rows = create("ADMM geometry point rows", 16 * c * n, usage);
        let outputs = std::array::from_fn(|_| create("ADMM world output", 16 * c, usage));
        let status = create("ADMM geometry common status", 32, usage);
        let preparation = ExperimentalSoilNewtonPrepareGpu::new(
            engine,
            SoilNewtonPrepareInputs {
                mass: input.mass,
                point_rows: &rows,
                contacts: &contacts,
                upstream_status: &status,
            },
            cap.dofs,
            cap.contacts,
            u64::MAX,
        )?;
        let solver = ExperimentalSoilAdmmGpu::new(
            engine,
            SoilAdmmInputs {
                mass: input.mass,
                free_rates: input.free_rates,
                point_rows: &rows,
                point_response: preparation.response_buffer(),
                contacts: &contacts,
                upstream_status: preparation.status_buffer(),
            },
            solver_capacity(cap),
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
                rows: &normals,
                status: &status,
            },
            SoilContactGeometryCapacity {
                dofs: cap.dofs,
                bodies: cap.bodies,
                queries: cap.contacts,
                max_owned_bytes: 16 * c + 16,
            },
        )?;
        let params = create(
            "ADMM geometry dimensions",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[cap.dofs, cap.bodies, cap.contacts, 0])
        );
        let program = geometry_program(engine)?;
        let bindings = [
            &nodes,
            input.queries,
            input.centers,
            input.jacobians,
            input.coefficients,
            &contacts,
            &rows,
            solver.outputs_buffer(),
            &outputs[0],
            &outputs[1],
            &outputs[2],
            &status,
            &params,
            solver.solver_status_buffer(),
            preparation.status_buffer(),
            solver.status_buffer(),
        ];
        let groups = std::array::from_fn(|stage| {
            let indices: &[usize] = if stage == 0 {
                &[0, 1, 2, 3, 4, 5, 6, 11, 12]
            } else {
                &[0, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ADMM geometry stage"),
                layout: &program.pipelines[stage].get_bind_group_layout(0),
                entries: &indices
                    .iter()
                    .map(|&i| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: bindings[i].as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        });
        debug_assert_eq!(
            bytes,
            geometry.owned_buffer_bytes()
                + preparation.owned_buffer_bytes()
                + solver.owned_buffer_bytes()
                + (160 + 20 * n) * c
                + 48
        );
        Ok(Self {
            geometry,
            preparation,
            solver,
            program,
            groups,
            outputs,
            status,
            contacts: cap.contacts,
            bytes,
        })
    }

    pub fn encode_reset(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.status, 0, None);
        self.preparation.encode_reset(encoder);
        self.solver.encode_reset(encoder);
    }

    fn stage(&self, encoder: &mut wgpu::CommandEncoder, stage: usize, groups: u32) {
        if groups == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.program.pipelines[stage]);
        pass.set_bind_group(0, &self.groups[stage], &[]);
        pass.dispatch_workgroups(groups, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_preparation(encoder);
        self.encode_solve(encoder);
        self.encode_publication(encoder);
    }

    pub(crate) fn encode_preparation(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_preparation_traced(encoder, |_, _, _| {});
    }

    pub(crate) fn encode_preparation_traced(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        mut trace: impl FnMut(
            &mut wgpu::CommandEncoder,
            crate::soil_newton_prepare_gpu::PreparationPhase,
            bool,
        ),
    ) {
        use crate::soil_newton_prepare_gpu::PreparationPhase as P;
        trace(encoder, P::Reset, true);
        self.preparation.encode_reset(encoder);
        self.solver.encode_reset(encoder);
        trace(encoder, P::Reset, false);
        trace(encoder, P::Geometry, true);
        self.geometry.encode(encoder);
        trace(encoder, P::Geometry, false);
        trace(encoder, P::Contacts, true);
        self.stage(encoder, 0, self.contacts.div_ceil(64));
        trace(encoder, P::Contacts, false);
        self.preparation.encode_traced(encoder, &mut trace);
    }

    pub(crate) fn encode_solve(&self, encoder: &mut wgpu::CommandEncoder) {
        self.solver.encode_solve(encoder);
    }

    pub(crate) fn encode_publication(&self, encoder: &mut wgpu::CommandEncoder) {
        self.solver.encode_publication(encoder);
        self.stage(encoder, 1, self.contacts.max(1).div_ceil(64));
    }

    /// Common fault; preparation, solver and publication faults; steps, factors and reserved words.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        self.solver.rates_buffer()
    }
    pub fn velocities_buffer(&self) -> &wgpu::Buffer {
        &self.outputs[0]
    }
    pub fn contact_impulses_buffer(&self) -> &wgpu::Buffer {
        &self.outputs[1]
    }
    pub fn wall_impulses_buffer(&self) -> &wgpu::Buffer {
        &self.outputs[2]
    }
    pub fn geometry_buffer(&self) -> &wgpu::Buffer {
        self.geometry.geometry_buffer()
    }
    pub fn preparation_failure_buffer(&self) -> &wgpu::Buffer {
        self.preparation.failure_buffer()
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }

    #[cfg(test)]
    pub(crate) fn trace_buffers(&self) -> [wgpu::Buffer; 3] {
        [
            self.solver.active_indices_buffer().clone(),
            self.solver.diagnostic_buffers()[1].clone(),
            self.solver.status_buffer().clone(),
        ]
    }

    #[cfg(test)]
    pub(crate) fn shares_program(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.program, &other.program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admm_geometry_capacity_contract() {
        for n in [1, 3, 32] {
            for c in [0, 1, 65, 4096, 16384] {
                let cap = SoilAdmmGeometryCapacity {
                    dofs: n,
                    bodies: 1,
                    contacts: c,
                    iterations: 64,
                    max_owned_bytes: u64::MAX,
                };
                assert_eq!(
                    ExperimentalSoilAdmmGeometryGpu::required_bytes(cap).unwrap(),
                    ExperimentalSoilAdmmGpu::required_bytes(solver_capacity(cap)).unwrap()
                        + ExperimentalSoilNewtonPrepareGpu::required_bytes(n, c).unwrap()
                        + (176 + 20 * u64::from(n)) * u64::from(c.max(1))
                        + 64
                );
            }
        }
        for (n, b, c, i) in [
            (0, 1, 1, 64),
            (33, 1, 1, 64),
            (1, 0, 1, 64),
            (1, 65, 1, 64),
            (1, 1, 16385, 64),
            (1, 1, 1, 0),
            (1, 1, 1, 65),
        ] {
            assert!(
                ExperimentalSoilAdmmGeometryGpu::required_bytes(SoilAdmmGeometryCapacity {
                    dofs: n,
                    bodies: b,
                    contacts: c,
                    iterations: i,
                    max_owned_bytes: u64::MAX
                })
                .is_err()
            );
        }
    }
}
