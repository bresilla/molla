//! Prepared explicit MPM unilateral contact with certified candidates and floor response.

use super::*;
use crate::soil_admm_geometry_gpu::*;
use crate::soil_candidates_gpu::*;
use crate::soil_contact_geometry_gpu::*;
use crate::soil_contact_projection_gpu::SoilPairedRatePublicationInputs;
use crate::soil_contact_projection_gpu::{SoilRatePublicationGpu, SoilRatePublicationInputs};
use crate::soil_floor_projection_gpu::*;
use crate::soil_friction_geometry_gpu::*;
use crate::soil_rate_encoding::RateEncoding;

#[cfg(test)]
#[path = "mpm_paired_soil_tests.rs"]
mod paired_tests;
use crate::soil_newton_geometry_gpu::*;
use crate::soil_shared_wall_geometry_gpu::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ContactMethod {
    Projection,
    Newton,
    Admm,
}

enum ContactSolve {
    Normal {
        geometry: ExperimentalSoilContactGeometryGpu,
        floor: ExperimentalSoilFloorProjectionGpu,
    },
    Friction(ExperimentalSoilFrictionGeometryGpu),
    WallFriction(ExperimentalSoilSharedWallGeometryGpu),
    Newton(ExperimentalSoilNewtonGeometryGpu),
    Admm(ExperimentalSoilAdmmGeometryGpu),
}
impl ContactSolve {
    fn encode_preparation_traced(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        trace: impl FnMut(
            &mut wgpu::CommandEncoder,
            crate::soil_newton_prepare_gpu::PreparationPhase,
            bool,
        ),
    ) {
        match self {
            Self::Newton(p) => p.encode_preparation_traced(encoder, trace),
            Self::Admm(p) => p.encode_preparation_traced(encoder, trace),
            _ => unreachable!("phased contact method required"),
        }
    }
    fn encode_solve(&self, encoder: &mut wgpu::CommandEncoder) {
        match self {
            Self::Newton(p) => p.encode_solve(encoder),
            Self::Admm(p) => p.encode_solve(encoder),
            _ => unreachable!("phased contact method required"),
        }
    }
    fn encode_publication(&self, encoder: &mut wgpu::CommandEncoder) {
        match self {
            Self::Newton(p) => p.encode_publication(encoder),
            Self::Admm(p) => p.encode_publication(encoder),
            _ => unreachable!("phased contact method required"),
        }
    }
    fn status_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::Normal { floor, .. } => floor.status_buffer(),
            Self::Friction(p) => p.status_buffer(),
            Self::WallFriction(p) => p.status_buffer(),
            Self::Newton(p) => p.status_buffer(),
            Self::Admm(p) => p.status_buffer(),
        }
    }
    fn rates_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::Normal { floor, .. } => floor.rates_buffer(),
            Self::Friction(p) => p.rates_buffer(),
            Self::WallFriction(p) => p.rates_buffer(),
            Self::Newton(p) => p.rates_buffer(),
            Self::Admm(p) => p.rates_buffer(),
        }
    }
    fn velocities_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::Normal { floor, .. } => floor.velocities_buffer(),
            Self::Friction(p) => p.velocities_buffer(),
            Self::WallFriction(p) => p.velocities_buffer(),
            Self::Newton(p) => p.velocities_buffer(),
            Self::Admm(p) => p.velocities_buffer(),
        }
    }
    fn contact_impulses_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::Normal { floor, .. } => floor.contact_impulses_buffer(),
            Self::Friction(p) => p.contact_impulses_buffer(),
            Self::WallFriction(p) => p.contact_impulses_buffer(),
            Self::Newton(p) => p.contact_impulses_buffer(),
            Self::Admm(p) => p.contact_impulses_buffer(),
        }
    }
    fn geometry_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::Normal { geometry, .. } => geometry.geometry_buffer(),
            Self::Friction(p) => p.geometry_buffer(),
            Self::WallFriction(p) => p.geometry_buffer(),
            Self::Newton(p) => p.geometry_buffer(),
            Self::Admm(p) => p.geometry_buffer(),
        }
    }
    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        match self {
            Self::Normal { geometry, floor } => {
                geometry.encode(encoder);
                floor.encode(encoder);
            }
            Self::Friction(p) => p.encode(encoder),
            Self::WallFriction(p) => p.encode(encoder),
            Self::Newton(p) => p.encode(encoder),
            Self::Admm(p) => p.encode(encoder),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpmUnilateralConfig {
    pub node_capacity: u32,
    pub search_distance: f32,
    pub max_iterations: u32,
}
impl MpmUnilateralConfig {
    pub fn validate(self) -> Result<()> {
        if !(1..=16384).contains(&self.node_capacity)
            || !(1..=64).contains(&self.max_iterations)
            || !self.search_distance.is_finite()
            || self.search_distance < 0.0
        {
            return Err(invalid(
                "invalid unilateral capacity, search distance or iteration bound",
            ));
        }
        Ok(())
    }
}
pub struct MpmUnilateralInputs<'a> {
    pub mass: &'a wgpu::Buffer,
    pub jacobians: &'a wgpu::Buffer,
    pub free_rates: &'a wgpu::Buffer,
    pub dofs: u32,
    pub max_owned_bytes: u64,
}

pub struct PreparedMpmUnilateralCoupling {
    encoding: RateEncoding,
    engine: Arc<Engine>,
    step_epoch: Arc<()>,
    grid_window_cells: [i32; 3],
    rigid_epoch: Arc<()>,
    _lease: Arc<()>,
    grid: wgpu::Buffer,
    bodies: u32,
    dofs: u32,
    candidates: ExperimentalSoilCandidatesGpu,
    contact: ContactSolve,
    particle_fault: particle_fault_bridge::ParticleFaultBridge,
    history: Option<crate::soil_newton_history_gpu::NewtonHistory>,
    pipelines: [wgpu::ComputePipeline; 2],
    group: wgpu::BindGroup,
    indirect: wgpu::Buffer,
    contact_motion: crate::soil_contact_motion_gpu::ContactMotionGpu,
    bytes: u64,
}
fn invalid(message: &str) -> Error {
    Error::Solver(format!("MPM unilateral coupling: {message}"))
}

impl PreparedMpmUnilateralCoupling {
    fn candidate_cells(soil: &MpmSimulation) -> Result<u32> {
        if soil.use_active_grid {
            soil.active_tiles.capacity.checked_mul(64)
        } else {
            soil.cfg.grid_size.checked_pow(3)
        }
        .ok_or_else(|| invalid("candidate grid capacity overflow"))
    }

    pub(crate) fn required_bytes_for_grid(
        soil: &MpmSimulation,
        dense_requirement: impl FnOnce(u32) -> Result<u64>,
    ) -> Result<u64> {
        let size = if soil.use_active_grid {
            5
        } else {
            soil.cfg.grid_size
        };
        let dense = dense_requirement(size)?;
        let cells = Self::candidate_cells(soil)?;
        dense
            .checked_sub(ExperimentalSoilCandidatesGpu::grid_storage_bytes(
                size.pow(3),
                false,
            ))
            .and_then(|base| {
                base.checked_add(ExperimentalSoilCandidatesGpu::grid_storage_bytes(
                    cells,
                    soil.use_active_grid,
                ))
            })
            .ok_or_else(|| invalid("candidate grid budget overflow"))
    }

    pub(crate) fn required_bytes(
        config: MpmUnilateralConfig,
        dofs: u32,
        size: u32,
        bodies: u32,
    ) -> Result<u64> {
        config.validate()?;
        if !(1..=32).contains(&dofs) || !(5..=128).contains(&size) || !(1..=64).contains(&bodies) {
            return Err(invalid("unsupported machine or grid size"));
        }
        let total = u64::from(size).pow(3);
        let c = u64::from(config.node_capacity);
        let n = u64::from(dofs);
        Ok(20 * total
            + 4 * total.div_ceil(256)
            + 260 * c
            + 20 * c * n
            + 4 * n
            + 224
            + crate::soil_contact_motion_gpu::ContactMotionGpu::bytes(bodies))
    }
    pub(crate) fn required_friction_bytes(
        config: MpmUnilateralConfig,
        dofs: u32,
        size: u32,
        bodies: u32,
    ) -> Result<u64> {
        let normal = Self::required_bytes(config, dofs, size, bodies)?;
        Ok(normal
            - 32 * u64::from(config.node_capacity)
            - 8 * u64::from(config.node_capacity) * u64::from(dofs)
            + 4 * u64::from(bodies))
    }
    #[cfg(test)]
    pub(crate) fn required_wall_friction_bytes(
        config: MpmUnilateralConfig,
        dofs: u32,
        size: u32,
        bodies: u32,
    ) -> Result<u64> {
        Self::required_wall_friction_bytes_for_rates(
            config,
            dofs,
            size,
            bodies,
            RateEncoding::Scalar,
        )
    }
    pub(crate) fn required_wall_friction_bytes_for_rates(
        config: MpmUnilateralConfig,
        dofs: u32,
        size: u32,
        bodies: u32,
        encoding: RateEncoding,
    ) -> Result<u64> {
        let base = Self::required_friction_bytes(config, dofs, size, bodies)?;
        let old =
            ExperimentalSoilFrictionGeometryGpu::required_bytes(SoilFrictionGeometryCapacity {
                dofs,
                bodies,
                contacts: config.node_capacity,
                max_iterations: config.max_iterations,
                max_owned_bytes: u64::MAX,
            })?;
        let new = ExperimentalSoilSharedWallGeometryGpu::required_bytes_for_rates(
            SoilSharedWallGeometryCapacity {
                dofs,
                bodies,
                contacts: config.node_capacity,
                max_sweeps: config.max_iterations,
                local_iterations: 64,
                max_owned_bytes: u64::MAX,
            },
            encoding,
        )?;
        Ok(base - old + new)
    }

    pub fn new(
        engine: Arc<Engine>,
        soil: &mut MpmSimulation,
        input: MpmUnilateralInputs<'_>,
        config: MpmUnilateralConfig,
    ) -> Result<Self> {
        Self::new_internal(
            engine,
            soil,
            input,
            config,
            None,
            RateEncoding::Scalar,
            ContactMethod::Projection,
        )
    }
    pub(crate) fn required_newton_bytes(
        config: MpmUnilateralConfig,
        dofs: u32,
        size: u32,
        bodies: u32,
    ) -> Result<u64> {
        let base = Self::required_friction_bytes(config, dofs, size, bodies)?;
        let old =
            ExperimentalSoilFrictionGeometryGpu::required_bytes(SoilFrictionGeometryCapacity {
                dofs,
                bodies,
                contacts: config.node_capacity,
                max_iterations: config.max_iterations,
                max_owned_bytes: u64::MAX,
            })?;
        let new = ExperimentalSoilNewtonGeometryGpu::required_bytes(SoilNewtonGeometryCapacity {
            dofs,
            bodies,
            contacts: config.node_capacity,
            newton_iterations: config.max_iterations,
            krylov_iterations: 64,
            max_owned_bytes: u64::MAX,
        })?;
        let history = if cfg!(feature = "warm-newton") {
            crate::soil_newton_history_gpu::NewtonHistory::required_bytes(config.node_capacity)?
        } else {
            0
        };
        Ok(base - old + new + history)
    }
    pub(crate) fn new_with_newton_rates(
        engine: Arc<Engine>,
        soil: &mut MpmSimulation,
        input: MpmUnilateralInputs<'_>,
        config: MpmUnilateralConfig,
        coefficients: &[f32],
    ) -> Result<Self> {
        Self::new_internal(
            engine,
            soil,
            input,
            config,
            Some((coefficients, true)),
            RateEncoding::Paired,
            ContactMethod::Newton,
        )
    }

    /// Dense-grid owned bytes for experimental ADMM, excluding borrowed inputs.
    pub fn required_admm_bytes(
        config: MpmUnilateralConfig,
        dofs: u32,
        size: u32,
        bodies: u32,
    ) -> Result<u64> {
        let base = Self::required_friction_bytes(config, dofs, size, bodies)?;
        let old =
            ExperimentalSoilFrictionGeometryGpu::required_bytes(SoilFrictionGeometryCapacity {
                dofs,
                bodies,
                contacts: config.node_capacity,
                max_iterations: config.max_iterations,
                max_owned_bytes: u64::MAX,
            })?;
        let new = ExperimentalSoilAdmmGeometryGpu::required_bytes(SoilAdmmGeometryCapacity {
            dofs,
            bodies,
            contacts: config.node_capacity,
            iterations: config.max_iterations,
            max_owned_bytes: u64::MAX,
        })?;
        Ok(base - old + new)
    }

    /// Experimental guarded ADMM with paired rates and per-body friction.
    /// Candidate, numerical or particle faults latch in the common status buffer.
    pub fn new_experimental_admm(
        engine: Arc<Engine>,
        soil: &mut MpmSimulation,
        input: MpmUnilateralInputs<'_>,
        config: MpmUnilateralConfig,
        coefficients: &[f32],
    ) -> Result<Self> {
        Self::new_internal(
            engine,
            soil,
            input,
            config,
            Some((coefficients, true)),
            RateEncoding::Paired,
            ContactMethod::Admm,
        )
    }

    /// Owned bytes for the simulation's current dense, active or compact grid.
    pub fn required_admm_bytes_for_soil(
        soil: &MpmSimulation,
        config: MpmUnilateralConfig,
        dofs: u32,
    ) -> Result<u64> {
        Self::required_bytes_for_grid(soil, |size| {
            Self::required_admm_bytes(config, dofs, size, soil.rigid_count)
        })
    }
    pub(crate) fn new_with_friction_rates(
        engine: Arc<Engine>,
        soil: &mut MpmSimulation,
        input: MpmUnilateralInputs<'_>,
        config: MpmUnilateralConfig,
        coefficients: &[f32],
        fixed_walls: bool,
        encoding: RateEncoding,
    ) -> Result<Self> {
        Self::new_internal(
            engine,
            soil,
            input,
            config,
            Some((coefficients, fixed_walls)),
            encoding,
            ContactMethod::Projection,
        )
    }
    fn new_internal(
        engine: Arc<Engine>,
        soil: &mut MpmSimulation,
        input: MpmUnilateralInputs<'_>,
        config: MpmUnilateralConfig,
        friction: Option<(&[f32], bool)>,
        encoding: RateEncoding,
        method: ContactMethod,
    ) -> Result<Self> {
        if encoding == RateEncoding::Paired && !friction.is_some_and(|(_, walls)| walls) {
            return Err(invalid("paired rates require shared wall friction"));
        }
        engine.check_device()?;
        let device = engine.device().ok_or_else(|| invalid("GPU required"))?;
        crate::kernels::validate_soil_device(device, &soil.device)?;
        super::mass_coupling::PreparedMpmMassCoupling::validate_mode(soil)?;
        let candidate_cells = Self::candidate_cells(soil)?;
        let bytes = Self::required_bytes_for_grid(soil, |budget_size| {
            if let Some((coefficients, fixed_walls)) = friction {
                if coefficients.len() != soil.rigid_count as usize
                    || coefficients.iter().any(|v| !v.is_finite() || *v < 0.0)
                {
                    return Err(invalid("friction coefficients must match live body order"));
                }
                if method == ContactMethod::Admm {
                    Self::required_admm_bytes(config, input.dofs, budget_size, soil.rigid_count)
                } else if method == ContactMethod::Newton {
                    Self::required_newton_bytes(config, input.dofs, budget_size, soil.rigid_count)
                } else if fixed_walls {
                    Self::required_wall_friction_bytes_for_rates(
                        config,
                        input.dofs,
                        budget_size,
                        soil.rigid_count,
                        encoding,
                    )
                } else {
                    Self::required_friction_bytes(config, input.dofs, budget_size, soil.rigid_count)
                }
            } else {
                Self::required_bytes(config, input.dofs, budget_size, soil.rigid_count)
            }
        })?;
        if method == ContactMethod::Admm {
            crate::soil_newton_prepare_gpu::ExperimentalSoilNewtonPrepareGpu::check_selected_execution(
                &engine,
                config.node_capacity,
            )?;
        }
        let limits = device.limits();
        let total = soil.g_mass.size() / 4;
        let n = u64::from(input.dofs);
        let c = u64::from(config.node_capacity);
        let b = u64::from(soil.rigid_count);
        if bytes > input.max_owned_bytes
            || !(1..=64).contains(&soil.rigid_count)
            || limits.max_compute_workgroup_storage_size
                < if encoding == RateEncoding::Paired {
                    2116
                } else {
                    2112
                }
            || limits.max_compute_workgroup_storage_size
                < if method == ContactMethod::Admm {
                    38400
                } else if friction.is_some_and(|(_, walls)| !walls) {
                    9552
                } else if friction.is_some() {
                    crate::soil_shared_wall_friction_gpu::SHARED_WALL_WORKGROUP_BYTES
                } else {
                    crate::soil_normal_projection_gpu::hard_storage_bytes(input.dofs)
                }
            || (method == ContactMethod::Admm && limits.max_bindings_per_bind_group < 16)
            || (method == ContactMethod::Newton
                && (limits.max_compute_workgroup_storage_size
                    < crate::soil_newton_gpu::execution_limits(&engine).1
                    || limits.max_bindings_per_bind_group < 22
                    || 64 * c * 65 > limits.max_storage_buffer_binding_size
                    || 64 * c * 65 > limits.max_buffer_size))
            || limits.max_storage_buffers_per_shader_stage
                < if method == ContactMethod::Newton {
                    21
                } else if soil.use_active_grid {
                    14
                } else {
                    12
                }
            || limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_size_x < 256
            || u64::from(candidate_cells).max(c).div_ceil(256)
                > u64::from(limits.max_compute_workgroups_per_dimension)
            || (soil.use_active_grid
                && candidate_cells.div_ceil(64) > limits.max_compute_workgroups_per_dimension)
            || [
                20 * u64::from(candidate_cells),
                16 * c * n,
                if friction.is_some_and(|(_, walls)| walls) {
                    336 * c
                } else {
                    112 * c
                },
                4 * n * n,
                if method == ContactMethod::Newton {
                    33280 + 432 * c.max(1)
                } else {
                    0
                },
                if method == ContactMethod::Admm {
                    4 * n * n + 16 * n + 128
                } else {
                    0
                },
                if method == ContactMethod::Newton {
                    crate::soil_newton_failure::NewtonFailureCapture::required_bytes(
                        n as u32, c as u32,
                    )? - 16
                } else {
                    0
                },
            ]
            .into_iter()
            .any(|size| {
                size > limits.max_storage_buffer_binding_size || size > limits.max_buffer_size
            })
        {
            return Err(invalid("coupling budget or workgroup storage exceeded"));
        }
        for (buffer, required) in [
            input.mass,
            input.free_rates,
            input.jacobians,
            &soil.g_velocity,
            &soil.rigid_center,
            &soil.rigid_vel,
            &soil.rigid_wrench,
        ]
        .iter()
        .zip([
            4 * n * n,
            encoding.bytes(n),
            24 * b * n,
            16 * total,
            16 * b,
            64 * b,
            32 * b,
        ]) {
            if buffer.size() < required
                || buffer.size() > limits.max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid("input buffer capacity or usage mismatch"));
            }
        }
        let make = |label, size, usage| {
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
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let c = u64::from(config.node_capacity);
        let n = u64::from(input.dofs);
        let queries = make("MPM unilateral queries", 48 * c, storage);
        let ids = make("MPM unilateral node IDs", 4 * c, storage);
        let contact = if let Some((values, fixed_walls)) = friction {
            let coefficients = make("MPM body friction coefficients", 4 * b, storage);
            molla_core::gpu_write_buffer!(
                engine.queue().unwrap(),
                &coefficients,
                0,
                bytemuck::cast_slice(values)
            );
            if method == ContactMethod::Admm {
                ContactSolve::Admm(ExperimentalSoilAdmmGeometryGpu::new(
                    &engine,
                    SoilAdmmGeometryInputs {
                        mass: input.mass,
                        free_rates: input.free_rates,
                        centers: &soil.rigid_center,
                        shapes: &soil.rigid_vel,
                        jacobians: input.jacobians,
                        queries: &queries,
                        coefficients: &coefficients,
                    },
                    SoilAdmmGeometryCapacity {
                        dofs: input.dofs,
                        bodies: soil.rigid_count,
                        contacts: config.node_capacity,
                        iterations: config.max_iterations,
                        max_owned_bytes: bytes,
                    },
                )?)
            } else if method == ContactMethod::Newton {
                ContactSolve::Newton(ExperimentalSoilNewtonGeometryGpu::new(
                    &engine,
                    SoilNewtonGeometryInputs {
                        mass: input.mass,
                        free_rates: input.free_rates,
                        centers: &soil.rigid_center,
                        shapes: &soil.rigid_vel,
                        jacobians: input.jacobians,
                        queries: &queries,
                        coefficients: &coefficients,
                    },
                    SoilNewtonGeometryCapacity {
                        dofs: input.dofs,
                        bodies: soil.rigid_count,
                        contacts: config.node_capacity,
                        newton_iterations: config.max_iterations,
                        krylov_iterations: 64,
                        max_owned_bytes: bytes,
                    },
                )?)
            } else if fixed_walls {
                ContactSolve::WallFriction(
                    ExperimentalSoilSharedWallGeometryGpu::with_rate_encoding(
                        &engine,
                        SoilSharedWallGeometryInputs {
                            mass: input.mass,
                            free_rates: input.free_rates,
                            centers: &soil.rigid_center,
                            shapes: &soil.rigid_vel,
                            jacobians: input.jacobians,
                            queries: &queries,
                            coefficients: &coefficients,
                        },
                        SoilSharedWallGeometryCapacity {
                            dofs: input.dofs,
                            bodies: soil.rigid_count,
                            contacts: config.node_capacity,
                            max_sweeps: config.max_iterations,
                            local_iterations: 64,
                            max_owned_bytes: bytes,
                        },
                        encoding,
                    )?,
                )
            } else {
                ContactSolve::Friction(ExperimentalSoilFrictionGeometryGpu::new(
                    &engine,
                    SoilFrictionGeometryInputs {
                        mass: input.mass,
                        free_rates: input.free_rates,
                        centers: &soil.rigid_center,
                        shapes: &soil.rigid_vel,
                        jacobians: input.jacobians,
                        queries: &queries,
                        coefficients: &coefficients,
                    },
                    SoilFrictionGeometryCapacity {
                        dofs: input.dofs,
                        bodies: soil.rigid_count,
                        contacts: config.node_capacity,
                        max_iterations: config.max_iterations,
                        max_owned_bytes: bytes,
                    },
                )?)
            }
        } else {
            let nodes = make("MPM unilateral floor nodes", 48 * c, storage);
            let rows = make("MPM unilateral machine rows", 4 * c * n, storage);
            let floor = ExperimentalSoilFloorProjectionGpu::new(
                &engine,
                SoilFloorInputs {
                    mass: input.mass,
                    free_rates: input.free_rates,
                    rows: &rows,
                    nodes: &nodes,
                },
                SoilFloorCapacity {
                    dofs: input.dofs,
                    nodes: config.node_capacity,
                    max_iterations: config.max_iterations,
                    max_owned_bytes: bytes,
                },
            )?;
            let geometry = ExperimentalSoilContactGeometryGpu::new(
                &engine,
                SoilContactGeometryInputs {
                    centers: &soil.rigid_center,
                    shapes: &soil.rigid_vel,
                    jacobians: input.jacobians,
                    queries: &queries,
                },
                SoilContactGeometryOutputs {
                    nodes: &nodes,
                    rows: &rows,
                    status: floor.status_buffer(),
                },
                SoilContactGeometryCapacity {
                    dofs: input.dofs,
                    bodies: soil.rigid_count,
                    queries: config.node_capacity,
                    max_owned_bytes: bytes,
                },
            )?;
            ContactSolve::Normal { geometry, floor }
        };
        let candidates = ExperimentalSoilCandidatesGpu::with_grid_window(
            &engine,
            SoilCandidateInputs {
                grid: &soil.g_velocity,
                centers: &soil.rigid_center,
                shapes: &soil.rigid_vel,
                jacobians: input.jacobians,
                corrected_rates: contact.rates_buffer(),
                corrected_velocities: contact.velocities_buffer(),
            },
            SoilCandidateOutputs {
                queries: &queries,
                node_ids: &ids,
                status: contact.status_buffer(),
            },
            SoilCandidateConfig {
                origin: soil.cfg.grid_origin,
                cell_size: soil.cfg.cell_size,
                grid_size: soil.cfg.grid_size,
                bodies: soil.rigid_count,
                dofs: input.dofs,
                capacity: config.node_capacity,
                dt: 1.0,
                search_distance: config.search_distance,
                max_owned_bytes: bytes,
            },
            encoding,
            soil.use_active_grid.then_some(SoilCandidateGridIndex {
                metadata: &soil.g_kdiag,
                metadata_word: (soil.active_tiles.metadata_offset / 4) as u32,
                tile_ids_word: (soil.active_tiles.metadata_offset / 4
                    + 4
                    + u64::from(soil.active_tiles.map_capacity)
                        * if soil.sparse_grid { 2 } else { 1 })
                    as u32,
                tile_capacity: soil.active_tiles.capacity,
                sparse: soil.sparse_grid,
            }),
            soil.grid_window_cells,
        )?;
        let indirect = make(
            "MPM unilateral guarded particles",
            12,
            storage | wgpu::BufferUsages::INDIRECT,
        );
        let particles = make(
            "MPM unilateral particle groups",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &particles,
            0,
            bytemuck::cast_slice(&[soil.particle_count.div_ceil(WORKGROUP_SIZE), 0u32, 0, 0])
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MPM unilateral reactions"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mpm_unilateral_reactions.wgsl").into(),
            ),
        });
        let entries: Vec<_> = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding >= 6 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 3,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unilateral reactions"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("unilateral reactions"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ["reactions", "finish"].map(|entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unilateral reactions"),
            layout: &layout,
            entries: &[
                &soil.rigid_center,
                &queries,
                contact.contact_impulses_buffer(),
                &soil.rigid_wrench,
                contact.status_buffer(),
                &indirect,
                candidates.parameters_buffer(),
                &particles,
            ]
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        let lease = soil.rigid_bridge.upgrade().unwrap_or_else(|| Arc::new(()));
        let contact_motion = crate::soil_contact_motion_gpu::ContactMotionGpu::with_rate_encoding(
            &engine,
            [
                &queries,
                contact.geometry_buffer(),
                contact.velocities_buffer(),
                contact.contact_impulses_buffer(),
                input.jacobians,
                contact.rates_buffer(),
                &soil.rigid_center,
                contact.status_buffer(),
            ],
            input.dofs,
            soil.rigid_count,
            config.node_capacity,
            bytes,
            encoding,
        )?;
        let history = if cfg!(feature = "warm-newton") {
            if let ContactSolve::Newton(owner) = &contact {
                Some(crate::soil_newton_history_gpu::NewtonHistory::new(
                    &engine,
                    owner.history_inputs(&ids, &queries),
                    config.node_capacity,
                    soil.cfg.grid_size.pow(3),
                    bytes,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        soil.rigid_bridge = Arc::downgrade(&lease);
        let particle_fault =
            particle_fault_bridge::ParticleFaultBridge::new(soil, contact.status_buffer());
        Ok(Self {
            particle_fault,
            encoding,
            engine,
            step_epoch: soil.step_epoch.clone(),
            grid_window_cells: soil.grid_window_cells,
            rigid_epoch: soil.rigid_epoch.clone(),
            _lease: lease,
            grid: soil.g_velocity.clone(),
            bodies: soil.rigid_count,
            dofs: input.dofs,
            candidates,
            contact,
            history,
            pipelines,
            group,
            indirect,
            contact_motion,
            bytes,
        })
    }
    pub fn step(&mut self, soil: &mut MpmSimulation, dt: f32) -> Result<()> {
        self.step_with_commands(soil, dt, |_| {}, |_| {})
    }

    pub(crate) fn step_with_commands(
        &mut self,
        soil: &mut MpmSimulation,
        dt: f32,
        before: impl FnOnce(&mut wgpu::CommandEncoder),
        after: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> Result<()> {
        self.engine.check_device()?;
        super::mass_coupling::PreparedMpmMassCoupling::validate_mode(soil)?;
        soil.validate_prescribed_step(dt)?;
        if !Arc::ptr_eq(&self.step_epoch, &soil.step_epoch)
            || self.grid_window_cells != soil.grid_window_cells
            || !Arc::ptr_eq(&self.rigid_epoch, &soil.rigid_epoch)
            || self.grid != soil.g_velocity
            || self.bodies != soil.rigid_count
        {
            return Err(invalid("stale prepared configuration"));
        }
        self.candidates.set_dt(self.engine.queue().unwrap(), dt)?;
        super::mass_coupling::step_projected(
            &self.engine,
            soil,
            dt,
            false,
            &self.indirect,
            before,
            |soil, encoder, profiled| {
                let detailed = profiled
                    && matches!(
                        self.contact,
                        ContactSolve::Newton(_) | ContactSolve::Admm(_)
                    );
                let details = detailed && soil.preparation_profile_enabled();
                let preparation_end = soil.profile_begin_scope(
                    detailed,
                    encoder,
                    MpmProfileStage::ContactPreparation,
                    Some(0),
                );
                let candidates_end = soil.profile_begin_scope(
                    details,
                    encoder,
                    MpmProfileStage::PreparationCandidates,
                    Some(0),
                );
                self.candidates.encode_candidates(encoder);
                soil.profile_end_scope(encoder, candidates_end);
                if matches!(
                    self.contact,
                    ContactSolve::Newton(_) | ContactSolve::Admm(_)
                ) {
                    let mut phase_end = None;
                    self.contact
                        .encode_preparation_traced(encoder, |encoder, phase, begin| {
                            use crate::soil_newton_prepare_gpu::PreparationPhase as P;
                            let stage = match phase {
                                P::Reset => MpmProfileStage::PreparationReset,
                                P::Geometry => MpmProfileStage::PreparationGeometry,
                                P::Contacts => MpmProfileStage::PreparationContacts,
                                P::Factor => MpmProfileStage::PreparationFactor,
                                P::Response => MpmProfileStage::PreparationResponse,
                                P::Cleanup => MpmProfileStage::PreparationCleanup,
                            };
                            if begin {
                                phase_end =
                                    soil.profile_begin_scope(details, encoder, stage, Some(0));
                            } else {
                                soil.profile_end_scope(encoder, phase_end.take());
                            }
                        });
                    let history_end = soil.profile_begin_scope(
                        details,
                        encoder,
                        MpmProfileStage::PreparationHistory,
                        Some(0),
                    );
                    if let Some(history) = &self.history {
                        history.restore(encoder);
                    }
                    soil.profile_end_scope(encoder, history_end);
                    soil.profile_end_scope(encoder, preparation_end);
                    soil.profile_begin_encoder(
                        detailed,
                        encoder,
                        MpmProfileStage::ContactSolve,
                        Some(0),
                    );
                    self.contact.encode_solve(encoder);
                    soil.profile_end_encoder(detailed, encoder);
                    soil.profile_begin_encoder(
                        detailed,
                        encoder,
                        MpmProfileStage::ContactPublication,
                        Some(0),
                    );
                    self.contact.encode_publication(encoder);
                } else {
                    self.contact.encode(encoder);
                }
                self.candidates.encode_certify_and_publish_grid(encoder);
                for (pipeline, groups) in self.pipelines.iter().zip([self.bodies, 1]) {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &self.group, &[]);
                    pass.dispatch_workgroups(groups, 1, 1);
                    molla_core::gpu_metrics::record_dispatch();
                }
                soil.profile_end_encoder(detailed, encoder);
            },
            |encoder| {
                self.particle_fault.encode(encoder);
                if let Some(history) = &self.history {
                    history.save(encoder);
                }
            },
            after,
        )
    }
    pub(crate) fn indexed_sort_backend(&self) -> Option<&'static str> {
        self.candidates.indexed_sort_backend()
    }
    pub(crate) fn newton_method_name(&self) -> Option<&'static str> {
        match &self.contact {
            ContactSolve::Newton(p) => Some(p.method_name()),
            _ => None,
        }
    }
    pub(crate) fn invalidate_newton_history(&self, queue: &wgpu::Queue) {
        if let Some(history) = &self.history {
            history.invalidate(queue);
        }
    }
    pub(crate) fn use_previous_newton_history(&mut self, previous: &Self) -> Result<()> {
        if !Arc::ptr_eq(&self.engine, &previous.engine)
            || self.grid != previous.grid
            || self.dofs != previous.dofs
            || self.bodies != previous.bodies
        {
            return Err(invalid(
                "history owners must share the same soil and machine layout",
            ));
        }
        match (&mut self.history, &previous.history) {
            (Some(current), Some(previous)) => {
                current.use_previous(self.engine.device().unwrap(), previous)
            }
            (None, None) => Ok(()),
            _ => Err(invalid("incompatible Newton history owners")),
        }
    }
    pub(crate) fn newton_seed_buffer(&self) -> Option<&wgpu::Buffer> {
        self.history.as_ref().map(|h| h.seed_buffer())
    }
    #[cfg(test)]
    pub(crate) fn newton_success_seed_buffer(&self) -> Option<&wgpu::Buffer> {
        match &self.contact {
            ContactSolve::Newton(p) => p.success_seed_buffer(),
            _ => None,
        }
    }
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        self.contact.status_buffer()
    }

    #[cfg(test)]
    pub(crate) fn admm_trace_sources(&self) -> Option<(u32, [wgpu::Buffer; 3])> {
        match &self.contact {
            ContactSolve::Admm(p) => Some((self.dofs, p.trace_buffers())),
            _ => None,
        }
    }

    /// ADMM preparation failure packet; absent for other contact methods.
    pub fn admm_preparation_failure_buffer(&self) -> Option<&wgpu::Buffer> {
        match &self.contact {
            ContactSolve::Admm(p) => Some(p.preparation_failure_buffer()),
            _ => None,
        }
    }

    pub(crate) fn candidate_failure_buffer(&self) -> &wgpu::Buffer {
        self.candidates.count_buffer()
    }
    pub(crate) fn newton_preparation_failure_buffer(&self) -> Option<&wgpu::Buffer> {
        match &self.contact {
            ContactSolve::Newton(p) => Some(p.preparation_failure_buffer()),
            _ => None,
        }
    }
    pub(crate) fn newton_failure_buffer(&self) -> Option<&wgpu::Buffer> {
        match &self.contact {
            ContactSolve::Newton(p) => Some(p.failure_buffer()),
            _ => None,
        }
    }
    #[cfg(all(
        test,
        feature = "active-publication-newton",
        not(target_arch = "wasm32")
    ))]
    pub(crate) fn newton_shadow_buffer(&self) -> Option<&wgpu::Buffer> {
        match &self.contact {
            ContactSolve::Newton(p) => p.shadow_summary(),
            _ => None,
        }
    }
    pub(crate) fn local_inputs_buffer(&self) -> Option<&wgpu::Buffer> {
        match &self.contact {
            ContactSolve::WallFriction(p) => Some(p.local_inputs_buffer()),
            _ => None,
        }
    }
    pub(crate) fn failure_solver(
        &self,
    ) -> Option<&crate::soil_shared_wall_friction_gpu::ExperimentalSoilSharedWallFrictionGpu> {
        match &self.contact {
            ContactSolve::WallFriction(p) => Some(p.failure_solver()),
            _ => None,
        }
    }
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        self.contact.rates_buffer()
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
    #[cfg(test)]
    pub(crate) fn trace_wall_stages(&mut self, path: std::path::PathBuf) {
        match &mut self.contact {
            ContactSolve::WallFriction(p) => p.trace_stages(self.engine.clone(), path),
            _ => panic!("wall trace requires shared wall friction"),
        }
    }
    #[cfg(test)]
    pub(crate) fn capture_first_wall_input(
        &mut self,
    ) -> crate::soil_shared_wall_friction_gpu::capture::SharedWallSnapshot {
        match &mut self.contact {
            ContactSolve::WallFriction(p) => p.capture_first_input(&self.engine),
            _ => panic!("first wall input capture requires shared wall friction"),
        }
    }
    #[cfg(test)]
    pub(crate) fn wall_trace(&self) -> [wgpu::Buffer; 5] {
        match &self.contact {
            ContactSolve::WallFriction(p) => p.wall_trace(),
            _ => panic!("wall trace requires shared wall friction"),
        }
    }

    pub(crate) fn contact_motion(&self) -> &crate::soil_contact_motion_gpu::ContactMotionGpu {
        &self.contact_motion
    }

    #[cfg(test)]
    pub(crate) fn prepare_reference_check(&mut self, source: &str) -> wgpu::Buffer {
        match &mut self.contact {
            ContactSolve::Normal { floor, .. } => {
                floor.prepare_reference_check(&self.engine, source)
            }
            ContactSolve::Friction(_)
            | ContactSolve::WallFriction(_)
            | ContactSolve::Newton(_)
            | ContactSolve::Admm(_) => {
                panic!("normal reference requires frictionless coupling")
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn prepare_failure_capture(
        &mut self,
    ) -> crate::soil_normal_projection_gpu::replay::NormalReplay {
        match &mut self.contact {
            ContactSolve::Normal { floor, .. } => floor.prepare_failure_capture(&self.engine),
            ContactSolve::Friction(_)
            | ContactSolve::WallFriction(_)
            | ContactSolve::Newton(_)
            | ContactSolve::Admm(_) => {
                panic!("normal capture requires frictionless coupling")
            }
        }
    }
    pub fn prepare_machine_publication(
        &self,
        input: SoilRatePublicationInputs<'_>,
    ) -> Result<SoilRatePublicationGpu> {
        if self.encoding != RateEncoding::Scalar {
            return Err(invalid("scalar publication cannot consume paired rates"));
        }
        SoilRatePublicationGpu::from_buffers(
            &self.engine,
            self.rates_buffer(),
            self.status_buffer(),
            self.dofs,
            input,
        )
    }
    pub fn prepare_paired_machine_publication(
        &self,
        input: SoilPairedRatePublicationInputs<'_>,
    ) -> Result<SoilRatePublicationGpu> {
        if self.encoding != RateEncoding::Paired {
            return Err(invalid("paired publication requires paired rates"));
        }
        SoilRatePublicationGpu::from_paired_buffers(
            &self.engine,
            self.rates_buffer(),
            self.status_buffer(),
            self.dofs,
            input,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admm_mpm_owned_budget_contract() {
        for dofs in [1, 3, 32] {
            for slots in [1, 128, 16384] {
                for bodies in [1, 64] {
                    let config = MpmUnilateralConfig {
                        node_capacity: slots,
                        search_distance: 0.25,
                        max_iterations: 64,
                    };
                    let n = u64::from(dofs);
                    let c = u64::from(slots);
                    let b = u64::from(bodies);
                    let adapter = (484 + 52 * n) * c + 8 * n * n + 24 * n + 340;
                    let orchestration = 20 * 4096 + 4 * 16 + 52 * c + 160 + 4 * b + 96 * b + 16;
                    assert_eq!(
                        PreparedMpmUnilateralCoupling::required_admm_bytes(
                            config, dofs, 16, bodies
                        )
                        .unwrap(),
                        adapter + orchestration,
                    );
                }
            }
        }
        let config = MpmUnilateralConfig {
            node_capacity: 128,
            search_distance: 0.25,
            max_iterations: 64,
        };
        assert_eq!(
            PreparedMpmUnilateralCoupling::required_admm_bytes(config, 3, 16, 1).unwrap(),
            171320
        );
        for (n, size, b) in [
            (0, 16, 1),
            (33, 16, 1),
            (3, 4, 1),
            (3, 129, 1),
            (3, 16, 0),
            (3, 16, 65),
        ] {
            assert!(
                PreparedMpmUnilateralCoupling::required_admm_bytes(config, n, size, b).is_err()
            );
        }
        for config in [
            MpmUnilateralConfig {
                node_capacity: 0,
                ..config
            },
            MpmUnilateralConfig {
                node_capacity: 16385,
                ..config
            },
            MpmUnilateralConfig {
                max_iterations: 0,
                ..config
            },
            MpmUnilateralConfig {
                max_iterations: 65,
                ..config
            },
            MpmUnilateralConfig {
                search_distance: f32::NAN,
                ..config
            },
        ] {
            assert!(PreparedMpmUnilateralCoupling::required_admm_bytes(config, 3, 16, 1).is_err());
        }
    }

    #[test]
    fn unilateral_configuration_and_owned_bytes() {
        let config = MpmUnilateralConfig {
            node_capacity: 512,
            search_distance: 0.05,
            max_iterations: 64,
        };
        assert!(config.validate().is_ok());
        assert_eq!(
            PreparedMpmUnilateralCoupling::required_bytes(config, 10, 48, 5).unwrap(),
            2_449_848
        );
        assert_eq!(
            PreparedMpmUnilateralCoupling::required_friction_bytes(config, 10, 48, 5).unwrap(),
            2_392_524
        );
        let maximum = MpmUnilateralConfig {
            node_capacity: 16384,
            search_distance: 0.0,
            max_iterations: 1,
        };
        assert_eq!(
            PreparedMpmUnilateralCoupling::required_wall_friction_bytes(config, 10, 48, 5).unwrap(),
            2_786_556
        );
        assert_eq!(
            PreparedMpmUnilateralCoupling::required_wall_friction_bytes(maximum, 32, 128, 64)
                .unwrap(),
            73_247_760
        );
        assert_eq!(
            PreparedMpmUnilateralCoupling::required_bytes(maximum, 32, 128, 64).unwrap(),
            56_727_920
        );
        assert_eq!(
            PreparedMpmUnilateralCoupling::required_friction_bytes(maximum, 32, 128, 64).unwrap(),
            52_009_584
        );
        for node_capacity in [0, 16385, u32::MAX] {
            assert!(
                MpmUnilateralConfig {
                    node_capacity,
                    ..config
                }
                .validate()
                .is_err()
            );
        }
        for max_iterations in [0, 65, u32::MAX] {
            assert!(
                MpmUnilateralConfig {
                    max_iterations,
                    ..config
                }
                .validate()
                .is_err()
            );
        }
        for search_distance in [-0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                MpmUnilateralConfig {
                    search_distance,
                    ..config
                }
                .validate()
                .is_err()
            );
        }
        for (dofs, size) in [(0, 48), (33, 48), (10, 4), (10, 129)] {
            assert!(PreparedMpmUnilateralCoupling::required_bytes(config, dofs, size, 5).is_err());
        }
        for bodies in [0, 65, u32::MAX] {
            assert!(PreparedMpmUnilateralCoupling::required_bytes(config, 10, 48, bodies).is_err());
        }
    }
}
