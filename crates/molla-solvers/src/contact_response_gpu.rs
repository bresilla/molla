//! Retained articulated contact response, disk/impact solves and generalized impulses.

use crate::constraint_rows_gpu::{ScalarConstraintBuffers, ScalarPositionBuffers};
use crate::contact_islands_gpu::{ContactIslandConfig, NO_CONTACT_ISLAND, ResidentContactIslands};
use crate::coulomb_disk_gpu::{CoulombDiskConfig, ResidentCoulombDisks};
use crate::impact_constraints_gpu::{ImpactGpuConfig, ResidentImpactConstraints};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{
    JacobianDispatch, JacobianKernel, JacobianRequest, MassMatrixDispatch, MassMatrixKernel,
};
use molla_sim::{BodyFlags, Model, State};

pub const CONTACT_RESPONSE_FAULT_INPUT: u32 = 1 << 22;
pub const CONTACT_RESPONSE_FAULT_MASS: u32 = 1 << 23;
pub const CONTACT_RESPONSE_FAULT_NONFINITE: u32 = 1 << 24;

pub struct ContactResponseState<'a> {
    pub input: &'a State,
    pub predicted_rates: &'a wgpu::Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct ContactResponseConfig {
    pub row_capacity: u32,
    pub disks: CoulombDiskConfig,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Body {
    com: [f32; 4],
    response: [u32; 4],
}

/// Geometry and mass use the input pose; free slip includes the predicted rate increment.
/// Corrected rates require status-guarded publication, such as
/// [`ResidentContactPublication`](crate::contact_publication_gpu::ResidentContactPublication).
/// Coordinate-null mass rows have zero response; other pivots <= 1e-6 of the mass scale fault.
pub struct ResidentContactResponse {
    core: ResponseCore,
}

struct ResponseCore {
    islands: ResidentContactIslands,
    solver: ResponseSolver,
    mass_dispatch: Vec<Option<MassMatrixDispatch>>,
    jacobian_dispatch: Vec<Option<JacobianDispatch>>,
    pipelines: [wgpu::ComputePipeline; 6],
    groups: Vec<wgpu::BindGroup>,
    scratch: wgpu::Buffer,
    gram: wgpu::Buffer,
    free: wgpu::Buffer,
    disk_input: wgpu::Buffer,
    directions: wgpu::Buffer,
    delta: wgpu::Buffer,
    corrected: wgpu::Buffer,
    world_impulses: wgpu::Buffer,
    block_kinds: wgpu::Buffer,
    scalar_solutions: wgpu::Buffer,
    predicted_contacts: wgpu::Buffer,
    filter_predicted: bool,
    work: [u32; 6],
}

impl ResidentContactResponse {
    /// Rebuild after model metadata, dependency edges or bound buffers change.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engine: &Engine,
        model: &Model,
        states: &[ContactResponseState<'_>],
        normal_rows: &wgpu::Buffer,
        dependencies: &[[u32; 2]],
        step_size: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ContactResponseConfig,
    ) -> Result<Self> {
        Self::with_constraints(
            engine,
            model,
            states,
            normal_rows,
            dependencies,
            None,
            step_size,
            status,
            config,
        )
    }

    /// Scalar and contact rows share island assembly, inverse mass and impulse application.
    #[allow(clippy::too_many_arguments)]
    pub fn with_constraints(
        engine: &Engine,
        model: &Model,
        states: &[ContactResponseState<'_>],
        normal_rows: &wgpu::Buffer,
        dependencies: &[[u32; 2]],
        scalar: Option<ScalarConstraintBuffers<'_>>,
        step_size: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ContactResponseConfig,
    ) -> Result<Self> {
        Self::with_runtime_constraints(
            engine,
            model,
            states,
            normal_rows,
            dependencies,
            scalar,
            None,
            config.disks.contacts_per_island,
            step_size,
            status,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_runtime_constraints(
        engine: &Engine,
        model: &Model,
        states: &[ContactResponseState<'_>],
        normal_rows: &wgpu::Buffer,
        dependencies: &[[u32; 2]],
        scalar: Option<ScalarConstraintBuffers<'_>>,
        predictor_delta: Option<&wgpu::Buffer>,
        contact_limit: u32,
        step_size: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ContactResponseConfig,
    ) -> Result<Self> {
        Ok(Self {
            core: ResponseCore::build(
                engine,
                model,
                states,
                normal_rows,
                dependencies,
                scalar,
                predictor_delta,
                contact_limit,
                step_size,
                status,
                config,
                false,
                None,
                false,
                None,
            )?,
        })
    }

    /// Solve only groups not selected by an earlier impact response.
    /// Predicted-rate buffers must contain that response's corrected rates; excluded
    /// groups retain them unchanged. The retained predictor delta still describes
    /// the original force predictor for the remaining friction/scalar groups.
    #[allow(clippy::too_many_arguments)]
    pub fn with_excluded_impact_groups(
        engine: &Engine,
        model: &Model,
        states: &[ContactResponseState<'_>],
        normal_rows: &wgpu::Buffer,
        dependencies: &[[u32; 2]],
        scalar: Option<ScalarConstraintBuffers<'_>>,
        predictor_delta: Option<&wgpu::Buffer>,
        impact_articulations: &wgpu::Buffer,
        contact_limit: u32,
        step_size: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ContactResponseConfig,
    ) -> Result<Self> {
        Ok(Self {
            core: ResponseCore::build(
                engine,
                model,
                states,
                normal_rows,
                dependencies,
                scalar,
                predictor_delta,
                contact_limit,
                step_size,
                status,
                config,
                false,
                None,
                false,
                Some(impact_articulations),
            )?,
        })
    }

    /// Record retained island, response, disk solve and application stages without host transfers.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, state: usize) -> Result<()> {
        self.core.encode(encoder, state)
    }
    pub fn islands(&self) -> &ResidentContactIslands {
        &self.core.islands
    }
    pub fn disks(&self) -> &ResidentCoulombDisks {
        let ResponseSolver::Disks(solver) = &self.core.solver else {
            unreachable!()
        };
        solver
    }
    pub fn gram_buffer(&self) -> &wgpu::Buffer {
        &self.core.gram
    }
    pub fn free_velocity_buffer(&self) -> &wgpu::Buffer {
        &self.core.free
    }
    pub fn disk_input_buffer(&self) -> &wgpu::Buffer {
        &self.core.disk_input
    }
    pub fn velocity_delta_buffer(&self) -> &wgpu::Buffer {
        &self.core.delta
    }
    pub fn corrected_rates_buffer(&self) -> &wgpu::Buffer {
        &self.core.corrected
    }
    /// World tangent impulse on B in original normal-row order.
    pub fn world_impulse_buffer(&self) -> &wgpu::Buffer {
        &self.core.world_impulses
    }
    pub fn scalar_solution_buffer(&self) -> &wgpu::Buffer {
        &self.core.scalar_solutions
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImpactResponseConfig {
    pub row_capacity: u32,
    pub impacts: ImpactGpuConfig,
}

/// Articulated normal/tangent impact assembly from device contact rows.
/// Active NormalContactRows need no penalty force or nonzero friction. Their relative
/// velocities use the input pose/rates; predicted increments and prescribed acceleration
/// are included once. Episode history and position correction are separate stages.
/// Standalone constructors require a nonseparating predicted active set; the onset
/// constructor filters predicted rows and retains only groups with an eligible contact.
/// Publication must honor the caller-owned status before consuming any result.
pub struct ResidentImpactResponse {
    core: ResponseCore,
}

impl ResidentImpactResponse {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engine: &Engine,
        model: &Model,
        states: &[ContactResponseState<'_>],
        impact_rows: &wgpu::Buffer,
        dependencies: &[[u32; 2]],
        scalar: Option<ScalarConstraintBuffers<'_>>,
        step_size: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ImpactResponseConfig,
    ) -> Result<Self> {
        Self::with_predictor_delta(
            engine,
            model,
            states,
            impact_rows,
            dependencies,
            scalar,
            None,
            step_size,
            status,
            config,
        )
    }

    /// Consume the retained generalized predictor increment instead of subtracting rounded rates.
    #[allow(clippy::too_many_arguments)]
    pub fn with_predictor_delta(
        engine: &Engine,
        model: &Model,
        states: &[ContactResponseState<'_>],
        impact_rows: &wgpu::Buffer,
        dependencies: &[[u32; 2]],
        scalar: Option<ScalarConstraintBuffers<'_>>,
        predictor_delta: Option<&wgpu::Buffer>,
        step_size: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ImpactResponseConfig,
    ) -> Result<Self> {
        Self::build(
            engine,
            model,
            states,
            impact_rows,
            dependencies,
            scalar,
            predictor_delta,
            None,
            step_size,
            status,
            config,
        )
    }

    /// Select whole dynamic groups from device episode onsets, including connected old contacts
    /// and scalar rows. Unselected generalized rates retain their predictor values.
    /// Separating predicted contacts remain graph edges but do not enter the impulse solve.
    #[allow(clippy::too_many_arguments)]
    pub fn with_episode_onsets(
        engine: &Engine,
        model: &Model,
        states: &[ContactResponseState<'_>],
        impact_rows: &wgpu::Buffer,
        dependencies: &[[u32; 2]],
        scalar: Option<ScalarConstraintBuffers<'_>>,
        predictor_delta: Option<&wgpu::Buffer>,
        onsets: &wgpu::Buffer,
        step_size: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ImpactResponseConfig,
    ) -> Result<Self> {
        Self::build(
            engine,
            model,
            states,
            impact_rows,
            dependencies,
            scalar,
            predictor_delta,
            Some(onsets),
            step_size,
            status,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        engine: &Engine,
        model: &Model,
        states: &[ContactResponseState<'_>],
        impact_rows: &wgpu::Buffer,
        dependencies: &[[u32; 2]],
        scalar: Option<ScalarConstraintBuffers<'_>>,
        predictor_delta: Option<&wgpu::Buffer>,
        onsets: Option<&wgpu::Buffer>,
        step_size: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ImpactResponseConfig,
    ) -> Result<Self> {
        let disks = CoulombDiskConfig {
            island_capacity: config.impacts.island_capacity,
            contacts_per_island: config.impacts.blocks_per_island,
            max_iterations: config.impacts.max_iterations,
            velocity_tolerance: config.impacts.velocity_tolerance,
        };
        Ok(Self {
            core: ResponseCore::build(
                engine,
                model,
                states,
                impact_rows,
                dependencies,
                scalar,
                predictor_delta,
                disks.contacts_per_island,
                step_size,
                status,
                ContactResponseConfig {
                    row_capacity: config.row_capacity,
                    disks,
                },
                true,
                onsets,
                false,
                None,
            )?,
        })
    }
    /// Record retained island, three-row response, impact solve and application stages.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, state: usize) -> Result<()> {
        self.core.encode(encoder, state)
    }
    pub fn islands(&self) -> &ResidentContactIslands {
        &self.core.islands
    }
    pub fn impacts(&self) -> &ResidentImpactConstraints {
        let ResponseSolver::Impacts(solver) = &self.core.solver else {
            unreachable!()
        };
        solver
    }
    pub fn gram_buffer(&self) -> &wgpu::Buffer {
        &self.core.gram
    }
    pub fn free_velocity_buffer(&self) -> &wgpu::Buffer {
        &self.core.free
    }
    pub fn block_input_buffer(&self) -> &wgpu::Buffer {
        &self.core.disk_input
    }
    pub fn velocity_delta_buffer(&self) -> &wgpu::Buffer {
        &self.core.delta
    }
    pub fn corrected_rates_buffer(&self) -> &wgpu::Buffer {
        &self.core.corrected
    }
    /// Full normal-plus-tangent world impulse on B in original contact-row order.
    pub fn world_impulse_buffer(&self) -> &wgpu::Buffer {
        &self.core.world_impulses
    }
    /// Scalar CoulombDiskSolution records in original scalar-row order.
    pub fn scalar_solution_buffer(&self) -> &wgpu::Buffer {
        &self.core.scalar_solutions
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImpactPositionConfig {
    pub row_capacity: u32,
    pub island_capacity: u32,
    pub blocks_per_island: u32,
    pub max_iterations: u32,
    pub position_tolerance: f32,
}

/// Mass-weighted linearized displacement at the input pose for new impact depths
/// and scalar position rows. Velocities, restitution and friction do not enter this solve.
/// Outputs are displacement multipliers, not physical impulses or reaction forces.
/// Applying the displacement and redrifting the corrected state are separate stages.
pub struct ResidentImpactPositionResponse {
    core: ResponseCore,
}

impl ResidentImpactPositionResponse {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engine: &Engine,
        model: &Model,
        states: &[&State],
        contact_rows: &wgpu::Buffer,
        dependencies: &[[u32; 2]],
        scalar: Option<ScalarPositionBuffers<'_>>,
        onsets: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ImpactPositionConfig,
    ) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("missing position response device".into()))?;
        let unit = storage(device, "position response unit step", 4);
        molla_core::gpu_write_buffer!(engine.queue().unwrap(), &unit, 0, bytemuck::bytes_of(&1f32));
        let states: Vec<_> = states
            .iter()
            .map(|input| ContactResponseState {
                input,
                predicted_rates: input.joint_qd.device_buffer().unwrap_or(&unit),
            })
            .collect();
        Ok(Self {
            core: ResponseCore::build(
                engine,
                model,
                &states,
                contact_rows,
                dependencies,
                scalar.map(|s| ScalarConstraintBuffers {
                    rows: s.rows,
                    jacobians: s.jacobians,
                    capacity: s.capacity,
                }),
                None,
                config.blocks_per_island,
                &unit,
                status,
                ContactResponseConfig {
                    row_capacity: config.row_capacity,
                    disks: CoulombDiskConfig {
                        island_capacity: config.island_capacity,
                        contacts_per_island: config.blocks_per_island,
                        max_iterations: config.max_iterations,
                        velocity_tolerance: config.position_tolerance,
                    },
                },
                true,
                Some(onsets),
                true,
                None,
            )?,
        })
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, state: usize) -> Result<()> {
        self.core.encode(encoder, state)
    }
    pub fn displacement_buffer(&self) -> &wgpu::Buffer {
        &self.core.delta
    }
    pub fn islands(&self) -> &ResidentContactIslands {
        &self.core.islands
    }
    pub fn gram_buffer(&self) -> &wgpu::Buffer {
        &self.core.gram
    }
    pub fn position_bias_buffer(&self) -> &wgpu::Buffer {
        &self.core.free
    }
    /// ImpactGpuSolution layout: displacement multipliers and remaining linearized gaps.
    pub fn solution_buffer(&self) -> &wgpu::Buffer {
        self.core.solver.solution_buffer()
    }
    pub fn diagnostic_buffer(&self) -> &wgpu::Buffer {
        let ResponseSolver::Impacts(solver) = &self.core.solver else {
            unreachable!()
        };
        solver.diagnostic_buffer()
    }
}

enum ResponseSolver {
    Disks(ResidentCoulombDisks),
    Impacts(ResidentImpactConstraints),
}

impl ResponseSolver {
    fn solution_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::Disks(s) => s.solution_buffer(),
            Self::Impacts(s) => s.solution_buffer(),
        }
    }
    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        match self {
            Self::Disks(s) => s.encode(encoder),
            Self::Impacts(s) => s.encode(encoder),
        }
    }
}

impl ResponseCore {
    #[allow(clippy::too_many_arguments)]
    fn build(
        engine: &Engine,
        model: &Model,
        states: &[ContactResponseState<'_>],
        normal_rows: &wgpu::Buffer,
        dependencies: &[[u32; 2]],
        scalar: Option<ScalarConstraintBuffers<'_>>,
        predictor_delta: Option<&wgpu::Buffer>,
        contact_limit: u32,
        step_size: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ContactResponseConfig,
        impact: bool,
        onsets: Option<&wgpu::Buffer>,
        position: bool,
        excluded_impacts: Option<&wgpu::Buffer>,
    ) -> Result<Self> {
        let invalid =
            || Error::Gpu("invalid articulated contact response metadata or buffers".into());
        let device = engine.device().ok_or_else(invalid)?;
        let queue = engine.queue().ok_or_else(invalid)?;
        let limits = device.limits();
        let starts = if model.articulation_count == 0 {
            vec![0, model.joint_count as u32]
        } else {
            model.articulation_start.host()?.to_vec()
        };
        let offsets = model.joint_dof_offset.host()?;
        let children = model.joint_child.host()?;
        let parents = model.joint_parent.host()?;
        let lower = model.joint_limit_lower.host()?;
        let upper = model.joint_limit_upper.host()?;
        let flags = model.body_flags.host()?;
        let centers = model.body_com.host()?;
        let masses = model.body_mass.host()?;
        let inertias = model.body_inertia.host()?;
        if (position && (!impact || onsets.is_none()))
            || (excluded_impacts.is_some() && (impact || onsets.is_some()))
            || states.is_empty()
            || config.row_capacity == 0
            || starts.len() < 2
            || starts.first() != Some(&0)
            || starts.last().copied() != Some(model.joint_count as u32)
            || starts.windows(2).any(|r| r[0] > r[1])
            || offsets.len() != model.joint_count + 1
            || offsets.first() != Some(&0)
            || offsets.last().copied() != Some(model.joint_dof_total as u32)
            || offsets.windows(2).any(|r| r[0] > r[1])
            || children.len() != model.joint_count
            || parents.len() != model.joint_count
            || lower.len() != model.joint_dof_total
            || upper.len() != model.joint_dof_total
            || flags.len() != model.body_count
            || centers.len() != model.body_count
            || masses.len() != model.body_count
            || inertias.len() != model.body_count
            || model.body_count > u32::MAX as usize
            || model.joint_dof_total > u32::MAX as usize
            || limits.max_storage_buffers_per_shader_stage < 32
        {
            return Err(invalid());
        }
        let representable = |v: f64| v == 0.0 || (v as f32).is_normal();
        if masses.iter().any(|&m| m < 0.0 || !representable(m))
            || inertias
                .iter()
                .flat_map(|i| i.to_cols_array())
                .any(|v| !representable(v))
            || centers
                .iter()
                .flat_map(|c| c.to_array())
                .any(|v| !representable(v))
        {
            return Err(invalid());
        }
        let art_count = starts.len() - 1;
        let row_width = if impact { 3 } else { 2 };
        let width = u64::from(config.disks.contacts_per_island)
            .checked_mul(row_width)
            .ok_or_else(invalid)?;
        let slots =
            u64::from(config.disks.island_capacity) * u64::from(config.disks.contacts_per_island);
        if width == 0 || width > u32::MAX as u64 || slots == 0 || slots > u32::MAX as u64 {
            return Err(invalid());
        }
        let mass_kernel = MassMatrixKernel::new(engine, model)?;
        let jacobian_kernel = JacobianKernel::new(engine, model)?;
        let mut owners = vec![NO_CONTACT_ISLAND; model.body_count];
        let boundaries = crate::kinematic_motion::driven_ancestors(model)?;
        let mut bodies = vec![Body::default(); model.body_count.max(1)];
        for (i, body) in bodies.iter_mut().enumerate().take(model.body_count) {
            body.com = [
                centers[i].x as f32,
                centers[i].y as f32,
                centers[i].z as f32,
                0.0,
            ];
            if !centers[i].is_finite() || body.com.iter().any(|v| !v.is_finite()) {
                return Err(invalid());
            }
            body.response[0] = NO_CONTACT_ISLAND;
            body.response[3] = boundaries[i];
        }
        let mut arts = Vec::new();
        let mut requests = Vec::new();
        let mut jacobian_words = 0_u64;
        let mut projected_words = 0_u64;
        let mut seen = vec![false; model.body_count];
        let mut mobile = vec![false; model.body_count];
        for (art, range) in starts.windows(2).enumerate() {
            let first = offsets[range[0] as usize];
            let n = offsets[range[1] as usize] - first;
            if n as usize != jacobian_kernel.dof_width(art) {
                return Err(invalid());
            }
            arts.push([
                first,
                n,
                mass_kernel.mass_offset(art),
                u32::try_from(projected_words).map_err(|_| invalid())?,
            ]);
            projected_words = projected_words
                .checked_add(width * u64::from(n))
                .ok_or_else(invalid)?;
            for joint in range[0] as usize..range[1] as usize {
                let child = children[joint];
                if child < 0 || child as usize >= model.body_count || seen[child as usize] {
                    return Err(invalid());
                }
                let b = child as usize;
                seen[b] = true;
                let parent = parents[joint];
                if parent < -1
                    || (parent >= 0
                        && (parent as usize >= model.body_count
                            || (!seen[parent as usize]
                                && flags[parent as usize] & BodyFlags::IMMOVABLE == 0)
                            || parent == child))
                {
                    return Err(invalid());
                }
                let inherited = parent >= 0 && mobile[parent as usize];
                let movable = (offsets[joint] as usize..offsets[joint + 1] as usize)
                    .any(|d| !(lower[d].is_finite() && lower[d] == upper[d]));
                mobile[b] = flags[b] & BodyFlags::IMMOVABLE == 0 && (inherited || movable);
                if !mobile[b] {
                    continue;
                }
                owners[b] = art as u32;
                bodies[b].response = [
                    art as u32,
                    u32::try_from(jacobian_words).map_err(|_| invalid())?,
                    n,
                    boundaries[b],
                ];
                requests.push(JacobianRequest {
                    articulation: art as u32,
                    target_body: b as u32,
                    output_offset: bodies[b].response[1],
                    body_q_offset: 0,
                    point_local: bodies[b].com,
                });
                jacobian_words = jacobian_words
                    .checked_add(6 * u64::from(n))
                    .ok_or_else(invalid)?;
            }
        }
        if seen
            .iter()
            .zip(flags)
            .any(|(&found, &f)| !found && f & BodyFlags::IMMOVABLE == 0)
        {
            return Err(invalid());
        }
        let factor_words = mass_kernel.output_len() as u64;
        let scratch_words = factor_words
            .checked_add(projected_words.checked_mul(2).ok_or_else(invalid)?)
            .and_then(|v| v.checked_add(art_count as u64))
            .ok_or_else(invalid)?;
        let gram_words = slots
            .checked_mul(width)
            .and_then(|n| n.checked_mul(row_width))
            .ok_or_else(invalid)?;
        if scratch_words > u32::MAX as u64
            || gram_words > u32::MAX as u64
            || jacobian_words > u32::MAX as u64
        {
            return Err(invalid());
        }
        let sizes = [
            scratch_words * 4,
            gram_words * 4,
            slots * 16,
            slots * row_width * 16,
            u64::from(config.row_capacity) * 16,
            jacobian_words * 4,
            factor_words * 4,
            model.body_count.max(1) as u64 * 32,
            arts.len() as u64 * 16,
            model.joint_dof_total.max(1) as u64 * 4,
        ];
        if sizes
            .iter()
            .any(|&s| s > limits.max_storage_buffer_binding_size)
            || slots.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
            || (art_count as u64).div_ceil(64)
                > u64::from(limits.max_compute_workgroups_per_dimension)
        {
            return Err(invalid());
        }
        let check = |b: &wgpu::Buffer, bytes: u64| -> Result<()> {
            if b.size() < bytes
                || b.size() > limits.max_storage_buffer_binding_size
                || !b.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                Err(invalid())
            } else {
                Ok(())
            }
        };
        check(step_size, 4)?;
        check(status, 4)?;
        if let Some(delta) = predictor_delta {
            check(delta, model.joint_dof_total.max(1) as u64 * 4)?;
        }
        let scalar_count = scalar.map_or(0, |s| s.capacity);
        if let Some(scalar) = scalar {
            if scalar.capacity == 0 {
                return Err(invalid());
            }
            check(scalar.rows, u64::from(scalar_count) * 32)?;
            let jacobian_bytes = u64::from(scalar_count)
                .checked_mul(model.joint_dof_total as u64)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(invalid)?;
            check(scalar.jacobians, jacobian_bytes.max(4))?;
        }
        let alloc = |label, size: u64| storage(device, label, size.max(4));
        let empty = alloc("contact response empty", 32);
        let body_data = alloc("contact response bodies", bodies.len() as u64 * 32);
        molla_core::gpu_write_buffer!(queue, &body_data, 0, bytemuck::cast_slice(&bodies));
        let art_data = alloc("contact response articulations", arts.len() as u64 * 16);
        molla_core::gpu_write_buffer!(queue, &art_data, 0, bytemuck::cast_slice(&arts));
        let mass = alloc("contact response mass", factor_words * 4);
        let jacobians = alloc("contact response jacobians", jacobian_words * 4);
        let scratch = alloc("contact response scratch", scratch_words * 4);
        let gram = alloc("contact response gram", gram_words * 4);
        let free = alloc("contact response free", slots * row_width * 4);
        let disk_input = alloc(
            "contact response disks",
            slots * if impact { 32 } else { 16 },
        );
        let directions = alloc("contact row directions", slots * row_width * 16);
        let block_kinds = alloc("joint contact block kinds", slots * 4);
        let scalar_solutions = alloc(
            "joint scalar solutions",
            u64::from(scalar_count.max(1)) * 32,
        );
        let delta = alloc(
            "contact generalized impulses",
            model.joint_dof_total as u64 * 4,
        );
        let corrected = alloc("contact corrected rates", model.joint_dof_total as u64 * 4);
        let world_impulses = alloc(
            "contact world impulses",
            u64::from(config.row_capacity) * 16,
        );
        let predicted_contacts = alloc(
            "predicted impact contacts",
            if onsets.is_some() {
                u64::from(config.row_capacity) * 8
            } else {
                8
            },
        );
        let island_config = ContactIslandConfig {
            articulation_count: art_count as u32,
            row_capacity: config.row_capacity,
            island_capacity: config.disks.island_capacity,
            contacts_per_island: config.disks.contacts_per_island,
        };
        let islands = if let Some(onsets) = onsets {
            ResidentContactIslands::with_predicted_impacts(
                engine,
                &owners,
                dependencies,
                normal_rows,
                scalar.map(|s| (s.rows, s.capacity)),
                onsets,
                &predicted_contacts,
                !position,
                status,
                island_config,
                contact_limit,
            )?
        } else if let Some(excluded) = excluded_impacts {
            ResidentContactIslands::with_excluded_impacts(
                engine,
                &owners,
                dependencies,
                normal_rows,
                scalar.map(|s| (s.rows, s.capacity)),
                excluded,
                status,
                island_config,
                contact_limit,
            )?
        } else {
            ResidentContactIslands::with_contact_mode(
                engine,
                &owners,
                dependencies,
                normal_rows,
                scalar.map(|s| (s.rows, s.capacity)),
                status,
                island_config,
                contact_limit,
                impact,
            )?
        };
        let solver = if impact {
            ResponseSolver::Impacts(ResidentImpactConstraints::new(
                device,
                &gram,
                &free,
                &disk_input,
                islands.contact_counts_buffer(),
                islands.island_count_buffer(),
                status,
                ImpactGpuConfig {
                    island_capacity: config.disks.island_capacity,
                    blocks_per_island: config.disks.contacts_per_island,
                    max_iterations: config.disks.max_iterations,
                    velocity_tolerance: config.disks.velocity_tolerance,
                },
            )?)
        } else {
            ResponseSolver::Disks(ResidentCoulombDisks::with_block_kinds(
                device,
                &gram,
                &free,
                &disk_input,
                islands.contact_counts_buffer(),
                islands.island_count_buffer(),
                status,
                Some(&block_kinds),
                config.disks,
            )?)
        };
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("contact response parameters"),
                size: 48,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            }
        );
        molla_core::gpu_mapped_upload!(params.slice(..)).copy_from_slice(bytemuck::cast_slice(&[
            art_count as u32,
            model.body_count as u32,
            config.disks.contacts_per_island,
            config.disks.island_capacity,
            config.row_capacity,
            model.joint_dof_total as u32,
            factor_words as u32,
            (factor_words + projected_words) as u32,
            (factor_words + 2 * projected_words) as u32,
            scalar_count,
            u32::from(predictor_delta.is_some()) | (u32::from(excluded_impacts.is_some()) << 1),
            if position {
                3
            } else if onsets.is_some() {
                2
            } else {
                u32::from(impact)
            },
        ]));
        params.unmap();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("contact response layout"),
            entries: &(0..33)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 23 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 13
                                    || binding == 19
                                    || (24..28).contains(&binding)
                                    || binding == 30
                                    || binding == 31,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                })
                .collect::<Vec<_>>(),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("articulated contact response"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/contact_response.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("contact response pipeline"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = [
            "geometry",
            "project",
            "assemble",
            "apply",
            "readout",
            "predict_contacts",
        ]
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let mut groups = Vec::new();
        let mut mass_dispatch = Vec::new();
        let mut jacobian_dispatch = Vec::new();
        for state in states {
            let poses = state.input.body_q.device_buffer().unwrap_or(&empty);
            let rates = state.input.joint_qd.device_buffer().unwrap_or(&empty);
            check(poses, model.body_count as u64 * 32)?;
            check(rates, model.joint_dof_total as u64 * 4)?;
            check(
                state.predicted_rates,
                model.joint_dof_total.max(1) as u64 * 4,
            )?;
            let buffers = [
                normal_rows,
                &body_data,
                &art_data,
                poses,
                &mass,
                &jacobians,
                rates,
                state.predicted_rates,
                islands.island_count_buffer(),
                islands.contact_counts_buffer(),
                islands.contact_indices_buffer(),
                islands.articulation_islands_buffer(),
                step_size,
                status,
                &scratch,
                &gram,
                &free,
                &disk_input,
                &directions,
                solver.solution_buffer(),
                &delta,
                &corrected,
                &world_impulses,
                &params,
                islands.articulation_offsets_buffer(),
                islands.articulation_indices_buffer(),
                scalar.map_or(&empty, |s| s.rows),
                scalar.map_or(&empty, |s| s.jacobians),
                &block_kinds,
                &scalar_solutions,
                if position {
                    onsets.unwrap()
                } else {
                    predictor_delta.unwrap_or(&empty)
                },
                model
                    .body_kinematic_acceleration
                    .device_buffer()
                    .unwrap_or(&empty),
                &predicted_contacts,
            ];
            groups.push(
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("contact response bindings"),
                    layout: &layout,
                    entries: &buffers
                        .iter()
                        .enumerate()
                        .map(|(binding, b)| wgpu::BindGroupEntry {
                            binding: binding as u32,
                            resource: b.as_entire_binding(),
                        })
                        .collect::<Vec<_>>(),
                }),
            );
            mass_dispatch.push(if model.joint_dof_total == 0 {
                None
            } else {
                Some(mass_kernel.prepare_on_device(device, queue, model, state.input, &mass)?)
            });
            jacobian_dispatch.push(jacobian_kernel.prepare_with_body_q_on_device(
                device, queue, model, poses, &requests, &jacobians,
            )?);
        }
        Ok(Self {
            islands,
            solver,
            mass_dispatch,
            jacobian_dispatch,
            pipelines,
            groups,
            scratch,
            gram,
            free,
            disk_input,
            directions,
            delta,
            corrected,
            world_impulses,
            block_kinds,
            scalar_solutions,
            predicted_contacts,
            filter_predicted: onsets.is_some(),
            work: [
                slots as u32,
                art_count as u32,
                config.disks.island_capacity,
                art_count as u32,
                slots as u32,
                config.row_capacity,
            ],
        })
    }

    /// Record the full island/response/disk/application chain without host transfers.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, state: usize) -> Result<()> {
        let group = self
            .groups
            .get(state)
            .ok_or_else(|| Error::Gpu("invalid contact response state binding".into()))?;
        for b in [
            &self.scratch,
            &self.gram,
            &self.free,
            &self.disk_input,
            &self.directions,
            &self.delta,
            &self.corrected,
            &self.world_impulses,
            &self.block_kinds,
            &self.scalar_solutions,
            &self.predicted_contacts,
        ] {
            encoder.clear_buffer(b, 0, None);
        }
        if let Some(j) = &self.jacobian_dispatch[state] {
            j.encode(encoder);
        }
        if self.filter_predicted {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines[5]);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(self.work[5].div_ceil(64), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        self.islands.encode(encoder);
        if let Some(m) = &self.mass_dispatch[state] {
            m.encode(encoder);
        }
        for stage in 0..5 {
            if stage == 3 {
                self.solver.encode(encoder);
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines[stage]);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(self.work[stage].div_ceil(64).max(1), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        Ok(())
    }
}

fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        }
    )
}

#[cfg(test)]
#[path = "contact_response_gpu_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "coupled_response_gpu_tests.rs"]
mod coupled_tests;

#[cfg(test)]
#[path = "impact_response_gpu_tests.rs"]
mod impact_tests;

#[cfg(test)]
#[path = "impact_position_gpu_tests.rs"]
mod position_tests;

#[cfg(test)]
#[path = "impact_partition_gpu_tests.rs"]
mod partition_tests;
