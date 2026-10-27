//! Substepped FEM and articulated-rigid CPU coupling.

use molla_core::{Error, Result};
use molla_math::Real;
use molla_sim::{Contacts, Model, State, runtime::ColliderHandle};

use crate::soft_rigid_contact::{SoftRigidContactConfig, gather_primitive_contacts, solve_cpu};
use crate::{Solver, VbdSolver, rigid_world::RigidWorld};

#[derive(Clone, Copy, Debug)]
pub struct FemRigidConfig {
    pub max_substep: Real,
    pub elastic_iterations: usize,
    pub contact_margin: Real,
    pub contact: SoftRigidContactConfig,
}

impl Default for FemRigidConfig {
    fn default() -> Self {
        Self {
            max_substep: 1.0 / 2400.0,
            elastic_iterations: 128,
            contact_margin: 0.001,
            contact: SoftRigidContactConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FemRigidStepReport {
    pub substeps: usize,
    pub active_contacts: usize,
    pub impulse_magnitude_sum: Real,
    pub max_penetration: Real,
}

/// All coupled FEM meshes share one model and advance the rigid world once per substep.
pub struct FemRigidSystem {
    model: Model,
    state: State,
    next: State,
    surface: Vec<u32>,
    solver: VbdSolver,
    config: FemRigidConfig,
    failed: bool,
}

impl FemRigidSystem {
    pub fn new_cpu(model: Model, config: FemRigidConfig) -> Result<Self> {
        if model.tet_count == 0
            || model.body_count != 0
            || model.shape_count != 0
            || !config.max_substep.is_finite()
            || config.max_substep <= 0.0
            || config.elastic_iterations == 0
            || !config.contact_margin.is_finite()
            || config.contact_margin < 0.0
            || config.contact.iterations == 0
            || !config.contact.penetration_recovery.is_finite()
            || !(0.0..=1.0).contains(&config.contact.penetration_recovery)
            || !config.contact.max_recovery_speed.is_finite()
            || config.contact.max_recovery_speed < 0.0
        {
            return Err(Error::Build(
                "invalid CPU FEM-rigid model or configuration".into(),
            ));
        }
        let tets: Vec<[u32; 4]> = model
            .tet_indices
            .host()?
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        let mut surface: Vec<_> = molla_sim::compute_tet_surface_triangles(&tets)
            .into_iter()
            .flatten()
            .collect();
        surface.sort_unstable();
        surface.dedup();
        let state = model.state()?;
        let next = model.state()?;
        Ok(Self {
            model,
            state,
            next,
            surface,
            solver: VbdSolver::new_cpu()
                .with_iterations(config.elastic_iterations)
                .with_contact_stiffness(0.0),
            config,
            failed: false,
        })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn state(&self) -> &State {
        &self.state
    }
    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    /// Advance elasticity, rigid dynamics and two-way primitive contact in bounded substeps.
    /// A failed substep invalidates the system; rebuild both coupled states before resuming.
    pub fn step_cpu(
        &mut self,
        world: &mut RigidWorld,
        colliders: &[ColliderHandle],
        dt: Real,
    ) -> Result<FemRigidStepReport> {
        if self.failed {
            return Err(Error::Solver(
                "FEM-rigid system requires rebuild after a failed substep".into(),
            ));
        }
        let count = (dt / self.config.max_substep).ceil();
        if !dt.is_finite() || dt <= 0.0 || !(1.0..=4096.0).contains(&count) {
            return Err(Error::Solver("invalid FEM-rigid timestep".into()));
        }
        gather_primitive_contacts(
            &self.model,
            &self.state,
            &self.surface,
            &world.scene,
            colliders,
            self.config.contact_margin,
        )?;
        let substeps = count as usize;
        let h = dt / substeps as Real;
        let mut report = FemRigidStepReport {
            substeps,
            ..Default::default()
        };
        self.failed = true;
        for _ in 0..substeps {
            let rigid_start = world.scene.state().clone_host()?;
            let contacts = gather_primitive_contacts(
                &self.model,
                &self.state,
                &self.surface,
                &world.scene,
                colliders,
                self.config.contact_margin,
            )?;
            self.solver.step(
                &self.model,
                &self.state,
                &self.model.control(),
                &Contacts::with_capacity(0),
                h,
                &mut self.next,
            )?;
            world.step(h)?;
            let mut rigid = world.scene.state().clone_host()?;
            rigid
                .body_q
                .host_mut()?
                .copy_from_slice(rigid_start.body_q.host()?);
            rigid
                .joint_q
                .host_mut()?
                .copy_from_slice(rigid_start.joint_q.host()?);
            molla_sim::eval_body_velocities(world.scene.model(), &mut rigid)?;
            self.next
                .particle_q
                .host_mut()?
                .copy_from_slice(self.state.particle_q.host()?);
            let solved = solve_cpu(
                &self.model,
                &mut self.next,
                world.scene.model(),
                &mut rigid,
                &contacts,
                h,
                self.config.contact,
            )?;
            let velocities = self.next.particle_qd.host()?.to_vec();
            for (q, v) in self.next.particle_q.host_mut()?.iter_mut().zip(velocities) {
                *q += v * h;
            }
            world
                .solver
                .drift_cpu(world.scene.model(), &rigid_start, h, &mut rigid)?;
            *world.scene.state_mut() = rigid;
            std::mem::swap(&mut self.state, &mut self.next);
            report.active_contacts += solved
                .impulses
                .iter()
                .filter(|j| j.length_squared() > 0.0)
                .count();
            report.impulse_magnitude_sum +=
                solved.impulses.iter().map(|j| j.length()).sum::<Real>();
            report.max_penetration = contacts
                .iter()
                .map(|c| -c.separation)
                .fold(report.max_penetration, Real::max);
        }
        self.failed = false;
        Ok(report)
    }
}
