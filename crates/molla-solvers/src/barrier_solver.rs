//! C0 integration — the barrier-Newton contact backend as a
//! first-class engine [`Solver`], peer to `VbdSolver`/`XpbdSolver`.
//!
//! Reads the same `Model` deformable data VBD does (particles, XPBD
//! springs with compliance→stiffness, `SoftSurface` triangles for
//! self-collision, static Plane/Sphere shapes as analytic obstacles)
//! and drives the `barrier_contact::BarrierShell` oracle per step:
//! penetration is impossible at every Newton iterate, hard strain
//! limits and μ_s/μ_k friction come along for free.
//!
//! v1 scope (documented, honest):
//! * static analytic obstacles only — Plane/Sphere shapes attached to
//!   `BodyId::NONE`; rigid-body state passes through untouched (no
//!   two-way coupling yet);
//! * cloth/shell surfaces (`SoftSurface` triangles); Model tets use
//!   VBD's Neo-Hookean material and are NOT consumed here (the
//!   barrier backend's own `BarrierTet` volume energy is a different
//!   material — wire when a scene needs it);
//! * CPU f64. The GPU `BarrierNewtonGpu` shares topology and can slot
//!   behind the same interface once a scene needs the scale.

use crate::Solver;
use crate::barrier_contact::{BarrierObstacle, BarrierParams, BarrierShell, BarrierSpring};
use crate::core::Result;
use crate::math::Real;
use molla_sim::{Contacts, Control, Model, ShapeGeometry, State};

/// Barrier-Newton deformable solver (see module docs).
pub struct BarrierSolver {
    /// Topology-carrying shell, built lazily from the `Model` on the
    /// first step (positions/velocities are streamed per step).
    shell: Option<BarrierShell>,
    /// Solver parameters applied when the shell is (re)built, and
    /// pushed into the live shell every step so viewer panes can
    /// mutate them.
    pub params: BarrierParams,
    /// Stats from the most recent step (viewer/diagnostics surface).
    pub last_newton_iters: usize,
    pub last_min_gap: Real,
}

impl Default for BarrierSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BarrierSolver {
    pub fn new() -> Self {
        Self {
            shell: None,
            params: BarrierParams::default(),
            last_newton_iters: 0,
            last_min_gap: Real::INFINITY,
        }
    }

    pub fn with_params(mut self, params: BarrierParams) -> Self {
        self.params = params;
        self
    }

    /// Build the shell topology from the model + initial state.
    fn build_shell(&self, model: &Model, state: &State) -> Result<BarrierShell> {
        let positions = state.particle_q.host()?.to_vec();
        let velocities = state.particle_qd.host()?.to_vec();
        let masses = model.particle_mass.host()?.to_vec();
        let inv_mass = model.particle_inv_mass.host()?;

        // XPBD compliance → stiffness, VBD's convention (capped so
        // rigid-limit springs stay finite).
        let spring_a = model.spring_a.host()?;
        let spring_b = model.spring_b.host()?;
        let spring_rest = model.spring_rest_length.host()?;
        let spring_comp = model.spring_compliance.host()?;
        const MAX_K: Real = 1.0e5;
        let mut springs = Vec::with_capacity(model.spring_count);
        for i in 0..model.spring_count {
            let k = if spring_comp[i] > 1.0e-12 {
                (1.0 / spring_comp[i]).min(MAX_K)
            } else {
                MAX_K
            };
            springs.push(BarrierSpring {
                a: spring_a[i],
                b: spring_b[i],
                rest: spring_rest[i],
                k,
            });
        }

        // Static analytic obstacles from world-attached shapes.
        let mut obstacles = Vec::new();
        for s in 0..model.shape_count {
            if model.shape_body.host()?[s] >= 0 {
                continue; // body-attached: rigid coupling is future work
            }
            let xf = model.shape_transform.host()?[s];
            match model.shape_geometry[s] {
                ShapeGeometry::Plane { normal } => {
                    let n = (xf.rotation * normal).normalize();
                    obstacles.push(BarrierObstacle::HalfSpace {
                        n,
                        offset: n.dot(xf.position),
                    });
                }
                ShapeGeometry::Sphere { radius } => {
                    obstacles.push(BarrierObstacle::Sphere {
                        center: xf.position,
                        radius,
                    });
                }
                _ => {} // other shapes: not representable analytically (v1)
            }
        }

        // Self-collision triangles from every soft surface.
        let triangles: Vec<[u32; 3]> = model.soft_surface_tris.clone();

        let mut masses_pos = masses.clone();
        // BarrierShell requires strictly positive masses; pinned
        // particles (inv_mass = 0) become true DoF removals below and
        // their mass value is only used for preconditioning scale.
        let pinned: Vec<usize> = inv_mass
            .iter()
            .enumerate()
            .filter(|&(_, &w)| w == 0.0)
            .map(|(i, _)| i)
            .collect();
        for &i in &pinned {
            if masses_pos[i] <= 0.0 {
                masses_pos[i] = 1.0;
            }
        }

        let mut shell = BarrierShell::new(positions, masses_pos, springs, obstacles, self.params)?;
        shell.velocities = velocities;
        if !triangles.is_empty() {
            shell = shell.with_triangles(triangles);
        }
        shell.pin_vertices(&pinned);
        Ok(shell)
    }
}

impl Solver for BarrierSolver {
    fn step(
        &mut self,
        model: &Model,
        state_in: &State,
        _control: &Control,
        _contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
    ) -> Result<()> {
        if self.shell.is_none() {
            self.shell = Some(self.build_shell(model, state_in)?);
        }
        let shell = self.shell.as_mut().expect("shell built above");
        shell.params = self.params;

        // Stream state in (positions may have been edited externally,
        // e.g. viewer reset — the topology is what the shell owns).
        {
            let q = state_in.particle_q.host()?;
            let qd = state_in.particle_qd.host()?;
            shell.positions.copy_from_slice(q);
            shell.velocities.copy_from_slice(qd);
        }

        let stats = shell.step(dt)?;
        self.last_newton_iters = stats.newton_iters;
        self.last_min_gap = stats.min_gap;

        state_out
            .particle_q
            .host_mut()?
            .copy_from_slice(&shell.positions);
        state_out
            .particle_qd
            .host_mut()?
            .copy_from_slice(&shell.velocities);

        // Rigid-body state passes through (static obstacles only).
        let q_in = state_in.body_q.host()?.to_vec();
        let qd_in = state_in.body_qd.host()?.to_vec();
        state_out.body_q.host_mut()?.copy_from_slice(&q_in);
        state_out.body_qd.host_mut()?.copy_from_slice(&qd_in);
        Ok(())
    }
}
