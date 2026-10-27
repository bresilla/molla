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
//! Rigid coupling (v1.5, weak/explicit): obstacles attached to
//! DYNAMIC bodies follow their body's live pose each step, and the
//! reaction wrench the shell reports at the converged state (implicit
//! Euler gives the shell exactly `h·F(x_new)` of impulse, so sampling
//! the same forces makes the exchange third-law-exact to Newton
//! tolerance) integrates the body semi-implicitly. Bodies not owning
//! barrier obstacles pass through. Strong (monolithic ABD-style)
//! coupling is the eventual upgrade; the weak scheme is conditionally
//! stable — heavy cloth on a light body wants smaller h.
//!
//! v1 scope (documented, honest):
//! * cloth/shell surfaces (`SoftSurface` triangles); Model tets use
//!   VBD's Neo-Hookean material and are NOT consumed here (the
//!   barrier backend's own `BarrierTet` volume energy is a different
//!   material — wire when a scene needs it);
//! * CPU f64. The GPU `BarrierNewtonGpu` shares topology and can slot
//!   behind the same interface once a scene needs the scale.

use crate::Solver;
use crate::barrier_contact::{BarrierObstacle, BarrierParams, BarrierShell, BarrierSpring};
use crate::core::Result;
use crate::math::{Real, Vec3};
use molla_sim::{Contacts, Control, Model, ShapeGeometry, State};

/// One obstacle's provenance: which body owns it (−1 = world/static)
/// and the shape's body-local geometry, so poses can be refreshed and
/// reactions attributed each step.
#[derive(Copy, Clone, Debug)]
struct ObstacleBinding {
    body: i32,
    local: molla_math::Transform,
    geom: ObstacleGeom,
}

#[derive(Copy, Clone, Debug)]
enum ObstacleGeom {
    Plane { normal: Vec3 },
    Sphere { radius: Real },
}

/// GPU backend state (feature-gated): device handles + the lazily
/// built kernel sim. The GPU path owns its trajectory (uploaded once,
/// read back each step for consumers) and supports STATIC obstacles
/// only — rigid two-way coupling stays on the CPU path.
#[cfg(feature = "gpu")]
struct GpuBackend {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    sim: Option<molla_kernels::BarrierNewtonGpu>,
}

/// Barrier-Newton deformable solver (see module docs).
pub struct BarrierSolver {
    /// Topology-carrying shell, built lazily from the `Model` on the
    /// first step (positions/velocities are streamed per step).
    shell: Option<BarrierShell>,
    #[cfg(feature = "gpu")]
    gpu: Option<GpuBackend>,
    /// Solver parameters applied when the shell is (re)built, and
    /// pushed into the live shell every step so viewer panes can
    /// mutate them.
    pub params: BarrierParams,
    /// Stats from the most recent step (viewer/diagnostics surface).
    pub last_newton_iters: usize,
    pub last_min_gap: Real,
    /// Obstacle provenance parallel to the shell's obstacle list.
    bindings: Vec<ObstacleBinding>,
    /// Treat every Model spring as a COLLISION edge (rod/chain scenes
    /// — noodles, yarn, knots — where there are no triangles to derive
    /// edges from). Set before the first step.
    pub springs_as_collision_edges: bool,
    /// (model body index, shell affine-body index) for dynamic
    /// BOX-shape bodies promoted to STRONG (ABD) coupling — their
    /// corner/face proxies + 12 surface triangles join the barrier
    /// solve, and poses write back after each step.
    abd_bodies: Vec<(usize, usize)>,
    /// Model particle count at shell build time (the shell may hold
    /// MORE vertices — ABD proxies — so topology-change detection
    /// must compare against this, not `positions.len()`).
    built_particles: usize,
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
            #[cfg(feature = "gpu")]
            gpu: None,
            params: BarrierParams::default(),
            last_newton_iters: 0,
            last_min_gap: Real::INFINITY,
            bindings: Vec::new(),
            springs_as_collision_edges: false,
            abd_bodies: Vec::new(),
            built_particles: usize::MAX,
        }
    }

    pub fn with_params(mut self, params: BarrierParams) -> Self {
        self.params = params;
        self
    }

    /// GPU-backed variant: the f32 `BarrierNewtonGpu` kernel owns the
    /// trajectory (state read back each step). Static obstacles only —
    /// scenes with dynamic obstacle-owning bodies should use the CPU
    /// solver until the coupling moves on-device.
    #[cfg(feature = "gpu")]
    pub fn new_gpu(engine: &molla_compute::Engine) -> Result<Self> {
        use molla_core::Error;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("engine has no GPU queue".into()))?
            .clone();
        let mut s = Self::new();
        s.gpu = Some(GpuBackend {
            device,
            queue,
            sim: None,
        });
        Ok(s)
    }

    /// Map the f64 solver params onto the f32 kernel params (Newton
    /// budget stays at the kernel's f32-calibrated defaults).
    #[cfg(feature = "gpu")]
    fn gpu_params(&self) -> molla_kernels::BarrierGpuParams {
        molla_kernels::BarrierGpuParams {
            dhat: self.params.dhat as f32,
            kappa_scale: self.params.kappa_scale as f32,
            gravity: [
                self.params.gravity.x as f32,
                self.params.gravity.y as f32,
                self.params.gravity.z as f32,
            ],
            friction_mu: self.params.friction_mu as f32,
            friction_mu_s_scale: self.params.friction_mu_s_scale as f32,
            friction_epsv: self.params.friction_epsv as f32,
            strain_limit: self.params.strain_limit.unwrap_or(0.0) as f32,
            strain_barrier_scale: self.params.strain_barrier_scale as f32,
            elasticity_inclusive_kappa: self.params.elasticity_inclusive_kappa,
            ..Default::default()
        }
    }

    /// Build the GPU sim from the same Model extraction the CPU shell
    /// uses.
    #[cfg(feature = "gpu")]
    fn build_gpu(&self, model: &Model, state: &State) -> Result<molla_kernels::BarrierNewtonGpu> {
        if model.spring_tension_only.host()?.iter().any(|&v| v != 0) {
            return Err(molla_core::Error::Solver(
                "barrier solver does not support tension-only elements".into(),
            ));
        }
        use molla_kernels::{BarrierGpuObstacle, BarrierGpuSpring, BarrierNewtonGpu};
        let positions: Vec<[f32; 3]> = state
            .particle_q
            .host()?
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        let inv_mass = model.particle_inv_mass.host()?;
        let masses: Vec<f32> = model
            .particle_mass
            .host()?
            .iter()
            .zip(inv_mass.iter())
            .map(|(&m, &w)| {
                if m > 0.0 {
                    Ok(m as f32)
                } else if w == 0.0 {
                    Ok(1.0) // pinned: mass only scales preconditioning
                } else {
                    Err(molla_core::Error::Solver(
                        "barrier gpu: non-positive mass on a free particle".into(),
                    ))
                }
            })
            .collect::<Result<_>>()?;
        let pinned: Vec<u32> = inv_mass
            .iter()
            .enumerate()
            .filter(|&(_, &w)| w == 0.0)
            .map(|(i, _)| i as u32)
            .collect();
        let spring_a = model.spring_a.host()?;
        let spring_b = model.spring_b.host()?;
        let spring_rest = model.spring_rest_length.host()?;
        let spring_comp = model.spring_compliance.host()?;
        let mut springs = Vec::with_capacity(model.spring_count);
        for i in 0..model.spring_count {
            let k = if spring_comp[i] > 1.0e-12 {
                (1.0 / spring_comp[i]).min(1.0e5)
            } else {
                1.0e5
            };
            springs.push(BarrierGpuSpring {
                a: spring_a[i],
                b: spring_b[i],
                rest: spring_rest[i] as f32,
                k: k as f32,
            });
        }
        let shape_body = model.shape_body.host()?;
        let shape_xf = model.shape_transform.host()?;
        let body_inv_mass_g = model.body_inv_mass.host()?;
        let mut obstacles = Vec::new();
        for s in 0..model.shape_count {
            if shape_body[s] >= 0 {
                if matches!(model.shape_geometry[s], ShapeGeometry::Box { .. })
                    && body_inv_mass_g[shape_body[s] as usize] > 0.0
                {
                    return Err(molla_core::Error::Solver(
                        "barrier gpu: dynamic box (ABD) bodies are CPU-path only (v1) — use BarrierSolver::new()".into(),
                    ));
                }
                continue; // GPU path: static obstacles only (documented)
            }
            let xf = shape_xf[s];
            match model.shape_geometry[s] {
                ShapeGeometry::Plane { normal } => {
                    let n = (xf.rotation * normal).normalize();
                    obstacles.push(BarrierGpuObstacle::HalfSpace {
                        n: [n.x as f32, n.y as f32, n.z as f32],
                        offset: n.dot(xf.position) as f32,
                    });
                }
                ShapeGeometry::Sphere { radius } => {
                    obstacles.push(BarrierGpuObstacle::Sphere {
                        center: [
                            xf.position.x as f32,
                            xf.position.y as f32,
                            xf.position.z as f32,
                        ],
                        radius: radius as f32,
                    });
                }
                _ => {}
            }
        }
        let triangles: Vec<[u32; 3]> = model.soft_surface_tris.clone();
        let gpu = self.gpu.as_ref().expect("gpu backend present");
        let sim = BarrierNewtonGpu::new_with_mesh(
            &gpu.device,
            &gpu.queue,
            &positions,
            &masses,
            &springs,
            &obstacles,
            &triangles,
            &pinned,
            self.gpu_params(),
        )?;
        let velocities: Vec<[f32; 3]> = state
            .particle_qd
            .host()?
            .iter()
            .map(|v| [v.x as f32, v.y as f32, v.z as f32])
            .collect();
        sim.set_velocities(&velocities)?;
        Ok(sim)
    }

    /// Build the shell topology from the model + initial state.
    fn build_shell(&mut self, model: &Model, state: &State) -> Result<BarrierShell> {
        if model.spring_tension_only.host()?.iter().any(|&v| v != 0) {
            return Err(molla_core::Error::Solver(
                "barrier solver does not support tension-only elements".into(),
            ));
        }
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

        // Analytic obstacles: world-attached shapes are static;
        // body-attached Plane/Sphere shapes follow their body's live
        // pose (weak two-way coupling — see module docs).
        let mut obstacles = Vec::new();
        self.bindings.clear();
        let shape_body = model.shape_body.host()?;
        let shape_xf = model.shape_transform.host()?;
        let body_q = state.body_q.host()?;
        for s in 0..model.shape_count {
            let body = shape_body[s];
            let local = shape_xf[s];
            let world = if body >= 0 {
                body_q[body as usize].compose(&local)
            } else {
                local
            };
            match model.shape_geometry[s] {
                ShapeGeometry::Plane { normal } => {
                    let n = (world.rotation * normal).normalize();
                    obstacles.push(BarrierObstacle::HalfSpace {
                        n,
                        offset: n.dot(world.position),
                    });
                    self.bindings.push(ObstacleBinding {
                        body,
                        local,
                        geom: ObstacleGeom::Plane { normal },
                    });
                }
                ShapeGeometry::Sphere { radius } => {
                    obstacles.push(BarrierObstacle::Sphere {
                        center: world.position,
                        radius,
                    });
                    self.bindings.push(ObstacleBinding {
                        body,
                        local,
                        geom: ObstacleGeom::Sphere { radius },
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
        if self.springs_as_collision_edges {
            // Only CHAIN springs become collision edges: cloth-owned
            // springs are excluded — a cloth quad's two DIAGONAL
            // springs cross at distance zero, which would violate the
            // invariant at t = 0 (cloth already collides through its
            // soft-surface triangles).
            let in_cloth = |v: u32| {
                model
                    .soft_surfaces
                    .iter()
                    .any(|s| v >= s.vertex_offset && v < s.vertex_offset + s.vertex_count)
            };
            let edges: Vec<[u32; 2]> = (0..model.spring_count)
                .filter(|&i| !in_cloth(spring_a[i]) || !in_cloth(spring_b[i]))
                .map(|i| [spring_a[i], spring_b[i]])
                .collect();
            shell = shell.with_collision_edges(edges);
        }
        // STRONG coupling: dynamic bodies with a BOX shape become
        // affine bodies — 8 corner + 6 face-centre proxies and the 12
        // box surface triangles join the barrier solve (this is what
        // makes dominoes/cards work: boxes are not analytic obstacles,
        // they are ABD bodies).
        self.abd_bodies.clear();
        let body_inv_mass = model.body_inv_mass.host()?;
        let body_mass = model.body_mass.host()?;
        let body_q0 = state.body_q.host()?;
        for s in 0..model.shape_count {
            let b = shape_body[s];
            if b < 0 || body_inv_mass[b as usize] <= 0.0 {
                continue;
            }
            let ShapeGeometry::Box { half_extents } = model.shape_geometry[s] else {
                continue;
            };
            let world = body_q0[b as usize].compose(&shape_xf[s]);
            let h = half_extents;
            let mut samples = Vec::with_capacity(14);
            for &sx in &[-1.0, 1.0] {
                for &sy in &[-1.0, 1.0] {
                    for &sz in &[-1.0, 1.0] {
                        samples.push(world.transform_point(Vec3::new(
                            sx * h.x,
                            sy * h.y,
                            sz * h.z,
                        )));
                    }
                }
            }
            for c in 0..3 {
                for &sgn in &[-1.0, 1.0] {
                    let mut v = Vec3::ZERO;
                    v[c] = sgn * h[c];
                    samples.push(world.transform_point(v));
                }
            }
            let m_each = body_mass[b as usize] / samples.len() as Real;
            let start = shell.positions.len() as u32;
            let abd_idx =
                shell.add_affine_body(samples, vec![m_each; 14], 1.0e5 * body_mass[b as usize]);
            // Inherit the body's initial motion: v_p from linear, and
            // v_A = [ω]× · A from angular velocity.
            let qd0 = state.body_qd.host()?[b as usize];
            let w = qd0.angular;
            let a0 = shell.affine_bodies[abd_idx].a_mat;
            let wx = molla_math::Mat3::from_cols(
                Vec3::new(0.0, w.z, -w.y),
                Vec3::new(-w.z, 0.0, w.x),
                Vec3::new(w.y, -w.x, 0.0),
            );
            shell.affine_bodies[abd_idx].v_p = qd0.linear;
            shell.affine_bodies[abd_idx].v_a = wx * a0;
            // Corner indexing: (sx,sy,sz) → 4*ix + 2*iy + iz.
            let c = |ix: u32, iy: u32, iz: u32| start + 4 * ix + 2 * iy + iz;
            let box_tris = vec![
                [c(0, 0, 0), c(0, 1, 0), c(0, 1, 1)],
                [c(0, 0, 0), c(0, 1, 1), c(0, 0, 1)],
                [c(1, 0, 0), c(1, 1, 1), c(1, 1, 0)],
                [c(1, 0, 0), c(1, 0, 1), c(1, 1, 1)],
                [c(0, 0, 0), c(1, 0, 1), c(1, 0, 0)],
                [c(0, 0, 0), c(0, 0, 1), c(1, 0, 1)],
                [c(0, 1, 0), c(1, 1, 0), c(1, 1, 1)],
                [c(0, 1, 0), c(1, 1, 1), c(0, 1, 1)],
                [c(0, 0, 0), c(1, 0, 0), c(1, 1, 0)],
                [c(0, 0, 0), c(1, 1, 0), c(0, 1, 0)],
                [c(0, 0, 1), c(1, 1, 1), c(1, 0, 1)],
                [c(0, 0, 1), c(0, 1, 1), c(1, 1, 1)],
            ];
            shell = shell.with_triangles_appended(box_tris);
            self.abd_bodies.push((b as usize, abd_idx));
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
        // Topology change (particle count) invalidates the lazily
        // built backends — rebuild instead of panicking on a length
        // mismatch (scene edits / model swaps).
        let n_particles = state_in.particle_q.host()?.len();
        if self.shell.is_some() && self.built_particles != n_particles {
            self.shell = None;
        }
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            let stale = self
                .gpu
                .as_ref()
                .and_then(|g| g.sim.as_ref())
                .is_some_and(|s| s.vertex_count() != n_particles);
            if stale && let Some(g) = self.gpu.as_mut() {
                g.sim = None;
            }
            if self.gpu.as_ref().is_some_and(|g| g.sim.is_none()) {
                let sim = self.build_gpu(model, state_in)?;
                if let Some(g) = self.gpu.as_mut() {
                    g.sim = Some(sim);
                }
            }
            let gp = self.gpu_params();
            let Some(sim) = self.gpu.as_mut().and_then(|g| g.sim.as_mut()) else {
                unreachable!("built above");
            };
            sim.params = gp;
            let stats = sim.step(dt as f32)?;
            self.last_newton_iters = stats.newton_iters;
            self.last_min_gap = Real::from(stats.min_gap);
            let p = sim.read_positions()?;
            let v = sim.read_velocities()?;
            {
                let q_out = state_out.particle_q.host_mut()?;
                for (dst, src) in q_out.iter_mut().zip(p.iter()) {
                    *dst = crate::math::Vec3::new(
                        Real::from(src[0]),
                        Real::from(src[1]),
                        Real::from(src[2]),
                    );
                }
                let qd_out = state_out.particle_qd.host_mut()?;
                for (dst, src) in qd_out.iter_mut().zip(v.iter()) {
                    *dst = crate::math::Vec3::new(
                        Real::from(src[0]),
                        Real::from(src[1]),
                        Real::from(src[2]),
                    );
                }
            }
            let q_in = state_in.body_q.host()?.to_vec();
            let qd_in = state_in.body_qd.host()?.to_vec();
            state_out.body_q.host_mut()?.copy_from_slice(&q_in);
            state_out.body_qd.host_mut()?.copy_from_slice(&qd_in);
            return Ok(());
        }
        if self.shell.is_none() {
            self.shell = Some(self.build_shell(model, state_in)?);
            self.built_particles = n_particles;
        }
        let shell = self.shell.as_mut().expect("shell built above");
        shell.params = self.params;

        // Stream state in (positions may have been edited externally,
        // e.g. viewer reset — the topology is what the shell owns).
        // ABD proxy vertices live BEYOND the Model's particle range
        // and are owned by the shell — stream only the Model slice.
        {
            let q = state_in.particle_q.host()?;
            let qd = state_in.particle_qd.host()?;
            shell.positions[..q.len()].copy_from_slice(q);
            shell.velocities[..qd.len()].copy_from_slice(qd);
        }

        // Weak-coupling sequencing with a RIGID-SIDE TOI: tentatively
        // integrate each coupled body under gravity, cap its
        // displacement so no owned obstacle can cross into the barrier
        // band unseen (the cloth-side CCD only guards a FROZEN
        // obstacle — without this cap a fast ball tunnels straight
        // through the cloth between steps), then run the deformable
        // solve against the capped poses. The reaction then builds
        // over the following steps; resting contact converges to
        // force balance with α → the honest fraction.
        let inv_mass = model.body_inv_mass.host()?;
        let mut q = state_in.body_q.host()?.to_vec();
        let mut qd = state_in.body_qd.host()?.to_vec();
        let mut coupled = vec![false; q.len()];
        for b in &self.bindings {
            if b.body >= 0 && inv_mass[b.body as usize] > 0.0 {
                coupled[b.body as usize] = true;
            }
        }
        let mut v_tent: Vec<Vec3> = qd.iter().map(|v| v.linear).collect();
        let mut alpha_body = vec![1.0 as Real; q.len()];
        for (i, coupled_i) in coupled.iter().enumerate() {
            if !coupled_i {
                continue;
            }
            v_tent[i] += self.params.gravity * dt;
            let disp = v_tent[i] * dt;
            let dlen = disp.length();
            if dlen <= 1.0e-12 {
                continue;
            }
            // Conservative TOI: an obstacle can close a vertex gap by
            // at most |disp|; keep 10% of the band in reserve.
            let mut min_gap = Real::INFINITY;
            for (o, b) in self.bindings.iter().enumerate() {
                if b.body as usize != i || b.body < 0 {
                    continue;
                }
                for (v_idx, xv) in shell.positions.iter().enumerate() {
                    if shell.pinned[v_idx] {
                        continue;
                    }
                    let (gap, n_dir) = match shell.obstacles[o] {
                        BarrierObstacle::Sphere { center, radius } => {
                            let d = *xv - center;
                            let len = d.length().max(1.0e-12);
                            (len - radius, d / len)
                        }
                        BarrierObstacle::HalfSpace { n, offset } => (n.dot(*xv) - offset, n),
                    };
                    // Only gaps the motion actually closes matter.
                    if disp.dot(n_dir) > 0.0 {
                        min_gap = min_gap.min(gap);
                    }
                }
            }
            if min_gap.is_finite() {
                let usable = (min_gap - 0.1 * self.params.dhat).max(0.0);
                alpha_body[i] = (usable / dlen).min(1.0);
            }
        }
        // Advance body poses by the capped displacement and refresh
        // obstacle world poses for the deformable solve.
        for (i, coupled_i) in coupled.iter().enumerate() {
            if *coupled_i {
                q[i].position += v_tent[i] * (dt * alpha_body[i]);
            }
        }
        {
            for (o, b) in self.bindings.iter().enumerate() {
                if b.body < 0 {
                    continue;
                }
                let world = q[b.body as usize].compose(&b.local);
                shell.obstacles[o] = match b.geom {
                    ObstacleGeom::Plane { normal } => {
                        let n = (world.rotation * normal).normalize();
                        BarrierObstacle::HalfSpace {
                            n,
                            offset: n.dot(world.position),
                        }
                    }
                    ObstacleGeom::Sphere { radius } => BarrierObstacle::Sphere {
                        center: world.position,
                        radius,
                    },
                };
            }
        }

        let stats = shell.step(dt)?;
        self.last_newton_iters = stats.newton_iters;
        self.last_min_gap = stats.min_gap;

        {
            let q_out = state_out.particle_q.host_mut()?;
            let np = q_out.len();
            q_out.copy_from_slice(&shell.positions[..np]);
            let qd_out = state_out.particle_qd.host_mut()?;
            qd_out.copy_from_slice(&shell.velocities[..np]);
        }

        // Reaction wrench (third-law-exact — see `obstacle_reactions`)
        // → velocity update; position already advanced by the capped
        // displacement above. Torque about the body origin (v1 treats
        // it as the COM — documented).
        let inv_inertia = model.body_inv_inertia.host()?;
        let reactions = shell.obstacle_reactions(dt);
        let mut wrench: Vec<(Vec3, Vec3)> = vec![(Vec3::ZERO, Vec3::ZERO); q.len()];
        for (o, b) in self.bindings.iter().enumerate() {
            if b.body >= 0 && inv_mass[b.body as usize] > 0.0 {
                let bi = b.body as usize;
                let (f, tau_origin) = reactions[o];
                wrench[bi].0 += f;
                wrench[bi].1 += tau_origin - q[bi].position.cross(f);
            }
        }
        // STRONG-coupled (ABD) bodies: write poses back from the
        // shell's affine state — position from p, rotation from the
        // Gram-Schmidt orthonormalization of the near-orthogonal A
        // (κ_o keeps ‖AᵀA−I‖ ~1e-6; the projection is for rendering
        // and downstream consumers, the solve itself uses A).
        for &(bi, ai) in &self.abd_bodies {
            let body = &shell.affine_bodies[ai];
            let a = body.a_mat;
            let c0 = a.col(0).normalize();
            let c1 = (a.col(1) - c0 * c0.dot(a.col(1))).normalize();
            let c2 = c0.cross(c1);
            let rot =
                molla_math::Quat::from_mat3(&molla_math::Mat3::from_cols(c0, c1, c2)).normalize();
            q[bi].position = body.p;
            q[bi].rotation = rot;
            qd[bi].linear = body.v_p;
            // ω from the skew part of Ȧ·Aᵀ.
            let w_mat = body.v_a * a.transpose();
            qd[bi].angular = Vec3::new(
                0.5 * (w_mat.col(1).z - w_mat.col(2).y),
                0.5 * (w_mat.col(2).x - w_mat.col(0).z),
                0.5 * (w_mat.col(0).y - w_mat.col(1).x),
            );
        }
        for i in 0..q.len() {
            if !coupled[i] {
                continue;
            }
            let r = q[i].rotation;
            let i_inv_world = molla_math::Mat3::from_quat(r)
                * inv_inertia[i]
                * molla_math::Mat3::from_quat(r).transpose();
            qd[i].linear = v_tent[i] + wrench[i].0 * (inv_mass[i] * dt);
            qd[i].angular += i_inv_world * wrench[i].1 * dt;
            let w = qd[i].angular;
            let wlen = w.length();
            if wlen > 1.0e-12 {
                let dq = molla_math::Quat::from_axis_angle(w / wlen, wlen * dt);
                q[i].rotation = (dq * q[i].rotation).normalize();
            }
        }
        state_out.body_q.host_mut()?.copy_from_slice(&q);
        state_out.body_qd.host_mut()?.copy_from_slice(&qd);
        Ok(())
    }
}
