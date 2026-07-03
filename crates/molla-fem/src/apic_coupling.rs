//! Stage 37.6 — APIC FEM-MPM hybrid coupling.
//!
//! Couples the Stage 24d MPM particle representation (positions,
//! velocities, **affine** matrix `C`) with the Stage 36 FEM tetrahedral
//! mesh (P1 nodal). Particles transfer momentum to mesh **nodes** via
//! APIC weights (P1 barycentric); the mesh provides the velocity-field
//! step (viscous diffusion + boundary conditions); particles read the
//! updated node velocities back to update their own velocity + affine
//! matrix.
//!
//! # Algorithm (per substep)
//!
//! 1. **P2G** (particle → grid/mesh): each particle `p` deposits
//!    momentum at the four vertices of its enclosing tet:
//!
//!    ```text
//!    node_momentum[i] += w_i · m_p · (v_p + C_p · (x_i − x_p))
//!    ```
//!
//!    where `w_i` is the P1 barycentric coordinate of `x_p` w.r.t. tet
//!    vertex `i`. (See Jiang et al. 2015, "The Affine Particle-In-Cell
//!    Method", and Warp's `examples/fem/example_apic_fluid.py` for the
//!    Stokes-coupled variant we mirror here.)
//!
//! 2. **Mesh solve**: from `(node_mass, node_momentum)` recover
//!    `node_velocity = node_momentum / node_mass`. Apply gravity
//!    (`v += dt · g`), apply Dirichlet boundary conditions (no-slip
//!    walls pin to zero), and run a **viscous Laplacian smoothing pass**
//!    over the mesh (`v ← v − dt · ν · L · v`) to dissipate sub-grid
//!    energy. This is a v1 stand-in for a full P1 Stokes pressure-
//!    Poisson projection — molla's Stokes solver lives on Taylor-Hood
//!    (P2 velocity) so plumbing the APIC P1 deposit through the P2
//!    space requires either a P2 deposit (rebuilds the contract) or an
//!    L2 projection between spaces (Stage 37.6.1). Pure viscous
//!    smoothing keeps the contract tests honest: momentum-roundtrip is
//!    exact (no Stokes inside the test), uniform-motion is exact (the
//!    Laplacian of a constant field is zero), and falling particles
//!    settle (viscosity dissipates KE, BCs hold the box).
//!
//! 3. **G2P** (grid/mesh → particle): each particle reads back
//!
//!    ```text
//!    v_p_new = Σ w_i · v_node_i
//!    C_p_new = Σ w_i · v_node_i · (x_i − x_p)ᵀ · D⁻¹
//!    ```
//!
//!    where `D⁻¹` is the (tet-local) inverse of `Σ w_i · (x_i − x_p) ·
//!    (x_i − x_p)ᵀ`. For P1 tets the barycentric weights are linear in
//!    `(x_i − x_p)` so this matrix has a clean closed form (it equals
//!    the second moment of the tet's vertices about the particle).
//!
//! 4. **Advect**: `x_p += dt · v_p_new`.
//!
//! # Particle-in-tet localisation (v1)
//!
//! v1 ships a naive O(P · T) brute-force barycentric test
//! ([`locate_particle_tet`]). Acceptable for the contract tests
//! (≤ 200 particles, ≤ a few hundred tets); a spatial hash lives in
//! the deferred Stage 37.6.1 follow-up.
//!
//! # Caveats / future work
//!
//! - **Pressure projection (Stage 37.6.1, CPU).** `step_projected`
//!   runs a per-tet (P0) pressure projection through the lumped-mass
//!   Schur operator `S = D M⁻¹ Dᵀ` (matrix-free CG) after the grid
//!   update, driving per-tet divergence on the deposit-supported
//!   fluid region to the solver floor (machine epsilon on the blob
//!   contract). Equal-order P1 node pressure was rejected: its lumped
//!   correction only cancels nodal moments and Richardson outer loops
//!   amplify LBB checkerboard modes. `step` keeps the viscous-only
//!   behavior (`pressure_iters = 0`). The GPU `step_gpu` does NOT yet
//!   run the projection — port design: `apply_s` is two gather passes
//!   over the existing node↔tet adjacency plus CG dot-product
//!   reductions (reuse the `fem_cg_fused.wgsl` pattern).
//! - **GPU port (Stage 41 v2).** The `step_gpu` method runs the full
//!   pipeline (locate, P2G, grid-update, viscous-smoothing, BCs, G2P)
//!   on the device. Mirrors the CPU mathematics exactly; the only
//!   semantic delta is f32 vs f64 round-off. See
//!   [`ApicHybrid::step_gpu`].
//! - **No particle re-seeding** if a tet empties (particles can drift
//!   slightly out of bounds when the explicit advect overshoots a
//!   wall). The `falling-particles-settle` contract test clamps to
//!   the box AABB to keep the demo honest.

use crate::core::{Error, Result};
use crate::fem::bc::BoundaryCondition;
use crate::fem::mesh::{FemCell, FemMesh};
use crate::math::{Mat3, Real, Vec3};

/// One MPM-style particle in the APIC hybrid.
///
/// `c` is the APIC affine-velocity matrix (Jiang et al. 2015): it
/// captures the local velocity gradient at the particle's location and
/// is what makes the transfer angular-momentum-preserving (PIC loses
/// it, FLIP injects it noisily, APIC carries it cleanly).
#[derive(Copy, Clone, Debug)]
pub struct ApicParticle {
    pub x: Vec3,
    pub v: Vec3,
    pub c: Mat3,
    pub mass: Real,
}

impl ApicParticle {
    /// Particle at rest with zero affine matrix.
    #[must_use]
    pub fn at_rest(x: Vec3, mass: Real) -> Self {
        Self {
            x,
            v: Vec3::ZERO,
            c: Mat3::ZERO,
            mass,
        }
    }
}

/// APIC FEM-MPM hybrid simulation. Owns both the mesh (the FEM grid)
/// and the particle list (the MPM Lagrangian carriers).
#[derive(Clone, Debug)]
pub struct ApicHybrid {
    pub mesh: FemMesh,
    pub particles: Vec<ApicParticle>,
    /// Gravity applied at every mesh-velocity update. Defaults to
    /// `(0, 0, -9.80665)` so falling-particle scenes work out of the box.
    pub gravity: Vec3,
}

impl ApicHybrid {
    /// Construct from a mesh + particle list. Gravity defaults to
    /// `−Z · g_standard`; override directly via [`Self::gravity`] after
    /// construction.
    #[must_use]
    pub fn new(mesh: FemMesh, particles: Vec<ApicParticle>) -> Self {
        Self {
            mesh,
            particles,
            gravity: Vec3::new(0.0, 0.0, -crate::math::GRAVITY),
        }
    }

    /// Number of particles.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Total kinetic energy `½ Σ m · |v|²` summed over all particles.
    /// Used by the falling-particle contract test + the viewer readout.
    #[must_use]
    pub fn total_kinetic_energy(&self) -> Real {
        self.particles
            .iter()
            .map(|p| 0.5 * p.mass * p.v.length_squared())
            .sum()
    }

    /// Total linear momentum `Σ m · v` summed over all particles.
    #[must_use]
    pub fn total_momentum(&self) -> Vec3 {
        self.particles
            .iter()
            .fold(Vec3::ZERO, |acc, p| acc + p.mass * p.v)
    }

    /// **P2G** — scatter particle momentum to mesh nodes. Returns
    /// `(node_mass, node_momentum)`.
    ///
    /// Particles outside every tet contribute nothing (silent skip).
    /// Mass is also scattered (proportional to the same barycentric
    /// weights) so the caller can recover `node_velocity = momentum /
    /// mass` later.
    #[must_use]
    pub fn p2g(&self) -> (Vec<Real>, Vec<Vec3>) {
        let n = self.mesh.vert_count();
        let mut node_mass: Vec<Real> = vec![0.0; n];
        let mut node_momentum = vec![Vec3::ZERO; n];

        for particle in &self.particles {
            let Some((tet_verts, weights, positions)) = locate_particle_tet(&self.mesh, particle.x)
            else {
                continue;
            };
            for k in 0..4 {
                let dx = positions[k] - particle.x;
                // APIC momentum: w · m · (v + C · dx)
                let contribution = weights[k] * particle.mass * (particle.v + particle.c * dx);
                let idx = tet_verts[k] as usize;
                node_momentum[idx] += contribution;
                node_mass[idx] += weights[k] * particle.mass;
            }
        }

        (node_mass, node_momentum)
    }

    /// **G2P** — gather mesh velocities back to particles. Updates
    /// `v` and the affine matrix `C` per APIC. Optional `dt` is used
    /// only to advect `x`; pass `0.0` for the round-trip test (no
    /// motion).
    ///
    /// `node_velocities[i]` is the velocity at mesh vertex `i`.
    pub fn g2p(&mut self, node_velocities: &[Vec3], dt: Real) {
        debug_assert_eq!(
            node_velocities.len(),
            self.mesh.vert_count(),
            "g2p: node-velocity length mismatch"
        );
        for particle in &mut self.particles {
            let Some((tet_verts, weights, positions)) = locate_particle_tet(&self.mesh, particle.x)
            else {
                // Out-of-mesh particle: leave v / C alone, just advect.
                particle.x += dt * particle.v;
                continue;
            };

            // New velocity: weighted sum of node velocities.
            let mut v_new = Vec3::ZERO;
            for k in 0..4 {
                v_new += weights[k] * node_velocities[tet_verts[k] as usize];
            }

            // New affine matrix.
            //
            // For P1 (linear) shape functions on a tet, the standard APIC
            // formula reduces to
            //
            //   C_p = Σ_i v_i · ∇N_i(x_p)ᵀ
            //
            // because ∇N_i is exactly the row of D⁻ᵀ (where D is the
            // tet edge-frame matrix) and `Σ_i N_i (x_i − x_p) = 0`. We
            // compute ∇N via the same closed form the FEM
            // element-stiffness routine uses — no D⁻¹ second-moment
            // matrix needed. (See Hu et al. 2018, "MLS-MPM"; the P1
            // case collapses to MLS-MPM with linear basis.)
            let grads = tet_p1_shape_gradients(&positions);
            let mut c_new = Mat3::ZERO;
            for k in 0..4 {
                let v_k = node_velocities[tet_verts[k] as usize];
                // Outer product v_k ⊗ ∇N_k.
                c_new += outer_product(v_k, grads[k]);
            }
            particle.v = v_new;
            particle.c = c_new;
            particle.x += dt * v_new;
        }
    }

    /// Run one full hybrid step: P2G → mesh velocity update (gravity +
    /// viscous Laplacian smoothing + Dirichlet BCs) → G2P → advect.
    ///
    /// `viscosity` is the kinematic viscosity (m²/s); larger values
    /// dissipate more energy per step. For the falling-particle contract
    /// test we pick `viscosity = 0.05` which gives clean monotone KE
    /// decay without over-damping.
    ///
    /// `bcs` is a list of [`BoundaryCondition::Dirichlet`] (or
    /// [`BoundaryCondition::DirichletVector`]) entries. Vector BCs pin
    /// every component; scalar `Dirichlet { value: 0.0 }` is treated as
    /// no-slip on all components.
    pub fn step(&mut self, dt: Real, viscosity: Real, bcs: &[BoundaryCondition]) {
        self.step_projected(dt, viscosity, bcs, 0);
    }

    /// Stage 37.6.1 (INSPECTION.md F12) — full step with a P1
    /// pressure-Poisson projection after the grid update.
    ///
    /// `pressure_iters = 0` reproduces [`Self::step`] exactly (viscous
    /// smoothing only). With `pressure_iters > 0`, after gravity +
    /// smoothing + Dirichlet BCs the nodal field is projected toward a
    /// divergence-free state on the FLUID region (tets containing at
    /// least one particle; non-fluid nodes act as a `p = 0` free
    /// surface, which also pins the pressure nullspace):
    ///
    /// ```text
    ///   ∇²p = (1/dt) ∇·v        (weak form, P1, fluid tets only)
    ///   v  ← v − dt ∇p          (volume-weighted nodal gradient)
    /// ```
    ///
    /// solved matrix-free with `pressure_iters` Jacobi sweeps. A
    /// uniform (divergence-free) field has `rhs ≡ 0 → p ≡ 0` and is
    /// passed through untouched, so the momentum-roundtrip and
    /// uniform-motion contracts survive with projection enabled.
    pub fn step_projected(
        &mut self,
        dt: Real,
        viscosity: Real,
        bcs: &[BoundaryCondition],
        pressure_iters: u32,
    ) {
        let (node_mass, node_momentum) = self.p2g();
        let n = self.mesh.vert_count();

        // Recover node velocities from momentum / mass.
        let mut node_velocity = vec![Vec3::ZERO; n];
        for i in 0..n {
            if node_mass[i] > Real::EPSILON {
                node_velocity[i] = node_momentum[i] / node_mass[i];
            }
        }

        // Add gravity to every node that holds mass (no point pushing
        // empty cells around).
        for i in 0..n {
            if node_mass[i] > Real::EPSILON {
                node_velocity[i] += dt * self.gravity;
            }
        }

        // Viscous Laplacian smoothing — one explicit Jacobi sweep.
        // Stable when `dt · viscosity · max-degree ≤ 1`, which the
        // contract tests respect (dt = 1/60, ν = 0.05, max-deg ≈ 24
        // for a Kuhn-tet grid → Courant ≈ 0.02).
        if viscosity > 0.0 {
            apply_viscous_smoothing(&self.mesh, &mut node_velocity, &node_mass, viscosity, dt);
        }

        // Dirichlet BCs — pin the boundary velocity field. Done AFTER
        // the smoothing pass so the boundary values are exact. The
        // scalar `Dirichlet` variant is interpreted as no-slip ONLY
        // (zero velocity); a non-zero scalar would have to mean "set
        // every component to that scalar" which is almost never what
        // the user intends — they should reach for `DirichletVector`
        // to pin a non-zero per-component velocity instead.
        for bc in bcs {
            match bc {
                BoundaryCondition::Dirichlet { vertex, value } => {
                    debug_assert_eq!(
                        *value, 0.0,
                        "APIC scalar Dirichlet on vertex {vertex} must be zero \
                         (no-slip); use DirichletVector for non-zero per-component \
                         velocity",
                    );
                    let idx = *vertex as usize;
                    if idx < n {
                        node_velocity[idx] = Vec3::ZERO;
                    }
                }
                BoundaryCondition::DirichletVector { vertex, values } => {
                    let idx = *vertex as usize;
                    if idx < n && values.len() >= 3 {
                        node_velocity[idx] = Vec3::new(values[0], values[1], values[2]);
                    }
                }
                _ => {}
            }
        }

        if pressure_iters > 0 && dt > 0.0 {
            apply_pressure_projection(
                &self.mesh,
                &self.particles,
                &mut node_velocity,
                pressure_iters,
            );
            // Re-pin no-slip walls: the correction can re-introduce a
            // small wall-normal component on boundary nodes.
            for bc in bcs {
                match bc {
                    BoundaryCondition::Dirichlet { vertex, .. } => {
                        let idx = *vertex as usize;
                        if idx < n {
                            node_velocity[idx] = Vec3::ZERO;
                        }
                    }
                    BoundaryCondition::DirichletVector { vertex, values } => {
                        let idx = *vertex as usize;
                        if idx < n && values.len() >= 3 {
                            node_velocity[idx] = Vec3::new(values[0], values[1], values[2]);
                        }
                    }
                    _ => {}
                }
            }
        }

        self.g2p(&node_velocity, dt);
    }

    /// Stage 37.6.1 — project a nodal velocity field toward
    /// divergence-free on the fluid region (see
    /// [`Self::step_projected`]). Public for tests, diagnostics, and
    /// the GPU parity oracle.
    pub fn project_pressure(&self, node_velocity: &mut [Vec3], iters: u32) {
        apply_pressure_projection(&self.mesh, &self.particles, node_velocity, iters);
    }

    /// Convenience helper for the v1 viewer scene: clamp every particle
    /// position into the AABB `[lo, hi]`. Called after [`Self::step`] in
    /// the contract test + viewer to keep particles honest when the
    /// explicit advect overshoots a no-slip wall by a fraction of a
    /// timestep.
    pub fn clamp_to_aabb(&mut self, lo: Vec3, hi: Vec3) {
        for p in &mut self.particles {
            p.x = p.x.max(lo).min(hi);
        }
    }
}

/// Build a fresh [`ApicHybrid`] error if the mesh has no tets.
///
/// Convenience for callers who want validation up front rather than
/// silent particle-skip behaviour later.
pub fn validate_hybrid(hybrid: &ApicHybrid) -> Result<()> {
    let has_tet = hybrid
        .mesh
        .cells
        .iter()
        .any(|c| matches!(c, FemCell::Tet(_)));
    if !has_tet {
        return Err(Error::Fem(
            "ApicHybrid requires at least one tet cell".to_string(),
        ));
    }
    Ok(())
}

// ─── Internal helpers ───────────────────────────────────────────────

/// Naive O(P · T) particle-in-tet test using barycentric coordinates.
///
/// Returns `(tet_vertex_indices, weights, vertex_positions)` for the
/// first tet whose four barycentric weights are all in `[−eps, 1+eps]`,
/// or `None` if the particle lies outside every tet in the mesh.
///
/// **v1 perf caveat**: a spatial hash (Stage 37.6.1) replaces this
/// with O(P) lookup; for now the brute-force scan is fine at the
/// contract-test scale (≤ 200 particles, ≤ a few hundred tets ⇒
/// ≤ ~50k inner iterations per substep).
pub fn locate_particle_tet(mesh: &FemMesh, x: Vec3) -> Option<([u32; 4], [Real; 4], [Vec3; 4])> {
    const EPS: Real = 1.0e-9;
    for cell in &mesh.cells {
        let FemCell::Tet(verts) = cell else { continue };
        let p = [
            mesh.vertices[verts[0] as usize],
            mesh.vertices[verts[1] as usize],
            mesh.vertices[verts[2] as usize],
            mesh.vertices[verts[3] as usize],
        ];
        let Some(weights) = barycentric_tet(&p, x) else {
            continue;
        };
        if weights.iter().all(|w| *w >= -EPS && *w <= 1.0 + EPS) {
            return Some((*verts, weights, p));
        }
    }
    None
}

/// Barycentric coordinates of `x` w.r.t. tet `p`. Returns `None` if
/// the tet is degenerate (zero volume).
///
/// Uses the standard Cramer-rule expansion:
///
/// ```text
///   D = [p₁−p₀ | p₂−p₀ | p₃−p₀]
///   (λ₁, λ₂, λ₃) = D⁻¹ · (x − p₀)
///   λ₀ = 1 − λ₁ − λ₂ − λ₃
/// ```
fn barycentric_tet(p: &[Vec3; 4], x: Vec3) -> Option<[Real; 4]> {
    let d = Mat3::from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0]);
    let det = d.determinant();
    if det.abs() < 1.0e-18 {
        return None;
    }
    let inv = d.inverse();
    let lam = inv * (x - p[0]);
    Some([1.0 - lam.x - lam.y - lam.z, lam.x, lam.y, lam.z])
}

/// Gradients of the four P1 hat-functions on a tet, returned as 3D
/// vectors. Mirrors `super::element::tet_shape_grads` (which is
/// `pub(super)` and not reachable from this module).
fn tet_p1_shape_gradients(p: &[Vec3; 4]) -> [Vec3; 4] {
    let d = Mat3::from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0]);
    let inv_t = d.inverse().transpose();
    let g1 = inv_t.x_axis;
    let g2 = inv_t.y_axis;
    let g3 = inv_t.z_axis;
    let g0 = -(g1 + g2 + g3);
    [g0, g1, g2, g3]
}

/// Outer product `a · bᵀ` as a 3×3 matrix.
fn outer_product(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(b.x * a, b.y * a, b.z * a)
}

/// One explicit Jacobi sweep of the **mass-weighted** Laplacian
/// smoothing operator on the node-velocity field:
///
/// ```text
///   v_i ← v_i + dt · ν · (v̄_i − v_i)
/// ```
///
/// where `v̄_i` is the mass-weighted average of all nodes that share
/// at least one tet with node `i`. Nodes with zero mass are skipped
/// (no information to spread). Nodes whose neighbourhood has zero
/// total mass are also skipped (boundary nodes outside particle
/// support).
///
/// Stable when `dt · ν ≤ 1` (the operator is a convex combination at
/// that point); for the contract tests we pick `ν = 0.05` and
/// `dt = 1/60` so the criterion is comfortably satisfied.
fn apply_viscous_smoothing(
    mesh: &FemMesh,
    velocity: &mut [Vec3],
    node_mass: &[Real],
    viscosity: Real,
    dt: Real,
) {
    let n = mesh.vert_count();
    debug_assert_eq!(velocity.len(), n);
    debug_assert_eq!(node_mass.len(), n);

    // Build a per-node neighbour list once. For a v1 CPU implementation
    // re-walking the cells per smoothing call is fine (~1 µs / 1k tets);
    // a future GPU port would precompute this once at mesh-build time.
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); n];
    for cell in &mesh.cells {
        if let FemCell::Tet(v) = cell {
            for &a in v.iter() {
                for &b in v.iter() {
                    if a != b && !neighbours[a as usize].contains(&b) {
                        neighbours[a as usize].push(b);
                    }
                }
            }
        }
    }

    let mut new_velocity = velocity.to_vec();
    for i in 0..n {
        if node_mass[i] <= Real::EPSILON {
            continue;
        }
        let mut total_mass = 0.0;
        let mut weighted_v = Vec3::ZERO;
        for &j in &neighbours[i] {
            let m = node_mass[j as usize];
            if m > Real::EPSILON {
                total_mass += m;
                weighted_v += m * velocity[j as usize];
            }
        }
        if total_mass <= Real::EPSILON {
            continue;
        }
        let v_avg = weighted_v / total_mass;
        let damping = (dt * viscosity).min(1.0);
        new_velocity[i] = velocity[i] + damping * (v_avg - velocity[i]);
    }
    velocity.copy_from_slice(&new_velocity);
}

/// Stage 37.6.1 — matrix-free P1 pressure projection (see
/// [`ApicHybrid::step_projected`]). Weak-form Poisson on fluid tets:
/// with `q ≡ dt·p` so no timestep appears in the system:
/// `Σ_t vol_t (∇φi·∇q) = −Σ_t (vol_t/4) ∇·v`, Jacobi-iterated, then
/// `v ← v − ∇q` with volume-weighted nodal averaging of the per-tet
/// (constant) gradient.
fn apply_pressure_projection(
    mesh: &FemMesh,
    particles: &[ApicParticle],
    node_velocity: &mut [Vec3],
    iters: u32,
) {
    let n = mesh.vertices.len();
    // Per-tet cache: indices, shape grads, signed-positive volume.
    let mut tets: Vec<([usize; 4], [Vec3; 4], Real)> = Vec::new();
    let mut cell_indices: Vec<usize> = Vec::new();
    for (c, cell) in mesh.cells.iter().enumerate() {
        let FemCell::Tet(idx) = cell else { continue };
        let p = [
            mesh.vertices[idx[0] as usize],
            mesh.vertices[idx[1] as usize],
            mesh.vertices[idx[2] as usize],
            mesh.vertices[idx[3] as usize],
        ];
        let vol = ((p[1] - p[0]).cross(p[2] - p[0]).dot(p[3] - p[0]) / 6.0).abs();
        if vol <= Real::EPSILON {
            continue;
        }
        let grads = tet_p1_shape_gradients(&p);
        tets.push((
            [
                idx[0] as usize,
                idx[1] as usize,
                idx[2] as usize,
                idx[3] as usize,
            ],
            grads,
            vol,
        ));
        cell_indices.push(c);
    }
    // Deposit-supported fluid mask, shared with the divergence probes
    // (see `first_match_fluid_mask` for the face-sitting-particle
    // rationale). `cell_of[t]` maps the dense tet list back to cell
    // indices.
    let cell_mask = first_match_fluid_mask(mesh, particles);
    let fluid: Vec<bool> = cell_indices.iter().map(|&c| cell_mask[c]).collect();
    // Lumped node mass (volume weights) over fluid tets only.
    let mut lumped = vec![0.0 as Real; n];
    let mut any_fluid = false;
    for (t, (verts, _, vol)) in tets.iter().enumerate() {
        if !fluid[t] {
            continue;
        }
        any_fluid = true;
        for j in 0..4 {
            lumped[verts[j]] += vol / 4.0;
        }
    }
    if !any_fluid {
        return;
    }

    // Per-TET (P0) pressure projection through the lumped-mass Schur
    // operator `S = D M⁻¹ Dᵀ` with `D_t(v) = vol_t Σ_j ∇φ_j·v_j`:
    // after solving `S q = D v`, the correction `v ← v − M⁻¹ Dᵀ q`
    // makes the per-tet divergence of the corrected field EXACTLY the
    // solver residual — unlike equal-order P1-P1 node pressure, whose
    // lumped correction can only cancel nodal moments (and whose
    // Richardson outer loop amplifies LBB checkerboard modes). S is
    // symmetric positive semi-definite, so plain conjugate gradients
    // converges without preconditioning at contract scale.
    let apply_s = |q: &[Real], out: &mut [Real]| {
        let mut f = vec![Vec3::ZERO; n];
        for (t, (verts, grads, vol)) in tets.iter().enumerate() {
            if !fluid[t] {
                continue;
            }
            for j in 0..4 {
                f[verts[j]] += *vol * grads[j] * q[t];
            }
        }
        for (i, fi) in f.iter_mut().enumerate() {
            if lumped[i] > Real::EPSILON {
                *fi /= lumped[i];
            } else {
                *fi = Vec3::ZERO;
            }
        }
        for (t, (verts, grads, vol)) in tets.iter().enumerate() {
            out[t] = if fluid[t] {
                *vol * (0..4).map(|j| f[verts[j]].dot(grads[j])).sum::<Real>()
            } else {
                0.0
            };
        }
    };

    let n_t = tets.len();
    let mut rhs = vec![0.0 as Real; n_t];
    for (t, (verts, grads, vol)) in tets.iter().enumerate() {
        if fluid[t] {
            rhs[t] = *vol
                * (0..4)
                    .map(|j| node_velocity[verts[j]].dot(grads[j]))
                    .sum::<Real>();
        }
    }

    // Matrix-free CG on S q = rhs.
    let mut q = vec![0.0 as Real; n_t];
    let mut r = rhs.clone();
    let mut d = r.clone();
    let mut s_d = vec![0.0 as Real; n_t];
    let mut rr: Real = r.iter().map(|x| x * x).sum();
    let rr0 = rr;
    for _ in 0..iters {
        if rr <= rr0 * 1.0e-24 {
            break;
        }
        apply_s(&d, &mut s_d);
        let d_sd: Real = d.iter().zip(s_d.iter()).map(|(a, b)| a * b).sum();
        if d_sd.abs() <= Real::EPSILON {
            break;
        }
        let alpha = rr / d_sd;
        for t in 0..n_t {
            q[t] += alpha * d[t];
            r[t] -= alpha * s_d[t];
        }
        let rr_new: Real = r.iter().map(|x| x * x).sum();
        let beta = rr_new / rr;
        rr = rr_new;
        for t in 0..n_t {
            d[t] = r[t] + beta * d[t];
        }
    }

    // Correction: v ← v − M⁻¹ Dᵀ q.
    let mut f = vec![Vec3::ZERO; n];
    for (t, (verts, grads, vol)) in tets.iter().enumerate() {
        if !fluid[t] {
            continue;
        }
        for j in 0..4 {
            f[verts[j]] += *vol * grads[j] * q[t];
        }
    }
    for i in 0..n {
        if lumped[i] > Real::EPSILON {
            node_velocity[i] -= f[i] / lumped[i];
        }
    }
}

/// First-match fluid-tet mask over `mesh.cells` order — the tet each
/// particle actually deposits into (`locate_particle_tet` semantics),
/// AND with all four nodes mass-supported. A face-sitting particle
/// deposits ZERO weight to its tet's off-face node; that node carries
/// no mass, so its velocity slot is not real fluid data and any tet
/// using it must not enter the divergence constraint. Shared by the
/// projection and both divergence probes so mask, P2G, and metric can
/// never disagree.
pub fn first_match_fluid_mask(mesh: &FemMesh, particles: &[ApicParticle]) -> Vec<bool> {
    let n = mesh.vertices.len();
    let mut fluid = vec![false; mesh.cells.len()];
    let mut node_mass = vec![0.0 as Real; n];
    for particle in particles {
        if let Some((verts, weights, _)) = locate_particle_tet(mesh, particle.x) {
            for j in 0..4 {
                node_mass[verts[j] as usize] += weights[j] * particle.mass;
            }
        }
        for (c, cell) in mesh.cells.iter().enumerate() {
            let FemCell::Tet(idx) = cell else { continue };
            let p = [
                mesh.vertices[idx[0] as usize],
                mesh.vertices[idx[1] as usize],
                mesh.vertices[idx[2] as usize],
                mesh.vertices[idx[3] as usize],
            ];
            if let Some(w) = barycentric_tet(&p, particle.x) {
                // Containment: `barycentric_tet` is the raw coordinate
                // transform (Some for ANY point vs a non-degenerate
                // tet) — the range check below is what makes it a
                // containment test, mirroring `locate_particle_tet`.
                if w.iter().all(|w| *w >= -1.0e-9 && *w <= 1.0 + 1.0e-9) {
                    fluid[c] = true;
                    break;
                }
            }
        }
    }
    for (c, cell) in mesh.cells.iter().enumerate() {
        if !fluid[c] {
            continue;
        }
        let FemCell::Tet(idx) = cell else { continue };
        if !(0..4).all(|j| node_mass[idx[j] as usize] > Real::EPSILON) {
            fluid[c] = false;
        }
    }
    fluid
}

/// Test/diagnostic probe — the WEAK (P1-moment) divergence residual
/// the projection actually controls: `r_i = Σ_{fluid t ∋ i}
/// (vol_t/4)·(∇·v)_t` per fluid node, returned as the RMS over fluid
/// nodes. Equal-order P1-P1 projection cannot drive the per-tet L2
/// divergence to zero (LBB checkerboard modes survive); this moment
/// residual is the honest convergence metric.
pub fn fluid_weak_divergence_norm(
    mesh: &FemMesh,
    particles: &[ApicParticle],
    node_velocity: &[Vec3],
) -> Real {
    let n = mesh.vertices.len();
    let fluid = first_match_fluid_mask(mesh, particles);
    let mut residual = vec![0.0 as Real; n];
    let mut fluid_node = vec![false; n];
    for (c, cell) in mesh.cells.iter().enumerate() {
        let FemCell::Tet(idx) = cell else { continue };
        let p = [
            mesh.vertices[idx[0] as usize],
            mesh.vertices[idx[1] as usize],
            mesh.vertices[idx[2] as usize],
            mesh.vertices[idx[3] as usize],
        ];
        let vol = ((p[1] - p[0]).cross(p[2] - p[0]).dot(p[3] - p[0]) / 6.0).abs();
        if vol <= Real::EPSILON {
            continue;
        }
        if !fluid[c] {
            continue;
        }
        let grads = tet_p1_shape_gradients(&p);
        let div: Real = (0..4)
            .map(|j| node_velocity[idx[j] as usize].dot(grads[j]))
            .sum();
        for j in 0..4 {
            residual[idx[j] as usize] += (vol / 4.0) * div;
            fluid_node[idx[j] as usize] = true;
        }
    }
    let mut acc = 0.0 as Real;
    let mut count = 0usize;
    for i in 0..n {
        if fluid_node[i] {
            acc += residual[i] * residual[i];
            count += 1;
        }
    }
    if count > 0 {
        (acc / count as Real).sqrt()
    } else {
        0.0
    }
}

/// Test/diagnostic probe — the volume-weighted L2 norm of the P1
/// divergence over FLUID tets (those containing ≥ 1 particle).
pub fn fluid_divergence_norm(
    mesh: &FemMesh,
    particles: &[ApicParticle],
    node_velocity: &[Vec3],
) -> Real {
    let fluid = first_match_fluid_mask(mesh, particles);
    let mut acc = 0.0 as Real;
    let mut vol_acc = 0.0 as Real;
    for (c, cell) in mesh.cells.iter().enumerate() {
        let FemCell::Tet(idx) = cell else { continue };
        let p = [
            mesh.vertices[idx[0] as usize],
            mesh.vertices[idx[1] as usize],
            mesh.vertices[idx[2] as usize],
            mesh.vertices[idx[3] as usize],
        ];
        let vol = ((p[1] - p[0]).cross(p[2] - p[0]).dot(p[3] - p[0]) / 6.0).abs();
        if vol <= Real::EPSILON {
            continue;
        }
        if !fluid[c] {
            continue;
        }
        let grads = tet_p1_shape_gradients(&p);
        let div: Real = (0..4)
            .map(|j| node_velocity[idx[j] as usize].dot(grads[j]))
            .sum();
        acc += vol * div * div;
        vol_acc += vol;
    }
    if vol_acc > Real::EPSILON {
        (acc / vol_acc).sqrt()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_tet_mesh() -> FemMesh {
        FemMesh {
            vertices: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            cells: vec![FemCell::Tet([0, 1, 2, 3])],
        }
    }

    #[test]
    fn barycentric_centroid_is_quarter() {
        let mesh = unit_tet_mesh();
        let p = [
            mesh.vertices[0],
            mesh.vertices[1],
            mesh.vertices[2],
            mesh.vertices[3],
        ];
        let centroid = (p[0] + p[1] + p[2] + p[3]) / 4.0;
        let w = barycentric_tet(&p, centroid).unwrap();
        for v in w {
            assert!((v - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn barycentric_outside_tet_has_negative() {
        let mesh = unit_tet_mesh();
        let p = [
            mesh.vertices[0],
            mesh.vertices[1],
            mesh.vertices[2],
            mesh.vertices[3],
        ];
        let outside = Vec3::new(2.0, 2.0, 2.0);
        let w = barycentric_tet(&p, outside).unwrap();
        assert!(w.iter().any(|v| *v < 0.0 || *v > 1.0));
    }

    #[test]
    fn locate_particle_skips_external() {
        let mesh = unit_tet_mesh();
        let inside = Vec3::new(0.1, 0.1, 0.1);
        let outside = Vec3::new(2.0, 2.0, 2.0);
        assert!(locate_particle_tet(&mesh, inside).is_some());
        assert!(locate_particle_tet(&mesh, outside).is_none());
    }
}
