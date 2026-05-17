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
//! - **No real Stokes projection (v1).** The viscous-smoothing
//!   placeholder is incompressible only in the limit `ν · dt → ∞`. A
//!   true pressure-Poisson on the P1 mesh is Stage 37.6.1.
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

        self.g2p(&node_velocity, dt);
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
