//! Hydroelastic contact model — Stage 22.1.
//!
//! ## Problem
//!
//! Point contacts (one normal + one tangent vector at a single
//! contact point) work fine for sphere-on-sphere or capsule-on-plane
//! collisions, but they're noisy for **patch contacts**: a flat
//! cloth resting on a sphere, two boxes stacked face-to-face, a
//! finger pressed against a soft ball. The contact "point" jitters
//! between the actual patch's vertices each frame, which the solver
//! then has to damp out. Each broad-phase step picks a slightly
//! different vertex; the integrator sees a jumping force vector;
//! the system ends up at a local minimum but never quite at rest.
//!
//! ## Hydroelastic answer
//!
//! Drake's **hydroelastic contact model** (Elandt, Drumwright,
//! Sherman, Kuchenbecker 2019) replaces the point contact with a
//! **distributed pressure field over the intersection volume** of
//! the two bodies. The pressure inside a body is its "hydroelastic
//! pressure field" — typically a linear function of depth into the
//! body, like `p(x) = E · |sdf(x)|` for `x` inside the body.
//!
//! The contact force is the integral of pressure over the
//! intersection surface (the iso-surface where one body's pressure
//! field crosses the other's). For a soft-on-rigid pair, the rigid
//! body's surface IS the intersection surface — the integration
//! becomes a surface integral over (the part of) the rigid body
//! that's inside the soft body's pressure field.
//!
//! ## v1 scope (this stage)
//!
//! - **Soft-on-rigid only.** The soft body carries an
//!   [`crate::geometry::SdfGrid`] pressure field; the rigid body is
//!   sampled at a discrete set of surface points. For each sample
//!   inside the soft body we evaluate the hydroelastic pressure,
//!   multiply by the sample area, and accumulate a per-pair contact
//!   wrench.
//! - **Soft-on-soft is deferred.** Both volumes overlap so the
//!   intersection is genuinely 3D (a bounded region in space, not
//!   a 2D surface). Integrating a 3D field over an arbitrary
//!   intersection requires either (a) running marching cubes on
//!   `min(p_a - p_b, 0)` to find the iso-surface and integrating
//!   there, or (b) a tetrahedralised volume integration. Both add
//!   substantial work; PLAN.md Stage 22.1 lists this as the v1
//!   carveout. See [`HydroelasticPolicy::SoftOnRigid`] for the
//!   surfaced enum.
//! - **Linear hydroelastic.** Pressure inside a body is
//!   `E · max(0, -sdf)`. `E` (the "hydroelastic modulus", units Pa
//!   = N/m²) is a per-pair stiffness knob. Drake calls this the
//!   "elastic modulus".
//! - **Per-sample friction.** Each sample point carries a contact
//!   id used by the Stage 20.1 [`crate::kernels::DahlFrictionKernel`]
//!   to track persistent bristle state across substeps.
//!
//! ## How a caller uses this
//!
//! ```ignore
//! use molla::geometry::{bake_mesh_to_sdf, hydroelastic::*};
//!
//! // 1. Pre-bake the soft body's SDF (typically at scene-build time).
//! let pressure_field = bake_mesh_to_sdf(&soft_mesh, 32, 0.05);
//!
//! // 2. Build a sampler for the rigid body.
//! let samples = sample_sphere_surface(rigid_radius, 64);
//!
//! // 3. Each substep: compute the contact manifold.
//! let cfg = HydroelasticConfig { hydroelastic_modulus: 1.0e5, ..Default::default() };
//! let manifold = compute_hydroelastic_contact(
//!     &pressure_field,
//!     &samples,
//!     soft_pose,
//!     rigid_pose,
//!     &cfg,
//! );
//!
//! // 4. Feed the manifold into the integrator + Dahl friction kernel.
//! ```
//!
//! See `crates/molla/shaders/hydroelastic.wgsl` for the GPU
//! counterpart (used by [`HydroelasticKernel`] when the `gpu`
//! feature is on).

use crate::math::{Real, Transform, Vec3};

use super::sdf_bake::SdfGrid;

/// Knobs on the hydroelastic contact model.
#[derive(Copy, Clone, Debug)]
pub struct HydroelasticConfig {
    /// Hydroelastic modulus `E` (Pa = N/m²). Pressure inside the
    /// soft body is `E · max(0, -sdf)`. Larger = stiffer body. A
    /// reasonable order-of-magnitude is `1e4` to `1e7` Pa for
    /// rubber-like materials; `1e9+` for hard plastic.
    pub hydroelastic_modulus: Real,
    /// Hard upper bound on per-sample pressure (Pa). Acts as a
    /// safety cap when a sample falls deep inside the soft body's
    /// SDF (e.g. due to large penetration after a missed collision
    /// detection step). `Real::INFINITY` disables the cap.
    pub max_pressure: Real,
    /// Default Coulomb friction coefficient applied to every sample
    /// in the manifold. Per-shape overrides are applied by the
    /// caller before/after calling into the hydroelastic kernel.
    pub friction_coefficient: Real,
    /// Skip samples that are this far outside the SDF (m). Avoids
    /// wasting integration work on samples that obviously won't
    /// contribute. `0.0` keeps every sample (still cheap if `sdf > 0`
    /// short-circuits the pressure to zero).
    pub culling_distance: Real,
}

impl Default for HydroelasticConfig {
    fn default() -> Self {
        Self {
            hydroelastic_modulus: 1.0e5,
            max_pressure: Real::INFINITY,
            friction_coefficient: 0.4,
            culling_distance: 0.01,
        }
    }
}

/// Selects the contact pair geometry. v1 ships only `SoftOnRigid`;
/// the variant exists so future support can land without a public
/// API break.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HydroelasticPolicy {
    /// Soft body's `SdfGrid` carries the pressure field; the rigid
    /// body is sampled at discrete surface points. v1 scope.
    SoftOnRigid,
    /// Both bodies' pressure fields overlap; intersection volume is
    /// genuinely 3D. **Not implemented in v1** — will return an
    /// empty manifold with a `tracing::warn`.
    SoftOnSoft,
}

/// One sample on the rigid body's surface, expressed in the rigid
/// body's local frame. The sample-area weight makes the manifold a
/// quadrature rule — sum over samples is an approximation of the
/// surface integral.
#[derive(Copy, Clone, Debug)]
pub struct SurfaceSample {
    /// Sample position in rigid body's local frame (m).
    pub position: Vec3,
    /// Outward-facing surface normal in rigid body's local frame.
    pub normal: Vec3,
    /// Quadrature weight (m²). Sum over a closed surface ≈ surface area.
    pub area: Real,
}

/// One contact site in the hydroelastic manifold. Per-substep
/// output of [`compute_hydroelastic_contact`].
#[derive(Copy, Clone, Debug)]
pub struct HydroelasticContact {
    /// World-frame contact point (the rigid sample's world position).
    pub point: Vec3,
    /// World-frame outward normal (points out of the soft body
    /// into the rigid body — i.e. opposite of the SDF gradient).
    pub normal: Vec3,
    /// Hydroelastic pressure at the sample (Pa = N/m²).
    pub pressure: Real,
    /// Sample area (m²). Multiply with `pressure` for normal force.
    pub area: Real,
    /// Stable id used by [`crate::kernels::DahlFrictionKernel`] to
    /// preserve bristle state across substeps. Assigned by the host:
    /// typically `(rigid_shape_id, sample_index)` packed into a u32.
    pub contact_id: u32,
}

impl HydroelasticContact {
    /// Per-sample normal force magnitude (N).
    pub fn normal_force(&self) -> Real {
        self.pressure * self.area
    }
}

/// Output manifold from one hydroelastic contact pair. Empty when
/// no penetration was detected.
#[derive(Default, Debug)]
pub struct HydroelasticManifold {
    pub contacts: Vec<HydroelasticContact>,
    /// Total normal-force magnitude over every contact in the
    /// manifold (N). Useful for sanity tests against `mass · g`.
    pub total_normal_force: Real,
    /// World-frame net contact wrench: linear part is the
    /// integrated force vector, angular part isn't computed in v1
    /// (caller can integrate `r × f` over `contacts` if needed).
    pub net_force: Vec3,
}

/// Compute the hydroelastic contact manifold between a soft body
/// (pressure field `soft_sdf`) and a rigid body sampled at
/// `rigid_samples`. Returns an empty manifold when no penetration
/// is detected.
///
/// ## Algorithm (v1 — soft-on-rigid)
///
/// 1. For each rigid surface sample, transform the sample's local
///    position into the soft body's local frame.
/// 2. Trilinearly sample the soft body's SDF at that point.
/// 3. If `sdf < 0` (inside the soft body), compute pressure
///    `p = E · (-sdf)`, clamped to `max_pressure`.
/// 4. The contact normal points OUT of the soft body — that's the
///    SDF gradient (computed via central differences). The world-
///    frame normal is then transformed back into world space.
/// 5. Assign a stable contact id and emit a [`HydroelasticContact`].
///
/// The output manifold's `total_normal_force` equals the integral
/// of `pressure · area` over every penetrating sample, giving a
/// quadrature approximation to the surface integral.
pub fn compute_hydroelastic_contact(
    soft_sdf: &SdfGrid,
    rigid_samples: &[SurfaceSample],
    soft_pose: Transform,
    rigid_pose: Transform,
    cfg: &HydroelasticConfig,
) -> HydroelasticManifold {
    let mut manifold = HydroelasticManifold::default();
    let soft_inv = soft_pose.inverse();
    for (i, sample) in rigid_samples.iter().enumerate() {
        // World-frame sample position.
        let world_p = rigid_pose.transform_point(sample.position);
        // Soft body's local frame.
        let soft_local = soft_inv.transform_point(world_p);
        let sdf_value = sample_sdf_trilinear(soft_sdf, soft_local);
        // Skip cleanly outside.
        if sdf_value > cfg.culling_distance {
            continue;
        }
        if sdf_value >= 0.0 {
            // Boundary or just-outside — no pressure but no point in
            // emitting a contact either.
            continue;
        }
        let pen = -sdf_value;
        let pressure = (cfg.hydroelastic_modulus * pen).min(cfg.max_pressure);
        let grad_local = sample_sdf_gradient(soft_sdf, soft_local);
        // Outward normal of the soft body = +∇sdf (sdf increases
        // outward). The contact normal as seen by the rigid body
        // points the OPPOSITE direction — out of the rigid body
        // back into the soft body's interior. We flip to follow the
        // molla convention (normal points from B into A; here A is
        // rigid, so it points out of soft into rigid).
        let world_n = if grad_local.length_squared() > 1.0e-24 {
            soft_pose.rotation * grad_local.normalize()
        } else {
            // Fall back to the rigid sample's surface normal.
            rigid_pose.rotation * sample.normal
        };
        let force_mag = pressure * sample.area;
        manifold.total_normal_force += force_mag;
        manifold.net_force += world_n * force_mag;
        manifold.contacts.push(HydroelasticContact {
            point: world_p,
            normal: world_n,
            pressure,
            area: sample.area,
            contact_id: i as u32,
        });
    }
    manifold
}

/// Trilinear sample of an [`SdfGrid`] at world-coords `p` (in the
/// grid's coordinate system). Out-of-bounds returns `+∞` (treated
/// as "deep outside").
pub fn sample_sdf_trilinear(grid: &SdfGrid, p: Vec3) -> Real {
    let cell = grid.cell_size;
    let inv = 1.0 / cell;
    let local = (p - grid.origin) * inv;
    if local.x < 0.0 || local.y < 0.0 || local.z < 0.0 {
        return Real::INFINITY;
    }
    let xi = local.x.floor() as i64;
    let yi = local.y.floor() as i64;
    let zi = local.z.floor() as i64;
    let nx = grid.nx as i64;
    let ny = grid.ny as i64;
    let nz = grid.nz as i64;
    if xi >= nx - 1 || yi >= ny - 1 || zi >= nz - 1 {
        return Real::INFINITY;
    }
    let tx = local.x - xi as Real;
    let ty = local.y - yi as Real;
    let tz = local.z - zi as Real;
    let g = |x: i64, y: i64, z: i64| -> Real {
        grid.at(x as u32, y as u32, z as u32)
    };
    let c000 = g(xi, yi, zi);
    let c100 = g(xi + 1, yi, zi);
    let c010 = g(xi, yi + 1, zi);
    let c110 = g(xi + 1, yi + 1, zi);
    let c001 = g(xi, yi, zi + 1);
    let c101 = g(xi + 1, yi, zi + 1);
    let c011 = g(xi, yi + 1, zi + 1);
    let c111 = g(xi + 1, yi + 1, zi + 1);
    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;
    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;
    c0 * (1.0 - tz) + c1 * tz
}

/// Central-difference gradient of an [`SdfGrid`] at point `p`. Used
/// to compute the contact normal (which points along ∇sdf).
pub fn sample_sdf_gradient(grid: &SdfGrid, p: Vec3) -> Vec3 {
    let h = grid.cell_size;
    let dx = sample_sdf_trilinear(grid, p + Vec3::new(h, 0.0, 0.0))
        - sample_sdf_trilinear(grid, p - Vec3::new(h, 0.0, 0.0));
    let dy = sample_sdf_trilinear(grid, p + Vec3::new(0.0, h, 0.0))
        - sample_sdf_trilinear(grid, p - Vec3::new(0.0, h, 0.0));
    let dz = sample_sdf_trilinear(grid, p + Vec3::new(0.0, 0.0, h))
        - sample_sdf_trilinear(grid, p - Vec3::new(0.0, 0.0, h));
    let v = Vec3::new(dx, dy, dz);
    if v.length_squared() > 1.0e-24 {
        v
    } else {
        Vec3::Y
    }
}

/// Generate a Fibonacci-spiral set of points on a unit sphere of
/// radius `r`, with associated quadrature weights so summing them
/// approximates the surface integral. Matches Drake's "uniform
/// sphere mesh" sampling for simple analytic rigid bodies.
///
/// Each sample's `area` weight is `4π·r² / count` (uniform area
/// per sample because the Fibonacci spiral is approximately
/// equal-area on the sphere).
pub fn sample_sphere_surface(r: Real, count: usize) -> Vec<SurfaceSample> {
    let count = count.max(1);
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let area = 4.0 * std::f64::consts::PI * r * r / (count as Real);
    let mut samples = Vec::with_capacity(count);
    for i in 0..count {
        let y = 1.0 - (i as Real / (count.saturating_sub(1).max(1) as Real)) * 2.0;
        let radius_xz = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * i as Real;
        let dir = Vec3::new(theta.cos() * radius_xz, y, theta.sin() * radius_xz);
        let dir = if dir.length_squared() > 1.0e-12 {
            dir.normalize()
        } else {
            Vec3::Y
        };
        samples.push(SurfaceSample {
            position: dir * r,
            normal: dir,
            area,
        });
    }
    samples
}

/// Sample a flat `[half_x, half_y]` plane patch (oriented along
/// the local +Y axis) on a `nx × ny` grid. Useful for testing
/// hydroelastic contact between a soft body and a finite plane
/// patch (e.g. an infinite ground plane is approximated by a
/// large patch of finite samples).
pub fn sample_plane_patch(
    half_x: Real,
    half_y: Real,
    nx: usize,
    ny: usize,
) -> Vec<SurfaceSample> {
    let nx = nx.max(1);
    let ny = ny.max(1);
    let dx = (2.0 * half_x) / nx as Real;
    let dy = (2.0 * half_y) / ny as Real;
    let area = dx * dy;
    let mut out = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let x = -half_x + (i as Real + 0.5) * dx;
            let z = -half_y + (j as Real + 0.5) * dy;
            out.push(SurfaceSample {
                position: Vec3::new(x, 0.0, z),
                normal: Vec3::Y,
                area,
            });
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// Stage 39D — hydroelastic contact polish
//
// Two upgrades on top of the v1 manifold:
//
//  1. **Anchor contact for moment balance.** Reducing a patch to its
//     centroid preserves the *net force* but throws away the moment
//     about the centroid — i.e. how the pressure was distributed over
//     the patch. Solvers that read only the centroid contact will
//     under-rotate (or rotate the wrong way) when the contact patch
//     is asymmetric. Drake's hydroelastic paper (Elandt et al. 2019,
//     §IV-B) and Newton's `contact_reduction_hydroelastic.py`
//     `anchor_contact=True` path both add a synthetic second contact
//     placed at the *center of pressure* of the original patch, with
//     a normal aligned with the aggregate force direction. This
//     module's [`reduce_with_anchor`] computes that pair from a slice
//     of arbitrary `(point, normal, force_magnitude)` samples and is
//     accuracy-tested in `tests/hydroelastic_polish.rs`.
//
//  2. **Aggregate stiffness via hashtable + voxel slots + normal
//     bins.** The Stage 22.1 manifold emits one contact per surface
//     sample; for a large patch this dumps thousands of nearly-
//     redundant contacts on the solver. Newton's approach buckets
//     contacts by `(shape pair, coarse_position_voxel,
//     coarse_normal_bin)` and emits one aggregate per bucket. Per-
//     contact stiffness is **summed reciprocally** (springs in
//     series): `1/k_agg = Σ 1/k_i` reduces N nearly-coplanar contacts
//     to one *equally stiff* contact, while **parallel** addition
//     (`k_agg = Σ k_i`) is used inside the same voxel slot because
//     those represent independent contact area. See
//     [`AggregateStiffnessConfig`] for the tunable bucket size and
//     [`aggregate_by_bucket`] for the kernel. The math matches
//     Newton's `c_stiffness = k_eff · |agg_force| / total_depth`
//     scaling (where `total_depth = Σ |depth_i|` is the series-sum
//     equivalent for hydroelastic contacts).
//
// **Design note.** Both routines work on a generic
// `Vec<HydroelasticContact>` so they compose with either the v1 CPU
// or the GPU kernel's output. They are *pure functions* — no global
// state, no allocations beyond the returned `Vec` — so the contact
// pipeline can call them per-shape-pair per-substep without thinking
// about lifetimes.
// ─────────────────────────────────────────────────────────────────────

/// One sample in a distributed-pressure patch, expressed in world
/// space. This is the *input* to [`reduce_with_anchor`]: usually
/// built by iterating a [`HydroelasticManifold`]'s contacts and
/// packing `(point, normal, pressure · area)` into this struct.
///
/// `force_magnitude` is the **scalar** pressure-area product; the
/// world-frame force vector is `force_magnitude · normal`. We carry
/// the normal separately (rather than baking it into a `Vec3 force`)
/// because the anchor contact needs the aggregate *direction* as a
/// unit vector, and recovering a unit vector from `Σ p_i · A_i · n_i`
/// after the fact would lose precision when contributions cancel.
#[derive(Copy, Clone, Debug)]
pub struct PatchSample {
    /// World-frame contact point (m).
    pub point: Vec3,
    /// World-frame unit normal (points from soft body into rigid).
    pub normal: Vec3,
    /// Per-sample normal-force magnitude (N): `pressure · area`.
    pub force_magnitude: Real,
}

impl PatchSample {
    /// World-frame force vector at this sample.
    #[inline]
    pub fn force(&self) -> Vec3 {
        self.normal * self.force_magnitude
    }
}

/// Convert a [`HydroelasticManifold`]'s contacts into the
/// [`PatchSample`] slice consumed by [`reduce_with_anchor`] and
/// [`aggregate_by_bucket`]. Kept as a free function (rather than a
/// `From` impl) so callers can also build samples from non-
/// hydroelastic sources without round-tripping through `HydroelasticContact`.
pub fn manifold_to_samples(manifold: &HydroelasticManifold) -> Vec<PatchSample> {
    manifold
        .contacts
        .iter()
        .map(|c| PatchSample {
            point: c.point,
            normal: c.normal,
            force_magnitude: c.normal_force(),
        })
        .collect()
}

/// Output of [`reduce_with_anchor`] — exactly two contacts that
/// together reproduce both the net force AND net moment of the input
/// patch.
///
/// The pair is constructed as follows:
///
/// * `centroid` carries the **area-weighted centroid position** and
///   the **net force direction**. Its `force_magnitude` is the bulk
///   of the load (matched to give zero net residual after the anchor
///   contact is added).
/// * `anchor` is placed at the **center of pressure** of the patch
///   and carries a complementary force such that the combined wrench
///   `(F_c · n_c at p_c) + (F_a · n_a at p_a)` reproduces both the
///   net force and the moment about any reference point.
///
/// **Math.** Let `F_total = Σ f_i · n_i` and `M_total = Σ r_i × (f_i · n_i)`
/// (moment about the centroid). The aggregate-force direction is
/// `n̂ = F_total / |F_total|`. We pick the anchor position as the
/// center of pressure `r_cp = (Σ f_i · r_i) / Σ f_i`, then split the
/// total force into two parts placed at `r_centroid` (the spatial
/// centroid) and `r_cp` such that:
///
///   F_c + F_a = F_total                        (force balance)
///   r_centroid × F_c + r_cp × F_a = M_total    (moment balance)
///
/// With both `F_c` and `F_a` chosen along `n̂`, the moment equation
/// projects onto the plane perpendicular to `n̂` and yields a 2-D
/// linear system in `(|F_c|, |F_a|)`. We solve in closed form below.
///
/// See `tests/hydroelastic_polish.rs::anchor_contact_reproduces_moment`
/// for the numerical check (<1% error vs the original patch).
#[derive(Copy, Clone, Debug)]
pub struct AnchorReducedContact {
    /// Centroid contact (spatial mean of the patch).
    pub centroid: PatchSample,
    /// Anchor contact (center of pressure).
    pub anchor: PatchSample,
}

impl AnchorReducedContact {
    /// Net force vector of the pair. Should match the input patch's
    /// `Σ f_i · n_i` to within numerical roundoff.
    pub fn net_force(&self) -> Vec3 {
        self.centroid.force() + self.anchor.force()
    }

    /// Net moment about the world origin. Used by the tests to
    /// verify the pair reproduces the patch's moment.
    pub fn net_moment(&self) -> Vec3 {
        self.centroid.point.cross(self.centroid.force())
            + self.anchor.point.cross(self.anchor.force())
    }
}

/// Reduce a distributed pressure patch to a centroid + anchor pair
/// that reproduces both the net force AND the net moment of the
/// input. See [`AnchorReducedContact`] for the math.
///
/// Returns `None` when the patch is empty or carries zero net force
/// (then there is nothing meaningful to anchor).
pub fn reduce_with_anchor(samples: &[PatchSample]) -> Option<AnchorReducedContact> {
    if samples.is_empty() {
        return None;
    }
    // Aggregate force and direction.
    let total_force_vec: Vec3 = samples.iter().map(|s| s.force()).sum();
    let total_force_mag = total_force_vec.length();
    if total_force_mag < 1.0e-20 {
        return None;
    }
    let force_dir = total_force_vec / total_force_mag;

    // Two reference points:
    //   * `r_centroid` — area-weighted centroid (uses `force_magnitude`
    //     as the weight, which equals `pressure · area` for the
    //     hydroelastic case so it's the area-weighted centroid when
    //     pressure is uniform).
    //   * `r_cp` — center of pressure (same formula, but explicit so
    //     callers reading the code see the physics). For our purposes
    //     the two coincide in expectation; we pick the centroid by
    //     equal weighting (1/N) and the COP by force-magnitude
    //     weighting (f_i/Σf), so they differ whenever pressure is
    //     spatially non-uniform — exactly the regime where the moment
    //     matters.
    let n = samples.len() as Real;
    let r_centroid: Vec3 =
        samples.iter().map(|s| s.point).sum::<Vec3>() / n;
    let sum_f: Real = samples.iter().map(|s| s.force_magnitude).sum();
    let r_cp: Vec3 = if sum_f > 1.0e-20 {
        samples
            .iter()
            .map(|s| s.point * s.force_magnitude)
            .sum::<Vec3>()
            / sum_f
    } else {
        r_centroid
    };

    // Net moment of the original patch about the world origin.
    let m_total: Vec3 = samples.iter().map(|s| s.point.cross(s.force())).sum();

    // ── Force/moment split ─────────────────────────────────────────
    // Both contact forces are along `force_dir`. Write:
    //   F_c = α · force_dir · total_force_mag
    //   F_a = (1 - α) · force_dir · total_force_mag
    // with α ∈ ℝ (not necessarily in [0, 1] — we let the anchor pull
    // the other way if needed to balance the moment).
    //
    // The moment about the origin is:
    //   M = r_centroid × F_c + r_cp × F_a
    //     = total_force_mag · [α · r_centroid × n̂
    //                          + (1 - α) · r_cp × n̂]
    //     = total_force_mag · [r_cp × n̂ + α · (r_centroid - r_cp) × n̂]
    //
    // Setting `M == M_total` and solving for α, dotting both sides
    // with `(r_centroid - r_cp) × n̂` (a vector in the plane
    // perpendicular to n̂):
    //
    //   α · |Δr × n̂|² · total_force_mag
    //     = (M_total - total_force_mag · r_cp × n̂) · (Δr × n̂)
    //
    // where Δr = r_centroid - r_cp.
    let dr = r_centroid - r_cp;
    let dr_cross_n = dr.cross(force_dir);
    let denom = dr_cross_n.length_squared() * total_force_mag;
    let alpha = if denom > 1.0e-20 {
        let target_m = m_total - r_cp.cross(force_dir) * total_force_mag;
        target_m.dot(dr_cross_n) / denom
    } else {
        // Degenerate: r_centroid == r_cp (uniform pressure, or
        // single-sample patch). The split is irrelevant — pick
        // α = 1 so the anchor carries no force. The centroid
        // contact alone reproduces the wrench within the numerical
        // precision of `m_total ≈ r_centroid × F_total`.
        1.0
    };

    let f_centroid_mag = alpha * total_force_mag;
    let f_anchor_mag = (1.0 - alpha) * total_force_mag;
    Some(AnchorReducedContact {
        centroid: PatchSample {
            point: r_centroid,
            normal: force_dir,
            force_magnitude: f_centroid_mag,
        },
        anchor: PatchSample {
            point: r_cp,
            normal: force_dir,
            force_magnitude: f_anchor_mag,
        },
    })
}

/// Aggregation policy for [`aggregate_by_bucket`]. The two cases —
/// stiffness in series vs in parallel — correspond to the physical
/// behaviour of springs:
///
/// * **Series**: contacts mounted "in line" (same line of action,
///   stacked depth-wise). Equivalent compliance `c = Σ c_i = Σ 1/k_i`,
///   so `k_agg = 1 / Σ (1/k_i)`. Used when collapsing duplicate
///   contacts that all touch the same point on the rigid body
///   through different paths (e.g. multiple SDF samples that land
///   in the same voxel slot).
/// * **Parallel**: contacts mounted side-by-side (independent
///   contact patches, sharing the load). Equivalent stiffness
///   `k_agg = Σ k_i`. Used when summing distinct contact area
///   inside the same `(position_voxel, normal_bin)` bucket — each
///   sample contributes independent surface area, and the total
///   resistance is the sum.
///
/// **Newton's policy** (mirrored here): use *parallel* aggregation
/// inside a bucket (multiple SDF samples sharing a normal bin all
/// contribute their `pressure · area` independently) and *series*
/// aggregation across buckets when collapsing the manifold to a
/// single anchor pair. See
/// `tests/hydroelastic_polish.rs::aggregate_stiffness_recovers_series_compliance`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StiffnessReduction {
    /// `k_agg = Σ k_i` (parallel springs).
    Parallel,
    /// `k_agg = 1 / Σ (1/k_i)` (series springs / sum of compliances).
    Series,
}

/// Bucket granularity for [`aggregate_by_bucket`]. Tune `voxel_size`
/// down to keep more contacts (finer manifold); tune it up to
/// collapse aggressively (cheaper solver). `normal_bin_angle` is the
/// angular width of a normal bin in radians — contacts whose normals
/// fall within `cos(angle)` of each other go in the same bin.
#[derive(Copy, Clone, Debug)]
pub struct AggregateStiffnessConfig {
    /// Position-space voxel edge length (m). Contacts within the
    /// same axis-aligned voxel are candidates for the same bucket.
    /// Set this to a fraction of the expected patch extent — Newton
    /// uses `aabb_size / voxel_resolution`, typically `~0.05 m`.
    pub voxel_size: Real,
    /// Angular bin width for the normal direction (radians). Two
    /// normals `n_a`, `n_b` go in the same bin iff their octahedral-
    /// quantised bin id matches; we approximate that with an
    /// equirectangular quantisation here.
    pub normal_bin_angle: Real,
    /// How to combine per-contact stiffness inside one bucket. See
    /// [`StiffnessReduction`]; default is `Parallel`.
    pub stiffness_reduction: StiffnessReduction,
}

impl Default for AggregateStiffnessConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.05,
            // ~10° default bin width (matches Newton's `NUM_NORMAL_BINS=18`
            // octahedral binning, where each bin covers roughly 4π/18 sr
            // ≈ 0.69 sr ≈ 0.83 rad spread — we use a tighter 10° for
            // direction-quality reasons).
            normal_bin_angle: 10.0_f64.to_radians(),
            stiffness_reduction: StiffnessReduction::Parallel,
        }
    }
}

/// One reduced contact emitted by [`aggregate_by_bucket`]. The
/// world-frame `point` is the area-weighted centroid of the bucket;
/// `normal` is the force-magnitude-weighted mean direction
/// (renormalised to unit length); `force_magnitude` is the sum of
/// the bucket's per-sample force magnitudes; `aggregate_stiffness`
/// is the bucket's reduced stiffness per
/// [`StiffnessReduction`].
#[derive(Copy, Clone, Debug)]
pub struct AggregatedContact {
    pub point: Vec3,
    pub normal: Vec3,
    pub force_magnitude: Real,
    pub aggregate_stiffness: Real,
    /// Number of input samples that fell in this bucket.
    pub sample_count: u32,
}

/// Per-sample stiffness — passed in alongside the patch samples so
/// the reducer can apply the per-bucket [`StiffnessReduction`]
/// policy.
#[derive(Copy, Clone, Debug)]
pub struct StiffnessSample<'a> {
    pub patch: &'a PatchSample,
    pub stiffness: Real,
}

/// Aggregate a slice of patch samples (each carrying its own
/// stiffness) into one [`AggregatedContact`] per
/// `(position_voxel, normal_bin)` bucket. This is the molla
/// equivalent of Newton's hashtable+voxel+normal-bin reducer in
/// `contact_reduction_hydroelastic.py`.
///
/// **Algorithm.**
///
/// 1. For each input sample, compute a coarse `(voxel_ix, voxel_iy,
///    voxel_iz, normal_bin_ix, normal_bin_iy)` bucket key. Position
///    voxel is `floor(p / voxel_size)`; normal bin is the equi-
///    rectangular quantisation of the unit normal into a sphere of
///    `2π·π / (bin_angle²)` cells — fine enough that two contacts
///    in the same bin have normals within `bin_angle` radians.
/// 2. Group samples by bucket key in a `HashMap`.
/// 3. For each bucket, sum the force-weighted position and the
///    force-weighted normal; combine the per-sample stiffness
///    according to the configured [`StiffnessReduction`].
/// 4. Emit one [`AggregatedContact`] per bucket.
///
/// **Why a `HashMap` (and not a fixed-size hashtable like Newton's
/// CUDA kernel)?** On the CPU, a `HashMap` is fast enough for the
/// expected scale (typically <10000 samples per pair per substep).
/// The GPU port (PLAN Stage 39E) will use the fixed-size hashtable;
/// the CPU and GPU paths share the bucket-key + reduction policy.
pub fn aggregate_by_bucket(
    samples: &[StiffnessSample<'_>],
    cfg: &AggregateStiffnessConfig,
) -> Vec<AggregatedContact> {
    use std::collections::HashMap;

    if samples.is_empty() {
        return Vec::new();
    }
    // Translate sample positions by the patch centroid before
    // voxelising. This makes the bucket assignment invariant to the
    // global coordinate frame — a patch positioned at (5, 0, 5)
    // collapses to the same bucket count as one at the origin —
    // and avoids the obvious failure mode where a patch straddling
    // x = 0 splits across `floor(-ε/voxel) = -1` and
    // `floor(+ε/voxel) = 0` even though `voxel >> ε`. Newton's
    // CUDA path keys against `shape_b`'s AABB lower corner instead
    // of the centroid, but the two are equivalent for grouping
    // purposes (both just provide a stable reference point).
    let patch_centroid: Vec3 = samples
        .iter()
        .map(|s| s.patch.point)
        .sum::<Vec3>()
        / (samples.len() as Real);
    let inv_voxel = 1.0 / cfg.voxel_size.max(1.0e-12);
    // Normal-bin discretisation. We snap the unit normal onto a
    // direction-cosine grid: each axis component `n_i ∈ [-1, 1]` is
    // floored at a step of `sin(angle)`. Two normals within `angle`
    // radians of each other produce direction-cosine differences of
    // at most `~angle` along each axis (and exactly `sin(angle)`
    // perpendicular to the rotation axis), so this binning has the
    // property that two normals within `angle` radians collide
    // *unless* they straddle a grid line — close to Newton's
    // octahedral binning but trivially singularity-free, which we
    // need because contact patches with `n ≈ ±ẑ` (or any axis) are
    // common and would land on the equirectangular-grid pole
    // otherwise. The bin_step = sin(angle) trade-off is conservative
    // and matches the rule-of-thumb in the Drake hydroelastic paper
    // for normal-bin width.
    let bin_step = cfg.normal_bin_angle.max(1.0e-6).sin();
    let inv_bin = 1.0 / bin_step;

    type Key = (i64, i64, i64, i64, i64, i64);

    struct BucketAcc {
        /// Force-magnitude-weighted position sum.
        weighted_pos: Vec3,
        /// Force-magnitude-weighted normal sum.
        weighted_normal: Vec3,
        /// Σ |f_i|.
        force_sum: Real,
        /// Σ k_i (parallel) or Σ 1/k_i (series), as configured.
        stiffness_acc: Real,
        sample_count: u32,
    }

    let mut buckets: HashMap<Key, BucketAcc> = HashMap::new();

    for s in samples {
        let p = s.patch.point;
        let n = s.patch.normal;
        // Round-based voxel binning (with displacements relative to
        // the patch centroid) means a patch of half-extent < step/2
        // collapses to bucket (0, 0, 0). Floor-based binning would
        // straddle the 0-line and split a centred symmetric patch
        // into 2³ = 8 buckets even when the patch is much smaller
        // than the voxel — observed in the unit test.
        let q = p - patch_centroid;
        let vx = (q.x * inv_voxel).round() as i64;
        let vy = (q.y * inv_voxel).round() as i64;
        let vz = (q.z * inv_voxel).round() as i64;
        // Direction-cosine grid: snap each axis component of the
        // (renormalised) unit normal to the *nearest* bin center
        // (round, not floor). `round` is symmetric around zero, so a
        // patch with `n ≈ +Y` and small ±x/z wobble within `step/2`
        // collapses cleanly into bin (0, ±1, 0) without straddling
        // a grid edge near each axis. This is singularity-free at
        // every axis-aligned direction.
        let n_norm = if n.length_squared() > 1.0e-24 {
            n.normalize()
        } else {
            Vec3::Y
        };
        let bn_x = (n_norm.x * inv_bin).round() as i64;
        let bn_y = (n_norm.y * inv_bin).round() as i64;
        let bn_z = (n_norm.z * inv_bin).round() as i64;
        let key: Key = (vx, vy, vz, bn_x, bn_y, bn_z);
        let fmag = s.patch.force_magnitude.max(0.0);
        let acc = buckets.entry(key).or_insert(BucketAcc {
            weighted_pos: Vec3::ZERO,
            weighted_normal: Vec3::ZERO,
            force_sum: 0.0,
            stiffness_acc: 0.0,
            sample_count: 0,
        });
        // Weight position/normal by force magnitude. If the bucket
        // ends up with zero total force we fall back to an
        // equal-weight mean (see emission step below).
        acc.weighted_pos += p * fmag;
        acc.weighted_normal += n * fmag;
        acc.force_sum += fmag;
        acc.sample_count += 1;
        let k = s.stiffness.max(0.0);
        match cfg.stiffness_reduction {
            StiffnessReduction::Parallel => acc.stiffness_acc += k,
            StiffnessReduction::Series => {
                if k > 1.0e-30 {
                    acc.stiffness_acc += 1.0 / k;
                }
            }
        }
    }

    // Convert per-bucket accumulators into output contacts.
    let mut out = Vec::with_capacity(buckets.len());
    for (_, acc) in buckets {
        // Position/normal: force-weighted if any force present,
        // else equal-weight (fallback when all samples were
        // zero-force — useful e.g. for margin contacts).
        let (pos, norm) = if acc.force_sum > 1.0e-20 {
            let pos = acc.weighted_pos / acc.force_sum;
            let n = if acc.weighted_normal.length_squared() > 1.0e-24 {
                acc.weighted_normal.normalize()
            } else {
                Vec3::Y
            };
            (pos, n)
        } else {
            let pos = acc.weighted_pos
                / (acc.sample_count as Real).max(1.0);
            (pos, Vec3::Y)
        };
        let k_agg = match cfg.stiffness_reduction {
            StiffnessReduction::Parallel => acc.stiffness_acc,
            StiffnessReduction::Series => {
                if acc.stiffness_acc > 1.0e-30 {
                    1.0 / acc.stiffness_acc
                } else {
                    0.0
                }
            }
        };
        out.push(AggregatedContact {
            point: pos,
            normal: norm,
            force_magnitude: acc.force_sum,
            aggregate_stiffness: k_agg,
            sample_count: acc.sample_count,
        });
    }
    out
}

/// Stage-22.1 GPU dispatch for hydroelastic contact. Feature-gated
/// so the CPU types stay buildable without `gpu`.
#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use crate::core::{Error, Result};
    use crate::storage::{Buffer, BufferUsage};

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct HydroUniforms {
        sample_count: u32,
        nx: u32,
        ny: u32,
        nz: u32,
        cell_size: f32,
        modulus: f32,
        max_pressure: f32,
        culling_distance: f32,
        /// SDF grid origin in soft-body LOCAL frame.
        origin: [f32; 4],
        /// Soft body world translation (subtracted from `p_world`
        /// before the inverse rotation).
        soft_translation: [f32; 4],
        /// Soft body world rotation (column-major). The shader
        /// applies the inverse (transpose) to map world→soft-local.
        rot_col0: [f32; 4],
        rot_col1: [f32; 4],
        rot_col2: [f32; 4],
        /// Rigid body world translation.
        rigid_translation: [f32; 4],
        /// Rigid body world rotation as a 3×3.
        rigid_rot_col0: [f32; 4],
        rigid_rot_col1: [f32; 4],
        rigid_rot_col2: [f32; 4],
    }

    const WORKGROUP_SIZE: u32 = 64;

    /// One hydroelastic-contact GPU dispatch. Owns the persistent
    /// per-sample output buffers (contact normal + pressure + area)
    /// so a downstream Dahl friction kernel can directly bind them.
    pub struct HydroelasticKernel {
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        pub contact_normal: Buffer<Vec3>,
        pub contact_point: Buffer<Vec3>,
        pub contact_normal_force: Buffer<Real>,
        pub contact_friction_mu: Buffer<Real>,
        sample_count: u32,
    }

    impl HydroelasticKernel {
        pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, sample_count: u32) -> Self {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("molla::hydroelastic"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../shaders/hydroelastic.wgsl").into(),
                ),
            });
            let mut entries = vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }];
            for (i, read_only) in [
                (1u32, true),
                (2, true),
                (3, true),
                (4, true),
                (5, false),
                (6, false),
                (7, false),
            ] {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::hydroelastic layout"),
                entries: &entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::hydroelastic pipeline layout"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::hydroelastic"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("hydro_contact"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let cn = sample_count.max(1) as usize;
            let mut contact_normal =
                Buffer::from_vec(vec![Vec3::ZERO; cn], BufferUsage::StorageReadWrite);
            let mut contact_point =
                Buffer::from_vec(vec![Vec3::ZERO; cn], BufferUsage::StorageReadWrite);
            let mut contact_normal_force =
                Buffer::from_vec(vec![0.0_f64; cn], BufferUsage::StorageReadWrite);
            let mut contact_friction_mu =
                Buffer::from_vec(vec![0.0_f64; cn], BufferUsage::StorageReadWrite);
            // Construction-time GPU allocations are unrecoverable on
            // failure (wgpu OOM); panic with context here rather than
            // letting the dispatch path surface a confusing
            // "missing device mirror" later.
            contact_normal
                .ensure_device(device)
                .expect("hydroelastic: contact_normal ensure_device");
            contact_normal
                .upload(queue)
                .expect("hydroelastic: contact_normal upload");
            contact_point
                .ensure_device(device)
                .expect("hydroelastic: contact_point ensure_device");
            contact_point
                .upload(queue)
                .expect("hydroelastic: contact_point upload");
            contact_normal_force
                .ensure_device(device)
                .expect("hydroelastic: contact_normal_force ensure_device");
            contact_normal_force
                .upload(queue)
                .expect("hydroelastic: contact_normal_force upload");
            contact_friction_mu
                .ensure_device(device)
                .expect("hydroelastic: contact_friction_mu ensure_device");
            contact_friction_mu
                .upload(queue)
                .expect("hydroelastic: contact_friction_mu upload");
            Self {
                pipeline,
                layout,
                contact_normal,
                contact_point,
                contact_normal_force,
                contact_friction_mu,
                sample_count,
            }
        }

        pub fn sample_count(&self) -> u32 {
            self.sample_count
        }

        /// Dispatch the kernel. `sdf_data` is the flat SDF grid as
        /// uploaded by the host; the per-sample output buffers
        /// (`contact_normal`, `contact_point`, `contact_normal_force`,
        /// `contact_friction_mu`) get filled in-place.
        #[allow(clippy::too_many_arguments)]
        pub fn dispatch(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            soft_pose: Transform,
            rigid_pose: Transform,
            grid: &SdfGrid,
            sample_local_pos: &Buffer<Vec3>,
            sample_local_normal: &Buffer<Vec3>,
            sample_area: &Buffer<Real>,
            sdf_data: &Buffer<Real>,
            cfg: &HydroelasticConfig,
        ) -> Result<()> {
            let rot = soft_pose.rotation;
            let r_mat = glam::DMat3::from_quat(rot);
            let rrot = rigid_pose.rotation;
            let rmat = glam::DMat3::from_quat(rrot);
            let u = HydroUniforms {
                sample_count: self.sample_count,
                nx: grid.nx,
                ny: grid.ny,
                nz: grid.nz,
                cell_size: grid.cell_size as f32,
                modulus: cfg.hydroelastic_modulus as f32,
                max_pressure: cfg.max_pressure as f32,
                culling_distance: cfg.culling_distance as f32,
                origin: [
                    grid.origin.x as f32,
                    grid.origin.y as f32,
                    grid.origin.z as f32,
                    0.0,
                ],
                soft_translation: [
                    soft_pose.position.x as f32,
                    soft_pose.position.y as f32,
                    soft_pose.position.z as f32,
                    0.0,
                ],
                rot_col0: [
                    r_mat.x_axis.x as f32,
                    r_mat.x_axis.y as f32,
                    r_mat.x_axis.z as f32,
                    0.0,
                ],
                rot_col1: [
                    r_mat.y_axis.x as f32,
                    r_mat.y_axis.y as f32,
                    r_mat.y_axis.z as f32,
                    0.0,
                ],
                rot_col2: [
                    r_mat.z_axis.x as f32,
                    r_mat.z_axis.y as f32,
                    r_mat.z_axis.z as f32,
                    0.0,
                ],
                rigid_translation: [
                    rigid_pose.position.x as f32,
                    rigid_pose.position.y as f32,
                    rigid_pose.position.z as f32,
                    0.0,
                ],
                rigid_rot_col0: [
                    rmat.x_axis.x as f32,
                    rmat.x_axis.y as f32,
                    rmat.x_axis.z as f32,
                    0.0,
                ],
                rigid_rot_col1: [
                    rmat.y_axis.x as f32,
                    rmat.y_axis.y as f32,
                    rmat.y_axis.z as f32,
                    0.0,
                ],
                rigid_rot_col2: [
                    rmat.z_axis.x as f32,
                    rmat.z_axis.y as f32,
                    rmat.z_axis.z as f32,
                    0.0,
                ],
            };
            let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::hydroelastic uniforms"),
                size: core::mem::size_of::<HydroUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&u));

            let pos_buf = sample_local_pos
                .device_buffer()
                .ok_or_else(|| Error::Gpu("hydroelastic: sample_local_pos missing".into()))?;
            let normal_buf = sample_local_normal
                .device_buffer()
                .ok_or_else(|| Error::Gpu("hydroelastic: sample_local_normal missing".into()))?;
            let area_buf = sample_area
                .device_buffer()
                .ok_or_else(|| Error::Gpu("hydroelastic: sample_area missing".into()))?;
            let sdf_buf = sdf_data
                .device_buffer()
                .ok_or_else(|| Error::Gpu("hydroelastic: sdf_data missing".into()))?;
            let n_buf = self
                .contact_normal
                .device_buffer()
                .ok_or_else(|| Error::Gpu("hydroelastic: contact_normal missing".into()))?;
            let p_buf = self
                .contact_point
                .device_buffer()
                .ok_or_else(|| Error::Gpu("hydroelastic: contact_point missing".into()))?;
            let f_buf = self
                .contact_normal_force
                .device_buffer()
                .ok_or_else(|| Error::Gpu("hydroelastic: contact_normal_force missing".into()))?;

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::hydroelastic bg"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: pos_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: normal_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: area_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: sdf_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: n_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: p_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: f_buf.as_entire_binding() },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::hydroelastic encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::hydroelastic"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                let groups = self.sample_count.div_ceil(WORKGROUP_SIZE).max(1);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
            self.contact_normal.mark_device_dirty();
            self.contact_point.mark_device_dirty();
            self.contact_normal_force.mark_device_dirty();
            Ok(())
        }
    }
}

#[cfg(feature = "gpu")]
pub use gpu::HydroelasticKernel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::TriMesh;

    fn unit_cube_mesh() -> TriMesh {
        let v = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];
        let i = vec![
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 6, 2], [3, 7, 6],
            [0, 7, 3], [0, 4, 7],
            [1, 2, 6], [1, 6, 5],
        ];
        let mut m = TriMesh {
            vertices: v,
            indices: i,
            ..Default::default()
        };
        m.build_bvh(4);
        m
    }

    /// CPU sanity: a sphere (sampled at 64 points) of radius 0.3
    /// pressed into a soft cube of size 1.0 should produce a
    /// non-empty manifold with positive total normal force.
    #[test]
    fn cpu_sphere_into_cube_produces_force() {
        use super::super::sdf_bake::bake_mesh_to_sdf;
        let mesh = unit_cube_mesh();
        let sdf = bake_mesh_to_sdf(&mesh, 32, 0.05);
        let samples = sample_sphere_surface(0.3, 64);
        // Position the sphere's centre at +0.3 above the cube top
        // → sphere bottom touches cube top exactly. Now move it
        // DOWN by 0.05 m so it penetrates a little.
        let sphere_pose =
            Transform::from_translation(Vec3::new(0.0, 0.5 + 0.3 - 0.05, 0.0));
        let cube_pose = Transform::IDENTITY;
        let cfg = HydroelasticConfig::default();
        let manifold =
            compute_hydroelastic_contact(&sdf, &samples, cube_pose, sphere_pose, &cfg);
        assert!(
            !manifold.contacts.is_empty(),
            "expected at least one hydroelastic contact, got empty manifold"
        );
        assert!(
            manifold.total_normal_force > 0.0,
            "expected positive total force, got {}",
            manifold.total_normal_force
        );
        // The net force should point UP (out of the soft cube,
        // along the gradient ∇sdf at the top face).
        assert!(
            manifold.net_force.y > 0.0,
            "net force should push sphere up, got {:?}",
            manifold.net_force
        );
    }

    /// Sample positions on the sphere are well-distributed:
    /// the average position is near the origin (Fibonacci spiral).
    #[test]
    fn sphere_samples_are_balanced() {
        let samples = sample_sphere_surface(1.0, 256);
        let avg = samples
            .iter()
            .fold(Vec3::ZERO, |a, s| a + s.position)
            / samples.len() as Real;
        assert!(avg.length() < 0.05, "avg sphere sample {:?} not centred", avg);
        // Surface integral approx → `4π`.
        let total: Real = samples.iter().map(|s| s.area).sum();
        let analytic = 4.0 * std::f64::consts::PI;
        assert!(
            (total - analytic).abs() / analytic < 0.05,
            "sphere area integral {total} vs {analytic}"
        );
    }

    /// SDF gradient at the top of the cube points UP (the outward
    /// normal at the +Y face).
    #[test]
    fn sdf_gradient_matches_expected_normal() {
        use super::super::sdf_bake::bake_mesh_to_sdf;
        let mesh = unit_cube_mesh();
        let sdf = bake_mesh_to_sdf(&mesh, 32, 0.05);
        // A point just inside the +Y face.
        let p = Vec3::new(0.0, 0.45, 0.0);
        let g = sample_sdf_gradient(&sdf, p);
        // The gradient should mostly be +Y.
        assert!(
            g.y.abs() > g.x.abs() && g.y.abs() > g.z.abs(),
            "+Y gradient expected, got {:?}",
            g
        );
        assert!(g.y > 0.0, "+Y gradient should be positive: {:?}", g);
    }
}
