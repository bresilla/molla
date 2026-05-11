# Hydroelastic contact

Point contacts (one normal + one tangent at a single point) work
fine for sphere-on-sphere or capsule-on-plane collisions, but
they're noisy for **patch contacts** — a cloth resting on a
sphere, two boxes stacked face-to-face, a finger pressed against
a soft ball. The contact "point" jitters between patch vertices
each frame; the solver damps it out but the system never quite
reaches rest.

Drake's **hydroelastic contact model** (Elandt, Drumwright,
Sherman & Kuchenbecker 2019) replaces the point contact with a
**distributed pressure field over the intersection volume** of
the two bodies. molla ships a soft-on-rigid implementation at
Stage 22.1 (`crates/molla/src/geometry/hydroelastic.rs`), with a
GPU port (Stage 41 v2 sub-4) in `shaders/hydroelastic.wgsl`. The
Stage 39D polish adds wrench-preserving reduction
(`reduce_with_anchor`) and bucket aggregation
(`aggregate_by_bucket`).

## Model

The soft body carries a [`SdfGrid`](./sdf.md) representing its
**hydroelastic pressure field**. The pressure inside the body is
a linear function of penetration depth into the body:

```text
p(x) = E · max(0, -sdf(x))           [units: Pa = N/m²]
```

`E` is the hydroelastic modulus (Drake's "elastic modulus"). The
contact force is the integral of pressure over the intersection
surface — for soft-on-rigid that's the rigid body's surface,
discretised as a quadrature point cloud with per-point area
weights.

## v1 scope (Stage 22.1)

* **Soft-on-rigid only.** Soft body has an SDF pressure field;
  rigid body is sampled at discrete surface points.
* **Soft-on-soft is deferred.** Both volumes overlap → genuinely
  3D intersection. Requires either marching cubes on
  `min(p_a - p_b, 0)` for the iso-surface, or a tetrahedralised
  volume integration. Neither is in v1.
* **Linear hydroelastic** — pressure is `E · max(0, -sdf)`.
* **Per-sample friction** — each contact carries a Dahl-friction
  contact id for persistent bristle state across substeps.

```rust
pub struct HydroelasticConfig {
    pub hydroelastic_modulus: Real, // E (Pa); 1e4..1e7 for rubber, 1e9+ for plastic
    pub max_pressure:         Real, // safety cap on per-sample pressure
    pub friction_coefficient: Real, // Coulomb μ (0.4 default)
    pub culling_distance:     Real, // skip samples this far outside the SDF
}

pub enum HydroelasticPolicy {
    SoftOnRigid,    // v1 scope
    SoftOnSoft,     // not implemented in v1
}

pub struct SurfaceSample {
    pub position: Vec3, // rigid-body local frame
    pub normal:   Vec3, // outward-facing surface normal, local frame
    pub area:     Real, // quadrature weight (m²)
}

pub struct HydroelasticContact {
    pub point:      Vec3, // world contact point
    pub normal:     Vec3, // world outward (out of soft, into rigid)
    pub pressure:   Real, // Pa
    pub area:       Real, // m²
    pub contact_id: u32,  // for friction warm-start
}

pub struct HydroelasticManifold {
    pub contacts:           Vec<HydroelasticContact>,
    pub total_normal_force: Real,
    pub net_force:          Vec3,
}
```

## `compute_hydroelastic_contact`

```rust
pub fn compute_hydroelastic_contact(
    soft_sdf:      &SdfGrid,
    rigid_samples: &[SurfaceSample],
    soft_pose:     Transform,
    rigid_pose:    Transform,
    cfg:           &HydroelasticConfig,
) -> HydroelasticManifold;
```

Algorithm (`hydroelastic.rs:211-262`):

1. For each rigid surface sample, transform its local position
   into world via `rigid_pose`, then into the soft body's local
   frame via `soft_pose.inverse()`.
2. Trilinearly sample the soft body's SDF at that point.
3. If `sdf < 0` (inside the soft body), compute
   `pressure = E · (-sdf)`, clamped to `max_pressure`.
4. The contact normal points OUT of the soft body — that's
   `+∇sdf` (SDF increases outward), computed via central
   differences in `sample_sdf_gradient`. Transform back to world.
5. Assign a stable contact id (typically
   `(rigid_shape_id, sample_index)` packed into a `u32`) and
   emit a `HydroelasticContact`.

Output `total_normal_force` is the quadrature approximation to
the surface integral of `pressure · area`. The empty manifold
case (no penetration) trivially yields `total_normal_force = 0,
net_force = ZERO`.

## Surface sampling helpers

```rust
pub fn sample_sphere_surface(r: Real, count: usize) -> Vec<SurfaceSample>;

pub fn sample_plane_patch(
    extent_x: Real, extent_z: Real,
    nx: usize,    nz: usize,
    plane_y: Real, normal: Vec3,
) -> Vec<SurfaceSample>;
```

`sample_sphere_surface` uses Fibonacci-spiral sampling — `count`
samples on a unit sphere scaled to radius `r`, each carrying its
spherical-area weight. `sample_plane_patch` produces a uniform
`nx × nz` grid of samples on a finite plane patch.

## `PatchSample` — signed `force_magnitude` contract

```rust
pub struct PatchSample {
    pub point:          Vec3, // world contact point
    pub normal:         Vec3, // unit, world frame (soft → rigid)
    pub force_magnitude: Real, // SIGNED; see below
}

impl PatchSample {
    pub fn force(&self) -> Vec3 { self.normal * self.force_magnitude }
}

pub fn manifold_to_samples(manifold: &HydroelasticManifold) -> Vec<PatchSample>;
```

**Sign convention:** for raw samples emitted by hydroelastic
integration `force_magnitude ≥ 0`. After
[`reduce_with_anchor`](#stage-39d--anchor-reduction) the
*centroid* and *anchor* samples can each carry a **negative**
`force_magnitude` when the moment-balance split coefficient `α`
leaves `[0, 1]` — the centroid pulls one way and the anchor pulls
the other, summing to the total normal force. Callers projecting
onto a friction cone or capping by magnitude should treat this
field as signed and use `force_magnitude.abs()` when they need
the actual magnitude.

## Stage 39D — anchor reduction

```rust
pub struct AnchorReducedContact {
    pub centroid: PatchSample, // spatial centroid of the patch
    pub anchor:   PatchSample, // center of pressure
}

impl AnchorReducedContact {
    pub fn net_force(&self) -> Vec3;
    pub fn net_moment(&self) -> Vec3;
}

pub fn reduce_with_anchor(samples: &[PatchSample]) -> Option<AnchorReducedContact>;
```

Reduces a distributed pressure patch to **exactly two contacts**
that together reproduce both the **net force** AND the **net
moment** about any reference point. Half the contact count of the
input (one per region rather than dozens of quadrature samples),
without losing the wrench.

### Math

Let `F_total = Σ fᵢ · n̂ᵢ` and `M_total = Σ rᵢ × (fᵢ · n̂ᵢ)`. Both
contact forces are along `n̂ = F_total / |F_total|`. Write:

```text
F_c = α     · n̂ · |F_total|        at r_centroid
F_a = (1-α) · n̂ · |F_total|        at r_cp
```

with `α ∈ ℝ` (not necessarily in `[0, 1]` — the anchor can pull
the opposite way to balance the moment). The two reference
points:

* `r_centroid = (1/N) Σ rᵢ` — spatial centroid (equal weights).
* `r_cp = (Σ fᵢ rᵢ) / Σ fᵢ` — center of pressure (force-weighted).

The moment equation `r_centroid × F_c + r_cp × F_a = M_total`
projects onto the plane perpendicular to `n̂` and yields a 2D
linear system in `(|F_c|, |F_a|)`. The closed-form solution
(`hydroelastic.rs:610-624`):

```rust
let dr          = r_centroid - r_cp;
let dr_cross_n  = dr.cross(force_dir);
let denom       = dr_cross_n.length_squared() * total_force_mag;
let alpha = if denom > 1.0e-20 {
    let target_m = m_total - r_cp.cross(force_dir) * total_force_mag;
    target_m.dot(dr_cross_n) / denom
} else {
    1.0 // degenerate: r_centroid == r_cp, anchor carries nothing
};
```

Verification: `tests/hydroelastic_polish.rs::anchor_contact_reproduces_moment`
checks the pair reproduces the original patch's wrench to within
1%.

## Stage 39D — bucket aggregation

```rust
pub enum StiffnessReduction {
    Parallel,    // k_agg = Σ k_i
    Series,      // k_agg = 1 / Σ (1/k_i)
}

pub struct AggregateStiffnessConfig {
    pub voxel_size:           Real, // position-bin edge length
    pub normal_bin_angle:     Real, // angular bin width (rad)
    pub stiffness_reduction:  StiffnessReduction,
}

pub struct AggregatedContact {
    pub point:               Vec3,
    pub normal:              Vec3,
    pub force_magnitude:     Real,
    pub aggregate_stiffness: Real,
    pub sample_count:        u32,
}

pub struct StiffnessSample<'a> {
    pub patch:     &'a PatchSample,
    pub stiffness: Real,
}

pub fn aggregate_by_bucket(
    samples: &[StiffnessSample<'_>],
    cfg:     &AggregateStiffnessConfig,
) -> Vec<AggregatedContact>;
```

The molla equivalent of Newton's hashtable+voxel+normal-bin
reducer in `contact_reduction_hydroelastic.py`. Coarse bucketing
collapses many quadrature samples into one canonical contact per
`(position_voxel, normal_bin)` cell.

### Bucket key

Position voxel: positions are translated by the patch centroid
**before** voxelising (`hydroelastic.rs:778-782`). Without this,
a patch positioned at `(5, 0, 5)` would split into different
buckets from one at the origin, and a patch straddling `x = 0`
would split across `floor(-ε)` = -1 and `floor(+ε)` = 0 even when
the patch is much smaller than the voxel. Translation +
*round-based* binning (not `floor`) sidesteps both issues.

Normal bin: snap each axis component of the unit normal to a
direction-cosine grid with step `sin(angle)`. `round`, not
`floor`, again — `round` is symmetric around zero so a patch with
`n ≈ +Y` and small `±x/z` wobble collapses cleanly into bin
`(0, ±1, 0)` without straddling a grid edge near each axis.
Singularity-free at every axis-aligned direction (where
equirectangular grids would pole-out).

### Stiffness reduction

The two cases correspond to spring topologies:

* **Series** — contacts mounted in line (same line of action,
  stacked depth-wise). Compliance `c = Σ cᵢ = Σ 1/kᵢ`, so
  `k_agg = 1 / Σ (1/kᵢ)`. Used when collapsing duplicate contacts
  that touch the same rigid-body point through different paths.
* **Parallel** — contacts mounted side-by-side (independent
  patches sharing the load). `k_agg = Σ kᵢ`. Used when summing
  distinct contact area inside the same bucket — each sample
  contributes independent surface area; total resistance is the
  sum.

Newton's policy: parallel inside a bucket, series across buckets
when collapsing to a single anchor pair.

## GPU port

`shaders/hydroelastic.wgsl` mirrors `compute_hydroelastic_contact`
line-for-line: one thread per surface sample, transforms world →
soft-local, trilerps the SDF, computes pressure + central-
difference normal, writes the per-sample `(normal, point, force)`
to three output buffers:

```wgsl
contact_normal[i]       — world-frame outward normal (zero if no contact)
contact_point[i]        — world-frame contact point
contact_normal_force[i] — pressure · sample area (N) (zero if no contact)
```

The downstream Dahl-friction kernel
([`crate::kernels::DahlFrictionKernel`]) consumes these directly,
keyed on a host-provided `contact_id` for persistent bristle
state across substeps. The host wires the friction-coefficient
buffer and the bristle-state buffer separately — the
hydroelastic kernel itself only handles the pressure-field path.

The kernel uses an SOA layout (separate `sample_local_pos`,
`sample_local_normal`, `sample_area` arrays) to keep the SDF read
the dominant memory cost: each sample reads 8 SDF cells for the
trilerp plus 6 for the central-difference gradient, vs ~24 bytes
for the per-sample fixed data. SOA gives the hardware coalesced
loads on the per-sample arrays and lets the SDF reads dominate
profile.
