# VBD

[`VbdSolver`](https://docs.rs/molla/latest/molla/solvers/struct.VbdSolver.html)
is molla's **Vertex Block Descent** soft-body solver — a true
implicit-Euler step solved by per-vertex 3×3 Newton blocks in a
graph-coloured Gauss-Seidel sweep. Stage 20 onward.

References:

- Chen, Liu, Yang, Yuksel (2024). *Vertex Block Descent.* SIGGRAPH.
- Giles, Diaz, Yuksel (2025). *Augmented Vertex Block Descent.*
  ACM Trans. Graph. 44(4).

```rust,ignore
use molla::solvers::{VbdSolver, VbdMode, Solver};

let mut solver = VbdSolver::default()                  // CPU
    // .new_gpu(&engine, particle_count, spring_count)?
    .with_iterations(8)
    .with_substeps(2)
    .with_mode(VbdMode::Soft)
    .with_contact_stiffness(1.0e5);

solver.step(&model, &state_in, &control, &contacts, dt, &mut state_out)?;
```

## Algorithm

Implicit Euler on the soft-body equations of motion: each substep
solves

```text
(M/dt² + Σ_e H_e(x)) · Δx = M/dt² · (y − x) + Σ_e f_e(x)
```

where:

- `M` is the lumped mass matrix.
- `y = x_n + dt·(v_n + dt·g)` is the inertial target.
- `H_e`, `f_e` are the per-element Hessian + force contributions
  (spring elasticity, tet Neo-Hookean, cable bending, …).

VBD's trick is to do this with **per-vertex Gauss-Seidel sweeps**: at
each outer iteration, every vertex assembles the 3×3 row of the
Hessian + residual that it owns, inverts the 3×3 block locally, and
writes back `x_v ← x_v + H_v⁻¹ · g_v`. Within a graph-coloured
class, no two vertices share a constraint, so the sweep is **race-free
in WGSL**.

```text
for outer_iter in 0..iterations:
    for colour in 0..particle_color_count:
        parallel for vertex v in particles_in_colour(colour):
            (H_v, g_v) = (zero, zero)
            # springs adjacent to v
            for spring in vertex_adjacent_springs[v]:
                accumulate (H, g) from spring elasticity
            # tets adjacent to v
            for (tet, v_order) in vertex_adjacent_tets[v]:
                accumulate (H, g) from Neo-Hookean Ψ
            # cable bending hinges adjacent to v
            for entry in vertex_cable_bending[v]:
                accumulate (H, g) from discrete hinge energy
            # inertial target
            H_v += M_v / dt²
            g_v += M_v / dt² · (y_v − x_v)
            x_v += H_v⁻¹ · g_v
```

## Why VBD vs XPBD or Style3D

|                 | XPBD                 | VBD                    | Style3D (Newton ref) |
|-----------------|----------------------|------------------------|----------------------|
| Step type       | position projection  | implicit Euler         | projective dynamics  |
| Stiffness scale | non-physical (iters → stiffer) | **physical** | physical             |
| Cloth + soft    | cloth, particle      | cloth + tet            | cloth only           |
| Rigid coupling  | rigid-via-equality   | AVBD (soft / hard)     | none                 |
| GPU port        | mature               | mature                 | not ported in molla  |
| Cable joint     | particle polyline    | particle + bend hinges | n/a                  |

molla picked **VBD** over Style3D in Stage 20 because:

1. AVBD coupling unlocks soft-on-rigid demos (cloth wrapping a
   kinematic sphere, jelly resting on a box) that Style3D can't do.
2. The vertex-block structure ports naturally onto WGSL — every
   vertex is a small 3×3 Newton block dispatched in colour passes
   that mirror the existing XPBD particles kernel.
3. The CABLE joint roadmap (Stage 20.1–20.2) needs the bending
   energy infrastructure VBD already has.

## Cloth + spring elasticity

VBD reuses the **same** spring storage as XPBD — `Model.spring_*` with
compliance `α = 1/k` — and the **same** per-vertex CSR
(`vertex_adjacent_springs`). The energy is

```text
Ψ(x) = ½ · k · (|x_b − x_a| − rest)²
```

with `k = 1 / max(α, ε)`, capped at `solver.max_spring_stiffness` to
avoid pathological Hessian conditioning. The per-vertex 3×3 Hessian
contribution is

```text
H_aa = k · (n̂·n̂ᵀ + (1 − rest/d) · (I − n̂·n̂ᵀ))
H_ab = -H_aa
g_a  = k · (d − rest) · n̂
```

with `d = |x_b − x_a|` and `n̂ = (x_b − x_a) / d`.

## Tet Neo-Hookean (Stage 20.2)

Per tet, the deformation gradient is `F = D_s · D_m⁻¹` with `D_s` the
current basis (`[x1−x0 | x2−x0 | x3−x0]`) and `D_m⁻¹` pre-inverted at
build time. The energy density is co-rotated Neo-Hookean:

```text
Ψ(F) = μ · (‖F‖² − 3) − 2μ · log J + ½ · λ · log² J     (J = det F)
```

Per-vertex contributions are scattered via the
`vertex_adjacent_tets` CSR (one `(tet_id, v_order)` pair per
adjacency) so the kernel fetches the right column of `D_m⁻¹` (the
gradient direction `m` for that vertex) and assembles the 3×3 block
without branches.

### Plasticity (Stage 20.3 + 20.4)

Multiplicative split `F = F_e · F_p` with **von Mises return mapping**:

- If `‖dev(τ_trial)‖ ≤ σ_y(ε_p)` → elastic, `F_p` unchanged.
- Otherwise → return-map onto the yield surface; accumulate `ε_p`;
  update `F_p`.

`σ_y(ε_p) = σ_y0 + H · ε_p` is the linear isotropic hardening rule.
**Kinematic hardening** (Stage 20.4) adds a per-tet back-stress
tensor `α` via the Prager / Ziegler evolution

```text
Δα = c · (2/3) · n̂ · Δγ
```

producing the **Bauschinger effect** (reverse-yield drop after
forward yielding) under cyclic loading. State lives on `State`:

```rust,ignore
pub tet_f_plastic:      Buffer<Mat3>,     // F_p, identity at start
pub tet_plastic_strain: Buffer<Real>,     // ε_p accumulator
pub tet_back_stress:    Buffer<Mat3>,     // α, zero at start
```

## Augmented VBD: soft + hard contact

```rust,ignore
pub enum VbdMode {
    /// Quadratic penalty `½ · k_c · pen²` per contact.
    Soft,
    /// Augmented-Lagrangian dual update — drives penetration → 0
    /// independent of contact stiffness.
    Hard,
}
```

**Soft mode** folds a per-contact penalty force / Hessian into the
per-vertex block. Cheap, allows ~1 mm steady-state penetration at
typical stiffness `k_c = 1 × 10⁵ N/m`.

**Hard mode** adds an augmented-Lagrangian dual variable `λ` per
particle. Each outer iteration:

```text
λ_v ← max(0, λ_v + k_c · pen_v)
g_v += λ_v · n
```

This drives `pen → 0` regardless of `k_c`. The cost is one extra
per-particle scratch buffer + a residual-evaluation pass per outer
iteration. **Use Hard mode when**:

- No-penetration is critical (a sealed cloth on a rigid body —
  fingertip grasp).
- Contact stiffness is too low for soft mode to keep penetration in
  spec.

## SDF-backed rigid contact (Stage 20.4)

For arbitrary-mesh rigid colliders, the VBD pipeline accepts a
**baked SDF backing volume**:

```rust,ignore
solver.with_sdf_grid(arc_sdf, world_transform);          // dense 3D texture
solver.with_sparse_sdf(arc_sparse_sdf, world_transform); // sparse NanoVDB-style
```

The post-substep fixup pass projects every soft-body vertex out of
any registered SDF surface, using the SDF's analytic normal +
distance. Stage 20.5 will wire this into the GPU contact kernel
directly via a dedicated `tet_sdf_contact` pipeline.

## Cross-mesh soft contact (Stage 33)

Two-way soft–soft VT (vertex–triangle) + EE (edge–edge) contact
between separately-registered soft surfaces (cloth + tet boundary
tris). The pipeline:

1. **Build** the cross-mesh BVH over `soft_surface_tris` (lazily,
   refit per substep, rebuilt on topology change).
2. **Gather** VT and EE candidate pairs via BVH traversal.
3. **Project** penetrating pairs in `vt_contact_sweeps` Jacobi sweeps
   per substep.

Per-surface `self_collision` toggles whether a vertex can collide
with triangles on its own surface (CSR `soft_vertex_incident_tris`
filters out the trivial "vertex is one of the triangle's corners"
cases).

Stage 33.4 adds **CCD pre-substep** for cloth–cloth pairs travelling
faster than `vt_contact_margin / dt`.

## Cable bending (Stage 20.2)

Per-vertex CSR `vertex_cable_bending_*` stores
`(v_prev, v_mid, v_next, role)` triples for every interior cable
particle. The discrete hinge energy

```text
E_b = ½ · k_b · (2·sin(θ/2))²       (θ = angle at v_mid)
```

contributes to the per-vertex Hessian + gradient. Setting
`set_cable_bending_stiffness(cable_id, 0)` (the default) disables
bending and the cable behaves as a pure stretch polyline.

## GPU port

| kernel | shader | role |
|--------|--------|------|
| `VbdClothKernel` | [`vbd_cloth.wgsl`](https://github.com/molla/molla/blob/main/crates/molla-kernels/shaders/vbd_cloth.wgsl) | per-vertex Newton block sweep |
| `SoftBvhGpu` | (BVH refit shader) | cross-mesh BVH refit |
| `SoftVtContactKernel` | (VT/EE pipelines) | cross-mesh contact gather + project |

All three are lazily constructed — scenes without soft surfaces pay
zero overhead. The vertex-block sweep iterates one colour pass per
dispatch, with within-colour parallel vertex updates.

## Configuration

```rust,ignore
pub struct VbdSolver {
    pub iterations: usize,           // outer Newton iters (default 8)
    pub substeps: usize,             // (default 1)
    pub contact_stiffness: f64,      // N/m (default 1e5)
    pub max_spring_stiffness: f64,
    pub vbd_mode: VbdMode,           // Soft / Hard
    pub vt_contact_margin: Real,
    pub vt_contact_sweeps: usize,
    pub sdf_backings: Vec<VbdSdfBacking>,
    /* … */
}
```

`iterations = 4` is the absolute minimum (visible jelly behaviour);
`8–12` is comfortable; `20+` is overkill outside research.

## Differentiability

VBD's per-vertex Newton block is locally smooth; the reverse pass is
exposed via [`crate::autograd::vbd_adjoint`](../autograd/xpbd_vbd.md).
Stage 23.x ships the cloth + spring + tet adjoint; contact gradients
follow the soft-penalty smoothing pattern.
