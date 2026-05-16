# XPBD

[`XpbdSolver`](https://docs.rs/molla/latest/molla/solvers/struct.XpbdSolver.html)
is molla's **eXtended Position-Based Dynamics** solver — the workhorse
for rigid contacts, distance / bending constraints, equality
constraints (`CONNECT` / `WELD`), particles, and ropes.

Reference: Müller, Macklin, Chentanez, Jakob, Kim,
*Detailed Rigid Body Simulation with Extended Position-Based Dynamics*
(Eurographics 2020).

```rust,ignore
use molla::solvers::{XpbdSolver, Solver};

let mut solver = XpbdSolver::new_cpu()                  // or new_gpu(&engine, body_count)
    .with_iterations(4)        // position iterations per substep
    .with_substeps(1);

pipeline.collide(&mut contacts, &state, &model)?;
solver.step(&model, &state_in, &control, &contacts, dt, &mut state_out)?;
```

## Algorithm

Each substep:

1. **Predict** positions + rotations from velocities + gravity
   (semi-implicit-ish, like [SemiImplicit](./semi_implicit.md)).
2. **Position projection** — for every constraint, project the
   predicted state toward feasibility:
   - Rigid contacts (per-body Jacobi).
   - Distance constraints (cloth edges, rope segments).
   - Spring strain limits (post-soft non-linear clamp).
   - Equality constraints (`CONNECT` / `WELD`).
   - Cable bending hinges.
   - Particle self-collision via spatial hash.
3. **Velocity recovery** — linear from `(p_new − p_old) / dt`, angular
   from the small-rotation extracted from the quaternion delta.
4. **Velocity projection** — restitution + Coulomb friction along
   contact normals.

## Per-body Jacobi projection

The rigid-body contact projection is the central novelty over plain
PBD. For each contact `(a, b, p, n, depth)`:

```text
# Lever arms from each body's COM in world to the contact point:
r_a = p − (body_q[a].pos + body_q[a].rot · body_com[a])
r_b = p − (body_q[b].pos + body_q[b].rot · body_com[b])

# Generalised inverse mass: linear + (r × n)ᵀ I⁻¹ (r × n) angular.
w_a = inv_mass_a + (r_a × n)ᵀ · I⁻¹_a · (r_a × n)
w_b = inv_mass_b + (r_b × n)ᵀ · I⁻¹_b · (r_b × n)
λ   = -depth / (w_a + w_b)

# Linear push apart:
body_q[a].pos −= n · (λ · inv_mass_a)
body_q[b].pos += n · (λ · inv_mass_b)

# Angular correction via small-rotation update:
Δθ_a = I⁻¹_a · (r_a × n) · λ
Δθ_b = I⁻¹_b · (r_b × n) · λ
body_q[a].rot = small_rot_apply(body_q[a].rot, Δθ_a)
body_q[b].rot = small_rot_apply(body_q[b].rot, Δθ_b)
```

The `(r × n)` term is the **angular Jacobian** of the contact
constraint. Without it, contacts wouldn't transmit torque — a ball
landing off-centre on a box wouldn't make the box rotate.

### Why "Jacobi" (not Gauss-Seidel)?

The GPU kernel iterates contacts **per body** in parallel: each thread
writes to its own body's pose. Two contacts that share a body would
write-race in Gauss-Seidel; Jacobi avoids this by accumulating
position / orientation deltas across all of a body's contacts in a
per-thread accumulator before committing. This sacrifices the per-
iteration convergence of Gauss-Seidel but is **fully parallel** on
GPU.

The CSR `body_contact_offsets` + `body_contact_indices` (built by the
collision pipeline) gives each body O(1) access to its contact list,
with the sign-bit trick `B_SIDE_BIT = 0x8000_0000` distinguishing the
`a` / `b` side without per-body lookups.

## Distance, spring, cable bending

For each spring `(a, b, rest, α, max_stretch)`:

```text
d = |p_b − p_a|; n = (p_b − p_a) / d
C = d − rest
λ = −C / (w_a + w_b + α / dt²)               (XPBD time-step rescaling)
Δp_a = -λ · w_a · n
Δp_b = +λ · w_b · n
```

The XPBD compliance `α` (m/N) replaces the iteration-dependent
"effective stiffness" of plain PBD: same `α` → same physical
stiffness regardless of iteration count.

**Strain limit**: after the soft pass, a separate non-linear projection
caps `|d − rest| ≤ rest · max_stretch`. Setting `max_stretch = 0`
makes the spring perfectly inextensible regardless of `α`.

**Bending hinges** (cable polyline, Stage 20.2): a per-triple
`(v_prev, v_mid, v_next)` discrete hinge energy
`E_b = ½ · k_b · (2·sin(θ/2))²` is folded in via the per-vertex CSR
`vertex_cable_bending_*`.

## Equality (CONNECT / WELD)

```text
CONNECT: 3 positional constraints between two anchor points.
WELD:    6 constraints — position + orientation.
```

The post-substep equality pass solves them via XPBD compliance: stiff
default (`α ≈ 1e-7`) for the rigid case, softer for breakable welds.
The GPU `XpbdEqualityKernel` runs the projection.

## Particle self-collision

For dense particle clouds (cloth, gels) the solver runs a
**spatial-hash broad phase** over particles each substep, then
projects pairs whose centre distance is below `2 · radius`:

```text
For each particle pair (i, j) with |p_i − p_j| < r_i + r_j:
    pen = r_i + r_j − |p_i − p_j|
    n = (p_j − p_i) / |...|
    Δp_i = -pen · w_i / (w_i + w_j) · n
    Δp_j = +pen · w_j / (w_i + w_j) · n
```

The spatial-hash grid resolution matches the median particle radius;
particles with `ParticleFlags::ACTIVE = 0` are skipped. The
[`crate::geometry::SpatialHash`](../geometry/spatial_hash.md) is the
shared primitive (also used by MPM and the soft VT contact pipeline).

## Configuration

```rust,ignore
pub struct XpbdSolver {
    pub position_iterations: usize,    // default 4
    pub substeps: usize,               // default 1
    backend: XpbdBackend,
}

impl XpbdSolver {
    pub fn new_cpu() -> Self;
    pub fn new_gpu(engine: &Engine, body_count: u32) -> Result<Self>;
    pub fn with_iterations(self, n: usize) -> Self;
    pub fn with_substeps(self, n: usize) -> Self;
    pub fn is_gpu(&self) -> bool;
}
```

- `position_iterations`: more iterations stiffen multi-body stacks
  (each iter pushes one contact-graph hop further apart). Default 4
  works for ~5-high stacks; raise to 8–12 for tall stacks.
- `substeps`: internal substep count. **Limitation**: the contact set
  is fixed across all substeps (no re-collision). For high-rate
  contact workloads, the external substep loop is the right pattern
  (see [overview](./overview.md#sub-stepping-pattern)).

## GPU kernels

| kernel | shader | role |
|--------|--------|------|
| `XpbdKernel` | [`xpbd.wgsl`](https://github.com/molla/molla/blob/main/crates/molla-kernels/shaders/xpbd.wgsl) | rigid-body predict + per-body Jacobi contact projection + velocity recovery |
| `XpbdEqualityKernel` | [`xpbd_equality.wgsl`](https://github.com/molla/molla/blob/main/crates/molla-kernels/shaders/xpbd_equality.wgsl) | CONNECT / WELD post-step projection |
| `XpbdParticlesKernel` | [`xpbd_particles.wgsl`](https://github.com/molla/molla/blob/main/crates/molla-kernels/shaders/xpbd_particles.wgsl) | particles + springs + strain limit + plane collision + velocity recovery |

The particle kernel is **lazily constructed** on the first step that
has particles, so rigid-only models pay no compile cost. Inside the
particle kernel the spring colour passes are dispatched serially
(one pass per colour class), with the within-colour parallel reads /
writes that the colouring guarantees are race-free.

## Substepping

`with_substeps(n)` runs `n` inner substeps of `dt/n`, ping-ponging
between `state_out` and a host-side scratch state. Each substep uses
the same caller-supplied contact set — fine for slow-contact
workloads (a Free-flying body on a static floor), but for fast-
contact stacks the right pattern is external (collide + step in a
loop). See `tests/xpbd_boxes.rs::external_substeps_box_stack` for the
recommended pattern.

## Differentiability (Stage 23.x)

XPBD's position-projection passes are piecewise-differentiable through
the soft Lagrangian formulation. The forward-mode tape records each
constraint's projection; reverse pass walks contributions back through
the Lagrange multipliers. See [XPBD adjoint](../autograd/xpbd_vbd.md).
