# Coulomb friction (Stage 50)

Stage 50 is the follow-up to [Stage 49 Signorini contact](./signorini.md):
once the **normal** force \\( \lambda_n \\) is known via Lagrange
multipliers, **tangential** friction follows a Coulomb law that
constrains the friction force to lie inside a per-contact friction
disc.

The FEM-side implementation extends `crates/molla-fem/src/signorini.rs`
with `solve_augmented_system_with_friction` + `ContactPair` (carrying
the slave DoF, master normal, and an orthonormal tangent basis). Stage 50 v2
also exposes `FrictionState` plus separate static/kinetic coefficients
(`μ_s`, `μ_k`), while `with_friction(μ)` keeps the old `μ_s == μ_k` behaviour.
The 3-component multiplier per pair (1 normal + 2 tangents) shares a single
projected-Uzawa loop with the Stage 49 normal-only path. Particle/cloth-
side Coulomb friction has separate implementations in
`crates/molla-solvers/src/vbd_soft_contact.rs` and
`crates/molla-kernels/src/soft_vt_contact.rs` (position-form analogue
on different state types).

## The Coulomb friction model

For each active contact pair with normal multiplier \\( \lambda_n \ge 0 \\),
static friction coefficient \\( \mu_s \ge 0 \\), and kinetic friction coefficient
\\( \mu_k \ge 0 \\):

- **Sticking** (no slip):
  \\( |\boldsymbol\lambda_t| \le \mu_s \lambda_n \\) with the tangential
  velocity \\( v_t = 0 \\).
- **Sliding** (kinetic):
  \\( |\boldsymbol\lambda_t| = \mu_k \lambda_n \\) and
  \\( \boldsymbol\lambda_t = -\mu_k \lambda_n \, \hat v_t \\) (opposes
  motion).

The friction force lies inside the **Coulomb friction cone** of the
contact:

```text
                            λ_n
                            ↑
                            │
                            │   ─── static cone (μ_s slope)
                            │  /
                            │ /
                            │/_______________→
                                                |λ_t|
            (λ_t, λ_n) feasible iff
            |λ_t| ≤ μ_s · λ_n  (stick)
            |λ_t| = μ_k · λ_n  (slip)
```

For an FEM contact with 3D normal \\( \mathbf{n} \\) and two
orthonormal in-plane tangents \\( \mathbf{t}_1, \mathbf{t}_2 \\), the
constraint becomes: \\( (\lambda_{t_1}, \lambda_{t_2}) \\) lies inside
the **static disc** of radius \\( \mu_s \lambda_n \\), or on the kinetic
boundary \\( \mu_k \lambda_n \\) once it slips.

## Algorithmic plan

The textbook semi-smooth Newton recipe (Wriggers, §6.5):

### 1. Tangent constraint matrix

In addition to the normal-direction constraint matrix
\\( C_n \\) (one row per active pair), build the **tangent constraint
matrix** \\( C_t \\) with **two rows per active pair** — one for each
tangent direction:

```text
C_t[2k    , slave_dof+0..+2] = (t1.x, t1.y, t1.z)
C_t[2k + 1, slave_dof+0..+2] = (t2.x, t2.y, t2.z)
```

The tangents \\( (\mathbf{t}_1, \mathbf{t}_2) \\) are picked
deterministically: \\( \mathbf{t}_1 \\) is a fixed projection of
\\( \mathbf{e}_x \\) onto the plane orthogonal to \\( \mathbf{n} \\)
(or \\( \mathbf{e}_y \\) if \\( \mathbf{n} \approx \pm \mathbf{e}_x \\)),
and \\( \mathbf{t}_2 = \mathbf{n} \times \mathbf{t}_1 \\).

### 2. Augmented system

\\[
\begin{pmatrix} K & -C_n^T & -C_t^T \\ C_n & 0 & 0 \\ C_t & 0 & 0 \end{pmatrix}
\begin{pmatrix} u \\ \boldsymbol\lambda_n \\ \boldsymbol\lambda_t \end{pmatrix} =
\begin{pmatrix} f \\ g_n \\ g_t \end{pmatrix}
\\]

with \\( g_t = 0 \\) for the sticking case (zero tangential slip).

### 3. Augmented Uzawa with friction-cone projection

Each Uzawa outer iteration:

1. Inner primal: \\( K u = f + C_n^T \boldsymbol\lambda_n + C_t^T \boldsymbol\lambda_t \\)
   via CG.
2. Normal dual update: \\( \lambda_n \leftarrow \max(0, \lambda_n + \rho_n (g_n - C_n u)) \\)
   — same as Signorini.
3. Tangent dual update: \\( \boldsymbol\lambda_t \leftarrow \boldsymbol\lambda_t + \rho_t (g_t - C_t u) \\),
   then **project onto the friction disc** using the static/kinetic rule:

   \\[
   \boldsymbol\lambda_t \leftarrow
   \begin{cases}
   \boldsymbol\lambda_t & \text{if } |\boldsymbol\lambda_t| \le \mu_s \lambda_n \\
   \mu_k \lambda_n \cdot \dfrac{\boldsymbol\lambda_t}{|\boldsymbol\lambda_t|} & \text{otherwise}
   \end{cases}
   \\]

4. Loop until both primal and dual residuals are below tolerance.

The friction-disc projection is what makes the scheme **semi-smooth**
— it's continuous but not differentiable on the disc boundary, which
is why classical Newton fails but Uzawa's projected gradient works.

### 4. Static vs kinetic friction

Static and kinetic friction coefficients differ in practice
(\\( \mu_s > \mu_k \\) — it takes more force to *start* slipping than
to *sustain* it). The textbook model treats this as two regimes:

- **Stick** (no slip): \\( |\boldsymbol\lambda_t| \le \mu_s \lambda_n \\),
  any value inside the disc is admissible. The projection step uses
  \\( \mu_s \\).
- **Slip** (kinetic): \\( |\boldsymbol\lambda_t| = \mu_k \lambda_n \\),
  opposing the slip velocity. The projection step uses \\( \mu_k \\).

A per-contact `FrictionState` records the final projection regime
(`Inactive`, `Stick`, or `Slip`). The current implementation classifies
directly from the projected dual update; it does not yet add a
velocity-threshold hysteresis loop.

## Expected API

When the module lands, the signatures will look like:

```rust
pub struct ContactPair {
    pub slave_dof: u32,
    pub normal: Vec3,
    pub tangent_u: Vec3,
    pub tangent_v: Vec3,
    pub target_gap: Real,
}

pub struct SignoriniContact {
    pub active_pairs: Vec<ContactPair>,
    pub friction_static_coeff: Real,
    pub friction_kinetic_coeff: Real,
}

pub fn solve_augmented_system_with_friction(
    k_global: &BsrMatrix,
    contact: &SignoriniContact,
    f: &[Real],
    max_iters: usize,
    tol: Real,
) -> Result<FrictionUzawaResult>;

pub fn solve_active_set_signorini_with_friction(
    k_global: &BsrMatrix,
    candidates: &SignoriniContactCandidates,
    f: &[Real],
    friction_static_coeff: Real,
    friction_kinetic_coeff: Real,
    max_outer_iters: usize,
    gap_tol: Real,
    inner_max_iters: usize,
    inner_tol: Real,
) -> Result<ActiveSetFrictionResult>;
```

The result returns both \\( u \\), \\( \boldsymbol\lambda_n \\), and
\\( \boldsymbol\lambda_t \\) (the projected tangential forces),
plus the per-pair stick/slip flag for diagnostics.

The active-set variant is the coupled Stage 49/50 path: the PDAS loop
discovers the current normal active set, and each inner solve uses the
stacked normal+tangent Coulomb system. Released contacts zero their
normal multiplier, tangent multipliers, and `FrictionState` in one
outer iteration, so friction is no longer a post-pass over a
preselected contact set.

## Rate-and-state friction

For advanced tribology / geology workloads, the FEM crate also exposes
the Dieterich-Ruina aging law as reusable per-contact helpers:

```rust
pub struct RateStateFrictionParams {
    pub mu0: Real,
    pub a: Real,
    pub b: Real,
    pub v0: Real,
    pub dc: Real,
}

pub struct RateStateFrictionState {
    pub theta: Real,
}

pub fn rate_state_friction_coeff(
    params: RateStateFrictionParams,
    state: RateStateFrictionState,
    slip_rate: Real,
) -> Real;

pub fn advance_rate_state_friction(
    params: RateStateFrictionParams,
    state: RateStateFrictionState,
    slip_rate: Real,
    dt: Real,
) -> (RateStateFrictionState, Real);
```

The coefficient is
`μ = μ0 + a ln(v/v0) + b ln(θ v0 / dc)`, and the state evolves with
the aging law `dθ/dt = 1 - vθ/dc`. The helpers are intentionally
solver-agnostic: Signorini, VBD, or particle contact pipelines can
own the state vector and feed the updated coefficient into their
existing friction-cone projection.

## What's already in the codebase

Friction is implemented (and tested) for these non-FEM paths:

| Module                                    | Friction model |
|-------------------------------------------|----------------|
| `solvers/vbd_soft_contact.rs`             | Position-form Coulomb friction inside VBD substep |
| `kernels/soft_vt_contact.rs`              | Position-form Coulomb friction for soft-vs-tri contact |
| `geometry/hydroelastic.rs`                | Default `μ` per hydroelastic patch |
| `sim/model.rs`, `sim/builder.rs`          | Per-shape `μ` storage + builders |
| `compute/implicit_mpm.rs`                 | Coulomb friction for MPM particles |

These all live at the **rigid / particle / cloth** layer; the
FEM-multiplier version is the missing piece that will unify the
contact-friction story across all body types.

## Why this is non-trivial

Three reasons it isn't a simple copy-paste from particles:

1. **Direction-dependent tangent basis.** Picking
   \\( \mathbf{t}_1, \mathbf{t}_2 \\) consistently across active pairs
   matters when neighbouring contacts share slave nodes — a
   stick-to-slip transition shouldn't flip basis on you.
2. **The friction-disc projection is non-smooth.** Classical Newton
   needs semi-smooth or smoothed variants; Uzawa's projected gradient
   handles it natively but converges sub-linearly near the disc
   boundary.
3. **Coupling with normal dual update.** The friction limit
   \\( \mu \lambda_n \\) moves every time \\( \lambda_n \\) updates,
   so the tangent projection sees a moving target — the convergence
   analysis needs both updates to descend in lockstep.

The reference implementation in
[`xtra/newton/newton/solvers/`](https://github.com/NVIDIA/Newton)
(particle-side friction) and the textbook chapter in Wriggers cover
these details; the FEM-side port is a self-contained extension that
slots into the existing
[Signorini machinery](./signorini.md).

## Cross-references

- [Signorini](./signorini.md) — the normal-only contact this layers
  on top of.
- [Nonconforming](./nonconforming.md) — penalty-contact precursor for
  small-displacement bodies.
- [Solvers](./solvers.md) — the augmented-system solver framework
  the friction extension will plug into.
