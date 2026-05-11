# Coulomb friction (Stage 50)

Stage 50 is the follow-up to [Stage 49 Signorini contact](./signorini.md):
once the **normal** force \\( \lambda_n \\) is known via Lagrange
multipliers, **tangential** friction follows a Coulomb law that
constrains the friction force to lie inside a per-contact friction
disc.

The FEM-side implementation extends `crates/molla/src/fem/signorini.rs`
with `solve_augmented_system_with_friction` + `ContactPair` (carrying
the slave DoF, master normal, and an orthonormal tangent basis). The
3-component multiplier per pair (1 normal + 2 tangents) shares a single
projected-Uzawa loop with the Stage 49 normal-only path. Particle/cloth-
side Coulomb friction has separate implementations in
`crates/molla/src/solvers/vbd_soft_contact.rs` and
`crates/molla/src/kernels/soft_vt_contact.rs` (position-form analogue
on different state types).

## The Coulomb friction model

For each active contact pair with normal multiplier \\( \lambda_n \ge 0 \\)
and friction coefficient \\( \mu \ge 0 \\):

- **Sticking** (no slip):
  \\( |\boldsymbol\lambda_t| \le \mu \lambda_n \\) with the tangential
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
                            │   ─── friction cone (μ slope)
                            │  /
                            │ /
                            │/_______________→
                                                |λ_t|
            (λ_t, λ_n) feasible iff
            |λ_t| ≤ μ · λ_n  (stick)
            |λ_t| = μ · λ_n  (slip)
```

For an FEM contact with 3D normal \\( \mathbf{n} \\) and two
orthonormal in-plane tangents \\( \mathbf{t}_1, \mathbf{t}_2 \\), the
constraint becomes: \\( (\lambda_{t_1}, \lambda_{t_2}) \\) lies inside
the **disc** of radius \\( \mu \lambda_n \\).

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
   then **project onto the friction disc** of radius
   \\( \mu \lambda_n \\):

   \\[
   \boldsymbol\lambda_t \leftarrow
   \begin{cases}
   \boldsymbol\lambda_t & \text{if } |\boldsymbol\lambda_t| \le \mu \lambda_n \\
   \mu \lambda_n \cdot \dfrac{\boldsymbol\lambda_t}{|\boldsymbol\lambda_t|} & \text{otherwise}
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

A simple state-machine over each contact decides which regime applies:
once \\( |\boldsymbol\lambda_t| \\) reaches \\( \mu_s \lambda_n \\) the
contact transitions to slip; once \\( v_t \\) drops below a threshold
it transitions back to stick. In practice molla's particle-side
friction (`vbd_soft_contact.rs`) uses a single \\( \mu \\) and lets
the implicit regularisation handle the transition; the FEM version
will likely follow the same pattern.

## Expected API

When the module lands, the signatures will look like:

```rust
// Per-pair (slave_dof, normal, tangent1, tangent2, target_gap, mu)
pub struct FrictionContact {
    pub active_pairs: Vec<(u32, Vec3, Vec3, Vec3, Real, Real)>,
}

pub fn assemble_normal_constraint(active_pairs, n_dofs) -> Result<BsrMatrix>;
pub fn assemble_tangent_constraint(active_pairs, n_dofs) -> Result<BsrMatrix>;

pub fn solve_signorini_with_friction(
    k_global: &BsrMatrix,
    c_normal: &BsrMatrix,
    c_tangent: &BsrMatrix,
    f: &[Real],
    g_normal: &[Real],
    friction_coeffs: &[Real],   // one μ per active pair
    config: FrictionConfig,
) -> Result<FrictionResult>;
```

The result returns both \\( u \\), \\( \boldsymbol\lambda_n \\), and
\\( \boldsymbol\lambda_t \\) (the projected tangential forces),
plus the per-pair stick/slip flag for diagnostics.

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
