# Signorini contact — Lagrange multipliers

Stage 49 added **exact** non-penetration contact via Lagrange
multipliers, replacing the penalty-spring approximation of
[Stage 47 nonconforming penalty contact](./nonconforming.md). It
lives in `crates/molla/src/fem/signorini.rs`.

## Why Lagrange multipliers?

Penalty contact approximates the Signorini condition
\\( \text{gap}_n \ge 0 \\) with a soft spring
\\( \mathbf{f}_{\text{contact}} = -k \cdot \Delta u \\). The penalty matrix
\\( k \, C^T C \\) becomes **ill-conditioned** as \\( k \to \infty \\) —
larger \\( k \\) means stiffer constraint enforcement but harder linear
solves. In practice the penalty is capped well below \\( \infty \\),
which means some residual penetration is always present.

Lagrange multipliers replace the spring with **explicit contact-force
unknowns** \\( \lambda \ge 0 \\) that enforce the constraint exactly:

\\[
\begin{aligned}
\text{minimise}~& \tfrac{1}{2} u^T K u - f^T u \\
\text{subject to}~& C u \ge g \quad \text{(non-penetration)} \\
& \lambda \ge 0 \quad \text{(only push, never pull)}
\end{aligned}
\\]

KKT stationarity gives \\( K u - C^T \lambda = f \\). In matrix form
the linearised saddle-point system reads

\\[
\begin{pmatrix} K & -C^T \\ C & 0 \end{pmatrix}
\begin{pmatrix} u \\ \lambda \end{pmatrix} =
\begin{pmatrix} f \\ g \end{pmatrix}, \quad
\lambda \ge 0
\\]

— a linear complementarity problem (LCP).

## The Signorini contact data type

```rust
pub struct SignoriniContact {
    pub active_pairs: Vec<(u32, Vec3, Real)>,  // (slave_dof, n, gap)
}

impl SignoriniContact {
    pub fn new() -> Self;
    pub fn n_active(&self) -> usize;
    pub fn pairs(&self) -> &[(u32, Vec3, Real)];
}
```

Each tuple is `(slave_dof_global, master_normal, target_gap)`:

- `slave_dof_global`: BASE global DoF index of the slave vertex
  (i.e. `vertex_index * dim`). The constraint matrix writes the
  normal components into the three consecutive DoFs starting here.
- `master_normal`: unit outward normal of the master plane at the
  contact point.
- `target_gap`: minimum allowed projected displacement. If the
  initial geometry penetrates by \\( -p < 0 \\), set
  `target_gap = p` so that
  \\( \mathbf{n} \cdot u_{\text{slave}} \ge p \\) restores contact at
  zero gap. For nodes that are *on* the master surface, `target_gap = 0`.

## Constraint matrix `C`

```rust
pub fn assemble_constraint_matrix(
    active_pairs: &[(u32, Vec3, Real)],
    n_dofs: usize,
) -> Result<BsrMatrix>;

pub fn target_gap_vector(active_pairs: &[(u32, Vec3, Real)]) -> Vec<Real>;
```

Shape: `n_active × n_dofs`. Row \\( k \\) corresponds to active pair
\\( k \\): three entries at columns
`(slave_dof, slave_dof+1, slave_dof+2)` equal to
\\( (n_x, n_y, n_z) \\). All other entries are zero.

## Projected-Uzawa solver

The Uzawa iteration is the textbook **dual ascent + inner-primal-solve**
scheme:

```text
1. Inner primal:  fix λ, solve  K u = f + Cᵀ λ  via CG.
2. Dual update:   λ ← max(0, λ + ρ · (g − C u))  (projected gradient ascent).
3. Iterate until ‖g − C u‖∞ < tol AND λ has stabilised.
```

```rust
pub fn solve_augmented_system(
    k_global: &BsrMatrix,
    c_matrix: &BsrMatrix,
    f: &[Real],
    g: &[Real],
    max_iters: usize,
    tol: Real,
) -> Result<UzawaResult>;

pub fn solve_augmented_system_with(
    k_global, c_matrix, f, g, max_iters, tol, rho,
    inner_cg_tol, inner_cg_max_iters,
) -> Result<UzawaResult>;

pub struct UzawaResult {
    pub u: Vec<Real>,
    pub lambda: Vec<Real>,
    pub iters: usize,
    pub primal_residual: Real,    // ‖g − C u‖∞ on active constraints
    pub dual_residual: Real,      // ‖λ_new − λ_old‖∞
}
```

The implementation warm-starts \\( u \\) from the unconstrained solve
\\( K u = f \\) (saves the first inner CG) and **reuses the previous
iterate** as the warm start for each subsequent inner CG. The primal
residual measures only the **active violation** — pairs where
\\( \lambda > 0 \\) care about both signs of \\( g - C u \\), while
pairs where \\( \lambda = 0 \\) (currently inactive) only care about
*positive* residual (still penetrating).

## Choice of dual step `ρ`

For SPD \\( K \\), the dual functional is concave with Lipschitz
gradient bounded by the largest eigenvalue of the Schur complement
\\( S = C K^{-1} C^T \\). Convergence requires
\\( 0 < \rho < 2 / \lambda_{\max}(S) \\). The default is

\\[
\rho = 1 / \lambda_{\max}(S),
\\]

estimated via a short **power iteration** that needs only matrix-free
actions of \\( K^{-1} \\) (via inner CG) and \\( C \\). The
`estimate_dual_step` helper runs 6 power-iteration steps to nail
\\( \lambda_{\max} \\) to ~10 % — the inverse is a *conservative*
step (always inside the \\( (0, 2/\lambda_{\max}) \\) stability band,
never optimal but never oscillating):

```rust
pub fn estimate_dual_step(
    k: &BsrMatrix,
    c: &BsrMatrix,
    n_active: usize,
    inner_cg_tol: Real,
    inner_cg_max_iters: usize,
) -> Real;
```

Seeded with a deterministic length-`n_active` ramp (not the all-ones
vector, which can lock onto a translational rigid-body mode after
BCs).

## Stage 49 scope — fixed active set

The v1 of Stage 49 solves the augmented system for a **fixed list of
active contact pairs**. The caller decides which pairs are active
(typically: any slave node currently penetrating the master surface).

Once Uzawa converges, you have:

- \\( u \\): displacements that satisfy \\( C u \ge g \\) on the active
  pairs.
- \\( \lambda \ge 0 \\): contact pressure magnitudes (each pointing
  along its master normal). A pair whose \\( \lambda \\) clamps to 0
  is a **candidate for active-set release** — it has no contact force
  and may have separated from the master.

## Stage 51 — active-set update loop (planned)

The natural follow-up is the **active-set update** loop — the
"primal-dual active-set" (PDAS) algorithm:

1. Initialise active set from a contact-detection pass.
2. Solve the augmented system on the current active set (Stage 49).
3. **Release** pairs where \\( \lambda = 0 \\) and \\( C u > g \\)
   (separated).
4. **Activate** pairs where the geometry penetrates and \\( \lambda \\)
   would be positive (new contact).
5. Re-solve.
6. Iterate until the active set stops changing.

This is the LCP variant of the Uzawa-Lagrange method and is what
Wriggers (*Computational Contact Mechanics*, §6.3) calls the standard
"semi-smooth Newton" formulation. Status: **planned** — the code for
the active-set update logic hasn't landed yet; the Stage 49 v1
solver returns the multipliers and primal residual that the v2 / v3
update would key off.

## References

- Wriggers, *Computational Contact Mechanics*, §6.3 (Uzawa-Lagrange).
- Newton's `signorini.py` (Nvidia) for the active-set logic that
  will come in v2.
- Warp's `xpbd_contact.py` for the position-based contact analogue
  on the rigid-body side.

## Cross-references

- [Nonconforming](./nonconforming.md) — the penalty-contact precursor
  that `NonconformingPenaltyContact` ships.
- [Friction](./friction.md) — Coulomb friction layered on top of the
  Signorini multipliers.
- [Solvers](./solvers.md) — `cg_solve` is the inner primal solver.
- [BC](./bc.md) — strong Dirichlet on the *parent* \\( K \\) is
  applied before handing to the Uzawa loop; the routine assumes
  \\( K \\) is SPD after BCs.
