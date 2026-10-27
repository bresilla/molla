# Small-matrix decompositions

`crates/molla/src/math/dense3.rs` ships specialised decompositions for
**3×3 dense matrices** — SVD, QR, symmetric eigendecomposition, and
polar decomposition — plus the reverse-mode adjoint of SVD with
Tikhonov regularisation for degenerate singular values. These are the
small-matrix counterparts of [`dense_solve`](#larger-systems-dense_solve);
the 3×3 case dominates because it shows up once per particle, per
vertex, or per body in inner loops.

Callers:

- **IK regularisation** (Stage 16) — when `(JᵀJ + λI)` Cholesky fails
  or the condition number is huge, fall back to an SVD-based damped
  pseudoinverse.
- **Per-vertex polar decomposition** for VBD soft-body co-rotational
  energy (Stage 20).
- **Per-particle inertia eigendecomposition** for MPM plasticity
  (Stage 24d).

The matching WGSL kernel lives in `crates/molla-kernels/shaders/dense3_svd.wgsl`
and is dispatched by `crate::kernels::Dense3Kernel`.

## Algorithm choices — Jacobi vs Householder vs Givens

All three forward decompositions deliberately avoid **Householder
reflections** even though textbooks prefer them at scale: Householder
needs conditional vector compaction, which is painful in WGSL with no
warp-shuffle and only `workgroupBarrier()` for synchronisation. Jacobi
and Givens are loop-free and branch-light, so the host code and the
WGSL kernel share their structure.

- **SVD via one-sided Jacobi.** `svd3` rotates column pairs of the
  working matrix by Givens rotations chosen to zero the off-diagonal
  entries of `MᵀM`. After convergence, the columns of `M` are
  orthogonal; their norms are the singular values, normalised columns
  form `U`, and the accumulated rotations form `V`. Reference:
  Demmel & Veselić "Jacobi's Method is More Accurate than QR" (1992),
  Golub & Van Loan 3rd ed. §8.6.3.
- **QR via three Givens rotations.** Zero the (1,0), (2,0), (2,1)
  sub-diagonal entries in turn. Each Givens is its own transpose, so
  `R = G₃·G₂·G₁·M` and `Q = G₁ᵀ·G₂ᵀ·G₃ᵀ`.
- **Symmetric eigendecomposition via cyclic Jacobi.** Same rotation
  recipe as SVD but applied to a symmetric input directly. Reads the
  upper triangle and assumes the lower mirrors it.

Both Jacobi-style sweeps share two private constants:

```rust
const JACOBI_EPS: Real = 1.0e-14;
const JACOBI_MAX_SWEEPS: usize = 32;
```

`JACOBI_MAX_SWEEPS = 32` is a hard cap. Well-conditioned 3×3 inputs
converge in 4-6 sweeps; the cap guards against pathologies (matrices
with duplicate eigenvalues that cause Jacobi to spin on the rotation
choice). The cap is the same on the host and in the WGSL kernel so
parity tests don't see drift from differing iteration counts.

## Public API

```rust
pub fn svd3(m: Mat3) -> (Mat3, Vec3, Mat3);
pub fn qr3(m: Mat3) -> (Mat3, Mat3);
pub fn eig3_sym(m: Mat3) -> (Vec3, Mat3);
pub fn polar_decompose_3(m: Mat3) -> (Mat3, Mat3);
pub fn cond2_3(m: Mat3) -> Real;
```

- `svd3(m)` returns `(U, σ, V)` with `M = U · diag(σ) · Vᵀ`. Singular
  values are non-negative but **not sorted** — sort externally if
  needed. If the constructed `U` ended up as a reflection (`det(U) <
  0`), the routine flips the smallest-`σ` column's sign in both `U`
  and `V` so the factorisation still holds with non-negative `σ`.
- `qr3(m)` returns `(Q, R)` with `Q` orthogonal and `R` upper-triangular.
  The strict lower triangle of `R` is explicitly zeroed at the end so
  callers can rely on `R[i,j] == 0` for `i > j`.
- `eig3_sym(m)` returns `(λ, V)` such that `A = V · diag(λ) · Vᵀ`.
  Asymmetric input gives garbage — symmetrise at the call site.
  Eigenvalues are **not sorted**.
- `polar_decompose_3(m)` returns `(R, S)` with `R` orthogonal and `S`
  symmetric positive semi-definite. Implemented as a thin SVD wrapper:
  `R = U · Vᵀ`, `S = V · diag(σ) · Vᵀ`. If `det(M) < 0`, `R` ends up
  improper (`det(R) = -1`); VBD-style co-rotational energy normally
  wants a proper rotation, so callers that need one post-process the
  result.
- `cond2_3(m)` returns the 2-norm condition number `σ_max / σ_min`,
  or `Real::INFINITY` when `σ_min == 0`. IK uses this as the
  Cholesky-vs-SVD switch.

## `adj_svd3` — reverse-mode adjoint of SVD

```rust
pub fn adj_svd3(
    _a: Mat3,
    u: Mat3,
    sigma: Vec3,
    v: Mat3,
    adj_u: Mat3,
    adj_sigma: Vec3,
    adj_v: Mat3,
) -> Mat3;
```

The forward decomposition is `A = U · diag(σ) · Vᵀ`. Given the
forward outputs `(U, σ, V)` and the upstream gradients `(adj_U,
adj_σ, adj_V)` of some downstream scalar loss `J`, this returns
`adj_A = ∂J/∂A`. Three contributions get summed:

1. **Direct σ term.** `U · diag(adj_σ) · Vᵀ`.
2. **U-rotation term.** `Ω_U = Uᵀ · adj_U − adj_Uᵀ · U` (twice the
   skew part of `Uᵀ · adj_U`); `u_term = U · ((F ⊙ Ω_U) · Σ) · Vᵀ`.
3. **V-rotation term.** Symmetric to (2), with `Ω_V` and `Σ · (F ⊙ Ω_V)`.

The literal arithmetic mirrors `xtra/warp/warp/native/svd.h::adj_svd3`
(Apache-2.0); references for the math are McAdams et al. "Computing the
Singular Value Decomposition of 3×3 Matrices" (2011) for the forward,
Townsend "Differentiating the Singular Value Decomposition" (2016) for
the adjoint.

### Degenerate σ — Tikhonov regularisation

The adjoint divisor `F_ij = 1 / (σⱼ² − σᵢ²)` is undefined when two
singular values coincide. molla uses

```text
F_ij = (σⱼ² − σᵢ²) / max((σⱼ² − σᵢ²)², ε²)
```

with `ε = ADJ_SVD_EPS = 1.0e-12`. The result is bounded by `1/ε` for
all inputs and is exactly `1/(σⱼ² − σᵢ²)` away from the degeneracy.
At exact degeneracy the regularised `F_ij → 0`, which is the
mathematically correct limit — the rotational gauge between equal-σ
columns is genuinely ambiguous, and projecting the gradient onto the
symmetric part is what the limit gives you. The implementation lives
in the private helper `safe_inv_sq_diff`:

```rust
fn safe_inv_sq_diff(a: Real, b: Real, eps: Real) -> Real {
    let d = b * b - a * a;
    d / (d * d + eps * eps)
}
```

`adj_polar(f, r, s, adj_r, adj_s)` is the polar-decomposition
adjoint, expressed through SVD — it recovers `(U, σ, V)` from `F`,
propagates `adj_R = U·Vᵀ` and `adj_S = V·Σ·Vᵀ` back to `(adj_U,
adj_σ, adj_V)`, then hands off to `adj_svd3`. The `(r, s)` arguments
are unused in the math (they're recoverable from `F`) but kept in the
signature for parity with the autograd convention.

## Larger systems — `dense_solve`

For `n × n` problems beyond 3×3 — typically the IK normal equations
`(JᵀJ + λI) · dq = Jᵀ · e` with `dof` up to ~30 — see
`crates/molla/src/math/dense_solve.rs`:

```rust
pub fn dense_chol(a: &mut [Real], n: usize) -> Result<()>;
pub fn forward_sub(l: &[Real], b: &[Real], n: usize) -> Result<Vec<Real>>;
pub fn back_sub(l: &[Real], y: &[Real], n: usize) -> Result<Vec<Real>>;
pub fn dense_chol_solve(a: &mut [Real], b: &[Real], n: usize) -> Result<Vec<Real>>;
pub fn dense_eig_sym(a: &[Real], n: usize) -> Result<(Vec<Real>, Vec<Real>)>;
pub fn dense_psd_pinv_solve(a: &[Real], b: &[Real], n: usize) -> Result<Vec<Real>>;
```

Row-major flat storage; an `n × n` matrix is a `Vec<Real>` of length
`n * n`. `dense_chol` errors when the input isn't SPD (negative
pivot); `dense_psd_pinv_solve` is the **IK fallback path** that
[`crate::solvers::ik`](./../sim/ik.md) uses when Cholesky reports a
breakdown on near-singular `(JᵀJ + λI)`. The fallback builds the full
symmetric eigendecomposition, clamps tiny eigenvalues at
`max(eigval) · 1e-12`, and forms

```text
x = Σᵢ (vᵢ · vᵢᵀ · b) / max(λᵢ, λ_floor)
```

which is equivalent to the SVD-based pseudoinverse for SPD matrices.
The clamp keeps the solution finite when `J` becomes rank-deficient
(joint-limit lock-up, singular configurations).
