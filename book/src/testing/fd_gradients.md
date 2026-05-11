# Finite-difference gradient checks

molla's differentiable paths (Featherstone reverse adjoint, XPBD /
VBD adjoints, MPM adjoint, FEM adjoint, raycast adjoint, Gaussian
splat adjoint) all ship with **finite-difference gradient tests**
that cross-check the analytic gradient against central-difference
numerical gradient.

This chapter documents the pattern, the tolerance budgets, and the
common gotchas.

## The central-difference oracle

Given a scalar loss `L(x)` and a parameter `x[k]`, the central
finite-difference gradient is:

```text
∂L/∂x[k] ≈ (L(x + e_k h) - L(x - e_k h)) / (2 h)
```

where `e_k` is the k-th basis vector and `h` is the step size.

The standard pattern in molla tests:

```rust
fn fd_grad<F>(x0: &[f64], h: f64, mut eval: F) -> Vec<f64>
where F: FnMut(&[f64]) -> f64
{
    let mut g = vec![0.0; x0.len()];
    let mut x = x0.to_vec();
    for k in 0..x0.len() {
        x[k] = x0[k] + h;
        let lp = eval(&x);
        x[k] = x0[k] - h;
        let lm = eval(&x);
        x[k] = x0[k];
        g[k] = (lp - lm) / (2.0 * h);
    }
    g
}
```

Then:

```rust
let g_analytic = my_adjoint(x0);
let g_fd       = fd_grad(&x0, 1e-5, |x| my_forward_loss(x));
for k in 0..x0.len() {
    let abs = (g_analytic[k] - g_fd[k]).abs();
    let rel = abs / g_analytic[k].abs().max(1e-12);
    assert!(rel < 1e-3, "grad[{k}]: analytic {} vs fd {} (rel err {rel})",
        g_analytic[k], g_fd[k]);
}
```

## Tolerance budget

The standard tolerances in molla:

| Test family               | Step `h`        | Tolerance   | Notes                                          |
|---------------------------|-----------------|-------------|------------------------------------------------|
| Featherstone adjoint vs FD | 1e-5           | 1e-3 rel    | Many integrator steps, accumulated error       |
| XPBD adjoint vs FD         | 1e-5           | 1e-3 rel    | Position-based projection, mild non-smoothness |
| MPM SVD-chain adjoint vs FD | 1e-5          | 1e-3 rel    | Heavy nonlinear, narrow window for `h`         |
| Raycast adjoint vs FD      | adaptive       | 1e-3 rel    | Adaptive FD step from `add_sphere` validate    |
| Splat backward vs FD       | 1e-5           | 1e-3 rel    | Per-channel (pos / color / opacity / scale / quat) |
| SH coefficient gradient    | closed-form     | 1e-7 abs    | No integrator, closed-form derivatives         |
| Spring rest-length grad    | 1e-5           | 1e-3 rel    | DiffSpring scene                               |

The 1e-3 relative budget is the workhorse. Tighter than that, and
finite-difference truncation error dominates — you're measuring
the wall, not the gradient. Looser than that signals a real bug.

For closed-form derivatives (no integrator state — like the
spherical-harmonic basis coefficients), the budget tightens to
1e-7 absolute because there's no accumulation site.

## Choosing the step `h`

The step size has a Goldilocks zone:

- Too small (e.g. 1e-12): the loss values `L(x±h)` are equal to
  machine precision. Numerator is round-off noise, gradient is
  meaningless.
- Too large (e.g. 1e-2): higher-order Taylor terms become
  significant, central-difference is no longer a good
  approximation.
- Just right: `h ≈ 1e-5` for `f64` evaluation. Round-off is below
  the truncation error, truncation error is below 1e-6.

For functions with kinks or discontinuities (XPBD constraint
projection at the inequality boundary, MPM plasticity at the yield
surface), the FD test side-steps the kink:

- Pick a test point that's **strictly in the smooth interior**.
- For projection-onto-cone tests, perturb only in the tangent
  directions.
- For yield-surface tests, perturb only within the elastic regime.

Tests that violate this pattern get false-positive FD
disagreements and are eventually rewritten to use Moreau-envelope
smoothing of the projection (Stage 23.1.3 D6 joint adjoint is the
worked example).

## Raycast adaptive FD

The raycast adjoint test (`crates/molla/tests/raycast_adjoint.rs`)
uses **adaptive FD** because the residual is sensitive to the step
size when rays graze the silhouette of a sphere:

```rust
// validate add_sphere offsets, then pick h that keeps the FD
// gradient away from the silhouette discontinuity.
let h = adaptive_fd_step(&x0, |x| residual(x));
```

The adaptive step picks the largest `h` such that
`|L(x+h) - L(x-h)|` is still many ULPs above noise, then halves
until the second-order finite-difference correction is small.
This was added in commit `72dd668` to fix flaky tests that
straddled the silhouette.

## When the FD test fails

The triage:

1. **Print both gradients side by side.** Look for sign flips or
   order-of-magnitude differences — these are bugs in the
   adjoint, not in the FD.
2. **Try a tighter `h` (1e-6) and a looser `h` (1e-4).** If the
   FD disagrees with the analytic at both ends, the adjoint is
   wrong. If it disagrees at one and agrees at the other, the
   step is off the Goldilocks zone.
3. **Check the analytic implementation against the chain rule.**
   Most adjoint bugs are in the chain — the per-op derivatives
   are usually right.
4. **Check for non-smoothness.** Inequality projections, max /
   min ops, abs / sign, and discrete branches all introduce
   kinks. Move the test point away from the kink or use
   Moreau-envelope smoothing.

## Failure modes documented in the codebase

Real examples from the molla history:

- **Splat rotation FD disagreement** (Stage 44 v2). The CPU
  oracle used `glam::Quat::Mul` which gives `q v q*` (correct for
  unit quaternions). The WGSL kernel uses the BAC-CAB formula
  `v + 2qw(qv×v) + 2(qv × (qv × v))` (correct for unit quaternions,
  different for non-unit). The FD test perturbed `q` to non-unit
  and measured a different gradient — fixed by rewriting the CPU
  oracle to use the same BAC-CAB formula.
- **D6 joint angular-limit adjoint** (Stage 23.1.3). The limit
  projection has a branch on which side of the limit you're on.
  Naive adjoint was wrong at the limit boundary. Fix: branch-aware
  projection that knows which side the gradient should flow.
- **MPM plasticity adjoint** (Stage 38.1). The Drucker-Prager
  return map projects onto a cone surface. The adjoint through
  the projection needs Moreau-envelope smoothing of the
  characteristic-function projector — naive backprop misses the
  Jacobian of the projection.

Each was caught by an FD gradient test that flagged a 10–50%
relative error before the adjoint shipped.

## See also

- [Parity testing](./parity.md) — for non-adjoint test patterns.
- [Test strategy](./strategy.md) — the higher-level picture.
- `crates/molla/tests/autograd_*.rs` — the actual FD tests.
- `crates/molla/tests/splat_*_backward.rs` — splat adjoint FD
  tests, the worked example.
