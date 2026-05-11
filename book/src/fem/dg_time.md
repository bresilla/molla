# DG time integration — SSP-RK

DG semi-discretisations produce an ODE

\\[
\frac{du}{dt} = R(u, t)
\\]

where `u` is the flattened DG-DoF vector and `R` already includes the
inverse mass matrix (block-diagonal in DG ⇒ cheap to apply). Molla
pairs that ODE with **Strong Stability Preserving** (SSP) explicit
Runge-Kutta steppers from `dg_time_integrator.rs`.

## Why SSP?

A monotone spatial operator (e.g. after the
[Barth-Jespersen limiter](./dg_limiters.md) clips the per-cell
reconstruction) only stays monotone in time if the time stepper
preserves convex combinations of forward-Euler steps. SSP-RK methods
(Shu-Osher 1988, Cockburn-Shu 1989) are explicit RK schemes whose
stages are exactly such convex combinations — so the TVD / maximum-
principle property of the spatial operator carries over to the fully
discrete scheme without further effort.

Both methods shipped have **CFL coefficient 1**: the maximum stable
step size equals that of a forward-Euler step of the spatial operator.

## SSP-RK(2,2) — Shu-Osher

```rust
pub fn ssp_rk2_step<F, P>(
    u: &mut Vec<Real>,
    t: Real,
    dt: Real,
    mut rhs: F,
    mut post_stage: P,
) where
    F: FnMut(&[Real], Real) -> Vec<Real>,
    P: FnMut(&mut Vec<Real>);
```

Second-order accurate, two RHS evaluations per step:

\\[
\begin{aligned}
u^{(1)}   &= u^n + \Delta t \cdot R(u^n, t^n) \\
u^{n+1}   &= \tfrac{1}{2} u^n + \tfrac{1}{2} u^{(1)} + \tfrac{1}{2} \Delta t \cdot R(u^{(1)}, t^n + \Delta t)
\end{aligned}
\\]

## SSP-RK(3,3) — Shu-Osher low-storage

```rust
pub fn ssp_rk3_step<F, P>(...);
```

Third-order accurate, three RHS evaluations per step:

\\[
\begin{aligned}
u^{(1)} &= u^n + \Delta t \cdot R(u^n, t^n) \\
u^{(2)} &= \tfrac{3}{4} u^n + \tfrac{1}{4} u^{(1)} + \tfrac{1}{4} \Delta t \cdot R(u^{(1)}, t^n + \Delta t) \\
u^{n+1} &= \tfrac{1}{3} u^n + \tfrac{2}{3} u^{(2)} + \tfrac{2}{3} \Delta t \cdot R(u^{(2)}, t^n + \tfrac{1}{2} \Delta t)
\end{aligned}
\\]

This is the standard pairing for DG-P2: third-order in space ⇒
third-order in time keeps the global error balanced.

## Low-storage form

Both implementations use the **Shu-Osher low-storage form** —
only the input vector and *one* scratch (the stage value) are needed
per step. For SSP-RK2 that's a single `u1` vector; for SSP-RK3
there's `u1` and `u2`. The interface takes `rhs` as
`FnMut(&[Real], Real) -> Vec<Real>` so the caller can route the
spatial assembly through whatever path they prefer (flux blocks +
element residual + inverse mass), allocating fresh output vectors
inside.

For large DG systems where allocator churn matters, a future
optimisation pass could rework the closures to write into pre-
allocated scratch — but at v1 mesh sizes (≤ 10⁵ DG-DoFs) the
per-step `Vec` allocations are dwarfed by the RHS computation itself.

## The `post_stage` hook

```rust
ssp_rk3_step(&mut u, t, dt, rhs, |u_stage| {
    limit_solution(&mesh, u_stage, degree, &neighbours, 0.0, f64::INFINITY);
});
```

The `post_stage` closure is called on each intermediate stage and the
final state — the standard SSP discipline is to slope-limit there.
Pass `|_| {}` to skip limiting.

This pattern is what makes the limiter's TVD property carry through
the time integration: the limiter is monotone on each stage value, and
SSP-RK is a convex combination of those monotone-stage outputs, so the
final state is bounded by the per-cell envelope of the initial state.

## Empirical order verification

The `ssp_rk2_is_2nd_order` and `ssp_rk3_is_3rd_order` tests integrate
\\( dy/dt = -y \\) with several \\( \Delta t \\) values and fit the
log-log slope of error vs `dt`. Expected slopes:

- SSP-RK2: 2.0 (tolerated band 1.9–2.1).
- SSP-RK3: 3.0 (tolerated band 2.9–3.1).

The test fixture is intentionally trivial — the goal is to verify
the time integrator in isolation. Validation against full DG spatial
problems (Burgers shock, SWE dam-break) happens in the viewer scenes
which compare against analytic reference solutions.

## CFL guidance

For DG of degree `d` on a mesh of size `h`, the forward-Euler CFL
bound is roughly

\\[
\Delta t \lesssim \frac{1}{(2d + 1)} \cdot \frac{h}{|a|_{\max}}
\\]

— so DG-P2 typically asks for ~5× smaller \\( \Delta t \\) than DG-P0
at the same `h`, and ~3× smaller than DG-P1. The advection-dominated
viewer scenes set `dt` based on the dominant wave speed; SWE scenes
include the gravity-wave contribution \\( \sqrt{gh} \\).

## Cross-references

- [DG basis](./dg_basis.md) — the per-cell DoF layout the integrator
  walks.
- [DG flux](./dg_flux.md) — what the RHS closure builds.
- [DG limiters](./dg_limiters.md) — what to wire into `post_stage`.
- [DG BC](./dg_bc.md) — periodic face pairs that the integrator's RHS
  closure typically wraps.
