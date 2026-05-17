//! Stage 43 v2 — Strong-stability-preserving (SSP) explicit time
//! integrators for DG-style semi-discrete systems.
//!
//! A DG semi-discretisation of a conservation law produces an ODE of
//! the form
//!
//! ```text
//!   du/dt = RHS(u, t)
//! ```
//!
//! where `u` is the flattened DG-DoF vector and `RHS` already includes
//! the inverse mass matrix (block-diagonal in DG so cheap to apply).
//! Explicit Runge-Kutta time integration is the standard pairing —
//! "SSP" (Shu-Osher 1988, Cockburn-Shu 1989) variants are convex
//! combinations of forward-Euler steps, which preserve the maximum-
//! principle / TVD properties of any monotone spatial operator (e.g.
//! after the Barth-Jespersen slope limiter has been applied).
//!
//! v2 ships two integrators:
//!
//! - [`ssp_rk2_step`] — Shu-Osher SSP-RK(2,2). Second-order accurate
//!   in time, two RHS evaluations per step.
//! - [`ssp_rk3_step`] — Shu-Osher SSP-RK(3,3). Third-order accurate
//!   in time, three RHS evaluations per step. The standard pairing
//!   for DG-P2 (which is third-order in space) so the global error
//!   stays balanced.
//!
//! Both integrators use the **low-storage Shu-Osher form**, which only
//! needs the input vector and one scratch (the stage value) — important
//! for large DG systems where allocating a full-state copy per stage
//! gets expensive. The interface takes a closure `rhs: FnMut(&[Real],
//! Real) -> Vec<Real>` so callers can route the spatial discretisation
//! through whatever assembly path they prefer (flux block + element
//! residual + inverse mass).
//!
//! # Shu-Osher SSP-RK2
//!
//! ```text
//!   u^(1) = u^n + dt · R(u^n, t^n)
//!   u^{n+1} = ½ u^n + ½ u^(1) + ½ dt · R(u^(1), t^n + dt)
//! ```
//!
//! # Shu-Osher SSP-RK3
//!
//! ```text
//!   u^(1) = u^n + dt · R(u^n, t^n)
//!   u^(2) = ¾ u^n + ¼ u^(1) + ¼ dt · R(u^(1), t^n + dt)
//!   u^{n+1} = ⅓ u^n + ⅔ u^(2) + ⅔ dt · R(u^(2), t^n + ½ dt)
//! ```
//!
//! Both forms have CFL-coefficient 1 (i.e. the maximum stable step
//! size equals that of a forward-Euler step of the spatial operator).

use crate::math::Real;

/// Strong-stability-preserving second-order Runge-Kutta step
/// (Shu-Osher SSP-RK(2,2)).
///
/// Advances `u` from time `t` to `t + dt` in place. The closure
/// `rhs(u, t)` returns the time-derivative `du/dt` evaluated at state
/// `u` and time `t` (must already include any inverse-mass step). After
/// each intermediate stage the caller can hook a post-stage limiter
/// (e.g. Barth-Jespersen) via `post_stage(u_stage)` — this is the
/// standard SSP discipline. Pass `|_| {}` to skip limiting.
///
/// Two RHS evaluations per step.
pub fn ssp_rk2_step<F, P>(u: &mut Vec<Real>, t: Real, dt: Real, mut rhs: F, mut post_stage: P)
where
    F: FnMut(&[Real], Real) -> Vec<Real>,
    P: FnMut(&mut Vec<Real>),
{
    // Stage 1: u^(1) = u^n + dt · R(u^n, t).
    let k1 = rhs(u, t);
    let mut u1: Vec<Real> = u.iter().zip(k1.iter()).map(|(a, b)| a + dt * b).collect();
    post_stage(&mut u1);

    // Stage 2: u^{n+1} = ½ u^n + ½ u^(1) + ½ dt · R(u^(1), t + dt).
    let k2 = rhs(&u1, t + dt);
    for i in 0..u.len() {
        u[i] = 0.5 * u[i] + 0.5 * u1[i] + 0.5 * dt * k2[i];
    }
    post_stage(u);
}

/// Strong-stability-preserving third-order Runge-Kutta step
/// (Shu-Osher SSP-RK(3,3)).
///
/// Advances `u` from time `t` to `t + dt` in place. The closure
/// `rhs(u, t)` returns the time-derivative `du/dt` evaluated at state
/// `u` and time `t` (must already include any inverse-mass step).
///
/// `post_stage` is called on the intermediate stages and the final
/// state — the standard SSP convention is to slope-limit there.
///
/// Three RHS evaluations per step. Recommended default for DG-P2
/// (third-order in space ⇒ third-order in time keeps the global error
/// balanced).
pub fn ssp_rk3_step<F, P>(u: &mut Vec<Real>, t: Real, dt: Real, mut rhs: F, mut post_stage: P)
where
    F: FnMut(&[Real], Real) -> Vec<Real>,
    P: FnMut(&mut Vec<Real>),
{
    // Stage 1: u^(1) = u^n + dt · R(u^n, t).
    let k1 = rhs(u, t);
    let mut u1: Vec<Real> = u.iter().zip(k1.iter()).map(|(a, b)| a + dt * b).collect();
    post_stage(&mut u1);

    // Stage 2: u^(2) = ¾ u^n + ¼ u^(1) + ¼ dt · R(u^(1), t + dt).
    let k2 = rhs(&u1, t + dt);
    let mut u2: Vec<Real> = (0..u.len())
        .map(|i| 0.75 * u[i] + 0.25 * u1[i] + 0.25 * dt * k2[i])
        .collect();
    post_stage(&mut u2);

    // Stage 3: u^{n+1} = ⅓ u^n + ⅔ u^(2) + ⅔ dt · R(u^(2), t + ½ dt).
    let k3 = rhs(&u2, t + 0.5 * dt);
    for i in 0..u.len() {
        u[i] = (1.0 / 3.0) * u[i] + (2.0 / 3.0) * u2[i] + (2.0 / 3.0) * dt * k3[i];
    }
    post_stage(u);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `dy/dt = -y, y(0) = 1` → `y(t) = exp(-t)`. Step `ssp_rk2_step`
    /// with several `dt`'s and fit the log-log slope of the error vs
    /// `dt` — should sit around 2.
    #[test]
    fn ssp_rk2_is_2nd_order() {
        let t_end: Real = 1.0;
        let truth: Real = (-t_end).exp();
        let dts = [0.1_f64, 0.05, 0.025, 0.0125];
        let mut errs = Vec::new();
        for &dt in &dts {
            let mut y: Vec<Real> = vec![1.0];
            let mut t = 0.0;
            let nsteps = (t_end / dt).round() as usize;
            for _ in 0..nsteps {
                ssp_rk2_step(&mut y, t, dt, |u, _t| vec![-u[0]], |_| {});
                t += dt;
            }
            errs.push((y[0] - truth).abs());
        }
        // Average log-log slope of err vs dt.
        let mut rates = Vec::new();
        for i in 0..errs.len() - 1 {
            let r = (errs[i] / errs[i + 1]).ln() / (dts[i] / dts[i + 1]).ln();
            rates.push(r);
        }
        let avg = rates.iter().sum::<Real>() / rates.len() as Real;
        assert!(
            avg >= 1.9 && avg <= 2.1,
            "SSP-RK2 measured order = {avg}, want ~2.0"
        );
    }

    #[test]
    fn ssp_rk3_is_3rd_order() {
        let t_end: Real = 1.0;
        let truth: Real = (-t_end).exp();
        let dts = [0.1_f64, 0.05, 0.025, 0.0125];
        let mut errs = Vec::new();
        for &dt in &dts {
            let mut y: Vec<Real> = vec![1.0];
            let mut t = 0.0;
            let nsteps = (t_end / dt).round() as usize;
            for _ in 0..nsteps {
                ssp_rk3_step(&mut y, t, dt, |u, _t| vec![-u[0]], |_| {});
                t += dt;
            }
            errs.push((y[0] - truth).abs());
        }
        let mut rates = Vec::new();
        for i in 0..errs.len() - 1 {
            let r = (errs[i] / errs[i + 1]).ln() / (dts[i] / dts[i + 1]).ln();
            rates.push(r);
        }
        let avg = rates.iter().sum::<Real>() / rates.len() as Real;
        assert!(
            avg >= 2.9 && avg <= 3.1,
            "SSP-RK3 measured order = {avg}, want ~3.0"
        );
    }
}
