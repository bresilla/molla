//! Eulerian fluid adjoint contract:
//! 1. FD gradcheck of ∂J/∂force and ∂J/∂ρ₀ through 12 full Stam steps
//!    (projection + self-advection + density advection) — f64, central
//!    differences against the adjoint.
//! 2. Smoke steering: gradient descent on a per-cell force field
//!    moves a smoke blob onto a target region (loss drops
//!    monotonically and substantially) — the diffsim use case.

use molla_autograd::stam_diff::{StamDiffConfig, StamFluidDiff};
use molla_math::Real;

const N: usize = 12;
const STEPS: usize = 12;

fn blob(n: usize, cx: Real, cy: Real, r: Real) -> Vec<Real> {
    let mut out = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let dx = i as Real - cx;
            let dy = j as Real - cy;
            out[j * n + i] = (-(dx * dx + dy * dy) / (r * r)).exp();
        }
    }
    out
}

/// J = Σ ρ_T · mask (smoke mass captured by the target region).
fn run_loss(force: &[[Real; 2]], rho0: &[Real], mask: &[Real]) -> (Real, StamFluidDiff) {
    let cfg = StamDiffConfig {
        grid_n: N,
        dt: 0.08,
        pressure_iters: 40,
    };
    let mut sim = StamFluidDiff::new(cfg);
    sim.density = rho0.to_vec();
    let source = vec![0.0; N * N];
    for _ in 0..STEPS {
        sim.step(force, &source);
    }
    let j = sim
        .density
        .iter()
        .zip(mask.iter())
        .map(|(r, m)| r * m)
        .sum();
    (j, sim)
}

#[test]
fn fd_validates_force_and_density_gradients() {
    let rho0 = blob(N, 3.0, 6.0, 1.6);
    let mask = blob(N, 8.0, 6.0, 2.0);
    // Small but nonzero rightward force so advection is active and
    // asymmetric.
    let mut force = vec![[0.0; 2]; N * N];
    for (k, f) in force.iter_mut().enumerate() {
        f[0] = 0.4 + 0.05 * ((k % 7) as Real - 3.0);
        f[1] = 0.03 * ((k % 5) as Real - 2.0);
    }

    let (_, sim) = run_loss(&force, &rho0, &mask);
    let adj_u = vec![[0.0; 2]; N * N];
    let adj_rho: Vec<Real> = mask.clone();
    let (g_force, _g_src, _g_u0, g_rho0) = sim.backward(adj_u, adj_rho);

    let eps = 1.0e-5;
    let mut max_rel: Real = 0.0;
    // Sample force gradients (both components) across the grid.
    for k in (0..N * N).step_by(17) {
        for c in 0..2 {
            let mut fp = force.clone();
            let mut fm = force.clone();
            fp[k][c] += eps;
            fm[k][c] -= eps;
            let (jp, _) = run_loss(&fp, &rho0, &mask);
            let (jm, _) = run_loss(&fm, &rho0, &mask);
            let fd = (jp - jm) / (2.0 * eps);
            let got = g_force[k][c];
            let denom = fd.abs().max(got.abs()).max(1.0e-6);
            let rel = (fd - got).abs() / denom;
            max_rel = max_rel.max(rel);
        }
    }
    eprintln!("stam dJ/dforce FD max rel = {max_rel:.3e}");
    assert!(
        max_rel < 5.0e-4,
        "force gradient FD mismatch: {max_rel:.3e}"
    );

    // ρ₀ gradients.
    let mut max_rel_r: Real = 0.0;
    for k in (0..N * N).step_by(23) {
        let mut rp = rho0.clone();
        let mut rm = rho0.clone();
        rp[k] += eps;
        rm[k] -= eps;
        let (jp, _) = run_loss(&force, &rp, &mask);
        let (jm, _) = run_loss(&force, &rm, &mask);
        let fd = (jp - jm) / (2.0 * eps);
        let got = g_rho0[k];
        let denom = fd.abs().max(got.abs()).max(1.0e-6);
        max_rel_r = max_rel_r.max((fd - got).abs() / denom);
    }
    eprintln!("stam dJ/drho0 FD max rel = {max_rel_r:.3e}");
    assert!(
        max_rel_r < 5.0e-4,
        "rho0 gradient FD mismatch: {max_rel_r:.3e}"
    );
}

#[test]
fn gradient_ascent_steers_smoke_to_target() {
    let rho0 = blob(N, 3.0, 6.0, 1.6);
    let mask = blob(N, 8.5, 6.0, 2.0);
    let mut force = vec![[0.0; 2]; N * N];

    // Maximise captured smoke J by gradient ASCENT on the force field.
    let (j_init, _) = run_loss(&force, &rho0, &mask);
    let mut j_hist = vec![j_init];
    for _ in 0..25 {
        let (_, sim) = run_loss(&force, &rho0, &mask);
        let (g_force, _, _, _) = sim.backward(vec![[0.0; 2]; N * N], mask.clone());
        for (f, g) in force.iter_mut().zip(g_force.iter()) {
            f[0] += 3.0 * g[0];
            f[1] += 3.0 * g[1];
        }
        let (j, _) = run_loss(&force, &rho0, &mask);
        j_hist.push(j);
    }
    let j_final = *j_hist.last().unwrap();
    eprintln!(
        "steering: J {j_init:.4} → {j_final:.4} ({} iters)",
        j_hist.len() - 1
    );
    assert!(
        j_final > 1.5 * j_init,
        "steering failed: J {j_init:.4} → {j_final:.4}"
    );
    // Broadly increasing (allow tiny line-search-free wobble).
    assert!(
        j_hist.windows(2).filter(|w| w[1] < w[0] - 1.0e-6).count() <= 3,
        "loss not broadly increasing: {j_hist:?}"
    );
}
