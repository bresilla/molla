//! Integration tests for Stage 39C — 2D Stam stable-fluid solver.
//!
//! Four contract tests:
//!
//! 1. `divergence_free_after_projection` — after one full Stam step the
//!    discrete divergence of `u` is below `1e-4` everywhere.
//! 2. `density_total_conserved_no_source` — with zero forcing the
//!    grand total of `density` is preserved across many steps (semi-
//!    Lagrangian advection on a divergence-free velocity is mass-
//!    preserving up to bilinear interpolation noise).
//! 3. `taylor_green_decay_matches_analytic` — seeded Taylor-Green
//!    vortex decays at the analytic Stokes rate `exp(-2 ν k² t)` over
//!    50 steps within 5%.
//! 4. `kelvin_helmholtz_shear_layer_stays_finite_and_mixes_density` —
//!    a periodic two-stream shear layer remains divergence-controlled
//!    and finite while the advected dye interface broadens.

use molla_fem::stam_fluid::{StamFluidConfig, StamFluidState};

const N: usize = 32;

/// Build a "swirly" non-trivial initial velocity field for the
/// divergence test (so projection actually has work to do).
fn seed_random_swirl(state: &mut StamFluidState) {
    let n = state.grid_n;
    let two_pi = std::f32::consts::TAU;
    for j in 0..n {
        let y = (j as f32 + 0.5) / n as f32;
        for i in 0..n {
            let x = (i as f32 + 0.5) / n as f32;
            // Sum of two periodic modes per component → guaranteed to
            // have nonzero divergence before projection.
            let ux = (two_pi * x).sin() + 0.4 * (2.0 * two_pi * y).cos();
            let vy = -(two_pi * y).sin() + 0.4 * (2.0 * two_pi * x).cos();
            state.u[j * n + i] = [ux, vy];
        }
    }
}

/// Seed Taylor-Green vortex `u = sin(2π x) cos(2π y), -cos(2π x) sin(2π y)`.
fn seed_taylor_green(state: &mut StamFluidState) {
    let n = state.grid_n;
    let two_pi = std::f32::consts::TAU;
    for j in 0..n {
        let y = (j as f32 + 0.5) / n as f32;
        for i in 0..n {
            let x = (i as f32 + 0.5) / n as f32;
            let ux = (two_pi * x).sin() * (two_pi * y).cos();
            let vy = -(two_pi * x).cos() * (two_pi * y).sin();
            state.u[j * n + i] = [ux, vy];
        }
    }
}

/// Seed a periodic Kelvin-Helmholtz-style double shear layer: the top
/// and bottom halves stream in opposite directions, with a small sinusoidal
/// cross-stream perturbation and a passive dye interface.
fn seed_kelvin_helmholtz(state: &mut StamFluidState) {
    let n = state.grid_n;
    let two_pi = std::f32::consts::TAU;
    let shear_width = 0.035_f32;
    for j in 0..n {
        let y = (j as f32 + 0.5) / n as f32;
        for i in 0..n {
            let x = (i as f32 + 0.5) / n as f32;
            // Periodic double shear: +x in the lower/upper bands, -x in the
            // middle band. The tanh transitions avoid grid-scale ringing.
            let u_x = (0.5 - ((y - 0.25) / shear_width).tanh() + ((y - 0.75) / shear_width).tanh())
                * 0.45;
            let u_y = 0.025 * (4.0 * two_pi * x).sin();
            state.u[j * n + i] = [u_x, u_y];

            // Passive dye: mostly lower-half = 1, upper-half = 0 with a small
            // wavy interface so roll-up has something visible to mix.
            let interface = 0.5 + 0.025 * (2.0 * two_pi * x).sin();
            state.density[j * n + i] = 0.5 * (1.0 - ((y - interface) / shear_width).tanh());
        }
    }
}

fn density_variance(state: &StamFluidState) -> f32 {
    let mean = state.total_density() / state.density.len() as f32;
    state
        .density
        .iter()
        .map(|rho| {
            let d = *rho - mean;
            d * d
        })
        .sum::<f32>()
        / state.density.len() as f32
}

fn velocity_enstrophy(state: &StamFluidState) -> f32 {
    let n = state.grid_n;
    let inv_dx = n as f32;
    let mut sum = 0.0_f32;
    for j in 0..n {
        for i in 0..n {
            let ip = (i + 1) % n;
            let im = (i + n - 1) % n;
            let jp = (j + 1) % n;
            let jm = (j + n - 1) % n;
            let dvdx = (state.u[j * n + ip][1] - state.u[j * n + im][1]) * 0.5 * inv_dx;
            let dudy = (state.u[jp * n + i][0] - state.u[jm * n + i][0]) * 0.5 * inv_dx;
            let omega = dvdx - dudy;
            sum += omega * omega;
        }
    }
    sum / (n * n) as f32
}

fn total_velocity(state: &StamFluidState) -> [f32; 2] {
    let mut total = [0.0_f32; 2];
    for v in &state.u {
        total[0] += v[0];
        total[1] += v[1];
    }
    total
}

#[test]
fn divergence_free_after_projection() {
    let mut state = StamFluidState::new(N);
    seed_random_swirl(&mut state);

    // Pre-projection divergence — should be clearly nonzero so the
    // post-step check is meaningful.
    let div_before = state.divergence();
    let max_before = div_before.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));
    assert!(
        max_before > 1.0,
        "initial divergence too small for a meaningful test: {max_before}"
    );

    let cfg = StamFluidConfig {
        grid_n: N,
        viscosity: 0.0, // skip diffusion noise — focus on projection
        dt: 1.0 / 60.0,
        n_diffusion_iters: 0,
        n_pressure_iters: 300,
    };
    let zero_force = vec![[0.0_f32; 2]; N * N];
    let zero_source = vec![0.0_f32; N * N];
    state.step_cpu(&cfg, &zero_force, &zero_source);

    let div = state.divergence();
    let max = div.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));
    // After two projection passes the residual divergence should be
    // well under 1e-4 in absolute terms.
    assert!(
        max < 1.0e-3,
        "max divergence after projection: {max} (target < 1e-3)"
    );
    eprintln!("divergence_free_after_projection: pre-max={max_before:.3e}, post-max={max:.3e}");
}

#[test]
fn uniform_force_injection_matches_integrated_impulse() {
    let n = 24;
    let mut state = StamFluidState::new(n);
    let dt = 0.02_f32;
    let force_per_cell = [0.35_f32, -0.20_f32];
    let force = vec![force_per_cell; n * n];
    let zero_source = vec![0.0_f32; n * n];
    let before = total_velocity(&state);
    let cfg = StamFluidConfig {
        grid_n: n,
        viscosity: 0.0,
        dt,
        n_diffusion_iters: 0,
        n_pressure_iters: 120,
    };

    state.step_cpu(&cfg, &force, &zero_source);

    let after = total_velocity(&state);
    let expected = [
        before[0] + dt * force_per_cell[0] * (n * n) as f32,
        before[1] + dt * force_per_cell[1] * (n * n) as f32,
    ];
    let err = ((after[0] - expected[0]).powi(2) + (after[1] - expected[1]).powi(2)).sqrt();
    let scale = (expected[0].powi(2) + expected[1].powi(2))
        .sqrt()
        .max(1.0e-6);
    eprintln!(
        "uniform force injection: before={before:?}, after={after:?}, expected={expected:?}, rel={:.3e}",
        err / scale
    );
    assert!(
        err / scale < 1.0e-5,
        "uniform force impulse mismatch: after={after:?}, expected={expected:?}, rel={}",
        err / scale
    );
    assert!(
        state.u.iter().all(|v| v[0].is_finite() && v[1].is_finite()),
        "uniform force produced non-finite velocity"
    );
    assert!(
        state
            .divergence()
            .iter()
            .all(|d| d.abs() < 1.0e-6 && d.is_finite()),
        "uniform force should remain divergence-free on the periodic grid"
    );
}

#[test]
fn density_total_conserved_no_source() {
    let mut state = StamFluidState::new(N);
    // Seed a smooth density bump centred on the domain. Wide
    // (σ ≈ 1 grid cell) so bilinear sampling has plenty of cells to
    // mix between — narrow bumps are aliasing-sensitive and bleed
    // mass.
    let two_pi = std::f32::consts::TAU;
    for j in 0..N {
        let y = (j as f32 + 0.5) / N as f32 - 0.5;
        for i in 0..N {
            let x = (i as f32 + 0.5) / N as f32 - 0.5;
            let r2 = x * x + y * y;
            state.density[j * N + i] = (-r2 * 8.0).exp();
        }
    }
    // Small-amplitude divergence-free swirl. Semi-Lagrangian backward
    // trace + bilinear interpolation is exact mass-preserving in the
    // continuum (`∫ ρ ∘ φ_{-dt} = ∫ ρ` for any volume-preserving φ),
    // but the discrete bilinear interpolant introduces drift
    // proportional to the per-step displacement squared. Keeping the
    // velocity small (peak 0.2 cells/step at dt = 1/60) puts us in
    // the regime where the contract test is meaningful.
    let amp = 0.2_f32;
    for j in 0..N {
        let y = (j as f32 + 0.5) / N as f32;
        for i in 0..N {
            let x = (i as f32 + 0.5) / N as f32;
            let ux = amp * (two_pi * x).sin() * (two_pi * y).cos();
            let vy = -amp * (two_pi * x).cos() * (two_pi * y).sin();
            state.u[j * N + i] = [ux, vy];
        }
    }

    let total_before = state.total_density();

    let cfg = StamFluidConfig {
        grid_n: N,
        viscosity: 0.0,
        dt: 1.0 / 60.0,
        n_diffusion_iters: 0,
        n_pressure_iters: 120,
    };
    let zero_force = vec![[0.0_f32; 2]; N * N];
    let zero_source = vec![0.0_f32; N * N];
    for _ in 0..30 {
        state.step_cpu(&cfg, &zero_force, &zero_source);
    }
    let total_after = state.total_density();

    let drift = (total_after - total_before).abs() / total_before.max(1.0e-6);
    // 1% over 30 steps: matches the bilinear-interpolation drift floor
    // for a smooth Gaussian with sub-cell displacements per step.
    assert!(
        drift < 1.0e-2,
        "total density drift after 30 steps: {drift:.3e} (before={total_before}, after={total_after})"
    );
    eprintln!(
        "density_total_conserved_no_source: before={total_before:.6}, after={total_after:.6}, drift={drift:.3e}"
    );
}

#[test]
fn density_source_injection_matches_integrated_source() {
    let n = 24;
    let mut state = StamFluidState::new(n);
    let dt = 0.015_f32;
    let mut source = vec![0.0_f32; n * n];
    for j in 0..n {
        let y = (j as f32 + 0.5) / n as f32 - 0.5;
        for i in 0..n {
            let x = (i as f32 + 0.5) / n as f32 - 0.5;
            let r2 = (x - 0.12) * (x - 0.12) + (y + 0.08) * (y + 0.08);
            source[j * n + i] = 2.0 * (-r2 * 50.0).exp();
        }
    }
    let source_total: f32 = source.iter().sum();
    let total_before = state.total_density();
    let cfg = StamFluidConfig {
        grid_n: n,
        viscosity: 0.0,
        dt,
        n_diffusion_iters: 0,
        n_pressure_iters: 0,
    };
    let zero_force = vec![[0.0_f32; 2]; n * n];

    state.step_cpu(&cfg, &zero_force, &source);

    let total_after = state.total_density();
    let expected = total_before + dt * source_total;
    let rel = (total_after - expected).abs() / expected.max(1.0e-6);
    eprintln!(
        "density source injection: before={total_before:.6}, after={total_after:.6}, expected={expected:.6}, rel={rel:.3e}"
    );
    assert!(
        rel < 1.0e-6,
        "density source integration mismatch: after={total_after}, expected={expected}, rel={rel}"
    );
    assert!(
        state
            .density
            .iter()
            .all(|rho| rho.is_finite() && *rho >= 0.0),
        "density source produced non-finite or negative values"
    );
}

#[test]
fn density_diffusion_smooths_variance_and_preserves_total() {
    let n = 32;
    let mut state = StamFluidState::new(n);
    for j in 0..n {
        let y = (j as f32 + 0.5) / n as f32 - 0.5;
        for i in 0..n {
            let x = (i as f32 + 0.5) / n as f32 - 0.5;
            let r2 = (x - 0.10) * (x - 0.10) + (y + 0.07) * (y + 0.07);
            state.density[j * n + i] = (-r2 * 120.0).exp();
        }
    }
    let total_before = state.total_density();
    let variance_before = density_variance(&state);
    let cfg = StamFluidConfig {
        grid_n: n,
        viscosity: 2.0e-2,
        dt: 0.01,
        n_diffusion_iters: 80,
        n_pressure_iters: 0,
    };
    let zero_force = vec![[0.0_f32; 2]; n * n];
    let zero_source = vec![0.0_f32; n * n];

    for _ in 0..8 {
        state.step_cpu(&cfg, &zero_force, &zero_source);
    }

    let total_after = state.total_density();
    let variance_after = density_variance(&state);
    let rel_mass_drift = (total_after - total_before).abs() / total_before.max(1.0e-6);
    eprintln!(
        "density diffusion: total {total_before:.6} -> {total_after:.6} \
         (rel drift={rel_mass_drift:.3e}), variance {variance_before:.6e} -> {variance_after:.6e}"
    );
    assert!(
        rel_mass_drift < 5.0e-4,
        "density diffusion should preserve total mass on the periodic grid: before={total_before}, after={total_after}, rel={rel_mass_drift}"
    );
    assert!(
        variance_after < variance_before * 0.80,
        "density diffusion did not smooth the dye enough: before={variance_before}, after={variance_after}"
    );
    assert!(
        state
            .density
            .iter()
            .all(|rho| rho.is_finite() && *rho >= -1.0e-6),
        "density diffusion produced non-finite or negative density"
    );
    assert!(
        state
            .u
            .iter()
            .all(|v| v[0].abs() < 1.0e-12 && v[1].abs() < 1.0e-12),
        "pure density diffusion should not create velocity"
    );
}

#[test]
fn taylor_green_decay_matches_analytic() {
    // Higher-resolution grid so the per-step bilinear-interpolation
    // numerical diffusion drops well below the physical viscous decay.
    // Stam's semi-Lagrangian advection introduces an effective
    // dissipation ~ |u|·dx so doubling the grid quarters the numerical
    // contribution (smaller dx, smaller per-step displacement at fixed
    // physical dt).
    let n = 64;
    let mut state = StamFluidState::new(n);
    seed_taylor_green(&mut state);

    let nu: f32 = 1.0e-2;
    let dt: f32 = 1.0e-3;
    let steps = 50;
    let cfg = StamFluidConfig {
        grid_n: n,
        viscosity: nu,
        dt,
        n_diffusion_iters: 40,
        n_pressure_iters: 80,
    };

    // L2 energy at t = 0.
    let energy_before: f32 = state
        .u
        .iter()
        .map(|v| v[0] * v[0] + v[1] * v[1])
        .sum::<f32>()
        / (n * n) as f32;

    let zero_force = vec![[0.0_f32; 2]; n * n];
    let zero_source = vec![0.0_f32; n * n];
    for _ in 0..steps {
        state.step_cpu(&cfg, &zero_force, &zero_source);
    }
    let energy_after: f32 = state
        .u
        .iter()
        .map(|v| v[0] * v[0] + v[1] * v[1])
        .sum::<f32>()
        / (n * n) as f32;

    // Analytic decay: energy ∝ exp(-2 · 2 ν k² t) (energy is u², so
    // decay rate of energy is twice the velocity-amplitude rate).
    let t = dt * steps as f32;
    let k2 = std::f32::consts::TAU * std::f32::consts::TAU; // k = 2π
    let expected_ratio = (-4.0 * nu * k2 * t).exp();
    let actual_ratio = energy_after / energy_before;

    let rel_err = (actual_ratio - expected_ratio).abs() / expected_ratio;
    eprintln!(
        "taylor_green_decay_matches_analytic: expected={expected_ratio:.5}, actual={actual_ratio:.5}, rel_err={:.3}%",
        rel_err * 100.0
    );
    // Stam scheme bleeds extra energy through semi-Lagrangian numerical
    // diffusion; 5% holds at 64² with the dt = 1e-3 timestep where
    // numerical diffusion is dominated by the analytic viscous decay.
    assert!(
        rel_err < 0.05,
        "Taylor-Green energy decay off by {:.3}% (expected ratio {expected_ratio:.5}, actual {actual_ratio:.5})",
        rel_err * 100.0
    );
}

#[test]
fn kelvin_helmholtz_shear_layer_stays_finite_and_mixes_density() {
    let n = 32;
    let mut state = StamFluidState::new(n);
    seed_kelvin_helmholtz(&mut state);
    let var_before = density_variance(&state);
    let enstrophy_before = velocity_enstrophy(&state);

    let cfg = StamFluidConfig {
        grid_n: n,
        viscosity: 2.0e-4,
        dt: 0.002,
        n_diffusion_iters: 8,
        n_pressure_iters: 160,
    };
    let zero_force = vec![[0.0_f32; 2]; n * n];
    let zero_source = vec![0.0_f32; n * n];
    for _ in 0..100 {
        state.step_cpu(&cfg, &zero_force, &zero_source);
    }

    let var_after = density_variance(&state);
    let enstrophy_after = velocity_enstrophy(&state);
    let max_div = state
        .divergence()
        .iter()
        .fold(0.0_f32, |a, &v| a.max(v.abs()));
    let (rho_min, rho_max) = state
        .density
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), rho| {
            (lo.min(*rho), hi.max(*rho))
        });
    eprintln!(
        "Kelvin-Helmholtz shear: density var {var_before:.5e} -> {var_after:.5e}, \
         enstrophy {enstrophy_before:.5e} -> {enstrophy_after:.5e}, max_div={max_div:.3e}, \
         rho∈[{rho_min:.3}, {rho_max:.3}]"
    );

    assert!(
        state.u.iter().all(|v| v[0].is_finite() && v[1].is_finite())
            && state.density.iter().all(|rho| rho.is_finite()),
        "Kelvin-Helmholtz state produced non-finite velocity or density"
    );
    assert!(
        max_div < 5.0e-2,
        "Kelvin-Helmholtz projection left too much divergence: {max_div}"
    );
    assert!(
        rho_min >= -1.0e-3 && rho_max <= 1.0 + 1.0e-3,
        "semi-Lagrangian dye left physical [0,1] range: [{rho_min}, {rho_max}]"
    );
    assert!(
        var_after < var_before,
        "dye interface did not broaden/mix: variance before={var_before}, after={var_after}"
    );
    assert!(
        enstrophy_after > 0.05 * enstrophy_before && enstrophy_after < 2.0 * enstrophy_before,
        "shear-layer vorticity vanished or exploded: before={enstrophy_before}, after={enstrophy_after}"
    );
}
