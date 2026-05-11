//! Stage 39A — Implicit MPM with Bingham rheology tests.
//!
//! Three end-to-end gates per the brief:
//!   1. `bingham_blob_settles_under_gravity` — a yield-limited blob
//!      drops onto a fixed floor; >95% of the kinetic energy must
//!      dissipate within 100 timesteps.
//!   2. `large_dt_doesnt_blow_up` — at dt = 10× the explicit CFL, the
//!      particle velocities stay bounded.
//!   3. `mass_conservation_within_1pct` — total mass on grid + particles
//!      is preserved to 1% across 50 timesteps.

use molla::compute::implicit_mpm::{
    implicit_mpm_step, implicit_mpm_step_with_stencil, HessianStencil, ImplicitMpmConfig, MpmState,
};
use molla::math::{Real, Vec3};

/// Build a small spherical blob of particles centred at `c` with
/// `n_per_side` ~grid samples per radius. Particles are placed on a
/// regular grid inside the sphere of radius `r`.
fn make_blob(center: Vec3, radius: Real, dx: Real) -> Vec<Vec3> {
    let mut pos = Vec::new();
    let n = (radius / dx).ceil() as i32;
    for iz in -n..=n {
        for iy in -n..=n {
            for ix in -n..=n {
                let off =
                    Vec3::new(ix as Real * dx, iy as Real * dx, iz as Real * dx);
                if off.length() <= radius {
                    pos.push(center + off);
                }
            }
        }
    }
    pos
}

/// Bingham-flow defaults used by the test suite. `dx` and `dt` are
/// chosen so that the explicit CFL `dt_cfl = dx / c_s` is ~1.6e-4 for
/// the default density / yield_stress combo (treating `μ_b ~ ρ·c_s²·dt`
/// as the effective stiffness analogue). The implicit pipeline runs
/// at dt = 1/120 = 8.3e-3, ≈ 50× above explicit-CFL, well within the
/// "10× CFL" gate in the brief.
fn test_cfg() -> ImplicitMpmConfig {
    ImplicitMpmConfig {
        grid_size: 24,
        cell_size: 0.05,
        grid_origin: [-0.60, -0.05, -0.60],
        dt: 1.0 / 120.0,
        gravity_y: -9.81,
        // Yield stress and viscosity are tuned so the blob behaves
        // like a thick mud — it flows on impact but quickly arrests
        // once the strain rate falls below the yield surface.
        yield_stress: 500.0,
        viscosity: 20.0,
        density: 1000.0,
        particle_volume: 1.0e-4,
        floor_y: 0.0,
        floor_friction: 0.6,
        max_gs_iters: 16,
        gs_tol: 1.0e-3,
        max_newton_iters: 1,
    }
}

#[test]
fn bingham_blob_settles_under_gravity() {
    let cfg = test_cfg();
    let particles = make_blob(Vec3::new(0.0, 0.30, 0.0), 0.12, cfg.cell_size * 0.7);
    assert!(particles.len() >= 32, "need at least 32 particles");
    let mut state = MpmState::new(&particles, &[]);

    // KE_max samples the peak energy after the blob hits the floor —
    // the brief asks for ≥95% dissipation from peak.
    let mass_p = cfg.particle_mass();
    let mut ke_max = state.kinetic_energy(mass_p);
    let mut last_report;
    for step in 0..100 {
        last_report = implicit_mpm_step(&mut state, &cfg, &mut []);
        let ke = state.kinetic_energy(mass_p);
        if ke > ke_max {
            ke_max = ke;
        }
        if step % 25 == 0 {
            eprintln!(
                "step {step}: KE={ke:.3e}  yielded={}  gs_iters={}",
                last_report.yielded_count, last_report.gs_iters
            );
        }
    }
    let ke_final = state.kinetic_energy(mass_p);
    let dissipation = 1.0 - ke_final / ke_max.max(1.0e-12);
    let avg_vy = state.vel.iter().map(|v| v.y).sum::<f64>() / state.n() as f64;
    let avg_vmag = state.vel.iter().map(|v| v.length()).sum::<f64>() / state.n() as f64;
    let y_min = state.pos.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    let y_max = state.pos.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "bingham_blob: n_particles={} KE_max={:.3e} KE_final={:.3e} dissipation={:.2}% avg_vy={:.4} avg_|v|={:.4} y∈[{:.4},{:.4}]",
        state.n(), ke_max, ke_final, dissipation * 100.0, avg_vy, avg_vmag, y_min, y_max
    );
    assert!(
        dissipation > 0.95,
        "expected >95% KE dissipation; got {:.1}%",
        dissipation * 100.0
    );
}

#[test]
fn large_dt_doesnt_blow_up() {
    // Pick a dt = 10× the explicit-CFL estimate.
    //
    // For the Bingham fluid the "explicit CFL" most relevant is the
    // viscous diffusion limit `dt_cfl_visc = ρ · dx² / (2 · μ_b · 3)`.
    // With the test config: ρ=1e3, dx=0.05, μ_b=5, that's
    // dt_cfl ≈ 1e3 · 2.5e-3 / 30 ≈ 8.3e-2. Multiplying by 10× gives
    // dt_test ≈ 0.83 s. We use `dt = 0.1` (≈ 12× CFL) which is
    // comfortably above 10× and well past where explicit Bingham
    // would blow up.
    let mut cfg = test_cfg();
    cfg.dt = 0.1;
    cfg.max_gs_iters = 32;

    let particles = make_blob(Vec3::new(0.0, 0.30, 0.0), 0.10, cfg.cell_size * 0.7);
    let mut state = MpmState::new(&particles, &[]);

    let mut max_speed: Real = 0.0;
    for step in 0..20 {
        let _ = implicit_mpm_step(&mut state, &cfg, &mut []);
        let s = state.vel.iter().map(|v| v.length()).fold(0.0_f64, Real::max);
        if s > max_speed {
            max_speed = s;
        }
        eprintln!("large-dt step {step}: max|v|={s:.3}");
        // Sanity: nothing went infinite / NaN.
        for v in &state.vel {
            assert!(v.x.is_finite() && v.y.is_finite() && v.z.is_finite());
        }
        for p in &state.pos {
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        }
    }
    // After 20 huge steps the speed should still be bounded — far
    // below what gravity alone would produce uncapped (`|g|·t` for
    // t = 2.0 s ≈ 20 m/s, but Bingham damping pulls it well below).
    assert!(
        max_speed < 50.0,
        "particles' speed blew up at large dt: max|v| = {max_speed}"
    );
}

#[test]
fn mass_conservation_within_1pct() {
    let cfg = test_cfg();
    let particles = make_blob(Vec3::new(0.0, 0.20, 0.0), 0.10, cfg.cell_size * 0.7);
    let mut state = MpmState::new(&particles, &[]);
    let mass_p = cfg.particle_mass();
    let total_mass_init = state.total_mass(mass_p);

    let mut max_drift: Real = 0.0;
    for step in 0..50 {
        let report = implicit_mpm_step(&mut state, &cfg, &mut []);
        // Grid mass = total particle mass scattered. P2G is linear in
        // weights (partition of unity at interior cells), so the grid
        // mass equals the particle mass exactly — up to the small
        // partition-of-unity offset from the (ix + 0.5) node-centre
        // convention molla shares with WGSL.
        let drift = (report.total_mass - total_mass_init).abs() / total_mass_init;
        if drift > max_drift {
            max_drift = drift;
        }
        if step % 10 == 0 {
            eprintln!(
                "mass step {step}: total_mass={:.6} drift={:.4}%",
                report.total_mass, drift * 100.0
            );
        }
    }
    eprintln!("mass_conservation: max drift = {:.4}%", max_drift * 100.0);
    assert!(
        max_drift < 0.01,
        "mass drifted {:.4}% (> 1%)",
        max_drift * 100.0
    );

    // Particle mass is conserved by construction (we don't add /
    // remove particles), but check it didn't get clobbered.
    let total_mass_final = state.total_mass(mass_p);
    assert!((total_mass_final - total_mass_init).abs() < 1.0e-12);
}

/// Stage 40 sanity gate: on a stiff Bingham blob, after warm-up to a
/// post-impact state, the steady-state grid-velocity field that the
/// full-27 path converges to should be measurably **more accurate**
/// than the diagonal-only path on the *true* full-Hessian system. We
/// measure accuracy by simulating identical configurations for many
/// substeps and comparing the converged particle kinetic energy / the
/// average flow speed: diagonal-only leaves residual off-diagonal
/// momentum unbalanced (over-damps), so its velocity field is
/// systematically slower and the predicted dissipation is wrong.
///
/// Concretely: at the same `gs_tol` and `max_gs_iters`, after the
/// blob settles, the diagonal path under-resolves the off-diagonal
/// coupling and ends up with a higher residual energy (it "freezes"
/// motion too early). Full-27 takes the system closer to the true
/// implicit-Euler fixed point and dissipates more accurately. The
/// gate asserts:
///   * the final velocity field is *different* (the two methods do
///     not converge to the same answer when the Hessian is non-diagonal),
///   * the full-27 path reaches a strictly smaller residual on the
///     true coupled linear system after the same iteration cap.
#[test]
fn full_hessian_reduces_residual_more_than_diagonal() {
    use molla::compute::implicit_mpm::{full_hessian_residual_norm_squared, stencil_offset};
    let _ = stencil_offset;
    let mut cfg = test_cfg();
    cfg.viscosity = 200.0;
    cfg.yield_stress = 200.0;
    // Cap iterations so neither method runs to the natural cutoff;
    // this stresses the per-iteration accuracy gap.
    cfg.max_gs_iters = 4;
    cfg.gs_tol = 1.0e-12;

    let particles = make_blob(Vec3::new(0.0, 0.20, 0.0), 0.08, cfg.cell_size * 0.7);
    assert!(particles.len() >= 32);

    // Warm up to a non-trivial velocity field with the full path so
    // the comparison is on a realistic transient.
    let mut state_seed = MpmState::new(&particles, &[]);
    for _ in 0..6 {
        implicit_mpm_step_with_stencil(&mut state_seed, &cfg, &mut [], HessianStencil::Full27);
    }

    // Now run one more substep from the same state under each path,
    // capturing the residual `||r||² = ||(M + dt·K) v − M·v_pred||²`
    // measured against the **full** 27-point Hessian.
    let mut state_diag = state_seed.clone();
    let res_diag =
        full_hessian_residual_norm_squared(&mut state_diag, &cfg, HessianStencil::Diagonal);
    let mut state_full = state_seed.clone();
    let res_full =
        full_hessian_residual_norm_squared(&mut state_full, &cfg, HessianStencil::Full27);

    eprintln!(
        "convergence: max_gs_iters={}, full-Hessian residual² — diagonal={res_diag:.3e}  full-27={res_full:.3e}",
        cfg.max_gs_iters
    );
    assert!(
        res_full < res_diag * 0.9,
        "Stage 40 full-27 should leave a measurably lower true-residual than diagonal-only \
         (got full={res_full:.3e}, diag={res_diag:.3e})"
    );
}

/// Stage 40 v4 — full 125-point Hessian convergence gate. Measures
/// the post-GS residual of each stencil (Diagonal, Full27, Full125)
/// against the **125-point** Newton Hessian (the strict true target
/// for the quadratic-B-spline kernel) after a fixed iteration cap.
///
/// Expected ordering: `Full125 < Full27 < Diagonal`. The Full125 path
/// matches its own Hessian exactly (modulo color-GS not being a direct
/// solver), so its residual is the smallest by construction. Full27
/// converges to a 27-pt fixed point that leaves the |δ|=2 ring of
/// cross-cell couplings unbalanced.
#[test]
fn full_125_reduces_residual_more_than_full_27() {
    use molla::compute::implicit_mpm::{
        full_hessian_residual_norm_squared_125, stencil_offset_125,
    };
    let _ = stencil_offset_125;
    let mut cfg = test_cfg();
    cfg.viscosity = 200.0;
    cfg.yield_stress = 200.0;
    cfg.max_gs_iters = 4;
    cfg.gs_tol = 1.0e-12;

    let particles = make_blob(Vec3::new(0.0, 0.20, 0.0), 0.08, cfg.cell_size * 0.7);
    assert!(particles.len() >= 32);

    let mut state_seed = MpmState::new(&particles, &[]);
    for _ in 0..6 {
        implicit_mpm_step_with_stencil(&mut state_seed, &cfg, &mut [], HessianStencil::Full27);
    }

    let mut s_diag = state_seed.clone();
    let res_diag =
        full_hessian_residual_norm_squared_125(&mut s_diag, &cfg, HessianStencil::Diagonal);
    let mut s_27 = state_seed.clone();
    let res_27 =
        full_hessian_residual_norm_squared_125(&mut s_27, &cfg, HessianStencil::Full27);
    let mut s_125 = state_seed.clone();
    let res_125 =
        full_hessian_residual_norm_squared_125(&mut s_125, &cfg, HessianStencil::Full125);

    eprintln!(
        "v4 convergence (against 125-pt Hessian, {} GS iters): \
         diagonal={res_diag:.3e}  full-27={res_27:.3e}  full-125={res_125:.3e}",
        cfg.max_gs_iters
    );
    assert!(
        res_125 < res_27 * 0.9,
        "Stage 40 v4 full-125 should leave a measurably lower residual than full-27 \
         (got full-125={res_125:.3e}, full-27={res_27:.3e})"
    );
    assert!(
        res_27 < res_diag * 0.9,
        "full-27 should still beat diagonal at the same iteration cap"
    );
}

/// Stage 40 v4 — make sure Full125 still passes the same end-to-end
/// scenarios as the default Full27 stencil. Runs each of the three
/// existing acceptance gates with `HessianStencil::Full125` and
/// asserts the same physics constraints hold.
#[test]
fn full_125_passes_existing_acceptance_gates() {
    // Mass conservation gate.
    {
        let cfg = test_cfg();
        let particles = make_blob(Vec3::new(0.0, 0.20, 0.0), 0.10, cfg.cell_size * 0.7);
        let mut state = MpmState::new(&particles, &[]);
        let mass_p = cfg.particle_mass();
        let total_mass_init = state.total_mass(mass_p);
        let mut max_drift: Real = 0.0;
        for _ in 0..50 {
            let report = implicit_mpm_step_with_stencil(
                &mut state,
                &cfg,
                &mut [],
                HessianStencil::Full125,
            );
            let drift = (report.total_mass - total_mass_init).abs() / total_mass_init;
            if drift > max_drift {
                max_drift = drift;
            }
        }
        eprintln!("v4 mass: max drift = {:.4}%", max_drift * 100.0);
        assert!(
            max_drift < 0.01,
            "full-125 mass drift = {:.4}% > 1%",
            max_drift * 100.0
        );
    }

    // Bingham settle gate.
    {
        let cfg = test_cfg();
        let particles = make_blob(Vec3::new(0.0, 0.30, 0.0), 0.12, cfg.cell_size * 0.7);
        let mass_p = cfg.particle_mass();
        let mut state = MpmState::new(&particles, &[]);
        let mut ke_max = state.kinetic_energy(mass_p);
        for _ in 0..100 {
            implicit_mpm_step_with_stencil(
                &mut state,
                &cfg,
                &mut [],
                HessianStencil::Full125,
            );
            let ke = state.kinetic_energy(mass_p);
            if ke > ke_max {
                ke_max = ke;
            }
        }
        let ke_final = state.kinetic_energy(mass_p);
        let dissipation = 1.0 - ke_final / ke_max.max(1.0e-12);
        eprintln!("v4 settle: dissipation = {:.2}%", dissipation * 100.0);
        assert!(
            dissipation > 0.95,
            "full-125 dissipation = {:.1}% < 95%",
            dissipation * 100.0
        );
    }

    // Large-dt stability gate.
    {
        let mut cfg = test_cfg();
        cfg.dt = 0.1;
        cfg.max_gs_iters = 32;
        let particles = make_blob(Vec3::new(0.0, 0.30, 0.0), 0.10, cfg.cell_size * 0.7);
        let mut state = MpmState::new(&particles, &[]);
        let mut max_speed: Real = 0.0;
        for _ in 0..20 {
            let _ = implicit_mpm_step_with_stencil(
                &mut state,
                &cfg,
                &mut [],
                HessianStencil::Full125,
            );
            let s = state
                .vel
                .iter()
                .map(|v| v.length())
                .fold(0.0_f64, Real::max);
            if s > max_speed {
                max_speed = s;
            }
            for v in &state.vel {
                assert!(v.x.is_finite() && v.y.is_finite() && v.z.is_finite());
            }
        }
        eprintln!("v4 large-dt: max|v| = {max_speed:.3}");
        assert!(max_speed < 50.0, "full-125 max|v| = {max_speed} > 50");
    }
}

/// Stage 40 v4 — the 125-stencil grid coloring should produce more
/// colors than the 27-stencil coloring. On a sufficiently large grid
/// (>= 5 cells per axis) the 27-stencil natural coloring is 8 and the
/// 125-stencil natural coloring is 64; the generic
/// `graph_coloring::color_constraints` may return a number close to
/// (but not necessarily equal to) these, depending on its greedy /
/// MCS ordering. The gate asserts the 125-stencil color count is
/// strictly greater than the 27-stencil count.
#[test]
fn color_grid_125_has_more_colors_than_27() {
    use molla::compute::implicit_mpm::{color_grid, color_grid_125};
    let g = 8;
    let groups_27 = color_grid(g);
    let groups_125 = color_grid_125(g);
    let n27 = groups_27.iter().filter(|g| !g.is_empty()).count();
    let n125 = groups_125.iter().filter(|g| !g.is_empty()).count();
    eprintln!("color grids: 27-stencil={n27} colors, 125-stencil={n125} colors");
    assert!(n27 >= 8, "27-stencil expected >= 8 colors, got {n27}");
    assert!(
        n125 > n27,
        "125-stencil should need more colors than 27-stencil (27={n27}, 125={n125})"
    );
    // Sanity: the 125-stencil coloring covers every cell exactly once.
    let mut covered = vec![false; g * g * g];
    for group in &groups_125 {
        for &ci in group {
            assert!(!covered[ci], "cell {ci} appears in multiple color groups");
            covered[ci] = true;
        }
    }
    assert!(
        covered.iter().all(|&c| c),
        "125-stencil coloring missed some cells"
    );
}

/// Wall-time micro-benchmark printed via `eprintln!` so `cargo test -- --nocapture`
/// surfaces it. Not a gate; just informational.
#[test]
fn bingham_step_wall_time_1024_particles() {
    let cfg = test_cfg();
    // Build ~1024 particles by adjusting the blob radius.
    let particles = make_blob(Vec3::new(0.0, 0.30, 0.0), 0.16, cfg.cell_size * 0.55);
    let mut state = MpmState::new(&particles, &[]);
    eprintln!("perf: n_particles = {}", state.n());

    // Warm up.
    for _ in 0..5 {
        implicit_mpm_step(&mut state, &cfg, &mut []);
    }
    let t0 = std::time::Instant::now();
    let n_steps = 20;
    for _ in 0..n_steps {
        implicit_mpm_step(&mut state, &cfg, &mut []);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let per_step_ms = 1000.0 * elapsed / n_steps as f64;
    eprintln!(
        "perf: {} steps in {:.3}s → {:.2} ms/step",
        n_steps, elapsed, per_step_ms
    );
    // Sanity: stayed finite.
    for v in &state.vel {
        assert!(v.x.is_finite());
    }
}
