//! C0 phase 1 — barrier-Newton CPU oracle contract.
//!
//! 1. FD-validate ∇E and the (projected) ∇²E·v operator on a random
//!    configuration with active contacts — the correctness anchor for
//!    the GPU port.
//! 2. Sphere/plane drop: a pre-tensioned shell dropped onto obstacles
//!    settles with every gap in (0, d̂] band behavior and the
//!    INVARIANT holds: `min_gap` stays strictly positive at every
//!    accepted Newton iterate of every step.
//! 3. High-speed impact (the tunneling case that motivates C0): a
//!    shell fired at the floor at 50 m/s with a step so large it
//!    crosses the floor several times over — the CCD-truncated line
//!    search must keep it penetration-free.
//!
//! Every test ends with an explicit intersection scan (the
//! ppf-contact-solver practice adopted verbatim).

use molla_math::{Real, Vec3};
use molla_solvers::barrier_contact::{BarrierObstacle, BarrierParams, BarrierShell, BarrierSpring};

fn cloth_patch(n: usize, y: Real, spacing: Real) -> (Vec<Vec3>, Vec<Real>, Vec<BarrierSpring>) {
    let mut pos = Vec::new();
    let mut mass = Vec::new();
    for j in 0..n {
        for i in 0..n {
            pos.push(Vec3::new(
                (i as Real - (n as Real - 1.0) / 2.0) * spacing,
                y,
                (j as Real - (n as Real - 1.0) / 2.0) * spacing,
            ));
            mass.push(0.02);
        }
    }
    let idx = |i: usize, j: usize| (j * n + i) as u32;
    let mut springs = Vec::new();
    let mut push = |a: u32, b: u32| {
        let rest = (pos[a as usize] - pos[b as usize]).length();
        springs.push(BarrierSpring {
            a,
            b,
            rest,
            k: 400.0,
        });
    };
    for j in 0..n {
        for i in 0..n {
            if i + 1 < n {
                push(idx(i, j), idx(i + 1, j));
            }
            if j + 1 < n {
                push(idx(i, j), idx(i, j + 1));
            }
            if i + 1 < n && j + 1 < n {
                push(idx(i, j), idx(i + 1, j + 1));
                push(idx(i + 1, j), idx(i, j + 1));
            }
        }
    }
    (pos, mass, springs)
}

#[test]
fn fd_validates_gradient_and_hessian_operator() {
    let (mut pos, mass, springs) = cloth_patch(4, 0.0006, 0.05);
    // Perturb into the barrier band + irregular spring strains.
    let mut s = 0xC0FFEE_u64;
    let mut rng = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    for p in pos.iter_mut() {
        p.x += 0.004 * rng();
        p.y += 0.0004 * rng();
        p.z += 0.004 * rng();
    }
    let obstacles = vec![
        BarrierObstacle::HalfSpace {
            n: Vec3::Y,
            offset: 0.0,
        },
        BarrierObstacle::Sphere {
            center: Vec3::new(0.02, -0.05, 0.0),
            radius: 0.0495,
        },
    ];
    let shell = BarrierShell::new(
        pos.clone(),
        mass,
        springs,
        obstacles,
        BarrierParams::default(),
    )
    .expect("shell");
    let h = 1.0 / 240.0;
    let x_tilde: Vec<Vec3> = pos
        .iter()
        .map(|p| *p + Vec3::new(0.001, -0.002, 0.0))
        .collect();

    // Some barrier terms must be ACTIVE or the test is vacuous.
    let active = pos.iter().filter(|p| p.y < shell.params.dhat).count();
    assert!(active > 4, "need active barrier contacts, got {active}");

    // ∇E vs central FD of E.
    let mut grad = vec![Vec3::ZERO; pos.len()];
    shell.fd_probe_gradient(&pos, &x_tilde, h, &mut grad);
    let eps = 1.0e-7;
    let mut max_rel: Real = 0.0;
    for i in 0..pos.len().min(12) {
        for c in 0..3 {
            let mut xp = pos.clone();
            let mut xm = pos.clone();
            match c {
                0 => {
                    xp[i].x += eps;
                    xm[i].x -= eps;
                }
                1 => {
                    xp[i].y += eps;
                    xm[i].y -= eps;
                }
                _ => {
                    xp[i].z += eps;
                    xm[i].z -= eps;
                }
            }
            let fd = (shell.fd_probe_energy(&xp, &x_tilde, h)
                - shell.fd_probe_energy(&xm, &x_tilde, h))
                / (2.0 * eps);
            let got = match c {
                0 => grad[i].x,
                1 => grad[i].y,
                _ => grad[i].z,
            };
            let denom = fd.abs().max(got.abs()).max(1.0e-10);
            max_rel = max_rel.max((fd - got).abs() / denom);
        }
    }
    eprintln!("gradient FD max rel = {max_rel:.3e}");
    assert!(max_rel < 1.0e-5, "gradient FD mismatch: {max_rel:.3e}");

    // Hessian OPERATOR consistency: away from the projection kinks the
    // operator must match FD of the gradient along a random direction.
    // (Compression-clamped springs + convex-only barrier blocks mean
    // the projected operator differs from the true Hessian exactly
    // where clamps are ACTIVE; probe with a tensioned configuration so
    // no clamp is active: all springs stretched.)
    let stretched: Vec<Vec3> = pos.iter().map(|p| *p * 1.2).collect();
    let dir: Vec<Vec3> = (0..pos.len())
        .map(|_| Vec3::new(rng(), rng(), rng()))
        .collect();
    let mut hv = vec![Vec3::ZERO; pos.len()];
    shell.fd_probe_hessian_mul(&stretched, h, &dir, &mut hv);
    let mut gp = vec![Vec3::ZERO; pos.len()];
    let mut gm = vec![Vec3::ZERO; pos.len()];
    let xp: Vec<Vec3> = stretched
        .iter()
        .zip(dir.iter())
        .map(|(x, d)| *x + *d * eps)
        .collect();
    let xm: Vec<Vec3> = stretched
        .iter()
        .zip(dir.iter())
        .map(|(x, d)| *x - *d * eps)
        .collect();
    shell.fd_probe_gradient(&xp, &x_tilde, h, &mut gp);
    shell.fd_probe_gradient(&xm, &x_tilde, h, &mut gm);
    let mut max_rel_h: Real = 0.0;
    for i in 0..pos.len() {
        let fd = (gp[i] - gm[i]) / (2.0 * eps);
        let denom = fd.length().max(hv[i].length()).max(1.0e-8);
        max_rel_h = max_rel_h.max((fd - hv[i]).length() / denom);
    }
    eprintln!("hessian-op FD max rel = {max_rel_h:.3e}");
    assert!(max_rel_h < 1.0e-4, "hessian FD mismatch: {max_rel_h:.3e}");
}

#[test]
fn drop_settles_penetration_free_on_plane_and_sphere() {
    let (pos, mass, springs) = cloth_patch(8, 0.15, 0.03);
    let obstacles = vec![
        BarrierObstacle::HalfSpace {
            n: Vec3::Y,
            offset: 0.0,
        },
        BarrierObstacle::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 0.06,
        },
    ];
    let mut shell =
        BarrierShell::new(pos, mass, springs, obstacles, BarrierParams::default()).expect("shell");
    let h = 1.0 / 240.0;
    let mut worst_gap = Real::INFINITY;
    for step in 0..240 {
        let stats = shell.step(h).expect("step");
        worst_gap = worst_gap.min(stats.min_gap);
        assert!(
            stats.min_gap > 0.0,
            "step {step}: INVARIANT violated — gap {} at an accepted iterate",
            stats.min_gap
        );
    }
    // Settled: velocities near zero, draped over the sphere.
    let v_max = shell
        .velocities
        .iter()
        .map(|v| v.length())
        .fold(0.0, Real::max);
    let gap = shell.min_gap();
    eprintln!("drop settled: worst gap {worst_gap:.2e}, final gap {gap:.2e}, v_max {v_max:.3e}");
    assert!(v_max < 0.05, "shell did not settle: v_max = {v_max}");
    assert!(
        gap > 0.0 && gap < shell.params.dhat,
        "resting gap must sit inside the barrier band: {gap:.3e} vs d̂ {}",
        shell.params.dhat
    );
    // Explicit end-of-test intersection scan.
    assert!(shell.min_gap() > 0.0, "final intersection scan failed");
}

#[test]
fn high_speed_impact_cannot_tunnel() {
    // 50 m/s straight down with h = 1/60: one free-flight step moves
    // 0.83 m — the shell starts 5 cm above the floor, so an
    // unprotected integrator crosses the floor ~16× over in a single
    // step. The CCD-truncated line search must make that impossible.
    let (pos, mass, springs) = cloth_patch(6, 0.05, 0.03);
    let obstacles = vec![BarrierObstacle::HalfSpace {
        n: Vec3::Y,
        offset: 0.0,
    }];
    let mut shell =
        BarrierShell::new(pos, mass, springs, obstacles, BarrierParams::default()).expect("shell");
    for v in shell.velocities.iter_mut() {
        v.y = -50.0;
    }
    let h = 1.0 / 60.0;
    for step in 0..30 {
        let stats = shell.step(h).expect("step");
        assert!(
            stats.min_gap > 0.0,
            "step {step}: tunneled — gap {}",
            stats.min_gap
        );
    }
    assert!(shell.min_gap() > 0.0, "final intersection scan failed");
    // And it actually arrested the shell rather than parking it midair.
    let y_max = shell
        .positions
        .iter()
        .map(|p| p.y)
        .fold(Real::NEG_INFINITY, Real::max);
    eprintln!("impact arrested: max y = {y_max:.4}");
    assert!(
        y_max < 0.1,
        "shell should have landed near the floor: {y_max}"
    );
}

/// Elasticity-inclusive dynamic stiffness (TOG 43.6 scheme): a vertex
/// pressed into the floor by a HARD elastic load (stiff spring to an
/// anchor pinned below the floor — force ~1 kN, constant to within
/// the band width). With κ sized for inertia alone (mass-only) the
/// elastic press drives the vertex deep into the activation band,
/// nearly to the wall; the inclusive κ = scale·(m + h²·Σk)/h² meets
/// the press with matched stiffness and holds most of the band.
/// `kappa_scale = 1` so the contrast is measurable (the default 1e3
/// hides any realistic load).
#[test]
fn elasticity_inclusive_kappa_holds_gap_under_elastic_press() {
    let run = |inclusive: bool| -> Real {
        let params = BarrierParams {
            kappa_scale: 1.0,
            elasticity_inclusive_kappa: inclusive,
            gravity: Vec3::ZERO,
            ..Default::default()
        };
        let pos = vec![
            Vec3::new(0.0, 0.5e-3, 0.0),  // pressed vertex, in-band
            Vec3::new(0.0, -0.0499, 0.0), // anchor below the floor
        ];
        let springs = vec![BarrierSpring {
            a: 0,
            b: 1,
            rest: 0.0,
            k: 2.0e4, // press ≈ k·0.05 ≈ 1 kN
        }];
        let mut shell = BarrierShell::new(
            pos,
            vec![0.02, 0.02],
            springs,
            vec![BarrierObstacle::HalfSpace {
                n: Vec3::Y,
                offset: 0.0,
            }],
            params,
        )
        .expect("shell");
        shell.pin_vertices(&[1]);
        let h = 1.0 / 240.0;
        for step in 0..120 {
            let stats = shell.step(h).expect("step");
            assert!(stats.min_gap > 0.0, "step {step}: pressed through");
        }
        shell.positions[0].y
    };

    let gap_mass_only = run(false);
    let gap_inclusive = run(true);
    eprintln!(
        "elastic press: mass-only gap {gap_mass_only:.3e}, inclusive gap {gap_inclusive:.3e}"
    );
    assert!(gap_mass_only > 0.0 && gap_inclusive > 0.0);
    assert!(
        gap_inclusive > 5.0 * gap_mass_only,
        "inclusive κ should resist the elastic press: {gap_inclusive:.3e} vs {gap_mass_only:.3e}"
    );
}
