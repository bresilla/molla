//! C0 phase 4b — lagged IPC friction with the μ_s/μ_k split.
//!
//! 1. FD gradcheck of the friction potential with contacts sitting in
//!    ALL THREE mollifier regions (stick well, blend, kinetic
//!    plateau) — the lag is pinned first, making E smooth in x.
//! 2. THE discriminating incline (the oracle the roadmap-A1 XPBD
//!    attempt could not pass): tan θ = 0.5 sits between μ_k = 0.4 and
//!    μ_s = 0.6, so a patch at rest must STICK with the split active
//!    and SLIDE with μ_s = μ_k. Velocity-level friction cannot
//!    express this; the lagged potential can.

use molla_math::{Real, Vec3};
use molla_solvers::barrier_contact::{BarrierObstacle, BarrierParams, BarrierShell, BarrierSpring};

fn incline_patch(mu_s_scale: Real) -> BarrierShell {
    let theta: Real = 0.5_f64.atan();
    let n = Vec3::new(theta.sin(), theta.cos(), 0.0);
    let t1 = Vec3::new(theta.cos(), -theta.sin(), 0.0); // downhill
    let t2 = Vec3::Z;
    let spacing = 0.03;
    let (nx, nz) = (4usize, 4usize);
    let params = BarrierParams {
        friction_mu: 0.4,
        friction_mu_s_scale: mu_s_scale,
        ..Default::default()
    };
    let mut pos = Vec::new();
    for j in 0..nz {
        for i in 0..nx {
            // In-plane lattice, half a barrier band off the surface.
            pos.push(
                t1 * (i as Real * spacing) + t2 * (j as Real * spacing) + n * (0.5 * params.dhat),
            );
        }
    }
    let idx = |i: usize, j: usize| (j * nx + i) as u32;
    let mut springs = Vec::new();
    for j in 0..nz {
        for i in 0..nx {
            for (di, dj) in [(1usize, 0usize), (0, 1), (1, 1)] {
                if i + di < nx && j + dj < nz {
                    let (a, b) = (idx(i, j), idx(i + di, j + dj));
                    springs.push(BarrierSpring {
                        a,
                        b,
                        rest: (pos[a as usize] - pos[b as usize]).length(),
                        k: 400.0,
                    });
                }
            }
        }
    }
    BarrierShell::new(
        pos,
        vec![0.02; nx * nz],
        springs,
        vec![BarrierObstacle::HalfSpace { n, offset: 0.0 }],
        params,
    )
    .expect("shell")
}

#[test]
fn fd_validates_friction_potential_in_all_regions() {
    let mut shell = incline_patch(1.5);
    shell.rebuild_friction_lag();
    let h = 1.0 / 240.0;
    let e_w = shell.params.friction_epsv * h; // stick-well width

    // Push vertices tangentially into the interior of each region:
    // 0.3e (stick), 1.5e (blend), 3e (kinetic) — all far from the
    // piecewise boundaries relative to the 1e-7 FD step.
    let theta: Real = 0.5_f64.atan();
    let t1 = Vec3::new(theta.cos(), -theta.sin(), 0.0);
    let mut pos = shell.positions.clone();
    for (i, p) in pos.iter_mut().enumerate() {
        let s = match i % 3 {
            0 => 0.3 * e_w,
            1 => 1.5 * e_w,
            _ => 3.0 * e_w,
        };
        *p += t1 * s;
    }
    let x_tilde = pos.clone();
    let mut grad = vec![Vec3::ZERO; pos.len()];
    shell.fd_probe_gradient(&pos, &x_tilde, h, &mut grad);

    let eps = 1.0e-9;
    let mut max_rel: Real = 0.0;
    for i in 0..pos.len() {
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
            // Floor three orders below the O(1) friction-gradient
            // scale — sub-1e-8 absolute disagreements are FD noise.
            let denom = fd.abs().max(got.abs()).max(1.0e-3);
            let rel = (fd - got).abs() / denom;
            if rel > 1.0e-4 {
                eprintln!(
                    "  probe v{i} c{c} (region {}): fd {fd:.6e} got {got:.6e}",
                    i % 3
                );
            }
            max_rel = max_rel.max(rel);
        }
    }
    eprintln!("friction gradient FD max rel = {max_rel:.3e}");
    assert!(max_rel < 1.0e-4, "friction FD mismatch: {max_rel:.3e}");
}

#[test]
fn incline_discriminates_static_from_kinetic() {
    let theta: Real = 0.5_f64.atan();
    let t1 = Vec3::new(theta.cos(), -theta.sin(), 0.0);
    let run = |scale: Real| -> Real {
        let mut shell = incline_patch(scale);
        let h = 1.0 / 240.0;
        // Settle phase: gravity along −n only (no tangential pull), so
        // the patch finds the true barrier equilibrium gap without the
        // launch transient injecting tangential velocity (a transient
        // in the kinetic regime never re-enters the stick well — that
        // is correct stick-slip physics, but not what this test
        // measures).
        let theta_i: Real = 0.5_f64.atan();
        let n_vec = Vec3::new(theta_i.sin(), theta_i.cos(), 0.0);
        shell.params.gravity = n_vec * -9.81;
        for _ in 0..120 {
            shell.step(h).expect("settle");
        }
        shell.params.gravity = Vec3::new(0.0, -9.81, 0.0);
        for v in shell.velocities.iter_mut() {
            *v = Vec3::ZERO;
        }
        let com0: Vec3 =
            shell.positions.iter().copied().sum::<Vec3>() / shell.positions.len() as Real;
        for step in 0..240 {
            let stats = shell.step(h).expect("step");
            assert!(stats.min_gap > 0.0, "step {step}: penetrated");
        }
        let com: Vec3 =
            shell.positions.iter().copied().sum::<Vec3>() / shell.positions.len() as Real;
        (com - com0).dot(t1) // downhill displacement
    };

    let d_stick = run(1.5); // μ_s = 0.6 > tan θ = 0.5
    let d_slide = run(1.0); // μ_s = μ_k = 0.4 < tan θ
    eprintln!("incline: stick moved {d_stick:.5} m, slide moved {d_slide:.5} m");
    assert!(
        d_stick.abs() < 2.0e-3,
        "μ_s = 0.6 > tanθ must STICK (creep budget 2 mm): {d_stick:.5}"
    );
    assert!(
        d_slide > 0.03 && d_slide > 10.0 * d_stick.abs(),
        "μ_s = μ_k = 0.4 < tanθ must SLIDE: {d_slide:.5}"
    );
}
