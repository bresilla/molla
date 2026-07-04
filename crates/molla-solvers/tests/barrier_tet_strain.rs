//! C0 phase 4a — volumetric tets + HARD strain limiting.
//!
//! 1. FD gradcheck with active volume AND strain-barrier terms (a
//!    compressed 5-tet cube stretched past the strain band).
//! 2. Hard bound: a strip hung from two pinned corners with a weight
//!    at the tip — unlimited springs stretch far past 5%; with
//!    `strain_limit = 1.05` the max strain must stay < 1.05 at EVERY
//!    step (the line-search truncation is the guarantee, the barrier
//!    only keeps springs off the wall).
//! 3. Tet block dropped on the floor: penetration-free, settled, and
//!    volume preserved to a few percent.

use molla_math::{Real, Vec3};
use molla_solvers::barrier_contact::{
    BarrierObstacle, BarrierParams, BarrierShell, BarrierSpring, BarrierTet,
};

fn cube_5tets(
    origin: Vec3,
    edge: Real,
    kv: Real,
    k: Real,
) -> (Vec<Vec3>, Vec<BarrierSpring>, Vec<BarrierTet>) {
    let e = edge;
    let pos: Vec<Vec3> = [
        [0.0, 0.0, 0.0],
        [e, 0.0, 0.0],
        [0.0, e, 0.0],
        [e, e, 0.0],
        [0.0, 0.0, e],
        [e, 0.0, e],
        [0.0, e, e],
        [e, e, e],
    ]
    .iter()
    .map(|p| origin + Vec3::new(p[0], p[1], p[2]))
    .collect();
    let tet_ids: [[u32; 4]; 5] = [
        [0, 1, 3, 5],
        [0, 3, 2, 6],
        [0, 5, 4, 6],
        [3, 5, 6, 7],
        [0, 3, 5, 6],
    ];
    let mut seen = std::collections::BTreeSet::new();
    let mut springs = Vec::new();
    for t in &tet_ids {
        for a in 0..4 {
            for b in (a + 1)..4 {
                let (x, y) = (t[a].min(t[b]), t[a].max(t[b]));
                if seen.insert((x, y)) {
                    springs.push(BarrierSpring {
                        a: x,
                        b: y,
                        rest: (pos[x as usize] - pos[y as usize]).length(),
                        k,
                    });
                }
            }
        }
    }
    let tets = tet_ids.iter().map(|&ids| BarrierTet { ids, kv }).collect();
    (pos, springs, tets)
}

#[test]
fn fd_validates_volume_and_strain_terms() {
    let (mut pos, springs, tets) = cube_5tets(Vec3::new(0.0, 0.5, 0.0), 0.1, 2000.0, 300.0);
    // Deform: squash y (volume term active), stretch x hard enough to
    // push several springs into the strain-barrier band.
    for p in pos.iter_mut() {
        p.x *= 1.045;
        p.y = 0.5 + (p.y - 0.5) * 0.9;
    }
    let params = BarrierParams {
        strain_limit: Some(1.05),
        ..Default::default()
    };
    let shell = BarrierShell::new(pos.clone(), vec![0.1; 8], springs, Vec::new(), params)
        .expect("shell")
        .with_tets(tets);
    // NOTE: with_tets measured rest volumes at the DEFORMED positions;
    // rebuild v0 from the undeformed cube so the volume term is live.
    let mut shell = shell;
    let (pos0, _, tets0) = cube_5tets(Vec3::new(0.0, 0.5, 0.0), 0.1, 2000.0, 300.0);
    shell.tet_v0 = tets0
        .iter()
        .map(|t| {
            let a = pos0[t.ids[0] as usize];
            (pos0[t.ids[1] as usize] - a)
                .cross(pos0[t.ids[2] as usize] - a)
                .dot(pos0[t.ids[3] as usize] - a)
                / 6.0
        })
        .collect();

    let h = 1.0 / 240.0;
    let x_tilde: Vec<Vec3> = pos
        .iter()
        .map(|p| *p + Vec3::new(0.0, -2.0e-4, 0.0))
        .collect();
    let mut grad = vec![Vec3::ZERO; 8];
    shell.fd_probe_gradient(&pos, &x_tilde, h, &mut grad);

    let eps = 1.0e-7;
    let mut max_rel: Real = 0.0;
    for i in 0..8 {
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
            let denom = fd.abs().max(got.abs()).max(1.0e-4);
            max_rel = max_rel.max((fd - got).abs() / denom);
        }
    }
    eprintln!("tet+strain gradient FD max rel = {max_rel:.3e}");
    assert!(max_rel < 1.0e-5, "gradient FD mismatch: {max_rel:.3e}");
}

#[test]
fn strain_limit_is_a_hard_bound_under_load() {
    // A 6×2 strip hung from its two left corners, with 40× mass on the
    // two right-tip vertices. Soft springs (k = 40) would stretch far
    // past 5% under this load.
    let n = 6;
    let m = 2;
    let spacing = 0.03;
    let mut pos = Vec::new();
    let mut masses = Vec::new();
    for j in 0..m {
        for i in 0..n {
            pos.push(Vec3::new(i as Real * spacing, 0.0, j as Real * spacing));
            masses.push(0.02);
        }
    }
    let idx = |i: usize, j: usize| (j * n + i) as u32;
    masses[idx(n - 1, 0) as usize] = 0.8;
    masses[idx(n - 1, 1) as usize] = 0.8;
    let mut springs = Vec::new();
    for j in 0..m {
        for i in 0..n {
            if i + 1 < n {
                let (a, b) = (idx(i, j), idx(i + 1, j));
                springs.push(BarrierSpring {
                    a,
                    b,
                    rest: spacing,
                    k: 40.0,
                });
            }
            if j + 1 < m {
                let (a, b) = (idx(i, j), idx(i, j + 1));
                springs.push(BarrierSpring {
                    a,
                    b,
                    rest: spacing,
                    k: 40.0,
                });
            }
        }
    }

    let run = |limit: Option<Real>| -> (Real, BarrierShell) {
        let params = BarrierParams {
            strain_limit: limit,
            ..Default::default()
        };
        let mut shell = BarrierShell::new(
            pos.clone(),
            masses.clone(),
            springs.clone(),
            Vec::new(),
            params,
        )
        .expect("shell");
        shell.pin_vertices(&[idx(0, 0) as usize, idx(0, 1) as usize]);
        let h = 1.0 / 240.0;
        let mut worst: Real = 0.0;
        for _ in 0..240 {
            shell.step(h).expect("step");
            worst = worst.max(shell.max_strain());
        }
        (worst, shell)
    };

    let (strain_free, _) = run(None);
    let (strain_limited, _) = run(Some(1.05));
    eprintln!("strip strain: unlimited {strain_free:.4}, limited {strain_limited:.4}");
    assert!(
        strain_free > 1.10,
        "load too weak to prove anything: unlimited strain {strain_free:.4}"
    );
    assert!(
        strain_limited < 1.05,
        "HARD BOUND violated: {strain_limited:.4} ≥ 1.05"
    );
}

#[test]
fn tet_block_drops_and_keeps_volume() {
    let (pos, springs, tets) = cube_5tets(Vec3::new(-0.05, 0.08, -0.05), 0.1, 5000.0, 800.0);
    let kv_ids: Vec<[u32; 4]> = tets.iter().map(|t| t.ids).collect();
    let mut shell = BarrierShell::new(
        pos,
        vec![0.1; 8],
        springs,
        vec![BarrierObstacle::HalfSpace {
            n: Vec3::Y,
            offset: 0.0,
        }],
        BarrierParams::default(),
    )
    .expect("shell")
    .with_tets(tets);
    let v0_total: Real = shell.tet_v0.iter().sum();

    let h = 1.0 / 240.0;
    for step in 0..240 {
        let stats = shell.step(h).expect("step");
        assert!(
            stats.min_gap > 0.0,
            "step {step}: block penetrated the floor — gap {}",
            stats.min_gap
        );
    }
    let v_final: Real = kv_ids
        .iter()
        .map(|&ids| {
            let a = shell.positions[ids[0] as usize];
            (shell.positions[ids[1] as usize] - a)
                .cross(shell.positions[ids[2] as usize] - a)
                .dot(shell.positions[ids[3] as usize] - a)
                / 6.0
        })
        .sum();
    let v_max = shell
        .velocities
        .iter()
        .map(|v| v.length())
        .fold(0.0, Real::max);
    let vol_err = ((v_final - v0_total) / v0_total).abs();
    eprintln!("tet block: settled v_max {v_max:.3e}, volume drift {vol_err:.3e}");
    assert!(v_max < 0.05, "block did not settle: {v_max}");
    assert!(vol_err < 0.05, "volume not preserved: {vol_err:.3e}");
    assert!(shell.min_gap() > 0.0, "final intersection scan failed");
}
