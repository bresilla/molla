//! C0 phase 3 — self-collision oracle contract (VT + EE barrier pairs).
//!
//! 1. FD gradcheck with ACTIVE vertex-triangle and edge-edge pairs
//!    (two offset patches inside each other's barrier band).
//! 2. Two-layer drop: a free patch falls onto a (heavy-pinned) patch —
//!    layers must never interpenetrate and must settle stacked with a
//!    positive inter-layer gap.
//! 3. Crossed ribbons (45°): the falling ribbon meets the lower one
//!    edge-on — the EE pairs carry the contact.
//!
//! Every test ends with the explicit intersection scan (`min_gap`,
//! which now includes all non-adjacent VT/EE distances).

use molla_math::{Real, Vec3};
use molla_solvers::barrier_contact::{BarrierObstacle, BarrierParams, BarrierShell, BarrierSpring};

/// Triangulated n×m patch in the y=const plane; returns positions,
/// springs (edges+diagonals of the triangulation) and triangles.
#[allow(clippy::type_complexity)]
fn patch(
    n: usize,
    m: usize,
    y: Real,
    spacing: Real,
    origin: [Real; 2],
    mass: Real,
) -> (Vec<Vec3>, Vec<BarrierSpring>, Vec<[u32; 3]>, Vec<Real>) {
    let mut pos = Vec::new();
    let mut masses = Vec::new();
    for j in 0..m {
        for i in 0..n {
            pos.push(Vec3::new(
                origin[0] + i as Real * spacing,
                y,
                origin[1] + j as Real * spacing,
            ));
            masses.push(mass);
        }
    }
    let idx = |i: usize, j: usize| (j * n + i) as u32;
    let mut tris = Vec::new();
    for j in 0..m - 1 {
        for i in 0..n - 1 {
            tris.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
            tris.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut springs = Vec::new();
    for t in &tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            if seen.insert((a.min(b), a.max(b))) {
                springs.push(BarrierSpring {
                    a,
                    b,
                    rest: (pos[a as usize] - pos[b as usize]).length(),
                    k: 400.0,
                });
            }
        }
    }
    (pos, springs, tris, masses)
}

/// Two patches merged into one shell, the lower one pinned via huge
/// masses; vertical separation `dz` between the layers.
fn two_layers(dz: Real) -> (BarrierShell, usize) {
    let (mut pos, mut springs, mut tris, mut masses) = patch(5, 5, dz, 0.03, [-0.06, -0.06], 0.02);
    let n_top = pos.len();
    let (pos_b, springs_b, tris_b, _) = patch(7, 7, 0.0, 0.03, [-0.09, -0.09], 0.02);
    let base = n_top as u32;
    pos.extend(pos_b.iter().copied());
    masses.extend(std::iter::repeat_n(0.02, pos_b.len()));
    springs.extend(springs_b.iter().map(|s| BarrierSpring {
        a: s.a + base,
        b: s.b + base,
        ..*s
    }));
    tris.extend(
        tris_b
            .iter()
            .map(|t| [t[0] + base, t[1] + base, t[2] + base]),
    );
    let n_total = pos.len();
    let mut shell = BarrierShell::new(pos, masses, springs, Vec::new(), BarrierParams::default())
        .expect("shell")
        .with_triangles(tris);
    shell.pin_vertices(&(n_top..n_total).collect::<Vec<_>>());
    (shell, n_top)
}

#[test]
fn fd_validates_self_pair_gradient() {
    // Two layers 0.7·d̂ apart: VT and EE pairs are ACTIVE.
    let (mut shell, n_top) = two_layers(0.7e-3);
    // De-generify: aligned grids put every closest point EXACTLY on a
    // region boundary of the clamped closest-point map — the measure-
    // zero kinks where the a.e.-exact constant-weight gradient is
    // one-sided (probing there compares FD across a kink against one
    // side's slope). Jitter the free layer tangentially so all pairs
    // sit strictly inside a region.
    let mut seed = 0x5EED_u64;
    let mut rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / f64::from(1u32 << 31)) - 0.5
    };
    for p in shell.positions[..n_top].iter_mut() {
        p.x += 0.011 * rng();
        p.z += 0.011 * rng();
        p.y += 1.0e-4 * rng();
    }
    let pos = shell.positions.clone();
    let h = 1.0 / 240.0;
    let x_tilde: Vec<Vec3> = pos
        .iter()
        .map(|p| *p + Vec3::new(0.0, -1.0e-4, 0.0))
        .collect();

    let mut grad = vec![Vec3::ZERO; pos.len()];
    shell.fd_probe_gradient(&pos, &x_tilde, h, &mut grad);
    // Self-pair forces must actually be present on the free layer.
    let f_top: Real = grad[..n_top].iter().map(|g| g.length()).sum();
    assert!(f_top > 1.0e-9, "no active self-pair forces: {f_top:.3e}");

    let eps = 1.0e-7;
    let mut max_rel: Real = 0.0;
    for i in (0..pos.len()).step_by(7).filter(|&i| !shell.pinned[i]) {
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
            // Denominator floor above the FD noise floor: E is
            // O(100) here (κ-scaled barrier terms), so central
            // differences carry ~E·ε_f64/eps ≈ 3e-7 of noise —
            // near-zero true derivatives (tangential components,
            // at-rest springs) would otherwise score rel = 1.
            let denom = fd.abs().max(got.abs()).max(1.0e-4);
            let rel = (fd - got).abs() / denom;
            if rel > 1.0e-4 {
                eprintln!("  probe v{i} c{c}: fd {fd:.6e} got {got:.6e} rel {rel:.3e}");
            }
            max_rel = max_rel.max(rel);
        }
    }
    eprintln!("self-pair gradient FD max rel = {max_rel:.3e}");
    assert!(
        max_rel < 1.0e-4,
        "self-pair gradient FD mismatch: {max_rel:.3e}"
    );
}

#[test]
fn two_layer_drop_stacks_without_interpenetration() {
    let (mut shell, n_top) = two_layers(0.02);
    let h = 1.0 / 240.0;
    for step in 0..200 {
        let stats = shell.step(h).expect("step");
        assert!(
            stats.min_gap > 0.0,
            "step {step}: layers interpenetrated — gap {}",
            stats.min_gap
        );
    }
    let v_max = shell.velocities[..n_top]
        .iter()
        .map(|v| v.length())
        .fold(0.0, Real::max);
    let y_top: Real = shell.positions[..n_top].iter().map(|p| p.y).sum::<Real>() / n_top as Real;
    let gap = shell.min_gap();
    eprintln!("two-layer: settled y_top {y_top:.4}, gap {gap:.2e}, v_max {v_max:.3e}");
    assert!(v_max < 0.05, "top layer did not settle: {v_max}");
    assert!(gap > 0.0, "final intersection scan failed: {gap:.3e}");
    // Stacked: the free layer rests roughly one barrier band above the
    // pinned one, not on the floor of the world.
    assert!(
        y_top > 0.0 && y_top < 5.0e-3,
        "top layer should rest just above the pinned layer: {y_top:.4}"
    );
}

#[test]
fn crossed_ribbons_meet_edge_on_without_passing() {
    // Lower ribbon along x (pinned, heavy); upper ribbon along z,
    // falling — the crossing region meets edge-against-edge.
    let (mut pos, mut springs, mut tris, mut masses) =
        patch(9, 2, 0.02, 0.03, [-0.12, -0.015], 0.02);
    let n_top = pos.len();
    let base = n_top as u32;
    let (pos_b, springs_b, tris_b, _) = patch(2, 9, 0.0, 0.03, [-0.015, -0.12], 0.02);
    pos.extend(pos_b.iter().copied());
    masses.extend(std::iter::repeat_n(0.02, pos_b.len()));
    springs.extend(springs_b.iter().map(|s| BarrierSpring {
        a: s.a + base,
        b: s.b + base,
        ..*s
    }));
    tris.extend(
        tris_b
            .iter()
            .map(|t| [t[0] + base, t[1] + base, t[2] + base]),
    );
    let mut shell = BarrierShell::new(
        pos,
        masses,
        springs,
        vec![BarrierObstacle::HalfSpace {
            n: Vec3::Y,
            offset: -0.5,
        }],
        BarrierParams::default(),
    )
    .expect("shell")
    .with_triangles(tris);
    let n_total = shell.positions.len();
    shell.pin_vertices(&(n_top..n_total).collect::<Vec<_>>());

    let h = 1.0 / 240.0;
    for step in 0..200 {
        let stats = shell.step(h).expect("step");
        assert!(
            stats.min_gap > 0.0,
            "step {step}: ribbons crossed — gap {}",
            stats.min_gap
        );
    }
    // The falling ribbon's center must be held up by the lower ribbon
    // (~y > 0), while its far ends drape below — proof the contact
    // carried load rather than letting it fall through to y = −0.5·g·t².
    let center_y = shell.positions[..n_top]
        .iter()
        .filter(|p| p.z.abs() < 0.02 && p.x.abs() < 0.02)
        .map(|p| p.y)
        .fold(Real::NEG_INFINITY, Real::max);
    let gap = shell.min_gap();
    eprintln!("ribbons: center y {center_y:.4}, final gap {gap:.2e}");
    assert!(gap > 0.0, "final intersection scan failed");
    assert!(
        center_y > 0.0,
        "upper ribbon fell through the lower one: center y = {center_y:.4}"
    );
}
