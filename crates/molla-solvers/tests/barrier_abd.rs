//! C0 — ABD strong coupling: rigid bodies as 12-DOF affine transforms
//! INSIDE the barrier Newton solve. One incremental potential, one
//! CCD-truncated line search — no force exchange, no rigid-side TOI
//! patch.
//!
//! 1. Rigid cube drop: rests penetration-free on the floor and stays
//!    RIGID (‖AᵀA − I‖ small under the orthogonality potential).
//! 2. THE strong-coupling oracle: a FAST cube (5 m/s) fired at a
//!    pinned cloth — the exact scenario whose weak-coupling version
//!    tunneled until a hand-written rigid-side TOI was added. Here the
//!    unified line search handles it with no special-casing.
//! 3. Zero-g momentum: cube into free cloth conserves total linear
//!    momentum (unified implicit Euler).

use molla_math::{Mat3, Real, Vec3};
use molla_solvers::barrier_contact::{BarrierObstacle, BarrierParams, BarrierShell, BarrierSpring};

fn cube_samples(center: Vec3, half: Real) -> (Vec<Vec3>, Vec<Real>) {
    let mut s = Vec::new();
    // 8 corners + 6 face centres: enough proxies for stable contact.
    for &x in &[-half, half] {
        for &y in &[-half, half] {
            for &z in &[-half, half] {
                s.push(center + Vec3::new(x, y, z));
            }
        }
    }
    for c in 0..3 {
        for &sgn in &[-1.0, 1.0] {
            let mut v = Vec3::ZERO;
            v[c] = sgn * half;
            s.push(center + v);
        }
    }
    let m = vec![0.05; s.len()];
    (s, m)
}

fn ortho_error(a: Mat3) -> Real {
    let r = a.transpose() * a - Mat3::IDENTITY;
    (r.col(0).length_squared() + r.col(1).length_squared() + r.col(2).length_squared()).sqrt()
}

fn cloth_patch(
    n: usize,
    y: Real,
    span: Real,
) -> (Vec<Vec3>, Vec<BarrierSpring>, Vec<[u32; 3]>, Vec<Real>) {
    let spacing = span / (n - 1) as Real;
    let mut pos = Vec::new();
    for j in 0..n {
        for i in 0..n {
            pos.push(Vec3::new(
                i as Real * spacing - span * 0.5,
                y,
                j as Real * spacing - span * 0.5,
            ));
        }
    }
    let idx = |i: usize, j: usize| (j * n + i) as u32;
    let mut tris = Vec::new();
    for j in 0..n - 1 {
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
                    k: 800.0,
                });
            }
        }
    }
    let masses = vec![0.02; pos.len()];
    (pos, springs, tris, masses)
}

#[test]
fn rigid_cube_rests_on_floor_and_stays_orthogonal() {
    let mut shell = BarrierShell::new(
        Vec::new(),
        Vec::new(),
        Vec::new(),
        vec![BarrierObstacle::HalfSpace {
            n: Vec3::Y,
            offset: 0.0,
        }],
        BarrierParams::default(),
    )
    .expect("shell");
    let (s, m) = cube_samples(Vec3::new(0.0, 0.09, 0.0), 0.05);
    shell.add_affine_body(s, m, 1.0e5);

    let h = 1.0 / 240.0;
    for step in 0..240 {
        let stats = shell.step(h).expect("step");
        assert!(stats.min_gap > 0.0, "step {step}: cube penetrated floor");
    }
    let body = &shell.affine_bodies[0];
    let oe = ortho_error(body.a_mat);
    eprintln!(
        "cube rest: p.y {:.4}, |v_p| {:.3e}, ortho err {:.3e}",
        body.p.y,
        body.v_p.length(),
        oe
    );
    assert!(body.v_p.length() < 0.05, "cube did not settle");
    // Center one half-extent above the floor (inside the band).
    assert!(
        (body.p.y - 0.05).abs() < 2.0e-3,
        "cube rest height off: {}",
        body.p.y
    );
    assert!(oe < 1.0e-3, "cube lost rigidity: {oe:.3e}");
}

#[test]
fn fast_cube_into_cloth_needs_no_toi_patch() {
    let (pos, springs, tris, masses) = cloth_patch(9, 0.0, 0.4);
    let n_cloth = pos.len();
    let mut shell = BarrierShell::new(pos, masses, springs, Vec::new(), BarrierParams::default())
        .expect("shell");
    // Pin the boundary ring.
    let ring: Vec<usize> = (0..n_cloth)
        .filter(|&k| {
            let (i, j) = (k % 9, k / 9);
            i == 0 || j == 0 || i == 8 || j == 8
        })
        .collect();
    shell.pin_vertices(&ring);
    let (s, m) = cube_samples(Vec3::new(0.0, 0.12, 0.0), 0.04);
    let body = shell.add_affine_body(s, m, 1.0e5);
    shell = shell.with_triangles(tris);
    // Fire it DOWN at 5 m/s: 2 cm per step at h = 1/240 vs a 1 mm
    // band — the weak-coupling tunneling case, handled here by the
    // unified CCD line search alone.
    shell.affine_bodies[body].v_p = Vec3::new(0.0, -5.0, 0.0);

    let h = 1.0 / 240.0;
    for step in 0..240 {
        let stats = shell.step(h).expect("step");
        assert!(
            stats.min_gap > 0.0,
            "step {step}: STRONG-coupling invariant violated — gap {}",
            stats.min_gap
        );
    }
    let b = &shell.affine_bodies[0];
    eprintln!(
        "fast cube: p.y {:.4}, |v_p| {:.3e}, ortho {:.3e}",
        b.p.y,
        b.v_p.length(),
        ortho_error(b.a_mat)
    );
    // Caught by the cloth (never through it to the region far below).
    assert!(b.p.y > -0.2, "cube tunneled through the cloth: {}", b.p.y);
    // The undamped cube-on-trampoline system keeps oscillating; the
    // arrest gate is "lost ≥ 80% of the 5 m/s impact speed", not
    // stillness (measured residual ≈ 0.52 m/s).
    assert!(
        b.v_p.length() < 1.0,
        "cube not arrested: {}",
        b.v_p.length()
    );
}

#[test]
fn zero_g_cube_cloth_impact_conserves_momentum() {
    let (pos, springs, tris, masses) = cloth_patch(7, 0.0, 0.3);
    let cloth_m: Real = masses.iter().sum();
    let mut shell = BarrierShell::new(
        pos,
        masses,
        springs,
        Vec::new(),
        BarrierParams {
            gravity: Vec3::ZERO,
            ..Default::default()
        },
    )
    .expect("shell");
    let (s, m) = cube_samples(Vec3::new(0.0, 0.1, 0.0), 0.04);
    let cube_m: Real = m.iter().sum();
    let body = shell.add_affine_body(s, m, 1.0e5);
    shell = shell.with_triangles(tris);
    shell.affine_bodies[body].v_p = Vec3::new(0.0, -1.5, 0.0);

    let p_total = |sh: &BarrierShell| -> Vec3 {
        let mut p = sh.affine_bodies[0].v_p * cube_m;
        let per = cloth_m / (7.0 * 7.0);
        for (i, v) in sh.velocities.iter().enumerate() {
            if sh.proxy_of[i].is_none() {
                p += *v * per;
            }
        }
        p
    };
    let p0 = p_total(&shell);
    let h = 1.0 / 240.0;
    for step in 0..120 {
        let stats = shell.step(h).expect("step");
        assert!(stats.min_gap > 0.0, "step {step}: penetrated");
    }
    let p1 = p_total(&shell);
    let drift = (p1 - p0).length() / p0.length();
    eprintln!(
        "abd momentum: |p0| {:.4} |p1| {:.4} drift {:.3e}",
        p0.length(),
        p1.length(),
        drift
    );
    assert!(drift < 0.02, "momentum drift {drift:.3e}");
}
