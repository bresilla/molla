//! PPF `fail-examples/` analogues: the solver must REFUSE bad input
//! loudly — never simulate through it.
//!
//! * intersect-1: a vertex starting INSIDE a sphere obstacle.
//! * intersect-2: two cloth layers built interpenetrating (negative
//!   self gap at t = 0).
//! * intersect-3: a chain threaded THROUGH a sheet at t = 0.
//! * assertion: nonsense parameters rejected up front.
//!
//! Plus the opt-out: `reject_intersecting_start = false` steps anyway
//! (deliberate recovery), and the barrier pushes outward.

use molla_math::{Real, Vec3};
use molla_solvers::barrier_contact::{BarrierObstacle, BarrierParams, BarrierShell, BarrierSpring};

fn flat_patch(n: usize, y: Real) -> (Vec<Vec3>, Vec<BarrierSpring>, Vec<[u32; 3]>) {
    let spacing = 0.03;
    let mut pos = Vec::new();
    for j in 0..n {
        for i in 0..n {
            pos.push(Vec3::new(
                (i as Real - (n as Real - 1.0) / 2.0) * spacing,
                y,
                (j as Real - (n as Real - 1.0) / 2.0) * spacing,
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
                    k: 400.0,
                });
            }
        }
    }
    (pos, springs, tris)
}

#[test]
fn intersect_1_vertex_inside_obstacle_is_refused() {
    let (pos, springs, _) = flat_patch(4, 0.0);
    // Sphere centred on the patch: several vertices start INSIDE.
    let mut shell = BarrierShell::new(
        pos,
        vec![0.02; 16],
        springs,
        vec![BarrierObstacle::Sphere {
            center: Vec3::ZERO,
            radius: 0.05,
        }],
        BarrierParams::default(),
    )
    .expect("build is fine — stepping is what must refuse");
    let err = shell.step(1.0 / 240.0).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("INTERSECTS"),
        "expected a loud intersection refusal, got: {msg}"
    );
}

#[test]
fn intersect_2_interpenetrating_layers_are_refused() {
    let (mut pos, mut springs, mut tris) = flat_patch(4, 0.0);
    let base = pos.len() as u32;
    // TILTED second layer: y = 0.3·z crosses the first layer's y = 0
    // plane — triangles genuinely PIERCE (the closest-distance metric
    // alone cannot see this; the Möller edge-triangle scan must).
    let (pos2, springs2, tris2) = flat_patch(4, 0.0);
    let pos2: Vec<Vec3> = pos2
        .into_iter()
        .map(|p| Vec3::new(p.x + 0.005, 0.3 * p.z, p.z + 0.003))
        .collect();
    pos.extend(pos2);
    springs.extend(springs2.into_iter().map(|s| BarrierSpring {
        a: s.a + base,
        b: s.b + base,
        ..s
    }));
    tris.extend(
        tris2
            .into_iter()
            .map(|t| [t[0] + base, t[1] + base, t[2] + base]),
    );
    let mut shell = BarrierShell::new(
        pos,
        vec![0.02; 32],
        springs,
        Vec::new(),
        BarrierParams::default(),
    )
    .expect("build")
    .with_triangles(tris);
    let err = shell.step(1.0 / 240.0).unwrap_err();
    assert!(format!("{err}").contains("INTERSECTS"));
}

#[test]
fn intersect_3_chain_threaded_through_sheet_is_refused() {
    let (mut pos, springs, tris) = flat_patch(5, 0.1);
    let base = pos.len() as u32;
    // A strand passing exactly through the sheet plane.
    let mut chain_springs = springs;
    for k in 0..8u32 {
        pos.push(Vec3::new(
            0.001,
            0.1 - 0.035 + 0.01 * Real::from(k),
            0.0007 * Real::from(k), // slight curve (no colinear degeneracy)
        ));
        if k > 0 {
            chain_springs.push(BarrierSpring {
                a: base + k - 1,
                b: base + k,
                rest: 0.0101,
                k: 400.0,
            });
        }
    }
    let chain_edges: Vec<[u32; 2]> = (1..8).map(|k| [base + k - 1, base + k]).collect();
    let mut shell = BarrierShell::new(
        pos,
        vec![0.02; 25 + 8],
        chain_springs,
        Vec::new(),
        BarrierParams::default(),
    )
    .expect("build")
    .with_triangles(tris)
    .with_collision_edges(chain_edges);
    let err = shell.step(1.0 / 240.0).unwrap_err();
    assert!(format!("{err}").contains("INTERSECTS"));
}

#[test]
fn assertion_bad_params_are_refused() {
    let (pos, springs, _) = flat_patch(3, 0.1);
    for params in [
        BarrierParams {
            dhat: 0.0,
            ..Default::default()
        },
        BarrierParams {
            kappa_scale: -1.0,
            ..Default::default()
        },
        BarrierParams {
            friction_mu: 0.5,
            friction_mu_s_scale: 0.5, // μ_s < μ_k: nonsense
            ..Default::default()
        },
    ] {
        let mut shell = BarrierShell::new(
            pos.clone(),
            vec![0.02; 9],
            springs.clone(),
            Vec::new(),
            params,
        )
        .expect("build");
        assert!(
            shell.step(1.0 / 240.0).is_err(),
            "params must be refused: {params:?}"
        );
    }
}

#[test]
fn opt_out_recovers_instead_of_refusing() {
    let (pos, springs, _) = flat_patch(4, 0.0);
    let mut shell = BarrierShell::new(
        pos,
        vec![0.02; 16],
        springs,
        vec![BarrierObstacle::Sphere {
            center: Vec3::ZERO,
            radius: 0.05,
        }],
        BarrierParams {
            reject_intersecting_start: false,
            gravity: Vec3::ZERO,
            ..Default::default()
        },
    )
    .expect("build");
    for _ in 0..120 {
        shell.step(1.0 / 240.0).expect("recovery stepping");
    }
    assert!(
        shell.min_gap() > 0.0,
        "barrier should have pushed the patch out: {}",
        shell.min_gap()
    );
}
