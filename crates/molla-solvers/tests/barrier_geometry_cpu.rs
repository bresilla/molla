use molla_math::{Mat3, Real, Vec3};
use molla_solvers::barrier_contact::{
    BarrierObstacle, BarrierParams, BarrierShell, BarrierSpring, BarrierTet,
};

fn shell(positions: Vec<Vec3>) -> BarrierShell {
    let n = positions.len();
    BarrierShell::new(
        positions,
        vec![1.0; n],
        vec![],
        vec![],
        BarrierParams {
            gravity: Vec3::ZERO,
            ..Default::default()
        },
    )
    .unwrap()
}

fn rejected_unchanged(shell: &mut BarrierShell, h: Real, message: &str) {
    let positions = shell.positions.clone();
    let velocities = shell.velocities.clone();
    let bodies: Vec<_> = shell
        .affine_bodies
        .iter()
        .map(|b| (b.p, b.a_mat, b.v_p, b.v_a))
        .collect();
    let err = shell.step(h).unwrap_err().to_string();
    assert!(err.contains(message), "{err}; expected {message}");
    assert_eq!(shell.positions, positions);
    assert_eq!(shell.velocities, velocities);
    assert_eq!(
        shell
            .affine_bodies
            .iter()
            .map(|b| (b.p, b.a_mat, b.v_p, b.v_a))
            .collect::<Vec<_>>(),
        bodies
    );
}

#[test]
fn piercing_coplanar_and_isolated_vertices_are_refused() {
    let sheet = [[-1.0, 0.0, -0.4], [1.0, 0.0, -0.4], [0.0, 0.0, 0.5]];
    for second in [
        vec![[-0.3, -0.5, 0.0], [0.3, 0.5, 0.0], [0.4, -0.5, 0.3]],
        vec![[-0.2, 0.0, -0.2], [0.2, 0.0, -0.2], [0.0, 0.0, 0.2]],
        vec![[-1.0, 0.0, 0.4], [1.0, 0.0, 0.4], [0.0, 0.0, -0.5]],
        vec![[0.0; 3]],
    ] {
        for axis in 0..3 {
            for scale in [1.0e-6, 1.0, 1.0e6] {
                let positions = sheet
                    .iter()
                    .chain(&second)
                    .map(|p| Vec3::new(p[axis], p[(axis + 1) % 3], p[(axis + 2) % 3]) * scale)
                    .collect();
                let mut triangles = vec![[0, 1, 2]];
                if second.len() == 3 {
                    triangles.push([3, 4, 5]);
                }
                if axis == 1 {
                    triangles[0].swap(1, 2);
                }
                let mut s = shell(positions).with_triangles(triangles);
                assert!(s.has_intersections());
                rejected_unchanged(&mut s, 1.0 / 240.0, "mesh INTERSECTS");
            }
        }
    }
}

#[test]
fn shared_topology_is_valid_but_overlap_at_a_shared_vertex_is_not() {
    for scale in [1.0e-6, 1.0, 1.0e6] {
        let positions = vec![
            Vec3::ZERO,
            Vec3::X * scale,
            Vec3::Z * scale,
            (Vec3::X + Vec3::Z) * scale,
        ];
        let mut s = shell(positions.clone()).with_triangles(vec![[0, 1, 2], [1, 3, 2]]);
        s.params.dhat = scale * 1.0e-3;
        assert!(!s.has_intersections());
        s.step(1.0 / 240.0).unwrap();
        assert_eq!(s.positions, positions);
    }
    let mut s = shell(vec![
        Vec3::ZERO,
        Vec3::X,
        Vec3::Z,
        Vec3::new(0.2, 0.0, 0.2),
        Vec3::new(0.1, 0.0, 0.3),
    ])
    .with_triangles(vec![[0, 1, 2], [0, 3, 4]]);
    rejected_unchanged(&mut s, 1.0 / 240.0, "mesh INTERSECTS");
}

#[test]
fn explicit_rod_edges_and_degenerate_triangles_are_checked() {
    let mut s = shell(vec![
        Vec3::ZERO,
        Vec3::X,
        Vec3::Z,
        Vec3::new(0.2, -1.0, 0.2),
        Vec3::new(0.2, 1.0, 0.2),
    ])
    .with_triangles(vec![[0, 1, 2]])
    .with_collision_edges(vec![[3, 4]]);
    rejected_unchanged(&mut s, 1.0 / 240.0, "mesh INTERSECTS");
    let mut s = shell(vec![Vec3::ZERO, Vec3::X, Vec3::X * 2.0]).with_triangles(vec![[0, 1, 2]]);
    rejected_unchanged(&mut s, 1.0 / 240.0, "degenerate triangle");
}

#[test]
fn malformed_mutable_topology_is_rejected_before_indexing() {
    let mut initial = shell(vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z])
        .with_triangles(vec![[0, 1, 2]])
        .with_tets(vec![BarrierTet {
            ids: [0, 1, 2, 3],
            kv: 1.0,
        }]);
    initial.springs.push(BarrierSpring {
        a: 0,
        b: 1,
        rest: 1.0,
        k: 1.0,
    });
    let mutations: &[fn(&mut BarrierShell)] = &[
        |s| s.springs[0].b = u32::MAX,
        |s| s.triangles[0][2] = u32::MAX,
        |s| s.triangles[0][2] = 3,
        |s| s.collision_edges[0][0] = u32::MAX,
        |s| {
            s.collision_edges.pop();
        },
        |s| s.tets[0].ids[3] = u32::MAX,
        |s| s.tet_v0[0] = 0.0,
        |s| s.tets[0].kv = Real::INFINITY,
        |s| s.proxy_of[0] = Some(0),
        |s| s.masses[0] = 0.0,
        |s| {
            s.pinned.pop();
        },
    ];
    for mutate in mutations {
        let mut s = initial.clone();
        mutate(&mut s);
        rejected_unchanged(&mut s, 1.0 / 240.0, "invalid");
    }
}

#[test]
fn tet_inversion_is_not_published_with_or_without_affine_bodies() {
    for affine in [false, true] {
        let mut s =
            shell(vec![Vec3::ZERO, Vec3::X, Vec3::Z, Vec3::Y]).with_tets(vec![BarrierTet {
                ids: [0, 1, 2, 3],
                kv: 0.0,
            }]);
        s.pin_vertices(&[0, 1, 2]);
        s.params.gravity = Vec3::new(0.0, -9.81, 0.0);
        if affine {
            s.add_affine_body(
                vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z]
                    .into_iter()
                    .map(|p| p + Vec3::splat(5.0))
                    .collect(),
                vec![1.0; 4],
                1.0,
            );
        }
        rejected_unchanged(
            &mut s,
            0.5,
            "accepted geometry rejected: degenerate or inverted tet",
        );
        s.params.gravity = Vec3::ZERO;
        let before = s.positions.clone();
        s.step(1.0 / 240.0).unwrap();
        assert_eq!(s.positions, before);
    }
}

#[test]
fn numerical_overflow_is_refused_and_valid_retry_succeeds() {
    let mut s = shell(vec![Vec3::Y]);
    s.velocities[0] = Vec3::X * Real::MAX;
    rejected_unchanged(&mut s, 16.0, "predicted positions");
    s.velocities[0] = Vec3::X * 1.0e200;
    rejected_unchanged(&mut s, 1.0, "CG residual");
    s.velocities[0] = Vec3::X;
    s.step(1.0 / 240.0).unwrap();
    assert!(s.positions[0].x > 0.0);

    let mut s = shell(vec![]);
    s.add_affine_body(
        vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
        vec![1.0; 4],
        1.0,
    );
    s.affine_bodies[0].v_a = Mat3::IDENTITY * Real::MAX;
    rejected_unchanged(&mut s, 16.0, "nonfinite affine prediction");
    s.affine_bodies[0].v_a = Mat3::ZERO;
    s.step(1.0 / 240.0).unwrap();
    s.affine_bodies[0].j_mat = Mat3::from_diagonal(Vec3::new(-1.0, -1.0, 1.0));
    rejected_unchanged(&mut s, 1.0 / 240.0, "positive-definite inertia");
}

#[test]
fn recovery_opt_out_does_not_publish_a_still_invalid_state() {
    let mut s = shell(vec![-Vec3::Y * 0.1]);
    s.obstacles.push(BarrierObstacle::HalfSpace {
        n: Vec3::Y,
        offset: 0.0,
    });
    s.params.reject_intersecting_start = false;
    s.params.newton_tol = 1.0e30;
    rejected_unchanged(
        &mut s,
        1.0 / 240.0,
        "accepted geometry rejected: obstacle INTERSECTS",
    );
}

#[test]
fn enabled_strain_bounds_and_pinned_obstacle_exemption_are_preserved() {
    let mut s = shell(vec![Vec3::ZERO, Vec3::X * 2.0]);
    s.springs.push(BarrierSpring {
        a: 0,
        b: 1,
        rest: 1.0,
        k: 1.0,
    });
    s.params.strain_limit = Some(1.1);
    rejected_unchanged(&mut s, 1.0 / 240.0, "strain limit");
    let mut s = shell(vec![Vec3::ZERO]);
    s.obstacles.push(BarrierObstacle::Sphere {
        center: Vec3::ZERO,
        radius: 1.0,
    });
    s.pin_vertices(&[0]);
    s.step(1.0 / 240.0).unwrap();
    assert_eq!(s.positions, vec![Vec3::ZERO]);
}
