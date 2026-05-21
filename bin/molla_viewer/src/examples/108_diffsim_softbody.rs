fn build_diffsim_softbody_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.35);

    let n = DIFFSIM_SOFTBODY_GRID;
    let dx = DIFFSIM_SOFTBODY_SIDE / (n as f64 - 1.0);
    let origin = MVec3::new(
        -DIFFSIM_SOFTBODY_SIDE * 0.5,
        DIFFSIM_SOFTBODY_TOP_Y,
        -DIFFSIM_SOFTBODY_SIDE * 0.5,
    );
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = origin + MVec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                b.add_particle(p, 0.04, dx * 0.45, mp::WorldId(0));
            }
        }
    }

    let v_at = |i: u32, j: u32, k: u32| -> u32 { (k * n + j) * n + i };
    let mut tets = Vec::new();
    for kk in 0..n - 1 {
        for jj in 0..n - 1 {
            for ii in 0..n - 1 {
                let corners = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                tets.extend(jelly_cube_to_six_tets(corners));
            }
        }
    }
    b.add_tet_mesh(&tets, 2.0e4, 0.42, None)
        .expect("diffsim softbody tet mesh");

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("diffsim softbody model"), Vec::new())
}

// BrickWall scene — a wide, low brick wall struck by a heavy
// projectile from one side. Geometry-only port of
// `xtra/newton/newton/examples/contacts/example_brick_stacking.py`
// (the full Newton example loads a Franka arm and stacks bricks
// with IK; we ship only the wall + projectile so the WIDE-not-TALL
// layout differentiates this from `Pyramid` and `Tower`).
const BRICK_HALF: MVec3 = MVec3::new(0.18, 0.10, 0.10);
const BRICK_GAP: f64 = 0.005;
const BRICK_ROWS: usize = 6;

