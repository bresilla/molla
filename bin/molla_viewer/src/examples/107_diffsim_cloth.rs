fn build_diffsim_cloth_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.45);

    let origin = MVec3::new(
        -DIFFSIM_CLOTH_SIZE * 0.5,
        DIFFSIM_CLOTH_TOP_Y,
        -DIFFSIM_CLOTH_SIZE * 0.5,
    );
    let u_axis = MVec3::new(DIFFSIM_CLOTH_SIZE, 0.0, 0.0);
    let v_axis = MVec3::new(0.0, 0.0, DIFFSIM_CLOTH_SIZE);
    let (first, _, _) = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        DIFFSIM_CLOTH_NX,
        DIFFSIM_CLOTH_NX,
        0.035,
        0.018,
        5.0e-6,
        0.05,
        mp::WorldId(0),
    );

    // Pin the -X edge. The reduced differentiable oracle below optimises a
    // scalar lift/control for this pinned edge; the live viewer uses the same
    // topology as the GPU-forward cloth path while adjoints stay CPU-side.
    for iy in 0..DIFFSIM_CLOTH_NX {
        b.pin_particle(first + iy * DIFFSIM_CLOTH_NX);
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("diffsim cloth model"), Vec::new())
}

const DIFFSIM_SOFTBODY_GRID: u32 = 4;
const DIFFSIM_SOFTBODY_SIDE: f64 = 0.75;
const DIFFSIM_SOFTBODY_TOP_Y: f64 = 1.8;

