const SHAPES_FIELD_N: u32 = 16;
const SHAPES_FIELD_CELL: f32 = 0.6;
const SHAPES_TORUS_MAJOR: f32 = 1.0;
const SHAPES_TORUS_MINOR: f32 = 0.25;
const SHAPES_TORUS_Y: f32 = 3.5;

// Stage-13 demo scene — drops cones, convex hulls, and spheres
// onto a wavy heightfield with a static torus (SDF) sticking out
// of the middle. Single scene exercises every primitive shipped
// in Stage 13: Cone, ConvexHull, Heightfield, Sdf.
fn build_shapes_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.2);

    let mut visuals: Vec<BodyVisual> = Vec::new();

    // ── Heightfield (gently wavy terrain) — static, no body. ────
    let n = SHAPES_FIELD_N;
    let cell = SHAPES_FIELD_CELL as f64;
    let half = (n as f64 - 1.0) * cell * 0.5;
    let mut heights = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            let x = (i as f64 - (n as f64 - 1.0) * 0.5) * cell;
            let z = (j as f64 - (n as f64 - 1.0) * 0.5) * cell;
            // Gentler waves than before — small amplitude so things
            // settle visibly instead of rolling off into the abyss.
            let h = 0.25 * ((x * 0.5).sin() + (z * 0.5).cos());
            heights.push(h);
        }
    }
    let hf = mp::Heightfield {
        nx: n,
        nz: n,
        cell_size: cell,
        cell_size_z: cell,
        origin: MVec3::new(-half, 0.0, -half),
        heights,
    };
    let field_id = b.add_heightfield(hf);
    b.add_shape_heightfield(mp::BodyId::NONE, field_id, mp::Transform::IDENTITY)
        .expect("heightfield shape");

    // ── Static torus (SDF) embedded in the terrain centre. ──────
    let sdf_id = b.add_sdf(mp::SdfShape::Torus {
        major: SHAPES_TORUS_MAJOR as f64,
        minor: SHAPES_TORUS_MINOR as f64,
    });
    b.add_shape_sdf(
        mp::BodyId::NONE,
        sdf_id,
        mp::Transform::from_translation(MVec3::new(0.0, SHAPES_TORUS_Y as f64, 0.0)),
    )
    .expect("torus sdf");

    // ── Falling cones (3 of them, scattered above the field). ───
    for (i, pos) in [
        MVec3::new(-1.5, 8.0, -1.0),
        MVec3::new(0.5, 9.0, 0.5),
        MVec3::new(1.8, 8.5, -0.5),
    ]
    .iter()
    .enumerate()
    {
        let half_h = 0.35_f64;
        let radius = 0.25_f64;
        let mass = 0.8_f64;
        let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
            mass * (radius * radius * 0.25 + (2.0 * half_h).powi(2) / 12.0),
            mass * radius * radius * 0.5,
            mass * (radius * radius * 0.25 + (2.0 * half_h).powi(2) / 12.0),
        ));
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_pose(mp::Transform::new(
                    *pos,
                    // Tilt each cone a bit so they tumble.
                    MQuat::from_axis_angle(MVec3::Z, (i as f64) * 0.3 + 0.4),
                )),
        );
        b.add_shape_cone(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("cone shape");
        visuals.push(BodyVisual::Cone {
            half_h: half_h as f32,
            radius: radius as f32,
        });
    }

    // ── Falling convex hulls (3 — actual cuboid hulls so the
    // visual matches the collision geometry). ──────────────────
    for (i, pos) in [
        MVec3::new(-2.5, 10.0, 1.5),
        MVec3::new(0.0, 10.5, -2.0),
        MVec3::new(2.5, 9.5, 1.5),
    ]
    .iter()
    .enumerate()
    {
        // 8-vertex cuboid hull — vertices at every (±sx, ±sy, ±sz)
        // corner. Slightly different proportions per cube so they
        // tumble interestingly.
        let half = MVec3::new(0.30, 0.40, 0.30);
        let verts = vec![
            MVec3::new(-half.x, -half.y, -half.z),
            MVec3::new(half.x, -half.y, -half.z),
            MVec3::new(-half.x, half.y, -half.z),
            MVec3::new(half.x, half.y, -half.z),
            MVec3::new(-half.x, -half.y, half.z),
            MVec3::new(half.x, -half.y, half.z),
            MVec3::new(-half.x, half.y, half.z),
            MVec3::new(half.x, half.y, half.z),
        ];
        let mass = 0.6;
        let h2 = half * half;
        let m_third = mass / 3.0;
        let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
            m_third * (h2.y + h2.z),
            m_third * (h2.x + h2.z),
            m_third * (h2.x + h2.y),
        ));
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_pose(mp::Transform::new(
                    *pos,
                    MQuat::from_axis_angle(MVec3::X, (i as f64) * 0.5),
                )),
        );
        let hull_id = b.add_hull(verts);
        b.add_shape_convex_hull(body, hull_id, mp::Transform::IDENTITY)
            .expect("hull shape");
        // Render as the matching box. The hull's bounding box is
        // exactly the cube formed by ±half, so the visual
        // matches the collision shape 1:1.
        visuals.push(BodyVisual::Box(Vec3::new(
            half.x as f32,
            half.y as f32,
            half.z as f32,
        )));
    }

    // ── Falling spheres for variety. ────────────────────────────
    for (i, pos) in [
        MVec3::new(-0.8, 11.0, 0.0),
        MVec3::new(0.8, 11.5, 1.5),
        MVec3::new(-1.5, 11.5, -1.5),
    ]
    .iter()
    .enumerate()
    {
        let _ = i;
        let radius = 0.3;
        let mass = 1.0;
        let i_v = (2.0 / 5.0) * mass * radius * radius;
        let inertia = molla::math::Mat3::from_diagonal(MVec3::new(i_v, i_v, i_v));
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_position(*pos),
        );
        b.add_shape_sphere(body, radius, mp::Transform::IDENTITY)
            .expect("sphere shape");
        // Rolling friction so spheres come to rest on the wavy
        // heightfield instead of rolling forever.
        b.set_shape_friction_modes(mp::ShapeId::from_index(b.shape_count() - 1), 0.05, 0.02)
            .ok();
        visuals.push(BodyVisual::Sphere(radius as f32));
    }

    // ── Ellipsoid (Stage 30) — one tilted oblate body that lands on
    // the wavy field and demonstrates the new Sphere/Plane and GJK
    // pairs in action.
    {
        let half_axes = MVec3::new(0.45, 0.20, 0.35);
        let mass = 0.9_f64;
        let h2 = half_axes * half_axes;
        let m_fifth = mass / 5.0;
        let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
            m_fifth * (h2.y + h2.z),
            m_fifth * (h2.x + h2.z),
            m_fifth * (h2.x + h2.y),
        ));
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_pose(mp::Transform::new(
                    MVec3::new(1.2, 12.0, 1.2),
                    MQuat::from_axis_angle(MVec3::Z, 0.6),
                )),
        );
        b.add_shape_ellipsoid(body, half_axes, mp::Transform::IDENTITY)
            .expect("ellipsoid shape");
        visuals.push(BodyVisual::Ellipsoid {
            half_axes: Vec3::new(half_axes.x as f32, half_axes.y as f32, half_axes.z as f32),
        });
    }

    (b.build().expect("shapes model"), visuals)
}

fn shapes_pane(ui: &mut egui::Ui, sim: &Sim) {
    ui.label("Stage 13 primitives:");
    ui.separator();
    readout(
        ui,
        "Heightfield",
        &format!("{}×{} grid", SHAPES_FIELD_N, SHAPES_FIELD_N),
    );
    readout(
        ui,
        "Torus (SDF)",
        &format!(
            "major {:.2} m, minor {:.2} m",
            SHAPES_TORUS_MAJOR, SHAPES_TORUS_MINOR
        ),
    );
    ui.separator();
    readout(ui, "Bodies", &format!("{}", sim.model.body_count));
    readout(ui, "Shapes", &format!("{}", sim.model.shape_count));
    ui.label("");
    ui.label("Drop test: cones, hulls, spheres rain");
    ui.label("onto the wavy heightfield. Static");
    ui.label("torus sticks out of the middle.");
}

// Right-rail pane for the Stage-21 mesh-tools showcase.
// Three panels:
//   1. Original (left)  / Decimated (centre)  side-by-side toggles.
//   2. Marching-cubes-from-SDF surface (right) toggle.

