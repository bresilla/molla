fn build_heightfield_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.2);

    let n: u32 = 32;
    let cell = 0.5_f64;
    let half = (n as f64 - 1.0) * cell * 0.5;
    let mut heights = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            let x = (i as f64 - (n as f64 - 1.0) * 0.5) * cell;
            let z = (j as f64 - (n as f64 - 1.0) * 0.5) * cell;
            // Twin-frequency wavy field — bowls + ridges.
            let h = 0.5 * (x * 0.45).sin() * (z * 0.45).cos() + 0.2 * ((x + z) * 0.9).sin();
            heights.push(h);
        }
    }
    let hf = mp::Heightfield {
        nx: n,
        nz: n,
        cell_size: cell,
        origin: MVec3::new(-half, 0.0, -half),
        heights,
    };
    let field_id = b.add_heightfield(hf);
    b.add_shape_heightfield(mp::BodyId::NONE, field_id, mp::Transform::IDENTITY)
        .expect("heightfield");

    // Drop ~30 mixed primitives in a 6×5 grid pattern over the field.
    let mut rng_state: u64 = 0xCAFEBABEDEADBEEF;
    fn rng_uniform(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn random_quat(s: &mut u64) -> MQuat {
        let axis = MVec3::new(rng_uniform(s), rng_uniform(s), rng_uniform(s)).normalize_or_zero();
        let axis = if axis.length_squared() < 1e-9 {
            MVec3::Y
        } else {
            axis
        };
        let angle = rng_uniform(s) * std::f64::consts::PI;
        MQuat::from_axis_angle(axis, angle)
    }
    let mut visuals = Vec::new();
    let drop_grid = 6;
    let drop_spacing = 1.6_f64;
    for ix in 0..drop_grid {
        for iz in 0..5 {
            let x = (ix as f64 - (drop_grid as f64 - 1.0) * 0.5) * drop_spacing
                + 0.2 * rng_uniform(&mut rng_state);
            let z = (iz as f64 - 2.0) * drop_spacing + 0.2 * rng_uniform(&mut rng_state);
            let y = 6.0 + 0.5 * rng_uniform(&mut rng_state).abs();
            let pos = MVec3::new(x, y, z);
            let rot = random_quat(&mut rng_state);
            match (ix + iz) % 3 {
                0 => {
                    let r = 0.30_f64;
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia::sphere(1.0, r))
                            .with_pose(mp::Transform::new(pos, rot)),
                    );
                    b.add_shape_sphere(body, r, mp::Transform::IDENTITY)
                        .expect("sphere");
                    visuals.push(BodyVisual::Sphere(r as f32));
                }
                1 => {
                    let h = 0.30_f64;
                    let inertia = box_inertia(1.0, h);
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia {
                                mass: 1.0,
                                com: MVec3::ZERO,
                                inertia,
                            })
                            .with_pose(mp::Transform::new(pos, rot)),
                    );
                    b.add_shape_box(body, MVec3::splat(h), mp::Transform::IDENTITY)
                        .expect("box");
                    visuals.push(BodyVisual::Box(Vec3::splat(h as f32)));
                }
                _ => {
                    let half_h = 0.30_f64;
                    let radius = 0.18_f64;
                    let inertia = capsule_inertia(0.8, half_h, radius);
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia {
                                mass: 0.8,
                                com: MVec3::ZERO,
                                inertia,
                            })
                            .with_pose(mp::Transform::new(pos, rot)),
                    );
                    b.add_shape_capsule(body, half_h, radius, mp::Transform::IDENTITY)
                        .expect("capsule");
                    visuals.push(BodyVisual::Capsule {
                        half_h: half_h as f32,
                        radius: radius as f32,
                    });
                }
            }
        }
    }

    (b.build().expect("heightfield model"), visuals)
}
// Conveyor scene — kinematic belt (long box) plus dropped boxes that
// slide along. Port of `example_basic_conveyor.py` (linearised — we
// use a straight belt instead of the annular ring mesh, which would
// require trimesh authoring).
//
// The belt is body 0 (kinematic). The update system translates its
// pose along +X each frame at `belt_speed` m/s and snaps it back to

