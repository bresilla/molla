// --- Pile scene -----------------------------------------------------
const PILE_GRID_N: usize = 12;
// Tight horizontal spacing so bodies actually overlap on impact.
// Body half-extents are ~0.45, so 0.95 leaves a 0.05m gap → with the
// random rotation + drop-height stagger, plenty of off-axis impacts.
const PILE_SPACING: f32 = 0.95;
const PILE_SPHERE_RADIUS: f32 = 0.40;
const PILE_BOX_HALF: f32 = 0.40;
const PILE_CAPSULE_HALF_H: f32 = 0.40;
const PILE_CAPSULE_RADIUS: f32 = 0.22;
const PILE_DROP_HEIGHT: f32 = 4.0;
const PILE_DROP_BAND: f32 = 6.0;

fn build_pile_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);

    // Deterministic LCG so every reset gives the same pile but
    // different bodies have different rotations.
    let mut rng_state: u64 = 0x9E3779B97F4A7C15;
    fn next_unit(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (*s >> 11) as f64 / (1u64 << 53) as f64;
        v * 2.0 - 1.0 // ∈ [-1, 1)
    }
    fn random_rotation(s: &mut u64) -> MQuat {
        let axis = MVec3::new(next_unit(s), next_unit(s), next_unit(s)).normalize_or_zero();
        let axis = if axis.length_squared() < 1e-9 {
            MVec3::Y
        } else {
            axis
        };
        let angle = next_unit(s) * std::f64::consts::PI;
        MQuat::from_axis_angle(axis, angle)
    }

    let mut visuals: Vec<BodyVisual> = Vec::with_capacity(PILE_GRID_N * PILE_GRID_N);
    for ix in 0..PILE_GRID_N {
        for iz in 0..PILE_GRID_N {
            let x = (ix as f64 - (PILE_GRID_N as f64 - 1.0) * 0.5) * PILE_SPACING as f64
                + 0.15 * next_unit(&mut rng_state);
            let z = (iz as f64 - (PILE_GRID_N as f64 - 1.0) * 0.5) * PILE_SPACING as f64
                + 0.15 * next_unit(&mut rng_state);
            // Spread the drop heights over a band so contacts come in
            // waves — the falling stack pushes the lower bodies sideways.
            let y = PILE_DROP_HEIGHT as f64
                + (((ix + iz) as f64) / ((PILE_GRID_N + PILE_GRID_N) as f64))
                    * PILE_DROP_BAND as f64
                + 0.1 * next_unit(&mut rng_state);
            let pos = MVec3::new(x, y, z);
            let rot = random_rotation(&mut rng_state);

            match (ix + iz) % 3 {
                0 => {
                    // Spheres are rotation-invariant — random rotation
                    // doesn't help, so give them a random tangential
                    // angular velocity instead. We can't set initial
                    // qd through BodyParams; that's done after build().
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia::sphere(
                                1.0,
                                PILE_SPHERE_RADIUS as f64,
                            ))
                            .with_position(pos),
                    );
                    b.add_shape_sphere(body, PILE_SPHERE_RADIUS as f64, mp::Transform::IDENTITY)
                        .expect("sphere shape");
                    visuals.push(BodyVisual::Sphere(PILE_SPHERE_RADIUS));
                }
                1 => {
                    let half = MVec3::splat(PILE_BOX_HALF as f64);
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia {
                                mass: 1.0,
                                com: MVec3::ZERO,
                                inertia: box_inertia(1.0, PILE_BOX_HALF as f64),
                            })
                            .with_pose(mp::Transform::new(pos, rot)),
                    );
                    b.add_shape_box(body, half, mp::Transform::IDENTITY)
                        .expect("box shape");
                    visuals.push(BodyVisual::Box(Vec3::splat(PILE_BOX_HALF)));
                }
                _ => {
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia {
                                mass: 1.0,
                                com: MVec3::ZERO,
                                inertia: capsule_inertia(
                                    1.0,
                                    PILE_CAPSULE_HALF_H as f64,
                                    PILE_CAPSULE_RADIUS as f64,
                                ),
                            })
                            .with_pose(mp::Transform::new(pos, rot)),
                    );
                    b.add_shape_capsule(
                        body,
                        PILE_CAPSULE_HALF_H as f64,
                        PILE_CAPSULE_RADIUS as f64,
                        mp::Transform::IDENTITY,
                    )
                    .expect("capsule shape");
                    visuals.push(BodyVisual::Capsule {
                        half_h: PILE_CAPSULE_HALF_H,
                        radius: PILE_CAPSULE_RADIUS,
                    });
                }
            }
        }
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    (b.build().expect("model"), visuals)
}

// Vertical chain of small spheres. The top link is **kinematic**
// (mass-zero, integrators leave its pose alone) so the rope dangles
// from a fixed anchor. Adjacent links are pinned with `CONNECT`
// equality constraints — this exercises the stage-7 XPBD constraint

