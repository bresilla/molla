// PPF gallery, part 2 — coupling, chains, knots, boxes.

// ── 131 trampoline (trampoline.ipynb): rigid ball dropped onto a
// boundary-pinned sheet; weak rigid coupling carries the bounce.
fn build_ppf_trampoline_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    ppf_add_cloth(
        &mut b,
        MVec3::new(-0.2, 0.0, -0.2),
        MVec3::new(0.4, 0.0, 0.0),
        MVec3::new(0.0, 0.0, 0.4),
        11,
        1.0e-3,
        1.4,
    );
    let r = 0.06;
    let ball = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: 0.35,
                com: MVec3::ZERO,
                inertia: mp::Mat3::from_diagonal(MVec3::splat(0.4 * 0.35 * r * r)),
            })
            .with_pose(mp::Transform::from_translation(MVec3::new(0.0, 0.25, 0.0))),
    );
    b.add_shape_sphere(ball, r, mp::Transform::IDENTITY)
        .expect("ball");
    let mut model = b.build().expect("model");
    ppf_pin_where(&mut model, |p| {
        p.x.abs() > 0.18 || p.z.abs() > 0.18
    });
    (model, vec![BodyVisual::Sphere(r as f32)])
}

// ── 132 trapped (trapped.ipynb): a sheet squeezed into the narrow
// gap between two offset spheres — barrier gaps on both sides.
fn build_ppf_trapped_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    ppf_add_cloth(
        &mut b,
        MVec3::new(-0.18, 0.3, -0.18),
        MVec3::new(0.36, 0.0, 0.0),
        MVec3::new(0.0, 0.0, 0.36),
        12,
        2.5e-3,
        1.3,
    );
    b.add_shape_sphere(
        mp::BodyId::NONE,
        0.1,
        mp::Transform::from_translation(MVec3::new(-0.055, 0.1, 0.0)),
    )
    .expect("s1");
    b.add_shape_sphere(
        mp::BodyId::NONE,
        0.1,
        mp::Transform::from_translation(MVec3::new(0.055, 0.02, 0.0)),
    )
    .expect("s2");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor");
    (b.build().expect("model"), Vec::new())
}

// ── 133 needle (needle.ipynb): a draped sheet poked by a tiny sphere
// "needle tip" — sharp tent, no puncture (adaptation: sphere tip
// instead of their cone mesh).
fn build_ppf_needle_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    ppf_add_cloth(
        &mut b,
        MVec3::new(-0.15, 0.12, -0.15),
        MVec3::new(0.3, 0.0, 0.0),
        MVec3::new(0.0, 0.0, 0.3),
        13,
        2.5e-3,
        1.15,
    );
    b.add_shape_sphere(
        mp::BodyId::NONE,
        0.012,
        mp::Transform::from_translation(MVec3::new(0.0, 0.09, 0.0)),
    )
    .expect("needle");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor");
    (b.build().expect("model"), Vec::new())
}

// ── 134 belt + roller (belt.ipynb / roller.ipynb): a closed band
// draped over two sphere "pulleys" (adaptation: static pulleys; their
// notebooks motorize them).
fn build_ppf_belt_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let n = 48u32;
    let (cx1, cx2, cy, r_loop) = (-0.09_f64, 0.09_f64, 0.18_f64, 0.085_f64);
    // Two-strand band: a pair of parallel closed chains with rungs.
    let mut ids = Vec::new();
    for strand in 0..2 {
        let z = 0.02 * (f64::from(strand) - 0.5);
        let mut strand_ids = Vec::new();
        for i in 0..n {
            let t = f64::from(i) / f64::from(n) * std::f64::consts::TAU;
            // Stadium curve around both pulleys.
            let (x, y) = if t.cos() >= 0.0 {
                (cx2 + r_loop * t.sin(), cy + r_loop * -t.cos())
            } else {
                (cx1 + r_loop * t.sin(), cy + r_loop * -t.cos())
            };
            strand_ids.push(b.add_particle(MVec3::new(x, y, z), 0.008, 0.004, mp::WorldId(0)));
        }
        for i in 0..n as usize {
            let a = strand_ids[i];
            let c = strand_ids[(i + 1) % n as usize];
            b.add_distance_constraint(a, c, 0.012, 1.0e-3, 1.25);
        }
        ids.push(strand_ids);
    }
    for i in 0..n as usize {
        b.add_distance_constraint(ids[0][i], ids[1][i], 0.02, 1.0e-3, 1.25);
    }
    b.add_shape_sphere(
        mp::BodyId::NONE,
        0.06,
        mp::Transform::from_translation(MVec3::new(cx1, cy, 0.0)),
    )
    .expect("pulley1");
    b.add_shape_sphere(
        mp::BodyId::NONE,
        0.06,
        mp::Transform::from_translation(MVec3::new(cx2, cy, 0.0)),
    )
    .expect("pulley2");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor");
    (b.build().expect("model"), Vec::new())
}

// ── 135 noodle + yarn (noodle.ipynb / yarn.ipynb): a dozen strands
// dropped into a pile — EE chain contact everywhere.
fn build_ppf_noodle_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    for k in 0..12 {
        let phase = f64::from(k) * 0.9;
        let y0 = 0.15 + 0.03 * f64::from(k);
        let pts: Vec<MVec3> = (0..40)
            .map(|i| {
                let t = f64::from(i) / 39.0;
                MVec3::new(
                    -0.25 + 0.5 * t,
                    y0 + 0.01 * (8.0 * t * std::f64::consts::TAU + phase).sin(),
                    0.05 * (phase + 3.0 * t).sin(),
                )
            })
            .collect();
        ppf_add_chain(&mut b, &pts, 1.0e-3);
    }
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor");
    (b.build().expect("model"), Vec::new())
}

// ── 136 woven (woven.ipynb / large-woven.ipynb): warp/weft strands
// with over-under initial interleaving, borders pinned — the weave
// holds under gravity through EE contact alone.
fn build_ppf_woven_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let m = 7usize;
    let span = 0.36_f64;
    let amp = 0.006_f64;
    let d = span / f64::from((m - 1) as u32); // crossing spacing
    // Heights keyed to the CROSSING lattice: warp at (x_l, z_k) sits
    // at +amp·(−1)^{k+l}, weft at −amp·(−1)^{k+l} — every crossing
    // separates by exactly 2·amp with alternating over/under. (A
    // per-sample phase instead gave half the crossings ZERO gap.)
    for k in 0..m {
        let zc = -span * 0.5 + d * f64::from(k as u32);
        let sign_k = if k % 2 == 0 { 1.0 } else { -1.0 };
        let pts: Vec<MVec3> = (0..36)
            .map(|i| {
                let t = f64::from(i) / 35.0;
                let x = -span * 0.5 + span * t;
                let u = (x + span * 0.5) / d;
                let weave = amp * (std::f64::consts::PI * u).cos() * sign_k;
                MVec3::new(x, 0.25 + weave, zc)
            })
            .collect();
        ppf_add_chain(&mut b, &pts, 1.0e-3);
    }
    for k in 0..m {
        let xc = -span * 0.5 + d * f64::from(k as u32);
        let sign_k = if k % 2 == 0 { 1.0 } else { -1.0 };
        let pts: Vec<MVec3> = (0..36)
            .map(|i| {
                let t = f64::from(i) / 35.0;
                let z = -span * 0.5 + span * t;
                let v = (z + span * 0.5) / d;
                let weave = amp * (std::f64::consts::PI * v).cos() * sign_k;
                MVec3::new(xc, 0.25 - weave, z)
            })
            .collect();
        ppf_add_chain(&mut b, &pts, 1.0e-3);
    }
    let mut model = b.build().expect("model");
    ppf_pin_where(&mut model, |p| {
        p.x.abs() > span * 0.5 - 1.0e-3 || p.z.abs() > span * 0.5 - 1.0e-3
    });
    (model, Vec::new())
}

// ── 137 fishing knot (fishingknot.ipynb): a loose overhand knot in a
// single strand, one end pinned, weight on the other — gravity
// tightens the knot and EE contact keeps it from pulling through.
fn build_ppf_knot_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    // Parametric loose overhand (trefoil-like) curve, then a straight
    // tail hanging down.
    let mut pts = Vec::new();
    for i in 0..30 {
        // 30 samples over 29/30·τ — the trefoil is a CLOSED curve, and
        // sampling the full period would make the open chain's first
        // and last points coincide (distance-zero unconnected ends).
        let t = f64::from(i) / 30.0 * std::f64::consts::TAU;
        pts.push(MVec3::new(
            0.05 * (t.sin() + 2.0 * (2.0 * t).sin()) * 0.5,
            0.3 + 0.05 * (t.cos() - 2.0 * (2.0 * t).cos()) * 0.5,
            0.05 * -(3.0 * t).sin() * 0.5,
        ));
    }
    // Tail with a slight natural helix — an EXACTLY straight chain is
    // degenerate (all its segments are mutually colinear, distance 0).
    for i in 1..=14 {
        let last = *pts.last().unwrap();
        let a = f64::from(i) * 0.9;
        pts.push(last + MVec3::new(0.003 * a.cos(), -0.02, 0.003 * a.sin()));
    }
    let first = ppf_add_chain(&mut b, &pts, 8.0e-4);
    // Heavier bob on the tail end.
    let bob = b.add_particle(
        pts.last().unwrap().to_owned() + MVec3::new(0.0, -0.02, 0.0),
        0.12,
        0.008,
        mp::WorldId(0),
    );
    b.add_distance_constraint(bob - 1, bob, 0.02, 8.0e-4, 1.2);
    let mut model = b.build().expect("model");
    // Pin the knot's leading end.
    {
        let inv = model.particle_inv_mass.host_mut().unwrap();
        inv[first as usize] = 0.0;
        inv[first as usize + 1] = 0.0;
    }
    (model, Vec::new())
}

// ── 138 domino (domino.ipynb): a chain of ABD boxes toppled by the
// first — strong coupling through the Model path.
fn build_ppf_domino_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let half = MVec3::new(0.006, 0.05, 0.025);
    let gap = 0.055;
    let count = 8;
    let mut visuals = Vec::new();
    for k in 0..count {
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 0.05,
                    com: MVec3::ZERO,
                    inertia: mp::Mat3::from_diagonal(MVec3::new(
                        0.05 * (half.y * half.y + half.z * half.z) / 3.0,
                        0.05 * (half.x * half.x + half.z * half.z) / 3.0,
                        0.05 * (half.x * half.x + half.y * half.y) / 3.0,
                    )),
                })
                .with_pose(mp::Transform::from_translation(MVec3::new(
                    -0.18 + f64::from(k) * gap,
                    half.y + 5.0e-4,
                    0.0,
                ))),
        );
        b.add_shape_box(body, half, mp::Transform::IDENTITY)
            .expect("domino");
        visuals.push(BodyVisual::Box(Vec3::new(
            half.x as f32,
            half.y as f32,
            half.z as f32,
        )));
    }
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor");
    let model = b.build().expect("model");
    (model, visuals)
}

// ── 139 cards (cards.ipynb): leaning ABD plate pairs on a
// friction floor — the card-house statics test.
fn build_ppf_cards_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    // Thickness along x (the lean plane), width along z.
    let half = MVec3::new(0.0015, 0.05, 0.035);
    let lean = 24.0_f64.to_radians();
    let mut visuals = Vec::new();
    for pair in 0..3 {
        let x0 = -0.12 + f64::from(pair) * 0.12;
        for side in 0..2 {
            // A-frame: with rotation φ = s·lean the card's top-centre
            // sits at centre_x − s·h_y·sin(lean); left card (side 0)
            // takes s = −1 so its top leans toward +x. Tops meet at
            // x0 ± clearance (thickness + 2 mm — a positive starting
            // gap, the barrier holds the lean from there).
            let s = if side == 0 { -1.0 } else { 1.0 };
            let clearance = half.x + 2.0e-3;
            let rot = mp::Quat::from_axis_angle(MVec3::Z, s * lean);
            let cx = x0 - s * (clearance + half.y * lean.sin());
            let body = b.add_body(
                mp::BodyParams::new()
                    .with_spatial_inertia(mp::SpatialInertia {
                        mass: 0.01,
                        com: MVec3::ZERO,
                        inertia: mp::Mat3::from_diagonal(MVec3::new(
                            0.01 * (half.y * half.y + half.z * half.z) / 3.0,
                            0.01 * (half.x * half.x + half.z * half.z) / 3.0,
                            0.01 * (half.x * half.x + half.y * half.y) / 3.0,
                        )),
                    })
                    .with_pose(mp::Transform::new(
                        MVec3::new(cx, half.y * lean.cos() + 1.0e-3, 0.0),
                        rot,
                    )),
            );
            b.add_shape_box(body, half, mp::Transform::IDENTITY)
                .expect("card");
            visuals.push(BodyVisual::Box(Vec3::new(
                half.x as f32,
                half.y as f32,
                half.z as f32,
            )));
        }
    }
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor");
    (b.build().expect("model"), visuals)
}

// ── 140 codim (codim.ipynb): mixed codimension in ONE solve — a
// sheet (2D), strands (1D) and an ABD box (rigid) all colliding.
fn build_ppf_codim_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    ppf_add_cloth(
        &mut b,
        MVec3::new(-0.15, 0.2, -0.15),
        MVec3::new(0.3, 0.0, 0.0),
        MVec3::new(0.0, 0.0, 0.3),
        10,
        2.5e-3,
        1.3,
    );
    for k in 0..4 {
        let pts: Vec<MVec3> = (0..30)
            .map(|i| {
                let t = f64::from(i) / 29.0;
                // Gentle wave: perfectly straight chains are
                // degenerate (colinear segments at distance zero).
                MVec3::new(
                    -0.2 + 0.4 * t,
                    0.3 + 0.02 * f64::from(k) + 0.004 * (9.0 * t + f64::from(k)).sin(),
                    -0.06 + 0.04 * f64::from(k),
                )
            })
            .collect();
        ppf_add_chain(&mut b, &pts, 1.0e-3);
    }
    let half = MVec3::new(0.03, 0.03, 0.03);
    let body = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: 0.08,
                com: MVec3::ZERO,
                inertia: mp::Mat3::from_diagonal(MVec3::splat(
                    0.08 * 2.0 * half.x * half.x / 3.0,
                )),
            })
            .with_pose(mp::Transform::from_translation(MVec3::new(0.0, 0.42, 0.0))),
    );
    b.add_shape_box(body, half, mp::Transform::IDENTITY)
        .expect("box");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor");
    let model = b.build().expect("model");
    (
        model,
        vec![BodyVisual::Box(Vec3::new(
            half.x as f32,
            half.y as f32,
            half.z as f32,
        ))],
    )
}

// ── 141 fitting (fitting.ipynb): a poncho sheet dropped over a
// sphere-mannequin (adaptation: their body mesh → head + torso +
// shoulder spheres).
fn build_ppf_fitting_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    ppf_add_cloth(
        &mut b,
        MVec3::new(-0.22, 0.5, -0.22),
        MVec3::new(0.44, 0.0, 0.0),
        MVec3::new(0.0, 0.0, 0.44),
        14,
        2.5e-3,
        1.2,
    );
    let mannequin = [
        (MVec3::new(0.0, 0.38, 0.0), 0.06),  // head
        (MVec3::new(0.0, 0.22, 0.0), 0.10),  // torso
        (MVec3::new(-0.12, 0.28, 0.0), 0.05), // shoulder L
        (MVec3::new(0.12, 0.28, 0.0), 0.05),  // shoulder R
    ];
    for (c, r) in mannequin {
        b.add_shape_sphere(mp::BodyId::NONE, r, mp::Transform::from_translation(c))
            .expect("mannequin");
    }
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor");
    (b.build().expect("model"), Vec::new())
}

// ── 142 ribbon (ribbon.ipynb): a long thin strip falling on its edge,
// buckling into folds with self-contact.
fn build_ppf_ribbon_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let _ = b.add_cloth_grid(
        MVec3::new(-0.3, 0.25, -0.02),
        MVec3::new(0.6, 0.0, 0.0),
        MVec3::new(0.0, 0.0, 0.04),
        28,
        3,
        0.01,
        0.004,
        1.5e-3,
        1.15,
        mp::WorldId(0),
    );
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor");
    (b.build().expect("model"), Vec::new())
}
