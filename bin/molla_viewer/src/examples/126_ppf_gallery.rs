// PPF gallery (scenes 126+) — molla renditions of every example in
// st-tech/ppf-contact-solver, all running the barrier-Newton backend.
// Where their notebooks DRIVE boundaries (twisting ends, motorized
// rollers) we use equivalent initial conditions / static stand-ins and
// say so in each scene description. Shared builders live here; the
// per-scene files are thin.

/// Cloth grid + soft surface via the standard builder.
fn ppf_add_cloth(
    b: &mut mp::ModelBuilder,
    origin: MVec3,
    u_axis: MVec3,
    v_axis: MVec3,
    n: u32,
    compliance: f64,
    max_stretch: f64,
) {
    let _ = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        n,
        n,
        0.02,
        0.005,
        compliance,
        max_stretch,
        mp::WorldId(0),
    );
}

/// Pin every particle whose INITIAL position satisfies `pred`
/// (inv_mass = 0 → true DoF removal in the barrier solver).
fn ppf_pin_where(model: &mut mp::Model, pred: impl Fn(MVec3) -> bool) {
    let q = model.initial_particle_q.host().unwrap().to_vec();
    let inv = model.particle_inv_mass.host_mut().unwrap();
    for (i, p) in q.iter().enumerate() {
        if pred(*p) {
            inv[i] = 0.0;
        }
    }
}

/// A particle chain with distance constraints (rod/yarn segment for
/// the chains solver kind). Returns the first particle id.
fn ppf_add_chain(b: &mut mp::ModelBuilder, points: &[MVec3], compliance: f64) -> u32 {
    let mut first = 0u32;
    let mut prev: Option<(u32, MVec3)> = None;
    for (i, p) in points.iter().enumerate() {
        let pid = b.add_particle(*p, 0.01, 0.004, mp::WorldId(0));
        if i == 0 {
            first = pid;
        }
        if let Some((a, ap)) = prev {
            b.add_distance_constraint(a, pid, (*p - ap).length(), compliance, 1.3);
        }
        prev = Some((pid, *p));
    }
    first
}

// ── 126 drape/curtain (drape.ipynb + curtain.ipynb): top-edge-pinned
// sheet falls into self-contacting folds.
fn build_ppf_curtain_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    ppf_add_cloth(
        &mut b,
        MVec3::new(-0.24, 0.42, 0.0),
        MVec3::new(0.48, 0.0, 0.0),
        MVec3::new(0.0, -0.40, 0.02),
        14,
        2.5e-3,
        1.2,
    );
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    let mut model = b.build().expect("model");
    ppf_pin_where(&mut model, |p| (p.y - 0.42).abs() < 1.0e-6);
    (model, Vec::new())
}

// ── 127 twist family (twist / five-twist / large-twist /
// large-five-twist.ipynb): a strip pinned at both ends, REPOSITIONED
// after build into five full helix turns (their notebooks drive the
// ends; we start twisted with flat rest lengths — the tensioned
// self-contact physics is the same showcase).
fn build_ppf_twist_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let n = 24u32;
    let len = 0.5_f64;
    let width = 0.09_f64;
    // Flat strip via the standard cloth builder (n × n grid squashed
    // into a strip via a narrow v-axis: rests come from THIS layout).
    let _ = b.add_cloth_grid(
        MVec3::new(-len * 0.5, 0.25, -width * 0.5),
        MVec3::new(len, 0.0, 0.0),
        MVec3::new(0.0, 0.0, width),
        n,
        4,
        0.01,
        0.004,
        1.0e-3,
        1.2,
        mp::WorldId(0),
    );
    let mut model = b.build().expect("model");
    // Re-pose into a 5-turn helix about the x-axis.
    {
        let q = model.initial_particle_q.host_mut().unwrap();
        for p in q.iter_mut() {
            let t = (p.x + len * 0.5) / len;
            let ang = 5.0 * std::f64::consts::TAU * t;
            let w = p.z; // signed cross-section coordinate
            *p = MVec3::new(p.x, 0.25 + w * ang.sin(), w * ang.cos());
        }
    }
    ppf_pin_where(&mut model, |p| p.x.abs() > len * 0.5 - 1.2e-2);
    (model, Vec::new())
}

// ── 128 friction ramp (friction.ipynb): patch dropped onto a 30°
// incline; the pane's μ sliders flip it live between stick and slide.
fn build_ppf_friction_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let theta = 30.0_f64.to_radians();
    let n_vec = MVec3::new(theta.sin(), theta.cos(), 0.0);
    ppf_add_cloth(
        &mut b,
        MVec3::new(-0.1, 0.35, -0.15),
        MVec3::new(0.3, 0.0, 0.0),
        MVec3::new(0.0, 0.0, 0.3),
        10,
        2.5e-3,
        1.5,
    );
    b.add_shape_plane(
        mp::BodyId::NONE,
        n_vec,
        mp::Transform::from_translation(MVec3::new(0.0, 0.2, 0.0)),
    )
    .expect("ramp");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor");
    (b.build().expect("model"), Vec::new())
}

// ── 129 hang (hang.ipynb): two-corner hang with a hard 5% strain
// limit — "not rubbery".
fn build_ppf_hang_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    ppf_add_cloth(
        &mut b,
        MVec3::new(-0.2, 0.4, 0.0),
        MVec3::new(0.4, 0.0, 0.0),
        MVec3::new(0.0, -0.35, 0.0),
        12,
        2.5e-3,
        1.05,
    );
    let mut model = b.build().expect("model");
    ppf_pin_where(&mut model, |p| {
        (p.y - 0.4).abs() < 1.0e-6 && p.x.abs() > 0.18
    });
    (model, Vec::new())
}

// ── 130 stack (stack.ipynb): five sheets dropped from staggered
// heights, settling into a layered pile.
fn build_ppf_stack_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    for layer in 0..5 {
        let s = 0.3 - 0.02 * f64::from(layer);
        ppf_add_cloth(
            &mut b,
            MVec3::new(-s * 0.5, 0.06 + 0.05 * f64::from(layer), -s * 0.5),
            MVec3::new(s, 0.0, 0.0),
            MVec3::new(0.0, 0.0, s),
            8,
            2.5e-3,
            1.3,
        );
    }
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    (b.build().expect("model"), Vec::new())
}
