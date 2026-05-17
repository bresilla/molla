//! Stage 39E — `JointType::Cable` integration tests.
//!
//! Three end-to-end checks on the CPU Featherstone solver:
//!
//!   1. `cable_joint_2dof_count`
//!      A model with N cable joints exposes `2·N` DoFs (1 linear + 1
//!      bend per joint).
//!
//!   2. `cable_stretch_oscillates_at_correct_frequency`
//!      Single child body attached to the world by a cable. With unit
//!      mass and the cable axis perpendicular to gravity, the linear
//!      DoF behaves like a 1-D mass-spring oscillator with `ω = √(k/m)`.
//!      We track the first zero crossing and compare its time to the
//!      analytic quarter-period `T/4 = (π/2)/ω`.
//!
//!   3. `cable_bend_returns_to_rest`
//!      Perturb the bend DoF by ~10°, watch it decay back toward zero
//!      under the configured bend damping. Final `|q_b|` and `|q̇_b|`
//!      must be < 1 % of the initial bend.
//!
//! The cable convention (Stage 39E) is fixed: cable axis = joint-local
//! **+z**, bend axis = joint-local **+x**.

use molla_core::BodyId;
use molla_core::WorldId;
use molla_math::{Mat3, Quat, Real, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, Contacts, ModelBuilder};
use molla_solvers::{FeatherstoneSolver, Solver};

/// A spherical-inertia body with mass `m` and diagonal inertia tensor
/// `I·diag` — used by both oscillator tests.
fn unit_body(b: &mut ModelBuilder, mass: Real, inertia_scalar: Real) -> molla_core::BodyId {
    b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY * inertia_scalar,
    }))
}

#[test]
fn cable_joint_2dof_count() {
    let mut b = ModelBuilder::new();
    // Each joint connects world → its own body; we add four independent
    // articulations so the DoF accounting isn't entangled by tree
    // topology — purely the joint's own (coord, dof) layout.
    let mut prev_body: Option<BodyId> = None;
    let mut joint_count = 0usize;
    for i in 0..4 {
        let body = unit_body(&mut b, 1.0, 0.05);
        let parent = if i == 0 {
            BodyId::NONE
        } else {
            prev_body.unwrap()
        };
        b.add_joint_cable(
            parent,
            body,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.0, -0.5)),
            10.0, // stiffness_linear
            0.5,  // stiffness_bend
            0.0,
            0.0,
        )
        .unwrap();
        prev_body = Some(body);
        joint_count += 1;
    }
    let model = b.build().unwrap();

    // 2 DoFs per cable joint.
    assert_eq!(model.joint_count, joint_count);
    assert_eq!(model.joint_dof_total, 2 * joint_count);
    // 2 coords per cable joint (linear stretch + bend angle).
    assert_eq!(model.joint_coord_total, 2 * joint_count);
    // CSR offsets monotonically increase by 2 each entry.
    let dof_off = model.joint_dof_offset.host().unwrap();
    let coord_off = model.joint_coord_offset.host().unwrap();
    for j in 0..joint_count {
        assert_eq!(
            dof_off[j + 1] - dof_off[j],
            2,
            "joint {j} should expose 2 DoF, got {}",
            dof_off[j + 1] - dof_off[j]
        );
        assert_eq!(
            coord_off[j + 1] - coord_off[j],
            2,
            "joint {j} should expose 2 coords"
        );
    }
}

/// Single mass on a cable spring. With gravity (-y), the cable axis
/// pinned along world +z, and the body's COM placed on the cable
/// axis, the linear DoF decouples from gravity and reduces to a 1-D
/// mass-spring oscillator
///
/// ```text
///     m · q̈_l + k · q_l = 0     →     ω = √(k / m).
/// ```
///
/// We seed `q_l(0) = q0` and `q̇_l(0) = 0`, integrate until the first
/// zero-crossing of `q_l`, and check the elapsed time matches the
/// analytic quarter-period `T/4 = (π/2) / ω` within 5 %.
#[test]
fn cable_stretch_oscillates_at_correct_frequency() {
    let mass: Real = 1.0;
    let k_lin: Real = 100.0; // → ω = 10 rad/s, T = 0.628 s, T/4 = 0.157 s
    let omega = (k_lin / mass).sqrt();
    let quarter_period = std::f64::consts::FRAC_PI_2 / omega;

    // Single body. Place the joint frame so the cable axis (+z) is
    // perpendicular to gravity (-y). No `x_c` offset → child COM sits
    // on the cable axis at q_l = 0, so the stretch DoF doesn't couple
    // into the gravitational potential energy at the linearised order.
    let mut b = ModelBuilder::new();
    let body = unit_body(&mut b, mass, 0.1);
    let joint = b
        .add_joint_cable(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
            k_lin,
            0.0, // no bend stiffness (we only excite the linear DoF)
            0.0, // no linear damping
            0.0,
        )
        .unwrap();
    let _ = joint;
    // Disable gravity on the body's world so the test is a pure
    // mass-spring — the cable axis is perpendicular to default
    // gravity but any tiny coupling through the second-order ABA
    // recursion would still bias the period at the percent level.
    b.set_world_gravity(WorldId(0), V3::ZERO);
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let q0: Real = 0.05;
    s_a.joint_q.host_mut().unwrap()[0] = q0;
    s_a.joint_qd.host_mut().unwrap()[0] = 0.0;
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();

    let dt: Real = 0.0001;
    let max_steps = (2.5 * quarter_period / dt) as usize;
    let mut zero_cross_time: Option<Real> = None;
    let mut prev_q = q0;
    for step in 1..=max_steps {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q = s_a.joint_q.host().unwrap()[0];
        if zero_cross_time.is_none() && prev_q > 0.0 && q <= 0.0 {
            // Linear interp between the last two samples to refine the
            // zero-crossing time.
            let frac = prev_q / (prev_q - q);
            zero_cross_time = Some((step as Real - 1.0 + frac) * dt);
            break;
        }
        prev_q = q;
    }

    let t_zero = zero_cross_time
        .expect("cable stretch never crossed zero — solver may be over-damped or unstable");
    let err = (t_zero - quarter_period).abs() / quarter_period;
    eprintln!(
        "cable stretch ω = {:.3} rad/s, T/4 = {:.5}s, observed zero crossing = {:.5}s, err = {:.2}%",
        omega,
        quarter_period,
        t_zero,
        err * 100.0
    );
    assert!(
        err < 0.05,
        "cable stretch frequency off by {:.2}% (T/4 expected {:.5}s, got {:.5}s)",
        err * 100.0,
        quarter_period,
        t_zero
    );
}

/// Perturb the bend DoF by ~10° and verify it decays back to zero
/// under the configured bend damping. The body has spherical inertia
/// and we set bend stiffness `k_b` plus damping `c_b` such that the
/// linearised bend ODE
///
/// ```text
///     I · q̈_b + c_b · q̇_b + k_b · q_b = 0
/// ```
///
/// is well into the under-damped regime (`ζ ≈ 0.4`). After integrating
/// for roughly 6 natural periods we expect both `|q_b|` and `|q̇_b|`
/// to be < 1 % of the initial bend amplitude.
#[test]
fn cable_bend_returns_to_rest() {
    let mass: Real = 1.0;
    let inertia: Real = 0.05;
    let k_bend: Real = 5.0; // ω_b = √(k/I) ≈ 10 rad/s
    let c_bend: Real = 0.4; // ζ = c/(2·√(k·I)) ≈ 0.4

    let mut b = ModelBuilder::new();
    let body = unit_body(&mut b, mass, inertia);
    b.add_joint_cable(
        BodyId::NONE,
        body,
        Transform::IDENTITY,
        Transform::IDENTITY,
        0.0,
        k_bend,
        0.0,
        c_bend,
    )
    .unwrap();
    b.set_world_gravity(WorldId(0), V3::ZERO);
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let q_b0: Real = 0.175; // ~10°
    s_a.joint_q.host_mut().unwrap()[1] = q_b0; // bend coord
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();

    let dt: Real = 0.0005;
    // Run for ≈ 6 natural periods → 6 · 2π/ω ≈ 3.8 s.
    let total_time: Real = 4.0;
    let n_steps = (total_time / dt) as usize;
    let mut max_q_abs: Real = 0.0;
    for _ in 0..n_steps {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q_b = s_a.joint_q.host().unwrap()[1];
        if q_b.abs() > max_q_abs {
            max_q_abs = q_b.abs();
        }
    }
    let q_b_final = s_a.joint_q.host().unwrap()[1];
    let qd_b_final = s_a.joint_qd.host().unwrap()[1];
    eprintln!(
        "cable bend: q0 = {:.4} rad, max|q| during run = {:.4}, final q = {:.5}, final qd = {:.5}",
        q_b0, max_q_abs, q_b_final, qd_b_final
    );

    // Sanity: never exceeded the initial amplitude (no energy gain).
    assert!(
        max_q_abs <= q_b0 * 1.05,
        "bend amplitude grew during decay: max = {} > 1.05·q0 = {}",
        max_q_abs,
        q_b0 * 1.05
    );
    // Final state is within 1 % of rest on both coord and velocity.
    assert!(
        q_b_final.abs() < 0.01 * q_b0,
        "bend did not decay: final |q| = {} ≥ 1% of q0 = {}",
        q_b_final.abs(),
        q_b0
    );
    assert!(
        qd_b_final.abs() < 0.01 * q_b0 * (k_bend / inertia).sqrt(),
        "bend velocity did not decay: final |qd| = {}",
        qd_b_final.abs()
    );

    // Final body orientation should be near identity rotation.
    let body_q = s_a.body_q.host().unwrap();
    let r_final = body_q[0].rotation;
    let r_identity_err = (Quat::IDENTITY * r_final.inverse()).to_axis_angle().1.abs();
    assert!(
        r_identity_err < 0.02,
        "body rotation didn't return near identity: residual angle = {} rad",
        r_identity_err
    );
}

/// Procedural multi-segment cable chain: several cable joints are connected in
/// series, then seeded with alternating stretch/bend disturbances. With
/// damping and gravity disabled, the chain should dissipate those disturbances
/// without generating energy or NaNs. This is the small CPU oracle for longer
/// cable-pile/Y-junction follow-up cases.
#[test]
fn cable_chain_damps_stretch_and_bend_without_energy_gain() {
    let mass: Real = 1.0;
    let inertia: Real = 0.05;
    let k_lin: Real = 120.0;
    let k_bend: Real = 8.0;
    let c_lin: Real = 4.0;
    let c_bend: Real = 0.8;
    let joint_count = 5usize;

    let mut b = ModelBuilder::new();
    let mut parent = BodyId::NONE;
    for _ in 0..joint_count {
        let body = unit_body(&mut b, mass, inertia);
        b.add_joint_cable(
            parent,
            body,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.0, -0.4)),
            k_lin,
            k_bend,
            c_lin,
            c_bend,
        )
        .unwrap();
        parent = body;
    }
    b.set_world_gravity(WorldId(0), V3::ZERO);
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    {
        let q = s_a.joint_q.host_mut().unwrap();
        for j in 0..joint_count {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            q[2 * j] = sign * 0.025;
            q[2 * j + 1] = -sign * 0.12;
        }
    }
    let initial_norm = s_a
        .joint_q
        .host()
        .unwrap()
        .iter()
        .map(|q| q * q)
        .sum::<Real>()
        .sqrt();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();
    let dt: Real = 0.0005;
    let mut max_norm = initial_norm;

    for _ in 0..6000 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q = s_a.joint_q.host().unwrap();
        let qd = s_a.joint_qd.host().unwrap();
        let norm = q.iter().map(|q| q * q).sum::<Real>().sqrt();
        max_norm = max_norm.max(norm);
        assert!(q.iter().all(|v| v.is_finite()), "non-finite cable q: {q:?}");
        assert!(
            qd.iter().all(|v| v.is_finite()),
            "non-finite cable qd: {qd:?}"
        );
    }

    let q_final = s_a.joint_q.host().unwrap();
    let qd_final = s_a.joint_qd.host().unwrap();
    let final_norm = q_final.iter().map(|q| q * q).sum::<Real>().sqrt();
    let final_speed_norm = qd_final.iter().map(|qd| qd * qd).sum::<Real>().sqrt();
    eprintln!(
        "cable chain: initial_norm={initial_norm:.5}, max_norm={max_norm:.5}, final_norm={final_norm:.5}, final_speed_norm={final_speed_norm:.5}"
    );

    assert!(
        max_norm <= initial_norm * 1.10,
        "damped cable chain gained generalized displacement energy: max={max_norm}, initial={initial_norm}",
    );
    assert!(
        final_norm < initial_norm * 0.20,
        "damped cable chain did not return close to rest: final={final_norm}, initial={initial_norm}",
    );
    assert!(
        final_speed_norm < 0.08,
        "damped cable chain retained too much generalized speed: {final_speed_norm}",
    );
}

#[test]
fn cable_chain_sags_under_gravity_without_unbounded_stretch() {
    let mass: Real = 0.6;
    let inertia: Real = 0.04;
    let k_lin: Real = 180.0;
    let k_bend: Real = 9.0;
    let c_lin: Real = 6.0;
    let c_bend: Real = 1.1;
    let joint_count = 4usize;

    let mut b = ModelBuilder::new();
    let mut parent = BodyId::NONE;
    for _ in 0..joint_count {
        let body = unit_body(&mut b, mass, inertia);
        b.add_joint_cable(
            parent,
            body,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.0, -0.35)),
            k_lin,
            k_bend,
            c_lin,
            c_bend,
        )
        .unwrap();
        parent = body;
    }
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    {
        let q = s_a.joint_q.host_mut().unwrap();
        for j in 0..joint_count {
            q[2 * j + 1] = 0.03 * (j as Real + 1.0);
        }
    }
    let initial_tip_y = s_a.body_q.host().unwrap()[joint_count - 1].position.y;
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();
    let dt: Real = 0.0005;
    let mut max_stretch: Real = 0.0;
    let mut max_bend: Real = 0.0;

    for _ in 0..5000 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q = s_a.joint_q.host().unwrap();
        let qd = s_a.joint_qd.host().unwrap();
        assert!(
            q.iter().all(|v| v.is_finite()) && qd.iter().all(|v| v.is_finite()),
            "gravity cable chain produced non-finite q/qd: q={q:?}, qd={qd:?}"
        );
        for j in 0..joint_count {
            max_stretch = max_stretch.max(q[2 * j].abs());
            max_bend = max_bend.max(q[2 * j + 1].abs());
        }
    }

    let q = s_a.joint_q.host().unwrap();
    let qd = s_a.joint_qd.host().unwrap();
    let body_q = s_a.body_q.host().unwrap();
    let body_qd = s_a.body_qd.host().unwrap();
    let final_tip_y = body_q[joint_count - 1].position.y;
    let final_speed = qd.iter().map(|v| v * v).sum::<Real>().sqrt();
    eprintln!(
        "gravity cable chain: tip_y {initial_tip_y:.4}->{final_tip_y:.4}, max_stretch={max_stretch:.4}, max_bend={max_bend:.4}, final_speed={final_speed:.4}"
    );
    for i in 0..joint_count {
        assert!(
            body_q[i].position.is_finite()
                && body_qd[i].linear.is_finite()
                && body_qd[i].angular.is_finite(),
            "gravity cable body {i} escaped or became non-finite: pose={:?}, vel={:?}",
            body_q[i],
            body_qd[i]
        );
    }
    assert!(
        final_tip_y < initial_tip_y - 0.05,
        "cable tip did not sag under gravity: initial_y={initial_tip_y}, final_y={final_tip_y}"
    );
    assert!(
        max_stretch < 0.35,
        "cable stretch became unbounded under gravity: max_stretch={max_stretch}"
    );
    assert!(
        max_bend < 1.5,
        "cable bend became unbounded under gravity: max_bend={max_bend}"
    );
    assert!(
        final_speed < 0.8,
        "gravity cable chain did not damp toward quasi-static sag: final generalized speed={final_speed}, q={q:?}"
    );
}

/// Procedural Y-junction cable tree: one fixed hub owns three cable
/// branches. This is the asset-free CPU gate for branched cable
/// topology; unlike the serial chain above, all three cable joints
/// share the same parent body and therefore exercise Featherstone's
/// branching traversal with cable stretch/bend DoFs.
#[test]
fn cable_y_junction_branches_damp_without_cross_branch_instability() {
    let mass: Real = 0.8;
    let inertia: Real = 0.04;
    let k_lin: Real = 100.0;
    let k_bend: Real = 6.0;
    let c_lin: Real = 5.0;
    let c_bend: Real = 0.9;
    let branches = 3usize;

    let mut b = ModelBuilder::new();
    b.begin_articulation();
    let hub = unit_body(&mut b, 2.0, 0.2);
    b.add_joint_fixed(BodyId::NONE, hub, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();

    let branch_offsets = [
        V3::new(-0.25, 0.0, 0.0),
        V3::new(0.25, 0.0, 0.0),
        V3::new(0.0, 0.0, 0.25),
    ];
    for offset in branch_offsets {
        let body = unit_body(&mut b, mass, inertia);
        b.add_joint_cable(
            hub,
            body,
            Transform::from_translation(offset),
            Transform::from_translation(V3::new(0.0, 0.0, -0.35)),
            k_lin,
            k_bend,
            c_lin,
            c_bend,
        )
        .unwrap();
    }
    b.set_world_gravity(WorldId(0), V3::ZERO);
    let model = b.build().unwrap();
    assert_eq!(
        model.joint_count,
        branches + 1,
        "fixed hub + cable branches"
    );
    assert_eq!(model.joint_dof_total, 2 * branches);
    assert_eq!(model.joint_coord_total, 2 * branches);

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    {
        let q = s_a.joint_q.host_mut().unwrap();
        let qd = s_a.joint_qd.host_mut().unwrap();
        for j in 0..branches {
            let sign = if j == 1 { -1.0 } else { 1.0 };
            q[2 * j] = sign * 0.020;
            q[2 * j + 1] = (j as Real - 1.0) * 0.10;
            qd[2 * j] = -sign * 0.04;
            qd[2 * j + 1] = 0.02 * sign;
        }
    }
    let initial_norm = s_a
        .joint_q
        .host()
        .unwrap()
        .iter()
        .map(|q| q * q)
        .sum::<Real>()
        .sqrt();

    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();
    let dt: Real = 0.0005;
    let mut max_norm = initial_norm;
    for _ in 0..5000 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q = s_a.joint_q.host().unwrap();
        let qd = s_a.joint_qd.host().unwrap();
        let norm = q.iter().map(|q| q * q).sum::<Real>().sqrt();
        max_norm = max_norm.max(norm);
        assert!(
            q.iter().all(|v| v.is_finite()) && qd.iter().all(|v| v.is_finite()),
            "Y-junction produced non-finite q/qd: q={q:?}, qd={qd:?}"
        );
    }

    let q_final = s_a.joint_q.host().unwrap();
    let qd_final = s_a.joint_qd.host().unwrap();
    let final_norm = q_final.iter().map(|q| q * q).sum::<Real>().sqrt();
    let final_speed_norm = qd_final.iter().map(|qd| qd * qd).sum::<Real>().sqrt();
    eprintln!(
        "cable Y-junction: initial_norm={initial_norm:.5}, max_norm={max_norm:.5}, final_norm={final_norm:.5}, final_speed_norm={final_speed_norm:.5}"
    );

    assert!(
        max_norm <= initial_norm * 1.15,
        "branched cable junction gained displacement energy: max={max_norm}, initial={initial_norm}",
    );
    assert!(
        final_norm < initial_norm * 0.25,
        "branched cable junction did not damp toward rest: final={final_norm}, initial={initial_norm}",
    );
    assert!(
        final_speed_norm < 0.10,
        "branched cable junction retained too much generalized speed: {final_speed_norm}",
    );

    let body_q = s_a.body_q.host().unwrap();
    for (i, pose) in body_q.iter().enumerate() {
        assert!(
            pose.position.is_finite() && pose.position.length() < 5.0,
            "Y-junction body {i} escaped or became non-finite: {:?}",
            pose.position
        );
    }
}

#[derive(Copy, Clone, Debug)]
struct ReducedCableTwist {
    theta: Real,
    omega: Real,
    inertia: Real,
    stiffness: Real,
    damping: Real,
}

impl ReducedCableTwist {
    fn energy(&self) -> Real {
        0.5 * self.stiffness * self.theta * self.theta
            + 0.5 * self.inertia * self.omega * self.omega
    }

    fn step(&mut self, dt: Real) {
        let torque = -self.stiffness * self.theta - self.damping * self.omega;
        self.omega += (torque / self.inertia) * dt;
        self.theta += self.omega * dt;
    }
}

fn run_reduced_cable_twist(stiffness: Real) -> ReducedCableTwist {
    let mut twist = ReducedCableTwist {
        theta: 0.35,
        omega: 0.0,
        inertia: 0.04,
        stiffness,
        damping: 0.35,
    };
    let dt = 0.0005;
    let mut previous_energy = twist.energy();
    for _ in 0..3000 {
        twist.step(dt);
        assert!(twist.theta.is_finite() && twist.omega.is_finite());
        let energy = twist.energy();
        assert!(
            energy <= previous_energy + 1.0e-8,
            "reduced cable twist energy increased: prev={previous_energy}, now={energy}"
        );
        previous_energy = energy;
    }
    twist
}

#[test]
fn reduced_cable_twist_stiffness_controls_torsional_return() {
    let soft = run_reduced_cable_twist(0.25);
    let stiff = run_reduced_cable_twist(4.0);
    eprintln!(
        "reduced cable twist: soft theta={:.5}, omega={:.5}; stiff theta={:.5}, omega={:.5}",
        soft.theta, soft.omega, stiff.theta, stiff.omega
    );

    assert!(
        stiff.theta.abs() < soft.theta.abs() * 0.25,
        "higher twist stiffness should reduce residual twist: soft={}, stiff={}",
        soft.theta,
        stiff.theta
    );
    assert!(
        stiff.omega.abs() < 0.15,
        "stiff torsional mode should be damped, omega={}",
        stiff.omega
    );
}
