//! Tier-3 CPU articulation stress tests.
//!
//! GPU parity lives in the ignored Featherstone GPU lanes. This file
//! keeps a procedural many-articulation batch in the default CPU lane.

use molla_core::BodyId;
use molla_geometry::CollisionPipeline;
use molla_math::{Mat3, Quat, Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Control, JointTargetMode, Model, ModelBuilder};
use molla_solvers::{FeatherstoneSolver, Solver};

const CHAINS: usize = 6;
const LINKS: usize = 3;

fn build_chain_batch() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    for chain in 0..CHAINS {
        b.begin_articulation();
        let base_x = (chain as Real - (CHAINS as Real - 1.0) * 0.5) * 0.8;
        let mut parent = BodyId::NONE;
        let mut next_pos = Vec3::new(base_x, 2.0, 0.0);
        for link in 0..LINKS {
            let body = b.add_body(
                BodyParams::new()
                    .with_spatial_inertia(SpatialInertia {
                        mass: 1.0,
                        com: Vec3::ZERO,
                        inertia: Mat3::IDENTITY * 0.05,
                    })
                    .with_position(next_pos),
            );
            let x_p = if link == 0 {
                // Rest pose hangs down from the world frame.
                Transform::new(
                    Vec3::new(base_x, 2.25, 0.0),
                    Quat::from_axis_angle(Vec3::X, std::f64::consts::PI),
                )
            } else {
                Transform::from_translation(Vec3::new(0.0, -0.25, 0.0))
            };
            let x_c = Transform::from_translation(Vec3::new(0.0, 0.25, 0.0));
            let axis = if link % 2 == 0 { Vec3::Z } else { Vec3::X };
            b.add_joint_revolute(parent, body, x_p, x_c, axis).unwrap();
            parent = body;
            next_pos -= Vec3::new(0.0, 0.5, 0.0);
        }
    }
    let mut model = b.build().unwrap();
    // This is a controller stress test, not a falling-under-gravity test.
    for g in model.gravity.host_mut().unwrap() {
        *g = Vec3::ZERO;
    }
    model
}

fn configure_targets(control: &mut Control) -> Vec<Real> {
    let n = control.joint_target_pos.host().unwrap().len();
    let mut targets = Vec::with_capacity(n);
    for i in 0..n {
        let target = if i % 2 == 0 { 0.35 } else { -0.25 };
        targets.push(target);
        control.joint_target_mode.host_mut().unwrap()[i] = JointTargetMode::Position as u32;
        control.joint_target_pos.host_mut().unwrap()[i] = target;
        control.joint_target_kp.host_mut().unwrap()[i] = 160.0;
        control.joint_target_kd.host_mut().unwrap()[i] = 16.0;
    }
    targets
}

fn build_quadruped_like_tree() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.begin_articulation();

    let root_pos = Vec3::new(0.0, 1.2, 0.0);
    let root = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 5.0,
                com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 0.4,
            })
            .with_position(root_pos),
    );
    b.add_shape_box(root, Vec3::new(0.40, 0.08, 0.25), Transform::IDENTITY)
        .unwrap();
    b.add_joint_fixed(
        BodyId::NONE,
        root,
        Transform::from_translation(root_pos),
        Transform::IDENTITY,
    )
    .unwrap();

    let hip_offsets = [
        Vec3::new(-0.32, 0.0, -0.22),
        Vec3::new(0.32, 0.0, -0.22),
        Vec3::new(-0.32, 0.0, 0.22),
        Vec3::new(0.32, 0.0, 0.22),
    ];
    for (leg, hip) in hip_offsets.iter().enumerate() {
        let side_sign = if hip.x < 0.0 { -1.0 } else { 1.0 };
        let front_sign = if hip.z < 0.0 { -1.0 } else { 1.0 };
        let upper_pos = root_pos + *hip + Vec3::new(0.0, -0.28, 0.0);
        let upper = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 0.7,
                    com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY * 0.04,
                })
                .with_position(upper_pos),
        );
        b.add_shape_capsule(upper, 0.24, 0.035, Transform::IDENTITY)
            .unwrap();
        b.add_joint_revolute(
            root,
            upper,
            Transform::from_translation(*hip),
            Transform::from_translation(Vec3::new(0.0, 0.28, 0.0)),
            Vec3::Z * side_sign,
        )
        .unwrap();

        let lower_pos = root_pos + *hip + Vec3::new(0.0, -0.82, 0.0);
        let lower = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 0.5,
                    com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY * 0.03,
                })
                .with_position(lower_pos),
        );
        b.add_shape_capsule(lower, 0.24, 0.03, Transform::IDENTITY)
            .unwrap();
        b.add_joint_revolute(
            upper,
            lower,
            Transform::from_translation(Vec3::new(0.0, -0.28, 0.0)),
            Transform::from_translation(Vec3::new(0.0, 0.28, 0.0)),
            Vec3::X * front_sign,
        )
        .unwrap();
        let _ = leg;
    }

    let mut model = b.build().unwrap();
    // Controller-tree stress test; remove gravity so target convergence is the
    // only acceptance signal.
    for g in model.gravity.host_mut().unwrap() {
        *g = Vec3::ZERO;
    }
    model
}

fn build_contact_quadruped_like_tree() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.05);
    b.set_angular_damping(0.08);
    b.begin_articulation();

    let root_pos = Vec3::new(0.0, 1.15, 0.0);
    let root = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 5.0,
                com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 0.4,
            })
            .with_position(root_pos),
    );
    b.add_shape_box(root, Vec3::new(0.40, 0.08, 0.25), Transform::IDENTITY)
        .unwrap();
    b.add_joint_fixed(
        BodyId::NONE,
        root,
        Transform::from_translation(root_pos),
        Transform::IDENTITY,
    )
    .unwrap();

    let hip_offsets = [
        Vec3::new(-0.32, 0.0, -0.22),
        Vec3::new(0.32, 0.0, -0.22),
        Vec3::new(-0.32, 0.0, 0.22),
        Vec3::new(0.32, 0.0, 0.22),
    ];
    for hip in hip_offsets {
        let side_sign = if hip.x < 0.0 { -1.0 } else { 1.0 };
        let front_sign = if hip.z < 0.0 { -1.0 } else { 1.0 };
        let upper_pos = root_pos + hip + Vec3::new(0.0, -0.28, 0.0);
        let upper = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 0.8,
                    com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY * 0.045,
                })
                .with_position(upper_pos),
        );
        b.add_shape_capsule(upper, 0.24, 0.035, Transform::IDENTITY)
            .unwrap();
        b.add_joint_revolute(
            root,
            upper,
            Transform::from_translation(hip),
            Transform::from_translation(Vec3::new(0.0, 0.28, 0.0)),
            Vec3::Z * side_sign,
        )
        .unwrap();

        let lower_pos = root_pos + hip + Vec3::new(0.0, -0.82, 0.0);
        let lower = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 0.6,
                    com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY * 0.035,
                })
                .with_position(lower_pos),
        );
        b.add_shape_capsule(lower, 0.24, 0.035, Transform::IDENTITY)
            .unwrap();
        b.add_joint_revolute(
            upper,
            lower,
            Transform::from_translation(Vec3::new(0.0, -0.28, 0.0)),
            Transform::from_translation(Vec3::new(0.0, 0.28, 0.0)),
            Vec3::X * front_sign,
        )
        .unwrap();
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let mut model = b.build().unwrap();
    model.default_friction = 0.7;
    model.default_restitution = 0.0;
    model.contact_margin = 0.02;
    model
}

fn build_humanoid_like_tree() -> Model {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.06);
    b.begin_articulation();

    let root_pos = Vec3::new(0.0, 1.35, 0.0);
    let torso = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 6.0,
                com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 0.45,
            })
            .with_position(root_pos),
    );
    b.add_shape_box(torso, Vec3::new(0.22, 0.35, 0.12), Transform::IDENTITY)
        .unwrap();
    b.add_joint_fixed(
        BodyId::NONE,
        torso,
        Transform::from_translation(root_pos),
        Transform::IDENTITY,
    )
    .unwrap();

    let head_pos = root_pos + Vec3::new(0.0, 0.52, 0.0);
    let head = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 0.06,
            })
            .with_position(head_pos),
    );
    b.add_shape_sphere(head, 0.12, Transform::IDENTITY).unwrap();
    b.add_joint_revolute(
        torso,
        head,
        Transform::from_translation(Vec3::new(0.0, 0.38, 0.0)),
        Transform::from_translation(Vec3::new(0.0, -0.12, 0.0)),
        Vec3::Z,
    )
    .unwrap();

    for side in [-1.0, 1.0] {
        let shoulder = Vec3::new(0.26 * side, 0.25, 0.0);
        let upper_arm_pos = root_pos + shoulder + Vec3::new(0.22 * side, -0.08, 0.0);
        let upper_arm = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 0.8,
                    com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY * 0.035,
                })
                .with_position(upper_arm_pos),
        );
        b.add_shape_capsule(upper_arm, 0.20, 0.035, Transform::IDENTITY)
            .unwrap();
        b.add_joint_revolute(
            torso,
            upper_arm,
            Transform::from_translation(shoulder),
            Transform::from_translation(Vec3::new(-0.20 * side, 0.0, 0.0)),
            Vec3::Z * side,
        )
        .unwrap();

        let forearm_pos = root_pos + shoulder + Vec3::new(0.58 * side, -0.10, 0.0);
        let forearm = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 0.55,
                    com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY * 0.025,
                })
                .with_position(forearm_pos),
        );
        b.add_shape_capsule(forearm, 0.18, 0.03, Transform::IDENTITY)
            .unwrap();
        b.add_joint_revolute(
            upper_arm,
            forearm,
            Transform::from_translation(Vec3::new(0.20 * side, 0.0, 0.0)),
            Transform::from_translation(Vec3::new(-0.18 * side, 0.0, 0.0)),
            Vec3::Y * side,
        )
        .unwrap();

        let hip = Vec3::new(0.12 * side, -0.34, 0.0);
        let thigh_pos = root_pos + hip + Vec3::new(0.04 * side, -0.28, 0.0);
        let thigh = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 1.1,
                    com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY * 0.05,
                })
                .with_position(thigh_pos),
        );
        b.add_shape_capsule(thigh, 0.26, 0.04, Transform::IDENTITY)
            .unwrap();
        b.add_joint_revolute(
            torso,
            thigh,
            Transform::from_translation(hip),
            Transform::from_translation(Vec3::new(0.0, 0.26, 0.0)),
            Vec3::Z * side,
        )
        .unwrap();

        let shin_pos = root_pos + hip + Vec3::new(0.06 * side, -0.78, 0.0);
        let shin = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 0.9,
                    com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY * 0.04,
                })
                .with_position(shin_pos),
        );
        b.add_shape_capsule(shin, 0.24, 0.035, Transform::IDENTITY)
            .unwrap();
        b.add_joint_revolute(
            thigh,
            shin,
            Transform::from_translation(Vec3::new(0.0, -0.26, 0.0)),
            Transform::from_translation(Vec3::new(0.0, 0.24, 0.0)),
            Vec3::X,
        )
        .unwrap();
    }

    let mut model = b.build().unwrap();
    for g in model.gravity.host_mut().unwrap() {
        *g = Vec3::ZERO;
    }
    model
}

#[test]
fn many_procedural_chains_pd_drive_stays_finite_and_moves_to_targets() {
    let model = build_chain_batch();
    assert_eq!(model.joint_count, CHAINS * LINKS);

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    let targets = configure_targets(&mut control);
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu().with_damping(0.1);
    let initial_q = s_a.joint_q.host().unwrap().to_vec();
    let dt: Real = 0.005;
    for _ in 0..1000 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.joint_q.host().unwrap();
    let qd = s_a.joint_qd.host().unwrap();
    for i in 0..model.joint_count {
        assert!(q[i].is_finite(), "joint {i} q is non-finite: {}", q[i]);
        assert!(qd[i].is_finite(), "joint {i} qd is non-finite: {}", qd[i]);
        let initial_err = (initial_q[i] - targets[i]).abs();
        let final_err = (q[i] - targets[i]).abs();
        assert!(
            final_err < initial_err * 0.75,
            "joint {i} did not move enough toward target {}: initial_err={initial_err}, final_err={final_err}",
            targets[i]
        );
    }

    for (i, pose) in s_a.body_q.host().unwrap().iter().enumerate() {
        assert!(
            pose.position.is_finite(),
            "body {i} position is non-finite: {:?}",
            pose.position
        );
        assert!(
            pose.position.length() < 20.0,
            "body {i} escaped procedural batch: {:?}",
            pose.position
        );
    }
}

#[test]
fn procedural_quadruped_tree_pd_targets_stay_finite_and_converge() {
    let model = build_quadruped_like_tree();
    assert_eq!(model.body_count, 9, "root + 4 two-link legs");
    assert_eq!(model.joint_count, 9, "fixed root + 8 leg hinges");
    assert_eq!(model.joint_dof_total, 8);

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    let targets = [
        0.25, -0.45, // front-left
        -0.25, -0.45, // front-right
        -0.20, 0.35, // rear-left
        0.20, 0.35, // rear-right
    ];
    for (i, target) in targets.iter().enumerate() {
        control.joint_target_mode.host_mut().unwrap()[i] = JointTargetMode::Position as u32;
        control.joint_target_pos.host_mut().unwrap()[i] = *target;
        control.joint_target_kp.host_mut().unwrap()[i] = 180.0;
        control.joint_target_kd.host_mut().unwrap()[i] = 18.0;
    }
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu().with_damping(0.1);
    let initial_q = s_a.joint_q.host().unwrap().to_vec();
    let dt: Real = 0.004;
    for _ in 0..900 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.joint_q.host().unwrap();
    let qd = s_a.joint_qd.host().unwrap();
    for (i, target) in targets.iter().enumerate() {
        assert!(q[i].is_finite(), "quadruped dof {i} q is non-finite");
        assert!(qd[i].is_finite(), "quadruped dof {i} qd is non-finite");
        let initial_err = (initial_q[i] - *target).abs();
        let final_err = (q[i] - *target).abs();
        assert!(
            final_err < initial_err * 0.70,
            "quadruped dof {i} did not converge toward target {target}: initial_err={initial_err}, final_err={final_err}, q={}",
            q[i]
        );
    }

    let body_q = s_a.body_q.host().unwrap();
    for (i, pose) in body_q.iter().enumerate() {
        assert!(
            pose.position.is_finite(),
            "quadruped body {i} non-finite: {:?}",
            pose.position
        );
        assert!(
            pose.position.distance(Vec3::new(0.0, 1.0, 0.0)) < 5.0,
            "quadruped body {i} escaped: {:?}",
            pose.position
        );
    }

    // The four lower legs should not collapse into one pose; this is a cheap
    // branch-topology sanity check that the root has four independent child
    // branches rather than a single serial chain.
    let lower_indices = [2usize, 4, 6, 8];
    let mut min_sep = Real::INFINITY;
    for a in 0..lower_indices.len() {
        for b in a + 1..lower_indices.len() {
            let pa = body_q[lower_indices[a]].position;
            let pb = body_q[lower_indices[b]].position;
            min_sep = min_sep.min(pa.distance(pb));
        }
    }
    assert!(
        min_sep > 0.25,
        "quadruped lower-leg branches collapsed together; min_sep={min_sep}"
    );
}

#[test]
fn procedural_quadruped_tree_survives_floor_contacts_while_controlled() {
    let model = build_contact_quadruped_like_tree();
    assert_eq!(model.body_count, 9, "root + 4 two-link legs");
    assert_eq!(model.joint_dof_total, 8);

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    let targets = [
        0.18, -0.55, // front-left
        -0.18, -0.55, // front-right
        -0.16, 0.45, // rear-left
        0.16, 0.45, // rear-right
    ];
    for (i, target) in targets.iter().enumerate() {
        control.joint_target_mode.host_mut().unwrap()[i] = JointTargetMode::Position as u32;
        control.joint_target_pos.host_mut().unwrap()[i] = *target;
        control.joint_target_kp.host_mut().unwrap()[i] = 120.0;
        control.joint_target_kd.host_mut().unwrap()[i] = 16.0;
    }

    let mut contacts = Contacts::with_capacity(256);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = FeatherstoneSolver::new_cpu().with_damping(0.12);
    let initial_q = s_a.joint_q.host().unwrap().to_vec();
    let dt: Real = 0.003;
    let mut max_contacts = 0_u32;
    for _ in 0..800 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        max_contacts = max_contacts.max(contacts.count);
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    assert!(
        max_contacts > 0,
        "controlled quadruped did not exercise floor contacts"
    );
    let q = s_a.joint_q.host().unwrap();
    let qd = s_a.joint_qd.host().unwrap();
    for (i, target) in targets.iter().enumerate() {
        assert!(q[i].is_finite(), "contact quadruped dof {i} q non-finite");
        assert!(qd[i].is_finite(), "contact quadruped dof {i} qd non-finite");
        let initial_err = (initial_q[i] - *target).abs();
        let final_err = (q[i] - *target).abs();
        assert!(
            final_err < initial_err * 0.95,
            "contact quadruped dof {i} moved away from target {target}: initial_err={initial_err}, final_err={final_err}, q={}",
            q[i]
        );
    }

    let body_q = s_a.body_q.host().unwrap();
    let body_qd = s_a.body_qd.host().unwrap();
    let mut min_body_y = Real::INFINITY;
    for (i, (pose, vel)) in body_q.iter().zip(body_qd.iter()).enumerate() {
        min_body_y = min_body_y.min(pose.position.y);
        assert!(
            pose.position.is_finite() && vel.linear.is_finite() && vel.angular.is_finite(),
            "contact quadruped body {i} non-finite: pose={pose:?}, vel={vel:?}"
        );
        assert!(
            pose.position.distance(Vec3::new(0.0, 0.8, 0.0)) < 5.0,
            "contact quadruped body {i} escaped: {:?}",
            pose.position
        );
    }
    eprintln!(
        "contact quadruped: max_contacts={max_contacts}, min_body_y={min_body_y:.4}, q={:?}",
        q
    );
    assert!(
        min_body_y > -0.15,
        "floor contacts failed to keep articulated legs bounded above floor: min_body_y={min_body_y}"
    );
}

#[test]
fn procedural_humanoid_like_tree_pd_targets_stay_finite_and_branch_separated() {
    let model = build_humanoid_like_tree();
    assert_eq!(model.body_count, 10, "torso + head + two arms + two legs");
    assert_eq!(model.joint_dof_total, 9);

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let mut control = model.control();
    let targets = [
        0.12, // neck
        0.35, -0.28, // left shoulder/elbow
        -0.35, 0.28, // right shoulder/elbow
        -0.22, 0.40, // left hip/knee
        0.22, -0.40, // right hip/knee
    ];
    for (i, target) in targets.iter().enumerate() {
        control.joint_target_mode.host_mut().unwrap()[i] = JointTargetMode::Position as u32;
        control.joint_target_pos.host_mut().unwrap()[i] = *target;
        control.joint_target_kp.host_mut().unwrap()[i] = 150.0;
        control.joint_target_kd.host_mut().unwrap()[i] = 18.0;
    }

    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu().with_damping(0.12);
    let initial_q = s_a.joint_q.host().unwrap().to_vec();
    let dt: Real = 0.004;
    for _ in 0..900 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.joint_q.host().unwrap();
    let qd = s_a.joint_qd.host().unwrap();
    for (i, target) in targets.iter().enumerate() {
        assert!(q[i].is_finite(), "humanoid dof {i} q is non-finite");
        assert!(qd[i].is_finite(), "humanoid dof {i} qd is non-finite");
        let initial_err = (initial_q[i] - *target).abs();
        let final_err = (q[i] - *target).abs();
        assert!(
            final_err < initial_err * 0.75,
            "humanoid dof {i} did not converge toward target {target}: initial_err={initial_err}, final_err={final_err}, q={}",
            q[i]
        );
    }

    let body_q = s_a.body_q.host().unwrap();
    for (i, pose) in body_q.iter().enumerate() {
        assert!(
            pose.position.is_finite(),
            "humanoid body {i} non-finite: {:?}",
            pose.position
        );
        assert!(
            pose.position.distance(Vec3::new(0.0, 1.0, 0.0)) < 5.0,
            "humanoid body {i} escaped: {:?}",
            pose.position
        );
    }

    let left_hand = body_q[3].position;
    let right_hand = body_q[5].position;
    let left_foot = body_q[7].position;
    let right_foot = body_q[9].position;
    assert!(
        left_hand.distance(right_hand) > 0.55,
        "humanoid arm branches collapsed together: left={left_hand:?}, right={right_hand:?}"
    );
    assert!(
        left_foot.distance(right_foot) > 0.20,
        "humanoid leg branches collapsed together: left={left_foot:?}, right={right_foot:?}"
    );
}
