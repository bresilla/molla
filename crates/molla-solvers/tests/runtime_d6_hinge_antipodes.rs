use molla_math::{Mat3, Quat, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis};
use molla_sim::runtime::{BodyDesc, BodyHandle, BodyKind, JointDesc, JointHandle, RigidScene};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;
use molla_solvers::{FeatherstoneSolver, Solver};

#[cfg(feature = "gpu")]
#[path = "common/gpu_readback.rs"]
mod gpu_readback;

const DT: f64 = 0.01;

struct Case {
    parent: BodyHandle,
    child: BodyHandle,
    joint: JointHandle,
    body: BodyHandle,
    parent_frame: Transform,
    child_frame: Transform,
    left: Quat,
    right: Quat,
    free: usize,
    rate: Vec3,
    torque: Vec3,
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    let tensor = Mat3::from_cols(
        Vec3::new(3., 0.2, 0.3),
        Vec3::new(0.2, 4., 0.4),
        Vec3::new(0.3, 0.4, 5.),
    );
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let controls = [JointAxis::AngX, JointAxis::AngY, JointAxis::AngZ];
    let authored = Vec3::new(0.3, -0.2, 0.4);
    for frame in [
        Quat::IDENTITY,
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.3),
    ] {
        for reversed in [false, true] {
            for free in 0..3 {
                let (left, right) = match free {
                    0 => (
                        Quat::IDENTITY,
                        Quat::from_rotation_y(authored.y) * Quat::from_rotation_z(authored.z),
                    ),
                    1 => (
                        Quat::from_rotation_x(authored.x),
                        Quat::from_rotation_z(authored.z),
                    ),
                    _ => (
                        Quat::from_rotation_x(authored.x) * Quat::from_rotation_y(authored.y),
                        Quat::IDENTITY,
                    ),
                };
                for twist in [-0.4, 0., 0.4] {
                    for sign in [-1., 1.] {
                        let canonical =
                            Quat::from_axis_angle(axes[(free + 2) % 3], std::f64::consts::PI)
                                * Quat::from_axis_angle(axes[free], twist);
                        let relative = (left * canonical * right) * sign;
                        let parent_rotation = if reversed {
                            frame * relative.conjugate()
                        } else {
                            frame
                        };
                        let child_rotation = if reversed { frame } else { frame * relative };
                        let anchor = Vec3::new(2. * cases.len() as f64, 0., 0.);
                        let parent_frame = Transform::from_rotation(Quat::from_euler(
                            glam::EulerRot::XYZ,
                            0.2,
                            -0.3,
                            0.4,
                        ));
                        let child_frame = Transform::from_rotation(Quat::from_euler(
                            glam::EulerRot::XYZ,
                            -0.4,
                            0.3,
                            -0.2,
                        ));
                        let parent_pose = Transform::new(
                            anchor,
                            parent_rotation * parent_frame.rotation.conjugate(),
                        );
                        let child_pose = Transform::new(
                            anchor,
                            child_rotation * child_frame.rotation.conjugate(),
                        );
                        let properties = MassProperties {
                            mass: 2.,
                            local_com: Vec3::ZERO,
                            inertia: tensor,
                        };
                        let parent = scene
                            .insert_body(BodyDesc {
                                params: BodyParams::new().with_pose(parent_pose),
                                kind: if reversed {
                                    BodyKind::Dynamic
                                } else {
                                    BodyKind::Fixed
                                },
                                additional_mass: Some(properties),
                                ..Default::default()
                            })
                            .unwrap();
                        let child = scene
                            .insert_body(BodyDesc {
                                params: BodyParams::new().with_pose(child_pose),
                                kind: if reversed {
                                    BodyKind::Fixed
                                } else {
                                    BodyKind::Dynamic
                                },
                                additional_mass: Some(properties),
                                ..Default::default()
                            })
                            .unwrap();
                        let joint = scene
                            .insert_joint(JointDesc {
                                parent: Some(parent),
                                child,
                                kind: JointType::D6,
                                axis: Vec3::X,
                                frame_parent: parent_frame,
                                frame_child: child_frame,
                            })
                            .unwrap();
                        scene.set_joint_locked_axes(joint, JointAxes::LIN).unwrap();
                        for axis in 0..3 {
                            let limits = if axis == free {
                                [-0.2, 0.2]
                            } else {
                                [authored[axis]; 2]
                            };
                            scene
                                .set_joint_limits(joint, controls[axis], Some(limits))
                                .unwrap();
                        }
                        let body = if reversed { parent } else { child };
                        let pose = if reversed { parent_pose } else { child_pose };
                        let basis = Mat3::from_quat(pose.rotation);
                        let inertia = basis * tensor * basis.transpose();
                        let direction = if reversed { -1. } else { 1. };
                        let tangent = parent_rotation * left * axes[free] * direction;
                        let swing = parent_rotation
                            * left
                            * axes[(free + 2) % 3]
                            * (-std::f64::consts::PI / DT * direction);
                        let free_rate = (-tangent.dot(inertia * swing)
                            / tangent.dot(inertia * tangent))
                        .clamp((twist - 0.2) / DT, (twist + 0.2) / DT);
                        let rate = swing + tangent * free_rate;
                        cases.push(Case {
                            parent,
                            child,
                            joint,
                            body,
                            parent_frame,
                            child_frame,
                            left,
                            right,
                            free,
                            rate,
                            torque: inertia * rate / DT * direction,
                        });
                    }
                }
            }
        }
    }
    (scene, cases)
}

fn verify(mut world: RigidWorld, cases: &[Case], tolerance: f64) {
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    for step in 0..32 {
        world.step(DT).unwrap();
        assert!(world.quarantined_bodies().is_empty());
        assert!(world.joint_break_events().is_empty());
        for (i, case) in cases.iter().enumerate() {
            let pose = world.scene.body_pose(case.body).unwrap();
            let rate = world.scene.body_velocity(case.body).unwrap();
            let reaction = world.joint_reaction(case.joint).unwrap();
            assert!(pose.rotation.is_finite() && pose.position.is_finite());
            assert!(rate.angular.is_finite() && rate.linear.is_finite());
            assert!(reaction.child.angular.is_finite() && reaction.child.linear.is_finite());
            assert!((reaction.child.linear + reaction.parent.linear).length() <= tolerance);
            assert!((reaction.child.angular + reaction.parent.angular).length() <= tolerance);
            assert!(reaction.child.linear.length() <= tolerance);
            if step == 0 {
                let error = (rate.angular - case.rate).length();
                let torque_error = (reaction.child.angular - case.torque).length();
                println!(
                    "hinge_antipode case={i} rate_error={error:.9e} torque_error={torque_error:.9e}"
                );
                assert!(
                    error <= tolerance,
                    "case={i}: {rate:?} expected={:?}",
                    case.rate
                );
                assert!(
                    torque_error <= 0.01 * case.torque.length().max(1.),
                    "case={i}: {reaction:?} expected={:?}",
                    case.torque
                );
            }
        }
    }
    for (i, case) in cases.iter().enumerate() {
        let parent =
            world.scene.body_pose(case.parent).unwrap().rotation * case.parent_frame.rotation;
        let child = world.scene.body_pose(case.child).unwrap().rotation * case.child_frame.rotation;
        let relative =
            (case.left.conjugate() * parent.conjugate() * child * case.right.conjugate())
                .normalize();
        let primary = [Vec3::X, Vec3::Y, Vec3::Z][case.free];
        let swing = (relative * primary)
            .cross(primary)
            .length()
            .atan2((relative * primary).dot(primary));
        let angle = 2. * relative.xyz()[case.free].atan2(relative.w);
        let angle =
            (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        let gap = (-0.2 - angle).max(angle - 0.2).max(0.);
        println!("hinge_antipode_recovered case={i} swing={swing:.9e} gap={gap:.9e}");
        assert!(
            swing <= 1e-3 && gap <= 1e-3,
            "case={i}: swing={swing} angle={angle}"
        );
        assert!(
            (world.scene.body_pose(case.parent).unwrap().position
                - world.scene.body_pose(case.child).unwrap().position)
                .length()
                <= 1e-3
        );
    }
}

#[test]
fn cpu_finite_hinges_recover_antipodes_with_mass_coupled_reactions() {
    let (scene, cases) = fixture();
    verify(RigidWorld::from_scene(scene).unwrap(), &cases, 1e-6);
}

fn verify_standalone(scene: &RigidScene, state: &molla_sim::State, cases: &[Case], tolerance: f64) {
    for (i, case) in cases.iter().enumerate() {
        let index = scene.body_index(case.body).unwrap().index();
        let pose = state.body_q.host().unwrap()[index];
        let rate = state.body_qd.host().unwrap()[index];
        let error = (rate.angular - case.rate).length();
        println!("hinge_antipode_standalone case={i} rate_error={error:.9e}");
        assert!(pose.rotation.is_finite() && pose.position.is_finite());
        assert!(rate.angular.is_finite() && rate.linear.is_finite());
        assert!(
            error <= tolerance,
            "case={i}: {rate:?} expected={:?}",
            case.rate
        );
        assert!(rate.linear.length() <= tolerance);
        assert!(
            (pose.position - scene.body_pose(case.body).unwrap().position).length() <= tolerance
        );
    }
}

#[test]
fn cpu_standalone_finite_hinge_antipodes_use_the_mass_response() {
    let (scene, cases) = fixture();
    let input = scene.state().clone_host().unwrap();
    let mut output = scene.model().state().unwrap();
    FeatherstoneSolver::new_cpu()
        .step(
            scene.model(),
            &input,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            DT,
            &mut output,
        )
        .unwrap();
    verify_standalone(&scene, &output, &cases, 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_finite_hinges_recover_antipodes_with_mass_coupled_reactions() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let (scene, cases) = fixture();
    verify(
        RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &cases,
        1e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_standalone_finite_hinge_antipodes_use_the_mass_response() {
    standalone_gpu(false);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_standalone_compensated_hinge_antipodes_use_the_mass_response() {
    standalone_gpu(true);
}

#[cfg(feature = "gpu")]
fn standalone_gpu(compensated: bool) {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let (mut scene, cases) = fixture();
    scene.upload_model_to_device(&device, &queue).unwrap();
    scene.upload_controls_to_device(&device, &queue).unwrap();
    let mut input = scene.state().clone_host().unwrap();
    let mut output = scene.model().state().unwrap();
    input.upload_to_device(&device, &queue).unwrap();
    output.upload_to_device(&device, &queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, scene.model()).unwrap();
    if compensated {
        solver
            .prepare_gpu_coordinate_compensation(&[&input, &output])
            .unwrap();
    }
    solver
        .step(
            scene.model(),
            &input,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            DT,
            &mut output,
        )
        .unwrap();
    output.body_q.download(&device, &queue).unwrap();
    output.body_qd.download(&device, &queue).unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    verify_standalone(&scene, &output, &cases, 1e-3);
}
