use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, JointHandle, RigidScene};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

const DT: f64 = 0.01;
const GAIN: f64 = 1e6;
const COMMON_INERTIA: f64 = 1.999;
const INITIAL_RATE: f64 = 0.2;

struct Case {
    body: BodyHandle,
    joint: JointHandle,
    kind: JointType,
    frame: Quat,
    cap: f64,
}

fn fixture(kinds: &[JointType]) -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    for &kind in kinds {
        for cap in [0., 20., f64::INFINITY] {
            for frame in [
                Quat::IDENTITY,
                Quat::from_rotation_z(0.7) * Quat::from_rotation_y(-0.4),
            ] {
                let body = scene
                    .insert_body(BodyDesc {
                        params: BodyParams::new().with_pose(Transform::from_rotation(frame)),
                        additional_mass: Some(MassProperties {
                            mass: 1.,
                            local_com: Vec3::ZERO,
                            inertia: Mat3::from_cols(
                                Vec3::new(1., 0.999, 0.),
                                Vec3::new(0.999, 1., 0.),
                                Vec3::Z * 2.,
                            ),
                        }),
                        ..Default::default()
                    })
                    .unwrap();
                let joint = scene
                    .insert_joint(JointDesc {
                        parent: None,
                        child: body,
                        kind,
                        axis: Vec3::X,
                        frame_parent: Transform::from_rotation(frame),
                        frame_child: Transform::IDENTITY,
                    })
                    .unwrap();
                scene
                    .set_body_velocity(
                        body,
                        SpatialVector::new(
                            Vec3::ZERO,
                            frame * Vec3::new(INITIAL_RATE, INITIAL_RATE, 0.),
                        ),
                    )
                    .unwrap();
                for axis in [JointAxis::AngX, JointAxis::AngY] {
                    scene
                        .set_joint_motor(
                            joint,
                            axis,
                            JointMotor {
                                mode: JointTargetMode::Velocity,
                                target_velocity: 1.,
                                damping: GAIN,
                                max_force: cap,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                }
                cases.push(Case {
                    body,
                    joint,
                    kind,
                    frame,
                    cap,
                });
            }
        }
    }
    (scene, cases)
}

fn verify(world: &mut RigidWorld, cases: &[Case], tolerance: f64) {
    world.solver.substeps = 1;
    world.step(DT).unwrap();
    assert!(world.quarantined_bodies().is_empty());
    assert!(world.joint_break_events().is_empty());
    let uncapped = GAIN * (1. - INITIAL_RATE) / (1. + GAIN * DT / COMMON_INERTIA);
    for case in cases {
        let torque = uncapped.min(case.cap);
        let rate = INITIAL_RATE + DT * torque / COMMON_INERTIA;
        let expected = case.frame * Vec3::new(rate, rate, 0.);
        let actual = world.scene.body_velocity(case.body).unwrap();
        let pose = world.scene.body_pose(case.body).unwrap();
        let error = (actual.angular - expected).length();
        println!(
            "coupled_velocity cap={} expected={rate:.9e} error={error:.9e} actual={actual:?}",
            case.cap
        );
        assert!(error.is_finite() && error <= tolerance);
        assert!(actual.linear.length() <= tolerance);
        assert!(pose.position.length() <= tolerance && pose.rotation.is_finite());
    }
}

#[test]
fn cpu_strongly_coupled_velocity_drives_match_bounded_implicit_oracle() {
    let (scene, cases) = fixture(&[JointType::Ball]);
    verify(&mut RigidWorld::from_scene(scene).unwrap(), &cases, 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_strongly_coupled_velocity_drives_match_bounded_implicit_oracle() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let (scene, cases) = fixture(&[JointType::Ball]);
    verify(
        &mut RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &cases,
        1e-3,
    );
}

fn verify_sustained(world: &mut RigidWorld, cases: &[Case], tolerance: f64) {
    world.solver.substeps = 1;
    world.solver.stable_motor_feedback = true;
    let mut rates = vec![INITIAL_RATE; cases.len()];
    let mut angles = vec![0.0; cases.len()];
    let mut target = 1.0;
    let mut enabled = true;
    let mut peak_rate: f64 = 0.0;
    let mut peak_rotation: f64 = 0.0;
    #[cfg(feature = "gpu")]
    let compilations = world.gpu_diagnostics().map(|d| d.model_compilations);
    for frame in 0..100 {
        if [50, 75, 90].contains(&frame) {
            target = if frame == 90 { 0.7 } else { -0.5 };
            enabled = frame != 75;
            let stamp = world.scene.model_stamp();
            for case in cases {
                for axis in [JointAxis::AngX, JointAxis::AngY] {
                    world
                        .scene
                        .set_joint_motor(
                            case.joint,
                            axis,
                            JointMotor {
                                mode: if enabled {
                                    JointTargetMode::Velocity
                                } else {
                                    JointTargetMode::None
                                },
                                target_velocity: target,
                                damping: GAIN,
                                max_force: case.cap,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                }
            }
            assert_eq!(world.scene.model_stamp(), stamp);
        }
        for ((case, rate), angle) in cases.iter().zip(&mut rates).zip(&mut angles) {
            let torque = if enabled {
                (GAIN * (target - *rate) / (1.0 + GAIN * DT / COMMON_INERTIA))
                    .clamp(-case.cap, case.cap)
            } else {
                0.0
            };
            *rate += DT * torque / COMMON_INERTIA;
            // Normalized Euler rotation about the common inertia eigenvector.
            *angle += 2.0 * (*rate * std::f64::consts::SQRT_2 * DT / 2.0).atan();
        }
        world.step(DT).unwrap();
        assert!(world.quarantined_bodies().is_empty());
        assert!(world.joint_break_events().is_empty());
        assert_eq!(world.completed_steps(), frame + 1);
        #[cfg(feature = "gpu")]
        assert_eq!(
            world.gpu_diagnostics().map(|d| d.model_compilations),
            compilations
        );
        for ((case, &rate), &angle) in cases.iter().zip(&rates).zip(&angles) {
            let actual = world.scene.body_velocity(case.body).unwrap();
            let pose = world.scene.body_pose(case.body).unwrap();
            let expected_rate = case.frame * Vec3::new(rate, rate, 0.0);
            let expected_rotation =
                case.frame * Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), angle);
            let error_rate = (actual.angular - expected_rate).length();
            let relative = pose.rotation.normalize() * expected_rotation.conjugate();
            let error_rotation = 2.0
                * Vec3::new(relative.x, relative.y, relative.z)
                    .length()
                    .atan2(relative.w.abs());
            peak_rate = peak_rate.max(error_rate);
            peak_rotation = peak_rotation.max(error_rotation);
            assert!(actual.linear.is_finite() && actual.angular.is_finite());
            assert!(pose.position.is_finite() && pose.rotation.is_finite());
            assert!(
                error_rate <= tolerance && error_rotation <= tolerance,
                "frame={frame} kind={:?} cap={} rate_error={error_rate:e} rotation_error={error_rotation:e} actual={actual:?}",
                case.kind,
                case.cap
            );
            assert!(actual.linear.length() <= tolerance && pose.position.length() <= tolerance);
        }
        if (frame + 1).is_multiple_of(25) {
            println!(
                "coupled_velocity_sustained frames={} cases={} target={target} enabled={enabled} peak_rate={peak_rate:e} peak_rotation={peak_rotation:e}",
                frame + 1,
                cases.len()
            );
        }
    }
}

#[test]
fn cpu_coupled_velocity_drives_preserve_one_second_motion_and_control_edits() {
    let (scene, cases) = fixture(&[JointType::Ball, JointType::D6, JointType::Free]);
    verify_sustained(&mut RigidWorld::from_scene(scene).unwrap(), &cases, 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_coupled_velocity_drives_preserve_one_second_motion_and_control_edits() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let (scene, cases) = fixture(&[JointType::Ball, JointType::D6, JointType::Free]);
    println!(
        "coupled_velocity_sustained_prepare_begin cases={}",
        cases.len()
    );
    let preparation = std::time::Instant::now();
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    println!(
        "coupled_velocity_sustained_prepare_complete seconds={:.6}",
        preparation.elapsed().as_secs_f64()
    );
    verify_sustained(&mut world, &cases, 1e-3);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_raw_coupled_roots_work_without_coordinate_compensation() {
    use molla_solvers::Solver;
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} coordinate_compensation=false");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut scene, cases) = fixture(&[JointType::Ball, JointType::D6, JointType::Free]);
    scene.upload_model_to_device(device, queue).unwrap();
    scene.state_mut().upload_to_device(device, queue).unwrap();
    scene.control_mut().upload_to_device(device, queue).unwrap();
    let mut next = scene.model().state().unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut solver = molla_solvers::FeatherstoneSolver::new_gpu(&engine, scene.model())
        .unwrap()
        .with_rigorous_d_solve(true)
        .with_stable_motor_feedback(true);
    solver
        .step(
            scene.model(),
            scene.state(),
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            DT,
            &mut next,
        )
        .unwrap();
    next.body_q.download(device, queue).unwrap();
    next.body_qd.download(device, queue).unwrap();
    let mut peak_rate: f64 = 0.0;
    for case in &cases {
        let body = scene.body_index(case.body).unwrap().index();
        let uncapped = GAIN * (1.0 - INITIAL_RATE) / (1.0 + GAIN * DT / COMMON_INERTIA);
        let rate = INITIAL_RATE + DT * uncapped.min(case.cap) / COMMON_INERTIA;
        let actual = next.body_qd.host().unwrap()[body];
        let pose = next.body_q.host().unwrap()[body];
        let expected = case.frame * Vec3::new(rate, rate, 0.0);
        let error = (actual.angular - expected).length();
        peak_rate = peak_rate.max(error);
        assert!(actual.linear.is_finite() && actual.angular.is_finite());
        assert!(pose.position.is_finite() && pose.rotation.is_finite());
        assert!(
            error <= 1e-3,
            "kind={:?} cap={} error={error:e}",
            case.kind,
            case.cap
        );
        assert!(actual.linear.length() <= 1e-3 && pose.position.length() <= 1e-3);
    }
    println!(
        "raw_coupled_roots cases={} steps=1 peak_rate={peak_rate:e}",
        cases.len()
    );
}
