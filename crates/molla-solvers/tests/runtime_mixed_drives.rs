use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, JointDesc, JointHandle, RigidScene,
};
use molla_sim::{
    BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, JointTargetMode, JointType,
    MassProperties,
};
use molla_solvers::rigid_world::RigidWorld;

const DT: f64 = 1. / 240.;
const SUBSTEPS: usize = 2;
const FRAMES: usize = 240;

struct Case {
    body: BodyHandle,
    joint: JointHandle,
    origin: Vec3,
    rotation: Quat,
    mass: f64,
    model: MotorModel,
    cap: f64,
    q: [f64; 2],
    v: [f64; 2],
}

fn angular_motor(case: &Case, target: f64) -> JointMotor {
    JointMotor {
        mode: JointTargetMode::Velocity,
        model: case.model,
        target_velocity: target,
        damping: 3.,
        max_force: case.cap,
        ..Default::default()
    }
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(-Vec3::Z * 2.).unwrap();
    let mut cases = Vec::new();
    for mass in [1., 17.] {
        for yaw in [0., 0.7] {
            for model in [MotorModel::Force, MotorModel::Acceleration] {
                for cap in [0., 2. * mass, f64::INFINITY] {
                    let origin = Vec3::new(cases.len() as f64 * 8., 0., 0.49998);
                    let rotation = Quat::from_rotation_z(yaw);
                    let body = scene
                        .insert_body(BodyDesc {
                            params: BodyParams::new().with_pose(Transform::new(origin, rotation)),
                            additional_mass: Some(MassProperties {
                                mass,
                                local_com: Vec3::Z * 0.2,
                                inertia: Mat3::from_cols(
                                    Vec3::new(2., 0.1, 0.2),
                                    Vec3::new(0.1, 3., 0.3),
                                    Vec3::new(0.2, 0.3, 4.),
                                ) * mass,
                            }),
                            ..Default::default()
                        })
                        .unwrap();
                    let material = ContactMaterial {
                        stiffness: 1e5 * mass,
                        damping: 0.,
                        friction: Some(FrictionCoefficients {
                            mu_static: 0.125,
                            mu_dynamic: 0.1,
                        }),
                        ..Default::default()
                    };
                    let mut sphere = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
                    sphere.parent = Some(body);
                    sphere.mass = ColliderMass::Mass(0.);
                    sphere.contact_material = material;
                    scene.insert_collider(sphere).unwrap();
                    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
                        half_extents: Vec3::new(2., 2., 0.1),
                    });
                    ground.pose = Transform::from_translation(Vec3::new(origin.x, 0., -0.1));
                    ground.contact_material = material;
                    scene.insert_collider(ground).unwrap();
                    let joint = scene
                        .insert_joint(JointDesc {
                            parent: None,
                            child: body,
                            kind: JointType::D6,
                            axis: Vec3::X,
                            frame_parent: Transform::new(origin, rotation),
                            frame_child: Transform::IDENTITY,
                        })
                        .unwrap();
                    scene
                        .set_joint_locked_axes(
                            joint,
                            JointAxes(
                                JointAxes::ALL.0
                                    & !((1 << JointAxis::LinX as u32)
                                        | (1 << JointAxis::AngZ as u32)),
                            ),
                        )
                        .unwrap();
                    scene
                        .set_joint_motor(
                            joint,
                            JointAxis::LinX,
                            JointMotor {
                                mode: JointTargetMode::Effort,
                                target_position: mass * 0.6,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                    scene
                        .set_body_velocity(
                            body,
                            SpatialVector::new(rotation * Vec3::X * 0.1, Vec3::Z * 0.1),
                        )
                        .unwrap();
                    let case = Case {
                        body,
                        joint,
                        origin,
                        rotation,
                        mass,
                        model,
                        cap,
                        q: [0.; 2],
                        v: [0.1; 2],
                    };
                    scene
                        .set_joint_motor(joint, JointAxis::AngZ, angular_motor(&case, 0.7))
                        .unwrap();
                    cases.push(case);
                }
            }
        }
    }
    assert_eq!(cases.len(), 24);
    (scene, cases)
}

type Trace = Vec<Vec<(Transform, SpatialVector)>>;

fn rotation_error(actual: Quat, expected: Quat) -> f64 {
    let relative = actual.normalize() * expected.normalize().conjugate();
    2. * Vec3::new(relative.x, relative.y, relative.z)
        .length()
        .atan2(relative.w.abs())
}

fn certify(
    world: &mut RigidWorld,
    cases: &mut [Case],
    tolerance: f64,
    reference: Option<&Trace>,
) -> Trace {
    world.solver.substeps = SUBSTEPS;
    world.solver.stable_motor_feedback = true;
    world.set_reaction_readout(true).unwrap();
    let h = DT / SUBSTEPS as f64;
    let mut target = 0.7;
    let mut trace = Vec::new();
    let mut peak_pose: f64 = 0.;
    let mut peak_rate: f64 = 0.;
    let mut peak_reaction: f64 = 0.;
    let mut peak_parity: f64 = 0.;
    #[cfg(feature = "gpu")]
    let mut compilations = None;
    for frame in 0..FRAMES {
        if frame == 120 {
            target = 0.4;
            let stamp = world.scene.model_stamp();
            for case in cases.iter() {
                world
                    .scene
                    .set_joint_motor(case.joint, JointAxis::AngZ, angular_motor(case, target))
                    .unwrap();
            }
            assert_eq!(world.scene.model_stamp(), stamp);
        }
        let mut torques = vec![0.; cases.len()];
        for _ in 0..SUBSTEPS {
            for (i, case) in cases.iter_mut().enumerate() {
                let inertia = 4. * case.mass;
                let scale = if case.model == MotorModel::Acceleration {
                    inertia
                } else {
                    1.
                };
                let torque = (scale * 3. * (target - case.v[1]) / (1. + h * 3. * scale / inertia))
                    .clamp(-case.cap, case.cap);
                torques[i] += torque / SUBSTEPS as f64;
                case.v[0] += h * (0.6 - 0.1 * 2.);
                case.v[1] += h * torque / inertia;
                case.q[0] += h * case.v[0];
                case.q[1] += 2. * (0.5 * h * case.v[1]).atan();
            }
        }
        world
            .step(DT)
            .unwrap_or_else(|e| panic!("mixed drives frame={frame}: {e}"));
        assert_eq!(world.completed_steps(), (frame + 1) as u64);
        assert!(world.quarantined_bodies().is_empty());
        assert!(world.joint_break_events().is_empty());
        let state = world.scene.state();
        assert!(
            state
                .body_f
                .host()
                .unwrap()
                .iter()
                .all(|f| f.linear.is_finite() && f.angular.is_finite())
        );
        assert!(state.joint_q.host().unwrap().iter().all(|q| q.is_finite()));
        assert!(state.joint_qd.host().unwrap().iter().all(|v| v.is_finite()));
        let samples = world.joint_reaction_samples().unwrap();
        assert_eq!(samples.len(), SUBSTEPS * cases.len());
        for sample in samples {
            let case = cases
                .iter()
                .find(|case| case.joint == sample.joint)
                .unwrap();
            let force_balance = (sample.child.linear + sample.parent.linear).length() / case.mass;
            let moment_balance = (sample.child.angular
                + sample.parent.angular
                + (sample.anchors[1] - sample.anchors[0]).cross(sample.child.linear))
            .length()
                / (4. * case.mass);
            assert!(
                force_balance <= tolerance && moment_balance <= tolerance,
                "frame={frame} force_balance={force_balance:e} moment_balance={moment_balance:e}"
            );
        }
        assert_eq!(world.contact_pairs().len(), cases.len());
        let mut row = Vec::new();
        for (i, case) in cases.iter().enumerate() {
            let pose = world.scene.body_pose(case.body).unwrap();
            let velocity = world.scene.body_velocity(case.body).unwrap();
            assert!(pose.position.is_finite() && pose.rotation.is_finite());
            assert!(velocity.linear.is_finite() && velocity.angular.is_finite());
            let direction = case.rotation * Vec3::X;
            let expected_pose = Transform::new(
                case.origin + direction * case.q[0],
                case.rotation * Quat::from_rotation_z(case.q[1]),
            );
            let pose_error = (pose.position - expected_pose.position)
                .length()
                .max(rotation_error(pose.rotation, expected_pose.rotation));
            let rate_error = (velocity.linear - direction * case.v[0])
                .length()
                .max((velocity.angular - Vec3::Z * case.v[1]).length());
            peak_pose = peak_pose.max(pose_error);
            peak_rate = peak_rate.max(rate_error);
            assert!(
                pose_error <= tolerance && rate_error <= tolerance,
                "frame={frame} case={i} pose_error={pose_error:e} rate_error={rate_error:e} actual={velocity:?} expected_linear={:?} expected_angular={:?}",
                direction * case.v[0],
                Vec3::Z * case.v[1]
            );
            let reaction = world.joint_reaction(case.joint).unwrap();
            for wrench in [reaction.child, reaction.parent] {
                assert!(wrench.linear.is_finite() && wrench.angular.is_finite());
            }
            let force_error =
                (reaction.child.linear.dot(direction) - 0.6 * case.mass).abs() / case.mass;
            let torque_error = (reaction.child.angular.z - torques[i]).abs() / (4. * case.mass);
            peak_reaction = peak_reaction.max(force_error.max(torque_error));
            assert!(
                force_error <= tolerance && torque_error <= tolerance,
                "frame={frame} case={i} force_error={force_error:e} torque_error={torque_error:e}"
            );
            let pair = world
                .contact_pairs()
                .into_iter()
                .find(|pair| pair.bodies.contains(&Some(case.body)))
                .unwrap();
            assert_eq!(pair.points.len(), SUBSTEPS);
            for point in pair.points {
                assert!(point.force_on_b.is_finite() && point.normal.is_finite());
                assert!(point.tangent_impulse.is_finite() && point.normal_impulse.is_finite());
                assert_eq!(
                    point.friction_regime,
                    molla_solvers::penalty_contact::FrictionRegime::Sliding
                );
                assert!(point.acted);
                let normal_error = (point.normal_force / case.mass - 2.).abs();
                assert!(
                    normal_error <= tolerance.max(1e-7),
                    "frame={frame} case={i} normal_error={normal_error:e}"
                );
                let friction_error = (point.tangent_impulse.length() / h / case.mass - 0.2).abs();
                assert!(
                    friction_error <= tolerance.max(1e-7),
                    "frame={frame} case={i} friction_error={friction_error:e}"
                );
            }
            if let Some(reference) = reference {
                let (cpu_pose, cpu_velocity) = reference[frame][i];
                let parity = (pose.position - cpu_pose.position)
                    .length()
                    .max(rotation_error(pose.rotation, cpu_pose.rotation))
                    .max((velocity.linear - cpu_velocity.linear).length())
                    .max((velocity.angular - cpu_velocity.angular).length());
                peak_parity = peak_parity.max(parity);
                assert!(parity <= 1e-3, "frame={frame} case={i} parity={parity:e}");
            }
            row.push((pose, velocity));
        }
        #[cfg(feature = "gpu")]
        if let Some(diagnostics) = world.gpu_diagnostics() {
            assert_eq!(
                *compilations.get_or_insert(diagnostics.model_compilations),
                diagnostics.model_compilations
            );
        }
        trace.push(row);
    }
    println!(
        "mixed_drives cases={} frames={FRAMES} dt={DT} substeps={SUBSTEPS} peak_pose={peak_pose:e} peak_rate={peak_rate:e} peak_reaction={peak_reaction:e} peak_parity={peak_parity:e}",
        cases.len()
    );
    trace
}

#[test]
fn cpu_mixed_linear_angular_drives_with_contact_match_independent_oracles() {
    let (scene, mut cases) = fixture();
    certify(
        &mut RigidWorld::from_scene(scene).unwrap(),
        &mut cases,
        1e-8,
        None,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_mixed_linear_angular_drives_with_contact_match_oracles_and_cpu() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let (cpu_scene, mut cpu_cases) = fixture();
    let reference = certify(
        &mut RigidWorld::from_scene(cpu_scene).unwrap(),
        &mut cpu_cases,
        1e-8,
        None,
    );
    let (scene, mut cases) = fixture();
    let preparation = std::time::Instant::now();
    println!("mixed_drives_prepare_begin");
    let mut world = RigidWorld::from_scene_gpu_with_contact_capacity(
        scene,
        engine,
        molla_solvers::rigid_world::GpuContactCapacity {
            pairs: 48,
            contacts_per_island: 8,
        },
    )
    .unwrap();
    println!(
        "mixed_drives_prepare_complete seconds={:.6}",
        preparation.elapsed().as_secs_f64()
    );
    certify(&mut world, &mut cases, 1e-3, Some(&reference));
}
