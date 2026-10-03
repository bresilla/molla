use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, JointDesc, JointHandle, RigidScene, SphericalConeLimits,
};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

#[cfg(feature = "gpu")]
fn hardware_engine() -> std::sync::Arc<molla_compute::Engine> {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(
        !adapter.is_software,
        "cone contract requires a hardware GPU"
    );
    println!("adapter={adapter:?}");
    engine
}

fn fixture(
    axis: Vec3,
    frame: Quat,
    limits: SphericalConeLimits,
) -> (RigidScene, BodyHandle, JointHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams {
                pose: Transform::from_rotation(frame),
                ..Default::default()
            },
            additional_mass: Some(MassProperties {
                mass: 1.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::Ball,
            axis,
            frame_parent: Transform::from_rotation(frame),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene.set_cone_limits(joint, limits).unwrap();
    (scene, body, joint)
}

fn loaded(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    for (axis, frame) in [
        (Vec3::X, Quat::IDENTITY),
        (Vec3::Y, Quat::from_rotation_x(0.8)),
        (Vec3::Z, Quat::from_rotation_y(-0.6)),
        (Vec3::new(2.0, -3.0, 4.0), Quat::from_rotation_z(1.1)),
    ] {
        let [p, b0, _] = molla_sim::spherical_cone::basis(axis).unwrap();
        for (a0, a1) in [(0.4, 0.4), (0.4, 0.9), (0.0, 0.9), (0.0, 0.0)] {
            let (scene, body, joint) = fixture(
                axis,
                frame,
                SphericalConeLimits {
                    angle0: Some(a0),
                    angle1: Some(a1),
                },
            );
            let mut world = make(scene);
            let torque = frame * p.cross(b0) * 3.0;
            world
                .scene
                .add_body_wrench(body, SpatialVector::new(Vec3::ZERO, torque))
                .unwrap();
            let mut peak = 0.0_f64;
            for _ in 0..240 {
                world.step(1.0 / 120.0).unwrap();
                let reading = world.cone_reading(joint).unwrap();
                peak = peak.max(reading.violation);
                assert!(
                    reading.violation < tolerance,
                    "axis={axis:?}, extents={a0},{a1}: {reading:?}"
                );
                assert!(world.scene.body_pose(body).unwrap().position.length() < tolerance);
            }
            let reading = world.cone_reading(joint).unwrap();
            assert!(reading.active, "{reading:?}");
            assert!((reading.swing[0] - a0).abs() < tolerance);
            assert!(
                (reading.torque + torque).length() < tolerance * 100.0,
                "{reading:?}, applied={torque:?}"
            );
            eprintln!(
                "loaded cone axis={axis:?} extents={a0},{a1} peak={peak:e} torque_error={:e}",
                (reading.torque + torque).length()
            );
        }
    }
}

fn free_twist(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    let (scene, body, joint) = fixture(
        Vec3::X,
        Quat::IDENTITY,
        SphericalConeLimits {
            angle0: Some(0.0),
            angle1: Some(0.0),
        },
    );
    let mut world = make(scene);
    world
        .scene
        .set_body_velocity(body, SpatialVector::new(Vec3::ZERO, Vec3::X * 2.0))
        .unwrap();
    for _ in 0..240 {
        world.step(1.0 / 120.0).unwrap();
        let reading = world.cone_reading(joint).unwrap();
        assert!(reading.violation < tolerance);
        assert!(reading.torque.length() < tolerance);
        assert!(
            (world.scene.body_velocity(body).unwrap().angular - Vec3::X * 2.0).length() < tolerance
        );
    }
}

#[test]
fn cpu_loaded_cones_hold_boundary_and_react_torque() {
    loaded(|s| RigidWorld::from_scene(s).unwrap(), 1e-6);
}
#[test]
fn cpu_zero_cone_preserves_free_twist() {
    free_twist(|s| RigidWorld::from_scene(s).unwrap(), 1e-9);
}

fn momentum(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    let (mut scene, child, old) = fixture(Vec3::X, Quat::IDENTITY, SphericalConeLimits::default());
    scene.remove_joint(old).unwrap();
    scene
        .set_body_mass_properties(
            child,
            MassProperties {
                mass: 3.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 3.0,
            },
        )
        .unwrap();
    let parent = scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 2.0,
            }),
            ..Default::default()
        })
        .unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(parent),
            child,
            kind: JointType::Ball,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_cone_limits(
            joint,
            SphericalConeLimits {
                angle0: Some(0.0),
                angle1: Some(0.0),
            },
        )
        .unwrap();
    scene
        .set_body_velocity(child, SpatialVector::new(Vec3::ZERO, Vec3::Z * 2.0))
        .unwrap();
    scene
        .add_body_wrench(child, SpatialVector::new(Vec3::ZERO, Vec3::Z * 3.0))
        .unwrap();
    let mut world = make(scene);
    for frame in 0..120 {
        world.step(1.0 / 120.0).unwrap();
        let expected = (6.0 + 3.0 * (frame + 1) as f64 / 120.0) / 5.0;
        for body in [parent, child] {
            assert!(
                (world.scene.body_velocity(body).unwrap().angular - Vec3::Z * expected).length()
                    < tolerance
            );
        }
        assert!(world.cone_reading(joint).unwrap().violation < tolerance);
    }
    assert!(
        (world.cone_reading(joint).unwrap().torque + Vec3::Z * 1.2).length() < tolerance * 100.0
    );
}

fn diagonal_twist(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    let limits = SphericalConeLimits {
        angle0: Some(0.4),
        angle1: Some(0.9),
    };
    let (mut scene, body, joint) = fixture(Vec3::X, Quat::IDENTITY, limits);
    let direction = Vec3::new(0.0, 0.4, 0.9).normalize();
    let angle = 0.4_f64.hypot(0.9) / 2.0_f64.sqrt();
    let rotation = Quat::from_axis_angle(Vec3::X.cross(direction), angle);
    scene
        .set_body_pose(body, Transform::from_rotation(rotation))
        .unwrap();
    let geometry = molla_sim::spherical_cone::evaluate(limits, Vec3::X, rotation).unwrap();
    let torque = geometry.rows[0].angular.normalize() * 3.0;
    let spin = rotation * Vec3::X * 2.0;
    scene
        .set_body_velocity(body, SpatialVector::new(Vec3::ZERO, spin))
        .unwrap();
    scene
        .add_body_wrench(body, SpatialVector::new(Vec3::ZERO, torque))
        .unwrap();
    let mut world = make(scene);
    let mut mean = Vec3::ZERO;
    let mut peak_velocity = 0.0_f64;
    let mut peak_boundary = 0.0_f64;
    for _ in 0..240 {
        world.step(1.0 / 120.0).unwrap();
        let reading = world.cone_reading(joint).unwrap();
        let elliptical = (reading.swing[0] / 0.4).powi(2) + (reading.swing[1] / 0.9).powi(2);
        peak_boundary = peak_boundary.max((elliptical - 1.0).abs());
        assert!((elliptical - 1.0).abs() < tolerance, "{reading:?}");
        let velocity = world.scene.body_velocity(body).unwrap().angular;
        peak_velocity = peak_velocity.max((velocity - spin).length());
        assert!(
            (velocity - spin).length() < tolerance * 10.0,
            "velocity={velocity:?}, expected={spin:?}, error={}, reading={reading:?}",
            (velocity - spin).length()
        );
        mean += reading.torque / 240.0;
    }
    eprintln!(
        "diagonal loaded twist: peak_velocity={peak_velocity:e} peak_boundary={peak_boundary:e} mean_torque_error={:e}",
        (mean + torque).length()
    );
    assert!(
        (mean + torque).length() < tolerance * 10.0,
        "mean={mean:?}, torque={torque:?}"
    );
}

fn pendulum(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    let rates = match std::env::var("PRIMITIVE_CONE_HZ") {
        Err(std::env::VarError::NotPresent) => vec![60, 120, 240],
        Ok(value) => match value.parse::<usize>().unwrap() {
            hz @ (60 | 120 | 240) => vec![hz],
            _ => panic!("PRIMITIVE_CONE_HZ must be 60, 120 or 240"),
        },
        Err(error) => panic!("invalid PRIMITIVE_CONE_HZ: {error}"),
    };
    println!("cone pendulum rates={rates:?} duration=10s substeps=8");
    for hz in rates {
        let dt = 1.0 / hz as f64;
        let (mut scene, body, joint) = fixture(
            Vec3::Y,
            Quat::IDENTITY,
            SphericalConeLimits {
                angle0: Some(0.4),
                angle1: Some(0.4),
            },
        );
        scene.set_gravity(-Vec3::Y * 9.81).unwrap();
        scene
            .set_body_mass_properties(
                body,
                MassProperties {
                    mass: 1.0,
                    local_com: -Vec3::Y,
                    inertia: Mat3::IDENTITY * 0.2,
                },
            )
            .unwrap();
        scene
            .set_body_pose(body, Transform::from_rotation(Quat::from_rotation_z(0.2)))
            .unwrap();
        let offset = scene.body_pose(body).unwrap().rotation * (-Vec3::Y);
        scene
            .set_body_velocity(
                body,
                SpatialVector::new((Vec3::Z * 3.0).cross(offset), Vec3::Z * 3.0),
            )
            .unwrap();
        let mut world = make(scene);
        world.solver.substeps = 8;
        let energy = |world: &RigidWorld| {
            let v = world.scene.body_velocity(body).unwrap();
            0.5 * (v.linear.length_squared() + 0.2 * v.angular.length_squared())
                + 9.81 * (world.scene.body_world_com(body).unwrap().y + 1.0)
        };
        let initial = energy(&world);
        let mut peak = 0.0_f64;
        let mut peak_anchor = 0.0_f64;
        let mut active_frames = 0;
        let mut peak_energy = initial;
        for frame in 1..=10 * hz {
            world.step(dt).unwrap();
            assert!(world.quarantined_bodies().is_empty());
            assert!(world.joint_break_events().is_empty());
            assert_eq!(world.completed_steps(), frame as u64);
            let reading = world.cone_reading(joint).unwrap();
            active_frames += usize::from(reading.active);
            peak = peak.max(reading.violation);
            let pose = world.scene.body_pose(body).unwrap();
            let velocity = world.scene.body_velocity(body).unwrap();
            assert!(pose.position.is_finite() && pose.rotation.is_finite());
            assert!(velocity.linear.is_finite() && velocity.angular.is_finite());
            let current_energy = energy(&world);
            assert!(current_energy.is_finite());
            peak_energy = peak_energy.max(current_energy);
            peak_anchor = peak_anchor.max(pose.position.length());
            assert!(peak < tolerance, "dt={dt} peak={peak}");
            assert!(
                peak_energy < initial * 1.01 + 1e-3,
                "dt={dt} initial={initial}, peak={peak_energy}"
            );
            assert!(peak_anchor < tolerance);
            if frame % hz == 0 {
                println!(
                    "cone pendulum progress: hz={hz} elapsed={}s active_frames={active_frames} peak={peak:e} peak_anchor={peak_anchor:e} peak_energy={peak_energy}",
                    frame / hz
                );
            }
        }
        assert!(active_frames > 0, "pendulum must impact the cone boundary");
        eprintln!(
            "cone pendulum duration=10s dt={dt} substeps=8 active_frames={active_frames} peak={peak:e} peak_anchor={peak_anchor:e} initial_energy={initial} peak_energy={peak_energy} final_energy={}",
            energy(&world)
        );
    }
}

#[test]
fn cpu_cone_impacts_preserve_two_body_angular_momentum() {
    momentum(|s| RigidWorld::from_scene(s).unwrap(), 1e-8);
}
#[test]
fn cpu_diagonal_ellipse_retains_loaded_free_twist() {
    diagonal_twist(|s| RigidWorld::from_scene(s).unwrap(), 1e-8);
}

#[test]
fn cpu_diagonal_ellipse_with_f32_quantized_inputs() {
    diagonal_twist(
        |mut scene| {
            let bodies: Vec<_> = scene.bodies().map(|(id, _)| id).collect();
            for body in bodies {
                let pose = scene.body_pose(body).unwrap();
                scene
                    .set_body_pose(
                        body,
                        Transform::from_rotation(pose.rotation.as_quat().as_dquat().normalize()),
                    )
                    .unwrap();
                let velocity = scene.body_velocity(body).unwrap();
                scene
                    .set_body_velocity(
                        body,
                        SpatialVector::new(
                            velocity.linear.as_vec3().as_dvec3(),
                            velocity.angular.as_vec3().as_dvec3(),
                        ),
                    )
                    .unwrap();
            }
            for wrench in scene.state_mut().body_f.host_mut().unwrap() {
                wrench.linear = wrench.linear.as_vec3().as_dvec3();
                wrench.angular = wrench.angular.as_vec3().as_dvec3();
            }
            let limits: Vec<_> = scene
                .cone_joints()
                .map(|(id, _, limits)| (id, limits))
                .collect();
            for (id, limits) in limits {
                scene
                    .set_cone_limits(
                        id,
                        SphericalConeLimits {
                            angle0: limits.angle0.map(|v| v as f32 as f64),
                            angle1: limits.angle1.map(|v| v as f32 as f64),
                        },
                    )
                    .unwrap();
            }
            RigidWorld::from_scene(scene).unwrap()
        },
        2e-5,
    );
}
#[test]
fn cpu_cone_pendulum_is_bounded_for_ten_seconds() {
    pendulum(|s| RigidWorld::from_scene(s).unwrap(), 1e-4);
}

#[test]
fn cpu_external_ball_cone_preserves_compliant_anchor_under_gravity() {
    let (mut scene, body, old) = fixture(Vec3::Y, Quat::IDENTITY, SphericalConeLimits::default());
    scene.remove_joint(old).unwrap();
    scene.set_gravity(-Vec3::Y * 9.81).unwrap();
    scene
        .set_body_mass_properties(
            body,
            MassProperties {
                mass: 1.0,
                local_com: -Vec3::Y,
                inertia: Mat3::IDENTITY * 0.2,
            },
        )
        .unwrap();
    let joint = scene
        .insert_loop_joint(
            JointDesc {
                parent: None,
                child: body,
                kind: JointType::Ball,
                axis: Vec3::Y,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            },
            molla_sim::joint_control::JointSoftness {
                natural_frequency: 80.0,
                damping_ratio: 1.0,
            },
        )
        .unwrap();
    scene
        .set_cone_limits(
            joint,
            SphericalConeLimits {
                angle0: Some(0.4),
                angle1: Some(0.4),
            },
        )
        .unwrap();
    scene
        .set_body_pose(body, Transform::from_rotation(Quat::from_rotation_z(0.2)))
        .unwrap();
    scene
        .add_body_wrench(body, SpatialVector::new(Vec3::ZERO, Vec3::Z * 8.0))
        .unwrap();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    let mut peak_anchor = 0.0_f64;
    let mut peak_cone = 0.0_f64;
    for _ in 0..1200 {
        world.step(1.0 / 120.0).unwrap();
        peak_anchor = peak_anchor.max(world.scene.body_pose(body).unwrap().position.length());
        peak_cone = peak_cone.max(world.cone_reading(joint).unwrap().violation);
        assert!(peak_anchor < 0.01, "anchor={peak_anchor}");
        assert!(peak_cone < 1e-4, "cone={peak_cone}");
    }
    assert!(world.cone_reading(joint).unwrap().active);
    eprintln!(
        "CPU external ball cone duration=10s peak_anchor={peak_anchor:e} peak_cone={peak_cone:e}"
    );
}

fn mutation_sleep(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    let (mut scene, body, joint) = fixture(
        Vec3::X,
        Quat::IDENTITY,
        SphericalConeLimits {
            angle0: Some(0.0),
            angle1: Some(0.0),
        },
    );
    let other = scene
        .insert_body(BodyDesc {
            velocity: SpatialVector::new(Vec3::X, Vec3::ZERO),
            ..Default::default()
        })
        .unwrap();
    let mut world = make(scene);
    world.sleep_body(body).unwrap();
    world
        .scene
        .set_body_velocity_with_wake(body, SpatialVector::new(Vec3::ZERO, Vec3::Y), false)
        .unwrap();
    for _ in 0..20 {
        world.step(0.01).unwrap();
        assert!(world.is_sleeping(body));
        assert!(world.cone_reading(joint).is_none());
        assert!(
            world
                .scene
                .body_pose(body)
                .unwrap()
                .rotation
                .angle_between(Quat::IDENTITY)
                .abs()
                < tolerance
        );
    }
    assert!((world.scene.body_pose(other).unwrap().position.x - 0.2).abs() < tolerance);
    world
        .scene
        .set_cone_limits(
            joint,
            SphericalConeLimits {
                angle0: None,
                angle1: Some(0.0),
            },
        )
        .unwrap();
    world.step(0.01).unwrap();
    assert!(!world.is_sleeping(body));
    assert!(world.cone_reading(joint).unwrap().active);
    world.scene.remove_body(other).unwrap();
    assert!(world.cone_reading(joint).is_none());
    world.step(0.01).unwrap();
    assert!(world.cone_reading(joint).is_some());
    world
        .scene
        .set_cone_limits(joint, SphericalConeLimits::default())
        .unwrap();
    world
        .scene
        .add_body_wrench(body, SpatialVector::new(Vec3::ZERO, Vec3::Z))
        .unwrap();
    world.step(0.1).unwrap();
    assert!(world.cone_reading(joint).is_none());
    assert!(world.scene.body_velocity(body).unwrap().angular.z > 0.09);
    world.scene.remove_joint(joint).unwrap();
    world.step(0.01).unwrap();
}

#[test]
fn cpu_cone_mutation_sleep_and_compaction() {
    mutation_sleep(|s| RigidWorld::from_scene(s).unwrap(), 1e-8);
}

fn mixed_radial(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    let (mut scene, child, old) = fixture(Vec3::X, Quat::IDENTITY, SphericalConeLimits::default());
    scene.remove_joint(old).unwrap();
    scene
        .set_body_mass_properties(
            child,
            MassProperties {
                mass: 3.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 3.0,
            },
        )
        .unwrap();
    scene
        .set_body_pose(child, Transform::from_translation(Vec3::X))
        .unwrap();
    let parent = scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 2.0,
            }),
            ..Default::default()
        })
        .unwrap();
    let cone = scene
        .insert_joint(JointDesc {
            parent: Some(parent),
            child,
            kind: JointType::Ball,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(Vec3::X),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_cone_limits(
            cone,
            SphericalConeLimits {
                angle0: Some(0.0),
                angle1: Some(0.0),
            },
        )
        .unwrap();
    let radial = scene
        .insert_joint(JointDesc {
            parent: None,
            child,
            kind: JointType::Distance,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(Vec3::X),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_distance_limits(
            radial,
            molla_sim::runtime::DistanceLimits {
                min: None,
                max: Some(0.0),
            },
        )
        .unwrap();
    scene
        .add_body_wrench(child, SpatialVector::new(Vec3::ZERO, Vec3::Z))
        .unwrap();
    let mut world = make(scene);
    for frame in 0..10 {
        world.step(0.01).unwrap();
        assert!(world.cone_reading(cone).unwrap().violation < tolerance);
        assert!(world.distance_reading(radial).unwrap().violation < tolerance);
        let expected = Vec3::Z * ((frame + 1) as f64 * 0.01 / 7.0);
        for body in [parent, child] {
            assert!(
                (world.scene.body_velocity(body).unwrap().angular - expected).length() < tolerance
            );
        }
    }
    let torque_error = (world.cone_reading(cone).unwrap().torque + Vec3::Z * (4.0 / 7.0)).length();
    assert!(
        torque_error < tolerance * 100.0,
        "torque error={torque_error}"
    );
}

#[test]
fn cpu_cone_and_radial_rows_share_offset_articulated_response() {
    mixed_radial(|s| RigidWorld::from_scene(s).unwrap(), 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_cone_and_radial_rows_share_offset_articulated_response() {
    let engine = hardware_engine();
    mixed_radial(
        |s| RigidWorld::from_scene_gpu(s, engine.clone()).unwrap(),
        3e-5,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_loaded_cones_hold_boundary_and_react_torque() {
    let engine = hardware_engine();
    loaded(
        |s| RigidWorld::from_scene_gpu(s, engine.clone()).unwrap(),
        2e-5,
    );
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_zero_cone_preserves_free_twist() {
    let engine = hardware_engine();
    free_twist(
        |s| RigidWorld::from_scene_gpu(s, engine.clone()).unwrap(),
        2e-5,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_cone_impacts_preserve_two_body_angular_momentum() {
    let engine = hardware_engine();
    momentum(
        |s| RigidWorld::from_scene_gpu(s, engine.clone()).unwrap(),
        3e-4,
    );
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_diagonal_ellipse_retains_loaded_free_twist() {
    let engine = hardware_engine();
    diagonal_twist(
        |s| RigidWorld::from_scene_gpu(s, engine.clone()).unwrap(),
        2e-5,
    );
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_cone_pendulum_is_bounded_for_ten_seconds() {
    let engine = hardware_engine();
    pendulum(
        |s| RigidWorld::from_scene_gpu(s, engine.clone()).unwrap(),
        3e-4,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_cone_mutation_sleep_and_compaction() {
    let engine = hardware_engine();
    mutation_sleep(
        |s| RigidWorld::from_scene_gpu(s, engine.clone()).unwrap(),
        2e-5,
    );
}
