#![cfg(feature = "gpu")]

#[path = "support/hardware_engine.rs"]
mod hardware_engine;
use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, DistanceLimits, JointDesc, RigidScene};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn fixture(
    min: Option<f64>,
    max: Option<f64>,
    mass: f64,
    velocity: Vec3,
) -> (
    RigidScene,
    molla_sim::runtime::BodyHandle,
    molla_sim::runtime::BodyHandle,
    molla_sim::runtime::JointHandle,
) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut bodies = Vec::new();
    for (position, mass, velocity) in [(Vec3::ZERO, mass, Vec3::ZERO), (Vec3::X, 1.0, velocity)] {
        bodies.push(
            scene
                .insert_body(BodyDesc {
                    params: BodyParams {
                        pose: Transform::from_translation(position),
                        ..Default::default()
                    },
                    velocity: SpatialVector::new(velocity, Vec3::ZERO),
                    additional_mass: Some(MassProperties {
                        mass,
                        local_com: Vec3::ZERO,
                        inertia: Mat3::IDENTITY * mass,
                    }),
                    ..Default::default()
                })
                .unwrap(),
        );
    }
    let j = scene
        .insert_joint(JointDesc {
            parent: Some(bodies[0]),
            child: bodies[1],
            kind: JointType::Distance,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_distance_limits(j, DistanceLimits { min, max })
        .unwrap();
    (scene, bodies[0], bodies[1], j)
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_distance_gpu_shared_response_and_independent_bounds() {
    let engine = hardware_engine::engine();
    for (min, max, velocity) in [
        (Some(1.0), Some(1.0), Vec3::X),
        (None, Some(1.0), Vec3::X),
        (Some(1.0), None, -Vec3::X),
        (Some(0.5), Some(1.5), Vec3::X),
        (None, Some(2.0), Vec3::Y),
        (None, None, Vec3::new(-1.0, 2.0, 3.0)),
    ] {
        for mass in [1.0, 1000.0, 0.001] {
            let (scene, a, b, j) = fixture(min, max, mass, velocity);
            let mut cpu = RigidWorld::from_scene(scene).unwrap();
            let (scene, ga, gb, gj) = fixture(min, max, mass, velocity);
            let mut gpu = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
            cpu.solver.substeps = 2;
            gpu.solver.substeps = 2;
            for frame in 0..4 {
                cpu.step(0.01).unwrap();
                gpu.step(0.01).unwrap();
                for (c, g) in [(a, ga), (b, gb)] {
                    let dv = (cpu.scene.body_velocity(c).unwrap().linear
                        - gpu.scene.body_velocity(g).unwrap().linear)
                        .length();
                    let dp = (cpu.scene.body_pose(c).unwrap().position
                        - gpu.scene.body_pose(g).unwrap().position)
                        .length();
                    assert!(
                        dv < 1e-3 && dp < 1e-3,
                        "mass={mass} min={min:?} max={max:?} frame={frame}: dv={dv} dp={dp}"
                    );
                }
                let cr = cpu.distance_reading(j).unwrap();
                let gr = gpu.distance_reading(gj).unwrap();
                assert!(
                    (cr.force - gr.force).length() < 0.02,
                    "CPU {cr:?}, GPU {gr:?}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_distance_gpu_world_pendulum_ten_seconds_and_short_parity() {
    let engine = hardware_engine::engine();
    for dt in [1.0 / 60.0, 1.0 / 120.0, 1.0 / 240.0] {
        let make = || {
            let (mut scene, a, b, j) = fixture(Some(1.0), Some(1.0), 1.0, Vec3::Z);
            scene
                .set_body_kind(a, molla_sim::runtime::BodyKind::Fixed)
                .unwrap();
            scene.set_gravity(Vec3::new(0.0, -9.81, 0.0)).unwrap();
            (scene, b, j)
        };
        let (scene, b, j) = make();
        let mut gpu = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        let (scene, cb, _) = make();
        let mut cpu = RigidWorld::from_scene(scene).unwrap();
        gpu.solver.substeps = 8;
        cpu.solver.substeps = 8;
        let mut peak: f64 = 0.0;
        let mut parity: f64 = 0.0;
        for step in 0..(10.0 / dt) as usize {
            gpu.step(dt).unwrap();
            peak = peak.max(gpu.distance_reading(j).unwrap().violation);
            if (step as f64) * dt < 1.0 {
                cpu.step(dt).unwrap();
                parity = parity.max(
                    (gpu.scene.body_pose(b).unwrap().position
                        - cpu.scene.body_pose(cb).unwrap().position)
                        .length(),
                );
            }
        }
        println!(
            "GPU distance pendulum dt={dt} substeps=8 duration=10s peak={peak} one-second-parity={parity}"
        );
        assert!(peak <= 1e-3 && parity <= 1e-3);
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_distance_gpu_zero_equality_failure_and_resume_are_atomic() {
    let engine = hardware_engine::engine();
    let (mut scene, a, b, j) = fixture(None, Some(0.0), 1.0, Vec3::new(1.0, 2.0, 3.0));
    scene.set_body_pose(b, Transform::IDENTITY).unwrap();
    scene
        .set_body_kind(a, molla_sim::runtime::BodyKind::Fixed)
        .unwrap();
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    gpu.step(0.01).unwrap();
    assert!(gpu.scene.body_velocity(b).unwrap().linear.length() < 1e-5);
    assert!(gpu.distance_reading(j).unwrap().violation < 1e-5);
    gpu.scene
        .set_body_kind(b, molla_sim::runtime::BodyKind::Fixed)
        .unwrap();
    gpu.scene
        .set_distance_limits(
            j,
            DistanceLimits {
                min: Some(1.0),
                max: Some(1.0),
            },
        )
        .unwrap();
    let stamp = gpu.scene.stamp();
    let pose = gpu.scene.body_pose(b).unwrap();
    let error = gpu.step(0.01).unwrap_err().to_string();
    assert!(error.contains("solver fault 512"), "{error}");
    assert_eq!(gpu.scene.stamp(), stamp);
    assert_eq!(gpu.scene.body_pose(b).unwrap(), pose);
    assert!(gpu.distance_reading(j).is_none());
    gpu.scene
        .set_body_kind(b, molla_sim::runtime::BodyKind::Dynamic)
        .unwrap();
    gpu.step(0.01).unwrap();
    assert!(gpu.distance_reading(j).unwrap().violation < 1e-5);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_distance_gpu_offset_rotated_frames_full_inertia_and_loop() {
    use molla_math::Quat;
    let engine = hardware_engine::engine();
    let make = || {
        let (mut scene, a, b, j) = fixture(None, None, 2.0, Vec3::new(0.1, 0.2, -0.3));
        for id in [a, b] {
            scene
                .set_body_mass_properties(
                    id,
                    MassProperties {
                        mass: 2.0,
                        local_com: Vec3::new(0.1, -0.2, 0.05),
                        inertia: Mat3::from_cols(
                            Vec3::new(3.0, 0.2, 0.3),
                            Vec3::new(0.2, 4.0, 0.4),
                            Vec3::new(0.3, 0.4, 5.0),
                        ),
                    },
                )
                .unwrap();
        }
        let pa = Transform {
            position: Vec3::new(0.0, 0.3, 0.1),
            rotation: Quat::from_rotation_x(0.4),
        };
        let pb = Transform {
            position: Vec3::new(0.0, -0.2, 0.2),
            rotation: Quat::from_rotation_y(-0.7),
        };
        scene.set_joint_frames(j, pa, pb).unwrap();
        let length = (scene.body_pose(b).unwrap().compose(&pb).position
            - scene.body_pose(a).unwrap().compose(&pa).position)
            .length();
        scene
            .set_distance_limits(
                j,
                DistanceLimits {
                    min: Some(length),
                    max: Some(length),
                },
            )
            .unwrap();
        let redundant = scene
            .insert_joint(JointDesc {
                parent: Some(a),
                child: b,
                kind: JointType::Distance,
                axis: Vec3::X,
                frame_parent: pa,
                frame_child: pb,
            })
            .unwrap();
        scene
            .set_distance_limits(
                redundant,
                DistanceLimits {
                    min: Some(length),
                    max: Some(length),
                },
            )
            .unwrap();
        (scene, a, b, j)
    };
    let (scene, a, b, j) = make();
    let mut cpu = RigidWorld::from_scene(scene).unwrap();
    let (scene, ga, gb, gj) = make();
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    cpu.solver.substeps = 4;
    gpu.solver.substeps = 4;
    let mut peak: f64 = 0.0;
    for frame in 0..120 {
        cpu.step(1.0 / 120.0).unwrap();
        gpu.step(1.0 / 120.0).unwrap();
        for (c, g) in [(a, ga), (b, gb)] {
            let cp = cpu.scene.body_pose(c).unwrap();
            let gp = gpu.scene.body_pose(g).unwrap();
            peak = peak.max((cp.position - gp.position).length());
            assert!(cp.rotation.dot(gp.rotation).abs().min(1.0).acos() * 2.0 < 1e-3);
        }
        assert!(gpu.distance_reading(gj).unwrap().violation < 1e-3);
        assert!(cpu.distance_reading(j).unwrap().violation < 1e-3);
        let cr = cpu.distance_reading(j).unwrap().force;
        let gr = gpu.distance_reading(gj).unwrap().force;
        assert!(
            (cr - gr).length() < 0.01 * (1.0 + cr.length()),
            "redundant row reactions at frame {frame}: CPU {cr:?}, GPU {gr:?}; peak position={peak}"
        );
        let momentum = (gpu.scene.body_velocity(ga).unwrap().linear
            + gpu.scene.body_velocity(gb).unwrap().linear)
            * 2.0;
        assert!((momentum - Vec3::new(0.2, 0.4, -0.6)).length() < 1e-3);
    }
    println!("offset redundant distance one-second CPU/GPU peak position={peak}");
    assert!(peak < 1e-3);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_distance_gpu_resolves_sub_ulp_bound_gaps() {
    let engine = hardware_engine::engine();
    for length in [0.01, 0.015625, 1.0, 64.0] {
        let dt = length * 0.001;
        for axis in [Vec3::new(1.0, 0.5, 0.25), Vec3::X, Vec3::Y, Vec3::Z] {
            for sign in [-1.0, 1.0] {
                let separation = (axis * length).length();
                let bound = separation * (1.0 + sign * 2e-8);
                let make = || {
                    let (mut scene, a, b, joint) =
                        fixture(Some(bound), Some(bound), 2.0, Vec3::ZERO);
                    scene
                        .set_body_pose(b, Transform::from_translation(axis * length))
                        .unwrap();
                    (scene, a, b, joint)
                };
                let (scene, _, _, joint) = make();
                let mut cpu = RigidWorld::from_scene(scene).unwrap();
                let (scene, _, _, gpu_joint) = make();
                let mut gpu = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
                cpu.solver.substeps = 1;
                gpu.solver.substeps = 1;
                cpu.step(dt).unwrap();
                gpu.step(dt).unwrap();
                let expected = axis.normalize() * ((bound - separation) * (2.0 / 3.0) / (dt * dt));
                for (backend, actual) in [
                    ("CPU", cpu.distance_reading(joint).unwrap().force),
                    ("GPU", gpu.distance_reading(gpu_joint).unwrap().force),
                ] {
                    assert!(
                        (actual - expected).length() < expected.length() * 1e-4,
                        "{backend} length={length} axis={axis:?} sign={sign}: force={actual:?}, expected={expected:?}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_distance_gpu_resolves_anchors_with_identical_position_high_parts() {
    let engine = hardware_engine::engine();
    let origin = Vec3::new(4096.0, -4096.0, 8192.0);
    let dt = 0.001;
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        let endpoint = origin + axis * 1e-4;
        assert_eq!(origin.as_vec3(), endpoint.as_vec3());
        let separation = (endpoint - origin).length();
        for sign in [-1.0, 1.0] {
            let bound = separation * (1.0 + sign * 0.1);
            let (mut scene, a, b, joint) = fixture(Some(bound), Some(bound), 2.0, Vec3::ZERO);
            scene
                .set_body_pose(a, Transform::from_translation(origin))
                .unwrap();
            scene
                .set_body_pose(b, Transform::from_translation(endpoint))
                .unwrap();
            let mut gpu = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
            gpu.solver.substeps = 1;
            let expected = axis * ((bound - separation) * (2.0 / 3.0) / (dt * dt));
            for edited in [false, true] {
                if edited {
                    for (body, position) in [(a, origin), (b, endpoint)] {
                        gpu.scene
                            .set_body_pose(
                                body,
                                Transform::from_translation(position + Vec3::splat(0.03125)),
                            )
                            .unwrap();
                        gpu.scene
                            .set_body_velocity(body, SpatialVector::ZERO)
                            .unwrap();
                    }
                }
                gpu.step(dt).unwrap();
                let actual = gpu.distance_reading(joint).unwrap().force;
                assert!(
                    (actual - expected).length() < expected.length() * 1e-4,
                    "axis={axis:?} sign={sign} edited={edited}: force={actual:?}, expected={expected:?}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_distance_gpu_scalar_stop_couples_to_other_articulation() {
    use molla_sim::joint_control::JointAxis;
    let engine = hardware_engine::engine();
    let (mut scene, a, c, j) = fixture(Some(1.0), Some(1.0), 2.0, Vec3::ZERO);
    scene.remove_joint(j).unwrap();
    let b = scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 3.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    scene
        .set_body_mass_properties(
            c,
            MassProperties {
                mass: 5.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            },
        )
        .unwrap();
    let slider = scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Prismatic,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_joint_limits(slider, JointAxis::LinX, Some([-1.0, 0.0]))
        .unwrap();
    let distance = scene
        .insert_joint(JointDesc {
            parent: Some(b),
            child: c,
            kind: JointType::Distance,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_distance_limits(
            distance,
            DistanceLimits {
                min: Some(1.0),
                max: Some(1.0),
            },
        )
        .unwrap();
    scene
        .add_body_wrench(c, SpatialVector::new(Vec3::X * 10.0, Vec3::ZERO))
        .unwrap();
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    gpu.step(0.01).unwrap();
    for body in [a, b, c] {
        assert!((gpu.scene.body_velocity(body).unwrap().linear.x - 0.01).abs() < 1e-4);
    }
    assert!(gpu.distance_reading(distance).unwrap().violation < 1e-5);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_distance_gpu_redundant_support_matches_independent_load_shares() {
    let engine = hardware_engine::engine();
    for direction in [Vec3::X, Vec3::Y, Vec3::new(1.0, 0.5, 0.25).normalize()] {
        for copies in [2, 3, 5] {
            let origin = Vec3::new(1e5, -2e5, 3e5);
            let mut scene = RigidScene::new().unwrap();
            scene.set_gravity(-direction * 10.0).unwrap();
            let body = scene
                .insert_body(BodyDesc {
                    params: BodyParams::new().with_position(origin + direction),
                    additional_mass: Some(MassProperties {
                        mass: 2.0,
                        local_com: Vec3::ZERO,
                        inertia: Mat3::IDENTITY,
                    }),
                    ..Default::default()
                })
                .unwrap();
            let length = (scene.body_pose(body).unwrap().position - origin).length();
            let mut joints = Vec::new();
            for _ in 0..copies {
                let joint = scene
                    .insert_joint(JointDesc {
                        parent: None,
                        child: body,
                        kind: JointType::Distance,
                        axis: Vec3::X,
                        frame_parent: Transform::from_translation(origin),
                        frame_child: Transform::IDENTITY,
                    })
                    .unwrap();
                scene
                    .set_distance_limits(
                        joint,
                        DistanceLimits {
                            min: Some(length),
                            max: Some(length),
                        },
                    )
                    .unwrap();
                joints.push(joint);
            }
            let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
            world.solver.substeps = 4;
            let expected = direction * (20.0 / copies as f64);
            for frame in 0..8 {
                world.step(1.0 / 240.0).unwrap();
                let mut total = Vec3::ZERO;
                for &joint in &joints {
                    let reading = world.distance_reading(joint).unwrap();
                    assert!(
                        reading.violation < 1e-3,
                        "direction={direction:?} copies={copies} frame={frame}: reading={reading:?}, pose={:?}, length={length}",
                        world.scene.body_pose(body).unwrap()
                    );
                    assert!(
                        (reading.force - expected).length() < 0.01 * expected.length(),
                        "direction={direction:?} copies={copies} frame={frame}: {:?} != {expected:?}",
                        reading.force
                    );
                    total += reading.force;
                }
                assert!((total - direction * 20.0).length() < 0.2);
                assert_eq!(world.completed_steps(), frame + 1);
                assert!(world.quarantined_bodies().is_empty());
            }
        }
    }
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "requires GPU and wgpu resource counters"]
fn runtime_distance_gpu_retains_warm_resources_and_tracks_mutation() {
    let engine = hardware_engine::engine();
    let (scene, _a, b, j) = fixture(Some(1.0), Some(1.0), 2.0, Vec3::ZERO);
    let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
    world.solver.substeps = 4;
    let resources = || {
        engine
            .device()
            .unwrap()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let c = engine.device().unwrap().get_internal_counters().hal;
        [
            c.buffers.read(),
            c.buffer_memory.read(),
            c.bind_groups.read(),
            c.compute_pipelines.read(),
            c.shader_modules.read(),
        ]
    };
    for _ in 0..4 {
        world.step(0.01).unwrap();
    }
    let warm = resources();
    let diag = world.gpu_diagnostics().unwrap();
    assert_eq!(diag.model_compilations, 1);
    assert_eq!(diag.current_model_bind_groups_created, 11 + 18 + 2 + 2);
    for _ in 0..20 {
        world
            .scene
            .set_body_velocity(b, SpatialVector::new(Vec3::X, Vec3::ZERO))
            .unwrap();
        world.step(0.01).unwrap();
        assert_eq!(resources(), warm);
        assert_eq!(
            world
                .gpu_diagnostics()
                .unwrap()
                .current_model_bind_groups_created,
            diag.current_model_bind_groups_created
        );
        assert_eq!(world.gpu_diagnostics().unwrap().model_compilations, 1);
    }
    world
        .scene
        .set_distance_limits(
            j,
            DistanceLimits {
                min: None,
                max: None,
            },
        )
        .unwrap();
    assert!(world.distance_reading(j).is_none());
    world.step(0.01).unwrap();
    assert_eq!(world.gpu_diagnostics().unwrap().model_compilations, 2);
    assert_eq!(world.distance_reading(j).unwrap().force, Vec3::ZERO);
    world.scene.remove_joint(j).unwrap();
    world.step(0.01).unwrap();
    assert!(world.distance_reading(j).is_none());
    println!("distance GPU warm [buffers, bytes, groups, pipelines, shaders]={warm:?}");
}
