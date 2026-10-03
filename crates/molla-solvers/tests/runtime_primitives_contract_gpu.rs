#![cfg(feature = "gpu")]

use molla_compute::{Backend, Engine};
use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor};
use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene};
use molla_sim::{BodyParams, JointTargetMode, JointType};
use molla_solvers::rigid_world::RigidWorld;
use std::sync::Arc;

#[path = "support/world_isolation.rs"]
mod world_isolation;

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_gpu_shared_engine_keeps_world_gravity_state_and_mutations_isolated() {
    let engine = engine();
    world_isolation::verify(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        1e-5,
    );
}

fn engine() -> Arc<Engine> {
    static ENGINE: std::sync::OnceLock<Arc<Engine>> = std::sync::OnceLock::new();
    let engine = ENGINE
        .get_or_init(|| Arc::new(Engine::new(Backend::AutoGpu).unwrap()))
        .clone();
    let adapter = engine
        .adapter_info()
        .expect("requires GPU adapter information");
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    engine
}

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    struct Wake(std::thread::Thread);
    impl std::task::Wake for Wake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(Arc::new(Wake(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_gpu_setup_capacity_rejection_preserves_world_and_can_resume() {
    use molla_sim::runtime::{ColliderDesc, ColliderGeometry, DistanceLimits};

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .unwrap();
    let info = molla_compute::AdapterInfo::from_wgpu(adapter.get_info());
    assert!(!info.is_software, "requires a hardware GPU");
    let mut limits = adapter.limits();
    limits.max_storage_buffer_binding_size = 65536;
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("runtime setup rejection fixture"),
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap();
    let engine = Arc::new(Engine::from_device(
        Arc::new(device),
        Arc::new(queue),
        Some(info),
    ));
    let mut scene = RigidScene::new().unwrap();
    let mut desc = BodyDesc::default();
    desc.params.pose.position = Vec3::X;
    let body = scene.insert_body(desc.clone()).unwrap();
    let distance = |child| JointDesc {
        parent: None,
        child,
        kind: JointType::Distance,
        axis: Vec3::X,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
    };
    let bounds = DistanceLimits {
        min: Some(1.),
        max: Some(1.),
    };
    let joint = scene.insert_joint(distance(body)).unwrap();
    scene.set_distance_limits(joint, bounds).unwrap();
    scene
        .insert_collider(ColliderDesc {
            parent: Some(body),
            ..ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 })
        })
        .unwrap();
    let mut world = RigidWorld::from_scene_gpu_with_contact_capacity(
        scene,
        engine.clone(),
        molla_solvers::rigid_world::GpuContactCapacity {
            pairs: 32,
            contacts_per_island: 4,
        },
    )
    .unwrap();
    world.step(1. / 240.).unwrap();
    let mut added = Vec::new();
    for _ in 0..64 {
        let child = world.scene.insert_body(desc.clone()).unwrap();
        let extra = world.scene.insert_joint(distance(child)).unwrap();
        world.scene.set_distance_limits(extra, bounds).unwrap();
        added.push(child);
    }
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    let stamp = world.scene.model_stamp();
    let diagnostics = world.gpu_diagnostics().unwrap();
    let steps = world.completed_steps();
    let error = world.step(1. / 240.).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("scalar Jacobian capacity exceeded"),
        "{error}"
    );
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
    assert_eq!(world.scene.model_stamp(), stamp);
    assert_eq!(world.gpu_diagnostics().unwrap(), diagnostics);
    assert_eq!(world.completed_steps(), steps);
    assert_eq!(world.scene.distance_limits(joint).unwrap(), bounds);
    assert!(world.distance_reading(joint).is_none());
    engine.check_device().unwrap();
    for child in added.into_iter().rev() {
        world.scene.remove_body(child).unwrap();
    }
    world.step(1. / 240.).unwrap();
    assert_eq!(world.completed_steps(), steps + 1);
    assert_eq!(
        world.gpu_diagnostics().unwrap().model_compilations,
        diagnostics.model_compilations + 1
    );
    assert!(world.distance_reading(joint).unwrap().violation <= 1e-3);
    engine.check_device().unwrap();
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_gpu_empty_reaction_readout_stays_on_device() {
    let mut scene = RigidScene::new().unwrap();
    scene.insert_body(BodyDesc::default()).unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine()).unwrap();
    world.set_reaction_readout(true).unwrap();
    assert!(world.reaction_readout_enabled());
    world.step(0.01).unwrap();
    assert!(world.solver.is_gpu());
    assert!(world.joint_reaction_samples().unwrap().is_empty());
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_gpu_distance_mutation_is_solved_and_can_be_removed() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine()).unwrap();
    world.step(0.01).unwrap();
    let joint = world
        .scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::Distance,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    world
        .scene
        .set_distance_limits(
            joint,
            molla_sim::runtime::DistanceLimits {
                min: Some(0.0),
                max: Some(0.0),
            },
        )
        .unwrap();
    world.step(0.01).unwrap();
    assert!(world.scene.body_pose(body).unwrap().position.length() < 1e-5);
    assert!(world.distance_reading(joint).unwrap().violation < 1e-5);
    world.scene.remove_joint(joint).unwrap();
    world.step(0.01).unwrap();
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_gpu_free_body_matches_cpu_and_publishes_velocity() {
    let engine = engine();
    let create = || {
        let mut scene = RigidScene::new().unwrap();
        let body = scene
            .insert_body(BodyDesc {
                params: BodyParams {
                    pose: Transform::from_translation(Vec3::new(1.0, 2.0, 3.0)),
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
        (scene, body)
    };
    let (scene, a) = create();
    let mut cpu = RigidWorld::from_scene(scene).unwrap();
    let (scene, b) = create();
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    assert!(gpu.solver.is_gpu());
    for steps in [1, 2, 8] {
        gpu.solver.substeps = steps;
        cpu.solver.substeps = steps;
        for _ in 0..10 {
            cpu.step(1.0 / 120.0).unwrap();
            gpu.step(1.0 / 120.0).unwrap();
            let error = (cpu.scene.body_pose(a).unwrap().position
                - gpu.scene.body_pose(b).unwrap().position)
                .length();
            assert!(error < 1e-4, "position error={error}");
            let error = (cpu.scene.body_velocity(a).unwrap().linear
                - gpu.scene.body_velocity(b).unwrap().linear)
                .length();
            assert!(error < 1e-4, "velocity error={error}");
        }
    }
    assert_eq!(gpu.solver.substeps, 8);
    assert!(gpu.scene.body_velocity(b).unwrap().linear.length() > 1.0);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_gpu_mutations_rebuild_and_preserve_handles() {
    let engine = engine();
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    world.solver.substeps = 2;
    let model_stamp = world.scene.model_stamp();
    world
        .scene
        .set_body_velocity(body, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    assert_eq!(model_stamp, world.scene.model_stamp());
    world.step(0.01).unwrap();
    let before = world.scene.body_pose(body).unwrap();
    let extra = world.scene.insert_body(BodyDesc::default()).unwrap();
    assert_ne!(model_stamp, world.scene.model_stamp());
    world.step(0.01).unwrap();
    world.scene.remove_body(extra).unwrap();
    world.step(0.01).unwrap();
    assert!(
        (world.scene.body_pose(body).unwrap().position.x - before.position.x - 0.02).abs() < 1e-6
    );
    world.scene.set_gravity(Vec3::new(0.0, -4.0, 0.0)).unwrap();
    world.step(0.01).unwrap();
    assert!((world.scene.body_velocity(body).unwrap().linear.y + 0.04).abs() < 1e-6);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_gpu_rejects_unsupported_mutation_without_advancing() {
    let engine = engine();
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    let before = world.scene.body_pose(body).unwrap();
    let mut parent = body;
    let mut added = Vec::new();
    for _ in 0..32 {
        let child = world.scene.insert_body(BodyDesc::default()).unwrap();
        world
            .scene
            .insert_joint(JointDesc {
                parent: Some(parent),
                child,
                kind: JointType::Fixed,
                axis: Vec3::X,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        added.push(child);
        parent = child;
    }
    assert_eq!(world.scene.model().articulation_count, 1);
    assert_eq!(world.scene.model().joint_count, 33);
    let error = world.step(0.01).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("articulations larger than 32 joints"),
        "{error}"
    );
    assert_eq!(before, world.scene.body_pose(body).unwrap());
    assert_eq!(
        world.scene.body_velocity(body).unwrap(),
        SpatialVector::ZERO
    );
    assert_eq!(world.completed_steps(), 0);
    assert_eq!(world.gpu_diagnostics().unwrap().completed_frames, 0);
    for child in added.into_iter().rev() {
        world.scene.remove_body(child).unwrap();
    }
    world.step(0.01).unwrap();
    assert_ne!(before, world.scene.body_pose(body).unwrap());
    assert_eq!(world.completed_steps(), 1);
    assert!(
        RigidWorld::from_scene_gpu(
            RigidScene::new().unwrap(),
            Arc::new(Engine::new(Backend::Cpu).unwrap())
        )
        .is_err()
    );
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_gpu_world_anchor_uses_authored_frames() {
    let engine = engine();
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let frame = Transform::new(Vec3::new(1.0, 2.0, 0.0), Quat::from_rotation_x(0.3));
    let child_frame = Transform::new(Vec3::new(0.1, -0.2, 0.0), Quat::from_rotation_z(0.2));
    let expected = frame.compose(&child_frame.inverse());
    scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: frame,
            frame_child: child_frame,
        })
        .unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    world.step(0.01).unwrap();
    let pose = world.scene.body_pose(body).unwrap();
    assert!((pose.position - expected.position).length() < 1e-6);
    assert!((pose.rotation.normalize().angle_between(expected.rotation)).abs() < 1e-6);
    assert_eq!(
        world.scene.body_velocity(body).unwrap(),
        SpatialVector::ZERO
    );
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_gpu_rotated_frames_com_inertia_and_motor_parity() {
    let engine = engine();
    for kind in [JointType::Ball, JointType::Revolute, JointType::Prismatic] {
        let create = || {
            let mut scene = RigidScene::new().unwrap();
            let desc = BodyDesc {
                params: BodyParams {
                    com: Vec3::new(0.1, 0.2, -0.1),
                    inertia: Mat3::from_cols(
                        Vec3::new(2.0, 0.1, 0.2),
                        Vec3::new(0.1, 3.0, 0.3),
                        Vec3::new(0.2, 0.3, 4.0),
                    ),
                    pose: Transform::new(Vec3::new(0.0, 2.0, 0.0), Quat::from_rotation_z(0.3)),
                    ..Default::default()
                },
                ..Default::default()
            };
            let parent = scene.insert_body(desc.clone()).unwrap();
            let child = scene.insert_body(desc).unwrap();
            let joint = scene
                .insert_joint(JointDesc {
                    parent: Some(parent),
                    child,
                    kind,
                    axis: Vec3::new(1.0, 2.0, 3.0).normalize(),
                    frame_parent: Transform::new(Vec3::Y, Quat::from_rotation_y(0.4)),
                    frame_child: Transform::new(
                        Vec3::new(0.0, 0.5, 0.0),
                        Quat::from_rotation_x(-0.2),
                    ),
                })
                .unwrap();
            scene
                .add_body_wrench(
                    child,
                    SpatialVector::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.1, -0.2, 0.3)),
                )
                .unwrap();
            (scene, [parent, child], joint)
        };
        let (scene, cpu_bodies, cpu_joint) = create();
        let mut cpu = RigidWorld::from_scene(scene).unwrap();
        let (scene, gpu_bodies, gpu_joint) = create();
        let mut gpu = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        let mut worst_position: f64 = 0.0;
        let mut worst_rotation: f64 = 0.0;
        let mut worst_velocity: f64 = 0.0;
        for step in 0..120 {
            if kind != JointType::Ball && step % 30 == 0 {
                let motor = JointMotor {
                    mode: JointTargetMode::Velocity,
                    target_velocity: if step < 60 { 0.5 } else { -0.5 },
                    damping: 2.0,
                    max_force: 3.0,
                    ..Default::default()
                };
                let axis = if kind == JointType::Prismatic {
                    JointAxis::LinX
                } else {
                    JointAxis::AngX
                };
                let stamp = gpu.scene.model_stamp();
                cpu.scene.set_joint_motor(cpu_joint, axis, motor).unwrap();
                gpu.scene.set_joint_motor(gpu_joint, axis, motor).unwrap();
                assert_eq!(stamp, gpu.scene.model_stamp());
            }
            cpu.step(1.0 / 120.0).unwrap();
            gpu.step(1.0 / 120.0).unwrap();
            for (&a, &b) in cpu_bodies.iter().zip(&gpu_bodies) {
                let a_pose = cpu.scene.body_pose(a).unwrap();
                let b_pose = gpu.scene.body_pose(b).unwrap();
                if step == 0 {
                    eprintln!("{kind:?} initial corrected CPU={a_pose:?} GPU={b_pose:?}");
                }
                worst_position = worst_position.max((a_pose.position - b_pose.position).length());
                worst_rotation = worst_rotation.max(
                    a_pose
                        .rotation
                        .angle_between(b_pose.rotation.normalize())
                        .abs(),
                );
                let a_v = cpu.scene.body_velocity(a).unwrap();
                let b_v = gpu.scene.body_velocity(b).unwrap();
                worst_velocity = worst_velocity
                    .max((a_v.linear - b_v.linear).length())
                    .max((a_v.angular - b_v.angular).length());
            }
        }
        eprintln!(
            "{kind:?}: max position={worst_position:e} rotation={worst_rotation:e} velocity={worst_velocity:e}"
        );
        assert!(worst_position < 1e-3 && worst_rotation < 1e-3 && worst_velocity < 1e-3);
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_gpu_empty_scene_and_invalid_step_are_safe() {
    let engine = engine();
    let mut world = RigidWorld::from_scene_gpu(RigidScene::new().unwrap(), engine).unwrap();
    world.step(0.01).unwrap();
    for dt in [
        0.0,
        -0.1,
        f64::NAN,
        f64::INFINITY,
        f64::MAX,
        f64::MIN_POSITIVE,
    ] {
        assert!(world.step(dt).is_err());
    }
    world.step(0.01).unwrap();
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "requires GPU and wgpu resource counters"]
fn runtime_gpu_warm_resources_are_retained() {
    let engine = engine();
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
    let resources = || {
        engine
            .device()
            .unwrap()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let counters = engine.device().unwrap().get_internal_counters().hal;
        [
            counters.buffers.read(),
            counters.buffer_memory.read(),
            counters.bind_groups.read(),
            counters.compute_pipelines.read(),
            counters.shader_modules.read(),
        ]
    };
    for _ in 0..4 {
        world.step(0.01).unwrap();
    }
    let warm = resources();
    assert!(warm[0] > 0);
    let diagnostics = world.gpu_diagnostics().unwrap();
    assert_eq!(diagnostics.current_model_bind_groups_created, 15);
    assert_eq!(diagnostics.model_compilations, 1);
    let stamp = world.scene.model_stamp();
    for _ in 0..20 {
        world
            .scene
            .set_body_velocity(body, SpatialVector::new(Vec3::X, Vec3::ZERO))
            .unwrap();
        world.step(0.01).unwrap();
        assert_eq!(stamp, world.scene.model_stamp());
        assert_eq!(warm, resources());
        let current = world.gpu_diagnostics().unwrap();
        assert_eq!(current.model_compilations, diagnostics.model_compilations);
        assert_eq!(
            current.current_model_bind_groups_created,
            diagnostics.current_model_bind_groups_created
        );
    }
    eprintln!("warm runtime GPU [buffers, bytes, groups, pipelines, shaders]={warm:?}");
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_gpu_fault_is_atomic_and_corrected_inputs_can_resume() {
    let engine = engine();
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::Prismatic,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    let before = world.scene.body_pose(body).unwrap();
    world
        .scene
        .set_joint_motor(
            joint,
            JointAxis::LinX,
            JointMotor {
                mode: JointTargetMode::Velocity,
                target_velocity: 1.0,
                damping: 1e100,
                ..Default::default()
            },
        )
        .unwrap();
    let error = world.step(0.01).unwrap_err();
    assert!(error.to_string().contains("solver fault 32"), "{error}");
    assert_eq!(before, world.scene.body_pose(body).unwrap());
    assert_eq!(
        world.scene.body_velocity(body).unwrap(),
        SpatialVector::ZERO
    );
    assert_eq!(world.gpu_diagnostics().unwrap().completed_frames, 0);
    world
        .scene
        .set_joint_motor(
            joint,
            JointAxis::LinX,
            JointMotor {
                mode: JointTargetMode::Velocity,
                target_velocity: 1.0,
                damping: 2.0,
                ..Default::default()
            },
        )
        .unwrap();
    world.step(0.01).unwrap();
    assert!(world.scene.body_velocity(body).unwrap().linear.x > 0.0);
    assert_eq!(world.gpu_diagnostics().unwrap().completed_frames, 1);
}
