use super::*;
use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor};
use molla_sim::runtime::{BodyDesc, BodyKind, JointDesc};
use molla_sim::{BodyParams, JointType, MassProperties};

#[test]
fn paired_runtime_rejects_unsupported_scene_before_gpu_work() {
    let mut scene = RigidScene::new().unwrap();
    assert!(validate_paired(&scene, false).is_err());
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    assert!(validate_paired(&scene, false).is_ok());
    assert!(validate_paired(&scene, true).is_err());
    scene.set_body_kind(body, BodyKind::Kinematic).unwrap();
    assert!(validate_paired(&scene, false).is_err());
}

#[test]
#[ignore = "requires real GPU"]
fn paired_runtime_commits_sub_ulp_force_and_host_edits() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new()
                .with_mass(1e6)
                .with_inertia(Mat3::IDENTITY),
            ..Default::default()
        })
        .unwrap();
    let initial = 1. + 2f64.powi(-30);
    scene
        .set_body_velocity(body, SpatialVector::linear(Vec3::X * initial))
        .unwrap();
    eprintln!("paired runtime construction begin");
    let mut world = RigidWorld::from_scene_gpu_with_paired_machine_options(
        scene,
        engine,
        GpuContactCapacity::default(),
        crate::kernels::FeatherstoneRounding::Storage,
        crate::kernels::FeatherstoneExecution::Generic,
    )
    .unwrap();
    eprintln!("paired runtime construction complete");
    let mut expected = initial;
    for frame in 0..4 {
        world.solver.substeps = if frame % 2 == 0 { 1 } else { 2 };
        world.scene.state_mut().body_f.host_mut().unwrap()[0] = SpatialVector::linear(Vec3::X);
        world.step(0.001).unwrap();
        expected += 1e-6f32 as f64 * 0.001f32 as f64;
        let observed = world.scene.state().joint_qd.host().unwrap()[0];
        assert!(
            (observed - expected).abs() < 2e-13,
            "frame {frame}: {observed} != {expected}"
        );
        assert!((world.scene.state().body_qd.host().unwrap()[0].linear.x - expected).abs() < 2e-13);
    }
    world
        .scene
        .set_body_velocity(body, SpatialVector::linear(Vec3::X * (2. + 2f64.powi(-28))))
        .unwrap();
    world.scene.state_mut().body_f.host_mut().unwrap()[0] = SpatialVector::ZERO;
    world.step(0.001).unwrap();
    assert!(
        (world.scene.state().joint_qd.host().unwrap()[0] - (2. + 2f64.powi(-28))).abs() < 2e-13
    );
    world
        .scene
        .set_body_mass_properties(
            body,
            MassProperties {
                mass: 2e6,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            },
        )
        .unwrap();
    world.step(0.001).unwrap();
    assert!(
        (world.scene.state().joint_qd.host().unwrap()[0] - (2. + 2f64.powi(-28))).abs() < 2e-13
    );
}

#[test]
#[ignore = "requires real GPU"]
fn paired_runtime_transmits_child_force_through_fixed_joint() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut bodies = Vec::new();
    for mass in [2e6, 3e6] {
        bodies.push(
            scene
                .insert_body(BodyDesc {
                    params: BodyParams::new()
                        .with_mass(mass)
                        .with_inertia(Mat3::IDENTITY * 1e6),
                    ..Default::default()
                })
                .unwrap(),
        );
    }
    scene
        .insert_joint(JointDesc {
            parent: Some(bodies[0]),
            child: bodies[1],
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(Vec3::X),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let initial = 1. + 2f64.powi(-30);
    for body in bodies {
        scene
            .set_body_velocity(body, SpatialVector::linear(Vec3::X * initial))
            .unwrap();
    }
    eprintln!("paired articulated runtime construction begin");
    let mut world = RigidWorld::from_scene_gpu_with_paired_machine_options(
        scene,
        engine,
        GpuContactCapacity::default(),
        crate::kernels::FeatherstoneRounding::Storage,
        crate::kernels::FeatherstoneExecution::Generic,
    )
    .unwrap();
    eprintln!("paired articulated runtime construction complete");
    let mut expected = initial;
    for frame in 0..4 {
        world.solver.substeps = 1 + frame % 2;
        world.scene.state_mut().body_f.host_mut().unwrap()[1] = SpatialVector::linear(Vec3::X);
        world.step(0.001).unwrap();
        expected += 0.001f32 as f64 / 5e6;
        for (body, velocity) in world
            .scene
            .state()
            .body_qd
            .host()
            .unwrap()
            .iter()
            .enumerate()
        {
            assert!(
                (velocity.linear.x - expected).abs() < 2e-13,
                "frame {frame}, body {body}: {} != {expected}",
                velocity.linear.x
            );
            assert!(velocity.linear.y.abs().max(velocity.linear.z.abs()) < 2e-13);
            assert!(velocity.angular.length() < 2e-13);
        }
    }
    assert_eq!(world.gpu_diagnostics().unwrap().completed_frames, 4);
}

fn device_words(engine: &Engine, buffers: &[&wgpu::Buffer]) -> Vec<Vec<u32>> {
    let device = engine.device().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paired runtime checkpoint observation"),
        size: buffers.iter().map(|b| b.size()).sum(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut offset = 0;
    for buffer in buffers {
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, offset, buffer.size());
        offset += buffer.size();
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .unwrap()
        .unwrap();
    let data = staging.slice(..).get_mapped_range();
    let mut offset = 0;
    let result = buffers
        .iter()
        .map(|buffer| {
            let end = offset + buffer.size() as usize;
            let words = bytemuck::cast_slice(&data[offset..end]).to_vec();
            offset = end;
            words
        })
        .collect();
    drop(data);
    staging.unmap();
    result
}

fn retained_parts(world: &RigidWorld, side: usize) -> Vec<Vec<u32>> {
    let gpu = world.gpu.as_ref().unwrap();
    let state = &gpu.compiled.as_ref().unwrap().states[side];
    device_words(
        &gpu.engine,
        &[
            world.solver.gpu_paired_rates_buffer(state).unwrap(),
            world
                .solver
                .gpu_body_velocity_compensation_buffer(state)
                .unwrap(),
            world
                .solver
                .gpu_coordinate_compensation_buffer(state)
                .unwrap(),
            world
                .solver
                .gpu_position_compensation_buffer(state)
                .unwrap(),
            world.solver.gpu_paired_velocity_delta_buffer().unwrap(),
        ],
    )
}

fn scene_momentum(scene: &RigidScene) -> SpatialVector {
    let mut total = SpatialVector::ZERO;
    for (body, _) in scene.bodies() {
        let mass = scene.body_mass_properties(body).unwrap();
        let rotation = Mat3::from_quat(scene.body_pose(body).unwrap().rotation);
        let velocity = scene.body_velocity(body).unwrap();
        let linear = velocity.linear * mass.mass;
        total.linear += linear;
        total.angular += rotation * mass.inertia * rotation.transpose() * velocity.angular
            + scene.body_world_com(body).unwrap().cross(linear);
    }
    total
}

#[test]
#[ignore = "requires real GPU"]
fn paired_six_body_tractor_drives_and_rolls_back() {
    use crate::soil_tractor::build_soil_tractor_scene;
    let engine = Arc::new(Engine::new_gpu().unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let configure = || {
        let (mut scene, handles) = build_soil_tractor_scene().unwrap();
        for axle in handles.axles {
            scene
                .set_joint_motor(
                    axle,
                    JointAxis::AngX,
                    JointMotor {
                        mode: molla_sim::JointTargetMode::Effort,
                        target_position: -300.,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        let tine = scene.body_index(handles.tine).unwrap().index();
        scene.state_mut().body_f.host_mut().unwrap()[tine] =
            SpatialVector::new(Vec3::new(12., -5., 3.), Vec3::new(1., 2., 3.));
        (scene, handles)
    };
    let (scene, handles) = configure();
    assert_eq!(scene.bodies().count(), 6);
    let mut reference = RigidWorld::from_scene(configure().0).unwrap();
    eprintln!("paired six-body construction begin");
    let mut world = RigidWorld::from_scene_gpu_with_paired_machine_options(
        scene,
        engine,
        GpuContactCapacity::default(),
        crate::kernels::FeatherstoneRounding::Storage,
        crate::kernels::FeatherstoneExecution::Generic,
    )
    .unwrap();
    eprintln!("paired six-body construction complete");
    let dt = 0.0003;
    for frame in 0..4 {
        let substeps = if frame % 2 == 0 { 3 } else { 2 };
        world.solver.substeps = substeps;
        reference.solver.substeps = substeps;
        if frame == 2 {
            let before = world.scene.state().clone_host().unwrap();
            let diagnostics = world.gpu_diagnostics().unwrap();
            let retained = retained_parts(&world, diagnostics.last_substeps % 2);
            world.gpu.as_mut().unwrap().fail_after_substep = Some(1);
            let error = world.step(dt).unwrap_err().to_string();
            assert!(error.contains("injected post-substep failure"), "{error}");
            assert_eq!(
                world.gpu_diagnostics().unwrap().completed_frames,
                diagnostics.completed_frames
            );
            assert_eq!(
                world.scene.state().joint_q.host().unwrap(),
                before.joint_q.host().unwrap()
            );
            assert_eq!(
                world.scene.state().joint_qd.host().unwrap(),
                before.joint_qd.host().unwrap()
            );
            assert_eq!(
                world.scene.state().body_q.host().unwrap(),
                before.body_q.host().unwrap()
            );
            assert_eq!(
                world.scene.state().body_qd.host().unwrap(),
                before.body_qd.host().unwrap()
            );
            for side in 0..2 {
                assert_eq!(retained_parts(&world, side), retained);
            }
            eprintln!("paired six-body post-substep rollback verified");
        }
        world.step(dt).unwrap();
        reference.step(dt).unwrap();
        for (i, (&actual, &expected)) in world
            .scene
            .state()
            .joint_qd
            .host()
            .unwrap()
            .iter()
            .zip(reference.scene.state().joint_qd.host().unwrap())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 2e-5 * (1. + expected.abs()),
                "frame {frame}, DOF {i}: {actual} != {expected}"
            );
        }
        let actual = scene_momentum(&world.scene);
        let expected = scene_momentum(&reference.scene);
        let linear = (actual.linear - expected.linear).length();
        let angular = (actual.angular - expected.angular).length();
        eprintln!(
            "paired six-body frame={frame} linear_error={linear:e} angular_error={angular:e}"
        );
        assert!(linear < 2e-5, "linear {linear}");
        assert!(angular < 2e-5, "angular {angular}");
    }
    for wheel in handles.wheels {
        assert!(world.scene.body_velocity(wheel).unwrap().angular.z < -1e-4);
    }
    assert_eq!(world.gpu_diagnostics().unwrap().completed_frames, 4);
}
