use super::*;
use crate::mpm::replay_tests::read_words;
use crate::{FeatherstoneSolver, Solver};
use molla_core::gpu_metrics::{DeviceScope, TransferScope};
use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::BodyDesc;
use molla_sim::{Contacts, MassProperties};

fn scene(engine: &Engine) -> (RigidScene, [State; 2], Vec<MpmRigidDeviceBinding>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut handles = vec![];
    for i in 0..2 {
        let body = scene
            .insert_body(BodyDesc {
                additional_mass: Some(MassProperties {
                    mass: 10.0,
                    local_com: Vec3::new(0.0625, 0.0, 0.0),
                    inertia: Mat3::IDENTITY * 0.2,
                }),
                ..Default::default()
            })
            .unwrap();
        scene
            .set_body_pose(
                body,
                Transform {
                    position: Vec3::new(i as f64 * 3.0, 0.0, 0.0),
                    rotation: Quat::from_rotation_z(std::f64::consts::FRAC_PI_2),
                },
            )
            .unwrap();
        scene
            .set_body_velocity(
                body,
                SpatialVector::new(Vec3::new(0.05, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0)),
            )
            .unwrap();
        handles.push(body);
    }
    scene
        .upload_model_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    scene
        .upload_controls_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut states = [
        scene.state().clone_host().unwrap(),
        scene.state().clone_host().unwrap(),
    ];
    for state in &mut states {
        state
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
    }
    let bindings = handles
        .into_iter()
        .rev()
        .map(|body| MpmRigidDeviceBinding {
            body,
            shape: MpmSolidShape::Box {
                half_extents: [0.25, 0.25, 0.25],
            },
        })
        .collect();
    (scene, states, bindings)
}

fn soil(engine: &Engine) -> MpmSimulation {
    soil_layout(engine, false, 0)
}

fn soil_layout(engine: &Engine, native: bool, layout: u32) -> MpmSimulation {
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 16;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 8;
    cfg.max_rigid = 2;
    cfg.gravity_y = 0.0;
    cfg.youngs = 100.0;
    cfg.particle_mass = 0.25;
    cfg.particle_volume = 0.25 / 1600.0;
    let mut soil = match (native, layout) {
        (true, 2) => MpmSimulation::new_native_sparse(engine, cfg, 64),
        (false, 2) => MpmSimulation::new_sparse(engine, cfg, 64),
        (true, _) => MpmSimulation::new_native_explicit(engine, cfg),
        (false, _) => MpmSimulation::new(engine, cfg),
    }
    .unwrap();
    if layout == 1 {
        soil.set_active_grid(true).unwrap();
    }
    soil.set_particles(
        engine.queue().unwrap(),
        &[[0.0, 0.0625, 0.0], [0.0625, 0.0625, 0.0625]],
    )
    .unwrap();
    soil
}

#[test]
#[ignore = "requires a real GPU with native atomics"]
fn active_kinematics_only_bridge_preserves_soil_and_forces() {
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let read = |buffer: &wgpu::Buffer| read_words(device, queue, buffer);
    for native in [false, true] {
        for layout in 0..3 {
            let (scene, mut states, bindings) = scene(&engine);
            states[1].body_q.host_mut().unwrap()[0].position.x += 0.25;
            states[1].body_q.upload(queue).unwrap();
            for (i, state) in states.iter_mut().enumerate() {
                for (j, force) in state.body_f.host_mut().unwrap().iter_mut().enumerate() {
                    *force = SpatialVector::new(
                        Vec3::splat(1.0 + i as f64),
                        Vec3::splat(3.0 + j as f64),
                    );
                }
                state.body_f.upload(queue).unwrap();
            }
            let mut soil = soil_layout(&engine, native, layout);
            let allocations = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            assert!(
                PreparedMpmRigidDeviceBridge::new_kinematics(
                    engine.clone(),
                    &mut soil,
                    &scene,
                    &states,
                    &bindings,
                    111
                )
                .is_err()
            );
            assert!(soil.rigid_bridge.upgrade().is_none());
            if layout != 0 {
                assert!(
                    PreparedMpmRigidDeviceBridge::new(
                        engine.clone(),
                        &mut soil,
                        &scene,
                        &states,
                        &bindings,
                        4096
                    )
                    .is_err()
                );
            }
            assert_eq!(allocations.finish().observations, 0);
            assert_eq!(transfers.finish().upload_bytes, 0);
            let mut bridge = PreparedMpmRigidDeviceBridge::new_kinematics(
                engine.clone(),
                &mut soil,
                &scene,
                &states,
                &bindings,
                112,
            )
            .unwrap();
            assert_eq!(bridge.buffer_bytes(), 112);
            assert!(bridge.saved.is_none());
            assert!(
                PreparedMpmRigidDeviceBridge::new_kinematics(
                    engine.clone(),
                    &mut soil,
                    &scene,
                    &states,
                    &bindings,
                    112
                )
                .is_err()
            );
            let particles: Vec<_> = soil.checkpoint_buffers().into_iter().map(read).collect();
            let forces = states
                .each_ref()
                .map(|s| read(s.body_f.device_buffer_current().unwrap()));
            let wrench = vec![7.0_f32; soil.rigid_wrench.size() as usize / 4];
            queue.write_buffer(&soil.rigid_wrench, 0, bytemuck::cast_slice(&wrench));
            for current in [0, 1] {
                let allocations = DeviceScope::start().unwrap();
                let transfers = TransferScope::start().unwrap();
                let mut encoder = device.create_command_encoder(&Default::default());
                bridge
                    .encode_kinematics(&mut encoder, &soil, &scene, &states, current)
                    .unwrap();
                queue.submit([encoder.finish()]);
                assert!(
                    bridge
                        .step_soil_and_apply(&mut soil, &scene, &mut states, current, 0.0001)
                        .is_err()
                );
                assert!(bridge.restore_forces(&mut states).is_err());
                assert!(!bridge.failed && !bridge.pending);
                assert_eq!(allocations.finish().observations, 0);
                let transfers = transfers.finish();
                assert_eq!(transfers.upload_bytes, 0);
                assert_eq!(transfers.readback_bytes, 0);
                let centers: Vec<_> = read(&soil.rigid_center)
                    .into_iter()
                    .map(f32::from_bits)
                    .collect();
                assert!((centers[4] - current as f32 * 0.25).abs() < 1e-6);
                assert!((centers[5] - 0.0625).abs() < 1e-6);
                assert!((centers[0] - 3.0).abs() < 1e-6);
                let motion: Vec<_> = read(&soil.rigid_vel)
                    .into_iter()
                    .map(f32::from_bits)
                    .collect();
                for offset in [0, 16] {
                    assert!((motion[offset] - 0.05).abs() < 1e-6);
                    assert!((motion[offset + 5] - 0.1).abs() < 1e-6);
                    assert!((motion[offset + 10] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
                    assert!((motion[offset + 11] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
                }
                assert_eq!(
                    read(&soil.rigid_wrench),
                    wrench.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(
                    soil.checkpoint_buffers()
                        .into_iter()
                        .map(read)
                        .collect::<Vec<_>>(),
                    particles
                );
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            assert!(
                bridge
                    .encode_kinematics(&mut encoder, &soil, &scene, &states, 2)
                    .is_err()
            );
            soil.set_prescribed_solids(queue, &[]).unwrap();
            assert!(
                bridge
                    .encode_kinematics(&mut encoder, &soil, &scene, &states, 0)
                    .is_err()
            );
            drop(bridge);
            assert!(soil.rigid_bridge.upgrade().is_none());
            for (state, expected) in states.iter().zip(&forces) {
                assert_eq!(
                    read(state.body_f.device_buffer_current().unwrap()),
                    *expected
                );
            }
            println!(
                "kinematics-only layout={layout} native={native}: 112 bytes, resident COM/motion, exact particles/wrench/forces, rejection and lease guards pass"
            );
        }
    }
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn kinematics_only_does_not_advance_soil_or_apply_reactions() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (scene, mut states, bindings) = scene(&engine);
    states[1].body_q.host_mut().unwrap()[0].position.x += 0.25;
    states[1].body_q.upload(queue).unwrap();
    let mut soil = soil(&engine);
    let mut bridge = PreparedMpmRigidDeviceBridge::new(
        engine.clone(),
        &mut soil,
        &scene,
        &states,
        &bindings,
        4096,
    )
    .unwrap();
    let read = |buffer| read_words(device, queue, buffer);
    let particles: Vec<_> = soil.checkpoint_buffers().into_iter().map(read).collect();
    let forces = states
        .each_ref()
        .map(|s| read(s.body_f.device_buffer_current().unwrap()));
    for current in [0, 1] {
        let allocations = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        bridge
            .encode_kinematics(&mut encoder, &soil, &scene, &states, current)
            .unwrap();
        queue.submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let transfers = transfers.finish();
        assert_eq!(transfers.readback_bytes, 0);
        assert_eq!(transfers.upload_bytes, 0);
        let centers: Vec<_> = read(&soil.rigid_center)
            .into_iter()
            .map(f32::from_bits)
            .collect();
        assert!((centers[4] - current as f32 * 0.25).abs() < 1e-6);
        assert!((centers[5] - 0.0625).abs() < 1e-6);
        assert!((centers[0] - 3.0).abs() < 1e-6);
        for (state, expected) in states.iter().zip(&forces) {
            assert_eq!(
                read(state.body_f.device_buffer_current().unwrap()),
                *expected
            );
        }
        let after: Vec<_> = soil.checkpoint_buffers().into_iter().map(read).collect();
        assert_eq!(after, particles);
        assert!(!bridge.pending);
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(
        bridge
            .encode_kinematics(&mut encoder, &soil, &scene, &states, 2)
            .is_err()
    );
    bridge
        .step_soil_and_apply(&mut soil, &scene, &mut states, 0, 0.0001)
        .unwrap();
    assert!(
        bridge
            .encode_kinematics(&mut encoder, &soil, &scene, &states, 0)
            .is_err()
    );
    bridge.restore_forces(&mut states).unwrap();
    soil.set_prescribed_solids(queue, &[]).unwrap();
    assert!(
        bridge
            .encode_kinematics(&mut encoder, &soil, &scene, &states, 0)
            .is_err()
    );
    println!(
        "kinematics-only bridge: both resident states, rotated COM, unchanged soil/forces and stale/pending guards pass"
    );
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn device_bridge_feeds_actual_gpu_machine_and_restores_external_forces() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (scene, mut states, bindings) = scene(&engine);
    println!("device bridge fixture: scene ready");
    let mut soil = soil(&engine);
    println!("device bridge fixture: soil ready");
    let mut solver = FeatherstoneSolver::new_gpu(&engine, scene.model())
        .unwrap()
        .with_substeps(1)
        .with_damping(0.0)
        .with_rigorous_d_solve(true);
    println!("device bridge fixture: Featherstone ready");
    let mut bridge = PreparedMpmRigidDeviceBridge::new(
        engine.clone(),
        &mut soil,
        &scene,
        &states,
        &bindings,
        4096,
    )
    .unwrap();
    assert_eq!(bridge.buffer_bytes(), 48 * 2 + 16 + 2 * 2 * 32);
    let force_before = states.each_ref().map(|s| {
        read_words(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            s.body_f.device_buffer_current().unwrap(),
        )
    });
    let initial_velocity = scene.body_velocity(bindings[1].body).unwrap();
    let dt = 0.0001;
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    bridge
        .step_soil_and_apply(&mut soil, &scene, &mut states, 0, dt)
        .unwrap();
    assert_eq!(allocations.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!(transfers.readback_bytes, 0);
    println!(
        "resident soil/rigid bridge: 0 buffer creations/readback bytes, {} ordinary soil parameter upload bytes",
        transfers.upload_bytes
    );
    assert!(
        bridge
            .step_soil_and_apply(&mut soil, &scene, &mut states, 0, dt)
            .is_err()
    );
    let [input, output] = &mut states;
    solver
        .step(
            scene.model(),
            input,
            scene.control(),
            &Contacts::with_capacity(0),
            dt as f64,
            output,
        )
        .unwrap();
    let reactions = soil.read_rigid_wrenches(&engine).unwrap();
    assert!(reactions[1].0.iter().any(|f| f.abs() > 1e-4));
    assert_eq!(reactions[0], ([0.0; 3], [0.0; 3]));
    let center_words = read_words(
        engine.device().unwrap(),
        engine.queue().unwrap(),
        &soil.rigid_center,
    );
    for (slot, binding) in bindings.iter().enumerate() {
        let expected = scene.body_world_com(binding.body).unwrap();
        for axis in 0..3 {
            assert!(
                (f32::from_bits(center_words[4 * slot + axis]) as f64 - expected[axis]).abs()
                    < 1e-6
            );
        }
    }
    states[1]
        .body_qd
        .download(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let body = scene.body_index(bindings[1].body).unwrap().index();
    let actual = states[1].body_qd.host().unwrap()[body];
    assert!(
        (actual.linear - initial_velocity.linear).length() > 1e-4,
        "machine must actually consume the soil reaction"
    );
    assert_eq!(
        read_words(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            solver.gpu_status_buffer().unwrap()
        )[0],
        0
    );
    for axis in 0..3 {
        let expected =
            initial_velocity.linear[axis] + reactions[1].0[axis] as f64 * dt as f64 / 10.0;
        assert!(
            (actual.linear[axis] - expected).abs() <= 2e-5,
            "COM velocity {axis}: {} != {expected}",
            actual.linear[axis]
        );
    }
    let transfers = TransferScope::start().unwrap();
    bridge.restore_forces(&mut states).unwrap();
    assert_eq!(transfers.finish().readback_bytes, 0);
    for (state, original) in states.iter().zip(&force_before) {
        assert_eq!(
            &read_words(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                state.body_f.device_buffer_current().unwrap()
            ),
            original
        );
    }
    assert!(bridge.restore_forces(&mut states).is_err());
    bridge
        .step_soil_and_apply(&mut soil, &scene, &mut states, 1, dt)
        .unwrap();
    drop(bridge);
    for (state, original) in states.iter().zip(&force_before) {
        assert_eq!(
            &read_words(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                state.body_f.device_buffer_current().unwrap()
            ),
            original
        );
    }
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn device_bridge_rejects_stale_handles_foreign_state_and_host_override() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (mut scene, mut states, bindings) = scene(&engine);
    for (i, state) in states.iter_mut().enumerate() {
        state.body_f.host_mut().unwrap()[0] = SpatialVector::new(
            Vec3::new(1.0 + i as f64, -2.0, 3.0),
            Vec3::new(0.25, -0.5, 0.75),
        );
        state.body_f.upload(engine.queue().unwrap()).unwrap();
    }
    let originals = states.each_ref().map(|state| {
        read_words(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            state.body_f.device_buffer_current().unwrap(),
        )
    });
    let mut soil = soil(&engine);
    let allocations = DeviceScope::start().unwrap();
    assert!(
        PreparedMpmRigidDeviceBridge::new(
            engine.clone(),
            &mut soil,
            &scene,
            &states,
            &[bindings[0], bindings[0]],
            4096
        )
        .is_err()
    );
    assert!(
        PreparedMpmRigidDeviceBridge::new(engine.clone(), &mut soil, &scene, &states, &bindings, 1)
            .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    let foreign = Arc::new(Engine::new_gpu().unwrap());
    let (_, foreign_states, _) = self::scene(&foreign);
    assert!(
        PreparedMpmRigidDeviceBridge::new(
            engine.clone(),
            &mut soil,
            &scene,
            &foreign_states,
            &bindings,
            4096
        )
        .is_err()
    );
    let mut bridge = PreparedMpmRigidDeviceBridge::new(
        engine.clone(),
        &mut soil,
        &scene,
        &states,
        &bindings,
        4096,
    )
    .unwrap();
    let allocations = DeviceScope::start().unwrap();
    assert!(
        PreparedMpmRigidDeviceBridge::new(
            engine.clone(),
            &mut soil,
            &scene,
            &states,
            &bindings,
            4096
        )
        .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    soil.set_active_grid(true).unwrap();
    let transfers = TransferScope::start().unwrap();
    assert!(
        bridge
            .step_soil_and_apply(&mut soil, &scene, &mut states, 0, 0.0001)
            .is_err()
    );
    assert_eq!(transfers.finish().upload_bytes, 0);
    soil.set_active_grid(false).unwrap();
    soil.set_rigid_spheres(engine.queue().unwrap(), &[])
        .unwrap();
    let transfers = TransferScope::start().unwrap();
    assert!(
        bridge
            .step_soil_and_apply(&mut soil, &scene, &mut states, 0, 0.0001)
            .is_err()
    );
    assert_eq!(transfers.finish().upload_bytes, 0);
    drop(bridge);
    let mut bridge = PreparedMpmRigidDeviceBridge::new(
        engine.clone(),
        &mut soil,
        &scene,
        &states,
        &bindings,
        4096,
    )
    .unwrap();
    bridge
        .step_soil_and_apply(&mut soil, &scene, &mut states, 0, 0.0001)
        .unwrap();
    scene.remove_body(bindings[1].body).unwrap();
    bridge.restore_forces(&mut states).unwrap();
    for (state, original) in states.iter().zip(&originals) {
        assert_eq!(
            &read_words(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                state.body_f.device_buffer_current().unwrap()
            ),
            original
        );
    }
    let transfers = TransferScope::start().unwrap();
    assert!(
        bridge
            .step_soil_and_apply(&mut soil, &scene, &mut states, 1, 0.0001)
            .is_err()
    );
    assert_eq!(transfers.finish().upload_bytes, 0);
}
