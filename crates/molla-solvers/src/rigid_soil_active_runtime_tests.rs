use super::*;
use crate::mpm::{MpmConfig, MpmSolidShape};
use molla_core::gpu_metrics::{DeviceScope, TransferScope};
use molla_math::{Mat3, SpatialVector, Vec3};
use molla_sim::MassProperties;
use molla_sim::runtime::BodyDesc;

fn read_status(engine: &Engine, buffer: &wgpu::Buffer) -> Vec<u32> {
    bytemuck::cast_slice(&crate::soil_shared_wall_friction_gpu::capture::read(
        engine, buffer,
    ))
    .to_vec()
}

fn soil(engine: &Engine, native: bool, compact: bool, capacity: u32) -> MpmSimulation {
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 16;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 8;
    cfg.max_rigid = 1;
    cfg.gravity_y = 0.0;
    cfg.youngs = 100.0;
    cfg.particle_mass = 0.25;
    cfg.particle_volume = 0.25 / 1600.0;
    let mut soil = match (native, compact) {
        (true, true) => MpmSimulation::new_native_sparse(engine, cfg, capacity),
        (false, true) => MpmSimulation::new_sparse(engine, cfg, capacity),
        (true, false) => MpmSimulation::new_native_explicit(engine, cfg),
        (false, false) => MpmSimulation::new(engine, cfg),
    }
    .unwrap();
    soil.set_active_grid(true).unwrap();
    soil.set_particles_with_velocity(
        engine.queue().unwrap(),
        &[[0.25, 0.0, 0.0], [0.3125, 0.0, 0.0]],
        &[[1.0, 0.0, 0.0]; 2],
    )
    .unwrap();
    soil
}

fn state(engine: &Engine, scene: &RigidScene) -> [State; 2] {
    std::array::from_fn(|_| {
        let mut state = scene.state().clone_host().unwrap();
        state
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        state
    })
}

#[test]
#[ignore = "requires real native GPU; direct no-projection machine fault lifecycle"]
fn active_runtime_direct_machine_fault_lifecycle() {
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut scene = RigidScene::new().unwrap();
    scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 10.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let mut states = state(&engine, &scene);
    for native in [false, true] {
        let soil = soil(&engine, native, true, 1);
        let status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("machine fault test status"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let guard = crate::mpm::MpmMachineFaultBridge::new(&soil, &status).unwrap();
        let backup = MpmFrameBackup::new(engine.clone(), &soil, 1_000_000).unwrap();
        let mut runtime = RuntimeSoil {
            soil,
            bindings: vec![],
            bridge: None,
            projection: None,
            machine_fault: Some(guard),
            projection_config: None,
            prepared_stamp: None,
            published: None,
            substep_dt: 0.0,
            completion_timeout: std::time::Duration::from_secs(5),
            backup,
            max_bytes: 1_000_000,
            applied: false,
            stats: Default::default(),
            profile_enabled: false,
            profiling_current: false,
            profile_frame: None,
            profile_host_ms: HOST_STAGES,
            profile_completion_start: None,
            machine_profiler: None,
            loads: None,
            fail_after_steps: None,
            contact_trace: None,
            force_separate_submissions: false,
        };
        let before = runtime.soil.checkpoint_soil_with(queue).unwrap().to_bytes();
        runtime.begin(&engine, 0.0001).unwrap();
        runtime.soil.step_with(device, queue, 0.0001).unwrap();
        assert!(
            runtime
                .soil
                .read_active_grid_status_with(device, queue)
                .unwrap()
                .overflow
        );
        queue.write_buffer(&status, 0, bytemuck::cast_slice(&[4u32]));
        let allocations = DeviceScope::start().unwrap();
        let traffic = TransferScope::start().unwrap();
        assert!(
            !runtime
                .after_prediction(&engine, &mut states, 0, 0.0001)
                .unwrap()
        );
        assert_eq!(allocations.finish().observations, 0);
        let traffic = traffic.finish();
        assert_eq!(traffic.upload_bytes + traffic.readback_bytes, 0);
        assert_eq!(read_status(&engine, &status), [36]);
        runtime.rollback(Some(&mut states)).unwrap();
        assert!(runtime.machine_fault.is_none());
        assert_eq!(
            runtime.soil.checkpoint_soil_with(queue).unwrap().to_bytes(),
            before
        );
        let stale = crate::mpm::MpmMachineFaultBridge::new(&runtime.soil, &status).unwrap();
        runtime.soil.reserve_sparse_tiles(64).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        assert!(stale.encode(&mut encoder, &runtime.soil).is_err());
        runtime.machine_fault =
            Some(crate::mpm::MpmMachineFaultBridge::new(&runtime.soil, &status).unwrap());
        queue.write_buffer(&status, 0, bytemuck::cast_slice(&[4u32]));
        assert!(
            !runtime
                .after_prediction(&engine, &mut states, 0, 0.0001)
                .unwrap()
        );
        assert_eq!(read_status(&engine, &status), [4]);
        runtime.soil.step_with(device, queue, 0.0001).unwrap();
        assert!(
            !runtime
                .soil
                .read_active_grid_status_with(device, queue)
                .unwrap()
                .faulted()
        );
        assert!(
            !runtime
                .after_prediction(&engine, &mut states, 0, 0.0001)
                .unwrap()
        );
        assert_eq!(read_status(&engine, &status), [4]);
        println!(
            "direct machine fault native={native}: empty bindings, status OR, exact soil rollback, stale growth, reset and healthy retry pass"
        );
    }
}

fn checkpoint(world: &mut RigidWorld, engine: &Engine) -> Vec<u8> {
    world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .soil
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap()
        .to_bytes()
}

#[test]
#[ignore = "requires real native GPU; actual active/compact projected RigidWorld transactions"]
fn active_runtime_attachment_transactions() {
    attachment_transactions(crate::kernels::FeatherstoneExecution::Generic);
}

#[test]
#[ignore = "requires real native GPU; opt-in joint-type machine attachment transactions"]
fn active_runtime_joint_types_attachment_transactions() {
    attachment_transactions(crate::kernels::FeatherstoneExecution::ExperimentalJointTypes);
}

fn attachment_transactions(execution: crate::kernels::FeatherstoneExecution) {
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            velocity: SpatialVector::linear(Vec3::X * 0.05),
            additional_mass: Some(MassProperties {
                mass: 10.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let started = std::time::Instant::now();
    eprintln!("active runtime: constructing real GPU machine execution={execution:?}");
    let mut world = RigidWorld::from_scene_gpu_with_machine_options(
        scene,
        engine.clone(),
        GpuContactCapacity::default(),
        crate::kernels::FeatherstoneRounding::Storage,
        execution,
    )
    .unwrap();
    eprintln!(
        "active runtime: machine ready execution={execution:?} construction_s={:.6}",
        started.elapsed().as_secs_f64()
    );
    world.solver.substeps = 2;
    let bindings = || {
        vec![MpmRigidDeviceBinding {
            body,
            shape: MpmSolidShape::Box {
                half_extents: [0.25; 3],
            },
        }]
    };
    for native in [false, true] {
        for compact in [false, true] {
            world
                .attach_mpm_soil(soil(&engine, native, compact, 64), bindings(), 2_000_000)
                .unwrap();
            let before = checkpoint(&mut world, &engine);
            let pose = world.scene.body_pose(body).unwrap();
            let velocity = world.scene.body_velocity(body).unwrap();
            let diagnostics = world.gpu_diagnostics().unwrap();
            let transfers = TransferScope::start().unwrap();
            assert!(
                world
                    .step(0.0002)
                    .unwrap_err()
                    .to_string()
                    .contains("requires projected coupling")
            );
            assert_eq!(transfers.finish().upload_bytes, 0);
            assert_eq!(
                world.gpu_diagnostics().unwrap().completed_frames,
                diagnostics.completed_frames
            );
            assert_eq!(world.scene.body_pose(body).unwrap(), pose);
            assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
            assert_eq!(checkpoint(&mut world, &engine), before);
            assert!(world.enable_mpm_soil_mass_coupling(0).is_err());
            let runtime = world.gpu.as_mut().unwrap().soil.as_mut().unwrap();
            let budget = runtime.max_bytes;
            runtime.max_bytes = runtime.stats.retained_bytes;
            assert!(world.enable_mpm_soil_mass_coupling(128).is_err());
            let runtime = world.gpu.as_mut().unwrap().soil.as_mut().unwrap();
            assert!(runtime.projection_config.is_none());
            assert!(runtime.projection.is_none());
            runtime.max_bytes = budget;
            assert_eq!(checkpoint(&mut world, &engine), before);
            world.enable_mpm_soil_mass_coupling(128).unwrap();
            world.step(0.0002).unwrap();
            assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 1);
            let coupled_velocity = world.scene.body_velocity(body).unwrap().linear;
            assert!(
                coupled_velocity.x > velocity.linear.x + 1e-5,
                "soil must transfer a nonzero forward impulse to the actual body"
            );
            let particle_velocities = world
                .gpu
                .as_mut()
                .unwrap()
                .soil
                .as_mut()
                .unwrap()
                .soil
                .read_velocities_with(engine.device().unwrap(), engine.queue().unwrap())
                .unwrap();
            let momentum = world.scene.body_velocity(body).unwrap().linear * 10.0
                + particle_velocities
                    .iter()
                    .map(|v| Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64) * 0.25)
                    .sum::<Vec3>();
            let momentum_error = (momentum - (velocity.linear * 10.0 + Vec3::X * 0.5)).length();
            assert!(momentum_error < 2e-5);
            eprintln!(
                "actual response native={native} compact={compact}: body_delta_vx={:.9} total_momentum_error={momentum_error:.9}",
                coupled_velocity.x - velocity.linear.x
            );
            let before = checkpoint(&mut world, &engine);
            let pose = world.scene.body_pose(body).unwrap();
            let velocity = world.scene.body_velocity(body).unwrap();
            world
                .gpu
                .as_mut()
                .unwrap()
                .soil
                .as_mut()
                .unwrap()
                .fail_after_steps = Some(1);
            assert!(
                world
                    .step(0.0002)
                    .unwrap_err()
                    .to_string()
                    .contains("injected failure")
            );
            assert_eq!(checkpoint(&mut world, &engine), before);
            assert_eq!(world.scene.body_pose(body).unwrap(), pose);
            assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
            assert_eq!(world.mpm_soil_stats().unwrap().rolled_back_frames, 1);
            world
                .gpu
                .as_mut()
                .unwrap()
                .soil
                .as_mut()
                .unwrap()
                .fail_after_steps = None;
            world.step(0.0002).unwrap();
            assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 2);
            assert!(!world.mpm_soil_stats().unwrap().faulted);
            world.detach_mpm_soil().unwrap().unwrap();
            println!(
                "actual machine native={native} compact={compact}: attach/configure, conservation, exact rollback/retry pass"
            );
        }
    }
    world
        .attach_mpm_soil(soil(&engine, true, true, 1), bindings(), 2_000_000)
        .unwrap();
    world.enable_mpm_soil_mass_coupling(128).unwrap();
    let before = checkpoint(&mut world, &engine);
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    let error = world.step(0.0002).unwrap_err();
    assert!(error.to_string().contains("fault"), "{error}");
    assert_eq!(checkpoint(&mut world, &engine), before);
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 0);
    assert_eq!(world.mpm_soil_stats().unwrap().rolled_back_frames, 1);
    let mut soil = world.detach_mpm_soil().unwrap().unwrap();
    soil.reserve_sparse_tiles(64).unwrap();
    world.attach_mpm_soil(soil, bindings(), 2_000_000).unwrap();
    world.enable_mpm_soil_mass_coupling(128).unwrap();
    world.step(0.0002).unwrap();
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 1);
    println!(
        "actual machine compact overflow: shared frame rejection, exact rollback and grown reattachment retry pass"
    );
}
