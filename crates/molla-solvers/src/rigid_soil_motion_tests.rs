use super::*;
use crate::mpm::{MpmConfig, MpmSolidShape};
use molla_math::{Mat3, SpatialVector, Vec3};
use molla_sim::{MassProperties, runtime::BodyDesc};

#[cfg(feature = "parallel-indices-newton")]
#[path = "rigid_soil_tractor_retry_tests.rs"]
mod tractor;

pub(super) fn retry_machine_words(world: &RigidWorld) -> Vec<Vec<u8>> {
    let gpu = world.gpu.as_ref().unwrap();
    let compiled = gpu.compiled.as_ref().unwrap();
    let state = &compiled.states[compiled.render_side];
    [
        state.body_q.device_buffer().unwrap(),
        state.body_qd.device_buffer().unwrap(),
        state.joint_q.device_buffer().unwrap(),
        state.joint_qd.device_buffer().unwrap(),
        state.body_f.device_buffer().unwrap(),
        world
            .solver
            .gpu_coordinate_compensation_buffer(state)
            .unwrap(),
        world
            .solver
            .gpu_position_compensation_buffer(state)
            .unwrap(),
        world.solver.gpu_paired_rates_buffer(state).unwrap(),
        world
            .solver
            .gpu_body_velocity_compensation_buffer(state)
            .unwrap(),
    ]
    .map(|buffer| crate::soil_shared_wall_friction_gpu::capture::read(&gpu.engine, buffer))
    .into()
}

fn save_retry_words(
    root: &std::path::Path,
    name: &str,
    world: &mut RigidWorld,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let soil = world.read_mpm_soil_checkpoint(2).unwrap().1.to_bytes();
    let machine = retry_machine_words(world);
    std::fs::write(root.join(format!("{name}-soil.bin")), &soil).unwrap();
    for (field, words) in machine.iter().enumerate() {
        std::fs::write(root.join(format!("{name}-machine-{field}.bin")), words).unwrap();
    }
    (soil, machine)
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; identical guarded direct prefixes, exact fault retry and control repeat"]
fn coupled_motion_matched_direct_retry() {
    matched_direct_retry(false, false);
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; compensated-reaction control repeat and exact motion-fault retry"]
fn coupled_motion_compensated_direct_retry() {
    matched_direct_retry(true, false);
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; fused guarded proposals against two direct controls and exact frame retry"]
fn coupled_motion_fused_proposals_exact_retry() {
    matched_direct_retry(true, true);
}

#[cfg(feature = "resource-counters")]
fn matched_direct_retry(compensated: bool, fusion: bool) {
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_MOTION_RETRY_CAPTURE").expect("explicit retry capture"),
    );
    std::fs::create_dir(&root).unwrap();
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC
                | wgpu::Features::SHADER_F64
                | wgpu::Features::PASSTHROUGH_SHADERS
                | molla_kernels::FeatherstoneKernel::paired_drift_required_features(),
        )
        .unwrap(),
    );
    assert!(!engine.adapter_info().unwrap().is_software);
    let (mut control, control_body) = fixture(&engine, false, false);
    let (mut repeat, repeat_body) = fixture(&engine, false, false);
    let (mut retry, retry_body) = fixture(&engine, false, false);
    let budget = 16 + 128 * 8;
    for (arm, world) in [&mut control, &mut repeat, &mut retry]
        .into_iter()
        .enumerate()
    {
        if compensated {
            let before = world.mpm_soil_reaction_info().unwrap();
            assert_eq!((before.compensated, before.retained_bytes), (false, 0));
            let resources = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            assert_eq!(
                world.set_mpm_soil_compensated_reactions(false, 0).unwrap(),
                0
            );
            assert!(world.set_mpm_soil_compensated_reactions(true, 31).is_err());
            assert_eq!(resources.finish().observations, 0);
            assert_eq!(world.mpm_soil_reaction_info().unwrap(), before);
            assert_eq!(
                world.set_mpm_soil_compensated_reactions(true, 32).unwrap(),
                32
            );
            let selected = world.mpm_soil_reaction_info().unwrap();
            assert_eq!((selected.compensated, selected.retained_bytes), (true, 32));
            let resources = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
            assert_eq!(
                world.set_mpm_soil_compensated_reactions(true, 32).unwrap(),
                32
            );
            assert!(world.set_mpm_soil_compensated_reactions(true, 31).is_err());
            assert!(world.set_mpm_soil_compensated_reactions(false, 31).is_err());
            assert_eq!(world.mpm_soil_reaction_info().unwrap(), selected);
            assert_eq!(resources.finish().observations, 0);
            let transfers = transfers.finish();
            assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
        }
        world
            .set_mpm_soil_motion_bound(Some([1.0; 3]), budget)
            .unwrap();
        assert_eq!(world.mpm_soil_motion_projection_fused(), Some(false));
        if fusion && arm == 2 {
            let resources = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            world.set_mpm_soil_motion_projection_fused(true).unwrap();
            world.set_mpm_soil_motion_projection_fused(true).unwrap();
            assert_eq!(resources.finish().observations, 0);
            assert_eq!(world.mpm_soil_motion_projection_fused(), Some(true));
        }
        world.step(0.0002).unwrap();
    }
    let initial = save_retry_words(&root, "control-prefix", &mut control);
    if compensated {
        let resources = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let error = control
            .enable_mpm_soil_newton_coupling(
                crate::mpm::MpmUnilateralConfig {
                    node_capacity: 4096,
                    search_distance: 0.05,
                    max_iterations: 64,
                },
                &[(control_body, 0.3)],
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("disable compensated direct reactions"),
            "{error}"
        );
        assert_eq!(resources.finish().observations, 0);
        assert!(
            control
                .gpu
                .as_ref()
                .unwrap()
                .soil
                .as_ref()
                .unwrap()
                .projection_config
                .is_none()
        );
        assert_eq!(
            save_retry_words(&root, "projection-rejected", &mut control),
            initial
        );
    }
    assert_eq!(
        save_retry_words(&root, "repeat-prefix", &mut repeat),
        initial,
        "control self-repeat prefix"
    );
    assert_eq!(
        save_retry_words(&root, "retry-prefix", &mut retry),
        initial,
        "retry must start from identical raw physical input"
    );
    let pose = retry.scene.body_pose(retry_body).unwrap();
    let velocity = retry.scene.body_velocity(retry_body).unwrap();
    assert_eq!(control.scene.body_pose(control_body).unwrap(), pose);
    assert_eq!(repeat.scene.body_pose(repeat_body).unwrap(), pose);
    assert_eq!(control.scene.body_velocity(control_body).unwrap(), velocity);
    assert_eq!(repeat.scene.body_velocity(repeat_body).unwrap(), velocity);
    retry
        .set_mpm_soil_motion_bound(Some([0.0; 3]), budget)
        .unwrap();
    let error = retry.step(0.0002).unwrap_err().to_string();
    std::fs::write(root.join("fault.txt"), &error).unwrap();
    assert!(error.contains("motion-bound"), "{error}");
    let after_fault = save_retry_words(&root, "after-fault", &mut retry);
    assert_eq!(after_fault.0, initial.0, "complete soil rollback");
    assert_eq!(
        &after_fault.1[5..],
        &initial.1[5..],
        "retained machine precision rollback"
    );
    assert_eq!(retry.scene.body_pose(retry_body).unwrap(), pose);
    assert_eq!(retry.scene.body_velocity(retry_body).unwrap(), velocity);
    assert_eq!(retry.completed_steps(), 1);
    assert!(retry.gpu_body_render_view().is_err());
    let stats = retry.mpm_soil_stats().unwrap();
    assert_eq!(
        (
            stats.committed_frames,
            stats.rolled_back_frames,
            stats.faulted
        ),
        (1, 1, false)
    );
    retry
        .set_mpm_soil_motion_bound(Some([1.0; 3]), budget)
        .unwrap();
    for world in [&mut control, &mut repeat, &mut retry] {
        world.step(0.0002).unwrap();
    }
    let expected = save_retry_words(&root, "control-final", &mut control);
    assert_eq!(
        save_retry_words(&root, "repeat-final", &mut repeat),
        expected,
        "control self-repeat final"
    );
    assert_eq!(
        save_retry_words(&root, "retry-final", &mut retry),
        expected,
        "matched-input fault retry final"
    );
    for (world, body) in [(&repeat, repeat_body), (&retry, retry_body)] {
        assert_eq!(
            world.scene.body_pose(body).unwrap(),
            control.scene.body_pose(control_body).unwrap()
        );
        assert_eq!(
            world.scene.body_velocity(body).unwrap(),
            control.scene.body_velocity(control_body).unwrap()
        );
        assert_eq!(world.completed_steps(), 2);
        assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 2);
    }
    if compensated {
        let stats = retry.mpm_soil_stats().unwrap();
        let resources = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        assert_eq!(
            retry.set_mpm_soil_compensated_reactions(false, 32).unwrap(),
            32
        );
        let info = retry.mpm_soil_reaction_info().unwrap();
        assert_eq!((info.compensated, info.retained_bytes), (false, 32));
        assert!(retry.set_mpm_soil_compensated_reactions(false, 0).is_err());
        assert_eq!(
            retry.set_mpm_soil_compensated_reactions(true, 32).unwrap(),
            32
        );
        assert_eq!(retry.mpm_soil_stats().unwrap(), stats);
        assert_eq!(resources.finish().observations, 0);
        let transfers = transfers.finish();
        assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
        assert_eq!(
            save_retry_words(&root, "reselected-final", &mut retry),
            expected
        );
    }
    std::fs::write(root.join("passed.json"), serde_json::to_vec_pretty(&serde_json::json!({"mode":"portable-dense-direct","compensated_reaction":compensated,"particles":1,"substeps":2,"machine_streams":9,"prefix_exact":true,"rollback_exact":true,"control_repeat_exact":true,"retry_exact":true})).unwrap()).unwrap();
    println!(
        "matched direct retry: complete soil and nine machine streams exact at prefix/final; precision and committed CPU state restored after real motion fault; original mixed-guard comparison remains separate"
    );
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; physical second-substep bound failure restores full coupled frame"]
fn coupled_motion_later_substep_restores_frame() {
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC
                | wgpu::Features::SHADER_F64
                | wgpu::Features::PASSTHROUGH_SHADERS
                | molla_kernels::FeatherstoneKernel::paired_drift_required_features(),
        )
        .unwrap(),
    );
    assert!(!engine.adapter_info().unwrap().is_software);
    let dt = 0.0001;
    let bound = 0.00000015;
    let (mut reference, reference_body) = fixture_options(&engine, true, true, true);
    reference.solver.substeps = 1;
    let reference_pose = reference.scene.body_pose(reference_body).unwrap();
    let initial_y = reference.read_mpm_soil_positions().unwrap()[0][1];
    reference.step(dt).unwrap();
    let first_y = reference.read_mpm_soil_positions().unwrap()[0][1];
    assert_ne!(
        reference.scene.body_pose(reference_body).unwrap(),
        reference_pose
    );
    reference.step(dt).unwrap();
    let second_y = reference.read_mpm_soil_positions().unwrap()[0][1];
    let first_displacement = (f64::from(first_y) - f64::from(initial_y)).abs();
    let second_displacement = (f64::from(second_y) - f64::from(first_y)).abs();
    assert!(first_displacement > 0.0 && first_displacement < bound);
    assert!(second_displacement > bound);
    println!(
        "late coupled motion: first={first_displacement:e} bound={bound:e} second={second_displacement:e}"
    );
    let (mut world, body) = fixture_options(&engine, true, true, true);
    world.solver.substeps = 3;
    world.prepare_mpm_soil_frame_loads().unwrap();
    let checkpoint = |world: &mut RigidWorld| {
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
    };
    let initial = checkpoint(&mut world);
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    let budget = 16 + 128 * 8;
    world
        .set_mpm_soil_motion_bound(Some([1.0, bound, 1.0]), budget)
        .unwrap();
    let error = world.step(3.0 * dt).unwrap_err().to_string();
    println!("late coupled motion: {error}");
    assert!(error.contains("soil grid flags=4(motion-bound)"), "{error}");
    assert!(error.contains("g2p-counter=2;"), "{error}");
    assert_eq!(checkpoint(&mut world), initial);
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
    assert_eq!(world.completed_steps(), 0);
    let stats = world.mpm_soil_stats().unwrap();
    assert_eq!(
        (
            stats.committed_frames,
            stats.rolled_back_frames,
            stats.faulted
        ),
        (0, 1, false)
    );
    assert!(world.read_mpm_soil_checkpoint(2).is_err());
    assert!(world.read_mpm_soil_frame_loads().is_err());
    assert!(world.gpu_body_render_view().is_err());
    assert_eq!(world.mpm_soil_particle_render_view().unwrap().0, 0);
    world
        .set_mpm_soil_motion_bound(Some([1.0; 3]), budget)
        .unwrap();
    world.step(3.0 * dt).unwrap();
    assert_ne!(
        world.read_mpm_soil_checkpoint(2).unwrap().1.to_bytes(),
        initial
    );
    assert_eq!(world.read_mpm_soil_checkpoint(2).unwrap().0, 1);
    assert_eq!(
        world.read_mpm_soil_frame_loads().unwrap().committed_frame,
        1
    );
    assert_eq!(world.completed_steps(), 1);
    println!(
        "late coupled motion: second gather rejected after physical first-substep movement; full initial checkpoint/pose/velocity/counters restored; no failed loads published; three-substep retry committed"
    );
}

fn load_words(frame: &MpmSoilFrameLoads) -> Vec<u32> {
    frame
        .bodies
        .iter()
        .flat_map(|body| {
            body.linear
                .into_iter()
                .chain(body.angular)
                .chain(body.world_origin_angular)
                .map(f32::to_bits)
        })
        .collect()
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; projected portable/native motion fault, committed loads and retry"]
fn coupled_motion_projected_restores_committed_frame() {
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC
                | wgpu::Features::SHADER_F64
                | wgpu::Features::PASSTHROUGH_SHADERS
                | molla_kernels::FeatherstoneKernel::paired_drift_required_features(),
        )
        .unwrap(),
    );
    assert!(!engine.adapter_info().unwrap().is_software);
    let budget = 16 + 128 * 8;
    for native in [false, true] {
        println!("projected motion rollback: native={native}");
        let (mut world, body) = fixture(&engine, native, true);
        world.prepare_mpm_soil_frame_loads().unwrap();
        world
            .set_mpm_soil_motion_bound(Some([1.0; 3]), budget)
            .unwrap();
        world.step(0.0002).unwrap();
        let saved = world.read_mpm_soil_checkpoint(2).unwrap();
        let pose = world.scene.body_pose(body).unwrap();
        let velocity = world.scene.body_velocity(body).unwrap();
        let loads = world.read_mpm_soil_frame_loads().unwrap();
        assert_eq!(
            (saved.0, loads.committed_frame, world.completed_steps()),
            (1, 1, 1)
        );
        assert_eq!(loads.bodies.len(), 1);
        assert_eq!(loads.bodies[0].body, body);
        assert!(loads.simulated_seconds.is_finite() && loads.simulated_seconds > 0.0);
        world
            .set_mpm_soil_motion_bound(Some([0.0; 3]), budget)
            .unwrap();
        let error = world.step(0.0002).unwrap_err();
        assert!(error.to_string().contains("particle-grid-fault"), "{error}");
        println!("projected motion rollback: {error}");
        let after = world.read_mpm_soil_checkpoint(2).unwrap();
        assert_eq!(after.0, 1);
        assert_eq!(after.1.to_bytes(), saved.1.to_bytes());
        assert_eq!(world.scene.body_pose(body).unwrap(), pose);
        assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
        assert_eq!(world.completed_steps(), 1);
        let stats = world.mpm_soil_stats().unwrap();
        assert_eq!(
            (
                stats.committed_frames,
                stats.rolled_back_frames,
                stats.faulted
            ),
            (1, 1, false)
        );
        let after_loads = world.read_mpm_soil_frame_loads().unwrap();
        assert_eq!(after_loads.committed_frame, loads.committed_frame);
        assert_eq!(
            after_loads.simulated_seconds.to_bits(),
            loads.simulated_seconds.to_bits()
        );
        assert_eq!(after_loads.bodies.len(), 1);
        assert_eq!(after_loads.bodies[0].body, body);
        assert_eq!(load_words(&after_loads), load_words(&loads));
        assert_eq!(world.mpm_soil_particle_render_view().unwrap().0, 1);
        assert!(world.gpu_body_render_view().is_err());
        world
            .set_mpm_soil_motion_bound(Some([1.0; 3]), budget)
            .unwrap();
        world.step(0.0002).unwrap();
        assert_eq!(world.read_mpm_soil_checkpoint(2).unwrap().0, 2);
        assert_eq!(
            world.read_mpm_soil_frame_loads().unwrap().committed_frame,
            2
        );
        assert_eq!(world.completed_steps(), 2);
        assert_eq!(
            world.set_mpm_soil_motion_bound(None, budget).unwrap(),
            budget
        );
        world.step(0.0002).unwrap();
        assert_eq!(world.read_mpm_soil_checkpoint(2).unwrap().0, 3);
        assert_eq!(
            world.read_mpm_soil_frame_loads().unwrap().committed_frame,
            3
        );
        assert_eq!(world.completed_steps(), 3);
        println!(
            "projected motion rollback: native={native}, exact checkpoint/pose/velocity/loads/time, guarded retry and disabled step committed; cross-world retry equivalence not tested"
        );
    }
}

fn compare_machine(
    a: &RigidWorld,
    ah: molla_sim::runtime::BodyHandle,
    b: &RigidWorld,
    bh: molla_sim::runtime::BodyHandle,
) {
    let values = |world: &RigidWorld, body| {
        let pose = world.scene.body_pose(body).unwrap();
        let velocity = world.scene.body_velocity(body).unwrap();
        pose.position
            .to_array()
            .into_iter()
            .chain(pose.rotation.to_array())
            .chain(velocity.linear.to_array())
            .chain(velocity.angular.to_array())
    };
    let mut maximum = 0.0f64;
    for (a, b) in values(a, ah).zip(values(b, bh)) {
        let delta = (a - b).abs();
        assert!(
            delta.is_finite() && delta <= 2e-5,
            "machine trajectory delta={delta}"
        );
        maximum = maximum.max(delta);
    }
    println!("motion frame: machine trajectory max={maximum:e} bound=2e-5");
}

fn fixture(
    engine: &Arc<Engine>,
    native: bool,
    projected: bool,
) -> (RigidWorld, molla_sim::runtime::BodyHandle) {
    fixture_options(engine, native, projected, false)
}

fn fixture_options(
    engine: &Arc<Engine>,
    native: bool,
    projected: bool,
    accelerating: bool,
) -> (RigidWorld, molla_sim::runtime::BodyHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            velocity: SpatialVector::new(Vec3::X * 0.05, Vec3::ZERO),
            additional_mass: Some(MassProperties {
                mass: 10.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 16;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 8;
    cfg.max_rigid = 1;
    cfg.gravity_y = if accelerating { -9.81 } else { 0.0 };
    cfg.youngs = 100.0;
    cfg.particle_mass = 0.25;
    cfg.particle_volume = 0.25 / 1600.0;
    let mut soil = match (native, projected) {
        (false, false) => MpmSimulation::new(engine, cfg),
        (true, false) => MpmSimulation::new_native_explicit(engine, cfg),
        (false, true) => MpmSimulation::new_sparse(engine, cfg, 64),
        (true, true) => MpmSimulation::new_native_sparse(engine, cfg, 64),
    }
    .unwrap();
    let count = if native || projected { 2 } else { 1 };
    let positions = if accelerating {
        [[0.5, 0.3, 0.0], [0.0, 0.0625, 0.0625]]
    } else {
        [[0.0; 3], [0.0625; 3]]
    };
    soil.set_particles_with_velocity(
        engine.queue().unwrap(),
        &positions[..count],
        &[[0.1, 0.0, 0.0]; 2][..count],
    )
    .unwrap();
    soil.prepare_parallel_particle_tile_ordering(engine.queue().unwrap(), 8)
        .unwrap();
    soil.set_particle_tile_ordering(true).unwrap();
    if native || projected {
        soil.prepare_deterministic_p2g_with_budget(512).unwrap();
        soil.set_deterministic_p2g(true).unwrap();
    }
    println!("motion fixture: soil ready; constructing native-prediction machine");
    let bytes =
        2 * crate::native_prediction_gpu::PreparedNativePrediction::required_bytes(scene.model())
            .unwrap();
    let mut world =
        RigidWorld::from_scene_gpu_with_native_prediction(scene, engine.clone(), 0.0001, bytes)
            .unwrap();
    world.solver.substeps = 2;
    world
        .attach_mpm_soil(
            soil,
            vec![MpmRigidDeviceBinding {
                body,
                shape: MpmSolidShape::Box {
                    half_extents: [0.25; 3],
                },
            }],
            if projected {
                128 * 1024 * 1024
            } else {
                32 * 1024 * 1024
            },
        )
        .unwrap();
    if projected {
        world
            .enable_mpm_soil_newton_coupling(
                crate::mpm::MpmUnilateralConfig {
                    node_capacity: 4096,
                    search_distance: 0.05,
                    max_iterations: 64,
                },
                &[(body, 0.3)],
            )
            .unwrap();
    }
    println!("motion fixture: machine and coupling ready");
    (world, body)
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; native/portable direct/projected coupled motion rollback"]
fn coupled_motion_bound_rolls_back_and_retries() {
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC
                | wgpu::Features::SHADER_F64
                | wgpu::Features::PASSTHROUGH_SHADERS
                | molla_kernels::FeatherstoneKernel::paired_drift_required_features(),
        )
        .unwrap(),
    );
    assert!(!engine.adapter_info().unwrap().is_software);
    let budget = 16 + 128 * 8;
    for native in [false, true] {
        for projected in [false, true] {
            println!("motion frame: native={native} projected={projected}");
            let (mut control, control_body) = fixture(&engine, native, projected);
            let (mut guarded, body) = fixture(&engine, native, projected);
            assert_eq!(guarded.set_mpm_soil_motion_bound(None, 0).unwrap(), 0);
            let resources = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            for bounds in [
                [f64::NAN, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, f64::INFINITY],
            ] {
                assert!(
                    guarded
                        .set_mpm_soil_motion_bound(Some(bounds), budget)
                        .is_err()
                );
            }
            assert!(
                guarded
                    .set_mpm_soil_motion_bound(Some([1.0; 3]), budget - 1)
                    .is_err()
            );
            assert_eq!(resources.finish().observations, 0);
            assert_eq!(
                guarded
                    .set_mpm_soil_motion_bound(Some([1.0; 3]), budget)
                    .unwrap(),
                budget
            );
            println!("motion fixture: guard prepared; stepping pair");
            control.step(0.0002).unwrap();
            guarded.step(0.0002).unwrap();
            let saved = guarded.read_mpm_soil_checkpoint(2).unwrap().1.to_bytes();
            assert_eq!(
                saved,
                control.read_mpm_soil_checkpoint(2).unwrap().1.to_bytes()
            );
            compare_machine(&guarded, body, &control, control_body);
            let pose = guarded.scene.body_pose(body).unwrap();
            let velocity = guarded.scene.body_velocity(body).unwrap();
            let resources = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            assert_eq!(
                guarded
                    .set_mpm_soil_motion_bound(Some([0.0; 3]), budget)
                    .unwrap(),
                budget
            );
            assert_eq!(resources.finish().observations, 0);
            let error = guarded.step(0.0002).unwrap_err();
            println!("motion frame rejected: {error}");
            let stats = guarded.mpm_soil_stats().unwrap();
            assert_eq!(
                (
                    stats.committed_frames,
                    stats.rolled_back_frames,
                    stats.faulted
                ),
                (1, 1, false)
            );
            assert_eq!(
                guarded.read_mpm_soil_checkpoint(2).unwrap().1.to_bytes(),
                saved
            );
            assert_eq!(guarded.scene.body_pose(body).unwrap(), pose);
            assert_eq!(guarded.scene.body_velocity(body).unwrap(), velocity);
            assert!(guarded.set_mpm_soil_motion_bound(None, budget - 1).is_err());
            assert_eq!(
                guarded.set_mpm_soil_motion_bound(None, budget).unwrap(),
                budget
            );
            guarded.step(0.0002).unwrap();
            control.step(0.0002).unwrap();
            assert_eq!(
                guarded.read_mpm_soil_checkpoint(2).unwrap().1.to_bytes(),
                control.read_mpm_soil_checkpoint(2).unwrap().1.to_bytes()
            );
            compare_machine(&guarded, body, &control, control_body);
            assert_eq!(guarded.mpm_soil_stats().unwrap().committed_frames, 2);
        }
    }
    println!(
        "motion frame: four modes exact particle admission/rollback/retry; machine trajectory <=2e-5, exact rollback; budget/invalid bounds reject before allocation"
    );
}
