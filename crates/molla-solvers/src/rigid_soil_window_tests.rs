use super::*;
use crate::mpm::{MpmConfig, MpmSolidShape};
use molla_core::gpu_metrics::TransferScope;
use molla_math::{Mat3, SpatialVector, Vec3};
use molla_sim::{MassProperties, runtime::BodyDesc};

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; coupled window publication, continued stepping and rollback"]
fn coupled_grid_window_move_preserves_machine_and_rollback() {
    run_window(false);
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; integer-window committed publication and rollback"]
fn coupled_integer_window_preserves_machine_and_rollback() {
    run_window(true);
}

fn run_window(integer: bool) {
    let native = integer || std::env::var_os("MOLLA_WINDOW_NATIVE_PREDICTION").is_some();
    let projected = std::env::var_os("MOLLA_WINDOW_PROJECTED").is_some();
    assert!(!projected || native);
    let engine = Arc::new(
        Engine::new_gpu_with_features(if native {
            wgpu::Features::SHADER_F64
                | wgpu::Features::PASSTHROUGH_SHADERS
                | molla_kernels::FeatherstoneKernel::paired_drift_required_features()
        } else {
            wgpu::Features::empty()
        })
        .unwrap(),
    );
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
    if integer {
        cfg.grid_size = 64;
        cfg.grid_origin = [-3.0, -0.5, -3.0];
        cfg.cell_size = 0.1;
    }
    cfg.max_particles = 8;
    cfg.max_rigid = 1;
    cfg.gravity_y = 0.0;
    cfg.youngs = 100.0;
    cfg.particle_mass = 0.25;
    cfg.particle_volume = 0.25 / 1600.0;
    let mut soil = if integer {
        MpmSimulation::new_sparse(&engine, cfg, 125).unwrap()
    } else if projected {
        MpmSimulation::new_sparse(&engine, cfg, 64).unwrap()
    } else {
        MpmSimulation::new(&engine, cfg).unwrap()
    };
    soil.set_particles(engine.queue().unwrap(), &[[0.0; 3], [0.0625; 3]])
        .unwrap();
    if projected {
        soil.prepare_parallel_particle_tile_ordering(engine.queue().unwrap(), cfg.max_particles)
            .unwrap();
        soil.set_particle_tile_ordering(true).unwrap();
        soil.prepare_deterministic_p2g_with_compensated_stiffness(
            u64::from(cfg.max_particles) * 64,
        )
        .unwrap();
        soil.set_deterministic_p2g(true).unwrap();
    }
    println!("coupled window: soil ready, constructing GPU machine");
    let mut world = if native {
        let bytes = 2 * crate::native_prediction_gpu::PreparedNativePrediction::required_bytes(
            scene.model(),
        )
        .unwrap();
        RigidWorld::from_scene_gpu_with_native_prediction(scene, engine.clone(), 0.0001, bytes)
            .unwrap()
    } else {
        RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap()
    };
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
                2_000_000
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
        println!("coupled window: projected compact fixed-order path selected");
    }
    let move_window = |world: &mut RigidWorld, limit| {
        if integer {
            world.move_mpm_soil_grid_window_cells([-1, 0, 1], limit)
        } else {
            world.move_mpm_soil_grid_window([-1.125, -1.0, -1.0], limit)
        }
    };
    assert!(move_window(&mut world, 2).is_err());
    world.step(0.0002).unwrap();
    println!("coupled window: first committed frame");
    let before = world.read_mpm_soil_checkpoint(2).unwrap().1;
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    let stats = world.mpm_soil_stats().unwrap();
    assert!(move_window(&mut world, 1).is_err());
    if integer {
        assert_eq!(world.move_mpm_soil_grid_window_cells([0; 3], 2).unwrap(), 1);
        assert!(
            world
                .move_mpm_soil_grid_window_cells([40, 0, 0], 2)
                .is_err()
        );
    } else {
        assert!(
            world
                .move_mpm_soil_grid_window([-1.0625, -1.0, -1.0], 2)
                .is_err()
        );
    }
    assert_eq!(
        world.read_mpm_soil_checkpoint(2).unwrap().1.to_bytes(),
        before.to_bytes()
    );
    assert_eq!(world.mpm_soil_stats().unwrap(), stats);
    let expected = if integer {
        before.with_grid_window_cells([-1, 0, 1]).unwrap()
    } else {
        before.with_grid_window([-1.125, -1.0, -1.0]).unwrap()
    };
    let transfers = TransferScope::start().unwrap();
    assert_eq!(move_window(&mut world, 2).unwrap(), 1);
    let traffic = transfers.finish();
    assert_eq!(traffic.upload_bytes, 0);
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 1);
    assert!(world.read_mpm_soil_checkpoint(2).is_err());
    assert!(world.read_mpm_soil_wrenches().is_err());
    let actual = world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .soil
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap();
    assert_eq!(actual.to_bytes(), expected.to_bytes());
    world.step(0.0002).unwrap();
    let continued = world.read_mpm_soil_checkpoint(2).unwrap();
    assert_eq!(continued.0, 2);
    assert_eq!(
        continued.1.config().grid_origin,
        expected.config().grid_origin
    );
    assert_eq!(
        continued.1.grid_window_cells(),
        expected.grid_window_cells()
    );
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .fail_after_steps = Some(2);
    assert!(
        world
            .step(0.0002)
            .unwrap_err()
            .to_string()
            .contains("injected failure")
    );
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
    assert_eq!(
        world.read_mpm_soil_checkpoint(2).unwrap().1.to_bytes(),
        continued.1.to_bytes()
    );
    world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .fail_after_steps = None;
    world.step(0.0002).unwrap();
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 3);
    if let Some(root) = std::env::var_os("MOLLA_COUPLED_WINDOW_OUTPUT") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir(&root).unwrap();
        for (name, state) in [
            ("before", before),
            ("moved", actual),
            ("continued", continued.1),
        ] {
            std::fs::write(root.join(format!("{name}.checkpoint")), state.to_bytes()).unwrap();
        }
    }
    println!(
        "coupled window: exact publication, unchanged machine, invalid observations, continued coupling and exact rollback pass; uploads={} readbacks={}",
        traffic.upload_bytes, traffic.readback_bytes
    );
}
