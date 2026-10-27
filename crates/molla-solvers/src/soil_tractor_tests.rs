use super::*;
use molla_math::SpatialVector;

#[test]
fn admm_tractor_transfer_admission_before_gpu() {
    let engine = Arc::new(Engine::new_cpu());
    let config = SoilTractorConfig::default();
    let coupling = crate::mpm::MpmUnilateralConfig {
        node_capacity: 4096,
        search_distance: 0.05,
        max_iterations: 64,
    };
    for transfer in [
        SoilTractorTransfer::FixedOrder {
            max_gather_bytes: 3_199_999,
        },
        SoilTractorTransfer::SharedChunk {
            max_gather_bytes: 3_199_999,
        },
        SoilTractorTransfer::GridReuse {
            max_gather_bytes: 14_075_711,
        },
    ] {
        let explicit = SoilTractor::new_native_explicit_with_paired_admm_transfer(
            engine.clone(),
            config,
            coupling,
            0.3,
            molla_kernels::FeatherstoneExecution::Generic,
            MpmGridStorage::Dense,
            transfer,
        )
        .err()
        .unwrap();
        assert!(
            explicit
                .to_string()
                .contains("gather scratch budget exceeded")
        );
        let native = SoilTractor::new_native_prediction_with_admm_transfer(
            engine.clone(),
            config,
            coupling,
            0.3,
            MpmGridStorage::Dense,
            1.0 / 60.0,
            transfer,
        )
        .err()
        .unwrap();
        assert!(
            native
                .to_string()
                .contains("gather scratch budget exceeded")
        );
    }
}

#[test]
fn tractor_paired_transfer_rejects_budget_before_device_creation() {
    let engine = Arc::new(Engine::new_cpu());
    let config = SoilTractorConfig::default();
    for transfer in [
        SoilTractorTransfer::FixedOrder {
            max_gather_bytes: 3_199_999,
        },
        SoilTractorTransfer::SharedChunk {
            max_gather_bytes: 3_199_999,
        },
        SoilTractorTransfer::GridReuse {
            max_gather_bytes: 14_075_711,
        },
    ] {
        let error = SoilTractor::new_native_explicit_with_paired_newton_transfer(
            engine.clone(),
            config,
            crate::mpm::MpmUnilateralConfig {
                node_capacity: 4096,
                search_distance: 0.05,
                max_iterations: 64,
            },
            0.3,
            molla_kernels::FeatherstoneExecution::Generic,
            MpmGridStorage::Dense,
            transfer,
        )
        .err()
        .expect("undersized transfer budget must reject before GPU setup");
        assert!(error.to_string().contains("gather scratch budget exceeded"));
    }
}

#[test]
fn tractor_transfer_budget_is_explicit() {
    assert_eq!(SoilTractorTransfer::default(), SoilTractorTransfer::Atomic);
    for particles in [0, 1, 129, 50000, u32::MAX] {
        let bytes = u64::from(particles.max(1)) * 64;
        SoilTractorTransfer::Atomic.validate(particles).unwrap();
        SoilTractorTransfer::FixedOrder {
            max_gather_bytes: bytes,
        }
        .validate(particles)
        .unwrap();
        assert!(
            SoilTractorTransfer::FixedOrder {
                max_gather_bytes: bytes - 1
            }
            .validate(particles)
            .is_err()
        );
        SoilTractorTransfer::SharedChunk {
            max_gather_bytes: bytes,
        }
        .validate(particles)
        .unwrap();
        assert!(
            SoilTractorTransfer::SharedChunk {
                max_gather_bytes: bytes - 1
            }
            .validate(particles)
            .is_err()
        );
    }
}

#[test]
fn tractor_grid_reuse_budget_includes_logical_grid() {
    for (particles, grid, expected) in [
        (50_000, 64, 14_075_712),
        (129, 35, 886_100),
        (0, 64, 5_275_888),
    ] {
        let transfer = SoilTractorTransfer::GridReuse {
            max_gather_bytes: expected,
        };
        assert_eq!(
            transfer.required_gather_bytes(particles, grid).unwrap(),
            expected
        );
        transfer.validate_with_grid(particles, grid).unwrap();
        assert!(
            SoilTractorTransfer::GridReuse {
                max_gather_bytes: expected - 1
            }
            .validate_with_grid(particles, grid)
            .is_err()
        );
    }
    let transfer = SoilTractorTransfer::GridReuse {
        max_gather_bytes: u64::MAX,
    };
    for grid in [0, 951, u32::MAX] {
        assert!(transfer.validate_with_grid(50_000, grid).is_err());
    }
    assert!(transfer.validate_with_grid(u32::MAX, 64).is_ok());
}

#[test]
fn tractor_grid_storage_configuration() {
    let cfg = SoilTractorConfig::default();
    assert_eq!(MpmGridStorage::default(), MpmGridStorage::Dense);
    for storage in [
        MpmGridStorage::Dense,
        MpmGridStorage::ActiveTiles,
        MpmGridStorage::Compact { tile_capacity: 1 },
        MpmGridStorage::Compact {
            tile_capacity: 4096,
        },
    ] {
        cfg.validate_grid_storage(storage).unwrap();
    }
    for tile_capacity in [0, 4097, u32::MAX] {
        assert!(
            cfg.validate_grid_storage(MpmGridStorage::Compact { tile_capacity })
                .is_err()
        );
    }
    assert!(
        SoilTractorConfig {
            grid_size: 0,
            ..cfg
        }
        .validate_grid_storage(MpmGridStorage::Dense)
        .is_err()
    );
}

#[test]
#[ignore = "requires native GPU; tractor soil factory layout and seed contract"]
fn tractor_grid_storage_factory() {
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let cfg = SoilTractorConfig {
        seed: [8, 2, 4],
        ..Default::default()
    };
    let mut expected = None;
    for native in [false, true] {
        for storage in [
            MpmGridStorage::Dense,
            MpmGridStorage::ActiveTiles,
            MpmGridStorage::Compact { tile_capacity: 512 },
        ] {
            let mut soil = cfg
                .soil_backend_with_grid(&engine, native, storage)
                .unwrap();
            assert_eq!(soil.grid_storage(), storage);
            assert_eq!(soil.particle_count(), 64);
            let bytes = soil
                .checkpoint_soil_with(engine.queue().unwrap())
                .unwrap()
                .to_bytes();
            assert_eq!(*expected.get_or_insert(bytes.clone()), bytes);
            if let MpmGridStorage::Compact { .. } = storage {
                let before = soil.grid_storage_bytes();
                soil.reserve_sparse_tiles(1024).unwrap();
                assert_eq!(
                    soil.grid_storage(),
                    MpmGridStorage::Compact {
                        tile_capacity: 1024
                    }
                );
                assert!(soil.grid_storage_bytes() > before);
                assert_eq!(
                    soil.checkpoint_soil_with(engine.queue().unwrap())
                        .unwrap()
                        .to_bytes(),
                    bytes
                );
            }
            println!(
                "tractor factory native={native} storage={storage:?}: exact seed/history and live layout pass"
            );
        }
    }
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let invalid = MpmGridStorage::Compact { tile_capacity: 0 };
    assert!(SoilTractor::new_with_grid_storage(engine.clone(), cfg, invalid).is_err());
    assert!(
        SoilTractor::new_native_explicit_with_grid_options(
            engine.clone(),
            cfg,
            invalid,
            molla_kernels::FeatherstoneRounding::Storage,
            molla_kernels::FeatherstoneExecution::Generic
        )
        .is_err()
    );
    let coupling = crate::mpm::MpmUnilateralConfig {
        node_capacity: 128,
        search_distance: 0.05,
        max_iterations: 64,
    };
    assert!(
        SoilTractor::new_native_explicit_with_paired_newton_grid(
            engine.clone(),
            cfg,
            coupling,
            0.3,
            molla_kernels::FeatherstoneExecution::Generic,
            invalid
        )
        .is_err()
    );
    assert!(
        SoilTractor::new_native_explicit_with_paired_wall_friction_grid(
            engine,
            cfg,
            coupling,
            0.3,
            molla_kernels::FeatherstoneExecution::Generic,
            invalid
        )
        .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    assert_eq!(traffic.finish().upload_bytes, 0);
}

#[test]
#[ignore = "requires native GPU; full tractor seed and initial rigid support, no machine solve"]
fn tractor_grid_initial_support() {
    use crate::mpm::MpmPrescribedSolid;
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let config = SoilTractorConfig::default();
    let (scene, handles) = build_soil_tractor_scene().unwrap();
    let solids: Vec<_> = handles
        .wheels
        .into_iter()
        .map(|body| {
            (
                body,
                MpmSolidShape::Cylinder {
                    radius: WHEEL_RADIUS as f32,
                    half_width: WHEEL_HALF_WIDTH as f32,
                },
            )
        })
        .chain(std::iter::once((
            handles.tine,
            MpmSolidShape::Box {
                half_extents: TINE_HALF_EXTENTS.as_vec3().to_array(),
            },
        )))
        .map(|(body, shape)| MpmPrescribedSolid {
            shape,
            center: scene.body_world_com(body).unwrap().as_vec3().to_array(),
            orientation: scene.body_pose(body).unwrap().rotation.as_quat().to_array(),
            linear_velocity: [0.; 3],
            angular_velocity: [0.; 3],
        })
        .collect();
    for with_rigids in [false, true] {
        let mut expected_tiles = None;
        for storage in [
            MpmGridStorage::ActiveTiles,
            MpmGridStorage::Compact { tile_capacity: 512 },
            MpmGridStorage::Compact {
                tile_capacity: 1024,
            },
        ] {
            let mut soil = config
                .soil_backend_with_grid(&engine, true, storage)
                .unwrap();
            if with_rigids {
                soil.set_prescribed_solids(queue, &solids).unwrap();
            }
            let before_positions = soil.read_positions_with(device, queue).unwrap();
            let before_velocities = soil.read_velocities_with(device, queue).unwrap();
            soil.step_n_with(device, queue, (1. / 60. / 34.) as f32, 1)
                .unwrap();
            let stats = soil.read_active_grid_status_with(device, queue).unwrap();
            eprintln!(
                "initial 50k support with_rigids={with_rigids} storage={storage:?}: {stats:?}"
            );
            assert!(!stats.numerical_fault, "{stats:?}");
            let overflow = with_rigids && storage == MpmGridStorage::Compact { tile_capacity: 512 };
            assert_eq!(stats.overflow, overflow, "{stats:?}");
            assert_eq!(stats.completed_substeps, u32::from(!overflow));
            if overflow {
                assert!(stats.active_tiles > stats.tile_capacity);
                assert!(soil.checkpoint_soil_with(queue).is_err());
                assert_eq!(
                    before_positions,
                    soil.read_positions_with(device, queue).unwrap()
                );
                assert_eq!(
                    before_velocities,
                    soil.read_velocities_with(device, queue).unwrap()
                );
            } else {
                assert_eq!(stats.active_tiles, if with_rigids { 544 } else { 320 });
                assert_eq!(
                    *expected_tiles.get_or_insert(stats.active_tiles),
                    stats.active_tiles
                );
            }
        }
    }
}

#[test]
fn tractor_geometry_mass_and_motor_conventions() {
    assert_eq!(
        SoilTractorConfig::default().particle_count().unwrap(),
        50_000
    );
    SoilTractorConfig::default().validate().unwrap();
    assert!(
        SoilTractorConfig {
            seed: [u32::MAX, 2, 2],
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    let (mut scene, h) = build_soil_tractor_scene().unwrap();
    assert_eq!(scene.bodies().count(), 6);
    assert_eq!(scene.colliders().count(), 0);
    assert_eq!(scene.loop_joints().count(), 0);
    let mass: Real = scene
        .bodies()
        .map(|(b, _)| scene.body_mass_properties(b).unwrap().mass)
        .sum();
    assert_eq!(mass, 860.0);
    for (wheel, axle) in h.wheels.into_iter().zip(h.axles) {
        let pose = scene.body_pose(wheel).unwrap();
        assert!((pose.rotation * Vec3::Y - Vec3::Z).length() < 1e-12);
        let joint = scene.joint(axle).unwrap();
        assert_eq!(joint.kind, JointType::Revolute);
        let a = scene
            .body_pose(h.chassis)
            .unwrap()
            .compose(&joint.frame_parent);
        let b = pose.compose(&joint.frame_child);
        assert!((a.position - b.position).length() < 1e-12);
        assert!((a.rotation * joint.axis - Vec3::Z).length() < 1e-12);
        scene
            .set_joint_motor(
                axle,
                JointAxis::AngX,
                JointMotor {
                    mode: JointTargetMode::Effort,
                    target_position: -100.0,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    assert!(scene.body_pose(h.tine).unwrap().position.y - TINE_HALF_EXTENTS.y < 0.0);
    let mut oracle = RigidWorld::from_scene(scene).unwrap();
    oracle.solver.substeps = 1;
    oracle.step(0.0001).unwrap();
    for wheel in h.wheels {
        assert!(oracle.scene.body_velocity(wheel).unwrap().angular.z < 0.0);
    }
}

fn momentum(scene: &RigidScene) -> SpatialVector {
    let mut total = SpatialVector::ZERO;
    for (body, _) in scene.bodies() {
        let mass = scene.body_mass_properties(body).unwrap();
        let pose = scene.body_pose(body).unwrap();
        let velocity = scene.body_velocity(body).unwrap();
        let rotation = Mat3::from_quat(pose.rotation);
        let linear = velocity.linear * mass.mass;
        total.linear += linear;
        total.angular += rotation * mass.inertia * rotation.transpose() * velocity.angular
            + scene.body_world_com(body).unwrap().cross(linear);
    }
    total
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn soil_only_tractor_boundary_completes_without_machine_dispatch() {
    soil_only_boundary(false);
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn soil_only_tractor_boundary_completes_with_frame_backup() {
    soil_only_boundary(true);
}

fn soil_only_boundary(with_backup: bool) {
    use crate::mpm::PreparedMpmRigidDeviceBridge;
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let config = SoilTractorConfig {
        seed: [56, 4, 32],
        grid_size: 48,
        cell_size: 0.125,
        ..Default::default()
    };
    let (scene, handles) = build_soil_tractor_scene().unwrap();
    let mut soil = config.soil(&engine).unwrap();
    let mut backup = with_backup.then(|| {
        crate::mpm::MpmFrameBackup::new(engine.clone(), &soil, config.coupling_budget_bytes)
            .unwrap()
    });
    let mut states = std::array::from_fn(|_| {
        let mut state = scene.state().clone_host().unwrap();
        state
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        state
    });
    let mut bindings: Vec<_> = handles
        .wheels
        .into_iter()
        .map(|body| MpmRigidDeviceBinding {
            body,
            shape: MpmSolidShape::Cylinder {
                radius: WHEEL_RADIUS as f32,
                half_width: WHEEL_HALF_WIDTH as f32,
            },
        })
        .collect();
    bindings.push(MpmRigidDeviceBinding {
        body: handles.tine,
        shape: MpmSolidShape::Box {
            half_extents: TINE_HALF_EXTENTS.as_vec3().to_array(),
        },
    });
    let mut bridge = PreparedMpmRigidDeviceBridge::new(
        engine.clone(),
        &mut soil,
        &scene,
        &states,
        &bindings,
        4096,
    )
    .unwrap();
    if let Some(backup) = &mut backup {
        backup.begin(&soil).unwrap();
    }
    bridge
        .step_soil_and_apply(&mut soil, &scene, &mut states, 0, 0.0001)
        .unwrap();
    let complete = engine.device().unwrap().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(5)),
    });
    if let Err(error) = complete {
        eprintln!("soil-only tractor boundary did not complete: {error}");
        std::process::exit(76);
    }
    bridge.restore_forces(&mut states).unwrap();
    if let Some(backup) = &mut backup {
        backup.commit();
    }
    let wrenches = soil.read_rigid_wrenches(&engine).unwrap();
    assert_eq!(wrenches.len(), 5);
    assert!(
        wrenches
            .iter()
            .flat_map(|(f, t)| f.iter().chain(t))
            .all(|v| v.is_finite())
    );
    println!(
        "7168-particle soil-only tractor boundary completed; backup={with_backup}; machine dispatch not exercised"
    );
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn articulated_tractor_consumes_force_torque_and_drives() {
    articulated_tractor_fixture(false);
}

#[test]
#[ignore = "requires native float-atomic GPU support"]
fn native_articulated_tractor_consumes_force_torque_and_drives() {
    articulated_tractor_fixture(true);
}

fn articulated_tractor_fixture(native: bool) {
    let engine = Arc::new(if native {
        Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap()
    } else {
        Engine::new_gpu().unwrap()
    });
    let config = SoilTractorConfig {
        seed: [56, 4, 32],
        grid_size: 48,
        cell_size: 0.125,
        ..Default::default()
    };
    println!("tractor fixture: building six-body GPU articulation and 7168 soil particles");
    let mut tractor = if native {
        SoilTractor::new_native_explicit(engine.clone(), config).unwrap()
    } else {
        SoilTractor::new(engine.clone(), config).unwrap()
    };
    println!("tractor fixture: GPU articulation ready");
    assert!(tractor.read_soil_loads().is_err());
    let before = momentum(&tractor.world.scene);
    let centers: Vec<_> = tractor
        .world
        .scene
        .bodies()
        .map(|(body, _)| (body, tractor.world.scene.body_world_com(body).unwrap()))
        .collect();
    let mut gravity_torque = Vec3::ZERO;
    for &(body, center) in &centers {
        gravity_torque += center.cross(
            Vec3::Y * (-9.81 * tractor.world.scene.body_mass_properties(body).unwrap().mass),
        );
    }
    let dt = 0.0001;
    println!("tractor fixture: first frame begin");
    tractor_step_or_report_fault(&mut tractor, &engine, dt);
    println!("tractor fixture: first frame complete, reading soil loads");
    let loads = tractor.read_soil_loads().unwrap();
    assert_eq!(loads.committed_frame, 1);
    assert_eq!(loads.substep_dt, dt as f32);
    assert_eq!(loads.bodies.len(), 5);
    let mut force = Vec3::new(0.0, -9.81 * 860.0, 0.0);
    let mut torque = gravity_torque;
    for load in &loads.bodies {
        let f = Vec3::from_array(load.force.map(f64::from));
        let t = Vec3::from_array(load.torque.map(f64::from));
        assert!(f.is_finite() && t.is_finite());
        let center = centers
            .iter()
            .find(|(body, _)| *body == load.body)
            .unwrap()
            .1;
        force += f;
        torque += t + center.cross(f);
    }
    let after = momentum(&tractor.world.scene);
    let linear_error = (after.linear - before.linear - force * dt).length();
    let angular_error = (after.angular - before.angular - torque * dt).length();
    println!("tractor impulse balance: linear={linear_error:.9e}, angular={angular_error:.9e}");
    assert!(linear_error < 2e-5, "linear {linear_error}");
    assert!(angular_error < 2e-5, "angular {angular_error}");
    let start = tractor.telemetry().unwrap();
    for _ in 0..6 {
        tractor_step_or_report_fault(&mut tractor, &engine, 0.001);
    }
    let end = tractor.telemetry().unwrap();
    assert!((end.simulated_seconds - 0.0061).abs() < 1e-12);
    assert!(end.chassis_pose.position.is_finite());
    assert!(end.wheel_surface_speed_m_s.iter().all(|v| *v > 1e-4));
    assert!(end.chassis_pose.position.x > start.chassis_pose.position.x + 1e-8);
    let loads = tractor.read_soil_loads().unwrap();
    assert_eq!(loads.committed_frame, 7);
    assert!(
        loads
            .bodies
            .iter()
            .any(|b| b.body == tractor.handles.tine && b.force.iter().any(|v| v.abs() > 1e-4))
    );
    assert!(
        loads
            .bodies
            .iter()
            .any(|b| b.torque.iter().any(|v| v.abs() > 1e-4))
    );
    let old_time = end.simulated_seconds;
    assert!(tractor.step(0.001, [f64::NAN; 4]).is_err());
    assert_eq!(tractor.telemetry().unwrap().simulated_seconds, old_time);
    tractor
        .world
        .scene
        .set_body_enabled(tractor.handles.wheels[0], false)
        .unwrap();
    assert!(tractor.read_soil_loads().is_err());
    println!(
        "tractor fixture: articulated force/torque balance, driven motion, tine loads and stale-readout guard pass"
    );
}

fn tractor_step_or_report_fault(tractor: &mut SoilTractor, engine: &Engine, dt: Real) {
    let result = tractor.step(dt, [300.0; 4]);
    if let Err(error) = &result {
        if engine.check_device().is_err() {
            eprintln!(
                "tractor GPU frame failed; exiting faulted test process without queue teardown: {error}"
            );
            std::process::exit(76);
        }
    }
    result.unwrap();
}
