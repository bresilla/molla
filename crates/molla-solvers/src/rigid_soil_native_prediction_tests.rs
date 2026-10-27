use super::*;

#[path = "rigid_soil_submission_tests.rs"]
mod submission;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeWorldFixture {
    Tine,
    AllContacts,
}

impl NativeWorldFixture {
    fn parse(value: Option<&str>) -> Option<Self> {
        match value {
            None | Some("tine") => Some(Self::Tine),
            Some("all-contacts") => Some(Self::AllContacts),
            _ => None,
        }
    }

    fn positions(self, scene: &RigidScene, bindings: &[MpmRigidDeviceBinding]) -> Vec<[f32; 3]> {
        if self == Self::Tine {
            return vec![[-1.45, 0.0, 0.0]; 65];
        }
        bindings
            .iter()
            .flat_map(|binding| {
                let center = scene.body_world_com(binding.body).unwrap();
                let offset = match binding.shape {
                    MpmSolidShape::Cylinder { radius, .. } => {
                        Vec3::new(0.0, -f64::from(radius) - 0.02, 0.0)
                    }
                    MpmSolidShape::Box { half_extents } => {
                        Vec3::new(-f64::from(half_extents[0]) - 0.02, -0.23, 0.0)
                    }
                    _ => panic!("unsupported native angular fixture shape"),
                };
                (0..13).map(move |i| {
                    (center
                        + offset
                        + Vec3::new(
                            (i % 5) as f64 * 0.005 - 0.01,
                            0.0,
                            (i / 5) as f64 * 0.005 - 0.005,
                        ))
                    .as_vec3()
                    .to_array()
                })
            })
            .collect()
    }

    fn velocity(self) -> Vec3 {
        match self {
            Self::Tine => Vec3::new(1.0, 0.0, 0.5),
            Self::AllContacts => Vec3::new(1.0, 1.0, 0.5),
        }
    }
}

#[test]
fn native_angular_fixture_selection_and_full_support() {
    assert_eq!(
        NativeWorldFixture::parse(None),
        Some(NativeWorldFixture::Tine)
    );
    assert_eq!(
        NativeWorldFixture::parse(Some("tine")),
        Some(NativeWorldFixture::Tine)
    );
    assert_eq!(
        NativeWorldFixture::parse(Some("all-contacts")),
        Some(NativeWorldFixture::AllContacts)
    );
    for value in ["", "all", "1", "all-contacts "] {
        assert_eq!(NativeWorldFixture::parse(Some(value)), None);
    }
    let (scene, handles) = crate::soil_tractor::build_soil_tractor_scene().unwrap();
    let bindings: Vec<_> = handles
        .wheels
        .into_iter()
        .map(|body| MpmRigidDeviceBinding {
            body,
            shape: MpmSolidShape::Cylinder {
                radius: 0.45,
                half_width: 0.14,
            },
        })
        .chain([MpmRigidDeviceBinding {
            body: handles.tine,
            shape: MpmSolidShape::Box {
                half_extents: [0.1, 0.35, 0.1],
            },
        }])
        .collect();
    let points = NativeWorldFixture::AllContacts.positions(&scene, &bindings);
    assert_eq!(points.len(), 65);
    let mut nodes = std::collections::BTreeSet::new();
    for (patch, binding) in points.chunks_exact(13).zip(&bindings) {
        let center = scene.body_world_com(binding.body).unwrap();
        for point in patch {
            let p = Vec3::from_array(point.map(f64::from));
            assert!((p - center).length() < 0.5);
            let base = ((p - Vec3::new(-3.0, -0.5, -3.0)) / 0.125 - Vec3::splat(0.5)).floor();
            assert!(base.min_element() >= 2.0 && (base + Vec3::splat(2.0)).max_element() < 46.0);
            for x in 0..3 {
                for y in 0..3 {
                    for z in 0..3 {
                        nodes.insert([base.x as i32 + x, base.y as i32 + y, base.z as i32 + z]);
                    }
                }
            }
        }
    }
    assert!(nodes.len() > 128 && nodes.len() <= 256);
    assert_eq!(
        NativeWorldFixture::Tine.positions(&scene, &bindings),
        vec![[-1.45, 0.0, 0.0]; 65]
    );
}

#[test]
#[ignore = "requires native f64 GPU, timestamps and bounded six-body soil runtime construction"]
fn native_prediction_world_shares_mass_and_rolls_back() {
    let fixture =
        NativeWorldFixture::parse(std::env::var("MOLLA_NATIVE_WORLD_FIXTURE").ok().as_deref())
            .expect("invalid native world fixture");
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_NATIVE_WORLD_CAPTURE").unwrap());
    native_prediction_world_case(native_world_engine(), root, fixture, None);
}

fn native_world_engine() -> Arc<Engine> {
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_F64
                | wgpu::Features::SHADER_FLOAT32_ATOMIC
                | wgpu::Features::PASSTHROUGH_SHADERS
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        )
        .unwrap(),
    );
    assert!(!engine.adapter_info().unwrap().is_software);
    assert!(crate::soil_newton_gpu::protected_enabled(&engine));
    engine
}

fn native_prediction_world_case(
    engine: Arc<Engine>,
    root: std::path::PathBuf,
    fixture: NativeWorldFixture,
    force_separate: Option<bool>,
) {
    native_prediction_world_case_method(engine, root, fixture, force_separate, false);
}

#[test]
#[ignore = "requires native f64 GPU; articulated ADMM world with all wheel/tine contacts"]
fn admm_articulated_world_feedback_and_rollback() {
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_ADMM_WORLD_OUTPUT").unwrap());
    native_prediction_world_case_method(
        native_world_engine(),
        root,
        NativeWorldFixture::AllContacts,
        None,
        true,
    );
}

fn native_prediction_world_case_method(
    engine: Arc<Engine>,
    root: std::path::PathBuf,
    fixture: NativeWorldFixture,
    force_separate: Option<bool>,
    admm: bool,
) {
    use crate::soil_tractor::{
        TINE_HALF_EXTENTS, WHEEL_HALF_WIDTH, WHEEL_RADIUS, build_soil_tractor_scene,
    };
    use molla_sim::JointTargetMode;
    use molla_sim::joint_control::{JointAxis, JointMotor};
    let start = std::time::Instant::now();
    std::fs::create_dir(&root).unwrap();
    let (mut scene, handles) = build_soil_tractor_scene().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let bindings: Vec<_> = handles
        .wheels
        .into_iter()
        .map(|body| MpmRigidDeviceBinding {
            body,
            shape: MpmSolidShape::Cylinder {
                radius: WHEEL_RADIUS as f32,
                half_width: WHEEL_HALF_WIDTH as f32,
            },
        })
        .chain([MpmRigidDeviceBinding {
            body: handles.tine,
            shape: MpmSolidShape::Box {
                half_extents: TINE_HALF_EXTENTS.as_vec3().to_array(),
            },
        }])
        .collect();
    let dt = 0.0015;
    let points = fixture.positions(&scene, &bindings);
    let node_capacity = if fixture == NativeWorldFixture::AllContacts {
        256
    } else {
        128
    };
    let mut world =
        RigidWorld::from_scene_gpu_with_native_prediction(scene, engine.clone(), dt / 3.0, 1 << 20)
            .unwrap();
    world.solver.substeps = 3;
    if cfg!(feature = "native-lazy-geometry-drift") {
        assert_eq!(
            world.solver.gpu_unrestricted_drift_backend(),
            Some("native-geometry-lazy-anchors")
        );
    }
    let mut config = MpmConfig::sand();
    config.grid_size = 48;
    config.grid_origin = [-3.0, -0.5, -3.0];
    config.cell_size = 0.125;
    config.max_particles = 65;
    config.max_rigid = 5;
    config.gravity_y = 0.0;
    config.youngs = 100.0;
    config.particle_mass = 0.5 / 65.0;
    config.particle_volume = config.particle_mass / 1600.0;
    let particle_mass = f64::from(config.particle_mass);
    let initial_velocity = fixture.velocity();
    std::fs::write(root.join("fixture.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "fixture":format!("{fixture:?}"), "positions":points, "initial_velocity":initial_velocity.to_array(),
        "node_capacity":node_capacity, "dt":dt, "substeps":3,
        "scope":"zero-gravity six-body reaction/angular diagnostic; not field calibration or full-workload acceptance",
        "angular_tolerance":2e-5, "linear_tolerance":2e-5, "combined_linear_tolerance":2e-4,
    })).unwrap()).unwrap();
    eprintln!("native world: compact soil begin {:?}", start.elapsed());
    let mut soil = MpmSimulation::new_native_sparse(&engine, config, 1024).unwrap();
    eprintln!("native world: compact soil ready {:?}", start.elapsed());
    soil.set_particles_with_velocity(
        engine.queue().unwrap(),
        &points,
        &vec![initial_velocity.as_vec3().to_array(); 65],
    )
    .unwrap();
    if force_separate.is_some() {
        soil.prepare_parallel_particle_tile_ordering(engine.queue().unwrap(), 65)
            .unwrap();
        soil.set_particle_tile_ordering(true).unwrap();
        soil.prepare_deterministic_p2g().unwrap();
        soil.set_deterministic_p2g(true).unwrap();
    }
    world
        .attach_mpm_soil(soil, bindings.clone(), 32 * 1024 * 1024)
        .unwrap();
    if let Some(force_separate) = force_separate {
        world
            .gpu
            .as_mut()
            .unwrap()
            .soil
            .as_mut()
            .unwrap()
            .force_separate_submissions = force_separate;
    }
    eprintln!(
        "native world: bridge ready; contact begin admm={admm} {:?}",
        start.elapsed()
    );
    (if admm {
        RigidWorld::enable_mpm_soil_admm_coupling
    } else {
        RigidWorld::enable_mpm_soil_newton_coupling
    })(
        &mut world,
        crate::mpm::MpmUnilateralConfig {
            node_capacity,
            search_distance: 0.05,
            max_iterations: 64,
        },
        &bindings.iter().map(|b| (b.body, 0.3)).collect::<Vec<_>>(),
    )
    .unwrap();
    world.prepare_mpm_soil_frame_loads().unwrap();
    if admm {
        assert!(world.mpm_soil_admm_enabled());
        assert!(!world.mpm_soil_newton_enabled());
        assert_eq!(world.mpm_soil_newton_backend(), None);
        assert_eq!(
            world.mpm_soil_admm_backend(),
            Some("experimental-admm-guarded")
        );
        assert_eq!(
            world.mpm_soil_admm_preparation_backend(),
            Some(crate::soil_admm_geometry_gpu::ExperimentalSoilAdmmGeometryGpu::preparation_backend_label())
        );
        assert_eq!(world.mpm_soil_friction_fixed_walls(), Some(true));
        assert_eq!(
            world.mpm_soil_friction_coefficients().unwrap(),
            bindings.iter().map(|b| (b.body, 0.3)).collect::<Vec<_>>()
        );
    } else if cfg!(feature = "soa-scratch-newton") {
        assert_eq!(
            world.mpm_soil_newton_backend(),
            Some(
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-parallel-indices-soa-scratch-256-pairs"
            )
        );
    } else if cfg!(feature = "parallel-indices-newton") {
        assert_eq!(
            world.mpm_soil_newton_backend(),
            Some(
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-parallel-indices-256-pairs"
            )
        );
    } else if cfg!(feature = "grouped-eight-newton") {
        assert_eq!(
            world.mpm_soil_newton_backend(),
            Some(
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-256-pairs"
            )
        );
    } else if cfg!(feature = "grouped-sum-newton") {
        assert_eq!(
            world.mpm_soil_newton_backend(),
            Some(
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-sum-256-pairs"
            )
        );
    } else if cfg!(feature = "stagnant-branch-newton") {
        assert_eq!(
            world.mpm_soil_newton_backend(),
            Some(
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-256-pairs"
            )
        );
    } else if cfg!(feature = "stream-block-newton") {
        assert_eq!(
            world.mpm_soil_newton_backend(),
            Some(
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-256-pairs"
            )
        );
    } else if cfg!(feature = "shared-conditioned-core-newton") {
        assert_eq!(
            world.mpm_soil_newton_backend(),
            Some(
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-256-pairs"
            )
        );
    } else if cfg!(feature = "conditioned-projection-cache-newton") {
        assert_eq!(
            world.mpm_soil_newton_backend(),
            Some(
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-projection-cache-256-pairs"
            )
        );
    }
    assert_eq!(
        world.mpm_soil_defers_native_prediction_drift(),
        cfg!(feature = "deferred-native-drift")
    );
    assert_eq!(
        world.mpm_soil_candidate_index_backend(),
        Some(if cfg!(feature = "parallel-candidate-index") {
            "parallel-stable-radix"
        } else {
            "serial-stable-radix"
        })
    );
    world
        .prepare_mpm_soil_frame_profiling_with_preparation(3)
        .unwrap();
    let budget_rejection = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        world
            .prepare_mpm_soil_frame_profiling_with_preparation(103)
            .is_err()
    );
    assert_eq!(budget_rejection.finish().observations, 0);
    {
        let gpu = world.gpu.as_ref().unwrap();
        let projection = gpu.soil.as_ref().unwrap().projection.as_ref().unwrap();
        assert!(projection.shares_native_operators());
        assert_eq!(projection.native_jacobian_copy_work(), (1, 1200));
        let compiled = gpu.compiled.as_ref().unwrap();
        assert!(projection.matches_operators(compiled.native_mass(), compiled.native_jacobians()));
        assert!(!projection.matches_operators(compiled.native_mass(), None));
        assert!(!projection.matches_operators(None, compiled.native_jacobians()));
        let mut reversed = compiled.native_jacobians().unwrap();
        reversed.swap(0, 1);
        assert!(!projection.matches_operators(compiled.native_mass(), Some(reversed)));
    }
    eprintln!("native world ready {:?}", start.elapsed());
    let mut tangential = 0.0;
    let mut body_impulse_magnitudes = vec![0.0; bindings.len()];
    let mut records = Vec::new();
    for frame in 1..=4 {
        if frame == 3 {
            world.prepare_mpm_soil_frame_profiling(3).unwrap();
        }
        if frame == 4 {
            world
                .prepare_mpm_soil_frame_profiling_with_preparation(3)
                .unwrap();
        }
        let before = machine_momentum_energy(&world.scene).0;
        let angular_before = machine_angular_momentum(&world.scene);
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        world.step(dt).unwrap();
        assert!(
            world
                .gpu
                .as_ref()
                .unwrap()
                .compiled
                .as_ref()
                .unwrap()
                .states
                .iter()
                .all(|s| s.body_f.host().is_err())
        );
        assert_eq!(allocations.finish().observations, 0);
        let (after, mut energy) = machine_momentum_energy(&world.scene);
        let loads = world.read_mpm_soil_frame_loads().unwrap();
        let angular_after = machine_angular_momentum(&world.scene);
        let angular_impulse: Vec3 = loads
            .bodies
            .iter()
            .map(|b| Vec3::from_array(b.world_origin_angular.map(f64::from)))
            .sum();
        let angular_error = (angular_after - angular_before - angular_impulse).length();
        assert_eq!(loads.bodies.len(), bindings.len());
        let body_records: Vec<_> = loads
            .bodies
            .iter()
            .zip(&bindings)
            .zip(&mut body_impulse_magnitudes)
            .map(|((load, binding), magnitude)| {
                assert_eq!(load.body, binding.body);
                *magnitude += Vec3::from_array(load.linear.map(f64::from)).length();
                serde_json::json!({"body":format!("{:?}",load.body),"linear":load.linear,
                    "com_angular":load.angular,"world_origin_angular":load.world_origin_angular})
            })
            .collect();
        std::fs::write(root.join(format!("angular-{frame}.json")), serde_json::to_vec_pretty(&serde_json::json!({
            "before":angular_before.to_array(),"after":angular_after.to_array(),
            "soil_angular_impulse":angular_impulse.to_array(),"residual":angular_error,"bodies":body_records,
        })).unwrap()).unwrap();
        eprintln!("native angular frame={frame} residual={angular_error:e}");
        assert!(
            angular_error < 2e-5,
            "angular impulse error {angular_error:e}"
        );
        let impulse: Vec3 = loads
            .bodies
            .iter()
            .map(|b| Vec3::from_array(b.linear.map(f64::from)))
            .sum();
        tangential += impulse.z.abs();
        let impulse_error = (after - before - impulse).length();
        assert!(impulse_error < 2e-5, "impulse error {impulse_error:e}");
        let velocities = world
            .gpu
            .as_mut()
            .unwrap()
            .soil
            .as_mut()
            .unwrap()
            .soil
            .read_velocities_with(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        let mut total = after;
        for velocity in velocities {
            let velocity = Vec3::from_array(velocity.map(f64::from));
            total += particle_mass * velocity;
            energy += 0.5 * particle_mass * velocity.length_squared();
        }
        let momentum_error = (total - 65.0 * particle_mass * initial_velocity).length();
        assert!(momentum_error < 2e-4, "momentum error {momentum_error:e}");
        assert!(energy <= 0.5 * 65.0 * particle_mass * initial_velocity.length_squared() + 1e-4);
        assert!(
            world
                .prepare_mpm_soil_frame_profiling_with_preparation(3)
                .is_err()
        );
        let profile = world.read_mpm_soil_frame_profile().unwrap().unwrap();
        assert_eq!(profile.profile.preparation_details_enabled, frame != 3);
        assert_eq!(
            profile.profile.preparation_stages.len(),
            if frame == 3 { 0 } else { 24 }
        );
        assert_eq!(
            profile.profile.readback_bytes,
            16 * (profile.profile.stages.len()
                + profile.profile.contact_stages.len()
                + profile.profile.preparation_stages.len()) as u64
        );
        let preparation_details: Vec<_> = profile.profile.preparation_stages.iter().map(|s| serde_json::json!({"stage":s.stage.name(),"substep":s.substep,"start_ms":s.start_ms,"gpu_ms":s.elapsed_ms})).collect();
        std::fs::write(
            root.join(format!("preparation-{frame}.json")),
            serde_json::to_vec_pretty(&preparation_details).unwrap(),
        )
        .unwrap();
        assert_eq!(profile.machine_stages.len(), 10);
        assert!(
            profile
                .machine_stages
                .iter()
                .all(|s| s.elapsed_ms.is_finite() && s.elapsed_ms > 0.0)
        );
        let prediction_ms: f64 = profile
            .machine_stages
            .iter()
            .filter(|s| s.stage == "dynamics")
            .map(|s| s.elapsed_ms)
            .sum();
        assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, frame);
        check_body_render(&world, &engine, frame, 1);
        let saved = checkpoint(&mut world, &engine);
        std::fs::write(root.join(format!("soil-{frame}.bin")), &saved).unwrap();
        let soil_energy = crate::mpm::MpmSoilCheckpoint::from_bytes(&saved, 65)
            .unwrap()
            .energy_at_height(0.0)
            .unwrap();
        assert_eq!(soil_energy.elastic_unavailable_particles, 0);
        let elastic_energy = soil_energy.elastic_potential.unwrap();
        let total_mechanical_energy = energy + soil_energy.affine_kinetic + elastic_energy;
        assert!(
            total_mechanical_energy
                <= 0.5 * 65.0 * particle_mass * initial_velocity.length_squared() + 1e-4
        );
        records.push(serde_json::json!({"frame":frame,"impulse_error":impulse_error,"angular_impulse_error":angular_error,"momentum_error":momentum_error,"energy":energy,"prediction_ms":prediction_ms,
            "soil_affine_kinetic":soil_energy.affine_kinetic,"soil_elastic":elastic_energy,"total_mechanical_energy":total_mechanical_energy}));
        if let Some(force_separate) = force_separate {
            submission::capture(
                &mut world,
                &engine,
                &root,
                &format!("frame-{frame}"),
                &handles,
                Some(&loads),
                force_separate,
            );
        }
    }
    assert!(tangential > 1e-6);
    if fixture == NativeWorldFixture::AllContacts {
        assert!(
            body_impulse_magnitudes.iter().all(|&value| value > 1e-6),
            "missing loaded body: {body_impulse_magnitudes:?}"
        );
    }
    let saved = checkpoint(&mut world, &engine);
    let pose = world.scene.body_pose(handles.chassis).unwrap();
    let velocity = world.scene.body_velocity(handles.chassis).unwrap();
    world
        .scene
        .set_joint_motor(
            handles.axles[0],
            JointAxis::AngX,
            JointMotor {
                mode: JointTargetMode::Velocity,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(world.step(dt).is_err());
    assert_eq!(checkpoint(&mut world, &engine), saved);
    assert_eq!(world.scene.body_pose(handles.chassis).unwrap(), pose);
    assert_eq!(
        world.scene.body_velocity(handles.chassis).unwrap(),
        velocity
    );
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 4);
    if let Some(force_separate) = force_separate {
        submission::capture(
            &mut world,
            &engine,
            &root,
            "rejected",
            &handles,
            None,
            force_separate,
        );
    }
    world
        .scene
        .set_joint_motor(
            handles.axles[0],
            JointAxis::AngX,
            JointMotor {
                mode: JointTargetMode::Effort,
                ..Default::default()
            },
        )
        .unwrap();
    world.step(dt).unwrap();
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 5);
    let recovered_profile = world.read_mpm_soil_frame_profile().unwrap().unwrap();
    assert_eq!(recovered_profile.committed_frame, 5);
    assert_eq!(recovered_profile.profile.preparation_stages.len(), 24);
    std::fs::write(root.join("recovered.bin"), checkpoint(&mut world, &engine)).unwrap();
    if let Some(force_separate) = force_separate {
        let loads = world.read_mpm_soil_frame_loads().unwrap();
        submission::capture(
            &mut world,
            &engine,
            &root,
            "recovered",
            &handles,
            Some(&loads),
            force_separate,
        );
    }
    if admm {
        let saved = checkpoint(&mut world, &engine);
        let machine = super::super::motion_tests::retry_machine_words(&world);
        let pose = world.scene.body_pose(handles.chassis).unwrap();
        let velocity = world.scene.body_velocity(handles.chassis).unwrap();
        let status = world
            .gpu
            .as_ref()
            .unwrap()
            .soil
            .as_ref()
            .unwrap()
            .projection
            .as_ref()
            .unwrap()
            .admm_status_buffers();
        for buffer in &status {
            engine.queue().unwrap().write_buffer(
                buffer,
                0,
                bytemuck::cast_slice(&[4096u32, 2, 0, 0, 0, 0, 0, 0]),
            );
        }
        let error = world.step(dt).unwrap_err().to_string();
        std::fs::write(root.join("injected-admm-fault.txt"), &error).unwrap();
        assert!(
            error.contains("admm-preparation")
                && error.contains("steps, factors, reserved, reserved"),
            "{error}"
        );
        assert!(!error.contains("newton-preparation"), "{error}");
        assert_eq!(checkpoint(&mut world, &engine), saved);
        assert_eq!(
            super::super::motion_tests::retry_machine_words(&world),
            machine
        );
        assert_eq!(world.scene.body_pose(handles.chassis).unwrap(), pose);
        assert_eq!(
            world.scene.body_velocity(handles.chassis).unwrap(),
            velocity
        );
        assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 5);
        assert!(!world.mpm_soil_stats().unwrap().faulted);
        assert_eq!(world.mpm_soil_admm_backend(), None);
        assert_eq!(world.mpm_soil_admm_preparation_backend(), None);
        world.step(dt).unwrap();
        assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 6);
        assert_eq!(
            world.mpm_soil_admm_backend(),
            Some("experimental-admm-guarded")
        );
        assert_eq!(
            world.mpm_soil_admm_preparation_backend(),
            Some(crate::soil_admm_geometry_gpu::ExperimentalSoilAdmmGeometryGpu::preparation_backend_label())
        );
        std::fs::write(root.join("admm-retry.bin"), checkpoint(&mut world, &engine)).unwrap();
    }
    std::fs::write(root.join("result.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "frames":records,"tangential_impulse":tangential,"shared_mass_without_crba":true,
        "fixture":format!("{fixture:?}"),"body_impulse_magnitudes":body_impulse_magnitudes,
        "world_origin_angular_gate":true,
        "full_soil_energy_gate":true,
        "shared_native_com_jacobians_without_legacy_dispatch":true,
        "native_jacobian_copy_commands_per_substep":1,"native_jacobian_copy_bytes_per_substep":1200,
        "control_fault_exact_rollback_and_retry":true,"acceptance":false,"physically_calibrated":false,
        "preparation_profile_enable_disable_pending_capacity_and_retry":true,
        "live_candidate_index_backend":world.mpm_soil_candidate_index_backend(),
        "soil_newton_preparation_backend":crate::soil_newton_prepare_gpu::ExperimentalSoilNewtonPrepareGpu::backend_label(),
        "live_soil_newton_backend":world.mpm_soil_newton_backend(),
        "live_soil_admm_backend":world.mpm_soil_admm_backend(),
        "injected_admm_fault_exact_rollback_and_retry":admm,
        "live_deferred_native_prediction_drift":world.mpm_soil_defers_native_prediction_drift(),
        "live_machine_unrestricted_drift_backend":world.solver.gpu_unrestricted_drift_backend(),
        "copied_force_host_mirrors_invalidated":true,
    })).unwrap()).unwrap();
    eprintln!(
        "native_world_shared_mass_rollback=pass elapsed={:?}",
        start.elapsed()
    );
}
