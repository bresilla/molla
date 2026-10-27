use super::*;
use crate::mpm::replay_tests::read_words;

#[test]
fn motion_fusion_source_preserves_proposals_and_separate_commit() {
    let source = fused_source();
    let appended = "\n@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)\nfn soil_motion_gather_projection(@builtin(global_invocation_id) gid: vec3<u32>) {\n    fused_g2p(gid);\n    fused_soil_motion_plasticity(gid);\n}\n";
    let mut original = source.strip_suffix(appended).unwrap().to_owned();
    for entry in ["g2p", "soil_motion_plasticity"] {
        original = original.replace(&format!("fn fused_{entry}(gid: vec3<u32>)"),
            &format!("@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)\nfn {entry}(@builtin(global_invocation_id) gid: vec3<u32>)"));
    }
    assert_eq!(original, super::super::source());
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(
        module
            .entry_points
            .iter()
            .any(|p| p.name == "soil_motion_commit")
    );
    assert!(
        !module
            .entry_points
            .iter()
            .any(|p| matches!(p.name.as_str(), "g2p" | "soil_motion_plasticity"))
    );
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; fused guarded proposals preserve exact state, faults, reuse and profiling"]
fn motion_fusion_gpu_parity_faults_and_lifecycle() {
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
    )
    .unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let output = std::path::PathBuf::from(
        std::env::var_os("MOLLA_MOTION_FUSION_OUTPUT").expect("explicit fusion capture"),
    );
    std::fs::create_dir(&output).unwrap();
    let raw = |sim: &MpmSimulation| {
        sim.checkpoint_buffers()
            .iter()
            .map(|b| read_words(device, queue, b))
            .collect::<Vec<_>>()
    };
    for (mode, implicit) in [(0, false), (1, false), (2, false), (2, true)] {
        for (material, rheology) in [
            MpmRheology::Sand {
                friction_angle: 0.5,
            },
            MpmRheology::Snow {
                theta_c: 0.025,
                theta_s: 0.0075,
                hardening: 10.0,
            },
            MpmRheology::Viscous {
                velocity_damping: 3.0,
                affine_damping: 5.0,
            },
        ]
        .into_iter()
        .enumerate()
        {
            let cfg = MpmConfig {
                max_particles: 129,
                max_rigid: 1,
                grid_size: 64,
                cell_size: 0.1,
                grid_origin: [-3.0, -0.5, -3.0],
                youngs: 1000.0,
                rheology,
                time_step: if implicit {
                    MpmTimeStep::Implicit { jacobi_iters: 2 }
                } else {
                    MpmTimeStep::Explicit
                },
                ..MpmConfig::sand()
            };
            let mut sim = if mode == 2 {
                MpmSimulation::new_sparse(&engine, cfg, 128).unwrap()
            } else {
                MpmSimulation::new(&engine, cfg).unwrap()
            };
            if mode == 1 {
                sim.set_active_grid(true).unwrap();
            }
            let mut positions = vec![[-0.5, 0.3, 0.0]; 129];
            positions[128] = [0.5, 0.3, 0.0];
            sim.set_particles(queue, &positions).unwrap();
            let mut velocities = vec![[0.1_f32, 0.0, 0.0, 0.0]; 129];
            velocities[128][0] = 10.0;
            queue.write_buffer(&sim.p_vel, 0, bytemuck::cast_slice(&velocities));
            let f = [
                0.97_f32, 0.001, 0.0, 0.0, 0.0, 1.01, 0.002, 0.0, 0.001, 0.0, 0.98, 0.0,
            ];
            queue.write_buffer(&sim.p_f, 0, bytemuck::cast_slice(&vec![f; 129]));
            queue.write_buffer(&sim.p_jp, 0, bytemuck::cast_slice(&vec![0.2_f32; 129]));
            let scope = DeviceScope::start().unwrap();
            assert!(sim.set_soil_motion_projection_fused(true).is_err());
            sim.set_soil_motion_projection_fused(false).unwrap();
            assert_eq!(scope.finish().observations, 0);
            sim.prepare_soil_motion_guard().unwrap();
            assert!(sim.set_soil_motion_projection_fused(true).is_err());
            sim.set_soil_motion_bound(Some([0.1; 3]));
            let bytes = sim.soil_motion_guard_bytes();
            let initial = sim.checkpoint_soil_with(queue).unwrap();
            sim.step_n(&engine, 0.0001, 1).unwrap();
            let control = sim.checkpoint_soil_with(queue).unwrap();
            sim.restore_soil_with(queue, &initial).unwrap();
            let count = DispatchScope::start().unwrap();
            sim.step_n(&engine, 0.0001, 1).unwrap();
            let control_dispatches = count.finish().encoded_dispatches;
            assert_eq!(
                control.to_bytes(),
                sim.checkpoint_soil_with(queue).unwrap().to_bytes()
            );
            sim.restore_soil_with(queue, &initial).unwrap();
            let stale = sim.prepare_step(0.0001, 1).unwrap();
            let scope = DeviceScope::start().unwrap();
            sim.set_soil_motion_projection_fused(true).unwrap();
            assert_eq!(scope.finish().observations, 0);
            assert_eq!(sim.soil_motion_guard_bytes(), bytes);
            let mut batch =
                sim.begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&stale).is_err());
            drop(batch);
            let epoch = sim.step_epoch.clone();
            sim.set_soil_motion_projection_fused(true).unwrap();
            assert!(Arc::ptr_eq(&epoch, &sim.step_epoch));
            let resources = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            let count = DispatchScope::start().unwrap();
            sim.step_n(&engine, 0.0001, 1).unwrap();
            let fused_dispatches = count.finish().encoded_dispatches;
            assert_eq!(resources.finish().observations, 0);
            assert_eq!(transfers.finish().readback_bytes, 0);
            assert_eq!(fused_dispatches + 1, control_dispatches);
            let candidate = sim.checkpoint_soil_with(queue).unwrap();
            assert_eq!(
                candidate.to_bytes(),
                control.to_bytes(),
                "mode={mode} material={material} implicit={implicit}"
            );
            for repeat in 0..2 {
                sim.restore_soil_with(queue, &initial).unwrap();
                sim.step_n(&engine, 0.0001, 1).unwrap();
                assert_eq!(
                    candidate.to_bytes(),
                    sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
                    "repeat={repeat}"
                );
            }
            sim.restore_soil_with(queue, &initial).unwrap();
            sim.set_soil_motion_bound(Some([0.001; 3]));
            let before = raw(&sim);
            sim.step_n(&engine, 0.001, 2).unwrap();
            let fault = sim.read_active_grid_status_with(device, queue).unwrap();
            assert!(fault.motion_bound_exceeded && !fault.overflow);
            assert_eq!(raw(&sim), before);
            sim.restore_soil_with(queue, &initial).unwrap();
            sim.set_soil_motion_bound(Some([0.1; 3]));
            queue.write_buffer(&sim.p_jp, 128 * 4, &f32::INFINITY.to_le_bytes());
            let before = raw(&sim);
            sim.step_n(&engine, 0.0001, 2).unwrap();
            assert!(
                sim.read_active_grid_status_with(device, queue)
                    .unwrap()
                    .numerical_fault
            );
            assert_eq!(raw(&sim), before);
            sim.restore_soil_with(queue, &initial).unwrap();
            sim.prepare_step_profiling(1).unwrap();
            sim.step_n_profiled_with(device, queue, 0.0001, 1).unwrap();
            let profile = sim.read_step_profile().unwrap().unwrap();
            for stage in [
                MpmProfileStage::GuardedGatherProjection,
                MpmProfileStage::GuardedParticleCommit,
            ] {
                assert_eq!(
                    profile.stages.iter().filter(|s| s.stage == stage).count(),
                    1
                );
            }
            assert!(!profile.stages.iter().any(|s| matches!(
                s.stage,
                MpmProfileStage::GridToParticle | MpmProfileStage::Plasticity
            )));
            assert_eq!(
                candidate.to_bytes(),
                sim.checkpoint_soil_with(queue).unwrap().to_bytes()
            );
            sim.restore_soil_with(queue, &initial).unwrap();
            if mode == 2 && material == 0 && !implicit {
                sim.prepare_soil_quiescence(u64::MAX).unwrap();
                assert!(sim.prepare_soil_projection_cache(u64::MAX).is_err());
                sim.set_soil_motion_projection_fused(false).unwrap();
                sim.prepare_soil_projection_cache(u64::MAX).unwrap();
                assert!(sim.set_soil_motion_projection_fused(true).is_err());
                assert!(!sim.soil_motion_projection_fused());
                sim.disable_soil_projection_cache();
                sim.set_soil_motion_projection_fused(true).unwrap();
                sim.step_n(&engine, 0.0001, 1).unwrap();
                assert_eq!(
                    candidate.to_bytes(),
                    sim.checkpoint_soil_with(queue).unwrap().to_bytes()
                );
                sim.restore_soil_with(queue, &initial).unwrap();
                sim.set_soil_motion_bound(None);
                assert!(sim.soil_motion_projection_fused());
                assert_eq!(
                    sim.particle_publication_stages(),
                    [MpmProfileStage::GridToParticle, MpmProfileStage::Plasticity]
                );
                sim.set_soil_motion_bound(Some([0.1; 3]));
            }
            sim.set_soil_motion_projection_fused(false).unwrap();
            sim.step_n(&engine, 0.0001, 1).unwrap();
            assert_eq!(
                control.to_bytes(),
                sim.checkpoint_soil_with(queue).unwrap().to_bytes()
            );
            let prefix = format!("mode-{mode}-material-{material}-implicit-{implicit}");
            std::fs::write(
                output.join(format!("{prefix}-control.soil")),
                control.to_bytes(),
            )
            .unwrap();
            std::fs::write(
                output.join(format!("{prefix}-candidate.soil")),
                candidate.to_bytes(),
            )
            .unwrap();
            println!(
                "motion_fusion {prefix} exact=true repeats=2 faults=2 dispatches={control_dispatches}/{fused_dispatches} bytes={bytes}"
            );
        }
    }
}
