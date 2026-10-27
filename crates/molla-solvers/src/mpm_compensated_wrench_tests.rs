use super::*;
use crate::mpm::replay_tests::read_words;
use molla_core::gpu_metrics::{DeviceScope, TransferScope};

#[test]
fn retained_wrench_source_contract() {
    let source = source().unwrap();
    assert!(!source.contains("wrench_terms"));
    assert!(!source.contains("wrench_term_base"));
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
#[ignore = "requires GPU; retained reaction reduction across grid and body modes"]
fn retained_wrench_lifecycle() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let cfg = MpmConfig {
        grid_size: 16,
        grid_origin: [-0.4, 0.0, -0.4],
        max_particles: 65,
        max_rigid: 2,
        ..MpmConfig::sand()
    };
    let points: Vec<_> = (0..65)
        .map(|i| {
            [
                -0.12 + (i % 5) as f32 * 0.03,
                0.3 + (i / 5 % 5) as f32 * 0.03,
                -0.1 + (i / 25) as f32 * 0.03,
            ]
        })
        .collect();
    for mode in 0..3 {
        let mut sims = [0, 1].map(|_| {
            let mut sim = if mode == 2 {
                MpmSimulation::new_sparse(&engine, cfg, 64).unwrap()
            } else {
                MpmSimulation::new(&engine, cfg).unwrap()
            };
            if mode == 1 {
                sim.set_active_grid(true).unwrap();
            }
            sim.set_particles(queue, &points).unwrap();
            sim.prepare_compensated_grid_transfer(sim.g_mass.size() * 5)
                .unwrap();
            sim.set_compensated_grid_transfer(true).unwrap();
            sim
        });
        let stale = sims[1].prepare_step(0.0001, 1).unwrap();
        let resources = DeviceScope::start().unwrap();
        assert!(sims[1].prepare_compensated_rigid_wrench(63).is_err());
        assert_eq!(resources.finish().observations, 0);
        assert_eq!(sims[1].compensated_rigid_wrench_storage_bytes(), 0);
        assert!(sims[1].set_compensated_rigid_wrench(true).is_err());
        sims[1].prepare_compensated_rigid_wrench(64).unwrap();
        assert!(!sims[1].compensated_rigid_wrench_enabled());
        assert_eq!(sims[1].compensated_rigid_wrench_storage_bytes(), 64);
        let resources = DeviceScope::start().unwrap();
        sims[1].prepare_compensated_rigid_wrench(64).unwrap();
        assert_eq!(resources.finish().observations, 0);
        sims[1].set_compensated_rigid_wrench(true).unwrap();
        assert!(sims[1].release_compensated_rigid_wrench().is_err());
        {
            let mut batch =
                sims[1].begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&stale).is_err());
        }
        for (prescribed, count) in [(false, 2), (true, 2), (true, 0), (false, 1), (true, 2)] {
            for sim in &mut sims {
                sim.set_rigid_spheres(queue, &[]).unwrap();
                sim.set_particles(queue, &points).unwrap();
                let checkpoint = sim.checkpoint_soil_with(queue).unwrap();
                sim.restore_soil_with(queue, &checkpoint).unwrap();
                if prescribed {
                    let solids: Vec<_> = (0..count)
                        .map(|i| MpmPrescribedSolid {
                            shape: if i == 0 {
                                MpmSolidShape::Box {
                                    half_extents: [0.05; 3],
                                }
                            } else {
                                MpmSolidShape::Sphere { radius: 0.06 }
                            },
                            center: [-0.08 + i as f32 * 0.1, 0.35, -0.07],
                            orientation: [0.0, 0.0, 0.0, 1.0],
                            linear_velocity: [0.01, -0.005, 0.0],
                            angular_velocity: [0.0; 3],
                        })
                        .collect();
                    sim.set_prescribed_solids(queue, &solids).unwrap();
                } else {
                    let spheres: Vec<_> = (0..count)
                        .map(|i| {
                            (
                                [-0.08 + i as f32 * 0.1, 0.35, -0.07],
                                0.06,
                                [0.01, -0.005, 0.0],
                            )
                        })
                        .collect();
                    sim.set_rigid_spheres(queue, &spheres).unwrap();
                }
                let record = sim.prepare_step(0.0001, 2).unwrap();
                let encoder = device.create_command_encoder(&Default::default());
                let resources = DeviceScope::start().unwrap();
                let transfers = TransferScope::start().unwrap();
                let mut batch = sim.begin_command_batch(encoder);
                batch.encode_step(&record).unwrap();
                batch.submit(queue);
                assert_eq!(resources.finish().observations, 0);
                let transfers = transfers.finish();
                assert_eq!(transfers.upload_bytes, 0);
                assert_eq!(transfers.readback_bytes, 0);
                assert_eq!(
                    sim.read_rigid_wrenches_with(device, queue).unwrap().len(),
                    count
                );
            }
            for (a, b) in sims[0]
                .checkpoint_buffers()
                .into_iter()
                .zip(sims[1].checkpoint_buffers())
            {
                assert_eq!(read_words(device, queue, a), read_words(device, queue, b));
            }
            let wrenches = sims
                .each_ref()
                .map(|s| read_words(device, queue, &s.rigid_wrench));
            let mut worst = 0_f32;
            for (&a, &b) in wrenches[0][..count * 8]
                .iter()
                .zip(&wrenches[1][..count * 8])
            {
                let a = f32::from_bits(a);
                let b = f32::from_bits(b);
                assert!(a.is_finite() && b.is_finite());
                worst = worst.max((a - b).abs());
            }
            assert!(worst <= 2e-5);
            println!(
                "wrench lifecycle mode={mode} prescribed={prescribed} bodies={count} error={worst} particle_words_exact=true hot_resources=0 transfers=0"
            );
        }
        let stale = sims[1].prepare_step(0.0001, 1).unwrap();
        sims[1].set_compensated_rigid_wrench(false).unwrap();
        sims[1].release_compensated_rigid_wrench().unwrap();
        assert_eq!(sims[1].compensated_rigid_wrench_storage_bytes(), 0);
        let mut batch =
            sims[1].begin_command_batch(device.create_command_encoder(&Default::default()));
        assert!(batch.encode_step(&stale).is_err());
    }
    let mut native = MpmSimulation::new_native_explicit(&engine, cfg).unwrap();
    let resources = DeviceScope::start().unwrap();
    assert!(native.prepare_compensated_rigid_wrench(u64::MAX).is_err());
    assert!(native.set_compensated_rigid_wrench(true).is_err());
    assert_eq!(resources.finish().observations, 0);
}
