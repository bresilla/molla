use super::*;
use crate::mpm::replay_tests::read_words;

#[test]
fn quiescence_projection_cache_layout_and_shader() {
    let limits = wgpu::Limits::default();
    assert_eq!(bytes(129, &limits, 14448).unwrap(), 14448);
    assert!(bytes(129, &limits, 14447).is_err());
    assert!(bytes(u32::MAX, &limits, u64::MAX).is_err());
    let module = naga::front::wgsl::parse_str(&source()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let (_, record) = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some("SoilProjectionRecord"))
        .unwrap();
    assert!(matches!(
        record.inner,
        naga::TypeInner::Struct { span: 112, .. }
    ));
}

fn fixture(engine: &Engine, mode: usize, rheology: MpmRheology) -> MpmSimulation {
    let config = MpmConfig {
        max_particles: 129,
        max_rigid: 1,
        grid_size: 64,
        grid_origin: [-3.0, -0.5, -3.0],
        cell_size: 0.1,
        gravity_y: 0.0,
        time_step: MpmTimeStep::Explicit,
        friction_angle: match rheology {
            MpmRheology::Sand { friction_angle } => friction_angle,
            _ => MpmConfig::sand().friction_angle,
        },
        rheology,
        ..MpmConfig::sand()
    };
    let mut sim = if mode == 2 {
        MpmSimulation::new_sparse(engine, config, 128).unwrap()
    } else {
        MpmSimulation::new(engine, config).unwrap()
    };
    if mode == 1 {
        sim.set_active_grid(true).unwrap();
    }
    sim.set_particles(engine.queue().unwrap(), &vec![[0.0, 0.3, 0.0]; 129])
        .unwrap();
    sim.prepare_soil_motion_guard().unwrap();
    sim.set_soil_motion_bound(Some([0.1; 3]));
    sim.prepare_soil_quiescence(2080).unwrap();
    sim
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; exact cached/uncached constitutive and trajectory parity"]
fn quiescence_gpu_projection_cache_parity_and_lifecycle() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..3 {
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
            let mut cached = fixture(&engine, mode, rheology);
            let mut control = fixture(&engine, mode, rheology);
            let resources = DeviceScope::start().unwrap();
            assert!(cached.prepare_soil_projection_cache(14447).is_err());
            assert_eq!(resources.finish().observations, 0);
            cached.prepare_soil_projection_cache(14448).unwrap();
            assert_eq!(cached.soil_projection_cache_bytes(), 14448);
            let initial = cached.checkpoint_soil_with(queue).unwrap();
            for phase in 0..3 {
                if phase == 1 {
                    let f = [
                        1.03_f32, 0.001, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.97, 0.0,
                    ];
                    for sim in [&mut cached, &mut control] {
                        sim.restore_soil_with(queue, &initial).unwrap();
                        queue.write_buffer(&sim.p_f, 128 * 48, bytemuck::cast_slice(&f));
                        queue.write_buffer(&sim.p_jp, 128 * 4, bytemuck::bytes_of(&0.2_f32));
                    }
                } else if phase == 2 {
                    cached.set_material(10000.0, 0.4);
                    control.set_material(10000.0, 0.4);
                }
                for step in 0..4 {
                    let dt = if step < 2 { 0.0001 } else { 0.0002 };
                    control.step_n(&engine, dt, 1).unwrap();
                    let prepared = cached.prepare_step(dt, 1).unwrap();
                    let resources = DeviceScope::start().unwrap();
                    let transfers = TransferScope::start().unwrap();
                    if step % 2 == 0 {
                        cached.step_n(&engine, dt, 1).unwrap();
                    } else {
                        let mut batch = cached.begin_command_batch(
                            engine
                                .device()
                                .unwrap()
                                .create_command_encoder(&Default::default()),
                        );
                        batch.encode_step(&prepared).unwrap();
                        batch.submit(queue);
                    }
                    assert_eq!(resources.finish().observations, 0);
                    assert_eq!(transfers.finish().readback_bytes, 0);
                    assert_eq!(
                        cached.checkpoint_soil_with(queue).unwrap().to_bytes(),
                        control.checkpoint_soil_with(queue).unwrap().to_bytes(),
                        "mode={mode} material={material} phase={phase} step={step}"
                    );
                }
            }
            cached.disable_soil_projection_cache();
            assert_eq!(cached.soil_projection_cache_bytes(), 0);
            cached.step_n(&engine, 0.0001, 1).unwrap();
            control.step_n(&engine, 0.0001, 1).unwrap();
            assert_eq!(
                cached.checkpoint_soil_with(queue).unwrap().to_bytes(),
                control.checkpoint_soil_with(queue).unwrap().to_bytes()
            );
            println!(
                "projection-cache mode={mode} rheology={material} exact_paired_boundaries=13 particles=129 retained_bytes=14448"
            );
        }
    }
}

#[test]
#[ignore = "requires GPU; cache hits bypass deliberately faulting miss path, invalidations miss"]
fn quiescence_gpu_projection_cache_hit_and_miss_admission() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = fixture(
        &engine,
        2,
        MpmRheology::Sand {
            friction_angle: 0.5,
        },
    );
    sim.prepare_soil_projection_cache(14448).unwrap();
    let initial = sim.checkpoint_soil_with(queue).unwrap();
    let config = sim.cfg;
    let mut poisoned_source = source();
    let marker = "fn soil_project_uncached(gid: vec3<u32>) {";
    assert_eq!(poisoned_source.matches(marker).count(), 1);
    poisoned_source = poisoned_source.replace(
        marker,
        &format!("{marker}\n if gid.x < u.particle_count {{ numerical_fault(); return; }}"),
    );
    let layout = sim
        .motion_guard
        .as_ref()
        .unwrap()
        .quiescence
        .as_ref()
        .unwrap()
        .projection_cache
        .as_ref()
        .unwrap()
        .pipeline
        .get_bind_group_layout(0);
    let poison = ProjectionCache::pipeline(&sim, &layout, poisoned_source);
    for mode in 0..8 {
        sim.set_material(config.youngs, config.friction_angle);
        sim.set_rheology(config.rheology);
        sim.restore_soil_with(queue, &initial).unwrap();
        sim.step_n(&engine, 0.0001, 2).unwrap();
        let cache = sim
            .motion_guard
            .as_mut()
            .unwrap()
            .quiescence
            .as_mut()
            .unwrap()
            .projection_cache
            .as_mut()
            .unwrap();
        let original = std::mem::replace(&mut cache.pipeline, poison.clone());
        let mut dt = 0.0001;
        match mode {
            0 => {}
            1 => queue.write_buffer(&sim.p_f, 128 * 48, bytemuck::bytes_of(&0.9_f32)),
            2 => queue.write_buffer(&sim.p_jp, 128 * 4, bytemuck::bytes_of(&0.2_f32)),
            3 => dt = 0.0002,
            4 => sim.set_material(10000.0, 0.4),
            5 => sim.restore_soil_with(queue, &initial).unwrap(),
            6 => {
                let mut batch =
                    sim.begin_command_batch(device.create_command_encoder(&Default::default()));
                let _ = batch.encoder_mut();
            }
            7 => {
                sim.set_material(12000.0, 0.5);
                let prepared = sim.prepare_step(dt, 1).unwrap();
                let mut batch =
                    sim.begin_command_batch(device.create_command_encoder(&Default::default()));
                batch.encode_step(&prepared).unwrap();
            }
            _ => unreachable!(),
        }
        let before: Vec<_> = sim
            .checkpoint_buffers()
            .iter()
            .map(|b| read_words(device, queue, b))
            .collect();
        sim.step_n(&engine, dt, 1).unwrap();
        let fault = sim
            .read_active_grid_status_with(device, queue)
            .unwrap()
            .numerical_fault;
        assert_eq!(fault, mode != 0, "cache admission mode={mode}");
        let after: Vec<_> = sim
            .checkpoint_buffers()
            .iter()
            .map(|b| read_words(device, queue, b))
            .collect();
        assert_eq!(before, after, "cache admission mode={mode}");
        sim.motion_guard
            .as_mut()
            .unwrap()
            .quiescence
            .as_mut()
            .unwrap()
            .projection_cache
            .as_mut()
            .unwrap()
            .pipeline = original;
        println!(
            "projection-cache admission={mode} injected_miss_fault={fault} physical_state_exact=true"
        );
    }
}
