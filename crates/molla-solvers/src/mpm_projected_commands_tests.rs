use super::*;
use std::cell::Cell;

#[test]
#[ignore = "requires real native GPU; composed projected step parity and rejection"]
fn projected_commands_order_and_parity() {
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for native in [false, true] {
        for layout in 0..3 {
            for unilateral in [false, true] {
                let mut arms = Vec::new();
                for batched in [false, true] {
                    let inputs = ordered::inputs(&engine);
                    let free = &inputs[2];
                    let source = data(&engine, &[0.; 3]);
                    let copied = data(&engine, &[0.; 3]);
                    let delta = data(&engine, &[7., 8., 9.]);
                    let status = data(&engine, &[0.]);
                    let gather = native || layout != 0;
                    let count = if gather { 17 } else { 1 };
                    let mut soil = fixture_layout(&engine, native, count, [0.; 3], layout, 16, 64);
                    ordered::prepare_order(&mut soil, queue, gather);
                    let mut coupling = ordered::coupling(&engine, &mut soil, &inputs, unilateral);
                    let publication = coupling.publication(SoilRatePublicationInputs {
                        rates: free,
                        velocity_delta: &delta,
                        machine_status: &status,
                        max_owned_bytes: 16,
                    });
                    let mut frames = Vec::new();
                    for (step, dt) in [0.0001, 0.0002, 0.00005].into_iter().enumerate() {
                        let rates = [0.01 * step as f32, -9.81 * dt, 0.0];
                        queue.write_buffer(&source, 0, bytemuck::cast_slice(&rates));
                        queue.write_buffer(free, 0, bytemuck::cast_slice(&[5.0f32; 3]));
                        let before = |e: &mut wgpu::CommandEncoder| {
                            e.copy_buffer_to_buffer(&source, 0, free, 0, 12);
                        };
                        let after = |e: &mut wgpu::CommandEncoder| {
                            publication.encode(e);
                            e.copy_buffer_to_buffer(free, 0, &copied, 0, 12);
                        };
                        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
                        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
                        if batched {
                            coupling
                                .step_with_commands(&mut soil, dt, before, after)
                                .unwrap();
                        } else {
                            let mut e = device.create_command_encoder(&Default::default());
                            before(&mut e);
                            queue.submit([e.finish()]);
                            ordered::step(&mut coupling, &mut soil, dt).unwrap();
                            let mut e = device.create_command_encoder(&Default::default());
                            after(&mut e);
                            queue.submit([e.finish()]);
                        }
                        assert_eq!(allocations.finish().observations, 0);
                        let traffic = traffic.finish();
                        assert_eq!(traffic.readback_bytes, 0);
                        assert!(!traffic.overflowed);
                        let raw = crate::soil_shared_wall_friction_gpu::capture::read_many(
                            &engine,
                            &[free, &copied, &delta, &status, coupling.status()],
                        );
                        assert_eq!(raw[0], raw[1], "post-publication copy");
                        assert_eq!(&raw[3], &[0; 4]);
                        assert!(raw[4].chunks_exact(4).next().is_some_and(|x| x == [0; 4]));
                        frames.push((
                            soil.checkpoint_soil_with(queue).unwrap().to_bytes(),
                            raw,
                            traffic.upload_bytes,
                        ));
                    }
                    let callbacks = Cell::new(0);
                    for dt in [0.0, -0.001, f32::NAN, f32::INFINITY] {
                        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
                        assert!(
                            coupling
                                .step_with_commands(
                                    &mut soil,
                                    dt,
                                    |_| callbacks.set(callbacks.get() + 1),
                                    |_| callbacks.set(callbacks.get() + 1)
                                )
                                .is_err()
                        );
                        assert_eq!(traffic.finish().upload_bytes, 0);
                        assert_eq!(callbacks.get(), 0);
                    }
                    soil.set_particles(queue, &[]).unwrap();
                    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
                    assert!(
                        coupling
                            .step_with_commands(
                                &mut soil,
                                0.0001,
                                |_| callbacks.set(callbacks.get() + 1),
                                |_| callbacks.set(callbacks.get() + 1)
                            )
                            .is_err()
                    );
                    assert_eq!(traffic.finish().upload_bytes, 0);
                    assert_eq!(callbacks.get(), 0);
                    arms.push(frames);
                }
                assert_eq!(
                    arms[0], arms[1],
                    "native={native} layout={layout} unilateral={unilateral}"
                );
                println!(
                    "projected commands native={native} layout={layout} unilateral={unilateral}: three states and five outputs exact, no hot allocations/readbacks, invalid/stale rejected before hooks"
                );
            }
        }
    }
}
