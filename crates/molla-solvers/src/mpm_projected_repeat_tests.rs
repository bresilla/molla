use super::*;

#[test]
#[ignore = "requires real GPU; targeted submission/control repeatability capture"]
fn projected_commands_reference_repeats() {
    let root = std::path::PathBuf::from(std::env::var("MOLLA_PROJECTED_REPEAT_CAPTURE").unwrap());
    std::fs::create_dir(&root).unwrap();
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut captures = Vec::new();
    for (arm, batched) in [false, false, true, false, true, false]
        .into_iter()
        .enumerate()
    {
        let inputs = ordered::inputs(&engine);
        let free = &inputs[2];
        let source = data(&engine, &[0.; 3]);
        let copied = data(&engine, &[0.; 3]);
        let delta = data(&engine, &[7., 8., 9.]);
        let status = data(&engine, &[0.]);
        let mut soil = fixture_layout(&engine, false, 17, [0.; 3], 1, 16, 64);
        ordered::prepare_order(&mut soil, queue, true);
        let mut coupling = ordered::coupling(&engine, &mut soil, &inputs, false);
        let publication = coupling.publication(SoilRatePublicationInputs {
            rates: free,
            velocity_delta: &delta,
            machine_status: &status,
            max_owned_bytes: 16,
        });
        let mut previous = soil.checkpoint_soil_with(queue).unwrap().to_bytes();
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
            let FaultTestCoupling::Sticking(owner) = &coupling else {
                unreachable!()
            };
            let raw = crate::soil_shared_wall_friction_gpu::capture::read_many(
                &engine,
                &[
                    free,
                    &copied,
                    &delta,
                    &status,
                    coupling.status(),
                    &owner.nodes,
                    &owner.spans,
                ],
            );
            assert_eq!(raw[0], raw[1]);
            assert_eq!(raw[3], [0; 4]);
            assert_eq!(raw[4], [0; 4]);
            let spans: Vec<u32> = bytemuck::cast_slice(&raw[6]).to_vec();
            assert_eq!(spans.len(), 2);
            assert_eq!(spans[0], 0);
            assert!(spans[1] > 0 && spans[1] <= 128);
            let active = &raw[5][..spans[1] as usize * 32];
            let mut sorted: Vec<_> = active.chunks_exact(32).collect();
            sorted.sort_unstable();
            let sorted = sorted.concat();
            let current = soil.checkpoint_soil_with(queue).unwrap().to_bytes();
            let directory = root.join(format!("arm-{arm}-step-{step}"));
            std::fs::create_dir(&directory).unwrap();
            for (name, bytes) in [
                ("before.bin", previous.as_slice()),
                ("after.bin", current.as_slice()),
                ("nodes.bin", active),
                ("sorted-nodes.bin", sorted.as_slice()),
            ] {
                std::fs::write(directory.join(name), bytes).unwrap();
            }
            for (name, bytes) in [
                "rates.bin",
                "copied.bin",
                "delta.bin",
                "status.bin",
                "soil-status.bin",
                "node-capacity.bin",
                "spans.bin",
            ]
            .into_iter()
            .zip(&raw)
            {
                std::fs::write(directory.join(name), bytes).unwrap();
            }
            let hash = |b: &[u8]| blake3::hash(b).to_hex().to_string();
            let row = serde_json::json!({
                "arm": arm, "batched": batched, "step": step, "dt_bits": dt.to_bits(),
                "before": hash(&previous), "after": hash(&current),
                "nodes": hash(active), "sorted_nodes": hash(&sorted),
                "rates": hash(&raw[0]), "node_count": spans[1],
            });
            println!("projected-repeat {row}");
            captures.push(row);
            previous = current;
        }
    }
    std::fs::write(
        root.join("capture.json"),
        serde_json::to_vec_pretty(&captures).unwrap(),
    )
    .unwrap();
    println!(
        "Captured 12 reference and 6 composed states; completion is diagnostic, not an exact parity pass"
    );
}
