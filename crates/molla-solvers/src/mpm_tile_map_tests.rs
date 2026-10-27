use super::super::replay_tests::read_words;
use super::*;
use std::collections::BTreeSet;

fn hash(id: u32) -> u32 {
    let value = (id ^ (id >> 16)).wrapping_mul(0x7feb352d);
    let value = (value ^ (value >> 15)).wrapping_mul(0x846ca68b);
    value ^ (value >> 16)
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn sparse_hash_collisions_retirement_and_overflow() {
    let engine = Engine::new_gpu().expect("GPU required for sparse hash test");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 64;
    cfg.max_particles = 128;
    cfg.max_rigid = 1;
    let mut sim = MpmSimulation::new_sparse(&engine, cfg, 16).unwrap();
    sim.set_particles(queue, &[[0.1; 3]]).unwrap();
    sim.step_n_with(device, queue, 0.0001, 1).unwrap();
    let source = format!(
        "{}\n@compute @workgroup_size(64) fn test_insert(@builtin(global_invocation_id) gid: vec3<u32>) {{ if gid.x < params.counts.x {{ mark_sparse_tile(bitcast<u32>(positions[gid.x].x)); }} }}",
        include_str!("../shaders/mpm_active_tiles.wgsl")
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MPM collision insertion"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let bindings = sim.active_tiles.pipelines[0].get_bind_group_layout(0);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MPM collision insertion"),
        bind_group_layouts: &[Some(&bindings)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MPM collision insertion"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("test_insert"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[("SPARSE_MAP", 1.0)],
            ..Default::default()
        },
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.clear_buffer(&sim.active_tiles.list, 0, None);
    queue.submit([encoder.finish()]);
    let collisions: Vec<_> = (1..4096u32)
        .filter(|&id| hash(id) & 31 == 31)
        .take(32)
        .collect();
    assert_eq!(collisions.len(), 32);
    for phase in 0..3 {
        let unique: Vec<_> = if phase < 2 {
            std::iter::once(0)
                .chain(collisions[phase * 15..phase * 15 + 15].iter().copied())
                .collect()
        } else {
            collisions[..17].to_vec()
        };
        let inputs: Vec<[u32; 4]> = unique
            .iter()
            .cycle()
            .take(unique.len() * 4)
            .map(|&id| [id, 0, 0, 0])
            .collect();
        queue.write_buffer(&sim.p_pos, 0, bytemuck::cast_slice(&inputs));
        sim.active_tiles
            .upload(queue, &cfg, inputs.len() as u32, 0, [0; 3])
            .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(
            &sim.active_tiles.list,
            sim.active_tiles.metadata_offset,
            Some(4),
        );
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &sim.active_tiles.bindings, &[]);
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
            pass.set_pipeline(&sim.active_tiles.pipelines[2]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let words = read_words(device, queue, &sim.active_tiles.list);
        let header = (sim.active_tiles.metadata_offset / 4) as usize;
        let base = header + 4;
        assert_eq!(words[header], unique.len() as u32);
        assert_eq!(words[header + 2] != 0, phase == 2);
        if phase == 2 {
            assert!(sim.read_active_tile_ids_with(device, queue).is_err());
            break;
        }
        let snapshot = sim.read_active_tile_ids_with(device, queue).unwrap();
        assert_eq!(snapshot.len(), unique.len());
        assert_eq!(
            snapshot.iter().copied().collect::<BTreeSet<_>>(),
            unique.iter().copied().collect()
        );
        assert_eq!(snapshot, words[base + 64..base + 64 + 16]);
        let mut ids = BTreeSet::new();
        let mut slots = BTreeSet::new();
        for bucket in 0..32 {
            let key = words[base + bucket];
            if key == 0 {
                continue;
            }
            assert!(ids.insert(key - 1));
            let slot = words[base + 32 + bucket];
            assert!(slot > 0 && slot <= 16 && slots.insert(slot));
            assert_eq!(words[base + 64 + slot as usize - 1], key - 1);
            assert_eq!(words[base + 64 + 16 + slot as usize - 1], bucket as u32);
        }
        assert_eq!(ids, unique.iter().copied().collect());
        assert_eq!(slots.len(), 16);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            sim.bind_step(&mut pass);
            pass.set_pipeline(&sim.active_pipelines[0]);
            pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
        }
        queue.submit([encoder.finish()]);
        let cleared = read_words(device, queue, &sim.active_tiles.list);
        assert!(cleared[base..base + 32].iter().all(|&key| key == 0));
        println!("phase={phase} colliding_ids=15 unique_slots=16 retired_keys=16");
    }
    sim.set_particles(queue, &[[0.1; 3]]).unwrap();
    sim.reset_active_grid();
    sim.step_n_with(device, queue, 0.0001, 1).unwrap();
    assert!(
        !sim.read_active_grid_stats_with(device, queue)
            .unwrap()
            .overflow
    );
}
