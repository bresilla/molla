use super::*;

fn grid(sim: &MpmSimulation) -> [&wgpu::Buffer; 6] {
    [
        &sim.g_mass,
        &sim.g_momentum,
        &sim.g_velocity,
        &sim.g_rigid_mask,
        &sim.g_kdiag,
        &sim.g_velocity_prev,
    ]
}

fn encode_scatter(sim: &MpmSimulation, encoder: &mut wgpu::CommandEncoder) {
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        sim.bind_step(&mut pass);
        pass.set_pipeline(&sim.active_pipelines[0]);
        pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
    }
    sim.active_tiles
        .encode_build(encoder, sim.particle_count, sim.rigid_count);
    let mut pass = encoder.begin_compute_pass(&Default::default());
    sim.bind_step(&mut pass);
    sim.encode_p2g(&mut pass);
}

fn encode_coupling(sim: &MpmSimulation, encoder: &mut wgpu::CommandEncoder) {
    let mut pass = encoder.begin_compute_pass(&Default::default());
    sim.bind_step(&mut pass);
    pass.set_pipeline(&sim.pipe_rigid_wrench_reset);
    pass.dispatch_workgroups(sim.rigid_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    for stage in [1, 2, 3, 4, 3, 4, 5] {
        pass.set_pipeline(&sim.active_pipelines[stage]);
        pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
    }
}

#[test]
#[ignore = "needs a real GPU adapter; fixed-input stage diagnostic"]
fn fixed_grid_and_particle_stage_replay() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = seeded(&engine, 2, true);
    sim.step_n_with(device, queue, 0.0001, 2).unwrap();
    sim.step_n_with(device, queue, 0.0002, 2).unwrap();
    queue.write_buffer(
        &sim.uniform_buf,
        0,
        bytemuck::bytes_of(&sim.step_uniforms(0.00008)),
    );
    let particles = [&sim.p_pos, &sim.p_vel, &sim.p_c, &sim.p_f, &sim.p_jp]
        .map(|b| read_words(device, queue, b));
    let saved = grid(&sim).map(|b| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fixed pre-imprint grid"),
            size: b.size(),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encode_scatter(&sim, &mut encoder);
    for (from, to) in grid(&sim).iter().zip(&saved) {
        encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
    }
    queue.submit([encoder.finish()]);
    let mut first_grid: Option<[Vec<u32>; 6]> = None;
    let mut first_wrench: Option<Vec<u32>> = None;
    let mut maximum_wrench = 0_f32;
    for replay in 0..6 {
        let mut encoder = device.create_command_encoder(&Default::default());
        for (from, to) in saved.iter().zip(grid(&sim)) {
            encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
        }
        encode_coupling(&sim, &mut encoder);
        queue.submit([encoder.finish()]);
        let output = grid(&sim).map(|b| read_words(device, queue, b));
        if let Some(first) = &first_grid {
            assert_eq!(&output, first, "fixed-grid replay changed cell outputs");
        } else {
            first_grid = Some(output);
        }
        let wrench = read_words(device, queue, &sim.rigid_wrench);
        let reference = first_wrench.get_or_insert_with(|| wrench.clone());
        let mut different = 0;
        let mut maximum = 0_f32;
        for (&a, &b) in wrench.iter().zip(reference.iter()) {
            let fa = f32::from_bits(a);
            let fb = f32::from_bits(b);
            assert!(fa.is_finite() && fb.is_finite());
            different += usize::from(a != b);
            maximum = maximum.max((fa - fb).abs());
        }
        maximum_wrench = maximum_wrench.max(maximum);
        println!(
            "fixed_grid_replay={replay} grid_bit_identical=true different_wrench_words={different} max_abs_wrench_error={maximum}"
        );
    }
    let mut first_scatter: Option<[Vec<u32>; 6]> = None;
    let mut total_changed = [0_usize; 6];
    let mut scatter_maximum = [0_f32; 6];
    for replay in 0..6 {
        let mut encoder = device.create_command_encoder(&Default::default());
        encode_scatter(&sim, &mut encoder);
        queue.submit([encoder.finish()]);
        let output = grid(&sim).map(|b| read_words(device, queue, b));
        let reference = first_scatter.get_or_insert_with(|| output.clone());
        let mut changed = [0_usize; 6];
        let mut maximum = [0_f32; 6];
        for (field, (a, b)) in output.iter().zip(reference.iter()).enumerate() {
            let count = if field == 4 {
                sim.cfg.grid_size.pow(3) as usize
            } else {
                a.len()
            };
            for (&a, &b) in a[..count].iter().zip(&b[..count]) {
                changed[field] += usize::from(a != b);
                let a = f32::from_bits(a);
                let b = f32::from_bits(b);
                assert!(a.is_finite() && b.is_finite());
                maximum[field] = maximum[field].max((a - b).abs());
            }
            total_changed[field] += changed[field];
            scatter_maximum[field] = scatter_maximum[field].max(maximum[field]);
        }
        println!(
            "fixed_particle_replay={replay} changed_grid_words={changed:?} max_abs_grid_errors={maximum:?}"
        );
    }
    assert_eq!(
        particles,
        [&sim.p_pos, &sim.p_vel, &sim.p_c, &sim.p_f, &sim.p_jp]
            .map(|b| read_words(device, queue, b))
    );
    println!(
        "stage_summary fixed_grid_max_abs_wrench_error={maximum_wrench} fixed_particle_total_changed_grid_words={total_changed:?} fixed_particle_max_abs_grid_errors={scatter_maximum:?}"
    );
    if sim.compensated_grid_transfer_enabled() {
        assert_eq!(total_changed, [0; 6], "compensated fixed-particle replay");
    }
    assert!(
        maximum_wrench <= 2e-5,
        "fixed-input rigid accumulation exceeds 2e-5"
    );
}
