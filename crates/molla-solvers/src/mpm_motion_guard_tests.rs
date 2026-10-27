use super::*;
use crate::mpm::replay_tests::read_words;
use wgpu::util::DeviceExt;

#[path = "mpm_motion_transaction_late_tests.rs"]
mod late;

#[path = "mpm_motion_projection_tests.rs"]
mod projection;

#[test]
fn motion_guard_bounds_round_down_and_source_preserves_original_uniform_layout() {
    for value in [
        0.0,
        f64::from_bits(1),
        1e-40,
        1e-6,
        0.1,
        1.0,
        f64::from(f32::MAX),
        f64::MAX,
    ] {
        let bounds = lower_bounds([value; 3]);
        for &bound in &bounds[..3] {
            assert!(bound.is_finite() && bound >= 0.0 && f64::from(bound) <= value);
            assert!(bound == 0.0 || bound.is_normal());
        }
    }
    assert_eq!(std::mem::size_of::<Uniforms>(), 112);
    let source = source();
    assert_eq!(
        source
            .matches("if !soil_motion_admitted(pos, new_pos)")
            .count(),
        1
    );
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    let projection = source
        .split_once("fn soil_motion_plasticity(")
        .unwrap()
        .1
        .split_once("@group(0) @binding(1)")
        .unwrap()
        .0;
    assert!(!projection.contains("p_F[pid]") && !projection.contains("p_jp[pid]"));
    assert!(projection.contains("soil_motion_proposals[pid].deformation = F_new"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let proposal = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("SoilMotionProposal"))
        .unwrap()
        .1;
    assert!(matches!(
        proposal.inner,
        naga::TypeInner::Struct { span: 128, .. }
    ));
    let limits = wgpu::Limits::default();
    assert_eq!(
        MotionGuard::required_bytes(129, &limits).unwrap(),
        16 + 129 * 128
    );
    assert_eq!(MotionGuard::required_bytes(0, &limits).unwrap(), 144);
    assert!(MotionGuard::required_bytes(u32::MAX, &limits).is_err());
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; guarded multiparticle G2P publishes all or no proposals"]
fn motion_guard_gpu_transactional_publication_across_workgroups() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let output = std::env::var_os("MOLLA_MOTION_TRANSACTION_OUTPUT").map(std::path::PathBuf::from);
    if let Some(path) = &output {
        std::fs::create_dir_all(path).unwrap();
    }
    let words = |sim: &MpmSimulation| {
        sim.checkpoint_buffers()
            .iter()
            .map(|buffer| read_words(device, queue, buffer))
            .collect::<Vec<_>>()
    };
    for (mode, implicit) in [
        (0, false),
        (1, false),
        (2, false),
        (0, true),
        (1, true),
        (2, true),
    ] {
        let cfg = MpmConfig {
            max_particles: 129,
            max_rigid: 1,
            grid_size: 64,
            cell_size: 0.1,
            grid_origin: [-3.0, -0.5, -3.0],
            gravity_y: 0.0,
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
        let initial = sim.checkpoint_soil_with(queue).unwrap();
        sim.step_n(&engine, 0.001, 1).unwrap();
        let successful = sim.checkpoint_soil_with(queue).unwrap();
        let moved = read_words(device, queue, &sim.p_pos);
        assert_ne!(moved[0], positions[0][0].to_bits());
        assert_ne!(moved[128 * 4], positions[128][0].to_bits());
        sim.restore_soil_with(queue, &initial).unwrap();
        let resources = DeviceScope::start().unwrap();
        assert!(
            sim.prepare_soil_motion_guard_with_budget(16 + 129 * 128 - 1)
                .is_err()
        );
        assert_eq!(resources.finish().observations, 0);
        assert_eq!(sim.soil_motion_guard_bytes(), 0);
        sim.prepare_soil_motion_guard_with_budget(16 + 129 * 128)
            .unwrap();
        assert_eq!(sim.soil_motion_guard_bytes(), 16 + 129 * 128);
        sim.set_soil_motion_bound(Some([1.0; 3]));
        let step = sim.prepare_step(0.001, 1).unwrap();
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        batch.encode_step(&step).unwrap();
        batch.submit(queue);
        assert_eq!(
            sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
            successful.to_bytes()
        );
        sim.restore_soil_with(queue, &initial).unwrap();
        let before = words(&sim);
        sim.set_soil_motion_bound(Some([0.001; 3]));
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        sim.step_n(&engine, 0.001, 3).unwrap();
        assert_eq!(resources.finish().observations, 0);
        assert_eq!(transfers.finish().readback_bytes, 0);
        let status = sim.read_active_grid_status_with(device, queue).unwrap();
        assert!(status.motion_bound_exceeded && !status.numerical_fault && !status.overflow);
        let rejected = words(&sim);
        assert_eq!(rejected, before);
        assert!(sim.checkpoint_soil_with(queue).is_err());
        sim.set_soil_motion_bound(None);
        sim.step_n(&engine, 0.001, 2).unwrap();
        assert_eq!(words(&sim), before);
        sim.restore_soil_with(queue, &initial).unwrap();
        sim.set_soil_motion_bound(Some([1.0; 3]));
        sim.step_n(&engine, 0.001, 1).unwrap();
        assert_eq!(
            sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
            successful.to_bytes()
        );
        if let Some(path) = &output {
            std::fs::write(
                path.join(format!("mode-{mode}-implicit-{implicit}-initial.soil")),
                initial.to_bytes(),
            )
            .unwrap();
            std::fs::write(
                path.join(format!("mode-{mode}-implicit-{implicit}-successful.soil")),
                successful.to_bytes(),
            )
            .unwrap();
            let raw: Vec<u32> = rejected.into_iter().flatten().collect();
            std::fs::write(
                path.join(format!("mode-{mode}-implicit-{implicit}-rejected.words")),
                bytemuck::cast_slice(&raw),
            )
            .unwrap();
        }
        println!(
            "motion_transaction mode={mode} implicit={implicit} particles=129 stages=proposal,commit first_fault_exact=true success_exact=true retained_bytes={}",
            sim.soil_motion_guard_bytes()
        );
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn motion_guard_gpu_admission_is_conservative_against_f64_displacements() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let bound = [0.1_f64, 1e-6, 1.0];
    let mut cases = vec![
        [0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, f32::from_bits(1), 0.0, 0.0, 0.0],
        [f32::MAX, 0.0, 0.0, 0.0, -f32::MAX, 0.0, 0.0, 0.0],
        [f32::NAN, 0.0, 0.0, 0.0, f32::NAN, 0.0, 0.0, 0.0],
        [f32::INFINITY, 0.0, 0.0, 0.0, f32::INFINITY, 0.0, 0.0, 0.0],
    ];
    let mut seed = 0x17b943af_u32;
    for i in 0..4096 {
        let mut row = [0.0_f32; 8];
        for axis in 0..3 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let old = (seed as i32 as f32) * 1e-8;
            row[axis] = old;
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            row[axis + 4] = old + (sign * bound[axis] * f64::from(i % 11) / 5.0) as f32;
        }
        cases.push(row);
    }
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&cases),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&lower_bounds(bound)),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: cases.len() as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let source = format!(
        "{}\n{}",
        include_str!("../shaders/mpm_motion_guard.wgsl"),
        r#"
struct Row { before: vec4<f32>, after: vec4<f32> }
@group(0) @binding(0) var<storage, read> rows: array<Row>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < arrayLength(&rows) {
        results[gid.x] = select(0u, 1u, soil_motion_admitted(rows[gid.x].before.xyz, rows[gid.x].after.xyz));
    }
}

"#
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            buffer_entry(0, &input),
            buffer_entry(1, &params),
            buffer_entry(2, &output),
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
    }
    queue.submit([encoder.finish()]);
    let results = read_words(device, queue, &output);
    assert_eq!(&results[..7], &[1, 1, 0, 0, 0, 0, 0]);
    let mut admitted = 0;
    for (row, &result) in cases.iter().zip(&results) {
        assert!(result <= 1);
        if result == 1 {
            admitted += 1;
            for axis in 0..3 {
                assert!((f64::from(row[axis + 4]) - f64::from(row[axis])).abs() <= bound[axis]);
            }
        }
    }
    assert!(admitted > 1000 && admitted < 3000);
    println!(
        "motion admission: {} cases, {admitted} admitted; no false acceptance against f64",
        cases.len()
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn motion_guard_gather_latches_preserves_streams_and_prepared_records() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..3 {
        let cfg = MpmConfig {
            max_particles: 1,
            max_rigid: 1,
            ..MpmConfig::sand()
        };
        let mut sim = if mode == 2 {
            MpmSimulation::new_sparse(&engine, cfg, 64).unwrap()
        } else {
            MpmSimulation::new(&engine, cfg).unwrap()
        };
        if mode == 1 {
            sim.set_active_grid(true).unwrap();
        }
        sim.set_particles(queue, &[[0.0, 0.3, 0.0]]).unwrap();
        let initial = sim.checkpoint_soil_with(queue).unwrap();
        sim.step_n(&engine, 0.0001, 3).unwrap();
        let expected = sim.checkpoint_soil_with(queue).unwrap();
        sim.restore_soil_with(queue, &initial).unwrap();
        sim.prepare_soil_motion_guard().unwrap();
        let unguarded = sim.prepare_step(0.0001, 3).unwrap();
        sim.set_soil_motion_bound(Some([0.01; 3]));
        let guarded = sim.prepare_step(0.0001, 3).unwrap();
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        assert!(batch.encode_step(&unguarded).is_err());
        batch.encode_step(&guarded).unwrap();
        batch.submit(queue);
        assert_eq!(
            sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
            expected.to_bytes()
        );
        sim.restore_soil_with(queue, &initial).unwrap();
        sim.set_soil_motion_bound(Some([0.0; 3]));
        let before: Vec<_> = sim
            .checkpoint_buffers()
            .iter()
            .map(|b| read_words(device, queue, b))
            .collect();
        sim.step_n(&engine, 0.0001, 3).unwrap();
        let status = sim.read_active_grid_status_with(device, queue).unwrap();
        assert!(status.motion_bound_exceeded && status.faulted());
        assert!(!status.numerical_fault && !status.overflow);
        assert_eq!(status.completed_substeps, 1);
        assert!(sim.checkpoint_soil_with(queue).is_err());
        sim.set_soil_motion_bound(None);
        sim.step_n(&engine, 0.0001, 2).unwrap();
        assert!(
            sim.read_active_grid_status_with(device, queue)
                .unwrap()
                .motion_bound_exceeded
        );
        let after: Vec<_> = sim
            .checkpoint_buffers()
            .iter()
            .map(|b| read_words(device, queue, b))
            .collect();
        assert_eq!(before, after);
        sim.restore_soil_with(queue, &initial).unwrap();
        sim.step_n(&engine, 0.0001, 3).unwrap();
        assert_eq!(
            sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
            expected.to_bytes()
        );
    }
}
