use super::*;

const INITIALIZE: &str = "    atomicStore(&wg_kdiag[lid.x], 0u);";
const INIT: &str = "\n    stiffness_lane = lid.x;\n    atomicStore(&wg_stiffness_tail[lid.x], 0u);";
const COMMIT: &str = "atomic_add_kdiag(cidx, bitcast<f32>(atomicLoad(&wg_kdiag[slot])));";
const CORRECTED: &str = "atomic_add_kdiag(cidx, bitcast<f32>(atomicLoad(&wg_kdiag[slot])) + bitcast<f32>(atomicLoad(&wg_stiffness_tail[slot])));";
const GLOBAL_COMMIT: &str = "atomic_add_kdiag(cidx, bitcast<f32>(atomicLoad(&wg_kdiag[slot])));\n    scatter_stiffness_global_tail(cidx, bitcast<f32>(atomicLoad(&wg_stiffness_tail[slot])));";

fn original_global() -> &'static str {
    let s = include_str!("../shaders/mpm.wgsl");
    let start = s.find("fn atomic_add_kdiag(").unwrap();
    let end = start + s[start..].find("\n}\n").unwrap() + 2;
    &s[start..end]
}

fn global_compensated(input: String) -> String {
    let source = compensated(input);
    assert_eq!(source.matches(original_global()).count(), 1);
    assert_eq!(source.matches(CORRECTED).count(), 1);
    source
        .replace(
            original_global(),
            include_str!("../shaders/mpm_stiffness_global.wgsl").trim_end(),
        )
        .replace(CORRECTED, GLOBAL_COMMIT)
}

#[test]
fn stiffness_scatter_global_source_contract() {
    let local = compensated(source());
    let global = global_compensated(source());
    assert_eq!(
        global
            .replace(
                include_str!("../shaders/mpm_stiffness_global.wgsl").trim_end(),
                original_global()
            )
            .replace(GLOBAL_COMMIT, CORRECTED),
        local
    );
    assert_eq!(
        global.matches("var<workgroup>").count(),
        local.matches("var<workgroup>").count()
    );
    assert_eq!(
        global.matches("var<storage").count(),
        local.matches("var<storage").count() + 1
    );
    let module = naga::front::wgsl::parse_str(&global).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

fn original() -> &'static str {
    let s = include_str!("../shaders/mpm.wgsl");
    let start = s.find("fn wg_atomic_add_kdiag(").unwrap();
    let end = start + s[start..].find("\n}\n").unwrap() + 2;
    &s[start..end]
}

fn compensated(mut input: String) -> String {
    for (old, new) in [
        (
            original(),
            include_str!("../shaders/mpm_stiffness_workgroup.wgsl")
                .trim_end()
                .to_owned(),
        ),
        (INITIALIZE, format!("{INITIALIZE}{INIT}")),
        (COMMIT, CORRECTED.to_owned()),
    ] {
        assert_eq!(input.matches(old).count(), 1);
        input = input.replace(old, &new);
    }
    input
}

#[test]
fn stiffness_scatter_compensation_source_contract() {
    let control = source();
    let candidate = compensated(control.clone());
    assert_eq!(
        candidate
            .replace(
                include_str!("../shaders/mpm_stiffness_workgroup.wgsl").trim_end(),
                original()
            )
            .replace(INIT, "")
            .replace(CORRECTED, COMMIT),
        control
    );
    assert_eq!(
        candidate.matches("var<storage").count(),
        control.matches("var<storage").count()
    );
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        control.matches("var<workgroup>").count() + 2
    );
    let module = naga::front::wgsl::parse_str(&candidate).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn stiffness_scatter_committed_residuals_survive_reordered_tail_adds() {
    for offset in 0..64 {
        let values: Vec<_> = (0..1024)
            .map(|i| {
                if (i + offset) % 64 == 0 {
                    4096_f32
                } else {
                    0.0001_f32
                }
            })
            .collect();
        let exact: f64 = values.iter().map(|&v| f64::from(v)).sum();
        let mut sum = 0_f32;
        let mut errors = Vec::new();
        for &value in &values {
            let next = sum + value;
            let error = (sum.max(value) - next) + sum.min(value);
            assert_eq!(
                f64::from(next) + f64::from(error),
                f64::from(sum) + f64::from(value)
            );
            errors.push(error);
            sum = next;
        }
        assert!((f64::from(sum) - exact).abs() > 0.09);
        for reverse in [false, true] {
            let mut errors = errors.clone();
            if reverse {
                errors.reverse();
            }
            let tail: f32 = errors.into_iter().sum();
            assert!((f64::from(sum + tail) - exact).abs() < 0.004);
        }
    }
}

fn run(engine: &Engine, sim: &mut MpmSimulation, probe: &Probe) -> [Vec<u32>; 3] {
    let device = engine.device().unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    sim.initialize_active_grid(&mut encoder, false);
    encoder.clear_buffer(&probe.records, 0, None);
    if let Some(tail) = &probe.tail {
        encoder.clear_buffer(tail, 0, None);
    }
    sim.particle_order.as_ref().unwrap().encode(
        &mut encoder,
        &sim.p_material,
        u64::from(sim.cfg.max_particles) * 4,
    );
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        sim.bind_step(&mut pass);
        pass.set_pipeline(&sim.active_pipelines[0]);
        pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
    }
    sim.active_tiles
        .encode_build(&mut encoder, sim.particle_count, 0);
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        sim.bind_step(&mut pass);
        pass.set_bind_group(1, &probe.binding, &[]);
        pass.set_pipeline(&probe.pipelines[2]);
        pass.dispatch_workgroups(sim.particle_count.div_ceil(64), 1, 1);
        if let Some(resolve) = &probe.resolve {
            pass.set_pipeline(resolve);
            pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
        }
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    diagnostic_grid(engine, sim)
}

#[test]
#[ignore = "requires GPU; bounded non-native compensated scatter comparison"]
fn stiffness_scatter_compensation_50k() {
    let input = std::path::PathBuf::from(std::env::var_os("SOIL_SCATTER_INPUT").unwrap());
    let root = std::path::PathBuf::from(std::env::var_os("SOIL_SCATTER_OUTPUT").unwrap());
    std::fs::create_dir(&root).unwrap();
    let frozen = std::fs::read(input.join("soil.bin")).unwrap();
    let expected_terms = std::fs::read(input.join("contributions.bin")).unwrap();
    let reference: Vec<_> = std::fs::read(input.join("sum.bin"))
        .unwrap()
        .chunks_exact(8)
        .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let gathered: [Vec<f32>; 3] = [0, 1, 2].map(|field| {
        std::fs::read(input.join(format!("gather-{field}.bin")))
            .unwrap()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    });
    assert_eq!(reference.len(), 64_usize.pow(3));
    assert_eq!(expected_terms.len(), 50_000 * 27 * 16);
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = tests::seeded_benchmark(&engine);
    sim.restore_soil_with(
        queue,
        &MpmSoilCheckpoint::from_bytes(&frozen, 50_000).unwrap(),
    )
    .unwrap();
    let mut u = sim.step_uniforms(0.0001);
    u.grid_layout[2] = 0;
    queue.write_buffer(&sim.uniform_buf, 0, bytemuck::bytes_of(&u));
    assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), frozen);
    let global = std::env::var_os("SOIL_SCATTER_GLOBAL").is_some();
    let probes = [false, true].map(|candidate| {
        let dir = root.join(if candidate { "candidate" } else { "control" });
        std::fs::create_dir(&dir).unwrap();
        let s = source();
        Probe::with_source(
            &sim,
            &dir,
            if candidate {
                if global {
                    global_compensated(s)
                } else {
                    compensated(s)
                }
            } else {
                s
            },
        )
    });
    let mut records = Vec::new();
    let mut maxima = [[0_f64; 4]; 2];
    for round in 0..6 {
        for arm in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            let grid = run(&engine, &mut sim, &probes[arm]);
            let mut errors = [0_f64; 4];
            for field in 0..3 {
                assert_eq!(grid[field].len(), gathered[field].len());
                for (&bits, &base) in grid[field].iter().zip(&gathered[field]) {
                    let v = f32::from_bits(bits);
                    assert!(v.is_finite() && base.is_finite());
                    errors[field] = errors[field].max(f64::from((v - base).abs()));
                }
                std::fs::write(
                    root.join(format!("round-{round}-arm-{arm}-field-{field}.bin")),
                    bytemuck::cast_slice(&grid[field]),
                )
                .unwrap();
            }
            for (&bits, &value) in grid[2].iter().zip(&reference) {
                errors[3] = errors[3].max((f64::from(f32::from_bits(bits)) - value).abs());
            }
            let terms = read_words(device, queue, &probes[arm].records);
            assert_eq!(
                bytemuck::cast_slice::<u32, u8>(&terms),
                expected_terms,
                "unchanged per-particle terms"
            );
            for field in 0..4 {
                maxima[arm][field] = maxima[arm][field].max(errors[field]);
            }
            records.push(serde_json::json!({"round":round,"arm":arm,"mass_momentum_stiffness_f64_errors":errors,"terms_exact":true}));
        }
    }
    assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), frozen);
    let passed = maxima[1].iter().all(|&v| v <= 2e-5);
    let report = serde_json::json!({"records":records,"maxima":maxima,"bound":2e-5,"passed":passed,"particle_checkpoint_exact":true,"global_compensation":global,"extra_tail_bytes":probes[1].tail.as_ref().map_or(0, wgpu::Buffer::size),"extra_resolve_dispatches":u32::from(global),"scope":"instrumented non-native scatter on frozen 50k state; no timing, exact scatter repeat or full physics claim"});
    std::fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("stiffness_scatter_compensation={report}");
    assert!(passed, "original field bound and captured-term reference");
}
