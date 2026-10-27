use super::*;
use crate::mpm::replay_tests::read_words;

const ADD: &str = "stiffness_sum += w * w * gather_particles[pid].stiffness;";
const COMPENSATED: &str = "let term = w * w * gather_particles[pid].stiffness;
                        let next = stiffness_sum + term;
                        stiffness_tail += select((stiffness_sum - next) + term,
                            (term - next) + stiffness_sum, abs(term) > abs(stiffness_sum));
                        stiffness_sum = next;";
const ROUNDED: &str = "let term = w * w * gather_particles[pid].stiffness;
                        let next = stiffness_round(lane, stiffness_sum + term);
                        let large = max(stiffness_sum, term);
                        let small = min(stiffness_sum, term);
                        let error = stiffness_round(lane, stiffness_round(lane, large - next) + small);
                        stiffness_tail = stiffness_round(lane, stiffness_tail + error);
                        stiffness_sum = next;";
const ROUNDING: &str = "
var<workgroup> stiffness_rounding: array<atomic<u32>, 64>;
fn stiffness_round(lane: u32, value: f32) -> f32 {
    atomicStore(&stiffness_rounding[lane], bitcast<u32>(value));
    return bitcast<f32>(atomicAdd(&stiffness_rounding[lane], 0u));
}
";

fn source() -> String {
    let original = include_str!("../shaders/mpm_gather_masks.wgsl");
    assert_eq!(original.matches(ADD).count(), 1);
    original
        .replace(
            "var stiffness_sum = 0.0;",
            "var stiffness_sum = 0.0;\n    var stiffness_tail = 0.0;",
        )
        .replace(ADD, COMPENSATED)
        .replace(
            "select(stiffness_sum, 0.0,",
            "select(stiffness_sum + stiffness_tail, 0.0,",
        )
}

fn rounded_source() -> String {
    format!("{}{ROUNDING}", source().replace(COMPENSATED, ROUNDED))
}

#[test]
fn gather_stiffness_source_and_compensation_contract() {
    let candidate = source();
    assert_eq!(
        candidate
            .replace("\n    var stiffness_tail = 0.0;", "")
            .replace(COMPENSATED, ADD)
            .replace(
                "select(stiffness_sum + stiffness_tail, 0.0,",
                "select(stiffness_sum, 0.0,"
            ),
        include_str!("../shaders/mpm_gather_masks.wgsl")
    );
    for base in [
        include_str!("../shaders/mpm.wgsl").to_owned(),
        native_atomics::shader_source(),
    ] {
        let shader = format!(
            "{base}\n{}\n{candidate}",
            include_str!("../shaders/mpm_deterministic_p2g.wgsl")
        );
        let module = naga::front::wgsl::parse_str(&shader).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
    let mut sum = 0_f32;
    let mut tail = 0_f32;
    let mut exact = 0_f64;
    for term in std::iter::once(4096_f32).chain(std::iter::repeat_n(0.0001_f32, 4096)) {
        let next = sum + term;
        tail += if term.abs() > sum.abs() {
            (term - next) + sum
        } else {
            (sum - next) + term
        };
        sum = next;
        exact += f64::from(term);
    }
    assert!((f64::from(sum) - exact).abs() > 0.4);
    assert!((f64::from(sum + tail) - exact).abs() < 0.00025);
}

#[test]
fn gather_stiffness_rounded_source_contract() {
    let rounded = rounded_source();
    assert_eq!(rounded, compensated_gather::source().unwrap());
    assert_eq!(
        rounded
            .trim_end_matches(ROUNDING)
            .replace(ROUNDED, COMPENSATED),
        source()
    );
    for base in [
        include_str!("../shaders/mpm.wgsl").to_owned(),
        native_atomics::shader_source(),
    ] {
        let shader = format!(
            "{base}\n{}\n{rounded}",
            include_str!("../shaders/mpm_deterministic_p2g.wgsl")
        );
        let module = naga::front::wgsl::parse_str(&shader).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

fn rounding_probe(engine: &Engine, root: &std::path::Path) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let addition = ROUNDED.replace("w * w * gather_particles[pid].stiffness", "increment");
    let shader = format!(
        "{ROUNDING}
@group(0) @binding(0) var<storage, read_write> output: array<vec2<f32>>;
@compute @workgroup_size(64)
fn probe(@builtin(local_invocation_index) lane: u32) {{
    var plain = 4096.0; var stiffness_sum = 4096.0; var stiffness_tail = 0.0;
    let increment = 0.0001 * f32(lane + 1u);
    for (var i = 0u; i < 4096u; i++) {{ plain += increment; {addition} }}
    output[lane] = vec2<f32>(plain, stiffness_sum + stiffness_tail);
}}"
    );
    std::fs::write(root.join("rounding-probe.wgsl"), &shader).unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stiffness rounding probe"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("stiffness rounding probe"),
        layout: None,
        module: &shader,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 512,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[buffer_entry(0, &buffer)],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &binding, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
    let words = read_words(device, queue, &buffer);
    std::fs::write(
        root.join("rounding-probe.bin"),
        bytemuck::cast_slice(&words),
    )
    .unwrap();
    let mut errors = [0_f64; 2];
    for (lane, pair) in words.chunks_exact(2).enumerate() {
        let expected = 4096.0 + 4096.0 * f64::from(0.0001_f32 * (lane + 1) as f32);
        for field in 0..2 {
            let value = f32::from_bits(pair[field]);
            assert!(value.is_finite());
            errors[field] = errors[field].max((f64::from(value) - expected).abs());
        }
    }
    std::fs::write(
        root.join("rounding-probe.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"maximum_errors":errors,"compensated_bound":0.00025}),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        errors[0] > 0.4 && errors[1] < 0.00025,
        "GPU compensation survives compilation"
    );
}

fn reference(engine: &Engine, sim: &MpmSimulation, root: &std::path::Path) -> Vec<f64> {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let streams = [&sim.p_pos, &sim.p_volume, &sim.p_jp, &sim.p_material]
        .map(|b| read_words(device, queue, b));
    for (name, words) in ["positions", "volume", "log-jp", "material"]
        .into_iter()
        .zip(&streams)
    {
        std::fs::write(
            root.join(format!("{name}.bin")),
            bytemuck::cast_slice(words),
        )
        .unwrap();
    }
    let [positions, volumes, jp, materials] = streams;
    let u = sim.step_uniforms(0.0001);
    std::fs::write(root.join("uniforms.bin"), bytemuck::bytes_of(&u)).unwrap();
    let mut result = vec![0_f64; sim.cfg.grid_size.pow(3) as usize];
    let weights = |x: f32| {
        [
            0.5 * (1.5 - x) * (1.5 - x),
            0.75 - (x - 1.0) * (x - 1.0),
            0.5 * (x - 0.5) * (x - 0.5),
        ]
    };
    for p in 0..sim.particle_count as usize {
        assert_eq!(materials[p], 0, "sand-only reference fixture");
        let xp = [0, 1, 2]
            .map(|a| (f32::from_bits(positions[p * 4 + a]) - u.grid_origin[a]) * u.inv_cell_size);
        let base = xp.map(|v| (v - 0.5).floor());
        let w = [0, 1, 2].map(|a| weights(xp[a] - base[a]));
        let mu = u.youngs / (2.0 * (1.0 + u.poisson));
        let lam = u.youngs * u.poisson / ((1.0 + u.poisson) * (1.0 - 2.0 * u.poisson));
        let h = f32::from_bits(jp[p]).clamp(-3.0, 3.0).exp();
        let k = 4.0
            * f32::from_bits(volumes[p])
            * (2.0 * (mu * h) + lam * h)
            * (u.inv_cell_size * u.inv_cell_size);
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    let xyz = [x, y, z];
                    let cell = [0, 1, 2].map(|a| base[a] as i32 + xyz[a] as i32);
                    if cell.iter().any(|&c| c < 0 || c >= u.grid_size as i32) {
                        continue;
                    }
                    let weight = w[0][x] * w[1][y] * w[2][z];
                    let term = weight * weight * k;
                    assert!(term.is_finite() && term >= 0.0);
                    let index = cell[0] as usize
                        + u.grid_size as usize
                            * (cell[1] as usize + u.grid_size as usize * cell[2] as usize);
                    result[index] += f64::from(term);
                }
            }
        }
    }
    std::fs::write(
        root.join("rounded-term-f64-reference.bin"),
        bytemuck::cast_slice(&result),
    )
    .unwrap();
    result
}

#[test]
#[ignore = "requires GPU; frozen 50k non-native stiffness comparison"]
fn gather_stiffness_compensated_50k() {
    let root = std::path::PathBuf::from(
        std::env::var_os("SOIL_STIFFNESS_OUTPUT").expect("fresh evidence path"),
    );
    std::fs::create_dir(&root).unwrap();
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = tests::seeded_benchmark(&engine);
    let rounded = std::env::var_os("SOIL_STIFFNESS_ROUNDED").is_some();
    let candidate_source = if rounded {
        rounding_probe(&engine, &root);
        rounded_source()
    } else {
        source()
    };
    let candidate = Gather::new_variant_source(&sim, true, false, true, &candidate_source).unwrap();
    std::fs::write(root.join("candidate.wgsl"), candidate_source).unwrap();
    let before = sim.checkpoint_soil_with(queue).unwrap().to_bytes();
    std::fs::write(root.join("soil.bin"), &before).unwrap();
    let oracle = reference(&engine, &sim, &root);
    let mut first: [Option<[Vec<u32>; 3]>; 3] = Default::default();
    let mut records = Vec::new();
    for round in 0..6 {
        for mode in if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
            let mut encoder = device.create_command_encoder(&Default::default());
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
                match mode {
                    0 => {
                        pass.set_pipeline(&sim.pipe_p2g_two_pass);
                        pass.dispatch_workgroups(sim.particle_count.div_ceil(64), 1, 1);
                    }
                    1 => sim
                        .deterministic_p2g
                        .as_ref()
                        .unwrap()
                        .encode(&mut pass, &sim),
                    2 => candidate.encode(&mut pass, &sim),
                    _ => unreachable!(),
                }
            }
            queue.submit([encoder.finish()]);
            let grid = tests::world_grid(&engine, &sim);
            assert!(
                grid.iter()
                    .flatten()
                    .all(|&v| f32::from_bits(v).is_finite())
            );
            let oracle_error = grid[2]
                .iter()
                .zip(&oracle)
                .map(|(&v, &r)| (f64::from(f32::from_bits(v)) - r).abs())
                .fold(0_f64, f64::max);
            let errors = first[1].as_ref().map(|control| {
                [0, 1, 2].map(|field| {
                    grid[field]
                        .iter()
                        .zip(&control[field])
                        .map(|(&a, &b)| {
                            let (a, b) = (f32::from_bits(a), f32::from_bits(b));
                            assert!(a.is_finite() && b.is_finite());
                            (a - b).abs()
                        })
                        .fold(0_f32, f32::max)
                })
            });
            let repeat_exact = first[mode].as_ref().map(|old| *old == grid);
            if mode != 0
                && let Some(old) = &first[mode]
            {
                assert_eq!(*old, grid, "fixed-order self repeat");
            }
            for (field, words) in grid.iter().enumerate() {
                std::fs::write(
                    root.join(format!("round-{round}-mode-{mode}-field-{field}.bin")),
                    bytemuck::cast_slice(words),
                )
                .unwrap();
            }
            records.push(serde_json::json!({"round":round,"mode":mode,"oracle_error":oracle_error,"control_delta":errors,"repeat_exact":repeat_exact}));
            if first[mode].is_none() {
                first[mode] = Some(grid);
            }
        }
    }
    assert_eq!(before, sim.checkpoint_soil_with(queue).unwrap().to_bytes());
    let candidate = first[2].as_ref().unwrap();
    let control = first[1].as_ref().unwrap();
    assert_eq!(candidate[..2], control[..2], "mass and momentum unchanged");
    let mut scatter_delta = 0_f32;
    for round in 0..6 {
        let bytes = std::fs::read(root.join(format!("round-{round}-mode-0-field-2.bin"))).unwrap();
        for (a, &b) in bytes.chunks_exact(4).zip(&candidate[2]) {
            let a = f32::from_le_bytes(a.try_into().unwrap());
            scatter_delta = scatter_delta.max((a - f32::from_bits(b)).abs());
        }
    }
    let candidate_oracle_error = records
        .iter()
        .filter(|r| r["mode"] == 2)
        .map(|r| r["oracle_error"].as_f64().unwrap())
        .fold(0_f64, f64::max);
    let report = serde_json::json!({"records":records,"candidate_scatter_delta":scatter_delta,"candidate_rounded_term_oracle_error":candidate_oracle_error,"bound":2e-5,"passed":scatter_delta<=2e-5 && candidate_oracle_error<=2e-5,"scope":"f64 sum of CPU f32 formula terms; CPU/GPU exp and contraction may differ; not a continuum or constitutive accuracy certificate","particle_checkpoint_exact":true});
    std::fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("gather_stiffness_report={report}");
    assert!(
        scatter_delta <= 2e-5 && candidate_oracle_error <= 2e-5,
        "original non-native field bound"
    );
}
