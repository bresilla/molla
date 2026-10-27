use super::*;
use crate::mpm::replay_tests::read_words;

#[path = "mpm_stiffness_scatter_tests.rs"]
mod scatter_tests;

const DECLARATION: &str =
    "\n@group(1) @binding(11) var<storage, read_write> stiffness_records: array<vec4<u32>>;\n";
const SCATTER: &str = "let dk    = w * w * kdiag_p;";
const SCATTER_RECORD: &str = "
                    stiffness_records[pid * 27u + i * 9u + j * 3u + k] = vec4<u32>(
                        1u + u32(ix) + u.grid_size * (u32(iy) + u.grid_size * u32(iz)),
                        bitcast<u32>(dk), bitcast<u32>(w), bitcast<u32>(kdiag_p));";
const GATHER: &str = "stiffness_sum += w * w * gather_particles[pid].stiffness;";
const GATHER_RECORD: &str = "
                        stiffness_records[pid * 27u + u32(offset.x) * 9u + u32(offset.y) * 3u + u32(offset.z)] = vec4<u32>(
                            1u + u32(cell.x) + u.grid_size * (u32(cell.y) + u.grid_size * u32(cell.z)),
                            bitcast<u32>(w * w * gather_particles[pid].stiffness), bitcast<u32>(w), bitcast<u32>(gather_particles[pid].stiffness));";

fn source() -> String {
    let scatter = include_str!("../shaders/mpm.wgsl");
    let gather = include_str!("../shaders/mpm_gather_masks.wgsl");
    assert_eq!(scatter.matches(SCATTER).count(), 1);
    assert_eq!(gather.matches(GATHER).count(), 1);
    format!(
        "{}\n{}\n{}{DECLARATION}",
        scatter.replace(SCATTER, &format!("{SCATTER}{SCATTER_RECORD}")),
        include_str!("../shaders/mpm_deterministic_p2g.wgsl"),
        gather.replace(
            GATHER,
            &format!("{GATHER_RECORD}\n                        {GATHER}")
        )
    )
}

#[test]
fn stiffness_contributions_source_contract() {
    let source = source();
    let restored = source
        .replace(SCATTER_RECORD, "")
        .replace(&format!("{GATHER_RECORD}\n                        "), "")
        .replace(DECLARATION, "");
    assert_eq!(
        restored,
        format!(
            "{}\n{}\n{}",
            include_str!("../shaders/mpm.wgsl"),
            include_str!("../shaders/mpm_deterministic_p2g.wgsl"),
            include_str!("../shaders/mpm_gather_masks.wgsl")
        )
    );
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let slots: std::collections::BTreeSet<_> = (0..3)
        .flat_map(|i| (0..3).flat_map(move |j| (0..3).map(move |k| i * 9 + j * 3 + k)))
        .collect();
    assert_eq!(
        slots.into_iter().collect::<Vec<_>>(),
        (0..27).collect::<Vec<_>>()
    );
}

struct Probe {
    _cache: wgpu::Buffer,
    records: wgpu::Buffer,
    binding: wgpu::BindGroup,
    pipelines: [wgpu::ComputePipeline; 3],
    tail: Option<wgpu::Buffer>,
    resolve: Option<wgpu::ComputePipeline>,
}

impl Probe {
    fn new(sim: &MpmSimulation, root: &std::path::Path) -> Self {
        Self::with_source(sim, root, source())
    }

    fn with_source(sim: &MpmSimulation, root: &std::path::Path, source: String) -> Self {
        let device = &sim.device;
        assert!(!sim.native_float_atomics && sim.sparse_grid);
        let global = source.contains("fn resolve_stiffness(");
        assert!(device.limits().max_storage_buffers_per_shader_stage >= 21 + u32::from(global));
        let buffer = |bytes| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stiffness contribution diagnostic"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cache = buffer(u64::from(sim.cfg.max_particles) * 64);
        let records = buffer(u64::from(sim.cfg.max_particles) * 27 * 16);
        let tail = global.then(|| buffer(sim.g_mass.size()));
        let mut entries = storage_entries(12 + u32::from(global), false);
        entries[10] = storage_entries(11, true).pop().unwrap();
        let particles = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&sim.layout_uniforms),
                Some(&particles),
                Some(&sim.layout_grid),
                Some(&sim.layout_rigid),
            ],
            immediate_size: 0,
        });
        std::fs::write(root.join("instrumented.wgsl"), &source).unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stiffness contribution diagnostic"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let constants = [
            ("ACTIVE_GRID", 1.0),
            ("SPARSE_GRID", 1.0),
            ("DENSE_GATHER", 0.0),
            ("ZERO_GATHER_STIFFNESS", 0.0),
            ("PRUNE_GATHER_RANGES", 1.0),
            ("DEFER_P2G_LOOKUP", 0.0),
        ];
        let pipelines = ["gather_prepare", "gather_grid_masked", "p2g_two_pass"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        });
        let resolve = global.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("resolve stiffness residual"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("resolve_stiffness"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        });
        let mut bindings = vec![
            buffer_entry(0, &sim.p_pos),
            buffer_entry(1, &sim.p_vel),
            buffer_entry(2, &sim.p_c),
            buffer_entry(3, &sim.p_f),
            buffer_entry(4, &sim.p_volume),
            buffer_entry(5, &sim.p_mass),
            buffer_entry(6, &sim.p_jp),
            buffer_entry(7, &sim.p_material),
            buffer_entry(8, &sim.material_table),
            buffer_entry(9, &cache),
            buffer_entry(10, &sim.particle_order.as_ref().unwrap().keys),
            buffer_entry(11, &records),
        ];
        if let Some(tail) = &tail {
            bindings.push(buffer_entry(12, tail));
        }
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &particles,
            entries: &bindings,
        });
        Self {
            _cache: cache,
            records,
            binding,
            pipelines,
            tail,
            resolve,
        }
    }
}

fn sums(records: &[u32], cells: usize) -> Vec<f64> {
    let mut result = vec![0_f64; cells];
    for record in records.chunks_exact(4) {
        assert!(
            record[0] > 0 && record[0] as usize <= cells,
            "complete in-bounds fixture support"
        );
        let term = f32::from_bits(record[1]);
        assert!(term.is_finite() && term >= 0.0);
        result[record[0] as usize - 1] += f64::from(term);
    }
    result
}

fn diagnostic_grid(engine: &Engine, sim: &MpmSimulation) -> [Vec<u32>; 3] {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (stats, ids) = sim.active_tiles.read_metadata(device, queue, true).unwrap();
    assert!(!stats.faulted() && stats.active_tiles as usize == ids.len());
    let source = [&sim.g_mass, &sim.g_momentum, &sim.g_kdiag].map(|b| read_words(device, queue, b));
    let cells = sim.cfg.grid_size.pow(3) as usize;
    let mut output = [vec![0; cells], vec![0; cells * 3], vec![0; cells]];
    let width = sim.cfg.grid_size.div_ceil(4);
    let mut unique = std::collections::BTreeSet::new();
    for (slot, tile) in ids.into_iter().enumerate() {
        assert!(tile < width.pow(3) && unique.insert(tile));
        for local in 0..64 {
            let xyz = [
                tile % width * 4 + local % 4,
                tile / width % width * 4 + local / 4 % 4,
                tile / (width * width) * 4 + local / 16,
            ];
            if xyz.iter().any(|&v| v >= sim.cfg.grid_size) {
                continue;
            }
            let world =
                (xyz[0] + sim.cfg.grid_size * (xyz[1] + sim.cfg.grid_size * xyz[2])) as usize;
            let physical = slot * 64 + local as usize;
            for (field, stride) in [1, 3, 1].into_iter().enumerate() {
                output[field][world * stride..(world + 1) * stride]
                    .copy_from_slice(&source[field][physical * stride..(physical + 1) * stride]);
            }
        }
    }
    output
}

#[test]
#[ignore = "requires GPU; saved non-native 50k contribution capture"]
fn stiffness_contributions_capture_50k() {
    let input = std::path::PathBuf::from(std::env::var_os("SOIL_CONTRIBUTIONS_INPUT").unwrap());
    let root = std::path::PathBuf::from(std::env::var_os("SOIL_CONTRIBUTIONS_OUTPUT").unwrap());
    std::fs::create_dir(&root).unwrap();
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = tests::seeded_benchmark(&engine);
    let frozen = std::fs::read(input.join("soil.bin")).unwrap();
    sim.restore_soil_with(
        queue,
        &MpmSoilCheckpoint::from_bytes(&frozen, 50_000).unwrap(),
    )
    .unwrap();
    let mut u = sim.step_uniforms(0.0001);
    u.grid_layout[2] = 0;
    queue.write_buffer(&sim.uniform_buf, 0, bytemuck::bytes_of(&u));
    assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), frozen);
    let probe = Probe::new(&sim, &root);
    let mut captures: Vec<Vec<u32>> = Vec::new();
    let mut grids = Vec::new();
    for mode in 0..3 {
        let mut encoder = device.create_command_encoder(&Default::default());
        sim.initialize_active_grid(&mut encoder, false);
        encoder.clear_buffer(&probe.records, 0, None);
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
            if mode == 0 {
                sim.deterministic_p2g
                    .as_ref()
                    .unwrap()
                    .encode(&mut pass, &sim);
            } else {
                pass.set_bind_group(1, &probe.binding, &[]);
                if mode == 1 {
                    pass.set_pipeline(&probe.pipelines[0]);
                    pass.dispatch_workgroups(sim.particle_count.div_ceil(64), 1, 1);
                    pass.set_pipeline(&probe.pipelines[1]);
                    pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
                } else {
                    pass.set_pipeline(&probe.pipelines[2]);
                    pass.dispatch_workgroups(sim.particle_count.div_ceil(64), 1, 1);
                }
            }
        }
        queue.submit([encoder.finish()]);
        let grid = diagnostic_grid(&engine, &sim);
        assert!(
            grid.iter()
                .flatten()
                .all(|&v| f32::from_bits(v).is_finite())
        );
        for (field, words) in grid.iter().enumerate() {
            std::fs::write(
                root.join(format!("mode-{mode}-grid-{field}.bin")),
                bytemuck::cast_slice(words),
            )
            .unwrap();
        }
        grids.push(grid);
        if mode != 0 {
            let words = read_words(device, queue, &probe.records);
            std::fs::write(
                root.join(format!("mode-{mode}-contributions.bin")),
                bytemuck::cast_slice(&words),
            )
            .unwrap();
            captures.push(words);
        }
    }
    let sums: Vec<_> = captures
        .iter()
        .map(|c| sums(c, u.grid_total as usize))
        .collect();
    let mut changed = [0_usize; 4];
    for (a, b) in captures[0].chunks_exact(4).zip(captures[1].chunks_exact(4)) {
        for field in 0..4 {
            changed[field] += usize::from(a[field] != b[field]);
        }
    }
    let mut errors = [0_f64; 3];
    for i in 0..u.grid_total as usize {
        errors[0] = errors[0].max((sums[0][i] - sums[1][i]).abs());
        errors[1] = errors[1].max((sums[0][i] - f64::from(f32::from_bits(grids[1][2][i]))).abs());
        errors[2] = errors[2].max((sums[1][i] - f64::from(f32::from_bits(grids[2][2][i]))).abs());
    }
    for (mode, values) in sums.iter().enumerate() {
        std::fs::write(
            root.join(format!("sum-{mode}.bin")),
            bytemuck::cast_slice(values),
        )
        .unwrap();
    }
    let checkpoint_exact = sim.checkpoint_soil_with(queue).unwrap().to_bytes() == frozen;
    let original_exact = grids[0] == grids[1];
    let archived = [0, 1, 2].map(|field| {
        std::fs::read(input.join(format!("round-0-mode-1-field-{field}.bin"))).unwrap()
    });
    let archived_exact =
        (0..3).all(|field| bytemuck::cast_slice::<u32, u8>(&grids[0][field]) == archived[field]);
    let report = serde_json::json!({"particle_contributions":50_000*27,"changed_cell_term_weight_stiffness":changed,"maximum_sum_delta_gather_error_scatter_error":errors,"instrumented_gather_exact":original_exact,"archived_gather_exact":archived_exact,"particle_checkpoint_exact":checkpoint_exact,"scope":"instrumented transfer terms and host f64 sums; no timing or continuum accuracy claim"});
    std::fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("stiffness_contributions={report}");
    assert!(
        original_exact && archived_exact && checkpoint_exact,
        "instrumentation and frozen-input identity"
    );
    assert_eq!(changed[0], 0, "identical stencil-cell mapping");
}
