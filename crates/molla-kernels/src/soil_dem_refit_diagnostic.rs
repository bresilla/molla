use super::*;
use std::{io::Write, path::Path};

#[path = "soil_dem_ordered_refit_tests.rs"]
mod ordered;

fn trace_source(particle: usize) -> String {
    let source = include_str!("../shaders/tire_dem.wgsl");
    let start = source.find("fn step_dem_lbvh(").unwrap();
    let end = start + source[start..].find("\n@compute").unwrap();
    let mut body = source[start..end].to_string();
    assert_eq!(body.matches("let i = gid.x;").count(), 1);
    body = body.replace(
        "let i = gid.x;",
        &format!("let i = {particle}u; if gid.x != 0u {{ return; }} var visit = 0u;"),
    );
    let begin = body
        .find("                        if AUDIT_CONTACTS {")
        .unwrap();
    let finish = begin + body[begin..].find("\n                    }\n").unwrap();
    body.replace_range(begin..finish, r#"
                        let pair = pair_force(p0.xyz, v0.xyz, pos_in[j].xyz, vel_in[j].xyz,
                            grain_radius * 2.0, normal_stiffness, damping, friction);
                        force += pair;
                        visit++;
                        let distance = length(p0.xyz - pos_in[j].xyz);
                        pos_out[j] = vec4<f32>(force, bitcast<f32>(visit));
                        vel_out[j] = vec4<f32>(pair, select(0.0, 1.0, distance >= 1e-9 && distance < grain_radius * 2.0));
"#);
    body = body.replace("    if AUDIT_CONTACTS { return; }", "");
    body = body.replace("pos_out[i] = p;", "pos_out[u.dims.x] = p;");
    body = body.replace("vel_out[i] = v;", "vel_out[u.dims.x] = v;");
    format!("{}{}{}", &source[..start], body, &source[end..])
}

#[test]
fn dem_refit_trace_source_preserves_other_entrypoints() {
    let original = include_str!("../shaders/tire_dem.wgsl");
    let source = trace_source(13661);
    let start = original.find("fn step_dem_lbvh(").unwrap();
    let end = original.find("fn raster_dem_fem_terrain(").unwrap();
    assert!(source.starts_with(&original[..start]));
    assert!(source.ends_with(&original[end..]));
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

struct Trace {
    prefixes: Vec<[f32; 4]>,
    pairs: Vec<[f32; 4]>,
    order: Vec<usize>,
}

impl Trace {
    fn record(&self, path: &Path, gravity: [f32; 3], mass: f32) {
        let mut out = std::fs::File::create(path).unwrap();
        writeln!(out, "rank,neighbor,contact,fx,fy,fz,sum_x,sum_y,sum_z").unwrap();
        let mut sum = gravity.map(|g| g * mass);
        for (rank, &id) in self.order.iter().enumerate() {
            let pair = self.pairs[id];
            let prefix = self.prefixes[id];
            for axis in 0..3 {
                sum[axis] += pair[axis];
                assert_eq!(
                    sum[axis].to_bits(),
                    prefix[axis].to_bits(),
                    "recorded force accumulation rank={rank} axis={axis}"
                );
            }
            writeln!(
                out,
                "{rank},{id},{},{},{},{},{},{},{}",
                pair[3], pair[0], pair[1], pair[2], prefix[0], prefix[1], prefix[2]
            )
            .unwrap();
        }
    }
}

fn trace(
    kernel: &TireDemGpuKernel,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    state: &TireDemResidentState,
    tree: &TireDemResidentLbvh,
    contacts: &[TireDemContactCell],
    p: TireDemStepParams,
    selected: usize,
) -> Trace {
    use wgpu::util::DeviceExt;
    let n = state.grain_count;
    assert!(selected < n && n <= 50_000);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("DEM selected row trace"),
        source: wgpu::ShaderSource::Wgsl(trace_source(selected).into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("DEM selected row trace"),
        bind_group_layouts: &[
            Some(&kernel.layout_lbvh_io),
            Some(&kernel.layout_step_uniforms),
        ],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("DEM selected row trace"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("step_dem_lbvh"),
        compilation_options: Default::default(),
        cache: None,
    });
    let outputs: [_; 2] = std::array::from_fn(|_| {
        allocate(
            device,
            "DEM row trace",
            ((n + 1) * 16) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
        .unwrap()
    });
    let contacts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("DEM trace contact"),
        contents: bytemuck::cast_slice(contacts),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let io = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("DEM row trace"),
        layout: &kernel.layout_lbvh_io,
        entries: &[
            entry(0, &state.positions),
            entry(1, &state.velocities),
            entry(2, &contacts_buffer),
            entry(3, &outputs[0]),
            entry(4, &outputs[1]),
            entry(10, &tree.node_rows),
            entry(11, &tree.sorted_indices),
        ],
    });
    let uniforms = immutable_params(
        device,
        &kernel.layout_step_uniforms,
        TireDemStepUniforms {
            dims: [
                n as u32,
                contacts.len() as u32,
                tree.node_count as u32,
                tree.sorted_index_count as u32,
            ],
            params0: [
                p.grain_mass,
                p.grain_radius,
                p.dt,
                p.tire_position_push_scale,
            ],
            params1: [p.normal_stiffness, p.damping, p.friction, p.floor_y],
            gravity: [p.gravity[0], p.gravity[1], p.gravity[2], 0.0],
            grid_origin_cell_size: [0.0; 4],
            grid_shape: [0; 4],
        },
    );
    let stages: [_; 2] =
        std::array::from_fn(|_| stage_vec4(device, "DEM row trace staging", n + 1));
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &io, &[]);
        pass.set_bind_group(1, &uniforms, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    for i in 0..2 {
        encoder.copy_buffer_to_buffer(&outputs[i], 0, &stages[i], 0, ((n + 1) * 16) as u64);
    }
    queue.submit([encoder.finish()]);
    let prefixes = read_vec4(device, &stages[0], "DEM row prefixes").unwrap();
    let pairs = read_vec4(device, &stages[1], "DEM row forces").unwrap();
    let mut order = (0..n)
        .filter(|&i| prefixes[i][3].to_bits() != 0)
        .collect::<Vec<_>>();
    order.sort_by_key(|&i| prefixes[i][3].to_bits());
    for (rank, &i) in order.iter().enumerate() {
        assert_eq!(prefixes[i][3].to_bits() as usize, rank + 1);
    }
    Trace {
        prefixes,
        pairs,
        order,
    }
}

pub(super) fn difference(a: &TireDemGpuOutput, b: &TireDemGpuOutput) -> (f32, f32, Option<usize>) {
    let mut maximum = [0.0f32; 2];
    let mut first = None;
    for (field, (a, b)) in [(&a.positions, &b.positions), (&a.velocities, &b.velocities)]
        .into_iter()
        .enumerate()
    {
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            for axis in 0..3 {
                assert!(a[axis].is_finite() && b[axis].is_finite());
                maximum[field] = maximum[field].max((a[axis] - b[axis]).abs());
                if a[axis].to_bits() != b[axis].to_bits() {
                    first.get_or_insert(i);
                }
            }
        }
    }
    (maximum[0], maximum[1], first)
}

pub(super) fn save(root: &Path, name: &str, state: &TireDemGpuOutput) {
    for (suffix, rows) in [
        ("positions", &state.positions),
        ("velocities", &state.velocities),
    ] {
        std::fs::write(
            root.join(format!("{name}-{suffix}.bin")),
            bytemuck::cast_slice(rows),
        )
        .unwrap();
    }
}

#[test]
#[ignore = "requires GPU; same-input 50k DEM rebuild/refit first-divergence isolation"]
fn dem_refit_50k_first_divergence() {
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_DEM_DIVERGENCE_OUTPUT").expect("fresh diagnostic output"),
    );
    std::fs::create_dir(&root).unwrap();
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software);
    std::fs::write(root.join("adapter.txt"), format!("{adapter:?}\n")).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    let positions = (0..50_000)
        .map(|i| {
            [
                (i % 40) as f32 * 0.045 - 0.9,
                (i / 1600) as f32 * 0.045 + 0.026,
                (i / 40 % 40) as f32 * 0.045 - 0.9,
                0.0,
            ]
        })
        .collect::<Vec<_>>();
    let velocities = vec![[0.0; 4]; positions.len()];
    let terrain = TireDemResidentTerrainParams {
        origin: [-1.0, -1.0],
        spacing: 0.0625,
        nx: 32,
        nz: 32,
        floor_y: 0.0,
        grain_radius: 0.025,
    };
    let params = TireDemStepParams {
        grain_mass: 0.18,
        grain_radius: 0.025,
        normal_stiffness: 8000.0,
        damping: 40.0,
        friction: 0.6,
        floor_y: 0.0,
        gravity: [0.0, -9.81, 0.0],
        dt: 0.0005,
        tire_position_push_scale: 0.0,
    };
    let contacts = [TireDemContactCell {
        center_radius: [0.0, 1.4, 0.0, 0.3],
        force: [10.0, -80.0, 0.0, 0.0],
    }];
    let mut owners: [_; 3] = std::array::from_fn(|_| {
        TireDemResidentFemTerrainOwner::from_grain_state(
            device,
            queue,
            &kernel,
            terrain,
            &positions,
            &velocities,
        )
        .unwrap()
    });
    owners[1].set_lbvh_rebuild_interval(32).unwrap();
    owners[2].set_lbvh_rebuild_interval(32).unwrap();
    let mut previous = TireDemGpuOutput {
        positions,
        velocities,
    };
    let mut history = std::fs::File::create(root.join("history.csv")).unwrap();
    writeln!(history,"step,matched_position_max,matched_velocity_max,trajectory_position_max,trajectory_velocity_max").unwrap();
    let mut traced = false;
    for step in 1..=52 {
        kernel
            .update_resident_state(
                queue,
                owners[1].resident_state(),
                &previous.positions,
                &previous.velocities,
            )
            .unwrap();
        for owner in &mut owners {
            owner
                .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, 8)
                .unwrap();
        }
        if step == 1 {
            for owner in &mut owners[1..] {
                owner.stepper.as_mut().unwrap().legacy_refit = true;
            }
        }
        let outputs = owners
            .iter()
            .map(|owner| {
                kernel
                    .readback_resident_state(device, queue, owner.resident_state())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let matched = difference(&outputs[0], &outputs[1]);
        let trajectory = difference(&outputs[0], &outputs[2]);
        writeln!(
            history,
            "{step},{},{},{},{}",
            matched.0, matched.1, trajectory.0, trajectory.1
        )
        .unwrap();
        if !traced {
            if let Some(selected) = matched.2 {
                traced = true;
                save(&root, "first-input", &previous);
                save(&root, "first-rebuild", &outputs[0]);
                save(&root, "first-refit", &outputs[1]);
                let state = kernel
                    .create_resident_state(device, queue, &previous.positions, &previous.velocities)
                    .unwrap();
                let traces: [_; 2] = std::array::from_fn(|i| {
                    trace(
                        &kernel,
                        device,
                        queue,
                        &state,
                        owners[i].resident_lbvh().unwrap(),
                        &contacts,
                        params,
                        selected,
                    )
                });
                for (i, t) in traces.iter().enumerate() {
                    assert_eq!(
                        t.prefixes[50_000], outputs[i].positions[selected],
                        "trace reproduces production position"
                    );
                    assert_eq!(
                        t.pairs[50_000], outputs[i].velocities[selected],
                        "trace reproduces production velocity"
                    );
                    t.record(
                        &root.join(format!("trace-{i}.csv")),
                        params.gravity,
                        params.grain_mass,
                    );
                }
                let mut oracle = Vec::new();
                let a = previous.positions[selected];
                for (j, b) in previous.positions.iter().enumerate() {
                    let distance =
                        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
                            .sqrt();
                    if j != selected && distance >= 1e-9 && distance < 2.0 * params.grain_radius {
                        oracle.push(j);
                    }
                }
                for t in &traces {
                    let found = (0..50_000)
                        .filter(|&j| t.pairs[j][3] > 0.0)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        found, oracle,
                        "selected production contact set matches exhaustive search"
                    );
                }
                for &j in &oracle {
                    assert_eq!(
                        traces[0].pairs[j], traces[1].pairs[j],
                        "same pair force at neighbor {j}"
                    );
                }
                let orders: [_; 2] = std::array::from_fn(|i| {
                    traces[i]
                        .order
                        .iter()
                        .copied()
                        .filter(|&j| traces[i].pairs[j][3] > 0.0)
                        .collect::<Vec<_>>()
                });
                assert_ne!(orders[0], orders[1], "changed contact summation order");
                std::fs::write(root.join("first.txt"),format!("step={step}\nparticle={selected}\nmatched_position_max={}\nmatched_velocity_max={}\ncontacts={}\nrebuild_order={:?}\nrefit_order={:?}\n",matched.0,matched.1,oracle.len(),orders[0],orders[1])).unwrap();
                println!(
                    "first same-input divergence: step={step} particle={selected}; contact sets/pair forces exact, accumulation order differs"
                );
            }
        }
        if step == 52 {
            for (i, state) in outputs.iter().enumerate() {
                save(&root, &format!("final-{i}"), state);
            }
            println!("final matched={matched:?}; trajectory={trajectory:?}");
        }
        previous = outputs.into_iter().next().unwrap();
    }
    assert!(traced, "expected original refit divergence");
}
