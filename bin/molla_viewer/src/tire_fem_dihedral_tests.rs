use super::*;
use wgpu::util::DeviceExt;

type V = [f64; 3];
type Quad = [V; 4];
const ANALYTIC: &str =
    include_str!("../../../crates/molla-kernels/shaders/tire_fem_dihedral_analytic.wgsl");
pub(super) const PRODUCTION: &str = concat!(
    include_str!("../../../crates/molla-kernels/shaders/tire_fem_fiber.wgsl"),
    "\n",
    include_str!("../../../crates/molla-kernels/shaders/tire_fem_dihedral_analytic.wgsl"),
    "\n",
    include_str!("../../../crates/molla-kernels/shaders/tire_fem_atlas.wgsl")
);
fn sub(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] - b[i])
}
fn add(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] + b[i])
}
fn scale(a: V, s: f64) -> V {
    a.map(|v| v * s)
}
fn dot(a: V, b: V) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: V) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: V) -> V {
    scale(a, 1. / norm(a))
}
fn angle(p: &Quad) -> f64 {
    let a = unit(cross(sub(p[1], p[0]), sub(p[2], p[0])));
    let b = unit(cross(sub(p[2], p[0]), sub(p[3], p[0])));
    norm(cross(a, b)).atan2(dot(a, b))
}
fn gradients(p: &Quad) -> Quad {
    let u = sub(p[1], p[0]);
    let v = sub(p[2], p[0]);
    let w = sub(p[3], p[0]);
    let a = cross(u, v);
    let b = cross(v, w);
    let na = unit(a);
    let nb = unit(b);
    let edge = unit(v);
    let orientation = dot(cross(na, nb), edge);
    if orientation == 0. {
        return [[0.; 3]; 4];
    }
    let sign = orientation.signum();
    let ga = scale(cross(na, edge), sign / norm(a));
    let gb = scale(cross(edge, nb), sign / norm(b));
    let g1 = cross(v, ga);
    let g2 = add(cross(ga, u), cross(w, gb));
    let g3 = cross(gb, v);
    [scale(add(add(g1, g2), g3), -1.), g1, g2, g3]
}
fn quads() -> Vec<Quad> {
    let mut out = Vec::new();
    for size in [0.05, 0.5, 2.] {
        for shift in [[0.; 3], [2., -1., 3.]] {
            for bend in [
                -3.140592653589793f64,
                -2.8,
                -1.2,
                -0.1,
                -0.001,
                0.001,
                0.1,
                1.2,
                2.8,
                3.140592653589793,
            ] {
                for twist in [0f64, 0.3, 1.7] {
                    let base = [
                        [0., 0., 0.],
                        [0.3, 1., 0.],
                        [1., 0., 0.],
                        [0.8, -bend.cos(), bend.sin()],
                    ];
                    let p = base.map(|p| {
                        let rotated = [
                            p[0] * twist.cos() - p[1] * twist.sin(),
                            p[0] * twist.sin() + p[1] * twist.cos(),
                            p[2],
                        ];
                        add(scale(rotated, size), shift).map(|v| v as f32 as f64)
                    });
                    out.push(p);
                }
            }
        }
    }
    out
}
fn finite_difference(p: &Quad, node: usize, axis: usize) -> f64 {
    let h = norm(sub(p[2], p[0])) * 1e-6;
    let sample = |offset: f64| {
        let mut p = *p;
        p[node][axis] += offset * h;
        angle(&p)
    };
    (sample(-2.) - 8. * sample(-1.) + 8. * sample(1.) - sample(2.)) / (12. * h)
}

#[test]
fn analytic_dihedral_derivative_oracle() {
    let mut maximum = 0f64;
    for p in quads() {
        let g = gradients(&p);
        for node in 0..4 {
            for axis in 0..3 {
                let oracle = finite_difference(&p, node, axis);
                let error = (g[node][axis] - oracle).abs() / (1. + oracle.abs());
                maximum = maximum.max(error);
                assert!(error <= 2e-5, "analytic derivative {error:e}");
            }
        }
        let sum = g.into_iter().fold([0.; 3], add);
        let torque = p
            .into_iter()
            .zip(g)
            .map(|(p, g)| cross(p, g))
            .fold([0.; 3], add);
        assert!(norm(sum) < 1e-10 && norm(torque) < 1e-10);
    }
    println!(
        "{} analytic hinges; 12 derivative components each; maximum scaled five-point derivative error {maximum:e}; rigid-mode gradients pass",
        quads().len()
    );
}

fn packet(path: &std::path::Path) -> (usize, Vec<Vec<u8>>) {
    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.len() <= 32 * 1024 * 1024 && bytes.len() >= 16);
    assert_eq!(&bytes[..8], b"TFEMSTG1");
    let nodes = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    assert!((1..=16384).contains(&nodes));
    assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 44);
    let mut offset = 16;
    let mut buffers = Vec::new();
    for _ in 0..44 {
        let length = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        assert!(length > 0 && length % 4 == 0 && length <= 16 * 1024 * 1024);
        buffers.push(bytes[offset..offset + length].to_vec());
        offset += length;
    }
    assert_eq!(offset, bytes.len());
    assert_eq!(buffers[1].len(), 16 * nodes);
    (nodes, buffers)
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn raw_floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn captured_quads(data: &[Vec<u8>]) -> Vec<Quad> {
    let pos = floats(&data[1]);
    data[34]
        .chunks_exact(16)
        .map(|indices| {
            std::array::from_fn(|i| {
                let node =
                    u32::from_le_bytes(indices[4 * i..4 * i + 4].try_into().unwrap()) as usize;
                std::array::from_fn(|j| pos[4 * node + j] as f64)
            })
        })
        .collect()
}

fn gradient_gpu(engine: &Engine, cases: &[Quad]) -> Vec<f32> {
    let device = engine.device().unwrap();
    let input: Vec<f32> = cases
        .iter()
        .flat_map(|p| {
            p.iter()
                .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 0.])
        })
        .collect();
    let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &raw_floats(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (cases.len() * 80) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let shader=device.create_shader_module(wgpu::ShaderModuleDescriptor { label:None,source:wgpu::ShaderSource::Wgsl(format!("{ANALYTIC}\n@group(0) @binding(0) var<storage,read> points:array<vec4<f32>>;\n@group(0) @binding(1) var<storage,read_write> output:array<vec4<f32>>;\n@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{ let i=id.x; if 4u*i>=arrayLength(&points) {{return;}} let x0=points[4u*i].xyz;let x1=points[4u*i+1u].xyz;let x2=points[4u*i+2u].xyz;let x3=points[4u*i+3u].xyz;let g=quad_dihedral_gradients(x0,x1,x2,x3);for(var j=0u;j<4u;j++) {{output[5u*i+j]=vec4<f32>(g[j],0.0);}} output[5u*i+4u]=vec4<f32>(quad_dihedral_angle(x0,x1,x2,x3),0.0,0.0,0.0);}}" ).into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    read_floats(engine, &output)
}

fn candidate_shader() -> String {
    PRODUCTION.to_owned()
}

fn legacy_shader() -> String {
    let mut source = PRODUCTION.replacen(
        ANALYTIC,
        include_str!("../../../crates/molla-kernels/tests/support/tire_fem_dihedral_legacy.wgsl"),
        1,
    );
    let old = "            let gradients = quad_dihedral_gradients(x0, x1, x2, x3);\n            let g0 = gradients[0];\n            let g1 = gradients[1];\n            let g2 = gradients[2];\n            let g3 = gradients[3];";
    assert_eq!(source.matches(old).count(), 1);
    source=source.replace(old,"            let g0 = quad_dihedral_grad_vec(0u, x0, x1, x2, x3);\n            let g1 = quad_dihedral_grad_vec(1u, x0, x1, x2, x3);\n            let g2 = quad_dihedral_grad_vec(2u, x0, x1, x2, x3);\n            let g3 = quad_dihedral_grad_vec(3u, x0, x1, x2, x3);");
    source
}

fn node_replay(engine: &Engine, data: &[Vec<u8>], nodes: usize, source: &str) -> Vec<f32> {
    node_replay_timed(engine, data, nodes, source, false).0
}

fn node_replay_timed(
    engine: &Engine,
    data: &[Vec<u8>],
    nodes: usize,
    source: &str,
    timed: bool,
) -> (Vec<f32>, Option<f64>) {
    let device = engine.device().unwrap();
    let buffers: Vec<_> = data
        .iter()
        .map(|d| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: d,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        })
        .collect();
    let entries: Vec<_> = (0..44)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: !matches!(binding, 12 | 13 | 14 | 20),
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let bindings: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &bindings,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("step_nodes_atlas_contract"),
        compilation_options: Default::default(),
        cache: None,
    });
    let query = timed.then(|| {
        device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        })
    });
    let resolved = timed.then(|| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    });
    if timed {
        let mut warmup = device.create_command_encoder(&Default::default());
        {
            let mut pass = warmup.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            for _ in 0..32 {
                pass.dispatch_workgroups((nodes as u32).div_ceil(64), 1, 1);
            }
        }
        engine.queue().unwrap().submit([warmup.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: query
                .as_ref()
                .map(|query_set| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        for _ in 0..if timed { 64 } else { 1 } {
            pass.dispatch_workgroups((nodes as u32).div_ceil(64), 1, 1);
        }
    }
    if let (Some(q), Some(r)) = (&query, &resolved) {
        encoder.resolve_query_set(q, 0..2, r, 0);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    let duration = resolved.map(|r| {
        let words = read_floats(engine, &r);
        let ticks: Vec<_> = words
            .chunks_exact(2)
            .map(|w| w[0].to_bits() as u64 | ((w[1].to_bits() as u64) << 32))
            .collect();
        assert!(ticks[1] > ticks[0]);
        (ticks[1] - ticks[0]) as f64 * engine.queue().unwrap().get_timestamp_period() as f64
            / 64.
            / 1e6
    });
    let output = buffers[12..15]
        .iter()
        .flat_map(|b| read_floats(engine, b))
        .collect();
    (output, duration)
}

#[test]
#[ignore = "native GPU timestamps; paired immutable real-node replay, not advancing trajectory"]
fn analytic_dihedral_paired_timing_gpu() {
    paired_timing([legacy_shader(), candidate_shader()], false);
}

#[test]
#[ignore = "native GPU timestamps; disabled-tangent guard against identical unguarded shader"]
fn fiber_tangent_disabled_paired_timing_gpu() {
    let guard = "    if (!stabilize) { return FiberResponse(P, 0.0); }";
    let signature = "gradient: vec3<f32>) -> FiberResponse";
    let body = "    let projected = dot(gradient, a0);";
    assert_eq!(PRODUCTION.matches(signature).count(), 1);
    assert_eq!(PRODUCTION.matches(body).count(), 1);
    assert_eq!(PRODUCTION.matches(", g_node);").count(), 4);
    let candidate = PRODUCTION
        .replace(
            signature,
            "gradient: vec3<f32>, stabilize: bool) -> FiberResponse",
        )
        .replace(body, &format!("{guard}\n{body}"))
        .replace(", g_node);", ", g_node, params.phys4.y > 0.0);");
    paired_timing([PRODUCTION.to_owned(), candidate], true);
}

fn paired_timing(shaders: [String; 2], full_equal: bool) {
    let engine = Engine::new_gpu_with_features(wgpu::Features::TIMESTAMP_QUERY).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/soil/2026-10-13-cavity-stage-probe/capture");
    for name in ["gpu", "cpu"] {
        let (nodes, data) = packet(&root.join(format!("{name}.inputs.bin")));
        let mut samples: [Vec<f64>; 2] = Default::default();
        let mut expected: [Option<Vec<u32>>; 2] = [None, None];
        for pair in 0..8 {
            for route in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                let (values, ms) = node_replay_timed(&engine, &data, nodes, &shaders[route], true);
                assert!(values.iter().all(|v| v.is_finite()));
                let bits: Vec<_> = values.iter().map(|v| v.to_bits()).collect();
                if let Some(reference) = &expected[route] {
                    assert_eq!(&bits, reference);
                } else {
                    expected[route] = Some(bits);
                }
                let ms = ms.unwrap();
                assert!(ms.is_finite() && ms > 0.);
                samples[route].push(ms);
                println!("timing,{name},{pair},{route},{ms:.9}");
            }
            assert_eq!(
                &expected[0].as_ref().unwrap()[if full_equal { 0 } else { 8 * nodes }..],
                &expected[1].as_ref().unwrap()[if full_equal { 0 } else { 8 * nodes }..]
            );
        }
        let medians = samples.map(|mut v| {
            v.sort_by(f64::total_cmp);
            (v[3] + v[4]) * 0.5
        });
        println!(
            "paired {name}: {nodes} nodes, 32 warmup + 64 timed immutable dispatches/sample, 8 alternating pairs; median control {:.9} ms candidate {:.9} ms ratio {:.6}; both routes repeat bit-exact, force outputs equal, all outputs equal={full_equal}",
            medians[0],
            medians[1],
            medians[1] / medians[0]
        );
    }
}

#[test]
#[ignore = "native GPU; analytic bending derivative and saved real FEM input comparison"]
fn analytic_dihedral_gpu_and_saved_inputs() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/soil/2026-10-13-cavity-stage-probe/capture");
    let (n, a) = packet(&root.join("gpu.inputs.bin"));
    let (m, b) = packet(&root.join("cpu.inputs.bin"));
    assert_eq!(n, m);
    let mut cases = quads();
    cases.extend(captured_quads(&a));
    cases.extend(captured_quads(&b));
    let result = gradient_gpu(&engine, &cases);
    let special = gradient_gpu(
        &engine,
        &[
            [[0.; 3]; 4],
            [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., -1., 0.]],
        ],
    );
    assert!(
        special.iter().all(|v| *v == 0.),
        "degenerate and exact-flat convention"
    );
    let mut maximum = 0f64;
    let mut count = 0;
    for (p, rows) in cases.iter().zip(result.chunks_exact(20)) {
        let target = gradients(p);
        for i in 0..4 {
            for j in 0..3 {
                let error =
                    (rows[4 * i + j] as f64 - target[i][j]).abs() / (1. + target[i][j].abs());
                maximum = maximum.max(error);
                assert!(
                    error <= 2e-5,
                    "GPU analytic derivative case {count}: {error:e}"
                );
            }
        }
        assert!((rows[16] as f64 - angle(p)).abs() <= 2e-5);
        count += 1;
    }
    println!(
        "{count} synthetic/saved hinges: GPU analytic gradients vs f64 max scaled error {maximum:e}"
    );
    let mut outputs = Vec::new();
    for (name, data) in [("gpu", a), ("cpu", b)] {
        let output = node_replay(&engine, &data, n, &candidate_shader());
        let repeat = node_replay(&engine, &data, n, &candidate_shader());
        assert!(
            output.iter().chain(&repeat).all(|v| v.is_finite()),
            "nonfinite candidate replay {name}"
        );
        assert_eq!(
            output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            repeat.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let baseline = node_replay(&engine, &data, n, &legacy_shader());
        assert!(
            baseline.iter().all(|v| v.is_finite()),
            "nonfinite current baseline {name}"
        );
        assert_eq!(
            output[8 * n..]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            baseline[8 * n..]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "bending changed pre-constraint force output"
        );
        outputs.push(output);
    }
    println!(
        "saved third-substep production analytic input-pair position max {:e}; velocity max {:e}; force outputs unchanged vs legacy bending; repeats bit-exact",
        difference(&outputs[0][..4 * n], &outputs[1][..4 * n]),
        difference(&outputs[0][4 * n..8 * n], &outputs[1][4 * n..8 * n])
    );
}
