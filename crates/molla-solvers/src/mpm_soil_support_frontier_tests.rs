use super::*;
use std::time::Instant;
use wgpu::util::DeviceExt;

const REFERENCE: &str = include_str!("../shaders/mpm_soil_support_reference.wgsl");
const FRONTIER: &str = include_str!("../shaders/mpm_soil_support.wgsl");

struct Fixture {
    params: wgpu::Buffer,
    positions: wgpu::Buffer,
    quiet: wgpu::Buffer,
    nodes: wgpu::Buffer,
    result: wgpu::Buffer,
    staging: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    pipelines: [[wgpu::ComputePipeline; 4]; 2],
}

impl Fixture {
    fn new(device: &wgpu::Device) -> Self {
        let buffer = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("support frontier fixture"),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0; 64],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let positions = buffer(50_000 * 16, storage);
        let quiet = buffer(50_000 * 8, storage);
        let nodes = buffer(64 * 64 * 64 * 4, storage | wgpu::BufferUsages::COPY_SRC);
        let result = buffer(50_000 * 4, storage | wgpu::BufferUsages::COPY_SRC);
        let staging = buffer(
            64 * 64 * 64 * 4 + 50_000 * 4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[0, 1, 2, 3, 4].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 3,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            }),
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                buffer_entry(0, &params),
                buffer_entry(1, &positions),
                buffer_entry(2, &quiet),
                buffer_entry(3, &nodes),
                buffer_entry(4, &result),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = [REFERENCE, FRONTIER].map(|source| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            [
                "support_mark",
                "closure_gather",
                "closure_mark",
                "closure_check",
            ]
            .map(|entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
        });
        Self {
            params,
            positions,
            quiet,
            nodes,
            result,
            staging,
            bindings,
            pipelines,
        }
    }

    fn input(&self, queue: &wgpu::Queue, positions: &[[f32; 4]], pattern: u32) -> Vec<u32> {
        let count = positions.len();
        let mut params = [0u32; 16];
        params[8] = count as u32;
        params[9] = 64;
        params[10] = 2;
        params[11] = u32::from(pattern >= 2);
        params[12] = 1f32.to_bits();
        queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&params));
        queue.write_buffer(&self.positions, 0, bytemuck::cast_slice(positions));
        let quiet: Vec<[u32; 2]> = (0..count)
            .map(|i| [if pattern == 1 && i % 7 == 0 { 0 } else { 2 }, 0])
            .collect();
        queue.write_buffer(&self.quiet, 0, bytemuck::cast_slice(&quiet));
        (0..count)
            .map(|i| u32::from(pattern >= 2 && (i == 0 || (pattern == 3 && i + 1 == count))))
            .collect()
    }

    fn run(&self, engine: &Engine, arm: usize, rounds: u32, seeds: &[u32]) -> (Vec<u32>, f64) {
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        queue.write_buffer(&self.result, 0, bytemuck::cast_slice(seeds));
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.nodes, 0, None);
        let mut dispatch = |index: usize| {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[arm][index]);
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.dispatch_workgroups((seeds.len() as u32).div_ceil(64), 1, 1);
        };
        dispatch(0);
        for _ in 0..rounds {
            dispatch(1);
            dispatch(2);
        }
        dispatch(3);
        let node_bytes = 64 * 64 * 64 * 4;
        let particle_bytes = seeds.len() as u64 * 4;
        encoder.copy_buffer_to_buffer(&self.result, 0, &self.staging, 0, particle_bytes);
        encoder.copy_buffer_to_buffer(&self.nodes, 0, &self.staging, particle_bytes, node_bytes);
        queue.submit([encoder.finish()]);
        let slice = self.staging.slice(..particle_bytes + node_bytes);
        let (send, recv) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            send.send(r).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        recv.recv().unwrap().unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.;
        let words = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        self.staging.unmap();
        (words, ms)
    }
}

fn independent(positions: &[[f32; 4]], seeds: &[u32], pattern: u32, rounds: u32) -> Vec<u32> {
    let bases: Vec<[i32; 3]> = positions
        .iter()
        .map(|p| std::array::from_fn(|a| (p[a] - 0.5).floor() as i32))
        .collect();
    let valid: Vec<_> = bases
        .iter()
        .map(|b| b.iter().all(|&v| (0..62).contains(&v)))
        .collect();
    let nonquiet: Vec<_> = (0..positions.len())
        .map(|i| pattern == 1 && i % 7 == 0)
        .collect();
    let touches = |i: usize, j: usize| {
        valid[i] && valid[j] && (0..3).all(|a| bases[i][a].abs_diff(bases[j][a]) <= 2)
    };
    let mut reached: Vec<_> = (0..positions.len())
        .map(|i| valid[i] && (nonquiet[i] || seeds[i] != 0))
        .collect();
    for _ in 0..rounds {
        reached = (0..positions.len())
            .map(|i| (0..positions.len()).any(|j| reached[j] && touches(i, j)))
            .collect();
    }
    (0..positions.len())
        .map(|i| {
            if !valid[i] {
                return 4;
            }
            let incomplete =
                !reached[i] && (0..positions.len()).any(|j| reached[j] && touches(i, j));
            (if nonquiet[i] { 2 } else { 1 })
                | (seeds[i] * 32)
                | (u32::from(reached[i]) * 64)
                | (u32::from(incomplete) * 128)
        })
        .collect()
}

#[test]
fn quiescence_frontier_shader_contract() {
    for source in [REFERENCE, FRONTIER] {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
    assert!(FRONTIER.contains("result[pid] &= ~256u;"));
    assert!(!REFERENCE.contains("256u"));
}

#[test]
#[ignore = "requires GPU; exact reference/frontier support closure and paired 50k component timing"]
fn quiescence_gpu_frontier_differential_and_profile() {
    let engine = Engine::new_gpu().unwrap();
    let fixture = Fixture::new(engine.device().unwrap());
    let mut pairs = 0;
    for count in [29usize, 129, 50_000] {
        let mut positions: Vec<[f32; 4]> = (0..count)
            .map(|i| {
                if count == 50_000 {
                    [
                        (1 + i % 50) as f32,
                        (1 + (i / 50) % 25) as f32,
                        (1 + i / 1250) as f32,
                        0.,
                    ]
                } else {
                    [(2 + 2 * (i % 29)) as f32, 8., 8., 0.]
                }
            })
            .collect();
        if count == 129 {
            positions[128] = [-3., 8., 8., 0.];
        }
        for pattern in 0..4 {
            let seeds = fixture.input(engine.queue().unwrap(), &positions, pattern);
            let rounds: &[u32] = if count == 29 {
                &[1, 2, 8, 32, 64]
            } else {
                &[1, 8, 64]
            };
            for &rounds in rounds {
                let (reference, _) = fixture.run(&engine, 0, rounds, &seeds);
                let (frontier, _) = fixture.run(&engine, 1, rounds, &seeds);
                assert_eq!(
                    frontier, reference,
                    "count={count} pattern={pattern} rounds={rounds}"
                );
                assert!(frontier[..count].iter().all(|w| w & 256 == 0));
                if count < 50_000 {
                    assert_eq!(
                        &frontier[..count],
                        independent(&positions, &seeds, pattern, rounds)
                    );
                }
                pairs += 1;
            }
        }
        println!(
            "support-frontier count={count} exact_results_and_nodes=true independent_small_fixture=true"
        );
        if count == 50_000 {
            let seeds = fixture.input(engine.queue().unwrap(), &positions, 2);
            let mut timings = [Vec::new(), Vec::new()];
            for pair in 0..6 {
                let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
                let mut outputs = [Vec::new(), Vec::new()];
                for arm in order {
                    let (words, ms) = fixture.run(&engine, arm, 64, &seeds);
                    println!(
                        "support-frontier-timing pair={pair} arm={arm} particles=50000 rounds=64 wall_ms={ms:.6}"
                    );
                    timings[arm].push(ms);
                    outputs[arm] = words;
                }
                assert_eq!(outputs[0], outputs[1]);
            }
            for t in &mut timings {
                t.sort_by(f64::total_cmp);
            }
            println!(
                "support-frontier-profile reference_p50_ms={:.6} frontier_p50_ms={:.6} includes_encoding_readback=true",
                timings[0][2], timings[1][2]
            );
        }
    }
    println!("support-frontier exact_pairs={pairs} numerical_gates_unchanged=true");
}
