use super::tests::{buffer, read};
use super::*;

#[derive(Clone)]
pub(crate) struct NormalReplay {
    pub inputs: [wgpu::Buffer; 4],
    pub status: wgpu::Buffer,
    pub control: wgpu::Buffer,
    pub capacity: SoilNormalCapacity,
    pub hard_rows: bool,
}

pub(super) struct FailureCapture {
    pub snapshot: NormalReplay,
    pipelines: [wgpu::ComputePipeline; 3],
    groups: [wgpu::BindGroup; 5],
}

pub(super) struct ReferenceCheck {
    pub reference: Box<ExperimentalSoilNormalProjectionGpu>,
    pub counts: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    groups: [(wgpu::BindGroup, u32); 4],
}

impl ReferenceCheck {
    pub fn new(
        engine: &Engine,
        candidate: &ExperimentalSoilNormalProjectionGpu,
        source: &str,
    ) -> Self {
        let input = &candidate.replay_inputs;
        let reference = Box::new(
            ExperimentalSoilNormalProjectionGpu::new_shader(
                engine,
                SoilNormalInputs {
                    mass: &input[0],
                    free_rates: &input[1],
                    rows: &input[2],
                    contacts: &input[3],
                },
                candidate.replay_config.0,
                candidate.replay_config.1,
                source,
            )
            .unwrap(),
        );
        let device = engine.device().unwrap();
        let counts = buffer(engine, &[0.0; 5]);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test-only same-input normal comparison"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_normal_compare.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("compare"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sources = [
            (&reference.status, &candidate.status),
            (&reference.rates, &candidate.rates),
            (&reference.impulses, &candidate.impulses),
            (&reference.normal_velocities, &candidate.normal_velocities),
        ];
        let groups = std::array::from_fn(|kind| {
            let (old, new) = sources[kind];
            assert_eq!(old.size(), new.size());
            let count = (old.size() / 4) as u32;
            let params = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            engine.queue().unwrap().write_buffer(
                &params,
                0,
                bytemuck::cast_slice(&[kind as u32, count, 0, 0]),
            );
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    old,
                    new,
                    &counts,
                    &reference.status,
                    &candidate.status,
                    &params,
                ]
                .iter()
                .enumerate()
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
            });
            (group, count.div_ceil(64))
        });
        Self {
            reference,
            counts,
            pipeline,
            groups,
        }
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.reference.encode(encoder);
        for (group, count) in &self.groups {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(*count, 1, 1);
        }
    }
}

pub(crate) fn read_reference_counts(engine: &Engine, counts: &wgpu::Buffer) -> [u32; 5] {
    read(engine, counts)
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

#[test]
#[ignore = "requires real GPU"]
fn reference_check_detects_output_changes_and_skips_faulted_outputs() {
    let engine = Engine::new_gpu().unwrap();
    let inputs = [vec![2.0], vec![1.0], vec![1.0], vec![1.0, 0.0, 0.0, 0.0]]
        .map(|words| buffer(&engine, &words));
    let mut candidate = ExperimentalSoilNormalProjectionGpu::new(
        &engine,
        SoilNormalInputs {
            mass: &inputs[0],
            free_rates: &inputs[1],
            rows: &inputs[2],
            contacts: &inputs[3],
        },
        SoilNormalCapacity {
            dofs: 1,
            contacts: 1,
            max_iterations: 32,
            max_owned_bytes: u64::MAX,
        },
    )
    .unwrap();
    let source = include_str!("../shaders/soil_normal_projection.wgsl").replace(
        "normal_velocities[c]=velocity;",
        "normal_velocities[c]=velocity+1.0;",
    );
    assert_ne!(
        source,
        include_str!("../shaders/soil_normal_projection.wgsl")
    );
    let counts = candidate.prepare_reference_check(&engine, &source);
    encode(&engine, &candidate, true);
    assert_eq!(read_reference_counts(&engine, &counts), [1, 0, 0, 0, 1]);
    assert!((read(&engine, candidate.normal_velocities_buffer())[0] - 2.0 / 3.0).abs() < 1e-6);
    engine.queue().unwrap().write_buffer(
        candidate.status_buffer(),
        0,
        bytemuck::cast_slice(&[NORMAL_INPUT]),
    );
    encode(&engine, &candidate, false);
    assert_eq!(read_reference_counts(&engine, &counts), [2, 0, 0, 0, 1]);
}

impl FailureCapture {
    pub fn new(
        engine: &Engine,
        inputs: &[wgpu::Buffer; 4],
        status: &wgpu::Buffer,
        config: (SoilNormalCapacity, bool),
    ) -> Self {
        let device = engine.device().unwrap();
        let make = |source: &wgpu::Buffer| buffer(engine, &vec![0.0; (source.size() / 4) as usize]);
        let snapshot = NormalReplay {
            inputs: inputs.each_ref().map(make),
            status: make(status),
            control: buffer(engine, &[0.0; 4]),
            capacity: config.0,
            hard_rows: config.1,
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test-only normal first-failure capture"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_normal_capture.wgsl").into(),
            ),
        });
        let entries: Vec<_> = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding % 2 == 0,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
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
        let pipelines = ["arm", "capture", "finish"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let sources = [&inputs[0], &inputs[1], &inputs[2], &inputs[3], status];
        let targets = [
            &snapshot.inputs[0],
            &snapshot.inputs[1],
            &snapshot.inputs[2],
            &snapshot.inputs[3],
            &snapshot.status,
        ];
        let groups = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[sources[i], targets[i], status, &snapshot.control]
                    .iter()
                    .enumerate()
                    .map(|(binding, b)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        });
        Self {
            snapshot,
            pipelines,
            groups,
        }
    }
    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: usize,
        group: usize,
        count: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines[pipeline]);
        pass.set_bind_group(0, &self.groups[group], &[]);
        pass.dispatch_workgroups(count, 1, 1);
    }
    pub fn arm(&self, encoder: &mut wgpu::CommandEncoder) {
        self.dispatch(encoder, 0, 0, 1);
    }
    pub fn capture(&self, encoder: &mut wgpu::CommandEncoder) {
        for (i, b) in self
            .snapshot
            .inputs
            .iter()
            .chain([&self.snapshot.status])
            .enumerate()
        {
            self.dispatch(encoder, 1, i, (b.size() / 4).div_ceil(64) as u32);
        }
        self.dispatch(encoder, 2, 0, 1);
    }
}

impl NormalReplay {
    pub fn save(&self, engine: &Engine, path: &std::path::Path) -> bool {
        let control = read(engine, &self.control)
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        if control[0] == 0 {
            return false;
        }
        std::fs::create_dir(path).unwrap();
        for (name, source) in ["mass", "free", "rows", "terms", "status"]
            .into_iter()
            .zip(self.inputs.iter().chain([&self.status]))
        {
            let values = read(engine, source);
            let bytes: Vec<_> = values
                .into_iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            std::fs::write(path.join(format!("{name}.bin")), bytes).unwrap();
        }
        let metadata = serde_json::json!({"schema":1,"dofs":self.capacity.dofs,"contacts":self.capacity.contacts,"max_iterations":self.capacity.max_iterations,"hard_rows":self.hard_rows,"control":control});
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata).unwrap(),
        )
        .unwrap();
        true
    }
}

fn encode(engine: &Engine, projection: &ExperimentalSoilNormalProjectionGpu, clear: bool) {
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if clear {
        encoder.clear_buffer(projection.status_buffer(), 0, None);
    }
    projection.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
}

#[test]
#[ignore = "requires real GPU"]
fn capture_preserves_first_local_failure_and_storage_only_inputs() {
    let engine = Engine::new_gpu().unwrap();
    let expected = [vec![1.0f32], vec![1.0], vec![1.0], vec![1.0, 0.0, 0.0, 0.0]];
    let inputs = expected.each_ref().map(|words| {
        let result = engine
            .device()
            .unwrap()
            .create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (words.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        engine
            .queue()
            .unwrap()
            .write_buffer(&result, 0, bytemuck::cast_slice(words));
        result
    });
    let mut projection = ExperimentalSoilNormalProjectionGpu::new(
        &engine,
        SoilNormalInputs {
            mass: &inputs[0],
            free_rates: &inputs[1],
            rows: &inputs[2],
            contacts: &inputs[3],
        },
        SoilNormalCapacity {
            dofs: 1,
            contacts: 1,
            max_iterations: 1,
            max_owned_bytes: u64::MAX,
        },
    )
    .unwrap();
    let capture = projection.prepare_failure_capture(&engine);
    engine.queue().unwrap().write_buffer(
        projection.status_buffer(),
        0,
        bytemuck::cast_slice(&[128u32]),
    );
    encode(&engine, &projection, false);
    assert_eq!(read(&engine, &capture.control)[0].to_bits(), 0);
    encode(&engine, &projection, true);
    assert_eq!(
        read(&engine, &capture.status)[0].to_bits(),
        NORMAL_ITERATIONS
    );
    assert_eq!(
        read(&engine, &capture.control)
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        [1, 1, 1, NORMAL_ITERATIONS]
    );
    for (a, b) in capture.inputs.iter().zip(&expected) {
        assert_eq!(&read(&engine, a), b);
    }
    engine
        .queue()
        .unwrap()
        .write_buffer(&inputs[1], 0, bytemuck::cast_slice(&[2.0f32]));
    encode(&engine, &projection, true);
    assert_eq!(read(&engine, &capture.inputs[1]), [1.0]);
    assert_eq!(read(&engine, &capture.control)[2].to_bits(), 1);
}

#[test]
#[ignore = "requires real GPU and SOIL_NORMAL_CAPTURE_DIR"]
fn replay_saved_normal_failure() {
    let engine = Engine::new_gpu().unwrap();
    let root = std::path::PathBuf::from(
        std::env::var_os("SOIL_NORMAL_CAPTURE_DIR").expect("capture directory required"),
    );
    let mut captures = 0;
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if !path.join("metadata.json").exists() {
            continue;
        }
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(metadata["schema"], 1);
        let words = |name: &str| {
            let bytes = std::fs::read(path.join(format!("{name}.bin"))).unwrap();
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
                .collect::<Vec<_>>()
        };
        let inputs = ["mass", "free", "rows", "terms"].map(|name| buffer(&engine, &words(name)));
        let capacity = SoilNormalCapacity {
            dofs: metadata["dofs"].as_u64().unwrap() as u32,
            contacts: metadata["contacts"].as_u64().unwrap() as u32,
            max_iterations: metadata["max_iterations"].as_u64().unwrap() as u32,
            max_owned_bytes: u64::MAX,
        };
        let projection = ExperimentalSoilNormalProjectionGpu::new_internal(
            &engine,
            SoilNormalInputs {
                mass: &inputs[0],
                free_rates: &inputs[1],
                rows: &inputs[2],
                contacts: &inputs[3],
            },
            capacity,
            metadata["hard_rows"].as_bool().unwrap(),
        )
        .unwrap();
        encode(&engine, &projection, true);
        let got = read(&engine, projection.status_buffer())
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let expected = words("status")
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        println!(
            "normal replay {}: {got:?}, captured {expected:?}",
            path.display()
        );
        let expectation =
            std::env::var("SOIL_NORMAL_REPLAY_EXPECT").unwrap_or_else(|_| "solved".into());
        if expectation == "solved" {
            assert_eq!(got[0], 0, "captured workload still faults");
            verify_finite_capture(&engine, &inputs, &projection);
        } else if expectation == "failure" {
            assert_eq!(got, expected);
        } else {
            panic!("SOIL_NORMAL_REPLAY_EXPECT must be solved or failure");
        }
        if let Some(output) = std::env::var_os("SOIL_NORMAL_TRACE_DIR") {
            trace_replay(
                &engine,
                &inputs,
                &projection,
                capacity,
                metadata["hard_rows"].as_bool().unwrap(),
                &std::path::PathBuf::from(output),
            );
        }
        captures += 1;
    }
    assert!(captures > 0, "no captured local failure");
}

pub(super) fn trace_replay(
    engine: &Engine,
    inputs: &[wgpu::Buffer; 4],
    reference: &ExperimentalSoilNormalProjectionGpu,
    capacity: SoilNormalCapacity,
    hard: bool,
    output: &std::path::Path,
) {
    let n = capacity.dofs as usize;
    let stride = 3 * n + 6;
    let trace = buffer(
        engine,
        &vec![0.0; n + capacity.max_iterations as usize * stride],
    );
    let params = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    engine.queue().unwrap().write_buffer(
        &params,
        0,
        bytemuck::cast_slice(&[capacity.dofs, capacity.contacts, capacity.max_iterations, 0]),
    );
    let source=include_str!("../shaders/soil_normal_projection.wgsl")
        .replace("const WIDTH: u32 = 32u;",&format!("const WIDTH: u32 = {}u;",capacity.dofs))
        .replace("const HARD_WIDTH: u32 = 1u;",&format!("const HARD_WIDTH: u32 = {}u;",if hard {capacity.dofs} else {1}))
        .replace("const HARD_ROWS: bool = false;",&format!("const HARD_ROWS: bool = {hard};"))
        .replace("var impulse_converged=true;",r#"
        let trace_base=n+iteration*(3u*n+6u);
        for (var i=0u;i<n;i++) {
            rates[trace_base+i]=machine_rate(i); rates[trace_base+n+i]=gradient[i]; rates[trace_base+2u*n+i]=direction[i];
        }
        rates[trace_base+3u*n]=change; rates[trace_base+3u*n+1u]=scale;
        rates[trace_base+3u*n+2u]=slope; rates[trace_base+3u*n+3u]=f32(residual_converged);
        var impulse_converged=true;"#)
        .replace("iteration_gate=vec2<f32>",r#"
        rates[trace_base+3u*n+4u]=f32(impulse_converged);
        iteration_gate=vec2<f32>"#)
        .replace("var needs_removal=false;",r#"
        let trace_base=n+(status.iterations-1u)*(3u*n+6u);
        rates[trace_base+3u*n+5u]=-1.0;
        if iteration_gate.y!=0.0 { rates[trace_base+3u*n+5u]=f32(published_momentum_converged(active_count)); }
        var needs_removal=false;"#);
    let device = engine.device().unwrap();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test-only normal iteration trace"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = reference.pipelines[0].get_bind_group_layout(0);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("normal iteration trace"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("solve"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            &inputs[0],
            &inputs[1],
            &inputs[2],
            &inputs[3],
            &trace,
            &reference.impulses,
            &reference.normal_velocities,
            &reference.status,
            &params,
        ]
        .iter()
        .enumerate()
        .map(|(binding, b)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: b.as_entire_binding(),
        })
        .collect::<Vec<_>>(),
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.clear_buffer(&reference.status, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    let status = read(engine, &reference.status)
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let values = read(engine, &trace);
    let iterations:Vec<_>=values[n..].chunks_exact(stride).take(status[1] as usize).enumerate().map(|(i,row)|serde_json::json!({"iteration":i,"q":&row[..n],"gradient":&row[n..2*n],"direction":&row[2*n..3*n],"change":row[3*n],"scale":row[3*n+1],"slope":row[3*n+2],"residual_converged":row[3*n+3],"impulse_converged":row[3*n+4],"published_momentum_converged":row[3*n+5]})).collect();
    let mass = read(engine, &inputs[0]);
    let free = read(engine, &inputs[1]);
    let rows = read(engine, &inputs[2]);
    let impulses = read(engine, &reference.impulses);
    let q = &values[n + (status[1] as usize - 1) * stride..][..n];
    let published_residuals: Vec<_> = (0..n)
        .map(|i| {
            let machine = (0..n)
                .map(|j| f64::from(mass[i * n + j]) * (f64::from(q[j]) - f64::from(free[j])))
                .sum::<f64>();
            let contact = (0..capacity.contacts as usize)
                .map(|c| f64::from(rows[c * n + i]) * f64::from(impulses[c]))
                .sum::<f64>();
            let scale = (0..n)
                .map(|j| (f64::from(mass[i * n + j]) * f64::from(free[j])).abs())
                .sum::<f64>()
                .max(1.0);
            serde_json::json!({"dof":i,"machine":machine,"contact":contact,"scale":scale,"ratio":(machine+contact).abs()/(1e-6*scale)})
        })
        .collect();
    std::fs::create_dir(output).unwrap();
    std::fs::write(
        output.join("iterations.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"status":status,"iterations":iterations,"published_residuals":published_residuals}))
            .unwrap(),
    )
    .unwrap();
    println!(
        "trace status {status:?}; final iteration {:?}",
        iterations.last()
    );
    println!("f64 published residuals: {published_residuals:?}");
}

fn verify_finite_capture(
    engine: &Engine,
    inputs: &[wgpu::Buffer; 4],
    projection: &ExperimentalSoilNormalProjectionGpu,
) {
    let [mass, free, rows, terms] = inputs.each_ref().map(|b| {
        read(engine, b)
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>()
    });
    let n = free.len();
    let count = terms.len() / 4;
    assert!(
        (0..count).all(|c| terms[4 * c + 3] == 0.0),
        "finite-only oracle"
    );
    let mut reference = free.clone();
    let mut converged = false;
    for _ in 0..64 {
        let previous = reference.clone();
        let mut hessian = mass.clone();
        let mut rhs: Vec<_> = (0..n)
            .map(|i| (0..n).map(|j| mass[i * n + j] * free[j]).sum::<f64>())
            .collect();
        for c in 0..count {
            let m = terms[4 * c];
            if m == 0.0 {
                continue;
            }
            let bound = terms[4 * c + 1] + terms[4 * c + 2];
            let value = (0..n).map(|i| rows[c * n + i] * previous[i]).sum::<f64>();
            if value <= bound {
                continue;
            }
            for i in 0..n {
                rhs[i] += m * rows[c * n + i] * bound;
                for j in 0..n {
                    hessian[i * n + j] += m * rows[c * n + i] * rows[c * n + j];
                }
            }
        }
        reference = super::tests::solve(hessian, rhs);
        if reference
            .iter()
            .zip(previous)
            .all(|(a, b)| (a - b).abs() < 1e-12)
        {
            converged = true;
            break;
        }
    }
    assert!(converged, "f64 active set did not converge");
    let expected: Vec<_> = (0..count)
        .map(|c| {
            terms[4 * c]
                * ((0..n).map(|i| rows[c * n + i] * reference[i]).sum::<f64>()
                    - terms[4 * c + 1]
                    - terms[4 * c + 2])
                    .max(0.0)
        })
        .collect();
    let actual = read(engine, projection.rates_buffer());
    let impulses = read(engine, projection.impulses_buffer());
    let max_rate = actual
        .iter()
        .zip(&reference)
        .map(|(&a, &b)| (f64::from(a) - b).abs())
        .fold(0.0, f64::max);
    let max_impulse = impulses
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (f64::from(a) - b).abs())
        .fold(0.0, f64::max);
    assert!(max_rate <= 2e-5, "rate error {max_rate}");
    assert!(max_impulse <= 2e-5, "impulse error {max_impulse}");
    let mut max_residual_ratio = 0.0f64;
    for i in 0..n {
        let reference_residual = (0..n)
            .map(|j| mass[i * n + j] * (reference[j] - free[j]))
            .sum::<f64>()
            + (0..count)
                .map(|c| rows[c * n + i] * expected[c])
                .sum::<f64>();
        let scale = (0..n)
            .map(|j| (mass[i * n + j] * free[j]).abs())
            .sum::<f64>()
            .max(1.0);
        assert!(
            reference_residual.abs() <= 1e-10 * scale,
            "oracle KKT residual {reference_residual}"
        );
        let residual = (0..n)
            .map(|j| mass[i * n + j] * (f64::from(actual[j]) - free[j]))
            .sum::<f64>()
            + (0..count)
                .map(|c| rows[c * n + i] * f64::from(impulses[c]))
                .sum::<f64>();
        let ratio = residual.abs() / (1e-6 * scale);
        max_residual_ratio = max_residual_ratio.max(ratio);
        assert!(ratio <= 1.0, "published momentum ratio {ratio} at {i}");
    }
    println!(
        "capture oracle: max rate error {max_rate:e}, impulse error {max_impulse:e}, published momentum tolerance ratio {max_residual_ratio}"
    );
}

#[test]
#[ignore = "requires GPU timestamps and SOIL_NORMAL_CAPTURE_DIR"]
fn paired_saved_normal_benchmark() {
    let check = std::env::var("SOIL_NORMAL_PAIR_CHECK").unwrap_or_else(|_| "oracle".into());
    assert!(matches!(check.as_str(), "bits" | "oracle"));
    let baseline = std::env::var_os("SOIL_NORMAL_REFERENCE_SHADER")
        .map(|path| std::fs::read_to_string(path).expect("reference shader must be readable"))
        .unwrap_or_else(|| include_str!("../shaders/soil_normal_serial_reference.wgsl").into());
    let candidate = std::env::var_os("SOIL_NORMAL_CANDIDATE_SHADER")
        .map(|path| std::fs::read_to_string(path).expect("candidate shader must be readable"))
        .unwrap_or_else(|| include_str!("../shaders/soil_normal_projection.wgsl").into());
    let engine = Engine::new_gpu_with_features(wgpu::Features::TIMESTAMP_QUERY).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let root = std::path::PathBuf::from(
        std::env::var_os("SOIL_NORMAL_CAPTURE_DIR").expect("capture directory required"),
    );
    let mut cases = 0;
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if !path.join("metadata.json").exists() {
            continue;
        }
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(metadata["schema"], 1);
        let words = |name: &str| {
            let bytes = std::fs::read(path.join(format!("{name}.bin"))).unwrap();
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
                .collect::<Vec<_>>()
        };
        let inputs = ["mass", "free", "rows", "terms"].map(|name| buffer(&engine, &words(name)));
        let capacity = SoilNormalCapacity {
            dofs: metadata["dofs"].as_u64().unwrap() as u32,
            contacts: metadata["contacts"].as_u64().unwrap() as u32,
            max_iterations: metadata["max_iterations"].as_u64().unwrap() as u32,
            max_owned_bytes: u64::MAX,
        };
        let mut variants = [baseline.as_str(), candidate.as_str()].map(|source| {
            ExperimentalSoilNormalProjectionGpu::new_shader(
                &engine,
                SoilNormalInputs {
                    mass: &inputs[0],
                    free_rates: &inputs[1],
                    rows: &inputs[2],
                    contacts: &inputs[3],
                },
                capacity,
                metadata["hard_rows"].as_bool().unwrap(),
                source,
            )
            .unwrap()
        });
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("normal replay paired timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 8,
        });
        let resolved = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        for (i, p) in variants.iter_mut().enumerate() {
            p.pass_profile = Some((queries.clone(), 4 * i as u32));
        }
        let mut samples = [Vec::new(), Vec::new()];
        for trial in 0..8 {
            let mut encoder = device.create_command_encoder(&Default::default());
            let order = if trial % 2 == 0 { [0, 1] } else { [1, 0] };
            for i in order {
                encoder.clear_buffer(variants[i].status_buffer(), 0, None);
                variants[i].encode(&mut encoder);
            }
            encoder.resolve_query_set(&queries, 0..8, &resolved, 0);
            queue.submit([encoder.finish()]);
            let timestamp_words = read(&engine, &resolved)
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            let timestamps: Vec<_> = timestamp_words
                .chunks_exact(2)
                .map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32))
                .collect();
            let statuses = variants.each_ref().map(|p| {
                read(&engine, p.status_buffer())
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            });
            if check == "bits" {
                assert_eq!(
                    statuses[0], statuses[1],
                    "reference/candidate status mismatch"
                );
            } else {
                assert_eq!(statuses[1][0], 0, "candidate must solve captured case");
                if statuses[0][0] != 0 {
                    let captured: Vec<_> = words("status").into_iter().map(f32::to_bits).collect();
                    assert_eq!(statuses[0], captured, "unexpected baseline failure");
                }
            }
            if check == "bits" && statuses[0][0] == 0 {
                for (a, b) in [
                    (&variants[0].rates, &variants[1].rates),
                    (&variants[0].impulses, &variants[1].impulses),
                    (
                        &variants[0].normal_velocities,
                        &variants[1].normal_velocities,
                    ),
                ] {
                    let old = read(&engine, a)
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>();
                    let new = read(&engine, b)
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>();
                    assert_eq!(old, new, "same-input output words differ");
                }
            }
            let times: [f64; 2] = std::array::from_fn(|i| {
                let start = 4 * i;
                assert!(
                    timestamps[start + 1] > timestamps[start]
                        && timestamps[start + 3] > timestamps[start + 2]
                );
                ((timestamps[start + 1] - timestamps[start])
                    + (timestamps[start + 3] - timestamps[start + 2])) as f64
                    * f64::from(queue.get_timestamp_period())
                    / 1e6
            });
            if trial >= 2 {
                for i in 0..2 {
                    samples[i].push(times[i]);
                }
            }
            println!(
                "paired normal {} trial {trial} order {order:?} gpu_ms {times:?} check {check} statuses {statuses:?}",
                path.display(),
            );
        }
        for p in &variants {
            if read(&engine, p.status_buffer())[0].to_bits() == 0 {
                verify_finite_capture(&engine, &inputs, p);
            }
        }
        for sample in &mut samples {
            sample.sort_by(f64::total_cmp);
        }
        println!(
            "paired normal median ms reference={} candidate={} (six alternating measured pairs; component only)",
            samples[0][2], samples[1][2]
        );
        cases += 1;
    }
    assert!(cases > 0, "no captures");
}
