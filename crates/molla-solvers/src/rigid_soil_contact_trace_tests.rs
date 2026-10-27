use super::*;
use crate::kernels::{FeatherstoneExecution, FeatherstoneRounding};
use crate::soil_contact_projection_gpu::SoilProjectionReplay;
use std::io::Write;

const FIRST_FRAME: usize = 23;
const LAST_FRAME: usize = 26;
const SUBSTEPS: usize = 34;
const RECORDS: usize = (LAST_FRAME - FIRST_FRAME + 1) * SUBSTEPS;

pub(super) struct Probe {
    captures: [SoilProjectionReplay; 2],
    geometry: [[wgpu::Buffer; 3]; 2],
    staging: wgpu::Buffer,
    trace: std::fs::File,
    checked: usize,
    grid: ([f32; 3], f32, u32),
}

fn grid_index(arm: [f32; 3], center: &[f32], grid: ([f32; 3], f32, u32)) -> usize {
    let (origin, h, size) = grid;
    let xyz: [usize; 3] = std::array::from_fn(|i| {
        let cell = (f64::from(arm[i]) + f64::from(center[i]) - f64::from(origin[i])) / f64::from(h);
        let integer = cell.round();
        assert!((cell - integer).abs() < 0.002, "ambiguous grid node");
        assert!(integer >= 0.0 && integer < f64::from(size));
        integer as usize
    });
    xyz[0] + size as usize * (xyz[1] + size as usize * xyz[2])
}

fn floats(words: &[u32]) -> Vec<f32> {
    let values: Vec<_> = words.iter().copied().map(f32::from_bits).collect();
    assert!(values.iter().all(|x| x.is_finite()));
    values
}

impl Probe {
    fn new(engine: &Engine, soil: &mut RuntimeSoil, path: &std::path::Path) -> Self {
        let projection = soil.projection.as_mut().unwrap();
        let captures = projection.prepare_replay_capture();
        let geometry = projection.trace_geometry();
        let bytes = captures[0]
            .inputs
            .iter()
            .chain(&captures[0].intermediates)
            .chain(&captures[0].outputs)
            .chain(&geometry[0])
            .map(wgpu::Buffer::size)
            .sum();
        assert!(bytes <= 4 * 1024 * 1024);
        let staging = engine
            .device()
            .unwrap()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("bounded soil contact trace"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
        let cfg = soil.soil.config();
        Self {
            captures,
            geometry,
            staging,
            trace: std::fs::File::create_new(path).unwrap(),
            checked: 0,
            grid: (cfg.grid_origin, cfg.cell_size, cfg.grid_size),
        }
    }

    pub(super) fn record(&mut self, engine: &Engine, parity: usize, dt: f32) {
        assert!(self.checked < RECORDS);
        assert_eq!(parity, self.checked % 2);
        let capture = &self.captures[parity];
        let buffers: Vec<_> = capture
            .inputs
            .iter()
            .chain(&capture.intermediates)
            .chain(&capture.outputs)
            .chain(&self.geometry[parity])
            .collect();
        let device = engine.device().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut offset = 0;
        for source in &buffers {
            encoder.copy_buffer_to_buffer(source, 0, &self.staging, offset, source.size());
            offset += source.size();
        }
        assert_eq!(offset, self.staging.size());
        engine.queue().unwrap().submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
        rx.recv_timeout(std::time::Duration::from_secs(10))
            .unwrap()
            .unwrap();
        let mapped = self.staging.slice(..).get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let mut start = 0;
        let data: Vec<&[u32]> = buffers
            .iter()
            .map(|buffer| {
                let end = start + buffer.size() as usize / 4;
                let result = &words[start..end];
                start = end;
                result
            })
            .collect();
        assert_eq!(data.len(), 16);
        assert!(data[12].iter().all(|&value| value == 0), "projection fault");
        let centers = floats(data[13]);
        let shapes = floats(data[14]);
        let owners = data[15];
        assert_eq!(owners.len(), (self.grid.2 as usize).pow(3));
        assert!(
            owners
                .iter()
                .all(|&owner| owner == u32::MAX || owner < capture.capacity.bodies)
        );
        let mut bodies = Vec::new();
        for body in 0..capture.capacity.bodies as usize {
            let first = data[1][2 * body] as usize;
            let count = data[1][2 * body + 1] as usize;
            assert!(first + count <= capture.capacity.nodes as usize);
            let center = &centers[4 * body..4 * body + 4];
            let mut nodes = Vec::new();
            let mut keys = std::collections::BTreeSet::new();
            for slot in first..first + count {
                let node = floats(&data[0][8 * slot..8 * slot + 8]);
                let key = grid_index(node[..3].try_into().unwrap(), center, self.grid);
                assert_eq!(owners[key], body as u32);
                assert!(keys.insert(key), "duplicate contact node");
                assert!(node[3] > 0.0);
                assert!(node[7] >= 0.0 && node[7] <= 63.0 && node[7].fract() == 0.0);
                nodes.push(serde_json::json!({"slot":slot,"grid_index":key,"node":node}));
            }
            assert_eq!(
                keys.len(),
                owners.iter().filter(|&&owner| owner == body as u32).count()
            );
            bodies.push(serde_json::json!({
                "body":body,"span":[first,count],"center":center,
                "shape":&shapes[16*body..16*body+16],"nodes":nodes
            }));
        }
        let row = serde_json::json!({
            "frame":FIRST_FRAME+self.checked/SUBSTEPS,
            "substep":self.checked%SUBSTEPS,"input_parity":parity,"dt":dt,
            "bodies":bodies,"mass":floats(data[2]),"jacobians":floats(data[3]),
            "free_rates":floats(data[4]),"moments":floats(data[5]),
            "system":floats(data[6]),"rhs":floats(data[7]),"rates":floats(data[8]),
            "motion":floats(data[9]),"impulses":floats(data[10]),
            "boundary_impulses":floats(data[11]),"status":data[12]
        });
        writeln!(self.trace, "{row}").unwrap();
        self.trace.flush().unwrap();
        drop(mapped);
        self.staging.unmap();
        self.checked += 1;
    }
}

#[test]
fn contact_trace_grid_identity_handles_signed_origin_and_pose_roundoff() {
    let grid = ([-3.0, -0.5, -3.0], 0.1, 64);
    let center = [0.03123f32, -0.194321, -0.81234];
    for xyz in [[0, 0, 0], [63, 63, 63], [21, 2, 35], [30, 4, 29]] {
        let arm = std::array::from_fn(|i| grid.0[i] + xyz[i] as f32 * grid.1 - center[i]);
        assert_eq!(
            grid_index(arm, &center, grid),
            xyz[0] + 64 * (xyz[1] + 64 * xyz[2])
        );
    }
}

#[test]
#[should_panic(expected = "ambiguous grid node")]
fn contact_trace_rejects_non_grid_arms() {
    grid_index([0.025, 0.0, 0.0], &[0.0; 3], ([0.0; 3], 0.1, 64));
}

#[test]
#[ignore = "requires native GPU and two 50k trajectories with bounded contact tracing"]
fn trace_partial_and_extended_contact_transition() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("SOIL_CONTACT_TRACE_DIR")
            .expect("fresh absolute trace directory required"),
    );
    assert!(directory.is_absolute());
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":1,"first_frame":FIRST_FRAME,"last_frame":LAST_FRAME,"substeps":SUBSTEPS,
            "particles":50000,"grid_origin":[-3.0,-0.5,-3.0],"cell_size":0.1,"grid_size":64,
            "node_order":"actual gathered order; grid_index validated against exact GPU ownership",
            "scope":"independent instrumented trajectories; not performance or physical calibration"
        }))
        .unwrap(),
    )
    .unwrap();
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    for (label, execution) in [
        (
            "partial",
            FeatherstoneExecution::ExperimentalWorkgroupScratch,
        ),
        (
            "extended",
            FeatherstoneExecution::ExperimentalConstraintScratch,
        ),
    ] {
        eprintln!("constructing {label} contact-transition trajectory");
        let mut tractor =
            crate::soil_tractor::SoilTractor::new_native_explicit_with_machine_options(
                engine.clone(),
                crate::soil_tractor::SoilTractorConfig::default(),
                FeatherstoneRounding::Storage,
                execution,
            )
            .unwrap();
        tractor.enable_mass_coupling(4096).unwrap();
        for frame in 1..=LAST_FRAME {
            if frame == FIRST_FRAME {
                let gpu = tractor.world_mut_for_test().gpu.as_mut().unwrap();
                assert_eq!(gpu.execution, execution);
                let soil = gpu.soil.as_mut().unwrap();
                soil.contact_trace = Some(Probe::new(
                    &engine,
                    soil,
                    &directory.join(format!("{label}.jsonl")),
                ));
            }
            tractor.step(1.0 / 60.0, [300.0; 4]).unwrap();
            if frame >= FIRST_FRAME {
                let gpu = tractor.world_mut_for_test().gpu.as_mut().unwrap();
                let checked = gpu
                    .soil
                    .as_ref()
                    .unwrap()
                    .contact_trace
                    .as_ref()
                    .unwrap()
                    .checked;
                assert_eq!(checked, (frame - FIRST_FRAME + 1) * SUBSTEPS);
                println!("{label}: committed frame {frame}, retained {checked} contact substeps");
            }
        }
        assert_eq!(
            tractor.world().mpm_soil_stats().unwrap().committed_frames,
            LAST_FRAME as u64
        );
    }
}

#[test]
#[ignore = "requires GPU and completed SOIL_CONTACT_TRACE_DIR observations"]
fn saved_contact_transition_reproduces_geometric_membership() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("SOIL_CONTACT_TRACE_DIR").expect("saved trace directory required"),
    );
    let load = |label| -> Vec<serde_json::Value> {
        std::fs::read_to_string(directory.join(format!("{label}.jsonl")))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    let a = load("partial");
    let b = load("extended");
    assert_eq!(a.len(), RECORDS);
    assert_eq!(b.len(), RECORDS);
    let boundary_keys = |row: &serde_json::Value| {
        row["bodies"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|body| {
                body["nodes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|node| node["node"][7].as_f64().unwrap() != 0.0)
                    .map(|node| {
                        (
                            body["body"].as_u64().unwrap(),
                            node["grid_index"].as_u64().unwrap(),
                        )
                    })
            })
            .collect::<std::collections::BTreeSet<_>>()
    };
    let (left, right) = a
        .iter()
        .zip(&b)
        .find(|(x, y)| boundary_keys(x) != boundary_keys(y))
        .expect("captured boundary-membership disagreement required");
    assert_eq!(left["frame"], right["frame"]);
    assert_eq!(left["substep"], right["substep"]);
    let keys: Vec<_> = boundary_keys(left)
        .symmetric_difference(&boundary_keys(right))
        .copied()
        .collect();
    assert_eq!(keys.len(), 1);
    let (body, index) = keys[0];
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let source = format!(
        "{}\n{}",
        include_str!("../shaders/mpm_mass_coupling.wgsl"),
        r#"
@compute @workgroup_size(1)
fn membership_probe() {
    let arm=point(params.pad)-centers[0].xyz;
    let q=shapes[2]; let s=shapes[3];
    let local=rotate(vec4<f32>(-q.xyz,q.w),arm);
    let margin=min(s.x-abs(local.x),min(s.y-abs(local.y),s.z-abs(local.z)));
    nodes[0]=Node(vec4<f32>(arm,select(0.0,1.0,contains(0u,arm))),vec4<f32>(local,margin));
}
"#
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("saved production contact membership"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("saved production contact membership"),
        layout: None,
        module: &module,
        entry_point: Some("membership_probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let make = |size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("saved membership probe"),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let centers = make(16, storage);
    let shapes = make(64, storage);
    let output = make(32, storage);
    let params = make(
        48,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let staging = make(
        32,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("saved membership probe"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[(1, &centers), (2, &shapes), (3, &output), (11, &params)].map(
            |(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            },
        ),
    });
    let parameters = [
        (-3.0f32).to_bits(),
        (-0.5f32).to_bits(),
        (-3.0f32).to_bits(),
        0.1f32.to_bits(),
        64,
        1,
        1,
        64 * 64 * 64,
        0,
        0,
        0,
        index as u32,
    ];
    queue.write_buffer(&params, 0, bytemuck::cast_slice(&parameters));
    let mut results = Vec::new();
    for (label, row) in [("partial", left), ("extended", right)] {
        let record = &row["bodies"][body as usize];
        let center: Vec<f32> = serde_json::from_value(record["center"].clone()).unwrap();
        let shape: Vec<f32> = serde_json::from_value(record["shape"].clone()).unwrap();
        assert_eq!(shape[15], 2.0, "box fixture required for signed margin");
        queue.write_buffer(&centers, 0, bytemuck::cast_slice(&center));
        queue.write_buffer(&shapes, 0, bytemuck::cast_slice(&shape));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 32);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
        rx.recv_timeout(std::time::Duration::from_secs(10))
            .unwrap()
            .unwrap();
        let mapped = staging.slice(..).get_mapped_range();
        let actual: &[f32] = bytemuck::cast_slice(&mapped);
        assert!(actual.iter().all(|x| x.is_finite()));
        let occupied = record["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["grid_index"].as_u64() == Some(index));
        assert_eq!(
            actual[3],
            if occupied { 1.0 } else { 0.0 },
            "{label}: production membership differs from trace"
        );
        assert_eq!(actual[7] > 0.0, occupied);
        let result = serde_json::json!({"profile":label,"frame":row["frame"],"substep":row["substep"],
            "body":body,"grid_index":index,"occupied":occupied,"arm":&actual[..3],
            "local":&actual[4..7],"box_margin_m":actual[7],"center":center,"shape":shape});
        println!("{result}");
        results.push(result);
        drop(mapped);
        staging.unmap();
    }
    let file = std::fs::File::create_new(directory.join("geometry-replay.json")).unwrap();
    serde_json::to_writer_pretty(file, &results).unwrap();
}
