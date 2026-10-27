use super::*;
use wgpu::util::DeviceExt;

trait Bytes {
    fn append(&self, bytes: &mut Vec<u8>);
}
impl Bytes for f32 {
    fn append(&self, bytes: &mut Vec<u8>) {
        bytes.extend(self.to_le_bytes());
    }
}
impl Bytes for u32 {
    fn append(&self, bytes: &mut Vec<u8>) {
        bytes.extend(self.to_le_bytes());
    }
}
impl<T: Bytes, const N: usize> Bytes for [T; N] {
    fn append(&self, bytes: &mut Vec<u8>) {
        for v in self {
            v.append(bytes);
        }
    }
}
fn pack<T: Bytes>(values: &[T]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for v in values {
        v.append(&mut bytes);
    }
    if bytes.is_empty() {
        bytes.resize(16, 0);
    }
    bytes
}

fn payloads(a: &molla::kernels::TireFemGpuBatchAtlasBuffers, hubs: &[f32]) -> Vec<Vec<u8>> {
    let mut params = Vec::new();
    for row in &a.param_rows {
        row.counts.append(&mut params);
        row.counts2.append(&mut params);
        for v in [row.phys0, row.phys1, row.phys2, row.phys3, row.phys4] {
            v.append(&mut params);
        }
    }
    let mut offsets = Vec::new();
    for row in &a.offset_rows {
        for v in [row.node, row.tet, row.contact, row.xpbd] {
            v.append(&mut offsets);
        }
    }
    vec![
        pack(&a.rest_positions),
        pack(&a.positions),
        pack(&a.velocities),
        pack(&a.masses),
        pack(&a.tet_indices),
        pack(&a.tet_grads),
        pack(&a.tet_volumes),
        pack(&a.tet_mu),
        pack(&a.tet_lambda),
        pack(&a.bead_ids),
        pack(&a.bead_offsets),
        pack(&a.tread_ids),
        vec![0; 16 * a.positions.len()],
        vec![0; 16 * a.positions.len()],
        vec![0; 16 * a.positions.len()],
        pack(&a.tet_fiber_dir),
        pack(&a.tet_fiber_k1),
        pack(&a.tet_fiber_k2),
        pack(&a.node_tet_offsets),
        pack(&a.node_tet_entries),
        pack(hubs),
        pack(&a.tet_fiber_dir_b),
        pack(&a.tet_fiber_k1_b),
        pack(&a.tet_fiber_k2_b),
        pack(&a.xpbd_edge_indices),
        pack(&a.xpbd_edge_params),
        pack(&a.node_edge_offsets),
        pack(&a.node_edge_entries),
        pack(&a.tet_fiber_dir_c),
        pack(&a.tet_fiber_k1_c),
        pack(&a.tet_fiber_k2_c),
        pack(&a.tet_fiber_dir_d),
        pack(&a.tet_fiber_k1_d),
        pack(&a.tet_fiber_k2_d),
        pack(&a.xpbd_dihedral_indices),
        pack(&a.xpbd_dihedral_params),
        pack(&a.node_dihedral_offsets),
        pack(&a.node_dihedral_entries),
        pack(&a.node_tire_indices),
        pack(&a.tet_tire_indices),
        pack(&a.bead_tire_indices),
        pack(&a.tread_tire_indices),
        params,
        offsets,
    ]
}

fn source() -> String {
    let mut source = super::dihedral::PRODUCTION.to_owned();
    source.push_str("\n@group(0) @binding(49) var<storage, read_write> trace:array<vec4<f32>>;\n");
    for (stage, marker) in [
        "    let xpbd_volume_compliance = params.phys3.w;",
        "    let xpbd_bead_compliance = params.phys4.x;",
        "    if (params.counts2.x > 0u) {",
        "    if (params.counts2.y > 0u) {",
        "    let ply_limit = max(params.phys4.z, 1.01);",
        "    let mu_static: f32 = 1.0;",
        "    pos_out[i] = vec4<f32>(p_next, 0.0);",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(source.matches(marker).count(), 1);
        let capture = format!(
            "    trace[21u*i+{}u]=vec4<f32>(p_next,0.0); trace[21u*i+{}u]=vec4<f32>(v_next,0.0); trace[21u*i+{}u]=vec4<f32>(force,node_stiffness_diag);\n",
            3 * stage,
            3 * stage + 1,
            3 * stage + 2
        );
        source = source.replacen(marker, &(capture + marker), 1);
    }
    source
}

fn replay(
    engine: &Engine,
    data: &[Vec<u8>],
    nodes: usize,
    path: &std::path::Path,
) -> (Vec<f32>, Vec<u32>) {
    let mut packet = b"TFEMSTG1".to_vec();
    packet.extend((nodes as u32).to_le_bytes());
    packet.extend((data.len() as u32).to_le_bytes());
    for bytes in data {
        packet.extend((bytes.len() as u64).to_le_bytes());
        packet.extend(bytes);
    }
    std::fs::write(path.with_extension("inputs.bin"), packet).unwrap();
    let device = engine.device().unwrap();
    let buffers: Vec<_> = data
        .iter()
        .map(|data| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        })
        .collect();
    let trace = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (nodes * 21 * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut entries: Vec<_> = (0..44)
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
    let mut baseline = Vec::new();
    for traced in [false, true, true] {
        if traced && entries.len() == 44 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 49,
                ..entries[12]
            });
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let mut bindings: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        if traced {
            bindings.push(wgpu::BindGroupEntry {
                binding: 49,
                resource: trace.as_entire_binding(),
            });
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &bindings,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let text = if traced {
            source()
        } else {
            super::dihedral::PRODUCTION.into()
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(text.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("step_nodes_atlas_contract"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((nodes as u32).div_ceil(64), 1, 1);
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        let values: Vec<_> = buffers[12..15]
            .iter()
            .flat_map(|b| read_floats(engine, b))
            .map(f32::to_bits)
            .collect();
        if !traced {
            baseline = values;
        } else {
            assert_eq!(baseline, values, "trace perturbed node outputs");
        }
    }
    std::fs::write(path.with_extension("outputs.bin"), pack(&baseline)).unwrap();
    let traced = read_floats(engine, &trace);
    assert!(traced.iter().all(|v| v.is_finite()));
    std::fs::write(path.with_extension("trace.bin"), pack(&traced)).unwrap();
    (traced, baseline)
}

pub(super) fn run(
    engine: &Engine,
    gpu: &molla::kernels::TireFemGpuBatchAtlasBuffers,
    gpu_hubs: &[f32],
    cpu: &molla::kernels::TireFemGpuBatchAtlasBuffers,
    cpu_hubs: &[f32],
) -> (Vec<u32>, Vec<u32>) {
    let dir = std::path::PathBuf::from(
        std::env::var_os("MOLLA_TIRE_STAGE_PROBE_DIR").expect("probe output directory"),
    );
    assert!(!dir.exists());
    std::fs::create_dir_all(&dir).unwrap();
    let nodes = gpu.positions.len();
    assert_eq!(cpu.positions.len(), nodes);
    let (a, gpu_output) = replay(engine, &payloads(gpu, gpu_hubs), nodes, &dir.join("gpu"));
    let (b, cpu_output) = replay(engine, &payloads(cpu, cpu_hubs), nodes, &dir.join("cpu"));
    std::fs::write(dir.join("instrumented.wgsl"), source()).unwrap();
    for (stage, name) in [
        "integration",
        "volume",
        "bead",
        "edges",
        "dihedral",
        "ply",
        "floor",
    ]
    .iter()
    .enumerate()
    {
        let mut delta = [0f64; 3];
        let mut at = [0; 3];
        for node in 0..nodes {
            for component in 0..3 {
                for axis in 0..3 {
                    let i = node * 84 + stage * 12 + component * 4 + axis;
                    let d = (a[i] as f64 - b[i] as f64).abs();
                    if d > delta[component] {
                        delta[component] = d;
                        at[component] = node;
                    }
                }
            }
        }
        println!(
            "third-substep stage {name}: position/velocity/force max {delta:?} at nodes {at:?}; instrumented output words exact to ordinary shader twice"
        );
    }
    (gpu_output, cpu_output)
}
