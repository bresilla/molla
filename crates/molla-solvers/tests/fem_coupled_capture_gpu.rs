#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use std::path::Path;
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

const PROBE: &str = r#"
@group(2) @binding(5) var<storage, read_write> probe_results: array<vec4<f32>>;
@group(2) @binding(6) var<storage, read_write> probe_tets: array<vec4<f32>>;
@compute @workgroup_size(64)
fn coupled_capture(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= u.particle_count { return; }
    probe_tets[i] = particle_qd[i];
}
@compute @workgroup_size(64)
fn coupled_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= u.particle_count { return; }
    inertia_y[i] = vec4<f32>(inertia_y[i].xyz + u.dt * (particle_qd[i].xyz - probe_tets[i].xyz), 0.0);
    particle_delta[i] = vec4<f32>(u.dt * particle_qd[i].xyz, 0.0);
}
@compute @workgroup_size(64)
fn coupled_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= u.particle_count { return; }
    particle_qd[i] = vec4<f32>(particle_delta[i].xyz / u.dt, 0.0);
}
@compute @workgroup_size(1)
fn local_probe() {
    let v = u.color_index;
    for (var i = 0u; i < 129u; i++) {
        solve_vertex(v, true, true);
        probe_results[i] = vec4<f32>(particle_delta[v].xyz, inertia_y[v].w);
        if i < 128u { solve_vertex(v, false, false); }
    }
}
@compute @workgroup_size(1)
fn contribution_probe() {
    let v = u.color_index;
    let start = vertex_adj_tets_offsets[v];
    let end = vertex_adj_tets_offsets[v + 1u];
    for (var t = start; t < end; t += 2u) {
        var force: vec3<f32>;
        var hessian: mat3x3<f32>;
        tet_neo_hookean_block(vertex_adj_tets[t], vertex_adj_tets[t + 1u], true, &force, &hessian);
        probe_results[(t - start) / 2u] = vec4<f32>(force, 0.0);
    }
    let spring_start = vertex_adj_offsets[v];
    let spring_end = vertex_adj_offsets[v + 1u];
    for (var s = spring_start; s < spring_end; s++) {
        var force: vec3<f32>;
        var hessian: mat3x3<f32>;
        let index = vertex_adj_springs[s];
        spring_force_hessian(v, index, true, &force, &hessian);
        probe_results[96u + s - spring_start] = vec4<f32>(force, f32(index));
    }
}
"#;

#[test]
#[ignore = "requires GPU and MOLLA_LOCAL_MATERIAL_FIXTURE through make test-gpu"]
fn captured_node_updates_with_fixed_neighbors() {
    let root = std::env::var("MOLLA_LOCAL_MATERIAL_FIXTURE").expect("native fixture directory");
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut source = format!(
        "{}\n{}\n{PROBE}",
        include_str!("../../molla-kernels/shaders/tet_viscosity.wgsl"),
        include_str!("../../molla-kernels/shaders/vbd_cloth.wgsl"),
    );
    let detail = std::env::var_os("MOLLA_LOCAL_TET_DETAIL").is_some();
    let stress_detail = std::env::var_os("MOLLA_LOCAL_STRESS_DETAIL").is_some();
    assert!(!stress_detail || detail);
    if std::env::var_os("MOLLA_ACCURATE_NEWTON").is_some() {
        for (old, new) in [
            (
                "spring_force_hessian(v, spring_idx, accurate_force, &f_s, &h_s);",
                "spring_force_hessian(v, spring_idx, true, &f_s, &h_s);",
            ),
            (
                "tet_neo_hookean_block(tet_id, v_order, accurate_force, &f_t, &h_t);",
                "tet_neo_hookean_block(tet_id, v_order, true, &f_t, &h_t);",
            ),
        ] {
            assert_eq!(source.matches(old).count(), 1);
            source = source.replace(old, new);
        }
    }
    let gradient_detail = std::env::var_os("MOLLA_LOCAL_GRADIENT_DETAIL").is_some();
    assert!(!gradient_detail || detail);
    assert!(!gradient_detail || !stress_detail);
    let cached_previous = std::env::var_os("MOLLA_PROBE_CACHE_PREVIOUS").is_some();
    let runtime_cache = std::env::var_os("MOLLA_RUNTIME_CACHE_PREVIOUS").is_some();
    assert!(!runtime_cache || (!detail && !cached_previous));
    if runtime_cache {
        source = source.replace(
            "const CACHE_PREVIOUS_TETS: bool = false;",
            "const CACHE_PREVIOUS_TETS: bool = true;",
        );
    }
    assert!(!cached_previous || !detail);
    if cached_previous {
        let original =
            "let previous_parts = tet_previous_parts(tet_id, mat3x3<f32>(v1, v2, v3), dm_inv);";
        assert_eq!(source.matches(original).count(), 1);
        source = source.replace(
            original,
            "let previous_parts = probe_previous_parts(tet_id);",
        );
        source.push_str(
            r#"
fn probe_previous_parts(tet: u32) -> TetGradient {
    let base = tet * 6u;
    return TetGradient(
        mat3x3<f32>(probe_tets[base].xyz, probe_tets[base+1u].xyz, probe_tets[base+2u].xyz),
        mat3x3<f32>(probe_tets[base+3u].xyz, probe_tets[base+4u].xyz, probe_tets[base+5u].xyz));
}
@compute @workgroup_size(64)
fn probe_cache_previous(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tet = gid.x;
    if tet >= arrayLength(&tet_rest_volume) { return; }
    let origin = tet_indices[tet*4u];
    let edges = mat3x3<f32>(
        tet_displacement_edge(origin, tet_indices[tet*4u+1u]),
        tet_displacement_edge(origin, tet_indices[tet*4u+2u]),
        tet_displacement_edge(origin, tet_indices[tet*4u+3u]));
    let raw = tet_dm_inv[tet];
    let inverse = mat3x3<f32>(raw.col0.xyz, raw.col1.xyz, raw.col2.xyz);
    let parts = tet_previous_gradient_parts(tet, edges, inverse);
    for (var column = 0u; column < 3u; column++) {
        probe_tets[tet*6u+column] = vec4<f32>(parts.value[column], 0.0);
        probe_tets[tet*6u+3u+column] = vec4<f32>(parts.correction[column], 0.0);
    }
}
"#,
        );
    }
    if detail {
        source = source.replace("var cof = cofactor3(f_mat);", "let record = (tet_id * 4u + v_order) * 12u; probe_tets[record] = vec4<f32>(f_mat[0], 0.0); probe_tets[record + 1u] = vec4<f32>(f_mat[1], 0.0); probe_tets[record + 2u] = vec4<f32>(f_mat[2], 0.0); var cof = cofactor3(f_mat);");
        source = source.replace("*out_force = f_clean;", "probe_tets[record + 3u] = vec4<f32>(f_clean, 0.0); probe_tets[record + 11u] = vec4<f32>(m, 0.0); *out_force = f_clean;");
        source = source.replace("*out_force += viscous.force;", "probe_tets[record + 4u] = vec4<f32>(viscous.force, 0.0); for (var column = 0u; column < 3u; column++) { probe_tets[record + 5u + column] = vec4<f32>(previous[column], 0.0); probe_tets[record + 8u + column] = vec4<f32>(increment[column], 0.0); } *out_force += viscous.force;");
    }
    if gradient_detail {
        source = source.replace("* 12u; probe_tets", "* 32u; probe_tets");
        source.push_str("\nvar<private> gradient_record: u32;\n");
        source = source.replace("let previous_parts = tet_previous_parts", "gradient_record = (tet_id * 4u + v_order) * 32u; let previous_parts = tet_previous_parts");
        source = source.replace("product.value[column] = value;", r#"
        if column == 1u && origin[1].y == 1.0 {
            for (var j = 0u; j < 3u; j++) {
                probe_tets[gradient_record + 12u + j] = vec4<f32>(a[j], 0.0);
                probe_tets[gradient_record + 15u + j] = vec4<f32>(b[j], 0.0);
            }
            probe_tets[gradient_record + 18u] = vec4<f32>(p0, 0.0);
            probe_tets[gradient_record + 19u] = vec4<f32>(p1, 0.0);
            probe_tets[gradient_record + 20u] = vec4<f32>(p2, 0.0);
            probe_tets[gradient_record + 21u] = vec4<f32>(sum01, 0.0);
            probe_tets[gradient_record + 22u] = vec4<f32>(sum012, 0.0);
            probe_tets[gradient_record + 23u] = vec4<f32>(value, 0.0);
            probe_tets[gradient_record + 24u] = vec4<f32>(error, 0.0);
            probe_tets[gradient_record + 25u] = vec4<f32>(fma(a[0], vec3<f32>(b[column].x), -p0), 0.0);
            probe_tets[gradient_record + 26u] = vec4<f32>(fma(a[1], vec3<f32>(b[column].y), -p1), 0.0);
            probe_tets[gradient_record + 27u] = vec4<f32>(fma(a[2], vec3<f32>(b[column].z), -p2), 0.0);
            probe_tets[gradient_record + 28u] = vec4<f32>(tet_sum_error(p0, p1, sum01), 0.0);
            probe_tets[gradient_record + 29u] = vec4<f32>(tet_sum_error(sum01, p2, sum012), 0.0);
            probe_tets[gradient_record + 30u] = vec4<f32>(tet_sum_error(sum012, origin[column], value), 0.0);
            probe_tets[gradient_record + 31u] = vec4<f32>(value + error, 0.0);
        }
        product.value[column] = value;
        "#);
    }
    if stress_detail {
        source = source.replace("* 12u; probe_tets", "* 32u; probe_tets");
        source = source.replace(
            "let p_vec = mat3x3<f32>(",
            r#"
        for (var c = 0u; c < 3u; c++) {
            probe_tets[record + 12u + c] = vec4<f32>(p_mat[c], 0.0);
            probe_tets[record + 15u + c] = vec4<f32>(f_low[c], 0.0);
            probe_tets[record + 18u + c] = vec4<f32>(cof[c], 0.0);
            probe_tets[record + 21u + c] = vec4<f32>(cof_rounding[c], 0.0);
            probe_tets[record + 25u + c] = vec4<f32>(previous_parts.correction[c], 0.0);
            probe_tets[record + 28u + c] = vec4<f32>(increment_parts.correction[c], 0.0);
        }
        probe_tets[record + 24u] = vec4<f32>(j_minus_one, lam, mu, 0.0);
        let p_vec = mat3x3<f32>(
        "#,
        );
    }
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fixed-neighbor native material probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let counts = [7, 9, 7, 1, 5, 17];
    let writable = |g, b| g == 2 || (g == 5 && [10, 11, 14].contains(&b));
    let layouts: Vec<_> = counts
        .iter()
        .enumerate()
        .map(|(g, &count)| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &(0..count)
                    .map(|b| wgpu::BindGroupLayoutEntry {
                        binding: b,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if g == 3 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: !writable(g, b),
                                }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &layouts.iter().map(Some).collect::<Vec<_>>(),
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("local_probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut nodes: Vec<i32> = std::fs::read_dir(&root)
        .unwrap()
        .filter_map(|entry| entry.unwrap().file_name().to_str().unwrap().parse().ok())
        .collect();
    assert!(!nodes.is_empty(), "no numeric fixture directories");
    nodes.sort();
    for node in nodes {
        assert!(
            !cached_previous || node == -1,
            "cache probe requires whole mesh"
        );
        assert!(
            !runtime_cache || node == -1,
            "runtime cache probe requires whole mesh"
        );
        let directory = Path::new(&root).join(node.to_string());
        if node == -1 && !directory.exists() {
            continue;
        }
        let buffers: Vec<Vec<_>> = counts
            .iter()
            .enumerate()
            .map(|(g, &count)| {
                (0..count)
                    .map(|b| {
                        let path = directory.join(format!("{g}-{b}.bin"));
                        let mut data = if path.exists() {
                            std::fs::read(path).unwrap()
                        } else {
                            vec![0; 64]
                        };
                        if data.is_empty() {
                            data.resize(64, 0);
                        }
                        if g == 2 && b == 5 {
                            data.resize(129 * 16, 0);
                        }
                        if g == 2 && b == 2 && runtime_cache {
                            let tets =
                                std::fs::metadata(directory.join("5-2.bin")).unwrap().len() / 4;
                            data.resize(data.len() + tets as usize * 96, 0);
                        }
                        if g == 2 && b == 6 && cached_previous {
                            let tets =
                                std::fs::metadata(directory.join("5-2.bin")).unwrap().len() / 4;
                            data.resize(tets as usize * 96, 0);
                        }
                        if g == 2 && b == 6 && directory.join("contact-final-rows.bin").exists() {
                            data.resize(
                                std::fs::metadata(directory.join("2-1.bin")).unwrap().len()
                                    as usize,
                                0,
                            );
                        } else if g == 2 && b == 6 && detail {
                            let indices =
                                std::fs::metadata(directory.join("5-0.bin")).unwrap().len();
                            data.resize(
                                indices as usize
                                    * if gradient_detail || stress_detail {
                                        128
                                    } else {
                                        48
                                    },
                                0,
                            );
                        }
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: &data,
                            usage: wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST
                                | if g == 3 {
                                    wgpu::BufferUsages::UNIFORM
                                } else {
                                    wgpu::BufferUsages::STORAGE
                                },
                        })
                    })
                    .collect()
            })
            .collect();
        let groups: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(g, buffers)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layouts[g],
                    entries: &buffers
                        .iter()
                        .enumerate()
                        .map(|(b, buffer)| wgpu::BindGroupEntry {
                            binding: b as u32,
                            resource: buffer.as_entire_binding(),
                        })
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        if node == -1 {
            global_probe(
                &engine, &directory, &shader, &layout, &layouts, &buffers, &groups,
            );
            continue;
        }
        let contributions = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("contribution_probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&contributions);
            for (i, group) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let values = readback::read::<[f32; 4]>(&engine, &buffers[2][5], 129);
        std::fs::write(
            directory.join("contributions.bin"),
            bytemuck::cast_slice(&values),
        )
        .unwrap();
        if detail {
            let values = readback::read::<[f32; 4]>(
                &engine,
                &buffers[2][6],
                buffers[2][6].size() as usize / 16,
            );
            std::fs::write(
                directory.join("tet-detail.bin"),
                bytemuck::cast_slice(&values),
            )
            .unwrap();
        }
        let before =
            readback::read::<[f32; 4]>(&engine, &buffers[2][2], buffers[2][2].size() as usize / 16);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for (i, group) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let results = readback::read::<[f32; 4]>(&engine, &buffers[2][5], 129);
        let after = readback::read::<[f32; 4]>(&engine, &buffers[2][2], before.len());
        let uniform = std::fs::read(directory.join("3-0.bin")).unwrap();
        let local = u32::from_le_bytes(uniform[12..16].try_into().unwrap()) as usize;
        for (i, (a, b)) in before.iter().zip(&after).enumerate() {
            if i != local {
                assert_eq!(a, b, "neighbor changed: node={node} local={i}");
            }
        }
        assert!(results.iter().flatten().all(|v| v.is_finite()));
        let original = std::fs::read(directory.join("2-3.bin")).unwrap();
        let residual = f32::from_le_bytes(
            original[16 * local + 12..16 * local + 16]
                .try_into()
                .unwrap(),
        );
        if std::env::var_os("MOLLA_LOCAL_MATERIAL_REQUIRE_BASELINE").is_some() {
            assert_eq!(
                results[0][3], residual,
                "native residual mismatch at node {node}"
            );
        }
        std::fs::write(
            directory.join("local-updates.bin"),
            bytemuck::cast_slice(&results),
        )
        .unwrap();
        for i in [0, 1, 2, 4, 8, 16, 32, 64, 128] {
            println!(
                "LOCAL node={node} iteration={i} delta_residual={:?}",
                results[i]
            );
        }
    }
}

fn storage_copy(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    source: &wgpu::Buffer,
    destination: &wgpu::Buffer,
) {
    assert_eq!(source.size(), destination.size());
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fixture storage restore"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;
@compute @workgroup_size(64)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < arrayLength(&source) { destination[gid.x] = source[gid.x]; }
}
"#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("copy"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: destination.as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((source.size() as u32 / 4).div_ceil(64), 1, 1);
}

fn coupled_probe(
    engine: &Engine,
    directory: &Path,
    material: &[Vec<wgpu::Buffer>],
    groups: &[wgpu::BindGroup],
    uniforms: &[wgpu::BindGroup],
    offsets: &[u32],
    solve: &wgpu::ComputePipeline,
    residual: &wgpu::ComputePipeline,
    capture: &wgpu::ComputePipeline,
    accumulate: &wgpu::ComputePipeline,
    velocity: &wgpu::ComputePipeline,
    count: u32,
) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let names = [
        "positions",
        "velocities",
        "inverse_mass",
        "contacts",
        "body_poses",
        "body_com",
        "jacobians",
        "mass_matrix",
        "rigid_velocities",
        "body_velocities",
    ];
    let upload = |bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("captured continuation input"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    };
    let buffers: Vec<_> = names
        .iter()
        .map(|name| upload(&std::fs::read(directory.join(format!("contact-{name}.bin"))).unwrap()))
        .collect();
    let rows = (buffers[3].size() / 80) as u32;
    let dofs = (buffers[8].size() / 4) as u32;
    let bodies = (buffers[4].size() / 32) as u32;
    let kernel = if std::env::var_os("MOLLA_CONTACT_BLOCKS").is_some() {
        SoftRigidSolveKernel::new_blocked_contacts(device, dofs, rows, count)
    } else {
        SoftRigidSolveKernel::new_separated_static(device, dofs, rows, count)
    }
    .unwrap();
    let bindings = SoftRigidSolveBindings {
        positions: &buffers[0],
        velocities: &material[2][1],
        inverse_mass: &buffers[2],
        contacts: &buffers[3],
        body_poses: &buffers[4],
        body_com: &buffers[5],
        jacobians: &buffers[6],
        mass_matrix: &buffers[7],
        rigid_velocities: &buffers[8],
        body_velocities: &buffers[9],
        particle_count: count,
        body_count: bodies,
    };
    let base = std::fs::read(directory.join("3-0.bin")).unwrap();
    let contact_sweeps = std::env::var("MOLLA_COUPLED_CONTACT_SWEEPS")
        .map(|v| v.parse::<u32>().unwrap())
        .unwrap_or(1);
    let material_sweeps = std::env::var("MOLLA_COUPLED_MATERIAL_SWEEPS")
        .map(|v| v.parse::<u32>().unwrap())
        .unwrap_or(8);
    assert!((1..=4096).contains(&contact_sweeps) && material_sweeps <= 4096);
    let output = std::env::var_os("MOLLA_COUPLED_OUTPUT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| directory.to_path_buf());
    std::fs::create_dir_all(&output).unwrap();
    println!(
        "COUPLED contact_sweeps={contact_sweeps} material_sweeps={material_sweeps} output={}",
        output.display()
    );
    let settings = SoftRigidSolveSettings {
        iterations: contact_sweeps,
        dt: f32::from_le_bytes(base[20..24].try_into().unwrap()),
        penetration_recovery: 0.2,
        max_recovery_speed: 0.5,
    };
    let mut restored_rows = std::fs::read(directory.join("contact-final-rows.bin")).unwrap();
    let ground_control = std::env::var_os("MOLLA_COUPLED_GROUND_SHAPE");
    if let Some(path) = ground_control.as_ref() {
        use molla_kernels::soft_rigid_contact::{
            SoftRigidContactBindings, SoftRigidContactGpu, SoftRigidContactKernel,
            SoftRigidSampleGpu, SoftRigidShapeGpu,
        };
        let raw = std::fs::read(path).unwrap();
        let shape: SoftRigidShapeGpu = bytemuck::pod_read_unaligned(&raw);
        assert_eq!(shape.ids[0..2], [1, u32::MAX]);
        assert_eq!(shape.rotation, [0.0, 0.0, 0.0, 1.0]);
        let original = std::fs::read(directory.join("contact-contacts.bin")).unwrap();
        let mut contacts: Vec<SoftRigidContactGpu> = original
            .chunks_exact(80)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let selected: Vec<_> = contacts
            .iter()
            .enumerate()
            .filter(|(_, c)| c.ids[1] == u32::MAX && c.ids[3] == 1)
            .map(|(i, _)| i)
            .collect();
        assert!(!selected.is_empty());
        let samples: Vec<_> = selected
            .iter()
            .map(|&i| {
                let c = contacts[i];
                assert_eq!(c.ids[2], 70);
                assert!(
                    c.normal_gap[0] == 0.0
                        && c.normal_gap[2] == 0.0
                        && (c.normal_gap[1] - 1.0).abs() < 1e-6
                );
                SoftRigidSampleGpu {
                    nodes: [c.ids[0], c.support[0], c.support[1], c.support[2]],
                    weights: c.weights,
                }
            })
            .collect();
        let surface = upload(bytemuck::cast_slice(&samples));
        let shapes = upload(&raw);
        let radii = upload(&vec![0u8; count as usize * 4]);
        let gather = SoftRigidContactKernel::new_sampled(device, samples.len() as u32, 1).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        gather
            .encode(
                device,
                &mut encoder,
                SoftRigidContactBindings {
                    positions: &buffers[0],
                    radii: &radii,
                    surface: &surface,
                    shapes: &shapes,
                    body_poses: &buffers[4],
                    particle_count: count,
                    body_count: bodies,
                },
                1.0,
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        let updated: Vec<SoftRigidContactGpu> =
            readback::read(engine, gather.rows(), samples.len());
        let positions = readback::read::<[f32; 4]>(engine, &buffers[0], count as usize);
        let mut rows: Vec<[f32; 8]> = restored_rows
            .chunks_exact(32)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let mut max_geometry_error = 0.0_f64;
        for ((&index, sample), updated) in selected.iter().zip(&samples).zip(updated) {
            assert_eq!(updated.ids[3], 1);
            assert!(
                updated.normal_gap[0] == 0.0
                    && updated.normal_gap[2] == 0.0
                    && (updated.normal_gap[1] - 1.0).abs() < 1e-6
            );
            let expected: f64 = (0..sample.nodes[3] as usize)
                .map(|i| {
                    sample.weights[i] as f64
                        * (positions[sample.nodes[i] as usize][1] as f64
                            - (shape.position[1] as f64 + shape.data[1] as f64))
                })
                .sum();
            max_geometry_error =
                max_geometry_error.max((updated.normal_gap[3] as f64 - expected).abs());
            assert!((updated.normal_gap[3] as f64 - expected).abs() < 1e-10);
            contacts[index].normal_gap[3] = updated.normal_gap[3];
            let gap = updated.normal_gap[3];
            rows[index][2] = if gap < 0.0 {
                (-settings.penetration_recovery * gap / settings.dt)
                    .min(settings.max_recovery_speed)
            } else {
                -gap / settings.dt
            };
        }
        restored_rows = bytemuck::cast_slice(&rows).to_vec();
        queue.write_buffer(&buffers[3], 0, bytemuck::cast_slice(&contacts));
        std::fs::write(
            output.join("ground-control-contacts.bin"),
            bytemuck::cast_slice(&contacts),
        )
        .unwrap();
        println!(
            "GROUND CONTROL fixed_set_rows={} f64_gap_error={max_geometry_error}; test-only readback, no native integration",
            selected.len()
        );
    }
    let final_rows = upload(&restored_rows);
    let mut encoder = device.create_command_encoder(&Default::default());
    kernel
        .encode(device, &mut encoder, bindings, settings)
        .unwrap();
    storage_copy(device, &mut encoder, &final_rows, kernel.row_state());
    queue.submit([encoder.finish()]);
    assert_eq!(readback::read::<u32>(engine, kernel.status(), 1), [0]);
    queue.write_buffer(
        &buffers[8],
        0,
        &std::fs::read(directory.join("contact-final-rigid-velocities.bin")).unwrap(),
    );
    queue.write_buffer(
        &material[2][1],
        0,
        &std::fs::read(directory.join("2-1.bin")).unwrap(),
    );
    let dispatch = |encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline| {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        for (i, group) in groups.iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
    };
    let mut previous = 0;
    let max_cycles = std::env::var("MOLLA_COUPLED_MAX_CYCLES")
        .map(|value| value.parse::<u32>().unwrap())
        .unwrap_or(if std::env::var_os("MOLLA_COUPLED_EXTENDED").is_some() { 8192 } else { 128 });
    assert!(max_cycles.is_power_of_two() && max_cycles <= 8192);
    let mut checkpoints = vec![0];
    let mut next = 1;
    while next <= max_cycles { checkpoints.push(next); next *= 2; }
    for iterations in checkpoints {
        let mut encoder = device.create_command_encoder(&Default::default());
        for _ in previous..iterations {
            dispatch(&mut encoder, capture);
            kernel
                .encode_continuation(device, &mut encoder, bindings, settings)
                .unwrap();
            dispatch(&mut encoder, accumulate);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                for (i, group) in groups.iter().enumerate() {
                    pass.set_bind_group(i as u32, group, &[]);
                }
                pass.set_pipeline(solve);
                for _ in 0..material_sweeps {
                    for (color, group) in uniforms.iter().enumerate() {
                        pass.set_bind_group(3, group, &[]);
                        pass.dispatch_workgroups(
                            (offsets[color + 1] - offsets[color]).div_ceil(64),
                            1,
                            1,
                        );
                    }
                }
            }
            dispatch(&mut encoder, velocity);
        }
        dispatch(&mut encoder, residual);
        kernel
            .encode_residual_check(
                device,
                &mut encoder,
                bindings,
                settings,
                if iterations == max_cycles { 1e-4 } else { f32::MAX },
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        let force = readback::read::<[f32; 4]>(engine, &material[2][3], count as usize);
        let contact = readback::read::<[f32; 8]>(engine, kernel.row_state(), rows as usize);
        let status = readback::read::<u32>(engine, kernel.status(), 1)[0];
        assert!(status == 0 || status == 64);
        assert!(
            force
                .iter()
                .flatten()
                .chain(contact.iter().flatten())
                .all(|v| v.is_finite())
        );
        let force_max = force.iter().map(|v| v[3]).fold(0., f32::max);
        let contact_max = contact.iter().map(|v| v[3]).fold(0., f32::max);
        println!(
            "COUPLED iterations={iterations} material={force_max} contact={contact_max} status={status}"
        );
        if iterations == 0 {
            if ground_control.is_none() {
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&contact),
                    restored_rows,
                    "contact baseline changed"
                );
            } else {
                for (actual, expected) in contact.iter().zip(restored_rows.chunks_exact(32)) {
                    let expected: [f32; 8] = bytemuck::pod_read_unaligned(expected);
                    for i in [0, 1, 2, 4, 5, 6, 7] {
                        assert_eq!(actual[i], expected[i]);
                    }
                }
            }
            let original = std::fs::read(directory.join("2-3.bin")).unwrap();
            let expected_max = original
                .chunks_exact(16)
                .map(|v| f32::from_le_bytes(v[12..16].try_into().unwrap()))
                .fold(0., f32::max);
            println!(
                "COUPLED baseline original_material={expected_max} difference={}",
                force_max - expected_max
            );
        }
        for (label, buffer) in [
            ("force", &material[2][3]),
            ("delta", &material[2][2]),
            ("velocity", &material[2][1]),
            ("rigid", &buffers[8]),
            ("rows", kernel.row_state()),
        ] {
            let bytes = readback::read::<u8>(engine, buffer, buffer.size() as usize);
            std::fs::write(
                output.join(format!("coupled-{label}-{iterations}.bin")),
                bytes,
            )
            .unwrap();
        }
        previous = iterations;
    }
}

fn global_probe(
    engine: &Engine,
    directory: &Path,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    layouts: &[wgpu::BindGroupLayout],
    buffers: &[Vec<wgpu::Buffer>],
    groups: &[wgpu::BindGroup],
) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let pipeline = |entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(layout),
            module: shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let solve = pipeline(if std::env::var_os("MOLLA_RUNTIME_POLISH").is_some() {
        "polish_color"
    } else {
        "solve_color"
    });
    let residual = pipeline("force_residual");
    let offsets = std::fs::read(directory.join("1-6.bin")).unwrap();
    let offsets: Vec<u32> = offsets
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let base = std::fs::read(directory.join("3-0.bin")).unwrap();
    let count = u32::from_le_bytes(base[0..4].try_into().unwrap());
    if std::env::var_os("MOLLA_RUNTIME_CACHE_PREVIOUS").is_some() {
        assert!(!directory.join("contact-final-rows.bin").exists());
        let tets = buffers[5][2].size() as u32 / 4;
        let mut data = base.clone();
        data[16..20].copy_from_slice(&tets.to_le_bytes());
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &data,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layouts[3],
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let cache = pipeline("cache_previous_tets");
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for (i, group) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.set_bind_group(3, &group, &[]);
            pass.set_pipeline(&cache);
            pass.dispatch_workgroups(tets.div_ceil(64), 1, 1);
        }
        let start = std::time::Instant::now();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        println!(
            "CACHE preparation_wall_seconds={} bytes={}",
            start.elapsed().as_secs_f64(),
            tets as u64 * 96
        );
    }
    if std::env::var_os("MOLLA_PROBE_CACHE_PREVIOUS").is_some() {
        assert!(!directory.join("contact-final-rows.bin").exists());
        let cache = pipeline("probe_cache_previous");
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for (i, group) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.set_pipeline(&cache);
            pass.dispatch_workgroups((buffers[5][2].size() as u32 / 4).div_ceil(64), 1, 1);
        }
        let start = std::time::Instant::now();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        println!(
            "CACHE preparation_wall_seconds={} bytes={}",
            start.elapsed().as_secs_f64(),
            buffers[2][6].size()
        );
    }
    let uniforms: Vec<_> = offsets
        .windows(2)
        .enumerate()
        .map(|(color, range)| {
            let mut data = base.clone();
            data[12..16].copy_from_slice(&(color as u32).to_le_bytes());
            data[16..20].copy_from_slice(&(range[1] - range[0]).to_le_bytes());
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &data,
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layouts[3],
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        })
        .collect();
    let initial = std::fs::read(directory.join("2-3.bin")).unwrap();
    if directory.join("contact-final-rows.bin").exists() {
        coupled_probe(
            engine,
            directory,
            buffers,
            groups,
            &uniforms,
            &offsets,
            &solve,
            &residual,
            &pipeline("coupled_capture"),
            &pipeline("coupled_accumulate"),
            &pipeline("coupled_velocity"),
            count,
        );
        return;
    }
    let mut previous = 0;
    for iterations in [0, 1, 8, 32, 128, 512] {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for (i, group) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.set_pipeline(&solve);
            for _ in previous..iterations {
                for (color, group) in uniforms.iter().enumerate() {
                    pass.set_bind_group(3, group, &[]);
                    pass.dispatch_workgroups(
                        (offsets[color + 1] - offsets[color]).div_ceil(64),
                        1,
                        1,
                    );
                }
            }
            pass.set_bind_group(3, &groups[3], &[]);
            pass.set_pipeline(&residual);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        let wall = std::time::Instant::now();
        queue.submit([encoder.finish()]);
        let samples = readback::read::<[f32; 4]>(engine, &buffers[2][3], count as usize);
        let delta = readback::read::<[f32; 4]>(engine, &buffers[2][2], count as usize);
        println!(
            "MATERIAL_TIMING iterations={iterations} sweeps={} submission_readback_wall_seconds={}",
            iterations - previous,
            wall.elapsed().as_secs_f64()
        );
        assert!(
            samples
                .iter()
                .flatten()
                .chain(delta.iter().flatten())
                .all(|v| v.is_finite())
        );
        if iterations == 0 {
            let mut max_difference = 0.0_f32;
            let mut changed = 0;
            for (i, (sample, bytes)) in samples.iter().zip(initial.chunks_exact(16)).enumerate() {
                let expected: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                assert_eq!(&sample[..3], &expected[..3], "initial target node={i}");
                max_difference = max_difference.max((sample[3] - expected[3]).abs());
                changed += usize::from(sample[3] != expected[3]);
            }
            println!("GLOBAL baseline changed={changed} max_scalar_difference={max_difference}");
        }
        let (worst, value) = samples
            .iter()
            .enumerate()
            .max_by(|a, b| a.1[3].total_cmp(&b.1[3]))
            .unwrap();
        println!(
            "GLOBAL iterations={iterations} worst={worst} residual={} node5332={}",
            value[3], samples[5332][3]
        );
        std::fs::write(
            directory.join(format!("delta-{iterations}.bin")),
            bytemuck::cast_slice(&delta),
        )
        .unwrap();
        std::fs::write(
            directory.join(format!("force-{iterations}.bin")),
            bytemuck::cast_slice(&samples),
        )
        .unwrap();
        previous = iterations;
    }
}
