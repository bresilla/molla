use crate::soil_newton_gpu::{
    ExperimentalSoilNewtonGpu as Solver, NewtonVariant, SoilNewtonCapacity, SoilNewtonInputs,
};
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

const PROBE: &str = r#"
@compute @workgroup_size(256)
fn retained_probe(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;let mode=upstream[0];
    if lane==0u {
        status.active_count=0u;atomicStore(&report.fault,0u);
        for(var c=0u;c<params.contacts;c++) {if contacts[c].soil_mass.w>0.0 {active_indices[status.active_count]=c;status.active_count++;}}
    }
    for(var i=lane;i<4160u;i+=256u) {condensed[i]=vec2<f32>(0.0);}
    let stride=max(params.contacts,1u);
    for(var i=lane;i<(params.iterations+1u)*stride;i+=256u) {
        basis[i].v=Six(vec4<f32>(123.0),vec4<f32>(456.0));
        basis[i].z=Six(vec4<f32>(789.0),vec4<f32>(987.0));
    }
    for(var c=lane;c<params.contacts;c+=256u) {
        states[c]=State();let offset=4160u+54u*c;
        for(var j=0u;j<54u;j++) {
            let high=f32(i32((c*7u+j)%31u)-15)/1024.0;
            let low=f32(i32((c+j)%5u)-2)*0.0000000000582076609134674072265625;
            condensed[offset+j]=select(vec2<f32>(bitcast<f32>(0x7fc00000u),0.0),vec2<f32>(high,low),contacts[c].soil_mass.w>0.0);
        }
        condensed[offset+53u]=vec2<f32>(1.0,0.0);
        if (mode==1u && c==2u) || (mode==2u && (c==2u || c==3u)) || (mode==3u && c==1u) || (mode==4u && c==0u) {condensed[offset+53u]=vec2<f32>(0.0);}
        for(var i=0u;i<6u;i++) {
            blocks[c].residual[i]=f32(i32(i)-2)/64.0;
            for(var j=0u;j<6u;j++) {blocks[c].direct[6u*i+j]=vec2<f32>(select(0.0,2.0,i==j),0.0);}
            for(var a=0u;a<3u;a++) {blocks[c].machine[3u*i+a]=vec2<f32>(f32(i32((i*3u+a)%7u)-3)/128.0,0.0);}
        }
    }
    storageBarrier();let accepted=conditioned_prepare(1u);
    if lane==0u {atomicStore(&report.fault,select(8u,0u,accepted));}
    storageBarrier();
    for(var i=lane;i<(params.iterations+1u)*stride;i+=256u) {
        let entry=basis[i];
        if any(entry.z.k!=vec4<f32>(789.0)) || any(entry.z.w!=vec4<f32>(987.0)) {atomicOr(&report.fault,16u);}
        if params.width>params.iterations || i>=params.iterations*stride {
            if any(entry.v.k!=vec4<f32>(123.0)) || any(entry.v.w!=vec4<f32>(456.0)) {atomicOr(&report.fault,32u);}
        }
    }
    if !accepted {return;}
    let next=Six(vec4<f32>(0.125,-0.25,0.375,0.0),vec4<f32>(0.0));
    for(var slot=lane;slot<status.active_count;slot+=256u) {
        let c=active_indices[slot];states[c]=conditioned_state(c,1u,next);
    }
    storageBarrier();
}
"#;

fn coefficient(c: usize, j: usize) -> f64 {
    (((c * 7 + j) % 31) as i32 - 15) as f64 / 1024.0
        + (((c + j) % 5) as i32 - 2) as f64 * 2f64.powi(-34)
}

#[test]
#[ignore = "requires protected GPU; retained-core assembly, solve, recovery and rejection boundaries"]
fn retained_conditioned_core_gpu_lifecycle() {
    lifecycle(0, 64);
}

#[test]
#[ignore = "requires protected GPU; cached projections and scratch isolation"]
fn newton_conditioned_cache_gpu_lifecycle() {
    lifecycle(1, 64);
}

#[test]
#[ignore = "requires protected GPU; cache capacity fallback and scratch isolation"]
fn newton_conditioned_cache_gpu_fallback() {
    lifecycle(1, 1);
}

#[test]
#[ignore = "requires protected GPU; shared conditioned matrix and scratch isolation"]
fn newton_shared_core_gpu_lifecycle() {
    lifecycle(2, 64);
}

#[test]
#[ignore = "requires protected GPU; shared matrix capacity fallback"]
fn newton_shared_core_gpu_fallback() {
    lifecycle(2, 1);
}

fn lifecycle(cached: u32, iterations: u32) {
    let engine =
        molla_compute::Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let storage = |data: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retained core oracle"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    let count = 301usize;
    let mut contacts = vec![0.0; 16 * count];
    for c in 0..count {
        contacts[16 * c + 3] = if c % 7 == 0 { 0.0 } else { 1.0 };
        contacts[16 * c + 5] = 1.0;
        contacts[16 * c + 12] = f32::from_bits(21);
    }
    let contacts = storage(&contacts);
    let upstream = storage(&[0.0]);
    for width in [1usize, 3, 10, 32] {
        let mut rows = vec![0.0; 4 * count * width];
        let mut response = rows.clone();
        for c in 0..count {
            for i in 0..width {
                for a in 0..3 {
                    rows[4 * (c * width + i) + a] =
                        (((c * 3 + i * 5 + a) % 17) as i32 - 8) as f32 / 128.0;
                    response[4 * (c * width + i) + a] =
                        (((c * 7 + i * 3 + a) % 13) as i32 - 6) as f32 / 256.0;
                }
            }
        }
        let mut mass = vec![0.0; width * width];
        for i in 0..width {
            mass[i * width + i] = 1.0;
        }
        let mass = storage(&mass);
        let rates = storage(&vec![0.0; 2 * width]);
        let row_buffer = storage(&rows);
        let response_buffer = storage(&response);
        let solver = Solver::new_with_storage(
            &engine,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &rates,
                point_rows: &row_buffer,
                point_response: &response_buffer,
                contacts: &contacts,
                upstream_status: &upstream,
            },
            SoilNewtonCapacity {
                dofs: width as u32,
                contacts: count as u32,
                newton_iterations: 64,
                krylov_iterations: iterations,
                max_owned_bytes: u64::MAX,
            },
            NewtonVariant::Fischer,
            |s| {
                let s = super::source(super::baseline(s));
                let s = if cached == 2 {
                    crate::soil_newton_gpu::distributed::mixed::shared_core::source(s)
                } else if cached == 1 {
                    crate::soil_newton_gpu::distributed::mixed::conditioned_cache::source(s)
                } else {
                    s
                };
                let probe = if cached == 2 {
                    PROBE.replace("params.width>params.iterations", "!conditioned_shared_capacity()")
                        .replace("storageBarrier();let accepted=conditioned_prepare(1u);", "storageBarrier();if mode!=2u {let warmed=conditioned_prepare(2u);}\n    let accepted=conditioned_prepare(1u);")
                        .replace("condensed[offset+53u]=vec2<f32>(1.0,0.0);", "condensed[offset+53u]=vec2<f32>(1.0,0.0);\n        if mode==5u && c==1u {condensed[offset]=vec2<f32>(bitcast<f32>(0x7fc00000u),0.0);}")
                        .replace("if !accepted {return;}", "if mode==5u && conditioned_shared_ready {atomicOr(&report.fault,64u);}\n    if !accepted {return;}")
                } else { PROBE.to_owned() };
                format!(
                    "{}{}{}256\n{}\n{probe}",
                    crate::soil_newton_protected::PREFIX,
                    crate::soil_newton_protected::VOLATILE_MARKER,
                    crate::soil_newton_width::MARKER,
                    s
                )
            },
            "retained_probe",
            true,
        )
        .unwrap();
        for mode in 0..if cached == 2 { 6u32 } else { 5u32 } {
            queue.write_buffer(&upstream, 0, bytemuck::bytes_of(&mode));
            let mut previous = None;
            for repeat in 0..2 {
                let mut encoder = device.create_command_encoder(&Default::default());
                solver.encode_reset(&mut encoder);
                solver.encode(&mut encoder);
                queue.submit([encoder.finish()]);
                let report = capture::read(&engine, solver.status_buffer());
                assert_eq!(
                    u32::from_le_bytes(report[..4].try_into().unwrap()),
                    if mode == 2 { 8 } else { 0 },
                    "width={width} mode={mode}"
                );
                let bytes = capture::read(&engine, solver.precision_local_buffer());
                let state = capture::read(&engine, solver.state_buffer());
                if let Some((old, old_state)) = &previous {
                    assert_eq!(&bytes, old);
                    assert_eq!(&state, old_state);
                }
                previous = Some((bytes.clone(), state.clone()));
                if mode == 2 {
                    assert!(bytes[..3200 * 8].iter().all(|b| *b == 0));
                    continue;
                }
                check(&bytes, &state, width, count, mode == 1, &rows, &response);
                eprintln!(
                    "retained core width={width} mode={mode} repeat={repeat}: assembly/solve/recovery pass"
                );
            }
        }
    }
}

fn check(
    bytes: &[u8],
    state: &[u8],
    width: usize,
    count: usize,
    retained: bool,
    rows: &[f32],
    response: &[f32],
) {
    let m = width + if retained { 6 } else { 0 };
    let stride = m + 4;
    let pair = |index: usize| {
        f32::from_le_bytes(bytes[8 * index..8 * index + 4].try_into().unwrap()) as f64
            + f32::from_le_bytes(bytes[8 * index + 4..8 * index + 8].try_into().unwrap()) as f64
    };
    let mut matrix = vec![vec![0.0; stride]; m];
    for i in 0..m {
        for j in 0..stride {
            let expected = if i < width {
                if j < width || j == m {
                    let mut sum = if i == j { 1.0 } else { 0.0 };
                    for c in 0..count {
                        if c % 7 == 0 || c == 1 || (retained && c == 2) {
                            continue;
                        }
                        for a in 0..3 {
                            if j == m {
                                sum -= response[4 * (c * width + i) + a] as f64
                                    * coefficient(c, 36 + a);
                            } else {
                                for b in 0..3 {
                                    sum -= rows[4 * (c * width + j) + a] as f64
                                        * response[4 * (c * width + i) + b] as f64
                                        * coefficient(c, 3 * b + a);
                                }
                            }
                        }
                    }
                    sum
                } else if j < m {
                    if j - width < 3 {
                        response[4 * (2 * width + i) + j - width] as f64
                    } else {
                        0.0
                    }
                } else {
                    response[4 * (width + i) + j - m - 1] as f64
                }
            } else {
                let local = i - width;
                if j < width {
                    (0..3)
                        .map(|a| {
                            (((local * 3 + a) % 7) as i32 - 3) as f64 / 128.0
                                * rows[4 * (2 * width + j) + a] as f64
                        })
                        .sum()
                } else if j < m {
                    if local == j - width { 2.0 } else { 0.0 }
                } else if j == m {
                    -(local as f64 - 2.0) / 64.0
                } else {
                    0.0
                }
            };
            matrix[i][j] = expected;
            assert!(
                (pair(1600 + i * stride + j) - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
                "assembly width={width} retained={retained} ({i},{j})"
            );
        }
    }
    for k in 0..4 {
        let solution: Vec<_> = (0..m).map(|i| pair(i * stride + m + k)).collect();
        let mut error = 0.0;
        let mut scale = 0.0;
        for row in &matrix {
            let rhs = row[m + k];
            let r = (0..m).map(|j| row[j] * solution[j]).sum::<f64>() - rhs;
            error += r * r;
            scale += rhs * rhs;
        }
        assert!(
            error.sqrt() / scale.sqrt().max(1e-30) <= 1e-5,
            "solve width={width} column={k}"
        );
    }
    let next = [0.125, -0.25, 0.375];
    let delta: Vec<_> = (0..m)
        .map(|i| {
            pair(i * stride + m)
                - (0..3)
                    .map(|a| pair(i * stride + m + 1 + a) * next[a])
                    .sum::<f64>()
        })
        .collect();
    for c in 0..count {
        for i in 0..6 {
            let expected = if c % 7 == 0 {
                0.0
            } else if c == 1 {
                if i < 3 { next[i] } else { 0.0 }
            } else if retained && c == 2 {
                delta[width + i]
            } else {
                coefficient(c, 36 + i)
                    - (0..3)
                        .map(|a| {
                            coefficient(c, 3 * i + a)
                                * (0..width)
                                    .map(|j| rows[4 * (c * width + j) + a] as f64 * delta[j])
                                    .sum::<f64>()
                        })
                        .sum::<f64>()
            };
            let index = 16 * c + if i < 3 { i } else { i + 1 };
            let word = |index: usize| {
                f32::from_le_bytes(state[4 * index..4 * index + 4].try_into().unwrap()) as f64
            };
            let actual = word(index) + word(index + 8);
            assert!(
                (actual - expected).abs() <= 2e-10 * (1.0 + expected.abs()),
                "state width={width} retained={retained} c={c} i={i} actual={actual:e} expected={expected:e}"
            );
        }
    }
}
