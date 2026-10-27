use super::*;
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

const PROBE: &str = r#"
fn direct_expected(c:u32,i:u32)->f32 {return f32(i32((c*3u+i)%11u)-5)/64.0;}
@compute @workgroup_size(256)
fn retained_direct_probe(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;let mode=upstream[0];
    if lane==0u {
        atomicStore(&report.fault,0u);status.active_count=0u;
        for(var c=0u;c<params.contacts;c++) {
            if contacts[c].soil_mass.w>0.0 && mode!=6u {active_indices[status.active_count]=c;status.active_count++;}
        }
    }
    for(var c=lane;c<params.contacts;c+=256u) {
        states[c]=State();
        let bad=(mode==1u && c==2u) || (mode==2u && (c==2u || c==3u)) || (mode==3u && c==0u) || ((mode==4u || mode==5u) && c==2u);
        for(var i=0u;i<6u;i++) {
            blocks[c].residual[i]=-2.0*direct_expected(c,i);
            for(var j=0u;j<6u;j++) {blocks[c].direct[6u*i+j]=vec2<f32>(select(0.0,2.0,i==j),0.0);}
            for(var a=0u;a<3u;a++) {blocks[c].machine[3u*i+a]=vec2<f32>(0.0);}
        }
        if bad {
            var diagonal=0.0;if mode==4u {diagonal=1e-30;}if mode==5u {diagonal=bitcast<f32>(0x7fc00000u);}
            blocks[c].direct[0]=vec2<f32>(diagonal,0.0);
            blocks[c].machine[0]=vec2<f32>(1.0,0.0);
            blocks[c].residual[0]=-free_rates[0].x;
        }
    }
    storageBarrier();prepare_condensed();let accepted=direct_direction();
    if lane==0u {
        atomicStore(&report.fault,select(8u,atomicLoad(&status.fault),accepted));report.steps=conditioned_core_width;
        report.linear_iterations=status.iterations;report.residual=status.relative;
    }
    if accepted {
        for(var slot=lane;slot<status.active_count;slot+=256u) {
            let c=active_indices[slot];states[c]=State(direction[c],basis[params.iterations*max(params.contacts,1u)+c].z);
        }
    }
    storageBarrier();
}
"#;

fn expected(c: usize, i: usize) -> f64 {
    (((c * 3 + i) % 11) as i32 - 5) as f64 / 64.0
}

#[test]
#[ignore = "requires protected GPU; independent retained-direct oracle"]
fn newton_retained_direct_gpu_lifecycle() {
    lifecycle(false, false);
}

#[test]
#[ignore = "requires protected GPU; lazy retained-direct independent oracle"]
fn newton_retained_direct_lazy_gpu_lifecycle() {
    lifecycle(true, false);
}

#[test]
#[ignore = "requires protected GPU; targeted quality correction independent oracle"]
fn newton_retained_direct_quality_gpu_lifecycle() {
    lifecycle(true, true);
}

fn lifecycle(lazy: bool, quality: bool) {
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let storage = |values: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retained direct oracle"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    const C: usize = 301;
    let mut nodes = vec![0.0; 16 * C];
    for c in 0..C {
        nodes[16 * c + 3] = if c % 7 == 0 { 0.0 } else { 1.0 };
        nodes[16 * c + 5] = 1.0;
        nodes[16 * c + 12] = f32::from_bits(21);
    }
    let contacts = storage(&nodes);
    let upstream = storage(&[0.0]);
    let dq = -(0..C)
        .filter(|c| c % 7 != 0)
        .map(|c| expected(c, 0) / 256.0)
        .sum::<f64>();
    for width in [1usize, 3, 10, 32] {
        let mut rows = vec![0.0; 4 * C * width];
        let mut response = rows.clone();
        for c in 0..C {
            rows[4 * c * width] = 1.0;
            response[4 * c * width] = 1.0 / 256.0;
        }
        let rows = storage(&rows);
        let response = storage(&response);
        let mut m = vec![0.0; width * width];
        for i in 0..width {
            m[i * width + i] = if i == 0 { 256.0 } else { 1.0 };
        }
        let mass = storage(&m);
        let mut free = vec![0.0; 2 * width];
        free[0] = dq as f32;
        let free = storage(&free);
        for candidate in [false, true] {
            let solver = ExperimentalSoilNewtonGpu::new_with_storage(
                &engine,
                SoilNewtonInputs {
                    mass: &mass,
                    free_rates: &free,
                    point_rows: &rows,
                    point_response: &response,
                    contacts: &contacts,
                    upstream_status: &upstream,
                },
                SoilNewtonCapacity {
                    dofs: width as u32,
                    contacts: C as u32,
                    newton_iterations: 64,
                    krylov_iterations: 64,
                    max_owned_bytes: u64::MAX,
                },
                NewtonVariant::Fischer,
                |input| {
                    let mut source = if quality { quality_program(input, candidate) } else { program_with_policy(input, candidate, lazy) };
                    if quality && candidate {
                        replace(&mut source, "let accepted=initial_direct_direction();", r#"let accepted=initial_direct_direction();
    if accepted && upstream[0]>=7u {
        if lane==0u {
            scratch[2].k.x=1.0;
            basis[params.iterations*max(params.contacts,1u)+2u].z.k.x=1e-12;
            if upstream[0]==8u {condensed[4160u+54u*2u+53u]=vec2<f32>(0.0);condensed[4160u+54u*3u+53u]=vec2<f32>(0.0);}
            if upstream[0]>=9u {condensed[4160u+54u*2u+53u]=vec2<f32>(0.0);}
        }
        storageBarrier();
    }"#);
                        replace(&mut source, "if retained_direct_solve() {", "if retained_direct_solve() {\n        if upstream[0]==9u {if lane==0u {direction[2].k.x=bitcast<f32>(0x7fc00000u);}storageBarrier();}");
                        replace(&mut source, "let after=direct_worst_relative();", "var after=direct_worst_relative();if upstream[0]==10u {after=before;}");
                    }
                    format!("{source}\n{PROBE}")
                },
                "retained_direct_probe",
                true,
            )
            .unwrap();
            for mode in 0..=if quality { 10u32 } else { 6u32 } {
                queue.write_buffer(&upstream, 0, bytemuck::bytes_of(&mode));
                let valid = matches!(mode, 0 | 3 | 6..=10) || (candidate && matches!(mode, 1 | 4));
                let mut previous = None;
                for repeat in 0..2 {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    solver.encode_reset(&mut encoder);
                    solver.encode(&mut encoder);
                    queue.submit([encoder.finish()]);
                    let report = capture::read(&engine, solver.status_buffer());
                    let words: [u32; 8] = bytemuck::pod_read_unaligned(&report);
                    assert_eq!(
                        words[0],
                        if valid { 0 } else { 8 },
                        "width={width} candidate={candidate} mode={mode}"
                    );
                    let state = capture::read(&engine, solver.state_buffer());
                    if let Some((r, s)) = &previous {
                        assert_eq!(r, &report);
                        assert_eq!(s, &state);
                    }
                    previous = Some((report, state.clone()));
                    if quality && candidate && mode >= 7 {
                        let offset = 64 * 2 + 32;
                        assert_eq!(
                            f32::from_le_bytes(state[offset..offset + 4].try_into().unwrap()),
                            1e-12f32,
                            "restore low pair {width}/{mode}"
                        );
                    }
                    if !valid || mode == 6 {
                        assert!(state.iter().all(|&b| b == 0));
                        continue;
                    }
                    let value = |c: usize, i: usize| {
                        let offset = 64 * c + 4 * (if i < 3 { i } else { i + 1 });
                        let f = |offset| {
                            f32::from_le_bytes(state[offset..offset + 4].try_into().unwrap()) as f64
                        };
                        f(offset) + f(offset + 32)
                    };
                    let actual_dq = -(0..C)
                        .filter(|c| c % 7 != 0)
                        .map(|c| value(c, 0) / 256.0)
                        .sum::<f64>();
                    let mut error = 0.0;
                    let mut norm = 0.0;
                    for c in 0..C {
                        for i in 0..6 {
                            if c % 7 == 0 {
                                assert_eq!(value(c, i), 0.0);
                                continue;
                            }
                            assert!(
                                (value(c, i) - expected(c, i)).abs() <= 1e-6,
                                "recovery {width}/{mode}/{c}/{i}"
                            );
                            let retained = c == 2 && matches!(mode, 1 | 4);
                            let rhs = if retained && i == 0 {
                                dq
                            } else {
                                2.0 * expected(c, i)
                            };
                            let actual = if retained && i == 0 {
                                actual_dq + if mode == 4 { 1e-30 * value(c, i) } else { 0.0 }
                            } else {
                                2.0 * value(c, i)
                            };
                            error += (actual - rhs).powi(2);
                            norm += rhs.powi(2);
                        }
                    }
                    assert!(error.sqrt() / norm.sqrt().max(1e-30) <= 1e-5);
                    assert!(f32::from_bits(words[4]) <= 1e-5);
                    assert_eq!(
                        words[1],
                        if candidate && matches!(mode, 1 | 4 | 9 | 10) {
                            width as u32 + 6
                        } else {
                            0
                        }
                    );
                    eprintln!(
                        "retained-direct width={width} mode={mode} candidate={candidate} repeat={repeat} recovery and independent residual pass"
                    );
                }
            }
        }
    }
}
