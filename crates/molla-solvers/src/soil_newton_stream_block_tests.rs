use super::*;
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, PartialEq)]
enum Probe {
    Stream,
    Arguments,
    Private,
}

fn probe_source(input: String, candidate: bool) -> String {
    probe_source_with(input, candidate, Probe::Stream)
}

fn probe_source_with(input: String, candidate: bool, probe: Probe) -> String {
    let mut input = if probe != Probe::Stream {
        crate::soil_newton_gpu::distributed::mixed::complementarity::baseline(input)
    } else {
        shared_core::source(native_division_program(input, false))
    };
    if probe == Probe::Private && candidate {
        input = crate::soil_newton_gpu::distributed::mixed::private_samples::source(input, false);
    } else if probe == Probe::Arguments && candidate {
        input = crate::soil_newton_gpu::distributed::mixed::complementarity::source(input, false);
    } else if probe == Probe::Stream && candidate {
        input = source(input);
    }
    let block = if candidate || probe != Probe::Stream {
        "blocks[c]"
    } else {
        "block"
    };
    let validation = format!("    for (var i=0u;i<6u;i++) {{ if !finite({block}.residual[i])");
    replace(
        &mut input,
        &validation,
        &format!(
            "    if stream_mode==2u {{{block}.residual[0]=bitcast<f32>(upstream[1]);}}\n    if stream_mode==3u {{{block}.direct[0]=vec2<f32>(bitcast<f32>(upstream[1]),0.0);}}\n    if stream_mode==4u {{{block}.machine[0]=vec2<f32>(bitcast<f32>(upstream[1]),0.0);}}\n{validation}"
        ),
    );
    input.push_str(r#"
var<private> stream_mode:u32;
@compute @workgroup_size(256)
fn stream_probe(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane!=0u {return;}
    rates[0]=vec2<f32>(0.0);fischer_smoothing=0.01;
    for(var mode=0u;mode<8u;mode++) {
        stream_mode=mode;direct_linearize=mode!=1u;
        atomicStore(&status.fault,select(0u,1u,mode==7u));
        states[mode]=State(Six(vec4<f32>(0.1,0.5,-0.1,0.0),vec4<f32>(0.0)),Six());
        for(var i=0u;i<6u;i++) {blocks[mode].residual[i]=17.0;}
        for(var i=0u;i<36u;i++) {blocks[mode].direct[i]=vec2<f32>(19.0);}
        for(var i=0u;i<18u;i++) {blocks[mode].machine[i]=vec2<f32>(23.0);}
        nonlinear_contact(mode);
        let base=mode*58u;
        condensed[base]=vec2<f32>(f32(atomicLoad(&status.fault)),f32(mode));
        for(var i=0u;i<3u;i++) {condensed[base+1u+i]=vec2<f32>(blocks[mode].residual[2u*i],blocks[mode].residual[2u*i+1u]);}
        for(var i=0u;i<36u;i++) {condensed[base+4u+i]=blocks[mode].direct[i];}
        for(var i=0u;i<18u;i++) {condensed[base+40u+i]=blocks[mode].machine[i];}
    }
}
"#);
    input
}

#[test]
fn newton_stream_block_probe_contract() {
    for candidate in [false, true] {
        crate::soil_newton_protected::compile(
            &probe_source(shader_source_for(NewtonVariant::Fischer), candidate),
            "stream_probe",
        )
        .unwrap();
    }
}

#[test]
#[ignore = "requires protected GPU; streaming block parity and poisoned fault clearing"]
fn newton_stream_block_gpu_edges() {
    gpu_edges(Probe::Stream);
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_complementarity_probe_contract() {
    for candidate in [false, true] {
        crate::soil_newton_protected::compile(
            &probe_source_with(
                shader_source_for(NewtonVariant::Fischer),
                candidate,
                Probe::Arguments,
            ),
            "stream_probe",
        )
        .unwrap();
    }
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
#[ignore = "requires protected GPU; complementarity sharing and fault parity"]
fn newton_complementarity_gpu_edges() {
    gpu_edges(Probe::Arguments);
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
fn newton_private_samples_probe_contract() {
    for candidate in [false, true] {
        crate::soil_newton_protected::compile(
            &probe_source_with(
                shader_source_for(NewtonVariant::Fischer),
                candidate,
                Probe::Private,
            ),
            "stream_probe",
        )
        .unwrap();
    }
}

#[cfg(feature = "stagnant-branch-newton")]
#[test]
#[ignore = "requires protected GPU; private complementarity samples and fault parity"]
fn newton_private_samples_gpu_edges() {
    gpu_edges(Probe::Private);
}

fn gpu_edges(probe: Probe) {
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_STREAM_BLOCK_CAPTURE").unwrap());
    std::fs::create_dir(&root).unwrap();
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let storage = |words: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stream block fixture"),
            contents: bytemuck::cast_slice(words),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let mass = storage(&[1.0]);
    let free = storage(&[0.0; 2]);
    let rows = storage(&[0.0; 32]);
    let response = storage(&[0.0; 32]);
    let upstream = storage(&[0.0, f32::INFINITY]);
    let mut contacts = [0.0; 128];
    for c in 0..8 {
        contacts[16 * c + 1] = -0.1;
        contacts[16 * c + 2] = 0.3;
        contacts[16 * c + 3] = if c == 5 { 0.0 } else { 2.0 };
        contacts[16 * c + 5] = if c == 6 { 0.0 } else { 1.0 };
        contacts[16 * c + 8] = 0.3;
        contacts[16 * c + 12] = f32::from_bits(0b01_10_01);
    }
    let contacts = storage(&contacts);
    let mut expected = None;
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
                dofs: 1,
                contacts: 8,
                newton_iterations: 64,
                krylov_iterations: 64,
                max_owned_bytes: u64::MAX,
            },
            NewtonVariant::Fischer,
            |s| probe_source_with(s, candidate, probe),
            "stream_probe",
            true,
        )
        .unwrap();
        for repeat in 0..2 {
            let mut encoder = device.create_command_encoder(&Default::default());
            solver.encode_reset(&mut encoder);
            solver.encode(&mut encoder);
            queue.submit([encoder.finish()]);
            let all = capture::read(&engine, solver.precision_local_buffer());
            let out = all[..8 * 58 * 8].to_vec();
            std::fs::write(
                root.join(format!("candidate-{candidate}-repeat-{repeat}.bin")),
                &out,
            )
            .unwrap();
            if let Some(previous) = &expected {
                assert_eq!(&out, previous);
            }
            for (mode, bytes) in out.chunks_exact(58 * 8).enumerate() {
                let v: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let fault = match mode {
                    2..=4 => 2.0,
                    6..=7 => 1.0,
                    _ => 0.0,
                };
                assert_eq!(v[0], fault, "candidate={candidate} mode={mode}");
                assert_eq!(v[1], mode as f32);
                assert!(v.iter().all(|x| x.is_finite()));
                if mode >= 2 {
                    assert!(v[2..].iter().all(|x| *x == 0.0));
                } else {
                    assert!(v[2..8].iter().any(|x| *x != 0.0));
                    if mode == 1 {
                        assert!(v[8..].iter().all(|x| *x == 0.0));
                    } else {
                        assert!(v[8..].iter().any(|x| *x != 0.0));
                        for i in 3..6 {
                            for j in 0..6 {
                                if j != i && j != i - 3 {
                                    assert_eq!(
                                        &v[8 + 2 * (6 * i + j)..10 + 2 * (6 * i + j)],
                                        &[0.0, 0.0]
                                    );
                                }
                            }
                        }
                        assert!(v[80 + 18..].iter().all(|x| *x == 0.0));
                    }
                }
            }
            expected = Some(out);
        }
    }
    eprintln!(
        "stream-block: 32 cases pass exact control/candidate/repeat equality; full/lazy, three poisoned validation paths, inactive/invalid/sticky fault clearing"
    );
}
