use super::*;
use crate::soil_newton_publish_gpu::{ExperimentalSoilNewtonPublishGpu, SoilNewtonPublishInputs};

fn source(mut source: String, mode: u32) -> String {
    source = source.replace(
        &crate::soil_rate_encoding::arithmetic(64),
        &crate::soil_rate_encoding::sorted_arithmetic(64),
    );
    if mode == 0 {
        return source;
    }
    let anchor = "outputs[c].wall_impulse=vec4<f32>(w,0.0);outputs[c].soil_velocity_energy=vec4<f32>(soil,e);outputs[c].machine_velocity_residual=vec4<f32>(machine,0.0);outputs[c].status=vec4<u32>(0u,0u,0u,wall_bits);";
    assert_eq!(source.matches(anchor).count(), 1);
    let capture = if mode == 1 {
        "outputs[c].machine_impulse_normal=vec4<f32>(lambda,strict.x,strict.y,abs(lambda*separation));
        outputs[c].soil_velocity_energy=vec4<f32>(length(tangent),cap,length(slip),length(tangent+cap*normalize(slip)));
        outputs[c].machine_velocity_residual=vec4<f32>(ps,vs,separation,node.friction.x);
        outputs[c].wall_impulse=vec4<f32>(soil,length(relative));outputs[c].status=bitcast<vec4<u32>>(vec4<f32>(normal,node.soil_mass.w));"
    } else {
        "outputs[c].machine_impulse_normal=vec4<f32>(k,lambda);
        outputs[c].soil_velocity_energy=vec4<f32>(soil,node.soil_mass.w);
        outputs[c].machine_velocity_residual=vec4<f32>(machine,node.normal_gap.w);
        outputs[c].wall_impulse=vec4<f32>(slip,cap);outputs[c].status=bitcast<vec4<u32>>(vec4<f32>(tangent,length(w)));"
    };
    source = source.replace(anchor, capture);
    source = source.replace("certify_machine(lane);", "");
    source = source.replace("if rejected {outputs[c]=Output();}", "");
    source.replace("select(q[lane],vec2<f32>(0.0),rejected)", "q[lane]")
}

pub(super) fn trace(
    engine: &Engine,
    input: SoilNewtonPublishInputs<'_>,
    f: &Fixture,
    p: &Reference,
) {
    let n = f.free.len();
    let c = f.contacts.len();
    let output =
        std::path::PathBuf::from(std::env::var_os("MOLLA_NEWTON_PUBLICATION_TRACE").unwrap());
    assert!(!output.exists());
    std::fs::create_dir_all(&output).unwrap();
    let mut reference = None;
    let mut baseline = None;
    for mode in 0..3 {
        let publisher = ExperimentalSoilNewtonPublishGpu::new_with_program(
            engine,
            SoilNewtonPublishInputs {
                mass: input.mass,
                free_rates: input.free_rates,
                point_rows: input.point_rows,
                point_response: input.point_response,
                contacts: input.contacts,
                state: input.state,
                upstream_status: input.upstream_status,
                solver_status: input.solver_status,
            },
            n as u32,
            c as u32,
            u64::MAX,
            |s| source(s, mode),
        )
        .unwrap();
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        publisher.encode_reset(&mut encoder);
        publisher.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        let report: [u32; 8] =
            bytemuck::pod_read_unaligned(&capture::read(engine, publisher.status_buffer())[..32]);
        eprintln!("publication trace mode={mode} report={report:?}");
        if mode == 0 {
            reference = Some(report);
        } else {
            assert_eq!(&report[..4], &reference.unwrap()[..4]);
        }
        let raw = capture::read(engine, publisher.outputs_buffer());
        std::fs::write(output.join(format!("mode-{mode}.bin")), &raw).unwrap();
        std::fs::write(
            output.join(format!("rates-{mode}.bin")),
            capture::read(engine, publisher.rates_buffer()),
        )
        .unwrap();
        if report[1] > 0 && mode > 0 {
            let contact = report[1] as usize - 1;
            let words: Vec<_> = raw[80 * contact..80 * (contact + 1)]
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            eprintln!("publication trace contact={contact} mode={mode} values={words:?}");
        }
        if mode == 0 {
            baseline = Some(publisher);
        }
    }
    if reference.unwrap()[0] == 0 {
        let publisher = baseline.unwrap();
        super::super::publication::verify_outputs(
            engine,
            publisher.outputs_buffer(),
            publisher.rates_buffer(),
            f,
            p,
        );
    }
}

#[test]
fn newton_publication_trace_shader() {
    for mode in 0..3 {
        let module = wgpu::naga::front::wgsl::parse_str(&source(
            crate::soil_newton_publish_gpu::shader_source(),
            mode,
        ))
        .unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
#[ignore = "requires frozen publication trace and Newton packet; CPU represented-value oracle"]
fn newton_publication_trace_oracle() {
    let root =
        std::path::PathBuf::from(std::env::var_os("MOLLA_NEWTON_PUBLICATION_TRACE").unwrap());
    let packet = std::fs::read(
        std::path::PathBuf::from(std::env::var_os("MOLLA_NEWTON_FAILURE_REPLAY").unwrap())
            .join("newton.bin"),
    )
    .unwrap();
    let u32_at = |offset| u32::from_le_bytes(packet[offset..offset + 4].try_into().unwrap());
    let n = u32_at(4) as usize;
    let c = u32_at(8) as usize;
    let floats = |raw: &[u8]| {
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
            .collect::<Vec<_>>()
    };
    let vectors = floats(&std::fs::read(root.join("mode-2.bin")).unwrap());
    let scalars = floats(&std::fs::read(root.join("mode-1.bin")).unwrap());
    let rates = floats(&std::fs::read(root.join("rates-2.bin")).unwrap());
    let rates: Vec<_> = rates.chunks_exact(2).map(|q| q[0] + q[1]).collect();
    let start = 128 + 4 * n * n + 8 * n;
    let stride = 32 * n + 128;
    let mut worst = 0f64;
    for contact in 0..c {
        let offset = start + stride * contact;
        let node = floats(&packet[offset + 32 * n..offset + 32 * n + 64]);
        if node[3] == 0. {
            continue;
        }
        let normal = xyz(&node[4..]).normalize();
        let v = &vectors[20 * contact..20 * (contact + 1)];
        let rows = floats(&packet[offset..offset + 16 * n]);
        let machine: DVec3 = rows
            .chunks_exact(4)
            .zip(&rates)
            .map(|(r, q)| xyz(r) * q)
            .sum();
        let k = xyz(v);
        let soil = xyz(&v[4..]);
        let relative = soil - machine;
        let slip = relative - normal * normal.dot(relative);
        let lambda = normal.dot(k);
        let tangent = k - normal * lambda;
        let ps = 1. + k.length() + v[19];
        let vs = 1. + soil.length() + machine.length();
        let alignment = if slip.length() > 1e-6 * vs {
            (tangent + node[8] * lambda * slip.normalize()).length() / ps
        } else {
            0.
        };
        let gpu = &scalars[20 * contact..20 * (contact + 1)];
        let gpu_alignment = if gpu[6] > 1e-6 * gpu[9] {
            gpu[7] / gpu[8]
        } else {
            0.
        };
        if alignment > worst || gpu_alignment > 1e-6 {
            eprintln!(
                "published oracle contact={contact} f64_alignment={alignment:e} gpu_alignment={gpu_alignment:e} true_slip={slip:?} gpu_slip={:?} true_tangent={tangent:?} gpu_tangent={:?} normal_length_squared={:e}",
                &v[12..15],
                &v[16..19],
                xyz(&gpu[16..]).length_squared()
            );
        }
        worst = worst.max(alignment);
    }
    eprintln!("published oracle worst friction alignment={worst:e}");
    assert!(worst <= 1e-6);
}

#[test]
#[ignore = "requires GPU; tangent projection against f64 represented-input oracle"]
fn newton_publication_tangent_gpu_precision() {
    let engine = Engine::new_gpu().unwrap();
    let mut cases = Vec::<[[f32; 4]; 3]>::new();
    let mut expected = Vec::new();
    for normal in [
        DVec3::X,
        DVec3::Y,
        DVec3::Z,
        DVec3::new(0.0240499302744865, -0.99971067905426, -9.18741443456383e-6),
        DVec3::new(1., 2., 3.).normalize(),
        DVec3::new(-3., 2., 0.01).normalize(),
    ] {
        for length in [0.99999, 1., 1.00001] {
            let raw = pack(normal * length, 0.);
            let n = vec(raw);
            let axis = if n.x.abs() < 0.8 { DVec3::X } else { DVec3::Y };
            let tangent = n.cross(axis).normalize();
            for magnitude in [1e-3, 1., 1024.] {
                for ratio in [0., 1e-8, 1e-7, 1e-6, 1e-4, 1e-2, 1., -1e-6, -1.] {
                    let v = (n + tangent * ratio) * magnitude;
                    let hi = pack(v, 0.);
                    let lo = pack(v - vec(hi), 0.);
                    let v = vec(hi) + vec(lo);
                    let reference = n.cross(v.cross(n)) / n.length_squared();
                    expected.push((reference, 3e-7 * reference.length() + 2e-13 * v.length()));
                    cases.push([raw, hi, lo]);
                }
            }
        }
    }
    let input = storage(&engine, &cases);
    let output = storage(&engine, &vec![[0f32; 8]; cases.len()]);
    let full = include_str!("../shaders/soil_newton_publish.wgsl");
    let helper =
        &full[full.find("fn publication_tangent(").unwrap()..full.find("fn fault(").unwrap()];
    let source = format!("{}\nfn value(v:vec2<f32>)->f32 {{return rate_round(v.x+v.y);}}\n{helper}\n
        struct Case {{ n:vec4<f32>, hi:vec4<f32>, lo:vec4<f32> }}
        struct Output {{ fixed:vec4<f32>, original:vec4<f32> }}
        @group(0) @binding(0) var<storage,read> input:array<Case>;
        @group(0) @binding(1) var<storage,read_write> output:array<Output>;
        @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane:u32,@builtin(global_invocation_id) id:vec3<u32>) {{
            rate_lane=lane;if id.x>=arrayLength(&input) {{return;}}let a=input[id.x];
            let pairs=array<vec2<f32>,3>(vec2<f32>(a.hi.x,a.lo.x),vec2<f32>(a.hi.y,a.lo.y),vec2<f32>(a.hi.z,a.lo.z));
            let n=normalize(a.n.xyz);let v=a.hi.xyz+a.lo.xyz;
            output[id.x]=Output(vec4<f32>(publication_tangent(pairs,a.n.xyz),0.0),vec4<f32>(v-dot(n,v)*n,0.0));
        }}",crate::soil_rate_encoding::sorted_arithmetic(64));
    let device = engine.device().unwrap();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("publication tangent regression"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
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
    let result = values(&engine, &output);
    let mut failures = 0;
    let mut old_failures = 0;
    let mut worst = 0f64;
    for (i, ((reference, bound), out)) in expected.iter().zip(result.chunks_exact(8)).enumerate() {
        let fixed = DVec3::new(out[0] as f64, out[1] as f64, out[2] as f64);
        let old = DVec3::new(out[4] as f64, out[5] as f64, out[6] as f64);
        let error = (fixed - *reference).length();
        if !fixed.is_finite() || error > *bound {
            failures += 1;
            eprintln!("tangent case={i} error={error:e} bound={bound:e}");
        }
        if (old - *reference).length() > *bound {
            old_failures += 1;
        }
        worst = worst.max(error / bound);
    }
    eprintln!(
        "tangent cases={} fixed_failures={failures} old_failures={old_failures} worst_bound_fraction={worst:e}",
        cases.len()
    );
    assert_eq!(failures, 0);
    assert!(old_failures > 0);
}
