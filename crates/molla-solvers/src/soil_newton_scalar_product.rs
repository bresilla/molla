use super::*;

const ORIGINAL: &str =
    "fischer_product(soa_direct(c,6u*i+j),vec2<f32>(component(vector[c],j),0.0))";
const CANDIDATE: &str = "operator_scalar_product(soa_direct(c,6u*i+j),component(vector[c],j))";
const HELPERS: &str = include_str!("../shaders/soil_newton_scalar_product.wgsl");

pub(crate) fn source(mut source: String) -> String {
    assert!(source.starts_with(crate::soil_newton_protected::PREFIX));
    assert!(!source.contains("params.pad2"));
    replace(&mut source, ORIGINAL, CANDIDATE);
    source.push_str(HELPERS);
    source
}

fn admitted([high, low, scalar]: [f32; 3]) -> bool {
    [high, low, scalar]
        .into_iter()
        .zip([0x2c000000, 0x17800000, 0x2c000000])
        .all(|(v, min)| {
            let bits = v.to_bits() & 0x7fffffff;
            bits == 0 || (min..=0x4e800000).contains(&bits)
        })
}

fn cases() -> Vec<[f32; 3]> {
    let words = [
        0, 1, 0x007fffff, 0x00800000, 0x177fffff, 0x17800000, 0x17800001, 0x2bffffff, 0x2c000000,
        0x2c000001, 0x3f000000, 0x3f7fffff, 0x3f800000, 0x3f800001, 0x4e7fffff, 0x4e800000,
        0x4e800001, 0x7f7fffff, 0x7f800000, 0x7fc12345,
    ];
    let values: Vec<_> = words
        .into_iter()
        .flat_map(|v| [v, v | 0x80000000])
        .map(f32::from_bits)
        .collect();
    let mut cases = Vec::new();
    for &a in &values {
        for &b in &values {
            for &c in &values {
                cases.push([a, b, c]);
            }
        }
    }
    let mut seed = 0x28c93a17u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for _ in 0..32768 {
        let a = next();
        let b = next();
        let c = next();
        cases.push([a, b, c].map(f32::from_bits));
        cases.push(
            [a, b, c].map(|v| f32::from_bits((v & 0x807fffff) | ((88 + (v >> 24) % 69) << 23))),
        );
    }
    cases
}

fn sum(a: f32, b: f32) -> [f32; 2] {
    let (large, small) = if b.abs() > a.abs() { (b, a) } else { (a, b) };
    let s = large + small;
    let z = s - large;
    [s, small - z]
}

fn original([a, low, b]: [f32; 3]) -> [f32; 2] {
    let high = a * b;
    let cross = a.mul_add(0.0, low * b);
    let tail = low.mul_add(0.0, a.mul_add(b, -high));
    sum(high, tail + cross)
}

fn fast([a, low, b]: [f32; 3]) -> [f32; 2] {
    let high = b * a;
    sum(high, b.mul_add(a, -high) + b * low)
}

#[test]
fn newton_scalar_product_guarded_arithmetic() {
    assert_eq!(f32::from_bits(0x2c000000), 2f32.powi(-39));
    assert_eq!(f32::from_bits(0x17800000), 2f32.powi(-80));
    assert_eq!(f32::from_bits(0x4e800000), 2f32.powi(30));
    let cases = cases();
    let mut accepted = 0;
    for (i, &case) in cases.iter().enumerate() {
        if admitted(case) {
            accepted += 1;
            assert_eq!(
                original(case).map(f32::to_bits),
                fast(case).map(f32::to_bits),
                "case {i}: {case:?}"
            );
            let product = case[0] * case[2];
            let error = case[0].mul_add(case[2], -product);
            let cross = case[1] * case[2];
            for v in [product, error, cross] {
                assert!(v.is_finite() && (v == 0.0 || v.is_normal()));
            }
        }
    }
    assert!(accepted > 32768 && accepted < cases.len());
    eprintln!(
        "scalar product CPU cases={} admitted={accepted}",
        cases.len()
    );
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_SCALAR_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("fixtures.bin"), bytemuck::cast_slice(&cases)).unwrap();
    }
}

const PROBE: &str = r#"
@compute @workgroup_size(256)
fn scalar_product_probe(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    for(var c=lane;c<params.contacts;c+=256u) {
        let input=contacts[c].soil_mass;
        condensed[2u*c]=fischer_product(input.xy,vec2<f32>(input.z,0.0));
        condensed[2u*c+1u]=guarded_scalar_product(input.xy,input.z);
        condensed[2u*params.contacts+c]=vec2<f32>(select(0.0,1.0,scalar_product_admitted(input.xy,input.z)),0.0);
    }
}
"#;

#[test]
fn newton_scalar_product_source_contract() {
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    assert!(control.contains("let swap=abs(b)>abs(a);let large=select(a,b,swap);let small=select(b,a,swap);\n    let s=rate_round(large+small);let z=rate_round(s-large);\n    return vec2<f32>(s,rate_round(small-z));"));
    let candidate = source(control.clone());
    let mut restored = candidate.strip_suffix(HELPERS).unwrap().to_owned();
    replace(&mut restored, CANDIDATE, ORIGINAL);
    assert_eq!(restored, control);
    for token in [
        "var<storage",
        "var<workgroup>",
        "@binding(",
        "workgroupBarrier();",
        "storageBarrier();",
    ] {
        assert_eq!(
            candidate.matches(token).count(),
            control.matches(token).count()
        );
    }
    for (name, text, entry) in [
        ("operator", candidate.clone(), "newton"),
        (
            "probe",
            format!("{candidate}{PROBE}"),
            "scalar_product_probe",
        ),
    ] {
        let words = crate::soil_newton_protected::compile(&text, entry).unwrap();
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_SCALAR_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), &text).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
}

pub(crate) fn probe(engine: &Engine) {
    use crate::soil_shared_wall_friction_gpu::capture;
    use wgpu::util::DeviceExt;
    const C: usize = 1024;
    let input = std::fs::read(std::env::var_os("MOLLA_NEWTON_SCALAR_FIXTURE").unwrap()).unwrap();
    assert_eq!(input.len() % 12, 0);
    let cases: Vec<[f32; 3]> = input
        .chunks_exact(12)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    assert_eq!(cases.len(), 129536);
    let device = engine.device().unwrap();
    let storage = |data: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scalar product cases"),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    let zero = storage(&[0; 8]);
    let contacts = storage(&vec![0; C * 64]);
    let rows = storage(&vec![0; C * 16]);
    let solver = ExperimentalSoilNewtonGpu::new_with_storage(
        engine,
        SoilNewtonInputs {
            mass: &zero,
            free_rates: &zero,
            point_rows: &rows,
            point_response: &rows,
            contacts: &contacts,
            upstream_status: &zero,
        },
        SoilNewtonCapacity {
            dofs: 1,
            contacts: C as u32,
            newton_iterations: 1,
            krylov_iterations: 1,
            max_owned_bytes: u64::MAX,
        },
        NewtonVariant::Fischer,
        |s| format!("{}{PROBE}", source(soa_scratch::program(s, true))),
        "scalar_product_probe",
        true,
    )
    .unwrap();
    let mut output = Vec::<u8>::new();
    let mut accepted = 0;
    for chunk in cases.chunks(C) {
        let mut data = vec![0f32; C * 16];
        for (i, case) in chunk.iter().enumerate() {
            data[16 * i..16 * i + 3].copy_from_slice(case);
        }
        engine
            .queue()
            .unwrap()
            .write_buffer(&contacts, 0, bytemuck::cast_slice(&data));
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        let bytes = capture::read(engine, solver.precision_local_buffer());
        for (i, &case) in chunk.iter().enumerate() {
            let a: [f32; 2] = bytemuck::pod_read_unaligned(&bytes[16 * i..16 * i + 8]);
            let b: [f32; 2] = bytemuck::pod_read_unaligned(&bytes[16 * i + 8..16 * i + 16]);
            let flag: [f32; 2] =
                bytemuck::pod_read_unaligned(&bytes[16 * C + 8 * i..16 * C + 8 * i + 8]);
            assert_eq!(flag, [u32::from(admitted(case)) as f32, 0.0]);
            accepted += usize::from(admitted(case));
            for k in 0..2 {
                if a[k].is_nan() {
                    assert!(b[k].is_nan());
                } else {
                    assert_eq!(
                        a[k].to_bits(),
                        b[k].to_bits(),
                        "GPU scalar {case:?} word {k}"
                    );
                }
            }
            if admitted(case) {
                assert_eq!(
                    a.map(f32::to_bits),
                    original(case).map(f32::to_bits),
                    "GPU/CPU scalar {case:?}"
                );
            }
            output.extend_from_slice(&bytes[16 * i..16 * i + 16]);
        }
    }
    let root =
        std::path::PathBuf::from(std::env::var_os("MOLLA_NEWTON_SCALAR_COMPONENT_OUT").unwrap());
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("pairs.bin"), output).unwrap();
    std::fs::write(root.join("audit.json"), serde_json::to_vec_pretty(&serde_json::json!({"cases":cases.len(),"admitted":accepted,"fallback":cases.len()-accepted,"finite_outputs_bit_exact":true,"admitted_cpu_bit_exact":true,"nonfinite_nan_payload_contract":"classification only; unmodified fallback","acceptance":false})).unwrap()).unwrap();
    eprintln!(
        "scalar product GPU cases={} admitted={accepted}",
        cases.len()
    );
}
