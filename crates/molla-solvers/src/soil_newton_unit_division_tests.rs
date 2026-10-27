use super::*;
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires protected GPU; quotient edge cases and exact finite output"]
fn newton_unit_division_gpu_edges() {
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_UNIT_DIVISION_CAPTURE").unwrap());
    std::fs::create_dir(&root).unwrap();
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut inputs = Vec::new();
    let high = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.25,
        -0.25,
        1e20,
        -1e20,
        1e-20,
        -1e-20,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x80000001),
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let denoms = [
        [1.0, 0.0],
        [-1.0, 0.0],
        [1.0, -0.0],
        [-1.0, -0.0],
        [1.0, 1e-8],
        [-1.0, 1e-8],
        [0.0, 1.0],
        [0.0, 0.0],
        [f32::MIN_POSITIVE, 0.0],
        [f32::INFINITY, 0.0],
    ];
    for h in high {
        for l in [
            0.0,
            -0.0,
            1e-8,
            -1e-8,
            h * 0.25,
            -h,
            f32::from_bits(1),
            f32::NAN,
        ] {
            for b in denoms {
                inputs.push([h, l, b[0], b[1]]);
            }
        }
    }
    let mut seed = 0x795ae132u32;
    for _ in 0..128 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let h = f32::from_bits((seed & 0x807fffff) | ((1 + (seed % 253)) << 23));
        for sign in [1.0, -1.0] {
            inputs.push([h, h * 2f32.powi(-25), sign, 0.0]);
        }
    }
    let bytes = |data: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("unit quotient fixture"),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let n = inputs.len();
    let rows = bytes(bytemuck::cast_slice(&inputs));
    let response = bytes(bytemuck::cast_slice(&vec![[0.0f32; 4]; n]));
    let contacts = bytes(bytemuck::cast_slice(&vec![[0.0f32; 16]; n]));
    let mass = bytes(bytemuck::bytes_of(&1.0f32));
    let free = bytes(bytemuck::cast_slice(&[0.0f32; 2]));
    let upstream = bytes(bytemuck::bytes_of(&0u32));
    std::fs::write(root.join("inputs.bin"), bytemuck::cast_slice(&inputs)).unwrap();
    let probe = r#"
@compute @workgroup_size(256)
fn quotient_probe(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    for(var c=id;c<params.contacts;c+=256u) {
        let v=rows[c];let q=fischer_div(v.xy,v.zw);let product=fischer_product(q,vec2<f32>(1.0,0.0));
        states[c]=State(Six(vec4<f32>(q,product),vec4<f32>(0.0)),Six());
    }
}
"#;
    let mut outputs = Vec::new();
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
                contacts: n as u32,
                newton_iterations: 64,
                krylov_iterations: 64,
                max_owned_bytes: u64::MAX,
            },
            NewtonVariant::Fischer,
            |s| {
                let s = unit_division_program(s, candidate);
                assert!(crate::soil_newton_protected::uses_compiler(&s).unwrap());
                format!("{s}\n{probe}")
            },
            "quotient_probe",
            true,
        )
        .unwrap();
        let mut previous = None;
        for repeat in 0..2 {
            let allocation = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let transfer = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            solver.encode_reset(&mut encoder);
            solver.encode(&mut encoder);
            queue.submit([encoder.finish()]);
            assert_eq!(allocation.finish().observations, 0);
            assert_eq!(transfer.finish(), Default::default());
            let out = capture::read(&engine, solver.state_buffer());
            std::fs::write(
                root.join(format!("candidate-{candidate}-repeat-{repeat}.bin")),
                &out,
            )
            .unwrap();
            if let Some(p) = previous.as_ref() {
                let p: &Vec<u8> = p;
                assert_eq!(out.len(), p.len());
                let mismatch = out.iter().zip(p).position(|(a, b)| a != b);
                assert!(
                    mismatch.is_none(),
                    "candidate={candidate} repeat={repeat} first differing byte={mismatch:?}"
                );
            }
            previous = Some(out);
        }
        outputs.push(previous.unwrap());
    }
    let mut finite = 0;
    let mut nonfinite = 0;
    let mut max_error = 0.0f64;
    for (i, (a, b)) in outputs[0]
        .chunks_exact(64)
        .zip(outputs[1].chunks_exact(64))
        .enumerate()
    {
        for k in 0..4 {
            let a = u32::from_le_bytes(a[4 * k..4 * k + 4].try_into().unwrap());
            let b = u32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap());
            if f32::from_bits(a).is_finite() {
                assert_eq!(a, b, "case={i} component={k} input={:?}", inputs[i]);
                finite += 1;
            } else {
                assert!(!f32::from_bits(b).is_finite());
                nonfinite += 1;
            }
        }
        let q = |k: usize| f64::from(f32::from_le_bytes(a[4 * k..4 * k + 4].try_into().unwrap()));
        let input = inputs[i];
        let reference = (f64::from(input[0]) + f64::from(input[1]))
            / (f64::from(input[2]) + f64::from(input[3]));
        if (q(0) + q(1)).is_finite()
            && reference.is_finite()
            && input[2].abs() == 1.0
            && input[3] == 0.0
            && input[0].abs() >= f32::MIN_POSITIVE
            && (input[1] == 0.0 || input[1].abs() >= f32::MIN_POSITIVE)
        {
            let e = ((q(0) + q(1)) - reference).abs() / (1.0 + reference.abs());
            max_error = max_error.max(e);
            assert!(e <= 1e-6);
        }
    }
    std::fs::write(root.join("result.json"),serde_json::to_vec_pretty(&serde_json::json!({"acceptance":false,"cases":n,"finite_words_exact":finite,"nonfinite_words_preserved":nonfinite,"max_reference_error":max_error,"exact_self_repeat":true})).unwrap()).unwrap();
}
