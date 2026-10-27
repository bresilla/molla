use super::*;

#[test]
#[ignore = "requires GPU; cooperative compensated reduction accuracy"]
fn sorted_pairs_gpu_tree_reduction() {
    let engine = Engine::new_gpu().unwrap();
    let mut cases = vec![vec![[0.0_f32; 2]; 64]; 67];
    cases[0][0] = [2.0_f32.powi(90), 2.0_f32.powi(-40)];
    cases[0][31] = [-2.0_f32.powi(90), 0.0];
    cases[1][0] = [-1e8, 0.25];
    cases[1][32] = [1e8, 0.0];
    for group in 3..cases.len() {
        for (i, value) in cases[group].iter_mut().enumerate() {
            let x = ((i * 19 + group * 7) as f32).sin() * 2.0_f32.powi((group % 41) as i32 - 20);
            *value = [x, -x * 2.0_f32.powi(-25)];
        }
    }
    let input = storage(
        &engine,
        &cases.iter().flatten().copied().collect::<Vec<_>>(),
    );
    let output = storage(&engine, &vec![[0.0_f32; 2]; cases.len()]);
    let source = format!("{}\nvar<workgroup> parts:array<vec2<f32>,64>;var<private> lane:u32;\n{}\n
        @group(0) @binding(0) var<storage,read> input:array<vec2<f32>>;
        @group(0) @binding(1) var<storage,read_write> output:array<vec2<f32>>;
        @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) id:u32,@builtin(workgroup_id) group:vec3<u32>) {{
            lane=id;rate_lane=id;parts[id]=input[64u*group.x+id];let sum=reduce_parts();if id==0u {{output[group.x]=sum;}}
        }}", crate::soil_rate_encoding::sorted_arithmetic(64), include_str!("../shaders/soil_newton_reduce.wgsl"));
    let device = engine.device().unwrap();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cooperative paired reduction"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cooperative paired reduction"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cooperative paired reduction"),
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
        pass.dispatch_workgroups(cases.len() as u32, 1, 1);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    let result = values(&engine, &output);
    assert_eq!(result[0] as f64 + result[1] as f64, 2.0_f64.powi(-40));
    assert_eq!(result[2] as f64 + result[3] as f64, 0.25);
    assert_eq!(result[4] as f64 + result[5] as f64, 0.0);
    let mut worst = 0.0_f64;
    for (group, case) in cases.iter().enumerate().skip(3) {
        let expected = case.iter().flatten().map(|&x| x as f64).sum::<f64>();
        let scale = case
            .iter()
            .flatten()
            .map(|&x| (x as f64).abs())
            .sum::<f64>();
        let actual = result[2 * group] as f64 + result[2 * group + 1] as f64;
        let relative = (actual - expected).abs() / scale;
        worst = worst.max(relative);
        assert!(relative <= 1e-13, "group={group} error={relative:e}");
    }
    eprintln!(
        "tree reduction groups={} worst_scaled_error={worst:e}",
        cases.len()
    );
}

fn sum(a: f32, b: f32) -> [f32; 2] {
    let s = a + b;
    let vb = s - a;
    let va = s - vb;
    [s, (a - va) + (b - vb)]
}

fn add(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    let h = sum(a[0], b[0]);
    let l = sum(a[1], b[1]);
    let m = sum(h[0], h[1] + l[0]);
    sum(m[0], m[1] + l[1])
}

fn mul(a: f32, b: [f32; 2]) -> [f32; 2] {
    let h = a * b[0];
    sum(h, a.mul_add(b[0], -h) + a * b[1])
}

fn wide_mantissa_cases() -> Vec<[f32; 8]> {
    let mut seed = 0x45f3a621_u32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        f32::from_bits((seed & 0x807fffff) | ((87 + (seed >> 24) % 81) << 23))
    };
    (0..4096)
        .flat_map(|_| {
            let a = next();
            let b = next();
            let factor = next();
            [
                [a, 0., b, 0., factor, 0., 0., 0.],
                [
                    a,
                    a * 2.0_f32.powi(-25),
                    b,
                    -b * 2.0_f32.powi(-26),
                    factor,
                    0.,
                    0.,
                    0.,
                ],
            ]
        })
        .collect()
}

fn evaluate(engine: &Engine, cases: &[[f32; 8]], arithmetic: u32) -> Vec<f32> {
    evaluate_rounding(engine, cases, arithmetic, None)
}

fn evaluate_rounding(
    engine: &Engine,
    cases: &[[f32; 8]],
    arithmetic: u32,
    rounding: Option<u8>,
) -> Vec<f32> {
    let mut source = format!(
        "{}\n@group(0) @binding(0) var<storage,read> inputs:array<vec4<f32>>;
        @group(0) @binding(1) var<storage,read_write> outputs:array<vec2<f32>>;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) {{
            rate_lane=id.x%64u;let c=id.x;if c*2u>=arrayLength(&inputs) {{return;}}
            let v=inputs[2u*c];let a=v.xy;let b=v.zw;let factor=inputs[2u*c+1u].x;
            outputs[4u*c]=rate_sum(a.x,b.x);
            outputs[4u*c+1u]=rate_add(a,b);
            outputs[4u*c+2u]=rate_mul(factor,a);
            outputs[4u*c+3u]=rate_add(rate_add(a,b),-b);
        }}",
        match arithmetic {
            1 => crate::soil_rate_encoding::native_arithmetic(64),
            2..=19 => crate::soil_rate_encoding::sorted_arithmetic(64),
            _ => crate::soil_rate_encoding::arithmetic(64),
        }
    );
    if matches!(arithmetic, 3 | 6 | 7) {
        let fischer = include_str!("../shaders/soil_newton_fischer.wgsl");
        source.push_str(
            &fischer[fischer.find("fn fischer_product(").unwrap()
                ..fischer.find("fn fischer_dot(").unwrap()],
        );
        source = source.replace(
            "outputs[4u*c+3u]=rate_add(rate_add(a,b),-b);",
            "outputs[4u*c+3u]=fischer_product(a,b);",
        );
    }
    if matches!(arithmetic, 8..=13) {
        let fischer = include_str!("../shaders/soil_newton_fischer.wgsl");
        source.push_str("\nfn value(x:vec2<f32>)->f32 {return rate_round(x.x+x.y);}\n");
        source.push_str(&fischer[..fischer.find("fn fischer_length(").unwrap()]);
        source = source.replace(
            "outputs[4u*c+3u]=rate_add(rate_add(a,b),-b);",
            "outputs[4u*c+3u]=fischer_div(a,b);",
        );
        if arithmetic >= 11 {
            source = source.replace(
                "outputs[4u*c]=rate_sum(a.x,b.x);",
                "let high=a.x/b.x;outputs[4u*c]=vec2<f32>(high,0.0);",
            );
            source = source.replace(
                "outputs[4u*c+1u]=rate_add(a,b);",
                "outputs[4u*c+1u]=rate_mul(high,b);",
            );
            source = source.replace(
                "outputs[4u*c+2u]=rate_mul(factor,a);",
                "outputs[4u*c+2u]=rate_add(a,-rate_mul(high,b));",
            );
        }
    }
    if matches!(arithmetic, 14..=19) {
        source = source.replace(
            "outputs[4u*c+3u]=rate_add(rate_add(a,b),-b);",
            "outputs[4u*c+3u]=rate_mul(a.x,vec2<f32>(a.x,0.0));",
        );
    }
    if matches!(arithmetic, 17..=19) {
        let fischer = include_str!("../shaders/soil_newton_fischer.wgsl");
        source.push_str(&fischer[..fischer.find("fn fischer_div(").unwrap()]);
        source = source.replace("let a=v.xy;", "let a=vec2<f32>(fischer_root(v.x),0.0);");
        source = source.replace("outputs[4u*c]=rate_sum(a.x,b.x);", "outputs[4u*c]=a;");
    }
    let device = engine.device().unwrap();
    let input = storage(engine, cases);
    let output = storage(engine, &vec![[0.0_f32; 8]; cases.len()]);
    let module = if let Some(mode) = rounding {
        #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
        {
            match mode {
                0 | 1 => crate::soil_newton_precise_tests::rounding_shader(
                    device,
                    &source,
                    "main",
                    mode == 1,
                ),
                2 | 3 => crate::soil_newton_precise_tests::float_controls::shader(
                    device,
                    &source,
                    mode == 3,
                ),
                _ => panic!("unknown rounding fixture mode"),
            }
        }
        #[cfg(not(all(feature = "protected-newton", not(target_arch = "wasm32"))))]
        {
            let _ = mode;
            panic!("rounding probe requires protected native features");
        }
    } else if matches!(arithmetic, 4..=9 | 12..=15 | 18..=19) {
        assert_eq!(engine.adapter_info().unwrap().backend, "Vulkan");
        crate::soil_newton_precise_tests::shader(device, &source, "main", arithmetic % 2 == 1)
    } else {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("paired arithmetic comparison"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("paired arithmetic layout"),
        entries: &(0..2)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding == 0,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("paired arithmetic layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("paired arithmetic comparison"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("paired arithmetic comparison"),
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
    values(engine, &output)
}

#[test]
#[ignore = "requires native GPU FP64; cancellation and exponent-range arithmetic"]
fn native_pairs_gpu_arithmetic() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_F64).unwrap();
    arithmetic_cases(&engine, 1);
}

#[test]
#[ignore = "requires GPU; sorted summation cancellation and exponent range"]
fn sorted_pairs_gpu_arithmetic() {
    let engine = Engine::new_gpu().unwrap();
    arithmetic_cases(&engine, 2);
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools; independent paired arithmetic"]
fn newton_precise_pairs_gpu_arithmetic() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    arithmetic_cases(&engine, 4);
    arithmetic_cases(&engine, 5);
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools; composed paired products"]
fn newton_precise_pairs_gpu_products() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    fischer_products(&engine, 6);
    fischer_products(&engine, 7);
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools; paired division"]
fn newton_precise_pairs_gpu_division() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let mut cases = vec![[0., 0., 1., 0., 1., 0., 0., 0.]];
    for i in 0..4096_i32 {
        let a = (1.0 + (i % 97) as f32 / 128.0) * 2.0_f32.powi(i % 81 - 40);
        let b = -(1.0 + (i % 83) as f32 / 128.0) * 2.0_f32.powi((i * 37) % 81 - 40);
        cases.push([
            a,
            a * 2.0_f32.powi(-25),
            b,
            -b * 2.0_f32.powi(-26),
            1.,
            0.,
            0.,
            0.,
        ]);
    }
    let baseline = evaluate(&engine, &cases, 10);
    for arithmetic in [8, 9] {
        let output = evaluate(&engine, &cases, arithmetic);
        let mut worst = 0.0_f64;
        let mut mismatches = 0;
        for (i, case) in cases.iter().enumerate() {
            let expected = (case[0] as f64 + case[1] as f64) / (case[2] as f64 + case[3] as f64);
            let actual = output[8 * i + 6] as f64 + output[8 * i + 7] as f64;
            let error = (actual - expected).abs() / expected.abs().max(1e-30);
            if output[8 * i..8 * i + 8] != baseline[8 * i..8 * i + 8] {
                if mismatches < 4 {
                    eprintln!(
                        "division mismatch case={i} input={:?} baseline={:?} candidate={:?} relative={error:e}",
                        &case[..4],
                        &baseline[8 * i + 6..8 * i + 8],
                        &output[8 * i + 6..8 * i + 8]
                    );
                }
                mismatches += 1;
            }
            worst = worst.max(error);
        }
        eprintln!(
            "paired division variant={arithmetic} cases={} worst_relative={worst:e}",
            cases.len()
        );
        assert!(worst <= 1e-13, "division worst relative error={worst:e}");
        assert_eq!(mismatches, 0, "division output parity");
    }
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools; composed division trace"]
fn newton_precise_pairs_gpu_division_trace() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let cases = [[
        1.,
        2.0_f32.powi(-25),
        -1.,
        2.0_f32.powi(-26),
        1.,
        0.,
        0.,
        0.,
    ]];
    let baseline = evaluate(&engine, &cases, 11);
    for arithmetic in [12, 13] {
        let output = evaluate(&engine, &cases, arithmetic);
        eprintln!("division trace mode={arithmetic} baseline={baseline:?} actual={output:?}");
        assert_eq!(output, baseline);
    }
}

#[test]
#[ignore = "requires GPU; paired Fischer coefficient products"]
fn fischer_pairs_gpu_products() {
    let engine = Engine::new_gpu().unwrap();
    fischer_products(&engine, 3);
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools; correlated squared products"]
fn newton_precise_pairs_gpu_square_product() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let mut cases = wide_mantissa_cases();
    cases.push([f32::from_bits(0x3fb51c7d), 0., 0., 0., 1., 0., 0., 0.]);
    for arithmetic in [16, 14, 15] {
        let output = evaluate(&engine, &cases, arithmetic);
        for (i, case) in cases.iter().enumerate() {
            let expected = mul(case[0], [case[0], 0.]);
            assert_eq!(
                &output[8 * i + 6..8 * i + 8],
                &expected,
                "square mode={arithmetic} case={i} input={:?}",
                case[0]
            );
        }
        eprintln!(
            "squared products mode={arithmetic} cases={} exact CPU component parity",
            cases.len()
        );
    }
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools; products of computed roots"]
fn newton_precise_pairs_gpu_root_product() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    root_products(&engine, &[17, 18, 19]);
}

#[test]
#[ignore = "requires protected GPU; volatile rounding arithmetic and computed-root regression"]
#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
fn newton_volatile_pairs_gpu() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    rounding_cases(&engine, 1);
}

#[test]
#[ignore = "requires RTX 4080 Vulkan float-controls2; no-memory rounding parity"]
#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
fn newton_float_controls_pairs_gpu() {
    let engine = crate::soil_newton_precise_tests::float_controls::engine(wgpu::Features::empty());
    rounding_cases(&engine, 2);
}

#[test]
#[ignore = "requires RTX 4080 float-controls2; per-instruction no-memory parity"]
#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
fn newton_float_controls_selective_pairs_gpu() {
    let engine = crate::soil_newton_precise_tests::float_controls::engine(wgpu::Features::empty());
    rounding_cases(&engine, 3);
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
fn rounding_cases(engine: &Engine, candidate_mode: u8) {
    for mode in [3, 9, 13, 15, 19] {
        let mut cases = wide_mantissa_cases();
        if mode == 19 {
            for case in &mut cases {
                case[0] = case[0].abs();
            }
        }
        cases.push([
            f32::from_bits(0x4000214d),
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]);
        let reference = evaluate_rounding(engine, &cases, mode, Some(0));
        let candidate = evaluate_rounding(engine, &cases, mode, Some(candidate_mode));
        let repeat = evaluate_rounding(engine, &cases, mode, Some(candidate_mode));
        if candidate_mode >= 2 {
            let root = std::path::PathBuf::from(
                std::env::var_os("MOLLA_FLOAT_CONTROLS_ARTIFACTS").unwrap(),
            );
            for (name, bytes) in [
                ("input", bytemuck::cast_slice(cases.as_slice())),
                ("reference", bytemuck::cast_slice(reference.as_slice())),
                ("candidate", bytemuck::cast_slice(candidate.as_slice())),
                ("repeat", bytemuck::cast_slice(repeat.as_slice())),
            ] {
                std::fs::write(root.join(format!("mode-{mode}-{name}.bin")), bytes).unwrap();
            }
        }
        assert_eq!(
            candidate.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            repeat.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        for (i, (&a, &b)) in reference.iter().zip(&candidate).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "rounding candidate={candidate_mode} parity mode={mode} case={} word={}",
                i / 8,
                i % 8
            );
        }
        let mut worst = 0.0_f64;
        for (i, case) in cases.iter().enumerate() {
            let a = case[0] as f64 + case[1] as f64;
            let b = case[2] as f64 + case[3] as f64;
            let expected = match mode {
                3 => a * b,
                9 | 13 => a / b,
                15 => (case[0] as f64).powi(2),
                19 => (candidate[8 * i] as f64).powi(2),
                _ => unreachable!(),
            };
            let actual = candidate[8 * i + 6] as f64 + candidate[8 * i + 7] as f64;
            let error = if expected == 0.0 {
                (actual - expected).abs()
            } else {
                ((actual - expected) / expected).abs()
            };
            assert!(
                error <= 1e-13,
                "rounding candidate={candidate_mode} oracle mode={mode} case={i} error={error:e}"
            );
            worst = worst.max(error);
            if mode == 19 {
                assert_eq!(
                    &candidate[8 * i + 6..8 * i + 8],
                    &mul(candidate[8 * i], [candidate[8 * i], 0.0]),
                    "computed root case={i}"
                );
            }
        }
        eprintln!(
            "rounding candidate={candidate_mode} mode={mode} {} cases: exact atomic parity and self-repeat, worst relative error={worst:e}",
            cases.len()
        );
    }
}

#[test]
#[ignore = "requires GPU; production materialized-root product accuracy"]
fn fischer_pairs_gpu_root_product() {
    root_products(&Engine::new_gpu().unwrap(), &[17]);
}

fn root_products(engine: &Engine, modes: &[u32]) {
    let mut cases = wide_mantissa_cases();
    for case in &mut cases {
        case[0] = case[0].abs();
    }
    cases.push([f32::from_bits(0x4000214d), 0., 0., 0., 1., 0., 0., 0.]);
    for &arithmetic in modes {
        let output = evaluate(engine, &cases, arithmetic);
        let mut mismatches = 0;
        for (i, _) in cases.iter().enumerate() {
            let root = output[8 * i];
            let expected = mul(root, [root, 0.]);
            if output[8 * i + 6..8 * i + 8] != expected {
                if mismatches < 4 {
                    eprintln!(
                        "root product mode={arithmetic} case={i} root={root:?} expected={expected:?} actual={:?}",
                        &output[8 * i + 6..8 * i + 8]
                    );
                }
                mismatches += 1;
            }
        }
        eprintln!(
            "computed-root products mode={arithmetic} cases={} mismatches={mismatches}",
            cases.len()
        );
        assert_eq!(mismatches, 0);
    }
}

fn fischer_products(engine: &Engine, arithmetic: u32) {
    let mut cases = vec![[0.0; 8], [1.0, 2e-8, -1.0, 1e-8, 0.0, 0.0, 0.0, 0.0]];
    for i in 0..4096_i32 {
        let a = (1.0 + (i % 97) as f32 / 128.0) * 2.0_f32.powi(i % 81 - 40);
        let b = -(1.0 + (i % 83) as f32 / 128.0) * 2.0_f32.powi((i * 37) % 81 - 40);
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        cases.push([
            a,
            sign * a * 2.0_f32.powi(-25),
            b,
            -sign * b * 2.0_f32.powi(-26),
            0.0,
            0.0,
            0.0,
            0.0,
        ]);
    }
    cases.extend(wide_mantissa_cases());
    let output = evaluate(engine, &cases, arithmetic);
    if arithmetic >= 4 {
        let baseline = evaluate(engine, &cases, 3);
        for (i, (&a, &b)) in output.iter().zip(&baseline).enumerate() {
            assert_eq!(
                a,
                b,
                "product case={} word={} input={:?}",
                i / 8,
                i % 8,
                cases[i / 8]
            );
        }
    }
    let mut worst = 0.0_f64;
    for (i, case) in cases.iter().enumerate() {
        let expected = (case[0] as f64 + case[1] as f64) * (case[2] as f64 + case[3] as f64);
        let actual = output[8 * i + 6] as f64 + output[8 * i + 7] as f64;
        let error = if expected == 0.0 {
            (actual - expected).abs()
        } else {
            ((actual - expected) / expected).abs()
        };
        assert!(error <= 1e-13, "case={i} error={error:e}");
        worst = worst.max(error);
    }
    eprintln!(
        "Fischer paired products: {} cases, worst relative error {worst:e}",
        cases.len()
    );
}

fn arithmetic_cases(engine: &Engine, arithmetic: u32) {
    let mut cases = vec![
        [-1e8, 0.25, 1e8, 0.0, 0.5, 0.0, 0.0, 0.0],
        [
            2.0_f32.powi(120),
            2.0_f32.powi(-100),
            -2.0_f32.powi(120),
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            2.0_f32.powi(120),
            0.0,
            2.0_f32.powi(-100),
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            2.0_f32.powi(-100),
            0.0,
            2.0_f32.powi(120),
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
    ];
    for i in 0..4096_i32 {
        let a = (1.0 + (i % 97) as f32 / 128.0) * 2.0_f32.powi(i % 161 - 80);
        let b = -(1.0 + (i % 83) as f32 / 128.0) * 2.0_f32.powi((i * 37) % 161 - 80);
        cases.push([
            a,
            a * 2.0_f32.powi(-25),
            b,
            -b * 2.0_f32.powi(-26),
            1.03125,
            0.0,
            0.0,
            0.0,
        ]);
    }
    cases.extend(wide_mantissa_cases());
    let native = evaluate(engine, &cases, arithmetic);
    let paired = evaluate(engine, &cases, 0);
    for (c, case) in cases.iter().enumerate() {
        let a = [case[0], case[1]];
        let b = [case[2], case[3]];
        let expected = [
            sum(a[0], b[0]),
            add(a, b),
            mul(case[4], a),
            add(add(a, b), [-b[0], -b[1]]),
        ];
        for (i, &value) in expected.iter().flatten().enumerate() {
            assert_eq!(
                native[c * 8 + i],
                value,
                "native case={c} word={i} input={case:?}"
            );
            assert_eq!(paired[c * 8 + i], value, "fallback case={c} word={i}");
        }
    }
    eprintln!(
        "paired arithmetic variant={arithmetic}: {} cases, exact component parity",
        cases.len()
    );
}
