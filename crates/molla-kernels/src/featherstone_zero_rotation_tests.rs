use molla_compute::Engine;

#[test]
#[ignore = "requires GPU and two cold six-body machine programs"]
fn zero_rotation_matches_compensated_tractors() {
    super::rounding_tests::zero_rotation_articulation_comparison(false, false);
}

#[test]
#[ignore = "requires GPU and two cold corrected-drift programs"]
fn zero_rotation_matches_corrected_drift() {
    super::rounding_tests::zero_rotation_articulation_comparison(true, false);
}

#[test]
#[ignore = "requires GPU and two cold constrained machine programs"]
fn zero_rotation_preserves_mutating_constraints() {
    super::rounding_tests::zero_rotation_articulation_comparison(false, true);
}

pub(super) fn reference_source(source: &str) -> String {
    let guard = "    if zero_position_rotation(q,value) { return PositionPair(vec3<f32>(0.0),vec3<f32>(0.0)); }\n";
    assert_eq!(source.matches(guard).count(), 1);
    source.replace(guard, "")
}

pub(super) fn xyz_candidate_source(source: &str) -> String {
    let call = "let result=multiply(child,multiply(child,q,vector),inverse);";
    assert_eq!(source.matches(call).count(), 1);
    assert_eq!(source.matches("fn zero_position_rotation").count(), 1);
    source
        .replace(
            call,
            "let result=position_pair_product(child,multiply(child,q,vector),inverse);",
        )
        .replace(
            "fn zero_position_rotation",
            &format!(
                "{}fn zero_position_rotation",
                include_str!("../shaders/featherstone_position_product.wgsl")
            ),
        )
}

#[test]
#[ignore = "requires GPU"]
fn xyz_position_rotation_preserves_bits() {
    paired_rotation(true);
}

#[test]
#[ignore = "requires GPU and two corrected-drift programs"]
fn xyz_position_rotation_matches_corrected_drift() {
    super::rounding_tests::xyz_rotation_articulation_comparison(true);
}

#[test]
#[ignore = "requires GPU and two machine programs"]
fn xyz_position_rotation_matches_dynamics() {
    super::rounding_tests::xyz_rotation_articulation_comparison(false);
}

#[test]
#[ignore = "requires GPU"]
fn exact_zero_rotation_preserves_bits_and_nonzero_fallback() {
    paired_rotation(false);
}

fn paired_rotation(xyz: bool) {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let source = include_str!("../shaders/featherstone.wgsl");
    let rounding = source
        .split_once("override local_position_rounding")
        .unwrap()
        .1
        .split_once("fn position_add")
        .unwrap()
        .0;
    let rotation = source
        .split_once("fn zero_position_rotation")
        .unwrap()
        .1
        .split_once("fn compensated_body_position")
        .unwrap()
        .0;
    let rotation = format!("fn zero_position_rotation{rotation}");
    let reference = if xyz {
        rotation.clone()
    } else {
        reference_source(&rotation)
    }
    .replace(
        "fn zero_position_rotation",
        "fn reference_zero_position_rotation",
    )
    .replace(
        "fn position_pair_rotate",
        "fn reference_position_pair_rotate",
    );
    let rotation = if xyz {
        xyz_candidate_source(&rotation)
    } else {
        rotation
    };
    let shader_source = format!(
        "override local_position_rounding{rounding}\n\
        struct PositionPair {{ high:vec3<f32>, low:vec3<f32> }}\n\
        fn rounded(slot:u32,value:f32)->f32 {{return position_round(slot,value);}}\n\
        fn two_sum(slot:u32,a:f32,b:f32)->vec2<f32> {{return position_sum(slot,a,b);}}\n\
        {}\n{rotation}\n{reference}\n{}",
        include_str!("../shaders/quaternion_precision.wgsl"),
        r#"
struct Case { q:QuaternionPair, high:vec4<f32>, low:vec4<f32> }
@group(0) @binding(0) var<storage,read> cases:array<Case>;
@group(0) @binding(1) var<storage,read_write> position_error:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> outputs:array<u32>;
@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) id:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    position_rounding_lane=lane;
    if id.x>=arrayLength(&cases) {return;}
    let c=cases[id.x]; let value=PositionPair(c.high.xyz,c.low.xyz);
    let old=reference_position_pair_rotate(id.x,c.q,value);
    let candidate=position_pair_rotate(id.x,c.q,value);
    for(var axis=0u;axis<3u;axis++) {
        outputs[id.x*13u+axis]=bitcast<u32>(old.high[axis]);
        outputs[id.x*13u+3u+axis]=bitcast<u32>(old.low[axis]);
        outputs[id.x*13u+6u+axis]=bitcast<u32>(candidate.high[axis]);
        outputs[id.x*13u+9u+axis]=bitcast<u32>(candidate.low[axis]);
    }
    outputs[id.x*13u+12u]=u32(zero_position_rotation(c.q,value));
}
"#,
    );
    let mut cases: Vec<[u32; 16]> = Vec::new();
    let mut expected = Vec::new();
    for mask in 0..64u32 {
        for q in [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, -0.2, 0.3, 0.9, 1e-9, -2e-9, 3e-9, -4e-9],
            [-0.70710677, 0.0, 0.0, 0.70710677, -1e-8, 0.0, 0.0, 1e-8],
        ] {
            let mut words = [0u32; 16];
            for (word, value) in words.iter_mut().zip(q) {
                *word = f32::to_bits(value);
            }
            for (i, index) in [8, 9, 10, 12, 13, 14].into_iter().enumerate() {
                words[index] = ((mask >> i) & 1) << 31;
            }
            cases.push(words);
            expected.push(true);
            for bits in [1, 0x80000001, 0x00800000, 0x3f800000] {
                for index in [8, 9, 10, 12, 13, 14] {
                    let mut nonzero = words;
                    nonzero[index] = bits;
                    cases.push(nonzero);
                    expected.push(false);
                }
            }
        }
    }
    for index in 0..8 {
        for invalid in [0x7f800000, 0xff800000, 0x7fc01234] {
            let mut words = [0u32; 16];
            words[3] = 1f32.to_bits();
            words[index] = invalid;
            cases.push(words);
            expected.push(false);
        }
    }
    if xyz {
        let mut seed = 0x714ad513u32;
        for index in 0..8193 {
            let mut words = [0u32; 16];
            for (component, word) in words.iter_mut().enumerate() {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let scale = match component {
                    0..=3 => 1.0,
                    4..=7 | 12..=15 => 1e-7,
                    _ => [0.001, 1.0, 65536.0][index % 3],
                };
                *word = ((seed as i32 as f32 / i32::MAX as f32) * scale).to_bits();
            }
            cases.push(words);
            expected.push(false);
        }
    }
    let allocate = |bytes, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage,
            mapped_at_creation: false,
        })
    };
    let input = allocate(
        cases.len() as u64 * 64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let scratch = allocate(cases.len() as u64 * 16, wgpu::BufferUsages::STORAGE);
    let output = allocate(
        cases.len() as u64 * 13 * 4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    queue.write_buffer(&input, 0, bytemuck::cast_slice(&cases));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("exact zero paired position rotation"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    for local in [false, true] {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("local_position_rounding", f64::from(local))],
                ..Default::default()
            },
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[&input, &scratch, &output]
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
        }
        queue.submit([encoder.finish()]);
        let words = super::rounding_tests::words(device, queue, &output);
        for (index, row) in words.chunks_exact(13).enumerate() {
            assert_eq!(
                row[12] != 0,
                expected[index],
                "guard case {index}, local={local}"
            );
            for axis in 0..6 {
                let a = f32::from_bits(row[axis]);
                let b = f32::from_bits(row[axis + 6]);
                if a.is_nan() {
                    assert!(b.is_nan());
                } else {
                    assert_eq!(
                        row[axis],
                        row[axis + 6],
                        "rotation case {index}, axis={axis}, local={local}"
                    );
                }
            }
        }
    }
    eprintln!(
        "{} cases match reference rotation for storage and shared rounding; xyz_only={xyz}; signed-zero, tiny nonzero and nonfinite guard coverage",
        cases.len()
    );
}
