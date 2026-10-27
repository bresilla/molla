use super::*;
use wgpu::util::DeviceExt;

#[allow(dead_code)]
mod readback {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/common/gpu_readback.rs"
    ));
}

#[test]
#[ignore = "requires a real GPU through make test"]
fn green_viscosity_gpu_matches_independent_double_precision_blocks() {
    let engine = crate::compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut input = Vec::<[f32; 32]>::new();
    let mut expected = Vec::new();
    let mut cases = states();
    let small_start = cases.len() * 16;
    cases.extend(
        [1e-10, -1e-10, 1e-8, -1e-8].map(|delta| (Mat3::IDENTITY * (1.0 + delta), Mat3::IDENTITY)),
    );
    for (f, previous) in cases {
        for gradient in [-Vec3::ONE, Vec3::X, Vec3::Y, Vec3::Z] {
            for (shear, bulk) in [(0.0, 0.0), (3.0, 0.0), (0.0, 7.0), (3.0, 7.0)] {
                let mut values = [0.0_f32; 32];
                for (column, offset) in previous
                    .to_cols_array_2d()
                    .into_iter()
                    .chain((f - previous).to_cols_array_2d())
                    .zip((0..24).step_by(4))
                {
                    values[offset..offset + 3].copy_from_slice(&column.map(|v| v as f32));
                }
                values[24..27].copy_from_slice(&gradient.to_array().map(|v| v as f32));
                values[28..32].copy_from_slice(&[shear, bulk, 50.0, 0.17]);
                let unpack = |offset| {
                    Mat3::from_cols_array_2d(&std::array::from_fn(|c| {
                        std::array::from_fn(|r| values[offset + c * 4 + r] as f64)
                    }))
                };
                let b = increment_block(
                    unpack(0),
                    unpack(12),
                    gradient,
                    shear as f64,
                    bulk as f64,
                    0.02,
                    values[31] as f64,
                );
                expected.push(b);
                input.push(values);
            }
        }
    }
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Green strain viscosity oracle"),
        source: wgpu::ShaderSource::Wgsl(format!("{}\n{}",
            include_str!("../../../../molla-kernels/shaders/tet_viscosity.wgsl"),
            "struct Input { previous: mat3x3<f32>, increment: mat3x3<f32>, gradient: vec4<f32>, material: vec4<f32> }
             @group(0) @binding(0) var<storage, read> input: array<Input>;
             @group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
             @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                 let i = id.x;
                 if i >= arrayLength(&input) { return; }
                 let v = input[i];
                 let b = tet_viscous_increment_block(v.previous, v.increment, v.gradient.xyz, v.material.x, v.material.y, v.material.z, v.material.w);
                 output[5u*i] = vec4<f32>(b.force, 0.0);
                 output[5u*i+1u] = vec4<f32>(b.tangent[0], 0.0);
                 output[5u*i+2u] = vec4<f32>(b.tangent[1], 0.0);
                 output[5u*i+3u] = vec4<f32>(b.tangent[2], 0.0);
                 output[5u*i+4u] = vec4<f32>(b.geometric, b.potential, 0.0, 0.0);
             }").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let source = molla_core::gpu_buffer_init!(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        }
    );
    let target = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: None,
            size: expected.len() as u64 * 80,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }
    );
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: target.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((expected.len() as u32).div_ceil(64), 1, 1);
    }
    queue.submit([encoder.finish()]);
    let actual = readback::read::<[f32; 4]>(&engine, &target, expected.len() * 5);
    for (i, (actual, expected)) in actual.chunks_exact(5).zip(expected).enumerate() {
        if i >= small_start && input[i][29] > 0.0 {
            let force = Vec3::new(
                actual[0][0] as f64,
                actual[0][1] as f64,
                actual[0][2] as f64,
            );
            assert!(
                (force - expected.force).length() < 2e-6 * expected.force.length(),
                "small strain lost: case{i}: {force:?} != {:?}",
                expected.force
            );
        }
        let vectors = [
            expected.force,
            expected.tangent.x_axis,
            expected.tangent.y_axis,
            expected.tangent.z_axis,
            Vec3::new(expected.geometric, expected.potential, 0.0),
        ];
        for (row, reference) in actual.iter().zip(vectors) {
            for (a, b) in row[..3].iter().zip(reference.to_array()) {
                assert!(
                    a.is_finite() && (*a as f64 - b).abs() < 2e-4 + 3e-6 * b.abs(),
                    "case{i}: GPU {a} != CPU {b}"
                );
            }
        }
    }
}
