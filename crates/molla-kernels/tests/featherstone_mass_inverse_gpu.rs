#![cfg(feature = "gpu")]

use molla_compute::Engine;

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn joint_psd_inverse_preserves_scaled_weak_modes_and_rejects_indefinite_blocks() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut matrices = Vec::<[f32; 36]>::new();
    let mut expected = Vec::<Option<[f64; 36]>>::new();
    for scale in [1e-20_f32, 1.0, 1e20] {
        let mut matrix = [0.0; 36];
        let mut inverse = [0.0; 36];
        for (axis, value) in [1.0, 1e-10, 0.0, 2.0, 0.0, 4.0].into_iter().enumerate() {
            matrix[axis * 6 + axis] = value * scale;
            if value != 0.0 {
                inverse[axis * 6 + axis] = 1.0 / matrix[axis * 6 + axis] as f64;
            }
        }
        matrices.push(matrix);
        expected.push(Some(inverse));
        let mut matrix = [0.0; 36];
        let mut inverse = [0.0; 36];
        for row in 0..2 {
            for column in 0..2 {
                matrix[row * 6 + column] = scale;
                inverse[row * 6 + column] = 0.25 / scale as f64;
            }
        }
        matrix[14] = scale * 1e-10;
        inverse[14] = 1.0 / matrix[14] as f64;
        matrices.push(matrix);
        expected.push(Some(inverse));
        matrix[0] = -scale;
        matrices.push(matrix);
        expected.push(None);
        matrices.push([0.0; 36]);
        expected.push(Some([0.0; 36]));
    }
    let mut invalid = [0.0; 36];
    invalid[0] = f32::NAN;
    matrices.push(invalid);
    expected.push(None);
    let storage = |bytes: &[u8], size| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("joint inverse component"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            queue.write_buffer(&buffer, 0, bytes);
        }
        buffer
    };
    let input = storage(bytemuck::cast_slice(&matrices), matrices.len() as u64 * 144);
    let bytes = matrices.len() as u64 * 160;
    let output = storage(&[], bytes);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("joint inverse readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let source = format!(
        "{}\n{}",
        include_str!("../shaders/featherstone_mass_inverse.wgsl"),
        r#"
struct Output { matrix: array<f32,36>, flags: vec4<u32> }
@group(0) @binding(0) var<storage,read> inputs: array<array<f32,36>>;
@group(0) @binding(1) var<storage,read_write> outputs: array<Output>;
@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&inputs) { return; }
    let result = joint_psd_inverse(inputs[id.x],6u);
    outputs[id.x] = Output(result.matrix,vec4<u32>(select(0u,1u,result.valid),0u,0u,0u));
}
"#
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("joint inverse component"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("joint inverse component"),
        layout: None,
        module: &shader,
        entry_point: Some("evaluate"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("joint inverse component"),
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
    let mut previous = Vec::new();
    for _ in 0..3 {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((matrices.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = staging.slice(..).get_mapped_range();
        if !previous.is_empty() {
            assert_eq!(previous, mapped[..]);
        }
        previous = mapped.to_vec();
        for (index, (actual, expected)) in mapped.chunks_exact(160).zip(&expected).enumerate() {
            let words = bytemuck::cast_slice::<u8, u32>(actual);
            assert_eq!(words[36] != 0, expected.is_some(), "case={index}");
            if let Some(expected) = expected {
                for (axis, expected) in expected.iter().enumerate() {
                    let actual = f32::from_bits(words[axis]) as f64;
                    if *expected == 0.0 {
                        assert_eq!(actual, 0.0, "case={index} entry={axis}");
                    } else {
                        assert!(
                            (actual / expected - 1.0).abs() < 2e-5,
                            "case={index} entry={axis} expected={expected} actual={actual}"
                        );
                    }
                }
            }
        }
        drop(mapped);
        staging.unmap();
    }
}
