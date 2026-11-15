#![cfg(feature = "resource-counters")]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
use molla_kernels::constitutive::{
    ConstitutiveInput, ConstitutiveKernel, OUTPUT_FLOATS, SHADER_SOURCE, decode_response,
};
use molla_math::{
    Mat3, Quat, Real, Vec3,
    constitutive::{ElasticMaterial, ElasticModel},
};
use wgpu::util::DeviceExt;

fn buffer(device: &wgpu::Device, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage,
        mapped_at_creation: false,
    })
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<f32> {
    let stage = buffer(
        device,
        source.size(),
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &stage, 0, source.size());
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    stage
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = stage.slice(..).get_mapped_range().unwrap();
    let result = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    stage.unmap();
    result
}

fn models() -> [ElasticModel; 4] {
    [
        ElasticModel::Corotated,
        ElasticModel::QuarticStretch,
        ElasticModel::StableNeoHookean,
        ElasticModel::BaraffWitkin,
    ]
}

#[test]
fn shader_parses_and_validates() {
    let module = naga::front::wgsl::parse_str(SHADER_SOURCE)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(SHADER_SOURCE)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-materials-gpu"]
fn coupled_material_responses_match_f64_oracle() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = ConstitutiveKernel::new(device);
    let rotation = Mat3::from_quat(Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.5, 0.7));
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for model in models() {
        for n in [2, 3] {
            if n == 3 && model == ElasticModel::BaraffWitkin {
                continue;
            }
            let material = ElasticMaterial {
                model,
                mu: 2.0,
                lambda: 3.0,
            };
            for projected in [false, true] {
                for base in [
                    Mat3::IDENTITY,
                    Mat3::IDENTITY * 0.3,
                    Mat3::IDENTITY * 1.5,
                    Mat3::from_diagonal(Vec3::new(1.1, 1.1 + 1.0e-7, 0.8)),
                    Mat3::from_cols(
                        Vec3::new(1.2, 0.1, -0.1),
                        Vec3::new(0.3, 0.8, 0.2),
                        Vec3::new(0.1, -0.2, 1.1),
                    ),
                ] {
                    for rotate in [Mat3::IDENTITY, rotation] {
                        let f = Mat3::from_cols_array(
                            &(rotate * base).to_cols_array().map(|v| f64::from(v as f32)),
                        );
                        inputs.push(ConstitutiveInput::new(material, f, n, projected).unwrap());
                        expected.push((
                            model,
                            n,
                            projected,
                            material.evaluate(f, n, projected).unwrap(),
                        ));
                    }
                }
            }
        }
    }
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = buffer(
        device,
        (inputs.len() * OUTPUT_FLOATS * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let bindings = kernel
        .bind(device, &input, &output, inputs.len() as u32)
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &bindings);
    queue.submit([encoder.finish()]);
    let rows = read(device, queue, &output);
    let mut worst: Real = 0.0;
    for (row, (model, n, projected, cpu)) in rows.chunks_exact(OUTPUT_FLOATS).zip(expected) {
        let gpu = decode_response(row)
            .unwrap_or_else(|error| panic!("{model:?} n={n} projected={projected}: {error}"));
        let values = |r: &molla_math::constitutive::MaterialResponse| {
            std::iter::once(r.energy)
                .chain(r.gradient.to_cols_array())
                .chain(r.hessian.into_iter().flatten())
                .collect::<Vec<_>>()
        };
        for (a, b) in values(&cpu).into_iter().zip(values(&gpu)) {
            let error = (a - b).abs() / (1.0 + a.abs());
            worst = worst.max(error);
            assert!(
                error < 3.0e-4,
                "{model:?} n={n} projected={projected}: {a} != {b}, rel={error}"
            );
        }
    }
    eprintln!(
        "constitutive GPU cases={} worst_scaled_error={worst:.3e}",
        inputs.len()
    );
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-materials-gpu"]
fn malformed_device_inputs_and_short_buffers_are_refused() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = ConstitutiveKernel::new(device);
    let m = ElasticMaterial {
        model: ElasticModel::Corotated,
        mu: 1.0,
        lambda: 1.0,
    };
    assert!(ConstitutiveInput::new(m, Mat3::IDENTITY * Real::MAX, 3, true).is_err());
    let good = ConstitutiveInput::new(m, Mat3::IDENTITY, 3, true).unwrap();
    let base: [f32; 16] = *bytemuck::from_bytes(bytemuck::bytes_of(&good));
    let mut cases = vec![];
    for (index, value) in [
        (0, f32::NAN),
        (9, f32::INFINITY),
        (9, -1.0),
        (11, 5.0),
        (11, 0.5),
        (12, 1.0),
        (13, 2.0),
        (0, -1.0),
        (0, 0.0),
    ] {
        let mut values = base;
        values[index] = value;
        cases.push(values);
    }
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&cases),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = buffer(
        device,
        (cases.len() * OUTPUT_FLOATS * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    assert!(kernel.bind(device, &input, &output, 0).is_err());
    assert!(
        kernel
            .bind(device, &input, &output, cases.len() as u32 + 1)
            .is_err()
    );
    let tiny = buffer(device, 16, wgpu::BufferUsages::STORAGE);
    assert!(kernel.bind(device, &input, &tiny, 1).is_err());
    let bindings = kernel
        .bind(device, &input, &output, cases.len() as u32)
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &bindings);
    queue.submit([encoder.finish()]);
    for row in read(device, queue, &output).chunks_exact(OUTPUT_FLOATS) {
        assert!(decode_response(row).is_err());
    }
}

#[test]
#[ignore = "requires release GPU benchmark; run via make bench-ppf-materials"]
fn resident_tangent_resource_profile() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    eprintln!("constitutive adapter={:?}", engine.adapter_info().unwrap());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = ConstitutiveKernel::new(device);
    for count in [256u32, 4096] {
        for model in models() {
            let n = if model == ElasticModel::BaraffWitkin {
                2
            } else {
                3
            };
            let f = Mat3::from_cols(
                Vec3::new(0.9, 0.1, 0.0),
                Vec3::new(0.2, 1.1, 0.1),
                Vec3::new(0.0, 0.2, 0.8),
            );
            let m = ElasticMaterial {
                model,
                mu: 2.0,
                lambda: 3.0,
            };
            let records = vec![ConstitutiveInput::new(m, f, n, true).unwrap(); count as usize];
            let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&records),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let output = buffer(
                device,
                u64::from(count) * OUTPUT_FLOATS as u64 * 4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let bindings = kernel.bind(device, &input, &output, count).unwrap();
            let dispatch = || {
                let mut encoder = device.create_command_encoder(&Default::default());
                kernel.encode(&mut encoder, &bindings);
                queue.submit([encoder.finish()]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            };
            for _ in 0..4 {
                dispatch();
            }
            for repeat in 0..3 {
                let allocations = DeviceScope::start().unwrap();
                let transfers = TransferScope::start().unwrap();
                let dispatches = DispatchScope::start().unwrap();
                let mut times = Vec::new();
                for _ in 0..16 {
                    let now = std::time::Instant::now();
                    dispatch();
                    times.push(now.elapsed().as_secs_f64() * 1000.0);
                }
                let a = allocations.finish();
                let t = transfers.finish();
                let d = dispatches.finish();
                assert!(!a.invalid && !t.overflowed && !d.overflowed);
                assert_eq!(a.observations, 0);
                assert_eq!(t.readback_bytes, 0);
                assert_eq!(t.upload_bytes, 0);
                assert_eq!(d.encoded_dispatches, 16);
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "constitutive model={model:?} dimensions={n} count={count} repeat={repeat} p50_ms={:.3} p95_ms={:.3} resident_bytes={} allocations=0 uploads=0 readbacks=0 dispatches=16",
                    times[8],
                    times[15],
                    input.size() + output.size()
                );
            }
            for row in read(device, queue, &output).chunks_exact(OUTPUT_FLOATS) {
                decode_response(row).unwrap();
            }
        }
    }
}
