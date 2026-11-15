use super::{Materials, Prepared};
use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, TransferScope};
use molla_math::{
    Mat3, Vec3,
    constitutive::{ElasticMaterial, ElasticModel},
};
use molla_sim::ElasticElement;
use wgpu::util::DeviceExt;

fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<f32> {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &stage, 0, source.size());
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    stage
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = stage.slice(..).get_mapped_range().unwrap();
    let values = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    stage.unmap();
    values
}

fn rows(values: &[Vec3]) -> Vec<[f32; 4]> {
    values
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32, 0.0])
        .collect()
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-materials-gpu"]
fn resident_assembly_matches_coupled_oracle_and_rejects_invalid_trials() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z, Vec3::ONE];
    let mut elements = Vec::new();
    for model in [
        ElasticModel::Corotated,
        ElasticModel::QuarticStretch,
        ElasticModel::StableNeoHookean,
        ElasticModel::BaraffWitkin,
    ] {
        let material = ElasticMaterial {
            model,
            mu: 50.0,
            lambda: 30.0,
        };
        elements.push(ElasticElement::membrane(&rest, [1, 4, 2], 0.07, material).unwrap());
        if model != ElasticModel::BaraffWitkin {
            elements.push(ElasticElement::tetrahedron(&rest, [0, 1, 2, 3], material).unwrap());
        }
    }
    let affine = Mat3::from_cols(
        Vec3::new(1.2, 0.1, 0.0),
        Vec3::new(0.2, 0.8, 0.1),
        Vec3::new(0.1, -0.1, 1.1),
    );
    let x: Vec<_> = rest
        .iter()
        .map(|p| (affine * *p + Vec3::splat(0.2)).as_vec3().as_dvec3())
        .collect();
    let p: Vec<_> = x
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();
    let prepared = Prepared::new(device, &p, &elements).unwrap().unwrap();
    let buffer = |data: &[u8], uniform: bool| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage: (if uniform {
                wgpu::BufferUsages::UNIFORM
            } else {
                wgpu::BufferUsages::STORAGE
            }) | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    };
    let h = 0.03f32;
    let uniforms = buffer(
        bytemuck::cast_slice(&[5u32, 0, 0, 0, h.to_bits(), 0, 0, 0]),
        true,
    );
    let pinned = [1u32, 0, 0, 0, 1];
    let pin_buffer = buffer(bytemuck::cast_slice(&pinned), false);
    let x_buffer = buffer(bytemuck::cast_slice(&rows(&x)), false);
    let trial_buffer = buffer(bytemuck::cast_slice(&rows(&x)), false);
    let direction: Vec<_> = (0..5)
        .map(|i| Vec3::new(i as f64 * 0.2 - 0.3, 0.4 - i as f64 * 0.1, 0.2))
        .collect();
    let directions = buffer(bytemuck::cast_slice(&rows(&direction)), false);
    let newton: Vec<_> = direction.iter().map(|p| *p * 8.0).collect();
    let newton_buffer = buffer(bytemuck::cast_slice(&rows(&newton)), false);
    let zero = vec![[0.0f32; 4]; 5];
    let gradient = buffer(bytemuck::cast_slice(&zero), false);
    let product = buffer(bytemuck::cast_slice(&zero), false);
    let partial = buffer(bytemuck::cast_slice(&[0.0f32; 5]), false);
    let mut scalar_data = [0.0f32; 16];
    scalar_data[6] = 1.0;
    let scalars = buffer(bytemuck::cast_slice(&scalar_data), false);
    let material = Materials::new(
        device,
        prepared,
        &uniforms,
        &pin_buffer,
        [&x_buffer, &trial_buffer, &directions, &newton_buffer],
        [&gradient, &product, &partial, &scalars],
    );
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    material.validate(&mut encoder, &scalars);
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        for name in ["gradient", "hessian_mul", "energy_vertex", "ccd_alpha"] {
            assert!(material.encode_after(&mut pass, name));
        }
    }
    queue.submit([encoder.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    assert_eq!(allocations.finish().observations, 0);
    let transfer = transfers.finish();
    assert_eq!(transfer.upload_bytes, 0);
    assert_eq!(transfer.readback_bytes, 0);
    let actual_g = read(device, queue, &gradient);
    let actual_h = read(device, queue, &product);
    let actual_e: f64 = read(device, queue, &partial)
        .into_iter()
        .map(f64::from)
        .sum();
    let status = read(device, queue, &scalars);
    assert_eq!(status[13].to_bits(), 0);
    let mut expected_g = vec![Vec3::ZERO; 5];
    let mut expected_h = vec![Vec3::ZERO; 5];
    let mut expected_e = 0.0;
    let mut bound = 1.0f64;
    let newton: Vec<_> = newton
        .iter()
        .enumerate()
        .map(|(i, p)| if pinned[i] == 0 { *p } else { Vec3::ZERO })
        .collect();
    for e in &elements {
        let response = e.response(&x, true).unwrap();
        let shapes = e.shape_gradients();
        let mut df = Mat3::ZERO;
        for (k, &id) in e.vertices().iter().enumerate() {
            if pinned[id as usize] == 0 {
                let d = direction[id as usize];
                let s = shapes[k];
                df += Mat3::from_cols(d * s.x, d * s.y, d * s.z);
            }
        }
        let stress = response.hessian_mul(df);
        for (k, &id) in e.vertices().iter().enumerate() {
            if pinned[id as usize] == 0 {
                expected_g[id as usize] += response.gradient * shapes[k] * (h as f64).powi(2);
                expected_h[id as usize] += stress * shapes[k] * (h as f64).powi(2);
            }
        }
        expected_e += response.energy * (h as f64).powi(2);
        bound = bound.min(e.step_bound(&x, &newton).unwrap());
    }
    for i in 0..5 {
        for axis in 0..3 {
            assert!((actual_g[4 * i + axis] as f64 - expected_g[i][axis]).abs() < 3e-6);
            assert!((actual_h[4 * i + axis] as f64 - expected_h[i][axis]).abs() < 3e-6);
        }
    }
    assert!((actual_e - expected_e).abs() < 3e-6);
    assert!(bound < 1.0);
    assert!((status[6] as f64 - bound).abs() < 2e-6);
    let mut bad = x.clone();
    bad[3] = bad[0];
    queue.write_buffer(&trial_buffer, 0, bytemuck::cast_slice(&rows(&bad)));
    queue.write_buffer(&partial, 0, bytemuck::cast_slice(&[0.0f32; 5]));
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        material.encode_after(&mut pass, "energy_vertex");
    }
    queue.submit([encoder.finish()]);
    assert!(
        read(device, queue, &partial)
            .iter()
            .any(|v| v.is_infinite())
    );
    queue.write_buffer(&x_buffer, 0, bytemuck::cast_slice(&rows(&bad)));
    let mut encoder = device.create_command_encoder(&Default::default());
    material.validate(&mut encoder, &scalars);
    queue.submit([encoder.finish()]);
    assert_ne!(read(device, queue, &scalars)[13].to_bits(), 0);
    queue.write_buffer(&x_buffer, 0, bytemuck::cast_slice(&rows(&x)));
    let mut encoder = device.create_command_encoder(&Default::default());
    material.validate(&mut encoder, &scalars);
    queue.submit([encoder.finish()]);
    assert_eq!(read(device, queue, &scalars)[13].to_bits(), 0);
    let inversion = vec![
        Vec3::ZERO,
        -Vec3::X * 3.0,
        -Vec3::Y * 3.0,
        Vec3::ZERO,
        Vec3::ZERO,
    ];
    queue.write_buffer(&x_buffer, 0, bytemuck::cast_slice(&rows(&rest)));
    queue.write_buffer(&newton_buffer, 0, bytemuck::cast_slice(&rows(&inversion)));
    queue.write_buffer(&gradient, 0, bytemuck::cast_slice(&zero));
    queue.write_buffer(&scalars, 0, bytemuck::cast_slice(&scalar_data));
    let mut encoder = device.create_command_encoder(&Default::default());
    material.validate(&mut encoder, &scalars);
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        material.encode_after(&mut pass, "gradient");
        material.encode_after(&mut pass, "ccd_alpha");
    }
    queue.submit([encoder.finish()]);
    let status = read(device, queue, &scalars);
    assert_eq!(status[13].to_bits(), 0);
    let alpha = status[6] as f64;
    assert!(alpha > 0.0 && alpha < 1.0 / 3.0);
    let tet = elements.iter().find(|e| e.dimensions() == 3).unwrap();
    let at = |t| {
        rest.iter()
            .zip(&inversion)
            .map(|(x, d)| *x + *d * t)
            .collect::<Vec<_>>()
    };
    assert!(tet.deformation(&at(1.0)).unwrap().determinant() > 0.0);
    for i in 0..=100 {
        assert!(
            tet.deformation(&at(alpha * i as f64 / 100.0))
                .unwrap()
                .determinant()
                > 0.0
        );
    }
}
