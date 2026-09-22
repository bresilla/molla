#![cfg(feature = "gpu")]

#[path = "../src/vbd_cloth/bindings.rs"]
mod bindings;

use bindings::BindingCache;
use molla_compute::Engine;

fn storage_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn storage_group(
    cache: &mut BindingCache,
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
    offset: u64,
    size: u64,
) -> wgpu::BindGroup {
    cache.storage_group(
        device,
        &wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset,
                    size: wgpu::BufferSize::new(size),
                }),
            }],
        },
    )
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn cached_bindings_preserve_identity_and_pending_commands() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let layout = storage_layout(device);
    let other_layout = storage_layout(device);
    let uniform_descriptor = wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    };
    let uniform_layout = device.create_bind_group_layout(&uniform_descriptor);
    let other_uniform_layout = device.create_bind_group_layout(&uniform_descriptor);
    let buffer = || {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("same label, distinct buffers"),
            size: 512,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let a = buffer();
    let b = buffer();
    let mut cache = BindingCache::new(2);
    let first = storage_group(&mut cache, device, &layout, &a, 0, 256);
    assert_eq!(
        first,
        storage_group(&mut cache, device, &layout, &a.clone(), 0, 256)
    );
    assert_ne!(
        first,
        storage_group(&mut cache, device, &layout, &b, 0, 256)
    );
    assert_eq!(
        first,
        storage_group(&mut cache, device, &layout, &a, 0, 256)
    );
    assert_ne!(
        first,
        storage_group(&mut cache, device, &layout, &a, 256, 256)
    );
    assert_eq!(
        first,
        storage_group(&mut cache, device, &layout, &a, 0, 256)
    );
    assert_ne!(
        first,
        storage_group(&mut cache, device, &layout, &a, 0, 128)
    );
    assert_eq!(
        first,
        storage_group(&mut cache, device, &layout, &a, 0, 256)
    );
    assert_ne!(
        first,
        storage_group(&mut cache, device, &other_layout, &a, 0, 256)
    );
    let original_uniform = cache.uniform_group(
        device,
        &uniform_layout,
        bytemuck::cast_slice(&[7_u32, 0, 0, 0]),
    );
    assert_eq!(
        original_uniform,
        cache.uniform_group(
            device,
            &uniform_layout,
            bytemuck::cast_slice(&[7_u32, 0, 0, 0]),
        )
    );
    assert_ne!(
        original_uniform,
        cache.uniform_group(
            device,
            &other_uniform_layout,
            bytemuck::cast_slice(&[7_u32, 0, 0, 0]),
        )
    );
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout), Some(&uniform_layout)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read_write> output: array<u32>;
             @group(1) @binding(0) var<uniform> value: vec4<u32>;
             @compute @workgroup_size(1) fn main() { output[0] += value.x; }"
                .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut expected = [0_u32; 4];
    for value in 1_u32..=24 {
        let index = (value as usize - 1) % 4;
        let buffer = if index < 2 { &a } else { &b };
        let group = storage_group(
            &mut cache,
            device,
            &layout,
            buffer,
            (index % 2 * 256) as u64,
            256,
        );
        let uniform = cache.uniform_group(
            device,
            &uniform_layout,
            bytemuck::cast_slice(&[value, 0, 0, 0]),
        );
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.set_bind_group(1, &uniform, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        expected[index] += value;
        assert_eq!(cache.entry_counts(), (2, 2));
    }
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &first, &[]);
        pass.set_bind_group(1, &original_uniform, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        expected[0] += 7;
    }
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    for (index, buffer) in [&a, &a, &b, &b].into_iter().enumerate() {
        encoder.copy_buffer_to_buffer(
            buffer,
            (index % 2 * 256) as u64,
            &readback,
            (index * 4) as u64,
            4,
        );
    }
    drop(cache);
    queue.submit([encoder.finish()]);
    let (send, recv) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).unwrap()
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    recv.recv().unwrap().unwrap();
    let mapped = readback.slice(..).get_mapped_range();
    assert_eq!(bytemuck::cast_slice::<u8, u32>(&mapped), expected);
    drop(mapped);
    readback.unmap();
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn cached_vbd_matches_fresh_bindings_with_changing_inputs() {
    use molla_core::WorldId;
    use molla_kernels::VbdClothKernel;
    use molla_math::Vec3;
    use molla_sim::ModelBuilder;

    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut scenes = Vec::new();
    for scale in [0.1, 0.2] {
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), Vec3::new(0.0, 0.0, -9.81));
        for point in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
            builder.add_particle(point * scale, 1.0, 0.0, WorldId(0));
        }
        builder
            .add_tet_mesh(&[[0, 1, 2, 3]], 2e5, 0.45, None)
            .unwrap();
        let mut model = builder.build().unwrap();
        let mut cached_in = model.state().unwrap();
        let mut cached_out = model.state().unwrap();
        let mut fresh_in = model.state().unwrap();
        let mut fresh_out = model.state().unwrap();
        for state in [&mut cached_in, &mut fresh_in] {
            state.particle_q.host_mut().unwrap()[3].z *= 1.03;
        }
        model.upload_to_device(device, queue).unwrap();
        for state in [
            &mut cached_in,
            &mut cached_out,
            &mut fresh_in,
            &mut fresh_out,
        ] {
            state.upload_to_device(device, queue).unwrap();
        }
        scenes.push((model, cached_in, cached_out, fresh_in, fresh_out));
    }
    let mut cached = VbdClothKernel::new(device, queue, 4, 0, &[]);
    for step in 0..12 {
        for (model, cached_in, cached_out, fresh_in, fresh_out) in &mut scenes {
            model.linear_damping = if step % 3 == 0 { 0.0 } else { 0.1 };
            model.gravity.host_mut().unwrap()[0].x = step as f64 * 0.1;
            model.gravity.upload(queue).unwrap();
            let dt = if step % 4 < 2 {
                1.0 / 1200.0
            } else {
                1.0 / 2400.0
            };
            let hard = step % 2 == 0;
            let mut fresh = VbdClothKernel::new(device, queue, 4, 0, &[]);
            cached
                .dispatch(
                    device, queue, model, cached_in, cached_out, dt, 8, 0.0, hard,
                )
                .unwrap();
            fresh
                .dispatch(device, queue, model, fresh_in, fresh_out, dt, 8, 0.0, hard)
                .unwrap();
            std::mem::swap(cached_in, cached_out);
            std::mem::swap(fresh_in, fresh_out);
        }
    }
    for (_, mut cached, _, mut fresh, _) in scenes {
        cached.particle_q.download(device, queue).unwrap();
        fresh.particle_q.download(device, queue).unwrap();
        cached.particle_qd.download(device, queue).unwrap();
        fresh.particle_qd.download(device, queue).unwrap();
        cached.tet_f_plastic.download(device, queue).unwrap();
        fresh.tet_f_plastic.download(device, queue).unwrap();
        cached.tet_plastic_strain.download(device, queue).unwrap();
        fresh.tet_plastic_strain.download(device, queue).unwrap();
        cached.tet_back_stress.download(device, queue).unwrap();
        fresh.tet_back_stress.download(device, queue).unwrap();
        assert_eq!(
            cached.tet_f_plastic.host().unwrap(),
            fresh.tet_f_plastic.host().unwrap()
        );
        assert_eq!(
            cached.tet_plastic_strain.host().unwrap(),
            fresh.tet_plastic_strain.host().unwrap()
        );
        assert_eq!(
            cached.tet_back_stress.host().unwrap(),
            fresh.tet_back_stress.host().unwrap()
        );
        assert_eq!(
            cached.particle_q.host().unwrap(),
            fresh.particle_q.host().unwrap()
        );
        assert_eq!(
            cached.particle_qd.host().unwrap(),
            fresh.particle_qd.host().unwrap()
        );
        assert!(
            cached
                .particle_q
                .host()
                .unwrap()
                .iter()
                .all(|q| q.is_finite())
        );
        let velocities = cached.particle_qd.host().unwrap();
        eprintln!("cached/fresh VBD velocities: {velocities:?}");
        assert!(velocities.iter().map(|v| v.z).sum::<f64>() < 0.0);
    }
}
