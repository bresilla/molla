#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::Vec3;
use molla_solvers::fem_rigid_gpu::{
    FemRigidGpuScene, FemRigidGpuSystem, request_fem_device_limits,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[path = "common/fem_rotor.rs"]
mod fem_rotor;

fn moving_scene() -> FemRigidGpuScene {
    let mut scene = fem_rotor::scene();
    scene
        .soft_state
        .particle_qd
        .host_mut()
        .unwrap()
        .fill(Vec3::X * 0.3);
    scene
}

fn mapped(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let result = buffer.slice(..).get_mapped_range().to_vec();
    buffer.unmap();
    result
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit([encoder.finish()]);
    mapped(device, &staging)
}

fn render(device: &wgpu::Device, queue: &wgpu::Queue, positions: &wgpu::Buffer) -> Vec<u8> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FEM direct vertex pulling"),
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
             @vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                 return vec4<f32>(positions[i].xy * 8.0 - vec2<f32>(0.4), 0.0, 1.0);
             }
             @fragment fn fragment() -> @location(0) vec4<f32> {
                 return vec4<f32>(1.0, 0.0, 0.0, 1.0);
             }"
            .into(),
        ),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: positions.as_entire_binding(),
        }],
    });
    let size = wgpu::Extent3d {
        width: 128,
        height: 128,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 128 * 128 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(512),
                rows_per_image: Some(128),
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);
    mapped(device, &output)
}

fn centroid(pixels: &[u8]) -> (f64, usize) {
    let mut sum = 0_usize;
    let mut count = 0;
    for (i, pixel) in pixels.chunks_exact(4).enumerate() {
        if pixel[0] > 200 {
            sum += i % 128;
            count += 1;
        }
    }
    assert!(count > 1000);
    (sum as f64 / count as f64, count)
}

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    struct WakeThread(std::thread::Thread);
    impl std::task::Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn negotiated_renderer_limits_support_fem_and_direct_rendering() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = block_on(instance.request_adapter(&Default::default())).unwrap();
    let mut descriptor = wgpu::DeviceDescriptor::default();
    request_fem_device_limits(&mut descriptor.required_limits, &adapter.limits()).unwrap();
    let (device, queue) = block_on(adapter.request_device(&descriptor)).unwrap();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let mut system = FemRigidGpuSystem::new_on_device(
        device.clone(),
        queue.clone(),
        moving_scene(),
        fem_rotor::config(),
    )
    .unwrap();
    let before = render(
        &device,
        &queue,
        system.soft_state().particle_q.device_buffer().unwrap(),
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let mut submitted = 0;
    while submitted < 65 {
        assert!(std::time::Instant::now() < deadline);
        if system.try_step_substep(1.0 / 4096.0).unwrap() {
            submitted += 1;
        }
        std::thread::yield_now();
    }
    let after = render(
        &device,
        &queue,
        system.soft_state().particle_q.device_buffer().unwrap(),
    );
    assert!(system.poll_completion().unwrap());
    assert!(system.soft_state().particle_q.host().is_err());
    assert_eq!(read(&device, &queue, system.status_buffer()), [0; 4]);
    let (before_x, before_count) = centroid(&before);
    let (after_x, after_count) = centroid(&after);
    assert!(after_x > before_x + 2.0);
    assert_eq!(before_count, after_count);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn insufficient_external_device_is_rejected_without_destroying_it() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("renderer with default limits"),
        ..Default::default()
    }))
    .unwrap();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    assert!(device.limits().max_bind_groups < 6);
    let error = FemRigidGpuSystem::new_on_device(
        device.clone(),
        queue.clone(),
        moving_scene(),
        fem_rotor::config(),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("FEM device requires"));
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &[1, 2, 3, 4]);
    assert_eq!(read(&device, &queue, &buffer), [1, 2, 3, 4]);
    let cpu_error = FemRigidGpuSystem::new(
        Arc::new(Engine::new_cpu()),
        moving_scene(),
        fem_rotor::config(),
    )
    .err()
    .unwrap();
    assert!(cpu_error.to_string().contains("FEM requires a GPU device"));
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn external_device_renders_resident_fem_and_retains_owner_lifecycle() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let engine_weak = Arc::downgrade(&engine);
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let losses = Arc::new(AtomicUsize::new(0));
    let callback_losses = losses.clone();
    device.set_device_lost_callback(move |_, _| {
        callback_losses.fetch_add(1, Ordering::SeqCst);
    });
    let mut baseline =
        FemRigidGpuSystem::new(engine.clone(), moving_scene(), fem_rotor::config()).unwrap();
    let mut external = FemRigidGpuSystem::new_on_device(
        device.clone(),
        queue.clone(),
        moving_scene(),
        fem_rotor::config(),
    )
    .unwrap();
    let mut sampled = FemRigidGpuSystem::new_surface_sampled_on_device(
        device.clone(),
        queue.clone(),
        moving_scene(),
        fem_rotor::config(),
    )
    .unwrap();
    drop(engine);
    assert!(engine_weak.upgrade().is_none());
    let before = render(
        &device,
        &queue,
        external.soft_state().particle_q.device_buffer().unwrap(),
    );
    for system in [&mut baseline, &mut external, &mut sampled] {
        system.set_drive_effort(0, 0.5).unwrap();
        system.enable_momentum_diagnostics();
        system.step(65.0 / 4096.0).unwrap();
        assert!(system.soft_state().particle_q.host().is_err());
        assert!(system.soft_state().particle_qd.host().is_err());
        assert_eq!(read(&device, &queue, system.status_buffer()), [0; 4]);
    }
    let snapshot = |system: &FemRigidGpuSystem| {
        [
            system.soft_state().particle_q.device_buffer().unwrap(),
            system.soft_state().particle_qd.device_buffer().unwrap(),
            system.rigid_state().joint_q.device_buffer().unwrap(),
            system.rigid_state().joint_qd.device_buffer().unwrap(),
            system.rigid_state().body_q.device_buffer().unwrap(),
        ]
        .map(|buffer| read(&device, &queue, buffer))
    };
    assert_eq!(snapshot(&baseline), snapshot(&external));
    assert_eq!(snapshot(&baseline), snapshot(&sampled));
    let positions = external
        .soft_state()
        .particle_q
        .device_buffer()
        .unwrap()
        .clone();
    let after = render(&device, &queue, &positions);
    let (before_x, before_count) = centroid(&before);
    let (after_x, after_count) = centroid(&after);
    eprintln!(
        "resident rendering centroid: {before_x} -> {after_x}; pixels {before_count}/{after_count}"
    );
    assert!(after_x - before_x > 1.0);
    assert!(before_count.abs_diff(after_count) < 100);
    assert!(external.soft_state().particle_q.host().is_err());
    drop((baseline, external, sampled));
    assert_eq!(render(&device, &queue, &positions), after);
    assert_eq!(losses.load(Ordering::SeqCst), 0);
    device.destroy();
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    assert_eq!(losses.load(Ordering::SeqCst), 1);
}
