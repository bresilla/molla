use crate::{Engine, PipelineCacheSnapshot};

#[test]
fn cpu_engine_rejects_compiler_cache_without_side_effects() {
    let mut engine = Engine::new_cpu();
    assert!(engine.enable_pipeline_cache(None).is_err());
    assert!(engine.pipeline_cache().is_none());
    assert!(engine.pipeline_cache_snapshot().is_err());
    engine.disable_pipeline_cache();
    fn send_sync<T: Send + Sync>() {}
    send_sync::<PipelineCacheSnapshot>();
    send_sync::<Engine>();
}

fn dispatch(engine: &Engine, factor: u32, addend: u32) -> Vec<u32> {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compiler-cache output contract"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "override factor: u32 = 1u;
            @group(0) @binding(0) var<storage, read_write> output: array<u32>;
            @compute @workgroup_size(8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                if id.x < arrayLength(&output) {{ output[id.x] = factor * id.x + {addend}u; }}
            }}"
            )
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compiler-cache output contract"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[("factor", f64::from(factor))],
            ..Default::default()
        },
        cache: engine.pipeline_cache(),
    });
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 68,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 68,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(3, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buffer, 0, &readback, 0, 68);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let view = readback.slice(..).get_mapped_range();
    let words = view
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    drop(view);
    readback.unmap();
    words
}

#[test]
#[ignore = "requires real GPU with PIPELINE_CACHE"]
fn pipeline_cache_snapshots_preserve_outputs_variants_and_device_lifecycle() {
    let mut plain = Engine::new_gpu().unwrap();
    assert!(!plain.adapter_info().unwrap().is_software);
    assert!(plain.pipeline_cache_snapshot().unwrap().is_none());
    assert!(plain.enable_pipeline_cache(None).is_err());
    assert!(plain.pipeline_cache().is_none());
    let expected = dispatch(&plain, 7, 13);
    assert_eq!(expected, (0..17).map(|i| 7 * i + 13).collect::<Vec<_>>());
    drop(plain);

    let mut first = Engine::new_gpu_with_features(wgpu::Features::PIPELINE_CACHE).unwrap();
    assert!(first.pipeline_cache().is_none());
    first.enable_pipeline_cache(None).unwrap();
    assert_eq!(dispatch(&first, 7, 13), expected);
    let saved = first.pipeline_cache_snapshot().unwrap().unwrap();
    assert!(!saved.as_bytes().is_empty());
    first.clear_gpu_program_cache();
    assert!(first.pipeline_cache().is_some());
    first.destroy_device().unwrap();
    assert!(first.pipeline_cache_snapshot().is_err());
    assert!(first.enable_pipeline_cache(Some(&saved)).is_err());
    drop(first);

    // These unmodified bytes are from get_data above.
    let imported =
        unsafe { PipelineCacheSnapshot::from_persisted_bytes(saved.as_bytes().to_vec()) };
    let mut second = Engine::new_gpu_with_features(wgpu::Features::PIPELINE_CACHE).unwrap();
    second.enable_pipeline_cache(Some(&imported)).unwrap();
    assert_eq!(dispatch(&second, 7, 13), expected);
    assert_eq!(
        dispatch(&second, 19, 13),
        (0..17).map(|i| 19 * i + 13).collect::<Vec<_>>()
    );
    assert_eq!(
        dispatch(&second, 7, 29),
        (0..17).map(|i| 7 * i + 29).collect::<Vec<_>>()
    );
    let renewed = second.pipeline_cache_snapshot().unwrap().unwrap();
    second.disable_pipeline_cache();
    assert!(second.pipeline_cache().is_none());
    assert!(second.pipeline_cache_snapshot().unwrap().is_none());
    assert_eq!(dispatch(&second, 7, 13), expected);
    second.enable_pipeline_cache(Some(&renewed)).unwrap();
    assert_eq!(dispatch(&second, 7, 13), expected);
    eprintln!(
        "compiler cache snapshot bytes={}, renewed={}; outputs exact; no cache-hit claim",
        saved.as_bytes().len(),
        renewed.as_bytes().len()
    );
}
