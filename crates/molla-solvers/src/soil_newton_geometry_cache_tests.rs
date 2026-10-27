use super::*;

fn buffer(engine: &Engine, data: &[u8]) -> wgpu::Buffer {
    let result = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("geometry cache fixture"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
    engine.queue().unwrap().write_buffer(&result, 0, data);
    result
}

fn read(engine: &Engine, source: &wgpu::Buffer) -> Vec<u8> {
    let device = engine.device().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("geometry cache observation"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let bytes = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();
    bytes
}

fn fixture(engine: &Engine, count: u32) -> (ExperimentalSoilNewtonGeometryGpu, wgpu::Buffer) {
    let mass = buffer(engine, bytemuck::cast_slice(&[860.0f32]));
    let free = buffer(engine, bytemuck::cast_slice(&[0.0f32; 2]));
    let centers = buffer(engine, bytemuck::cast_slice(&[0.0f32; 4]));
    let shapes = buffer(
        engine,
        bytemuck::cast_slice(&[
            [0.0f32; 4],
            [0.0; 4],
            [0.0, 0.0, 0.0, 1.0],
            [0.45, 0.45, 0.45, 1.0],
        ]),
    );
    let jacobians = buffer(
        engine,
        bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0]),
    );
    let queries = buffer(
        engine,
        bytemuck::cast_slice(&vec![
            SoilContactQuery {
                position_mass: [0.5, 0.0, 0.0, 1.0],
                velocity_dt: [0.25, 0.0, 0.0, 0.01],
                body_boundary: [0; 4],
            };
            count as usize
        ]),
    );
    let coefficients = buffer(engine, bytemuck::cast_slice(&[0.3f32]));
    let owner = ExperimentalSoilNewtonGeometryGpu::new(
        engine,
        SoilNewtonGeometryInputs {
            mass: &mass,
            free_rates: &free,
            centers: &centers,
            shapes: &shapes,
            jacobians: &jacobians,
            queries: &queries,
            coefficients: &coefficients,
        },
        SoilNewtonGeometryCapacity {
            dofs: 1,
            bodies: 1,
            contacts: count,
            newton_iterations: 64,
            krylov_iterations: 64,
            max_owned_bytes: u64::MAX,
        },
    )
    .unwrap();
    (owner, coefficients)
}

fn run(engine: &Engine, owner: &ExperimentalSoilNewtonGeometryGpu) -> Vec<u8> {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    encoder.clear_buffer(&owner.status, 0, None);
    owner.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = transfers.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    read(engine, &owner.status)
}

#[test]
#[ignore = "requires GPU; Newton bridge program reuse with independent uniforms and faults"]
fn newton_geometry_program_cache_lifecycle() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let (a, coefficients) = fixture(&engine, 1);
    let (b, _) = fixture(&engine, 65);
    assert!(std::sync::Arc::ptr_eq(&a.program, &b.program));
    let retained = std::sync::Arc::downgrade(&a.program);
    assert_eq!(&run(&engine, &a)[..16], &[0; 16]);
    assert_eq!(&run(&engine, &b)[..16], &[0; 16]);
    let before: Vec<_> = b.outputs.iter().map(|v| read(&engine, v)).collect();
    for bytes in before[0].chunks_exact(16) {
        assert_eq!(
            bytemuck::pod_read_unaligned::<[f32; 4]>(bytes),
            [0.25, 0.0, 0.0, 0.0]
        );
    }
    engine
        .queue()
        .unwrap()
        .write_buffer(&coefficients, 0, &(-1.0f32).to_le_bytes());
    assert_ne!(&run(&engine, &a)[..4], &[0; 4]);
    assert_eq!(&run(&engine, &b)[..16], &[0; 16]);
    assert_eq!(
        b.outputs
            .iter()
            .map(|v| read(&engine, v))
            .collect::<Vec<_>>(),
        before
    );
    engine.clear_gpu_program_cache();
    let rebuilt = geometry_program(&engine).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&rebuilt, &b.program));
    assert_eq!(&run(&engine, &b)[..16], &[0; 16]);
    assert_eq!(
        b.outputs
            .iter()
            .map(|v| read(&engine, v))
            .collect::<Vec<_>>(),
        before
    );
    engine
        .queue()
        .unwrap()
        .write_buffer(&coefficients, 0, &0.3f32.to_le_bytes());
    assert_eq!(&run(&engine, &a)[..16], &[0; 16]);
    drop(a);
    assert!(retained.upgrade().is_some());
    drop(b);
    assert!(retained.upgrade().is_none());
    let foreign = Engine::new_gpu().unwrap();
    assert!(!std::sync::Arc::ptr_eq(
        &rebuilt,
        &geometry_program(&foreign).unwrap()
    ));
}
