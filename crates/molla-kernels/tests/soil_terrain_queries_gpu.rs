#![cfg(all(feature = "gpu", feature = "resource-counters"))]
use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
use molla_kernels::{
    PreparedSoilTerrainQueries as Queries, SoilTerrainSample,
    TireFemContactFeedbackGpuTerrain as Terrain,
};
use std::time::Duration;
use wgpu::util::DeviceExt;

fn buffer(device: &wgpu::Device, values: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("terrain query fixture"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn terrain<'a>(buffers: &'a [wgpu::Buffer; 5], flags: u32) -> Terrain<'a> {
    Terrain {
        origin: [0.; 2],
        spacing: 1.,
        nx: 4,
        nz: 4,
        heights: &buffers[0],
        friction: 0.7,
        friction_map: (flags & 1 != 0).then_some(&buffers[1]),
        water_film_map: (flags & 2 != 0).then_some(&buffers[2]),
        sinkage_map: (flags & 4 != 0).then_some(&buffers[3]),
        surface_velocity_map: (flags & 8 != 0).then_some(&buffers[4]),
        surface_velocity: [1., 2., 3.],
        soil_rolling_resistance_scale: 0.4,
    }
}

fn read(engine: &Engine, source: &wgpu::Buffer) -> Vec<SoilTerrainSample> {
    let device = engine.device().unwrap();
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &stage, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    stage.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    let result = bytemuck::cast_slice(&stage.slice(..).get_mapped_range()).to_vec();
    stage.unmap();
    result
}

fn near(a: f32, b: f64) {
    assert!((f64::from(a) - b).abs() < 2e-5, "{a} != {b}");
}

#[test]
#[ignore = "requires GPU; binned MPM raster followed by resident queries"]
fn soil_terrain_queries_gpu_mpm_export_chain() {
    use molla_kernels::{TireMpmGpuKernel, TireMpmTerrainRasterParams};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let positions: Vec<_> = (0..16)
        .flat_map(|i| [(i % 4) as f32 + 0.5, 0., (i / 4) as f32 + 0.5, 0.])
        .collect();
    let positions = buffer(device, &positions);
    let velocity = buffer(device, &[1., 2., 3., 0.].repeat(16));
    let compaction = buffer(device, &[0.2, 0., 0., 0.].repeat(16));
    let maps = [
        buffer(device, &[0.; 16]),
        buffer(device, &[0.; 16]),
        buffer(device, &[0.; 16]),
        buffer(device, &[0.; 16]),
        buffer(device, &[0.; 64]),
    ];
    let kernel = TireMpmGpuKernel::new(device);
    let raster = kernel
        .prepare_raster_to_buffers(
            device,
            &positions,
            &velocity,
            &compaction,
            &maps[0],
            &maps[3],
            &maps[4],
            &TireMpmTerrainRasterParams {
                particle_count: 16,
                origin: [0.; 2],
                spacing: 1.,
                nx: 4,
                nz: 4,
                rest_height: 2.,
            },
            [0.; 3],
        )
        .unwrap();
    let input = buffer(device, &[1.5, 0., 1.5, 0.]);
    let queries = Queries::new(&engine, &terrain(&maps, 12), &input, 1, 176).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let resources = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    raster.encode(&mut encoder);
    queries.encode(&engine, &mut encoder).unwrap();
    assert_eq!(resources.finish().observations, 0);
    assert_eq!(transfers.finish(), Default::default());
    drop(raster);
    drop(kernel);
    drop(positions);
    drop(velocity);
    drop(compaction);
    drop(maps);
    drop(input);
    engine.queue().unwrap().submit([encoder.finish()]);
    let sample = read(&engine, queries.output())[0];
    assert_eq!(sample.status, [1, 0, 0, 0]);
    assert_eq!(&sample.normal_height[..3], &[0., 1., 0.]);
    near(sample.normal_height[3], 1.8);
    near(sample.material[1], 0.2);
    assert_eq!(&sample.velocity_friction[..3], &[1., 2., 3.]);
    println!(
        "binned MPM export to resident height/normal/sinkage/velocity query: zero hot allocations/transfers"
    );
}

#[test]
#[ignore = "requires GPU; resident terrain queries, all optional maps and ordered reuse"]
fn soil_terrain_queries_gpu_oracle_and_ordering() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let heights: Vec<_> = (0..16)
        .map(|i| (i % 4) as f32 + 2. * (i / 4) as f32)
        .collect();
    let velocity: Vec<_> = heights.iter().flat_map(|&h| [h, -h, 2. * h, 0.]).collect();
    let buffers = [
        buffer(device, &heights),
        buffer(device, &[0.3; 16]),
        buffer(device, &[0.02; 16]),
        buffer(device, &[0.1; 16]),
        buffer(device, &velocity),
    ];
    let points: Vec<_> = (0..129)
        .flat_map(|i| {
            [
                -0.5 + (i % 9) as f32 * 0.5,
                0.,
                -0.5 + (i / 9 % 9) as f32 * 0.5,
                0.,
            ]
        })
        .collect();
    let input = buffer(device, &points);
    let height = |x: f64, z: f64| x.clamp(0., 3.) + 2. * z.clamp(0., 3.);
    for flags in 0..16 {
        let queries = Queries::new(
            &engine,
            &terrain(&buffers, flags),
            &input,
            129,
            Queries::required_bytes(129),
        )
        .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let dispatches = DispatchScope::start().unwrap();
        queries.encode(&engine, &mut encoder).unwrap();
        assert_eq!(resources.finish().observations, 0);
        assert_eq!(transfers.finish(), Default::default());
        assert_eq!(dispatches.finish().encoded_dispatches, 1);
        engine.queue().unwrap().submit([encoder.finish()]);
        for (p, s) in points.chunks_exact(4).zip(read(&engine, queries.output())) {
            assert_eq!(s.status, [1, 0, 0, 0]);
            let (x, z) = (f64::from(p[0]), f64::from(p[2]));
            let h = height(x, z);
            let dx = height(x + 0.5, z) - height(x - 0.5, z);
            let dz = height(x, z + 0.5) - height(x, z - 0.5);
            let norm = (dx * dx + 1. + dz * dz).sqrt();
            for (v, e) in s
                .normal_height
                .into_iter()
                .zip([-dx / norm, 1. / norm, -dz / norm, h])
            {
                near(v, e);
            }
            let expected = if flags & 8 != 0 {
                [h, -h, 2. * h]
            } else {
                [1., 2., 3.]
            };
            for (v, e) in s.velocity_friction[..3].iter().zip(expected) {
                near(*v, e);
            }
            near(
                s.velocity_friction[3],
                if flags & 1 != 0 { 0.3 } else { 0.7 },
            );
            near(s.material[0], if flags & 2 != 0 { 0.02 } else { 0. });
            near(s.material[1], if flags & 4 != 0 { 0.1 } else { 0. });
            near(s.material[2], 0.4);
        }
    }
    let queries = Queries::new(&engine, &terrain(&buffers, 15), &input, 129, u64::MAX).unwrap();
    let shifted = buffer(device, &heights.iter().map(|h| h + 2.).collect::<Vec<_>>());
    let before = buffer(device, &vec![0.; 129 * 16]);
    let mut encoder = device.create_command_encoder(&Default::default());
    queries.encode(&engine, &mut encoder).unwrap();
    encoder.copy_buffer_to_buffer(queries.output(), 0, &before, 0, before.size());
    encoder.copy_buffer_to_buffer(&shifted, 0, &buffers[0], 0, shifted.size());
    queries.encode(&engine, &mut encoder).unwrap();
    drop(buffers);
    drop(input);
    engine.queue().unwrap().submit([encoder.finish()]);
    let first = read(&engine, &before);
    let second = read(&engine, queries.output());
    for (a, b) in first.iter().zip(second) {
        assert_eq!(b.status, [1, 0, 0, 0]);
        near(b.normal_height[3] - a.normal_height[3], 2.);
        assert_eq!(&a.normal_height[..3], &b.normal_height[..3]);
    }
    println!(
        "2064 optional-map/oracle samples; ordered resident update and retained-source drop pass"
    );
}

#[test]
#[ignore = "requires GPU; invalid inputs, zero queries, budgets and device health"]
fn soil_terrain_queries_gpu_faults_and_lifecycle() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let buffers = [
        buffer(device, &[0.; 16]),
        buffer(device, &[0.; 16]),
        buffer(device, &[0.; 16]),
        buffer(device, &[0.; 16]),
        buffer(device, &[0.; 64]),
    ];
    let input = buffer(device, &[0., 0., 0., 0.]);
    let mut t = terrain(&buffers, 15);
    let resources = DeviceScope::start().unwrap();
    assert!(Queries::new(&engine, &t, &input, 1, 175).is_err());
    assert!(Queries::new(&engine, &t, &input, 2, u64::MAX).is_err());
    t.spacing = 1e-7;
    assert!(Queries::new(&engine, &t, &input, 1, u64::MAX).is_err());
    t.spacing = 1.;
    t.nx = 0;
    assert!(Queries::new(&engine, &t, &input, 1, u64::MAX).is_err());
    assert_eq!(resources.finish().observations, 0);
    t.nx = 4;
    let empty = Queries::new(&engine, &t, &input, 0, 176).unwrap();
    let queries = Queries::new(&engine, &t, &input, 1, 176).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let dispatch = DispatchScope::start().unwrap();
    empty.encode(&engine, &mut encoder).unwrap();
    assert_eq!(dispatch.finish().encoded_dispatches, 0);
    assert!(queries.encode(&Engine::new_cpu(), &mut encoder).is_err());
    for (coordinate, expected) in [(f32::NAN, 2), (f32::INFINITY, 2), (f32::MAX, 2), (0., 1)] {
        engine.queue().unwrap().write_buffer(
            &input,
            0,
            bytemuck::cast_slice(&[coordinate, 0., 0., 0.]),
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        queries.encode(&engine, &mut encoder).unwrap();
        engine.queue().unwrap().submit([encoder.finish()]);
        let s = read(&engine, queries.output())[0];
        assert_eq!(s.status, [expected, 0, 0, 0]);
        if expected != 1 {
            assert_eq!(s.normal_height, [0.; 4]);
            assert_eq!(s.velocity_friction, [0.; 4]);
            assert_eq!(s.material, [0.; 4]);
        }
    }
    for b in &buffers {
        engine
            .queue()
            .unwrap()
            .write_buffer(b, 0, bytemuck::bytes_of(&f32::NAN));
        let mut encoder = device.create_command_encoder(&Default::default());
        queries.encode(&engine, &mut encoder).unwrap();
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(read(&engine, queries.output())[0].status, [3, 0, 0, 0]);
        engine
            .queue()
            .unwrap()
            .write_buffer(b, 0, bytemuck::bytes_of(&0f32));
    }
    engine
        .device_health()
        .unwrap()
        .record(wgpu::DeviceLostReason::Destroyed, "fixture".into());
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(queries.encode(&engine, &mut encoder).is_err());
    assert!(Queries::new(&engine, &t, &input, 1, 176).is_err());
    println!(
        "zero query, invalid coordinates/maps, preallocation rejection, foreign engine and device loss pass"
    );
}
