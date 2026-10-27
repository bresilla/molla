#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::PreparedTireFemCavity;
use molla_vehicle::{TetraTireFem, TetraTireFemParams};
use wgpu::util::DeviceExt;

fn buffer(engine: &Engine, data: &[[f32; 4]]) -> wgpu::Buffer {
    engine
        .device()
        .unwrap()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
}

fn read(engine: &Engine, buffer: &wgpu::Buffer) -> [f32; 4] {
    let device = engine.device().unwrap();
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &stage, 0, 16);
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    stage
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let row = bytemuck::pod_read_unaligned(&stage.slice(..).get_mapped_range());
    stage.unmap();
    row
}

#[test]
#[ignore = "requires native GPU; real cavity meshes, deformation and retained lifecycle"]
fn resident_cavity_matches_cpu_gas_law() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut count = 0;
    let mut maximum = 0f64;
    for resolution in [
        TetraTireFemParams::realtime_viewer(),
        TetraTireFemParams::default(),
    ] {
        let mut tire = TetraTireFem::new(0., 180_000., resolution);
        let snapshot = tire.gpu_snapshot();
        let n = snapshot.positions.len();
        let mut positions = vec![[0.; 4]; 3];
        positions.extend_from_slice(&snapshot.positions);
        let pos = [buffer(&engine, &positions), buffer(&engine, &positions)];
        let hub = tire.hub();
        let mut hubs = vec![[0.; 4]; 3];
        hubs[2] = [hub[0] as f32, hub[1] as f32, hub[2] as f32, 0.];
        let hub_buffer = buffer(&engine, &hubs);
        let mesh = tire.cavity_surface();
        let bytes = PreparedTireFemCavity::required_bytes(
            mesh.tube.len() + mesh.negative_cap.len() + mesh.positive_cap.len(),
        )
        .unwrap();
        #[cfg(feature = "resource-counters")]
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        assert!(
            PreparedTireFemCavity::new(
                &engine,
                [&pos[0], &pos[1]],
                &hub_buffer,
                3,
                2,
                &mesh,
                bytes - 1
            )
            .is_err()
        );
        #[cfg(feature = "resource-counters")]
        assert_eq!(allocations.finish().observations, 0);
        let area = mesh.area_per_bead_m2;
        let mut cavity = PreparedTireFemCavity::new(
            &engine,
            [&pos[0], &pos[1]],
            &hub_buffer,
            3,
            2,
            &mesh,
            bytes,
        )
        .unwrap();
        assert_eq!(cavity.owned_buffer_bytes(), bytes);
        for pressure in [0., 40_000., 180_000., 400_000.] {
            tire.set_pressure(pressure);
            cavity.update_pressure(&engine, pressure as f32).unwrap();
            for deformation in [0.7, 1., 1.3] {
                for parity in [false, true] {
                    let mut nodes = snapshot.positions.clone();
                    for p in &mut nodes {
                        p[1] = (hub[1] + (p[1] as f64 - hub[1]) * deformation) as f32;
                        p[0] += (0.05 * deformation) as f32;
                    }
                    tire.apply_gpu_node_state(
                        &nodes,
                        &vec![[0.; 4]; n],
                        [hubs[2][0], hubs[2][1], hubs[2][2], 0., 0., 0., 0., 0.],
                    )
                    .unwrap();
                    engine.queue().unwrap().write_buffer(
                        &pos[usize::from(parity)],
                        48,
                        bytemuck::cast_slice(&nodes),
                    );
                    #[cfg(feature = "resource-counters")]
                    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
                    #[cfg(feature = "resource-counters")]
                    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
                    #[cfg(feature = "resource-counters")]
                    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
                    let mut encoder = engine
                        .device()
                        .unwrap()
                        .create_command_encoder(&Default::default());
                    cavity.encode(&mut encoder, parity);
                    engine.queue().unwrap().submit([encoder.finish()]);
                    #[cfg(feature = "resource-counters")]
                    {
                        assert_eq!(allocations.finish().observations, 0);
                        let traffic = traffic.finish();
                        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
                        assert_eq!(dispatch.finish().encoded_dispatches, 1);
                    }
                    let row = read(&engine, cavity.output_buffer());
                    assert_eq!(row[3], 1.);
                    let force = tire.cavity_radial_force_per_bead_n();
                    let expected = [tire.cavity_volume(), force / area, force];
                    for i in 0..3 {
                        let error =
                            (f64::from(row[i]) - expected[i]).abs() / (1. + expected[i].abs());
                        maximum = maximum.max(error);
                        assert!(
                            error <= 2e-5,
                            "node count {n} field {i}: GPU {} CPU {} relative {error}",
                            row[i],
                            expected[i]
                        );
                    }
                    count += 1;
                }
            }
        }
        #[cfg(feature = "resource-counters")]
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        for pressure in [f32::NAN, f32::INFINITY, -1., f32::MAX] {
            assert!(cavity.update_pressure(&engine, pressure).is_err());
        }
        #[cfg(feature = "resource-counters")]
        assert_eq!(traffic.finish().upload_bytes, 0);
        assert!(cavity.update_pressure(&Engine::new_cpu(), 1.).is_err());
        let mut invalid = snapshot.positions.clone();
        invalid[0][0] = f32::NAN;
        engine
            .queue()
            .unwrap()
            .write_buffer(&pos[0], 48, bytemuck::cast_slice(&invalid));
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        cavity.encode(&mut encoder, false);
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(read(&engine, cavity.output_buffer()), [0.; 4]);
        drop(pos);
        drop(hub_buffer);
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        cavity.encode(&mut encoder, true);
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(read(&engine, cavity.output_buffer())[3], 1.);
    }
    eprintln!(
        "resident cavity: {count} real-mesh cases, maximum scaled error {maximum:e}; zero hot buffers/uploads/readbacks"
    );
}
