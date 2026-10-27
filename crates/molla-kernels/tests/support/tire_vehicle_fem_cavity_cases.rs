use super::*;
use molla_kernels::{
    PreparedTireVehicleFemReadback, PreparedTireVehicleFemStep, TireResidentVehicleFemAtlasOwner,
};

pub(super) fn read(engine: &Engine, buffer: &wgpu::Buffer) -> Vec<u8> {
    let device = engine.device().unwrap();
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &stage, 0, buffer.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    stage
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = stage.slice(..).get_mapped_range().to_vec();
    stage.unmap();
    bytes
}

fn snapshot(
    engine: &Engine,
    kernel: &TireFemGpuKernel,
    reader: &mut PreparedTireVehicleFemReadback,
    step: &PreparedTireVehicleFemStep,
    owner: &TireResidentVehicleFemAtlasOwner,
) -> Vec<u8> {
    let frame = reader.read_blocking(engine, step, owner).unwrap();
    let mut bytes = Vec::new();
    for i in 0..2 {
        bytes.extend_from_slice(bytemuck::cast_slice(frame.positions(i).unwrap()));
        bytes.extend_from_slice(bytemuck::cast_slice(frame.velocities(i).unwrap()));
    }
    for rows in [
        frame.hub_rows(),
        frame.planar_rows(),
        frame.spin_rows(),
        frame.suspension_rows(),
    ] {
        bytes.extend_from_slice(bytemuck::cast_slice(rows));
    }
    bytes.extend_from_slice(bytemuck::bytes_of(&frame.vertical_row()));
    for state in kernel
        .readback_resident_atlas_batch(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            owner.fem(),
        )
        .unwrap()
    {
        bytes.extend_from_slice(bytemuck::cast_slice(&state.forces));
    }
    bytes
}

fn raw_snapshot(
    engine: &Engine,
    kernel: &TireFemGpuKernel,
    owner: &TireResidentVehicleFemAtlasOwner,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    for state in kernel
        .readback_resident_atlas_batch(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            owner.fem(),
        )
        .unwrap()
    {
        for rows in [&state.positions, &state.velocities, &state.forces] {
            bytes.extend_from_slice(bytemuck::cast_slice(rows));
        }
    }
    let vehicle = owner.vehicle();
    for buffer in [
        owner.fem().hub_state_buffer(),
        vehicle.planar.current_state_buffer(),
        vehicle.vertical.current_state_buffer(),
        vehicle.brush.spin.current_state_buffer(),
        vehicle.brush.suspension.rows_buffer(),
    ] {
        bytes.extend_from_slice(&read(engine, buffer));
    }
    bytes
}

pub(super) fn run(
    engine: &Engine,
    fem: &TireFemGpuKernel,
    vehicle: &TireVehicleYawGpuKernel,
    brush: &TireBrushGpuKernel,
    build: impl Fn() -> TireResidentVehicleFemAtlasOwner,
    base_params: &[TireFemGpuParams],
    hub_params: &[TireFemHubParams],
) {
    let mut control = build();
    let mut candidate = build();
    let prepare = |owner: &TireResidentVehicleFemAtlasOwner| {
        PreparedTireVehicleFemStep::new(
            engine,
            fem,
            vehicle,
            brush,
            owner,
            None,
            1. / 600.,
            base_params,
            hub_params,
            1.,
            4096,
        )
        .unwrap()
    };
    let mut cold = prepare(&control);
    let mut live = prepare(&candidate);
    let mut stale_read =
        PreparedTireVehicleFemReadback::new(engine, &live, &candidate, 448).unwrap();
    let tube = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let surfaces: Vec<_> = base_params
        .iter()
        .map(|p| molla_vehicle::TetraTireFemCavitySurface {
            tube: &tube,
            negative_cap: &[],
            positive_cap: &[],
            node_count: 4,
            rest_volume_m3: 1. / 6.,
            pressure_pa: p.pressure_pa as f64,
            gamma: 1.4,
            coupling: 0.5,
            width_m: 1.,
            area_per_bead_m2: 1e-5,
        })
        .collect();
    #[cfg(feature = "resource-counters")]
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        live.prepare_cavities(engine, &candidate, &surfaces, 351)
            .is_err()
    );
    #[cfg(feature = "resource-counters")]
    assert_eq!(allocations.finish().observations, 0);
    live.prepare_cavities(engine, &candidate, &surfaces, 352)
        .unwrap();
    assert!(stale_read.read_blocking(engine, &live, &candidate).is_err());
    #[cfg(feature = "resource-counters")]
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(PreparedTireVehicleFemReadback::new(engine, &live, &candidate, 463).is_err());
    #[cfg(feature = "resource-counters")]
    assert_eq!(allocations.finish().observations, 0);
    assert!(
        live.prepare_cavities(engine, &candidate, &surfaces, 352)
            .is_err()
    );
    assert!(live.step(engine, &mut candidate).is_err());
    assert!(live.step_cavity_frame(engine, &mut control, 1).is_err());
    assert!(live.step_cavity_frame(engine, &mut candidate, 0).is_err());
    let mut cold_read = PreparedTireVehicleFemReadback::new(engine, &cold, &control, 1024).unwrap();
    let mut live_read =
        PreparedTireVehicleFemReadback::new(engine, &live, &candidate, 464).unwrap();
    assert!(live_read.read_blocking(engine, &live, &candidate).is_err());
    let mut params = base_params.to_vec();
    for (frame, substeps) in [1, 2, 3, 4].into_iter().enumerate() {
        for p in &mut params {
            p.pressure_pa += 100.;
        }
        live.update_configuration(engine, &candidate, &params, hub_params, 1.)
            .unwrap();
        #[cfg(feature = "resource-counters")]
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        #[cfg(feature = "resource-counters")]
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        #[cfg(feature = "resource-counters")]
        let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
        let output = live
            .step_cavity_frame(engine, &mut candidate, substeps)
            .unwrap();
        #[cfg(feature = "resource-counters")]
        {
            assert_eq!(allocations.finish().observations, 0);
            let traffic = traffic.finish();
            assert_eq!(traffic.upload_bytes, 0);
            assert_eq!(traffic.readback_bytes, 0);
            assert_eq!(
                dispatch.finish().encoded_dispatches,
                3 + 9 * u64::from(substeps)
            );
        }
        let status: Vec<u32> = read(engine, live.cavity_status_buffer().unwrap())
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(&status[..4], &[0, 1, frame as u32 + 1, 0]);
        #[cfg(feature = "resource-counters")]
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        #[cfg(feature = "resource-counters")]
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let observed = live_read.read_blocking(engine, &live, &candidate).unwrap();
        assert_eq!(observed.cavity_status(), Some([0, 1, frame as u32 + 1, 0]));
        #[cfg(feature = "resource-counters")]
        {
            assert_eq!(allocations.finish().observations, 0);
            let traffic = traffic.finish();
            assert_eq!(traffic.upload_bytes, 0);
            assert_eq!(traffic.readback_bytes, 464);
        }
        let records = read(engine, live.cavity_records_buffer().unwrap());
        let mut reference = params.clone();
        for (i, p) in reference.iter_mut().enumerate() {
            let row: [f32; 4] = bytemuck::pod_read_unaligned(&records[16 * i..16 * i + 16]);
            assert_eq!(row[3], 1.);
            assert!(row[2] > 0.);
            p.cavity_radial_force_per_bead = row[2];
        }
        cold.update_configuration(engine, &control, &reference, hub_params, 1.)
            .unwrap();
        let mut control_output = None;
        for _ in 0..substeps {
            control_output = Some(cold.step(engine, &mut control).unwrap());
        }
        assert_eq!(
            snapshot(engine, fem, &mut live_read, &live, &candidate),
            snapshot(engine, fem, &mut cold_read, &cold, &control),
            "frame {frame} all physical states"
        );
        assert_eq!(
            read(engine, &output.buffer),
            read(engine, &control_output.unwrap().buffer)
        );
    }
    let hub = candidate.fem().hub_state_buffer();
    let original = read(engine, hub);
    engine
        .queue()
        .unwrap()
        .write_buffer(hub, 32, bytemuck::bytes_of(&f32::NAN));
    let before = raw_snapshot(engine, fem, &candidate);
    for substeps in [1, 2, 3] {
        live.step_cavity_frame(engine, &mut candidate, substeps)
            .unwrap();
        assert_eq!(
            raw_snapshot(engine, fem, &candidate),
            before,
            "invalid cavity preserves all current physical states across parity flips"
        );
        let status = read(engine, live.cavity_status_buffer().unwrap());
        assert_eq!(&status[..16], bytemuck::cast_slice(&[1u32, 0, 4, 0]));
        assert!(live_read.read_blocking(engine, &live, &candidate).is_err());
    }
    engine
        .queue()
        .unwrap()
        .write_buffer(candidate.fem().hub_state_buffer(), 0, &original);
    let restored = raw_snapshot(engine, fem, &candidate);
    live.step_cavity_frame(engine, &mut candidate, 3).unwrap();
    assert_eq!(
        raw_snapshot(engine, fem, &candidate),
        restored,
        "cavity fault stays latched after source repair"
    );
    assert!(live_read.read_blocking(engine, &live, &candidate).is_err());
    let mut replacement = prepare(&candidate);
    replacement
        .prepare_cavities(engine, &candidate, &surfaces, 352)
        .unwrap();
    assert!(
        live_read
            .read_blocking(engine, &replacement, &candidate)
            .is_err()
    );
    assert!(
        live.prepare_cavities(engine, &candidate, &surfaces, 352)
            .is_err()
    );
    println!(
        "cavity frame: four changing-pressure frames/ten substeps exact against held GPU loads; invalid second wheel stops all nine stages; repaired input stays faulted; no hot GPU allocations/readback"
    );
}
