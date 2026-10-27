#![cfg(feature = "gpu")]

use molla_kernels::{TireBekkerContactCell, TireBekkerFemTerrainOwner};

fn owner(nx: u32, nz: u32) -> TireBekkerFemTerrainOwner {
    TireBekkerFemTerrainOwner::flat(
        [-0.5, -0.5],
        0.5,
        nx,
        nz,
        0.0,
        0.55,
        900_000.0,
        0.12,
        10_000.0,
        0.05,
        1.0,
    )
    .unwrap()
}

#[test]
fn soil_uniform_material_maps_track_edits_removal_and_dimensions() {
    for (nx, nz) in [(1, 1), (3, 3), (7, 2)] {
        let mut state = owner(nx, nz);
        let n = (nx * nz) as usize;
        assert!(state.set_uniform_friction_map(0.55).unwrap());
        assert!(state.set_uniform_water_film_map(0.002).unwrap());
        for _ in 0..8 {
            assert!(!state.set_uniform_friction_map(0.55).unwrap());
            assert!(!state.set_uniform_water_film_map(0.002).unwrap());
        }
        assert!(state.set_friction_map(Some(vec![0.1; n + 1])).is_err());
        assert!(state.set_uniform_friction_map(f32::NAN).is_err());
        assert!(state.set_uniform_water_film_map(f32::INFINITY).is_err());
        assert!(!state.set_uniform_friction_map(0.55).unwrap());
        assert!(!state.set_uniform_water_film_map(0.002).unwrap());
        state.set_water_film_map(Some(vec![0.003; n])).unwrap();
        assert!(!state.set_uniform_water_film_map(0.003).unwrap());
        let mut spatial = vec![0.55; n];
        spatial[0] = 0.3;
        state.set_friction_map(Some(spatial)).unwrap();
        assert!(state.set_uniform_friction_map(0.55).unwrap());
        state.set_water_film_map(None).unwrap();
        assert!(state.fem_contact_terrain().water_film_map.is_none());
        assert!(state.set_uniform_water_film_map(0.0).unwrap());
        let view = state.fem_contact_terrain();
        assert_eq!(view.friction_map.unwrap(), vec![0.55; n]);
        assert_eq!(view.water_film_map.unwrap(), vec![0.0; n]);
        let contacts = [TireBekkerContactCell {
            xz_pressure: [0.0, 0.0, 90_000.0, 20_000.0],
        }];
        let mut control = owner(nx, nz);
        control.set_friction_map(Some(vec![0.55; n])).unwrap();
        control.set_water_film_map(Some(vec![0.0; n])).unwrap();
        state.apply_contacts_cpu(&contacts).unwrap();
        control.apply_contacts_cpu(&contacts).unwrap();
        assert_eq!(state.heights(), control.heights());
        assert_eq!(state.sinkage(), control.sinkage());
    }
}

#[cfg(feature = "resource-counters")]
mod hardware {
    use super::*;
    use molla_compute::Engine;
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    use molla_kernels::{TireBekkerGpuKernel, TireBekkerGpuResidentTerrain};

    fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<f32> {
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("soil material map readback"),
            size: source.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &stage, 0, source.size());
        queue.submit([encoder.finish()]);
        let slice = stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        stage.unmap();
        values
    }

    fn step(
        owner: &mut TireBekkerFemTerrainOwner,
        engine: &Engine,
        kernel: &TireBekkerGpuKernel,
    ) -> (TireBekkerGpuResidentTerrain, u64, u64) {
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let view = owner
            .apply_contacts_gpu_resident(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                kernel,
                &[TireBekkerContactCell {
                    xz_pressure: [0.0, 0.0, 90_000.0, 20_000.0],
                }],
            )
            .unwrap();
        let resources = resources.finish();
        let transfers = transfers.finish();
        assert!(!resources.invalid && !transfers.overflowed);
        (view, resources.observations, transfers.upload_bytes)
    }

    #[test]
    #[ignore = "needs a real GPU adapter"]
    fn soil_material_cache_reuses_unchanged_maps_and_preserves_snapshots() {
        let engine = Engine::new_gpu().expect("GPU required");
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let kernel = TireBekkerGpuKernel::new(device);
        let mut state = owner(3, 3);
        state.set_uniform_friction_map(0.55).unwrap();
        state.set_uniform_water_film_map(0.002).unwrap();
        state
            .set_surface_velocity_map(Some(vec![[0.6, 0.0, 0.2, 0.0]; 9]))
            .unwrap();
        let (first, initial_allocations, initial_uploads) = step(&mut state, &engine, &kernel);
        let (same, steady_allocations, steady_uploads) = step(&mut state, &engine, &kernel);
        assert_eq!(initial_allocations, steady_allocations + 3);
        assert_eq!(initial_uploads, steady_uploads + 9 * (4 + 4 + 16));
        assert_eq!(first.friction_map, same.friction_map);
        assert_eq!(first.water_film_map, same.water_film_map);
        assert_eq!(first.surface_velocity_map, same.surface_velocity_map);
        for _ in 0..8 {
            assert!(!state.set_uniform_friction_map(0.55).unwrap());
            assert!(!state.set_uniform_water_film_map(0.002).unwrap());
            let (view, allocations, uploads) = step(&mut state, &engine, &kernel);
            assert_eq!((allocations, uploads), (steady_allocations, steady_uploads));
            assert_eq!(view.friction_map, first.friction_map);
            assert_eq!(view.water_film_map, first.water_film_map);
        }
        state.set_friction_map(Some(vec![0.55; 9])).unwrap();
        let (equal, allocations, uploads) = step(&mut state, &engine, &kernel);
        assert_eq!(equal.friction_map, first.friction_map);
        assert_eq!((allocations, uploads), (steady_allocations, steady_uploads));
        let mut fork = state.clone();
        fork.set_uniform_friction_map(0.3).unwrap();
        let (changed, allocations, uploads) = step(&mut fork, &engine, &kernel);
        assert_eq!(allocations, steady_allocations + 1);
        assert_eq!(uploads, steady_uploads + 9 * 4);
        assert_ne!(changed.friction_map, first.friction_map);
        assert_eq!(changed.water_film_map, first.water_film_map);
        assert_eq!(
            read(device, queue, changed.friction_map.as_ref().unwrap()),
            vec![0.3; 9]
        );
        assert_eq!(
            read(device, queue, first.friction_map.as_ref().unwrap()),
            vec![0.55; 9]
        );
        assert_eq!(
            read(device, queue, first.water_film_map.as_ref().unwrap()),
            vec![0.002; 9]
        );
        assert_eq!(
            read(device, queue, first.surface_velocity_map.as_ref().unwrap()),
            [0.6, 0.0, 0.2, 0.0].repeat(9)
        );
        fork.set_water_film_map(None).unwrap();
        let (removed, allocations, uploads) = step(&mut fork, &engine, &kernel);
        assert!(removed.water_film_map.is_none());
        assert_eq!((allocations, uploads), (steady_allocations, steady_uploads));
        fork.set_uniform_water_film_map(0.0).unwrap();
        let (dry, allocations, uploads) = step(&mut fork, &engine, &kernel);
        assert_eq!(
            (allocations, uploads),
            (steady_allocations + 1, steady_uploads + 36)
        );
        assert_eq!(
            read(device, queue, dry.water_film_map.as_ref().unwrap()),
            vec![0.0; 9]
        );
        let mut reference = owner(3, 3);
        reference
            .apply_contacts_cpu(&[TireBekkerContactCell {
                xz_pressure: [0.0, 0.0, 90_000.0, 20_000.0],
            }])
            .unwrap();
        for (&a, &b) in state.heights().iter().zip(reference.heights()) {
            assert!((a - b).abs() <= 2e-5);
        }
        println!(
            "unchanged material maps: 0 allocations, 0 uploads; three-map initialization: 3 allocations, 216 bytes; one scalar-map edit: 1 allocation, 36 bytes; solver allocations/uploads excluded from these deltas"
        );
    }

    #[test]
    #[ignore = "needs a real GPU adapter"]
    fn soil_material_cache_invalidates_on_kernel_and_device_replacement() {
        let engine = Engine::new_gpu().expect("GPU required");
        let kernel = TireBekkerGpuKernel::new(engine.device().unwrap());
        let mut state = owner(3, 3);
        state.set_uniform_friction_map(0.55).unwrap();
        let (old, _, _) = step(&mut state, &engine, &kernel);
        let replacement = TireBekkerGpuKernel::new(engine.device().unwrap());
        let (new, _, _) = step(&mut state, &engine, &replacement);
        assert_ne!(old.friction_map, new.friction_map);
        let foreign = Engine::new_gpu().expect("second GPU context required");
        let foreign_kernel = TireBekkerGpuKernel::new(foreign.device().unwrap());
        let (foreign_view, _, _) = step(&mut state, &foreign, &foreign_kernel);
        assert_eq!(
            read(
                foreign.device().unwrap(),
                foreign.queue().unwrap(),
                foreign_view.friction_map.as_ref().unwrap()
            ),
            vec![0.55; 9]
        );
        let (returned, _, _) = step(&mut state, &engine, &kernel);
        assert_eq!(
            read(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                returned.friction_map.as_ref().unwrap()
            ),
            vec![0.55; 9]
        );
        assert_eq!(
            read(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                old.friction_map.as_ref().unwrap()
            ),
            vec![0.55; 9]
        );
    }
}
