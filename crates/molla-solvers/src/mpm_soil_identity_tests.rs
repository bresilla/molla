use super::super::{checkpoint::tests::fixture, replay_tests::read_words};
use super::*;
use molla_vehicle::SoilTileLayout;
use std::time::{Duration, Instant};

fn layout() -> SoilTileLayout {
    SoilTileLayout::new([0.0; 3], 0.125).unwrap()
}

fn selection(count: u32, tag: u32) -> MpmSoilSelection {
    let ids: Vec<_> = (0..count)
        .map(|i| {
            SoilParticleId(if i + 1 == count {
                u64::MAX
            } else {
                (u64::from(i) << 40) | u64::from(tag)
            })
        })
        .collect();
    let materials = [
        SoilMaterialId(u64::MAX),
        SoilMaterialId(1 << 62),
        SoilMaterialId(u64::from(tag)),
    ];
    let archive = fixture(count)
        .partition_tiles(layout(), &ids, &materials)
        .unwrap();
    archive
        .select(&archive.tiles().collect::<Vec<_>>(), count)
        .unwrap()
}

fn packed(ids: impl Iterator<Item = u64>) -> Vec<u32> {
    ids.flat_map(|id| [id as u32, (id >> 32) as u32]).collect()
}

fn check_device_ids(engine: &Engine, sim: &MpmSimulation) {
    let view = sim.soil_identity_buffers().unwrap();
    let ids = sim.soil_identity_snapshot().unwrap();
    let p = read_words(
        engine.device().unwrap(),
        engine.queue().unwrap(),
        view.particles,
    );
    let m = read_words(
        engine.device().unwrap(),
        engine.queue().unwrap(),
        view.materials,
    );
    assert_eq!(
        &p[..view.particle_count as usize * 2],
        packed(ids.particle_ids().iter().map(|id| id.0))
    );
    assert_eq!(
        &m[..view.material_count as usize * 2],
        packed(ids.material_ids().iter().map(|id| id.0))
    );
}

fn shader_ids(engine: &Engine, sim: &MpmSimulation) -> Vec<u32> {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let ids = sim.soil_identity_buffers().unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("soil identity ABI probe"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var<storage, read> particles: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> materials: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> local_material: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_ids: array<vec4<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < arrayLength(&output_ids) {
        output_ids[gid.x] = vec4<u32>(particles[gid.x], materials[local_material[gid.x]]);
    }
}
"#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("soil identity ABI probe"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("soil identity ABI results"),
        size: u64::from(ids.particle_count) * 16,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("soil identity ABI probe"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            buffer_entry(0, ids.particles),
            buffer_entry(1, ids.materials),
            buffer_entry(2, &sim.p_material),
            buffer_entry(3, &output),
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(ids.particle_count.div_ceil(64), 1, 1);
    }
    queue.submit([encoder.finish()]);
    read_words(device, queue, &output)
}

#[test]
fn identity_snapshots_and_views_are_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<MpmSoilIdentitySnapshot>();
    check::<MpmSoilIdentityBuffers<'static>>();
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn identified_restore_gpu_abi_step_ordering_and_retained_resources_preserve_ids() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for sparse in [false, true] {
        let initial = selection(129, 0);
        let mut sim = if sparse {
            MpmSimulation::new_sparse(&engine, initial.checkpoint().config(), 128).unwrap()
        } else {
            MpmSimulation::new(&engine, initial.checkpoint().config()).unwrap()
        };
        assert_eq!(sim.soil_identity_buffer_bytes(), 0);
        assert!(sim.soil_identity_buffers().is_none());
        let allocations = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        sim.restore_soil_selection(&engine, &initial).unwrap();
        assert_eq!(allocations.finish().observations, 4);
        let transfers = transfers.finish();
        assert_eq!(
            (transfers.upload_bytes, transfers.readback_bytes),
            (129 * 168 + 3 * 8, 0)
        );
        let retained = sim.soil_identity_buffer_bytes();
        assert_eq!(retained, 129 * 8 + MAX_MPM_MATERIALS as u64 * 8);
        let view = sim.soil_identity_buffers().unwrap();
        let buffers = [view.particles.clone(), view.materials.clone()];
        let snapshot = sim.soil_identity_snapshot().unwrap().clone();
        check_device_ids(&engine, &sim);
        let expected: Vec<_> = initial
            .particle_ids()
            .iter()
            .enumerate()
            .flat_map(|(i, id)| {
                let material = initial.material_ids()[i % 3].0;
                [
                    id.0 as u32,
                    (id.0 >> 32) as u32,
                    material as u32,
                    (material >> 32) as u32,
                ]
            })
            .collect();
        assert_eq!(shader_ids(&engine, &sim), expected);
        if sparse {
            sim.prepare_parallel_particle_tile_ordering(queue, 129)
                .unwrap();
            sim.set_particle_tile_ordering(true).unwrap();
            sim.reserve_sparse_tiles(256).unwrap();
        }
        let allocations = DeviceScope::start().unwrap();
        sim.step_n_with(device, queue, 0.0001, 3).unwrap();
        assert_eq!(allocations.finish().observations, 0);
        assert!(sim.is_soil_identity_current(&snapshot));
        assert_eq!(shader_ids(&engine, &sim), expected);
        let restored = selection(129, 17);
        let allocations = DeviceScope::start().unwrap();
        sim.restore_soil_selection(&engine, &restored).unwrap();
        assert_eq!(allocations.finish().observations, 2);
        assert!(!sim.is_soil_identity_current(&snapshot));
        let current = sim.soil_identity_buffers().unwrap();
        assert_eq!(
            (&buffers[0], &buffers[1]),
            (current.particles, current.materials)
        );
        check_device_ids(&engine, &sim);
        assert_eq!(
            sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
            restored.checkpoint().to_bytes()
        );
        println!(
            "sparse={sparse}: u64 IDs consumed by WGSL vec2 ABI before/after three substeps; retained {retained} bytes, repeat restore reuses ID buffers"
        );
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn invalid_restore_is_atomic_and_identity_epochs_follow_reseed_and_material_changes() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let selected = selection(1, 5);
    let mut cfg = selected.checkpoint().config();
    cfg.max_particles = 1;
    let mut sim = MpmSimulation::new(&engine, cfg).unwrap();
    sim.restore_soil_selection(&engine, &selected).unwrap();
    let snapshot = sim.soil_identity_snapshot().unwrap().clone();
    let too_many = selection(129, 5);
    let resources = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    assert!(sim.restore_soil_selection(&engine, &too_many).is_err());
    assert!(sim.restore_soil_selection(&foreign, &selected).is_err());
    assert_eq!(resources.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
    assert!(sim.is_soil_identity_current(&snapshot));
    check_device_ids(&engine, &sim);
    sim.set_particles(queue, &[]).unwrap();
    assert!(!sim.is_soil_identity_current(&snapshot));
    assert!(sim.soil_identity_buffers().is_none());
    let empty = selection(0, 5);
    sim.restore_soil_selection(&engine, &empty).unwrap();
    assert_eq!(sim.soil_identity_buffers().unwrap().particle_count, 0);
    check_device_ids(&engine, &sim);
    sim.restore_soil_selection(&engine, &selected).unwrap();
    let snapshot = sim.soil_identity_snapshot().unwrap().clone();
    sim.set_materials(selected.checkpoint().materials(), vec![0])
        .unwrap();
    assert!(!sim.is_soil_identity_current(&snapshot));
    sim.restore_soil_selection(&engine, &selected).unwrap();
    let snapshot = sim.soil_identity_snapshot().unwrap().clone();
    sim.restore_soil_with(queue, selected.checkpoint()).unwrap();
    assert!(!sim.is_soil_identity_current(&snapshot));
    assert!(sim.soil_identity_snapshot().is_none());
    assert!(sim.soil_identity_buffer_bytes() > 0);
}

fn next(ring: &mut MpmSoilCheckpointReadback) -> MpmSoilCheckpointCompletion {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(done) = ring.poll_next().unwrap() {
            return done;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn asynchronous_archive_keeps_request_identity_across_restore_and_owner_drop() {
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let a = selection(129, 3);
    let b = selection(1, 9);
    let mut sim = MpmSimulation::new_sparse(&engine, a.checkpoint().config(), 128).unwrap();
    sim.restore_soil_selection(&engine, &a).unwrap();
    let mut ring = sim
        .prepare_soil_checkpoint_readback(&engine, 129, 3, 3 * (129 * 160 + 16))
        .unwrap();
    ring.try_capture(&sim, 0).unwrap().unwrap();
    sim.restore_soil_selection(&engine, &b).unwrap();
    ring.try_capture(&sim, 1).unwrap().unwrap();
    sim.restore_soil_with(queue, b.checkpoint()).unwrap();
    ring.try_capture(&sim, 2).unwrap().unwrap();
    drop(sim);
    for (revision, expected) in [(0, &a), (1, &b)] {
        let done = next(&mut ring);
        assert_eq!(done.ticket().revision(), revision);
        assert_eq!(
            done.identity().unwrap().particle_ids(),
            expected.particle_ids()
        );
        assert_eq!(
            done.identity().unwrap().material_ids(),
            expected.material_ids()
        );
        let archive = done.into_tile_archive(layout()).unwrap();
        let decoded = MpmSoilTileArchive::from_bytes(&archive.to_bytes(), 129).unwrap();
        let observed = decoded
            .select(&decoded.tiles().collect::<Vec<_>>(), 129)
            .unwrap();
        assert_eq!(observed.particle_ids(), expected.particle_ids());
        assert_eq!(observed.material_ids(), expected.material_ids());
        assert_eq!(
            observed.checkpoint().to_bytes(),
            expected.checkpoint().to_bytes()
        );
    }
    let done = next(&mut ring);
    assert!(done.identity().is_none());
    assert!(done.into_tile_archive(layout()).is_err());
    println!(
        "two identified captures survive different restores/source drop; identity-free capture refuses to invent global IDs"
    );
}
