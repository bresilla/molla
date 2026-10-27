use super::*;
use crate::normal_contact_gpu::NormalContactRow;
use std::sync::{Arc, OnceLock};

fn engine() -> Arc<Engine> {
    static ENGINE: OnceLock<Arc<Engine>> = OnceLock::new();
    ENGINE
        .get_or_init(|| Arc::new(Engine::new_gpu().unwrap()))
        .clone()
}

fn read(engine: &Engine, buffer: &wgpu::Buffer) -> Vec<u8> {
    let device = engine.device().unwrap();
    let staging = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("activation test readback"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let result = molla_core::gpu_readback!(staging.slice(..)).to_vec();
    staging.unmap();
    result
}

fn words(engine: &Engine, buffer: &wgpu::Buffer) -> Vec<u32> {
    read(engine, buffer)
        .chunks_exact(4)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

struct Fixture {
    engine: Arc<Engine>,
    core: ContactActivationGpu,
    rows: wgpu::Buffer,
    impacts: wgpu::Buffer,
    onsets: wgpu::Buffer,
    status: wgpu::Buffer,
    velocities: wgpu::Buffer,
    accelerations: wgpu::Buffer,
    initial: Vec<NormalContactRow>,
}

impl Fixture {
    fn new() -> Self {
        Self::with_identities(None)
    }

    fn with_identities(ids: Option<&[u32]>) -> Self {
        let engine = engine();
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut initial = vec![NormalContactRow::default(); 8];
        for (i, (a, b)) in [(5, 6), (4, 5), (3, 4), (2, 3), (1, 2), (0, 1), (7, 8)]
            .into_iter()
            .enumerate()
        {
            initial[i] = NormalContactRow {
                point_depth: [0., 0., 0., 0.001],
                normal_force: [0., 1., 0., 2.],
                coefficients: [0.6, 0.3, 0.4, 1.],
                endpoints: [
                    if i == 6 { 42 } else { 2 * i as u32 },
                    if i == 6 { 43 } else { 2 * i as u32 + 1 },
                    a,
                    b,
                ],
                force_on_b: [0., 2., 0., 0.],
                ..Default::default()
            };
        }
        let rows = storage(device, queue, 0, bytemuck::cast_slice(&initial));
        let impacts = storage(device, queue, 0, bytemuck::cast_slice(&initial));
        let onsets = storage(
            device,
            queue,
            0,
            bytemuck::cast_slice(&[1u32, 1, 1, 1, 1, 1, 1, 0]),
        );
        let status = storage(device, queue, 4, &[]);
        let velocities = storage(device, queue, 9 * 32, &[]);
        let accelerations = storage(device, queue, 9 * 32, &[]);
        let identities = ids.map(|ids| storage(device, queue, 0, bytemuck::cast_slice(ids)));
        let core = ContactActivationGpu::new(
            &engine,
            &[0, 1, 2, 3, 4, 5, 6, 7, NO_GROUP],
            8,
            8,
            ActivationBuffers {
                identities: identities.as_ref(),
                rows: &rows,
                impacts: &impacts,
                onsets: &onsets,
                status: &status,
                velocities: &[&velocities],
                accelerations: &accelerations,
            },
        )
        .unwrap();
        Self {
            engine,
            core,
            rows,
            impacts,
            onsets,
            status,
            velocities,
            accelerations,
            initial,
        }
    }
    fn restore_rows(&self) {
        let queue = self.engine.queue().unwrap();
        molla_core::gpu_write_buffer!(queue, &self.rows, 0, bytemuck::cast_slice(&self.initial));
        molla_core::gpu_write_buffer!(queue, &self.impacts, 0, bytemuck::cast_slice(&self.initial));
        molla_core::gpu_write_buffer!(
            queue,
            &self.onsets,
            0,
            bytemuck::cast_slice(&[1u32, 1, 1, 1, 1, 1, 1, 0]),
        );
        molla_core::gpu_write_buffer!(queue, &self.status, 0, bytemuck::bytes_of(&0u32));
    }
    fn dispatch(&self) {
        let mut encoder = self
            .engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        self.core.encode(&mut encoder, 0).unwrap();
        self.engine.queue().unwrap().submit([encoder.finish()]);
    }
}

#[test]
#[ignore = "requires GPU; run via make"]
fn watched_colliders_retain_support_across_compiled_pieces() {
    let mut identities: Vec<u32> = (1..=46).collect();
    identities[42] = 102;
    identities[43] = 103;
    identities[44] = 102;
    identities[45] = 103;
    let mut f = Fixture::with_identities(Some(&identities));
    for switched_piece in [false, true] {
        if switched_piece {
            f.initial[6].endpoints[..2].copy_from_slice(&[44, 45]);
        }
        f.restore_rows();
        f.core
            .prepare(f.engine.queue().unwrap(), &[0; 8], &[], &[[7, 102, 103, 0]])
            .unwrap();
        f.dispatch();
        assert_eq!(words(&f.engine, &f.status), [0]);
        assert_eq!(words(&f.engine, f.core.group_mask()), [0; 8]);
    }
    f.initial[6].coefficients[3] = 0.;
    f.restore_rows();
    f.core
        .prepare(f.engine.queue().unwrap(), &[0; 8], &[], &[[7, 102, 103, 0]])
        .unwrap();
    f.dispatch();
    assert_eq!(words(&f.engine, &f.status), [0]);
    assert_eq!(
        words(&f.engine, f.core.group_mask()),
        [0, 0, 0, 0, 0, 0, 0, 1]
    );
}

#[test]
#[ignore = "requires GPU; run via make"]
fn contact_components_wake_in_one_dispatch_and_dormant_rows_retire() {
    let f = Fixture::new();
    f.core
        .prepare(
            f.engine.queue().unwrap(),
            &[1, 0, 0, 0, 0, 0, 0, 0],
            &[],
            &[[7, 42, 43, 0]],
        )
        .unwrap();
    let output = storage(
        f.engine.device().unwrap(),
        f.engine.queue().unwrap(),
        12,
        &[],
    );
    let selected = f
        .core
        .selection(&f.engine, &[0, 7, NO_GROUP], &output)
        .unwrap();
    let mut encoder = f
        .engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    f.core.encode(&mut encoder, 0).unwrap();
    f.core.encode_selection(&mut encoder, &selected);
    f.engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(words(&f.engine, &f.status), vec![0]);
    assert_eq!(
        words(&f.engine, f.core.group_mask()),
        vec![1, 1, 1, 1, 1, 1, 1, 0]
    );
    assert_eq!(
        words(&f.engine, f.core.body_mask()),
        vec![1, 1, 1, 1, 1, 1, 1, 0, 1]
    );
    assert_eq!(words(&f.engine, &output), vec![1, 0, 1]);
    let bytes = read(&f.engine, &f.rows);
    let rows: Vec<NormalContactRow> = bytes
        .chunks_exact(128)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    assert_eq!(
        bytemuck::bytes_of(&rows[0]),
        bytemuck::bytes_of(&f.initial[0])
    );
    assert_eq!(rows[6].coefficients[3], 0.);
    assert_eq!(rows[6].normal_force[3], 0.);
    assert_eq!(rows[6].force_on_b, [0., 0., 0., -1.]);
    assert_eq!(rows[6].endpoints, f.initial[6].endpoints);
    assert_eq!(rows[6].point_depth, f.initial[6].point_depth);
    let bytes = read(&f.engine, &f.impacts);
    let impacts: Vec<NormalContactRow> = bytes
        .chunks_exact(128)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    assert_eq!(impacts[6].coefficients[3], 0.);
    assert_eq!(impacts[6].force_on_b, [0.; 4]);
    assert_eq!(words(&f.engine, &f.onsets), vec![1, 1, 1, 1, 1, 1, 0, 0]);
}

#[test]
#[ignore = "requires GPU; run via make"]
fn support_loss_motion_surface_and_sleep_dependencies_wake_components() {
    let mut f = Fixture::new();
    for mode in 0..5 {
        f.initial[6].surface_velocity = if mode == 3 {
            [0.2, 0., 0., 0.]
        } else {
            [0.; 4]
        };
        f.restore_rows();
        molla_core::gpu_write_buffer!(f.engine.queue().unwrap(), &f.velocities, 0, &[0; 9 * 32]);
        molla_core::gpu_write_buffer!(f.engine.queue().unwrap(), &f.accelerations, 0, &[0; 9 * 32]);
        let mut awake = [0; 8];
        awake[0] = 1;
        let edges = if mode == 4 { vec![[0, 7]] } else { Vec::new() };
        let watches = if mode == 0 {
            vec![[7, 99, 100, 0]]
        } else {
            vec![[7, 43, 42, 0]]
        };
        if mode == 1 {
            molla_core::gpu_write_buffer!(
                f.engine.queue().unwrap(),
                &f.velocities,
                8 * 32,
                bytemuck::bytes_of(&[1f32, 0., 0., 0.]),
            );
        }
        if mode == 2 {
            molla_core::gpu_write_buffer!(
                f.engine.queue().unwrap(),
                &f.accelerations,
                8 * 32,
                bytemuck::bytes_of(&[0f32, 1., 0., 0.]),
            );
        }
        f.core
            .prepare(f.engine.queue().unwrap(), &awake, &edges, &watches)
            .unwrap();
        f.dispatch();
        assert_eq!(words(&f.engine, &f.status), vec![0], "mode={mode}");
        assert_eq!(
            words(&f.engine, f.core.group_mask()),
            vec![1; 8],
            "mode={mode}"
        );
        let bytes = read(&f.engine, &f.rows);
        let row = bytemuck::pod_read_unaligned::<NormalContactRow>(&bytes[6 * 128..7 * 128]);
        assert_eq!(row.coefficients[3], 1.);
        assert_eq!(row.normal_force[3], 2.);
    }
}

#[test]
#[ignore = "requires GPU; run via make"]
fn activation_faults_guard_publication_and_recover_without_replacing_masks() {
    let mut f = Fixture::new();
    f.core
        .prepare(f.engine.queue().unwrap(), &[1; 8], &[], &[])
        .unwrap();
    f.dispatch();
    let retained = f.core.body_mask().clone();
    assert_eq!(words(&f.engine, &retained), vec![1; 9]);
    assert!(
        f.core
            .prepare(f.engine.queue().unwrap(), &[2; 8], &[], &[])
            .is_err()
    );
    assert!(
        f.core
            .prepare(f.engine.queue().unwrap(), &[0; 8], &[[0, 8]], &[])
            .is_err()
    );
    assert!(
        f.core
            .prepare(f.engine.queue().unwrap(), &[0; 8], &[], &[[8, 42, 43, 0]])
            .is_err()
    );
    let output = storage(
        f.engine.device().unwrap(),
        f.engine.queue().unwrap(),
        4,
        &[],
    );
    assert!(f.core.selection(&f.engine, &[8], &output).is_err());
    f.initial[0].endpoints[2] = 99;
    f.restore_rows();
    f.core
        .prepare(f.engine.queue().unwrap(), &[0; 8], &[], &[])
        .unwrap();
    f.dispatch();
    assert_eq!(words(&f.engine, &f.status), vec![1 << 20]);
    assert_eq!(words(&f.engine, &retained), vec![1; 9]);
    f.initial[0].endpoints[2] = 5;
    f.restore_rows();
    f.core
        .prepare(f.engine.queue().unwrap(), &[0; 8], &[], &[])
        .unwrap();
    f.dispatch();
    assert_eq!(words(&f.engine, &f.status), vec![0]);
    assert_eq!(words(&f.engine, &retained), vec![0, 0, 0, 0, 0, 0, 0, 0, 1]);
    assert_eq!(f.core.body_mask(), &retained);
}

#[test]
#[ignore = "requires GPU; run via make"]
fn boundary_only_layout_has_no_dynamic_group_or_response() {
    boundary_layout(&[NO_GROUP]);
}

#[test]
#[ignore = "requires GPU; run via make"]
fn empty_body_layout_binds_complete_dependency_elements() {
    boundary_layout(&[]);
}

fn boundary_layout(body_groups: &[u32]) {
    let engine = engine();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let row = NormalContactRow::default();
    let rows = storage(device, queue, 128, bytemuck::bytes_of(&row));
    let impacts = storage(device, queue, 128, bytemuck::bytes_of(&row));
    let onsets = storage(device, queue, 4, &[]);
    let status = storage(device, queue, 4, &[]);
    let velocities = storage(device, queue, 32, &[]);
    let accelerations = storage(device, queue, 32, &[]);
    let core = ContactActivationGpu::new(
        &engine,
        body_groups,
        0,
        1,
        ActivationBuffers {
            identities: None,
            rows: &rows,
            impacts: &impacts,
            onsets: &onsets,
            status: &status,
            velocities: &[&velocities],
            accelerations: &accelerations,
        },
    )
    .unwrap();
    core.prepare(queue, &[], &[], &[]).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(core.encode(&mut encoder, 1).is_err());
    core.encode(&mut encoder, 0).unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(words(&engine, &status), vec![0]);
    assert_eq!(core.dependencies.size(), 8);
    assert_eq!(
        words(&engine, core.body_mask()),
        vec![u32::from(!body_groups.is_empty())]
    );
    assert_eq!(read(&engine, &rows), bytemuck::bytes_of(&row));
}
