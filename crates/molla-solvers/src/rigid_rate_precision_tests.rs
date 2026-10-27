use super::*;
use molla_math::Vec3;
use molla_sim::runtime::{BodyDesc, BodyHandle, RigidScene};

fn add(scene: &mut RigidScene) -> BodyHandle {
    let id = scene.insert_body(BodyDesc::default()).unwrap();
    scene
        .set_body_velocity(
            id,
            SpatialVector::new(
                Vec3::new(1. + 2f64.powi(-30), 2., 3.),
                Vec3::new(4., 5., 6.),
            ),
        )
        .unwrap();
    id
}

#[test]
fn paired_rate_mapping_preserves_unedited_components_and_survivors() {
    let mut scene = RigidScene::new().unwrap();
    let retired = add(&mut scene);
    let survivor = add(&mut scene);
    let old = Layout::new(&scene).unwrap();
    let before = scene.state().clone_host().unwrap();
    scene.remove_body(retired).unwrap();
    let replacement = add(&mut scene);
    let velocity = SpatialVector::new(
        Vec3::new(8. + 2f64.powi(-25), 2., 3.),
        Vec3::new(4., 5., 6.),
    );
    scene.set_body_velocity(survivor, velocity).unwrap();
    let layout = Layout::new(&scene).unwrap();
    let transfers = RateTransfers::remap(
        &layout,
        &old,
        scene.state(),
        before.joint_qd.host().unwrap(),
        before.body_qd.host().unwrap(),
    )
    .unwrap();
    let i = scene.body_index(survivor).unwrap().index();
    let r = layout.dofs[i].start;
    assert_eq!(transfers.rates[r * 2], [0, 0, 8f32.to_bits(), 0]);
    assert_eq!(
        transfers.rates[r * 2 + 1],
        [0, 0, 2f32.powi(-25).to_bits(), 0]
    );
    assert_eq!(transfers.rates[(r + 1) * 2], [14, 1, 0, 0]);
    assert_eq!(
        transfers.velocities[i * 8],
        [0, 0, 2f32.powi(-25).to_bits(), 0]
    );
    assert_eq!(transfers.velocities[i * 8 + 1], [9, 2, 0, 0]);
    let i = scene.body_index(replacement).unwrap().index();
    assert!(
        transfers.rates[layout.dofs[i].start * 2..layout.dofs[i].end * 2]
            .iter()
            .all(|v| v[1] == 0)
    );
    assert!(
        transfers.velocities[i * 8..i * 8 + 8]
            .iter()
            .all(|v| v[1] == 0)
    );
}

#[test]
fn paired_rate_seed_and_readback_validate_before_publication() {
    let mut scene = RigidScene::new().unwrap();
    add(&mut scene);
    let mut state = scene.state().clone_host().unwrap();
    let transfers = RateTransfers::rounded(&state).unwrap();
    let pairs: Vec<u32> = transfers.rates.iter().map(|v| v[2]).collect();
    let low: Vec<u32> = transfers.velocities.iter().map(|v| v[2]).collect();
    let expected = state.body_qd.host().unwrap()[0];
    state.body_qd.host_mut().unwrap()[0].linear.x = 1.;
    super::super::decode_rates(
        &mut state,
        bytemuck::cast_slice(&pairs),
        bytemuck::cast_slice(&low),
    )
    .unwrap();
    assert_eq!(state.body_qd.host().unwrap()[0], expected);
    assert_eq!(state.joint_qd.host().unwrap()[0], expected.linear.x);
    let before = state.clone_host().unwrap();
    let mut invalid = low.clone();
    invalid[6] = f32::NAN.to_bits();
    assert!(
        super::super::decode_rates(
            &mut state,
            bytemuck::cast_slice(&pairs),
            bytemuck::cast_slice(&invalid)
        )
        .is_err()
    );
    assert!(super::super::decode_rates(&mut state, &[], bytemuck::cast_slice(&low)).is_err());
    assert_eq!(
        state.joint_qd.host().unwrap(),
        before.joint_qd.host().unwrap()
    );
    assert_eq!(
        state.body_qd.host().unwrap(),
        before.body_qd.host().unwrap()
    );
    state.body_qd.host_mut().unwrap()[0].angular.z = f64::INFINITY;
    assert!(RateTransfers::rounded(&state).is_err());
    state.body_qd.host_mut().unwrap()[0].angular.z = 0.;
    state.joint_qd.host_mut().unwrap()[5] = f64::MAX;
    assert!(RateTransfers::rounded(&state).is_err());
}

#[test]
fn paired_rate_mapping_honors_signed_zero_edits_and_body_provenance() {
    let mut scene = RigidScene::new().unwrap();
    add(&mut scene);
    let layout = Layout::new(&scene).unwrap();
    let mut before = scene.state().clone_host().unwrap();
    before.joint_qd.host_mut().unwrap()[2] = 0.;
    before.body_qd.host_mut().unwrap()[0].linear.z = 0.;
    let mut after = before.clone_host().unwrap();
    after.joint_qd.host_mut().unwrap()[2] = -0.;
    after.body_qd.host_mut().unwrap()[0].linear.z = -0.;
    let transfers = RateTransfers::remap(
        &layout,
        &layout,
        &after,
        before.joint_qd.host().unwrap(),
        before.body_qd.host().unwrap(),
    )
    .unwrap();
    assert_eq!(transfers.rates[4], [0, 0, (-0f32).to_bits(), 0]);
    assert_eq!(transfers.velocities[2][1], 0);
    let mut other = RigidScene::new().unwrap();
    add(&mut other);
    let other_layout = Layout::new(&other).unwrap();
    let transfers = RateTransfers::remap(
        &other_layout,
        &layout,
        other.state(),
        scene.state().joint_qd.host().unwrap(),
        scene.state().body_qd.host().unwrap(),
    )
    .unwrap();
    assert!(
        transfers
            .rates
            .iter()
            .chain(&transfers.velocities)
            .all(|v| v[1] == 0)
    );
}

fn buffer(engine: &Engine, words: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    engine
        .device()
        .unwrap()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("runtime paired precision fixture"),
            contents: bytemuck::cast_slice(words),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
}

fn words(engine: &Engine, source: &wgpu::Buffer) -> Vec<u32> {
    let device = engine.device().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .unwrap()
        .unwrap();
    let result = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    result
}

fn owner(engine: &Engine, state: &State) -> RuntimeRates {
    let buffers = || {
        [
            buffer(engine, &vec![0; state.joint_qd.len() * 2]),
            buffer(engine, &vec![0; state.body_qd.len() * 8]),
        ]
    };
    RuntimeRates::from_buffers(
        engine,
        [buffers(), buffers()],
        buffer(engine, &vec![0; state.joint_qd.len() * 2]),
        std::array::from_fn(|_| buffer(engine, &vec![0; state.joint_qd.len()])),
    )
    .unwrap()
}

fn prepare(engine: &Engine, owner: &RuntimeRates, layout: &Layout, state: &State, side: usize) {
    let transfers = owner.plan(layout, state).unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    owner
        .prepare_frame(engine, &mut encoder, transfers, side)
        .unwrap();
    engine.queue().unwrap().submit([encoder.finish()]);
}

#[test]
#[ignore = "requires real GPU"]
fn paired_runtime_rates_retain_edit_remap_and_rollback_on_device() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut scene = RigidScene::new().unwrap();
    let retired = add(&mut scene);
    let survivor = add(&mut scene);
    let layout = Layout::new(&scene).unwrap();
    let mut retained = owner(&engine, scene.state());
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let bad = RuntimeRates::from_buffers(
        &engine,
        [retained.states[0].clone(), retained.states[0].clone()],
        retained.delta.clone(),
        retained.scalar_rates.clone(),
    );
    assert!(bad.is_err());
    assert_eq!(allocations.finish().observations, 0);
    for mirrors in [
        [
            retained.scalar_rates[0].clone(),
            retained.scalar_rates[0].clone(),
        ],
        [
            retained.states[0][0].clone(),
            retained.scalar_rates[1].clone(),
        ],
    ] {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        assert!(
            RuntimeRates::from_buffers(
                &engine,
                retained.states.clone(),
                retained.delta.clone(),
                mirrors
            )
            .is_err()
        );
        assert_eq!(allocations.finish().observations, 0);
    }
    prepare(&engine, &retained, &layout, scene.state(), 0);
    let seed = RateTransfers::rounded(scene.state()).unwrap();
    let initial = [
        seed.rates.iter().map(|v| v[2]).collect::<Vec<_>>(),
        seed.velocities.iter().map(|v| v[2]).collect(),
    ];
    for state in &retained.states {
        for lane in 0..2 {
            assert_eq!(words(&engine, &state[lane]), initial[lane]);
        }
    }
    retained.committed(scene.state()).unwrap();
    let hidden = 2f32.powi(-80).to_bits();
    let mut predicted = initial.clone();
    predicted[0][15] = hidden;
    predicted[1][9] = hidden;
    let delta = vec![2f32.powi(-32).to_bits(); scene.state().joint_qd.len() * 2];
    for lane in 0..2 {
        queue.write_buffer(
            &retained.states[1][lane],
            0,
            bytemuck::cast_slice(&predicted[lane]),
        );
    }
    queue.write_buffer(&retained.delta, 0, bytemuck::cast_slice(&delta));
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    prepare(&engine, &retained, &layout, scene.state(), 1);
    assert_eq!(allocations.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
    for lane in 0..2 {
        assert_eq!(words(&engine, &retained.states[0][lane]), predicted[lane]);
    }
    scene
        .set_body_velocity(
            survivor,
            SpatialVector::new(
                Vec3::new(8. + 2f64.powi(-25), 2., 3.),
                Vec3::new(4., 5., 6.),
            ),
        )
        .unwrap();
    prepare(&engine, &retained, &layout, scene.state(), 1);
    retained.committed(scene.state()).unwrap();
    let edited = [
        words(&engine, &retained.states[0][0]),
        words(&engine, &retained.states[0][1]),
    ];
    assert_eq!(edited[0][12], 8f32.to_bits());
    assert_eq!(edited[0][13], 2f32.powi(-25).to_bits());
    assert_eq!(edited[0][15], hidden);
    assert_eq!(edited[1][9], hidden);
    for state in &retained.states {
        for b in state {
            queue.write_buffer(b, 0, &vec![0; b.size() as usize]);
        }
    }
    queue.write_buffer(&retained.delta, 0, &vec![0; retained.delta.size() as usize]);
    for scalar in &retained.scalar_rates {
        queue.write_buffer(scalar, 0, &vec![0xff; scalar.size() as usize]);
    }
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    retained.rollback(&mut encoder);
    queue.submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
    for state in &retained.states {
        for lane in 0..2 {
            assert_eq!(words(&engine, &state[lane]), edited[lane]);
        }
    }
    assert_eq!(words(&engine, &retained.delta), delta);
    for scalar in &retained.scalar_rates {
        assert_eq!(
            words(&engine, scalar),
            edited[0].iter().step_by(2).copied().collect::<Vec<_>>()
        );
    }
    scene.remove_body(retired).unwrap();
    add(&mut scene);
    let new_layout = Layout::new(&scene).unwrap();
    let next = owner(&engine, scene.state());
    let transfers = next
        .inherit_plan(&new_layout, &layout, scene.state(), Some(&retained))
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    next.inherit(&engine, &mut encoder, &transfers, Some(&retained), 1);
    queue.submit([encoder.finish()]);
    for state in &next.states {
        let rates = words(&engine, &state[0]);
        let low = words(&engine, &state[1]);
        assert_eq!(&rates[..12], &edited[0][12..]);
        assert_eq!(&low[..8], &edited[1][8..]);
        assert_eq!(rates[15], 0);
        assert_eq!(low[9], 0);
    }
    let mut invalid = scene.state().clone_host().unwrap();
    invalid.body_qd.host_mut().unwrap()[1].angular.z = f64::NAN;
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    assert!(next.plan(&new_layout, &invalid).is_err());
    assert_eq!(allocations.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
    eprintln!(
        "paired runtime rates: seed, no-traffic ping-pong, component edit, both-side/delta rollback, identity remap and finite preflight pass"
    );
}
