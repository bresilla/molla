use super::*;
use crate::normal_contact_gpu::{ContactStateBuffers, NormalContactRow, ResidentNormalContacts};
use molla_compute::Engine;
use molla_core::BodyId;
use molla_geometry::RESIDENT_MANIFOLD_POINTS;
use molla_math::{GpuSpatialVector, GpuTransform, SpatialVector, Vec3};
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry};
use molla_sim::{BodyParams, ContactMaterial, ModelBuilder};

fn input(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) -> wgpu::Buffer {
    let b = storage(device, "episode test input", bytes.len().max(4) as u64);
    molla_core::gpu_write_buffer!(queue, &b, 0, bytes);
    b
}
fn staging(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("episode test readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    )
}
fn read(device: &wgpu::Device, b: &wgpu::Buffer) -> Vec<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    b.slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let result = molla_core::gpu_readback!(b.slice(..)).to_vec();
    b.unmap();
    result
}
fn contact(a: u32, b: u32) -> NormalContactRow {
    NormalContactRow {
        point_depth: [0., 0., 0., 0.01],
        normal_force: [0., 1., 0., 4.],
        coefficients: [0.6, 0.4, 0.7, 1.],
        endpoints: [a, b, 0, 1],
        relative_velocity: [0., -1., 0., -1.],
        force_on_b: [0., 4., 0., 0.],
        ..Default::default()
    }
}
fn config(rows: u32, shapes: u32, table: u32) -> ImpactEpisodeConfig {
    ImpactEpisodeConfig {
        row_capacity: rows,
        shape_capacity: shapes,
        table_capacity: table,
        approach_speed: 1e-4,
        normal_similarity: 0.95,
    }
}
struct Harness {
    engine: Engine,
    episodes: ResidentImpactEpisodes,
    rows: wgpu::Buffer,
    identities: wgpu::Buffer,
    status: wgpu::Buffer,
}

#[test]
#[ignore = "requires GPU"]
fn invalid_episode_configuration_and_failed_history_shrink_are_fail_closed() {
    let h = Harness::new(4, 6, 8);
    let device = h.engine.device().unwrap();
    let queue = h.engine.queue().unwrap();
    let valid = config(4, 6, 8);
    for bad in [
        ImpactEpisodeConfig {
            row_capacity: 0,
            ..valid
        },
        ImpactEpisodeConfig {
            shape_capacity: 0,
            ..valid
        },
        ImpactEpisodeConfig {
            table_capacity: 3,
            ..valid
        },
        ImpactEpisodeConfig {
            approach_speed: f32::NAN,
            ..valid
        },
        ImpactEpisodeConfig {
            approach_speed: -1.,
            ..valid
        },
        ImpactEpisodeConfig {
            normal_similarity: f32::INFINITY,
            ..valid
        },
        ImpactEpisodeConfig {
            normal_similarity: -1.1,
            ..valid
        },
    ] {
        assert!(
            ResidentImpactEpisodes::new(device, &h.rows, &h.identities, &h.status, bad).is_err()
        );
    }
    let short = input(device, queue, &[0; 4]);
    assert!(ResidentImpactEpisodes::new(device, &short, &h.identities, &h.status, valid).is_err());
    assert!(ResidentImpactEpisodes::new(device, &h.rows, &short, &h.status, valid).is_err());
    assert_eq!(h.run(&[contact(0, 1), contact(2, 3)]).0, 0);
    let smaller = ResidentImpactEpisodes::new(
        device,
        &h.rows,
        &h.identities,
        &h.status,
        ImpactEpisodeConfig {
            table_capacity: 1,
            ..valid
        },
    )
    .unwrap();
    let snapshot = staging(device, 4 + smaller.history.size());
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(
        smaller
            .encode_inherit(device, &mut encoder, &smaller)
            .is_err()
    );
    smaller
        .encode_inherit(device, &mut encoder, &h.episodes)
        .unwrap();
    encoder.copy_buffer_to_buffer(&h.status, 0, &snapshot, 0, 4);
    encoder.copy_buffer_to_buffer(&smaller.history, 0, &snapshot, 4, smaller.history.size());
    queue.submit([encoder.finish()]);
    let got = read(device, &snapshot);
    assert_eq!(
        bytemuck::pod_read_unaligned::<u32>(&got[..4]),
        IMPACT_EPISODE_FAULT
    );
    assert!(got[4..].iter().all(|&byte| byte == 0));
    molla_core::gpu_write_buffer!(queue, &h.status, 0, bytemuck::bytes_of(&0u32));
    assert_eq!(&h.run(&[contact(0, 1), contact(2, 3)]).1[..2], &[0, 0]);
}

#[test]
#[ignore = "requires GPU"]
fn onset_suppression_does_not_hide_invalid_force_or_material_inputs() {
    let h = Harness::new(1, 2, 2);
    let valid = contact(0, 1);
    let checkpoint = h.run(&[valid]).4;
    let mut invalid = Vec::new();
    let mut r = valid;
    r.normal_force[3] = -1.;
    invalid.push(r);
    let mut r = valid;
    r.force_on_b[1] = f32::NAN;
    invalid.push(r);
    let mut r = valid;
    r.surface_velocity[0] = f32::INFINITY;
    invalid.push(r);
    for coefficients in [
        [0.3, 0.4, 0.7, 1.],
        [0.6, -0.1, 0.7, 1.],
        [0.6, 0.4, 1.1, 1.],
        [0.6, 0.4, f32::NAN, 1.],
    ] {
        let mut r = valid;
        r.coefficients = coefficients;
        invalid.push(r);
    }
    for r in invalid {
        molla_core::gpu_write_buffer!(
            h.engine.queue().unwrap(),
            &h.status,
            0,
            bytemuck::bytes_of(&0u32)
        );
        let (status, onset, _, impacts, history) = h.run(&[r]);
        assert_eq!(status, IMPACT_EPISODE_FAULT);
        assert_eq!(onset, [0]);
        assert_eq!(impacts[0].coefficients[3], 0.);
        assert_eq!(history, checkpoint);
    }
}
impl Harness {
    fn new(rows: u32, shapes: u32, table: u32) -> Self {
        let engine = Engine::new_gpu().unwrap();
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let rows_buffer = input(device, queue, &vec![0; rows as usize * 128]);
        let ids: Vec<_> = (1..=shapes).collect();
        let identities = input(device, queue, bytemuck::cast_slice(&ids));
        let status = input(device, queue, &[0; 4]);
        let episodes = ResidentImpactEpisodes::new(
            device,
            &rows_buffer,
            &identities,
            &status,
            config(rows, shapes, table),
        )
        .unwrap();
        Self {
            engine,
            episodes,
            rows: rows_buffer,
            identities,
            status,
        }
    }
    fn run(
        &self,
        rows: &[NormalContactRow],
    ) -> (
        u32,
        Vec<u32>,
        Vec<NormalContactRow>,
        Vec<NormalContactRow>,
        Vec<u8>,
    ) {
        let mut rows = rows.to_vec();
        rows.resize(
            self.episodes.config.row_capacity as usize,
            NormalContactRow::default(),
        );
        molla_core::gpu_write_buffer!(
            self.engine.queue().unwrap(),
            &self.rows,
            0,
            bytemuck::cast_slice(&rows)
        );
        let sources = [
            &self.status,
            self.episodes.onset_buffer(),
            &self.rows,
            self.episodes.impact_row_buffer(),
            &self.episodes.history,
        ];
        let device = self.engine.device().unwrap();
        let out = staging(device, sources.iter().map(|b| b.size()).sum());
        let mut encoder = device.create_command_encoder(&Default::default());
        self.episodes.encode(&mut encoder);
        let mut offset = 0;
        for source in &sources {
            encoder.copy_buffer_to_buffer(source, 0, &out, offset, source.size());
            offset += source.size();
        }
        self.engine.queue().unwrap().submit([encoder.finish()]);
        let bytes = read(device, &out);
        let mut offset = 0;
        let sections: Vec<_> = sources
            .iter()
            .map(|source| {
                let size = source.size() as usize;
                let data = &bytes[offset..offset + size];
                offset += size;
                data
            })
            .collect();
        assert_eq!(self.episodes.resource_counts(), [7, 1, 6]);
        (
            bytemuck::pod_read_unaligned(sections[0]),
            sections[1]
                .chunks_exact(4)
                .map(bytemuck::pod_read_unaligned)
                .collect(),
            sections[2]
                .chunks_exact(128)
                .map(bytemuck::pod_read_unaligned)
                .collect(),
            sections[3]
                .chunks_exact(128)
                .map(bytemuck::pod_read_unaligned)
                .collect(),
            sections[4].to_vec(),
        )
    }
    fn encode(&self, action: impl FnOnce(&mut wgpu::CommandEncoder)) {
        let mut encoder = self
            .engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        action(&mut encoder);
        self.engine.queue().unwrap().submit([encoder.finish()]);
    }
}

#[test]
fn identities_survive_compaction_and_steps_but_not_authored_edits_or_unrelated_scenes() {
    let mut scene = RigidScene::new().unwrap();
    let dummy = scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 }))
        .unwrap();
    let a = scene.insert_body(BodyDesc::default()).unwrap();
    let b = scene.insert_body(BodyDesc::default()).unwrap();
    let mut shape = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.2 });
    shape.parent = Some(a);
    let ca = scene.insert_collider(shape.clone()).unwrap();
    shape.parent = Some(b);
    let cb = scene.insert_collider(shape).unwrap();
    let first = ImpactEpisodeLayout::new(&scene, None).unwrap();
    let token = |layout: &ImpactEpisodeLayout, id: ColliderHandle| layout.entries[&id].1;
    let next = scene.state().clone_host().unwrap();
    scene.commit_step(next).unwrap();
    let stepped = ImpactEpisodeLayout::new(&scene, Some(&first)).unwrap();
    assert_eq!(first.tokens(), stepped.tokens());
    scene.remove_collider(dummy).unwrap();
    let compact = ImpactEpisodeLayout::new(&scene, Some(&stepped)).unwrap();
    assert_eq!(token(&compact, ca), token(&first, ca));
    assert_eq!(token(&compact, cb), token(&first, cb));
    scene
        .set_body_pose(a, Transform::from_translation(Vec3::X))
        .unwrap();
    let moved = ImpactEpisodeLayout::new(&scene, Some(&compact)).unwrap();
    assert_ne!(token(&moved, ca), token(&compact, ca));
    assert_eq!(token(&moved, cb), token(&compact, cb));
    let mut material = scene.collider(cb).unwrap().clone();
    material.restitution = 0.75;
    scene.replace_collider(cb, material).unwrap();
    let edited = ImpactEpisodeLayout::new(&scene, Some(&moved)).unwrap();
    assert_ne!(token(&edited, cb), token(&moved, cb));
    assert_eq!(token(&edited, ca), token(&moved, ca));
    let mut fork = scene.fork().unwrap();
    let same = ImpactEpisodeLayout::new(&fork, Some(&edited)).unwrap();
    assert_eq!(same.tokens(), edited.tokens());
    fork.set_body_velocity(a, SpatialVector::new(Vec3::Y, Vec3::ZERO))
        .unwrap();
    scene
        .set_body_velocity(a, SpatialVector::new(-Vec3::Y, Vec3::ZERO))
        .unwrap();
    let left = ImpactEpisodeLayout::new(&scene, Some(&edited)).unwrap();
    let right = ImpactEpisodeLayout::new(&fork, Some(&edited)).unwrap();
    assert_ne!(token(&left, ca), token(&right, ca));
    let mut independent = RigidScene::new().unwrap();
    let independent_shape = independent
        .insert_collider(ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 }))
        .unwrap();
    assert_eq!(independent_shape.to_bits(), dummy.to_bits());
    let separate = ImpactEpisodeLayout::new(&independent, Some(&first)).unwrap();
    assert_ne!(token(&separate, independent_shape), token(&first, dummy));
}

#[test]
fn detachment_and_repair_retain_tokens_but_not_explicit_state_edits() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let joint = scene
        .insert_joint(molla_sim::runtime::JointDesc {
            parent: None,
            child: body,
            kind: molla_sim::JointType::Fixed,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let mut shape = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    shape.parent = Some(body);
    scene.insert_collider(shape).unwrap();
    let mut old = ImpactEpisodeLayout::new(&scene, None).unwrap();
    scene.break_joints(&[joint]).unwrap();
    let authored = ImpactEpisodeLayout::new(&scene, Some(&old)).unwrap();
    assert_eq!(authored.tokens(), old.tokens());
    old.preserve_after_detachment(&scene);
    let automatic = ImpactEpisodeLayout::new(&scene, Some(&old)).unwrap();
    assert_eq!(automatic.tokens(), old.tokens());
    scene.repair_joint(joint).unwrap();
    let repaired = ImpactEpisodeLayout::new(&scene, Some(&automatic)).unwrap();
    assert_eq!(repaired.tokens(), automatic.tokens());
    scene
        .set_body_velocity(body, SpatialVector::new(Vec3::Y, Vec3::ZERO))
        .unwrap();
    old.preserve_after_detachment(&scene);
    let edited = ImpactEpisodeLayout::new(&scene, Some(&old)).unwrap();
    assert_ne!(edited.tokens(), old.tokens());
    let mut previous = edited;
    scene
        .set_body_mass_properties(
            body,
            molla_sim::MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: molla_math::Mat3::IDENTITY,
            },
        )
        .unwrap();
    let mass = ImpactEpisodeLayout::new(&scene, Some(&previous)).unwrap();
    assert_ne!(mass.tokens(), previous.tokens());
    previous = mass;
    let next = scene.state().clone_host().unwrap();
    scene.commit_step(next).unwrap();
    assert_eq!(
        ImpactEpisodeLayout::new(&scene, Some(&previous))
            .unwrap()
            .tokens(),
        previous.tokens()
    );
}

#[test]
fn restoring_published_state_reuses_tokens_without_aliasing_genuine_edits() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut shape = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    shape.parent = Some(body);
    let shape = scene.insert_collider(shape).unwrap();
    let pose = scene.body_pose(body).unwrap();
    let velocity = scene.body_velocity(body).unwrap();
    let mut published = ImpactEpisodeLayout::new(&scene, None).unwrap();
    published.record_published(&scene);
    scene
        .set_body_velocity(body, SpatialVector::new(-Vec3::Y, Vec3::ZERO))
        .unwrap();
    let changed = ImpactEpisodeLayout::new(&scene, Some(&published)).unwrap();
    assert_ne!(changed.tokens(), published.tokens());
    scene.set_body_velocity(body, velocity).unwrap();
    let restored = ImpactEpisodeLayout::new(&scene, Some(&published)).unwrap();
    assert_eq!(restored.tokens(), published.tokens());
    scene
        .set_body_pose(body, Transform::from_translation(Vec3::X))
        .unwrap();
    let moved = ImpactEpisodeLayout::new(&scene, Some(&restored)).unwrap();
    assert_ne!(moved.tokens(), restored.tokens());
    scene.set_body_pose(body, pose).unwrap();
    let mut collider = scene.collider(shape).unwrap().clone();
    collider.restitution = 0.7;
    scene.replace_collider(shape, collider).unwrap();
    let material = ImpactEpisodeLayout::new(&scene, Some(&restored)).unwrap();
    assert_ne!(material.tokens(), restored.tokens());
}

#[test]
#[ignore = "requires GPU"]
fn onset_suppression_pair_reordering_separation_and_normal_changes() {
    let h = Harness::new(8, 4, 16);
    let first = contact(0, 1);
    let mut second = first;
    second.point_depth[0] = 1.;
    second.relative_velocity[1] = 0.5;
    let result = h.run(&[first, second]);
    assert_eq!(result.0, 0);
    assert_eq!(&result.1[..2], &[1, 1]);
    for i in 0..2 {
        assert_eq!(result.2[i].normal_force[3], 0.);
        assert_eq!(result.2[i].point_depth[3], 0.01);
        assert_eq!(result.3[i].coefficients[2], 0.7);
    }
    let mut reverse = first;
    reverse.endpoints.swap(0, 1);
    reverse.endpoints.swap(2, 3);
    reverse.normal_force[1] = -1.;
    reverse.relative_velocity[1] = 1.;
    reverse.force_on_b[1] = -4.;
    let result = h.run(&[second, reverse]);
    assert_eq!(result.0, 0);
    assert!(result.1.iter().all(|&v| v == 0));
    assert_eq!(result.2[0].normal_force[3], 4.);
    assert_eq!(result.3[1].coefficients[2], 0.);
    assert_eq!(h.run(&[]).0, 0);
    assert_eq!(h.run(&[first]).1[0], 1);
    let mut tilted = first;
    tilted.normal_force = [1., 0., 0., 4.];
    tilted.relative_velocity = [-1., 0., 0., -1.];
    assert_eq!(h.run(&[tilted]).1[0], 1);
    assert_eq!(h.run(&[tilted]).1[0], 0);
    molla_core::gpu_write_buffer!(
        h.engine.queue().unwrap(),
        &h.identities,
        0,
        bytemuck::cast_slice(&[11u32, 2, 3, 4]),
    );
    assert_eq!(h.run(&[tilted]).1[0], 1);
}

#[test]
#[ignore = "requires GPU"]
fn frame_rollback_sticky_fault_and_capacity_failure_preserve_history() {
    let h = Harness::new(4, 6, 2);
    let a = contact(0, 1);
    let b = contact(2, 3);
    let c = contact(4, 5);
    let initial = h.run(&[a]);
    assert_eq!(initial.0, 0);
    h.encode(|e| h.episodes.encode_checkpoint(e));
    assert_eq!(h.run(&[a, b]).1[..2], [0, 1]);
    let before = h.run(&[a, b]).4;
    let failed = h.run(&[a, b, c]);
    assert_eq!(failed.0, IMPACT_EPISODE_FAULT);
    assert_eq!(failed.4, before);
    assert!(failed.1.iter().all(|&v| v == 0));
    assert_eq!(h.run(&[a]).0, IMPACT_EPISODE_FAULT);
    h.encode(|e| h.episodes.encode_rollback(e));
    molla_core::gpu_write_buffer!(
        h.engine.queue().unwrap(),
        &h.status,
        0,
        bytemuck::bytes_of(&0u32)
    );
    let restored = h.run(&[a]);
    assert_eq!(restored.0, 0);
    assert_eq!(restored.1[0], 0);
    let retried = h.run(&[a, b]);
    assert_eq!(retried.0, 0);
    assert_eq!(&retried.1[..2], &[0, 1]);
    let mut invalid = a;
    invalid.endpoints[0] = 99;
    assert_eq!(h.run(&[invalid]).0, IMPACT_EPISODE_FAULT);
}

#[test]
#[ignore = "requires GPU"]
fn history_rehash_and_dense_shape_remapping_retain_pair_identity() {
    let h = Harness::new(8, 4, 16);
    let a = contact(0, 1);
    let b = contact(2, 3);
    assert_eq!(h.run(&[a, b]).0, 0);
    let device = h.engine.device().unwrap();
    let queue = h.engine.queue().unwrap();
    let rows = input(device, queue, &[0; 1024]);
    let identities = input(device, queue, bytemuck::cast_slice(&[3u32, 1, 4, 2]));
    let next = ResidentImpactEpisodes::new(device, &rows, &identities, &h.status, config(8, 4, 64))
        .unwrap();
    h.encode(|e| next.encode_inherit(device, e, &h.episodes).unwrap());
    let mut remapped = vec![contact(1, 3), contact(0, 2)];
    remapped.resize(8, NormalContactRow::default());
    molla_core::gpu_write_buffer!(queue, &rows, 0, bytemuck::cast_slice(&remapped));
    let out = staging(device, 36);
    let mut encoder = device.create_command_encoder(&Default::default());
    next.encode(&mut encoder);
    encoder.copy_buffer_to_buffer(&h.status, 0, &out, 0, 4);
    encoder.copy_buffer_to_buffer(next.onset_buffer(), 0, &out, 4, 32);
    queue.submit([encoder.finish()]);
    assert_eq!(read(device, &out), vec![0; 36]);
}

#[test]
#[ignore = "requires GPU"]
fn one_hundred_queued_manifold_changes_match_pair_reference_across_workgroups() {
    let h = Harness::new(257, 130, 256);
    let device = h.engine.device().unwrap();
    let queue = h.engine.queue().unwrap();
    let stride = 4 + 257 * 4;
    let out = staging(device, stride * 100);
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut old = BTreeMap::new();
    let mut expectations = Vec::new();
    for frame in 0..100usize {
        let rotation = frame % 130;
        let mut ids = (1u32..=130).collect::<Vec<_>>();
        ids.rotate_left(rotation);
        let identity_source = input(device, queue, bytemuck::cast_slice(&ids));
        encoder.copy_buffer_to_buffer(&identity_source, 0, &h.identities, 0, 520);
        let mut next = BTreeMap::new();
        let mut data = Vec::new();
        for i in 0..257 {
            let pair = i % 65;
            let present = (pair + frame) % 11 != 0;
            let axis = (frame / 7 + pair) % 3 == 0;
            let a = ((pair * 2 + 130 - rotation) % 130) as u32;
            let b = ((pair * 2 + 1 + 130 - rotation) % 130) as u32;
            let mut row = contact(a, b);
            let approaching = (pair + frame) % 13 != 0;
            if axis {
                row.normal_force = [1., 0., 0., 4.];
                row.relative_velocity = [-1., 0., 0., -1.];
            }
            if !approaching {
                row.relative_velocity = [0.; 4];
            }
            if frame % 2 != 0 {
                row.endpoints.swap(0, 1);
                for v in &mut row.normal_force[..3] {
                    *v = -*v;
                }
                for v in &mut row.relative_velocity[..3] {
                    *v = -*v;
                }
            }
            let onset = present && approaching && old.get(&pair) != Some(&axis);
            if present {
                next.insert(pair, axis);
            } else {
                row = NormalContactRow::default();
            }
            data.push((row, u32::from(onset)));
        }
        data.rotate_left(frame % 257);
        if frame % 3 == 0 {
            data.reverse();
        }
        expectations.push(data.iter().map(|(_, v)| *v).collect::<Vec<_>>());
        old = next;
        let source = input(
            device,
            queue,
            bytemuck::cast_slice(&data.iter().map(|(r, _)| *r).collect::<Vec<_>>()),
        );
        encoder.copy_buffer_to_buffer(&source, 0, &h.rows, 0, h.rows.size());
        h.episodes.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&h.status, 0, &out, frame as u64 * stride, 4);
        encoder.copy_buffer_to_buffer(
            h.episodes.onset_buffer(),
            0,
            &out,
            frame as u64 * stride + 4,
            257 * 4,
        );
    }
    queue.submit([encoder.finish()]);
    let bytes = read(device, &out);
    for (frame, bytes) in bytes.chunks_exact(stride as usize).enumerate() {
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&bytes[..4]),
            0,
            "frame={frame}"
        );
        let actual = bytes[4..]
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned::<u32>)
            .collect::<Vec<_>>();
        assert_eq!(actual, expectations[frame], "frame={frame}");
    }
}

#[test]
#[ignore = "requires GPU"]
fn normal_preparation_episode_suppression_and_force_gather_share_one_encoder() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    let sphere = builder
        .add_shape_sphere(body, 0.5, Transform::IDENTITY)
        .unwrap();
    let plane = builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    for shape in [sphere, plane] {
        builder
            .set_shape_contact_material(
                shape,
                ContactMaterial {
                    stiffness: 100.,
                    damping: 0.,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let model = builder.build().unwrap();
    let pose = input(
        device,
        queue,
        bytemuck::bytes_of(&GpuTransform::from(Transform::IDENTITY)),
    );
    let velocity = input(
        device,
        queue,
        bytemuck::bytes_of(&GpuSpatialVector::from(SpatialVector::new(
            -Vec3::Y,
            Vec3::ZERO,
        ))),
    );
    let external = input(device, queue, &[0; 32]);
    let a = input(device, queue, bytemuck::bytes_of(&0u32));
    let b = input(device, queue, bytemuck::bytes_of(&1u32));
    let count = input(device, queue, bytemuck::bytes_of(&1u32));
    let mut points = vec![[0_f32; 16]; RESIDENT_MANIFOLD_POINTS as usize];
    points[0][1] = -1.;
    points[0][3] = 0.1;
    points[0][12] = f32::from_bits(1);
    let geometry = input(device, queue, bytemuck::cast_slice(&points));
    let status = input(device, queue, &[0; 4]);
    let normal = ResidentNormalContacts::new(
        device,
        &model,
        &[ContactStateBuffers {
            poses: &pose,
            velocities: &velocity,
            external_forces: &external,
        }],
        &a,
        &b,
        &count,
        &count,
        &geometry,
        &status,
        1,
    )
    .unwrap();
    let identities = input(device, queue, bytemuck::cast_slice(&[1u32, 2]));
    let episodes = ResidentImpactEpisodes::new(
        device,
        normal.row_buffer(),
        &identities,
        &status,
        config(RESIDENT_MANIFOLD_POINTS, 2, 8),
    )
    .unwrap();
    let out = staging(device, 72);
    let mut encoder = device.create_command_encoder(&Default::default());
    for step in 0..2 {
        normal.encode_rows(&mut encoder, 0).unwrap();
        episodes.encode(&mut encoder);
        normal.encode_forces(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(&status, 0, &out, step * 36, 4);
        encoder.copy_buffer_to_buffer(normal.force_buffer(), 0, &out, step * 36 + 4, 32);
    }
    queue.submit([encoder.finish()]);
    let bytes = read(device, &out);
    for (step, bytes) in bytes.chunks_exact(36).enumerate() {
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&bytes[..4]), 0);
        let force = bytemuck::pod_read_unaligned::<GpuSpatialVector>(&bytes[4..]);
        assert!((force.linear[1] - if step == 0 { 0. } else { 10. }).abs() < 1e-5);
    }
}
