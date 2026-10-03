#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_solvers::constraint_rows_gpu::ScalarConstraintRow;
use molla_solvers::contact_islands_gpu::{
    CONTACT_ISLAND_FAULT_CAPACITY, CONTACT_ISLAND_FAULT_INPUT, ContactIslandConfig,
    NO_CONTACT_ISLAND as NONE, ResidentContactIslands,
};
use molla_solvers::normal_contact_gpu::NormalContactRow;
use std::collections::BTreeMap;

fn buffer(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> wgpu::Buffer {
    let b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("contact island input"),
        size: data.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        queue.write_buffer(&b, 0, data);
    }
    b
}
fn staging(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("contact island readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}
fn read(device: &wgpu::Device, b: &wgpu::Buffer) -> Vec<u32> {
    let (tx, rx) = std::sync::mpsc::channel();
    b.slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let result = bytemuck::cast_slice(&b.slice(..).get_mapped_range()).to_vec();
    b.unmap();
    result
}
fn row(a: u32, b: u32) -> NormalContactRow {
    NormalContactRow {
        endpoints: [0, 1, a, b],
        normal_force: [0.0, 1.0, 0.0, 2.0],
        coefficients: [0.6, 0.2, 0.0, 1.0],
        ..Default::default()
    }
}

fn expected(
    config: ContactIslandConfig,
    owners: &[u32],
    edges: &[[u32; 2]],
    rows: &[NormalContactRow],
) -> Vec<u32> {
    expected_mixed(config, owners, edges, rows, &[])
}

fn expected_mixed(
    config: ContactIslandConfig,
    owners: &[u32],
    edges: &[[u32; 2]],
    rows: &[NormalContactRow],
    scalar: &[ScalarConstraintRow],
) -> Vec<u32> {
    expected_graph(config, owners, edges, rows, scalar, None)
}

fn expected_graph(
    config: ContactIslandConfig,
    owners: &[u32],
    edges: &[[u32; 2]],
    rows: &[NormalContactRow],
    scalar: &[ScalarConstraintRow],
    onsets: Option<&[u32]>,
) -> Vec<u32> {
    fn root(parent: &[u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            i = parent[i as usize];
        }
        i
    }
    fn join(parent: &mut [u32], a: u32, b: u32) {
        if a == NONE || b == NONE {
            return;
        }
        let a = root(parent, a);
        let b = root(parent, b);
        parent[a.max(b) as usize] = a.min(b);
    }
    let owner = |body: u32| {
        if body == NONE {
            NONE
        } else {
            owners[body as usize]
        }
    };
    let nodes = config.articulation_count + scalar.len() as u32;
    let mut parent: Vec<_> = (0..nodes).collect();
    for &[a, b] in edges {
        join(&mut parent, a, b);
    }
    for r in rows {
        if r.coefficients[3] != 0.0 && (onsets.is_some() || r.normal_force[3] > 0.0) {
            join(&mut parent, owner(r.endpoints[2]), owner(r.endpoints[3]));
        }
    }
    for (i, r) in scalar.iter().enumerate() {
        if r.endpoints[3] != 0 {
            join(
                &mut parent,
                config.articulation_count + i as u32,
                owner(r.endpoints[0]),
            );
            join(
                &mut parent,
                config.articulation_count + i as u32,
                owner(r.endpoints[1]),
            );
        }
    }
    let mut groups = BTreeMap::<u32, Vec<u32>>::new();
    for (index, r) in rows.iter().enumerate() {
        if r.coefficients[3] == 0.0
            || (onsets.is_none() && (r.normal_force[3] == 0.0 || r.coefficients[0] == 0.0))
        {
            continue;
        }
        let a = owner(r.endpoints[2]).min(owner(r.endpoints[3]));
        if a != NONE {
            groups
                .entry(root(&parent, a))
                .or_default()
                .push(index as u32);
        }
    }
    for (i, r) in scalar.iter().enumerate() {
        if r.endpoints[3] != 0 {
            groups
                .entry(root(&parent, config.articulation_count + i as u32))
                .or_default()
                .push(config.row_capacity + i as u32);
        }
    }
    if let Some(onsets) = onsets {
        groups.retain(|_, members| {
            members
                .iter()
                .any(|&r| (r as usize) < rows.len() && onsets[r as usize] != 0)
        });
    }
    let mut counts = vec![0; config.island_capacity as usize];
    let mut roots = vec![NONE; config.island_capacity as usize];
    let mut indices = vec![NONE; (config.island_capacity * config.contacts_per_island) as usize];
    let mut slots = vec![NONE; (config.row_capacity as usize + scalar.len()) * 2];
    let mut arts = vec![NONE; nodes.max(1) as usize];
    for (island, (&r, contacts)) in groups.iter().enumerate() {
        counts[island] = contacts.len() as u32;
        roots[island] = r;
        arts[r as usize] = island as u32;
        for (slot, &index) in contacts.iter().enumerate() {
            indices[island * config.contacts_per_island as usize + slot] = index;
            slots[2 * index as usize] = island as u32;
            slots[2 * index as usize + 1] = slot as u32;
        }
        for a in 0..config.articulation_count {
            if root(&parent, a) == r {
                arts[a as usize] = island as u32;
            }
        }
    }
    let mut result = vec![0, groups.len() as u32];
    result.extend(counts);
    result.extend(roots);
    result.extend(indices);
    result.extend(slots);
    result.extend(&arts);
    let mut members = Vec::new();
    let mut offsets = Vec::new();
    for island in 0..config.island_capacity {
        offsets.push(members.len() as u32);
        members.extend(
            arts.iter()
                .take(config.articulation_count as usize)
                .enumerate()
                .filter_map(|(a, &owner)| (owner == island).then_some(a as u32)),
        );
    }
    offsets.push(members.len() as u32);
    members.resize(config.articulation_count.max(1) as usize, NONE);
    result.extend(offsets);
    result.extend(members);
    result
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-islands-gpu"]
fn mixed_joint_contact_graphs_preserve_joint_only_and_boundary_only_islands() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let config = ContactIslandConfig {
        articulation_count: 129,
        row_capacity: 129,
        island_capacity: 130,
        contacts_per_island: 300,
    };
    let mut owners: Vec<_> = (0..129u32).collect();
    owners.push(NONE);
    let dependencies = [[0, 1]];
    let normal = buffer(device, queue, &vec![0; 129 * 128]);
    let scalar = buffer(device, queue, &vec![0; 130 * 32]);
    let status = buffer(device, queue, &[0; 4]);
    let component = ResidentContactIslands::with_scalar_rows(
        &engine,
        &owners,
        &dependencies,
        &normal,
        Some((&scalar, 130)),
        &status,
        config,
    )
    .unwrap();
    assert_eq!(component.resource_counts(), [23, 7, 9]);
    let frame_words = expected_mixed(
        config,
        &owners,
        &dependencies,
        &vec![NormalContactRow::default(); 129],
        &vec![ScalarConstraintRow::default(); 130],
    )
    .len();
    let snapshot = staging(device, frame_words as u64 * 4 * 60);
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut references = Vec::new();
    for step in 0..60 {
        let mode = step % 6;
        let mut normals = vec![NormalContactRow::default(); 129];
        if matches!(mode, 0 | 3) {
            for (i, r) in normals.iter_mut().enumerate() {
                *r = row(i as u32, 129);
            }
        }
        let mut scalars = vec![ScalarConstraintRow::default(); 130];
        if mode < 3 {
            for (i, r) in scalars[..128].iter_mut().enumerate() {
                *r = ScalarConstraintRow {
                    endpoints: [
                        i as u32 + 1,
                        i as u32,
                        1,
                        u32::from(mode != 2 || i % 2 == 0),
                    ],
                    velocity: [0.; 4],
                };
            }
            scalars[128] = ScalarConstraintRow {
                endpoints: [NONE, NONE, 1, 1],
                velocity: [0.; 4],
            };
        } else if mode == 5 {
            for (i, r) in scalars[..129].iter_mut().enumerate() {
                *r = ScalarConstraintRow {
                    endpoints: [129, i as u32, 2, 1],
                    velocity: [0.; 4],
                };
            }
            scalars[129] = ScalarConstraintRow {
                endpoints: [129, NONE, 1, 1],
                velocity: [0.; 4],
            };
        }
        if step % 2 == 1 {
            for r in &mut scalars {
                r.endpoints.swap(0, 1);
            }
            for r in &mut normals {
                r.endpoints.swap(2, 3);
            }
        }
        for (dst, bytes) in [
            (&normal, bytemuck::cast_slice(&normals)),
            (&scalar, bytemuck::cast_slice(&scalars)),
        ] {
            let source = buffer(device, queue, bytes);
            encoder.copy_buffer_to_buffer(&source, 0, dst, 0, dst.size());
        }
        component.encode(&mut encoder);
        capture(
            &mut encoder,
            &component,
            &status,
            &snapshot,
            step as u64 * frame_words as u64 * 4,
        );
        references.push(expected_mixed(
            config,
            &owners,
            &dependencies,
            &normals,
            &scalars,
        ));
    }
    queue.submit([encoder.finish()]);
    for (step, (got, expected)) in read(device, &snapshot)
        .chunks_exact(frame_words)
        .zip(references)
        .enumerate()
    {
        assert_eq!(got, expected, "step={step}");
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-islands-gpu"]
fn mixed_graph_validation_capacity_and_sticky_recovery() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let config = ContactIslandConfig {
        articulation_count: 1,
        row_capacity: 1,
        island_capacity: 3,
        contacts_per_island: 2,
    };
    let normal = buffer(device, queue, bytemuck::bytes_of(&row(0, NONE)));
    let scalar = buffer(device, queue, &[0; 64]);
    let status = buffer(device, queue, &[0; 4]);
    assert!(
        ResidentContactIslands::with_scalar_rows(
            &engine,
            &[0],
            &[],
            &normal,
            Some((&scalar, 0)),
            &status,
            config
        )
        .is_err()
    );
    assert!(
        ResidentContactIslands::with_scalar_rows(
            &engine,
            &[0],
            &[],
            &normal,
            Some((&status, 1)),
            &status,
            config
        )
        .is_err()
    );
    let component = ResidentContactIslands::with_scalar_rows(
        &engine,
        &[0],
        &[],
        &normal,
        Some((&scalar, 2)),
        &status,
        config,
    )
    .unwrap();
    let output = staging(device, 18 * 8);
    let mut encoder = device.create_command_encoder(&Default::default());
    for step in 0..18 {
        let active = ScalarConstraintRow {
            endpoints: [NONE, 0, 1, 1],
            velocity: [0.; 4],
        };
        let mut rows = [active, ScalarConstraintRow::default()];
        match step % 9 {
            0 => rows[1] = active,
            3 => rows[0].endpoints[2] = 4,
            4 => rows[0].endpoints[3] = 2,
            5 => rows[0].endpoints[1] = 1,
            6 => {
                rows[0].endpoints[2] = 3;
                rows[0].velocity = [0.0, 0.25, -2.0, 3.0];
            }
            7 => rows[0].endpoints[2] = 2,
            8 => rows[0].endpoints[2] = 0,
            _ => {}
        }
        if step % 9 != 1 {
            encoder.clear_buffer(&status, 0, None);
        }
        let source = buffer(device, queue, bytemuck::cast_slice(&rows));
        encoder.copy_buffer_to_buffer(&source, 0, &scalar, 0, 64);
        component.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&status, 0, &output, step * 8, 4);
        encoder.copy_buffer_to_buffer(component.island_count_buffer(), 0, &output, step * 8 + 4, 4);
    }
    queue.submit([encoder.finish()]);
    for (step, frame) in read(device, &output).chunks_exact(2).enumerate() {
        let expected = match step % 9 {
            0 | 1 => CONTACT_ISLAND_FAULT_CAPACITY,
            2 | 6 | 7 => 0,
            _ => CONTACT_ISLAND_FAULT_INPUT,
        };
        assert_eq!(frame[0], expected, "step={step}");
        if step % 9 == 1 {
            assert_eq!(frame[1], 0);
        }
        if matches!(step % 9, 2 | 6 | 7) {
            assert_eq!(frame[1], 1);
        }
    }
}

fn capture(
    encoder: &mut wgpu::CommandEncoder,
    component: &ResidentContactIslands,
    status: &wgpu::Buffer,
    output: &wgpu::Buffer,
    mut offset: u64,
) {
    for b in [
        status,
        component.island_count_buffer(),
        component.contact_counts_buffer(),
        component.roots_buffer(),
        component.contact_indices_buffer(),
        component.row_slots_buffer(),
        component.articulation_islands_buffer(),
        component.articulation_offsets_buffer(),
        component.articulation_indices_buffer(),
    ] {
        encoder.copy_buffer_to_buffer(b, 0, output, offset, b.size());
        offset += b.size();
    }
}

#[test]
#[ignore = "requires GPU"]
fn onset_selected_groups_match_cpu_over_80_queued_contact_and_scalar_graphs() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let config = ContactIslandConfig {
        articulation_count: 130,
        row_capacity: 260,
        island_capacity: 130,
        contacts_per_island: 400,
    };
    let mut owners: Vec<_> = (0..130).collect();
    owners.push(NONE);
    let edges = [[0, 1], [125, 126]];
    let rows = buffer(device, queue, &vec![0; 260 * 128]);
    let scalars = buffer(device, queue, &vec![0; 131 * 32]);
    let onsets = buffer(device, queue, &vec![0; 260 * 4]);
    let status = buffer(device, queue, &[0; 4]);
    let component = ResidentContactIslands::with_impact_onsets(
        &engine,
        &owners,
        &edges,
        &rows,
        Some((&scalars, 131)),
        &onsets,
        &status,
        config,
        260,
    )
    .unwrap();
    let mut references = Vec::new();
    let mut frames = Vec::new();
    for step in 0..80 {
        let mut contacts = Vec::new();
        let mut mask = Vec::new();
        let mut scalar = Vec::new();
        for i in 0..130u32 {
            let mut contact = row(i, 130);
            contact.normal_force[3] = 0.;
            contact.coefficients[..2].fill(0.);
            contacts.push(contact);
            mask.push(u32::from(step % 8 != 0 && (i + step) % 17 == 0));
            let neighbor = i / 5 * 5 + (i + 1) % 5;
            let mut link = row(i, neighbor);
            link.normal_force[3] = 0.;
            link.coefficients[3] = u32::from((i + step) % 3 != 0) as f32;
            contacts.push(link);
            mask.push(0);
            scalar.push(ScalarConstraintRow {
                endpoints: [
                    i,
                    if step % 4 == 1 { (i + 1) % 130 } else { 130 },
                    1 + i % 2,
                    u32::from((i + step) % 7 != 0),
                ],
                velocity: [0.; 4],
            });
        }
        scalar.push(ScalarConstraintRow {
            endpoints: [NONE, 130, 1, 1],
            velocity: [0.; 4],
        });
        if step % 2 == 1 {
            contacts.reverse();
            mask.reverse();
            scalar.reverse();
            for row in &mut contacts {
                row.endpoints.swap(2, 3);
            }
        }
        references.push(expected_graph(
            config,
            &owners,
            &edges,
            &contacts,
            &scalar,
            Some(&mask),
        ));
        frames.push((contacts, scalar, mask));
    }
    let words = references[0].len();
    let snapshot = staging(device, words as u64 * 4 * frames.len() as u64);
    let mut encoder = device.create_command_encoder(&Default::default());
    for (step, (contacts, scalar, mask)) in frames.iter().enumerate() {
        for (dst, bytes) in [
            (&rows, bytemuck::cast_slice(contacts)),
            (&scalars, bytemuck::cast_slice(scalar)),
            (&onsets, bytemuck::cast_slice(mask)),
        ] {
            let src = buffer(device, queue, bytes);
            encoder.copy_buffer_to_buffer(&src, 0, dst, 0, dst.size());
        }
        component.encode(&mut encoder);
        capture(
            &mut encoder,
            &component,
            &status,
            &snapshot,
            step as u64 * words as u64 * 4,
        );
    }
    queue.submit([encoder.finish()]);
    for (step, (got, expected)) in read(device, &snapshot)
        .chunks_exact(words)
        .zip(references)
        .enumerate()
    {
        assert_eq!(got, expected, "selected graph {step}");
    }
}

#[test]
#[ignore = "requires GPU"]
fn onset_selection_validates_masks_and_only_charges_selected_island_capacity() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut contacts = [row(0, 3), row(1, 2), row(1, 2)];
    let rows = buffer(device, queue, bytemuck::cast_slice(&contacts));
    let onsets = buffer(device, queue, bytemuck::cast_slice(&[1u32, 0, 0]));
    let short = buffer(device, queue, &[0; 4]);
    let status = buffer(device, queue, &[0; 4]);
    let config = ContactIslandConfig {
        articulation_count: 3,
        row_capacity: 3,
        island_capacity: 1,
        contacts_per_island: 1,
    };
    assert!(
        ResidentContactIslands::with_impact_onsets(
            &engine,
            &[0, 1, 2, NONE],
            &[],
            &rows,
            None,
            &short,
            &status,
            config,
            1
        )
        .is_err()
    );
    let component = ResidentContactIslands::with_impact_onsets(
        &engine,
        &[0, 1, 2, NONE],
        &[],
        &rows,
        None,
        &onsets,
        &status,
        config,
        1,
    )
    .unwrap();
    let snapshot = staging(device, 4);
    let run = |mask: [u32; 3], reset: bool| {
        queue.write_buffer(&onsets, 0, bytemuck::cast_slice(&mask));
        if reset {
            queue.write_buffer(&status, 0, bytemuck::bytes_of(&0u32));
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        component.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&status, 0, &snapshot, 0, 4);
        queue.submit([encoder.finish()]);
        read(device, &snapshot)[0]
    };
    assert_eq!(run([1, 0, 0], true), 0);
    assert_eq!(run([0, 1, 0], true), CONTACT_ISLAND_FAULT_CAPACITY);
    assert_eq!(run([1, 0, 0], false), CONTACT_ISLAND_FAULT_CAPACITY);
    assert_eq!(run([1, 0, 0], true), 0);
    assert_eq!(run([2, 0, 0], true), CONTACT_ISLAND_FAULT_INPUT);
    contacts[0].coefficients[3] = 0.;
    queue.write_buffer(&rows, 0, bytemuck::cast_slice(&contacts));
    assert_eq!(run([1, 0, 0], true), CONTACT_ISLAND_FAULT_INPUT);
    assert_eq!(run([0, 0, 0], true), 0);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-islands-gpu"]
fn deterministic_contact_islands_match_cpu_across_100_queued_graph_changes() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let config = ContactIslandConfig {
        articulation_count: 160,
        row_capacity: 512,
        island_capacity: 160,
        contacts_per_island: 64,
    };
    let mut owners: Vec<_> = (0..160).flat_map(|i| [i, i]).collect();
    owners.extend([NONE, NONE]);
    let edges = [[0, 1], [2, 3], [150, 151]];
    let rows = buffer(device, queue, &vec![0; config.row_capacity as usize * 128]);
    let status = buffer(device, queue, &[0; 4]);
    let component =
        ResidentContactIslands::new(&engine, &owners, &edges, &rows, &status, config).unwrap();
    assert_eq!(component.resource_counts(), [23, 7, 9]);
    let mut references = Vec::new();
    let frame_words = 2
        + 2 * config.island_capacity
        + config.island_capacity * config.contacts_per_island
        + config.row_capacity * 2
        + config.articulation_count * 2
        + config.island_capacity
        + 1;
    let output = staging(device, u64::from(frame_words) * 4 * 100);
    let mut encoder = device.create_command_encoder(&Default::default());
    for step in 0..100 {
        let mode = step % 5;
        let group_size = if mode == 0 { 8 } else { 4 };
        let mut frame = vec![NormalContactRow::default(); config.row_capacity as usize];
        for (i, r) in frame.iter_mut().enumerate().take(if mode == 4 {
            0
        } else if mode == 3 {
            1
        } else {
            425
        }) {
            let art = (i as u32 * 37 + 11) % 160;
            let other = art / group_size * group_size + (art + 1) % group_size;
            let boundary = if i % 3 == 0 {
                320
            } else if i % 3 == 1 {
                321
            } else {
                NONE
            };
            *r = row(
                2 * art + i as u32 % 2,
                if mode == 2 || i % 4 == 0 {
                    boundary
                } else {
                    2 * other
                },
            );
            if i % 13 == 0 {
                r.coefficients[0] = 0.0;
                r.coefficients[1] = 0.0;
            }
            if i % 17 == 0 {
                r.coefficients[3] = 0.0;
            }
            if i % 19 == 0 {
                r.normal_force[3] = 0.0;
            }
            if i % 23 == 0 {
                r.endpoints.swap(2, 3);
            }
        }
        if mode == 3 {
            frame[0] = row(318, 320);
        }
        let source = buffer(device, queue, bytemuck::cast_slice(&frame));
        encoder.copy_buffer_to_buffer(&source, 0, &rows, 0, rows.size());
        component.encode(&mut encoder);
        capture(
            &mut encoder,
            &component,
            &status,
            &output,
            step as u64 * u64::from(frame_words) * 4,
        );
        references.push(expected(config, &owners, &edges, &frame));
    }
    queue.submit([encoder.finish()]);
    let result = read(device, &output);
    for (step, (actual, reference)) in result
        .chunks_exact(frame_words as usize)
        .zip(references)
        .enumerate()
    {
        assert_eq!(actual, reference, "step={step}");
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-islands-gpu"]
fn reversed_cross_workgroup_chains_keep_one_whole_articulation_component() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let config = ContactIslandConfig {
        articulation_count: 1025,
        row_capacity: 2048,
        island_capacity: 1,
        contacts_per_island: 2048,
    };
    let owners: Vec<_> = (0..1025).collect();
    let rows = buffer(device, queue, &vec![0; config.row_capacity as usize * 128]);
    let status = buffer(device, queue, &[0; 4]);
    let component =
        ResidentContactIslands::new(&engine, &owners, &[], &rows, &status, config).unwrap();
    let words = expected(
        config,
        &owners,
        &[],
        &vec![NormalContactRow::default(); 2048],
    )
    .len();
    let output = staging(device, words as u64 * 4 * 12);
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut references = Vec::new();
    for step in 0..12 {
        let mut frame = vec![NormalContactRow::default(); 2048];
        for i in 0..1024 {
            let index = if step % 3 == 0 {
                1023 - i
            } else if step % 3 == 1 {
                i
            } else {
                i * 37 % 1024
            };
            frame[index] = row(i as u32 + 1, i as u32);
            if step % 2 == 0 {
                frame[index].coefficients[0] = 0.0;
                frame[index].coefficients[1] = 0.0;
            }
        }
        for (i, r) in frame[1024..].iter_mut().enumerate() {
            *r = row((i * 17 % 1025) as u32, NONE);
        }
        let source = buffer(device, queue, bytemuck::cast_slice(&frame));
        encoder.copy_buffer_to_buffer(&source, 0, &rows, 0, rows.size());
        component.encode(&mut encoder);
        capture(
            &mut encoder,
            &component,
            &status,
            &output,
            step as u64 * words as u64 * 4,
        );
        references.push(expected(config, &owners, &[], &frame));
    }
    queue.submit([encoder.finish()]);
    for (step, (actual, reference)) in read(device, &output)
        .chunks_exact(words)
        .zip(references)
        .enumerate()
    {
        assert_eq!(actual, reference, "step={step}");
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-islands-gpu"]
fn contact_island_faults_retire_outputs_and_reset_at_explicit_boundary() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let config = ContactIslandConfig {
        articulation_count: 8,
        row_capacity: 8,
        island_capacity: 2,
        contacts_per_island: 2,
    };
    let owners: Vec<_> = (0..8).collect();
    let rows = buffer(device, queue, &vec![0; 8 * 128]);
    let status = buffer(device, queue, &[0; 4]);
    let component =
        ResidentContactIslands::new(&engine, &owners, &[], &rows, &status, config).unwrap();
    let words = expected(config, &owners, &[], &[NormalContactRow::default(); 8]).len();
    for case in 0..6 {
        let mut bad = [NormalContactRow::default(); 8];
        bad[0] = row(0, NONE);
        let code = match case {
            0 => {
                bad[1] = row(1, NONE);
                bad[2] = row(2, NONE);
                CONTACT_ISLAND_FAULT_CAPACITY
            }
            1 => {
                bad[1] = row(0, NONE);
                bad[2] = row(0, NONE);
                CONTACT_ISLAND_FAULT_CAPACITY
            }
            2 => {
                bad[0].endpoints[2] = 100;
                CONTACT_ISLAND_FAULT_INPUT
            }
            3 => {
                bad[0].coefficients[0] = f32::NAN;
                CONTACT_ISLAND_FAULT_INPUT
            }
            4 => {
                bad[0].normal_force[3] = f32::INFINITY;
                CONTACT_ISLAND_FAULT_INPUT
            }
            _ => {
                bad[0].coefficients[3] = -1.0;
                CONTACT_ISLAND_FAULT_INPUT
            }
        };
        let mut valid = [NormalContactRow::default(); 8];
        valid[0] = row(0, NONE);
        let output = staging(device, words as u64 * 4 * 4);
        let mut encoder = device.create_command_encoder(&Default::default());
        for (step, frame) in [valid, bad, [NormalContactRow::default(); 8], valid]
            .iter()
            .enumerate()
        {
            if step == 0 || step == 3 {
                encoder.clear_buffer(&status, 0, None);
            }
            let source = buffer(device, queue, bytemuck::cast_slice(frame));
            encoder.copy_buffer_to_buffer(&source, 0, &rows, 0, rows.size());
            component.encode(&mut encoder);
            capture(
                &mut encoder,
                &component,
                &status,
                &output,
                step as u64 * words as u64 * 4,
            );
        }
        queue.submit([encoder.finish()]);
        let result = read(device, &output);
        let frames: Vec<_> = result.chunks_exact(words).collect();
        assert_eq!(frames[0], expected(config, &owners, &[], &valid));
        assert_eq!(frames[1][0], code, "case={case}");
        let mut retired = expected(config, &owners, &[], &[NormalContactRow::default(); 8]);
        retired[0] = code;
        assert_eq!(frames[2], retired, "case={case}");
        assert_eq!(frames[3], frames[0]);
        if case == 0 {
            assert_eq!(frames[1][1], 3);
        }
        if case == 1 {
            assert_eq!(frames[1][2], 3);
        }
    }
    for invalid in [
        ContactIslandConfig {
            row_capacity: 0,
            ..config
        },
        ContactIslandConfig {
            contacts_per_island: u32::MAX,
            ..config
        },
        ContactIslandConfig {
            articulation_count: u32::MAX,
            ..config
        },
    ] {
        assert!(
            ResidentContactIslands::new(&engine, &owners, &[], &rows, &status, invalid).is_err()
        );
    }
    assert!(ResidentContactIslands::new(&engine, &[9], &[], &rows, &status, config).is_err());
    assert!(
        ResidentContactIslands::new(&engine, &owners, &[[0, 9]], &rows, &status, config).is_err()
    );
    assert!(ResidentContactIslands::new(&engine, &owners, &[], &status, &status, config).is_err());
    let config = ContactIslandConfig {
        articulation_count: 0,
        row_capacity: 1,
        island_capacity: 1,
        contacts_per_island: 1,
    };
    let rows = buffer(device, queue, bytemuck::bytes_of(&row(NONE, NONE)));
    let component = ResidentContactIslands::new(&engine, &[], &[], &rows, &status, config).unwrap();
    let reference = expected(config, &[], &[], &[row(NONE, NONE)]);
    let output = staging(device, reference.len() as u64 * 4);
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.clear_buffer(&status, 0, None);
    component.encode(&mut encoder);
    capture(&mut encoder, &component, &status, &output, 0);
    queue.submit([encoder.finish()]);
    assert_eq!(read(device, &output), reference);
}
