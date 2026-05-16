//! Stage 28.1 — GPU vs CPU parity for the contact reducer primitive.
//!
//! Compares `kernels::ContactReductionKernel` against
//! `geometry::reduce_contacts_per_pair` on a few canonical inputs.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_geometry::{ReducedContact, reduce_contacts_per_pair};
use molla_kernels::{
    ContactReductionKernel, MAX_OUT_PER_GROUP, MAX_PER_GROUP, pack_groups_host, unpack_output_host,
};
use molla_math::{Real, Vec3};

fn make_contact(a: u32, b: u32, id: u64, point: Vec3, depth: Real) -> ReducedContact {
    ReducedContact {
        a_shape: a,
        b_shape: b,
        id,
        point,
        normal: Vec3::Y,
        depth,
    }
}

/// Bin the flat contact slice by (a_shape, b_shape) — same key order
/// the CPU reducer uses (sort by key, then take the indices).
fn bin_by_pair(contacts: &[ReducedContact]) -> Vec<(u32, u32, Vec<usize>)> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
    for (i, c) in contacts.iter().enumerate() {
        groups.entry((c.a_shape, c.b_shape)).or_default().push(i);
    }
    groups.into_iter().map(|((a, b), v)| (a, b, v)).collect()
}

fn run_gpu(contacts: &[ReducedContact], max_per_pair: u32) -> Vec<ReducedContact> {
    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let groups = bin_by_pair(contacts);
    let n_groups = groups.len().max(1);

    let kernel = ContactReductionKernel::new(&device);
    let (mut in_contacts, mut in_count) =
        ContactReductionKernel::alloc_input(&device, &queue, n_groups).unwrap();
    let (mut out_contacts, mut out_count) =
        ContactReductionKernel::alloc_output(&device, &queue, n_groups).unwrap();

    let (host_in, host_count) = pack_groups_host(contacts, &groups);
    in_contacts.host_mut().unwrap()[..host_in.len()].copy_from_slice(&host_in);
    in_count.host_mut().unwrap()[..host_count.len()].copy_from_slice(&host_count);
    in_contacts.upload(&queue).unwrap();
    in_count.upload(&queue).unwrap();

    kernel
        .dispatch(
            &device,
            &queue,
            &in_contacts,
            &in_count,
            &mut out_contacts,
            &mut out_count,
            n_groups as u32,
            max_per_pair,
        )
        .unwrap();
    out_contacts.download(&device, &queue).unwrap();
    out_count.download(&device, &queue).unwrap();
    unpack_output_host(out_contacts.host().unwrap(), out_count.host().unwrap())
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_contact_reduction_matches_cpu_eight_corner_input() {
    // The same eight contacts the CPU test uses (4 corners + 4 mid-
    // points of a unit square at varying depths). Expect both
    // reducers to keep 4 contacts — the deepest first, then 3
    // spreading the patch.
    let cs = vec![
        // corners
        make_contact(0, 1, 1, Vec3::new(0.0, 0.0, 0.0), 0.010),
        make_contact(0, 1, 2, Vec3::new(1.0, 0.0, 0.0), 0.012),
        make_contact(0, 1, 3, Vec3::new(0.0, 0.0, 1.0), 0.011),
        make_contact(0, 1, 4, Vec3::new(1.0, 0.0, 1.0), 0.013),
        // midpoints
        make_contact(0, 1, 5, Vec3::new(0.5, 0.0, 0.0), 0.005),
        make_contact(0, 1, 6, Vec3::new(0.5, 0.0, 1.0), 0.005),
        make_contact(0, 1, 7, Vec3::new(0.0, 0.0, 0.5), 0.005),
        make_contact(0, 1, 8, Vec3::new(1.0, 0.0, 0.5), 0.005),
    ];

    let cpu = reduce_contacts_per_pair(&cs, 4);
    let gpu = run_gpu(&cs, 4);

    assert_eq!(cpu.len(), 4, "CPU should keep 4");
    assert_eq!(gpu.len(), 4, "GPU should keep 4");
    // Deepest contact (id=4) must come first on both.
    assert_eq!(cpu[0].id, 4);
    assert_eq!(gpu[0].id, 4);
    // Remaining set: ids must match (order may differ due to
    // tie-break). For this input the algorithm is deterministic with
    // distinct depths, so the order should match exactly.
    let cpu_ids: Vec<u64> = cpu.iter().map(|c| c.id).collect();
    let gpu_ids: Vec<u64> = gpu.iter().map(|c| c.id).collect();
    eprintln!("CPU ids = {cpu_ids:?}; GPU ids = {gpu_ids:?}");
    assert_eq!(cpu_ids, gpu_ids);
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_contact_reduction_preserves_deepest() {
    // One pair, eight contacts, one of which is the "king" deepest.
    let cs = vec![
        make_contact(0, 1, 1, Vec3::new(0.0, 0.0, 0.0), 0.01),
        make_contact(0, 1, 2, Vec3::new(1.0, 0.0, 0.0), 0.02),
        make_contact(0, 1, 3, Vec3::new(0.0, 0.0, 1.0), 0.03),
        make_contact(0, 1, 4, Vec3::new(1.0, 0.0, 1.0), 0.99), // king
        make_contact(0, 1, 5, Vec3::new(0.5, 0.0, 0.5), 0.04),
        make_contact(0, 1, 6, Vec3::new(0.2, 0.0, 0.7), 0.05),
        make_contact(0, 1, 7, Vec3::new(0.7, 0.0, 0.3), 0.06),
        make_contact(0, 1, 8, Vec3::new(0.4, 0.0, 0.1), 0.07),
    ];

    let gpu = run_gpu(&cs, 4);
    assert_eq!(gpu.len(), 4);
    assert_eq!(gpu[0].id, 4, "GPU must keep deepest first");
    assert!((gpu[0].depth - 0.99 as Real).abs() < 1e-5);
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_contact_reduction_two_independent_pairs() {
    // Two pairs, each capped at 1 → each pair should keep its
    // deepest contact. Output ordered by (a, b) key, matching CPU.
    let cs = vec![
        make_contact(0, 1, 1, Vec3::ZERO, 0.01),
        make_contact(2, 3, 2, Vec3::ZERO, 0.02),
        make_contact(0, 1, 3, Vec3::X, 0.03),
        make_contact(2, 3, 4, Vec3::Y, 0.04),
    ];
    let cpu = reduce_contacts_per_pair(&cs, 1);
    let gpu = run_gpu(&cs, 1);
    assert_eq!(cpu.len(), 2);
    assert_eq!(gpu.len(), 2);
    // CPU sorts by (a, b) — (0, 1) then (2, 3). GPU does too (we
    // pack groups via BTreeMap which iterates in key order).
    assert_eq!(cpu[0].id, 3);
    assert_eq!(gpu[0].id, 3);
    assert_eq!(cpu[1].id, 4);
    assert_eq!(gpu[1].id, 4);
    assert!(MAX_PER_GROUP >= 4);
    assert!(MAX_OUT_PER_GROUP >= 1);
}
