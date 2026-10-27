#![cfg(feature = "gpu")]

use molla_kernels::{
    TireFemGpuInput, TireFemGpuParams, tire_fem_gpu_batch_atlas_buffers,
    tire_fem_gpu_batch_dispatch_plan, tire_fem_gpu_input_from_tetra_fem,
};
use molla_vehicle::{TetraTireFem, TetraTireFemParams};

fn fixture() -> TireFemGpuInput {
    tire_fem_gpu_input_from_tetra_fem(
        &TetraTireFem::new(0.0, 180_000.0, TetraTireFemParams::realtime_viewer()),
        TireFemGpuParams::default(),
    )
}

#[test]
fn tire_topology_preserves_real_mesh_and_arbitrary_adjacency_order() {
    let mut input = fixture();
    tire_fem_gpu_batch_dispatch_plan(&[input.clone()]).unwrap();
    for (offsets, entries) in [
        (&input.node_tet_offsets, &mut input.node_tet_entries),
        (&input.node_edge_offsets, &mut input.node_edge_entries),
        (
            &input.node_dihedral_offsets,
            &mut input.node_dihedral_entries,
        ),
    ] {
        for range in offsets.windows(2) {
            entries[range[0] as usize..range[1] as usize].reverse();
        }
    }
    tire_fem_gpu_batch_atlas_buffers(&[input.clone(), input]).unwrap();
}

#[test]
fn tire_topology_rejects_malformed_indices_and_incidence() {
    let input = fixture();
    let mutations: &[(&str, fn(&mut TireFemGpuInput))] = &[
        ("tet vertex", |i| i.tet_indices[0] = u32::MAX),
        ("tet repeated vertex", |i| {
            i.tet_indices[0] = i.tet_indices[1]
        }),
        ("tet offset start", |i| i.node_tet_offsets[0] = 1),
        ("tet offset decreasing", |i| {
            i.node_tet_offsets[1] = u32::MAX
        }),
        ("tet entry", |i| i.node_tet_entries[0] = u32::MAX),
        ("tet duplicate", |i| {
            i.node_tet_entries[1] = i.node_tet_entries[0]
        }),
        ("tet owner", |i| {
            let j = i.node_tet_offsets[1] as usize;
            i.node_tet_entries.swap(0, j);
        }),
        ("edge vertex", |i| i.xpbd_edge_indices[0][0] = u32::MAX),
        ("edge repeated vertex", |i| {
            i.xpbd_edge_indices[0][0] = i.xpbd_edge_indices[0][1]
        }),
        ("edge offset", |i| i.node_edge_offsets[1] = u32::MAX),
        ("edge entry", |i| i.node_edge_entries[0] = u32::MAX),
        ("edge duplicate", |i| {
            i.node_edge_entries[1] = i.node_edge_entries[0]
        }),
        ("edge missing incidence", |i| {
            i.node_edge_entries.pop();
            *i.node_edge_offsets.last_mut().unwrap() -= 1;
        }),
        ("dihedral vertex", |i| {
            i.xpbd_dihedral_indices[0][0] = u32::MAX
        }),
        ("dihedral repeated vertex", |i| {
            i.xpbd_dihedral_indices[0][0] = i.xpbd_dihedral_indices[0][1]
        }),
        ("dihedral offset", |i| i.node_dihedral_offsets[1] = u32::MAX),
        ("dihedral entry", |i| i.node_dihedral_entries[0] = u32::MAX),
        ("dihedral duplicate", |i| {
            i.node_dihedral_entries[1] = i.node_dihedral_entries[0]
        }),
        ("dihedral missing incidence", |i| {
            i.node_dihedral_entries.pop();
            *i.node_dihedral_offsets.last_mut().unwrap() -= 1;
        }),
        ("bead index", |i| i.bead_ids[0] = u32::MAX),
        ("bead duplicate", |i| i.bead_ids[1] = i.bead_ids[0]),
        ("tread index", |i| i.tread_ids[0] = u32::MAX),
        ("tread duplicate", |i| i.tread_ids[1] = i.tread_ids[0]),
    ];
    for (name, mutate) in mutations {
        let mut broken = input.clone();
        mutate(&mut broken);
        assert!(
            tire_fem_gpu_batch_dispatch_plan(&[broken.clone()]).is_err(),
            "{name}"
        );
        assert!(
            tire_fem_gpu_batch_atlas_buffers(&[input.clone(), broken]).is_err(),
            "{name}"
        );
    }
    println!(
        "{} malformed topologies rejected at both public boundaries",
        mutations.len()
    );
}

#[test]
fn tire_topology_accepts_absent_optional_constraints_and_isolated_nodes() {
    let mut input = fixture();
    input.xpbd_edge_indices.clear();
    input.xpbd_edge_params.clear();
    input.node_edge_offsets.clear();
    input.node_edge_entries.clear();
    input.xpbd_dihedral_indices.clear();
    input.xpbd_dihedral_params.clear();
    input.node_dihedral_offsets.clear();
    input.node_dihedral_entries.clear();
    input.positions.push([0.0; 4]);
    input.rest_positions.push([0.0; 4]);
    input.velocities.push([0.0; 4]);
    input.masses.push(1.0);
    input
        .node_tet_offsets
        .push(*input.node_tet_offsets.last().unwrap());
    tire_fem_gpu_batch_atlas_buffers(&[input]).unwrap();
}

#[test]
#[ignore = "requires a real GPU; rejects malformed topology before dispatch or upload"]
fn tire_topology_gpu_upload_boundaries_reject_invalid_mesh() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = molla_kernels::TireFemGpuKernel::new(device);
    let mut input = fixture();
    input.node_tet_entries[0] = u32::MAX;
    for result in [
        kernel.create_persistent(device, queue, &input).map(|_| ()),
        kernel.dispatch_blocking(device, queue, &input).map(|_| ()),
    ] {
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid tetrahedron topology")
        );
    }
}
