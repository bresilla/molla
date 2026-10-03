//! GPU tire-cell to DEM/granular particle impulse tests.
//!
//! Ignored GPU parity lane:
//! `cargo test -p molla-kernels --features gpu --test tire_dem_gpu -- --ignored`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    TireDemContactCell, TireDemFemTerrainOwner, TireDemGpuBuiltGridStepInput, TireDemGpuInput,
    TireDemGpuKernel, TireDemGpuRadixGridStepInput, TireDemResidentFemTerrainOwner,
    TireDemResidentTerrainParams, TireDemStepGpuInput, TireDemStepParams, TireFemContactTerrain,
    tire_dem_build_grid_step_input, tire_dem_build_sparse_lbvh,
    tire_dem_build_sparse_lbvh_leaf_rows_gpu_radix,
    tire_dem_build_sparse_lbvh_morton_order_gpu_radix, tire_dem_build_sparse_lbvh_step_input,
    tire_dem_build_sparse_lbvh_tree_rows_gpu_radix,
    tire_dem_build_unbounded_sorted_grid_step_input,
    tire_dem_build_unbounded_sorted_grid_step_input_gpu_radix, tire_dem_raster_surface_height_map,
    tire_dem_raster_surface_velocity_map, tire_dem_reference_grid_step,
    tire_dem_reference_impulses, tire_dem_reference_sparse_lbvh_step, tire_dem_reference_step,
    tire_dem_sparse_lbvh_dispatch_plan, tire_dem_sparse_lbvh_gpu_rows,
    tire_fem_contact_terrain_with_dem_surface_velocity,
};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn sample_input() -> TireDemGpuInput {
    TireDemGpuInput {
        positions: vec![
            [0.0, 0.10, 0.0, 0.0],
            [0.04, 0.10, 0.0, 0.0],
            [0.35, 0.10, 0.0, 0.0],
        ],
        velocities: vec![[0.0; 4]; 3],
        contacts: vec![TireDemContactCell {
            center_radius: [0.0, 0.10, 0.0, 0.12],
            force: [12.0, -90.0, 4.0, 0.0],
        }],
        grain_mass: 1.5,
        grain_radius: 0.05,
        dt: 1.0e-3,
        position_push_scale: 0.20,
    }
}

fn sample_step_input() -> TireDemStepGpuInput {
    TireDemStepGpuInput {
        positions: vec![
            [0.0, 0.10, 0.0, 0.0],
            [0.08, 0.10, 0.0, 0.0],
            [0.35, 0.03, 0.0, 0.0],
        ],
        velocities: vec![[0.0; 4], [0.0; 4], [0.10, 0.0, 0.0, 0.0]],
        contacts: vec![TireDemContactCell {
            center_radius: [0.0, 0.10, 0.0, 0.12],
            force: [12.0, -90.0, 4.0, 0.0],
        }],
        params: TireDemStepParams {
            grain_mass: 1.5,
            grain_radius: 0.05,
            normal_stiffness: 1.0e4,
            damping: 0.0,
            friction: 0.40,
            floor_y: 0.0,
            gravity: [0.0, -9.81, 0.0],
            dt: 1.0e-3,
            tire_position_push_scale: 0.20,
        },
    }
}

fn sample_grid_step_input() -> molla_kernels::TireDemGridStepGpuInput {
    let input = sample_step_input();
    tire_dem_build_grid_step_input(
        input.positions,
        input.velocities,
        input.contacts,
        input.params,
        [-0.20, -0.02, -0.20],
        0.12,
        [6, 3, 4],
    )
    .expect("grid build")
}

fn sample_gpu_built_grid_step_input() -> TireDemGpuBuiltGridStepInput {
    let input = sample_step_input();
    TireDemGpuBuiltGridStepInput {
        positions: input.positions,
        velocities: input.velocities,
        contacts: input.contacts,
        params: input.params,
        grid_origin: [-0.20, -0.02, -0.20],
        cell_size: 0.12,
        grid_shape: [6, 3, 4],
        max_grains_per_cell: 8,
    }
}

fn sample_gpu_radix_grid_step_input() -> TireDemGpuRadixGridStepInput {
    let input = sample_step_input();
    TireDemGpuRadixGridStepInput {
        positions: input.positions,
        velocities: input.velocities,
        contacts: input.contacts,
        params: input.params,
        grid_origin: [-0.20, -0.02, -0.20],
        cell_size: 0.12,
        grid_shape: [6, 3, 4],
    }
}

const SPARSE_LBVH_STRESS_X: usize = 14;
const SPARSE_LBVH_STRESS_Z: usize = 12;
const SPARSE_LBVH_STRESS_GRAINS_PER_TILE: usize = 5;

fn large_sparse_lbvh_stress_input() -> TireDemStepGpuInput {
    let mut positions = Vec::new();
    for ix in 0..SPARSE_LBVH_STRESS_X {
        for iz in 0..SPARSE_LBVH_STRESS_Z {
            let base_x = ix as f32 * 2.4;
            let base_z = iz as f32 * 2.1;
            let base_y = 0.10 + ((ix + iz) % 3) as f32 * 0.004;
            positions.push([base_x, base_y, base_z, 0.0]);
            positions.push([base_x + 0.082, base_y, base_z + 0.006, 0.0]);
            positions.push([base_x + 0.164, base_y, base_z - 0.004, 0.0]);
            positions.push([base_x + 0.041, base_y + 0.003, base_z + 0.088, 0.0]);
            positions.push([base_x + 0.124, base_y + 0.002, base_z + 0.052, 0.0]);
        }
    }
    assert_eq!(
        positions.len(),
        SPARSE_LBVH_STRESS_X * SPARSE_LBVH_STRESS_Z * SPARSE_LBVH_STRESS_GRAINS_PER_TILE
    );
    let velocities = positions
        .iter()
        .enumerate()
        .map(|(i, _)| {
            [
                if i % 2 == 0 { 0.04 } else { -0.02 },
                0.0,
                if i % 3 == 0 { 0.01 } else { -0.01 },
                0.0,
            ]
        })
        .collect::<Vec<_>>();
    TireDemStepGpuInput {
        positions,
        velocities,
        contacts: Vec::new(),
        params: TireDemStepParams {
            grain_mass: 1.5,
            grain_radius: 0.05,
            normal_stiffness: 1.0e4,
            damping: 0.0,
            friction: 0.35,
            floor_y: -1.0,
            gravity: [0.0; 3],
            dt: 1.0e-3,
            tire_position_push_scale: 0.0,
        },
    }
}

#[test]
fn tire_dem_reference_applies_localized_tire_impulses() {
    let input = sample_input();
    let out = tire_dem_reference_impulses(&input).expect("DEM reference");
    assert_eq!(out.positions.len(), input.positions.len());
    assert_eq!(out.velocities.len(), input.velocities.len());
    assert!(
        out.velocities[0][1] < 0.0 && out.positions[0][1] < input.positions[0][1],
        "grain under tire patch should receive downward impulse/push: {:?}",
        out
    );
    assert!(
        out.velocities[0][0] > 0.0 && out.velocities[0][2] > 0.0,
        "tangential tire force should transfer horizontal grain velocity: {:?}",
        out.velocities[0]
    );
    assert!(
        out.velocities[1][1] < 0.0,
        "nearby grain should also receive a smaller impulse"
    );
    assert_eq!(
        out.velocities[2], input.velocities[2],
        "far grain should not be affected by local tire contact"
    );
}

#[test]
fn tire_dem_reference_step_solves_grain_floor_and_tire_contacts() {
    let input = sample_step_input();
    let out = tire_dem_reference_step(&input).expect("DEM step reference");
    assert_eq!(out.positions.len(), input.positions.len());
    assert_eq!(out.velocities.len(), input.velocities.len());
    for (idx, (position, velocity)) in out.positions.iter().zip(out.velocities.iter()).enumerate() {
        for axis in 0..3 {
            assert!(
                position[axis].is_finite() && velocity[axis].is_finite(),
                "DEM step produced non-finite grain={idx} axis={axis}: p={position:?} v={velocity:?}"
            );
        }
    }
    assert!(
        out.velocities[0][0] < 0.0 && out.velocities[1][0] > 0.0,
        "overlapping grains should separate along their contact normal: {:?}",
        out.velocities
    );
    assert!(
        out.velocities[0][1] < input.velocities[0][1],
        "grain under tire patch should receive downward tire impulse: {:?}",
        out.velocities[0]
    );
    assert!(
        out.velocities[2][1] > input.velocities[2][1],
        "grain penetrating floor should receive upward floor impulse: {:?}",
        out.velocities[2]
    );
    assert!(
        out.velocities[2][0] < input.velocities[2][0],
        "floor Coulomb friction should reduce horizontal sliding speed: {:?}",
        out.velocities[2]
    );
}

#[test]
fn tire_dem_grid_reference_matches_dense_step_for_local_contacts() {
    let dense = sample_step_input();
    let grid = sample_grid_step_input();
    assert_eq!(grid.sorted_indices.len(), dense.positions.len());
    assert_eq!(
        grid.cell_ranges
            .iter()
            .map(|range| range[2] as usize)
            .sum::<usize>(),
        dense.positions.len(),
        "cell ranges should account for every grain exactly once"
    );
    let dense_out = tire_dem_reference_step(&dense).expect("dense reference");
    let grid_out = tire_dem_reference_grid_step(&grid).expect("grid reference");
    for (i, (a, b)) in grid_out
        .positions
        .iter()
        .zip(dense_out.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 1.0e-6,
                "grid position mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
    for (i, (a, b)) in grid_out
        .velocities
        .iter()
        .zip(dense_out.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 1.0e-6,
                "grid velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

#[test]
fn tire_dem_unbounded_sorted_grid_keeps_all_grains_in_dense_cell() {
    let grain_count = 12usize;
    let positions = (0..grain_count)
        .map(|i| [0.10 + i as f32 * 0.01, 0.10, 0.10, 0.0])
        .collect::<Vec<_>>();
    let velocities = vec![[0.0; 4]; grain_count];
    let params = TireDemStepParams {
        grain_mass: 1.5,
        grain_radius: 0.05,
        normal_stiffness: 0.0,
        damping: 0.0,
        friction: 0.0,
        floor_y: -1.0,
        gravity: [0.0; 3],
        dt: 1.0e-3,
        tire_position_push_scale: 0.0,
    };

    let (grid, build) = tire_dem_build_unbounded_sorted_grid_step_input(
        positions.clone(),
        velocities.clone(),
        Vec::new(),
        params,
        [0.0, 0.0, 0.0],
        1.0,
        [2, 2, 2],
    )
    .expect("unbounded sorted DEM grid build");

    assert_eq!(grid.sorted_indices.len(), grain_count);
    assert_eq!(build.sorted_indices.len(), grain_count);
    assert_eq!(build.sorted_cell_keys.len(), grain_count);
    assert_eq!(
        build
            .cell_ranges
            .iter()
            .map(|range| range[2] as usize)
            .sum::<usize>(),
        grain_count,
        "unbounded cell ranges should account for every grain exactly once"
    );
    assert_eq!(
        build.cell_ranges[0],
        [0, grain_count as u32, grain_count as u32, 0],
        "all sample grains should occupy the first cell without a capacity cap"
    );
    assert!(
        build.max_cell_occupancy > 4,
        "dense cells must expose overflow relative to the old fixed-capacity style, not silently drop grains: {build:?}"
    );

    let dense = TireDemStepGpuInput {
        positions,
        velocities,
        contacts: Vec::new(),
        params,
    };
    let dense_out = tire_dem_reference_step(&dense).expect("dense reference");
    let grid_out = tire_dem_reference_grid_step(&grid).expect("unbounded-grid reference");
    assert_eq!(grid_out.positions, dense_out.positions);
    assert_eq!(grid_out.velocities, dense_out.velocities);
}

#[test]
fn tire_dem_sparse_lbvh_builds_morton_sorted_sparse_tree() {
    let positions = (0..4)
        .flat_map(|ix| {
            (0..3).map(move |iz| {
                [
                    ix as f32 * 1.7,
                    0.10 + (ix % 2) as f32 * 0.02,
                    iz as f32 * 2.1,
                    0.0,
                ]
            })
        })
        .collect::<Vec<_>>();
    let lbvh = tire_dem_build_sparse_lbvh(&positions, 2).expect("sparse lbvh");
    assert_eq!(lbvh.sorted_indices.len(), positions.len());
    assert_eq!(lbvh.sorted_morton_codes.len(), positions.len());
    assert!(
        lbvh.nodes.len() > 1,
        "small leaf size should split sparse bed"
    );
    assert!(
        !lbvh.nodes[0].is_leaf(),
        "root should be an internal node for this sparse bed"
    );
    assert!(
        lbvh.sorted_morton_codes.windows(2).all(|w| w[0] <= w[1]),
        "Morton codes should be sorted for GPU-friendly linear traversal"
    );
    let leaf_grains = lbvh
        .nodes
        .iter()
        .filter(|node| node.is_leaf())
        .map(|node| node.count as usize)
        .sum::<usize>();
    assert_eq!(leaf_grains, positions.len());
    let rows = tire_dem_sparse_lbvh_gpu_rows(&lbvh);
    assert_eq!(rows.len(), lbvh.nodes.len() * 3);
    assert_eq!(rows[0][0], lbvh.nodes[0].min[0]);
    assert_eq!(rows[1][2], lbvh.nodes[0].max[2]);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_sparse_lbvh_morton_order_matches_cpu_oracle() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let positions = vec![
        [0.0, 0.10, 0.0, 0.0],
        [0.21, 0.18, 0.37, 0.0],
        [0.63, 0.14, 0.09, 0.0],
        [1.02, 0.26, 0.72, 0.0],
        [1.41, 0.31, 0.28, 0.0],
        [1.76, 0.22, 0.95, 0.0],
        [2.12, 0.42, 0.46, 0.0],
        [2.55, 0.35, 1.18, 0.0],
    ];
    let cpu = tire_dem_build_sparse_lbvh(&positions, 2).expect("CPU sparse LBVH oracle");
    let gpu = tire_dem_build_sparse_lbvh_morton_order_gpu_radix(device, queue, positions)
        .expect("GPU sparse-LBVH Morton order");

    assert_eq!(gpu.sorted_morton_codes, cpu.sorted_morton_codes);
    assert_eq!(gpu.sorted_indices, cpu.sorted_indices);
    assert_eq!(
        gpu.duplicate_morton_runs, 0,
        "fixture should avoid equal Morton keys so GPU radix order proves exact CPU tie-free order"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_sparse_lbvh_leaf_rows_match_cpu_sorted_leaf_chunks() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let positions = vec![
        [0.0, 0.10, 0.0, 0.0],
        [0.21, 0.18, 0.37, 0.0],
        [0.63, 0.14, 0.09, 0.0],
        [1.02, 0.26, 0.72, 0.0],
        [1.41, 0.31, 0.28, 0.0],
        [1.76, 0.22, 0.95, 0.0],
        [2.12, 0.42, 0.46, 0.0],
        [2.55, 0.35, 1.18, 0.0],
    ];
    let leaf_size = 2;
    let cpu = tire_dem_build_sparse_lbvh(&positions, leaf_size).expect("CPU sparse LBVH oracle");
    let gpu =
        tire_dem_build_sparse_lbvh_leaf_rows_gpu_radix(device, queue, positions.clone(), leaf_size)
            .expect("GPU sparse-LBVH leaf rows");

    assert_eq!(gpu.sorted_morton_codes, cpu.sorted_morton_codes);
    assert_eq!(gpu.sorted_indices, cpu.sorted_indices);
    assert_eq!(gpu.leaf_size, leaf_size as u32);
    assert_eq!(gpu.leaf_count, positions.len() / leaf_size);
    assert_eq!(gpu.leaf_node_rows.len(), gpu.leaf_count * 3);
    assert_eq!(
        gpu.duplicate_morton_runs, 0,
        "fixture should avoid equal Morton keys so sorted leaf chunks are stable"
    );

    for leaf in 0..gpu.leaf_count {
        let start = leaf * leaf_size;
        let end = (start + leaf_size).min(positions.len());
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for &grain_id in &gpu.sorted_indices[start..end] {
            let p = positions[grain_id as usize];
            for axis in 0..3 {
                min[axis] = min[axis].min(p[axis]);
                max[axis] = max[axis].max(p[axis]);
            }
        }
        let row0 = gpu.leaf_node_rows[leaf * 3];
        let row1 = gpu.leaf_node_rows[leaf * 3 + 1];
        let row2 = gpu.leaf_node_rows[leaf * 3 + 2];
        assert_eq!(row0[0..3], min);
        assert_eq!(row1[0..3], max);
        assert_eq!(row2[0], start as f32);
        assert_eq!(row2[1], (end - start) as f32);
        assert_eq!(row2[2], 1.0);
    }
}

fn complete_binary_lbvh_rows_from_sorted_chunks(
    positions: &[[f32; 4]],
    sorted_indices: &[u32],
    leaf_size: usize,
) -> Vec<[f32; 4]> {
    let leaf_count = positions.len().div_ceil(leaf_size);
    let padded_leaf_count = leaf_count.next_power_of_two();
    let node_count = padded_leaf_count * 2 - 1;
    let leaf_offset = padded_leaf_count - 1;
    let mut rows = vec![[0.0; 4]; node_count * 3];
    for leaf in 0..padded_leaf_count {
        let node = leaf_offset + leaf;
        if leaf >= leaf_count {
            rows[node * 3] = [f32::MAX, f32::MAX, f32::MAX, u32::MAX as f32];
            rows[node * 3 + 1] = [f32::MIN, f32::MIN, f32::MIN, u32::MAX as f32];
            rows[node * 3 + 2] = [positions.len() as f32, 0.0, 1.0, 0.0];
        } else {
            let start = leaf * leaf_size;
            let end = (start + leaf_size).min(positions.len());
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];
            for &grain_id in &sorted_indices[start..end] {
                let p = positions[grain_id as usize];
                for axis in 0..3 {
                    min[axis] = min[axis].min(p[axis]);
                    max[axis] = max[axis].max(p[axis]);
                }
            }
            rows[node * 3] = [min[0], min[1], min[2], u32::MAX as f32];
            rows[node * 3 + 1] = [max[0], max[1], max[2], u32::MAX as f32];
            rows[node * 3 + 2] = [start as f32, (end - start) as f32, 1.0, 0.0];
        }
    }
    let mut child_start = leaf_offset;
    let mut child_count = padded_leaf_count;
    while child_count > 1 {
        let parent_count = child_count / 2;
        let parent_start = child_start - parent_count;
        for parent_id in 0..parent_count {
            let parent = parent_start + parent_id;
            let left = child_start + parent_id * 2;
            let right = left + 1;
            let left_min = rows[left * 3];
            let left_max = rows[left * 3 + 1];
            let right_min = rows[right * 3];
            let right_max = rows[right * 3 + 1];
            rows[parent * 3] = [
                left_min[0].min(right_min[0]),
                left_min[1].min(right_min[1]),
                left_min[2].min(right_min[2]),
                left as f32,
            ];
            rows[parent * 3 + 1] = [
                left_max[0].max(right_max[0]),
                left_max[1].max(right_max[1]),
                left_max[2].max(right_max[2]),
                right as f32,
            ];
            rows[parent * 3 + 2] = [0.0, 0.0, 0.0, 0.0];
        }
        child_start = parent_start;
        child_count = parent_count;
    }
    rows
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_sparse_lbvh_tree_rows_match_cpu_complete_binary_oracle() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let positions = vec![
        [0.0, 0.10, 0.0, 0.0],
        [0.21, 0.18, 0.37, 0.0],
        [0.63, 0.14, 0.09, 0.0],
        [1.02, 0.26, 0.72, 0.0],
        [1.41, 0.31, 0.28, 0.0],
        [1.76, 0.22, 0.95, 0.0],
        [2.12, 0.42, 0.46, 0.0],
        [2.55, 0.35, 1.18, 0.0],
    ];
    let leaf_size = 2;
    let cpu = tire_dem_build_sparse_lbvh(&positions, leaf_size).expect("CPU sparse LBVH oracle");
    let gpu =
        tire_dem_build_sparse_lbvh_tree_rows_gpu_radix(device, queue, positions.clone(), leaf_size)
            .expect("GPU sparse-LBVH tree rows");
    let expected =
        complete_binary_lbvh_rows_from_sorted_chunks(&positions, &cpu.sorted_indices, leaf_size);

    assert_eq!(gpu.sorted_morton_codes, cpu.sorted_morton_codes);
    assert_eq!(gpu.sorted_indices, cpu.sorted_indices);
    assert_eq!(gpu.leaf_count, 4);
    assert_eq!(gpu.leaf_node_offset, 3);
    assert_eq!(gpu.node_count, 7);
    assert_eq!(gpu.node_rows, expected);
    assert_eq!(
        gpu.duplicate_morton_runs, 0,
        "fixture should avoid equal Morton keys so complete GPU tree rows are stable"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_sparse_lbvh_tree_rows_pad_non_power_of_two_leaves() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let positions = vec![
        [0.0, 0.10, 0.0, 0.0],
        [0.21, 0.18, 0.37, 0.0],
        [0.63, 0.14, 0.09, 0.0],
        [1.02, 0.26, 0.72, 0.0],
        [1.41, 0.31, 0.28, 0.0],
        [1.76, 0.22, 0.95, 0.0],
        [2.12, 0.42, 0.46, 0.0],
        [2.55, 0.35, 1.18, 0.0],
        [3.05, 0.38, 0.16, 0.0],
        [3.42, 0.24, 1.42, 0.0],
    ];
    let leaf_size = 2;
    let cpu = tire_dem_build_sparse_lbvh(&positions, leaf_size).expect("CPU sparse LBVH oracle");
    let gpu =
        tire_dem_build_sparse_lbvh_tree_rows_gpu_radix(device, queue, positions.clone(), leaf_size)
            .expect("GPU sparse-LBVH padded tree rows");
    let expected =
        complete_binary_lbvh_rows_from_sorted_chunks(&positions, &cpu.sorted_indices, leaf_size);

    assert_eq!(gpu.sorted_morton_codes, cpu.sorted_morton_codes);
    assert_eq!(gpu.sorted_indices, cpu.sorted_indices);
    assert_eq!(gpu.leaf_count, 5);
    assert_eq!(gpu.padded_leaf_count, 8);
    assert_eq!(gpu.leaf_node_offset, 7);
    assert_eq!(gpu.node_count, 15);
    assert_eq!(gpu.node_rows, expected);
    assert_eq!(gpu.node_rows[(gpu.leaf_node_offset + 5) * 3 + 2][1], 0.0);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_built_lbvh_tree_rows_feed_resident_sparse_step() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireDemGpuKernel::new(&device);
    let mut input = sample_step_input();
    input.positions = vec![
        [0.0, 0.10, 0.0, 0.0],
        [0.08, 0.10, 0.0, 0.0],
        [2.0, 0.10, 0.0, 0.0],
        [2.08, 0.10, 0.0, 0.0],
        [5.0, 0.03, 5.0, 0.0],
        [5.08, 0.03, 5.0, 0.0],
        [8.0, 0.10, -2.0, 0.0],
        [8.08, 0.10, -2.0, 0.0],
    ];
    input.velocities = vec![[0.0; 4]; input.positions.len()];
    input.contacts.clear();
    input.params.floor_y = -1.0;
    input.params.gravity = [0.0; 3];
    let expected = tire_dem_reference_step(&input).expect("dense reference");
    let tree = tire_dem_build_sparse_lbvh_tree_rows_gpu_radix(
        device.clone(),
        queue.clone(),
        input.positions.clone(),
        2,
    )
    .expect("GPU-built sparse-LBVH tree rows");
    let state = kernel
        .create_resident_state(&device, &queue, &input.positions, &input.velocities)
        .expect("resident DEM state");
    let resident_lbvh = kernel
        .upload_resident_lbvh_tree_rows(&device, &queue, input.positions.len(), &tree)
        .expect("resident topology from GPU-built sparse-LBVH rows");
    assert_eq!(resident_lbvh.node_count, tree.node_count);
    assert_eq!(resident_lbvh.node_row_count, tree.node_rows.len());
    assert_eq!(resident_lbvh.sorted_index_count, input.positions.len());

    let resident = kernel
        .dispatch_lbvh_step_resident_state_with_resident_lbvh(
            &device,
            &queue,
            &state,
            &input.contacts,
            input.params,
            &resident_lbvh,
        )
        .expect("resident DEM step using GPU-built sparse-LBVH rows");
    let got = kernel
        .readback_resident_output(&device, &queue, &resident)
        .expect("readback resident DEM from GPU-built LBVH rows");
    for (i, (a, b)) in got
        .velocities
        .iter()
        .zip(expected.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 1.0e-4,
                "GPU-built resident LBVH velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

fn test_bounds(positions: &[[f32; 4]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [positions[0][0], positions[0][1], positions[0][2]];
    let mut max = min;
    for p in &positions[1..] {
        for axis in 0..3 {
            min[axis] = min[axis].min(p[axis]);
            max[axis] = max[axis].max(p[axis]);
        }
    }
    (
        min,
        [
            (max[0] - min[0]).max(1.0e-9),
            (max[1] - min[1]).max(1.0e-9),
            (max[2] - min[2]).max(1.0e-9),
        ],
    )
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_resident_gpu_lbvh_builder_feeds_sparse_step_without_tree_readback() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireDemGpuKernel::new(&device);
    let mut input = sample_step_input();
    input.positions = vec![
        [0.0, 0.10, 0.0, 0.0],
        [0.08, 0.10, 0.0, 0.0],
        [2.0, 0.10, 0.0, 0.0],
        [2.08, 0.10, 0.0, 0.0],
        [5.0, 0.03, 5.0, 0.0],
        [5.08, 0.03, 5.0, 0.0],
        [8.0, 0.10, -2.0, 0.0],
        [8.08, 0.10, -2.0, 0.0],
        [10.0, 0.12, 3.0, 0.0],
        [10.08, 0.12, 3.0, 0.0],
    ];
    input.velocities = vec![[0.0; 4]; input.positions.len()];
    input.contacts.clear();
    input.params.floor_y = -1.0;
    input.params.gravity = [0.0; 3];
    let expected = tire_dem_reference_step(&input).expect("dense reference");
    let (scene_min, scene_extent) = test_bounds(&input.positions);
    let state = kernel
        .create_resident_state(&device, &queue, &input.positions, &input.velocities)
        .expect("resident DEM state");
    let resident_lbvh = kernel
        .build_resident_sparse_lbvh_tree_rows_gpu_radix(
            &device,
            &queue,
            &state,
            scene_min,
            scene_extent,
            2,
        )
        .expect("no-readback resident GPU sparse-LBVH builder");
    assert_eq!(resident_lbvh.node_count, 15);
    assert_eq!(resident_lbvh.node_row_count, 45);
    assert_eq!(resident_lbvh.sorted_index_count, input.positions.len());

    let resident = kernel
        .dispatch_lbvh_step_resident_state_with_resident_lbvh(
            &device,
            &queue,
            &state,
            &input.contacts,
            input.params,
            &resident_lbvh,
        )
        .expect("resident DEM step using no-readback GPU-built sparse-LBVH rows");
    let got = kernel
        .readback_resident_output(&device, &queue, &resident)
        .expect("readback resident DEM after no-readback LBVH build");
    for (i, (a, b)) in got
        .velocities
        .iter()
        .zip(expected.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 1.0e-4,
                "no-readback GPU-built resident LBVH velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

#[test]
fn tire_dem_surface_velocity_map_exports_top_layer_velocity_for_fem_terrain() {
    let positions = vec![
        [0.10, 0.00, 0.10, 0.0],
        [0.10, 0.24, 0.10, 0.0],
        [0.62, 0.20, 0.10, 0.0],
        [0.68, 0.22, 0.12, 0.0],
    ];
    let velocities = vec![
        [9.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
    ];
    let map =
        tire_dem_raster_surface_velocity_map(&positions, &velocities, [0.0, 0.0], 0.5, 2, 1, 0.025)
            .expect("surface velocity map");
    assert_eq!(map.rows.len(), 2);
    assert_eq!(map.active_cells, 2);
    assert_eq!(
        map.rows[0],
        [1.0, 0.0, 0.0, 1.0],
        "buried grains must not dominate the tire-visible surface velocity"
    );
    assert_eq!(
        map.rows[1],
        [0.0, 0.0, 3.0, 2.0],
        "near-surface grains in the same DEM cell should average into one FEM terrain velocity row"
    );
    assert!(map.max_speed_m_per_s >= 3.0 && map.mean_speed_m_per_s > 1.0);

    let terrain = TireFemContactTerrain {
        origin: map.origin,
        spacing: map.spacing,
        nx: map.nx,
        nz: map.nz,
        heights: vec![0.0; map.rows.len()],
        friction: 0.55,
        friction_map: None,
        water_film_map: None,
        sinkage_map: Some(vec![0.0; map.rows.len()]),
        soil_rolling_resistance_scale: 1.0,
        surface_velocity: [0.0, 0.0, 0.0],
        surface_velocity_map: None,
        sdf_primitives: Vec::new(),
        triangles: Vec::new(),
    };
    let terrain = tire_fem_contact_terrain_with_dem_surface_velocity(terrain, &map)
        .expect("DEM velocity rows should attach to matching FEM terrain");
    assert_eq!(
        terrain.surface_velocity_map.as_ref().unwrap(),
        &map.rows,
        "DEM surface velocity rows should be handed to FEM contact terrain without viewer-local slip math"
    );
}

#[test]
fn tire_dem_fem_terrain_owner_exports_height_sinkage_and_surface_velocity() {
    let positions0 = vec![
        [0.10, 0.18, 0.10, 0.0],
        [0.12, 0.08, 0.10, 0.0],
        [0.62, 0.16, 0.10, 0.0],
    ];
    let velocities0 = vec![[0.0; 4], [0.0; 4], [0.0; 4]];
    let heights0 =
        tire_dem_raster_surface_height_map(&positions0, [0.0, 0.0], 0.5, 2, 1, 0.0, 0.05)
            .expect("DEM surface height map");
    assert_eq!(heights0, vec![0.23, 0.21]);

    let mut owner = TireDemFemTerrainOwner::from_grain_state(
        [0.0, 0.0],
        0.5,
        2,
        1,
        0.0,
        0.62,
        0.05,
        &positions0,
        &velocities0,
    )
    .expect("DEM FEM terrain owner");
    assert_eq!(owner.heights(), heights0.as_slice());
    assert_eq!(owner.sinkage(), &[0.0, 0.0]);

    let positions1 = vec![
        [0.10, 0.10, 0.10, 0.0],
        [0.12, -0.06, 0.10, 0.0],
        [0.62, 0.16, 0.10, 0.0],
    ];
    let velocities1 = vec![
        [1.5, 0.0, 0.0, 0.0],
        [9.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -2.0, 0.0],
    ];
    owner
        .update_grain_state(&positions1, &velocities1)
        .expect("DEM owner update");
    assert!(
        owner.sinkage()[0] > 0.075 && owner.sinkage()[1] == 0.0,
        "lowered first top layer should export FEM sinkage without sinking unchanged cells: {:?}",
        owner.sinkage()
    );
    assert_eq!(
        owner.surface_velocity_map().rows[0],
        [1.5, 0.0, 0.0, 1.0],
        "surface velocity should come from the visible top layer, not the buried grain"
    );
    let terrain = owner
        .fem_contact_terrain()
        .expect("DEM terrain should bridge to FEM contact terrain");
    assert_eq!(terrain.heights, owner.heights());
    assert_eq!(terrain.sinkage_map.as_ref().unwrap(), owner.sinkage());
    assert_eq!(
        terrain.surface_velocity_map.as_ref().unwrap(),
        &owner.surface_velocity_map().rows
    );
}

#[test]
fn tire_dem_fem_terrain_owner_steps_owned_sparse_lbvh_state() {
    let positions = vec![
        [0.10, 0.10, 0.10, 0.0],
        [0.19, 0.10, 0.10, 0.0],
        [0.62, 0.12, 0.10, 0.0],
        [0.68, 0.12, 0.12, 0.0],
    ];
    let velocities = vec![[0.0; 4]; positions.len()];
    let mut owner = TireDemFemTerrainOwner::from_grain_state(
        [0.0, 0.0],
        0.25,
        4,
        2,
        0.0,
        0.65,
        0.05,
        &positions,
        &velocities,
    )
    .expect("owned DEM terrain");
    let contacts = vec![TireDemContactCell {
        center_radius: [0.12, 0.10, 0.10, 0.16],
        force: [8.0, -120.0, 2.0, 0.0],
    }];
    let report = owner
        .step_sparse_lbvh_reference(
            &contacts,
            TireDemStepParams {
                grain_mass: 1.5,
                grain_radius: 0.05,
                normal_stiffness: 1.0e4,
                damping: 0.0,
                friction: 0.55,
                floor_y: 0.0,
                gravity: [0.0, -9.81, 0.0],
                dt: 1.0e-3,
                tire_position_push_scale: 0.20,
            },
        )
        .expect("owned sparse-LBVH step");
    assert_eq!(report.grain_count, positions.len());
    assert_eq!(report.contact_count, contacts.len());
    assert!(report.lbvh_node_count >= report.lbvh_leaf_count);
    assert!(report.max_position_delta_y_m > 0.0);
    assert!(
        report.active_velocity_cells > 0 && report.max_surface_speed_m_per_s > 0.0,
        "owned DEM step should refresh FEM terrain velocity rows: {report:?}"
    );
    assert!(
        report.max_sinkage_m >= 0.0
            && owner
                .fem_contact_terrain()
                .expect("FEM terrain after owned DEM step")
                .surface_velocity_map
                .as_ref()
                .is_some_and(|rows| rows.len() == 8)
    );
    assert_ne!(
        owner.positions(),
        positions.as_slice(),
        "owned DEM terrain step should update the source-of-truth grain state"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_fem_terrain_owner_gpu_sparse_lbvh_step_matches_reference() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let positions = vec![
        [0.10, 0.10, 0.10, 0.0],
        [0.19, 0.10, 0.10, 0.0],
        [0.62, 0.12, 0.10, 0.0],
        [0.68, 0.12, 0.12, 0.0],
        [0.72, 0.18, 0.16, 0.0],
        [0.16, 0.16, 0.14, 0.0],
    ];
    let velocities = vec![[0.0; 4]; positions.len()];
    let mut cpu_owner = TireDemFemTerrainOwner::from_grain_state(
        [0.0, 0.0],
        0.25,
        4,
        2,
        0.0,
        0.65,
        0.05,
        &positions,
        &velocities,
    )
    .expect("CPU owner");
    let mut gpu_owner = cpu_owner.clone();
    let contacts = vec![TireDemContactCell {
        center_radius: [0.14, 0.10, 0.11, 0.18],
        force: [10.0, -145.0, 4.0, 0.0],
    }];
    let params = TireDemStepParams {
        grain_mass: 1.5,
        grain_radius: 0.05,
        normal_stiffness: 1.0e4,
        damping: 0.0,
        friction: 0.55,
        floor_y: 0.0,
        gravity: [0.0, -9.81, 0.0],
        dt: 1.0e-3,
        tire_position_push_scale: 0.20,
    };
    let cpu_report = cpu_owner
        .step_sparse_lbvh_reference(&contacts, params)
        .expect("CPU owner step");
    let gpu_report = gpu_owner
        .step_sparse_lbvh_gpu_blocking(
            &device,
            &queue,
            &TireDemGpuKernel::new(&device),
            &contacts,
            params,
            8,
        )
        .expect("GPU owner step");
    assert_eq!(gpu_report.grain_count, cpu_report.grain_count);
    assert_eq!(gpu_report.contact_count, cpu_report.contact_count);
    assert_eq!(gpu_report.lbvh_node_count, cpu_report.lbvh_node_count);
    assert_eq!(
        gpu_report.active_velocity_cells,
        cpu_report.active_velocity_cells
    );
    for (i, (gpu, cpu)) in gpu_owner
        .positions()
        .iter()
        .zip(cpu_owner.positions().iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (gpu[axis] - cpu[axis]).abs() < 2.0e-5,
                "owner GPU position mismatch grain={i} axis={axis}: got={} expected={}",
                gpu[axis],
                cpu[axis]
            );
        }
    }
    for (i, (gpu, cpu)) in gpu_owner
        .velocities()
        .iter()
        .zip(cpu_owner.velocities().iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (gpu[axis] - cpu[axis]).abs() < 2.0e-5,
                "owner GPU velocity mismatch grain={i} axis={axis}: got={} expected={}",
                gpu[axis],
                cpu[axis]
            );
        }
    }
    assert_eq!(gpu_owner.heights(), cpu_owner.heights());
    assert_eq!(gpu_owner.sinkage(), cpu_owner.sinkage());
    // The surface velocity map derives from the SAME f32 dynamics the
    // position/velocity checks above compare at 2e-5 — demanding
    // bitwise CPU-vs-GPU equality here was over-strict (FMA and
    // accumulation-order differences legitimately shift the last few
    // ULPs; observed drift ≤ 5e-9, three orders inside the physics
    // tolerance).
    let gpu_rows = &gpu_owner.surface_velocity_map().rows;
    let cpu_rows = &cpu_owner.surface_velocity_map().rows;
    assert_eq!(gpu_rows.len(), cpu_rows.len());
    for (i, (g, c)) in gpu_rows.iter().zip(cpu_rows.iter()).enumerate() {
        for k in 0..4 {
            assert!(
                (g[k] - c[k]).abs() < 2.0e-5,
                "surface velocity map mismatch row={i} k={k}: got={} expected={}",
                g[k],
                c[k]
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_resident_state_sparse_lbvh_step_ping_pongs_without_reupload() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireDemGpuKernel::new(&device);
    let mut input = sample_step_input();
    input.positions = vec![
        [0.0, 0.10, 0.0, 0.0],
        [0.08, 0.10, 0.0, 0.0],
        [1.2, 0.10, 0.4, 0.0],
        [1.28, 0.10, 0.4, 0.0],
        [-0.9, 0.12, -0.5, 0.0],
        [-0.82, 0.12, -0.5, 0.0],
    ];
    input.velocities = vec![[0.0; 4]; input.positions.len()];
    input.contacts = vec![TireDemContactCell {
        center_radius: [0.02, 0.10, 0.0, 0.16],
        force: [12.0, -90.0, 4.0, 0.0],
    }];
    input.params.floor_y = -1.0;
    input.params.gravity = [0.0; 3];

    let expected1 = tire_dem_reference_sparse_lbvh_step(&input)
        .expect("first CPU sparse step")
        .0;
    let input2 = TireDemStepGpuInput {
        positions: expected1.positions.clone(),
        velocities: expected1.velocities.clone(),
        contacts: input.contacts.clone(),
        params: input.params,
    };
    let expected2 = tire_dem_reference_sparse_lbvh_step(&input2)
        .expect("second CPU sparse step")
        .0;

    let mut state = kernel
        .create_resident_state(&device, &queue, &input.positions, &input.velocities)
        .expect("resident DEM state");
    let lbvh1 = tire_dem_build_sparse_lbvh(&input.positions, 8).expect("first LBVH");
    let out1 = kernel
        .dispatch_lbvh_step_resident_state(
            &device,
            &queue,
            &state,
            &input.contacts,
            input.params,
            &lbvh1,
        )
        .expect("first resident state step");
    assert_eq!(out1.grain_count, input.positions.len());
    state = out1.into_state();
    let lbvh2 = tire_dem_build_sparse_lbvh(&expected1.positions, 8).expect("second LBVH");
    let out2 = kernel
        .dispatch_lbvh_step_resident_state(
            &device,
            &queue,
            &state,
            &input.contacts,
            input.params,
            &lbvh2,
        )
        .expect("second resident state step");
    state = out2.into_state();
    let got = kernel
        .readback_resident_state(&device, &queue, &state)
        .expect("resident state readback seam");
    for (i, (gpu, cpu)) in got
        .positions
        .iter()
        .zip(expected2.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (gpu[axis] - cpu[axis]).abs() < 3.0e-5,
                "resident ping-pong position mismatch grain={i} axis={axis}: got={} expected={}",
                gpu[axis],
                cpu[axis]
            );
        }
    }
    for (i, (gpu, cpu)) in got
        .velocities
        .iter()
        .zip(expected2.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (gpu[axis] - cpu[axis]).abs() < 3.0e-5,
                "resident ping-pong velocity mismatch grain={i} axis={axis}: got={} expected={}",
                gpu[axis],
                cpu[axis]
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_resident_terrain_raster_matches_cpu_export_without_grain_readback() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireDemGpuKernel::new(&device);
    let positions = vec![
        [0.10, 0.00, 0.10, 0.0],
        [0.10, 0.24, 0.10, 0.0],
        [0.62, 0.20, 0.10, 0.0],
        [0.68, 0.22, 0.12, 0.0],
    ];
    let velocities = vec![
        [9.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
    ];
    let params = TireDemResidentTerrainParams {
        origin: [0.0, 0.0],
        spacing: 0.5,
        nx: 2,
        nz: 1,
        floor_y: 0.0,
        grain_radius: 0.025,
    };
    let reference_heights = vec![0.40, 0.30];
    let expected_heights = tire_dem_raster_surface_height_map(
        &positions,
        params.origin,
        params.spacing,
        params.nx,
        params.nz,
        params.floor_y,
        params.grain_radius,
    )
    .expect("CPU DEM height raster");
    let expected_velocity = tire_dem_raster_surface_velocity_map(
        &positions,
        &velocities,
        params.origin,
        params.spacing,
        params.nx,
        params.nz,
        params.grain_radius,
    )
    .expect("CPU DEM surface velocity raster");
    let expected_sinkage = reference_heights
        .iter()
        .zip(expected_heights.iter())
        .map(|(&reference, &height)| (reference - height).max(0.0))
        .collect::<Vec<_>>();

    let state = kernel
        .create_resident_state(&device, &queue, &positions, &velocities)
        .expect("resident DEM state");
    let resident = kernel
        .dispatch_resident_terrain_raster(&device, &queue, &state, &reference_heights, params)
        .expect("resident DEM terrain raster");
    let fem_gpu_terrain = resident
        .fem_contact_feedback_gpu_terrain(0.62, 1.0)
        .expect("resident DEM terrain should expose FEM GPU terrain buffers");
    assert_eq!(fem_gpu_terrain.nx, params.nx);
    assert!(fem_gpu_terrain.sinkage_map.is_some());
    assert!(fem_gpu_terrain.surface_velocity_map.is_some());
    let got = kernel
        .readback_resident_terrain(&device, &queue, &resident)
        .expect("resident terrain readback seam");

    assert_eq!(got.heights, expected_heights);
    assert_eq!(got.sinkage, expected_sinkage);
    // Averaged velocities (velocity_sum/count on device) are the same
    // fragility class as the physics-stepped map above: bitwise
    // GPU-vs-CPU only holds while the fixture's averages are exact in
    // f32. Compare at the shared 2e-5 physics tolerance instead.
    assert_eq!(
        got.surface_velocity_map.rows.len(),
        expected_velocity.rows.len()
    );
    for (i, (g, c)) in got
        .surface_velocity_map
        .rows
        .iter()
        .zip(expected_velocity.rows.iter())
        .enumerate()
    {
        for k in 0..4 {
            assert!(
                (g[k] - c[k]).abs() < 2.0e-5,
                "terrain velocity map mismatch row={i} k={k}: got={} expected={}",
                g[k],
                c[k]
            );
        }
    }
    assert_eq!(
        got.surface_velocity_map.active_cells,
        expected_velocity.active_cells
    );
    assert!(
        (got.surface_velocity_map.max_speed_m_per_s - expected_velocity.max_speed_m_per_s).abs()
            < 2.0e-5,
        "max_speed mismatch: got={} expected={}",
        got.surface_velocity_map.max_speed_m_per_s,
        expected_velocity.max_speed_m_per_s
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_resident_fem_terrain_owner_steps_and_keeps_gpu_terrain() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireDemGpuKernel::new(&device);
    let mut input = sample_step_input();
    input.positions = vec![
        [0.0, 0.10, 0.0, 0.0],
        [0.08, 0.10, 0.0, 0.0],
        [0.35, 0.10, 0.0, 0.0],
        [0.43, 0.11, 0.0, 0.0],
    ];
    input.velocities = vec![[0.0; 4]; input.positions.len()];
    let terrain_params = TireDemResidentTerrainParams {
        origin: [-0.2, -0.2],
        spacing: 0.25,
        nx: 4,
        nz: 2,
        floor_y: 0.0,
        grain_radius: input.params.grain_radius,
    };
    let mut owner = TireDemResidentFemTerrainOwner::from_grain_state(
        &device,
        &queue,
        &kernel,
        terrain_params,
        &input.positions,
        &input.velocities,
    )
    .expect("resident DEM FEM terrain owner");
    assert_eq!(owner.resident_state().grain_count, input.positions.len());
    assert_eq!(owner.reference_heights().len(), 8);
    assert!(owner.resident_lbvh().is_none());
    let (cached_min, cached_extent) = owner.lbvh_scene_bounds();
    assert_eq!(owner.reseed_count(), 0);
    assert!(
        cached_extent.iter().all(|v| v.is_finite() && *v > 0.0),
        "resident owner should cache finite positive Morton bounds"
    );
    assert!(
        cached_min[0] <= input.positions[0][0] && cached_min[2] <= input.positions[0][2],
        "cached DEM owner bounds should cover initial grains and terrain grid"
    );
    let report = owner
        .step_sparse_lbvh_gpu_mirrored(&device, &queue, &kernel, &input.contacts, input.params, 8)
        .expect("resident owner sparse-LBVH step");
    assert_eq!(report.grain_count, input.positions.len());
    assert!(report.lbvh_node_count >= report.lbvh_leaf_count);
    let resident_lbvh = owner
        .resident_lbvh()
        .expect("resident owner should retain GPU-built sparse-LBVH topology buffers");
    assert_eq!(resident_lbvh.grain_count, input.positions.len());
    assert_eq!(resident_lbvh.node_count, report.lbvh_node_count);
    assert_eq!(resident_lbvh.node_row_count, report.lbvh_node_count * 3);
    assert_eq!(resident_lbvh.sorted_index_count, input.positions.len());
    assert_eq!(resident_lbvh.leaf_size, 8);
    assert!(resident_lbvh.node_row_capacity >= resident_lbvh.node_row_count);
    assert!(resident_lbvh.sorted_index_capacity >= resident_lbvh.sorted_index_count);
    assert_eq!(resident_lbvh.topology_revision, 0);
    assert!(
        report.active_velocity_cells > 0 && report.max_surface_speed_m_per_s > 0.0,
        "resident owner should mirror terrain telemetry from GPU raster: {report:?}"
    );
    let gpu_terrain = owner
        .resident_terrain()
        .fem_contact_feedback_gpu_terrain(0.6, 1.0)
        .expect("resident owner terrain should feed FEM GPU terrain");
    assert_eq!(gpu_terrain.nx, terrain_params.nx);
    assert!(gpu_terrain.sinkage_map.is_some());
    assert!(gpu_terrain.surface_velocity_map.is_some());
    assert_ne!(
        owner.positions(),
        input.positions.as_slice(),
        "resident owner step should refresh explicit CPU mirror after GPU stepping"
    );
    let report2 = owner
        .step_sparse_lbvh_gpu_mirrored(&device, &queue, &kernel, &input.contacts, input.params, 8)
        .expect("resident owner should rebuild LBVH topology from resident GPU state");
    let resident_lbvh2 = owner
        .resident_lbvh()
        .expect("resident owner should keep resident sparse-LBVH topology buffers after GPU build");
    assert_eq!(resident_lbvh2.grain_count, input.positions.len());
    assert_eq!(resident_lbvh2.node_count, report2.lbvh_node_count);
    assert_eq!(resident_lbvh2.node_row_count, report2.lbvh_node_count * 3);
    assert_eq!(resident_lbvh2.sorted_index_count, input.positions.len());
    assert_eq!(resident_lbvh2.leaf_size, 8);
    assert!(resident_lbvh2.node_row_capacity >= resident_lbvh2.node_row_count);
    assert!(resident_lbvh2.sorted_index_capacity >= resident_lbvh2.sorted_index_count);
    assert_eq!(
        resident_lbvh2.topology_revision, 0,
        "owner steps should use the no-readback GPU topology builder, not the CPU in-place refresh"
    );
    assert_eq!(
        owner.reseed_count(),
        0,
        "regular GPU owner stepping should not masquerade as a reset/reseed"
    );
}

#[test]
fn tire_dem_resident_fem_owner_uses_no_readback_gpu_lbvh_builder_source_guard() {
    let src = include_str!("../src/tire_dem.rs");
    let owner_step = src
        .split("pub fn step_sparse_lbvh_gpu_mirrored")
        .nth(1)
        .expect("resident DEM/FEM terrain owner step should exist")
        .split("pub fn resident_state")
        .next()
        .expect("resident owner step body should precede accessors");
    assert!(
        owner_step.contains("build_resident_sparse_lbvh_tree_rows_gpu_radix"),
        "resident DEM/FEM owner should build sparse-LBVH topology from resident GPU buffers"
    );
    assert!(
        owner_step.contains("self.lbvh_scene_min") && owner_step.contains("self.lbvh_scene_extent"),
        "owner should use cached Morton bounds instead of deriving topology bounds from CPU mirrors"
    );
    assert!(
        !owner_step.contains("tire_dem_build_sparse_lbvh(&self.mirror_positions"),
        "owner hot path must not rebuild sparse-LBVH topology on the CPU"
    );
    assert!(
        !owner_step.contains("tire_dem_scene_bounds_extent(&self.mirror_positions"),
        "owner hot path must not depend on CPU mirror positions for Morton topology bounds"
    );
    assert!(
        !owner_step.contains("update_resident_lbvh(queue"),
        "owner hot path must not rely on CPU-built in-place LBVH refresh"
    );
}

#[test]
fn tire_dem_resident_owner_reseed_has_explicit_in_place_gpu_state_bridge_source_guard() {
    let src = include_str!("../src/tire_dem.rs");
    let reseed = src
        .split("pub fn reseed_grain_state")
        .nth(1)
        .expect("resident DEM/FEM terrain owner reseed API should exist")
        .split("pub fn step_sparse_lbvh_gpu_mirrored")
        .next()
        .expect("reseed body should precede step body");
    assert!(
        reseed.contains("update_resident_state(queue, &self.state"),
        "same-count DEM owner reseeds should rewrite resident GPU state buffers in place"
    );
    assert!(
        reseed.contains("self.lbvh = None"),
        "reseeding grains must invalidate stale resident LBVH topology"
    );
    assert!(
        reseed.contains("self.reseed_count = self.reseed_count.saturating_add(1)"),
        "reseeding should expose a monotonic telemetry counter for viewer/debug panels"
    );
    assert!(
        reseed.contains("tire_dem_resident_owner_lbvh_bounds"),
        "reseeding should refresh cached Morton bounds once, not rebuild topology on CPU"
    );
    let update = src
        .split("pub fn update_resident_state")
        .nth(1)
        .expect("resident DEM state update bridge should exist")
        .split("pub fn dispatch_lbvh_step_resident_state")
        .next()
        .expect("resident state update body should precede dispatch");
    let update: String = update.split_whitespace().collect();
    assert!(
        update.contains(
            "molla_core::gpu_write_buffer!(queue,&state.positions,0,bytemuck::cast_slice(positions))"
        ) && update.contains(
            "molla_core::gpu_write_buffer!(queue,&state.velocities,0,bytemuck::cast_slice(velocities))"
        ),
        "resident DEM state update should write existing GPU buffers directly"
    );
    assert!(
        !update.contains("create_buffer"),
        "resident DEM state update must not allocate replacement buffers"
    );
    assert!(
        update.contains("positions.len()!=state.grain_count"),
        "resident DEM state update must reject count changes instead of overflowing buffers"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_resident_fem_terrain_owner_reseed_refreshes_gpu_state_and_bounds() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireDemGpuKernel::new(&device);
    let mut input = sample_step_input();
    input.positions = vec![
        [0.0, 0.10, 0.0, 0.0],
        [0.08, 0.10, 0.0, 0.0],
        [0.35, 0.10, 0.0, 0.0],
        [0.43, 0.11, 0.0, 0.0],
    ];
    input.velocities = vec![[0.0; 4]; input.positions.len()];
    let terrain_params = TireDemResidentTerrainParams {
        origin: [-0.2, -0.2],
        spacing: 0.25,
        nx: 4,
        nz: 2,
        floor_y: 0.0,
        grain_radius: input.params.grain_radius,
    };
    let mut owner = TireDemResidentFemTerrainOwner::from_grain_state(
        &device,
        &queue,
        &kernel,
        terrain_params,
        &input.positions,
        &input.velocities,
    )
    .expect("resident DEM FEM terrain owner");
    owner
        .step_sparse_lbvh_gpu_mirrored(&device, &queue, &kernel, &input.contacts, input.params, 8)
        .expect("resident owner should build initial LBVH");
    assert!(owner.resident_lbvh().is_some());

    let reseeded_positions = vec![
        [0.60, 0.16, 0.20, 0.0],
        [0.68, 0.17, 0.20, 0.0],
        [0.76, 0.18, 0.22, 0.0],
        [0.84, 0.19, 0.24, 0.0],
    ];
    let reseeded_velocities = vec![
        [0.2, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.1, 0.0],
        [0.2, 0.0, 0.2, 0.0],
        [0.2, 0.0, 0.3, 0.0],
    ];
    owner
        .reseed_grain_state(
            &device,
            &queue,
            &kernel,
            &reseeded_positions,
            &reseeded_velocities,
        )
        .expect("same-count reseed should update resident DEM owner");
    assert_eq!(owner.positions(), reseeded_positions.as_slice());
    assert_eq!(owner.velocities(), reseeded_velocities.as_slice());
    assert_eq!(owner.reseed_count(), 1);
    assert!(
        owner.resident_lbvh().is_none(),
        "reseed should drop stale resident LBVH topology before the next GPU build"
    );
    let (min, extent) = owner.lbvh_scene_bounds();
    assert!(min[0] <= reseeded_positions[0][0]);
    assert!(extent.iter().all(|v| v.is_finite() && *v > 0.0));
    let resident = kernel
        .readback_resident_state(&device, &queue, owner.resident_state())
        .expect("reseeded resident DEM state readback");
    assert_eq!(resident.positions, reseeded_positions);
    assert_eq!(resident.velocities, reseeded_velocities);
    let terrain = kernel
        .readback_resident_terrain(&device, &queue, owner.resident_terrain())
        .expect("reseeded resident DEM terrain readback");
    let expected_heights = tire_dem_raster_surface_height_map(
        owner.positions(),
        terrain_params.origin,
        terrain_params.spacing,
        terrain_params.nx,
        terrain_params.nz,
        terrain_params.floor_y,
        terrain_params.grain_radius,
    )
    .expect("CPU expected reseeded heightfield");
    assert_eq!(terrain.heights, expected_heights);
}

#[test]
fn tire_dem_sparse_lbvh_reference_matches_dense_step_for_sparse_bed() {
    let mut input = sample_step_input();
    input.positions = vec![
        [0.0, 0.10, 0.0, 0.0],
        [0.08, 0.10, 0.0, 0.0],
        [2.0, 0.10, 0.0, 0.0],
        [2.08, 0.10, 0.0, 0.0],
        [5.0, 0.03, 5.0, 0.0],
        [5.08, 0.03, 5.0, 0.0],
        [8.0, 0.10, -2.0, 0.0],
        [8.08, 0.10, -2.0, 0.0],
        [-3.0, 0.10, 4.0, 0.0],
        [-3.08, 0.10, 4.0, 0.0],
    ];
    input.velocities = vec![[0.0; 4]; input.positions.len()];
    input.contacts.clear();
    input.params.floor_y = -1.0;
    input.params.gravity = [0.0; 3];

    let dense = tire_dem_reference_step(&input).expect("dense reference");
    let (lbvh_out, lbvh) = tire_dem_reference_sparse_lbvh_step(&input).expect("lbvh reference");
    assert!(
        lbvh.nodes.len() > 1,
        "sparse LBVH should build a tree for separated grain clusters"
    );
    for (i, (a, b)) in lbvh_out
        .velocities
        .iter()
        .zip(dense.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 1.0e-6,
                "LBVH velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

#[test]
fn tire_dem_sparse_lbvh_large_stress_matches_dense_step() {
    let input = large_sparse_lbvh_stress_input();
    let gpu_input = tire_dem_build_sparse_lbvh_step_input(
        input.positions.clone(),
        input.velocities.clone(),
        input.contacts.clone(),
        input.params,
        8,
    )
    .expect("large sparse LBVH GPU input");
    assert_eq!(
        gpu_input.positions.len(),
        SPARSE_LBVH_STRESS_X * SPARSE_LBVH_STRESS_Z * SPARSE_LBVH_STRESS_GRAINS_PER_TILE
    );
    assert!(
        gpu_input.lbvh.nodes.len() > 31,
        "large sparse bed should build a non-trivial LBVH, got {} nodes",
        gpu_input.lbvh.nodes.len()
    );
    assert!(
        gpu_input
            .lbvh
            .nodes
            .iter()
            .filter(|node| node.is_leaf())
            .all(|node| node.count <= 8),
        "large sparse LBVH leaves must respect the stress-test leaf size"
    );
    assert!(
        gpu_input
            .lbvh
            .sorted_morton_codes
            .windows(2)
            .all(|w| w[0] <= w[1]),
        "large sparse LBVH must stay Morton sorted for GPU traversal"
    );
    let rows = tire_dem_sparse_lbvh_gpu_rows(&gpu_input.lbvh);
    assert_eq!(rows.len(), gpu_input.lbvh.nodes.len() * 3);
    let plan = tire_dem_sparse_lbvh_dispatch_plan(&gpu_input).expect("LBVH dispatch plan");
    assert!(
        plan.is_nonempty(),
        "plan should describe non-empty sparse dispatch: {plan:?}"
    );
    assert_eq!(plan.grain_count, gpu_input.positions.len());
    assert_eq!(plan.node_count, gpu_input.lbvh.nodes.len());
    assert_eq!(plan.node_row_count, rows.len());
    assert_eq!(
        plan.workgroup_count,
        (gpu_input.positions.len() as u32).div_ceil(64)
    );
    assert!(
        plan.storage_bytes > (gpu_input.positions.len() * core::mem::size_of::<[f32; 4]>()) as u64,
        "plan should account for node rows, indices, IO and contacts: {plan:?}"
    );

    let dense = tire_dem_reference_step(&input).expect("dense reference");
    let (lbvh_out, lbvh) = tire_dem_reference_sparse_lbvh_step(&input).expect("lbvh reference");
    assert_eq!(lbvh.nodes.len(), gpu_input.lbvh.nodes.len());
    for (i, (a, b)) in lbvh_out
        .positions
        .iter()
        .zip(dense.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 1.0e-5,
                "large sparse LBVH position mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
    for (i, (a, b)) in lbvh_out
        .velocities
        .iter()
        .zip(dense.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 1.0e-5,
                "large sparse LBVH velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_large_sparse_lbvh_dispatch_stress_matches_reference() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let input = large_sparse_lbvh_stress_input();
    let expected = tire_dem_reference_sparse_lbvh_step(&input)
        .expect("large sparse LBVH reference")
        .0;
    let gpu_input = tire_dem_build_sparse_lbvh_step_input(
        input.positions,
        input.velocities,
        input.contacts,
        input.params,
        8,
    )
    .expect("large sparse LBVH GPU input");
    let plan = tire_dem_sparse_lbvh_dispatch_plan(&gpu_input).expect("large LBVH dispatch plan");
    assert!(
        plan.grain_count
            >= SPARSE_LBVH_STRESS_X * SPARSE_LBVH_STRESS_Z * SPARSE_LBVH_STRESS_GRAINS_PER_TILE
            && plan.node_count > 31
            && plan.workgroup_count > 1,
        "large hardware lane should exercise multi-workgroup sparse traversal: {plan:?}"
    );
    let kernel = TireDemGpuKernel::new(&device);
    let got = kernel
        .dispatch_lbvh_step_blocking(&device, &queue, &gpu_input)
        .expect("GPU large sparse-LBVH DEM step");
    for (i, (a, b)) in got
        .positions
        .iter()
        .zip(expected.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "large GPU LBVH position mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
    for (i, (a, b)) in got
        .velocities
        .iter()
        .zip(expected.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "large GPU LBVH velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_resident_gpu_lbvh_builder_large_sparse_bed_matches_reference() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireDemGpuKernel::new(&device);
    let input = large_sparse_lbvh_stress_input();
    let expected = tire_dem_reference_sparse_lbvh_step(&input)
        .expect("large sparse LBVH reference")
        .0;
    let (scene_min, scene_extent) = test_bounds(&input.positions);
    let state = kernel
        .create_resident_state(&device, &queue, &input.positions, &input.velocities)
        .expect("resident large DEM state");
    let resident_lbvh = kernel
        .build_resident_sparse_lbvh_tree_rows_gpu_radix(
            &device,
            &queue,
            &state,
            scene_min,
            scene_extent,
            8,
        )
        .expect("large no-readback resident GPU sparse-LBVH builder");
    let leaf_count = input.positions.len().div_ceil(8);
    let padded_leaf_count = leaf_count.next_power_of_two();
    assert_eq!(resident_lbvh.grain_count, input.positions.len());
    assert_eq!(resident_lbvh.sorted_index_count, input.positions.len());
    assert_eq!(resident_lbvh.node_count, padded_leaf_count * 2 - 1);
    assert_eq!(resident_lbvh.node_row_count, resident_lbvh.node_count * 3);
    assert!(
        resident_lbvh.node_count > 127,
        "large no-readback builder should exercise a multi-level padded tree: nodes={}",
        resident_lbvh.node_count
    );

    let resident = kernel
        .dispatch_lbvh_step_resident_state_with_resident_lbvh(
            &device,
            &queue,
            &state,
            &input.contacts,
            input.params,
            &resident_lbvh,
        )
        .expect("resident DEM step using large no-readback GPU-built sparse-LBVH rows");
    assert_eq!(resident.dispatch_plan.node_count, resident_lbvh.node_count);
    assert!(resident.dispatch_plan.workgroup_count > 1);
    let got = kernel
        .readback_resident_output(&device, &queue, &resident)
        .expect("readback large resident DEM after no-readback LBVH build");
    for (i, (a, b)) in got
        .positions
        .iter()
        .zip(expected.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "large no-readback GPU-built resident LBVH position mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
    for (i, (a, b)) in got
        .velocities
        .iter()
        .zip(expected.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "large no-readback GPU-built resident LBVH velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_lbvh_step_matches_sparse_lbvh_reference() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut input = sample_step_input();
    input.positions = vec![
        [0.0, 0.10, 0.0, 0.0],
        [0.08, 0.10, 0.0, 0.0],
        [2.0, 0.10, 0.0, 0.0],
        [2.08, 0.10, 0.0, 0.0],
        [5.0, 0.03, 5.0, 0.0],
        [5.08, 0.03, 5.0, 0.0],
        [8.0, 0.10, -2.0, 0.0],
        [8.08, 0.10, -2.0, 0.0],
        [-3.0, 0.10, 4.0, 0.0],
        [-3.08, 0.10, 4.0, 0.0],
    ];
    input.velocities = vec![[0.0; 4]; input.positions.len()];
    input.contacts.clear();
    input.params.floor_y = -1.0;
    input.params.gravity = [0.0; 3];
    let expected = tire_dem_reference_sparse_lbvh_step(&input)
        .expect("lbvh reference")
        .0;
    let gpu_input = tire_dem_build_sparse_lbvh_step_input(
        input.positions,
        input.velocities,
        input.contacts,
        input.params,
        8,
    )
    .expect("lbvh gpu input");
    let kernel = TireDemGpuKernel::new(&device);
    let resident = kernel
        .dispatch_lbvh_step_resident(&device, &queue, &gpu_input)
        .expect("GPU resident LBVH DEM step");
    assert_eq!(resident.grain_count, expected.positions.len());
    assert_eq!(
        resident.dispatch_plan.node_count,
        gpu_input.lbvh.nodes.len()
    );
    let got = kernel
        .readback_resident_output(&device, &queue, &resident)
        .expect("explicit resident DEM readback seam");
    for (i, (a, b)) in got
        .velocities
        .iter()
        .zip(expected.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 1.0e-4,
                "GPU LBVH velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_impulses_match_reference() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireDemGpuKernel::new(&device);
    let input = sample_input();
    let expected = tire_dem_reference_impulses(&input).expect("reference");
    let got = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("GPU DEM tire impulses");
    for (i, (a, b)) in got
        .positions
        .iter()
        .zip(expected.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "position mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
    for (i, (a, b)) in got
        .velocities
        .iter()
        .zip(expected.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_step_matches_reference() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireDemGpuKernel::new(&device);
    let input = sample_step_input();
    let expected = tire_dem_reference_step(&input).expect("reference");
    let got = kernel
        .dispatch_step_blocking(&device, &queue, &input)
        .expect("GPU DEM step");
    for (i, (a, b)) in got
        .positions
        .iter()
        .zip(expected.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "position mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
    for (i, (a, b)) in got
        .velocities
        .iter()
        .zip(expected.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_grid_step_matches_reference() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireDemGpuKernel::new(&device);
    let input = sample_grid_step_input();
    let expected = tire_dem_reference_grid_step(&input).expect("reference");
    let got = kernel
        .dispatch_grid_step_blocking(&device, &queue, &input)
        .expect("GPU DEM grid step");
    for (i, (a, b)) in got
        .positions
        .iter()
        .zip(expected.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "position mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
    for (i, (a, b)) in got
        .velocities
        .iter()
        .zip(expected.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_radix_unbounded_grid_matches_host_oracle() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let positions = vec![
        [0.30, 0.10, 0.10, 0.0],
        [0.08, 0.10, 0.10, 0.0],
        [0.12, 0.10, 0.10, 0.0],
        [1.10, 0.10, 0.10, 0.0],
        [0.18, 0.10, 0.10, 0.0],
        [1.20, 0.10, 0.10, 0.0],
    ];
    let velocities = vec![[0.0; 4]; positions.len()];
    let params = TireDemStepParams {
        gravity: [0.0; 3],
        normal_stiffness: 0.0,
        floor_y: -1.0,
        ..TireDemStepParams::default()
    };
    let expected = tire_dem_build_unbounded_sorted_grid_step_input(
        positions.clone(),
        velocities.clone(),
        Vec::new(),
        params,
        [0.0, 0.0, 0.0],
        1.0,
        [3, 2, 2],
    )
    .expect("host unbounded grid");
    let got = tire_dem_build_unbounded_sorted_grid_step_input_gpu_radix(
        device,
        queue,
        positions,
        velocities,
        Vec::new(),
        params,
        [0.0, 0.0, 0.0],
        1.0,
        [3, 2, 2],
    )
    .expect("gpu radix unbounded grid");
    assert_eq!(got.1.sorted_cell_keys, expected.1.sorted_cell_keys);
    assert_eq!(got.1.sorted_indices, expected.1.sorted_indices);
    assert_eq!(got.1.cell_ranges, expected.1.cell_ranges);
    assert_eq!(got.1.max_cell_occupancy, expected.1.max_cell_occupancy);
    assert_eq!(got.0.sorted_indices, expected.0.sorted_indices);
    assert_eq!(got.0.cell_ranges, expected.0.cell_ranges);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_unbounded_radix_grid_step_matches_host_grid_reference() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireDemGpuKernel::new(&device);
    let input = sample_gpu_radix_grid_step_input();
    let reference = tire_dem_build_unbounded_sorted_grid_step_input(
        input.positions.clone(),
        input.velocities.clone(),
        input.contacts.clone(),
        input.params,
        input.grid_origin,
        input.cell_size,
        input.grid_shape,
    )
    .expect("host unbounded grid reference");
    let expected = tire_dem_reference_grid_step(&reference.0).expect("reference");
    let got = kernel
        .dispatch_unbounded_radix_grid_step_blocking(device, queue, &input)
        .expect("GPU unbounded-radix DEM grid step");
    for (i, (a, b)) in got
        .positions
        .iter()
        .zip(expected.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "position mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
    for (i, (a, b)) in got
        .velocities
        .iter()
        .zip(expected.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_dem_gpu_built_grid_step_matches_host_grid_reference() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireDemGpuKernel::new(&device);
    let input = sample_gpu_built_grid_step_input();
    let reference = tire_dem_build_grid_step_input(
        input.positions.clone(),
        input.velocities.clone(),
        input.contacts.clone(),
        input.params,
        input.grid_origin,
        input.cell_size,
        input.grid_shape,
    )
    .expect("host reference grid");
    let expected = tire_dem_reference_grid_step(&reference).expect("reference");
    let got = kernel
        .dispatch_gpu_built_grid_step_blocking(&device, &queue, &input)
        .expect("GPU-built DEM grid step");
    for (i, (a, b)) in got
        .positions
        .iter()
        .zip(expected.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "position mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
    for (i, (a, b)) in got
        .velocities
        .iter()
        .zip(expected.velocities.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "velocity mismatch grain={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}
