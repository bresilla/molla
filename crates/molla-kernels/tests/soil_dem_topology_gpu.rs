#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    TireDemGpuKernel, TireDemResidentFemTerrainOwner, TireDemResidentTerrainParams,
    TireDemStepGpuInput, TireDemStepParams, tire_dem_reference_step,
};

fn terrain() -> TireDemResidentTerrainParams {
    TireDemResidentTerrainParams {
        origin: [-1.0, -1.0],
        spacing: 0.125,
        nx: 16,
        nz: 16,
        floor_y: 0.0,
        grain_radius: 0.04,
    }
}

fn fixture(n: usize, shape: usize) -> Vec<[f32; 4]> {
    (0..n)
        .map(|i| {
            let i = if shape == 1 { i * 17 % n } else { i };
            let spacing = if shape == 2 { 0.015625 } else { 0.0625 };
            let shift = if shape == 3 {
                if i % 2 == 0 { 2000.0 } else { -2000.0 }
            } else {
                0.0
            };
            [
                (i % 7) as f32 * spacing - 0.2 + shift,
                (i / 21) as f32 * spacing + 0.08,
                (i / 7 % 3) as f32 * spacing - 0.1,
                0.0,
            ]
        })
        .collect()
}

fn exhaustive_pairs(positions: &[[f32; 4]], radius: f32) -> Vec<[u32; 2]> {
    let mut pairs = Vec::new();
    for (i, a) in positions.iter().enumerate() {
        for (j, b) in positions.iter().enumerate() {
            let distance =
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            if i != j && distance >= 1e-9 && distance < 2.0 * radius {
                pairs.push([i as u32, j as u32]);
            }
        }
    }
    pairs
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_dem_refit_pair_sets_cover_reordering_clusters_and_radius_changes() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    let mut positions = fixture(65, 0);
    let velocities = vec![[0.0; 4]; positions.len()];
    let mut owner = TireDemResidentFemTerrainOwner::from_grain_state(
        device,
        queue,
        &kernel,
        terrain(),
        &positions,
        &velocities,
    )
    .unwrap();
    owner.set_lbvh_rebuild_interval(1000).unwrap();
    assert!(owner.set_lbvh_rebuild_interval(0).is_err());
    for shape in 0..4 {
        positions = fixture(65, shape);
        for radius in [0.02, 0.04, 0.0625, 0.2] {
            kernel
                .update_resident_state(queue, owner.resident_state(), &positions, &velocities)
                .unwrap();
            let params = TireDemStepParams {
                dt: 0.0,
                grain_radius: radius,
                ..Default::default()
            };
            owner
                .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &[], params, 8)
                .unwrap();
            let audit = kernel
                .audit_resident_lbvh_contacts(
                    device,
                    queue,
                    owner.resident_state(),
                    owner.resident_lbvh().unwrap(),
                    radius,
                )
                .unwrap();
            let expected = exhaustive_pairs(&positions, radius);
            assert_eq!(audit.contacts, expected, "shape={shape} radius={radius}");
            assert!(
                expected
                    .iter()
                    .all(|p| audit.candidates.binary_search(p).is_ok())
            );
            eprintln!(
                "shape={shape} radius={radius} directed_candidates={} directed_contacts={}",
                audit.candidates.len(),
                audit.contacts.len()
            );
        }
    }
    assert_eq!(owner.topology_stats().rebuilds, 1);
    assert_eq!(owner.topology_stats().refits, 15);
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_dem_refit_moving_contacts_match_local_oracle_and_rebuild() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    let mut positions = fixture(31, 0);
    let mut velocities: Vec<_> = (0..positions.len())
        .map(|i| [if i % 2 == 0 { 12.0 } else { -12.0 }, -0.1, 0.0, 0.0])
        .collect();
    let mut owner = TireDemResidentFemTerrainOwner::from_grain_state(
        device,
        queue,
        &kernel,
        terrain(),
        &positions,
        &velocities,
    )
    .unwrap();
    owner.set_lbvh_rebuild_interval(32).unwrap();
    let mut rebuild = TireDemResidentFemTerrainOwner::from_grain_state(
        device,
        queue,
        &kernel,
        terrain(),
        &positions,
        &velocities,
    )
    .unwrap();
    for step in 0..20 {
        let params = TireDemStepParams {
            grain_mass: 1.0,
            grain_radius: 0.03 + (step % 3) as f32 * 0.01,
            normal_stiffness: 300.0,
            damping: 5.0,
            friction: 0.3,
            floor_y: if step > 10 { 0.08 } else { 0.0 },
            dt: 0.002,
            ..Default::default()
        };
        let expected = tire_dem_reference_step(&TireDemStepGpuInput {
            positions: positions.clone(),
            velocities: velocities.clone(),
            contacts: Vec::new(),
            params,
        })
        .unwrap();
        let local_expected = tire_dem_reference_step(&TireDemStepGpuInput {
            positions: owner.positions().to_vec(),
            velocities: owner.velocities().to_vec(),
            contacts: Vec::new(),
            params,
        })
        .unwrap();
        owner
            .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &[], params, 4)
            .unwrap();
        owner.refresh_mirror(device, queue, &kernel).unwrap();
        rebuild
            .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &[], params, 4)
            .unwrap();
        rebuild.refresh_mirror(device, queue, &kernel).unwrap();
        let max_error = |a: &[[f32; 4]], b: &[[f32; 4]]| {
            a.iter()
                .zip(b)
                .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
                .fold(0.0f32, f32::max)
        };
        eprintln!(
            "step={step} cpu/refit={} cpu/rebuild={} rebuild/refit={}",
            max_error(&expected.velocities, owner.velocities()),
            max_error(&expected.velocities, rebuild.velocities()),
            max_error(rebuild.velocities(), owner.velocities())
        );
        for (label, actual, expected) in [
            (
                "position/local",
                owner.positions(),
                local_expected.positions.as_slice(),
            ),
            (
                "velocity/local",
                owner.velocities(),
                local_expected.velocities.as_slice(),
            ),
            ("position/rebuild", owner.positions(), rebuild.positions()),
            ("velocity/rebuild", owner.velocities(), rebuild.velocities()),
        ] {
            for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
                for k in 0..3 {
                    assert!(
                        (a[k] - b[k]).abs() <= 2e-5,
                        "{label} step={step} row={i} axis={k}: {} != {}",
                        a[k],
                        b[k]
                    );
                }
            }
        }
        positions = expected.positions;
        velocities = expected.velocities;
    }
    assert_eq!(owner.topology_stats().rebuilds, 1);
    assert_eq!(owner.topology_stats().refits, 19);
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_dem_refit_rebuild_schedule_and_membership_invalidation() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    let positions = fixture(17, 0);
    let velocities = vec![[0.0; 4]; positions.len()];
    let mut owner = TireDemResidentFemTerrainOwner::from_grain_state(
        device,
        queue,
        &kernel,
        terrain(),
        &positions,
        &velocities,
    )
    .unwrap();
    owner.set_lbvh_rebuild_interval(4).unwrap();
    let params = TireDemStepParams {
        dt: 0.0,
        ..Default::default()
    };
    for _ in 0..9 {
        owner
            .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &[], params, 8)
            .unwrap();
    }
    assert_eq!(owner.topology_stats().rebuilds, 3);
    assert_eq!(owner.topology_stats().refits, 6);
    for n in [17, 65] {
        let positions = fixture(n, 1);
        let before = owner.topology_stats().rebuilds;
        owner
            .reseed_grain_state(device, queue, &kernel, &positions, &vec![[0.0; 4]; n])
            .unwrap();
        owner
            .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &[], params, 8)
            .unwrap();
        assert_eq!(owner.topology_stats().rebuilds, before + 1);
    }
    let before = owner.topology_stats().rebuilds;
    owner
        .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &[], params, 2)
        .unwrap();
    assert_eq!(owner.topology_stats().rebuilds, before + 1);
    owner.set_lbvh_rebuild_interval(1).unwrap();
    owner
        .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &[], params, 2)
        .unwrap();
    assert_eq!(owner.topology_stats().rebuilds, before + 2);
}
