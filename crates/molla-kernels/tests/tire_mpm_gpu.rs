//! GPU tire-to-MPM material-point feedback tests.
//!
//! Ignored GPU parity lane:
//! `cargo test -p molla-kernels --features gpu --test tire_mpm_gpu -- --ignored`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    TireMpmContactCell, TireMpmFemTerrainOwner, TireMpmGpuInput, TireMpmGpuKernel,
    TireMpmTerrainRasterParams, tire_mpm_reference_feedback,
};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn sample_input() -> TireMpmGpuInput {
    TireMpmGpuInput {
        positions: vec![
            [0.0, 0.05, 0.0, 0.0],
            [0.05, 0.02, 0.0, 0.0],
            [0.35, 0.05, 0.0, 0.0],
        ],
        velocities: vec![[0.0; 4]; 3],
        compaction: vec![[0.0; 4]; 3],
        contacts: vec![TireMpmContactCell {
            center_radius: [0.0, 0.06, 0.0, 0.14],
            force_pressure: [20.0, -120.0, 5.0, 80_000.0],
        }],
        particle_mass: 2.0,
        dt: 1.0e-3,
        plastic_compaction_gain: 2.0e-3,
        max_compaction: 0.20,
        damping: 18.0,
    }
}

#[test]
fn tire_mpm_reference_applies_pressure_shear_and_compaction() {
    let input = sample_input();
    let out = tire_mpm_reference_feedback(&input).expect("MPM reference");
    assert_eq!(out.positions.len(), input.positions.len());
    assert_eq!(out.velocities.len(), input.velocities.len());
    assert_eq!(out.compaction.len(), input.compaction.len());
    assert!(
        out.velocities[0][1] < 0.0 && out.positions[0][1] < input.positions[0][1],
        "material point under tire should receive downward velocity: {:?}",
        out.velocities[0]
    );
    assert!(
        out.compaction[0][0] > 0.0 && out.compaction[0][1] > 0.0 && out.compaction[0][2] > 0.0,
        "contacted material point should record compaction, normal impulse, and shear impulse: {:?}",
        out.compaction[0]
    );
    assert!(
        out.compaction[1][0] > 0.0 && out.compaction[1][0] < out.compaction[0][0],
        "nearby off-centre material point should compact, but less than the patch centre: {:?}",
        out.compaction
    );
    assert_eq!(
        out.velocities[2], input.velocities[2],
        "far material point should not be affected by local tire contact"
    );
}

#[test]
fn tire_mpm_reference_caps_plastic_compaction() {
    let mut input = sample_input();
    input.compaction[0][0] = 0.199;
    input.max_compaction = 0.20;
    let out = tire_mpm_reference_feedback(&input).expect("MPM reference");
    assert!(
        out.compaction[0][0] <= input.max_compaction + 1.0e-6,
        "plastic compaction should cap at max_compaction: {:?}",
        out.compaction[0]
    );
}

#[test]
fn tire_mpm_owner_exports_fem_sinkage_terrain() {
    let input = sample_input();
    let contacts = input.contacts.clone();
    let mut owner =
        TireMpmFemTerrainOwner::from_input(input, [-0.20, -0.20], 0.20, 4, 3, 0.0, 0.42)
            .expect("owner");
    owner.set_soil_rolling_resistance_scale(2.25);
    owner.set_surface_velocity([0.4, 0.0, 0.0]);
    owner
        .apply_contacts_cpu(&contacts)
        .expect("CPU material point feedback");
    let terrain = owner.fem_contact_terrain();
    assert_eq!(terrain.origin, [-0.20, -0.20]);
    assert_eq!(terrain.spacing, 0.20);
    assert_eq!(terrain.nx, 4);
    assert_eq!(terrain.nz, 3);
    assert_eq!(terrain.friction, 0.42);
    assert_eq!(terrain.soil_rolling_resistance_scale, 2.25);
    assert_eq!(terrain.surface_velocity, [0.4, 0.0, 0.0]);
    let center = (terrain.nx as usize) + 1;
    assert!(
        terrain.sinkage_map.as_ref().unwrap()[center] > 0.0,
        "FEM terrain should receive MPM particle compaction sinkage: {:?}",
        terrain.sinkage_map
    );
    assert!(
        terrain.heights[center] < 0.0,
        "FEM terrain should be depressed by compacted MPM particles: {:?}",
        terrain.heights
    );
}

#[test]
fn tire_mpm_raster_exports_surface_velocity_map_contract() {
    let source = include_str!("../src/tire_mpm.rs");
    let shader = include_str!("../shaders/tire_mpm.wgsl");
    assert!(
        source.contains("surface_velocity_map: Vec<[f32; 4]>"),
        "MPM raster output should expose spatial material velocity samples"
    );
    assert!(
        shader.contains("@binding(4) var<storage, read> raster_vel")
            && shader.contains("@binding(5) var<storage, read_write> raster_surface_velocity"),
        "MPM raster shader should bind particle velocities and output terrain velocity maps"
    );
    assert!(
        shader.contains("vel_sum += raster_vel[pidx].xyz")
            && shader.contains("raster_surface_velocity[idx]"),
        "MPM raster shader should average particle velocity into terrain cells"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_mpm_gpu_feedback_matches_reference() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireMpmGpuKernel::new(&device);
    let input = sample_input();
    let expected = tire_mpm_reference_feedback(&input).expect("reference");
    let got = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("GPU MPM feedback");
    for (i, (a, b)) in got
        .positions
        .iter()
        .zip(expected.positions.iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "position mismatch particle={i} axis={axis}: got={} expected={}",
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
                "velocity mismatch particle={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
    for (i, (a, b)) in got
        .compaction
        .iter()
        .zip(expected.compaction.iter())
        .enumerate()
    {
        for axis in 0..4 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "compaction mismatch particle={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_mpm_gpu_raster_matches_owner_terrain() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireMpmGpuKernel::new(&device);
    let input = sample_input();
    let feedback = tire_mpm_reference_feedback(&input).expect("reference feedback");
    let mut owner = TireMpmFemTerrainOwner::from_input(
        TireMpmGpuInput {
            positions: feedback.positions.clone(),
            velocities: feedback.velocities.clone(),
            compaction: feedback.compaction.clone(),
            contacts: Vec::new(),
            particle_mass: input.particle_mass,
            dt: input.dt,
            plastic_compaction_gain: input.plastic_compaction_gain,
            max_compaction: input.max_compaction,
            damping: input.damping,
        },
        [-0.20, -0.20],
        0.20,
        4,
        3,
        0.0,
        0.42,
    )
    .expect("owner");
    owner.set_surface_velocity([0.25, 0.0, 0.0]);
    let expected = owner.fem_contact_terrain();
    let got = kernel
        .dispatch_raster_blocking(
            &device,
            &queue,
            &feedback.positions,
            &feedback.velocities,
            &feedback.compaction,
            &TireMpmTerrainRasterParams {
                particle_count: feedback.positions.len() as u32,
                origin: [-0.20, -0.20],
                spacing: 0.20,
                nx: 4,
                nz: 3,
                rest_height: 0.0,
            },
            [0.25, 0.0, 0.0],
        )
        .expect("GPU raster");
    for (i, (a, b)) in got.heights.iter().zip(expected.heights.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1.0e-6,
            "height mismatch cell={i}: got={a} expected={b}"
        );
    }
    for (i, (a, b)) in got
        .sinkage
        .iter()
        .zip(expected.sinkage_map.as_ref().unwrap().iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1.0e-6,
            "sinkage mismatch cell={i}: got={a} expected={b}"
        );
    }
    for (i, (a, b)) in got
        .surface_velocity_map
        .iter()
        .zip(expected.surface_velocity_map.as_ref().unwrap().iter())
        .enumerate()
    {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 2.0e-5,
                "surface velocity mismatch cell={i} axis={axis}: got={} expected={}",
                a[axis],
                b[axis]
            );
        }
    }
}
