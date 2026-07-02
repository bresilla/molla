//! GPU Bekker deformable-terrain tire writeback tests.
//!
//! Ignored GPU tests need a real adapter:
//! `cargo test -p molla-kernels --features gpu --test tire_bekker_gpu -- --ignored`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    TireBekkerContactCell, TireBekkerFemTerrainOwner, TireBekkerGpuInput, TireBekkerGpuKernel,
    tire_bekker_reference_writeback,
};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

fn sample_input() -> TireBekkerGpuInput {
    TireBekkerGpuInput {
        origin: [-0.5, -0.5],
        spacing: 0.5,
        nx: 3,
        nz: 3,
        rest_heights: vec![0.0; 9],
        heights: vec![0.0; 9],
        sinkage: vec![0.0; 9],
        contacts: vec![
            TireBekkerContactCell {
                xz_pressure: [0.0, 0.0, 90_000.0, 0.0],
            },
            TireBekkerContactCell {
                xz_pressure: [0.45, 0.0, 30_000.0, 30_000.0],
            },
        ],
        stiffness: 900_000.0,
        max_sinkage: 0.12,
        shear_strength: 10_000.0,
        shear_sinkage_gain: 0.05,
        bekker_n: 1.0,
    }
}

#[test]
fn tire_bekker_reference_writeback_ratchetes_pressure_and_shear_sinkage() {
    let input = sample_input();
    let out = tire_bekker_reference_writeback(&input).expect("reference writeback");
    let center = (input.nx as usize) + 1;
    let right = (input.nx as usize) + 2;
    assert!(
        (out.sinkage[center] - 0.10).abs() < 1.0e-6,
        "center cell should follow Bekker pressure-sinkage: {:?}",
        out.sinkage
    );
    assert!(
        out.sinkage[right] > 0.04,
        "over-shear should add excavation sinkage on the right cell: {:?}",
        out.sinkage
    );
    assert!(
        out.heights[center] < input.heights[center] && out.heights[right] < input.heights[right],
        "heightfield must plastically depress under tire contact"
    );

    let mut second = input.clone();
    second.heights = out.heights.clone();
    second.sinkage = out.sinkage.clone();
    second.contacts.clear();
    let unchanged = tire_bekker_reference_writeback(&second).expect("ratchet writeback");
    assert_eq!(
        unchanged.sinkage, out.sinkage,
        "Bekker writeback should ratchet; removing load must not heal the rut"
    );
}

#[test]
fn tire_bekker_owner_exports_fem_sinkage_terrain() {
    let mut owner = TireBekkerFemTerrainOwner::flat(
        [-0.5, -0.5],
        0.5,
        3,
        3,
        0.0,
        0.55,
        900_000.0,
        0.12,
        10_000.0,
        0.05,
        1.0,
    )
    .expect("owner");
    owner.set_soil_rolling_resistance_scale(1.75);
    owner
        .set_friction_map(Some(vec![0.55; 9]))
        .expect("friction map");
    owner
        .set_water_film_map(Some(vec![0.002; 9]))
        .expect("water film map");
    owner.set_surface_velocity([0.6, 0.0, 0.0]);
    owner
        .set_surface_velocity_map(Some(vec![[0.6, 0.0, 0.2, 0.0]; 9]))
        .expect("surface velocity map");
    owner
        .apply_contacts_cpu(&sample_input().contacts)
        .expect("owner writeback");
    let terrain = owner.fem_contact_terrain();
    let center = (terrain.nx as usize) + 1;
    assert_eq!(terrain.origin, [-0.5, -0.5]);
    assert_eq!(terrain.spacing, 0.5);
    assert_eq!(terrain.friction, 0.55);
    assert_eq!(terrain.friction_map.as_ref().unwrap().len(), 9);
    assert_eq!(terrain.water_film_map.as_ref().unwrap().len(), 9);
    assert_eq!(terrain.surface_velocity, [0.6, 0.0, 0.0]);
    assert_eq!(terrain.surface_velocity_map.as_ref().unwrap().len(), 9);
    assert_eq!(
        terrain.surface_velocity_map.as_ref().unwrap()[center],
        [0.6, 0.0, 0.2, 0.0]
    );
    assert_eq!(terrain.soil_rolling_resistance_scale, 1.75);
    assert!(
        terrain.sinkage_map.as_ref().unwrap()[center] > 0.09,
        "FEM terrain should receive Bekker sinkage map from owner: {:?}",
        terrain.sinkage_map
    );
    assert!(
        terrain.heights[center] < 0.0,
        "FEM terrain heightfield should be plastically depressed by contact"
    );
}

#[test]
fn tire_bekker_exposes_resident_fem_feedback_terrain_path() {
    let source = include_str!("../src/tire_bekker.rs");
    assert!(
        source.contains("TireBekkerGpuResidentTerrain")
            && source.contains("pub fn dispatch_resident")
            && source.contains("pub fn readback_resident_output")
            && source.contains("pub fn apply_contacts_gpu_resident"),
        "Bekker terrain owner should expose a resident GPU terrain path for FEM feedback"
    );
    assert!(
        source.contains("heights: resident.heights")
            && source.contains("sinkage: resident.sinkage")
            && source.contains("friction_map: upload_map")
            && source.contains("water_film_map: upload_map")
            && source.contains("surface_velocity_map: upload_vec4_map"),
        "resident Bekker terrain should preserve height/sinkage plus material/film/velocity maps"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_bekker_gpu_writeback_matches_reference() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireBekkerGpuKernel::new(&device);
    let input = sample_input();
    let expected = tire_bekker_reference_writeback(&input).expect("reference writeback");
    let got = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("GPU Bekker writeback");
    for (i, (a, b)) in got.sinkage.iter().zip(expected.sinkage.iter()).enumerate() {
        assert!(
            (a - b).abs() < 2.0e-5,
            "GPU sinkage mismatch at {i}: got={a} expected={b}"
        );
    }
    for (i, (a, b)) in got.heights.iter().zip(expected.heights.iter()).enumerate() {
        assert!(
            (a - b).abs() < 2.0e-5,
            "GPU height mismatch at {i}: got={a} expected={b}"
        );
    }
}
