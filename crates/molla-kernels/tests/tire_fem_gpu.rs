//! GPU smoke/parity test for the volumetric tire FEM node kernel.
//!
//! Ignored by default because it needs a real wgpu adapter:
//! `cargo test -p molla-kernels --features gpu --test tire_fem_gpu -- --ignored`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{TireFemGpuInput, TireFemGpuKernel, TireFemGpuParams};

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn tire_fem_gpu_advances_one_tetra_without_nan() {
    let engine = try_engine().expect("GPU adapter required for ignored test");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = TireFemGpuKernel::new(&device);

    // One positively-oriented tetra. Gradients are the constant shape-function
    // gradients for x0=(0,0,0), x1=(1,0,0), x2=(0,1,0), x3=(0,0,1).
    let input = TireFemGpuInput {
        rest_positions: vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        positions: vec![
            [0.0, 0.05, 0.0, 0.0],
            [1.0, 0.05, 0.0, 0.0],
            [0.0, 1.05, 0.0, 0.0],
            [0.0, 0.05, 1.0, 0.0],
        ],
        velocities: vec![[0.0; 4]; 4],
        masses: vec![0.25; 4],
        tet_indices: vec![0, 1, 2, 3],
        tet_grads: vec![
            [-1.0, -1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        tet_volumes: vec![1.0 / 6.0],
        tet_mu: vec![3_400.0],
        tet_lambda: vec![23_000.0],
        bead_ids: vec![],
        bead_offsets: vec![],
        tread_ids: vec![2],
        params: TireFemGpuParams {
            dt: 1.0 / 600.0,
            young_modulus: 1.0e4,
            pressure_pa: 1.0e3,
            pressure_area_per_tread_node: 0.01,
            hub: [0.0, 0.25, 0.0],
            ..Default::default()
        },
    };

    let out = kernel
        .dispatch_blocking(&device, &queue, &input)
        .expect("tire FEM GPU dispatch");
    assert_eq!(out.positions.len(), 4);
    assert_eq!(out.velocities.len(), 4);
    assert_eq!(out.forces.len(), 4);
    for p in &out.positions {
        assert!(
            p.iter().all(|x| x.is_finite()),
            "non-finite position: {p:?}"
        );
        assert!(p[1] >= 0.0, "floor contact should clamp y >= 0: {p:?}");
    }
}
