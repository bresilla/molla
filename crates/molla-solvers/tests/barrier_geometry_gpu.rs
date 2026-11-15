#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{BarrierGpuObstacle, BarrierGpuParams, BarrierGpuSpring, BarrierNewtonGpu};

fn params() -> BarrierGpuParams {
    BarrierGpuParams {
        gravity: [0.0; 3],
        ..Default::default()
    }
}

fn unchanged(gpu: &BarrierNewtonGpu, before: &(Vec<[f32; 3]>, Vec<[f32; 3]>)) {
    assert_eq!(gpu.read_positions().unwrap(), before.0);
    assert_eq!(gpu.read_velocities().unwrap(), before.1);
}

fn snapshot(gpu: &BarrierNewtonGpu) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    (
        gpu.read_positions().unwrap(),
        gpu.read_velocities().unwrap(),
    )
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-geometry"]
fn pierced_coplanar_and_isolated_vertex_starts_are_refused() {
    let engine = Engine::new_gpu().unwrap();
    let sheet = [[-1.0, 0.0, -0.4], [1.0, 0.0, -0.4], [0.0, 0.0, 0.5]];
    for (label, second) in [
        (
            "pierced",
            vec![[-0.3, -0.5, 0.0], [0.3, 0.5, 0.0], [0.4, -0.5, 0.3]],
        ),
        (
            "coplanar containment",
            vec![[-0.2, 0.0, -0.2], [0.2, 0.0, -0.2], [0.0, 0.0, 0.2]],
        ),
        (
            "coplanar crossed edges",
            vec![[-1.0, 0.0, 0.4], [1.0, 0.0, 0.4], [0.0, 0.0, -0.5]],
        ),
        ("isolated vertex", vec![[0.0; 3]]),
    ] {
        for axis in 0..3 {
            let positions: Vec<_> = sheet
                .iter()
                .chain(&second)
                .map(|p| [p[axis], p[(axis + 1) % 3], p[(axis + 2) % 3]])
                .collect();
            let mut triangles = if second.len() == 1 {
                vec![[0, 1, 2]]
            } else {
                vec![[0, 1, 2], [3, 4, 5]]
            };
            if axis == 1 {
                triangles[0].swap(1, 2);
            }
            let mut gpu = BarrierNewtonGpu::new_with_mesh(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                &positions,
                &vec![1.0; positions.len()],
                &[],
                &[],
                &triangles,
                &[],
                params(),
            )
            .unwrap();
            let before = snapshot(&gpu);
            let error = gpu.step(1.0 / 240.0).unwrap_err().to_string();
            assert!(
                error.contains("initial") && error.contains("mesh INTERSECTS"),
                "{label} axis={axis}: {error}"
            );
            unchanged(&gpu, &before);
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-geometry"]
fn adjacency_and_resolvable_separation_remain_valid() {
    let engine = Engine::new_gpu().unwrap();
    for scale in [1.0e-3, 1.0, 10.0] {
        let positions = [
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.0, 0.0, scale],
            [scale, 0.0, scale],
        ];
        let mut gpu = BarrierNewtonGpu::new_with_mesh(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            &positions,
            &[1.0; 4],
            &[],
            &[],
            &[[0, 1, 2], [1, 3, 2]],
            &[],
            BarrierGpuParams {
                dhat: 1.0e-3 * scale,
                ..params()
            },
        )
        .unwrap();
        for _ in 0..3 {
            gpu.step(1.0 / 240.0).unwrap();
        }
        assert_eq!(gpu.read_positions().unwrap(), positions);
    }
    let positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.02, 0.0],
        [1.0, 0.02, 0.0],
        [0.0, 0.02, 1.0],
    ];
    let mut gpu = BarrierNewtonGpu::new_with_mesh(
        engine.device().unwrap(),
        engine.queue().unwrap(),
        &positions,
        &[1.0; 6],
        &[],
        &[],
        &[[0, 1, 2], [3, 4, 5]],
        &[],
        params(),
    )
    .unwrap();
    gpu.set_velocities(&[[1.0, 0.0, 0.0]; 6]).unwrap();
    for _ in 0..20 {
        gpu.step(1.0 / 240.0).unwrap();
    }
    assert!(gpu.read_positions().unwrap()[0][0] > 0.01);
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-geometry"]
fn obstacle_and_strain_refusals_preserve_published_state() {
    let engine = Engine::new_gpu().unwrap();
    for obstacle in [
        BarrierGpuObstacle::HalfSpace {
            n: [0.0, 1.0, 0.0],
            offset: 0.2,
        },
        BarrierGpuObstacle::Sphere {
            center: [0.0; 3],
            radius: 0.2,
        },
    ] {
        let mut gpu = BarrierNewtonGpu::new(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            &[[0.0; 3]],
            &[1.0],
            &[],
            &[obstacle],
            params(),
        )
        .unwrap();
        let before = snapshot(&gpu);
        assert!(
            gpu.step(1.0 / 240.0)
                .unwrap_err()
                .to_string()
                .contains("obstacle INTERSECTS")
        );
        unchanged(&gpu, &before);
        let mut pinned = BarrierNewtonGpu::new_with_mesh(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            &[[0.0; 3]],
            &[1.0],
            &[],
            &[obstacle],
            &[],
            &[0],
            params(),
        )
        .unwrap();
        pinned.step(1.0 / 240.0).unwrap();
    }
    let mut gpu = BarrierNewtonGpu::new(
        engine.device().unwrap(),
        engine.queue().unwrap(),
        &[[0.0; 3], [2.0, 0.0, 0.0]],
        &[1.0; 2],
        &[BarrierGpuSpring {
            a: 0,
            b: 1,
            rest: 1.0,
            k: 1.0,
        }],
        &[],
        BarrierGpuParams {
            strain_limit: 1.1,
            ..params()
        },
    )
    .unwrap();
    let before = snapshot(&gpu);
    assert!(
        gpu.step(1.0 / 240.0)
            .unwrap_err()
            .to_string()
            .contains("strain limit")
    );
    unchanged(&gpu, &before);
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-geometry"]
fn final_gate_rejects_inversion_and_recovers_without_publishing() {
    let engine = Engine::new_gpu().unwrap();
    let positions = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
    let mut gpu = BarrierNewtonGpu::new_full(
        engine.device().unwrap(),
        engine.queue().unwrap(),
        &positions,
        &[1.0; 4],
        &[],
        &[],
        &[],
        &[([0, 1, 2, 3], 0.0)],
        &[0, 1, 2],
        BarrierGpuParams::default(),
    )
    .unwrap();
    let before = snapshot(&gpu);
    let error = gpu.step(0.5).unwrap_err().to_string();
    assert!(
        error.contains("accepted") && error.contains("inverted tet"),
        "{error}"
    );
    unchanged(&gpu, &before);
    gpu.params.gravity = [0.0; 3];
    gpu.step(1.0 / 240.0).unwrap();
    unchanged(&gpu, &before);
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-geometry"]
fn numerical_overflow_and_incomplete_recovery_are_not_published() {
    let engine = Engine::new_gpu().unwrap();
    let mut gpu = BarrierNewtonGpu::new(
        engine.device().unwrap(),
        engine.queue().unwrap(),
        &[[0.0, 1.0, 0.0]],
        &[1.0],
        &[],
        &[],
        params(),
    )
    .unwrap();
    gpu.set_velocities(&[[f32::MAX, 0.0, 0.0]]).unwrap();
    let before = snapshot(&gpu);
    assert!(
        gpu.step(16.0)
            .unwrap_err()
            .to_string()
            .contains("nonfinite")
    );
    unchanged(&gpu, &before);
    gpu.set_velocities(&[[0.0; 3]]).unwrap();
    gpu.step(1.0 / 240.0).unwrap();
    let mut gpu = BarrierNewtonGpu::new(
        engine.device().unwrap(),
        engine.queue().unwrap(),
        &[[0.0, -0.1, 0.0]],
        &[1.0],
        &[],
        &[BarrierGpuObstacle::HalfSpace {
            n: [0.0, 1.0, 0.0],
            offset: 0.0,
        }],
        BarrierGpuParams {
            reject_intersecting_start: false,
            newton_tol: 1.0e30,
            ..params()
        },
    )
    .unwrap();
    let before = snapshot(&gpu);
    let error = gpu.step(1.0 / 240.0).unwrap_err().to_string();
    assert!(
        error.contains("accepted") && error.contains("obstacle INTERSECTS"),
        "{error}"
    );
    unchanged(&gpu, &before);
}
