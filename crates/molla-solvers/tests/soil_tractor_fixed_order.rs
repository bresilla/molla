#![cfg(all(
    feature = "gpu",
    feature = "resource-counters",
    feature = "parallel-indices-newton"
))]

use molla_compute::Engine;
use molla_core::gpu_metrics::DeviceScope;
use molla_solvers::{
    mpm::{MpmGridStorage, MpmUnilateralConfig},
    soil_tractor::{SoilTractor, SoilTractorConfig, SoilTractorTransfer},
};
use std::sync::Arc;

#[test]
#[ignore = "requires protected native GPU; construction-only phase localization, not trajectory acceptance"]
fn tractor_fixed_order_constructor_diagnostic() {
    let started = std::time::Instant::now();
    eprintln!("constructor diagnostic engine begin");
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC
                | wgpu::Features::SHADER_F64
                | wgpu::Features::PASSTHROUGH_SHADERS
                | molla_kernels::FeatherstoneKernel::paired_drift_required_features(),
        )
        .unwrap(),
    );
    eprintln!(
        "constructor diagnostic engine ready elapsed={:?}",
        started.elapsed()
    );
    let tractor = SoilTractor::new_native_prediction_with_newton_transfer(
        engine,
        SoilTractorConfig {
            seed: [8, 2, 4],
            ..Default::default()
        },
        MpmUnilateralConfig {
            node_capacity: 4096,
            search_distance: 0.05,
            max_iterations: 64,
        },
        0.3,
        MpmGridStorage::Compact {
            tile_capacity: 1024,
        },
        1.0 / 60.0,
        SoilTractorTransfer::FixedOrder {
            max_gather_bytes: 4096,
        },
    )
    .unwrap();
    let info = tractor.world().mpm_soil_transfer_info().unwrap();
    assert!(info.fixed_order && info.particle_ordering && info.parallel_sort);
    assert_eq!(info.gather_bytes, 4096);
    assert!(info.ordering_bytes > 0);
    assert_eq!(
        tractor.world().mpm_soil_stats().unwrap().committed_frames,
        0
    );
    eprintln!(
        "constructor diagnostic ready elapsed={:?} info={info:?}; no stepping performed",
        started.elapsed()
    );
}

fn machine_bits(tractor: &SoilTractor) -> Vec<u64> {
    let scene = &tractor.world().scene;
    let mut values = Vec::new();
    for (body, _) in scene.bodies() {
        let pose = scene.body_pose(body).unwrap();
        let velocity = scene.body_velocity(body).unwrap();
        values.extend(pose.position.to_array());
        values.extend(pose.rotation.to_array());
        values.extend(velocity.linear.to_array());
        values.extend(velocity.angular.to_array());
    }
    values.extend_from_slice(scene.state().joint_q.host().unwrap());
    values.extend_from_slice(scene.state().joint_qd.host().unwrap());
    assert!(values.iter().all(|v| v.is_finite()));
    values.into_iter().map(f64::to_bits).collect()
}

#[test]
#[ignore = "requires protected native GPU; public tractor fixed-order constructor and exact repeats"]
fn tractor_fixed_order_public_constructor_and_repeatability() {
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC
                | wgpu::Features::SHADER_F64
                | wgpu::Features::PASSTHROUGH_SHADERS
                | molla_kernels::FeatherstoneKernel::paired_drift_required_features(),
        )
        .unwrap(),
    );
    let config = SoilTractorConfig {
        seed: [8, 2, 4],
        ..Default::default()
    };
    let coupling = MpmUnilateralConfig {
        node_capacity: 4096,
        search_distance: 0.05,
        max_iterations: 64,
    };
    let storage = MpmGridStorage::Compact {
        tile_capacity: 1024,
    };
    let create = |transfer| {
        SoilTractor::new_native_prediction_with_newton_transfer(
            engine.clone(),
            config,
            coupling,
            0.3,
            storage,
            1.0 / 60.0,
            transfer,
        )
    };
    let allocations = DeviceScope::start().unwrap();
    assert!(
        create(SoilTractorTransfer::FixedOrder {
            max_gather_bytes: 4095
        })
        .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    let policy = SoilTractorTransfer::FixedOrder {
        max_gather_bytes: 4096,
    };
    let mut a = create(policy).unwrap();
    let mut b = create(policy).unwrap();
    for tractor in [&a, &b] {
        let info = tractor.world().mpm_soil_transfer_info().unwrap();
        assert!(info.fixed_order && info.particle_ordering && info.parallel_sort);
        assert_eq!(info.gather_bytes, 4096);
        assert!(info.ordering_bytes > 0);
        assert_eq!(tractor.world().mpm_soil_grid_storage().unwrap().0, storage);
        assert!(tractor.world().mpm_soil_newton_enabled());
    }
    assert_eq!(machine_bits(&a), machine_bits(&b));
    for frame in 1..=4 {
        for tractor in [&mut a, &mut b] {
            let allocations = DeviceScope::start().unwrap();
            tractor.step(1.0 / 60.0, [300.0; 4]).unwrap();
            assert_eq!(allocations.finish().observations, 0);
            let stats = tractor.world().mpm_soil_stats().unwrap();
            assert_eq!(stats.committed_frames, frame);
            assert_eq!(stats.rolled_back_frames, 0);
            assert!(!stats.faulted);
        }
        assert_eq!(machine_bits(&a), machine_bits(&b));
        let positions_a = a.read_soil_positions().unwrap();
        let positions_b = b.read_soil_positions().unwrap();
        assert_eq!(positions_a.len(), 64);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&positions_a),
            bytemuck::cast_slice::<_, u8>(&positions_b)
        );
    }
    let atomic = SoilTractor::new_native_prediction_with_newton_grid(
        engine,
        config,
        coupling,
        0.3,
        storage,
        1.0 / 60.0,
    )
    .unwrap();
    let info = atomic.world().mpm_soil_transfer_info().unwrap();
    assert!(!info.fixed_order && !info.particle_ordering && !info.parallel_sort);
    assert_eq!((info.gather_bytes, info.ordering_bytes), (0, 0));
    println!(
        "public fixed-order tractor: 64 particles, two independent owners, four exact machine/position frames, zero hot buffers, pre-allocation budget rejection, unchanged atomic constructor"
    );
}
