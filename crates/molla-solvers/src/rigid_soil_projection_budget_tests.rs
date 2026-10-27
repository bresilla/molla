use super::*;
use crate::mpm::MpmConfig;
use molla_core::gpu_metrics::{DeviceScope, TransferScope};

fn config(size: u32) -> MpmConfig {
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = size;
    cfg.max_particles = 1;
    cfg.max_rigid = 2;
    cfg
}

fn indexed_bytes(cells: u64) -> u64 {
    36 * cells
        + 1028 * cells.div_ceil(256)
        + 64
        + if cfg!(feature = "parallel-candidate-index") {
            1024
        } else {
            0
        }
}

#[test]
#[ignore = "requires a real GPU; runtime grid-capacity budget contract"]
fn projected_runtime_grid_capacity_budgets() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let config_contact = MpmUnilateralConfig {
        node_capacity: 128,
        search_distance: 0.05,
        max_iterations: 64,
    };
    let modes = [
        (
            Mode::Admm {
                config: config_contact,
                coefficients: [0.3; 64],
            },
            RateEncoding::Paired,
        ),
        (
            Mode::Sticking {
                nodes_per_body: 128,
                boundary_scan: SoilBoundaryScan::Full,
            },
            RateEncoding::Scalar,
        ),
        (Mode::Unilateral(config_contact), RateEncoding::Scalar),
        (
            Mode::Friction {
                config: config_contact,
                coefficients: [0.3; 64],
                fixed_walls: false,
                newton: false,
            },
            RateEncoding::Scalar,
        ),
        (
            Mode::Friction {
                config: config_contact,
                coefficients: [0.3; 64],
                fixed_walls: true,
                newton: false,
            },
            RateEncoding::Scalar,
        ),
        (
            Mode::Friction {
                config: config_contact,
                coefficients: [0.3; 64],
                fixed_walls: true,
                newton: false,
            },
            RateEncoding::Paired,
        ),
        (
            Mode::Friction {
                config: config_contact,
                coefficients: [0.3; 64],
                fixed_walls: true,
                newton: true,
            },
            RateEncoding::Paired,
        ),
    ];
    let mut dense = MpmSimulation::new(&engine, config(16)).unwrap();
    let old_dense = [
        PreparedMpmUnilateralCoupling::required_admm_bytes(config_contact, 3, 16, 1).unwrap(),
        4096 * 4 + 128 * 32 + 168 + 4 * 3 * 3 + 8 * 3 + 96,
        PreparedMpmUnilateralCoupling::required_bytes(config_contact, 3, 16, 1).unwrap(),
        PreparedMpmUnilateralCoupling::required_friction_bytes(config_contact, 3, 16, 1).unwrap(),
        PreparedMpmUnilateralCoupling::required_wall_friction_bytes_for_rates(
            config_contact,
            3,
            16,
            1,
            RateEncoding::Scalar,
        )
        .unwrap(),
        PreparedMpmUnilateralCoupling::required_wall_friction_bytes_for_rates(
            config_contact,
            3,
            16,
            1,
            RateEncoding::Paired,
        )
        .unwrap(),
        PreparedMpmUnilateralCoupling::required_newton_bytes(config_contact, 3, 16, 1).unwrap(),
    ];
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    assert!(
        modes[0]
            .0
            .coupling_bytes(&dense, 3, 1, RateEncoding::Scalar)
            .is_err()
    );
    for ((mode, rates), expected) in modes.iter().zip(old_dense) {
        assert_eq!(mode.coupling_bytes(&dense, 3, 1, *rates).unwrap(), expected);
    }
    assert_eq!(allocations.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!(transfers.upload_bytes + transfers.readback_bytes, 0);
    dense.set_active_grid(true).unwrap();
    for (i, (mode, rates)) in modes.iter().enumerate() {
        let extra = if i == 1 {
            0
        } else {
            16 * 4096 + 1024 * 16 + 64
        };
        assert_eq!(
            mode.coupling_bytes(&dense, 3, 1, *rates).unwrap(),
            old_dense[i] + extra
        );
    }
    for size in [17, 96, 1536] {
        let mut soil = MpmSimulation::new_sparse(&engine, config(size), 125).unwrap();
        let allocations = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let mut before = vec![];
        for (i, (mode, rates)) in modes.iter().enumerate() {
            let expected = if i == 1 {
                old_dense[i] - 4 * 4096 + 4 * 8000
            } else {
                old_dense[i] - (20 * 4096 + 4 * 16) + indexed_bytes(8000)
            };
            let actual = mode.coupling_bytes(&soil, 3, 1, *rates).unwrap();
            assert_eq!(actual, expected, "world={size} mode={i}");
            for (dofs, bodies) in [(0, 1), (33, 1), (3, 0), (3, 65)] {
                assert!(mode.coupling_bytes(&soil, dofs, bodies, *rates).is_err());
            }
            before.push(actual);
        }
        assert_eq!(before[1], 36420);
        assert_eq!(before[2], 362268);
        assert!(PreparedMpmUnilateralCoupling::required_bytes_for_grid(&soil, |_| Ok(0)).is_err());
        assert!(
            PreparedMpmUnilateralCoupling::required_bytes_for_grid(&soil, |_| Ok(u64::MAX))
                .is_err()
        );
        assert!(PreparedMpmMassCoupling::required_bytes(&soil, 3, 1, 0).is_err());
        assert!(PreparedMpmMassCoupling::required_bytes(&soil, 3, 2, u32::MAX).is_err());
        assert_eq!(allocations.finish().observations, 0);
        let transfers = transfers.finish();
        assert_eq!(transfers.upload_bytes + transfers.readback_bytes, 0);
        if size == 96 {
            soil.reserve_sparse_tiles(216).unwrap();
            for (i, (mode, rates)) in modes.iter().enumerate() {
                let extra = if i == 1 {
                    4 * (216 - 125) * 64
                } else {
                    indexed_bytes(216 * 64) - indexed_bytes(125 * 64)
                };
                assert_eq!(
                    mode.coupling_bytes(&soil, 3, 1, *rates).unwrap(),
                    before[i] + extra
                );
            }
        }
        println!(
            "runtime world={size} 125-tile coupling budgets={before:?}; seven modes, dense identity, bounds and growth pass"
        );
    }
}
