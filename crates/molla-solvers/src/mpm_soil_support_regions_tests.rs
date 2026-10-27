use super::*;

#[test]
fn quiescence_region_support_budget_and_shader() {
    let limits = wgpu::Limits::default();
    assert_eq!(size(3, 96, &limits).unwrap(), 96);
    for (capacity, budget) in [(0, 0), (65, 2080), (3, 95)] {
        assert!(size(capacity, budget, &limits).is_err());
    }
    let source = include_str!("../shaders/mpm_soil_support_regions.wgsl");
    let module = naga::front::wgsl::parse_str(source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; region-position seeds, explicit-ID equivalence and bounded resources"]
fn quiescence_gpu_region_support_seeds_and_lifecycle() {
    use super::super::retained::tests::identified;
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let positions = [
        [-1., 0.5, 0.],
        [-0.8, 0.5, 0.],
        [-0.6, 0.5, 0.],
        [1., 0.5, 0.],
    ];
    let first = [-1., 0.5, 0., 0., -1., 0.5, 0., 0.];
    let isolated = [1., 0.5, 0., 0., 1., 0.5, 0., 0.];
    for mode in 0..3 {
        let (mut sim, snapshot, ids) = identified(&engine, &positions, mode);
        let before = sim
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .to_bytes();
        assert!(
            sim.read_soil_region_support(&engine, &snapshot, 2, 129, 2, &[first])
                .is_err()
        );
        let allocations = DeviceScope::start().unwrap();
        assert!(sim.prepare_soil_support_regions(3, 95).is_err());
        assert_eq!(allocations.finish().observations, 0);
        sim.prepare_soil_support_regions(3, 96).unwrap();
        assert_eq!(sim.soil_support_region_bytes(), 96);
        assert_eq!(sim.soil_support_bytes(), 1049688);
        let allocations = DeviceScope::start().unwrap();
        sim.prepare_soil_support_regions(3, 96).unwrap();
        assert_eq!(allocations.finish().observations, 0);
        let expected = sim
            .read_soil_retained_support(&engine, &snapshot, 2, 129, 2, &ids[..1])
            .unwrap();
        let allocations = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let dispatches = DispatchScope::start().unwrap();
        let result = sim
            .read_soil_region_support(&engine, &snapshot, 2, 129, 2, &[first])
            .unwrap();
        assert_eq!(allocations.finish().observations, 0);
        let transfers = transfers.finish();
        assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (96, 32));
        assert_eq!(dispatches.finish().encoded_dispatches, 7);
        assert_eq!(result.particles(), expected.particles());
        assert_eq!(result.requested_particles(), &ids[..1]);
        assert!(
            sim.read_soil_region_support(&engine, &snapshot, 2, 129, 1, &[first])
                .unwrap_err()
                .to_string()
                .contains("did not converge")
        );
        let union = sim
            .read_soil_region_support(&engine, &snapshot, 2, 129, 2, &[isolated, first])
            .unwrap();
        assert!(union.particles().iter().all(|p| p.retain));
        assert_eq!(union.requested_particles(), [ids[0], ids[3]]);
        assert_eq!(
            sim.read_soil_region_support(&engine, &snapshot, 2, 129, 2, &[first, isolated, first])
                .unwrap()
                .particles(),
            union.particles()
        );
        let dispatches = DispatchScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let empty = sim
            .read_soil_region_support(&engine, &snapshot, 2, 129, 2, &[])
            .unwrap();
        assert!(empty.particles().iter().all(|p| !p.retain && !p.requested));
        assert_eq!(dispatches.finish().encoded_dispatches, 6);
        let transfers = transfers.finish();
        assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (64, 32));
        for bad in [
            vec![first; 4],
            vec![[f32::NAN; 8]],
            vec![[1., 0., 0., 0., -1., 0., 0., 0.]],
        ] {
            let transfers = TransferScope::start().unwrap();
            let dispatches = DispatchScope::start().unwrap();
            assert!(
                sim.read_soil_region_support(&engine, &snapshot, 2, 129, 2, &bad)
                    .is_err()
            );
            let transfers = transfers.finish();
            assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
            assert_eq!(dispatches.finish().encoded_dispatches, 0);
        }
        assert!(sim.prepare_soil_support_regions(2, 63).is_err());
        assert_eq!(sim.soil_support_region_capacity(), 3);
        assert_eq!(
            sim.read_soil_retained_support(&engine, &snapshot, 2, 129, 2, &ids[..1])
                .unwrap()
                .particles(),
            expected.particles()
        );
        assert!(
            sim.read_soil_support_closure(&engine, &snapshot, 2, 129, 2)
                .unwrap()
                .particles()
                .iter()
                .all(|p| !p.connected_to_nonquiet)
        );
        assert_eq!(
            sim.checkpoint_soil_with(engine.queue().unwrap())
                .unwrap()
                .to_bytes(),
            before
        );
        sim.disable_soil_support();
        assert_eq!(sim.soil_support_region_bytes(), 0);
        sim.prepare_soil_support(1049688).unwrap();
        assert!(
            sim.read_soil_region_support(&engine, &snapshot, 2, 129, 2, &[first])
                .is_err()
        );
        println!(
            "region-support mode={mode} explicit_id_parity=true physical_bytes_exact=true seed_dispatches=1 total_dispatches=7 upload=96 readback=32 region_bytes=96"
        );
    }
    let (mut sim, snapshot, ids) = identified(&engine, &vec![[0., 0.5, 0.]; 129], 0);
    sim.prepare_soil_support_regions(1, 32).unwrap();
    let result = sim
        .read_soil_region_support(
            &engine,
            &snapshot,
            2,
            129,
            1,
            &[[0., 0.5, 0., 0., 0., 0.5, 0., 0.]],
        )
        .unwrap();
    assert_eq!(result.requested_particles(), ids);
    assert!(
        result
            .particles()
            .iter()
            .all(|p| p.quiet && p.requested && p.retain)
    );
    println!("region-support cross_workgroup=129 closed_boundary=true");
}
