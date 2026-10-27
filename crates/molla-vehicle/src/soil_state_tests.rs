use super::*;

fn tile(x: i64) -> SoilTileCoord {
    SoilTileCoord([x, 0, 0])
}
fn registry() -> SoilTileResidency {
    SoilTileResidency::new(SoilResidencyLimits {
        regions: 3,
        tiles: 4,
        claims: 12,
    })
    .unwrap()
}

#[test]
fn tile_membership_is_half_open_signed_and_checked() {
    let layout = SoilTileLayout::new([0.0; 3], 0.25).unwrap();
    for (point, expected) in [
        (-0.5, -2),
        (-0.25, -1),
        (-f64::MIN_POSITIVE, -1),
        (-0.0, 0),
        (0.0, 0),
        (0.24999999999999997, 0),
        (0.25, 1),
        (0.5, 2),
    ] {
        assert_eq!(
            layout.tile_at([point; 3]).unwrap(),
            SoilTileCoord([expected; 3])
        );
    }
    assert_eq!(
        SoilTileLayout::new([-100.0, 200.0, 0.0], 0.5)
            .unwrap()
            .tile_at([-100.5, 200.5, -0.5])
            .unwrap(),
        SoilTileCoord([-1, 1, -1])
    );
    for bad in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        (1_u64 << 52) as f64,
    ] {
        assert!(layout.tile_at([bad; 3]).is_err());
    }
    for edge in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(SoilTileLayout::new([0.0; 3], edge).is_err());
    }
}

#[test]
fn overlap_shares_tiles_handoffs_do_not_reload_and_pins_outlive_regions() {
    let mut state = registry();
    let a = [tile(0), tile(1), tile(1)];
    let b = [tile(1), tile(2)];
    let plan = state
        .plan(
            &[
                SoilRegionRequest {
                    id: SoilRegionId(9),
                    tiles: &a,
                },
                SoilRegionRequest {
                    id: SoilRegionId(2),
                    tiles: &b,
                },
            ],
            &[tile(3)],
        )
        .unwrap();
    assert_eq!(plan.activate(), &[tile(0), tile(1), tile(2), tile(3)]);
    assert!(plan.deactivate().is_empty());
    assert_eq!(
        plan.claims(tile(1)).unwrap().regions().collect::<Vec<_>>(),
        vec![SoilRegionId(2), SoilRegionId(9)]
    );
    assert_eq!(
        plan.claims(tile(1)).unwrap().coordinator(),
        Some(SoilRegionId(2))
    );
    assert!(plan.claims(tile(3)).unwrap().pinned());
    assert_eq!(plan.claims(tile(3)).unwrap().coordinator(), None);
    state.commit(plan).unwrap();
    let plan = state
        .plan(
            &[SoilRegionRequest {
                id: SoilRegionId(9),
                tiles: &a,
            }],
            &[tile(2), tile(3)],
        )
        .unwrap();
    assert!(plan.activate().is_empty() && plan.deactivate().is_empty());
    assert_eq!(
        plan.claims(tile(1)).unwrap().coordinator(),
        Some(SoilRegionId(9))
    );
    state.commit(plan).unwrap();
    let plan = state.plan(&[], &[tile(2), tile(3)]).unwrap();
    assert_eq!(plan.deactivate(), &[tile(0), tile(1)]);
    state.commit(plan).unwrap();
    assert_eq!(state.tiles().collect::<Vec<_>>(), vec![tile(2), tile(3)]);
}

#[test]
fn failures_cancellation_and_foreign_or_stale_plans_do_not_mutate() {
    let mut state = registry();
    let live = state.plan(&[], &[tile(1)]).unwrap();
    let stale = state.plan(&[], &[tile(2)]).unwrap();
    state.commit(live).unwrap();
    assert_eq!(state.commit(stale), Err(SoilStateError::StalePlan));
    let foreign = registry().plan(&[], &[tile(2)]).unwrap();
    assert_eq!(state.commit(foreign), Err(SoilStateError::StalePlan));
    drop(state.plan(&[], &[tile(2)]).unwrap());
    let duplicate = [
        SoilRegionRequest {
            id: SoilRegionId(1),
            tiles: &[],
        },
        SoilRegionRequest {
            id: SoilRegionId(1),
            tiles: &[],
        },
    ];
    let too_many: Vec<_> = (0..4)
        .map(|id| SoilRegionRequest {
            id: SoilRegionId(id),
            tiles: &[],
        })
        .collect();
    assert!(matches!(
        state.plan(&too_many, &[]),
        Err(SoilStateError::CapacityExceeded)
    ));
    assert!(matches!(
        state.plan(&duplicate, &[]),
        Err(SoilStateError::DuplicateRegion)
    ));
    assert!(matches!(
        state.plan(&[], &[tile(0), tile(1), tile(2), tile(3), tile(4)]),
        Err(SoilStateError::CapacityExceeded)
    ));
    assert!(matches!(
        state.plan(&[], &[tile(0); 13]),
        Err(SoilStateError::CapacityExceeded)
    ));
    assert_eq!(state.tiles().collect::<Vec<_>>(), vec![tile(1)]);
}

#[test]
fn long_metadata_traverse_forgets_retired_tiles_and_is_order_independent() {
    let mut a = registry();
    let mut b = registry();
    for x in -1000..1000 {
        let first = [tile(x), tile(x + 1)];
        let second = [tile(x + 1), tile(x + 2)];
        let p = a
            .plan(
                &[
                    SoilRegionRequest {
                        id: SoilRegionId(7),
                        tiles: &first,
                    },
                    SoilRegionRequest {
                        id: SoilRegionId(3),
                        tiles: &second,
                    },
                ],
                &[tile(x + 3)],
            )
            .unwrap();
        let q = b
            .plan(
                &[
                    SoilRegionRequest {
                        id: SoilRegionId(3),
                        tiles: &second,
                    },
                    SoilRegionRequest {
                        id: SoilRegionId(7),
                        tiles: &first,
                    },
                ],
                &[tile(x + 3)],
            )
            .unwrap();
        assert_eq!(p.tiles, q.tiles);
        assert_eq!(p.activate(), q.activate());
        assert_eq!(p.deactivate(), q.deactivate());
        a.commit(p).unwrap();
        b.commit(q).unwrap();
        assert_eq!(a.tiles().len(), 4);
    }
}
