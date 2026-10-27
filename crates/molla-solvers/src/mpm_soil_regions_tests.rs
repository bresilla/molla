use super::*;

#[test]
fn support_region_tile_boxes_cover_membership_boundaries() {
    let points = [
        -1.0e6f32,
        -1.0000001,
        -1.,
        -0.99999994,
        -0.3,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.,
        0.,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        0.3,
        0.99999994,
        1.,
        1.0000001,
        1.0e6,
    ];
    let mut checked = 0;
    for edge in [0.125, 0.3, 1e-6, 1e6] {
        for offset in [0., 0.123456789, 1e12] {
            let layout = SoilTileLayout::new([offset; 3], edge).unwrap();
            for origin in [None, Some([1e12, -1e12, 1e12]), Some([0.123456789; 3])] {
                for p in points {
                    let point = [f64::from(p); 3];
                    let tile = match origin {
                        Some(o) => layout.tile_at_local(point, o),
                        None => layout.tile_at(point),
                    };
                    let Ok(tile) = tile else {
                        continue;
                    };
                    let b = support_box(layout, tile, tile, origin).unwrap();
                    for a in 0..3 {
                        assert!(
                            b[a] <= p && p <= b[4 + a],
                            "point={p:?} tile={tile:?} box={b:?} origin={origin:?} edge={edge}"
                        );
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 100);
    println!("region-support tile_membership_boundary_cases={checked}");
    assert!(
        support_box(
            SoilTileLayout::new([0.; 3], f64::MAX).unwrap(),
            SoilTileCoord([2; 3]),
            SoilTileCoord([3; 3]),
            None
        )
        .is_err()
    );
}

#[test]
fn support_region_boxes_preserve_request_limits_and_tile_extent() {
    let layout = SoilTileLayout::new([0.; 3], 0.5).unwrap();
    let cfg = MpmConfig {
        cell_size: 0.1,
        grid_origin: [-3.; 3],
        time_step: MpmTimeStep::Explicit,
        ..MpmConfig::sand()
    };
    let e = MpmSoilRegionEnvelope {
        minimum: [0.01; 3],
        maximum: [0.01; 3],
        substeps: 1,
        particle_travel_per_substep: [0.; 3],
        ..envelope()
    };
    let limits = SoilResidencyLimits {
        regions: 3,
        tiles: 4096,
        claims: 8192,
    };
    let boxes = support_boxes(&[e], layout, &cfg, limits, None, 2).unwrap();
    assert_eq!(boxes.len(), 1);
    assert!(boxes[0][0] <= -0.5 && boxes[0][4] >= 0.5);
    assert!(support_boxes(&[e, e], layout, &cfg, limits, None, 2).is_err());
    assert!(support_boxes(&[e], layout, &cfg, limits, None, 0).is_err());
    assert!(
        support_boxes(
            &[e],
            layout,
            &cfg,
            SoilResidencyLimits {
                claims: 1,
                ..limits
            },
            None,
            2
        )
        .is_err()
    );
    assert!(
        support_boxes(&[], layout, &cfg, limits, None, 2)
            .unwrap()
            .is_empty()
    );
    let invalid = MpmSoilRegionEnvelope {
        minimum: [f64::NAN; 3],
        ..e
    };
    assert!(support_boxes(&[invalid], layout, &cfg, limits, None, 2).is_err());
}

#[test]
fn framed_envelopes_and_pins_preserve_local_padding_at_large_offsets() {
    let cfg = MpmConfig {
        cell_size: 0.03125,
        grid_origin: [-4.0; 3],
        ..MpmConfig::sand()
    };
    let layout = SoilTileLayout::new([0.0; 3], 1.0).unwrap();
    let origin = [1e12, -1e12, 1e12];
    let shift =
        |tile: SoilTileCoord| SoilTileCoord(std::array::from_fn(|a| tile.0[a] + origin[a] as i64));
    let e = MpmSoilRegionEnvelope {
        minimum: [-1e-6; 3],
        maximum: [0.125; 3],
        ..envelope()
    };
    let local = e.tiles(layout, &cfg, 4096).unwrap();
    let framed = e.tiles_in_frame(layout, &cfg, 4096, Some(origin)).unwrap();
    assert_eq!(framed, local.into_iter().map(shift).collect::<Vec<_>>());
    let points = positions(&[[-1e-6; 3], [0.125; 3]]);
    let mut horizon = Horizon::default();
    for swept in [false, true] {
        if swept {
            horizon.request_swept(&[e]);
        }
        let local = horizon.particle_pins(&points, layout, &cfg, 4096).unwrap();
        let framed = horizon
            .particle_pins_in_frame(&points, layout, &cfg, 4096, Some(origin))
            .unwrap();
        assert_eq!(framed, local.into_iter().map(shift).collect::<Vec<_>>());
    }
    assert!(e.tiles_in_frame(layout, &cfg, 0, Some(origin)).is_err());
    assert!(
        e.tiles_in_frame(layout, &cfg, 4096, Some([f64::INFINITY; 3]))
            .is_err()
    );
}

fn envelope() -> MpmSoilRegionEnvelope {
    MpmSoilRegionEnvelope {
        id: SoilRegionId(7),
        minimum: [0.125; 3],
        maximum: [0.125; 3],
        particle_travel_per_substep: [0.0; 3],
        substeps: 1,
    }
}

fn positions(points: &[[f32; 3]]) -> Vec<u32> {
    points
        .iter()
        .flat_map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits(), 0])
        .collect()
}

#[test]
fn active_pins_cover_detached_soil_and_union_relative_motion_bounds() {
    let cfg = MpmConfig {
        cell_size: 0.03125,
        ..MpmConfig::sand()
    };
    let layout = SoilTileLayout::new([0.0; 3], 0.125).unwrap();
    let points = [
        [-0.001_f32, 0.33, 0.02],
        [-0.12, 0.34, 0.03],
        [4.0, 0.33, 0.02],
    ];
    let mut horizon = Horizon::default();
    horizon.request_swept(&[
        MpmSoilRegionEnvelope {
            substeps: 2,
            particle_travel_per_substep: [0.01, 0.02, 0.0],
            ..envelope()
        },
        MpmSoilRegionEnvelope {
            substeps: 3,
            particle_travel_per_substep: [0.03, 0.0, 0.01],
            ..envelope()
        },
    ]);
    let pins = horizon
        .particle_pins(&positions(&points), layout, &cfg, 4096)
        .unwrap();
    assert!(pins.windows(2).all(|p| p[0] < p[1]));
    for point in points {
        let expected = MpmSoilRegionEnvelope {
            minimum: point.map(f64::from),
            maximum: point.map(f64::from),
            substeps: 2,
            particle_travel_per_substep: [0.06, 0.04, 0.02],
            ..envelope()
        }
        .tiles(layout, &cfg, 4096)
        .unwrap();
        assert!(expected.iter().all(|t| pins.binary_search(t).is_ok()));
    }
    assert!(pins.contains(&SoilTileCoord([0, 2, 0])));
    assert!(pins.contains(&SoilTileCoord([32, 2, 0])));
    assert!(horizon.remaining.is_none());
    horizon.commit();
    assert_eq!(horizon.remaining, Some(2));
    assert_eq!(horizon.travel, Some([0.01, 0.0, 0.0]));
    assert!(!horizon.has_swept_pins());
    horizon.request(None);
    assert_eq!(horizon.travel, Some([0.01, 0.0, 0.0]));
    horizon.commit();
    assert_eq!(horizon.travel, None);
}

#[test]
fn raw_empty_and_replaced_requests_do_not_reuse_swept_pins() {
    let cfg = MpmConfig::sand();
    let layout = SoilTileLayout::new([0.0; 3], 0.125).unwrap();
    let words = positions(&[[0.124, 0.33, 0.02]]);
    let mut horizon = Horizon::default();
    for empty_swept in [false, true] {
        horizon.request_swept(&[envelope()]);
        assert!(horizon.particle_pins(&words, layout, &cfg, 1).is_err());
        if empty_swept {
            horizon.request_swept(&[]);
        } else {
            horizon.request(None);
        }
        assert_eq!(
            horizon.particle_pins(&words, layout, &cfg, 1).unwrap(),
            [SoilTileCoord([0, 2, 0])]
        );
        assert!(
            horizon
                .particle_pins(&[], layout, &cfg, 0)
                .unwrap()
                .is_empty()
        );
        horizon.commit();
        assert_eq!(horizon.remaining, empty_swept.then_some(0));
    }
}

#[test]
fn swept_pin_union_capacity_and_invalid_coordinates_fail_closed() {
    let cfg = MpmConfig {
        cell_size: 0.03125,
        ..MpmConfig::sand()
    };
    let layout = SoilTileLayout::new([0.0; 3], 0.125).unwrap();
    let mut horizon = Horizon::default();
    horizon.request(Some(7));
    horizon.commit();
    horizon.request_swept(&[envelope()]);
    let a = positions(&[[0.0625; 3]]);
    let b = positions(&[[4.0625; 3]]);
    let n = horizon.particle_pins(&a, layout, &cfg, 100).unwrap().len();
    assert_eq!(horizon.particle_pins(&b, layout, &cfg, n).unwrap().len(), n);
    assert!(
        horizon
            .particle_pins(&[a, b].concat(), layout, &cfg, n)
            .is_err()
    );
    assert_eq!(horizon.remaining, Some(7));
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            horizon
                .particle_pins(&positions(&[[value; 3]]), layout, &cfg, 100)
                .is_err()
        );
    }
}

#[test]
fn f32_shared_node_beyond_ideal_radius_still_activates_neighbor_tile() {
    let cfg = MpmConfig {
        cell_size: 0.001,
        grid_origin: [-1.0; 3],
        ..MpmConfig::sand()
    };
    let p = f32::from_bits(996499723);
    let q = f32::from_bits(1003814489);
    let coordinate = |x: f32| (x - cfg.grid_origin[0]) * (1.0 / cfg.cell_size);
    let xp = coordinate(p);
    let xq = coordinate(q);
    let bp = (xp - 0.5).floor();
    let bq = (xq - 0.5).floor();
    assert_eq!((bp, bq), (1003.0, 1005.0));
    assert!(xp - bp > 0.5 && xq - bq < 1.5);
    let old_upper = f64::from(p) + 3.0 * f64::from(cfg.cell_size);
    assert!(f64::from(q) > old_upper);
    let boundary = (old_upper + f64::from(q)) * 0.5;
    let layout = SoilTileLayout::new([boundary, 0.0, 0.0], 0.01).unwrap();
    let region = MpmSoilRegionEnvelope {
        minimum: [f64::from(p); 3],
        maximum: [f64::from(p); 3],
        ..envelope()
    };
    let neighbor = layout
        .tile_at([f64::from(q), f64::from(p), f64::from(p)])
        .unwrap();
    assert_eq!(
        layout
            .tile_at([old_upper.next_up(), f64::from(p), f64::from(p)])
            .unwrap()
            .0[0],
        -1
    );
    assert_eq!(neighbor.0[0], 0);
    assert!(region.tiles(layout, &cfg, 64).unwrap().contains(&neighbor));
}

#[test]
fn unsupported_shader_precision_ranges_reject_instead_of_shrinking_padding() {
    let layout = SoilTileLayout::new([0.0; 3], 1.0).unwrap();
    for cfg in [
        MpmConfig {
            cell_size: f32::from_bits(1),
            ..MpmConfig::sand()
        },
        MpmConfig {
            cell_size: f32::MAX,
            ..MpmConfig::sand()
        },
        MpmConfig {
            grid_origin: [f32::NAN; 3],
            ..MpmConfig::sand()
        },
        MpmConfig {
            grid_origin: [f32::MAX; 3],
            ..MpmConfig::sand()
        },
    ] {
        assert!(envelope().tiles(layout, &cfg, usize::MAX).is_err());
    }
    let region = MpmSoilRegionEnvelope {
        substeps: 131072,
        ..envelope()
    };
    assert!(
        region
            .tiles(layout, &MpmConfig::sand(), usize::MAX)
            .is_err()
    );
    let region = MpmSoilRegionEnvelope {
        minimum: [1e12; 3],
        maximum: [1e12; 3],
        ..envelope()
    };
    assert!(
        region
            .tiles(layout, &MpmConfig::sand(), usize::MAX)
            .is_err()
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn shader_stencil_roundoff_is_covered_by_activation_bounds() {
    use crate::mpm::replay_tests::read_words;
    use wgpu::util::DeviceExt;
    let mut cases = vec![
        [f32::from_bits(996499723), -1.0, 1.0 / 0.001_f32, 0.001],
        [f32::from_bits(1003814489), -1.0, 1.0 / 0.001_f32, 0.001],
    ];
    for h in [0.001_f32, 0.01, 0.1, 0.125] {
        for origin in [-1.0_f32, 0.0, 1000.0] {
            for index in [2, 7, 49, 1003] {
                for delta in -2_i32..=2 {
                    let p = origin + (index as f32 + 0.5) * h;
                    let q = origin + (index as f32 + 3.5) * h;
                    cases.push([
                        f32::from_bits(p.to_bits().wrapping_add_signed(delta)),
                        origin,
                        1.0 / h,
                        h,
                    ]);
                    cases.push([
                        f32::from_bits(q.to_bits().wrapping_add_signed(-delta)),
                        origin,
                        1.0 / h,
                        h,
                    ]);
                }
            }
        }
    }
    let production = include_str!("../shaders/mpm.wgsl");
    let p2g = &production[production.find("fn p2g(").unwrap()..];
    let stencil = &p2g[p2g.find("let xp =").unwrap()..p2g.find("let wx =").unwrap()];
    let weights = &production[production.find("fn quad_weights(").unwrap()..];
    let weights = &weights[..weights.find("\n}").unwrap() + 2];
    let source = format!(
        r#"
struct Params {{ grid_origin: vec3<f32>, inv_cell_size: f32 }}
@group(0) @binding(0) var<storage, read> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<u32>>;
{weights}
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if gid.x >= arrayLength(&inputs) {{ return; }}
    let row = inputs[gid.x];
    let pos = vec3<f32>(row.x);
    let u = Params(vec3<f32>(row.y), row.z);
    {stencil}
    let w = quad_weights(fx.x);
    output[gid.x] = vec4<u32>(bitcast<u32>(base.x), bitcast<u32>(fx.x), bitcast<u32>(w.x), bitcast<u32>(w.z));
}}
"#
    );
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("soil stencil cases"),
        contents: bytemuck::cast_slice(&cases),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("soil stencil results"),
        size: cases.len() as u64 * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production MPM stencil probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("soil stencil probe"),
        layout: None,
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("soil stencil probe"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
    }
    queue.submit([encoder.finish()]);
    let words = read_words(device, queue, &output);
    let mut connected = 0;
    for (pair, results) in cases.chunks_exact(2).zip(words.chunks_exact(8)) {
        let bp = results[0] as i32;
        let bq = results[4] as i32;
        if (i64::from(bp) - i64::from(bq)).abs() > 2 {
            continue;
        }
        connected += 1;
        let cfg = MpmConfig {
            cell_size: pair[0][3],
            grid_origin: [pair[0][1]; 3],
            ..MpmConfig::sand()
        };
        for (a, b) in [(pair[0][0], pair[1][0]), (pair[1][0], pair[0][0])] {
            let region = MpmSoilRegionEnvelope {
                minimum: [f64::from(a); 3],
                maximum: [f64::from(a); 3],
                ..envelope()
            };
            let (lo, hi) = region.bounds(&cfg).unwrap();
            assert!(
                lo[0] <= f64::from(b) && f64::from(b) <= hi[0],
                "uncovered GPU stencil: {pair:?} bases {bp}/{bq}"
            );
        }
    }
    assert!(connected > 100, "non-vacuous connected stencil coverage");
    println!(
        "GPU stencil pairs: {connected}/{} connected; fixed witness base/fx/outer weights {:?}",
        cases.len() / 2,
        words[..8]
            .chunks_exact(4)
            .map(|r| (
                r[0] as i32,
                f32::from_bits(r[1]),
                f32::from_bits(r[2]),
                f32::from_bits(r[3])
            ))
            .collect::<Vec<_>>()
    );
}

#[test]
fn refresh_horizon_changes_only_on_commit_and_counts_all_substeps() {
    let mut horizon = Horizon::default();
    assert_eq!(horizon.after_steps(usize::MAX).unwrap(), None);
    horizon.request(Some(3));
    assert_eq!(horizon.remaining, None);
    horizon.commit();
    assert!(horizon.after_steps(4).is_err());
    assert_eq!(horizon.remaining, Some(3));
    horizon.remaining = horizon.after_steps(2).unwrap();
    assert_eq!(horizon.remaining, Some(1));
    horizon.request(Some(100));
    assert!(horizon.after_steps(2).is_err());
    horizon.request(Some(1));
    horizon.commit();
    horizon.remaining = horizon.after_steps(1).unwrap();
    assert_eq!(horizon.after_steps(0).unwrap(), Some(0));
    assert!(horizon.after_steps(1).is_err());
    horizon.request(None);
    assert!(horizon.after_steps(1).is_err());
    horizon.commit();
    assert_eq!(horizon.after_steps(100).unwrap(), None);
}

#[test]
fn closed_swept_bounds_cover_negative_coordinates_and_tile_seams() {
    let layout = SoilTileLayout::new([0.0; 3], 0.125).unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.cell_size = 0.03125;
    let mut region = envelope();
    region.minimum = [-0.125, 0.0, -0.125];
    region.maximum = [0.125, 0.25, -0.125];
    let tiles = region.tiles(layout, &cfg, 32).unwrap();
    assert_eq!(tiles.len(), 32);
    assert_eq!(tiles[0], SoilTileCoord([-2, -1, -2]));
    assert_eq!(*tiles.last().unwrap(), SoilTileCoord([1, 2, -1]));
    assert!(tiles.windows(2).all(|p| p[0] < p[1]));
    assert!(region.tiles(layout, &cfg, 31).is_err());
    let translated = SoilTileLayout::new([1.0, -2.0, 0.5], 0.125).unwrap();
    let origin = translated.origin();
    for (axis, offset) in origin.into_iter().enumerate() {
        region.minimum[axis] += offset;
        region.maximum[axis] += offset;
    }
    assert_eq!(region.tiles(translated, &cfg, 32).unwrap(), tiles);
}

#[test]
fn implicit_neighbor_reach_motion_and_refresh_horizon_expand_independently() {
    let layout = SoilTileLayout::new([0.0; 3], 1.0).unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.cell_size = 0.125;
    let region = envelope();
    let base = region.tiles(layout, &cfg, 1000).unwrap();
    assert_eq!(base.len(), 8);
    cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 4 };
    let implicit = region.tiles(layout, &cfg, 1000).unwrap();
    assert_eq!(implicit.len(), 27);
    assert!(base.iter().all(|key| implicit.contains(key)));
    let moving = MpmSoilRegionEnvelope {
        particle_travel_per_substep: [0.25, 0.0, 0.0],
        substeps: 2,
        ..region
    };
    let tiles = moving.tiles(layout, &cfg, 1000).unwrap();
    assert_eq!(tiles[0], SoilTileCoord([-3, -2, -2]));
    assert_eq!(*tiles.last().unwrap(), SoilTileCoord([2, 1, 1]));
    assert_eq!(tiles.len(), 96);
}

#[test]
fn all_shared_quadratic_nodes_and_jacobi_paths_fit_one_step_padding() {
    let layout = SoilTileLayout::new([0.013, -0.027, 0.011], 0.07).unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.cell_size = 0.125;
    let h = f64::from(cfg.cell_size);
    for sweeps in [0, 1, 4] {
        cfg.time_step = MpmTimeStep::Implicit {
            jacobi_iters: sweeps,
        };
        for phase in -33..=33 {
            let x = f64::from(phase) * h / 16.0;
            let region = MpmSoilRegionEnvelope {
                minimum: [x; 3],
                maximum: [x; 3],
                ..envelope()
            };
            let tiles = region.tiles(layout, &cfg, 200_000).unwrap();
            let base = (x / h - 0.5).floor() as i32;
            for neighbor in (base - 2 - sweeps as i32)..=(base + 2 + sweeps as i32) {
                for fraction in [0.5, 0.500001, 1.0, 1.499999] {
                    let other = (f64::from(neighbor) + fraction) * h;
                    for axis in 0..3 {
                        let mut point = [x; 3];
                        point[axis] = other;
                        let key = layout.tile_at(point).unwrap();
                        assert!(
                            tiles.binary_search(&key).is_ok(),
                            "sweeps={sweeps} x={x} other={other} axis={axis}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_or_unbounded_requests_reject_without_enumerating() {
    let layout = SoilTileLayout::new([0.0; 3], 0.125).unwrap();
    let cfg = MpmConfig::sand();
    for invalid_region in [
        MpmSoilRegionEnvelope {
            substeps: 0,
            ..envelope()
        },
        MpmSoilRegionEnvelope {
            minimum: [f64::NAN; 3],
            ..envelope()
        },
        MpmSoilRegionEnvelope {
            maximum: [f64::INFINITY; 3],
            ..envelope()
        },
        MpmSoilRegionEnvelope {
            maximum: [-1.0; 3],
            ..envelope()
        },
        MpmSoilRegionEnvelope {
            particle_travel_per_substep: [-1.0; 3],
            ..envelope()
        },
        MpmSoilRegionEnvelope {
            particle_travel_per_substep: [f64::MAX; 3],
            substeps: 2,
            ..envelope()
        },
        MpmSoilRegionEnvelope {
            minimum: [-1e14; 3],
            maximum: [1e14; 3],
            ..envelope()
        },
    ] {
        assert!(invalid_region.tiles(layout, &cfg, usize::MAX).is_err());
    }
    assert!(envelope().tiles(layout, &cfg, 0).is_err());
    for size in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            envelope()
                .tiles(
                    layout,
                    &MpmConfig {
                        cell_size: size,
                        ..cfg
                    },
                    1000
                )
                .is_err()
        );
    }
}

#[test]
fn overlapping_regions_share_tiles_but_charge_all_claims() {
    let layout = SoilTileLayout::new([0.0; 3], 1.0).unwrap();
    let cfg = MpmConfig {
        cell_size: 0.125,
        ..MpmConfig::sand()
    };
    let regions = [
        envelope(),
        MpmSoilRegionEnvelope {
            id: SoilRegionId(9),
            ..envelope()
        },
    ];
    let limits = SoilResidencyLimits {
        regions: 2,
        tiles: 8,
        claims: 16,
    };
    let owned = expand(&regions, layout, &cfg, limits).unwrap();
    let residency = SoilTileResidency::new(limits).unwrap();
    let requests: Vec<_> = owned
        .iter()
        .map(|(id, tiles)| SoilRegionRequest { id: *id, tiles })
        .collect();
    let plan = residency.plan(&requests, &[]).unwrap();
    assert_eq!(plan.tiles().len(), 8);
    assert!(
        plan.tiles()
            .all(|key| plan.claims(key).unwrap().regions().len() == 2)
    );
    assert!(
        expand(
            &regions,
            layout,
            &cfg,
            SoilResidencyLimits {
                claims: 15,
                ..limits
            }
        )
        .is_err()
    );
    assert!(
        expand(
            &regions,
            layout,
            &cfg,
            SoilResidencyLimits {
                regions: 1,
                ..limits
            }
        )
        .is_err()
    );
    assert!(expand(&[envelope(), envelope()], layout, &cfg, limits).is_err());
}
