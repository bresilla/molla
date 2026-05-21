// ═══ Scene 82: WarpTileCompaction (Stage 37.8) ═════════════════════════
//
// Random 3D point cloud + a "keep if inside radius R" predicate. Each
// frame builds the mask, calls `tile_stream_compaction` (CPU or GPU)
// to extract the surviving point indices, and renders kept points as
// bright spheres with rejected points faded. CPU/GPU toggle exercises
// both paths.

const TILE_COMPACT_N: usize = 256;

struct WarpTileCompactionState {
    /// 3D positions, fixed at construction.
    positions: Vec<Vec3>,
    /// Radius threshold; points with `‖p‖ < radius` are kept.
    radius: f32,
    /// Indices of surviving points after the last compaction (CPU
    /// reads + draw walks this).
    kept: Vec<u32>,
    use_gpu: bool,
    last_us: f64,
    needs_recompute: bool,
}

impl Default for WarpTileCompactionState {
    fn default() -> Self {
        let n = TILE_COMPACT_N;
        let mut positions = Vec::with_capacity(n);
        // Deterministic LCG so the cloud doesn't shimmer between
        // sessions or test runs.
        let mut seed = 0xC0DE_u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as i32 as f32) / 32768.0
        };
        for _ in 0..n {
            positions.push(Vec3::new(rng(), rng(), rng()));
        }
        Self {
            positions,
            radius: 0.5,
            kept: Vec::new(),
            use_gpu: false,
            last_us: 0.0,
            needs_recompute: true,
        }
    }
}

fn update_warp_tile_compaction(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileCompaction {
        return;
    }
    let st = &mut warp.tile_compaction;
    if !st.needs_recompute {
        return;
    }
    let n = st.positions.len();
    // Index array used as the input — the compaction primitive returns
    // the surviving entries; passing point indices means we get
    // surviving point indices back (then we look up positions to
    // render).
    let input_f32: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mask: Vec<bool> = st
        .positions
        .iter()
        .map(|p| p.length() < st.radius)
        .collect();

    let t0 = std::time::Instant::now();
    let kept_f32: Vec<f32> = if st.use_gpu {
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let kernel =
                molla::compute::tile_stream_compaction::TileStreamCompactionKernel::new(device);
            let mask_u32 = molla::compute::tile_stream_compaction::mask_to_u32(&mask);
            kernel
                .dispatch_blocking(device, queue, &input_f32, &mask_u32)
                .unwrap_or_else(|_| {
                    molla::compute::tile_stream_compaction::compact_cpu(&input_f32, &mask)
                        .expect("compact_cpu fallback")
                })
        } else {
            molla::compute::tile_stream_compaction::compact_cpu(&input_f32, &mask)
                .expect("compact_cpu")
        }
    } else {
        molla::compute::tile_stream_compaction::compact_cpu(&input_f32, &mask).expect("compact_cpu")
    };
    st.last_us = t0.elapsed().as_secs_f64() * 1.0e6;
    st.kept = kept_f32.into_iter().map(|x| x as u32).collect();
    st.needs_recompute = false;
}

fn draw_warp_tile_compaction_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpTileCompaction {
        return;
    }
    let st = &warp.tile_compaction;
    // Draw all points as faded gray dots, then re-draw kept ones as
    // bright spheres on top.
    for p in &st.positions {
        gizmos.sphere(*p, 0.005, Color::srgb(0.3, 0.3, 0.3));
    }
    for &idx in &st.kept {
        let p = st.positions[idx as usize];
        gizmos.sphere(p, 0.012, Color::srgb(0.2, 0.9, 0.4));
    }
    // Show the radius threshold as a wireframe sphere.
    let segments = 64;
    for axis in 0..3 {
        for s in 0..segments {
            let t0 = (s as f32 / segments as f32) * std::f32::consts::TAU;
            let t1 = ((s + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            let r = st.radius;
            let (a, b) = match axis {
                0 => (
                    Vec3::new(0.0, r * t0.cos(), r * t0.sin()),
                    Vec3::new(0.0, r * t1.cos(), r * t1.sin()),
                ),
                1 => (
                    Vec3::new(r * t0.cos(), 0.0, r * t0.sin()),
                    Vec3::new(r * t1.cos(), 0.0, r * t1.sin()),
                ),
                _ => (
                    Vec3::new(r * t0.cos(), r * t0.sin(), 0.0),
                    Vec3::new(r * t1.cos(), r * t1.sin(), 0.0),
                ),
            };
            gizmos.line(a, b, Color::srgb(0.7, 0.5, 0.3));
        }
    }
}

fn warp_tile_compaction_pane(ui: &mut egui::Ui, st: &mut WarpTileCompactionState) {
    ui.label("Warp tile compaction — Stage 37.8 stream compaction");
    ui.separator();
    readout(ui, "N points", &format!("{}", st.positions.len()));
    readout(ui, "Kept", &format!("{}", st.kept.len()));
    readout(ui, "Last solve", &format!("{:.1} µs", st.last_us));
    ui.separator();
    ui.label("Radius threshold");
    if ui
        .add(egui::Slider::new(&mut st.radius, 0.05..=1.5))
        .changed()
    {
        st.needs_recompute = true;
    }
    ui.separator();
    let was_gpu = st.use_gpu;
    let _ = ui.checkbox(&mut st.use_gpu, "Use GPU");
    if was_gpu != st.use_gpu {
        st.needs_recompute = true;
    }
    let backend = if st.use_gpu {
        "GPU TileStreamCompactionKernel"
    } else {
        "CPU compact_cpu"
    };
    readout(ui, "Backend", backend);
    if ui.button("Re-shuffle predicate").clicked() {
        st.needs_recompute = true;
    }
    ui.small("Green spheres = points kept by `‖p‖ < radius`.");
    ui.small("Wireframe sphere = the radius threshold.");
}

