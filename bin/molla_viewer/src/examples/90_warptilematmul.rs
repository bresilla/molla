// ═══ Scene 90: WarpTileMatmul (Stage 37.8) ═════════════════════════════
//
// Standalone GEMM `C = A · B` demo. Fixed M=64, N=32, K=128 with
// deterministic-hash inputs. CPU/GPU toggle exercises both
// `tile_matmul::matmul_cpu` and the GPU `TileMatmulKernel`. Renders A
// (M × K) on the left, B (K × N) in the middle, and C (M × N) on the
// right as colored point grids; each cell's hue encodes the value.

const TILE_MATMUL_M: usize = 64;
const TILE_MATMUL_N: usize = 32;
const TILE_MATMUL_K: usize = 128;

struct WarpTileMatmulState {
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    use_gpu: bool,
    last_us: f64,
    needs_recompute: bool,
    /// Per-frame display scale so all three grids share a consistent
    /// colour range; updated from the latest |C| max.
    c_abs_max: f32,
}

impl Default for WarpTileMatmulState {
    fn default() -> Self {
        let m = TILE_MATMUL_M;
        let n = TILE_MATMUL_N;
        let k = TILE_MATMUL_K;
        // Deterministic LCG so the demo doesn't shimmer between runs.
        let mut seed = 0xA110_BEEF_u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as i32 as f32) / 32768.0
        };
        let mut a = vec![0.0_f32; m * k];
        let mut b = vec![0.0_f32; k * n];
        for v in a.iter_mut() {
            *v = rng();
        }
        for v in b.iter_mut() {
            *v = rng();
        }
        Self {
            a,
            b,
            c: vec![0.0_f32; m * n],
            use_gpu: false,
            last_us: 0.0,
            needs_recompute: true,
            c_abs_max: 1.0,
        }
    }
}

fn update_warp_tile_matmul(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileMatmul {
        return;
    }
    let st = &mut warp.tile_matmul;
    if !st.needs_recompute {
        return;
    }
    let m = TILE_MATMUL_M;
    let n = TILE_MATMUL_N;
    let k = TILE_MATMUL_K;
    let mut c = vec![0.0_f32; m * n];

    let t0 = std::time::Instant::now();
    let used_gpu = if st.use_gpu {
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let kernel = molla::compute::tile_matmul::TileMatmulKernel::new(device);
            kernel
                .dispatch_blocking(device, queue, &st.a, &st.b, &mut c, m, n, k)
                .is_ok()
        } else {
            false
        }
    } else {
        false
    };
    if !used_gpu {
        molla::compute::tile_matmul::matmul_cpu(&st.a, &st.b, &mut c, m, n, k).expect("matmul_cpu");
    }
    st.last_us = t0.elapsed().as_secs_f64() * 1.0e6;
    // Track absolute-value range of C for consistent colour scaling.
    let mut mx = 1.0_f32;
    for &v in c.iter() {
        let av = v.abs();
        if av > mx {
            mx = av;
        }
    }
    st.c_abs_max = mx;
    st.c = c;
    st.needs_recompute = false;
}

// Map a scalar `v` normalized to roughly `[-1, 1]` to a Bevy colour.
fn tile_matmul_color(v: f32) -> Color {
    let t = ((v + 1.0) * 0.5).clamp(0.0, 1.0);
    if t < 0.5 {
        let s = t * 2.0;
        Color::srgb(0.05, 0.20 + 0.65 * s, 1.0 - 0.6 * s)
    } else {
        let s = (t - 0.5) * 2.0;
        Color::srgb(0.05 + 0.95 * s, 0.85 - 0.55 * s, 0.40 - 0.40 * s)
    }
}

fn draw_warp_tile_matmul_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpTileMatmul {
        return;
    }
    let st = &warp.tile_matmul;
    let m = TILE_MATMUL_M;
    let n = TILE_MATMUL_N;
    let k = TILE_MATMUL_K;
    // Each grid is rendered in a 1×1 box at y ≈ 0.05. The three grids
    // sit side-by-side along +X with a small gap. K is the largest axis
    // (128), so each tile of A/B is ~1/128 wide.
    let panel_w = 1.0_f32;
    let panel_h = 1.0_f32;
    let gap = 0.15_f32;
    // A at x ∈ [-1.7, -0.7]
    let ax_lo = -1.5_f32 - gap;
    // B at x ∈ [-0.5, 0.5]
    let bx_lo = -0.5_f32;
    // C at x ∈ [0.7, 1.7]
    let cx_lo = 0.5_f32 + gap;
    // A: M × K, rendered with M on z, K on x.
    let cell_a_x = panel_w / k as f32;
    let cell_a_z = panel_h / m as f32;
    let a_inv_max = 1.0_f32 / st.a.iter().map(|v| v.abs()).fold(1e-6_f32, f32::max);
    for i in 0..m {
        for p in 0..k {
            let v = st.a[i * k + p];
            let x = ax_lo + (p as f32 + 0.5) * cell_a_x;
            let z = -0.5 + (i as f32 + 0.5) * cell_a_z;
            gizmos.sphere(
                Vec3::new(x, 0.05, z),
                cell_a_z.min(cell_a_x) * 0.4,
                tile_matmul_color(v * a_inv_max),
            );
        }
    }
    // B: K × N, rendered with K on z, N on x.
    let cell_b_x = panel_w / n as f32;
    let cell_b_z = panel_h / k as f32;
    let b_inv_max = 1.0_f32 / st.b.iter().map(|v| v.abs()).fold(1e-6_f32, f32::max);
    for p in 0..k {
        for j in 0..n {
            let v = st.b[p * n + j];
            let x = bx_lo + (j as f32 + 0.5) * cell_b_x;
            let z = -0.5 + (p as f32 + 0.5) * cell_b_z;
            gizmos.sphere(
                Vec3::new(x, 0.05, z),
                cell_b_x.min(cell_b_z) * 0.4,
                tile_matmul_color(v * b_inv_max),
            );
        }
    }
    // C: M × N, rendered with M on z, N on x.
    let cell_c_x = panel_w / n as f32;
    let cell_c_z = panel_h / m as f32;
    let c_inv_max = 1.0_f32 / st.c_abs_max.max(1e-6);
    for i in 0..m {
        for j in 0..n {
            let v = st.c[i * n + j];
            let x = cx_lo + (j as f32 + 0.5) * cell_c_x;
            let z = -0.5 + (i as f32 + 0.5) * cell_c_z;
            gizmos.sphere(
                Vec3::new(x, 0.05, z),
                cell_c_x.min(cell_c_z) * 0.4,
                tile_matmul_color(v * c_inv_max),
            );
        }
    }
}

fn warp_tile_matmul_pane(ui: &mut egui::Ui, st: &mut WarpTileMatmulState) {
    ui.label("Warp tile matmul — Stage 37.8 GEMM `C = A · B`");
    ui.separator();
    readout(
        ui,
        "Shapes",
        &format!(
            "A={M}×{K}, B={K}×{N}, C={M}×{N}",
            M = TILE_MATMUL_M,
            N = TILE_MATMUL_N,
            K = TILE_MATMUL_K,
        ),
    );
    readout(ui, "Last solve", &format!("{:.1} µs", st.last_us));
    readout(ui, "max|C|", &format!("{:.3}", st.c_abs_max));
    ui.separator();
    let was_gpu = st.use_gpu;
    let _ = ui.checkbox(&mut st.use_gpu, "Use GPU");
    if was_gpu != st.use_gpu {
        st.needs_recompute = true;
    }
    let backend = if st.use_gpu {
        "GPU TileMatmulKernel"
    } else {
        "CPU matmul_cpu"
    };
    readout(ui, "Backend", backend);
    if ui.button("Re-randomise inputs").clicked() {
        // Re-seed off the wall clock so the demo can show new inputs.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let mut seed = (nanos ^ 0xA110_BEEF) as u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as i32 as f32) / 32768.0
        };
        for v in st.a.iter_mut() {
            *v = rng();
        }
        for v in st.b.iter_mut() {
            *v = rng();
        }
        st.needs_recompute = true;
    }
    if ui.button("Re-run").clicked() {
        st.needs_recompute = true;
    }
    ui.small("Left grid: A (M × K). Middle: B (K × N). Right: C (M × N).");
    ui.small("Stage 37.8 primitive: tile_matmul (`matmul_cpu` / `TileMatmulKernel`).");
}

