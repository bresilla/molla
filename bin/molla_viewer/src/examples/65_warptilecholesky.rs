// ── Scene 65: WarpTileCholesky ────────────────────────────────────
struct WarpTileCholeskyState {
    n: u32,
    /// Reconstruction error ‖L Lᵀ − A‖_F.
    last_err: f64,
    last_solve_us: f64,
    user_reset: bool,
}
impl Default for WarpTileCholeskyState {
    fn default() -> Self {
        Self {
            n: 16,
            last_err: 0.0,
            last_solve_us: 0.0,
            user_reset: true,
        }
    }
}

fn update_warp_tile_cholesky(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileCholesky {
        return;
    }
    let st = &mut warp.tile_cholesky;
    if !st.user_reset {
        return;
    }
    let n = st.n.max(2) as usize;
    // SPD test matrix: A = I + diag(1..N).
    let mut a32 = vec![0.0_f32; n * n];
    for i in 0..n {
        a32[i * n + i] = 1.0 + (i + 1) as f32;
        for j in (i + 1)..n {
            a32[i * n + j] = 1.0;
            a32[j * n + i] = 1.0;
        }
    }

    // Stage 37.8 — route through the molla `tile_block_cholesky`
    // primitive. Try the GPU kernel first; if anything goes wrong
    // (no adapter / N > MAX_N), fall back to the CPU oracle. Both
    // paths produce the lower-triangular `L` in place.
    let t0 = std::time::Instant::now();
    let mut l32 = a32.clone();
    let used_gpu = if n <= molla::compute::tile_block_cholesky::MAX_N {
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let kernel = molla::compute::tile_block_cholesky::TileBlockCholeskyKernel::new(device);
            kernel.dispatch_blocking(device, queue, &mut l32, n).is_ok()
        } else {
            false
        }
    } else {
        false
    };
    if !used_gpu {
        // CPU fallback (also exercised when the GPU is unavailable or
        // when N exceeds the single-workgroup limit).
        l32 = a32.clone();
        molla::compute::tile_block_cholesky::cholesky_cpu(&mut l32, n)
            .expect("tile_block_cholesky cpu");
    }
    st.last_solve_us = t0.elapsed().as_secs_f64() * 1.0e6;

    let err32 = molla::compute::tile_block_cholesky::frobenius_recon_error(&a32, &l32, n);
    st.last_err = err32 as f64;
    st.user_reset = false;
}

fn warp_tile_cholesky_pane(ui: &mut egui::Ui, st: &mut WarpTileCholeskyState) {
    ui.label("Warp tile Cholesky — port of tile/example_tile_cholesky.py");
    ui.separator();
    readout(ui, "N", &format!("{0}×{0}", st.n));
    readout(ui, "‖L Lᵀ − A‖_F", &format!("{:.3e}", st.last_err));
    readout(ui, "Factor time", &format!("{:.2} µs", st.last_solve_us));
    ui.separator();
    if ui
        .add(egui::Slider::new(&mut st.n, 4..=64).text("Size"))
        .changed()
    {
        st.user_reset = true;
    }
    if ui.button("Re-solve").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "Dense Cholesky on SPD `I + diag(1..N) + 1`, factored via \
         `molla::compute::tile_block_cholesky` (Stage 37.8) — single-\
         workgroup GPU kernel up to N=1024 with CPU fallback.",
    );
}

