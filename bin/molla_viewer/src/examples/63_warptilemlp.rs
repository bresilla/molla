// ── Scene 63: WarpTileMlp ─────────────────────────────────────────
struct WarpTileMlpState {
    inputs: [f32; 16],
    /// Output of the (3-layer) MLP forward pass.
    last_output: Vec<f32>,
    last_forward_us: f64,
    user_reset: bool,
}
impl Default for WarpTileMlpState {
    fn default() -> Self {
        Self {
            inputs: [0.5; 16],
            last_output: Vec::new(),
            last_forward_us: 0.0,
            user_reset: true,
        }
    }
}

fn update_warp_tile_mlp(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileMlp {
        return;
    }
    let st = &mut warp.tile_mlp;
    if !st.user_reset {
        return;
    }
    // Tiny 16 → 8 → 4 ReLU MLP, weights deterministic from a hash.
    // Stage 37.8: forward pass routes through `tile_matmul::matmul_cpu`
    // (the molla compute primitive) instead of inline triple loops, so
    // any future swap to the GPU `TileMatmulKernel` only needs to
    // change the call site here. The CPU path stays the default —
    // a 1×16 · 16×8 GEMM is microseconds, well below GPU dispatch
    // overhead.
    let mut w1 = vec![0.0_f32; 16 * 8];
    let mut b1 = vec![0.0_f32; 8];
    let mut w2 = vec![0.0_f32; 8 * 4];
    let mut b2 = vec![0.0_f32; 4];
    let mut seed = 0x12345_u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((seed >> 16) as i32 as f32) / 32768.0 * 0.5
    };
    for v in w1.iter_mut() {
        *v = rng();
    }
    for v in b1.iter_mut() {
        *v = rng();
    }
    for v in w2.iter_mut() {
        *v = rng();
    }
    for v in b2.iter_mut() {
        *v = rng();
    }
    let t0 = std::time::Instant::now();
    // Layer 1: hidden_pre = input · W1, shape (1, 8) = (1, 16) · (16, 8).
    let mut hidden_pre = vec![0.0_f32; 8];
    molla::compute::tile_matmul::matmul_cpu(&st.inputs, &w1, &mut hidden_pre, 1, 8, 16)
        .expect("tile_matmul layer1");
    let mut hidden = vec![0.0_f32; 8];
    for j in 0..8 {
        hidden[j] = (hidden_pre[j] + b1[j]).max(0.0);
    }
    // Layer 2: out = hidden · W2, shape (1, 4) = (1, 8) · (8, 4).
    let mut out = vec![0.0_f32; 4];
    molla::compute::tile_matmul::matmul_cpu(&hidden, &w2, &mut out, 1, 4, 8)
        .expect("tile_matmul layer2");
    for j in 0..4 {
        out[j] += b2[j];
    }
    st.last_forward_us = t0.elapsed().as_secs_f64() * 1.0e6;
    st.last_output = out;
    st.user_reset = false;
}

fn warp_tile_mlp_pane(ui: &mut egui::Ui, st: &mut WarpTileMlpState) {
    ui.label("Warp tile MLP — port of tile/example_tile_mlp.py");
    ui.separator();
    readout(ui, "Forward time", &format!("{:.2} µs", st.last_forward_us));
    readout(ui, "Output", &format!("{:?}", st.last_output));
    ui.separator();
    ui.label("Input slider 0:");
    if ui
        .add(egui::Slider::new(&mut st.inputs[0], -1.0..=1.0))
        .changed()
    {
        st.user_reset = true;
    }
    ui.label("Input slider 8:");
    if ui
        .add(egui::Slider::new(&mut st.inputs[8], -1.0..=1.0))
        .changed()
    {
        st.user_reset = true;
    }
    if ui.button("Re-randomise weights").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "16 → 8 → 4 ReLU MLP, deterministic-hash weights. Forward pass \
         routed through `molla::compute::tile_matmul::matmul_cpu` \
         (Stage 37.8). The matching GPU `TileMatmulKernel` is available \
         for larger MLPs.",
    );
}

