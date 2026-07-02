// ── Scene 62: WarpTileFft ─────────────────────────────────────────
struct WarpTileFftState {
    n: u32,
    /// Real-valued sinusoidal input.
    input: Vec<f64>,
    /// FFT magnitude (length n / 2).
    output_mag: Vec<f64>,
    user_reset: bool,
}
impl Default for WarpTileFftState {
    fn default() -> Self {
        Self {
            n: 256,
            input: Vec::new(),
            output_mag: Vec::new(),
            user_reset: true,
        }
    }
}

fn update_warp_tile_fft(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileFft {
        return;
    }
    let st = &mut warp.tile_fft;
    if !st.user_reset && st.input.len() == st.n as usize {
        return;
    }
    let n = st.n.next_power_of_two().max(8) as usize;
    st.input = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * 4.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 11.0 * t).sin()
        })
        .collect();
    // Naive DFT (since we don't want a GPU dispatch from a host system; this
    // is the host-side FFT readout side of the pane — magnitude is what
    // we display).
    let mut mags = vec![0.0_f64; n / 2];
    for k in 0..(n / 2) {
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, &x) in st.input.iter().enumerate() {
            let theta = 2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64;
            re += x * theta.cos();
            im -= x * theta.sin();
        }
        mags[k] = (re * re + im * im).sqrt() / (n as f64 * 0.5);
    }
    st.output_mag = mags;
    st.user_reset = false;
}

fn warp_tile_fft_pane(ui: &mut egui::Ui, st: &mut WarpTileFftState) {
    ui.label("Warp tile FFT — port of tile/example_tile_fft.py");
    ui.separator();
    readout(ui, "N", &format!("{}", st.n));
    readout(ui, "Output bins", &format!("{}", st.output_mag.len()));
    ui.label("");
    if !st.output_mag.is_empty() {
        let max = st.output_mag.iter().cloned().fold(1e-12_f64, f64::max);
        let mut buf = String::new();
        for (k, &m) in st.output_mag.iter().enumerate().take(32) {
            let bars = (m / max * 16.0) as usize;
            let bar: String = std::iter::repeat_n('▌', bars).collect();
            buf.push_str(&format!("{:>3}: {}\n", k, bar));
        }
        ui.add(egui::Label::new(
            egui::RichText::new(buf).monospace().small(),
        ));
    }
    if ui
        .add(egui::Slider::new(&mut st.n, 32..=1024).text("Length"))
        .changed()
    {
        st.user_reset = true;
    }
    ui.small(
        "Sum of two sinusoids → magnitude spectrum (host DFT). Stage 37.8 \
         did not add a tile-FFT primitive (Stage 24c GPU FFT lives in \
         `molla::kernels::FftKernel` instead); the host DFT stays as the \
         live readout for this pane.",
    );
}

