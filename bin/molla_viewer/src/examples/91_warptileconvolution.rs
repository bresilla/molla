// ═══ Scene 91: WarpTileConvolution (Stage 37.8) ════════════════════════
//
// Standalone separable 2D convolution demo. Procedural 64×64 input
// image; user-editable 5-tap kernel exposed as five coefficient sliders
// (the molla `convolve_separable_cpu` / `TileConvolutionKernel` API
// applies the same 1-D kernel along x and along y, hence "symmetric
// separable" — there is no independent K_x / K_y dial; this is what the
// existing primitive supports). CPU/GPU toggle exercises both paths.

const TILE_CONV_DIM: usize = 64;

struct WarpTileConvolutionState {
    input: Vec<f32>,
    output: Vec<f32>,
    /// Five 1-D kernel coefficients [c0, c1, c2, c3, c4]. Applied
    /// along x and y in sequence by the molla separable primitive.
    coeffs: [f32; 5],
    use_gpu: bool,
    last_us: f64,
    needs_recompute: bool,
}

impl Default for WarpTileConvolutionState {
    fn default() -> Self {
        let n = TILE_CONV_DIM;
        let mut input = vec![0.0_f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let x = (i as f32 + 0.5) / n as f32;
                let y = (j as f32 + 0.5) / n as f32;
                // Low-frequency pattern + a bright bump in the centre +
                // a square stamp off-centre — gives box / gaussian /
                // laplacian visibly different outputs.
                let pat =
                    (5.0 * std::f32::consts::PI * x).sin() * (5.0 * std::f32::consts::PI * y).cos();
                let dx = x - 0.5;
                let dy = y - 0.5;
                let bump = (-40.0 * (dx * dx + dy * dy)).exp();
                let in_square = ((x - 0.25).abs() < 0.08 && (y - 0.75).abs() < 0.08) as i32 as f32;
                input[j * n + i] = 0.35 * pat + 0.55 * bump + 0.5 * in_square;
            }
        }
        Self {
            input,
            output: vec![0.0_f32; n * n],
            // Default to the binomial 5-tap gaussian.
            coeffs: [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0],
            use_gpu: false,
            last_us: 0.0,
            needs_recompute: true,
        }
    }
}

fn update_warp_tile_convolution(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileConvolution {
        return;
    }
    let st = &mut warp.tile_convolution;
    if !st.needs_recompute {
        return;
    }
    let n = TILE_CONV_DIM;
    let kernel = molla::compute::tile_convolution::Kernel5 { coeffs: st.coeffs };
    let mut output = vec![0.0_f32; n * n];
    let t0 = std::time::Instant::now();
    let used_gpu = if st.use_gpu {
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let kk = molla::compute::tile_convolution::TileConvolutionKernel::new(device);
            kk.dispatch_blocking(device, queue, &st.input, &mut output, n, n, &kernel)
                .is_ok()
        } else {
            false
        }
    } else {
        false
    };
    if !used_gpu {
        let _ = molla::compute::tile_convolution::convolve_separable_cpu(
            &st.input,
            &mut output,
            n,
            n,
            &kernel,
        );
    }
    st.last_us = t0.elapsed().as_secs_f64() * 1.0e6;
    st.output = output;
    st.needs_recompute = false;
}

fn tile_conv_color(v: f32) -> Color {
    let t = ((v + 1.0) * 0.5).clamp(0.0, 1.0);
    if t < 0.5 {
        let s = t * 2.0;
        Color::srgb(0.05, 0.20 + 0.65 * s, 1.0 - 0.6 * s)
    } else {
        let s = (t - 0.5) * 2.0;
        Color::srgb(0.05 + 0.95 * s, 0.85 - 0.55 * s, 0.40 - 0.40 * s)
    }
}

fn draw_warp_tile_convolution_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpTileConvolution {
        return;
    }
    let st = &warp.tile_convolution;
    let n = TILE_CONV_DIM;
    let cell = 1.0 / n as f32;
    let half = 0.5_f32;
    for j in 0..n {
        for i in 0..n {
            let x_local = (i as f32 + 0.5) * cell - 0.5;
            let z_local = (j as f32 + 0.5) * cell - 0.5;
            let in_val = st.input[j * n + i];
            let in_pos = Vec3::new(-half - 0.05 + x_local, 0.05 + 0.5 * in_val, z_local);
            gizmos.sphere(in_pos, cell * 0.4, tile_conv_color(in_val));
            let out_val = st.output[j * n + i];
            let out_pos = Vec3::new(half + 0.05 + x_local, 0.05 + 0.5 * out_val, z_local);
            gizmos.sphere(out_pos, cell * 0.4, tile_conv_color(out_val));
        }
    }
}

fn warp_tile_convolution_pane(ui: &mut egui::Ui, st: &mut WarpTileConvolutionState) {
    ui.label("Warp tile convolution — Stage 37.8 separable 5-tap convolution");
    ui.separator();
    readout(ui, "Image", &format!("{0}×{0}", TILE_CONV_DIM));
    readout(ui, "Last solve", &format!("{:.1} µs", st.last_us));
    let sum: f32 = st.coeffs.iter().sum();
    readout(ui, "Σ coeffs", &format!("{:+.3}", sum));
    ui.separator();
    ui.label("Kernel coefficients (applied along x then y):");
    let mut changed = false;
    for (idx, c) in st.coeffs.iter_mut().enumerate() {
        if ui
            .add(egui::Slider::new(c, -1.0..=1.0).text(format!("c[{}]", idx)))
            .changed()
        {
            changed = true;
        }
    }
    if changed {
        st.needs_recompute = true;
    }
    ui.horizontal(|ui| {
        if ui.button("Box").clicked() {
            st.coeffs = [0.2, 0.2, 0.2, 0.2, 0.2];
            st.needs_recompute = true;
        }
        if ui.button("Gauss").clicked() {
            st.coeffs = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
            st.needs_recompute = true;
        }
        if ui.button("Laplace").clicked() {
            st.coeffs = [0.0, 1.0, -2.0, 1.0, 0.0];
            st.needs_recompute = true;
        }
        if ui.button("Identity").clicked() {
            st.coeffs = [0.0, 0.0, 1.0, 0.0, 0.0];
            st.needs_recompute = true;
        }
    });
    ui.separator();
    let was_gpu = st.use_gpu;
    let _ = ui.checkbox(&mut st.use_gpu, "Use GPU");
    if was_gpu != st.use_gpu {
        st.needs_recompute = true;
    }
    let backend = if st.use_gpu {
        "GPU TileConvolutionKernel"
    } else {
        "CPU convolve_separable_cpu"
    };
    readout(ui, "Backend", backend);
    if ui.button("Re-apply").clicked() {
        st.needs_recompute = true;
    }
    ui.small("Left grid: input image. Right grid: filtered output.");
    ui.small("Stage 37.8 primitive: tile_convolution (separable, 5-tap, clamp-to-edge).");
    ui.small("Note: the molla API applies one 1-D kernel along both axes (symmetric separable).");
}

