// ═══ Scene 81: WarpTileFilter (Stage 37.8) ═════════════════════════════
//
// Procedural 32×32 input image, separable 5-tap filter (box / gaussian
// / laplacian / identity) via `tile_convolution` + `tile_filtering`.
// Renders input on the left, filtered output on the right, both as
// colored point grids. CPU/GPU toggle exercises both paths.

const TILE_FILTER_DIM: usize = 32;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TileFilterKind {
    Box,
    Gaussian,
    Laplacian,
    Identity,
}

struct WarpTileFilterState {
    input: Vec<f32>,
    output: Vec<f32>,
    kind: TileFilterKind,
    use_gpu: bool,
    last_us: f64,
    needs_recompute: bool,
}

impl Default for WarpTileFilterState {
    fn default() -> Self {
        let n = TILE_FILTER_DIM;
        // Procedural input: low-frequency sin·cos pattern + a circular
        // bump that gives the Laplacian/box visible high-freq content.
        let mut input = vec![0.0_f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let x = (i as f32 + 0.5) / n as f32;
                let y = (j as f32 + 0.5) / n as f32;
                let pattern =
                    (4.0 * std::f32::consts::PI * x).sin() * (4.0 * std::f32::consts::PI * y).cos();
                let dx = x - 0.5;
                let dy = y - 0.5;
                let bump = (-30.0 * (dx * dx + dy * dy)).exp();
                input[j * n + i] = 0.4 * pattern + 0.6 * bump;
            }
        }
        Self {
            input,
            output: vec![0.0_f32; n * n],
            kind: TileFilterKind::Gaussian,
            use_gpu: false,
            last_us: 0.0,
            needs_recompute: true,
        }
    }
}

fn update_warp_tile_filter(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileFilter {
        return;
    }
    let st = &mut warp.tile_filter;
    if !st.needs_recompute {
        return;
    }
    let n = TILE_FILTER_DIM;
    let t0 = std::time::Instant::now();
    let mut output = vec![0.0_f32; n * n];

    let used_gpu = if st.use_gpu {
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let kernel = molla::compute::tile_convolution::TileConvolutionKernel::new(device);
            let result = match st.kind {
                TileFilterKind::Box => molla::compute::tile_filtering::box_filter_gpu(
                    device,
                    queue,
                    &kernel,
                    &st.input,
                    &mut output,
                    n,
                    n,
                ),
                TileFilterKind::Gaussian => molla::compute::tile_filtering::gaussian_filter_gpu(
                    device,
                    queue,
                    &kernel,
                    &st.input,
                    &mut output,
                    n,
                    n,
                ),
                TileFilterKind::Laplacian => molla::compute::tile_filtering::laplacian_filter_gpu(
                    device,
                    queue,
                    &kernel,
                    &st.input,
                    &mut output,
                    n,
                    n,
                ),
                TileFilterKind::Identity => molla::compute::tile_filtering::identity_filter_gpu(
                    device,
                    queue,
                    &kernel,
                    &st.input,
                    &mut output,
                    n,
                    n,
                ),
            };
            result.is_ok()
        } else {
            false
        }
    } else {
        false
    };
    if !used_gpu {
        let _ = match st.kind {
            TileFilterKind::Box => {
                molla::compute::tile_filtering::box_filter_cpu(&st.input, &mut output, n, n)
            }
            TileFilterKind::Gaussian => {
                molla::compute::tile_filtering::gaussian_filter_cpu(&st.input, &mut output, n, n)
            }
            TileFilterKind::Laplacian => {
                molla::compute::tile_filtering::laplacian_filter_cpu(&st.input, &mut output, n, n)
            }
            TileFilterKind::Identity => {
                molla::compute::tile_filtering::identity_filter_cpu(&st.input, &mut output, n, n)
            }
        };
    }
    st.last_us = t0.elapsed().as_secs_f64() * 1.0e6;
    st.output = output;
    st.needs_recompute = false;
}

// Map a value `v` (typically in [-1, 1]) to a Bevy color ramp.
fn tile_filter_color(v: f32) -> Color {
    let t = ((v + 1.0) * 0.5).clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0;
        (0.05, 0.20 + 0.65 * s, 1.0 - 0.6 * s)
    } else {
        let s = (t - 0.5) * 2.0;
        (0.05 + 0.95 * s, 0.85 - 0.55 * s, 0.40 - 0.40 * s)
    };
    Color::srgb(r, g, b)
}

fn draw_warp_tile_filter_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpTileFilter {
        return;
    }
    let st = &warp.tile_filter;
    let n = TILE_FILTER_DIM;
    let cell = 1.0 / n as f32;
    // Two grids: input on the left at x ∈ [-1.1, -0.1], output on the
    // right at x ∈ [0.1, 1.1]. Y up = z in world space (camera looks
    // down +Y).
    let half = 0.5;
    for j in 0..n {
        for i in 0..n {
            let x_local = (i as f32 + 0.5) * cell - 0.5;
            let z_local = (j as f32 + 0.5) * cell - 0.5;
            // Input
            let in_val = st.input[j * n + i];
            let in_pos = Vec3::new(-half - 0.05 + x_local, 0.05 + 0.5 * in_val, z_local);
            gizmos.sphere(in_pos, cell * 0.4, tile_filter_color(in_val));
            // Output
            let out_val = st.output[j * n + i];
            let out_pos = Vec3::new(half + 0.05 + x_local, 0.05 + 0.5 * out_val, z_local);
            gizmos.sphere(out_pos, cell * 0.4, tile_filter_color(out_val));
        }
    }
}

fn warp_tile_filter_pane(ui: &mut egui::Ui, st: &mut WarpTileFilterState) {
    ui.label("Warp tile filter — Stage 37.8 separable 5-tap convolution");
    ui.separator();
    readout(ui, "Image", &format!("{0}×{0}", TILE_FILTER_DIM));
    readout(ui, "Last solve", &format!("{:.1} µs", st.last_us));
    ui.separator();
    ui.label("Filter");
    let prev_kind = st.kind;
    egui::ComboBox::from_id_salt("tile_filter_kind")
        .selected_text(match st.kind {
            TileFilterKind::Box => "Box",
            TileFilterKind::Gaussian => "Gaussian (binomial 1-4-6-4-1)",
            TileFilterKind::Laplacian => "Laplacian (1-D edge stencil)",
            TileFilterKind::Identity => "Identity (passthrough)",
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut st.kind, TileFilterKind::Box, "Box");
            ui.selectable_value(&mut st.kind, TileFilterKind::Gaussian, "Gaussian");
            ui.selectable_value(&mut st.kind, TileFilterKind::Laplacian, "Laplacian");
            ui.selectable_value(&mut st.kind, TileFilterKind::Identity, "Identity");
        });
    if prev_kind != st.kind {
        st.needs_recompute = true;
    }
    ui.separator();
    let was_gpu = st.use_gpu;
    let _ = ui.checkbox(&mut st.use_gpu, "Use GPU");
    if was_gpu != st.use_gpu {
        st.needs_recompute = true;
    }
    let backend = if st.use_gpu {
        "GPU TileConvolutionKernel"
    } else {
        "CPU convolve_separable"
    };
    readout(ui, "Backend", backend);
    if ui.button("Re-apply").clicked() {
        st.needs_recompute = true;
    }
    ui.small("Left grid: input image. Right grid: filtered output.");
    ui.small("Stage 37.8 primitives: tile_convolution + tile_filtering.");
}

