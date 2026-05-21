// ═══ Scene 89: WarpFftPoisson2d — periodic 2-D Poisson on a torus ══════
//
// Port of Warp `core/example_fft_poisson_navier_stokes_2d.py`, but
// reduced to the Poisson core: deposit a localised Gaussian source
// every frame, invert `-∇² p = rhs` via the GPU tile-FFT, decay the
// field slightly so it doesn't saturate, render as a coloured point
// grid. This shows the FFT-Poisson + tile-FFT plumbing the same way
// the original example uses it for the vorticity-streamfunction
// projection.
//
// Lower complexity than the upstream example (no time-stepping NS, no
// turbulence IC — just a moving Gaussian source). Still exercises the
// same primitives: `molla::fem::solve_pressure_poisson_fft` +
// `molla::kernels::FftKernel`.

const FFT_POISSON_GRID: usize = 32;

struct WarpFftPoisson2dState {
    /// Current pressure field, row-major `n × n`.
    p: Vec<f32>,
    /// FFT kernel (lazy).
    kernel: Option<molla::kernels::FftKernel>,
    /// Simulation time (s).
    sim_t: f64,
    /// Decay factor applied to `p` each frame so steady-state forcing
    /// doesn't blow up. `1.0 = no decay`, `0.99 ≈ slow drift`.
    decay: f32,
    /// Source Gaussian half-width (in grid cells).
    source_sigma: f32,
    /// Source amplitude.
    source_amp: f32,
    /// Whether the source orbits or sits at the centre.
    orbiting_source: bool,
    /// Last solve time (µs).
    last_us: f64,
    /// "Reset to zero" flag.
    needs_reset: bool,
}

impl Default for WarpFftPoisson2dState {
    fn default() -> Self {
        Self {
            p: vec![0.0_f32; FFT_POISSON_GRID * FFT_POISSON_GRID],
            kernel: None,
            sim_t: 0.0,
            decay: 0.95,
            source_sigma: 1.8,
            source_amp: 5.0,
            orbiting_source: true,
            last_us: 0.0,
            needs_reset: true,
        }
    }
}

// Wrap-aware squared distance between grid cell `(i, j)` and a real-
// valued cell centre `(cx, cy)`. Periodic — the closest copy among the
// 3×3 wrap candidates wins, but in practice the source is far from the
// boundary so the direct distance is enough for the demo.
fn fft_poisson_gaussian(n: usize, cx: f32, cy: f32, sigma: f32, amp: f32) -> Vec<f32> {
    let mut rhs = vec![0.0_f32; n * n];
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    for j in 0..n {
        for i in 0..n {
            let dx = i as f32 - cx;
            let dy = j as f32 - cy;
            let r2 = dx * dx + dy * dy;
            rhs[j * n + i] = amp * (-r2 * inv_2s2).exp();
        }
    }
    rhs
}

fn update_warp_fft_poisson_2d(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFftPoisson2d {
        return;
    }
    let st = &mut warp.fft_poisson_2d;
    if st.needs_reset {
        st.p = vec![0.0_f32; FFT_POISSON_GRID * FFT_POISSON_GRID];
        st.sim_t = 0.0;
        st.needs_reset = false;
    }
    if run.paused {
        return;
    }

    let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) else {
        return;
    };
    if st.kernel.is_none() {
        st.kernel = Some(molla::kernels::FftKernel::new(device));
    }
    let kernel = st.kernel.as_ref().unwrap();

    // Build the source. If `orbiting_source`, the Gaussian orbits the
    // grid centre at radius n/5 with period 6 s.
    let n = FFT_POISSON_GRID;
    let centre = (n as f32 - 1.0) * 0.5;
    let (cx, cy) = if st.orbiting_source {
        let omega = std::f32::consts::TAU / 6.0;
        let r = n as f32 / 5.0;
        let theta = (st.sim_t as f32) * omega;
        (centre + r * theta.cos(), centre + r * theta.sin())
    } else {
        (centre, centre)
    };
    let rhs = fft_poisson_gaussian(n, cx, cy, st.source_sigma, st.source_amp);

    // Solve `-∇² p_new = rhs`.
    let t0 = std::time::Instant::now();
    match molla::fem::solve_pressure_poisson_fft(device, queue, kernel, &rhs, n) {
        Ok(p_new) => {
            // Mix new pressure into the field with the decay factor so
            // the source's effect accumulates briefly before fading.
            for (q, np) in st.p.iter_mut().zip(p_new.iter()) {
                *q = *q * st.decay + np * (1.0 - st.decay);
            }
        }
        Err(e) => warn!("fft_poisson_2d solve: {e}"),
    }
    st.last_us = t0.elapsed().as_secs_f64() * 1.0e6;
    st.sim_t += 1.0 / 60.0;
}

fn fft_poisson_colour(v: f32, scale: f32) -> Color {
    // Map signed pressure into a diverging blue→white→red ramp.
    let t = (v / scale).clamp(-1.0, 1.0);
    if t < 0.0 {
        let s = -t;
        Color::srgb(0.1 + 0.3 * (1.0 - s), 0.3 + 0.5 * (1.0 - s), 0.9 * s + 0.1)
    } else {
        Color::srgb(0.9 * t + 0.1, 0.3 + 0.5 * (1.0 - t), 0.1 + 0.3 * (1.0 - t))
    }
}

fn draw_warp_fft_poisson_2d_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFftPoisson2d {
        return;
    }
    let st = &warp.fft_poisson_2d;
    let n = FFT_POISSON_GRID;
    if st.p.len() != n * n {
        return;
    }
    let cell = 1.0_f32 / n as f32;
    // Auto-scale colour ramp by current max magnitude (so small initial
    // values are still visible).
    let max =
        st.p.iter()
            .fold(0.0_f32, |acc, v| acc.max(v.abs()))
            .max(1.0e-6);
    let half = 0.5_f32;
    for j in 0..n {
        for i in 0..n {
            let x = (i as f32 + 0.5) * cell - half;
            let z = (j as f32 + 0.5) * cell - half;
            let v = st.p[j * n + i];
            // Heightmap-ish: y = 0.05 + 0.3 * (v / max).
            let y = 0.05 + 0.30 * (v / max);
            let pos = Vec3::new(x, y, z);
            gizmos.sphere(pos, cell * 0.45, fft_poisson_colour(v, max));
        }
    }
    // Source marker (yellow).
    let centre = (n as f32 - 1.0) * 0.5;
    let (cx, cy) = if st.orbiting_source {
        let omega = std::f32::consts::TAU / 6.0;
        let r = n as f32 / 5.0;
        let theta = (st.sim_t as f32) * omega;
        (centre + r * theta.cos(), centre + r * theta.sin())
    } else {
        (centre, centre)
    };
    let sx = (cx + 0.5) * cell - half;
    let sz = (cy + 0.5) * cell - half;
    gizmos.sphere(Vec3::new(sx, 0.5, sz), 0.04, Color::srgb(1.0, 1.0, 0.2));
}

fn warp_fft_poisson_2d_pane(ui: &mut egui::Ui, st: &mut WarpFftPoisson2dState) {
    ui.label("Warp FFT Poisson 2-D — periodic `−∇² p = f` via tile-FFT");
    ui.separator();
    readout(ui, "Grid", &format!("{0}×{0}", FFT_POISSON_GRID));
    readout(ui, "Last solve", &format!("{:.1} µs", st.last_us));
    readout(ui, "t", &format!("{:.2} s", st.sim_t));
    let (pmin, pmax) =
        st.p.iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
                (a.min(v), b.max(v))
            });
    readout(ui, "p range", &format!("[{:+.3}, {:+.3}]", pmin, pmax));
    ui.separator();
    ui.label("Source");
    let _ = ui.checkbox(&mut st.orbiting_source, "Orbiting source");
    let _ = ui.add(egui::Slider::new(&mut st.source_sigma, 0.6..=4.0).text("σ (cells)"));
    let _ = ui.add(egui::Slider::new(&mut st.source_amp, -20.0..=20.0).text("amplitude"));
    ui.separator();
    ui.label("Field dynamics");
    let _ = ui.add(egui::Slider::new(&mut st.decay, 0.0..=1.0).text("decay (mix)"));
    if ui.button("Reset field").clicked() {
        st.needs_reset = true;
    }
    ui.small(
        "Each frame: deposit a Gaussian source, solve `−∇² p = source` via \
         GPU `solve_pressure_poisson_fft` (Stage 37.5), then EMA-mix into \
         the visible field. Coloured grid renders the pressure heightmap.",
    );
}

