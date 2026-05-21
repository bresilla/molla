// ── Scene 76: WarpFftFluid ────────────────────────────────────────
//
// Real fractional-step Navier-Stokes (Stage 37.5) on a periodic
// 16×16 grid with Taylor-Green initial condition. Pressure-Poisson
// solved by the GPU FFT kernel (Stage 24c). Velocity field is
// rendered as gizmo arrows.
//
// One frame = `FFT_FLUID_STEPS_PER_FRAME` projection steps. The FFT
// kernel is reused across frames — initialised lazily on first use.

const FFT_FLUID_GRID: usize = 16;
const FFT_FLUID_DT: f32 = 0.002;
const FFT_FLUID_STEPS_PER_FRAME: usize = 1;

struct WarpFftFluidState {
    sim_t: f64,
    /// Kinematic viscosity ν.
    viscosity: f32,
    /// Current velocity field — row-major `(j, i) → j·n + i`.
    u: Vec<[f32; 2]>,
    /// FFT kernel — built lazily on first GPU access.
    kernel: Option<molla::kernels::FftKernel>,
    /// Reset flag.
    needs_reset: bool,
}

impl Default for WarpFftFluidState {
    fn default() -> Self {
        Self {
            sim_t: 0.0,
            viscosity: 0.05,
            u: Vec::new(),
            kernel: None,
            needs_reset: true,
        }
    }
}

fn fft_fluid_seed_taylor_green(n: usize) -> Vec<[f32; 2]> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut u = vec![[0.0_f32; 2]; n * n];
    for j in 0..n {
        let y = (j as f32 + 0.5) / n as f32;
        for i in 0..n {
            let x = (i as f32 + 0.5) / n as f32;
            let ux = (two_pi * x).sin() * (two_pi * y).cos();
            let vy = -(two_pi * x).cos() * (two_pi * y).sin();
            u[j * n + i] = [ux, vy];
        }
    }
    u
}

fn update_warp_fft_fluid(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFftFluid {
        return;
    }
    let st = &mut warp.fft_fluid;
    if st.needs_reset {
        st.u = fft_fluid_seed_taylor_green(FFT_FLUID_GRID);
        st.sim_t = 0.0;
        st.needs_reset = false;
    }
    if run.paused {
        return;
    }

    // Acquire the GPU runtime; if unavailable the sim leaves the
    // state alone (the gizmos will still render the IC).
    let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) else {
        return;
    };
    if st.kernel.is_none() {
        st.kernel = Some(molla::kernels::FftKernel::new(device));
    }
    let kernel = st.kernel.as_ref().unwrap();

    let cfg = molla::fem::FractionalStepConfig {
        grid_n: FFT_FLUID_GRID,
        viscosity: st.viscosity,
        dt: FFT_FLUID_DT,
        num_steps: FFT_FLUID_STEPS_PER_FRAME,
    };
    let _ =
        molla::fem::advance_navier_stokes_fractional_step(device, queue, kernel, &mut st.u, cfg);
    st.sim_t += FFT_FLUID_DT as f64;
}

fn draw_warp_fft_fluid_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpFftFluid {
        return;
    }
    let st = &warp.fft_fluid;
    if st.u.is_empty() {
        return;
    }
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    let n = FFT_FLUID_GRID;
    for j in 0..n {
        let y = (j as f32 + 0.5) / n as f32;
        for i in 0..n {
            let x = (i as f32 + 0.5) / n as f32;
            let v = st.u[j * n + i];
            let from = origin + Vec3::new(x, 0.0, y);
            let to = from + Vec3::new(v[0] * 0.05, 0.0, v[1] * 0.05);
            gizmos.line(from, to, Color::srgb(0.5, 0.7, 0.95));
        }
    }
}

fn warp_fft_fluid_pane(ui: &mut egui::Ui, st: &mut WarpFftFluidState) {
    ui.label("Warp FFT fluid — Stage 37.5 fractional-step NS via GPU FFT");
    ui.separator();
    readout(ui, "t", &format!("{:.3} s", st.sim_t));
    ui.label("Viscosity ν");
    // Explicit forward-Euler diffusion is stable only while
    // ν·dt/dx² ≤ 0.25; with dx = 1/16 and dt = 0.002 that caps ν at
    // ~0.49, so the slider stops at 0.4 to keep a stability margin.
    let _ = ui.add(egui::Slider::new(&mut st.viscosity, 0.001..=0.4).logarithmic(true));
    if ui.button("Reset (Taylor-Green IC)").clicked() {
        st.needs_reset = true;
    }
    ui.small("Pressure-Poisson via molla::fem::solve_pressure_poisson_fft");
    ui.small("(Stage 24c GPU FFT). Predictor + corrector run on CPU.");
}

