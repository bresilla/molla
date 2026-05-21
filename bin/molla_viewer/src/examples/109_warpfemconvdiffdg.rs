// ── Scene 109: WarpFemConvDiffDg (reduced DG-P0) ─────────────────
//
// Separate from the continuous SUPG scene above: this uses a cell-local
// DG-P0/FV update with upwind advection, explicit diffusion, and an optional
// positivity limiter. It is a reduced procedural equivalent of Warp's DG
// convection-diffusion example until a full scalar DG assembly/GPU kernel
// lands.
const DG_CONVDIFF_NX: usize = 40;
const DG_CONVDIFF_NY: usize = 20;

struct WarpFemConvDiffDgState {
    u: Vec<f64>,
    alpha: f64,
    bx: f64,
    dt: f64,
    steps: usize,
    use_limiter: bool,
    paused: bool,
    user_reset: bool,
    last_min: f64,
    last_max: f64,
}

impl Default for WarpFemConvDiffDgState {
    fn default() -> Self {
        let mut me = Self {
            u: vec![0.0; DG_CONVDIFF_NX * DG_CONVDIFF_NY],
            alpha: 0.01,
            bx: 1.0,
            dt: 5.0e-4,
            steps: 0,
            use_limiter: true,
            paused: false,
            user_reset: false,
            last_min: 0.0,
            last_max: 1.0,
        };
        me.reset();
        me
    }
}

fn dg_convdiff_idx(i: usize, j: usize, nx: usize) -> usize {
    j * nx + i
}

fn dg_convdiff_reset(u: &mut [f64], nx: usize, ny: usize) {
    assert_eq!(u.len(), nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let x = (i as f64 + 0.5) / nx as f64;
            let y = (j as f64 + 0.5) / ny as f64;
            let plume = (-(x - 0.18).powi(2) / 0.01 - (y - 0.50).powi(2) / 0.05).exp();
            u[dg_convdiff_idx(i, j, nx)] = plume.clamp(0.0, 1.0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dg_convdiff_p0_step(
    u: &[f64],
    nx: usize,
    ny: usize,
    alpha: f64,
    bx: f64,
    dt: f64,
    use_limiter: bool,
) -> Vec<f64> {
    assert_eq!(u.len(), nx * ny);
    let dx = 1.0 / nx as f64;
    let dy = 1.0 / ny as f64;
    let mut next = u.to_vec();
    for j in 0..ny {
        for i in 0..nx {
            let id = dg_convdiff_idx(i, j, nx);
            let c = u[id];
            let l = if i == 0 {
                1.0
            } else {
                u[dg_convdiff_idx(i - 1, j, nx)]
            };
            let r = if i + 1 == nx {
                0.0
            } else {
                u[dg_convdiff_idx(i + 1, j, nx)]
            };
            let d = if j == 0 {
                c
            } else {
                u[dg_convdiff_idx(i, j - 1, nx)]
            };
            let t = if j + 1 == ny {
                c
            } else {
                u[dg_convdiff_idx(i, j + 1, nx)]
            };

            let adv = if bx >= 0.0 {
                -bx * (c - l) / dx
            } else {
                -bx * (r - c) / dx
            };
            let diff = alpha * ((l - 2.0 * c + r) / (dx * dx) + (d - 2.0 * c + t) / (dy * dy));
            let mut v = c + dt * (adv + diff);
            if use_limiter {
                // Positivity/maximum-principle limiter for the advection
                // dominated reduced case. The viewer reports min/max so the
                // effect is visible when toggled.
                v = v.clamp(0.0, 1.0);
            }
            next[id] = v;
        }
    }
    next
}

impl WarpFemConvDiffDgState {
    fn reset(&mut self) {
        dg_convdiff_reset(&mut self.u, DG_CONVDIFF_NX, DG_CONVDIFF_NY);
        self.steps = 0;
        self.last_min = self.u.iter().copied().fold(f64::INFINITY, f64::min);
        self.last_max = self.u.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }

    fn step(&mut self) {
        self.u = dg_convdiff_p0_step(
            &self.u,
            DG_CONVDIFF_NX,
            DG_CONVDIFF_NY,
            self.alpha,
            self.bx,
            self.dt,
            self.use_limiter,
        );
        self.steps += 1;
        self.last_min = self.u.iter().copied().fold(f64::INFINITY, f64::min);
        self.last_max = self.u.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }
}

fn update_warp_fem_convdiff_dg(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFemConvDiffDg {
        return;
    }
    let st = &mut warp.fem_convdiff_dg;
    if st.user_reset {
        st.reset();
        st.user_reset = false;
    }
    if run.paused || st.paused {
        return;
    }
    for _ in 0..8 {
        st.step();
    }
}

fn draw_warp_fem_convdiff_dg_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemConvDiffDg {
        return;
    }
    let st = &warp.fem_convdiff_dg;
    let x_half = 0.75_f32;
    let z_half = 0.38_f32;
    let dx = (2.0 * x_half) / DG_CONVDIFF_NX as f32;
    let dz = (2.0 * z_half) / DG_CONVDIFF_NY as f32;
    for j in 0..DG_CONVDIFF_NY {
        for i in 0..DG_CONVDIFF_NX {
            let v = st.u[dg_convdiff_idx(i, j, DG_CONVDIFF_NX)].clamp(0.0, 1.0);
            let col = fem_heat_colour(v);
            let x0 = -x_half + i as f32 * dx;
            let x1 = x0 + dx;
            let z0 = -z_half + j as f32 * dz;
            let z1 = z0 + dz;
            let y = 0.04 + 0.25 * v as f32;
            let p00 = Vec3::new(x0, y, z0);
            let p10 = Vec3::new(x1, y, z0);
            let p11 = Vec3::new(x1, y, z1);
            let p01 = Vec3::new(x0, y, z1);
            gizmos.line(p00, p10, col);
            gizmos.line(p10, p11, col);
            gizmos.line(p11, p01, col);
            gizmos.line(p01, p00, col);
        }
    }
}

fn warp_fem_convdiff_dg_pane(ui: &mut egui::Ui, st: &mut WarpFemConvDiffDgState) {
    ui.label("Warp FEM convection-diffusion DG — reduced DG-P0");
    ui.separator();
    readout(
        ui,
        "Cells",
        &format!("{}×{}", DG_CONVDIFF_NX, DG_CONVDIFF_NY),
    );
    readout(ui, "Step", &format!("{}", st.steps));
    readout(
        ui,
        "u min / max",
        &format!("{:.3} / {:.3}", st.last_min, st.last_max),
    );
    let mut changed = false;
    changed |= ui
        .add(
            egui::Slider::new(&mut st.alpha, 0.0005..=0.05)
                .logarithmic(true)
                .text("α"),
        )
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut st.bx, 0.1..=2.5).text("advection +x"))
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut st.dt, 1.0e-5..=2.0e-3)
                .logarithmic(true)
                .text("dt"),
        )
        .changed();
    let _ = ui.checkbox(&mut st.use_limiter, "maximum-principle limiter");
    let _ = ui.checkbox(&mut st.paused, "Pause");
    if changed || ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small("DG-P0/FV update with upwind flux + explicit diffusion; limiter clamps to [0, 1].");
    ui.small("GPU status: reusable DG/SWE flux+limiter kernels exist; scalar conv-diff DG GPU is future work.");
}

