// ═══ Scene 97: ShallowWaterDamBreak (Stage 43 v2) ══════════════════════
//
// 2D radial dam break for the shallow-water equations. Initial state:
// water column of height `h_in = 1.0 m` inside a circular dam of
// radius `R_d`; outside water at `h_out = 0.1 m`; `u = v = 0`. CPU
// Lax-Friedrichs finite-volume scheme on a 48×48 Cartesian grid.
// Water height rendered as a coloured height-field of small upright
// gizmo segments — blue (low) → cyan (mid) → white (high).

const SWE_GRID: usize = 48;
const SWE_PLANE_HALF: f32 = 0.50;
const SWE_G: f64 = 9.81;

struct SweDamBreakState {
    h: Vec<f64>,
    hu: Vec<f64>,
    hv: Vec<f64>,
    dt: f64,
    t: f64,
    steps: usize,
    h_in: f64,
    h_out: f64,
    dam_radius: f64,
    paused: bool,
    user_reset: bool,
    last_max_h: f64,
    last_min_h: f64,
    last_peak_speed: f64,
}

impl Default for SweDamBreakState {
    fn default() -> Self {
        let mut me = Self {
            h: vec![0.0; SWE_GRID * SWE_GRID],
            hu: vec![0.0; SWE_GRID * SWE_GRID],
            hv: vec![0.0; SWE_GRID * SWE_GRID],
            dt: 4.0e-4,
            t: 0.0,
            steps: 0,
            h_in: 1.0,
            h_out: 0.10,
            dam_radius: 0.20,
            paused: false,
            user_reset: false,
            last_max_h: 1.0,
            last_min_h: 0.10,
            last_peak_speed: 0.0,
        };
        me.reset();
        me
    }
}

impl SweDamBreakState {
    fn reset(&mut self) {
        let n = SWE_GRID;
        for j in 0..n {
            for i in 0..n {
                let xc = (i as f64 + 0.5) / n as f64 - 0.5;
                let yc = (j as f64 + 0.5) / n as f64 - 0.5;
                let r = (xc * xc + yc * yc).sqrt();
                let k = j * n + i;
                self.h[k] = if r < self.dam_radius {
                    self.h_in
                } else {
                    self.h_out
                };
                self.hu[k] = 0.0;
                self.hv[k] = 0.0;
            }
        }
        self.t = 0.0;
        self.steps = 0;
    }

    fn step(&mut self) {
        let n = SWE_GRID;
        let dx = 1.0 / n as f64;
        let g = SWE_G;
        let mut alpha: f64 = 0.0;
        for k in 0..n * n {
            let h = self.h[k].max(1.0e-6);
            let u = self.hu[k] / h;
            let v = self.hv[k] / h;
            let c = (g * h).sqrt();
            let s = (u.hypot(v)) + c;
            if s > alpha {
                alpha = s;
            }
        }
        self.last_peak_speed = alpha;
        let mut fx_h = vec![0.0_f64; n * n];
        let mut fx_hu = vec![0.0_f64; n * n];
        let mut fx_hv = vec![0.0_f64; n * n];
        let mut fy_h = vec![0.0_f64; n * n];
        let mut fy_hu = vec![0.0_f64; n * n];
        let mut fy_hv = vec![0.0_f64; n * n];
        for j in 0..n {
            for i in 0..n {
                let kl = j * n + i;
                let kr = j * n + ((i + 1) % n);
                let hl = self.h[kl].max(1.0e-6);
                let hr = self.h[kr].max(1.0e-6);
                let ul = self.hu[kl] / hl;
                let ur = self.hu[kr] / hr;
                let vl = self.hv[kl] / hl;
                let vr = self.hv[kr] / hr;
                let f_l = [
                    self.hu[kl],
                    self.hu[kl] * ul + 0.5 * g * hl * hl,
                    self.hu[kl] * vl,
                ];
                let f_r = [
                    self.hu[kr],
                    self.hu[kr] * ur + 0.5 * g * hr * hr,
                    self.hu[kr] * vr,
                ];
                fx_h[kl] = 0.5 * (f_l[0] + f_r[0]) - 0.5 * alpha * (self.h[kr] - self.h[kl]);
                fx_hu[kl] = 0.5 * (f_l[1] + f_r[1]) - 0.5 * alpha * (self.hu[kr] - self.hu[kl]);
                fx_hv[kl] = 0.5 * (f_l[2] + f_r[2]) - 0.5 * alpha * (self.hv[kr] - self.hv[kl]);
            }
        }
        for j in 0..n {
            for i in 0..n {
                let kl = j * n + i;
                let kr = ((j + 1) % n) * n + i;
                let hl = self.h[kl].max(1.0e-6);
                let hr = self.h[kr].max(1.0e-6);
                let ul = self.hu[kl] / hl;
                let ur = self.hu[kr] / hr;
                let vl = self.hv[kl] / hl;
                let vr = self.hv[kr] / hr;
                let g_l = [
                    self.hv[kl],
                    self.hv[kl] * ul,
                    self.hv[kl] * vl + 0.5 * g * hl * hl,
                ];
                let g_r = [
                    self.hv[kr],
                    self.hv[kr] * ur,
                    self.hv[kr] * vr + 0.5 * g * hr * hr,
                ];
                fy_h[kl] = 0.5 * (g_l[0] + g_r[0]) - 0.5 * alpha * (self.h[kr] - self.h[kl]);
                fy_hu[kl] = 0.5 * (g_l[1] + g_r[1]) - 0.5 * alpha * (self.hu[kr] - self.hu[kl]);
                fy_hv[kl] = 0.5 * (g_l[2] + g_r[2]) - 0.5 * alpha * (self.hv[kr] - self.hv[kl]);
            }
        }
        let lambda = self.dt / dx;
        for j in 0..n {
            for i in 0..n {
                let k = j * n + i;
                let kxm = j * n + ((i + n - 1) % n);
                let kym = ((j + n - 1) % n) * n + i;
                self.h[k] -= lambda * (fx_h[k] - fx_h[kxm]) + lambda * (fy_h[k] - fy_h[kym]);
                self.hu[k] -= lambda * (fx_hu[k] - fx_hu[kxm]) + lambda * (fy_hu[k] - fy_hu[kym]);
                self.hv[k] -= lambda * (fx_hv[k] - fx_hv[kxm]) + lambda * (fy_hv[k] - fy_hv[kym]);
                if self.h[k] < 1.0e-6 {
                    self.h[k] = 1.0e-6;
                    self.hu[k] = 0.0;
                    self.hv[k] = 0.0;
                }
            }
        }
        self.t += self.dt;
        self.steps += 1;
        self.last_max_h = self.h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        self.last_min_h = self.h.iter().cloned().fold(f64::INFINITY, f64::min);
    }
}

fn update_swe_dam_break(sim: Res<Sim>, run: Res<SimRunState>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::ShallowWaterDamBreak {
        return;
    }
    let st = &mut warp.swe_dam_break;
    if st.user_reset {
        st.reset();
        st.user_reset = false;
        return;
    }
    if run.paused || st.paused {
        return;
    }
    for _ in 0..8 {
        st.step();
    }
}

fn swe_height_color(h: f64, hmin: f64, hmax: f64) -> Color {
    let span = (hmax - hmin).max(1.0e-6);
    let t = ((h - hmin) / span).clamp(0.0, 1.0) as f32;
    if t < 0.5 {
        let tt = t * 2.0;
        Color::srgb(0.05, 0.10 + 0.85 * tt, 0.60 + 0.40 * tt)
    } else {
        let tt = (t - 0.5) * 2.0;
        Color::srgb(0.05 + 0.95 * tt, 0.95, 1.0)
    }
}

fn draw_swe_dam_break_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::ShallowWaterDamBreak {
        return;
    }
    let st = &warp.swe_dam_break;
    let n = SWE_GRID;
    let half = SWE_PLANE_HALF;
    let cell = (2.0 * half) / n as f32;
    let outline = Color::srgb(0.30, 0.30, 0.35);
    let corners = [
        Vec3::new(-half, 0.0, -half),
        Vec3::new(half, 0.0, -half),
        Vec3::new(half, 0.0, half),
        Vec3::new(-half, 0.0, half),
    ];
    for i in 0..4 {
        gizmos.line(corners[i], corners[(i + 1) % 4], outline);
    }
    let hmin = st.last_min_h;
    let hmax = st.last_max_h.max(hmin + 1.0e-3);
    for j in 0..n {
        for i in 0..n {
            let h = st.h[j * n + i];
            let x = -half + (i as f32 + 0.5) * cell;
            let z = -half + (j as f32 + 0.5) * cell;
            let height = (h as f32) * 0.30;
            let base = Vec3::new(x, 0.0, z);
            let top = base + Vec3::new(0.0, height, 0.0);
            let col = swe_height_color(h, hmin, hmax);
            gizmos.line(base, top, col);
        }
    }
}

fn swe_dam_break_pane(ui: &mut egui::Ui, st: &mut SweDamBreakState) {
    ui.label("Stage 43 v2 — radial dam break (LF flux)");
    ui.separator();
    readout(ui, "Grid", &format!("{0}×{0}", SWE_GRID));
    readout(ui, "t", &format!("{:.3} s", st.t));
    readout(ui, "Step", &format!("{}", st.steps));
    readout(
        ui,
        "h min / max",
        &format!("{:.3} / {:.3} m", st.last_min_h, st.last_max_h),
    );
    readout(ui, "Peak λ_max", &format!("{:.3} m/s", st.last_peak_speed));
    ui.separator();
    let _ = ui.add(egui::Slider::new(&mut st.h_in, 0.5..=2.0).text("h_in"));
    let _ = ui.add(egui::Slider::new(&mut st.h_out, 0.01..=0.5).text("h_out"));
    let _ = ui.add(egui::Slider::new(&mut st.dam_radius, 0.05..=0.40).text("dam radius"));
    let _ = ui.add(
        egui::Slider::new(&mut st.dt, 1.0e-5..=2.0e-3)
            .text("dt (s)")
            .logarithmic(true),
    );
    let _ = ui.checkbox(&mut st.paused, "Pause");
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "Height colour ramp: blue (low) → cyan (mid) → white (high). \
         LF flux; HLLC flux is the Stage 43 v3 follow-up. Periodic BC \
         on the 1×1 m domain — the wave will eventually wrap.",
    );
}

