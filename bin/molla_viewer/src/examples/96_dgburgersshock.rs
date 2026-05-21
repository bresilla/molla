// ═══ Scene 96: DgBurgersShock (Stage 43) ═══════════════════════════════
//
// 1D Burgers' equation `∂u/∂t + ∂(u²/2)/∂x = 0` on `[0, 1]` with smooth
// IC `u₀(x) = ½ + ½ sin(2π x)`, marched via a CPU finite-volume
// Lax-Friedrichs scheme plus a minmod slope limiter (an MUSCL/TVD
// stencil analogous to the BJ limiter the molla DG path uses on tri
// meshes). Shock forms at `t ≈ 1/π ≈ 0.318`. The pane shows the
// current step + a fading ghost trace of past snapshots.

const BURGERS_NX: usize = 200;
const BURGERS_PLANE_HALF: f32 = 0.50;

struct DgBurgersState {
    u: Vec<f64>,
    slope: Vec<f64>,
    snapshots: Vec<Vec<f64>>,
    t: f64,
    dt: f64,
    alpha_lf: f64,
    use_limiter: bool,
    paused: bool,
    user_reset: bool,
    steps: usize,
    last_max: f64,
    last_min: f64,
}

impl Default for DgBurgersState {
    fn default() -> Self {
        let mut me = Self {
            u: vec![0.0; BURGERS_NX],
            slope: vec![0.0; BURGERS_NX],
            snapshots: Vec::new(),
            t: 0.0,
            dt: 5.0e-4,
            alpha_lf: 1.2,
            use_limiter: true,
            paused: false,
            user_reset: false,
            steps: 0,
            last_max: 1.0,
            last_min: 0.0,
        };
        me.reset();
        me
    }
}

fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else if a.abs() < b.abs() {
        a
    } else {
        b
    }
}

impl DgBurgersState {
    fn reset(&mut self) {
        let n = self.u.len();
        let pi = std::f64::consts::PI;
        for i in 0..n {
            let x = (i as f64 + 0.5) / n as f64;
            self.u[i] = 0.5 + 0.5 * (2.0 * pi * x).sin();
        }
        self.slope.fill(0.0);
        self.snapshots.clear();
        self.t = 0.0;
        self.steps = 0;
    }

    fn step(&mut self) {
        let n = self.u.len();
        let dx = 1.0 / n as f64;
        if self.use_limiter {
            for i in 0..n {
                let im = (i + n - 1) % n;
                let ip = (i + 1) % n;
                let dl = (self.u[i] - self.u[im]) / dx;
                let dr = (self.u[ip] - self.u[i]) / dx;
                self.slope[i] = minmod(dl, dr);
            }
        } else {
            self.slope.fill(0.0);
        }
        let mut flux = vec![0.0_f64; n];
        for i in 0..n {
            let ip = (i + 1) % n;
            let u_l = self.u[i] + 0.5 * dx * self.slope[i];
            let u_r = self.u[ip] - 0.5 * dx * self.slope[ip];
            let fl = 0.5 * u_l * u_l;
            let fr = 0.5 * u_r * u_r;
            flux[i] = 0.5 * (fl + fr) - 0.5 * self.alpha_lf * (u_r - u_l);
        }
        let dt_over_dx = self.dt / dx;
        let mut new_u = self.u.clone();
        for i in 0..n {
            let im = (i + n - 1) % n;
            new_u[i] = self.u[i] - dt_over_dx * (flux[i] - flux[im]);
        }
        self.u = new_u;
        self.t += self.dt;
        self.steps += 1;
        self.last_max = self.u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        self.last_min = self.u.iter().cloned().fold(f64::INFINITY, f64::min);
    }
}

fn update_dg_burgers(sim: Res<Sim>, run: Res<SimRunState>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::DgBurgersShock {
        return;
    }
    let st = &mut warp.dg_burgers;
    if st.user_reset {
        st.reset();
        st.user_reset = false;
        return;
    }
    if run.paused || st.paused {
        return;
    }
    for _ in 0..10 {
        st.step();
    }
    if st.steps.is_multiple_of(50) {
        st.snapshots.push(st.u.clone());
        if st.snapshots.len() > 6 {
            st.snapshots.remove(0);
        }
    }
}

fn draw_dg_burgers_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::DgBurgersShock {
        return;
    }
    let st = &warp.dg_burgers;
    let n = st.u.len();
    let half = BURGERS_PLANE_HALF;
    let dx = (2.0 * half) / n as f32;
    let y_base = 0.0_f32;
    let y_scale = 0.6_f32;
    let axis = Color::srgb(0.30, 0.30, 0.35);
    gizmos.line(
        Vec3::new(-half, y_base, 0.0),
        Vec3::new(half, y_base, 0.0),
        axis,
    );
    gizmos.line(
        Vec3::new(-half, y_base, 0.0),
        Vec3::new(-half, y_base + y_scale, 0.0),
        axis,
    );
    for (k, snap) in st.snapshots.iter().enumerate() {
        let a = (k as f32 + 1.0) / (st.snapshots.len().max(1) as f32 + 1.0);
        let g = Color::srgba(0.45 + 0.45 * a, 0.45 + 0.45 * a, 0.50 + 0.40 * a, 0.5);
        for i in 0..n - 1 {
            let x0 = -half + (i as f32 + 0.5) * dx;
            let x1 = -half + (i as f32 + 1.5) * dx;
            let y0 = y_base + snap[i] as f32 * y_scale;
            let y1 = y_base + snap[i + 1] as f32 * y_scale;
            gizmos.line(Vec3::new(x0, y0, 0.0), Vec3::new(x1, y1, 0.0), g);
        }
    }
    let cur = Color::srgb(0.95, 0.30, 0.25);
    for i in 0..n - 1 {
        let x0 = -half + (i as f32 + 0.5) * dx;
        let x1 = -half + (i as f32 + 1.5) * dx;
        let y0 = y_base + st.u[i] as f32 * y_scale;
        let y1 = y_base + st.u[i + 1] as f32 * y_scale;
        gizmos.line(Vec3::new(x0, y0, 0.0), Vec3::new(x1, y1, 0.0), cur);
    }
}

fn dg_burgers_pane(ui: &mut egui::Ui, st: &mut DgBurgersState) {
    ui.label("Stage 43 — 1D Burgers + minmod limiter");
    ui.separator();
    readout(ui, "Cells", &format!("{}", st.u.len()));
    readout(ui, "t", &format!("{:.3} s", st.t));
    readout(ui, "Step", &format!("{}", st.steps));
    readout(
        ui,
        "u min / max",
        &format!("{:.3} / {:.3}", st.last_min, st.last_max),
    );
    readout(ui, "Shock at", "t ≈ 1/π ≈ 0.318");
    ui.separator();
    let _ = ui.add(
        egui::Slider::new(&mut st.dt, 1.0e-5..=2.0e-3)
            .text("dt (s)")
            .logarithmic(true),
    );
    let _ = ui.add(egui::Slider::new(&mut st.alpha_lf, 0.5..=3.0).text("α (LF speed)"));
    let _ = ui.checkbox(&mut st.use_limiter, "minmod slope limiter");
    let _ = ui.checkbox(&mut st.paused, "Pause");
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "Red = current solution; grey ghosts = past snapshots. Toggle \
         the limiter off to see Gibbs-type oscillations form near the \
         shock. The molla DG-P1 path uses a Barth-Jespersen scalar \
         limiter that's structurally equivalent on the cell-average \
         reconstruction stencil.",
    );
}

