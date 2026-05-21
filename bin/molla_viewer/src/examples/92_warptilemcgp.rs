// ═══ Scene 92: WarpTileMcgp (Stage 37.8) ═══════════════════════════════
//
// Monte Carlo Laplace solver on the unit square `[-1, 1]²` via the
// walk-on-spheres algorithm (Sawhney & Crane 2020, the same recipe
// driving Warp's `tile/example_tile_mcgp.py`). Each interior sample
// point launches `n_samples` random walks; at every step a uniformly
// random direction at distance equal to the closest-boundary radius
// is taken. The walk stops when the radius shrinks below `epsilon`,
// at which point the prescribed boundary value `g(x, y)` is returned.
// The average over walks is an unbiased estimator of `u(x, y)` where
// `Δu = 0` interior and `u = g` on the boundary.
//
// Wires `molla::compute::tile_mcgp::McgpConfig` directly (n_samples,
// epsilon, seed map cleanly onto walks-per-point / boundary-ε / RNG
// seed). The same module's `mcgp_estimate` is the SPSA gradient
// estimator — different problem, same MC tooling — and is mentioned
// in the pane for context.

const TILE_MCGP_GRID: usize = 16;

struct WarpTileMcgpState {
    /// Interior sample points (grid laid out in `[-0.95, 0.95]²`).
    /// Stored as `(x, y, u_hat)` after the last solve.
    points: Vec<(f32, f32, f32)>,
    cfg: molla::compute::tile_mcgp::McgpConfig,
    /// Cap on per-walk step count (textbook walk-on-spheres typically
    /// terminates in <50 steps for the unit square boundary).
    max_steps: u32,
    last_us: f64,
    needs_recompute: bool,
}

impl Default for WarpTileMcgpState {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            cfg: molla::compute::tile_mcgp::McgpConfig {
                n_samples: 64,
                epsilon: 0.01,
                seed: 0x5EED_F00D,
            },
            max_steps: 128,
            last_us: 0.0,
            needs_recompute: true,
        }
    }
}

// Prescribed boundary function `g(x, y)`. Used on the four edges
// `|x| = 1` or `|y| = 1`. A smooth, signed pattern makes the Laplace
// solution interesting to view.
fn mcgp_boundary_value(x: f32, y: f32) -> f32 {
    // 0.5 + 0.5 · sin(2π·x) · cos(2π·y). Range [0, 1].
    0.5 + 0.5 * (std::f32::consts::TAU * x).sin() * (std::f32::consts::TAU * y).cos()
}

// Signed distance to the unit-square boundary on the inside. Positive
// inside, zero on the boundary, undefined outside. For `(x, y) ∈
// [-1, 1]²` this is `min(1 − |x|, 1 − |y|)`.
fn mcgp_dist_to_boundary(x: f32, y: f32) -> f32 {
    (1.0 - x.abs()).min(1.0 - y.abs())
}

// Nearest-boundary point on `[-1, 1]²`. Used when the walk terminates
// within the boundary epsilon to evaluate `g`.
fn mcgp_nearest_boundary(x: f32, y: f32) -> (f32, f32) {
    let dx_left = x - -1.0;
    let dx_right = 1.0 - x;
    let dy_bot = y - -1.0;
    let dy_top = 1.0 - y;
    let mut best = (dx_left.abs(), -1.0_f32, y);
    let cand = (dx_right.abs(), 1.0_f32, y);
    if cand.0 < best.0 {
        best = cand;
    }
    let cand = (dy_bot.abs(), x, -1.0_f32);
    if cand.0 < best.0 {
        best = cand;
    }
    let cand = (dy_top.abs(), x, 1.0_f32);
    if cand.0 < best.0 {
        best = cand;
    }
    (best.1, best.2)
}

// Tiny xorshift64; same construction as inside `tile_mcgp`. The
// crate's RNG is private — we reproduce the canonical 13/7/17
// xorshift here so this scene stays viewer-only.
struct McgpXorShift(u64);
impl McgpXorShift {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform f32 in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// One walk-on-spheres random walk starting at `(x0, y0)`.
fn mcgp_walk(rng: &mut McgpXorShift, x0: f32, y0: f32, eps: f32, max_steps: u32) -> f32 {
    let mut x = x0;
    let mut y = y0;
    for _ in 0..max_steps {
        let r = mcgp_dist_to_boundary(x, y);
        if r < eps {
            let (bx, by) = mcgp_nearest_boundary(x, y);
            return mcgp_boundary_value(bx, by);
        }
        // Sample uniformly on the circle of radius r centred at (x, y).
        let theta = rng.next_f32() * std::f32::consts::TAU;
        x += r * theta.cos();
        y += r * theta.sin();
    }
    // Hit the step cap — fall back to the nearest boundary value.
    let (bx, by) = mcgp_nearest_boundary(x, y);
    mcgp_boundary_value(bx, by)
}

fn update_warp_tile_mcgp(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileMcgp {
        return;
    }
    let st = &mut warp.tile_mcgp;
    if !st.needs_recompute {
        return;
    }
    let n = TILE_MCGP_GRID;
    let mut pts = Vec::with_capacity(n * n);
    let t0 = std::time::Instant::now();
    let mut rng = McgpXorShift::new(st.cfg.seed);
    let walks = st.cfg.n_samples.max(1) as u32;
    let eps = st.cfg.epsilon as f32;
    for j in 0..n {
        for i in 0..n {
            // Interior grid laid out in [-0.95, 0.95]² so we never
            // start on the boundary.
            let x = -0.95 + 1.9 * (i as f32 + 0.5) / n as f32;
            let y = -0.95 + 1.9 * (j as f32 + 0.5) / n as f32;
            let mut acc = 0.0_f32;
            for _ in 0..walks {
                acc += mcgp_walk(&mut rng, x, y, eps, st.max_steps);
            }
            pts.push((x, y, acc / walks as f32));
        }
    }
    st.last_us = t0.elapsed().as_secs_f64() * 1.0e6;
    st.points = pts;
    st.needs_recompute = false;
}

fn tile_mcgp_color(v: f32) -> Color {
    let t = v.clamp(0.0, 1.0);
    if t < 0.5 {
        let s = t * 2.0;
        Color::srgb(0.05, 0.20 + 0.65 * s, 1.0 - 0.6 * s)
    } else {
        let s = (t - 0.5) * 2.0;
        Color::srgb(0.05 + 0.95 * s, 0.85 - 0.55 * s, 0.40 - 0.40 * s)
    }
}

fn draw_warp_tile_mcgp_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpTileMcgp {
        return;
    }
    let st = &warp.tile_mcgp;
    // Draw the unit-square boundary (z = 0) with the boundary
    // function `g` sampled along it.
    let edge_n = 96;
    for k in 0..edge_n {
        let t0 = -1.0 + 2.0 * k as f32 / edge_n as f32;
        let t1 = -1.0 + 2.0 * (k + 1) as f32 / edge_n as f32;
        // Top: y = 1
        gizmos.line(
            Vec3::new(t0, 0.0, 1.0),
            Vec3::new(t1, 0.0, 1.0),
            tile_mcgp_color(mcgp_boundary_value(t0, 1.0)),
        );
        // Bottom: y = -1
        gizmos.line(
            Vec3::new(t0, 0.0, -1.0),
            Vec3::new(t1, 0.0, -1.0),
            tile_mcgp_color(mcgp_boundary_value(t0, -1.0)),
        );
        // Right: x = 1
        gizmos.line(
            Vec3::new(1.0, 0.0, t0),
            Vec3::new(1.0, 0.0, t1),
            tile_mcgp_color(mcgp_boundary_value(1.0, t0)),
        );
        // Left: x = -1
        gizmos.line(
            Vec3::new(-1.0, 0.0, t0),
            Vec3::new(-1.0, 0.0, t1),
            tile_mcgp_color(mcgp_boundary_value(-1.0, t0)),
        );
    }
    // Sample points; height = u_hat so the surface bulges.
    for &(x, y, u) in &st.points {
        let pos = Vec3::new(x, 0.05 + 0.5 * u, y);
        gizmos.sphere(pos, 0.025, tile_mcgp_color(u));
    }
}

fn warp_tile_mcgp_pane(ui: &mut egui::Ui, st: &mut WarpTileMcgpState) {
    ui.label("Warp tile MCGP — Stage 37.8 Monte Carlo Laplace");
    ui.separator();
    readout(
        ui,
        "Points",
        &format!(
            "{0}×{0} = {1}",
            TILE_MCGP_GRID,
            TILE_MCGP_GRID * TILE_MCGP_GRID
        ),
    );
    readout(ui, "Last solve", &format!("{:.1} ms", st.last_us / 1000.0));
    ui.separator();
    ui.label("McgpConfig (from molla::compute::tile_mcgp):");
    let mut n_samples = st.cfg.n_samples as i32;
    if ui
        .add(egui::Slider::new(&mut n_samples, 4..=512).text("n_samples"))
        .changed()
    {
        st.cfg.n_samples = n_samples.max(1) as usize;
        st.needs_recompute = true;
    }
    let mut eps = st.cfg.epsilon as f32;
    if ui
        .add(
            egui::Slider::new(&mut eps, 0.001..=0.1)
                .text("epsilon")
                .logarithmic(true),
        )
        .changed()
    {
        st.cfg.epsilon = eps as f64;
        st.needs_recompute = true;
    }
    let mut max_steps = st.max_steps as i32;
    if ui
        .add(egui::Slider::new(&mut max_steps, 16..=512).text("max walk steps"))
        .changed()
    {
        st.max_steps = max_steps as u32;
        st.needs_recompute = true;
    }
    ui.separator();
    if ui.button("Re-seed").clicked() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xC0FFEE);
        st.cfg.seed = nanos ^ 0x5EED_F00D;
        st.needs_recompute = true;
    }
    if ui.button("Re-solve").clicked() {
        st.needs_recompute = true;
    }
    ui.small("Walk-on-spheres MC Laplace solver on `[-1, 1]²`.");
    ui.small("Boundary `g(x, y) = 0.5 + 0.5·sin(2π·x)·cos(2π·y)`.");
    ui.small("Sphere colour = û at that interior point; height = û.");
    ui.small(
        "Companion API: `mcgp_estimate` (SPSA gradient — separate problem, shared MC tooling).",
    );
}

