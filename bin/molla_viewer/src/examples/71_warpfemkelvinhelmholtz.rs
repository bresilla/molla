// ── Scene 71: WarpFemKelvinHelmholtz ──────────────────────────────
//
// Real FEM Navier-Stokes time-stepping (Stage 37.3) on a 12×12 P2-P1
// Taylor-Hood mesh of the unit square. Initial condition is a shear
// layer (`u_x = ±0.5` above / below the midline) plus a small
// vertical perturbation that excites the Kelvin-Helmholtz instability.
// Boundary conditions: pin the side walls to the bulk shear value,
// pressure pin at vertex 0.
//
// Per-frame budget: a handful of `solve_navier_stokes_step` calls. The
// 12×12 mesh keeps each step well under 50 ms in release mode. The
// pane lets the user dial viscosity, perturbation amplitude, and
// readouts include simulated time + the L∞ vorticity diagnostic.

const KH_GRID: usize = 12;
const KH_DT: f64 = 0.01;
// Steps per frame — keep small so the viewer stays interactive.
const KH_STEPS_PER_FRAME: usize = 1;

struct WarpFemKhState {
    sim_t: f64,
    /// Kinematic viscosity ν.
    viscosity: f64,
    /// Vertical-perturbation amplitude (initial condition).
    perturb_amp: f64,
    /// Last-frame max |ω| diagnostic.
    vorticity_max: f64,
    /// FEM mesh (built lazily).
    mesh: Option<fem::FemMesh>,
    th: Option<fem::MixedTaylorHood>,
    /// Cached `M / dt + A`, divergence, and mass matrices.
    a_eff: Option<molla::math::sparse::BsrMatrix>,
    b_div: Option<molla::math::sparse::BsrMatrix>,
    m: Option<molla::math::sparse::BsrMatrix>,
    /// Static BCs (side walls) — rebuilt only on reset.
    bcs: Vec<fem::StokesBc>,
    /// Velocity DoFs (interleaved P2-vector).
    u: Vec<f64>,
    /// Reset trigger.
    needs_reset: bool,
}

impl Default for WarpFemKhState {
    fn default() -> Self {
        Self {
            sim_t: 0.0,
            viscosity: 0.005,
            perturb_amp: 0.05,
            vorticity_max: 0.0,
            mesh: None,
            th: None,
            a_eff: None,
            b_div: None,
            m: None,
            bcs: Vec::new(),
            u: Vec::new(),
            needs_reset: true,
        }
    }
}

// Build the static BC list. All four walls are pinned to the imposed
// shear profile `u = ±0.5` (sign chosen by `y > 0.5`): the left/right
// walls set the inflow/outflow shear and the top/bottom walls hold a
// matching no-slip velocity that keeps the shear layer in the
// interior. P2 mid-edge nodes on the boundary get the same imposed
// velocity; vertex 0 anchors the pressure null-space.
fn kh_bcs(mesh: &fem::FemMesh, th: &fem::MixedTaylorHood) -> Vec<fem::StokesBc> {
    let mut bcs: Vec<fem::StokesBc> = Vec::new();
    let target_u = |y: f64| if y > 0.5 { 0.5_f64 } else { -0.5_f64 };

    for (i, p) in mesh.vertices.iter().enumerate() {
        let on_left = p.x.abs() < 1e-9;
        let on_right = (p.x - 1.0).abs() < 1e-9;
        let on_bot = p.y.abs() < 1e-9;
        let on_top = (p.y - 1.0).abs() < 1e-9;
        if on_left || on_right {
            bcs.push(fem::StokesBc::LidVelocity {
                node: i as u32,
                velocity: vec![target_u(p.y), 0.0],
            });
        } else if on_top || on_bot {
            // No-slip top/bottom keeps the shear layer in the interior.
            bcs.push(fem::StokesBc::LidVelocity {
                node: i as u32,
                velocity: vec![target_u(p.y), 0.0],
            });
        }
    }
    let (edges, _) = fem::compute_edges(mesh);
    let edge_offset = th.map.vertex_count;
    for (k, edge) in edges.iter().enumerate() {
        let pa = mesh.vertices[edge[0] as usize];
        let pb = mesh.vertices[edge[1] as usize];
        let mx = 0.5 * (pa.x + pb.x);
        let my = 0.5 * (pa.y + pb.y);
        let on_b = mx.abs() < 1e-9
            || (mx - 1.0).abs() < 1e-9
            || my.abs() < 1e-9
            || (my - 1.0).abs() < 1e-9;
        if !on_b {
            continue;
        }
        bcs.push(fem::StokesBc::LidVelocity {
            node: (edge_offset + k) as u32,
            velocity: vec![target_u(my), 0.0],
        });
    }
    bcs.push(fem::StokesBc::PressureZero { vertex: 0 });
    bcs
}

fn kh_initial_velocity(
    mesh: &fem::FemMesh,
    th: &fem::MixedTaylorHood,
    perturb_amp: f64,
) -> Vec<f64> {
    let dim = th.dim;
    let mut u = vec![0.0_f64; th.n_velocity];
    let pi = std::f64::consts::PI;
    let seed = |x: f64, y: f64| -> (f64, f64) {
        let dy = y - 0.5;
        let ux = if dy > 0.0 { 0.5 } else { -0.5 };
        // Vertical perturbation localised near the shear layer.
        let vy = perturb_amp * (4.0 * pi * x).sin() * (-dy.abs() * 8.0).exp();
        (ux, vy)
    };
    for (i, p) in mesh.vertices.iter().enumerate() {
        let (ux, vy) = seed(p.x, p.y);
        u[i * dim] = ux;
        u[i * dim + 1] = vy;
    }
    let (edges, _) = fem::compute_edges(mesh);
    let edge_offset = th.map.vertex_count;
    for (k, edge) in edges.iter().enumerate() {
        let pa = mesh.vertices[edge[0] as usize];
        let pb = mesh.vertices[edge[1] as usize];
        let mx = 0.5 * (pa.x + pb.x);
        let my = 0.5 * (pa.y + pb.y);
        let (ux, vy) = seed(mx, my);
        let n = edge_offset + k;
        u[n * dim] = ux;
        u[n * dim + 1] = vy;
    }
    u
}

fn kh_assemble(st: &mut WarpFemKhState) {
    let mesh = fem::FemMesh::tri_grid_2d(KH_GRID, KH_GRID, 1.0, 1.0);
    let th = fem::MixedTaylorHood::build(&mesh, 2);
    let a = fem::assemble_velocity_block(&mesh, &th, st.viscosity).unwrap();
    let m = fem::assemble_velocity_mass(&mesh, &th).unwrap();
    let b_div = fem::assemble_div_block(&mesh, &th).unwrap();
    let inv_dt = 1.0 / KH_DT;
    // a_eff = M/dt + A
    let mut triplets: Vec<(usize, usize, Vec<f64>)> = Vec::new();
    for r in 0..m.block_rows {
        for slot in m.row_offsets[r]..m.row_offsets[r + 1] {
            let c = m.col_idx[slot];
            triplets.push((r, c, vec![inv_dt * m.values[slot]]));
        }
    }
    for r in 0..a.block_rows {
        for slot in a.row_offsets[r]..a.row_offsets[r + 1] {
            let c = a.col_idx[slot];
            triplets.push((r, c, vec![a.values[slot]]));
        }
    }
    let a_eff =
        molla::math::sparse::BsrMatrix::from_triplets(1, a.block_rows, a.block_cols, &triplets)
            .unwrap();

    st.bcs = kh_bcs(&mesh, &th);
    st.u = kh_initial_velocity(&mesh, &th, st.perturb_amp);
    st.mesh = Some(mesh);
    st.th = Some(th);
    st.a_eff = Some(a_eff);
    st.b_div = Some(b_div);
    st.m = Some(m);
    st.sim_t = 0.0;
    st.vorticity_max = 0.0;
    st.needs_reset = false;
}

fn update_warp_fem_kh(sim: Res<Sim>, run: Res<SimRunState>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpFemKelvinHelmholtz {
        return;
    }
    let st = &mut warp.fem_kh;
    if st.needs_reset {
        kh_assemble(st);
    }
    if run.paused {
        return;
    }

    let (Some(mesh), Some(th), Some(a_eff), Some(b_div), Some(m)) = (
        st.mesh.as_ref(),
        st.th.as_ref(),
        st.a_eff.as_ref(),
        st.b_div.as_ref(),
        st.m.as_ref(),
    ) else {
        return;
    };

    let inv_dt = 1.0 / KH_DT;
    let zero_f = vec![0.0_f64; th.n_velocity];
    let simple = fem::StokesSimpleConfig {
        max_outer_iters: 40,
        tol_outer: 1e-5,
        max_inner_iters: 200,
        tol_inner: 1e-7,
        ..Default::default()
    };

    for _ in 0..KH_STEPS_PER_FRAME {
        let res = fem::solve_navier_stokes_step(
            mesh, th, a_eff, b_div, m, &st.u, &zero_f, inv_dt, &st.bcs, &simple,
        );
        if let Ok(res) = res {
            st.u = res.u;
            st.sim_t += KH_DT;
        } else {
            break;
        }
    }

    // Vorticity diagnostic ω_z = ∂v/∂x - ∂u/∂y, sampled on vertex grid
    // via finite differences across rows (cheap; just for the readout).
    let n = KH_GRID + 1;
    let dx = 1.0 / KH_GRID as f64;
    let dim = th.dim;
    let mut max_w: f64 = 0.0;
    for j in 1..(n - 1) {
        for i in 1..(n - 1) {
            let idx = |ii: usize, jj: usize| jj * n + ii;
            let dvdx = (st.u[idx(i + 1, j) * dim + 1] - st.u[idx(i - 1, j) * dim + 1]) / (2.0 * dx);
            let dudy = (st.u[idx(i, j + 1) * dim] - st.u[idx(i, j - 1) * dim]) / (2.0 * dx);
            let w = (dvdx - dudy).abs();
            if w > max_w {
                max_w = w;
            }
        }
    }
    st.vorticity_max = max_w;
}

fn draw_warp_fem_kh_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpFemKelvinHelmholtz {
        return;
    }
    let st = &warp.fem_kh;
    let Some(mesh) = st.mesh.as_ref() else {
        return;
    };
    let Some(th) = st.th.as_ref() else {
        return;
    };
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    let dim = th.dim;
    for (i, p) in mesh.vertices.iter().enumerate() {
        let ux = st.u[i * dim] as f32;
        let vy = st.u[i * dim + 1] as f32;
        let from = origin + Vec3::new(p.x as f32, 0.0, p.y as f32);
        let to = from + Vec3::new(ux * 0.05, 0.0, vy * 0.05);
        let col = if p.y > 0.5 {
            Color::srgb(0.95, 0.5, 0.3)
        } else {
            Color::srgb(0.3, 0.6, 0.95)
        };
        gizmos.line(from, to, col);
    }
}

fn warp_fem_kh_pane(ui: &mut egui::Ui, st: &mut WarpFemKhState) {
    ui.label("Warp FEM Kelvin-Helmholtz — Stage 37.3 NS time-stepping");
    ui.separator();
    readout(ui, "t", &format!("{:.2} s", st.sim_t));
    readout(ui, "max |ω|", &format!("{:.3e}", st.vorticity_max));
    ui.label("Viscosity ν");
    let mut nu = st.viscosity;
    if ui
        .add(egui::Slider::new(&mut nu, 0.0001..=0.05).logarithmic(true))
        .changed()
    {
        st.viscosity = nu;
        st.needs_reset = true;
    }
    ui.label("Perturbation amplitude");
    let mut amp = st.perturb_amp;
    if ui.add(egui::Slider::new(&mut amp, 0.0..=0.2)).changed() {
        st.perturb_amp = amp;
        st.needs_reset = true;
    }
    if ui.button("Reset").clicked() {
        st.needs_reset = true;
    }
    ui.small("Real FEM NS solve via solve_navier_stokes_step on a");
    ui.small("12×12 P2-P1 Taylor-Hood mesh.");
}

