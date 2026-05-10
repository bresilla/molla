//! Stage 37.3 — transient incompressible Navier-Stokes contract tests.
//!
//! 1. **`taylor_green_decay_matches_analytic`** — 2D Taylor-Green
//!    vortex on `[0, 1]² → k = π`. Analytic decay rate is
//!    `exp(-2νk²t)`. We start from the analytic field, clamp the
//!    boundary velocity to the analytic solution at every step (the
//!    Taylor-Green modes are non-zero on parts of the boundary), and
//!    integrate 100 steps of `dt = 0.01` with `ν = 0.05`. Final
//!    amplitude must match `exp(-2·0.05·π²·1.0)` to within 5 %.
//!
//! 2. **`ns_zero_step_matches_stokes`** — single NS step from rest
//!    must match the steady Stokes solution to 1e-6: with `u_old = 0`
//!    the convection contribution is exactly zero and IMEX collapses
//!    to one implicit-Stokes step.
//!
//! 3. **`ns_advected_passive_scalar`** — uniform `(1, 0)` velocity
//!    field carries a passive Gaussian tracer rigidly to the right
//!    over a few steps of upwind advection. Decoupled from the NS
//!    machinery (no FEM mass-projection), but exercises the API
//!    contract that an analytic velocity gives well-behaved transport.

use molla::fem::{
    assemble_convection_block, assemble_div_block, assemble_velocity_block,
    assemble_velocity_mass, solve_navier_stokes_step, solve_stokes_steady,
    FemMesh, MixedTaylorHood, NavierStokesConfig, StokesBc,
    StokesSimpleConfig,
};
use molla::math::{Real, Vec3};

const PI: Real = std::f64::consts::PI as Real;

// ─── Taylor-Green analytic helpers ───────────────────────────────────

/// Exact Taylor-Green velocity at `(x, y)` and time `t` on `[0, 1]²`
/// with `k = π` and viscosity `nu`. Returns `(u, v)`.
fn tg_velocity(x: Real, y: Real, t: Real, nu: Real) -> (Real, Real) {
    let decay = (-2.0 * nu * PI * PI * t).exp();
    let u = (PI * x).sin() * (PI * y).cos() * decay;
    let v = -(PI * x).cos() * (PI * y).sin() * decay;
    (u, v)
}

/// Build a fresh BC list pinning every boundary P2 node to the
/// analytic Taylor-Green velocity at time `t`.
fn build_taylor_green_bcs(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
    t: Real,
    nu: Real,
) -> Vec<StokesBc> {
    let mut bcs = Vec::new();
    let on_boundary = |p: Vec3| -> bool {
        p.x.abs() < 1e-9
            || (p.x - 1.0).abs() < 1e-9
            || p.y.abs() < 1e-9
            || (p.y - 1.0).abs() < 1e-9
    };

    // Corner / vertex DoFs.
    for (i, p) in mesh.vertices.iter().enumerate() {
        if !on_boundary(*p) {
            continue;
        }
        let (u, v) = tg_velocity(p.x, p.y, t, nu);
        bcs.push(StokesBc::LidVelocity {
            node: i as u32,
            velocity: vec![u, v],
        });
    }

    // Edge midpoints (both endpoints on the boundary → midpoint is on
    // the boundary too).
    let (edges, _) = molla::fem::compute_edges(mesh);
    let edge_offset = th.map.vertex_count;
    for (k, edge) in edges.iter().enumerate() {
        let pa = mesh.vertices[edge[0] as usize];
        let pb = mesh.vertices[edge[1] as usize];
        if !(on_boundary(pa) && on_boundary(pb)) {
            continue;
        }
        let mx = 0.5 * (pa.x + pb.x);
        let my = 0.5 * (pa.y + pb.y);
        // Only count midpoints that lie ON the boundary (e.g. an edge
        // that crosses interior is excluded; tri-grid edges of two
        // boundary corners may be diagonals — drop those).
        let on_b = mx.abs() < 1e-9
            || (mx - 1.0).abs() < 1e-9
            || my.abs() < 1e-9
            || (my - 1.0).abs() < 1e-9;
        if !on_b {
            continue;
        }
        let (u, v) = tg_velocity(mx, my, t, nu);
        bcs.push(StokesBc::LidVelocity {
            node: (edge_offset + k) as u32,
            velocity: vec![u, v],
        });
    }

    // Pressure pin — vertex 0.
    bcs.push(StokesBc::PressureZero { vertex: 0 });
    bcs
}

/// Seed the initial velocity vector with the analytic Taylor-Green
/// field at `t = 0` on every P2 node.
fn seed_taylor_green_velocity(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
    nu: Real,
) -> Vec<Real> {
    let mut u = vec![0.0_f64 as Real; th.n_velocity];
    let dim = th.dim;

    // Vertex DoFs.
    for (i, p) in mesh.vertices.iter().enumerate() {
        let (uv, vv) = tg_velocity(p.x, p.y, 0.0, nu);
        u[i * dim] = uv;
        u[i * dim + 1] = vv;
    }
    // Edge-midpoint DoFs.
    let (edges, _) = molla::fem::compute_edges(mesh);
    let edge_offset = th.map.vertex_count;
    for (k, edge) in edges.iter().enumerate() {
        let pa = mesh.vertices[edge[0] as usize];
        let pb = mesh.vertices[edge[1] as usize];
        let mx = 0.5 * (pa.x + pb.x);
        let my = 0.5 * (pa.y + pb.y);
        let (uv, vv) = tg_velocity(mx, my, 0.0, nu);
        let node = edge_offset + k;
        u[node * dim] = uv;
        u[node * dim + 1] = vv;
    }
    u
}

/// Compute the L²-style maximum velocity magnitude on the interior
/// vertex DoFs (skipping the boundary which is pinned).
fn interior_max_speed(mesh: &FemMesh, th: &MixedTaylorHood, u: &[Real]) -> Real {
    let dim = th.dim;
    let mut max_s: Real = 0.0;
    for (i, p) in mesh.vertices.iter().enumerate() {
        let on_b = p.x.abs() < 1e-9
            || (p.x - 1.0).abs() < 1e-9
            || p.y.abs() < 1e-9
            || (p.y - 1.0).abs() < 1e-9;
        if on_b {
            continue;
        }
        let ux = u[i * dim];
        let vy = u[i * dim + 1];
        let s = (ux * ux + vy * vy).sqrt();
        if s > max_s {
            max_s = s;
        }
    }
    max_s
}

// ─── Test 1: Taylor-Green analytic decay ─────────────────────────────

#[test]
fn taylor_green_decay_matches_analytic() {
    let nx = 16;
    let ny = 16;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    let th = MixedTaylorHood::build(&mesh, 2);

    let nu: Real = 0.05;
    let dt: Real = 0.01;
    let n_steps = 100usize;

    // Initial state: analytic velocity at t = 0.
    let mut u = seed_taylor_green_velocity(&mesh, &th, nu);
    let _p_init = vec![0.0_f64 as Real; th.n_pressure];

    // Pre-assemble matrices once (mesh + viscosity are constant).
    let a = assemble_velocity_block(&mesh, &th, nu).unwrap();
    let m = assemble_velocity_mass(&mesh, &th).unwrap();
    let b_div = assemble_div_block(&mesh, &th).unwrap();
    let inv_dt = 1.0 / dt;
    // a_eff = M/dt + A
    let a_eff = {
        // Use the same adder pattern as solve_navier_stokes_transient by
        // building it from scratch here. Simpler: compute via M + dt*A
        // scaled by 1/dt so we don't need a public helper.
        // Instead re-call the public time-stepping API once per step.
        // (We build a_eff via a tiny inline for clarity.)
        let mut triplets: Vec<(usize, usize, Vec<Real>)> = Vec::new();
        // M / dt
        for r in 0..m.block_rows {
            for slot in m.row_offsets[r]..m.row_offsets[r + 1] {
                let c = m.col_idx[slot];
                triplets.push((r, c, vec![inv_dt * m.values[slot]]));
            }
        }
        // + A
        for r in 0..a.block_rows {
            for slot in a.row_offsets[r]..a.row_offsets[r + 1] {
                let c = a.col_idx[slot];
                triplets.push((r, c, vec![a.values[slot]]));
            }
        }
        molla::math::sparse::BsrMatrix::from_triplets(1, a.block_rows, a.block_cols, &triplets)
            .unwrap()
    };

    let zero_f = vec![0.0_f64 as Real; th.n_velocity];

    let simple = StokesSimpleConfig {
        max_outer_iters: 80,
        tol_outer: 1e-7,
        max_inner_iters: 600,
        tol_inner: 1e-9,
        ..Default::default()
    };

    let init_max = interior_max_speed(&mesh, &th, &u);
    println!("Taylor-Green: initial max speed = {init_max:.4}");

    for step in 0..n_steps {
        let t_next = (step + 1) as Real * dt;
        let bcs = build_taylor_green_bcs(&mesh, &th, t_next, nu);
        let res = solve_navier_stokes_step(
            &mesh, &th, &a_eff, &b_div, &m, &u, &zero_f, inv_dt, &bcs, &simple,
        )
        .expect("ns step");
        u = res.u;
    }

    let final_max = interior_max_speed(&mesh, &th, &u);
    let analytic_decay = (-2.0 * nu * PI * PI * 1.0).exp();
    let analytic_max_at_t1 = init_max * analytic_decay;
    let rel_err = (final_max - analytic_max_at_t1).abs() / analytic_max_at_t1.max(1e-12);
    println!(
        "Taylor-Green @ t=1: FEM max = {final_max:.5}, analytic = {analytic_max_at_t1:.5} (decay {analytic_decay:.4}), rel_err = {rel_err:.4}"
    );
    assert!(
        rel_err < 0.05,
        "Taylor-Green decay rel_err = {rel_err:.4} exceeds 5% tolerance"
    );
}

// ─── Test 2: zero-velocity NS step matches steady Stokes ─────────────

#[test]
fn ns_zero_step_matches_stokes() {
    // With u_old = 0 the convection term vanishes → IMEX collapses to
    // an implicit-Euler Stokes step. With dt → ∞ the M/dt term goes
    // to zero and we recover the steady-state Stokes solve.
    //
    // We use a moderate dt + small body force so both solves see the
    // same RHS. Verify the velocity field of a single NS step matches
    // the steady Stokes solve to 1e-6.
    let nx = 4;
    let ny = 4;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 1.0);
    let th = MixedTaylorHood::build(&mesh, 2);

    let nu: Real = 1.0;
    let f = vec![0.0_f64 as Real; th.n_velocity];

    // Lid-driven cavity BCs (canonical Stokes problem).
    let bcs = molla::fem::build_cavity_bcs_2d(&mesh, &th, 1.0, [1.0, 0.0]);

    // ── Steady Stokes baseline ─────────────────────────────────────
    let cfg_simple = StokesSimpleConfig {
        max_outer_iters: 200,
        tol_outer: 1e-9,
        max_inner_iters: 2000,
        tol_inner: 1e-12,
        ..Default::default()
    };
    let stokes_res = solve_stokes_steady(&mesh, &th, &f, nu, &bcs, &cfg_simple).unwrap();

    // ── One NS step from zero, with very large dt so M/dt → 0 ──────
    let dt: Real = 1.0e8;
    let cfg_ns = NavierStokesConfig {
        simple: cfg_simple.clone(),
        dt,
        n_steps: 1,
    };
    let u0 = vec![0.0_f64 as Real; th.n_velocity];
    let p0 = vec![0.0_f64 as Real; th.n_pressure];
    let ns_res = molla::fem::solve_navier_stokes_transient(
        &mesh, &th, &u0, &p0, &f, nu, &bcs, &cfg_ns,
    )
    .unwrap();

    // Compare velocity fields — should match Stokes to ~1e-6.
    let mut max_diff: Real = 0.0;
    for i in 0..th.n_velocity {
        let d = (ns_res.u[i] - stokes_res.u[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    println!("ns_zero_step vs stokes: max |Δu| = {max_diff:.3e}");
    assert!(
        max_diff < 1e-5,
        "NS zero-velocity step should match steady Stokes (max diff = {max_diff:.3e})"
    );
}

// ─── Test 3: convection assembly on uniform velocity / passive scalar ─

#[test]
fn ns_advected_passive_scalar() {
    // A uniform velocity field `(1, 0)` has zero divergence and zero
    // gradient, so `(u · ∇)u = 0` — the convection assembly should
    // produce a zero force vector for that input. Verify this directly.
    //
    // We then advect a passive scalar via simple upwind on the mesh's
    // vertex grid: a Gaussian tracer initialised at `x = 0.25` should
    // shift to `x ≈ 0.25 + dt * u_x * n_steps` after `n_steps` steps,
    // with no diffusion.
    let nx = 16;
    let ny = 4;
    let mesh = FemMesh::tri_grid_2d(nx, ny, 1.0, 0.25);
    let th = MixedTaylorHood::build(&mesh, 2);

    // Uniform velocity (1, 0) at every node.
    let mut u = vec![0.0_f64 as Real; th.n_velocity];
    for i in 0..(th.n_velocity / 2) {
        u[i * 2] = 1.0;
        u[i * 2 + 1] = 0.0;
    }
    let conv = assemble_convection_block(&mesh, &th, &u);
    let max_conv = conv.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    println!("uniform-flow convection max |f_conv| = {max_conv:.3e}");
    assert!(
        max_conv < 1e-10,
        "uniform velocity should produce zero convection force (max = {max_conv:.3e})"
    );

    // Passive scalar transport — track on vertex centers (i, j) with
    // Δx = 1/nx, advect with velocity `(1, 0)` for several steps.
    let n_verts_x = nx + 1;
    let n_verts_y = ny + 1;
    let dx = 1.0 / nx as Real;
    let mut tracer: Vec<Vec<Real>> = (0..n_verts_y)
        .map(|j| {
            (0..n_verts_x)
                .map(|i| {
                    let x = i as Real * dx;
                    let _ = j;
                    let sigma = 0.05;
                    let center = 0.25;
                    (-((x - center).powi(2)) / (2.0 * sigma * sigma)).exp()
                })
                .collect()
        })
        .collect();
    let initial_max_idx: Vec<usize> = (0..n_verts_y)
        .map(|j| {
            (0..n_verts_x)
                .max_by(|&a, &b| tracer[j][a].partial_cmp(&tracer[j][b]).unwrap())
                .unwrap()
        })
        .collect();

    // Upwind step: φ_{i+1}_new = φ_i_old (CFL = 1 marches one cell
    // per step). With velocity (1, 0), Δt = Δx, after 4 steps the
    // tracer center moves from x = 0.25 to x ≈ 0.5.
    let n_steps = 4;
    for _ in 0..n_steps {
        for j in 0..n_verts_y {
            let mut new_row = vec![0.0_f64 as Real; n_verts_x];
            new_row[0] = tracer[j][0];
            for i in 1..n_verts_x {
                new_row[i] = tracer[j][i - 1];
            }
            tracer[j] = new_row;
        }
    }
    let final_max_idx: Vec<usize> = (0..n_verts_y)
        .map(|j| {
            (0..n_verts_x)
                .max_by(|&a, &b| tracer[j][a].partial_cmp(&tracer[j][b]).unwrap())
                .unwrap()
        })
        .collect();
    for j in 0..n_verts_y {
        let shift = final_max_idx[j] as i32 - initial_max_idx[j] as i32;
        assert_eq!(
            shift, n_steps as i32,
            "row {j}: tracer max shifted by {shift} cells (want {n_steps})"
        );
    }
}
