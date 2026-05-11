//! Stage 48 — time-harmonic Maxwell / eddy-current solver tests.
//!
//! Two contract tests:
//!
//! 1. **Skin-depth decay** — drive a conductive slab with an in-plane
//!    surface current and verify that `|A|` decays exponentially with
//!    depth at the analytic skin depth `δ = √(2/(ωσμ))`.
//! 2. **Low-frequency limit** — at `ω → 0` the eddy-current solver
//!    must converge to the Stage 46 magnetostatics solution within
//!    `1e-3` relative error.

use molla::fem::{
    compute_curl_per_cell, solve_eddy_current_steady, solve_magnetostatics_steady,
    EddyCurrentConfig, FemCell, FemMesh, NedelecDofMap,
};
use std::collections::BTreeMap;
use molla::math::{Real, Vec3};

// ─── Test 1: Skin-depth decay in a conductive slab ─────────────────

/// Conductive box `[-Lx/2, Lx/2] × [-Ly/2, Ly/2] × [0, Lz]` with a
/// tangential surface current driving the front face (z = 0).
///
/// **Continuous setup.** Treat the slab as semi-infinite in `+z` with
/// a sheet current `K = K₀ ê_y` at `z = 0`. Inside the conductor the
/// A-formulation reduces (in the 1D limit, neglecting edge effects)
/// to
///
/// ```text
///   d²A_y / dz² = i ω σ μ A_y          ⇒   A_y(z) = A₀ e^{-(1+i) z / δ}
/// ```
///
/// with **skin depth** `δ = √(2 / (ω σ μ))`. The magnitude decays as
/// `|A_y(z)| = A₀ exp(-z / δ)` — that's what we measure.
///
/// **Discrete setup.** We use a Cartesian tet grid. A "surface
/// current" is approximated by a constant body current density
/// `J = J₀ ê_y` in the first cell layer (`z < dz`), zero everywhere
/// else. We then read `|A_y|` along z by sampling Nedelec DoFs on
/// edges aligned with `ê_y`, fit a line to `log |A_y|` vs `z`, and
/// compare the recovered decay length to the analytic skin depth.
///
/// Tolerance: numerical skin depth within `10 %` of analytic.
#[test]
fn eddy_current_skin_depth_decay() {
    // Domain. The "perfect electric conductor" BC `n × A = 0` on
    // the four lateral faces turns the box into a rectangular
    // waveguide whose cutoff contributes a lateral wavenumber to
    // the dispersion:
    //
    // ```text
    //   k_z² ≈ k_⊥² + i ω σ μ ,    k_⊥ ≈ π / L⊥
    // ```
    //
    // For the y-polarised driving current the relevant transverse
    // wavenumber is `k_x = π / Lx` (A_y is automatically zero on
    // the y-walls). The numerical decay rate matches
    // `Re(√(k_x² + i ω σ μ))` to within a few percent on a
    // reasonably-isotropic mesh.
    //
    // Parameters: roughly isotropic cells (dx = dy ≈ 4 dz) so the
    // discrete curl-curl operator is well-conditioned. Box is long
    // enough to fit ~3 skin depths along z but not so long that the
    // gauge-null-space residue dominates the tail.
    let nx = 4usize;
    let ny = 4usize;
    let nz = 32usize;

    let omega: Real = 1.0;
    let sigma: Real = 4.0;
    let mu: Real = 1.0;
    let nu: Real = 1.0 / mu;
    let lx = 1.0;
    let ly = 1.0;
    let lz = 4.0;

    let mesh = FemMesh::tet_grid_3d(nx, ny, nz, lx, ly, lz);
    let mesh = recenter_xy(mesh, lx, ly);
    let n_cells = mesh.cells.len();

    // Analytic skin depth — naïve 1D formula and the box-corrected
    // value that accounts for lateral confinement.
    let skin_depth_1d: Real = (2.0 / (omega * sigma * mu)).sqrt();
    // Lateral cutoff for a rectangular waveguide with `n × A = 0` on
    // the four lateral walls — lowest TE/TM mode `(π/Lx, π/Ly)`.
    // For a y-polarised current the only relevant transverse
    // wavenumber is the x mode (`k_x = π/Lx`) since A_y is
    // automatically zero on the y-walls.
    let k_perp_sq: Real = (std::f64::consts::PI / lx).powi(2);
    let k_sq_complex_re: Real = k_perp_sq;
    let k_sq_complex_im: Real = omega * sigma * mu;
    let r = (k_sq_complex_re.powi(2) + k_sq_complex_im.powi(2)).sqrt();
    let theta = k_sq_complex_im.atan2(k_sq_complex_re);
    let kz_re: Real = r.sqrt() * (theta * 0.5).cos();
    let skin_depth_analytic: Real = 1.0 / kz_re;
    let dz = lz / nz as Real;
    eprintln!(
        "skin-depth test: δ_1D = {:.4}, δ_box = {:.4}, dz = {:.4} ⇒ {:.1} cells/δ_box",
        skin_depth_1d,
        skin_depth_analytic,
        dz,
        skin_depth_analytic / dz
    );

    // Per-cell coefficients. σ everywhere, ν = 1/μ everywhere, ε = 0
    // (eddy-current limit — no displacement current).
    let nu_per_cell = vec![nu; n_cells];
    let sigma_per_cell = vec![sigma; n_cells];
    let epsilon_per_cell = vec![0.0; n_cells];
    let cfg = EddyCurrentConfig {
        omega,
        epsilon_per_cell,
        sigma_per_cell,
        nu_per_cell,
    };

    // Surface current: drive J = J₀ ê_y in the first cell layer
    // (`z < dz`), zero elsewhere. The driving phase is purely real so
    // we put it into `J_re` (and `J_im = 0`).
    let j0: Real = 1.0;
    let mut j_source: Vec<(Vec3, Vec3)> = vec![(Vec3::ZERO, Vec3::ZERO); n_cells];
    for (i, cell) in mesh.cells.iter().enumerate() {
        let c = cell_centroid(cell, &mesh.vertices);
        if c.z < dz {
            j_source[i] = (Vec3::new(0.0, j0, 0.0), Vec3::ZERO);
        }
    }

    // Solve.
    let result = solve_eddy_current_steady(&mesh, &j_source, &cfg, &[]).expect("eddy solve");
    eprintln!(
        "eddy-current solve: iters = {}, residual = {:.3e}",
        result.iters, result.residual
    );

    // Extract |B_x|(z) — gauge-invariant observable, satisfies the
    // same 1D diffusion equation as A_y but is unaffected by the
    // gradient null space of the curl-curl operator.
    //
    // `compute_curl_per_cell` returns one B vector per cell. Group
    // cells by centroid z-layer (one bin per `dz` slab) and average
    // `|B_x|` across xy + Kuhn-decomposition strata.
    let map = NedelecDofMap::build(&mesh);
    let b_re = compute_curl_per_cell(&mesh, &map, &result.a_re);
    let b_im = compute_curl_per_cell(&mesh, &map, &result.a_im);
    let mut by_zbin: BTreeMap<i32, (Real, usize)> = BTreeMap::new();
    for (i, cell) in mesh.cells.iter().enumerate() {
        let c = cell_centroid(cell, &mesh.vertices);
        let bx_mag = (b_re[i].x.powi(2) + b_im[i].x.powi(2)).sqrt();
        let bin = (c.z / dz).floor() as i32;
        let entry = by_zbin.entry(bin).or_insert((0.0, 0));
        entry.0 += bx_mag;
        entry.1 += 1;
    }
    let avg: Vec<(Real, Real)> = by_zbin
        .into_iter()
        .map(|(bin, (sum, n))| {
            let zc = (bin as Real + 0.5) * dz;
            (zc, sum / n as Real)
        })
        .collect();
    // Uncomment for full |B_x|(z) profile dump.
    // for (z, mag) in &avg {
    //     eprintln!("  z = {:6.3}, |B_x| = {:.4e}, log = {:7.3}", z, mag, mag.ln());
    // }
    eprintln!(
        "|B_x|(z) profile: {} bins, near-source max = {:.3e}, far-end min = {:.3e}",
        avg.len(),
        avg.first().map(|(_, m)| *m).unwrap_or(0.0),
        avg.last().map(|(_, m)| *m).unwrap_or(0.0)
    );

    // Fit `log |B_x| = const − z / δ` to a window where the
    // eddy-current decay is the dominant mode (and the
    // gauge-null-space static residue is still negligible). Pick
    // `[z₁, z₂] = [0.5 δ, 2.5 δ]` from the source — well past the
    // source transient but not yet near the far boundary or the
    // noise floor where the ε_reg-stabilised gradient component
    // becomes comparable.
    let z_a_target = skin_depth_analytic * 0.5;
    let z_b_target = skin_depth_analytic * 2.5;
    let mut a = avg
        .iter()
        .min_by(|a, b| {
            let da = (a.0 - z_a_target).abs();
            let db = (b.0 - z_a_target).abs();
            da.partial_cmp(&db).unwrap()
        })
        .copied()
        .expect("no avg samples");
    let mut b = avg
        .iter()
        .min_by(|a, b| {
            let da = (a.0 - z_b_target).abs();
            let db = (b.0 - z_b_target).abs();
            da.partial_cmp(&db).unwrap()
        })
        .copied()
        .expect("no avg samples");
    // Defensive: log will explode if A ≤ 0. Replace tiny values with
    // a floor.
    a.1 = a.1.max(1e-18);
    b.1 = b.1.max(1e-18);
    let slope = (b.1.ln() - a.1.ln()) / (b.0 - a.0);
    let skin_depth_num = -1.0 / slope;

    let rel_err = (skin_depth_num - skin_depth_analytic).abs() / skin_depth_analytic;
    eprintln!(
        "skin depth: numerical = {:.4}, analytic = {:.4}, rel err = {:.2}%",
        skin_depth_num,
        skin_depth_analytic,
        100.0 * rel_err
    );
    assert!(
        skin_depth_num > 0.0,
        "numerical skin depth is non-positive ({skin_depth_num}); decay is wrong direction"
    );
    assert!(
        rel_err < 0.10,
        "skin depth: numerical {skin_depth_num} vs analytic {skin_depth_analytic} (err {:.2}%)",
        100.0 * rel_err
    );
}

// ─── Test 2: Low-frequency limit matches magnetostatics ───────────

/// At `ω → 0` the eddy-current A-formulation
///
/// ```text
///   curl(ν curl A) + i ω σ A − ω² ε A = J
/// ```
///
/// reduces to `curl(ν curl A) = J` — the Stage 46 magnetostatics
/// equation. We solve the same source on both pipelines and compare
/// the **physical observable** `B = curl A` per cell (which is the
/// gauge-invariant quantity).
///
/// We compare `B` rather than `A` directly because the curl-curl
/// operator has a huge gradient kernel and both solvers use a tiny
/// `ε M` regularisation that pins different (but small) gradient
/// components — the resulting `A` differs in null-space components
/// while `B = curl A` is identical to within the discretisation
/// + iterative-residual error.
///
/// Tolerance: `‖B_eddy − B_mag‖ / ‖B_mag‖ < 1e-3`. We use a small
/// finite `ω = 1e-6` rather than literal `0` so that BiCGSTAB never
/// hits an exactly singular off-block.
#[test]
fn eddy_current_low_frequency_limit_matches_magnetostatics() {
    let nx = 4usize;
    let ny = 4usize;
    let nz = 4usize;
    let lx = 1.0;
    let ly = 1.0;
    let lz = 1.0;
    let mesh = FemMesh::tet_grid_3d(nx, ny, nz, lx, ly, lz);
    let n_cells = mesh.cells.len();

    // Simple non-zero source: J(x, y, z) = (0, 1, 0) in the central
    // sphere of cells, 0 elsewhere.
    let mut j_real = vec![Vec3::ZERO; n_cells];
    for (i, cell) in mesh.cells.iter().enumerate() {
        let c = cell_centroid(cell, &mesh.vertices);
        let r = ((c.x - 0.5).powi(2) + (c.y - 0.5).powi(2) + (c.z - 0.5).powi(2)).sqrt();
        if r < 0.25 {
            j_real[i] = Vec3::new(0.0, 1.0, 0.0);
        }
    }

    let nu_per_cell = vec![1.0; n_cells];

    // Magnetostatics solution.
    let mag = solve_magnetostatics_steady(&mesh, &j_real, &nu_per_cell, &[])
        .expect("magnetostatics solve");
    eprintln!(
        "magnetostatics: iters = {}, residual = {:.3e}",
        mag.iters, mag.residual
    );

    // Eddy-current at small ω. Coefficients chosen so the ε/σ
    // contribution to A is `O(ω) · ‖A_mag‖` — well below the 1e-3
    // tolerance.
    let omega: Real = 1.0e-6;
    let cfg = EddyCurrentConfig {
        omega,
        epsilon_per_cell: vec![0.0; n_cells],
        sigma_per_cell: vec![1.0; n_cells],
        nu_per_cell: vec![1.0; n_cells],
    };
    let j_complex: Vec<(Vec3, Vec3)> = j_real.iter().map(|v| (*v, Vec3::ZERO)).collect();
    let eddy = solve_eddy_current_steady(&mesh, &j_complex, &cfg, &[]).expect("eddy solve");
    eprintln!(
        "eddy @ ω={}: iters = {}, residual = {:.3e}",
        omega, eddy.iters, eddy.residual
    );

    // Compute B per cell from both solvers and compare.
    let map = NedelecDofMap::build(&mesh);
    let b_mag = compute_curl_per_cell(&mesh, &map, &mag.a);
    let b_eddy = compute_curl_per_cell(&mesh, &map, &eddy.a_re);
    let b_eddy_im = compute_curl_per_cell(&mesh, &map, &eddy.a_im);

    let mut sum_diff_sq: Real = 0.0;
    let mut sum_ref_sq: Real = 0.0;
    let mut max_im: Real = 0.0;
    for i in 0..n_cells {
        let d = b_mag[i] - b_eddy[i];
        sum_diff_sq += d.length_squared();
        sum_ref_sq += b_mag[i].length_squared();
        max_im = max_im.max(b_eddy_im[i].length());
    }
    let rel = (sum_diff_sq / sum_ref_sq.max(1e-30)).sqrt();
    eprintln!(
        "low-ω: ‖B_eddy − B_mag‖ / ‖B_mag‖ = {:.3e}, max |B_im| = {:.3e}",
        rel, max_im
    );
    assert!(
        rel < 1.0e-3,
        "B_eddy vs B_mag: rel diff {rel} (‖B_mag‖² = {sum_ref_sq})"
    );
}

// ─── Helpers ───────────────────────────────────────────────────────

fn cell_centroid(cell: &FemCell, vertices: &[Vec3]) -> Vec3 {
    match cell {
        FemCell::Tet(v) => {
            let p = vertices[v[0] as usize]
                + vertices[v[1] as usize]
                + vertices[v[2] as usize]
                + vertices[v[3] as usize];
            p / 4.0
        }
        FemCell::Tri(v) => {
            let p = vertices[v[0] as usize]
                + vertices[v[1] as usize]
                + vertices[v[2] as usize];
            p / 3.0
        }
    }
}

/// Translate `tet_grid_3d(_, _, _, Lx, Ly, _)` so the xy domain is
/// centred on the origin (z unchanged — we want z ∈ [0, Lz]).
fn recenter_xy(mesh: FemMesh, lx: Real, ly: Real) -> FemMesh {
    let shift = Vec3::new(-lx / 2.0, -ly / 2.0, 0.0);
    let FemMesh { mut vertices, cells } = mesh;
    for v in &mut vertices {
        *v = *v + shift;
    }
    FemMesh { vertices, cells }
}
