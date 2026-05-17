//! Stage 43 v4 — GPU port of the per-face SWE flux (HLL / HLLC / Roe).
//!
//! The CPU oracle (Stage 43 v3) is
//! `fem::dg_element::compute_swe_numerical_flux_{hll,hllc,roe}`. The
//! GPU kernel lives in `kernels::DgSweFluxKernel`. This test:
//!
//! 1. Builds a thin-strip dam-break problem (same setup as
//!    `fem_dg_v3.rs::run_swe_dam_break` with `nx=40, ny=2`).
//! 2. Computes a deterministic non-zero state by stepping the CPU
//!    solver for a handful of small dt values so the trace is
//!    interesting (mix of subsonic / transonic across faces).
//! 3. Evaluates the **face-only** DG residual contribution
//!    `-Σ_F ∫ Ĥ · ⟦v⟧ ds` once on the CPU using the v3 helpers and
//!    once on the GPU via [`DgSweFluxKernel`].
//! 4. Asserts `|gpu - cpu| / max(|cpu|, scale) < 1e-4` per cell node
//!    per component, with `scale` set to the L∞ norm of the CPU
//!    residual to suppress noise on bins where the residual is
//!    essentially zero.
//!
//! Three variants are exercised (HLL / HLLC / Roe). Each shares the
//! same state + face geometry so any divergence is purely the per-flux
//! arithmetic path.
//!
//! Marked `#[ignore]` — requires a working `wgpu` adapter.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_fem::{
    DgSweFace, DgSweFluxKernel, DgSweFluxVariant, DgSweLimiterKernel, build_cell_neighbours,
    dg_ndof_per_cell, limit_solution, limit_solution_characteristic, pack_dg_swe_limiter_cells,
};
use molla_fem::{
    FemCell, FemMesh, compute_interior_faces, compute_swe_numerical_flux_hll,
    compute_swe_numerical_flux_hllc, compute_swe_numerical_flux_roe,
};
use molla_math::{Real, Vec3};

// Serialise wgpu adapter lifetime across tests in this binary (mirrors
// the GPU_LOCK pattern in `fem_elasticity_gpu.rs`).
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// One face's pre-computed geometry, mirrors the `FaceData` from
/// `fem_dg_v3.rs::run_swe_dam_break` so that the CPU oracle here uses
/// the same orientation + local-vertex mapping as the GPU upload.
#[derive(Clone, Debug)]
struct FaceData {
    cell_l: usize,
    cell_r: usize,
    /// Local node indices within `cell_l` (P1: 0..3) for the two face
    /// vertices. Order: matches `f.face_verts[0]` then `[1]`.
    local_l: [usize; 2],
    /// Local node indices within `cell_r` for the same two vertices.
    local_r: [usize; 2],
    n_lr: Vec3,
    len: Real,
}

fn build_face_data(mesh: &FemMesh) -> Vec<FaceData> {
    let interior = compute_interior_faces(mesh);
    interior
        .iter()
        .map(|f| {
            let cl = f.cell_left as usize;
            let cr = f.cell_right as usize;
            let v_l = match &mesh.cells[cl] {
                FemCell::Tri(v) => v,
                FemCell::Tet(_) => unreachable!(),
            };
            let v_r = match &mesh.cells[cr] {
                FemCell::Tri(v) => v,
                FemCell::Tet(_) => unreachable!(),
            };
            let c_l = (mesh.vertices[v_l[0] as usize]
                + mesh.vertices[v_l[1] as usize]
                + mesh.vertices[v_l[2] as usize])
                / 3.0;
            let c_r = (mesh.vertices[v_r[0] as usize]
                + mesh.vertices[v_r[1] as usize]
                + mesh.vertices[v_r[2] as usize])
                / 3.0;
            let a = mesh.vertices[f.face_verts[0] as usize];
            let b = mesh.vertices[f.face_verts[1] as usize];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let len = (dx * dx + dy * dy).sqrt();
            let raw = Vec3::new(dy / len, -dx / len, 0.0);
            let n_lr = if raw.dot(c_r - c_l) >= 0.0 { raw } else { -raw };
            let local_l = [
                v_l.iter()
                    .position(|&q| q as u32 == f.face_verts[0])
                    .unwrap(),
                v_l.iter()
                    .position(|&q| q as u32 == f.face_verts[1])
                    .unwrap(),
            ];
            let local_r = [
                v_r.iter()
                    .position(|&q| q as u32 == f.face_verts[0])
                    .unwrap(),
                v_r.iter()
                    .position(|&q| q as u32 == f.face_verts[1])
                    .unwrap(),
            ];
            FaceData {
                cell_l: cl,
                cell_r: cr,
                local_l,
                local_r,
                n_lr,
                len,
            }
        })
        .collect()
}

/// CPU reference: compute the face-only residual contribution
/// `r[i] = -Σ_F ∫ Ĥ · ⟦v⟧ ds` for the given flux variant. Layout:
/// `r[cell * 9 + i * 3 + c]` with `i ∈ 0..3, c ∈ {h, hu, hv}`.
fn cpu_face_residual(
    mesh: &FemMesh,
    state: &[Real],
    faces: &[FaceData],
    variant: DgSweFluxVariant,
    g: Real,
) -> Vec<Real> {
    let n_cells = mesh.cells.len();
    let mut r = vec![0.0_f64 as Real; n_cells * 3 * 3];

    let flux_fn: fn([Real; 3], [Real; 3], Vec3, Real) -> [Real; 3] = match variant {
        DgSweFluxVariant::Hll => compute_swe_numerical_flux_hll,
        DgSweFluxVariant::Hllc => compute_swe_numerical_flux_hllc,
        DgSweFluxVariant::Roe => compute_swe_numerical_flux_roe,
    };

    for f in faces {
        let off_l = f.cell_l * 3;
        let off_r = f.cell_r * 3;
        let gp = 1.0 / (3.0_f64).sqrt() / 2.0;
        let xis = [0.5 - gp, 0.5 + gp];
        let w_q = 0.5 * f.len;
        for &xi in &xis {
            let mut u_l_q = [0.0_f64 as Real; 3];
            let mut u_r_q = [0.0_f64 as Real; 3];
            for c in 0..3 {
                u_l_q[c] = (1.0 - xi) * state[(off_l + f.local_l[0]) * 3 + c]
                    + xi * state[(off_l + f.local_l[1]) * 3 + c];
                u_r_q[c] = (1.0 - xi) * state[(off_r + f.local_r[0]) * 3 + c]
                    + xi * state[(off_r + f.local_r[1]) * 3 + c];
            }
            let h_hat = flux_fn(u_l_q, u_r_q, f.n_lr, g);
            let nl_at_a = 1.0 - xi;
            let nl_at_b = xi;
            let nr_at_a = 1.0 - xi;
            let nr_at_b = xi;
            for c in 0..3 {
                r[(off_l + f.local_l[0]) * 3 + c] -= w_q * h_hat[c] * nl_at_a;
                r[(off_l + f.local_l[1]) * 3 + c] -= w_q * h_hat[c] * nl_at_b;
                r[(off_r + f.local_r[0]) * 3 + c] -= w_q * h_hat[c] * (-nr_at_a);
                r[(off_r + f.local_r[1]) * 3 + c] -= w_q * h_hat[c] * (-nr_at_b);
            }
        }
    }
    r
}

/// Build a non-trivial dam-break initial state. Linear `h` ramp around
/// the dam so that traces are non-constant across each face — pinpoints
/// flux-arithmetic differences that a flat reservoir wouldn't.
fn build_state(mesh: &FemMesh, dam_x: Real, h_l: Real, h_r: Real) -> Vec<Real> {
    let n_cells = mesh.cells.len();
    let mut state = vec![0.0_f64 as Real; n_cells * 3 * 3];
    for (ci, cell) in mesh.cells.iter().enumerate() {
        let off = ci * 3;
        let v = match cell {
            FemCell::Tri(v) => v,
            FemCell::Tet(_) => unreachable!(),
        };
        for (i, &vi) in v.iter().enumerate() {
            let p = mesh.vertices[vi as usize];
            // Smooth ramp around the dam: linear drop in `h` over a
            // 0.4-unit window, plus a small velocity perturbation that
            // varies with `y` so the tangential component is non-zero
            // for at least some faces (otherwise the HLLC star region
            // would simplify to HLL exactly for every face).
            let r = ((p.x - dam_x) / 0.2).clamp(-1.0, 1.0);
            let h = 0.5 * (h_l + h_r) - 0.5 * (h_l - h_r) * r;
            let hu = 0.05 * h * (1.0 - r * r); // pulse moving right
            let hv = 0.02 * h * (p.y - 0.05).sin();
            state[(off + i) * 3 + 0] = h;
            state[(off + i) * 3 + 1] = hu;
            state[(off + i) * 3 + 2] = hv;
        }
    }
    state
}

fn cpu_scalar_limited_swe(mesh: &FemMesh, state: &[Real], physical_h_min: Real) -> Vec<Real> {
    let degree = 1;
    let components = 3;
    let neighbours = build_cell_neighbours(mesh);
    let n_scalar: usize = mesh.cells.iter().map(|c| dg_ndof_per_cell(c, degree)).sum();
    let mut out = state.to_vec();
    for c in 0..components {
        let mut scratch = vec![0.0_f64 as Real; n_scalar];
        for i in 0..n_scalar {
            scratch[i] = out[i * components + c];
        }
        let (lo, hi) = if c == 0 {
            (physical_h_min, Real::INFINITY)
        } else {
            (Real::NEG_INFINITY, Real::INFINITY)
        };
        limit_solution(mesh, &mut scratch, degree, &neighbours, lo, hi);
        for i in 0..n_scalar {
            out[i * components + c] = scratch[i];
        }
    }
    out
}

fn max_scaled_error(cpu: &[Real], gpu: &[f32]) -> (Real, Real, usize) {
    assert_eq!(cpu.len(), gpu.len());
    let scale = cpu
        .iter()
        .fold(0.0_f64 as Real, |acc, v| acc.max(v.abs()))
        .max(1.0e-9);
    let mut max_abs = 0.0_f64 as Real;
    let mut max_rel = 0.0_f64 as Real;
    let mut argmax = 0usize;
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let abs = (*c - *g as Real).abs();
        let rel = abs / scale;
        if rel > max_rel {
            max_rel = rel;
            max_abs = abs;
            argmax = i;
        }
    }
    (max_abs, max_rel, argmax)
}

/// Parity helper: runs CPU + GPU for one variant and asserts agreement.
fn run_parity(
    kernel: &DgSweFluxKernel,
    variant: DgSweFluxVariant,
    mesh: &FemMesh,
    faces: &[FaceData],
    state: &[Real],
    g: Real,
    tol_rel: Real,
    label: &str,
) -> (Real, Real) {
    // CPU.
    let r_cpu = cpu_face_residual(mesh, state, faces, variant, g);

    // GPU: convert state to f32, pack face records, dispatch.
    let state_f32: Vec<f32> = state.iter().map(|x| *x as f32).collect();
    let face_records: Vec<DgSweFace> = faces
        .iter()
        .map(|f| DgSweFace {
            cell_l: f.cell_l as u32,
            cell_r: f.cell_r as u32,
            local_l0: f.local_l[0] as u32,
            local_l1: f.local_l[1] as u32,
            local_r0: f.local_r[0] as u32,
            local_r1: f.local_r[1] as u32,
            nx: f.n_lr.x as f32,
            ny: f.n_lr.y as f32,
            length: f.len as f32,
            _pad: 0.0,
        })
        .collect();

    let state_buf = kernel.upload_state(&state_f32);
    let faces_buf = kernel.upload_faces(&face_records);
    let resid_buf = kernel.alloc_residual(mesh.cells.len());

    kernel
        .dispatch(
            variant,
            &state_buf,
            &faces_buf,
            &resid_buf,
            face_records.len() as u32,
            g as f32,
        )
        .expect("gpu dispatch");
    let r_gpu_f32 = kernel
        .read_buffer_f32(&resid_buf, (mesh.cells.len() * 9) as u32)
        .expect("readback");

    assert_eq!(r_cpu.len(), r_gpu_f32.len());

    // Use the L∞ norm of the CPU residual as the absolute scale so the
    // relative tolerance is meaningful for slots where the residual is
    // essentially zero (rounding-error noise in the GPU shouldn't fail
    // a comparison against a true-zero entry).
    let mut scale: Real = 0.0;
    for &v in &r_cpu {
        if v.abs() > scale {
            scale = v.abs();
        }
    }
    let scale = scale.max(1.0e-9);

    let mut max_abs: Real = 0.0;
    let mut max_rel: Real = 0.0;
    let mut argmax_idx: usize = 0;
    for (i, (cpu, gpu)) in r_cpu.iter().zip(r_gpu_f32.iter()).enumerate() {
        let g64 = *gpu as Real;
        let abs = (g64 - cpu).abs();
        let rel = abs / scale;
        if rel > max_rel {
            max_rel = rel;
            argmax_idx = i;
            max_abs = abs;
        }
    }
    eprintln!(
        "[{label}] L∞ residual scale = {scale:.3e}, max_abs_diff = {max_abs:.3e}, \
         max_rel = {max_rel:.3e} @ slot {argmax_idx} (cpu = {:.3e}, gpu = {:.3e})",
        r_cpu[argmax_idx], r_gpu_f32[argmax_idx] as Real,
    );
    assert!(
        max_rel < tol_rel,
        "{label} parity: max_rel {max_rel:.3e} >= tol {tol_rel:.3e}",
    );
    (max_abs, max_rel)
}

#[test]
#[ignore = "needs a real wgpu adapter — run with --features gpu --test fem_dg_swe_gpu -- --ignored"]
fn dg_swe_flux_gpu_matches_cpu_hll_hllc_roe() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let l_total: Real = 4.0;
    let w: Real = 0.1;
    let nx = 40usize;
    let ny = 2usize;
    let g: Real = 9.81;
    let h_l: Real = 1.0;
    let h_r: Real = 0.1;
    let dam_x: Real = l_total / 2.0;

    let mesh = FemMesh::tri_grid_2d(nx, ny, l_total, w);
    let faces = build_face_data(&mesh);
    let state = build_state(&mesh, dam_x, h_l, h_r);

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().expect("gpu device").clone();
    let queue = engine.queue().expect("gpu queue").clone();
    let kernel = DgSweFluxKernel::new(device, queue);

    let tol: Real = 1.0e-4;
    let (a_hll, r_hll) = run_parity(
        &kernel,
        DgSweFluxVariant::Hll,
        &mesh,
        &faces,
        &state,
        g,
        tol,
        "HLL",
    );
    let (a_hllc, r_hllc) = run_parity(
        &kernel,
        DgSweFluxVariant::Hllc,
        &mesh,
        &faces,
        &state,
        g,
        tol,
        "HLLC",
    );
    let (a_roe, r_roe) = run_parity(
        &kernel,
        DgSweFluxVariant::Roe,
        &mesh,
        &faces,
        &state,
        g,
        tol,
        "Roe",
    );
    eprintln!(
        "DG SWE GPU parity summary: HLL  abs={a_hll:.3e} rel={r_hll:.3e} | \
         HLLC abs={a_hllc:.3e} rel={r_hllc:.3e} | \
         Roe  abs={a_roe:.3e} rel={r_roe:.3e} | \
         cells={} faces={}",
        mesh.cells.len(),
        faces.len()
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run with --features gpu --test fem_dg_swe_gpu -- --ignored"]
fn dg_swe_limiters_gpu_match_cpu_scalar_and_characteristic() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let l_total: Real = 4.0;
    let w: Real = 0.1;
    let nx = 40usize;
    let ny = 2usize;
    let g: Real = 9.81;
    let h_l: Real = 1.0;
    let h_r: Real = 0.1;
    let dam_x: Real = l_total / 2.0;
    let physical_h_min: Real = 0.0;

    let mesh = FemMesh::tri_grid_2d(nx, ny, l_total, w);
    let neighbours = build_cell_neighbours(&mesh);
    let state = build_state(&mesh, dam_x, h_l, h_r);
    let limiter_cells = pack_dg_swe_limiter_cells(&mesh, &neighbours);

    let cpu_scalar = cpu_scalar_limited_swe(&mesh, &state, physical_h_min);
    let mut cpu_characteristic = state.clone();
    limit_solution_characteristic(
        &mesh,
        &mut cpu_characteristic,
        1,
        &neighbours,
        g,
        physical_h_min,
    );

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().expect("gpu device").clone();
    let queue = engine.queue().expect("gpu queue").clone();
    let kernel = DgSweLimiterKernel::new(device, queue);

    let state_f32: Vec<f32> = state.iter().map(|x| *x as f32).collect();
    let state_buf = kernel.upload_state(&state_f32);
    let cells_buf = kernel.upload_cells(&limiter_cells);

    let out_scalar = kernel.alloc_state(mesh.cells.len());
    kernel
        .dispatch_scalar(
            &state_buf,
            &cells_buf,
            &out_scalar,
            mesh.cells.len() as u32,
            physical_h_min as f32,
        )
        .expect("scalar limiter dispatch");
    let gpu_scalar = kernel
        .read_buffer_f32(&out_scalar, (mesh.cells.len() * 9) as u32)
        .expect("scalar limiter readback");

    let out_characteristic = kernel.alloc_state(mesh.cells.len());
    kernel
        .dispatch_characteristic(
            &state_buf,
            &cells_buf,
            &out_characteristic,
            mesh.cells.len() as u32,
            g as f32,
            physical_h_min as f32,
        )
        .expect("characteristic limiter dispatch");
    let gpu_characteristic = kernel
        .read_buffer_f32(&out_characteristic, (mesh.cells.len() * 9) as u32)
        .expect("characteristic limiter readback");

    let tol: Real = 2.0e-5;
    let (scalar_abs, scalar_rel, scalar_idx) = max_scaled_error(&cpu_scalar, &gpu_scalar);
    eprintln!(
        "DG SWE scalar limiter GPU parity: max_abs={scalar_abs:.3e}, max_rel={scalar_rel:.3e} @ {scalar_idx}"
    );
    assert!(
        scalar_rel < tol,
        "scalar limiter parity: max_rel {scalar_rel:.3e} >= tol {tol:.3e}",
    );

    let (chr_abs, chr_rel, chr_idx) = max_scaled_error(&cpu_characteristic, &gpu_characteristic);
    eprintln!(
        "DG SWE characteristic limiter GPU parity: max_abs={chr_abs:.3e}, max_rel={chr_rel:.3e} @ {chr_idx}"
    );
    assert!(
        chr_rel < tol,
        "characteristic limiter parity: max_rel {chr_rel:.3e} >= tol {tol:.3e}",
    );
}
