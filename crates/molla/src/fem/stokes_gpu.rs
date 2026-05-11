//! Stage 41 v2 — end-to-end GPU steady-state Stokes (Taylor-Hood
//! P2-P1) on triangle meshes.
//!
//! # Architecture
//!
//! 1. Build the velocity-block `A` and divergence-block `B` BSR
//!    **patterns** on the CPU (cheap — the cell-graph traversal
//!    dominates and runs in microseconds at the test mesh sizes).
//! 2. Upload the patterns + the mesh + a per-cell `p2_dof_table` to
//!    the GPU. The `p2_dof_table` maps each cell's local P2 node
//!    index to its global node index (so the WGSL kernel can stay
//!    oblivious to the corner/edge addressing scheme).
//! 3. Dispatch the two assembly entry points
//!    (`assemble_velocity_block`, `assemble_div_block`) — the GPU
//!    fills the CSR `values` buffers via atomic-add scatter.
//! 4. Read the unreduced `A`, `B` values back to the host, apply the
//!    standard symmetric-Dirichlet BC reduction (mirrors the CPU
//!    solver's `apply_symmetric_dirichlet` so the BC handling stays
//!    pixel-identical between the two paths), and re-upload the
//!    reduced values to the GPU.
//! 5. Run Uzawa-CG entirely on the GPU: the outer pressure-CG iterates
//!    on the Schur action `S p = B · A⁻¹ · Bᵀ p` where each `A x = y`
//!    is solved by an inner GPU CG via `FemCgGpuKernel`.
//! 6. Read the final velocity + pressure back to the host.

#![cfg(feature = "gpu")]

use std::sync::Arc;
use std::time::Instant;

use crate::compute::Engine;
use crate::core::{Error, Result};
use crate::fem::mesh::{FemCell, FemMesh};
use crate::fem::stokes::{
    assemble_div_block, assemble_velocity_block, collect_velocity_pins_pub,
    MixedTaylorHood, StokesBc,
};
use crate::kernels::{FemCgGpuKernel, FemStokesGpuKernel, StokesEntry};
use crate::math::{BsrAlgebraKernel, BsrGpu};
use crate::math::sparse::BsrMatrix;
use crate::math::Real;

/// Result of [`solve_stokes_gpu`].
#[derive(Clone, Debug)]
pub struct StokesGpuResult {
    pub u: Vec<Real>,
    pub p: Vec<Real>,
    pub outer_iters: usize,
    pub div_residual: Real,
}

/// Solver knobs for the GPU Uzawa-CG path.
#[derive(Clone, Debug)]
pub struct StokesGpuConfig {
    pub max_outer_iters: usize,
    pub tol_outer: f32,
    pub max_inner_iters: usize,
    pub tol_inner: f32,
}

impl Default for StokesGpuConfig {
    fn default() -> Self {
        Self {
            max_outer_iters: 200,
            tol_outer: 1.0e-5,
            max_inner_iters: 2000,
            tol_inner: 1.0e-7,
        }
    }
}

/// Assembly-only entry point — dispatches the two WGSL kernels and
/// reads back `(A, B)` to host (no BC reduction, no Uzawa). Mostly for
/// parity tests against the CPU oracles.
///
/// Returns `(a_csr_values, b_csr_values)` matching the layout of the
/// CPU [`assemble_velocity_block`] / [`assemble_div_block`] outputs.
pub fn assemble_stokes_gpu_only(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
    viscosity: Real,
    engine: &Engine,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if th.dim != 2 {
        return Err(Error::Build(
            "assemble_stokes_gpu_only: v1 only supports dim = 2".into(),
        ));
    }
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("assemble_stokes_gpu_only: no GPU device".into()))?;
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("assemble_stokes_gpu_only: no GPU queue".into()))?;

    let a_cpu = assemble_velocity_block(mesh, th, viscosity)?;
    let b_cpu = assemble_div_block(mesh, th)?;
    let mesh_buffers = upload_mesh(device, queue, mesh);
    let p2_dof_table = build_p2_dof_table(mesh, th);
    let p2_dof_buf = create_storage_buffer(
        device,
        queue,
        "molla::fem::stokes_gpu p2_dof_table (asm-only)",
        bytemuck::cast_slice(&p2_dof_table),
    );
    let a_gpu = BsrGpu::upload(device, queue, &a_cpu)?;
    zero_buffer(queue, &a_gpu.values, a_gpu.nnz as usize * 4);
    let b_gpu = BsrGpu::upload(device, queue, &b_cpu)?;
    zero_buffer(queue, &b_gpu.values, b_gpu.nnz as usize * 4);

    let asm = FemStokesGpuKernel::new(Arc::clone(device), Arc::clone(queue));
    asm.assemble(
        StokesEntry::Velocity,
        &mesh_buffers.vertices,
        &mesh_buffers.cells,
        &mesh_buffers.cell_offsets,
        &p2_dof_buf,
        &a_gpu.row_offsets,
        &a_gpu.col_indices,
        &a_gpu.values,
        mesh.cells.len() as u32,
        th.ndof_velocity() as u32,
        th.ndof_pressure() as u32,
        th.dim as u32,
        viscosity as f32,
    )?;
    asm.assemble(
        StokesEntry::Divergence,
        &mesh_buffers.vertices,
        &mesh_buffers.cells,
        &mesh_buffers.cell_offsets,
        &p2_dof_buf,
        &b_gpu.row_offsets,
        &b_gpu.col_indices,
        &b_gpu.values,
        mesh.cells.len() as u32,
        th.ndof_velocity() as u32,
        th.ndof_pressure() as u32,
        th.dim as u32,
        viscosity as f32,
    )?;
    let a_vals = read_f32_buf(device, queue, &a_gpu.values, a_gpu.nnz)?;
    let b_vals = read_f32_buf(device, queue, &b_gpu.values, b_gpu.nnz)?;
    Ok((a_vals, b_vals))
}

/// Solve the steady-state incompressible Stokes problem on the GPU.
pub fn solve_stokes_gpu(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
    f_velocity: &[Real],
    viscosity: Real,
    bcs: &[StokesBc],
    cfg: &StokesGpuConfig,
    engine: &Engine,
) -> Result<StokesGpuResult> {
    if th.dim != 2 {
        return Err(Error::Build(format!(
            "solve_stokes_gpu: v1 only supports dim = 2, got dim = {}",
            th.dim,
        )));
    }
    for c in &mesh.cells {
        if !matches!(c, FemCell::Tri(_)) {
            return Err(Error::Build(
                "solve_stokes_gpu: v1 only supports tri meshes".into(),
            ));
        }
    }
    if f_velocity.len() != th.ndof_velocity() {
        return Err(Error::Build(format!(
            "solve_stokes_gpu: f_velocity length ({}) != n_v ({})",
            f_velocity.len(),
            th.ndof_velocity(),
        )));
    }
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("solve_stokes_gpu: engine has no GPU device".into()))?;
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("solve_stokes_gpu: engine has no GPU queue".into()))?;

    let n_v = th.ndof_velocity();
    let n_p = th.ndof_pressure();
    let dim = th.dim;

    // ─── (1) CPU pattern build ──────────────────────────────────────
    let mut a_cpu = assemble_velocity_block(mesh, th, viscosity)?;
    let mut b_cpu = assemble_div_block(mesh, th)?;

    // ─── (2) Upload mesh + patterns ─────────────────────────────────
    let mesh_buffers = upload_mesh(device, queue, mesh);
    let p2_dof_table = build_p2_dof_table(mesh, th);
    let p2_dof_buf = create_storage_buffer(
        device,
        queue,
        "molla::fem::stokes_gpu p2_dof_table",
        bytemuck::cast_slice(&p2_dof_table),
    );

    let a_gpu = BsrGpu::upload(device, queue, &a_cpu)?;
    zero_buffer(queue, &a_gpu.values, a_gpu.nnz as usize * 4);
    let b_gpu = BsrGpu::upload(device, queue, &b_cpu)?;
    zero_buffer(queue, &b_gpu.values, b_gpu.nnz as usize * 4);

    // ─── (3) Dispatch GPU assembly ──────────────────────────────────
    let asm_kernel = FemStokesGpuKernel::new(Arc::clone(device), Arc::clone(queue));
    asm_kernel.assemble(
        StokesEntry::Velocity,
        &mesh_buffers.vertices,
        &mesh_buffers.cells,
        &mesh_buffers.cell_offsets,
        &p2_dof_buf,
        &a_gpu.row_offsets,
        &a_gpu.col_indices,
        &a_gpu.values,
        mesh.cells.len() as u32,
        n_v as u32,
        n_p as u32,
        dim as u32,
        viscosity as f32,
    )?;
    asm_kernel.assemble(
        StokesEntry::Divergence,
        &mesh_buffers.vertices,
        &mesh_buffers.cells,
        &mesh_buffers.cell_offsets,
        &p2_dof_buf,
        &b_gpu.row_offsets,
        &b_gpu.col_indices,
        &b_gpu.values,
        mesh.cells.len() as u32,
        n_v as u32,
        n_p as u32,
        dim as u32,
        viscosity as f32,
    )?;

    // ─── (4) Read GPU values back, apply BCs, re-upload ─────────────
    let a_vals = read_f32_buf(device, queue, &a_gpu.values, a_gpu.nnz)?;
    let b_vals = read_f32_buf(device, queue, &b_gpu.values, b_gpu.nnz)?;
    for (dst, &src) in a_cpu.values.iter_mut().zip(a_vals.iter()) {
        *dst = src as Real;
    }
    for (dst, &src) in b_cpu.values.iter_mut().zip(b_vals.iter()) {
        *dst = src as Real;
    }

    let pinned_pairs = collect_velocity_pins_pub(bcs, dim);
    let mut f_red = f_velocity.to_vec();
    apply_symmetric_dirichlet_inline(&mut a_cpu, &mut f_red, &pinned_pairs);
    let pinned_set: std::collections::HashSet<usize> =
        pinned_pairs.iter().map(|(d, _)| *d).collect();
    // NOTE: we deliberately do NOT zero BC velocity columns of B. The
    // CPU oracle (`solve_stokes_with_matrices` in stokes.rs) keeps B
    // unmodified and zeroes `Bᵀ q` rows at pinned-velocity indices
    // explicitly inside the Schur action. Zeroing B's BC columns
    // ahead of time would change the value of `g = B · (A⁻¹ · f_red)`
    // when `f_red[pinned] = v_bc ≠ 0` (cavity lid), producing the
    // wrong outer-CG RHS and a converged-but-incorrect solution.

    let a_red_f32: Vec<f32> = a_cpu.values.iter().map(|&v| v as f32).collect();
    let b_red_f32: Vec<f32> = b_cpu.values.iter().map(|&v| v as f32).collect();
    queue.write_buffer(&a_gpu.values, 0, bytemuck::cast_slice(&a_red_f32));
    queue.write_buffer(&b_gpu.values, 0, bytemuck::cast_slice(&b_red_f32));

    let bsr_kernel = BsrAlgebraKernel::new(Arc::clone(device), Arc::clone(queue));
    let bt_gpu = bsr_kernel.transpose(&b_gpu)?;

    // Pin "dead" pressure rows. Matches the CPU oracle: a row is
    // "alive" iff at least one of its non-BC-velocity columns is
    // significant. (BC columns are still present in `b_cpu` — we
    // never zero them, see comment above — so the check must skip
    // them explicitly.)
    let b_inf_norm = b_cpu
        .values
        .iter()
        .copied()
        .map(Real::abs)
        .fold(0.0_f64, Real::max);
    let dead_threshold = (1.0e-12 * b_inf_norm).max(1.0e-30);
    let mut p_pinned_set: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for r in 0..n_p {
        let lo = b_cpu.row_offsets[r];
        let hi = b_cpu.row_offsets[r + 1];
        let mut alive = false;
        for slot in lo..hi {
            let c = b_cpu.col_idx[slot];
            if !pinned_set.contains(&c) && b_cpu.values[slot].abs() > dead_threshold {
                alive = true;
                break;
            }
        }
        if !alive {
            p_pinned_set.insert(r);
        }
    }
    let p_pin = bcs
        .iter()
        .find_map(|b| {
            if let StokesBc::PressureZero { vertex } = b {
                Some(*vertex as usize)
            } else {
                None
            }
        })
        .unwrap_or(0);
    p_pinned_set.insert(p_pin);
    let mut p_pin_mask = vec![0u32; n_p];
    for &r in &p_pinned_set {
        p_pin_mask[r] = 1;
    }
    // Velocity-pin mask — used inside the Schur op to zero `Bᵀ q`
    // entries at pinned-velocity DoFs, matching the CPU oracle.
    let mut v_pin_mask = vec![0u32; n_v];
    for (dof, _) in &pinned_pairs {
        v_pin_mask[*dof] = 1;
    }

    // ─── (5) Uzawa-CG on the GPU ───────────────────────────────────
    let cg_kernel = FemCgGpuKernel::new(Arc::clone(device), Arc::clone(queue));
    let uzawa_t0 = Instant::now();

    let alloc_f32_buf = |label: &str, n: u32| -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n as u64).max(1) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };

    let f_red_f32: Vec<f32> = f_red.iter().map(|&v| v as f32).collect();
    let f_red_buf = alloc_f32_buf("molla::stokes_gpu f_red", n_v as u32);
    queue.write_buffer(&f_red_buf, 0, bytemuck::cast_slice(&f_red_f32));

    let a_for_mv = clone_bsr(&a_gpu);

    // a_inv_f := A⁻¹ · f_red.
    let bsr_a_for_inv = clone_bsr(&a_gpu);
    let cg_a_inv_f = cg_kernel.cg_solve_resident(
        n_v as u32,
        &f_red_buf,
        None,
        cfg.tol_inner,
        cfg.max_inner_iters,
        |x_buf, y_buf| bsr_kernel.matvec(&bsr_a_for_inv, x_buf, y_buf, 1.0, 0.0),
    )?;

    // g := B · (A⁻¹ · f_red).
    let g_buf = alloc_f32_buf("molla::stokes_gpu g", n_p as u32);
    bsr_kernel.matvec(&b_gpu, &cg_a_inv_f.x, &g_buf, 1.0, 0.0)?;
    apply_zero_mask_f32(queue, &g_buf, &p_pin_mask);

    let scratch_v = alloc_f32_buf("molla::stokes_gpu schur_v", n_v as u32);
    let b_for_mv = clone_bsr(&b_gpu);
    let bt_for_mv = clone_bsr(&bt_gpu);

    let schur_op = |q_buf: &wgpu::Buffer, y_buf: &wgpu::Buffer| -> Result<()> {
        // scratch_v = Bᵀ · q.
        bsr_kernel.matvec(&bt_for_mv, q_buf, &scratch_v, 1.0, 0.0)?;
        // Zero pinned-velocity rows of `Bᵀ q` — matches the CPU
        // oracle's `x[dof] = 0` inside its Schur op. Without this,
        // BC velocity contributions leak into the inner CG's RHS,
        // and the inner CG would solve for a `y_v` whose pinned slots
        // are non-zero (the A_red identity rows make `y_v[pinned] =
        // x[pinned]`), poisoning the outer `B · y_v` matvec.
        apply_zero_mask_f32(queue, &scratch_v, &v_pin_mask);
        let inner = cg_kernel.cg_solve_resident(
            n_v as u32,
            &scratch_v,
            None,
            cfg.tol_inner,
            cfg.max_inner_iters,
            |x_buf, y_buf2| bsr_kernel.matvec(&a_for_mv, x_buf, y_buf2, 1.0, 0.0),
        )?;
        bsr_kernel.matvec(&b_for_mv, &inner.x, y_buf, 1.0, 0.0)?;
        // Pin pressure rows of the Schur result to zero (matches the
        // CPU's `yout[r] = q[r]` identity action for pinned rows,
        // which evaluates to 0 since the CG invariant keeps q[r] = 0).
        apply_zero_mask_f32(queue, y_buf, &p_pin_mask);
        Ok(())
    };

    let p_init_buf = alloc_f32_buf("molla::stokes_gpu p_init", n_p as u32);
    queue.write_buffer(&p_init_buf, 0, &vec![0u8; (n_p as usize).max(1) * 4]);

    let p_cg = cg_kernel.cg_solve_resident(
        n_p as u32,
        &g_buf,
        Some(&p_init_buf),
        cfg.tol_outer,
        cfg.max_outer_iters,
        schur_op,
    )?;

    // Recover u = A⁻¹ (f_red − Bᵀ p).
    let bt_p = alloc_f32_buf("molla::stokes_gpu bt_p", n_v as u32);
    bsr_kernel.matvec(&bt_gpu, &p_cg.x, &bt_p, 1.0, 0.0)?;
    let bt_p_host = read_f32_buf(device, queue, &bt_p, n_v as u32)?;
    let mut rhs_u_host: Vec<f32> = (0..n_v).map(|i| f_red_f32[i] - bt_p_host[i]).collect();
    for &(dof, val) in &pinned_pairs {
        rhs_u_host[dof] = val as f32;
    }
    let rhs_u_buf = alloc_f32_buf("molla::stokes_gpu rhs_u", n_v as u32);
    queue.write_buffer(&rhs_u_buf, 0, bytemuck::cast_slice(&rhs_u_host));
    let bsr_a_for_u = clone_bsr(&a_gpu);
    let u_cg = cg_kernel.cg_solve_resident(
        n_v as u32,
        &rhs_u_buf,
        None,
        cfg.tol_inner,
        cfg.max_inner_iters,
        |x_buf, y_buf| bsr_kernel.matvec(&bsr_a_for_u, x_buf, y_buf, 1.0, 0.0),
    )?;
    eprintln!(
        "stokes_gpu diag: inner_recovery iters = {}, residual = {:.3e}; outer iters = {}, residual = {:.3e}; a_inv_f iters = {}, residual = {:.3e}",
        u_cg.iters, u_cg.residual, p_cg.iters, p_cg.residual, cg_a_inv_f.iters, cg_a_inv_f.residual,
    );
    // Diag: residual ‖A·u + Bᵀ·p − f_red‖.
    let au_buf = alloc_f32_buf("diag::Au", n_v as u32);
    bsr_kernel.matvec(&a_for_mv, &u_cg.x, &au_buf, 1.0, 0.0)?;
    let btp_diag_buf = alloc_f32_buf("diag::Btp", n_v as u32);
    bsr_kernel.matvec(&bt_gpu, &p_cg.x, &btp_diag_buf, 1.0, 0.0)?;
    let au_host = read_f32_buf(device, queue, &au_buf, n_v as u32)?;
    let btp_host = read_f32_buf(device, queue, &btp_diag_buf, n_v as u32)?;
    let mut mom_res_sq: f64 = 0.0;
    let mut f_norm_sq: f64 = 0.0;
    for i in 0..n_v {
        let r = au_host[i] + btp_host[i] - f_red_f32[i];
        mom_res_sq += (r as f64) * (r as f64);
        f_norm_sq += (f_red_f32[i] as f64) * (f_red_f32[i] as f64);
    }
    eprintln!(
        "stokes_gpu diag: ‖A·u + Bᵀ·p − f_red‖ = {:.3e}, ‖f_red‖ = {:.3e}",
        mom_res_sq.sqrt(),
        f_norm_sq.sqrt(),
    );
    let _ = uzawa_t0.elapsed();

    // ─── (6) Read back ─────────────────────────────────────────────
    let u_f32 = cg_kernel.read_buffer(&u_cg.x, n_v as u32)?;
    let p_f32 = cg_kernel.read_buffer(&p_cg.x, n_p as u32)?;
    let mut u: Vec<Real> = u_f32.into_iter().map(|v| v as Real).collect();
    let mut p: Vec<Real> = p_f32.into_iter().map(|v| v as Real).collect();
    for &(dof, val) in &pinned_pairs {
        u[dof] = val;
    }
    for &r in &p_pinned_set {
        p[r] = 0.0;
    }

    // Divergence residual.
    let u_f32_final: Vec<f32> = u.iter().map(|&v| v as f32).collect();
    let u_buf = alloc_f32_buf("molla::stokes_gpu u_final", n_v as u32);
    queue.write_buffer(&u_buf, 0, bytemuck::cast_slice(&u_f32_final));
    let bu_buf = alloc_f32_buf("molla::stokes_gpu bu", n_p as u32);
    bsr_kernel.matvec(&b_gpu, &u_buf, &bu_buf, 1.0, 0.0)?;
    let bu = read_f32_buf(device, queue, &bu_buf, n_p as u32)?;
    let mut div_sq: Real = 0.0;
    for (r, &v) in bu.iter().enumerate() {
        if !p_pinned_set.contains(&r) {
            div_sq += (v as Real) * (v as Real);
        }
    }
    let div_residual = div_sq.sqrt();

    Ok(StokesGpuResult {
        u,
        p,
        outer_iters: p_cg.iters,
        div_residual,
    })
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn build_p2_dof_table(mesh: &FemMesh, th: &MixedTaylorHood) -> Vec<u32> {
    let mut table: Vec<u32> = Vec::with_capacity(mesh.cells.len() * 6);
    for cell in &mesh.cells {
        debug_assert!(matches!(cell, FemCell::Tri(_)), "stokes_gpu v1: tri-only");
        for local in 0..6 {
            table.push(th.map.local_to_global(cell, local) as u32);
        }
    }
    table
}

struct MeshBuffers {
    vertices: Arc<wgpu::Buffer>,
    cells: Arc<wgpu::Buffer>,
    cell_offsets: Arc<wgpu::Buffer>,
}

fn upload_mesh(device: &wgpu::Device, queue: &wgpu::Queue, mesh: &FemMesh) -> MeshBuffers {
    let mut vert_bytes: Vec<f32> = Vec::with_capacity(mesh.vertices.len() * 4);
    for v in &mesh.vertices {
        vert_bytes.push(v.x as f32);
        vert_bytes.push(v.y as f32);
        vert_bytes.push(v.z as f32);
        vert_bytes.push(0.0);
    }
    let vertices = create_storage_buffer(
        device,
        queue,
        "molla::stokes_gpu mesh_vertices",
        bytemuck::cast_slice(&vert_bytes),
    );
    let mut cell_offsets: Vec<u32> = Vec::with_capacity(mesh.cells.len() + 1);
    let mut cell_indices: Vec<u32> = Vec::with_capacity(mesh.cells.len() * 3);
    cell_offsets.push(0);
    let mut acc: u32 = 0;
    for cell in &mesh.cells {
        match cell {
            FemCell::Tri(v) => {
                cell_indices.extend_from_slice(v);
                acc += 3;
            }
            FemCell::Tet(v) => {
                cell_indices.extend_from_slice(v);
                acc += 4;
            }
        }
        cell_offsets.push(acc);
    }
    let cells = create_storage_buffer(
        device,
        queue,
        "molla::stokes_gpu mesh_cells",
        bytemuck::cast_slice(&cell_indices),
    );
    let cell_offsets_buf = create_storage_buffer(
        device,
        queue,
        "molla::stokes_gpu mesh_cell_offsets",
        bytemuck::cast_slice(&cell_offsets),
    );
    MeshBuffers {
        vertices: Arc::new(vertices),
        cells: Arc::new(cells),
        cell_offsets: Arc::new(cell_offsets_buf),
    }
}

fn zero_buffer(queue: &wgpu::Queue, buf: &wgpu::Buffer, bytes: usize) {
    let zeros = vec![0u8; bytes.max(16)];
    queue.write_buffer(buf, 0, &zeros);
}

fn create_storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[u8],
) -> wgpu::Buffer {
    let unpadded = data.len() as u64;
    let size = unpadded.max(16).next_multiple_of(4);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        queue.write_buffer(&buf, 0, data);
    }
    buf
}

fn read_f32_buf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    n: u32,
) -> Result<Vec<f32>> {
    let bytes = (n as u64).max(4) * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::stokes_gpu readback staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::stokes_gpu readback enc"),
    });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("stokes_gpu readback poll: {e:?}")))?;
    rx.recv()
        .map_err(|e| Error::Gpu(format!("stokes_gpu readback recv: {e}")))?
        .map_err(|e| Error::Gpu(format!("stokes_gpu readback map: {e:?}")))?;
    let mapped = slice.get_mapped_range();
    let mut v: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    v.truncate(n as usize);
    Ok(v)
}

fn apply_zero_mask_f32(queue: &wgpu::Queue, buf: &wgpu::Buffer, mask: &[u32]) {
    for (i, &m) in mask.iter().enumerate() {
        if m != 0 {
            let zero: f32 = 0.0;
            queue.write_buffer(buf, (i as u64) * 4, bytemuck::bytes_of(&zero));
        }
    }
}

fn apply_symmetric_dirichlet_inline(
    k: &mut BsrMatrix,
    f: &mut [Real],
    pinned: &[(usize, Real)],
) {
    let pin_map: std::collections::HashMap<usize, Real> = pinned.iter().copied().collect();
    for r in 0..k.block_rows {
        if pin_map.contains_key(&r) {
            continue;
        }
        let lo = k.row_offsets[r];
        let hi = k.row_offsets[r + 1];
        for slot in lo..hi {
            let c = k.col_idx[slot];
            if let Some(v) = pin_map.get(&c) {
                f[r] -= k.values[slot] * v;
                k.values[slot] = 0.0;
            }
        }
    }
    for (r, v) in pinned {
        let lo = k.row_offsets[*r];
        let hi = k.row_offsets[*r + 1];
        for slot in lo..hi {
            let c = k.col_idx[slot];
            k.values[slot] = if c == *r { 1.0 } else { 0.0 };
        }
        f[*r] = *v;
    }
}

fn clone_bsr(b: &BsrGpu) -> BsrGpu {
    BsrGpu {
        row_offsets: Arc::clone(&b.row_offsets),
        col_indices: Arc::clone(&b.col_indices),
        values: Arc::clone(&b.values),
        nrows: b.nrows,
        ncols: b.ncols,
        nnz: b.nnz,
    }
}
