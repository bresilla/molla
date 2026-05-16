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
    MixedTaylorHood, StokesBc, assemble_div_block, assemble_velocity_block, assemble_velocity_mass,
    collect_velocity_pins_pub,
};
use crate::gpu_kernels::{
    FemCgGpuKernel, FemNsAdvectKernel, FemNsRhsAssembleKernel, FemStokesGpuKernel, NsRhsMode,
    StokesEntry,
};
use crate::kernels::{BsrAlgebraKernel, BsrGpu};
use crate::math::Real;
use crate::math::sparse::BsrMatrix;

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

fn apply_symmetric_dirichlet_inline(k: &mut BsrMatrix, f: &mut [Real], pinned: &[(usize, Real)]) {
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

// ─── Stage 41 v4 — transient Navier-Stokes (fused RHS) ───────────────
//
// IMEX time-stepping (implicit viscous + explicit advection) on the
// GPU. v4 replaces the per-step host roundtrip from v3 with a fused
// WGSL RHS-axpy kernel (`fem_ns_rhs_assemble.wgsl`). Per step:
//   1. zero the GPU `f_conv` buffer,
//   2. dispatch `FemNsAdvectKernel` to scatter
//      `∫ N_i · (u_old·∇)u_old dx` (optionally with Hughes SUPG)
//      into `f_conv`,
//   3. matvec `m_u = M · u_old` on the GPU,
//   4. dispatch `FemNsRhsAssembleKernel::Step` to compute
//      `rhs = (M/Δt)·u_old + f_ext − f_conv − bc_delta`
//      with the pin-override applied inline — no host readback,
//   5. run the Uzawa-CG solver against the constant
//      `A_eff = M/Δt + νK` (pre-reduced once at setup time),
//   6. matvec `bt_p = Bᵀ · p_new` and dispatch
//      `FemNsRhsAssembleKernel::Recovery` to compute the recovery RHS
//      `rhs_u = rhs_step − bt_p` (with pin override) — again no host
//      readback,
//   7. run the inner CG for `u_new = A_eff⁻¹ rhs_u`,
//   8. copy `u_cg.x`, `p_cg.x` into the persistent `u_buf`, `p_buf`.
//
// All four "axpy + pin" operations that v3 paid for host-side (two per
// step) are now a single GPU dispatch each.
//
// Time-stepping scheme:
//   * **θ = 1** (implicit Euler) on the viscous + pressure terms.
//   * **Explicit** advection — the convective term is evaluated from
//     `u_old` and scattered as an RHS contribution. Stability requires
//     the standard advective CFL bound `Δt · |u| / h ≤ 1`; on the 8×8
//     lid-driven cavity at Re = 100 we use Δt = 0.005 which gives
//     `Δt · u_lid / h = 0.04` — well within the bound. Order 1 in time.
//   * **Quadrature** — the convection RHS uses the **6-point Dunavant
//     rule** (same as the velocity / divergence assembly kernels) rather
//     than a centroid-only sample. Centroid quadrature would be cheaper
//     but would drift the GPU path away from the CPU oracle (which uses
//     6-pt) by ~5e-2 relative on the 8×8 cavity, blowing the parity
//     budget. The kernel is small enough (~12 floats per cell scattered
//     atomically) that the extra quadrature cost is negligible compared
//     to the inner-CG matvecs.
//   * **SUPG** is wired in via the `supg_alpha` knob (0 = plain
//     Galerkin). The default is `0.0` for the parity test so the CPU
//     and GPU paths exercise the same convection term (the CPU oracle
//     uses `solve_navier_stokes_step` which forces plain Galerkin).
//     For convection-dominated runs callers should set `supg_alpha =
//     1.0` to recover the textbook Hughes formula. At small Δt + low
//     Re the local Péclet number is O(1) and SUPG buys no stability
//     win — leave it off.

/// Time integration scheme for [`solve_navier_stokes_gpu`].
///
/// All variants keep stage vectors resident on the device. `ImexEuler`
/// preserves the Stage 41 v4-v6 behavior. The SSP-RK modes apply the
/// same GPU IMEX Euler stage operator multiple times and form the
/// Shu-Osher convex combinations with a small WGSL combine kernel, so no
/// intermediate velocity or pressure stage is read back to the host.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NavierStokesGpuTimeIntegrator {
    /// One first-order IMEX Euler stage per time step.
    ImexEuler,
    /// Shu-Osher SSP-RK(2,2) using two resident IMEX stages.
    SspRk2,
    /// Shu-Osher SSP-RK(3,3) using three resident IMEX stages.
    SspRk3,
}

impl Default for NavierStokesGpuTimeIntegrator {
    fn default() -> Self {
        Self::ImexEuler
    }
}

/// Solver knobs for the transient NS IMEX path.
#[derive(Clone, Debug)]
pub struct NavierStokesGpuConfig {
    /// Time step `Δt`.
    pub dt: Real,
    /// Number of time steps to advance.
    pub n_steps: usize,
    /// Stage 41 v7 — device-resident time integrator. SSP-RK modes keep
    /// all stage velocity/pressure vectors on the GPU and use WGSL
    /// convex-combination dispatches between Euler stages.
    pub time_integrator: NavierStokesGpuTimeIntegrator,
    /// Hughes SUPG strength. `0.0` = plain Galerkin (matches the CPU
    /// `assemble_convection_block` path). `1.0` = textbook formula.
    pub supg_alpha: Real,
    /// Inner Stokes solver tolerances — re-used at every step.
    pub stokes: StokesGpuConfig,
    /// Stage 41 v5 — lazy-CG convergence-check period. `Some(K)` runs
    /// the inner CG in batches of `K` iters between residual reads
    /// (eliminates `K - 1` host syncs per batch). `None` falls back to
    /// the strict-mode CG (sync every iter — the v3/v4 behaviour). At
    /// `tol = 5e-7` the inner CG typically burns 20-50 iters/solve.
    ///
    /// The "sweet spot" is rig-dependent: on a Vulkan adapter where
    /// per-submit cost dominates, `Some(16)` wins; on a true-async-dot
    /// adapter (large GPU + low submit overhead) smaller K (4-8) wins
    /// because the lazy mode otherwise over-runs the true convergence
    /// point by ~K-1 iters. `Some(8)` is the cross-rig default.
    pub cg_check_period: Option<usize>,
    /// Stage 41 v6 — opt-in fully-fused single-pass CG iter. When
    /// `true`, the three inner-CG instances (Schur action `A⁻¹·f`,
    /// `A⁻¹·Bᵀ·q`, and the velocity recovery solve) route through
    /// [`crate::kernels::FemCgGpuKernel::cg_solve_resident_fused`],
    /// which collapses each CG iter from v5's 7 dispatches down to 3
    /// (matvec+atomic-pap, alpha+x/r+atomic-rr, beta+p+save). Same
    /// `cg_check_period` semantics for the residual readback cadence
    /// as v5. Default off; the perf bench enables it.
    pub cg_fused: bool,
}

impl Default for NavierStokesGpuConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            n_steps: 100,
            time_integrator: NavierStokesGpuTimeIntegrator::ImexEuler,
            supg_alpha: 0.0,
            stokes: StokesGpuConfig::default(),
            cg_check_period: Some(8),
            cg_fused: false,
        }
    }
}

/// Result of [`solve_navier_stokes_gpu`].
#[derive(Clone, Debug)]
pub struct NavierStokesGpuResult {
    pub u: Vec<Real>,
    pub p: Vec<Real>,
    /// Outer-CG iterations consumed in the **last** step (the inner
    /// time-stepping iterations vary per step — this is mostly for
    /// quick "did it converge?" checks).
    pub last_outer_iters: usize,
    /// Final divergence residual `‖B·u_final‖` on free pressure rows.
    pub div_residual: Real,
}

/// Transient incompressible Navier-Stokes on the GPU via IMEX time
/// stepping. Replicates the structure of
/// [`crate::fem::stokes::solve_navier_stokes_transient`] but keeps all
/// per-step state (matrices, mesh, working vectors) resident on the
/// device.
pub fn solve_navier_stokes_gpu(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
    u_init: &[Real],
    p_init: &[Real],
    f_velocity: &[Real],
    viscosity: Real,
    bcs: &[StokesBc],
    cfg: &NavierStokesGpuConfig,
    engine: &Engine,
) -> Result<NavierStokesGpuResult> {
    if th.dim != 2 {
        return Err(Error::Build(format!(
            "solve_navier_stokes_gpu: v3 only supports dim = 2, got dim = {}",
            th.dim,
        )));
    }
    for c in &mesh.cells {
        if !matches!(c, FemCell::Tri(_)) {
            return Err(Error::Build(
                "solve_navier_stokes_gpu: v3 only supports tri meshes".into(),
            ));
        }
    }
    if f_velocity.len() != th.ndof_velocity() {
        return Err(Error::Build(format!(
            "solve_navier_stokes_gpu: f_velocity length ({}) != n_v ({})",
            f_velocity.len(),
            th.ndof_velocity(),
        )));
    }
    if u_init.len() != th.ndof_velocity() {
        return Err(Error::Build(format!(
            "solve_navier_stokes_gpu: u_init length ({}) != n_v ({})",
            u_init.len(),
            th.ndof_velocity(),
        )));
    }
    if p_init.len() != th.ndof_pressure() {
        return Err(Error::Build(format!(
            "solve_navier_stokes_gpu: p_init length ({}) != n_p ({})",
            p_init.len(),
            th.ndof_pressure(),
        )));
    }
    if cfg.dt <= 0.0 {
        return Err(Error::Build(format!(
            "solve_navier_stokes_gpu: dt must be > 0 (got {})",
            cfg.dt,
        )));
    }
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("solve_navier_stokes_gpu: engine has no GPU device".into()))?;
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("solve_navier_stokes_gpu: engine has no GPU queue".into()))?;

    let n_v = th.ndof_velocity();
    let n_p = th.ndof_pressure();
    let dim = th.dim;
    let inv_dt: Real = 1.0 / cfg.dt;

    // ─── (1) CPU assembly of K, M, B + build A_eff = M/dt + νK ──────
    //
    // We reuse the CPU assembly path for the constant matrices so the
    // pattern build is identical to the steady-state GPU path (which
    // already runs the GPU assembly and reads back into the same BSR
    // shape). Doing it host-side here is fine because (a) the matrices
    // are constant across all `n_steps` and (b) at the test mesh sizes
    // assembly is microseconds, dominated by the per-step CG work.
    let k_cpu = assemble_velocity_block(mesh, th, viscosity)?;
    let m_cpu = assemble_velocity_mass(mesh, th)?;
    let b_cpu_unreduced = assemble_div_block(mesh, th)?;

    // A_eff_host = M/dt + K. We rebuild via triplets (pattern of
    // M ∪ K). M and K share the same (i, j) sparsity for the P2-vector
    // discretisation, so the result keeps the same pattern as K.
    let a_eff_host = build_a_eff_host(&m_cpu, &k_cpu, inv_dt);

    // BC reduction on A_eff. The reduced matrix lives on the GPU
    // throughout the per-step loop. v4 computes the symmetric-Dirichlet
    // **column-elimination delta** on the GPU (one matvec `bc_delta =
    // A_eff_unreduced · pin_val_full` with pinned rows masked to 0)
    // and feeds it into the fused RHS kernel, so the host never touches
    // the per-step RHS axpy.
    let pinned_pairs = collect_velocity_pins_pub(bcs, dim);
    let pinned_set: std::collections::HashSet<usize> =
        pinned_pairs.iter().map(|(d, _)| *d).collect();
    let mut a_eff_reduced = a_eff_host.clone();
    apply_symmetric_dirichlet_inline(&mut a_eff_reduced, &mut vec![0.0; n_v], &pinned_pairs);

    // Dead-pressure detection + pressure pin — same logic as
    // `solve_stokes_gpu`.
    let b_inf_norm = b_cpu_unreduced
        .values
        .iter()
        .copied()
        .map(Real::abs)
        .fold(0.0_f64, Real::max);
    let dead_threshold = (1.0e-12 * b_inf_norm).max(1.0e-30);
    let mut p_pinned_set: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for r in 0..n_p {
        let lo = b_cpu_unreduced.row_offsets[r];
        let hi = b_cpu_unreduced.row_offsets[r + 1];
        let mut alive = false;
        for slot in lo..hi {
            let c = b_cpu_unreduced.col_idx[slot];
            if !pinned_set.contains(&c) && b_cpu_unreduced.values[slot].abs() > dead_threshold {
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
    let mut v_pin_mask = vec![0u32; n_v];
    for (dof, _) in &pinned_pairs {
        v_pin_mask[*dof] = 1;
    }

    // ─── (2) Upload all persistent buffers ──────────────────────────
    let mesh_buffers = upload_mesh(device, queue, mesh);
    let p2_dof_table = build_p2_dof_table(mesh, th);
    let p2_dof_buf = create_storage_buffer(
        device,
        queue,
        "molla::fem::ns_gpu p2_dof_table",
        bytemuck::cast_slice(&p2_dof_table),
    );

    // Mass matrix (used only for `(M/dt)·u_old` matvec each step).
    let m_gpu = BsrGpu::upload(device, queue, &m_cpu)?;
    // BC-reduced effective LHS — never written after setup.
    let a_eff_gpu = BsrGpu::upload(device, queue, &a_eff_reduced)?;
    // Unreduced A_eff — used ONCE at setup to derive the per-step BC
    // column-elimination correction on the GPU. Released at the end of
    // setup (the binding scope drops with the function).
    let a_eff_unreduced_gpu = BsrGpu::upload(device, queue, &a_eff_host)?;
    // Divergence matrix — unmodified, see comment in `solve_stokes_gpu`.
    let b_gpu = BsrGpu::upload(device, queue, &b_cpu_unreduced)?;

    let bsr_kernel = BsrAlgebraKernel::new(Arc::clone(device), Arc::clone(queue));
    let bt_gpu = bsr_kernel.transpose(&b_gpu)?;
    let cg_kernel = FemCgGpuKernel::new(Arc::clone(device), Arc::clone(queue));
    let advect_kernel = FemNsAdvectKernel::new(Arc::clone(device), Arc::clone(queue));
    let rhs_kernel = FemNsRhsAssembleKernel::new(Arc::clone(device), Arc::clone(queue));

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

    // Persistent per-step buffers.
    let u_buf = alloc_f32_buf("molla::ns_gpu u_old", n_v as u32);
    let u_init_f32: Vec<f32> = u_init.iter().map(|&v| v as f32).collect();
    queue.write_buffer(&u_buf, 0, bytemuck::cast_slice(&u_init_f32));
    let p_buf = alloc_f32_buf("molla::ns_gpu p", n_p as u32);
    let p_init_f32: Vec<f32> = p_init.iter().map(|&v| v as f32).collect();
    queue.write_buffer(&p_buf, 0, bytemuck::cast_slice(&p_init_f32));

    let f_conv_buf = alloc_f32_buf("molla::ns_gpu f_conv", n_v as u32);
    let m_u_buf = alloc_f32_buf("molla::ns_gpu m_u", n_v as u32);
    let rhs_buf = alloc_f32_buf("molla::ns_gpu rhs", n_v as u32);
    // Separate destination for the recovery RHS — WGSL forbids binding
    // the same buffer as both read and read_write in one dispatch, so
    // we ping-pong between `rhs_buf` (step output / recovery input) and
    // `rhs_u_buf` (recovery output, fed into the u-CG).
    let rhs_u_buf = alloc_f32_buf("molla::ns_gpu rhs_u", n_v as u32);

    // Persistent constants for the fused RHS kernel — uploaded once.
    //   * f_ext   : external body force (constant per step).
    //   * pin_val : full-size velocity vector with pinned values at
    //               pinned DoFs and zero elsewhere. Used both to
    //               compute `bc_delta` and to override pinned RHS
    //               entries inside the kernel.
    //   * pin_mask: u32 per DoF, 1 if pinned.
    //   * bc_delta: `A_eff_unreduced · pin_val_full`, with pinned rows
    //               zeroed. Computed once at setup with a single
    //               BSR matvec + apply-zero-mask — no host axpy.
    let f_ext_f32: Vec<f32> = f_velocity.iter().map(|&v| v as f32).collect();
    let f_ext_buf = alloc_f32_buf("molla::ns_gpu f_ext", n_v as u32);
    queue.write_buffer(&f_ext_buf, 0, bytemuck::cast_slice(&f_ext_f32));

    let mut pin_val_full = vec![0.0f32; n_v];
    for &(dof, val) in &pinned_pairs {
        pin_val_full[dof] = val as f32;
    }
    let pin_val_buf = alloc_f32_buf("molla::ns_gpu pin_val", n_v as u32);
    queue.write_buffer(&pin_val_buf, 0, bytemuck::cast_slice(&pin_val_full));
    let pin_mask_buf = alloc_f32_buf("molla::ns_gpu pin_mask", n_v as u32); // u32 storage buffer
    queue.write_buffer(&pin_mask_buf, 0, bytemuck::cast_slice(&v_pin_mask));
    let p_pin_mask_buf = alloc_f32_buf("molla::ns_gpu p_pin_mask", n_p as u32); // u32 storage buffer
    queue.write_buffer(&p_pin_mask_buf, 0, bytemuck::cast_slice(&p_pin_mask));
    let p_zero_val_buf = alloc_f32_buf("molla::ns_gpu p_zero_pin_val", n_p as u32);
    let p_zero_val = vec![0.0f32; n_p];
    queue.write_buffer(&p_zero_val_buf, 0, bytemuck::cast_slice(&p_zero_val));

    // bc_delta = A_eff_unreduced · pin_val_full, with pinned rows
    // zeroed. Mirrors the host helper `compute_bc_rhs_delta` from v3
    // but stays entirely on the GPU.
    let bc_delta_buf = alloc_f32_buf("molla::ns_gpu bc_delta", n_v as u32);
    let pin_for_bc = clone_bsr(&a_eff_unreduced_gpu);
    bsr_kernel.matvec(&pin_for_bc, &pin_val_buf, &bc_delta_buf, 1.0, 0.0)?;
    // (One-time setup — host zero-mask is fine here; the per-step loop
    // uses the GPU `zero_mask` kernel.)
    apply_zero_mask_f32(queue, &bc_delta_buf, &v_pin_mask);

    let mut last_outer_iters = 0usize;

    // Reusable Schur buffers.
    let scratch_v = alloc_f32_buf("molla::ns_gpu schur_v", n_v as u32);
    let g_buf = alloc_f32_buf("molla::ns_gpu g", n_p as u32);
    let bt_p_buf = alloc_f32_buf("molla::ns_gpu bt_p", n_v as u32);

    let a_for_mv_step = clone_bsr(&a_eff_gpu);
    let a_for_inv_step = clone_bsr(&a_eff_gpu);
    let a_for_u_step = clone_bsr(&a_eff_gpu);
    let b_for_mv_step = clone_bsr(&b_gpu);
    let bt_for_mv_step = clone_bsr(&bt_gpu);

    // Stage 41 v5 — when lazy CG is active, pre-build the matvec
    // dispatches for the three inner-CG instances (a_inv_f, schur
    // inner, u recovery) so the inner loop can chain them into a
    // single compute pass instead of paying a fresh wgpu submit per
    // matvec per iter. The CG result buffers (`x_buf`) are workspace-
    // allocated inside `cg_solve_resident_lazy_prepared`, so we
    // can't bake them into the prepared matvec — instead the CG
    // workspace exposes `p_buf`/`ap_buf` and we plumb those in by
    // re-creating the prepared matvec at the start of each solve.
    // Allocation cost is one-time-per-solve (~2 bind groups) — fine.

    let total_t0 = Instant::now();
    let u_stage_a = alloc_f32_buf("molla::ns_gpu u_stage_a", n_v as u32);
    let u_stage_b = alloc_f32_buf("molla::ns_gpu u_stage_b", n_v as u32);
    let p_stage_a = alloc_f32_buf("molla::ns_gpu p_stage_a", n_p as u32);
    let p_stage_b = alloc_f32_buf("molla::ns_gpu p_stage_b", n_p as u32);

    {
        let mut advance_euler_stage = |src_u_buf: &wgpu::Buffer,
                                       src_p_buf: &wgpu::Buffer,
                                       dst_u_buf: &wgpu::Buffer,
                                       dst_p_buf: &wgpu::Buffer|
         -> Result<usize> {
            // (a) zero f_conv on the GPU.
            zero_buffer(queue, &f_conv_buf, (n_v as usize) * 4);

            // (b) dispatch convection assembly.
            advect_kernel.assemble(
                &mesh_buffers.vertices,
                &mesh_buffers.cells,
                &mesh_buffers.cell_offsets,
                &p2_dof_buf,
                src_u_buf,
                &f_conv_buf,
                mesh.cells.len() as u32,
                n_v as u32,
                dim as u32,
                viscosity as f32,
                cfg.supg_alpha as f32,
            )?;

            // (c) m_u = M · u_old (per-step matvec).
            bsr_kernel.matvec(&m_gpu, src_u_buf, &m_u_buf, 1.0, 0.0)?;

            // (d) fused RHS axpy + pin override on the GPU:
            //     rhs[i] = inv_dt * m_u[i] + f_ext[i] − f_conv[i] − bc_delta[i]
            //     rhs[dof] = pin_val[dof]   for pinned DoFs.
            // Single dispatch — no host roundtrip.
            rhs_kernel.dispatch(
                NsRhsMode::Step,
                &m_u_buf,
                &f_ext_buf,
                &f_conv_buf,
                &bc_delta_buf,
                &pin_mask_buf,
                &pin_val_buf,
                &rhs_buf,
                n_v as u32,
                inv_dt as f32,
            )?;

            // (e) Uzawa-CG solve against (A_eff, B). Same Schur structure
            // as `solve_stokes_gpu` — only the RHS changes per step.
            //
            // v5: if `cfg.cg_check_period` is `Some(K)`, drive every inner
            // CG through the prepared-matvec lazy variant. The prepared
            // matvec caches the BSR bind groups so each inner iter records
            // matvec + dot + axpy + dot + axpby into ONE compute pass —
            // one submit per iter instead of two (matvec + post-matvec),
            // and `K - 1` per-iter dot-readback syncs collapse to one read
            // per batch of `K`. The prepared matvec costs one extra
            // bind-group + uniform alloc per inner CG instantiation, which
            // amortises across the ~30-100 inner iters that each solve
            // burns.
            let check_period = cfg.cg_check_period;
            let cg_a_inv_f = if cfg.cg_fused {
                // Stage 41 v6 — single-pass fused CG.
                cg_kernel.cg_solve_resident_fused(
                    n_v as u32,
                    &a_for_inv_step,
                    &rhs_buf,
                    None,
                    cfg.stokes.tol_inner,
                    cfg.stokes.max_inner_iters,
                    check_period,
                )?
            } else if let Some(k) = check_period {
                cg_kernel.cg_solve_resident_lazy_prepared(
                    n_v as u32,
                    &rhs_buf,
                    None,
                    cfg.stokes.tol_inner,
                    cfg.stokes.max_inner_iters,
                    Some(k),
                    |p_buf_l, ap_buf_l| {
                        let prepared = bsr_kernel.prepare_matvec(
                            &a_for_inv_step,
                            p_buf_l,
                            ap_buf_l,
                            1.0,
                            0.0,
                        )?;
                        let bsr_ref = &bsr_kernel;
                        Ok(move |pass: &mut wgpu::ComputePass| {
                            bsr_ref.matvec_into_pass(pass, &prepared);
                        })
                    },
                )?
            } else {
                cg_kernel.cg_solve_resident(
                    n_v as u32,
                    &rhs_buf,
                    None,
                    cfg.stokes.tol_inner,
                    cfg.stokes.max_inner_iters,
                    |x_buf, y_buf| bsr_kernel.matvec(&a_for_inv_step, x_buf, y_buf, 1.0, 0.0),
                )?
            };

            bsr_kernel.matvec(&b_gpu, &cg_a_inv_f.x, &g_buf, 1.0, 0.0)?;
            apply_zero_mask_f32(queue, &g_buf, &p_pin_mask);

            let schur_op = |q_buf: &wgpu::Buffer, y_buf: &wgpu::Buffer| -> Result<()> {
                bsr_kernel.matvec(&bt_for_mv_step, q_buf, &scratch_v, 1.0, 0.0)?;
                apply_zero_mask_f32(queue, &scratch_v, &v_pin_mask);
                let inner = if cfg.cg_fused {
                    cg_kernel.cg_solve_resident_fused(
                        n_v as u32,
                        &a_for_mv_step,
                        &scratch_v,
                        None,
                        cfg.stokes.tol_inner,
                        cfg.stokes.max_inner_iters,
                        check_period,
                    )?
                } else if let Some(k) = check_period {
                    cg_kernel.cg_solve_resident_lazy_prepared(
                        n_v as u32,
                        &scratch_v,
                        None,
                        cfg.stokes.tol_inner,
                        cfg.stokes.max_inner_iters,
                        Some(k),
                        |p_buf_l, ap_buf_l| {
                            let prepared = bsr_kernel.prepare_matvec(
                                &a_for_mv_step,
                                p_buf_l,
                                ap_buf_l,
                                1.0,
                                0.0,
                            )?;
                            let bsr_ref = &bsr_kernel;
                            Ok(move |pass: &mut wgpu::ComputePass| {
                                bsr_ref.matvec_into_pass(pass, &prepared);
                            })
                        },
                    )?
                } else {
                    cg_kernel.cg_solve_resident(
                        n_v as u32,
                        &scratch_v,
                        None,
                        cfg.stokes.tol_inner,
                        cfg.stokes.max_inner_iters,
                        |x_buf, y_buf2| bsr_kernel.matvec(&a_for_mv_step, x_buf, y_buf2, 1.0, 0.0),
                    )?
                };
                bsr_kernel.matvec(&b_for_mv_step, &inner.x, y_buf, 1.0, 0.0)?;
                apply_zero_mask_f32(queue, y_buf, &p_pin_mask);
                Ok(())
            };
            // NOTE: per-Schur-iter `apply_zero_mask_f32` is still host-side.
            // A v4 prototype that swapped each call for a GPU `zero_mask`
            // entry-point dispatch was *slower* at 16×16 because the
            // per-dispatch uniform buffer + bind-group allocation overhead
            // dominated at the actual pin counts (the host write_buffer
            // path effectively batches via the wgpu staging belt). Cached
            // bind groups would close the gap, but bind-group caching
            // crosses into the CG inner-loop refactor that v5 keeps inside
            // the inner CG only.

            // Warm-start p from the previous step. The OUTER pressure CG
            // stays on the strict-mode kernel because its matvec is the
            // Schur op (itself a multi-submit composition: Bᵀ matvec +
            // inner CG + B matvec + zero masks), and there's no
            // straightforward way to fuse that into the lazy CG's single
            // compute pass. The brief defers full pipelined CG and
            // explicitly scopes lazy CG to the inner Schur-action solves.
            let p_cg = cg_kernel.cg_solve_resident(
                n_p as u32,
                &g_buf,
                Some(src_p_buf),
                cfg.stokes.tol_outer,
                cfg.stokes.max_outer_iters,
                schur_op,
            )?;
            last_outer_iters = p_cg.iters;

            // Recover u_new = A_eff⁻¹ (rhs − Bᵀ p). The fused recovery
            // kernel does `rhs_u = rhs − bt_p` + pin override into the
            // separate `rhs_u_buf` (WGSL forbids read+read_write aliasing
            // on the same buffer within a single dispatch). No host
            // roundtrip.
            bsr_kernel.matvec(&bt_gpu, &p_cg.x, &bt_p_buf, 1.0, 0.0)?;
            rhs_kernel.dispatch(
                NsRhsMode::Recovery,
                &rhs_buf,
                &bt_p_buf,
                &bc_delta_buf, // unused source slot — kernel ignores src_c.
                &bc_delta_buf, // unused source slot — kernel ignores src_d.
                &pin_mask_buf,
                &pin_val_buf,
                &rhs_u_buf,
                n_v as u32,
                inv_dt as f32,
            )?;
            let u_cg = if cfg.cg_fused {
                cg_kernel.cg_solve_resident_fused(
                    n_v as u32,
                    &a_for_u_step,
                    &rhs_u_buf,
                    Some(src_u_buf), // warm-start from u_old
                    cfg.stokes.tol_inner,
                    cfg.stokes.max_inner_iters,
                    check_period,
                )?
            } else if let Some(k) = check_period {
                cg_kernel.cg_solve_resident_lazy_prepared(
                    n_v as u32,
                    &rhs_u_buf,
                    Some(src_u_buf), // warm-start from u_old
                    cfg.stokes.tol_inner,
                    cfg.stokes.max_inner_iters,
                    Some(k),
                    |p_buf_l, ap_buf_l| {
                        let prepared = bsr_kernel.prepare_matvec(
                            &a_for_u_step,
                            p_buf_l,
                            ap_buf_l,
                            1.0,
                            0.0,
                        )?;
                        let bsr_ref = &bsr_kernel;
                        Ok(move |pass: &mut wgpu::ComputePass| {
                            bsr_ref.matvec_into_pass(pass, &prepared);
                        })
                    },
                )?
            } else {
                cg_kernel.cg_solve_resident(
                    n_v as u32,
                    &rhs_u_buf,
                    Some(src_u_buf), // warm-start from u_old
                    cfg.stokes.tol_inner,
                    cfg.stokes.max_inner_iters,
                    |x_buf, y_buf| bsr_kernel.matvec(&a_for_u_step, x_buf, y_buf, 1.0, 0.0),
                )?
            };

            // Copy u_cg.x and p_cg.x back into the persistent u_buf, p_buf
            // (next step reads from them).
            let mut copy_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::ns_gpu state-copy"),
            });
            copy_enc.copy_buffer_to_buffer(&u_cg.x, 0, dst_u_buf, 0, (n_v as u64) * 4);
            copy_enc.copy_buffer_to_buffer(&p_cg.x, 0, dst_p_buf, 0, (n_p as u64) * 4);
            queue.submit(std::iter::once(copy_enc.finish()));
            Ok(last_outer_iters)
        };

        let copy_state = |src_u: &wgpu::Buffer,
                          src_p: &wgpu::Buffer,
                          dst_u: &wgpu::Buffer,
                          dst_p: &wgpu::Buffer| {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::ns_gpu ssp_state-copy"),
            });
            enc.copy_buffer_to_buffer(src_u, 0, dst_u, 0, (n_v as u64) * 4);
            enc.copy_buffer_to_buffer(src_p, 0, dst_p, 0, (n_p as u64) * 4);
            queue.submit(std::iter::once(enc.finish()));
        };

        for _step in 0..cfg.n_steps {
            match cfg.time_integrator {
                NavierStokesGpuTimeIntegrator::ImexEuler => {
                    advance_euler_stage(&u_buf, &p_buf, &u_buf, &p_buf)?;
                }
                NavierStokesGpuTimeIntegrator::SspRk2 => {
                    // y1 = E(y_n); y2_tmp = E(y1).
                    advance_euler_stage(&u_buf, &p_buf, &u_stage_a, &p_stage_a)?;
                    advance_euler_stage(&u_stage_a, &p_stage_a, &u_stage_b, &p_stage_b)?;
                    // y_{n+1} = 1/2 y_n + 1/2 y2_tmp, with both convex
                    // combinations done by WGSL so the stage vectors never
                    // leave the device.
                    rhs_kernel.combine(
                        &u_buf,
                        &u_stage_b,
                        &pin_mask_buf,
                        &pin_val_buf,
                        &u_stage_a,
                        n_v as u32,
                        0.5,
                        0.5,
                    )?;
                    rhs_kernel.combine(
                        &p_buf,
                        &p_stage_b,
                        &p_pin_mask_buf,
                        &p_zero_val_buf,
                        &p_stage_a,
                        n_p as u32,
                        0.5,
                        0.5,
                    )?;
                    copy_state(&u_stage_a, &p_stage_a, &u_buf, &p_buf);
                }
                NavierStokesGpuTimeIntegrator::SspRk3 => {
                    // y1 = E(y_n); y2_tmp = E(y1).
                    advance_euler_stage(&u_buf, &p_buf, &u_stage_a, &p_stage_a)?;
                    advance_euler_stage(&u_stage_a, &p_stage_a, &u_stage_b, &p_stage_b)?;
                    // y2 = 3/4 y_n + 1/4 y2_tmp.
                    rhs_kernel.combine(
                        &u_buf,
                        &u_stage_b,
                        &pin_mask_buf,
                        &pin_val_buf,
                        &u_stage_a,
                        n_v as u32,
                        0.75,
                        0.25,
                    )?;
                    rhs_kernel.combine(
                        &p_buf,
                        &p_stage_b,
                        &p_pin_mask_buf,
                        &p_zero_val_buf,
                        &p_stage_a,
                        n_p as u32,
                        0.75,
                        0.25,
                    )?;
                    // y3_tmp = E(y2); y_{n+1} = 1/3 y_n + 2/3 y3_tmp.
                    advance_euler_stage(&u_stage_a, &p_stage_a, &u_stage_b, &p_stage_b)?;
                    rhs_kernel.combine(
                        &u_buf,
                        &u_stage_b,
                        &pin_mask_buf,
                        &pin_val_buf,
                        &u_stage_a,
                        n_v as u32,
                        1.0 / 3.0,
                        2.0 / 3.0,
                    )?;
                    rhs_kernel.combine(
                        &p_buf,
                        &p_stage_b,
                        &p_pin_mask_buf,
                        &p_zero_val_buf,
                        &p_stage_a,
                        n_p as u32,
                        1.0 / 3.0,
                        2.0 / 3.0,
                    )?;
                    copy_state(&u_stage_a, &p_stage_a, &u_buf, &p_buf);
                }
            }
        }
    }
    let total_dt = total_t0.elapsed().as_secs_f64() * 1000.0;
    let _ = total_dt;

    // ─── (3) Read back ──────────────────────────────────────────────
    let u_f32 = cg_kernel.read_buffer(&u_buf, n_v as u32)?;
    let p_f32 = cg_kernel.read_buffer(&p_buf, n_p as u32)?;
    let mut u: Vec<Real> = u_f32.into_iter().map(|v| v as Real).collect();
    let mut p: Vec<Real> = p_f32.into_iter().map(|v| v as Real).collect();
    for &(dof, val) in &pinned_pairs {
        u[dof] = val;
    }
    for &r in &p_pinned_set {
        p[r] = 0.0;
    }

    // Divergence residual.
    let u_final_f32: Vec<f32> = u.iter().map(|&v| v as f32).collect();
    let u_final_buf = alloc_f32_buf("molla::ns_gpu u_final", n_v as u32);
    queue.write_buffer(&u_final_buf, 0, bytemuck::cast_slice(&u_final_f32));
    let bu_buf = alloc_f32_buf("molla::ns_gpu bu", n_p as u32);
    bsr_kernel.matvec(&b_gpu, &u_final_buf, &bu_buf, 1.0, 0.0)?;
    let bu = read_f32_buf(device, queue, &bu_buf, n_p as u32)?;
    let mut div_sq: Real = 0.0;
    for (r, &v) in bu.iter().enumerate() {
        if !p_pinned_set.contains(&r) {
            div_sq += (v as Real) * (v as Real);
        }
    }
    let div_residual = div_sq.sqrt();

    Ok(NavierStokesGpuResult {
        u,
        p,
        last_outer_iters,
        div_residual,
    })
}

/// `A_eff = inv_dt · M + K` host-side. Both matrices live on the same
/// `(i, j)` pattern for the P2-vector discretisation so the result
/// inherits that pattern. We rebuild via triplets to be robust against
/// any future divergence between the M and K column orders.
fn build_a_eff_host(m: &BsrMatrix, k: &BsrMatrix, inv_dt: Real) -> BsrMatrix {
    debug_assert_eq!(m.b, 1);
    debug_assert_eq!(k.b, 1);
    debug_assert_eq!(m.nrows(), k.nrows());
    debug_assert_eq!(m.ncols(), k.ncols());
    let mut triplets: Vec<(usize, usize, Real)> =
        Vec::with_capacity(m.values.len() + k.values.len());
    for r in 0..m.block_rows {
        for slot in m.row_offsets[r]..m.row_offsets[r + 1] {
            let c = m.col_idx[slot];
            triplets.push((r, c, inv_dt * m.values[slot]));
        }
    }
    for r in 0..k.block_rows {
        for slot in k.row_offsets[r]..k.row_offsets[r + 1] {
            let c = k.col_idx[slot];
            triplets.push((r, c, k.values[slot]));
        }
    }
    BsrMatrix::from_scalar_triplets(m.block_rows, m.block_cols, &triplets)
        .expect("build_a_eff_host: triplet rebuild")
}

// `compute_bc_rhs_delta` (host) removed in Stage 41 v4 — the BC
// column-elimination delta is now computed on the GPU via a single
// `BsrAlgebraKernel::matvec(A_eff_unreduced, pin_val_full)` followed by
// `apply_zero_mask_f32(.., v_pin_mask)`, see the setup section of
// `solve_navier_stokes_gpu`.
