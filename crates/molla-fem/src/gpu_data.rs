//! Stage 36.5 — GPU mirrors of [`super::mesh::FemMesh`] + assembled
//! sparse [`crate::math::sparse::BsrMatrix`].
//!
//! # Precision tradeoff
//!
//! The CPU FEM stack runs `f64` end-to-end (mass + stiffness assembly,
//! BC reduction, CG iterates). The GPU mirror DROPS to `f32` because
//! WGSL has no `f64` type and adding the Vulkan-only path would split
//! the codebase. For the v1 heat-equation problem at 16³ tet
//! resolution the f32 round-off is ≪ the discretisation error (we
//! observe ~1e-4 absolute on the assembled CSR values vs the f64
//! oracle, well inside the documented 1e-3 solution tolerance).
//!
//! Higher-precision GPU FEM (Vulkan `shaderFloat64` + WGSL extension
//! support) is out of scope for v1; revisit if a consumer ever needs
//! the precision.
//!
//! # Layout
//!
//! Flat CSR storage on the GPU:
//!   - `csr_row_offsets[i]` = first slot of row `i` in `csr_col_indices`
//!     and `csr_values`. Length `nrows + 1`. `u32`.
//!   - `csr_col_indices[slot]` = column index of the `slot`-th nonzero.
//!     `u32`.
//!   - `csr_values[slot]` = matrix value at the `slot`-th nonzero. `f32`.
//!
//! Mesh storage on the GPU:
//!   - `mesh_vertices[i]` = `vec3<f32>` (16 bytes including 4-byte pad
//!     so the WGSL `array<vec3<f32>>` storage layout is well-defined).
//!   - `mesh_cell_offsets[k]` = first vertex of cell `k` in
//!     `mesh_cells`. Length `cell_count + 1`. `u32`.
//!   - `mesh_cells[off]` = vertex index. Triangles contribute 3
//!     entries, tets 4. `u32`.
//!
//! The `cell_offsets` indirection lets a single buffer hold both Tris
//! and Tets without padding. v1 always builds one of (`tri_grid_2d`,
//! `tet_grid_3d`) so the cell-type is uniform per mesh — but the
//! layout supports mixed meshes for free.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use crate::core::{Error, Result};
use crate::math::sparse::BsrMatrix;

use super::mesh::{FemCell, FemMesh};

/// GPU mirrors of the FEM mesh + assembled CSR matrix.
///
/// All buffers are owned (`Arc<wgpu::Buffer>`) so the dispatcher can
/// keep them alive across multiple matvec dispatches without
/// re-uploading.
///
/// `csr_*` fields are uploaded as `f32` (see module docstring for the
/// precision rationale). `csr_nrows` is the linear-system size; for
/// `FunctionSpace::P1Nodal` this equals `mesh.vert_count()`.
pub struct FemGpuData {
    pub mesh_vertices: Arc<wgpu::Buffer>,
    pub mesh_cells: Arc<wgpu::Buffer>,
    pub mesh_cell_offsets: Arc<wgpu::Buffer>,
    pub csr_row_offsets: Arc<wgpu::Buffer>,
    pub csr_col_indices: Arc<wgpu::Buffer>,
    pub csr_values: Arc<wgpu::Buffer>,

    /// Number of mesh vertices (= linear-system size for P1Nodal).
    pub vertex_count: u32,
    /// Number of cells in the mesh.
    pub cell_count: u32,
    /// Number of f32 values stored in `csr_values` / column-index slots.
    pub csr_nnz: u32,
    /// Linear-system row count (matches `vertex_count` for P1Nodal).
    pub csr_nrows: u32,
}

impl FemGpuData {
    /// Build a `FemGpuData` from a CPU `mesh` and an assembled CSR
    /// matrix (`BsrMatrix` with `B = 1`). The matrix values are
    /// converted from `f64` → `f32` on upload.
    ///
    /// `device` and `queue` are typically obtained from
    /// [`crate::compute::Engine::device`] / [`crate::compute::Engine::queue`].
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &FemMesh,
        csr: &BsrMatrix,
    ) -> Result<Self> {
        if csr.b != 1 {
            return Err(Error::Build(format!(
                "FemGpuData::upload: only BsrMatrix B = 1 supported in v1, got B = {}",
                csr.b
            )));
        }
        // The scalar-CSR (`B = 1`) layout supports both scalar-field
        // (`nrows == vert_count`) and vector-field (`nrows ==
        // vert_count * dim`) systems with the same buffer layout —
        // each scalar DoF gets one CSR row regardless of how the
        // vector components were grouped at the visitor level. Stage
        // 41 elasticity uses `P1Vector { dim = 3 }` which lands on
        // `nrows == vert_count * 3`; the heat path uses `P1Nodal`
        // with `nrows == vert_count`. Accept any positive integer
        // multiple of `vert_count` (the vector dim is derived per-
        // dispatch from a uniform — see
        // `crate::fem::elasticity_gpu::FemElasticityGpuKernel`).
        let nrows = csr.nrows();
        let vc = mesh.vert_count();
        if vc == 0 || nrows == 0 || nrows % vc != 0 {
            return Err(Error::Build(format!(
                "FemGpuData::upload: CSR rows ({nrows}) is not a positive integer \
                 multiple of mesh vertices ({vc})",
            )));
        }

        // ── Mesh vertex buffer (vec3<f32>, padded to 16 bytes) ──────
        // WGSL `array<vec3<f32>>` storage layout aligns each element
        // to 16 bytes; mirror that here so a flat `bytemuck::cast`
        // matches.
        let mut vert_bytes: Vec<f32> = Vec::with_capacity(mesh.vertices.len() * 4);
        for v in &mesh.vertices {
            vert_bytes.push(v.x as f32);
            vert_bytes.push(v.y as f32);
            vert_bytes.push(v.z as f32);
            vert_bytes.push(0.0);
        }
        let mesh_vertices = create_storage_buffer(
            device,
            queue,
            "molla::fem::gpu_data mesh_vertices",
            bytemuck::cast_slice(&vert_bytes),
        );

        // ── Mesh cell buffers (flat with offsets) ───────────────────
        let mut cell_offsets: Vec<u32> = Vec::with_capacity(mesh.cells.len() + 1);
        let mut cell_indices: Vec<u32> = Vec::with_capacity(mesh.cells.len() * 4);
        let mut acc: u32 = 0;
        cell_offsets.push(0);
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
        let mesh_cells = create_storage_buffer(
            device,
            queue,
            "molla::fem::gpu_data mesh_cells",
            bytemuck::cast_slice(&cell_indices),
        );
        let mesh_cell_offsets = create_storage_buffer(
            device,
            queue,
            "molla::fem::gpu_data mesh_cell_offsets",
            bytemuck::cast_slice(&cell_offsets),
        );

        // ── CSR storage (f32 values, u32 row offsets + col indices) ─
        let row_offsets_u32: Vec<u32> = csr.row_offsets.iter().map(|&v| v as u32).collect();
        let col_idx_u32: Vec<u32> = csr.col_idx.iter().map(|&v| v as u32).collect();
        let values_f32: Vec<f32> = csr.values.iter().map(|&v| v as f32).collect();

        let csr_row_offsets = create_storage_buffer(
            device,
            queue,
            "molla::fem::gpu_data csr_row_offsets",
            bytemuck::cast_slice(&row_offsets_u32),
        );
        let csr_col_indices = create_storage_buffer(
            device,
            queue,
            "molla::fem::gpu_data csr_col_indices",
            bytemuck::cast_slice(&col_idx_u32),
        );
        let csr_values = create_storage_buffer(
            device,
            queue,
            "molla::fem::gpu_data csr_values",
            bytemuck::cast_slice(&values_f32),
        );

        Ok(Self {
            mesh_vertices: Arc::new(mesh_vertices),
            mesh_cells: Arc::new(mesh_cells),
            mesh_cell_offsets: Arc::new(mesh_cell_offsets),
            csr_row_offsets: Arc::new(csr_row_offsets),
            csr_col_indices: Arc::new(csr_col_indices),
            csr_values: Arc::new(csr_values),
            vertex_count: mesh.vert_count() as u32,
            cell_count: mesh.cells.len() as u32,
            csr_nnz: values_f32.len() as u32,
            csr_nrows: csr.nrows() as u32,
        })
    }

    /// Read the CSR `values` buffer back into a host `Vec<f32>`. Used
    /// by parity tests; not on the per-iteration solve path.
    pub fn read_csr_values(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<f32>> {
        let bytes = (self.csr_nnz as u64) * 4;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem::gpu_data csr_values readback"),
            size: bytes.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::fem::gpu_data csr_values readback enc"),
        });
        encoder.copy_buffer_to_buffer(&self.csr_values, 0, &staging, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("fem gpu_data poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("fem gpu_data recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("fem gpu_data map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        Ok(v)
    }

    /// Zero out the CSR `values` buffer in place. Used by
    /// [`crate::kernels::FemGpuKernel::assemble`] before scattering
    /// fresh element contributions so repeated assembles don't
    /// accumulate.
    pub fn clear_csr_values(&self, queue: &wgpu::Queue) {
        let bytes = (self.csr_nnz as usize) * 4;
        let zeros = vec![0u8; bytes];
        queue.write_buffer(&self.csr_values, 0, &zeros);
    }
}

fn create_storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[u8],
) -> wgpu::Buffer {
    // wgpu requires storage buffers to be at least 16-byte aligned in
    // size — pad short uploads (e.g. a 4-element row_offsets for a
    // tiny mesh) up to a multiple of 16.
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
