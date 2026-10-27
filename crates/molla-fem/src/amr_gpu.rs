//! GPU constrained-assembly for AMR. Mirrors the CPU
//! [`compute_ttkt`](crate::amr_mesh::compute_ttkt): the `Tᵀ K T`
//! reduction of a stiffness matrix against the hanging-node constraint
//! prolongation `T`. The sparse products run on the device via
//! `BsrAlgebraKernel` (transpose + two SpGEMMs) — the numerical
//! hot-path of AMR (the refinement/hanging-node bookkeeping itself is
//! topological and stays on the host).

use crate::math::sparse::BsrMatrix;
use molla_core::Result;
use molla_kernels::{BsrAlgebraKernel, BsrGpu};

/// GPU counterpart of [`compute_ttkt`](crate::amr_mesh::compute_ttkt):
/// `Tᵀ · K · T` for sparse `T` (`n_total × n_free`) and `K`
/// (`n_total × n_total`), returning the reduced `n_free × n_free` matrix.
pub fn compute_ttkt_gpu(
    engine: &molla_compute::Engine,
    t: &BsrMatrix,
    k: &BsrMatrix,
) -> Result<BsrMatrix> {
    let device = engine
        .device()
        .ok_or_else(|| molla_core::Error::Gpu("amr_gpu: engine has no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| molla_core::Error::Gpu("amr_gpu: engine has no queue".into()))?
        .clone();

    let t_gpu = BsrGpu::upload(&device, &queue, t)?;
    let k_gpu = BsrGpu::upload(&device, &queue, k)?;
    let kernel = BsrAlgebraKernel::new(device.clone(), queue.clone());

    // KT = K · T ; Tᵀ ; (Tᵀ)(KT).
    let kt = kernel.spgemm(&k_gpu, &t_gpu)?;
    let tt = kernel.transpose(&t_gpu)?;
    let ttkt = kernel.spgemm(&tt, &kt)?;
    ttkt.read_back(&device, &queue)
}
