//! Stage 37.8 — block-Cholesky factorization for SPD matrices.
//!
//! Public API:
//! - [`cholesky_cpu`] — row-major reference impl. In-place: input is
//!   the SPD matrix `A`; on exit the lower triangle holds `L` with
//!   `L · Lᵀ = A`, the strict upper triangle is zero.
//! - [`TileBlockCholeskyKernel`] (gpu only) — single-workgroup WGSL
//!   factor for `N ≤ 1024`. Multi-workgroup tiling is a v2.
//!
//! v1 caveat: the GPU path uses one workgroup of 64 threads, so it's
//! latency-bound on small matrices and runs out of shared concurrency
//! at large `N`. For `N > 1024` the WGSL kernel returns garbage (no
//! protection in the shader; the host wrapper rejects it explicitly).

use molla_core::{Error, Result};

/// CPU reference Cholesky for an SPD `n × n` row-major matrix. The
/// input slice is overwritten in place: the lower triangle holds `L`
/// (with `L · Lᵀ = A`) and the strict upper triangle is zeroed out.
///
/// **Returns `Error::Internal` if the matrix is not symmetric
/// positive definite** — i.e. if any pivot `Aⱼⱼ − Σ Lⱼₖ²` is
/// negative or smaller than `1e-30`. Earlier this function silently
/// clamped negative pivots to zero and produced a "factor" with
/// wildly wrong off-diagonals; that meant non-SPD inputs returned
/// `Ok(())` with garbage data and no diagnostic. Use
/// [`cholesky_cpu_lenient`] when you genuinely want the approximate
/// (clamped) factor — typically only for incomplete-Cholesky-style
/// preconditioning of nearly-SPD systems.
pub fn cholesky_cpu(a: &mut [f32], n: usize) -> Result<()> {
    cholesky_cpu_impl(a, n, /* lenient = */ false)
}

/// Same as [`cholesky_cpu`] but clamps negative pivots to zero
/// instead of erroring. Use only when you understand that the
/// returned "factor" is **not** a true Cholesky decomposition on
/// non-SPD inputs.
pub fn cholesky_cpu_lenient(a: &mut [f32], n: usize) -> Result<()> {
    cholesky_cpu_impl(a, n, /* lenient = */ true)
}

fn cholesky_cpu_impl(a: &mut [f32], n: usize, lenient: bool) -> Result<()> {
    if a.len() != n * n {
        return Err(Error::Internal(format!(
            "tile_block_cholesky.cholesky_cpu: a.len() = {} but n*n = {}",
            a.len(),
            n * n
        )));
    }
    for j in 0..n {
        // Zero the strict upper triangle entry (i = j, c > j).
        for c in (j + 1)..n {
            a[j * n + c] = 0.0;
        }
        // Diagonal.
        let mut s = a[j * n + j];
        for k in 0..j {
            let v = a[j * n + k];
            s -= v * v;
        }
        if !lenient && s < 1.0e-30 {
            return Err(Error::Internal(format!(
                "tile_block_cholesky.cholesky_cpu: matrix is not SPD — pivot {j} \
                 evaluated to {s:.3e} (< 1e-30); use cholesky_cpu_lenient if you \
                 want clamped pseudo-factor"
            )));
        }
        let ljj = s.max(0.0).sqrt();
        a[j * n + j] = ljj;
        let inv = if ljj > 1.0e-30 { 1.0 / ljj } else { 0.0 };
        // Below-diagonal in column j.
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s * inv;
        }
    }
    Ok(())
}

/// Compute `‖L · Lᵀ − A‖_F` given the original SPD `A` and the
/// row-major lower-triangular factor `L`. Helper for tests.
pub fn frobenius_recon_error(orig: &[f32], factor: &[f32], n: usize) -> f32 {
    let mut err_sq = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f64;
            // Lower-triangular factor: L[i,k] = 0 for k > i.
            let kmax = i.min(j);
            for k in 0..=kmax {
                s += factor[i * n + k] as f64 * factor[j * n + k] as f64;
            }
            let d = s - orig[i * n + j] as f64;
            err_sq += d * d;
        }
    }
    err_sq.sqrt() as f32
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    /// Maximum N supported by the single-workgroup kernel. Bound by
    /// the per-thread O(N²) loop and per-buffer storage budget; 1024
    /// keeps the read traffic under ~4 MiB.
    pub const MAX_N: usize = 1024;

    #[repr(C)]
    #[derive(Copy, Clone, Debug, Pod, Zeroable)]
    struct Uniforms {
        n: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }

    pub struct TileBlockCholeskyKernel {
        pipeline: wgpu::ComputePipeline,
        layout_io: wgpu::BindGroupLayout,
        layout_uniforms: wgpu::BindGroupLayout,
    }

    impl TileBlockCholeskyKernel {
        pub fn new(device: &wgpu::Device) -> Self {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("molla::tile_block_cholesky"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/tile_block_cholesky.wgsl").into(),
                ),
            });

            let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tile_block_cholesky io"),
                entries: &[storage_entry(0, false)],
            });
            let layout_uniforms =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("molla::tile_block_cholesky uniforms"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tile_block_cholesky layout"),
                bind_group_layouts: &[&layout_io, &layout_uniforms],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tile_block_cholesky pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("tile_block_cholesky"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            Self {
                pipeline,
                layout_io,
                layout_uniforms,
            }
        }

        /// Factor `a` in place. On exit, the lower triangle of `a`
        /// holds `L`. Errors if `a.len() != n*n` or `n > MAX_N`.
        pub fn dispatch_blocking(
            &self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            a: &mut [f32],
            n: usize,
        ) -> Result<()> {
            if a.len() != n * n {
                return Err(Error::Gpu(format!(
                    "tile_block_cholesky: a.len() = {} but n*n = {}",
                    a.len(),
                    n * n
                )));
            }
            if n == 0 {
                return Ok(());
            }
            if n > MAX_N {
                return Err(Error::Gpu(format!(
                    "tile_block_cholesky: n = {n} exceeds single-workgroup limit {MAX_N}"
                )));
            }

            let bytes = (a.len() * 4) as u64;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::tile_block_cholesky buf"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(a));

            let uniforms = Uniforms {
                n: n as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let buf_u = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::tile_block_cholesky uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf_u, 0, bytemuck::bytes_of(&uniforms));

            let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::tile_block_cholesky io bg"),
                layout: &self.layout_io,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::tile_block_cholesky uniforms bg"),
                layout: &self.layout_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_u.as_entire_binding(),
                }],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::tile_block_cholesky encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::tile_block_cholesky pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bg_io, &[]);
                pass.set_bind_group(1, &bg_u, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            let stage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::tile_block_cholesky stage"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&buf, 0, &stage, 0, bytes);
            queue.submit(std::iter::once(encoder.finish()));

            let slice = stage.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                sender
                    .send(r.map_err(|e| Error::Gpu(format!("cholesky readback: {e:?}"))))
                    .ok();
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("cholesky poll: {e:?}")))?;
            receiver
                .recv()
                .map_err(|e| Error::Gpu(format!("cholesky recv: {e:?}")))??;
            let data = slice.get_mapped_range();
            let read: &[f32] = bytemuck::cast_slice(&data);
            a.copy_from_slice(read);
            drop(data);
            stage.unmap();
            Ok(())
        }
    }

    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}

#[cfg(feature = "gpu")]
pub use gpu::{MAX_N, TileBlockCholeskyKernel};
