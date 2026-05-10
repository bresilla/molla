//! Stage 36.5 Slice A — GPU-resident Conjugate Gradient solver.
//!
//! Eliminates the per-iteration host roundtrip that
//! [`crate::fem::solver::solve_steady_state_gpu`] paid in v1: `x`,
//! `r`, `p`, and `Ap` stay resident on the GPU across CG iterations.
//! The only data crossing the host boundary per iteration is the
//! dot-product partial-sum readback (≤ a few hundred `f32` values
//! for the workgroup-wise tree reduction).
//!
//! Bookkeeping (`α = rᵀr / pᵀAp`, `β = r_new² / r_old²`, convergence
//! check) runs on the host, but every vector operation is a GPU
//! dispatch. The CG bookkeeping scalars are tiny — host arithmetic
//! is faster than dispatching a 1-thread shader for a single
//! division.
//!
//! See `shaders/fem_vector_ops.wgsl` for the underlying primitives
//! (`vec_axpy`, `vec_axpby`, `vec_dot`).

use std::sync::Arc;

use crate::core::{Error, Result};

const WG_SIZE: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpyUniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpbyUniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f32,
    beta: f32,
    _pad3: f32,
    _pad4: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DotUniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Result of a GPU-resident CG solve. The solution `x` is left on the
/// device — the caller decides whether to read it back (via
/// [`FemCgGpuKernel::read_buffer`]) or keep it for further GPU work.
pub struct CgGpuResult {
    /// Final iterate `x`. Storage buffer length `n` (`f32`).
    pub x: Arc<wgpu::Buffer>,
    /// Final relative residual `‖b - A·x‖ / max(‖b‖, ε)`.
    pub residual: f32,
    /// Number of iterations consumed.
    pub iters: usize,
}

/// GPU-resident CG dispatcher. Owns three small compute pipelines
/// (`vec_axpy`, `vec_axpby`, `vec_dot`) — reuse across many solves.
///
/// The matrix-vector product is supplied as a closure so the same
/// dispatcher works for any GPU-resident `A` (FEM stiffness, mass +
/// stiffness, …). The closure's only contract is "given input `x` +
/// output `y` storage buffers, write `A · x` into `y`." Reuses the
/// existing [`super::FemGpuKernel::matvec`] infrastructure.
pub struct FemCgGpuKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipe_axpy: wgpu::ComputePipeline,
    pipe_axpby: wgpu::ComputePipeline,
    pipe_dot: wgpu::ComputePipeline,

    layout_axpy_io: wgpu::BindGroupLayout,
    layout_axpy_u: wgpu::BindGroupLayout,
    layout_axpby_io: wgpu::BindGroupLayout,
    layout_axpby_u: wgpu::BindGroupLayout,
    layout_dot_io: wgpu::BindGroupLayout,
    layout_dot_u: wgpu::BindGroupLayout,
}

impl FemCgGpuKernel {
    /// Build the kernel against `device` + `queue`. Compiles the
    /// `fem_vector_ops.wgsl` module and creates three pipelines.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::fem_cg::vector_ops"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_vector_ops.wgsl").into(),
            ),
        });

        // ── axpy ───────────────────────────────────────────────────
        let layout_axpy_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_cg::axpy io"),
            entries: &[
                rw_storage(0), // a
                ro_storage(1), // b
            ],
        });
        let layout_axpy_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_cg::axpy uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipe_axpy = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem_cg::vec_axpy"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("molla::fem_cg::axpy layout"),
                    bind_group_layouts: &[&layout_axpy_io, &layout_axpy_u],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("vec_axpy"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── axpby ──────────────────────────────────────────────────
        let layout_axpby_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_cg::axpby io"),
            entries: &[
                rw_storage(0), // dst
                ro_storage(1), // src
            ],
        });
        let layout_axpby_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_cg::axpby uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipe_axpby = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem_cg::vec_axpby"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("molla::fem_cg::axpby layout"),
                    bind_group_layouts: &[&layout_axpby_io, &layout_axpby_u],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("vec_axpby"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── dot ────────────────────────────────────────────────────
        let layout_dot_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_cg::dot io"),
            entries: &[
                ro_storage(0), // a
                ro_storage(1), // b
                rw_storage(2), // partials
            ],
        });
        let layout_dot_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_cg::dot uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipe_dot = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::fem_cg::vec_dot"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("molla::fem_cg::dot layout"),
                    bind_group_layouts: &[&layout_dot_io, &layout_dot_u],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("vec_dot"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipe_axpy,
            pipe_axpby,
            pipe_dot,
            layout_axpy_io,
            layout_axpy_u,
            layout_axpby_io,
            layout_axpby_u,
            layout_dot_io,
            layout_dot_u,
        }
    }

    /// Borrow the device.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Borrow the queue.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Solve `A · x = b` via Conjugate Gradient, keeping `x`, `r`,
    /// `p`, `Ap` resident on the GPU across iterations.
    ///
    /// `matvec(x, y)` is a user-supplied closure: given two storage
    /// buffers, it should write `A · x` into `y`. Typically wraps
    /// [`super::FemGpuKernel::matvec`].
    ///
    /// `b` is the right-hand-side (length `n`, `f32`); `x_init` is an
    /// optional initial guess (zero vector if `None`).
    ///
    /// The returned [`CgGpuResult::x`] buffer remains on the GPU —
    /// the caller decides whether to read it back. `residual` and
    /// `iters` are reported on the host so the caller can match
    /// the [`crate::math::iterative::CgResult`] semantics.
    pub fn cg_solve_resident<F>(
        &self,
        n: u32,
        b: &wgpu::Buffer,
        x_init: Option<&wgpu::Buffer>,
        tol: f32,
        max_iters: usize,
        matvec: F,
    ) -> Result<CgGpuResult>
    where
        F: Fn(&wgpu::Buffer, &wgpu::Buffer) -> Result<()>,
    {
        if n == 0 {
            return Err(Error::Build("cg_solve_resident: n must be > 0".into()));
        }
        let bytes = (n as u64).max(4) * 4;

        // Allocate persistent buffers for x, r, p, Ap (all length n,
        // f32). The dot-product partial-sum buffers are allocated
        // per-call inside [`Self::dispatch_dot_blocking`] (cheap, and
        // keeps the iteration arithmetic obvious).
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let x_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::x"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let r_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::r"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let p_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::p"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let ap_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::Ap"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });

        // Seed x ← x_init (or zero) + r ← b (we'll subtract A·x_init
        // below). Both happen via `copy_buffer_to_buffer` in the same
        // encoder so the host doesn't have to allocate a CPU zero-fill.
        let mut init_enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::init"),
            });
        match x_init {
            Some(buf) => init_enc.copy_buffer_to_buffer(buf, 0, &x_buf, 0, bytes),
            None => {
                // Zero x via write_buffer (cheap; happens once per
                // solve, not per iteration).
                self.queue
                    .write_buffer(&x_buf, 0, &vec![0u8; bytes as usize]);
            }
        }
        // r ← b initially; below we subtract A·x. p, Ap zeroed (only
        // the in-bounds rows matter, but we zero everything to keep
        // the buffers clean for staging readback).
        init_enc.copy_buffer_to_buffer(b, 0, &r_buf, 0, bytes);
        self.queue.submit(std::iter::once(init_enc.finish()));
        // Zero p, Ap so any later "min-bytes" readbacks see clean data.
        self.queue
            .write_buffer(&p_buf, 0, &vec![0u8; bytes as usize]);
        self.queue
            .write_buffer(&ap_buf, 0, &vec![0u8; bytes as usize]);

        // r ← b - A·x: first compute Ap ← A·x_init, then r -= Ap.
        // Reuse the `Ap` buffer as scratch — its CG-loop role doesn't
        // start until iter 0 below.
        matvec(&x_buf, &ap_buf)?;
        // r -= 1.0 · Ap (axpy with alpha = -1).
        self.dispatch_axpy(n, &r_buf, &ap_buf, -1.0)?;

        // ‖b‖ — one dot-reduction over b vs b.
        let bnorm_sq = self.dispatch_dot_blocking(n, b, b)?;
        let bnorm = bnorm_sq.sqrt().max(1.0e-30);

        // rr = ‖r‖² — from r vs r.
        let mut rr = self.dispatch_dot_blocking(n, &r_buf, &r_buf)?;
        let mut residual = rr.sqrt() / bnorm;
        if residual <= tol {
            return Ok(CgGpuResult {
                x: Arc::new(x_buf),
                residual,
                iters: 0,
            });
        }

        // p ← r — one buffer copy (single encoder).
        let mut copy_enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::p<-r"),
            });
        copy_enc.copy_buffer_to_buffer(&r_buf, 0, &p_buf, 0, bytes);
        self.queue.submit(std::iter::once(copy_enc.finish()));

        for it in 0..max_iters {
            // Ap ← A · p.
            matvec(&p_buf, &ap_buf)?;
            // pAp = dot(p, Ap).
            let pap = self.dispatch_dot_blocking(n, &p_buf, &ap_buf)?;
            if pap.abs() < 1.0e-30 {
                // Degenerate search direction — give up gracefully.
                return Ok(CgGpuResult {
                    x: Arc::new(x_buf),
                    residual: rr.sqrt() / bnorm,
                    iters: it,
                });
            }
            let alpha = rr / pap;
            // x += α · p ; r -= α · Ap. Batch into one encoder so we
            // submit + sync once instead of twice.
            self.dispatch_two_axpy(n, &x_buf, &p_buf, alpha, &r_buf, &ap_buf, -alpha)?;
            // rr_new = ‖r‖².
            let rr_new = self.dispatch_dot_blocking(n, &r_buf, &r_buf)?;
            residual = rr_new.sqrt() / bnorm;
            if residual <= tol {
                return Ok(CgGpuResult {
                    x: Arc::new(x_buf),
                    residual,
                    iters: it + 1,
                });
            }
            let beta = rr_new / rr;
            // p ← r + β·p (axpby with alpha=1, beta=β).
            self.dispatch_axpby(n, &p_buf, &r_buf, 1.0, beta)?;
            rr = rr_new;
        }
        Ok(CgGpuResult {
            x: Arc::new(x_buf),
            residual,
            iters: max_iters,
        })
    }

    /// Read an `n`-element f32 storage buffer back into a host
    /// `Vec<f32>`. Used to extract [`CgGpuResult::x`] at solve end.
    pub fn read_buffer(&self, buf: &wgpu::Buffer, n: u32) -> Result<Vec<f32>> {
        let bytes = (n as u64).max(4) * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::readback staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::readback enc"),
            });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.queue.submit(std::iter::once(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("fem_cg readback poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("fem_cg readback recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("fem_cg readback map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let mut out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        out.truncate(n as usize);
        Ok(out)
    }

    // ── Internal dispatchers ──────────────────────────────────────

    fn dispatch_axpy(&self, n: u32, a: &wgpu::Buffer, b: &wgpu::Buffer, alpha: f32) -> Result<()> {
        let u = AxpyUniforms {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            alpha,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        };
        let ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::axpy u"),
            size: core::mem::size_of::<AxpyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&u));
        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_cg::axpy io bg"),
            layout: &self.layout_axpy_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_cg::axpy u bg"),
            layout: &self.layout_axpy_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let groups = n.div_ceil(WG_SIZE).max(1);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::axpy enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::axpy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_axpy);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    /// Batched `(x += α·p ; r += β·Ap)` — encoded into a single
    /// command buffer so we submit once per CG iteration's update
    /// step instead of twice. Each axpy is independent (writes to
    /// disjoint buffers) so they can share the encoder.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_two_axpy(
        &self,
        n: u32,
        a1: &wgpu::Buffer,
        b1: &wgpu::Buffer,
        alpha1: f32,
        a2: &wgpu::Buffer,
        b2: &wgpu::Buffer,
        alpha2: f32,
    ) -> Result<()> {
        let u1 = AxpyUniforms {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            alpha: alpha1,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        };
        let u2 = AxpyUniforms {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            alpha: alpha2,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        };
        let ubuf1 = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::axpy u1"),
            size: core::mem::size_of::<AxpyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ubuf2 = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::axpy u2"),
            size: core::mem::size_of::<AxpyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&ubuf1, 0, bytemuck::bytes_of(&u1));
        self.queue.write_buffer(&ubuf2, 0, bytemuck::bytes_of(&u2));

        let bg_io1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_cg::axpy io1 bg"),
            layout: &self.layout_axpy_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b1.as_entire_binding(),
                },
            ],
        });
        let bg_u1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_cg::axpy u1 bg"),
            layout: &self.layout_axpy_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf1.as_entire_binding(),
            }],
        });
        let bg_io2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_cg::axpy io2 bg"),
            layout: &self.layout_axpy_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b2.as_entire_binding(),
                },
            ],
        });
        let bg_u2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_cg::axpy u2 bg"),
            layout: &self.layout_axpy_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf2.as_entire_binding(),
            }],
        });
        let groups = n.div_ceil(WG_SIZE).max(1);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::axpy2 enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::axpy2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_axpy);
            pass.set_bind_group(0, &bg_io1, &[]);
            pass.set_bind_group(1, &bg_u1, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_bind_group(0, &bg_io2, &[]);
            pass.set_bind_group(1, &bg_u2, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    fn dispatch_axpby(
        &self,
        n: u32,
        dst: &wgpu::Buffer,
        src: &wgpu::Buffer,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let u = AxpbyUniforms {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            alpha,
            beta,
            _pad3: 0.0,
            _pad4: 0.0,
        };
        let ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::axpby u"),
            size: core::mem::size_of::<AxpbyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&u));
        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_cg::axpby io bg"),
            layout: &self.layout_axpby_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_cg::axpby u bg"),
            layout: &self.layout_axpby_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let groups = n.div_ceil(WG_SIZE).max(1);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::axpby enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::axpby"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_axpby);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        Ok(())
    }

    /// `dot(a, b)` with a synchronous host-side reduction over the
    /// (small) per-workgroup partial sums. The only place per-iter
    /// where the GPU → CPU traffic is non-trivial — but `partials`
    /// is at most `⌈n / 256⌉` × 4 B (≤ 1 KiB even at 256 K DoFs).
    fn dispatch_dot_blocking(&self, n: u32, a: &wgpu::Buffer, b: &wgpu::Buffer) -> Result<f32> {
        let n_groups = n.div_ceil(WG_SIZE).max(1);
        let partial_bytes = (n_groups as u64).max(4) * 4;
        let partials = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::dot partials"),
            size: partial_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero the partials so any beyond-n_groups slots (rounded up
        // for min-buffer-size compliance) sum to zero.
        self.queue
            .write_buffer(&partials, 0, &vec![0u8; partial_bytes as usize]);

        let u = DotUniforms {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::dot u"),
            size: core::mem::size_of::<DotUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&u));
        let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_cg::dot io bg"),
            layout: &self.layout_dot_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: partials.as_entire_binding(),
                },
            ],
        });
        let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_cg::dot u bg"),
            layout: &self.layout_dot_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::dot staging"),
            size: partial_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::dot enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::dot"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_dot);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(n_groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&partials, 0, &staging, 0, partial_bytes);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("fem_cg dot poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("fem_cg dot recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("fem_cg dot map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let parts: &[f32] = bytemuck::cast_slice(&mapped);
        // Sum only the actually-used groups (the buffer might be
        // padded up to a 16-byte minimum).
        let mut sum: f32 = 0.0;
        for v in parts.iter().take(n_groups as usize) {
            sum += *v;
        }
        drop(mapped);
        staging.unmap();
        Ok(sum)
    }
}

fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
