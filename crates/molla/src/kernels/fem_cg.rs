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

/// GPU-resident CG dispatcher. Owns four small compute pipelines
/// (`vec_axpy`, `vec_axpby`, `vec_dot`, `vec_apply_diag_inv`) — reuse
/// across many solves. The fourth (`vec_apply_diag_inv`) is only
/// touched by [`Self::pcg_solve_resident`] but kept on the kernel so
/// shader compilation is amortised across PCG calls.
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
    pipe_apply_diag_inv: wgpu::ComputePipeline,

    // Stage 41 v5 — lazy CG primitives. Compiled once with the rest of
    // the kernel so successive lazy solves don't pay the shader-compile
    // cost.
    pipe_dot_finalize: wgpu::ComputePipeline,
    pipe_cg_alpha: wgpu::ComputePipeline,
    pipe_cg_beta: wgpu::ComputePipeline,
    pipe_axpy_s: wgpu::ComputePipeline,
    pipe_axpby_s: wgpu::ComputePipeline,
    // v5 — fused scalar-divide + vector update kernels. These collapse
    // the early-prototype (cg_alpha + 2× axpy_s) and (cg_beta + axpby_s)
    // into a single dispatch each by doing the per-thread divide
    // inline. Removes 3 dispatches per inner-CG iter, which dominated
    // wall-clock at 8×8 / 16×16 (small-n dispatch latency >> arithmetic).
    pipe_cg_axpy_xr: wgpu::ComputePipeline,
    pipe_cg_axpby_p: wgpu::ComputePipeline,

    layout_axpy_io: wgpu::BindGroupLayout,
    layout_axpy_u: wgpu::BindGroupLayout,
    layout_axpby_io: wgpu::BindGroupLayout,
    layout_axpby_u: wgpu::BindGroupLayout,
    layout_dot_io: wgpu::BindGroupLayout,
    layout_dot_u: wgpu::BindGroupLayout,
    layout_diag_io: wgpu::BindGroupLayout,
    layout_diag_u: wgpu::BindGroupLayout,

    layout_finalize_io: wgpu::BindGroupLayout,
    layout_finalize_u: wgpu::BindGroupLayout,
    layout_alpha_io: wgpu::BindGroupLayout,
    layout_beta_io: wgpu::BindGroupLayout,
    layout_axpys_io: wgpu::BindGroupLayout,
    layout_axpbys_io: wgpu::BindGroupLayout,
    layout_xrx_io: wgpu::BindGroupLayout,
    layout_bp_io: wgpu::BindGroupLayout,
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

        // ── apply_diag_inv ─────────────────────────────────────────
        // `dst = src .* inv_diag` — used by PCG to apply the Jacobi
        // preconditioner once per iter. Reuses the axpby uniform
        // layout (only `n` matters; alpha/beta unused) so the host
        // doesn't have to ship a fresh uniform per iter.
        let layout_diag_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_cg::diag io"),
            entries: &[
                rw_storage(0), // dst
                ro_storage(1), // src
                ro_storage(2), // inv_diag
            ],
        });
        let layout_diag_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::fem_cg::diag uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipe_apply_diag_inv =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem_cg::vec_apply_diag_inv"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("molla::fem_cg::diag layout"),
                        bind_group_layouts: &[&layout_diag_io, &layout_diag_u],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: Some("vec_apply_diag_inv"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // ═══════════════════════════════════════════════════════════
        // Stage 41 v5 — lazy CG kernels.
        //
        // Five extra pipelines + six layouts. Compiled here so the
        // standard CG path pays the cost too (one-time, ~ms) but the
        // lazy path doesn't pay an extra hit on first call.

        // ── dot_finalize ─────────────────────────────────────────────
        // io = (partials read, out 1-elem rw); u = (n_partials).
        let layout_finalize_io = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::fem_cg::finalize io"),
                entries: &[ro_storage(0), rw_storage(1)],
            },
        );
        let layout_finalize_u = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::fem_cg::finalize uniforms"),
                entries: &[uniform_entry(0)],
            },
        );
        let pipe_dot_finalize =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem_cg::dot_finalize"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("molla::fem_cg::finalize layout"),
                        bind_group_layouts: &[&layout_finalize_io, &layout_finalize_u],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: Some("dot_finalize"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // ── cg_alpha ─────────────────────────────────────────────────
        // io = (rr_old ro, pap ro, alpha rw, neg_alpha rw); no uniforms.
        let layout_alpha_io = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::fem_cg::cg_alpha io"),
                entries: &[ro_storage(0), ro_storage(1), rw_storage(2), rw_storage(3)],
            },
        );
        let pipe_cg_alpha = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem_cg::cg_alpha"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("molla::fem_cg::cg_alpha layout"),
                        bind_group_layouts: &[&layout_alpha_io],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: Some("cg_alpha"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );

        // ── cg_beta ──────────────────────────────────────────────────
        // io = (rr_new ro, rr_old rw, beta rw); no uniforms.
        let layout_beta_io = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::fem_cg::cg_beta io"),
                entries: &[ro_storage(0), rw_storage(1), rw_storage(2)],
            },
        );
        let pipe_cg_beta = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem_cg::cg_beta"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("molla::fem_cg::cg_beta layout"),
                        bind_group_layouts: &[&layout_beta_io],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: Some("cg_beta"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );

        // ── vec_axpy_s ───────────────────────────────────────────────
        // io = (a rw, b ro, scalar ro); u = (n).
        let layout_axpys_io = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::fem_cg::axpy_s io"),
                entries: &[rw_storage(0), ro_storage(1), ro_storage(2)],
            },
        );
        let pipe_axpy_s = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem_cg::vec_axpy_s"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("molla::fem_cg::axpy_s layout"),
                        bind_group_layouts: &[&layout_axpys_io, &layout_axpy_u],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: Some("vec_axpy_s"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );

        // ── vec_axpby_s ──────────────────────────────────────────────
        // io = (dst rw, src ro, scalar ro); u = (n).
        let layout_axpbys_io = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::fem_cg::axpby_s io"),
                entries: &[rw_storage(0), ro_storage(1), ro_storage(2)],
            },
        );
        let pipe_axpby_s = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem_cg::vec_axpby_s"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("molla::fem_cg::axpby_s layout"),
                        bind_group_layouts: &[&layout_axpbys_io, &layout_axpby_u],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: Some("vec_axpby_s"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );

        // ── cg_axpy_xr ───────────────────────────────────────────────
        // io = (x rw, r rw, p ro, ap ro, rr_old ro, pap ro); u = (n).
        let layout_xrx_io = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::fem_cg::cg_axpy_xr io"),
                entries: &[
                    rw_storage(0),
                    rw_storage(1),
                    ro_storage(2),
                    ro_storage(3),
                    ro_storage(4),
                    ro_storage(5),
                ],
            },
        );
        let pipe_cg_axpy_xr = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem_cg::cg_axpy_xr"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("molla::fem_cg::cg_axpy_xr layout"),
                        bind_group_layouts: &[&layout_xrx_io, &layout_axpy_u],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: Some("cg_axpy_xr"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );

        // ── cg_axpby_p ───────────────────────────────────────────────
        // io = (p rw, r ro, rr_new ro, rr_old rw); u = (n).
        let layout_bp_io = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::fem_cg::cg_axpby_p io"),
                entries: &[rw_storage(0), ro_storage(1), ro_storage(2), rw_storage(3)],
            },
        );
        let pipe_cg_axpby_p = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::fem_cg::cg_axpby_p"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("molla::fem_cg::cg_axpby_p layout"),
                        bind_group_layouts: &[&layout_bp_io, &layout_axpy_u],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: Some("cg_axpby_p"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );

        Self {
            device,
            queue,
            pipe_axpy,
            pipe_axpby,
            pipe_dot,
            pipe_apply_diag_inv,
            pipe_dot_finalize,
            pipe_cg_alpha,
            pipe_cg_beta,
            pipe_axpy_s,
            pipe_axpby_s,
            pipe_cg_axpy_xr,
            pipe_cg_axpby_p,
            layout_axpy_io,
            layout_axpy_u,
            layout_axpby_io,
            layout_axpby_u,
            layout_dot_io,
            layout_dot_u,
            layout_diag_io,
            layout_diag_u,
            layout_finalize_io,
            layout_finalize_u,
            layout_alpha_io,
            layout_beta_io,
            layout_axpys_io,
            layout_axpbys_io,
            layout_xrx_io,
            layout_bp_io,
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
        // f32). The dot-product partial-sum buffer + staging are
        // sized once per solve and cached in the workspace below —
        // earlier they were re-allocated per iteration which dominated
        // wall-clock at small N.
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

        // Build the cached workspace — uniform buffers + bind groups
        // for every dispatch slot in the inner loop, plus the
        // dot-product partials/staging. Hoisting these out of the
        // loop is the Slice D perf finish.
        let ws = CgWorkspace::new(self, n, &x_buf, &r_buf, &p_buf, &ap_buf, b);

        // Seed x ← x_init (or zero) + r ← b.
        let mut init_enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::init"),
            });
        match x_init {
            Some(buf) => init_enc.copy_buffer_to_buffer(buf, 0, &x_buf, 0, bytes),
            None => {
                self.queue
                    .write_buffer(&x_buf, 0, &vec![0u8; bytes as usize]);
            }
        }
        init_enc.copy_buffer_to_buffer(b, 0, &r_buf, 0, bytes);
        self.queue.submit(std::iter::once(init_enc.finish()));
        // Zero p, Ap so any later "min-bytes" readbacks see clean data.
        self.queue
            .write_buffer(&p_buf, 0, &vec![0u8; bytes as usize]);
        self.queue
            .write_buffer(&ap_buf, 0, &vec![0u8; bytes as usize]);

        // r ← b - A·x: first compute Ap ← A·x_init, then r -= Ap.
        matvec(&x_buf, &ap_buf)?;
        // r -= 1.0 · Ap — uses the cached `axpy_r_ap` slot with α = -1.
        ws.update_axpy(&self.queue, AxpySlot::RminusAlphaAp, -1.0);
        ws.dispatch_single_axpy(self, AxpySlot::RminusAlphaAp);

        // ‖b‖ — one dot-reduction over b vs b (cached "bb" slot).
        let bnorm_sq = ws.dispatch_dot_blocking(self, DotSlot::Bb)?;
        let bnorm = bnorm_sq.sqrt().max(1.0e-30);

        // rr = ‖r‖² — cached "rr" dot slot.
        let mut rr = ws.dispatch_dot_blocking(self, DotSlot::Rr)?;
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
            // pAp = dot(p, Ap) — cached "pAp" slot.
            let pap = ws.dispatch_dot_blocking(self, DotSlot::PAp)?;
            if pap.abs() < 1.0e-30 {
                return Ok(CgGpuResult {
                    x: Arc::new(x_buf),
                    residual: rr.sqrt() / bnorm,
                    iters: it,
                });
            }
            let alpha = rr / pap;
            // x += α · p ; r -= α · Ap. Both axpy uniforms get refreshed
            // via `write_buffer` (no allocation) and both dispatches go
            // into a single command encoder.
            ws.update_axpy(&self.queue, AxpySlot::XplusAlphaP, alpha);
            ws.update_axpy(&self.queue, AxpySlot::RminusAlphaAp, -alpha);
            ws.dispatch_two_axpy(self, AxpySlot::XplusAlphaP, AxpySlot::RminusAlphaAp);
            // rr_new = ‖r‖².
            let rr_new = ws.dispatch_dot_blocking(self, DotSlot::Rr)?;
            residual = rr_new.sqrt() / bnorm;
            if residual <= tol {
                return Ok(CgGpuResult {
                    x: Arc::new(x_buf),
                    residual,
                    iters: it + 1,
                });
            }
            let beta = rr_new / rr;
            // p ← r + β·p — cached axpby slot, just refresh β.
            ws.update_axpby(&self.queue, 1.0, beta);
            ws.dispatch_axpby(self);
            rr = rr_new;
        }
        Ok(CgGpuResult {
            x: Arc::new(x_buf),
            residual,
            iters: max_iters,
        })
    }

    /// Stage 41 v5 — async-dot CG. Same math as
    /// [`Self::cg_solve_resident`], but the per-iter `pap`/`rr` dot
    /// products stay GPU-resident as 1-element scalar buffers, and
    /// alpha/beta are derived on the GPU via two fused scalar+vector
    /// kernels (`cg_axpy_xr`, `cg_axpby_p`). The host only reads the
    /// residual back every `check_period` iters — at sizes where the
    /// inner-CG dominates (NS time stepping, hundreds of iters/step)
    /// the reduction in `map_async + poll` syncs from `2k` per solve
    /// to `2k / check_period` shaves off the dominant pipeline-stall
    /// cost.
    ///
    /// `check_period` defaults to **8** when callers pass `None`: at
    /// `tol = 5e-7` the inner CG burns ~20-40 iters/solve, so checking
    /// every 8 iters means we read back 3-5 times per solve instead of
    /// once per iter. The trade-off is that lazy mode may run up to
    /// `check_period - 1` extra iters past the true convergence point
    /// (the residual is checked only at batch boundaries).
    ///
    /// **Matvec contract**: the closure variant `cg_solve_resident_lazy`
    /// takes a closure that submits its own command encoder per call.
    /// At small-N (8×8/16×16) the per-submit overhead dominates wall
    /// clock — prefer [`Self::cg_solve_resident_lazy_prepared`] when
    /// the matvec is a fixed `(A, x, y)` BSR product, since that
    /// variant chains the matvec into the same compute pass as the
    /// CG body (one submit per iter total).
    ///
    /// **Contract w.r.t. the strict-CG path**: when iters ≪ check_period
    /// (already-converged seeds) the two paths produce bit-identical
    /// results modulo f32 atomic-add nondeterminism in the dot reduction
    /// (which both paths share). When iters > check_period, lazy may
    /// over-run the strict path by ≤ check_period - 1 iters.
    pub fn cg_solve_resident_lazy<F>(
        &self,
        n: u32,
        b: &wgpu::Buffer,
        x_init: Option<&wgpu::Buffer>,
        tol: f32,
        max_iters: usize,
        check_period: Option<usize>,
        matvec: F,
    ) -> Result<CgGpuResult>
    where
        F: Fn(&wgpu::Buffer, &wgpu::Buffer) -> Result<()>,
    {
        if n == 0 {
            return Err(Error::Build("cg_solve_resident_lazy: n must be > 0".into()));
        }
        let check_period = check_period.unwrap_or(8).max(1);
        let bytes = (n as u64).max(4) * 4;

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let x_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::lazy::x"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let r_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::lazy::r"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let p_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::lazy::p"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let ap_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::lazy::Ap"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });

        let ws = LazyCgWorkspace::new(self, n, &x_buf, &r_buf, &p_buf, &ap_buf, b);

        // Seed x ← x_init (or zero) + r ← b.
        let mut init_enc =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("molla::fem_cg::lazy::init"),
                });
        match x_init {
            Some(buf) => init_enc.copy_buffer_to_buffer(buf, 0, &x_buf, 0, bytes),
            None => {
                self.queue
                    .write_buffer(&x_buf, 0, &vec![0u8; bytes as usize]);
            }
        }
        init_enc.copy_buffer_to_buffer(b, 0, &r_buf, 0, bytes);
        self.queue.submit(std::iter::once(init_enc.finish()));
        self.queue
            .write_buffer(&p_buf, 0, &vec![0u8; bytes as usize]);
        self.queue
            .write_buffer(&ap_buf, 0, &vec![0u8; bytes as usize]);

        // r ← b - A·x_init via existing uniform-axpy (same as strict CG).
        matvec(&x_buf, &ap_buf)?;
        ws.update_uniform_axpy(&self.queue, LazyUniformAxpySlot::RminusAlphaAp, -1.0);
        ws.dispatch_uniform_axpy(self, LazyUniformAxpySlot::RminusAlphaAp);

        // ‖b‖² and initial ‖r‖² — both via the blocking dot for the
        // setup phase (these are one-shot, the latency doesn't matter).
        let bnorm_sq = ws.dispatch_dot_blocking(self, LazyDotSlot::Bb)?;
        let bnorm = bnorm_sq.sqrt().max(1.0e-30);
        let rr0 = ws.dispatch_dot_blocking(self, LazyDotSlot::Rr)?;
        let mut residual = rr0.sqrt() / bnorm;
        if residual <= tol {
            return Ok(CgGpuResult {
                x: Arc::new(x_buf),
                residual,
                iters: 0,
            });
        }

        // p ← r (initial).
        let mut copy_enc =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("molla::fem_cg::lazy::p<-r"),
                });
        copy_enc.copy_buffer_to_buffer(&r_buf, 0, &p_buf, 0, bytes);
        self.queue.submit(std::iter::once(copy_enc.finish()));

        // Seed rr_old_buf on the GPU with the value of rr0.
        ws.write_scalar(&self.queue, &ws.scalar_rr_old, rr0);

        let mut iters_done = 0usize;
        while iters_done < max_iters {
            let batch = (max_iters - iters_done).min(check_period);
            // Run `batch` lazy iters. Each iter does the matvec via
            // the caller-supplied closure (which submits its own
            // command encoder — BSR matvec lives in a separate kernel)
            // followed by a single fused "post-matvec" submit that
            // chains the dot → alpha → axpy → dot → beta → axpby into
            // one compute pass. No host syncs until the residual read
            // at the end of the batch.
            for _ in 0..batch {
                matvec(&p_buf, &ap_buf)?;
                ws.dispatch_post_matvec_step_fused(self);
            }
            iters_done += batch;
            // Read back the most recent rr_new for the convergence check.
            // After `cg_beta`, rr_new still holds the latest squared
            // residual and rr_old has been overwritten with the same
            // value — both paths to the residual norm are valid.
            let last_rr = ws.read_scalar_blocking(self, &ws.scalar_rr_new)?;
            residual = last_rr.sqrt() / bnorm;
            if residual <= tol {
                return Ok(CgGpuResult {
                    x: Arc::new(x_buf),
                    residual,
                    iters: iters_done,
                });
            }
        }
        Ok(CgGpuResult {
            x: Arc::new(x_buf),
            residual,
            iters: iters_done,
        })
    }

    /// Stage 41 v5 — single-submit-per-iter lazy CG. Companion to
    /// [`Self::cg_solve_resident_lazy`] that takes the matvec as a
    /// **compute-pass recorder** instead of a self-submitting closure.
    /// The recorder appends the matvec dispatch into the SAME compute
    /// pass that hosts the dot / fused-axpy dispatches, so a full CG
    /// iter is ONE submit instead of two.
    ///
    /// Wallclock impact on the test rig (16×16 cavity, 100 steps,
    /// release, tol 5e-7): 23.7s (v4) → 8.3s (v5 lazy, closure
    /// matvec) → 4.2s (v5 lazy, prepared matvec). The prepared path
    /// dominates at small-N because the per-submit launch latency on
    /// Vulkan + headless wgpu is ~50-100 µs and the inner CG fires
    /// ~30 iters per Schur step.
    ///
    /// `record_matvec(pass)` is invoked per iter to record the
    /// `Ap = A · p` dispatch. The caller must NOT call
    /// `set_pipeline`/`set_bind_group`/`dispatch_workgroups` for any
    /// state that conflicts with the CG body — but appending fresh
    /// `(set_pipeline + set_bind_group + dispatch_workgroups)` is fine
    /// (the wgpu compute pass tracks state per-recorder-call).
    pub fn cg_solve_resident_lazy_prepared<B, F>(
        &self,
        n: u32,
        b: &wgpu::Buffer,
        x_init: Option<&wgpu::Buffer>,
        tol: f32,
        max_iters: usize,
        check_period: Option<usize>,
        build_matvec: B,
    ) -> Result<CgGpuResult>
    where
        B: FnOnce(&wgpu::Buffer, &wgpu::Buffer) -> Result<F>,
        F: for<'a> FnMut(&mut wgpu::ComputePass<'a>),
    {
        if n == 0 {
            return Err(Error::Build(
                "cg_solve_resident_lazy_prepared: n must be > 0".into(),
            ));
        }
        let check_period = check_period.unwrap_or(8).max(1);
        let bytes = (n as u64).max(4) * 4;

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let x_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::lazy_prep::x"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let r_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::lazy_prep::r"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let p_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::lazy_prep::p"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let ap_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::lazy_prep::Ap"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });

        // The prepared matvec needs to bake in `p` and `ap` as fixed
        // input/output buffers (we own them, the caller doesn't). Hand
        // those to the caller's build closure so they can build a
        // `PreparedMatvec` with the right resources.
        //
        // NOTE for the setup phase (`r ← b − A·x_init`): we need a
        // SECOND matvec view `x → Ap`, not `p → Ap`. We dodge that by
        // copying x_init → p_buf first, running the recorder against
        // p_buf, then resetting p_buf to zero before the main loop.
        // Bit-equivalent to the closure-variant setup but lets us reuse
        // the single `record_matvec` recorder.
        let mut record_matvec = build_matvec(&p_buf, &ap_buf)?;

        let ws = LazyCgWorkspace::new(self, n, &x_buf, &r_buf, &p_buf, &ap_buf, b);

        // Seed x ← x_init (or zero), r ← b. Also copy x_init into
        // p_buf so the prepared matvec (p → Ap) computes `A · x_init`
        // — the value we need for the initial residual.
        let mut init_enc =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("molla::fem_cg::lazy_prep::init"),
                });
        match x_init {
            Some(buf) => {
                init_enc.copy_buffer_to_buffer(buf, 0, &x_buf, 0, bytes);
                init_enc.copy_buffer_to_buffer(buf, 0, &p_buf, 0, bytes);
            }
            None => {
                self.queue
                    .write_buffer(&x_buf, 0, &vec![0u8; bytes as usize]);
                self.queue
                    .write_buffer(&p_buf, 0, &vec![0u8; bytes as usize]);
            }
        }
        init_enc.copy_buffer_to_buffer(b, 0, &r_buf, 0, bytes);
        self.queue.submit(std::iter::once(init_enc.finish()));
        self.queue
            .write_buffer(&ap_buf, 0, &vec![0u8; bytes as usize]);

        // Setup matvec (p_buf holds x_init): Ap ← A · x_init.
        let mut setup_enc =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("molla::fem_cg::lazy_prep::setup mv"),
                });
        {
            let mut pass = setup_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::lazy_prep::setup mv"),
                timestamp_writes: None,
            });
            record_matvec(&mut pass);
        }
        self.queue.submit(std::iter::once(setup_enc.finish()));
        ws.update_uniform_axpy(&self.queue, LazyUniformAxpySlot::RminusAlphaAp, -1.0);
        ws.dispatch_uniform_axpy(self, LazyUniformAxpySlot::RminusAlphaAp);

        let bnorm_sq = ws.dispatch_dot_blocking(self, LazyDotSlot::Bb)?;
        let bnorm = bnorm_sq.sqrt().max(1.0e-30);
        let rr0 = ws.dispatch_dot_blocking(self, LazyDotSlot::Rr)?;
        let mut residual = rr0.sqrt() / bnorm;
        if residual <= tol {
            return Ok(CgGpuResult {
                x: Arc::new(x_buf),
                residual,
                iters: 0,
            });
        }

        let mut copy_enc =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("molla::fem_cg::lazy_prep::p<-r"),
                });
        copy_enc.copy_buffer_to_buffer(&r_buf, 0, &p_buf, 0, bytes);
        self.queue.submit(std::iter::once(copy_enc.finish()));

        ws.write_scalar(&self.queue, &ws.scalar_rr_old, rr0);

        let mut iters_done = 0usize;
        while iters_done < max_iters {
            let batch = (max_iters - iters_done).min(check_period);
            // Run `batch` iters as ONE mega-compute-pass inside ONE
            // command encoder + ONE submit. All `batch × (matvec +
            // post-matvec)` dispatches share a single compute pass
            // — wgpu's intra-pass barrier inserter handles the
            // per-dispatch RAW hazards (partials → finalize →
            // axpy_xr → next-iter matvec → ...) automatically; the
            // `partials` buffer is freely reused across iters
            // because each dot/finalize pair fully consumes the
            // previous contents before the next dot's write begins.
            //
            // Empirically (on the test rig's wgpu+Vulkan path) this
            // is faster than one-pass-per-iter (which would let
            // wgpu insert explicit inter-pass barriers — extra Vk
            // barrier commands per iter).
            let mut enc = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("molla::fem_cg::lazy_prep::batch"),
                },
            );
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::fem_cg::lazy_prep::batch pass"),
                    timestamp_writes: None,
                });
                for _ in 0..batch {
                    record_matvec(&mut pass);
                    ws.record_post_matvec(self, &mut pass);
                }
            }
            self.queue.submit(std::iter::once(enc.finish()));
            iters_done += batch;
            let last_rr = ws.read_scalar_blocking(self, &ws.scalar_rr_new)?;
            residual = last_rr.sqrt() / bnorm;
            if residual <= tol {
                return Ok(CgGpuResult {
                    x: Arc::new(x_buf),
                    residual,
                    iters: iters_done,
                });
            }
        }
        Ok(CgGpuResult {
            x: Arc::new(x_buf),
            residual,
            iters: iters_done,
        })
    }

    /// Solve `A · x = b` via Jacobi-preconditioned Conjugate Gradient,
    /// keeping `x`, `r`, `p`, `Ap`, `z` resident on the GPU across
    /// iterations. Sibling of [`Self::cg_solve_resident`] — same
    /// matvec contract, same workspace pattern, with one extra
    /// per-iter `apply_diag_inv` dispatch and one extra resident
    /// buffer (`z = M⁻¹ · r`).
    ///
    /// `inv_diag` is a length-`n` f32 buffer holding `1 / diag(A)`
    /// (zero for any null-space rows — pass-through). Compute it once
    /// per solve via [`super::FemGpuKernel::extract_diagonal`] +
    /// host-side reciprocal (see
    /// [`crate::fem::solver::solve_steady_state_gpu_resident_pcg`]
    /// for the canonical wiring).
    ///
    /// Convergence is still tested on `‖r‖`, not `‖z‖` — same contract
    /// as [`crate::math::iterative::pcg_solve`].
    #[allow(clippy::too_many_arguments)]
    pub fn pcg_solve_resident<F>(
        &self,
        n: u32,
        b: &wgpu::Buffer,
        x_init: Option<&wgpu::Buffer>,
        inv_diag: &wgpu::Buffer,
        tol: f32,
        max_iters: usize,
        matvec: F,
    ) -> Result<CgGpuResult>
    where
        F: Fn(&wgpu::Buffer, &wgpu::Buffer) -> Result<()>,
    {
        if n == 0 {
            return Err(Error::Build("pcg_solve_resident: n must be > 0".into()));
        }
        let bytes = (n as u64).max(4) * 4;

        // Allocate persistent buffers for x, r, p, Ap, z (all length n,
        // f32). Mirror `cg_solve_resident`'s layout exactly + one new
        // `z` slot for the preconditioned residual.
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let x_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::pcg::x"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let r_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::pcg::r"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let p_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::pcg::p"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let ap_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::pcg::Ap"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let z_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::pcg::z"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });

        // Build the cached workspace — identical to `CgWorkspace` plus
        // the (z, p, r) bind groups and the `apply_diag_inv` slots.
        let ws = PcgWorkspace::new(
            self, n, &x_buf, &r_buf, &p_buf, &ap_buf, &z_buf, b, inv_diag,
        );

        // Seed x ← x_init (or zero) + r ← b.
        let mut init_enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::pcg::init"),
            });
        match x_init {
            Some(buf) => init_enc.copy_buffer_to_buffer(buf, 0, &x_buf, 0, bytes),
            None => {
                self.queue
                    .write_buffer(&x_buf, 0, &vec![0u8; bytes as usize]);
            }
        }
        init_enc.copy_buffer_to_buffer(b, 0, &r_buf, 0, bytes);
        self.queue.submit(std::iter::once(init_enc.finish()));
        // Zero p, Ap, z so any later "min-bytes" readbacks see clean data.
        self.queue
            .write_buffer(&p_buf, 0, &vec![0u8; bytes as usize]);
        self.queue
            .write_buffer(&ap_buf, 0, &vec![0u8; bytes as usize]);
        self.queue
            .write_buffer(&z_buf, 0, &vec![0u8; bytes as usize]);

        // r ← b - A·x: compute Ap ← A·x_init, then r -= Ap.
        matvec(&x_buf, &ap_buf)?;
        ws.update_axpy(&self.queue, PcgAxpySlot::RminusAlphaAp, -1.0);
        ws.dispatch_single_axpy(self, PcgAxpySlot::RminusAlphaAp);

        // ‖b‖ — one dot-reduction over b vs b.
        let bnorm_sq = ws.dispatch_dot_blocking(self, PcgDotSlot::Bb)?;
        let bnorm = bnorm_sq.sqrt().max(1.0e-30);

        // Convergence check on ‖r‖ before any preconditioner work.
        let rr = ws.dispatch_dot_blocking(self, PcgDotSlot::Rr)?;
        let mut residual = rr.sqrt() / bnorm;
        if residual <= tol {
            return Ok(CgGpuResult {
                x: Arc::new(x_buf),
                residual,
                iters: 0,
            });
        }

        // z₀ = M⁻¹ · r — one apply_diag_inv dispatch.
        ws.dispatch_apply_diag_inv(self);
        // p₀ ← z₀.
        let mut copy_enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::pcg::p<-z"),
            });
        copy_enc.copy_buffer_to_buffer(&z_buf, 0, &p_buf, 0, bytes);
        self.queue.submit(std::iter::once(copy_enc.finish()));
        // rz = ⟨r, z⟩ — drives the α/β scalars in the preconditioned
        // recurrence (rather than ⟨r, r⟩ in plain CG).
        let mut rz = ws.dispatch_dot_blocking(self, PcgDotSlot::Rz)?;

        for it in 0..max_iters {
            // Ap ← A · p.
            matvec(&p_buf, &ap_buf)?;
            // pAp = ⟨p, Ap⟩.
            let pap = ws.dispatch_dot_blocking(self, PcgDotSlot::PAp)?;
            if pap.abs() < 1.0e-30 {
                return Ok(CgGpuResult {
                    x: Arc::new(x_buf),
                    residual,
                    iters: it,
                });
            }
            let alpha = rz / pap;
            // x += α · p ; r -= α · Ap (single command encoder).
            ws.update_axpy(&self.queue, PcgAxpySlot::XplusAlphaP, alpha);
            ws.update_axpy(&self.queue, PcgAxpySlot::RminusAlphaAp, -alpha);
            ws.dispatch_two_axpy(self, PcgAxpySlot::XplusAlphaP, PcgAxpySlot::RminusAlphaAp);
            // Convergence test on the unpreconditioned residual ‖r‖
            // (matches the CPU `pcg_solve` contract).
            let rr_new = ws.dispatch_dot_blocking(self, PcgDotSlot::Rr)?;
            residual = rr_new.sqrt() / bnorm;
            if residual <= tol {
                return Ok(CgGpuResult {
                    x: Arc::new(x_buf),
                    residual,
                    iters: it + 1,
                });
            }
            // z = M⁻¹ · r.
            ws.dispatch_apply_diag_inv(self);
            // rz_new = ⟨r, z⟩.
            let rz_new = ws.dispatch_dot_blocking(self, PcgDotSlot::Rz)?;
            let beta = rz_new / rz;
            // p ← z + β·p — axpby with α = 1, β = β; the cached uniform
            // already has `alpha = 1` from construction.
            ws.update_axpby(&self.queue, 1.0, beta);
            ws.dispatch_axpby(self);
            rz = rz_new;
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

}

// ── CG workspace + slot enums ─────────────────────────────────────────
//
// `CgWorkspace` is a per-solve cache of every GPU resource the inner
// loop touches: uniform buffers (one per dispatch slot), bind groups
// (one per slot for IO + uniform), partials/staging buffers for the
// dot reductions. Hoisting all of these out of the iteration cuts
// per-iter alloc traffic from ~12 bind groups + ~9 buffers down to
// **zero** — only `write_buffer` updates of the four uniform scalars
// and the unavoidable map_async/poll for the two dot syncs remain.

#[derive(Copy, Clone, Debug)]
enum AxpySlot {
    /// `x += α · p` — io = (x, p), uniform = `axpy_xp`.
    XplusAlphaP,
    /// `r -= α · Ap` — io = (r, Ap), uniform = `axpy_rap`. The α
    /// stored in the uniform is `-alpha` (the kernel does `a += α·b`).
    RminusAlphaAp,
}

#[derive(Copy, Clone, Debug)]
enum DotSlot {
    /// `‖b‖²` — io = (b, b). Computed once at solve start.
    Bb,
    /// `‖r‖²` — io = (r, r). Computed once per CG iter.
    Rr,
    /// `pᵀ Ap` — io = (p, Ap). Computed once per CG iter.
    PAp,
}

struct CgWorkspace {
    n: u32,
    n_groups: u32,
    bytes_partials: u64,

    // Uniform buffers — one per axpy slot + one shared dot uniform
    // (n is the only varying field on the dot side; written once at
    // workspace build). The `u_dot` field looks unused because we
    // only access it through the cached bind group, but keeping the
    // owning handle on the workspace is what keeps wgpu's internal
    // references valid for the lifetime of the bind group.
    #[allow(dead_code)]
    u_axpy_xp: wgpu::Buffer,
    #[allow(dead_code)]
    u_axpy_rap: wgpu::Buffer,
    #[allow(dead_code)]
    u_axpby_p: wgpu::Buffer,
    #[allow(dead_code)]
    u_dot: wgpu::Buffer,

    // IO bind groups — one per (a, b) pair the loop ever touches.
    bg_io_xp: wgpu::BindGroup,
    bg_io_rap: wgpu::BindGroup,
    bg_io_pr: wgpu::BindGroup,
    bg_io_bb: wgpu::BindGroup,
    bg_io_rr: wgpu::BindGroup,
    bg_io_pap: wgpu::BindGroup,

    // Uniform bind groups — one per uniform buffer.
    bg_u_axpy_xp: wgpu::BindGroup,
    bg_u_axpy_rap: wgpu::BindGroup,
    bg_u_axpby_p: wgpu::BindGroup,
    bg_u_dot: wgpu::BindGroup,

    // Dot reduction scratch — partials buffer + staging readback,
    // both sized once for `n_groups`.
    partials: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl CgWorkspace {
    fn new(
        k: &FemCgGpuKernel,
        n: u32,
        x: &wgpu::Buffer,
        r: &wgpu::Buffer,
        p: &wgpu::Buffer,
        ap: &wgpu::Buffer,
        b: &wgpu::Buffer,
    ) -> Self {
        let n_groups = n.div_ceil(WG_SIZE).max(1);
        let bytes_partials = (n_groups as u64).max(4) * 4;

        let make_uniform = |label: &str, size: u64| {
            k.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let axpy_size = core::mem::size_of::<AxpyUniforms>() as u64;
        let axpby_size = core::mem::size_of::<AxpbyUniforms>() as u64;
        let dot_size = core::mem::size_of::<DotUniforms>() as u64;

        let u_axpy_xp = make_uniform("molla::fem_cg::ws::u_axpy_xp", axpy_size);
        let u_axpy_rap = make_uniform("molla::fem_cg::ws::u_axpy_rap", axpy_size);
        let u_axpby_p = make_uniform("molla::fem_cg::ws::u_axpby_p", axpby_size);
        let u_dot = make_uniform("molla::fem_cg::ws::u_dot", dot_size);

        // Dot uniform's only field is `n`; write it once and never
        // touch again.
        k.queue.write_buffer(
            &u_dot,
            0,
            bytemuck::bytes_of(&DotUniforms {
                n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );

        // axpby uniform: kernel reads (alpha, beta). For our `p ← r + β·p`
        // alpha is always 1.0 — write that once and only refresh β
        // per iter via `update_axpby`.
        k.queue.write_buffer(
            &u_axpby_p,
            0,
            bytemuck::bytes_of(&AxpbyUniforms {
                n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                alpha: 1.0,
                beta: 0.0,
                _pad3: 0.0,
                _pad4: 0.0,
            }),
        );

        let make_io_axpy = |label: &str, a: &wgpu::Buffer, b: &wgpu::Buffer| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &k.layout_axpy_io,
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
            })
        };
        let make_io_axpby = |label: &str, dst: &wgpu::Buffer, src: &wgpu::Buffer| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &k.layout_axpby_io,
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
            })
        };
        let make_io_dot = |label: &str, a: &wgpu::Buffer, b: &wgpu::Buffer, partials: &wgpu::Buffer| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &k.layout_dot_io,
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
            })
        };
        let make_u_bg = |label: &str, layout: &wgpu::BindGroupLayout, ubuf: &wgpu::Buffer| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                }],
            })
        };

        let partials = k.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::ws::partials"),
            size: bytes_partials,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = k.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::ws::staging"),
            size: bytes_partials,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            n,
            n_groups,
            bytes_partials,
            bg_io_xp: make_io_axpy("ws::bg_io_xp", x, p),
            bg_io_rap: make_io_axpy("ws::bg_io_rap", r, ap),
            bg_io_pr: make_io_axpby("ws::bg_io_pr", p, r),
            bg_io_bb: make_io_dot("ws::bg_io_bb", b, b, &partials),
            bg_io_rr: make_io_dot("ws::bg_io_rr", r, r, &partials),
            bg_io_pap: make_io_dot("ws::bg_io_pap", p, ap, &partials),
            bg_u_axpy_xp: make_u_bg("ws::bg_u_axpy_xp", &k.layout_axpy_u, &u_axpy_xp),
            bg_u_axpy_rap: make_u_bg("ws::bg_u_axpy_rap", &k.layout_axpy_u, &u_axpy_rap),
            bg_u_axpby_p: make_u_bg("ws::bg_u_axpby_p", &k.layout_axpby_u, &u_axpby_p),
            bg_u_dot: make_u_bg("ws::bg_u_dot", &k.layout_dot_u, &u_dot),
            u_axpy_xp,
            u_axpy_rap,
            u_axpby_p,
            u_dot,
            partials,
            staging,
        }
    }

    /// Refresh the α scalar on a cached axpy uniform. Cheap — just a
    /// `write_buffer` of the 32-byte uniform struct (the `n` field
    /// matches what we wrote at construction; `alpha` is the only
    /// thing that changes per iter).
    fn update_axpy(&self, queue: &wgpu::Queue, slot: AxpySlot, alpha: f32) {
        let buf = match slot {
            AxpySlot::XplusAlphaP => &self.u_axpy_xp,
            AxpySlot::RminusAlphaAp => &self.u_axpy_rap,
        };
        queue.write_buffer(
            buf,
            0,
            bytemuck::bytes_of(&AxpyUniforms {
                n: self.n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                alpha,
                _pad3: 0.0,
                _pad4: 0.0,
                _pad5: 0.0,
            }),
        );
    }

    /// Refresh the (α, β) scalars on the cached `p ← α·r + β·p`
    /// uniform. In the inner loop we only ever pass `α = 1.0`, but
    /// keeping both knobs lets the workspace stay general.
    fn update_axpby(&self, queue: &wgpu::Queue, alpha: f32, beta: f32) {
        queue.write_buffer(
            &self.u_axpby_p,
            0,
            bytemuck::bytes_of(&AxpbyUniforms {
                n: self.n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                alpha,
                beta,
                _pad3: 0.0,
                _pad4: 0.0,
            }),
        );
    }

    fn dispatch_single_axpy(&self, k: &FemCgGpuKernel, slot: AxpySlot) {
        let (bg_io, bg_u) = match slot {
            AxpySlot::XplusAlphaP => (&self.bg_io_xp, &self.bg_u_axpy_xp),
            AxpySlot::RminusAlphaAp => (&self.bg_io_rap, &self.bg_u_axpy_rap),
        };
        let groups = self.n.div_ceil(WG_SIZE).max(1);
        let mut enc = k
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::ws::axpy enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::ws::axpy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_axpy);
            pass.set_bind_group(0, bg_io, &[]);
            pass.set_bind_group(1, bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    /// Two independent axpy dispatches in a single command encoder.
    /// Used in the CG loop to update `x += α·p` and `r -= α·Ap` with
    /// one submit instead of two.
    fn dispatch_two_axpy(&self, k: &FemCgGpuKernel, s1: AxpySlot, s2: AxpySlot) {
        let (bg1_io, bg1_u) = match s1 {
            AxpySlot::XplusAlphaP => (&self.bg_io_xp, &self.bg_u_axpy_xp),
            AxpySlot::RminusAlphaAp => (&self.bg_io_rap, &self.bg_u_axpy_rap),
        };
        let (bg2_io, bg2_u) = match s2 {
            AxpySlot::XplusAlphaP => (&self.bg_io_xp, &self.bg_u_axpy_xp),
            AxpySlot::RminusAlphaAp => (&self.bg_io_rap, &self.bg_u_axpy_rap),
        };
        let groups = self.n.div_ceil(WG_SIZE).max(1);
        let mut enc = k
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::ws::axpy2 enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::ws::axpy2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_axpy);
            pass.set_bind_group(0, bg1_io, &[]);
            pass.set_bind_group(1, bg1_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_bind_group(0, bg2_io, &[]);
            pass.set_bind_group(1, bg2_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    fn dispatch_axpby(&self, k: &FemCgGpuKernel) {
        let groups = self.n.div_ceil(WG_SIZE).max(1);
        let mut enc = k
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::ws::axpby enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::ws::axpby"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_axpby);
            pass.set_bind_group(0, &self.bg_io_pr, &[]);
            pass.set_bind_group(1, &self.bg_u_axpby_p, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    /// `dot` with the dispatch + readback chained into a single
    /// command encoder, then `map_async + poll` to retrieve the
    /// per-workgroup partial sums and reduce on the host.
    fn dispatch_dot_blocking(&self, k: &FemCgGpuKernel, slot: DotSlot) -> Result<f32> {
        let bg_io = match slot {
            DotSlot::Bb => &self.bg_io_bb,
            DotSlot::Rr => &self.bg_io_rr,
            DotSlot::PAp => &self.bg_io_pap,
        };
        let mut enc = k
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::ws::dot enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::ws::dot"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_dot);
            pass.set_bind_group(0, bg_io, &[]);
            pass.set_bind_group(1, &self.bg_u_dot, &[]);
            pass.dispatch_workgroups(self.n_groups, 1, 1);
        }
        // The kernel writes `partials[wid.x]` for every dispatched
        // workgroup (`wid.x ∈ [0, n_groups)`), so the host reduction
        // below — which only reads the first `n_groups` entries —
        // never observes any pre-existing data. No pre-clear needed
        // even though the buffer is padded to the 16-byte minimum
        // for tiny `n_groups`.
        enc.copy_buffer_to_buffer(&self.partials, 0, &self.staging, 0, self.bytes_partials);
        k.queue.submit(std::iter::once(enc.finish()));

        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        k.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("fem_cg ws dot poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("fem_cg ws dot recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("fem_cg ws dot map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let parts: &[f32] = bytemuck::cast_slice(&mapped);
        let mut sum: f32 = 0.0;
        for v in parts.iter().take(self.n_groups as usize) {
            sum += *v;
        }
        drop(mapped);
        self.staging.unmap();
        Ok(sum)
    }
}

// ── LazyCgWorkspace + slot enums ──────────────────────────────────────
//
// Stage 41 v5 — async-dot CG state.
//
// Layout mirrors [`CgWorkspace`] but adds the GPU-resident scalar
// buffers (`pap`, `rr_old`, `rr_new`, `alpha`, `neg_alpha`, `beta`)
// plus the bind groups for the scalar-driven axpy / axpby / cg_alpha /
// cg_beta / dot_finalize kernels. The "uniform-axpy" slots are kept
// around for the one-shot setup `r ← b − A·x_init` (alpha = -1 is a
// host constant; no need to roundtrip through a scalar buffer).
//
// All scalar buffers are 1-element f32 with `STORAGE | COPY_SRC |
// COPY_DST` so we can both bind them to kernels and read them back
// with the standard staging path. Costs ~7 × 4 B of GPU memory + the
// bind-group / pipeline-layout overhead. Negligible.

#[derive(Copy, Clone, Debug)]
enum LazyUniformAxpySlot {
    /// `x += α · p` — uniform-driven (only used at the very end of
    /// setup; the inner loop uses the scalar-driven variant).
    #[allow(dead_code)]
    XplusAlphaP,
    /// `r -= α · Ap` — uniform-driven (used once at setup with α = -1).
    RminusAlphaAp,
}

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
enum LazyAxpyScalarSlot {
    /// `x += alpha[0] * p` — reads alpha from `scalar_alpha`.
    XplusAlphaP,
    /// `r -= alpha[0] * Ap` — reads -alpha from `scalar_neg_alpha`.
    RminusAlphaAp,
}

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
enum LazyDotSlot {
    /// `‖b‖²` — io = (b, b). One-shot at solve start.
    Bb,
    /// `‖r‖²` — io = (r, r). Per iter.
    Rr,
    /// `pᵀ Ap` — io = (p, Ap). Per iter.
    PAp,
}

struct LazyCgWorkspace {
    n: u32,
    n_groups: u32,
    bytes_partials: u64,

    // ── Vector-op uniforms (held alive for bind-group references). ──
    #[allow(dead_code)]
    u_axpy_xp: wgpu::Buffer,
    #[allow(dead_code)]
    u_axpy_rap: wgpu::Buffer,
    #[allow(dead_code)]
    u_axpby_p: wgpu::Buffer,
    #[allow(dead_code)]
    u_dot: wgpu::Buffer,
    #[allow(dead_code)]
    u_finalize: wgpu::Buffer,

    // ── Scalar storage buffers (1 × f32 each). All handles are owned
    // by the workspace to keep the bind groups that reference them
    // alive; the fused inner-loop dispatches reach them only through
    // those bind groups, and the readback path reaches `scalar_rr_new`
    // by re-using the workspace's own field. ────────────────────────
    #[allow(dead_code)]
    scalar_pap: wgpu::Buffer,
    scalar_rr_new: wgpu::Buffer,
    scalar_rr_old: wgpu::Buffer,
    #[allow(dead_code)]
    scalar_alpha: wgpu::Buffer,
    #[allow(dead_code)]
    scalar_neg_alpha: wgpu::Buffer,
    #[allow(dead_code)]
    scalar_beta: wgpu::Buffer,

    // ── Uniform-axpy bind groups (setup phase). ─────────────────────
    bg_io_xp: wgpu::BindGroup,
    bg_io_rap: wgpu::BindGroup,

    // ── Dot bind groups (3 slots × partials). ───────────────────────
    bg_io_bb: wgpu::BindGroup,
    bg_io_rr: wgpu::BindGroup,
    bg_io_pap: wgpu::BindGroup,

    // ── Uniform bind groups. ────────────────────────────────────────
    bg_u_axpy_xp: wgpu::BindGroup,
    bg_u_axpy_rap: wgpu::BindGroup,
    bg_u_axpby_p: wgpu::BindGroup,
    bg_u_dot: wgpu::BindGroup,
    bg_u_finalize: wgpu::BindGroup,

    // ── Scalar-driven kernel bind groups. ───────────────────────────
    // dot_finalize: per-slot (partials → scalar_pap, partials → scalar_rr_new).
    bg_io_fin_pap: wgpu::BindGroup,
    bg_io_fin_rr: wgpu::BindGroup,
    // cg_alpha: (rr_old, pap, alpha, neg_alpha).
    #[allow(dead_code)]
    bg_io_alpha: wgpu::BindGroup,
    // cg_beta: (rr_new, rr_old, beta).
    #[allow(dead_code)]
    bg_io_beta: wgpu::BindGroup,
    // vec_axpy_s: per-slot (x, p, alpha) / (r, Ap, neg_alpha). Kept
    // alive for the granular dispatch helpers; the fused inner-loop
    // path uses `bg_io_xrx` instead.
    #[allow(dead_code)]
    bg_io_axpys_xp: wgpu::BindGroup,
    #[allow(dead_code)]
    bg_io_axpys_rap: wgpu::BindGroup,
    // vec_axpby_s: (p, r, beta). Same — kept for the granular path;
    // the fused inner-loop uses `bg_io_bp`.
    #[allow(dead_code)]
    bg_io_axpbys_pr: wgpu::BindGroup,
    // cg_axpy_xr: (x rw, r rw, p ro, Ap ro, rr_old ro, pap ro).
    bg_io_xrx: wgpu::BindGroup,
    // cg_axpby_p: (p rw, r ro, rr_new ro, rr_old rw).
    bg_io_bp: wgpu::BindGroup,

    // Dot-reduction scratch + staging for one-element scalar readbacks.
    partials: wgpu::Buffer,
    scalar_staging: wgpu::Buffer,
}

impl LazyCgWorkspace {
    fn new(
        k: &FemCgGpuKernel,
        n: u32,
        x: &wgpu::Buffer,
        r: &wgpu::Buffer,
        p: &wgpu::Buffer,
        ap: &wgpu::Buffer,
        b: &wgpu::Buffer,
    ) -> Self {
        let n_groups = n.div_ceil(WG_SIZE).max(1);
        let bytes_partials = (n_groups as u64).max(4) * 4;

        let make_uniform = |label: &str, size: u64| {
            k.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let make_scalar = |label: &str| {
            k.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16, // pad to 16 B for the wgpu min-buffer-size guard
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let axpy_size = core::mem::size_of::<AxpyUniforms>() as u64;
        let axpby_size = core::mem::size_of::<AxpbyUniforms>() as u64;
        let dot_size = core::mem::size_of::<DotUniforms>() as u64;
        let fin_size = 16u64; // DotFinalizeUniforms is 16 B

        let u_axpy_xp = make_uniform("molla::fem_cg::lazy::u_axpy_xp", axpy_size);
        let u_axpy_rap = make_uniform("molla::fem_cg::lazy::u_axpy_rap", axpy_size);
        let u_axpby_p = make_uniform("molla::fem_cg::lazy::u_axpby_p", axpby_size);
        let u_dot = make_uniform("molla::fem_cg::lazy::u_dot", dot_size);
        let u_finalize = make_uniform("molla::fem_cg::lazy::u_finalize", fin_size);

        // Dot uniform: `n` only, written once.
        k.queue.write_buffer(
            &u_dot,
            0,
            bytemuck::bytes_of(&DotUniforms {
                n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );
        // Finalize uniform: `n_partials` only, written once.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct FinU {
            n_partials: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }
        k.queue.write_buffer(
            &u_finalize,
            0,
            bytemuck::bytes_of(&FinU {
                n_partials: n_groups,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );
        // axpby uniform: alpha = 1, beta unused (we use the scalar
        // variant in the loop), n only.
        k.queue.write_buffer(
            &u_axpby_p,
            0,
            bytemuck::bytes_of(&AxpbyUniforms {
                n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                alpha: 1.0,
                beta: 0.0,
                _pad3: 0.0,
                _pad4: 0.0,
            }),
        );
        // axpy uniforms (setup-phase only) — `n` set + α gets overwritten
        // by `update_uniform_axpy`.
        for buf in [&u_axpy_xp, &u_axpy_rap] {
            k.queue.write_buffer(
                buf,
                0,
                bytemuck::bytes_of(&AxpyUniforms {
                    n,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    alpha: 0.0,
                    _pad3: 0.0,
                    _pad4: 0.0,
                    _pad5: 0.0,
                }),
            );
        }

        let scalar_pap = make_scalar("molla::fem_cg::lazy::scalar_pap");
        let scalar_rr_new = make_scalar("molla::fem_cg::lazy::scalar_rr_new");
        let scalar_rr_old = make_scalar("molla::fem_cg::lazy::scalar_rr_old");
        let scalar_alpha = make_scalar("molla::fem_cg::lazy::scalar_alpha");
        let scalar_neg_alpha = make_scalar("molla::fem_cg::lazy::scalar_neg_alpha");
        let scalar_beta = make_scalar("molla::fem_cg::lazy::scalar_beta");

        let partials = k.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::lazy::partials"),
            size: bytes_partials,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scalar_staging = k.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::lazy::scalar_staging"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Bind-group constructors ─────────────────────────────────
        let make_axpy_io = |label: &str, a: &wgpu::Buffer, b: &wgpu::Buffer| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &k.layout_axpy_io,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                ],
            })
        };
        let make_dot_io =
            |label: &str, a: &wgpu::Buffer, b: &wgpu::Buffer, partials: &wgpu::Buffer| {
                k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &k.layout_dot_io,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: partials.as_entire_binding() },
                    ],
                })
            };
        let make_u_bg = |label: &str, layout: &wgpu::BindGroupLayout, ubuf: &wgpu::Buffer| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }],
            })
        };
        let make_finalize_io = |label: &str, out: &wgpu::Buffer| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &k.layout_finalize_io,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: partials.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
                ],
            })
        };
        let make_alpha_io = |label: &str| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &k.layout_alpha_io,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: scalar_rr_old.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: scalar_pap.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: scalar_alpha.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: scalar_neg_alpha.as_entire_binding() },
                ],
            })
        };
        let make_beta_io = |label: &str| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &k.layout_beta_io,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: scalar_rr_new.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: scalar_rr_old.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: scalar_beta.as_entire_binding() },
                ],
            })
        };
        let make_axpys_io = |label: &str, a: &wgpu::Buffer, b: &wgpu::Buffer, s: &wgpu::Buffer| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &k.layout_axpys_io,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: s.as_entire_binding() },
                ],
            })
        };
        let make_axpbys_io = |label: &str, dst: &wgpu::Buffer, src: &wgpu::Buffer, s: &wgpu::Buffer| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &k.layout_axpbys_io,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: dst.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: s.as_entire_binding() },
                ],
            })
        };
        let bg_io_xrx = k.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lazy::bg_io_xrx"),
            layout: &k.layout_xrx_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ap.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: scalar_rr_old.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: scalar_pap.as_entire_binding() },
            ],
        });
        let bg_io_bp = k.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lazy::bg_io_bp"),
            layout: &k.layout_bp_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: p.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: r.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scalar_rr_new.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: scalar_rr_old.as_entire_binding() },
            ],
        });

        Self {
            n,
            n_groups,
            bytes_partials,
            bg_io_xp: make_axpy_io("lazy::bg_io_xp", x, p),
            bg_io_rap: make_axpy_io("lazy::bg_io_rap", r, ap),
            bg_io_bb: make_dot_io("lazy::bg_io_bb", b, b, &partials),
            bg_io_rr: make_dot_io("lazy::bg_io_rr", r, r, &partials),
            bg_io_pap: make_dot_io("lazy::bg_io_pap", p, ap, &partials),
            bg_u_axpy_xp: make_u_bg("lazy::bg_u_axpy_xp", &k.layout_axpy_u, &u_axpy_xp),
            bg_u_axpy_rap: make_u_bg("lazy::bg_u_axpy_rap", &k.layout_axpy_u, &u_axpy_rap),
            bg_u_axpby_p: make_u_bg("lazy::bg_u_axpby_p", &k.layout_axpby_u, &u_axpby_p),
            bg_u_dot: make_u_bg("lazy::bg_u_dot", &k.layout_dot_u, &u_dot),
            bg_u_finalize: make_u_bg("lazy::bg_u_finalize", &k.layout_finalize_u, &u_finalize),
            bg_io_fin_pap: make_finalize_io("lazy::bg_io_fin_pap", &scalar_pap),
            bg_io_fin_rr: make_finalize_io("lazy::bg_io_fin_rr", &scalar_rr_new),
            bg_io_alpha: make_alpha_io("lazy::bg_io_alpha"),
            bg_io_beta: make_beta_io("lazy::bg_io_beta"),
            bg_io_axpys_xp: make_axpys_io("lazy::bg_io_axpys_xp", x, p, &scalar_alpha),
            bg_io_axpys_rap: make_axpys_io("lazy::bg_io_axpys_rap", r, ap, &scalar_neg_alpha),
            bg_io_axpbys_pr: make_axpbys_io("lazy::bg_io_axpbys_pr", p, r, &scalar_beta),
            bg_io_xrx,
            bg_io_bp,
            u_axpy_xp,
            u_axpy_rap,
            u_axpby_p,
            u_dot,
            u_finalize,
            scalar_pap,
            scalar_rr_new,
            scalar_rr_old,
            scalar_alpha,
            scalar_neg_alpha,
            scalar_beta,
            partials,
            scalar_staging,
        }
    }

    fn update_uniform_axpy(&self, queue: &wgpu::Queue, slot: LazyUniformAxpySlot, alpha: f32) {
        let buf = match slot {
            LazyUniformAxpySlot::XplusAlphaP => &self.u_axpy_xp,
            LazyUniformAxpySlot::RminusAlphaAp => &self.u_axpy_rap,
        };
        queue.write_buffer(
            buf,
            0,
            bytemuck::bytes_of(&AxpyUniforms {
                n: self.n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                alpha,
                _pad3: 0.0,
                _pad4: 0.0,
                _pad5: 0.0,
            }),
        );
    }

    /// One-shot axpy via the uniform path (used at setup for
    /// `r ← b − A·x_init`).
    fn dispatch_uniform_axpy(&self, k: &FemCgGpuKernel, slot: LazyUniformAxpySlot) {
        let (bg_io, bg_u) = match slot {
            LazyUniformAxpySlot::XplusAlphaP => (&self.bg_io_xp, &self.bg_u_axpy_xp),
            LazyUniformAxpySlot::RminusAlphaAp => (&self.bg_io_rap, &self.bg_u_axpy_rap),
        };
        let groups = self.n.div_ceil(WG_SIZE).max(1);
        let mut enc = k.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::fem_cg::lazy::uniform-axpy"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::lazy::uniform-axpy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_axpy);
            pass.set_bind_group(0, bg_io, &[]);
            pass.set_bind_group(1, bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    /// One-shot blocking dot, reused at setup for `‖b‖²` and `‖r₀‖²`.
    fn dispatch_dot_blocking(&self, k: &FemCgGpuKernel, slot: LazyDotSlot) -> Result<f32> {
        let bg_io = match slot {
            LazyDotSlot::Bb => &self.bg_io_bb,
            LazyDotSlot::Rr => &self.bg_io_rr,
            LazyDotSlot::PAp => &self.bg_io_pap,
        };
        let mut enc = k.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::fem_cg::lazy::dot-setup"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::lazy::dot-setup"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_dot);
            pass.set_bind_group(0, bg_io, &[]);
            pass.set_bind_group(1, &self.bg_u_dot, &[]);
            pass.dispatch_workgroups(self.n_groups, 1, 1);
        }
        // Stage 41 v5: reuse the existing partials buffer + a tiny
        // staging buffer for the host reduction. Setup-phase only —
        // performance not critical here.
        let stg_setup = k.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::lazy::setup-dot-staging"),
            size: self.bytes_partials,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&self.partials, 0, &stg_setup, 0, self.bytes_partials);
        k.queue.submit(std::iter::once(enc.finish()));
        let slice = stg_setup.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        k.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("fem_cg lazy setup dot poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("fem_cg lazy setup dot recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("fem_cg lazy setup dot map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let parts: &[f32] = bytemuck::cast_slice(&mapped);
        let mut sum: f32 = 0.0;
        for v in parts.iter().take(self.n_groups as usize) {
            sum += *v;
        }
        drop(mapped);
        stg_setup.unmap();
        Ok(sum)
    }

    /// Inner-loop: dispatch a `dot` followed by a `dot_finalize` so
    /// the result lands in a 1-element scalar buffer. No host sync.
    /// All dispatches in one command encoder + submit.
    #[allow(dead_code)]
    fn dispatch_dot_to_scalar(
        &self,
        k: &FemCgGpuKernel,
        slot: LazyDotSlot,
        _scalar_out: &wgpu::Buffer,
    ) {
        let (bg_io_dot, bg_io_fin) = match slot {
            LazyDotSlot::Rr => (&self.bg_io_rr, &self.bg_io_fin_rr),
            LazyDotSlot::PAp => (&self.bg_io_pap, &self.bg_io_fin_pap),
            LazyDotSlot::Bb => {
                // Bb has no associated scalar slot in the lazy loop;
                // setup uses the blocking variant. Fall back to PAp to
                // satisfy the match (never reached on the hot path).
                (&self.bg_io_bb, &self.bg_io_fin_pap)
            }
        };
        let mut enc = k.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::fem_cg::lazy::dot-to-scalar"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::lazy::dot-to-scalar"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_dot);
            pass.set_bind_group(0, bg_io_dot, &[]);
            pass.set_bind_group(1, &self.bg_u_dot, &[]);
            pass.dispatch_workgroups(self.n_groups, 1, 1);
            // Chain dot_finalize — partials → 1-element scalar buffer.
            pass.set_pipeline(&k.pipe_dot_finalize);
            pass.set_bind_group(0, bg_io_fin, &[]);
            pass.set_bind_group(1, &self.bg_u_finalize, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    #[allow(dead_code)]
    fn dispatch_cg_alpha(&self, k: &FemCgGpuKernel) {
        let mut enc = k.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::fem_cg::lazy::cg_alpha"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::lazy::cg_alpha"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_cg_alpha);
            pass.set_bind_group(0, &self.bg_io_alpha, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    #[allow(dead_code)]
    fn dispatch_cg_beta(&self, k: &FemCgGpuKernel) {
        let mut enc = k.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::fem_cg::lazy::cg_beta"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::lazy::cg_beta"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_cg_beta);
            pass.set_bind_group(0, &self.bg_io_beta, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    #[allow(dead_code)]
    fn dispatch_two_axpy_scalar(
        &self,
        k: &FemCgGpuKernel,
        s1: LazyAxpyScalarSlot,
        s2: LazyAxpyScalarSlot,
    ) {
        let pick = |s: LazyAxpyScalarSlot| match s {
            LazyAxpyScalarSlot::XplusAlphaP => (&self.bg_io_axpys_xp, &self.bg_u_axpy_xp),
            LazyAxpyScalarSlot::RminusAlphaAp => (&self.bg_io_axpys_rap, &self.bg_u_axpy_rap),
        };
        let (bg1_io, bg1_u) = pick(s1);
        let (bg2_io, bg2_u) = pick(s2);
        let groups = self.n.div_ceil(WG_SIZE).max(1);
        let mut enc = k.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::fem_cg::lazy::axpy-s pair"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::lazy::axpy-s pair"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_axpy_s);
            pass.set_bind_group(0, bg1_io, &[]);
            pass.set_bind_group(1, bg1_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_bind_group(0, bg2_io, &[]);
            pass.set_bind_group(1, bg2_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    #[allow(dead_code)]
    fn dispatch_axpby_scalar(&self, k: &FemCgGpuKernel) {
        let groups = self.n.div_ceil(WG_SIZE).max(1);
        let mut enc = k.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::fem_cg::lazy::axpby-s"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::lazy::axpby-s"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_axpby_s);
            pass.set_bind_group(0, &self.bg_io_axpbys_pr, &[]);
            pass.set_bind_group(1, &self.bg_u_axpby_p, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    /// Record the post-matvec body of one CG iter into an existing
    /// compute pass. Six dispatches:
    ///   dot(p, Ap) → finalize → scalar_pap
    ///   cg_axpy_xr (fused α + x/r update)
    ///   dot(r, r) → finalize → scalar_rr_new
    ///   cg_axpby_p (fused β + p update + rr_old save)
    ///
    /// Used by both the self-contained-encoder variant
    /// (`dispatch_post_matvec_step_fused`) and the prepared-matvec
    /// variant (which records the matvec into the same pass before
    /// calling this). The two variants share this helper so the
    /// dispatch sequence stays identical.
    fn record_post_matvec(&self, k: &FemCgGpuKernel, pass: &mut wgpu::ComputePass<'_>) {
        let groups = self.n.div_ceil(WG_SIZE).max(1);

        pass.set_pipeline(&k.pipe_dot);
        pass.set_bind_group(0, &self.bg_io_pap, &[]);
        pass.set_bind_group(1, &self.bg_u_dot, &[]);
        pass.dispatch_workgroups(self.n_groups, 1, 1);

        pass.set_pipeline(&k.pipe_dot_finalize);
        pass.set_bind_group(0, &self.bg_io_fin_pap, &[]);
        pass.set_bind_group(1, &self.bg_u_finalize, &[]);
        pass.dispatch_workgroups(1, 1, 1);

        pass.set_pipeline(&k.pipe_cg_axpy_xr);
        pass.set_bind_group(0, &self.bg_io_xrx, &[]);
        pass.set_bind_group(1, &self.bg_u_axpy_xp, &[]);
        pass.dispatch_workgroups(groups, 1, 1);

        pass.set_pipeline(&k.pipe_dot);
        pass.set_bind_group(0, &self.bg_io_rr, &[]);
        pass.set_bind_group(1, &self.bg_u_dot, &[]);
        pass.dispatch_workgroups(self.n_groups, 1, 1);

        pass.set_pipeline(&k.pipe_dot_finalize);
        pass.set_bind_group(0, &self.bg_io_fin_rr, &[]);
        pass.set_bind_group(1, &self.bg_u_finalize, &[]);
        pass.dispatch_workgroups(1, 1, 1);

        pass.set_pipeline(&k.pipe_cg_axpby_p);
        pass.set_bind_group(0, &self.bg_io_bp, &[]);
        pass.set_bind_group(1, &self.bg_u_axpy_xp, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Self-contained-encoder variant — used by the closure-matvec
    /// lazy CG path. Creates its own encoder + submit; the
    /// prepared-matvec variant uses [`Self::record_post_matvec`]
    /// directly into the caller's pass.
    fn dispatch_post_matvec_step_fused(&self, k: &FemCgGpuKernel) {
        let mut enc = k.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::fem_cg::lazy::post-matvec fused"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::lazy::post-matvec fused"),
                timestamp_writes: None,
            });
            self.record_post_matvec(k, &mut pass);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    /// Write a single f32 into a scalar storage buffer (host-driven
    /// — used only at setup to seed `scalar_rr_old`).
    fn write_scalar(&self, queue: &wgpu::Queue, buf: &wgpu::Buffer, value: f32) {
        // Pad to 16 B so the wgpu write_buffer min-size guard is happy
        // (alignment > the 4-B scalar). Trailing bytes are unused by
        // any kernel — they read only [0..4].
        let bytes = [value.to_ne_bytes(), [0u8; 4], [0u8; 4], [0u8; 4]].concat();
        queue.write_buffer(buf, 0, &bytes);
    }

    fn read_scalar_blocking(
        &self,
        k: &FemCgGpuKernel,
        buf: &wgpu::Buffer,
    ) -> Result<f32> {
        let mut enc = k.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::fem_cg::lazy::scalar-readback"),
        });
        enc.copy_buffer_to_buffer(buf, 0, &self.scalar_staging, 0, 16);
        k.queue.submit(std::iter::once(enc.finish()));
        let slice = self.scalar_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        k.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("fem_cg lazy scalar poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("fem_cg lazy scalar recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("fem_cg lazy scalar map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let parts: &[f32] = bytemuck::cast_slice(&mapped);
        let v = parts[0];
        drop(mapped);
        self.scalar_staging.unmap();
        Ok(v)
    }
}

// ── PCG workspace + slot enums ────────────────────────────────────────
//
// Sibling of [`CgWorkspace`] for the Jacobi-preconditioned CG path.
// Adds the resident `z = M⁻¹ · r` buffer + an `apply_diag_inv` IO
// bind group + an additional dot slot (`r ⋅ z`). The axpy/axpby
// machinery is the same; we duplicate the slot enum so the host can
// pass uniform-update calls without leaking the CG-only slot names.

#[derive(Copy, Clone, Debug)]
enum PcgAxpySlot {
    /// `x += α · p` — io = (x, p).
    XplusAlphaP,
    /// `r -= α · Ap` — io = (r, Ap). The α stored in the uniform is
    /// `-alpha` (the kernel does `a += α·b`).
    RminusAlphaAp,
}

#[derive(Copy, Clone, Debug)]
enum PcgDotSlot {
    /// `‖b‖²` — io = (b, b). Computed once at solve start.
    Bb,
    /// `‖r‖²` — io = (r, r). Drives the convergence test.
    Rr,
    /// `pᵀ Ap` — io = (p, Ap). Drives the α scalar.
    PAp,
    /// `rᵀ z` — io = (r, z). Drives the α/β scalars in PCG (replaces
    /// the `‖r‖²` role from the unpreconditioned recurrence).
    Rz,
}

struct PcgWorkspace {
    n: u32,
    n_groups: u32,
    bytes_partials: u64,

    #[allow(dead_code)]
    u_axpy_xp: wgpu::Buffer,
    #[allow(dead_code)]
    u_axpy_rap: wgpu::Buffer,
    #[allow(dead_code)]
    u_axpby_p: wgpu::Buffer,
    #[allow(dead_code)]
    u_dot: wgpu::Buffer,
    #[allow(dead_code)]
    u_diag: wgpu::Buffer,

    bg_io_xp: wgpu::BindGroup,
    bg_io_rap: wgpu::BindGroup,
    /// `axpby` IO bind group for `p ← z + β·p` (dst = p, src = z).
    bg_io_pz: wgpu::BindGroup,
    bg_io_bb: wgpu::BindGroup,
    bg_io_rr: wgpu::BindGroup,
    bg_io_pap: wgpu::BindGroup,
    bg_io_rz: wgpu::BindGroup,
    /// `apply_diag_inv` IO bind group: (z, r, inv_diag).
    bg_io_diag: wgpu::BindGroup,

    bg_u_axpy_xp: wgpu::BindGroup,
    bg_u_axpy_rap: wgpu::BindGroup,
    bg_u_axpby_p: wgpu::BindGroup,
    bg_u_dot: wgpu::BindGroup,
    bg_u_diag: wgpu::BindGroup,

    partials: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl PcgWorkspace {
    #[allow(clippy::too_many_arguments)]
    fn new(
        k: &FemCgGpuKernel,
        n: u32,
        x: &wgpu::Buffer,
        r: &wgpu::Buffer,
        p: &wgpu::Buffer,
        ap: &wgpu::Buffer,
        z: &wgpu::Buffer,
        b: &wgpu::Buffer,
        inv_diag: &wgpu::Buffer,
    ) -> Self {
        let n_groups = n.div_ceil(WG_SIZE).max(1);
        let bytes_partials = (n_groups as u64).max(4) * 4;

        let make_uniform = |label: &str, size: u64| {
            k.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let axpy_size = core::mem::size_of::<AxpyUniforms>() as u64;
        let axpby_size = core::mem::size_of::<AxpbyUniforms>() as u64;
        let dot_size = core::mem::size_of::<DotUniforms>() as u64;

        let u_axpy_xp = make_uniform("molla::fem_cg::pcg::ws::u_axpy_xp", axpy_size);
        let u_axpy_rap = make_uniform("molla::fem_cg::pcg::ws::u_axpy_rap", axpy_size);
        let u_axpby_p = make_uniform("molla::fem_cg::pcg::ws::u_axpby_p", axpby_size);
        let u_dot = make_uniform("molla::fem_cg::pcg::ws::u_dot", dot_size);
        // `apply_diag_inv` shares the axpby uniform layout (only `n`
        // matters; alpha/beta unused). Allocate a separate buffer
        // because the axpby uniform's `alpha`/`beta` get rewritten per
        // iter and we don't want to fight for it.
        let u_diag = make_uniform("molla::fem_cg::pcg::ws::u_diag", axpby_size);

        // Dot uniform's only field is `n`; written once.
        k.queue.write_buffer(
            &u_dot,
            0,
            bytemuck::bytes_of(&DotUniforms {
                n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );

        // axpby uniform: alpha = 1.0 once at construction; β refreshed
        // per iter via `update_axpby`.
        k.queue.write_buffer(
            &u_axpby_p,
            0,
            bytemuck::bytes_of(&AxpbyUniforms {
                n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                alpha: 1.0,
                beta: 0.0,
                _pad3: 0.0,
                _pad4: 0.0,
            }),
        );

        // diag uniform: `n` is the only live field.
        k.queue.write_buffer(
            &u_diag,
            0,
            bytemuck::bytes_of(&AxpbyUniforms {
                n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                alpha: 1.0,
                beta: 0.0,
                _pad3: 0.0,
                _pad4: 0.0,
            }),
        );

        let make_io_axpy = |label: &str, a: &wgpu::Buffer, b: &wgpu::Buffer| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &k.layout_axpy_io,
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
            })
        };
        let make_io_axpby = |label: &str, dst: &wgpu::Buffer, src: &wgpu::Buffer| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &k.layout_axpby_io,
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
            })
        };
        let make_io_dot =
            |label: &str, a: &wgpu::Buffer, b: &wgpu::Buffer, partials: &wgpu::Buffer| {
                k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &k.layout_dot_io,
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
                })
            };
        let make_u_bg = |label: &str, layout: &wgpu::BindGroupLayout, ubuf: &wgpu::Buffer| {
            k.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                }],
            })
        };

        let partials = k.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::pcg::ws::partials"),
            size: bytes_partials,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = k.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::fem_cg::pcg::ws::staging"),
            size: bytes_partials,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // apply_diag_inv IO: (dst = z, src = r, inv_diag).
        let bg_io_diag = k.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::fem_cg::pcg::ws::bg_io_diag"),
            layout: &k.layout_diag_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: inv_diag.as_entire_binding(),
                },
            ],
        });

        Self {
            n,
            n_groups,
            bytes_partials,
            bg_io_xp: make_io_axpy("pcg::ws::bg_io_xp", x, p),
            bg_io_rap: make_io_axpy("pcg::ws::bg_io_rap", r, ap),
            bg_io_pz: make_io_axpby("pcg::ws::bg_io_pz", p, z),
            bg_io_bb: make_io_dot("pcg::ws::bg_io_bb", b, b, &partials),
            bg_io_rr: make_io_dot("pcg::ws::bg_io_rr", r, r, &partials),
            bg_io_pap: make_io_dot("pcg::ws::bg_io_pap", p, ap, &partials),
            bg_io_rz: make_io_dot("pcg::ws::bg_io_rz", r, z, &partials),
            bg_io_diag,
            bg_u_axpy_xp: make_u_bg("pcg::ws::bg_u_axpy_xp", &k.layout_axpy_u, &u_axpy_xp),
            bg_u_axpy_rap: make_u_bg("pcg::ws::bg_u_axpy_rap", &k.layout_axpy_u, &u_axpy_rap),
            bg_u_axpby_p: make_u_bg("pcg::ws::bg_u_axpby_p", &k.layout_axpby_u, &u_axpby_p),
            bg_u_dot: make_u_bg("pcg::ws::bg_u_dot", &k.layout_dot_u, &u_dot),
            bg_u_diag: make_u_bg("pcg::ws::bg_u_diag", &k.layout_diag_u, &u_diag),
            u_axpy_xp,
            u_axpy_rap,
            u_axpby_p,
            u_dot,
            u_diag,
            partials,
            staging,
        }
    }

    fn update_axpy(&self, queue: &wgpu::Queue, slot: PcgAxpySlot, alpha: f32) {
        let buf = match slot {
            PcgAxpySlot::XplusAlphaP => &self.u_axpy_xp,
            PcgAxpySlot::RminusAlphaAp => &self.u_axpy_rap,
        };
        queue.write_buffer(
            buf,
            0,
            bytemuck::bytes_of(&AxpyUniforms {
                n: self.n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                alpha,
                _pad3: 0.0,
                _pad4: 0.0,
                _pad5: 0.0,
            }),
        );
    }

    fn update_axpby(&self, queue: &wgpu::Queue, alpha: f32, beta: f32) {
        queue.write_buffer(
            &self.u_axpby_p,
            0,
            bytemuck::bytes_of(&AxpbyUniforms {
                n: self.n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                alpha,
                beta,
                _pad3: 0.0,
                _pad4: 0.0,
            }),
        );
    }

    fn dispatch_single_axpy(&self, k: &FemCgGpuKernel, slot: PcgAxpySlot) {
        let (bg_io, bg_u) = match slot {
            PcgAxpySlot::XplusAlphaP => (&self.bg_io_xp, &self.bg_u_axpy_xp),
            PcgAxpySlot::RminusAlphaAp => (&self.bg_io_rap, &self.bg_u_axpy_rap),
        };
        let groups = self.n.div_ceil(WG_SIZE).max(1);
        let mut enc = k
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::pcg::ws::axpy enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::pcg::ws::axpy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_axpy);
            pass.set_bind_group(0, bg_io, &[]);
            pass.set_bind_group(1, bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    fn dispatch_two_axpy(&self, k: &FemCgGpuKernel, s1: PcgAxpySlot, s2: PcgAxpySlot) {
        let (bg1_io, bg1_u) = match s1 {
            PcgAxpySlot::XplusAlphaP => (&self.bg_io_xp, &self.bg_u_axpy_xp),
            PcgAxpySlot::RminusAlphaAp => (&self.bg_io_rap, &self.bg_u_axpy_rap),
        };
        let (bg2_io, bg2_u) = match s2 {
            PcgAxpySlot::XplusAlphaP => (&self.bg_io_xp, &self.bg_u_axpy_xp),
            PcgAxpySlot::RminusAlphaAp => (&self.bg_io_rap, &self.bg_u_axpy_rap),
        };
        let groups = self.n.div_ceil(WG_SIZE).max(1);
        let mut enc = k
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::pcg::ws::axpy2 enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::pcg::ws::axpy2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_axpy);
            pass.set_bind_group(0, bg1_io, &[]);
            pass.set_bind_group(1, bg1_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            pass.set_bind_group(0, bg2_io, &[]);
            pass.set_bind_group(1, bg2_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    fn dispatch_axpby(&self, k: &FemCgGpuKernel) {
        let groups = self.n.div_ceil(WG_SIZE).max(1);
        let mut enc = k
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::pcg::ws::axpby enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::pcg::ws::axpby"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_axpby);
            pass.set_bind_group(0, &self.bg_io_pz, &[]);
            pass.set_bind_group(1, &self.bg_u_axpby_p, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    fn dispatch_apply_diag_inv(&self, k: &FemCgGpuKernel) {
        let groups = self.n.div_ceil(WG_SIZE).max(1);
        let mut enc = k
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::pcg::ws::apply_diag_inv enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::pcg::ws::apply_diag_inv"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_apply_diag_inv);
            pass.set_bind_group(0, &self.bg_io_diag, &[]);
            pass.set_bind_group(1, &self.bg_u_diag, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        k.queue.submit(std::iter::once(enc.finish()));
    }

    fn dispatch_dot_blocking(&self, k: &FemCgGpuKernel, slot: PcgDotSlot) -> Result<f32> {
        let bg_io = match slot {
            PcgDotSlot::Bb => &self.bg_io_bb,
            PcgDotSlot::Rr => &self.bg_io_rr,
            PcgDotSlot::PAp => &self.bg_io_pap,
            PcgDotSlot::Rz => &self.bg_io_rz,
        };
        let mut enc = k
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::fem_cg::pcg::ws::dot enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::fem_cg::pcg::ws::dot"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&k.pipe_dot);
            pass.set_bind_group(0, bg_io, &[]);
            pass.set_bind_group(1, &self.bg_u_dot, &[]);
            pass.dispatch_workgroups(self.n_groups, 1, 1);
        }
        // Host reduction reads only the first `n_groups` entries that
        // the kernel always writes — no pre-clear of the padding slots
        // is needed. Same reasoning as `CgWorkspace::dispatch_dot_blocking`.
        enc.copy_buffer_to_buffer(&self.partials, 0, &self.staging, 0, self.bytes_partials);
        k.queue.submit(std::iter::once(enc.finish()));

        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        k.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("fem_cg pcg ws dot poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("fem_cg pcg ws dot recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("fem_cg pcg ws dot map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let parts: &[f32] = bytemuck::cast_slice(&mapped);
        let mut sum: f32 = 0.0;
        for v in parts.iter().take(self.n_groups as usize) {
            sum += *v;
        }
        drop(mapped);
        self.staging.unmap();
        Ok(sum)
    }

    // Touch the unused-but-required-for-binding field handles; quiet
    // dead_code lints when the workspace is built for solves where
    // the diag uniform is never re-bound. Mirrors the helper on
    // [`CgWorkspace`].
    #[allow(dead_code)]
    fn _keep_warnings_quiet(&self) -> &wgpu::Buffer {
        &self.u_diag
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
