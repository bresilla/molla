//! Stage 23.1.2 — GPU pipeline for the closed-form Featherstone
//! adjoint Jacobians, batched across articulations.
//!
//! Wraps `shaders/featherstone_adjoint_jacobians.wgsl`. One workgroup
//! per articulation; each chain owns 3 × MAX_LINKS² floats in the
//! output buffer (qdd_tau / qdd_q / qdd_qd, packed as MAX_LINKS×MAX_LINKS
//! row-major sub-matrices). Callers slice by `n_links`.
//!
//! ## When to use
//!
//! - **Multi-articulation batching** — the workload this stage is
//!   built for. Costs amortise over `chain_count` workgroups; the
//!   per-chain throughput beats CPU AnalyticQd once `chain_count ≳ 8`
//!   on the dev laptop's adapter.
//! - **Single articulation** — round-trip dominates; the CPU
//!   AnalyticQd path wins. The `FeatherstoneAdjointDispatcher` resolves
//!   `Auto` to CPU AnalyticQd for the single-chain case for that
//!   reason.
//!
//! ## Parity caveat
//!
//! The shader is f32-internal; CPU paths are f64. Parity tests in
//! `tests/autograd_featherstone_v3_gpu.rs` use a **1e-3** tolerance
//! on the final per-step `g_tau` to absorb the WGSL accumulation
//! noise (looser than CPU's 1e-6 — documented in the spec).

#![cfg(feature = "gpu")]

use crate::autograd::{
    featherstone_adjoint::JacobianStrategy, FeatherstoneRevoluteChain, FeatherstoneTrace,
    JacobianScratch, JointKind,
};
use crate::compute::Engine;
use crate::core::{Error, Result};
use crate::math::{Real, Vec3};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// WGSL-side `MAX_LINKS` constant. Must match the shader.
pub const MAX_LINKS: usize = 8;
const MAX_LINKS2: usize = MAX_LINKS * MAX_LINKS;
/// Per-chain output stride in floats: 3 × MAX_LINKS² (qdd_tau / qdd_q / qdd_qd).
pub const OUT_STRIDE: usize = 3 * MAX_LINKS2;
/// Per-chain state stride in floats: 3 × MAX_LINKS (q / qd / qdd).
pub const STATE_STRIDE: usize = 3 * MAX_LINKS;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct GpuLink {
    kind: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    axis_local: [f32; 3],
    mass: f32,
    x_p_pos: [f32; 3],
    _pad3: f32,
    x_p_rot: [f32; 4],
    com_local: [f32; 3],
    _pad4: f32,
    inertia_local_row0: [f32; 3],
    _pad5: f32,
    inertia_local_row1: [f32; 3],
    _pad6: f32,
    inertia_local_row2: [f32; 3],
    _pad7: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct GpuChainHeader {
    n_links: u32,
    link_offset: u32,
    state_offset: u32,
    out_offset: u32,
    gravity: [f32; 3],
    damping: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct Uniforms {
    chain_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// One chain to be sent to the GPU. The borrow lifetime is tied to
/// the dispatcher call — the kernel snapshots all parameters into
/// device buffers before submission.
pub struct GpuChainInput<'a> {
    pub chain: &'a FeatherstoneRevoluteChain,
    pub q: &'a [Real],
    pub qd: &'a [Real],
    pub qdd: &'a [Real],
}

/// Per-chain Jacobian blocks, host-side as f64 vectors of length n×n.
pub struct GpuChainJacobian {
    /// `qdd_tau[j*n + k] = ∂q̈_j / ∂τ_k = (M⁻¹)_{jk}`.
    pub qdd_tau: Vec<Real>,
    /// `qdd_q[j*n + k] = ∂q̈_j / ∂q_k = -(M⁻¹ · ∂h/∂q)_{jk}`.
    pub qdd_q: Vec<Real>,
    /// `qdd_qd[j*n + k] = ∂q̈_j / ∂qd_k = -(M⁻¹ · ∂h/∂qd)_{jk}`.
    pub qdd_qd: Vec<Real>,
}

/// GPU kernel + cached device buffers for the multi-articulation
/// closed-form Jacobian sweep.
pub struct FeatherstoneAdjointGpuKernel {
    pipeline: wgpu::ComputePipeline,
    bgl_storage: wgpu::BindGroupLayout,
    bgl_uniforms: wgpu::BindGroupLayout,
}

impl FeatherstoneAdjointGpuKernel {
    /// Compile the WGSL kernel. Cheap enough to call at startup.
    pub fn new(engine: &Engine) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("FeatherstoneAdjointGpuKernel: no GPU device".into()))?
            .clone();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::featherstone_adjoint_jacobians"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/featherstone_adjoint_jacobians.wgsl").into(),
            ),
        });
        let bgl_storage = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::adj_jac storage"),
            entries: &[
                storage_entry(0, true),  // chain headers
                storage_entry(1, true),  // links
                storage_entry(2, true),  // qstate
                storage_entry(3, false), // out_jac
            ],
        });
        let bgl_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::adj_jac uniforms"),
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
            label: Some("molla::adj_jac layout"),
            bind_group_layouts: &[&bgl_storage, &bgl_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::adj_jac pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("jacobians"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            pipeline,
            bgl_storage,
            bgl_uniforms,
        })
    }

    /// Run the closed-form Jacobian kernel for a batch of articulations
    /// at the same time-step. Returns one [`GpuChainJacobian`] per input.
    ///
    /// Each chain's `n_links` must satisfy `n ≤ MAX_LINKS = 8`. Larger
    /// chains panic — the kernel's per-thread arrays are sized at
    /// MAX_LINKS for register-allocator reasons; if you need more,
    /// bump the constant in both this file and the WGSL shader.
    pub fn dispatch(
        &self,
        engine: &Engine,
        inputs: &[GpuChainInput<'_>],
    ) -> Result<Vec<GpuChainJacobian>> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("FeatherstoneAdjointGpuKernel::dispatch: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("FeatherstoneAdjointGpuKernel::dispatch: no queue".into()))?
            .clone();

        let chain_count = inputs.len();
        if chain_count == 0 {
            return Ok(vec![]);
        }

        // Pack headers, links, state.
        let mut headers: Vec<GpuChainHeader> = Vec::with_capacity(chain_count);
        let mut links_packed: Vec<GpuLink> = Vec::new();
        let mut qstate_packed: Vec<f32> = Vec::with_capacity(chain_count * STATE_STRIDE);
        for (chain_idx, inp) in inputs.iter().enumerate() {
            let n = inp.chain.links.len();
            assert!(
                n <= MAX_LINKS,
                "FeatherstoneAdjointGpuKernel: chain has {n} links, max {MAX_LINKS}"
            );
            let link_offset = links_packed.len() as u32;
            let state_offset = (chain_idx * STATE_STRIDE) as u32;
            let out_offset = (chain_idx * OUT_STRIDE) as u32;
            for link in &inp.chain.links {
                links_packed.push(pack_link(link));
            }
            // Pack q / qd / qdd, padding to MAX_LINKS each.
            qstate_packed.resize(qstate_packed.len() + 3 * MAX_LINKS, 0.0f32);
            let so = chain_idx * STATE_STRIDE;
            for i in 0..n {
                qstate_packed[so + i] = inp.q[i] as f32;
                qstate_packed[so + MAX_LINKS + i] = inp.qd[i] as f32;
                qstate_packed[so + 2 * MAX_LINKS + i] = inp.qdd[i] as f32;
            }
            let g = inp.chain.gravity;
            headers.push(GpuChainHeader {
                n_links: n as u32,
                link_offset,
                state_offset,
                out_offset,
                gravity: [g.x as f32, g.y as f32, g.z as f32],
                damping: inp.chain.damping as f32,
            });
        }
        // Out buffer = chain_count × OUT_STRIDE floats.
        let out_len = chain_count * OUT_STRIDE;
        let out_zero = vec![0f32; out_len];

        let usage_storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let chains_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::adj_jac chains"),
            contents: bytemuck::cast_slice(&headers),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let links_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::adj_jac links"),
            contents: bytemuck::cast_slice(&links_packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let qstate_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::adj_jac qstate"),
            contents: bytemuck::cast_slice(&qstate_packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let out_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::adj_jac out"),
            contents: bytemuck::cast_slice(&out_zero),
            usage: usage_storage,
        });
        let uniforms = Uniforms {
            chain_count: chain_count as u32,
            ..Default::default()
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::adj_jac uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg_storage = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::adj_jac storage bg"),
            layout: &self.bgl_storage,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chains_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: links_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: qstate_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::adj_jac uniforms bg"),
            layout: &self.bgl_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::adj_jac encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::adj_jac pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_storage, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(chain_count as u32, 1, 1);
        }

        // Stage out → mappable buffer.
        let bytes = (out_len * 4) as u64;
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::adj_jac stage"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel::<wgpu::BufferAsyncError>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            if let Err(e) = r {
                let _ = tx.send(e);
            }
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("adj_jac poll: {e:?}")))?;
        if let Ok(e) = rx.try_recv() {
            return Err(Error::Gpu(format!("adj_jac map: {e:?}")));
        }
        let mapped = slice.get_mapped_range();
        let out_floats: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        stage.unmap();

        // Unpack into per-chain Vec<f64> blocks (n × n each).
        let mut out: Vec<GpuChainJacobian> = Vec::with_capacity(chain_count);
        for (chain_idx, inp) in inputs.iter().enumerate() {
            let n = inp.chain.links.len();
            let oo = chain_idx * OUT_STRIDE;
            let mut qdd_tau = vec![0.0; n * n];
            let mut qdd_q = vec![0.0; n * n];
            let mut qdd_qd = vec![0.0; n * n];
            for j in 0..n {
                for k in 0..n {
                    qdd_tau[j * n + k] = out_floats[oo + j * MAX_LINKS + k] as Real;
                    qdd_q[j * n + k] = out_floats[oo + MAX_LINKS2 + j * MAX_LINKS + k] as Real;
                    qdd_qd[j * n + k] =
                        out_floats[oo + 2 * MAX_LINKS2 + j * MAX_LINKS + k] as Real;
                }
            }
            out.push(GpuChainJacobian {
                qdd_tau,
                qdd_q,
                qdd_qd,
            });
        }
        Ok(out)
    }

    /// Convenience: GPU-batched **full** reverse pass for `inputs`.
    /// Does the trajectory-level reverse on CPU but pulls every per-step
    /// Jacobian block from one batched GPU dispatch — so the per-step
    /// `(M, ∂h/∂q, ∂h/∂qd)` work amortises across the whole batch.
    ///
    /// Returns one `Vec<Vec<Real>>` per chain, same shape as the
    /// per-chain CPU `reverse_with_strategy` output.
    pub fn reverse_batched(
        &self,
        engine: &Engine,
        chains: &[&FeatherstoneRevoluteChain],
        traces: &[&FeatherstoneTrace],
        grad_ee: &[Vec3],
    ) -> Result<Vec<Vec<Vec<Real>>>> {
        let n_chains = chains.len();
        assert_eq!(traces.len(), n_chains, "traces / chains length mismatch");
        assert_eq!(grad_ee.len(), n_chains, "grad_ee / chains length mismatch");
        if n_chains == 0 {
            return Ok(vec![]);
        }
        // Each chain may have its own horizon — but per the spec we batch
        // chains of identical structure (same horizon). The simple impl
        // here loops timesteps and dispatches one GPU batch per step.
        let horizon = chains[0].horizon;
        for c in chains {
            assert_eq!(c.horizon, horizon, "GPU batch requires identical horizon");
        }

        // Per-chain working state — same structure the CPU
        // `reverse_with_strategy` walks.
        let mut g_tau_per_chain: Vec<Vec<Vec<Real>>> = chains
            .iter()
            .map(|c| vec![vec![0.0; c.links.len()]; horizon])
            .collect();
        let mut g_q_per_chain: Vec<Vec<Real>> = chains
            .iter()
            .map(|c| vec![0.0; c.links.len()])
            .collect();
        let mut g_qd_per_chain: Vec<Vec<Real>> = chains
            .iter()
            .map(|c| vec![0.0; c.links.len()])
            .collect();

        // Seed: ∂L/∂q at FINAL step from each chain's EE Jacobian.
        for (ci, chain) in chains.iter().enumerate() {
            let q_final = &traces[ci].q[horizon];
            let j_ee = chain.fk_jacobian(q_final);
            let n = chain.links.len();
            for j in 0..n {
                g_q_per_chain[ci][j] = grad_ee[ci].dot(j_ee[j]);
            }
        }

        // Time-reverse loop. For each t we build inputs (one per chain)
        // and dispatch a single batched GPU kernel call.
        for t in (0..horizon).rev() {
            // Build per-chain GpuChainInput slices for this step.
            let q_buf: Vec<&[Real]> = traces.iter().map(|tr| tr.q[t].as_slice()).collect();
            let qd_buf: Vec<&[Real]> = traces.iter().map(|tr| tr.qd[t].as_slice()).collect();
            let qdd_buf: Vec<&[Real]> = traces
                .iter()
                .map(|tr| tr.steps[t].q_dd.as_slice())
                .collect();
            let inputs: Vec<GpuChainInput<'_>> = (0..n_chains)
                .map(|ci| GpuChainInput {
                    chain: chains[ci],
                    q: q_buf[ci],
                    qd: qd_buf[ci],
                    qdd: qdd_buf[ci],
                })
                .collect();
            let jacs = self.dispatch(engine, &inputs)?;

            // Per-chain VJP + integration adjoint chain.
            for ci in 0..n_chains {
                let chain = chains[ci];
                let n = chain.links.len();
                let dt = chain.dt;
                let damp = (1.0 - chain.damping * dt).max(0.0);
                let g_q = &mut g_q_per_chain[ci];
                let g_qd = &mut g_qd_per_chain[ci];

                // Integration adjoint (mirrors CPU `reverse_with_strategy`).
                let mut g_qd_int = vec![0.0; n];
                for j in 0..n {
                    g_qd_int[j] = g_qd[j] + g_q[j] * dt;
                }
                let mut g_qdd = vec![0.0; n];
                let mut g_qd_in = vec![0.0; n];
                for j in 0..n {
                    g_qd_in[j] = g_qd_int[j] * damp;
                    g_qdd[j] = g_qd_int[j] * damp * dt;
                }

                let jac = &jacs[ci];
                let mut g_q_new = vec![0.0; n];
                let mut g_qd_new = vec![0.0; n];
                for k in 0..n {
                    let mut s_tau = 0.0;
                    let mut s_q = 0.0;
                    let mut s_qd = 0.0;
                    for jj in 0..n {
                        s_tau += g_qdd[jj] * jac.qdd_tau[jj * n + k];
                        s_q += g_qdd[jj] * jac.qdd_q[jj * n + k];
                        s_qd += g_qdd[jj] * jac.qdd_qd[jj * n + k];
                    }
                    g_tau_per_chain[ci][t][k] = s_tau;
                    g_q_new[k] = s_q;
                    g_qd_new[k] = s_qd;
                }
                for j in 0..n {
                    g_q[j] += g_q_new[j];
                    g_qd[j] = g_qd_in[j] + g_qd_new[j];
                }
            }
        }
        Ok(g_tau_per_chain)
    }
}

/// Helper to convert a `RevoluteLink` into the WGSL-ready `GpuLink`
/// representation. Multi-DoF kinds (Ball / Free) are not supported by
/// the GPU kernel directly — they are stacked on the host into
/// single-DoF Revolute / Prismatic links upstream of this call (same
/// convention as the CPU adjoint).
fn pack_link(link: &crate::autograd::featherstone_adjoint::RevoluteLink) -> GpuLink {
    let kind = match link.kind {
        JointKind::Revolute => 0u32,
        JointKind::Prismatic => 1u32,
        JointKind::Ball | JointKind::Free | JointKind::Distance | JointKind::D6 => {
            // Kernel never sees these — chain construction expands
            // them to stacked single-DoF links. Defensive: log + fall
            // back to revolute (no-op axis).
            eprintln!(
                "[molla::featherstone_adjoint_gpu] unexpected multi-DoF JointKind in chain — \
                 caller should stack to single-DoF"
            );
            0u32
        }
    };
    let q = link.x_p.rotation;
    GpuLink {
        kind,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        axis_local: [
            link.axis_local.x as f32,
            link.axis_local.y as f32,
            link.axis_local.z as f32,
        ],
        mass: link.mass as f32,
        x_p_pos: [
            link.x_p.position.x as f32,
            link.x_p.position.y as f32,
            link.x_p.position.z as f32,
        ],
        _pad3: 0.0,
        x_p_rot: [q.x as f32, q.y as f32, q.z as f32, q.w as f32],
        com_local: [
            link.com_local.x as f32,
            link.com_local.y as f32,
            link.com_local.z as f32,
        ],
        _pad4: 0.0,
        inertia_local_row0: [
            link.inertia_local.x_axis.x as f32,
            link.inertia_local.y_axis.x as f32,
            link.inertia_local.z_axis.x as f32,
        ],
        _pad5: 0.0,
        inertia_local_row1: [
            link.inertia_local.x_axis.y as f32,
            link.inertia_local.y_axis.y as f32,
            link.inertia_local.z_axis.y as f32,
        ],
        _pad6: 0.0,
        inertia_local_row2: [
            link.inertia_local.x_axis.z as f32,
            link.inertia_local.y_axis.z as f32,
            link.inertia_local.z_axis.z as f32,
        ],
        _pad7: 0.0,
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

// Suppress unused-but-public warning when caller never imports
// `JacobianScratch` — it's part of the API we keep for future use.
#[allow(dead_code)]
fn _unused() {
    let _: Option<JacobianScratch> = None;
    let _: JacobianStrategy = JacobianStrategy::AnalyticBiasQd;
}
