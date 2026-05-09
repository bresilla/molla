// Stage 23.1.1 / 23.1.2 — adjoint kernel for Featherstone Pass 2
// (upward articulated-inertia accumulation).
//
// **Status: superseded scaffold — see `featherstone_adjoint_fwd.wgsl`
// for the rationale.** Stage 23.1.2 collapsed the three per-sweep
// kernels into a single multi-articulation closed-form Jacobian
// kernel (`featherstone_adjoint_jacobians.wgsl`) because the per-
// sweep round-trip cost dominated the wall time at n_links ≤ 6. The
// per-sweep bind-group layout here documents the API the Stage 23.1.3
// pure-analytic path will use.

const MAX_LINKS: u32 = 64u;

struct ChainHeader {
    n_links: u32,
    _pad: vec3<u32>,
}

@group(0) @binding(0) var<storage, read> g_header: ChainHeader;
@group(0) @binding(1) var<storage, read> g_step_kin: array<f32>;
@group(0) @binding(2) var<storage, read> g_tau: array<f32>;
// Output: U_k (vec6 per link), D_k (scalar), u_tau_k (scalar) plus the
// articulated inertia matrix per link (6×6 = 36 floats).
// Stride per link: 6 + 1 + 1 + 36 = 44 floats.
@group(0) @binding(3) var<storage, read_write> g_step_inertia: array<f32>;

@compute @workgroup_size(1)
fn upward_inertia(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Deferred — see doc-comment at top of file.
}
