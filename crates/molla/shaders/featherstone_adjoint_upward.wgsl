// Stage 23.1.1 — adjoint kernel for Featherstone Pass 2 (upward
// articulated-inertia accumulation).
//
// **Status: scaffold only — see `featherstone_adjoint_fwd.wgsl` doc.**
//
// The Pass-2 adjoint walks the chain leaf → root accumulating the
// child-block contributions to `∂L/∂(I^A_p, p^A_p)` from the parent's
// rank-1 deflation `I^A := I^A − U·Uᵀ / D`. For a closed-form
// implementation see Carpentier & Mansard 2018 — the per-step cost is
// `O(n²)` matrix work which lines up well with WGSL's per-thread
// register file budget at n ≤ 6.
//
// The current adjoint dispatcher routes through the closed-form CPU
// path (CRBA + Cholesky-invert) which produces an exact `M⁻¹ = ∂q̈/∂τ`
// directly — bypassing the per-sweep adjoint entirely. The GPU sweep
// adjoints would be useful for **multi-articulation batching** (Stage
// 23.1.2) where the per-step Jacobian is amortised across 32+ chains;
// for the Stage-23.1 single-chain case the closed-form solve beats it
// by ~3×.

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
