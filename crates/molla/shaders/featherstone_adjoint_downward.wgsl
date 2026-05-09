// Stage 23.1.1 / 23.1.2 — adjoint kernel for Featherstone Pass 3
// (downward acceleration solve).
//
// **Status: superseded scaffold — see `featherstone_adjoint_fwd.wgsl`
// for the rationale.** Stage 23.1.2's batched closed-form Jacobian
// kernel (`featherstone_adjoint_jacobians.wgsl`) folds the per-sweep
// adjoints together; the standalone Pass-3 adjoint becomes useful
// once the pure-analytic Pass-1 / Pass-2 derivations land in Stage
// 23.1.3.

const MAX_LINKS: u32 = 64u;

struct ChainHeader {
    n_links: u32,
    _pad: vec3<u32>,
}

@group(0) @binding(0) var<storage, read> g_header: ChainHeader;
@group(0) @binding(1) var<storage, read> g_step_kin: array<f32>;
@group(0) @binding(2) var<storage, read> g_step_inertia: array<f32>;
// Output: q̈_k (scalar per link) + a_body (vec6 per link).
// Stride per link: 1 + 6 = 7 floats.
@group(0) @binding(3) var<storage, read_write> g_step_accel: array<f32>;

@compute @workgroup_size(1)
fn downward_accel(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Deferred — see doc-comment at top of file.
}
