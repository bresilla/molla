// Stage 23.1.1 — adjoint kernel for Featherstone Pass 3 (downward
// acceleration solve).
//
// **Status: scaffold only — see `featherstone_adjoint_fwd.wgsl` doc.**
//
// The Pass-3 adjoint walks root → leaf computing
//   `∂L/∂(q̈_k) = (∂L/∂a_i · S_k - ∂L/∂a_p · U_k) / D_k`
// and accumulates `∂L/∂a_p`. Mirrors the structure of
// `crate::solvers::featherstone::cpu_step` Pass 3 with the kinematic
// + inertia inputs supplied by the FWD + UPWARD adjoint kernels.
//
// As with the other two sweep kernels, the dispatcher currently routes
// to the closed-form CPU path because the per-step round-trip cost
// dominates at n_links ≤ 6. Multi-articulation batching (Stage 23.1.2)
// is the GPU-favourable workload this kernel is scaffolded for.

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
