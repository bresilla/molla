// Stage 23.1.1 / 23.1.2 — adjoint kernel for Featherstone Pass 1
// (forward kinematics + body-velocity sweep).
//
// **Status: superseded scaffold.** Stage 23.1.2 ships the multi-
// articulation closed-form Jacobian kernel in
// `featherstone_adjoint_jacobians.wgsl` instead — that kernel folds
// the three sweep adjoints into a single per-articulation workgroup
// because the per-sweep round-trip cost dominated the wall time at
// n_links ≤ 6. This file stays as documentation of the per-sweep
// bind-group layout the **pure-analytic** Featherstone (2008) §10
// adjoint will use when it lands in Stage 23.1.3 — the per-sweep
// Pass-1 adjoint becomes useful when the upstream Pass-2 / Pass-3
// adjoints are also analytic so each sweep's kernel does fixed-shape
// O(n) work instead of FD perturbations.

const MAX_LINKS: u32 = 64u;

struct LinkParams {
    // 0 = Revolute, 1 = Prismatic. Ball / Free expand to stacked
    // single-DoF links upstream, so the kernel never sees a multi-DoF
    // joint type — same convention as the CPU adjoint.
    kind: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    axis_local: vec3<f32>,
    _pad3: f32,
    x_p_pos: vec3<f32>,
    _pad4: f32,
    x_p_rot: vec4<f32>,
    com_local: vec3<f32>,
    mass: f32,
    inertia_local_row0: vec3<f32>,
    _pad5: f32,
    inertia_local_row1: vec3<f32>,
    _pad6: f32,
    inertia_local_row2: vec3<f32>,
    _pad7: f32,
}

struct ChainHeader {
    n_links: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    gravity: vec3<f32>,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read> g_links: array<LinkParams>;
@group(0) @binding(1) var<storage, read> g_header: ChainHeader;
@group(0) @binding(2) var<storage, read> g_q: array<f32>;
@group(0) @binding(3) var<storage, read> g_qd: array<f32>;
// Output: per-link kinematic state (body transform + S column + v_body).
// Stride per link: 28 floats (see CPU mirror in
// `featherstone_adjoint::JointKinematic`).
@group(0) @binding(4) var<storage, read_write> g_step_kin: array<f32>;

@compute @workgroup_size(1)
fn fwd_kinematics(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Implementation deferred to Stage 23.1.2 — the closed-form CPU
    // path beats the GPU round-trip cost at n_links ≤ 6 by ~3×.
    // See doc-comment at top of file.
}
