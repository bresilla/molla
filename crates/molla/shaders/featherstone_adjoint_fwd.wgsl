// Stage 23.1.1 — adjoint kernel for Featherstone Pass 1 (forward
// kinematics + body-velocity sweep).
//
// **Status: scaffold only.**
//
// The adjoint of Pass 1 produces, given `∂L/∂(body_world, v_body, c_term)`,
// the corresponding `∂L/∂(q, qd)`. For a single chain at n_links ≤ 6 the
// per-step FD-on-Pass-1 cost is ~7 µs on the dev laptop; the GPU
// dispatch + readback adds ~20 µs of latency, so the closed-form CPU
// path (`crate::autograd::featherstone_adjoint::fill_jacobians_closed_form`)
// dominates. The kernel scaffold lives here so when **multi-articulation
// batching** (Stage 23.1.2) lands, it slots in without a tape-format
// change — the adjoint dispatcher already iterates the per-sweep
// `TapeOp` variants and can swap `forward_step` for a GPU kernel call.
//
// This file is **not yet wired** into a pipeline; it documents the
// kernel signature + bind-group layout the future GPU path will use.
// See `crates/molla/src/kernels/featherstone_adjoint.rs` for the host-
// side dispatcher that currently routes to the closed-form CPU path
// and reports the GPU pipeline as `Status::DeferredScaffold`.

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
