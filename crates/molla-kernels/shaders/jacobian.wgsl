// Spatial body Jacobian — `6 × N` matrix mapping joint velocities to a
// target body-point's spatial velocity at world origin convention.
//
// Mirrors `crates/molla/src/solvers/jacobian.rs::jacobian`. One workgroup
// invocation per (articulation, request); each request specifies a
// target body + body-local point. Output: `6 · N` floats per request,
// row-major ([linear (3); angular (3)] × N).
//
// Multiple requests batched in one dispatch — useful for IK with many
// position objectives or many seeds.

struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,
};

struct Request {
    // 32 bytes total (vec4 alignment after the four u32s).
    articulation:    u32,
    target_body:     u32,
    output_offset:   u32,
    body_q_offset:   u32,
    point_local:     vec4<f32>,
};

struct Uniforms {
    request_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

const JOINT_PRISMATIC: u32 = 0u;
const JOINT_REVOLUTE:  u32 = 1u;
const JOINT_FIXED:     u32 = 3u;

const MAX_JOINTS_PER_ART: u32 = 64u;

// Group 0 — Model
@group(0) @binding(0) var<storage, read> joint_type:        array<u32>;
@group(0) @binding(1) var<storage, read> joint_parent:      array<i32>;
@group(0) @binding(2) var<storage, read> joint_child:       array<i32>;
@group(0) @binding(3) var<storage, read> joint_x_p:         array<Transform>;
@group(0) @binding(4) var<storage, read> joint_axis:        array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> joint_dof_offset:  array<u32>;
@group(0) @binding(6) var<storage, read> joint_x_c: array<Transform>;
@group(0) @binding(7) var<storage, read> joint_reversed: array<u32>;
@group(0) @binding(8) var<storage, read> joint_enabled: array<u32>;
@group(0) @binding(9) var<storage, read> body_flags: array<u32>;
@group(0) @binding(10) var<storage, read> joint_limit_lower: array<f32>;
@group(0) @binding(11) var<storage, read> joint_limit_upper: array<f32>;

// Group 1 — State
@group(1) @binding(0) var<storage, read> body_q: array<Transform>;

// Group 2 — Articulation CSR + requests + output
@group(2) @binding(0) var<storage, read>       art_offsets: array<u32>;
@group(2) @binding(1) var<storage, read>       requests:    array<Request>;
@group(2) @binding(2) var<storage, read_write> jacobian_out: array<f32>;

// Group 3 — Uniforms
@group(3) @binding(0) var<uniform> uniforms: Uniforms;

// Helpers (same shape as crba.wgsl).

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    );
}
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s*s - dot(u,u)) * v + 2.0 * s * cross(u, v);
}
fn transform_compose(a: Transform, b: Transform) -> Transform {
    var r: Transform;
    r.rotation = normalize(quat_mul(a.rotation, b.rotation));
    r.position = vec4<f32>(a.position.xyz + quat_rotate(a.rotation, b.position.xyz), 0.0);
    return r;
}

@compute @workgroup_size(64)
fn jacobian_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let req_idx = gid.x;
    if (req_idx >= uniforms.request_count) { return; }
    let req = requests[req_idx];

    let j_begin = art_offsets[req.articulation];
    let j_end   = art_offsets[req.articulation + 1u];
    let n_joints = j_end - j_begin;
    if (n_joints == 0u) { return; }
    if (n_joints > MAX_JOINTS_PER_ART) { return; }

    // Target point in world (body_q indexed via per-request slice offset
    // so multiple seeds with distinct FK can share one dispatch).
    let bq_off = req.body_q_offset;
    let cw = body_q[bq_off + req.target_body];
    let target_world = cw.position.xyz + quat_rotate(cw.rotation, req.point_local.xyz);

    // Walk parent chain from target_body to mark ancestor joints. Note
    // `joint_parent`/`joint_child` are GLOBAL body indices, not seed-local,
    // so we use req.target_body unshifted here.
    var is_ancestor: array<u32, MAX_JOINTS_PER_ART>;
    for (var k = 0u; k < n_joints; k = k + 1u) { is_ancestor[k] = 0u; }
    var walker: i32 = i32(req.target_body);
    loop {
        if (walker < 0) { break; }
        var found = false;
        for (var kk = 0u; kk < n_joints; kk = kk + 1u) {
            if (joint_child[j_begin + kk] == walker) {
                is_ancestor[kk] = 1u;
                walker = joint_parent[j_begin + kk];
                found = true;
                break;
            }
        }
        if (!found) { break; }
    }

    // Articulation-local generalized velocity width.
    let qd_base = joint_dof_offset[j_begin];
    let dof_total = joint_dof_offset[j_end] - qd_base;
    if (dof_total == 0u) { return; }

    // Zero the output tile.
    let out_base = req.output_offset;
    for (var i = 0u; i < 6u * dof_total; i = i + 1u) {
        jacobian_out[out_base + i] = 0.0;
    }

    // For each joint in the articulation, fill its column if it's an ancestor.
    for (var k = 0u; k < n_joints; k = k + 1u) {
        if (is_ancestor[k] == 0u) { continue; }
        let j = j_begin + k;
        let jt = joint_type[j];
        let frame = response_frame(j, bq_off);
        for (var d = 0u; d < response_width(jt); d += 1u) {
            let column = response_column(j, d, frame, target_world);
            let offset = out_base + joint_dof_offset[j] - qd_base + d;
            jacobian_out[offset] = column.linear.x;
            jacobian_out[offset + dof_total] = column.linear.y;
            jacobian_out[offset + 2u * dof_total] = column.linear.z;
            jacobian_out[offset + 3u * dof_total] = column.angular.x;
            jacobian_out[offset + 4u * dof_total] = column.angular.y;
            jacobian_out[offset + 5u * dof_total] = column.angular.z;
        }
    }
}
