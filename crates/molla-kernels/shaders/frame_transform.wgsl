// Frame-transform sensor — for each (body, local_pose) target,
// computes the world-frame pose `body_q · local_pose`, and (if a
// reference frame is supplied) the relative pose
// `inverse(reference_world_pose) · target_world_pose`.
//
// Mirrors Newton's `compute_shape_transforms_kernel` +
// `compute_relative_transforms_kernel` (xtra/newton/_src/sensors/
// sensor_frame_transform.py) but takes flat (body_idx, local_pose)
// arrays instead of shape indices — molla doesn't have a "site"
// concept, every frame is just a body + offset.
//
// One thread per target. The reference frame is either:
//   `has_reference == 0` → identity (world-frame poses only)
//   `has_reference == 1` → reference pose at index 0 in the same
//                          target arrays (so callers prepend the
//                          reference and the kernel skips slot 0
//                          when computing relative).

struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,
};

struct Uniforms {
    target_count:  u32,
    has_reference: u32,
    _pad0:         u32,
    _pad1:         u32,
};

@group(0) @binding(0) var<storage, read> body_q:           array<Transform>;
@group(0) @binding(1) var<storage, read> target_body:      array<u32>;
@group(0) @binding(2) var<storage, read> target_local_pos: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> target_local_rot: array<vec4<f32>>;

@group(1) @binding(0) var<storage, read_write> world_pos:    array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> world_rot:    array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> relative_pos: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> relative_rot: array<vec4<f32>>;

@group(2) @binding(0) var<uniform> uniforms: Uniforms;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}
fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

@compute @workgroup_size(64)
fn compute_frame_transforms(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.target_count) { return; }

    let body = target_body[tid];
    let bq = body_q[body];

    // World pose = body_q · local_pose.
    let lp = target_local_pos[tid].xyz;
    let lr = target_local_rot[tid];
    let wp_pos = bq.position.xyz + quat_rotate(bq.rotation, lp);
    let wp_rot = quat_mul(bq.rotation, lr);

    world_pos[tid] = vec4<f32>(wp_pos, 0.0);
    world_rot[tid] = wp_rot;

    if (uniforms.has_reference == 1u) {
        // Caller stashed the reference frame at index 0. Slot 0's
        // own relative pose is identity; everyone else is
        // inverse(world[0]) · world[tid].
        if (tid == 0u) {
            relative_pos[0u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            relative_rot[0u] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        } else {
            // Compute reference world pose ourselves so we don't
            // race the slot-0 thread (each thread is independent).
            let r_body = target_body[0u];
            let r_lp = target_local_pos[0u].xyz;
            let r_lr = target_local_rot[0u];
            let r_bq = body_q[r_body];
            let r_world_pos = r_bq.position.xyz + quat_rotate(r_bq.rotation, r_lp);
            let r_world_rot = quat_mul(r_bq.rotation, r_lr);

            let inv_rot = quat_conj(r_world_rot);
            let rel_pos = quat_rotate(inv_rot, wp_pos - r_world_pos);
            let rel_rot = quat_mul(inv_rot, wp_rot);
            relative_pos[tid] = vec4<f32>(rel_pos, 0.0);
            relative_rot[tid] = rel_rot;
        }
    } else {
        relative_pos[tid] = vec4<f32>(wp_pos, 0.0);
        relative_rot[tid] = wp_rot;
    }
}
