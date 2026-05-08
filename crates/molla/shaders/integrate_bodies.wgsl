// Semi-implicit Euler integrator for rigid bodies — GPU port of Newton's
// `integrate_rigid_body`. Mirrors the CPU path in `solvers/semi_implicit.rs`
// in single precision (WGSL has no f64).

// Layout of the host f64 → device f32 mirrors. Padding is `vec4` so we
// match WGSL alignment rules.

struct Transform {
    position: vec4<f32>, // xyz + pad
    rotation: vec4<f32>, // quat (x, y, z, w)
};

struct SpatialVec {
    linear: vec4<f32>,  // xyz + pad
    angular: vec4<f32>, // xyz + pad
};

// Mat3 stored as three padded columns (matches `GpuMat3`).
struct Mat3Cols {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
};

struct Uniforms {
    dt: f32,
    angular_damping: f32,
    body_count: u32,
    has_external_force: u32, // 1 if Control::body_external_f populated, else 0
};

const FLAG_KINEMATIC: u32 = 1u;

// ── Group 0 — Model (read-only) ──────────────────────────────────────
@group(0) @binding(0) var<storage, read> body_inv_mass: array<f32>;
@group(0) @binding(1) var<storage, read> body_inertia: array<Mat3Cols>;
@group(0) @binding(2) var<storage, read> body_inv_inertia: array<Mat3Cols>;
@group(0) @binding(3) var<storage, read> body_com: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> body_flags: array<u32>;
@group(0) @binding(5) var<storage, read> body_world: array<i32>;
@group(0) @binding(6) var<storage, read> gravity: array<vec4<f32>>;

// ── Group 1 — State (read inputs, write outputs) ─────────────────────
@group(1) @binding(0) var<storage, read> body_q_in: array<Transform>;
@group(1) @binding(1) var<storage, read> body_qd_in: array<SpatialVec>;
@group(1) @binding(2) var<storage, read> body_f_in: array<SpatialVec>;
@group(1) @binding(3) var<storage, read_write> body_q_out: array<Transform>;
@group(1) @binding(4) var<storage, read_write> body_qd_out: array<SpatialVec>;

// ── Group 2 — Control (read-only) ────────────────────────────────────
@group(2) @binding(0) var<storage, read> body_external_f: array<SpatialVec>;

// ── Group 3 — Uniforms ───────────────────────────────────────────────
@group(3) @binding(0) var<uniform> uniforms: Uniforms;

// ── Helpers ──────────────────────────────────────────────────────────

fn mat3_from_cols(m: Mat3Cols) -> mat3x3<f32> {
    return mat3x3<f32>(m.col0.xyz, m.col1.xyz, m.col2.xyz);
}

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

// ── Kernel ───────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn integrate_bodies(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.body_count) {
        return;
    }

    let flags = body_flags[tid];
    if ((flags & FLAG_KINEMATIC) != 0u) {
        body_q_out[tid] = body_q_in[tid];
        body_qd_out[tid] = body_qd_in[tid];
        return;
    }

    // ── Inputs ────────────────────────────────────────────────────────
    let q = body_q_in[tid];
    let qd = body_qd_in[tid];
    var applied = body_f_in[tid];
    if (uniforms.has_external_force == 1u) {
        let ext = body_external_f[tid];
        applied.linear = applied.linear + ext.linear;
        applied.angular = applied.angular + ext.angular;
    }

    let inv_mass = body_inv_mass[tid];
    let inertia = mat3_from_cols(body_inertia[tid]);
    let inv_inertia = mat3_from_cols(body_inv_inertia[tid]);
    let com = body_com[tid].xyz;

    let world_idx = max(body_world[tid], 0);
    let g = gravity[u32(world_idx)].xyz;

    let x0 = q.position.xyz;
    let r0 = q.rotation;
    let v0 = qd.linear.xyz;
    let w0 = qd.angular.xyz;
    let f0 = applied.linear.xyz;
    let t0 = applied.angular.xyz;

    // World COM = body origin + R * com_local.
    let x_com = x0 + quat_rotate(r0, com);

    // ── Linear ───────────────────────────────────────────────────────
    let gravity_gate = select(0.0, 1.0, inv_mass > 0.0);
    let v1 = v0 + (f0 * inv_mass + g * gravity_gate) * uniforms.dt;
    let x1 = x_com + v1 * uniforms.dt;

    // ── Angular (body frame) ─────────────────────────────────────────
    let r0_inv = quat_conj(r0);
    let wb = quat_rotate(r0_inv, w0);
    let tb = quat_rotate(r0_inv, t0) - cross(wb, inertia * wb);
    var w1 = quat_rotate(r0, wb + inv_inertia * tb * uniforms.dt);

    // Quaternion: r1 = normalise(r0 + 0.5 * dt * (w_quat ⊗ r0))
    let w_quat = vec4<f32>(w1, 0.0);
    let r_delta = quat_mul(w_quat, r0);
    let r1 = normalize(r0 + r_delta * (0.5 * uniforms.dt));

    // Damping after the quaternion update (matches Newton's order).
    w1 = w1 * (1.0 - uniforms.angular_damping * uniforms.dt);

    // Recover body-origin pose from world COM.
    let pos_new = x1 - quat_rotate(r1, com);

    body_q_out[tid].position = vec4<f32>(pos_new, 0.0);
    body_q_out[tid].rotation = r1;
    body_qd_out[tid].linear = vec4<f32>(v1, 0.0);
    body_qd_out[tid].angular = vec4<f32>(w1, 0.0);
}
