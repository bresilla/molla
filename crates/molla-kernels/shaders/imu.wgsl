// IMU sensor compute kernel — accelerometer + gyroscope readings.
//
// Mirrors Newton's `compute_sensor_imu_kernel`
// (xtra/newton/_src/sensors/sensor_imu.py) but computes the linear
// + angular accelerations by **finite difference** between
// successive `body_qd` snapshots, since molla doesn't carry a
// `body_qdd` state field. The sensor object owns a `prev_qd` buffer
// that's swapped with the current state's `body_qd` after each
// dispatch.
//
// Per-sensor inputs:
//   sensor_body       — owning body index (u32)
//   sensor_local_pos  — mount offset in body frame (Vec3)
//   sensor_local_rot  — mount rotation in body frame (Quat)
// Per-body inputs (read from Model + State):
//   body_q, body_qd, body_qd_prev, body_com, body_world, gravity
//
// Outputs (one entry per sensor):
//   accel  — linear acceleration in sensor frame [m/s²]; minus
//            gravity, so a body in free fall reads near zero
//            (this is "specific force", what a real accelerometer
//            measures).
//   gyro   — angular velocity in sensor frame [rad/s].

struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,  // (x, y, z, w)
};
struct SpatialVec {
    linear:  vec4<f32>,
    angular: vec4<f32>,
};

struct Uniforms {
    sensor_count: u32,
    dt:           f32,
    _pad0:        f32,
    _pad1:        f32,
};

@group(0) @binding(0) var<storage, read> body_q:           array<Transform>;
@group(0) @binding(1) var<storage, read> body_qd:          array<SpatialVec>;
@group(0) @binding(2) var<storage, read> body_qd_prev:     array<SpatialVec>;
@group(0) @binding(3) var<storage, read> body_com:         array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> body_world:       array<i32>;
@group(0) @binding(5) var<storage, read> gravity:          array<vec4<f32>>;

@group(1) @binding(0) var<storage, read> sensor_body:      array<u32>;
@group(1) @binding(1) var<storage, read> sensor_local_pos: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> sensor_local_rot: array<vec4<f32>>;

@group(2) @binding(0) var<storage, read_write> accel: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> gyro:  array<vec4<f32>>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

// ── Quaternion helpers (also in raycast.wgsl; duplicated until we
//    factor a common include via naga_oil). ────────────────────────
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
fn compute_imu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.sensor_count) { return; }

    let body = sensor_body[tid];
    let local_pos = sensor_local_pos[tid].xyz;
    let local_rot = sensor_local_rot[tid];

    // Look up body data.
    let bq    = body_q[body];
    let bqd   = body_qd[body];
    let bqd0  = body_qd_prev[body];
    let bcom  = body_com[body].xyz;
    let world = max(body_world[body], 0);
    let g_world = gravity[u32(world)].xyz;

    let inv_dt = 1.0 / uniforms.dt;
    let a_lin_com = (bqd.linear.xyz  - bqd0.linear.xyz)  * inv_dt;
    let a_ang     = (bqd.angular.xyz - bqd0.angular.xyz) * inv_dt;

    // Lever arm from COM to sensor mount, expressed in world frame.
    let r_world = quat_rotate(bq.rotation, local_pos - bcom);
    let omega   = bqd.angular.xyz;

    // Specific force at the sensor point, world frame.
    //   a = a_com − g + α × r + ω × (ω × r)
    let specific_force_world =
        a_lin_com - g_world
        + cross(a_ang, r_world)
        + cross(omega, cross(omega, r_world));

    // World→sensor frame: invert (body_q.rotation · sensor_local_rot).
    let world_to_sensor = quat_mul(bq.rotation, local_rot);
    let inv_q = quat_conj(world_to_sensor);
    let acc_local  = quat_rotate(inv_q, specific_force_world);
    let gyro_local = quat_rotate(inv_q, omega);

    accel[tid] = vec4<f32>(acc_local, 0.0);
    gyro[tid]  = vec4<f32>(gyro_local, 0.0);
}
