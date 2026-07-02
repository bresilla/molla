// GPU planar vehicle integration from tire world wrenches.
//
// One workgroup sums wheel forces and tire self-aligning moments, then
// integrates chassis translation and yaw/yaw-rate. This is the GPU counterpart
// to the CPU `Vehicle` planar wrench equation:
// yaw moment = r_x F_z - r_z F_x + tire_Mz.

const WG: u32 = 64u;

struct Uniforms {
    // x=wheel_count, yzw=reserved
    counts: vec4<u32>,
    // x=pos_x_m, y=pos_z_m, z=yaw_rad, w=reserved
    pose: vec4<f32>,
    // x=vel_x_m_s, y=vel_z_m_s, z=yaw_rate_rad_s, w=reserved
    velocity: vec4<f32>,
    // x=mass_kg, y=yaw_inertia_kg_m2, z=dt_s, w=linear_damping_ns_m
    params0: vec4<f32>,
    // x=yaw_damping_nms, yzw=reserved
    params1: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> wheel_offsets: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tire_wrenches: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> yaw_out: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> resident_rows: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> planar_rows: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> vertical_rows: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> u: Uniforms;

var<workgroup> partial_mz: array<f32, WG>;
var<workgroup> partial_fx: array<f32, WG>;
var<workgroup> partial_fy: array<f32, WG>;
var<workgroup> partial_fz: array<f32, WG>;

fn finite_scalar(v: f32) -> bool {
    return v == v && abs(v) < 3.0e38;
}

@compute @workgroup_size(WG)
fn integrate_yaw(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    let n = min(u.counts.x, WG);
    var mz = 0.0;
    if (i < n) {
        let off = wheel_offsets[i].xyz;
        let wrench = tire_wrenches[i];
        let force = wrench.xyz;
        let tire_mz = wrench.w;
        mz = off.x * force.z - off.z * force.x + tire_mz;
    }
    partial_mz[i] = mz;
    workgroupBarrier();

    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (i < stride) {
            partial_mz[i] = partial_mz[i] + partial_mz[i + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (i == 0u) {
        let inertia = max(u.params0.y, 1.0e-6);
        let dt = max(u.params0.z, 0.0);
        let damping = max(u.params1.x, 0.0);
        let total_mz = partial_mz[0];
        let yaw_rate =
            (inertia * u.velocity.z + total_mz * dt) / max(inertia + damping * dt, 1.0e-6);
        let yaw = u.pose.z + yaw_rate * dt;
        let finite_flag = select(
            0.0,
            1.0,
            finite_scalar(yaw) && finite_scalar(yaw_rate) && finite_scalar(total_mz),
        );
        yaw_out[0] = vec4<f32>(yaw, yaw_rate, total_mz, finite_flag);
    }
}

@compute @workgroup_size(WG)
fn integrate_planar(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    let n = min(u.counts.x, WG);
    var fx_sum = 0.0;
    var fz_sum = 0.0;
    var mz = 0.0;
    if (i < n) {
        let off = wheel_offsets[i].xyz;
        let wrench = tire_wrenches[i];
        let force = wrench.xyz;
        let tire_mz = wrench.w;
        fx_sum = force.x;
        fz_sum = force.z;
        mz = off.x * force.z - off.z * force.x + tire_mz;
    }
    partial_fx[i] = fx_sum;
    partial_fz[i] = fz_sum;
    partial_mz[i] = mz;
    workgroupBarrier();

    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (i < stride) {
            partial_fx[i] = partial_fx[i] + partial_fx[i + stride];
            partial_fz[i] = partial_fz[i] + partial_fz[i + stride];
            partial_mz[i] = partial_mz[i] + partial_mz[i + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (i == 0u) {
        let mass = max(u.params0.x, 1.0e-6);
        let inertia = max(u.params0.y, 1.0e-6);
        let dt = max(u.params0.z, 0.0);
        let linear_damping = max(u.params0.w, 0.0);
        let yaw_damping = max(u.params1.x, 0.0);
        let total_fx = partial_fx[0];
        let total_fz = partial_fz[0];
        let total_mz = partial_mz[0];
        let vx =
            (mass * u.velocity.x + total_fx * dt) / max(mass + linear_damping * dt, 1.0e-6);
        let vz =
            (mass * u.velocity.y + total_fz * dt) / max(mass + linear_damping * dt, 1.0e-6);
        let px = u.pose.x + vx * dt;
        let pz = u.pose.y + vz * dt;
        let yaw_rate =
            (inertia * u.velocity.z + total_mz * dt) / max(inertia + yaw_damping * dt, 1.0e-6);
        let yaw = u.pose.z + yaw_rate * dt;
        let finite_flag = select(
            0.0,
            1.0,
            finite_scalar(px) &&
                finite_scalar(pz) &&
                finite_scalar(vx) &&
                finite_scalar(vz) &&
                finite_scalar(yaw) &&
                finite_scalar(yaw_rate) &&
                finite_scalar(total_fx) &&
                finite_scalar(total_fz) &&
                finite_scalar(total_mz),
        );
        yaw_out[0] = vec4<f32>(px, pz, vx, vz);
        yaw_out[1] = vec4<f32>(yaw, yaw_rate, total_mz, finite_flag);
        yaw_out[2] = vec4<f32>(total_fx, total_fz, 0.0, 0.0);
    }
}

@compute @workgroup_size(WG)
fn integrate_planar_from_brush_outputs(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    let n = min(u.counts.x, WG);
    var fx_sum = 0.0;
    var fz_sum = 0.0;
    var mz = 0.0;
    if (i < n) {
        let off = wheel_offsets[i].xyz;
        // tire_wrenches is a flattened BrushWheelOutput array:
        // row + 2 = diag2 (y = aligning_moment_z),
        // row + 6 = world_force (x/y/z).
        let base = i * 7u;
        let diag2 = tire_wrenches[base + 2u];
        let world_force = tire_wrenches[base + 6u];
        let tire_mz = diag2.y;
        fx_sum = world_force.x;
        fz_sum = world_force.z;
        mz = off.x * world_force.z - off.z * world_force.x + tire_mz;
    }
    partial_fx[i] = fx_sum;
    partial_fz[i] = fz_sum;
    partial_mz[i] = mz;
    workgroupBarrier();

    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (i < stride) {
            partial_fx[i] = partial_fx[i] + partial_fx[i + stride];
            partial_fz[i] = partial_fz[i] + partial_fz[i + stride];
            partial_mz[i] = partial_mz[i] + partial_mz[i + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (i == 0u) {
        let mass = max(u.params0.x, 1.0e-6);
        let inertia = max(u.params0.y, 1.0e-6);
        let dt = max(u.params0.z, 0.0);
        let linear_damping = max(u.params0.w, 0.0);
        let yaw_damping = max(u.params1.x, 0.0);
        let total_fx = partial_fx[0];
        let total_fz = partial_fz[0];
        let total_mz = partial_mz[0];
        let vx =
            (mass * u.velocity.x + total_fx * dt) / max(mass + linear_damping * dt, 1.0e-6);
        let vz =
            (mass * u.velocity.y + total_fz * dt) / max(mass + linear_damping * dt, 1.0e-6);
        let px = u.pose.x + vx * dt;
        let pz = u.pose.y + vz * dt;
        let yaw_rate =
            (inertia * u.velocity.z + total_mz * dt) / max(inertia + yaw_damping * dt, 1.0e-6);
        let yaw = u.pose.z + yaw_rate * dt;
        let finite_flag = select(
            0.0,
            1.0,
            finite_scalar(px) &&
                finite_scalar(pz) &&
                finite_scalar(vx) &&
                finite_scalar(vz) &&
                finite_scalar(yaw) &&
                finite_scalar(yaw_rate) &&
                finite_scalar(total_fx) &&
                finite_scalar(total_fz) &&
                finite_scalar(total_mz),
        );
        yaw_out[0] = vec4<f32>(px, pz, vx, vz);
        yaw_out[1] = vec4<f32>(yaw, yaw_rate, total_mz, finite_flag);
        yaw_out[2] = vec4<f32>(total_fx, total_fz, 0.0, 0.0);
    }
}

@compute @workgroup_size(WG)
fn integrate_planar_state_from_brush_outputs(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    let n = min(u.counts.x, WG);
    var fx_sum = 0.0;
    var fz_sum = 0.0;
    var mz = 0.0;
    if (i < n) {
        let off = wheel_offsets[i].xyz;
        // tire_wrenches is a flattened BrushWheelOutput array:
        // row + 2 = diag2 (y = aligning_moment_z),
        // row + 6 = world_force (x/y/z).
        let base = i * 7u;
        let diag2 = tire_wrenches[base + 2u];
        let world_force = tire_wrenches[base + 6u];
        let tire_mz = diag2.y;
        fx_sum = world_force.x;
        fz_sum = world_force.z;
        mz = off.x * world_force.z - off.z * world_force.x + tire_mz;
    }
    partial_fx[i] = fx_sum;
    partial_fz[i] = fz_sum;
    partial_mz[i] = mz;
    workgroupBarrier();

    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (i < stride) {
            partial_fx[i] = partial_fx[i] + partial_fx[i + stride];
            partial_fz[i] = partial_fz[i] + partial_fz[i + stride];
            partial_mz[i] = partial_mz[i] + partial_mz[i + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (i == 0u) {
        let state = resident_rows[0];
        let yaw_state = resident_rows[1];
        let mass = max(u.params0.x, 1.0e-6);
        let inertia = max(u.params0.y, 1.0e-6);
        let dt = max(u.params0.z, 0.0);
        let linear_damping = max(u.params0.w, 0.0);
        let yaw_damping = max(u.params1.x, 0.0);
        let total_fx = partial_fx[0];
        let total_fz = partial_fz[0];
        let total_mz = partial_mz[0];
        let vx =
            (mass * state.z + total_fx * dt) / max(mass + linear_damping * dt, 1.0e-6);
        let vz =
            (mass * state.w + total_fz * dt) / max(mass + linear_damping * dt, 1.0e-6);
        let px = state.x + vx * dt;
        let pz = state.y + vz * dt;
        let yaw_rate =
            (inertia * yaw_state.y + total_mz * dt) / max(inertia + yaw_damping * dt, 1.0e-6);
        let yaw = yaw_state.x + yaw_rate * dt;
        let finite_flag = select(
            0.0,
            1.0,
            finite_scalar(px) &&
                finite_scalar(pz) &&
                finite_scalar(vx) &&
                finite_scalar(vz) &&
                finite_scalar(yaw) &&
                finite_scalar(yaw_rate) &&
                finite_scalar(total_fx) &&
                finite_scalar(total_fz) &&
                finite_scalar(total_mz),
        );
        yaw_out[0] = vec4<f32>(px, pz, vx, vz);
        yaw_out[1] = vec4<f32>(yaw, yaw_rate, total_mz, finite_flag);
        yaw_out[2] = vec4<f32>(total_fx, total_fz, 0.0, 0.0);
    }
}

@compute @workgroup_size(WG)
fn integrate_planar_from_contact_reduction_outputs(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    let n = min(u.counts.x, WG);
    var fx_sum = 0.0;
    var fz_sum = 0.0;
    var mz = 0.0;
    if (i < n) {
        let off = wheel_offsets[i].xyz;
        // tire_wrenches is a packed TireFemContactReductionOutput array:
        // row + 2 = world_force.xyz,
        // row + 3 = world_moment.xyz (yaw uses y/up moment).
        let base = i * 4u;
        let world_force = tire_wrenches[base + 2u];
        let world_moment = tire_wrenches[base + 3u];
        let tire_mz = world_moment.y;
        fx_sum = world_force.x;
        fz_sum = world_force.z;
        mz = off.x * world_force.z - off.z * world_force.x + tire_mz;
    }
    partial_fx[i] = fx_sum;
    partial_fz[i] = fz_sum;
    partial_mz[i] = mz;
    workgroupBarrier();

    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (i < stride) {
            partial_fx[i] = partial_fx[i] + partial_fx[i + stride];
            partial_fz[i] = partial_fz[i] + partial_fz[i + stride];
            partial_mz[i] = partial_mz[i] + partial_mz[i + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (i == 0u) {
        let mass = max(u.params0.x, 1.0e-6);
        let inertia = max(u.params0.y, 1.0e-6);
        let dt = max(u.params0.z, 0.0);
        let linear_damping = max(u.params0.w, 0.0);
        let yaw_damping = max(u.params1.x, 0.0);
        let total_fx = partial_fx[0];
        let total_fz = partial_fz[0];
        let total_mz = partial_mz[0];
        let vx =
            (mass * u.velocity.x + total_fx * dt) / max(mass + linear_damping * dt, 1.0e-6);
        let vz =
            (mass * u.velocity.y + total_fz * dt) / max(mass + linear_damping * dt, 1.0e-6);
        let px = u.pose.x + vx * dt;
        let pz = u.pose.y + vz * dt;
        let yaw_rate =
            (inertia * u.velocity.z + total_mz * dt) / max(inertia + yaw_damping * dt, 1.0e-6);
        let yaw = u.pose.z + yaw_rate * dt;
        let finite_flag = select(
            0.0,
            1.0,
            finite_scalar(px) &&
                finite_scalar(pz) &&
                finite_scalar(vx) &&
                finite_scalar(vz) &&
                finite_scalar(yaw) &&
                finite_scalar(yaw_rate) &&
                finite_scalar(total_fx) &&
                finite_scalar(total_fz) &&
                finite_scalar(total_mz),
        );
        yaw_out[0] = vec4<f32>(px, pz, vx, vz);
        yaw_out[1] = vec4<f32>(yaw, yaw_rate, total_mz, finite_flag);
        yaw_out[2] = vec4<f32>(total_fx, total_fz, 0.0, 0.0);
    }
}

@compute @workgroup_size(WG)
fn integrate_planar_state_from_contact_reduction_outputs(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    let n = min(u.counts.x, WG);
    var fx_sum = 0.0;
    var fz_sum = 0.0;
    var mz = 0.0;
    if (i < n) {
        let off = wheel_offsets[i].xyz;
        // tire_wrenches is a packed TireFemContactReductionOutput array:
        // row + 2 = world_force.xyz,
        // row + 3 = world_moment.xyz (yaw uses y/up moment).
        let base = i * 4u;
        let world_force = tire_wrenches[base + 2u];
        let world_moment = tire_wrenches[base + 3u];
        let tire_mz = world_moment.y;
        fx_sum = world_force.x;
        fz_sum = world_force.z;
        mz = off.x * world_force.z - off.z * world_force.x + tire_mz;
    }
    partial_fx[i] = fx_sum;
    partial_fz[i] = fz_sum;
    partial_mz[i] = mz;
    workgroupBarrier();

    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (i < stride) {
            partial_fx[i] = partial_fx[i] + partial_fx[i + stride];
            partial_fz[i] = partial_fz[i] + partial_fz[i + stride];
            partial_mz[i] = partial_mz[i] + partial_mz[i + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (i == 0u) {
        let state = resident_rows[0];
        let yaw_state = resident_rows[1];
        let mass = max(u.params0.x, 1.0e-6);
        let inertia = max(u.params0.y, 1.0e-6);
        let dt = max(u.params0.z, 0.0);
        let linear_damping = max(u.params0.w, 0.0);
        let yaw_damping = max(u.params1.x, 0.0);
        let total_fx = partial_fx[0];
        let total_fz = partial_fz[0];
        let total_mz = partial_mz[0];
        let vx =
            (mass * state.z + total_fx * dt) / max(mass + linear_damping * dt, 1.0e-6);
        let vz =
            (mass * state.w + total_fz * dt) / max(mass + linear_damping * dt, 1.0e-6);
        let px = state.x + vx * dt;
        let pz = state.y + vz * dt;
        let yaw_rate =
            (inertia * yaw_state.y + total_mz * dt) / max(inertia + yaw_damping * dt, 1.0e-6);
        let yaw = yaw_state.x + yaw_rate * dt;
        let finite_flag = select(
            0.0,
            1.0,
            finite_scalar(px) &&
                finite_scalar(pz) &&
                finite_scalar(vx) &&
                finite_scalar(vz) &&
                finite_scalar(yaw) &&
                finite_scalar(yaw_rate) &&
                finite_scalar(total_fx) &&
                finite_scalar(total_fz) &&
                finite_scalar(total_mz),
        );
        yaw_out[0] = vec4<f32>(px, pz, vx, vz);
        yaw_out[1] = vec4<f32>(yaw, yaw_rate, total_mz, finite_flag);
        yaw_out[2] = vec4<f32>(total_fx, total_fz, 0.0, 0.0);
    }
}

@compute @workgroup_size(WG)
fn integrate_planar_state_from_feedback_force_rows(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tire = lid.x;
    let n = min(u.counts.x, WG);
    var fx_sum = 0.0;
    var fz_sum = 0.0;
    var mz = 0.0;
    if (tire < n && tire < arrayLength(&vertical_rows)) {
        let off = wheel_offsets[tire].xyz;
        // vertical_rows[tire].xy = node_offset, node_count converted from
        // TireFemGpuBatchAtlasOffsetRow::node. tire_wrenches contains one vec4
        // force row per FEM atlas node from TireFemContactFeedbackOutput.
        let atlas = vertical_rows[tire];
        let start = u32(max(atlas.x, 0.0));
        let count = u32(max(atlas.y, 0.0));
        let end = min(start + count, arrayLength(&tire_wrenches));
        var force = vec3<f32>(0.0, 0.0, 0.0);
        var i = start;
        loop {
            if (i >= end) {
                break;
            }
            force = force + tire_wrenches[i].xyz;
            i = i + 1u;
        }
        fx_sum = force.x;
        fz_sum = force.z;
        mz = off.x * force.z - off.z * force.x;
    }
    partial_fx[tire] = fx_sum;
    partial_fz[tire] = fz_sum;
    partial_mz[tire] = mz;
    workgroupBarrier();

    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tire < stride) {
            partial_fx[tire] = partial_fx[tire] + partial_fx[tire + stride];
            partial_fz[tire] = partial_fz[tire] + partial_fz[tire + stride];
            partial_mz[tire] = partial_mz[tire] + partial_mz[tire + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (tire == 0u) {
        let state = resident_rows[0];
        let yaw_state = resident_rows[1];
        let mass = max(u.params0.x, 1.0e-6);
        let inertia = max(u.params0.y, 1.0e-6);
        let dt = max(u.params0.z, 0.0);
        let linear_damping = max(u.params0.w, 0.0);
        let yaw_damping = max(u.params1.x, 0.0);
        let total_fx = partial_fx[0];
        let total_fz = partial_fz[0];
        let total_mz = partial_mz[0];
        let vx =
            (mass * state.z + total_fx * dt) / max(mass + linear_damping * dt, 1.0e-6);
        let vz =
            (mass * state.w + total_fz * dt) / max(mass + linear_damping * dt, 1.0e-6);
        let px = state.x + vx * dt;
        let pz = state.y + vz * dt;
        let yaw_rate =
            (inertia * yaw_state.y + total_mz * dt) / max(inertia + yaw_damping * dt, 1.0e-6);
        let yaw = yaw_state.x + yaw_rate * dt;
        let finite_flag = select(
            0.0,
            1.0,
            finite_scalar(px) &&
                finite_scalar(pz) &&
                finite_scalar(vx) &&
                finite_scalar(vz) &&
                finite_scalar(yaw) &&
                finite_scalar(yaw_rate) &&
                finite_scalar(total_fx) &&
                finite_scalar(total_fz) &&
                finite_scalar(total_mz),
        );
        yaw_out[0] = vec4<f32>(px, pz, vx, vz);
        yaw_out[1] = vec4<f32>(yaw, yaw_rate, total_mz, finite_flag);
        yaw_out[2] = vec4<f32>(total_fx, total_fz, 0.0, 0.0);
    }
}

@compute @workgroup_size(WG)
fn integrate_vertical_state_from_suspension(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    let n = min(u.counts.x, WG);
    var fy_sum = 0.0;
    if (i < n) {
        // tire_wrenches[i] is resident suspension row:
        // x=normal_load_n, y=travel_m, z=travel_rate_m_s, w=finite.
        fy_sum = max(tire_wrenches[i].x, 0.0);
    }
    partial_fy[i] = fy_sum;
    workgroupBarrier();

    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (i < stride) {
            partial_fy[i] = partial_fy[i] + partial_fy[i + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (i == 0u) {
        let state = resident_rows[0];
        let mass = max(u.params0.x, 1.0e-6);
        let gravity = max(u.params0.y, 0.0);
        let dt = max(u.params0.z, 0.0);
        let damping = max(u.params0.w, 0.0);
        let force_y = partial_fy[0] - mass * gravity;
        let vy = (mass * state.y + force_y * dt) / max(mass + damping * dt, 1.0e-6);
        let y = state.x + vy * dt;
        let finite_flag = select(
            0.0,
            1.0,
            finite_scalar(y) && finite_scalar(vy) && finite_scalar(force_y),
        );
        yaw_out[0] = vec4<f32>(y, vy, force_y, finite_flag);
    }
}

@compute @workgroup_size(WG)
fn integrate_wheel_spin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = min(u.counts.x, WG);
    if (i >= n) {
        return;
    }

    // Reuse binding names for this entry:
    // wheel_offsets[i]: x=spin_rad_s, y=wheel_radius_m,
    //                   z=spin_inertia_kg_m2, w=reserved.
    // tire_wrenches[i]: x=drive_torque_nm, y=brake_torque_nm,
    //                   z=longitudinal_tire_force_n, w=reserved.
    let wheel = wheel_offsets[i];
    let command = tire_wrenches[i];
    let spin = wheel.x;
    let radius = max(wheel.y, 1.0e-6);
    let inertia = max(wheel.z, 1.0e-6);
    let dt = max(u.params0.z, 0.0);
    let drive = command.x;
    let brake_torque = max(command.y, 0.0);
    let tire_fx = command.z;
    let brake = brake_torque * tanh(spin);
    let net_torque = drive - brake - tire_fx * radius;
    let accel = net_torque / inertia;
    let next_spin = spin + accel * dt;
    let finite_flag = select(
        0.0,
        1.0,
        finite_scalar(next_spin) &&
            finite_scalar(accel) &&
            finite_scalar(net_torque) &&
            radius > 0.0,
    );
    yaw_out[i] = vec4<f32>(next_spin, accel, net_torque, finite_flag);
}

@compute @workgroup_size(WG)
fn integrate_wheel_spin_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = min(u.counts.x, WG);
    if (i >= n) {
        return;
    }

    // Same equation as integrate_wheel_spin, but the output row stays
    // state-compatible for ping-pong resident wheel-spin buffers:
    // x=next_spin_rad_s, y=wheel_radius_m, z=spin_inertia_kg_m2, w=finite.
    let wheel = wheel_offsets[i];
    let command = tire_wrenches[i];
    let spin = wheel.x;
    let radius = max(wheel.y, 1.0e-6);
    let inertia = max(wheel.z, 1.0e-6);
    let dt = max(u.params0.z, 0.0);
    let drive = command.x;
    let brake_torque = max(command.y, 0.0);
    let tire_fx = command.z;
    let brake = brake_torque * tanh(spin);
    let net_torque = drive - brake - tire_fx * radius;
    let accel = net_torque / inertia;
    let next_spin = spin + accel * dt;
    let finite_flag = select(
        0.0,
        1.0,
        finite_scalar(next_spin) &&
            finite_scalar(accel) &&
            finite_scalar(net_torque) &&
            radius > 0.0,
    );
    yaw_out[i] = vec4<f32>(next_spin, radius, inertia, finite_flag);
}

@compute @workgroup_size(WG)
fn compute_wheel_kinematics(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = min(u.counts.x, WG);
    if (i >= n) {
        return;
    }

    // Reuse binding names for this entry:
    // wheel_offsets[i]: x=attach_x_m, y=attach_y_m,
    //                   z=attach_z_m, w=wheel_radius_m.
    // tire_wrenches[i]: x=spin_rad_s, y=steer_rad, zw=reserved.
    let mount = wheel_offsets[i];
    let wheel = tire_wrenches[i];
    let yaw = u.pose.z;
    let yaw_rate = u.velocity.z;
    let cy = cos(yaw);
    let sy = sin(yaw);
    let off_x = mount.x * cy - mount.z * sy;
    let off_z = mount.x * sy + mount.z * cy;
    let corner_vx = u.velocity.x - yaw_rate * off_z;
    let corner_vz = u.velocity.y + yaw_rate * off_x;
    let heading = yaw + wheel.y;
    let long_x = cos(heading);
    let long_z = sin(heading);
    let lat_x = -long_z;
    let lat_z = long_x;
    let v_long = corner_vx * long_x + corner_vz * long_z;
    let v_lat = corner_vx * lat_x + corner_vz * lat_z;
    let radius = max(mount.w, 1.0e-6);
    let denom = max(abs(v_long), 0.5);
    let slip_ratio = (wheel.x * radius - v_long) / denom;
    let slip_angle = atan(v_lat / denom);
    yaw_out[i] = vec4<f32>(v_long, v_lat, slip_ratio, slip_angle);
    yaw_out[n + i] = vec4<f32>(long_x, long_z, lat_x, lat_z);
}

@compute @workgroup_size(WG)
fn compute_wheel_kinematics_from_spin_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = min(u.counts.x, WG);
    if (i >= n) {
        return;
    }

    // wheel_offsets[i]: resident wheel-spin state
    //   x=spin_rad_s, y=wheel_radius_m, z=spin_inertia_kg_m2, w=finite.
    // tire_wrenches[i*2 + 0]: mount (attach_x, attach_y, attach_z, radius fallback)
    // tire_wrenches[i*2 + 1]: controls (steer_rad, reserved...)
    let spin_state = wheel_offsets[i];
    let mount = tire_wrenches[i * 2u + 0u];
    let controls = tire_wrenches[i * 2u + 1u];
    let yaw = u.pose.z;
    let yaw_rate = u.velocity.z;
    let cy = cos(yaw);
    let sy = sin(yaw);
    let off_x = mount.x * cy - mount.z * sy;
    let off_z = mount.x * sy + mount.z * cy;
    let corner_vx = u.velocity.x - yaw_rate * off_z;
    let corner_vz = u.velocity.y + yaw_rate * off_x;
    let heading = yaw + controls.x;
    let long_x = cos(heading);
    let long_z = sin(heading);
    let lat_x = -long_z;
    let lat_z = long_x;
    let v_long = corner_vx * long_x + corner_vz * long_z;
    let v_lat = corner_vx * lat_x + corner_vz * lat_z;
    let fallback_radius = max(mount.w, 1.0e-6);
    let radius = select(fallback_radius, spin_state.y, finite_scalar(spin_state.y) && spin_state.y >= 1.0e-6);
    let denom = max(abs(v_long), 0.5);
    let slip_ratio = (spin_state.x * radius - v_long) / denom;
    let slip_angle = atan(v_lat / denom);
    yaw_out[i] = vec4<f32>(v_long, v_lat, slip_ratio, slip_angle);
    yaw_out[n + i] = vec4<f32>(long_x, long_z, lat_x, lat_z);
}

@compute @workgroup_size(WG)
fn compute_suspension(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = min(u.counts.x, WG);
    if (i >= n) {
        return;
    }

    // Reuse binding names for this entry:
    // wheel_offsets[i]: x=attach_y_m, y=wheel_radius_m,
    //                   z=rest_length_m, w=max_compression_m.
    // tire_wrenches[i]: x=stiffness_n_m, y=damping_ns_m,
    //                   z=ground_height_m, w=reserved.
    let geo = wheel_offsets[i];
    let params = tire_wrenches[i];
    let chassis_y = u.pose.x;
    let chassis_vy = u.velocity.x;
    let radius = max(geo.y, 1.0e-6);
    let rest_length = max(geo.z, 0.0);
    let max_compression = max(geo.w, 0.0);
    let stiffness = max(params.x, 0.0);
    let damping = max(params.y, 0.0);
    let ground_height = params.z;
    let attach_world_y = chassis_y + geo.x;
    let hub_y = ground_height + radius;
    let compression = rest_length - (attach_world_y - hub_y);
    let travel = clamp(compression, 0.0, max_compression);
    let travel_rate = -chassis_vy;
    let normal_load = max(stiffness * travel + damping * travel_rate, 0.0);
    let finite_flag = select(
        0.0,
        1.0,
        finite_scalar(normal_load) && finite_scalar(travel) && finite_scalar(travel_rate),
    );
    yaw_out[i] = vec4<f32>(normal_load, travel, travel_rate, finite_flag);
}

@compute @workgroup_size(WG)
fn compute_suspension_from_vertical_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = min(u.counts.x, WG);
    if (i >= n) {
        return;
    }

    let geo = wheel_offsets[i];
    let params = tire_wrenches[i];
    let vertical = resident_rows[0];
    let chassis_y = vertical.x;
    let chassis_vy = vertical.y;
    let radius = max(geo.y, 1.0e-6);
    let rest_length = max(geo.z, 0.0);
    let max_compression = max(geo.w, 0.0);
    let stiffness = max(params.x, 0.0);
    let damping = max(params.y, 0.0);
    let ground_height = params.z;
    let attach_world_y = chassis_y + geo.x;
    let hub_y = ground_height + radius;
    let compression = rest_length - (attach_world_y - hub_y);
    let travel = clamp(compression, 0.0, max_compression);
    let travel_rate = -chassis_vy;
    let normal_load = max(stiffness * travel + damping * travel_rate, 0.0);
    let finite_flag = select(
        0.0,
        1.0,
        finite_scalar(normal_load) && finite_scalar(travel) && finite_scalar(travel_rate),
    );
    yaw_out[i] = vec4<f32>(normal_load, travel, travel_rate, finite_flag);
}

@compute @workgroup_size(WG)
fn build_brush_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = min(u.counts.x, WG);
    if (i >= n) {
        return;
    }

    // Reuse binding names for this entry:
    // wheel_offsets[i]: x=attach_x_m, y=attach_y_m,
    //                   z=attach_z_m, w=wheel_radius_m.
    // tire_wrenches[i*5 + 0]: wheel state (spin, steer, width, pressure)
    // tire_wrenches[i*5 + 1]: tire row
    // tire_wrenches[i*5 + 2]: material row
    // tire_wrenches[i*5 + 3]: extra row
    // tire_wrenches[i*5 + 4]: suspension/contact row
    // yaw_out[i*7..i*7+7] is BrushWheelInput as seven vec4 rows.
    let mount = wheel_offsets[i];
    let base_in = i * 5u;
    let wheel = tire_wrenches[base_in + 0u];
    let tire = tire_wrenches[base_in + 1u];
    let material = tire_wrenches[base_in + 2u];
    let extra = tire_wrenches[base_in + 3u];
    let suspension = tire_wrenches[base_in + 4u];
    let yaw = u.pose.w;
    let cy = cos(yaw);
    let sy = sin(yaw);
    let off_x = mount.x * cy - mount.z * sy;
    let off_z = mount.x * sy + mount.z * cy;
    let heading = yaw + wheel.y;
    let long_x = cos(heading);
    let long_z = sin(heading);
    let radius = max(mount.w, 1.0e-6);
    let travel = max(suspension.x, 0.0);
    let ground_height = suspension.z;
    let hub_y = ground_height + radius - travel;
    let hub_vx = u.velocity.x - u.velocity.w * off_z;
    let hub_vz = u.velocity.z + u.velocity.w * off_x;
    let base_out = i * 7u;
    yaw_out[base_out + 0u] = vec4<f32>(u.pose.x + off_x, hub_y, u.pose.z + off_z, 0.0);
    yaw_out[base_out + 1u] = vec4<f32>(hub_vx, u.velocity.y, hub_vz, 0.0);
    yaw_out[base_out + 2u] = vec4<f32>(long_x, 0.0, long_z, 0.0);
    yaw_out[base_out + 3u] = vec4<f32>(wheel.x, radius, wheel.z, wheel.w);
    yaw_out[base_out + 4u] = tire;
    yaw_out[base_out + 5u] = material;
    yaw_out[base_out + 6u] = extra;
}

@compute @workgroup_size(WG)
fn build_brush_rows_from_spin_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = min(u.counts.x, WG);
    if (i >= n) {
        return;
    }

    // wheel_offsets[i]: resident wheel-spin state
    //   x=spin_rad_s, y=wheel_radius_m, z=spin_inertia_kg_m2, w=finite.
    // tire_wrenches[i*6 + 0]: wheel mount (attach_x, attach_y, attach_z, radius fallback)
    // tire_wrenches[i*6 + 1]: wheel controls (steer, width, pressure, reserved)
    // tire_wrenches[i*6 + 2]: tire row
    // tire_wrenches[i*6 + 3]: material row
    // tire_wrenches[i*6 + 4]: extra row
    // tire_wrenches[i*6 + 5]: suspension/contact row
    let spin_state = wheel_offsets[i];
    let base_in = i * 6u;
    let mount = tire_wrenches[base_in + 0u];
    let controls = tire_wrenches[base_in + 1u];
    let tire = tire_wrenches[base_in + 2u];
    let material = tire_wrenches[base_in + 3u];
    let extra = tire_wrenches[base_in + 4u];
    let suspension = tire_wrenches[base_in + 5u];
    let yaw = u.pose.w;
    let cy = cos(yaw);
    let sy = sin(yaw);
    let off_x = mount.x * cy - mount.z * sy;
    let off_z = mount.x * sy + mount.z * cy;
    let heading = yaw + controls.x;
    let long_x = cos(heading);
    let long_z = sin(heading);
    let fallback_radius = max(mount.w, 1.0e-6);
    let radius = select(fallback_radius, spin_state.y, finite_scalar(spin_state.y) && spin_state.y >= 1.0e-6);
    let travel = max(suspension.x, 0.0);
    let ground_height = suspension.z;
    let hub_y = ground_height + radius - travel;
    let hub_vx = u.velocity.x - u.velocity.w * off_z;
    let hub_vz = u.velocity.z + u.velocity.w * off_x;
    let base_out = i * 7u;
    yaw_out[base_out + 0u] = vec4<f32>(u.pose.x + off_x, hub_y, u.pose.z + off_z, 0.0);
    yaw_out[base_out + 1u] = vec4<f32>(hub_vx, u.velocity.y, hub_vz, 0.0);
    yaw_out[base_out + 2u] = vec4<f32>(long_x, 0.0, long_z, 0.0);
    yaw_out[base_out + 3u] = vec4<f32>(spin_state.x, radius, controls.y, controls.z);
    yaw_out[base_out + 4u] = tire;
    yaw_out[base_out + 5u] = material;
    yaw_out[base_out + 6u] = extra;
}

@compute @workgroup_size(WG)
fn build_brush_rows_from_spin_and_suspension_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = min(u.counts.x, WG);
    if (i >= n) {
        return;
    }

    // wheel_offsets[i]: resident wheel-spin state
    //   x=spin_rad_s, y=wheel_radius_m, z=spin_inertia_kg_m2, w=finite.
    // tire_wrenches[i*6 + 0]: wheel mount (attach_x, attach_y, attach_z, radius fallback)
    // tire_wrenches[i*6 + 1]: wheel controls (steer, width, pressure, reserved)
    // tire_wrenches[i*6 + 2]: tire row
    // tire_wrenches[i*6 + 3]: material row
    // tire_wrenches[i*6 + 4]: extra row
    // tire_wrenches[i*6 + 5]: contact row (x/y ignored, z=ground_height_m)
    // resident_rows[i]: resident suspension row
    //   x=normal_load_n, y=travel_m, z=travel_rate_m_s, w=finite.
    let spin_state = wheel_offsets[i];
    let base_in = i * 6u;
    let mount = tire_wrenches[base_in + 0u];
    let controls = tire_wrenches[base_in + 1u];
    let tire = tire_wrenches[base_in + 2u];
    let material = tire_wrenches[base_in + 3u];
    let extra = tire_wrenches[base_in + 4u];
    let contact = tire_wrenches[base_in + 5u];
    let suspension = resident_rows[i];
    let yaw = u.pose.w;
    let cy = cos(yaw);
    let sy = sin(yaw);
    let off_x = mount.x * cy - mount.z * sy;
    let off_z = mount.x * sy + mount.z * cy;
    let heading = yaw + controls.x;
    let long_x = cos(heading);
    let long_z = sin(heading);
    let fallback_radius = max(mount.w, 1.0e-6);
    let radius = select(fallback_radius, spin_state.y, finite_scalar(spin_state.y) && spin_state.y >= 1.0e-6);
    let travel = max(suspension.y, 0.0);
    let ground_height = contact.z;
    let hub_y = ground_height + radius - travel;
    let hub_vx = u.velocity.x - u.velocity.w * off_z;
    let hub_vz = u.velocity.z + u.velocity.w * off_x;
    let base_out = i * 7u;
    yaw_out[base_out + 0u] = vec4<f32>(u.pose.x + off_x, hub_y, u.pose.z + off_z, 0.0);
    yaw_out[base_out + 1u] = vec4<f32>(hub_vx, u.velocity.y, hub_vz, 0.0);
    yaw_out[base_out + 2u] = vec4<f32>(long_x, 0.0, long_z, 0.0);
    yaw_out[base_out + 3u] = vec4<f32>(spin_state.x, radius, controls.y, controls.z);
    yaw_out[base_out + 4u] = tire;
    yaw_out[base_out + 5u] = material;
    yaw_out[base_out + 6u] = extra;
}

@compute @workgroup_size(WG)
fn build_brush_rows_from_spin_suspension_planar_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = min(u.counts.x, WG);
    if (i >= n) {
        return;
    }

    // wheel_offsets[i]: resident wheel-spin state
    //   x=spin_rad_s, y=wheel_radius_m, z=spin_inertia_kg_m2, w=finite.
    // tire_wrenches[i*6 + 0]: wheel mount (attach_x, attach_y, attach_z, radius fallback)
    // tire_wrenches[i*6 + 1]: wheel controls (steer, width, pressure, reserved)
    // tire_wrenches[i*6 + 2]: tire row
    // tire_wrenches[i*6 + 3]: material row
    // tire_wrenches[i*6 + 4]: extra row
    // tire_wrenches[i*6 + 5]: contact row (x/y ignored, z=ground_height_m)
    // resident_rows[i]: resident suspension row
    //   x=normal_load_n, y=travel_m, z=travel_rate_m_s, w=finite.
    // planar_rows[0]: resident chassis x/z/vx/vz
    // planar_rows[1]: resident chassis yaw/yaw_rate/mz/finite.
    // u.pose.y and u.velocity.y provide the vertical chassis state until the
    // vertical suspension owner is resident too.
    let spin_state = wheel_offsets[i];
    let base_in = i * 6u;
    let mount = tire_wrenches[base_in + 0u];
    let controls = tire_wrenches[base_in + 1u];
    let tire = tire_wrenches[base_in + 2u];
    let material = tire_wrenches[base_in + 3u];
    let extra = tire_wrenches[base_in + 4u];
    let contact = tire_wrenches[base_in + 5u];
    let suspension = resident_rows[i];
    let planar = planar_rows[0];
    let yaw_state = planar_rows[1];
    let yaw = yaw_state.x;
    let cy = cos(yaw);
    let sy = sin(yaw);
    let off_x = mount.x * cy - mount.z * sy;
    let off_z = mount.x * sy + mount.z * cy;
    let heading = yaw + controls.x;
    let long_x = cos(heading);
    let long_z = sin(heading);
    let fallback_radius = max(mount.w, 1.0e-6);
    let radius = select(fallback_radius, spin_state.y, finite_scalar(spin_state.y) && spin_state.y >= 1.0e-6);
    let travel = max(suspension.y, 0.0);
    let ground_height = contact.z;
    let hub_y = ground_height + radius - travel;
    let hub_vx = planar.z - yaw_state.y * off_z;
    let hub_vz = planar.w + yaw_state.y * off_x;
    let base_out = i * 7u;
    yaw_out[base_out + 0u] = vec4<f32>(planar.x + off_x, hub_y, planar.y + off_z, 0.0);
    yaw_out[base_out + 1u] = vec4<f32>(hub_vx, u.velocity.y, hub_vz, 0.0);
    yaw_out[base_out + 2u] = vec4<f32>(long_x, 0.0, long_z, 0.0);
    yaw_out[base_out + 3u] = vec4<f32>(spin_state.x, radius, controls.y, controls.z);
    yaw_out[base_out + 4u] = tire;
    yaw_out[base_out + 5u] = material;
    yaw_out[base_out + 6u] = extra;
}

@compute @workgroup_size(WG)
fn build_brush_rows_from_spin_suspension_chassis_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = min(u.counts.x, WG);
    if (i >= n) {
        return;
    }

    // Same packed row contract as build_brush_rows_from_spin_suspension_planar_state,
    // but reads both planar x/z/yaw and vertical y/vy from resident chassis
    // buffers. This is the first full resident chassis-state brush-row builder.
    let spin_state = wheel_offsets[i];
    let base_in = i * 6u;
    let mount = tire_wrenches[base_in + 0u];
    let controls = tire_wrenches[base_in + 1u];
    let tire = tire_wrenches[base_in + 2u];
    let material = tire_wrenches[base_in + 3u];
    let extra = tire_wrenches[base_in + 4u];
    let contact = tire_wrenches[base_in + 5u];
    let suspension = resident_rows[i];
    let planar = planar_rows[0];
    let yaw_state = planar_rows[1];
    let vertical = vertical_rows[0];
    let yaw = yaw_state.x;
    let cy = cos(yaw);
    let sy = sin(yaw);
    let off_x = mount.x * cy - mount.z * sy;
    let off_z = mount.x * sy + mount.z * cy;
    let heading = yaw + controls.x;
    let long_x = cos(heading);
    let long_z = sin(heading);
    let fallback_radius = max(mount.w, 1.0e-6);
    let radius = select(fallback_radius, spin_state.y, finite_scalar(spin_state.y) && spin_state.y >= 1.0e-6);
    let travel = max(suspension.y, 0.0);
    let ground_height = contact.z;
    let hub_y = ground_height + radius - travel;
    let hub_vx = planar.z - yaw_state.y * off_z;
    let hub_vz = planar.w + yaw_state.y * off_x;
    let base_out = i * 7u;
    yaw_out[base_out + 0u] = vec4<f32>(planar.x + off_x, hub_y, planar.y + off_z, 0.0);
    yaw_out[base_out + 1u] = vec4<f32>(hub_vx, vertical.y, hub_vz, 0.0);
    yaw_out[base_out + 2u] = vec4<f32>(long_x, 0.0, long_z, 0.0);
    yaw_out[base_out + 3u] = vec4<f32>(spin_state.x, radius, controls.y, controls.z);
    yaw_out[base_out + 4u] = tire;
    yaw_out[base_out + 5u] = material;
    yaw_out[base_out + 6u] = extra;
}
