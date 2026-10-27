// Featherstone forward dynamics on GPU — Articulated Body Algorithm
// (ABA) per Featherstone (2008) ch. 7. Mirrors the CPU path in
// `solvers/featherstone.rs::cpu_step`.
//
// Multi-DOF blocks use dense Cholesky when rigorous_d_solve is enabled;
// the alternate path uses sequential scalar elimination.
//
// Per-articulation parallelism: one workgroup invocation per chain.

struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,
};

struct Mat3Cols {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
};

struct Uniforms {
    dt:                 f32,
    joint_damping:      f32,
    joint_count:        u32,
    articulation_count: u32,
    /// Stage 17.1: 1 = rigorous dense-Cholesky D solve in pass 2/3.
    rigorous_d_solve:   u32,
    /// Stage 17.1: 1 = velocity-level Lagrange projection on Distance.
    rigorous_distance:  u32,
    drift_only:         u32,
    stable_motor_feedback: u32,
    external_scalar_bounds: u32,
    compensated_coordinates: u32, control_body_wrenches: u32, paired_rates: u32,
};

const JOINT_PRISMATIC: u32 = 0u;
const JOINT_REVOLUTE:  u32 = 1u;
const JOINT_BALL:      u32 = 2u;
const JOINT_FIXED:     u32 = 3u;
const JOINT_FREE:      u32 = 4u;
const JOINT_DISTANCE:  u32 = 5u;
const JOINT_D6:        u32 = 6u;
const JOINT_CABLE:     u32 = 7u;

const MAX_JOINTS_PER_ART: u32 = 32u;
const MAX_DOFS_PER_JOINT: u32 = 6u;
const MAX_FEEDBACK_ROWS: u32 = min(MAX_JOINTS_PER_ART * MAX_DOFS_PER_JOINT, 32u);

@group(0) @binding(0) var<storage, read> joint_type:         array<u32>;
@group(0) @binding(1) var<storage, read> joint_parent:       array<i32>;
@group(0) @binding(2) var<storage, read> joint_child:        array<i32>;
@group(0) @binding(3) var<storage, read> joint_x_p:          array<Transform>;
@group(0) @binding(4) var<storage, read> joint_x_c:          array<Transform>;
@group(0) @binding(5) var<storage, read> joint_axis:         array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> joint_coord_offset: array<u32>;
@group(0) @binding(7) var<storage, read> joint_dof_offset:   array<u32>;
@group(0) @binding(8) var<storage, read> joint_limit_lower:    array<f32>;
@group(0) @binding(9) var<storage, read> joint_limit_upper:    array<f32>;
@group(0) @binding(10) var<storage, read> joint_friction:      array<f32>;
@group(0) @binding(11) var<storage, read> joint_effort_limit:  array<f32>;
@group(0) @binding(12) var<storage, read> joint_velocity_limit: array<f32>;
struct Compliance {
    enabled: u32, padding: u32, lower: f32, upper: f32,
    natural_frequency: f32, damping_ratio: f32
}
@group(0) @binding(13) var<storage, read> joint_compliance: array<Compliance>;
@group(0) @binding(14) var<storage, read> joint_reversed: array<u32>;
@group(0) @binding(15) var<storage, read> joint_enabled: array<u32>;
struct JointLimitReference {
    bounds: vec4<f32>, full: vec4<f32>, full_low: vec4<f32>, half: vec4<f32>, half_low: vec4<f32>,
}
@group(0) @binding(16) var<storage, read> joint_limit_references: array<JointLimitReference>;
struct JointFrameReference {
    parent_position: vec4<f32>, parent_rotation: vec4<f32>,
    child_position: vec4<f32>, child_rotation: vec4<f32>, axis: vec4<f32>,
}
@group(0) @binding(17) var<storage, read> joint_frame_references: array<JointFrameReference>;
var<private> fk_pair_rounding: bool;
var<private> root_response_paired: bool;
var<private> root_response_slot: u32;
var<private> root_inverse_low: array<f32,36>;
var<private> fk_rotations: array<QuaternionPair, MAX_JOINTS_PER_ART>;

@group(1) @binding(0) var<storage, read>       body_mass:    array<f32>;
@group(1) @binding(1) var<storage, read>       body_inertia: array<Mat3Cols>;
@group(1) @binding(2) var<storage, read>       body_com:     array<vec4<f32>>;
@group(1) @binding(3) var<storage, read>       body_world:   array<i32>;
@group(1) @binding(4) var<storage, read>       gravity:      array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> body_q:       array<Transform>;
@group(1) @binding(6) var<storage, read_write> joint_q:      array<f32>;
@group(1) @binding(7) var<storage, read_write> joint_qd:     array<f32>;
// External wrench per body, at COM, world frame.
// Layout matches `GpuSpatialVector`: 2 × vec4 (linear+pad, angular+pad).
@group(1) @binding(8) var<storage, read>       body_f:       array<SpatialVecGpu>;
@group(1) @binding(9) var<storage, read_write> body_qd:      array<SpatialVecGpu>;
@group(1) @binding(10) var<storage, read> body_flags: array<u32>;
@group(1) @binding(11) var<storage, read> boundary_acceleration: array<SpatialVecGpu>;
@group(1) @binding(12) var<storage, read> body_inverse_inertia: array<Mat3Cols>;

struct SpatialVecGpu {
    linear:  vec4<f32>,
    angular: vec4<f32>,
};

@group(2) @binding(0) var<storage, read> joint_force:        array<f32>;
@group(2) @binding(1) var<storage, read> joint_target_mode:  array<u32>;
@group(2) @binding(2) var<storage, read> joint_target_pos:   array<f32>;
@group(2) @binding(3) var<storage, read> joint_target_vel:   array<f32>;
@group(2) @binding(4) var<storage, read> joint_target_kp:    array<f32>;
@group(2) @binding(5) var<storage, read> joint_target_kd:    array<f32>;
@group(2) @binding(6) var<storage, read> joint_motor_max_force: array<f32>;
@group(2) @binding(7) var<storage, read> control_body_forces: array<SpatialVecGpu>;
@group(2) @binding(8) var<storage, read> joint_motor_model: array<u32>;
@group(2) @binding(9) var<storage, read> joint_brake: array<f32>;

const TARGET_NONE: u32 = 0u;
const TARGET_POS:  u32 = 1u;
const TARGET_VEL:  u32 = 2u;
const TARGET_PV:   u32 = 3u;
const TARGET_EFF:  u32 = 4u;

fn clamp_or_passthrough(x: f32, limit: f32) -> f32 {
    if (limit != limit) { return x; }
    if (limit > 1e30) { return x; }
    return clamp(x, -limit, limit);
}

fn drive_torque(mode: u32, q: f32, qd: f32, kp: f32, kd: f32, pos_t: f32, vel_t: f32) -> f32 {
    if (mode == TARGET_POS) { return kp * (pos_t - q) - kd * qd; }
    if (mode == TARGET_VEL) { return kd * (vel_t - qd); }
    if (mode == TARGET_PV)  { return kp * (pos_t - q) + kd * (vel_t - qd); }
    if (mode == TARGET_EFF) { return pos_t; }
    return 0.0;
}

@group(3) @binding(0) var<storage, read> art_offsets: array<u32>;
@group(3) @binding(1) var<storage, read_write> stop_status: atomic<u32>;
@group(3) @binding(2) var<storage, read> active_articulations: array<u32>;
@group(3) @binding(3) var<storage, read_write> velocity_delta: array<f32>;
@group(3) @binding(4) var<storage, read_write> coordinate_error: array<atomic<u32>>;
@group(3) @binding(5) var<storage, read_write> position_error: array<atomic<u32>>;
@group(3) @binding(6) var<storage, read_write> paired_rates: array<vec2<f32>>;
@group(3) @binding(7) var<storage, read_write> paired_rate_delta: array<vec2<f32>>;
@group(3) @binding(8) var<storage, read_write> body_velocity_low: array<SpatialVecGpu>;
@group(4) @binding(0) var<uniform> uniforms: Uniforms;

// Stage 42 v3 + 39E — per-joint cable spring/damper + per-DOF muscle
// torque accumulator. The host zero-fills these buffers when the
// model has no cables / muscles so the same bind-group layout works
// for every scene.
@group(5) @binding(0) var<storage, read> joint_cable_k_lin:  array<f32>;
@group(5) @binding(1) var<storage, read> joint_cable_k_bend: array<f32>;
@group(5) @binding(2) var<storage, read> joint_cable_c_lin:  array<f32>;
@group(5) @binding(3) var<storage, read> joint_cable_c_bend: array<f32>;
@group(5) @binding(4) var<storage, read> joint_muscle_torque: array<f32>;

fn coordinate_round(coord: u32, value: f32) -> f32 {
    atomicStore(&coordinate_error[coord], bitcast<u32>(value));
    return bitcast<f32>(atomicLoad(&coordinate_error[coord]));
}
fn coordinate_sum(coord: u32, a: f32, b: f32) -> vec2<f32> {
    let sum = coordinate_round(coord, a + b);
    let virtual_b = coordinate_round(coord, sum - a);
    let virtual_a = coordinate_round(coord, sum - virtual_b);
    return vec2<f32>(sum, coordinate_round(coord, (a - virtual_a) + (b - virtual_b)));
}
fn coordinate_advance(coord: u32, increment: f32) -> f32 {
    let old = joint_q[coord];
    if (uniforms.compensated_coordinates == 0u) { return old + increment; }
    let step = coordinate_sum(coord, increment, bitcast<f32>(atomicLoad(&coordinate_error[coord])));
    let next = coordinate_sum(coord, old, step.x);
    let normalized = coordinate_sum(coord, next.x, next.y + step.y);
    atomicStore(&coordinate_error[coord], bitcast<u32>(normalized.y));
    return normalized.x;
}
fn coordinate_reset(coord: u32) {
    if (uniforms.compensated_coordinates != 0u) { atomicStore(&coordinate_error[coord], 0u); }
}

fn rounded(slot: u32, value: f32) -> f32 {
    if fk_pair_rounding { return position_round(slot,value); }
    return coordinate_round(slot,value);
}
fn two_sum(slot: u32, a: f32, b: f32) -> vec2<f32> {
    if fk_pair_rounding { return position_sum(slot,a,b); }
    return coordinate_sum(slot,a,b);
}
fn coordinate_limit_gap(coord: u32, dof: u32, upper: bool) -> f32 {
    let reference=joint_limit_references[dof];
    let bound=select(reference.bounds.x,reference.bounds.z,upper);
    if !xyz_chart_finite(bound) { return bound; }
    let saved=atomicLoad(&coordinate_error[coord]);
    let low=select(0.0,bitcast<f32>(saved),uniforms.compensated_coordinates!=0u);
    let difference=two_sum(coord,bound,-joint_q[coord]);
    let gap=rounded(coord,difference.x+(difference.y+select(reference.bounds.y,reference.bounds.w,upper)-low));
    atomicStore(&coordinate_error[coord],saved);
    return gap;
}
fn coordinate_limit_set(coord: u32, dof: u32, upper: bool) {
    let reference=joint_limit_references[dof];
    joint_q[coord]=select(reference.bounds.x,reference.bounds.z,upper);
    if uniforms.compensated_coordinates!=0u {
        atomicStore(&coordinate_error[coord],bitcast<u32>(select(reference.bounds.y,reference.bounds.w,upper)));
    }
}
fn coordinate_axis_reference(dof: u32, axis: u32) -> QuaternionPair {
    let reference=joint_limit_references[dof];
    var rotation=QuaternionPair(vec4<f32>(0.0,0.0,0.0,reference.half.y),vec4<f32>(0.0,0.0,0.0,reference.half_low.y));
    rotation.high[axis]=reference.half.x; rotation.low[axis]=reference.half_low.x;
    return rotation;
}
fn coordinate_rotation_point_set(j: u32, angular_offset: u32) {
    let coord=joint_coord_offset[j];
    let dof=joint_dof_offset[j]+angular_offset;
    let saved=atomicLoad(&coordinate_error[coord]);
    let x=coordinate_axis_reference(dof,0u);
    let y=coordinate_axis_reference(dof+1u,1u);
    let z=coordinate_axis_reference(dof+2u,2u);
    let rotation=normalize_pair(coord,multiply(coord,multiply(coord,x,y),z));
    atomicStore(&coordinate_error[coord],saved);
    for (var lane=0u; lane<4u; lane++) {
        joint_q[coord+angular_offset+lane]=rotation.high[lane];
        if uniforms.compensated_coordinates!=0u {
            atomicStore(&coordinate_error[coord+angular_offset+lane],bitcast<u32>(rotation.low[lane]));
        }
    }
}
fn coordinate_quaternion_advance(coord: u32, omega: vec3<f32>) -> vec4<f32> {
    var old = QuaternionPair(vec4<f32>(0.0),vec4<f32>(0.0));
    for (var lane = 0u; lane < 4u; lane++) {
        old.high[lane] = joint_q[coord+lane];
        old.low[lane] = bitcast<f32>(atomicLoad(&coordinate_error[coord+lane]));
    }
    var step = QuaternionPair(vec4<f32>(0.0),vec4<f32>(0.0));
    for (var lane = 0u; lane < 3u; lane++) {
        let half = 0.5*omega[lane];
        step.high[lane] = coordinate_round(coord,half*uniforms.dt);
        step.low[lane] = fma(half,uniforms.dt,-step.high[lane]);
    }
    let increment = multiply(coord,step,old);
    var advanced = QuaternionPair(vec4<f32>(0.0),vec4<f32>(0.0));
    for (var lane = 0u; lane < 4u; lane++) {
        let sum = coordinate_sum(coord,old.high[lane],increment.high[lane]);
        let normalized = coordinate_sum(coord,sum.x,sum.y+old.low[lane]+increment.low[lane]);
        advanced.high[lane] = normalized.x;
        advanced.low[lane] = normalized.y;
    }
    let result = normalize_pair(coord,advanced);
    for (var lane = 0u; lane < 4u; lane++) {
        atomicStore(&coordinate_error[coord+lane],bitcast<u32>(result.low[lane]));
    }
    return result.high;
}

// ─── Spatial primitives ─────────────────────────────────────────

fn spatial_cross_lin(va: vec3<f32>, wa: vec3<f32>, vb: vec3<f32>, wb: vec3<f32>) -> vec3<f32> {
    return cross(wa, vb) + cross(va, wb);
}
fn spatial_cross_ang(_va: vec3<f32>, wa: vec3<f32>, _vb: vec3<f32>, wb: vec3<f32>) -> vec3<f32> {
    return cross(wa, wb);
}
fn spatial_cross_dual_lin(_va: vec3<f32>, wa: vec3<f32>, vb: vec3<f32>, _wb: vec3<f32>) -> vec3<f32> {
    return cross(wa, vb);
}
fn spatial_cross_dual_ang(va: vec3<f32>, wa: vec3<f32>, vb: vec3<f32>, wb: vec3<f32>) -> vec3<f32> {
    return cross(wa, wb) + cross(va, vb);
}

fn smat6_mul_sv(m: array<f32, 36>, v_lin: vec3<f32>, v_ang: vec3<f32>)
    -> array<f32, 6>
{
    let x = array<f32, 6>(v_lin.x, v_lin.y, v_lin.z, v_ang.x, v_ang.y, v_ang.z);
    var out: array<f32, 6>;
    for (var i = 0u; i < 6u; i = i + 1u) {
        var s = 0.0;
        for (var j = 0u; j < 6u; j = j + 1u) {
            s = s + m[i * 6u + j] * x[j];
        }
        out[i] = s;
    }
    return out;
}

fn smat6_from_body(mass: f32, com: vec3<f32>, i_world: mat3x3<f32>) -> array<f32, 36> {
    var m: array<f32, 36>;
    for (var i = 0u; i < 36u; i = i + 1u) {
        m[i] = 0.0;
    }
    m[0u]  = mass; m[7u]  = mass; m[14u] = mass;
    let sx = mat3x3<f32>(
        vec3<f32>(0.0, com.z, -com.y),
        vec3<f32>(-com.z, 0.0, com.x),
        vec3<f32>(com.y, -com.x, 0.0),
    );
    let s2 = sx * sx;
    // Spatial inertia at the reference origin, ordered [linear; angular].
    m[3u]  = -mass * sx[0][0]; m[4u]  = -mass * sx[1][0]; m[5u]  = -mass * sx[2][0];
    m[9u]  = -mass * sx[0][1]; m[10u] = -mass * sx[1][1]; m[11u] = -mass * sx[2][1];
    m[15u] = -mass * sx[0][2]; m[16u] = -mass * sx[1][2]; m[17u] = -mass * sx[2][2];
    m[18u] = mass * sx[0][0]; m[19u] = mass * sx[1][0]; m[20u] = mass * sx[2][0];
    m[24u] = mass * sx[0][1]; m[25u] = mass * sx[1][1]; m[26u] = mass * sx[2][1];
    m[30u] = mass * sx[0][2]; m[31u] = mass * sx[1][2]; m[32u] = mass * sx[2][2];
    m[21u] = i_world[0][0] - mass * s2[0][0];
    m[22u] = i_world[1][0] - mass * s2[1][0];
    m[23u] = i_world[2][0] - mass * s2[2][0];
    m[27u] = i_world[0][1] - mass * s2[0][1];
    m[28u] = i_world[1][1] - mass * s2[1][1];
    m[29u] = i_world[2][1] - mass * s2[2][1];
    m[33u] = i_world[0][2] - mass * s2[0][2];
    m[34u] = i_world[1][2] - mass * s2[1][2];
    m[35u] = i_world[2][2] - mass * s2[2][2];
    return m;
}

// ─── Helpers ─────────────────────────────────────────────────────

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    );
}
fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

fn quat_from_axis_angle(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let h = 0.5 * angle;
    if abs(h) <= 0.25 {
        let squared = h*h;
        let sine = h*(1.0+squared*(-1.0/6.0+squared*(1.0/120.0+squared*(-1.0/5040.0+squared/362880.0))));
        let cosine = 1.0+squared*(-0.5+squared*(1.0/24.0+squared*(-1.0/720.0+squared*(1.0/40320.0-squared/3628800.0))));
        return vec4<f32>(axis*sine,cosine);
    }
    let s = sin(h);
    return vec4<f32>(axis * s, cos(h));
}
fn transform_compose(a: Transform, b: Transform) -> Transform {
    var r: Transform;
    r.rotation = normalize(quat_mul(a.rotation, b.rotation));
    r.position = vec4<f32>(a.position.xyz + quat_rotate(a.rotation, b.position.xyz), 0.0);
    return r;
}
fn transform_inverse(t: Transform) -> Transform {
    var r: Transform;
    r.rotation = quat_conj(t.rotation);
    r.position = vec4<f32>(-quat_rotate(r.rotation, t.position.xyz), 0.0);
    return r;
}

struct PositionPair { high: vec3<f32>, low: vec3<f32> }
override local_position_rounding: bool = false;
var<workgroup> position_rounding_scratch: array<atomic<u32>,64>;
var<private> position_rounding_lane: u32;
fn position_round(child: u32, value: f32) -> f32 {
    if local_position_rounding {
        atomicStore(&position_rounding_scratch[position_rounding_lane], bitcast<u32>(value));
        return bitcast<f32>(atomicAdd(&position_rounding_scratch[position_rounding_lane], 0u));
    }
    atomicStore(&position_error[4u * child + 3u], bitcast<u32>(value));
    return bitcast<f32>(atomicLoad(&position_error[4u * child + 3u]));
}
fn position_sum(child: u32, a: f32, b: f32) -> vec2<f32> {
    let sum = position_round(child, a + b);
    let vb = position_round(child, sum - a);
    let va = position_round(child, sum - vb);
    let error = position_round(child, (a - va) + (b - vb));
    return vec2<f32>(sum, error);
}
fn position_add(child: u32, a: PositionPair, b: PositionPair) -> PositionPair {
    var value: PositionPair;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let sum = position_sum(child, a.high[axis], b.high[axis]);
        let normalized = position_sum(child, sum.x, sum.y + a.low[axis] + b.low[axis]);
        value.high[axis] = normalized.x;
        value.low[axis] = normalized.y;
    }
    return value;
}
fn position_advance(child: u32, position: PositionPair, increment: vec3<f32>) -> PositionPair {
    var value: PositionPair;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let step = position_sum(child, increment[axis], position.low[axis]);
        let next = position_sum(child, position.high[axis], step.x);
        let normalized = position_sum(child, next.x, next.y + step.y);
        value.high[axis] = normalized.x;
        value.low[axis] = normalized.y;
    }
    return value;
}
fn position_rotate(child: u32, q: vec4<f32>, value: PositionPair) -> PositionPair {
    let delta = 2.0 * cross(q.xyz, cross(q.xyz, value.high) + q.w * value.high);
    return position_add(child, PositionPair(value.high, quat_rotate(q, value.low)),
        PositionPair(delta, vec3<f32>(0.0)));
}
fn position_residue(body: u32) -> vec3<f32> {
    return vec3<f32>(bitcast<f32>(atomicLoad(&position_error[4u * body])),
        bitcast<f32>(atomicLoad(&position_error[4u * body + 1u])),
        bitcast<f32>(atomicLoad(&position_error[4u * body + 2u])));
}
fn zero_position_rotation(q: QuaternionPair, value: PositionPair) -> bool {
    let magnitude=vec3<u32>(0x7fffffffu);
    let exponent=vec4<u32>(0x7f800000u);
    return all((bitcast<vec3<u32>>(value.high)&magnitude)==vec3<u32>(0u))
        && all((bitcast<vec3<u32>>(value.low)&magnitude)==vec3<u32>(0u))
        && all((bitcast<vec4<u32>>(q.high)&exponent)!=exponent)
        && all((bitcast<vec4<u32>>(q.low)&exponent)!=exponent);
}
fn position_pair_rotate(child: u32, q: QuaternionPair, value: PositionPair) -> PositionPair {
    if zero_position_rotation(q,value) { return PositionPair(vec3<f32>(0.0),vec3<f32>(0.0)); }
    let inverse=QuaternionPair(vec4<f32>(-q.high.xyz,q.high.w),vec4<f32>(-q.low.xyz,q.low.w));
    let vector=QuaternionPair(vec4<f32>(value.high,0.0),vec4<f32>(value.low,0.0));
    let result=multiply(child,multiply(child,q,vector),inverse);
    return PositionPair(result.high.xyz,result.low.xyz);
}
fn compensated_body_position(j: u32, parent: Transform) -> PositionPair {
    let child = u32(joint_child[j]);
    let coord = joint_coord_offset[j];
    let jt = joint_type[j];
    let reference=joint_frame_references[j];
    fk_pair_rounding=true;
    var inherited = vec3<f32>(0.0);
    if (joint_parent[j] >= 0) { inherited = position_residue(u32(joint_parent[j])); }
    var translation=PositionPair(vec3<f32>(0.0),vec3<f32>(0.0));
    if (jt == JOINT_PRISMATIC) {
        let value=joint_q[coord];
        let low=bitcast<f32>(atomicLoad(&coordinate_error[coord]));
        for (var axis=0u;axis<3u;axis++) {
            let high=rounded(child,joint_axis[j][axis]*value);
            let tail=fma(joint_axis[j][axis],value,-high)+joint_axis[j][axis]*low+reference.axis[axis]*value;
            let normalized=position_sum(child,high,tail);
            translation.high[axis]=normalized.x;translation.low[axis]=normalized.y;
        }
    } else if (jt == JOINT_FREE || jt == JOINT_DISTANCE || jt == JOINT_D6) {
        for (var axis = 0u; axis < 3u; axis += 1u) {
            translation.high[axis]=joint_q[coord+axis];
            translation.low[axis]=bitcast<f32>(atomicLoad(&coordinate_error[coord+axis]));
        }
    } else if (jt == JOINT_CABLE) {
        translation.high.z=joint_q[coord];
        translation.low.z=bitcast<f32>(atomicLoad(&coordinate_error[coord]));
    }
    var rotation=QuaternionPair(vec4<f32>(0.0,0.0,0.0,1.0),vec4<f32>(0.0));
    if jt==JOINT_REVOLUTE || jt==JOINT_CABLE {
        let offset=coord+select(0u,1u,jt==JOINT_CABLE);
        let axis=select(joint_axis[j].xyz,vec3<f32>(1.0,0.0,0.0),jt==JOINT_CABLE);
        let axis_low=select(reference.axis.xyz,vec3<f32>(0.0),jt==JOINT_CABLE);
        rotation=axis_angle_pair(child,axis,axis_low,joint_q[offset],bitcast<f32>(atomicLoad(&coordinate_error[offset])));
    } else if jt==JOINT_BALL || jt==JOINT_FREE || jt==JOINT_DISTANCE || jt==JOINT_D6 {
        let offset=coord+select(3u,0u,jt==JOINT_BALL);
        for (var lane=0u;lane<4u;lane++) {
            rotation.high[lane]=joint_q[offset+lane];
            rotation.low[lane]=bitcast<f32>(atomicLoad(&coordinate_error[offset+lane]));
        }
    }
    if joint_reversed[j]!=0u {
        rotation.high=vec4<f32>(-rotation.high.xyz,rotation.high.w);
        rotation.low=vec4<f32>(-rotation.low.xyz,rotation.low.w);
        translation=position_pair_rotate(child,rotation,PositionPair(-translation.high,-translation.low));
    }
    let xp = joint_x_p[j];
    let xc = joint_x_c[j];
    var parent_pair=QuaternionPair(parent.rotation,vec4<f32>(0.0));
    if joint_parent[j]>=0 && (body_flags[u32(joint_parent[j])]&3u)!=0u {
        parent_pair=normalize_pair(child,parent_pair);
    }
    if joint_parent[j]>=0 && (body_flags[u32(joint_parent[j])]&3u)==0u {
        for (var back=1u;back<=min(j,MAX_JOINTS_PER_ART);back++) {
            let previous=j-back;
            if joint_child[previous]==joint_parent[j] {
                parent_pair=fk_rotations[previous%MAX_JOINTS_PER_ART];
                break;
            }
        }
    }
    let parent_rotation=multiply(child,parent_pair,QuaternionPair(xp.rotation,reference.parent_rotation));
    let after_rotation=multiply(child,parent_rotation,rotation);
    let inverse=QuaternionPair(vec4<f32>(-xc.rotation.xyz,xc.rotation.w),
        vec4<f32>(-reference.child_rotation.xyz,reference.child_rotation.w));
    let body_rotation=normalize_pair(child,multiply(child,after_rotation,inverse));
    fk_rotations[j%MAX_JOINTS_PER_ART]=body_rotation;
    body_q[child].rotation=body_rotation.high;
    var result = position_add(child, PositionPair(parent.position.xyz, inherited),
        position_pair_rotate(child,parent_pair,PositionPair(xp.position.xyz,reference.parent_position.xyz)));
    result=position_add(child,result,position_pair_rotate(child,parent_rotation,translation));
    result=position_add(child,result,position_pair_rotate(child,body_rotation,
        PositionPair(-xc.position.xyz,-reference.child_position.xyz)));
    fk_pair_rounding=false;
    return result;
}
fn publish_compensated_position(j: u32, parent: Transform) {
    let child = u32(joint_child[j]);
    let position = compensated_body_position(j, parent);
    body_q[child].position = vec4<f32>(position.high, 0.0);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        atomicStore(&position_error[4u * child + axis], bitcast<u32>(position.low[axis]));
    }
    atomicStore(&position_error[4u * child + 3u], 0u);
}

fn advance_kinematic(child: u32) {
    let pose = body_q[child];
    let velocity = body_qd[child];
    let acceleration = boundary_acceleration[child];
    let dt = uniforms.dt;
    let angle = (velocity.angular.xyz + acceleration.angular.xyz * (0.5 * dt)) * dt;
    let magnitude = length(angle);
    var rotation = pose.rotation;
    if magnitude > 0.0 {
        rotation = normalize(quat_mul(quat_from_axis_angle(angle / magnitude, magnitude), rotation));
    }
    let displacement = (velocity.linear.xyz + acceleration.linear.xyz * (0.5 * dt)) * dt;
    let local_com = body_com[child].xyz;
    if uniforms.compensated_coordinates != 0u {
        var position = position_add(child, PositionPair(pose.position.xyz, position_residue(child)),
            position_rotate(child, pose.rotation, PositionPair(local_com, vec3<f32>(0.0))));
        position = position_advance(child, position, displacement);
        position = position_add(child, position,
            position_rotate(child, rotation, PositionPair(-local_com, vec3<f32>(0.0))));
        body_q[child].position = vec4<f32>(position.high, 0.0);
        for (var axis = 0u; axis < 3u; axis += 1u) {
            atomicStore(&position_error[4u * child + axis], bitcast<u32>(position.low[axis]));
        }
        atomicStore(&position_error[4u * child + 3u], 0u);
    } else {
        body_q[child].position = vec4<f32>(pose.position.xyz + quat_rotate(pose.rotation, local_com)
            + displacement - quat_rotate(rotation, local_com), 0.0);
    }
    body_q[child].rotation = rotation;
    body_qd[child] = SpatialVecGpu(vec4<f32>(velocity.linear.xyz + acceleration.linear.xyz * dt, 0.0),
        vec4<f32>(velocity.angular.xyz + acceleration.angular.xyz * dt, 0.0));
}
fn mat3_from_cols(m: Mat3Cols) -> mat3x3<f32> {
    return mat3x3<f32>(m.col0.xyz, m.col1.xyz, m.col2.xyz);
}
fn rotate_inertia(q: vec4<f32>, i_local: mat3x3<f32>) -> mat3x3<f32> {
    let r0 = quat_rotate(q, vec3<f32>(1.0, 0.0, 0.0));
    let r1 = quat_rotate(q, vec3<f32>(0.0, 1.0, 0.0));
    let r2 = quat_rotate(q, vec3<f32>(0.0, 0.0, 1.0));
    let r = mat3x3<f32>(r0, r1, r2);
    let rt = transpose(r);
    return r * i_local * rt;
}

fn paired_matrix_vector(slot: u32, matrix: mat3x3<f32>, vector: PositionPair) -> PositionPair {
    var result: PositionPair;
    for (var r=0u; r<3u; r++) {
        var sum=vec2<f32>(0.0);
        for (var c=0u; c<3u; c++) {
            sum=quaternion_scalar_add(slot,sum,quaternion_scalar_product(slot,
                vec2<f32>(matrix[c][r],0.0),vec2<f32>(vector.high[c],vector.low[c])));
        }
        result.high[r]=sum.x; result.low[r]=sum.y;
    }
    return result;
}

struct PairedMatrix { high: mat3x3<f32>, low: mat3x3<f32> }
fn paired_matrix_pair_vector(slot: u32, matrix: PairedMatrix, vector: PositionPair) -> PositionPair {
    return position_add(slot,paired_matrix_vector(slot,matrix.high,vector),paired_matrix_vector(slot,matrix.low,vector));
}

fn paired_cross(slot: u32, a: PositionPair, b: PositionPair) -> PositionPair {
    var result: PositionPair;
    for (var axis=0u; axis<3u; axis++) {
        let r=(axis+1u)%3u; let c=(axis+2u)%3u;
        let positive=quaternion_scalar_product(slot,vec2<f32>(a.high[r],a.low[r]),vec2<f32>(b.high[c],b.low[c]));
        let negative=quaternion_scalar_product(slot,vec2<f32>(a.high[c],a.low[c]),vec2<f32>(b.high[r],b.low[r]));
        let sum=quaternion_scalar_add(slot,positive,-negative);
        result.high[axis]=sum.x; result.low[axis]=sum.y;
    }
    return result;
}

fn conjugate_pair(q: QuaternionPair) -> QuaternionPair {
    return QuaternionPair(vec4<f32>(-q.high.xyz,q.high.w),vec4<f32>(-q.low.xyz,q.low.w));
}

// Body-to-motion rotation and COM arm for isolated quaternion roots.
fn root_motion_rotation(j: u32) -> QuaternionPair {
    let child=u32(joint_child[j]);
    let xc=QuaternionPair(joint_x_c[j].rotation,joint_frame_references[j].child_rotation);
    if joint_reversed[j]!=0u { return normalize_pair(child,conjugate_pair(xc)); }
    let offset=joint_coord_offset[j]+select(3u,0u,joint_type[j]==JOINT_BALL);
    var rotation: QuaternionPair;
    for (var lane=0u; lane<4u; lane++) {
        rotation.high[lane]=joint_q[offset+lane];
        rotation.low[lane]=select(0.0,bitcast<f32>(atomicLoad(&coordinate_error[offset+lane])),uniforms.compensated_coordinates!=0u);
    }
    return normalize_pair(child,multiply(child,rotation,conjugate_pair(xc)));
}

fn root_motion_arm(j: u32, rotation: QuaternionPair) -> PositionPair {
    let child=u32(joint_child[j]);
    let offset=PositionPair(body_com[child].xyz,vec3<f32>(0.0));
    let anchor=PositionPair(-joint_x_c[j].position.xyz,-joint_frame_references[j].child_position.xyz);
    var arm=position_pair_rotate(child,rotation,position_add(child,offset,anchor));
    if joint_reversed[j]!=0u {
        let coord=joint_coord_offset[j];
        var translation=PositionPair(vec3<f32>(0.0),vec3<f32>(0.0));
        for (var axis=0u; axis<3u; axis++) {
            translation.high[axis]=-joint_q[coord+axis];
            translation.low[axis]=select(0.0,-bitcast<f32>(atomicLoad(&coordinate_error[coord+axis])),uniforms.compensated_coordinates!=0u);
        }
        arm=position_add(child,arm,translation);
    }
    return arm;
}

fn root_motion_inverse(j: u32, rotation: QuaternionPair) -> PairedMatrix {
    let child=u32(joint_child[j]);
    let inverse=mat3_from_cols(body_inverse_inertia[child]);
    var result: PairedMatrix;
    for (var axis=0u; axis<3u; axis++) {
        var unit=vec3<f32>(0.0); unit[axis]=1.0;
        let local=position_pair_rotate(child,conjugate_pair(rotation),PositionPair(unit,vec3<f32>(0.0)));
        let response=position_pair_rotate(child,rotation,paired_matrix_vector(child,inverse,local));
        result.high[axis]=response.high; result.low[axis]=response.low;
    }
    return result;
}

fn root_pivot_inverse(j: u32, rotation: QuaternionPair, arm: PositionPair) -> PairedMatrix {
    if all(arm.high==vec3<f32>(0.0)) && all(arm.low==vec3<f32>(0.0)) {
        return root_motion_inverse(j,rotation);
    }
    let child=u32(joint_child[j]);
    var matrix: array<f32,36>;
    let inertia=mat3_from_cols(body_inertia[child]);
    for (var c=0u; c<3u; c++) {
        var unit=vec3<f32>(0.0); unit[c]=1.0;
        let local=position_pair_rotate(child,conjugate_pair(rotation),PositionPair(unit,vec3<f32>(0.0)));
        let column=position_pair_rotate(child,rotation,paired_matrix_vector(child,inertia,local));
        let moment=paired_cross(child,arm,paired_cross(child,arm,PositionPair(unit,vec3<f32>(0.0))));
        for (var r=0u; r<3u; r++) {
            let correction=quaternion_scalar_product(child,vec2<f32>(body_mass[child],0.0),vec2<f32>(moment.high[r],moment.low[r]));
            let value=quaternion_scalar_add(child,vec2<f32>(column.high[r],column.low[r]),-correction);
            matrix[r*6u+c]=value.x+value.y;
        }
    }
    let inverse=joint_psd_inverse(matrix,3u);
    if !inverse.valid {
        let invalid=vec3<f32>(bitcast<f32>(0x7fc00000u));
        return PairedMatrix(mat3x3<f32>(invalid,invalid,invalid),mat3x3<f32>());
    }
    var result: PairedMatrix;
    for (var c=0u; c<3u; c++) {
        for (var r=0u; r<3u; r++) { result.high[c][r]=inverse.matrix[r*6u+c]; }
    }
    return result;
}

/// Joint local transform (parent-anchor frame → child-anchor frame).
fn joint_local_transform(jt: u32, q_start: u32, axis: vec3<f32>) -> Transform {
    var t: Transform;
    t.position = vec4<f32>(0.0);
    t.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (jt == JOINT_REVOLUTE) {
        t.rotation = quat_from_axis_angle(axis, joint_q[q_start]);
    } else if (jt == JOINT_PRISMATIC) {
        t.position = vec4<f32>(axis * joint_q[q_start], 0.0);
    } else if (jt == JOINT_BALL) {
        t.rotation = normalize(vec4<f32>(joint_q[q_start], joint_q[q_start+1u], joint_q[q_start+2u], joint_q[q_start+3u]));
    } else if (jt == JOINT_FREE || jt == JOINT_DISTANCE || jt == JOINT_D6) {
        t.position = vec4<f32>(joint_q[q_start], joint_q[q_start+1u], joint_q[q_start+2u], 0.0);
        t.rotation = normalize(vec4<f32>(joint_q[q_start+3u], joint_q[q_start+4u], joint_q[q_start+5u], joint_q[q_start+6u]));
    } else if (jt == JOINT_CABLE) {
        // 2 coords: q_l linear stretch along joint-local +z, q_b bend
        // angle about joint-local +x. Translate-then-rotate.
        let q_l = joint_q[q_start];
        let q_b = joint_q[q_start + 1u];
        t.position = vec4<f32>(0.0, 0.0, q_l, 0.0);
        t.rotation = quat_from_axis_angle(vec3<f32>(1.0, 0.0, 0.0), q_b);
    }
    return t;
}

/// DOF count per joint type. Mirrors `JointType::dof_count`.
fn tree_local_transform(j: u32) -> Transform {
    let authored = joint_local_transform(joint_type[j],joint_coord_offset[j],joint_axis[j].xyz);
    if joint_reversed[j] != 0u { return transform_inverse(authored); }
    return authored;
}

fn motion_frame(j: u32, parent_anchor: Transform, child_anchor: Transform) -> Transform {
    var frame = parent_anchor;
    if joint_reversed[j] != 0u { frame = child_anchor; }
    let kind = joint_type[j];
    if kind == JOINT_FREE || kind == JOINT_DISTANCE || kind == JOINT_D6 {
        frame.position = child_anchor.position;
        if joint_reversed[j] != 0u { frame.position = parent_anchor.position; }
    }
    return frame;
}

fn motion_arm(j: u32, frame_rotation: vec4<f32>) -> vec3<f32> {
    let child = u32(joint_child[j]);
    var arm = quat_rotate(body_q[child].rotation,body_com[child].xyz-joint_x_c[j].position.xyz);
    let kind = joint_type[j];
    if joint_reversed[j] != 0u && (kind == JOINT_FREE || kind == JOINT_DISTANCE || kind == JOINT_D6) {
        let coord = joint_coord_offset[j];
        arm -= quat_rotate(frame_rotation,vec3<f32>(joint_q[coord],joint_q[coord+1u],joint_q[coord+2u]));
    }
    return arm;
}

fn joint_dof_n(jt: u32) -> u32 {
    if (jt == JOINT_REVOLUTE || jt == JOINT_PRISMATIC) { return 1u; }
    if (jt == JOINT_BALL) { return 3u; }
    if (jt == JOINT_FREE || jt == JOINT_DISTANCE || jt == JOINT_D6) { return 6u; }
    if (jt == JOINT_CABLE) { return 2u; }
    return 0u; // FIXED
}

fn motion_axis_point(j: u32, axis: u32) -> bool {
    return (joint_enabled[j]&(1u<<(axis+1u)))!=0u;
}
fn d6_angular_lock_count(j: u32) -> u32 {
    return countOneBits((joint_enabled[j]>>4u)&7u);
}

/// Decompose a quaternion into intrinsic XYZ angles, with rotation Rx * Ry * Rz.
fn quat_to_euler_xyz(q: vec4<f32>) -> vec3<f32> {
    let xx = q.x * q.x;
    let yy = q.y * q.y;
    let zz = q.z * q.z;
    let ww = q.w * q.w;
    let xy = q.x * q.y;
    let xz = q.x * q.z;
    let yz = q.y * q.z;
    let wx = q.w * q.x;
    let wy = q.w * q.y;
    let wz = q.w * q.z;

    let m00 = ww + xx - yy - zz;
    let m01 = 2.0 * (xy - wz);
    let m02 = 2.0 * (xz + wy);
    let m12 = 2.0 * (yz - wx);
    let m22 = ww - xx - yy + zz;

    let cy = length(vec2<f32>(m00, m01));
    let ey = atan2(m02, cy);
    var ex = 0.0;
    var ez = 0.0;
    if (cy > 1.9073486328125e-6) {
        ex = atan2(-m12, m22);
        ez = atan2(-m01, m00);
    } else {
        // Set the Z angle to zero at gimbal lock.
        ex = atan2(2.0 * (yz + wx), ww - xx + yy - zz);
        ez = 0.0;
    }
    return vec3<f32>(ex, ey, ez);
}

fn quat_from_euler_xyz(e: vec3<f32>) -> vec4<f32> {
    let rx = quat_from_axis_angle(vec3<f32>(1.0, 0.0, 0.0), e.x);
    let ry = quat_from_axis_angle(vec3<f32>(0.0, 1.0, 0.0), e.y);
    let rz = quat_from_axis_angle(vec3<f32>(0.0, 0.0, 1.0), e.z);
    return normalize(quat_mul(quat_mul(rx, ry), rz));
}

fn bounded_euler_xyz(q: vec4<f32>, dof: u32, locked_only: bool) -> vec3<f32> {
    let angles=quat_to_euler_xyz(q);
    let cy=length(vec2<f32>(q.w*q.w+q.x*q.x-q.y*q.y-q.z*q.z,2.0*(q.x*q.y-q.w*q.z)));
    var bounds=vec4<f32>(joint_limit_lower[dof+3u],joint_limit_upper[dof+3u],joint_limit_lower[dof+5u],joint_limit_upper[dof+5u]);
    var pitch_bounds=vec2<f32>(joint_limit_lower[dof+4u],joint_limit_upper[dof+4u]);
    if !locked_only {
        for (var axis=0u;axis<2u;axis++) {
            let c=joint_compliance[dof+3u+2u*axis];
            if c.enabled!=0u { bounds[2u*axis]=c.lower; bounds[2u*axis+1u]=c.upper; }
        }
        let pitch=joint_compliance[dof+4u];
        if pitch.enabled!=0u { pitch_bounds=vec2<f32>(pitch.lower,pitch.upper); }
    }
    if cy>1.9073486328125e-6 {
        return xyz_regular_chart(angles,vec3<f32>(bounds.x,pitch_bounds.x,bounds.z),vec3<f32>(bounds.y,pitch_bounds.y,bounds.w)).xyz;
    }
    var chart=xyz_singular_chart(angles.x,angles.y,bounds).angles;
    chart.y=xyz_interval_angle(chart.y,pitch_bounds.x,pitch_bounds.y);
    return chart;
}

fn motor_coordinate(joint: u32, axis: u32) -> f32 {
    let kind = joint_type[joint];
    let coord = joint_coord_offset[joint];
    if kind == JOINT_PRISMATIC || kind == JOINT_REVOLUTE || kind == JOINT_CABLE {
        return joint_q[coord + axis];
    }
    var offset = coord;
    var angular_axis = axis;
    if kind == JOINT_FREE || kind == JOINT_DISTANCE || kind == JOINT_D6 {
        if axis < 3u { return joint_q[coord + axis]; }
        offset += 3u;
        angular_axis -= 3u;
    } else if kind != JOINT_BALL {
        return 0.0;
    }
    let rotation = vec4<f32>(joint_q[offset], joint_q[offset+1u], joint_q[offset+2u], joint_q[offset+3u]);
    if kind==JOINT_D6 { return bounded_euler_xyz(rotation,joint_dof_offset[joint],false)[angular_axis]; }
    return quat_to_euler_xyz(rotation)[angular_axis];
}

// ─── Kernel ────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn featherstone_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    position_rounding_lane=lane;
    root_response_paired=false;
    mixed_velocity_enabled=false;
    if atomicLoad(&stop_status) != 0u { return; }
    let art = gid.x;
    if (art >= uniforms.articulation_count) {
        return;
    }
    if active_articulations[art] == 0u { return; }
    let j_begin = art_offsets[art];
    let j_end   = art_offsets[art + 1u];
    let n = j_end - j_begin;
    if n == 0u { return; }
    if (n > MAX_JOINTS_PER_ART) {
        atomicOr(&stop_status, 32u);
        return;
    }
    let spatial_origin = body_q[u32(joint_child[j_begin])].position.xyz;
    let position_only = uniforms.drift_only == 2u;
    let refresh_only = uniforms.drift_only == 3u;
    if uniforms.paired_rates!=0u {
        for (var j=j_begin;j<j_end;j++) {
            if !body_rate_input_valid(u32(joint_child[j])) { atomicOr(&stop_status,32u); return; }
            if joint_parent[j]>=0 && !body_rate_input_valid(u32(joint_parent[j])) { atomicOr(&stop_status,32u); return; }
        }
        for (var d=joint_dof_offset[j_begin]; d<joint_dof_offset[j_end]; d++) {
            let rate=paired_rates[d];
            if any((bitcast<vec2<u32>>(rate) & vec2<u32>(0x7f800000u))==vec2<u32>(0x7f800000u))
                || rate.x+rate.y!=rate.x {
                atomicOr(&stop_status,32u); return;
            }
        }
        for (var d=joint_dof_offset[j_begin]; d<joint_dof_offset[j_end]; d++) {
            joint_qd[d]=paired_rates[d].x;
        }
    }
    if position_only {
        var moved = false;
        for (var j = j_begin; j < j_end; j++) {
            for (var d = joint_dof_offset[j]; d < joint_dof_offset[j+1u]; d++) {
                let delta = velocity_delta[d];
                if (bitcast<u32>(delta) & 0x7f800000u) == 0x7f800000u {
                    atomicOr(&stop_status, 0x80000000u);
                    return;
                }
                moved = moved || (delta != 0.0 && joint_enabled[j]!=0u && (body_flags[u32(joint_child[j])] & 3u) == 0u);
            }
        }
        if !moved { return; }
    }
    var has_stops = false;
    var mixed_stops = false;
    for (var j = j_begin; j < j_end; j += 1u) {
        if joint_type[j]==JOINT_BALL && joint_enabled[j]!=0u && (body_flags[u32(joint_child[j])] & 3u)==0u {
            let dof=joint_dof_offset[j];
            let fixed=motion_axis_point(j,0u) && motion_axis_point(j,1u) && motion_axis_point(j,2u);
            for (var axis=0u; axis<3u; axis++) {
                mixed_stops=mixed_stops || (!fixed && (xyz_chart_finite(joint_limit_lower[dof+axis])
                    || xyz_chart_finite(joint_limit_upper[dof+axis]) || xyz_chart_finite(joint_velocity_limit[dof+axis])));
            }
        }
        if joint_type[j]==JOINT_DISTANCE && uniforms.rigorous_distance==1u
            && joint_enabled[j]!=0u && (body_flags[u32(joint_child[j])] & 3u)==0u {
            let dof=joint_dof_offset[j];
            mixed_stops=mixed_stops || xyz_chart_finite(joint_limit_lower[dof]) || xyz_chart_finite(joint_limit_upper[dof]);
            for (var axis=0u; axis<6u; axis++) { mixed_stops=mixed_stops || xyz_chart_finite(joint_velocity_limit[dof+axis]); }
        }
        if joint_type[j]==JOINT_D6 && joint_enabled[j]!=0u && (body_flags[u32(joint_child[j])] & 3u)==0u {
            let count=d6_angular_lock_count(j);
            let dof=joint_dof_offset[j];
            for (var axis=0u; axis<6u; axis++) {
                let structural=motion_axis_point(j,axis) && (axis<3u || count==3u);
                mixed_stops=mixed_stops || (!structural && (xyz_chart_finite(joint_limit_lower[dof+axis])
                    || xyz_chart_finite(joint_limit_upper[dof+axis]) || xyz_chart_finite(joint_velocity_limit[dof+axis])));
            }
        }
        if joint_enabled[j]!=0u && (joint_type[j] == JOINT_REVOLUTE || joint_type[j] == JOINT_PRISMATIC) {
            let dof = joint_dof_offset[j];
            has_stops = has_stops || abs(joint_limit_lower[dof]) < 1e30 || abs(joint_limit_upper[dof]) < 1e30;
        }
    }
    var has_compliance = false;
    for (var j = j_begin; j < j_end; j++) {
        if joint_enabled[j]==0u { continue; }
        for (var d = joint_dof_offset[j]; d < joint_dof_offset[j+1u]; d++) {
            has_compliance = has_compliance || joint_compliance[d].enabled != 0u;
        }
    }
    let rigorous = uniforms.rigorous_d_solve == 1u || has_stops || has_compliance || mixed_stops;
    let local_scalar_bounds = uniforms.external_scalar_bounds!=0u || mixed_stops;
    has_stops = has_stops && uniforms.external_scalar_bounds == 0u && !position_only && !refresh_only;

    // Per-joint scratch.
    var v_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var v_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var c_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var c_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var i_a:      array<array<f32, 36>, MAX_JOINTS_PER_ART>;
    var p_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var p_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var a_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var a_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var n_dof_arr: StopCounts;

    // Per-joint × per-DOF scratch.
    // S_lin[k][d] / S_ang[k][d] is the d-th column of joint k's
    // motion subspace in world frame.
    var s_lin: StopVectors;
    var s_ang: StopVectors;
    var u_lin: StopVectors;
    var u_ang: StopVectors;
    var d_inv_arr: array<array<f32, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
    var d_dense_inv: StopBlocks;
    var u_tau_arr: array<array<f32, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
    var dof_active_arr: array<array<u32, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
    var q_dd_arr: StopScalars;
    var root_applied: array<f32,6>;

    if uniforms.drift_only == 0u || has_stops
        || (mixed_stops && uniforms.external_scalar_bounds==0u && !position_only && !refresh_only) {
    // ── Pass 1 — FK + body kinematics + motion subspace ─────────
    for (var k = 0u; k < n; k = k + 1u) {
        let j = j_begin + k;
        let parent = joint_parent[j];
        let child  = u32(joint_child[j]);
        let coord  = joint_coord_offset[j];
        let dof    = joint_dof_offset[j];
        let jt     = joint_type[j];
        let axis_l = joint_axis[j].xyz;

        var parent_world: Transform;
        if (parent < 0) {
            parent_world.position = vec4<f32>(0.0);
            parent_world.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        } else {
            parent_world = body_q[u32(parent)];
        }

        let local = tree_local_transform(j);
        let xp = joint_x_p[j];
        let xc_inv = transform_inverse(joint_x_c[j]);
        let xpw = transform_compose(parent_world, xp);
        let after = transform_compose(xpw, local);
        if (body_flags[child] & 3u) == 0u {
            body_q[child] = transform_compose(after, xc_inv);
            if (uniforms.compensated_coordinates != 0u) { publish_compensated_position(j, parent_world); }
        }

        // World-oriented motion columns about the articulation reference origin.
        let frame = motion_frame(j,xpw,after);
        let r_jw = frame.rotation;
        var p_jw = frame.position.xyz;
        p_jw -= spatial_origin;
        let ex = quat_rotate(r_jw, vec3<f32>(1.0, 0.0, 0.0));
        let ey = quat_rotate(r_jw, vec3<f32>(0.0, 1.0, 0.0));
        let ez = quat_rotate(r_jw, vec3<f32>(0.0, 0.0, 1.0));

        // Initialise all columns to zero / inactive.
        for (var d = 0u; d < MAX_DOFS_PER_JOINT; d = d + 1u) {
            s_lin[k][d] = vec3<f32>(0.0);
            s_ang[k][d] = vec3<f32>(0.0);
            dof_active_arr[k][d] = 0u;
        }

        var n_dof = 0u;
        if (jt == JOINT_REVOLUTE) {
            let axis_w = normalize(quat_rotate(r_jw, axis_l));
            let lo = joint_limit_lower[dof];
            let hi = joint_limit_upper[dof];
            if !motion_axis_point(j,0u) {
                s_lin[k][0u] = cross(axis_w, -p_jw);
                s_ang[k][0u] = axis_w;
                dof_active_arr[k][0u] = 1u;
            }
            n_dof = 1u;
        } else if (jt == JOINT_PRISMATIC) {
            let axis_w = normalize(quat_rotate(r_jw, axis_l));
            let lo = joint_limit_lower[dof];
            let hi = joint_limit_upper[dof];
            if !motion_axis_point(j,0u) {
                s_lin[k][0u] = axis_w;
                s_ang[k][0u] = vec3<f32>(0.0);
                dof_active_arr[k][0u] = 1u;
            }
            n_dof = 1u;
        } else if (jt == JOINT_BALL) {
            let fixed=motion_axis_point(j,0u) && motion_axis_point(j,1u) && motion_axis_point(j,2u);
            if !fixed {
            s_lin[k][0u] = cross(ex, -p_jw); s_ang[k][0u] = ex; dof_active_arr[k][0u] = 1u;
            s_lin[k][1u] = cross(ey, -p_jw); s_ang[k][1u] = ey; dof_active_arr[k][1u] = 1u;
            s_lin[k][2u] = cross(ez, -p_jw); s_ang[k][2u] = ez; dof_active_arr[k][2u] = 1u;
            }
            n_dof = 3u;
        } else if (jt == JOINT_FREE || jt == JOINT_DISTANCE) {
            s_lin[k][0u] = ex; dof_active_arr[k][0u] = 1u;
            s_lin[k][1u] = ey; dof_active_arr[k][1u] = 1u;
            s_lin[k][2u] = ez; dof_active_arr[k][2u] = 1u;
            s_lin[k][3u] = cross(ex, -p_jw); s_ang[k][3u] = ex; dof_active_arr[k][3u] = 1u;
            s_lin[k][4u] = cross(ey, -p_jw); s_ang[k][4u] = ey; dof_active_arr[k][4u] = 1u;
            s_lin[k][5u] = cross(ez, -p_jw); s_ang[k][5u] = ez; dof_active_arr[k][5u] = 1u;
            n_dof = 6u;
        } else if (jt == JOINT_D6) {
            let angular_locks=d6_angular_lock_count(j);
            for (var d = 0u; d < 6u; d = d + 1u) {
                let lo = joint_limit_lower[dof + d];
                let hi = joint_limit_upper[dof + d];
                let lo_finite = (lo == lo) && (abs(lo) < 1e30);
                let hi_finite = (hi == hi) && (abs(hi) < 1e30);
                let locked = motion_axis_point(j,d) && (d<3u || angular_locks==3u);
                var v_l = vec3<f32>(0.0);
                var v_a = vec3<f32>(0.0);
                if (d == 0u) { v_l = ex; }
                else if (d == 1u) { v_l = ey; }
                else if (d == 2u) { v_l = ez; }
                else if (d == 3u) { v_a = ex; }
                else if (d == 4u) { v_a = ey; }
                else { v_a = ez; }
                // Angular columns include the reference-origin moment arm.
                if (d >= 3u) { v_l = cross(v_a, -p_jw); }
                if (!locked) {
                    s_lin[k][d] = v_l;
                    s_ang[k][d] = v_a;
                    dof_active_arr[k][d] = 1u;
                }
            }
            n_dof = 6u;
        } else if (jt == JOINT_CABLE) {
            // Stage 39E — 2-column subspace: linear stretch along
            // joint-world +z, bend about joint-world +x. Mirrors the
            // CPU build_motion_subspace impl.
            s_lin[k][0u] = ez; dof_active_arr[k][0u] = 1u;
            s_lin[k][1u] = cross(ex, -p_jw); s_ang[k][1u] = ex; dof_active_arr[k][1u] = 1u;
            n_dof = 2u;
        }
        if joint_reversed[j] != 0u {
            for (var d = 0u; d < n_dof; d++) {
                s_lin[k][d] = -s_lin[k][d];
                s_ang[k][d] = -s_ang[k][d];
            }
        }
        if (body_flags[child] & 3u) != 0u || joint_enabled[j]==0u {
            n_dof = 0u;
            for (var d = 0u; d < 6u; d++) {
                s_lin[k][d] = vec3<f32>(0.0); s_ang[k][d] = vec3<f32>(0.0); dof_active_arr[k][d] = 0u;
            }
        }
        n_dof_arr[k] = n_dof;


        // Joint velocity = Σ S_k · qd_k.
        var v_j_l = vec3<f32>(0.0);
        var v_j_a = vec3<f32>(0.0);
        for (var d = 0u; d < n_dof; d = d + 1u) {
            if (dof_active_arr[k][d] != 0u) {
                let qd_k = joint_qd[dof + d];
                v_j_l = v_j_l + s_lin[k][d] * qd_k;
                v_j_a = v_j_a + s_ang[k][d] * qd_k;
            }
        }

        // Parent body's spatial velocity.
        var vp_l = vec3<f32>(0.0);
        var vp_a = vec3<f32>(0.0);
        if (parent >= 0) {
            for (var kk = 0u; kk < k; kk = kk + 1u) {
                let jj = j_begin + kk;
                if (u32(joint_child[jj]) == u32(parent)) {
                    vp_l = v_lin[kk];
                    vp_a = v_ang[kk];
                    break;
                }
            }
        }
        var vi_l = vp_l + v_j_l;
        var vi_a = vp_a + v_j_a;
        if (body_flags[child] & 2u) != 0u { vi_l = vec3<f32>(0.0); vi_a = vec3<f32>(0.0); }
        else if (body_flags[child] & 1u) != 0u {
            let center = body_q[child].position.xyz - spatial_origin + quat_rotate(body_q[child].rotation, body_com[child].xyz);
            vi_a = body_qd[child].angular.xyz;
            vi_l = body_qd[child].linear.xyz - cross(vi_a, center);
        }
        v_lin[k] = vi_l;
        v_ang[k] = vi_a;

        c_lin[k] = spatial_cross_lin(vi_l, vi_a, v_j_l, v_j_a);
        c_ang[k] = spatial_cross_ang(vi_l, vi_a, v_j_l, v_j_a);
        if joint_enabled[j]!=0u && (jt == JOINT_FREE || jt == JOINT_DISTANCE || jt == JOINT_D6) {
            let translation = ex * joint_qd[dof] + ey * joint_qd[dof+1u] + ez * joint_qd[dof+2u];
            c_lin[k] -= cross(v_j_a, translation);
        }

        let cw = body_q[child];
        let com_world = cw.position.xyz - spatial_origin + quat_rotate(cw.rotation, body_com[child].xyz);
        let i_local = mat3_from_cols(body_inertia[child]);
        let i_world = rotate_inertia(cw.rotation, i_local);
        var i_s = smat6_from_body(body_mass[child], com_world, i_world);

        let isv = smat6_mul_sv(i_s, vi_l, vi_a);
        let isv_l = vec3<f32>(isv[0u], isv[1u], isv[2u]);
        let isv_a = vec3<f32>(isv[3u], isv[4u], isv[5u]);
        let bias_l = spatial_cross_dual_lin(vi_l, vi_a, isv_l, isv_a);
        let bias_a = spatial_cross_dual_ang(vi_l, vi_a, isv_l, isv_a);
        let world_idx = max(body_world[child], 0);
        let g = gravity[u32(world_idx)].xyz;
        let f_g_lin = g * body_mass[child];
        let f_g_ang = cross(com_world, f_g_lin);
        // Shift the COM wrench to the articulation reference origin.
        var ext = body_f[child];
        if uniforms.control_body_wrenches != 0u {
            ext.linear += control_body_forces[child].linear;
            ext.angular += control_body_forces[child].angular;
        }
        if !(all(abs(ext.linear.xyz) <= vec3<f32>(3.402823e38)) && all(abs(ext.angular.xyz) <= vec3<f32>(3.402823e38))) {
            atomicOr(&stop_status, 32u); return;
        }
        let f_ext_lin = ext.linear.xyz;
        let f_ext_ang = ext.angular.xyz + cross(com_world, ext.linear.xyz);
        i_a[k] = i_s;
        p_lin[k] = bias_l - f_g_lin - f_ext_lin;
        p_ang[k] = bias_a - f_g_ang - f_ext_ang;
    }

    // ── Pass 2 — backward articulated-inertia accumulation ───────
    for (var ki = 0u; ki < n; ki = ki + 1u) {
        let k = n - 1u - ki;
        let j = j_begin + k;
        let parent = joint_parent[j];
        let jt = joint_type[j];
        let n_dof = n_dof_arr[k];
        let supported = (jt != JOINT_FIXED) && (n_dof > 0u);

        let dof = joint_dof_offset[j];
        let coord = joint_coord_offset[j];

        // Sequential rank-1 updates per DOF.
        var i_curr: array<f32, 36> = i_a[k];
        for (var d = 0u; d < n_dof; d = d + 1u) {
            if (dof_active_arr[k][d] == 0u) { continue; }
            let s_l = s_lin[k][d];
            let s_a = s_ang[k][d];
            var inertia = i_curr;
            if rigorous { inertia = i_a[k]; }
            let u_arr = smat6_mul_sv(inertia, s_l, s_a);
            let u_l = vec3<f32>(u_arr[0u], u_arr[1u], u_arr[2u]);
            let u_a = vec3<f32>(u_arr[3u], u_arr[4u], u_arr[5u]);
            let dval = dot(s_l, u_l) + dot(s_a, u_a);
            let d_i = select(0.0, 1.0 / dval, abs(dval) > 1e-12);
            u_lin[k][d] = u_l;
            u_ang[k][d] = u_a;
            d_inv_arr[k][d] = d_i;
            d_dense_inv[k][d*6u+d] = d_i;

            // Drive / friction / effort.
            let dof_g = dof + d;
            let qd_now = joint_qd[dof_g];
            let mode = joint_target_mode[dof_g];
            if joint_brake[dof_g] < 0.0 || !(joint_brake[dof_g] <= 3.402823e38)
                || (!rigorous && joint_brake[dof_g] != 0.0)
                || joint_motor_model[dof_g] > 1u || (!rigorous && joint_motor_model[dof_g] == 1u
                && mode >= TARGET_POS && mode <= TARGET_PV) {
                atomicOr(&stop_status,32u); return;
            }
            var q_now = 0.0;
            if mode == TARGET_POS || mode == TARGET_PV || jt == JOINT_CABLE {
                q_now = motor_coordinate(j, d);
            }
            var drv = drive_torque(
                mode,
                q_now, qd_now,
                joint_target_kp[dof_g], joint_target_kd[dof_g],
                joint_target_pos[dof_g], joint_target_vel[dof_g],
            );
            if mode >= TARGET_POS && mode <= TARGET_PV
                && (uniforms.stable_motor_feedback != 0u || joint_motor_model[dof_g] == 1u) {
                drv = 0.0;
            }
            let mu_jt = joint_friction[dof_g];
            // Stage 39E — per-DoF spring + damper for CABLE joints
            // (k=0 linear stretch uses k_lin/c_lin, k=1 bend uses
            // k_bend/c_bend). Buffers are zero for every other joint
            // type so we can read unconditionally.
            var cable_tau = 0.0;
            if (jt == JOINT_CABLE) {
                var k_stiff = 0.0;
                var c_damp = 0.0;
                if (d == 0u) {
                    k_stiff = joint_cable_k_lin[j];
                    c_damp  = joint_cable_c_lin[j];
                } else {
                    k_stiff = joint_cable_k_bend[j];
                    c_damp  = joint_cable_c_bend[j];
                }
                cable_tau = -k_stiff * q_now - c_damp * qd_now;
            }
            // Stage 42 v3 — pre-computed Hill-type muscle torque.
            // `muscle.wgsl` writes the per-DOF sum into
            // `joint_muscle_torque[dof_g]`. Only the cable linear-
            // stretch DoF is populated; everything else is zero.
            let muscle_tau = joint_muscle_torque[dof_g];
            let motor_cap = joint_motor_max_force[dof_g];
            let motor_torque = clamp(drv, -motor_cap, motor_cap);
            let tau_raw = joint_force[dof_g] + motor_torque - mu_jt * qd_now
                          + cable_tau + muscle_tau;
            let tau_app = clamp_or_passthrough(tau_raw, joint_effort_limit[dof_g]);
            if k == 0u { root_applied[d] = tau_app; }
            u_tau_arr[k][d] = tau_app - (dot(s_l, p_lin[k]) + dot(s_a, p_ang[k]));

            // Scalar block elimination.
            if !rigorous || n_dof == 1u {
            let ua = array<f32, 6>(u_l.x, u_l.y, u_l.z, u_a.x, u_a.y, u_a.z);
            for (var rr = 0u; rr < 6u; rr = rr + 1u) {
                for (var cc = 0u; cc < 6u; cc = cc + 1u) {
                    i_curr[rr * 6u + cc] = i_curr[rr * 6u + cc] - ua[rr] * ua[cc] * d_i;
                }
            }
            }
        }

        if rigorous && supported && n_dof > 1u {
            let isolated_free = n == 1u && parent < 0 && n_dof == 6u;
            let isolated_ball = n == 1u && parent < 0 && joint_type[j]==JOINT_BALL && n_dof==3u;
            if isolated_ball {
                let child=u32(joint_child[j]);
                fk_pair_rounding=true;
                let rotation=root_motion_rotation(j);
                let angular=root_pivot_inverse(j,rotation,root_motion_arm(j,rotation));
                root_response_paired=true; root_response_slot=child;
                for (var c=0u; c<3u; c++) {
                    for (var r=0u; r<3u; r++) {
                        if !(abs(angular.high[c][r])<=3.402823e38) { atomicOr(&stop_status,32u); return; }
                        let enabled_axis=dof_active_arr[k][r]!=0u && dof_active_arr[k][c]!=0u;
                        d_dense_inv[k][r*6u+c]=select(0.0,angular.high[c][r],enabled_axis);
                        root_inverse_low[r*6u+c]=select(0.0,angular.low[c][r],enabled_axis);
                    }
                }
                fk_pair_rounding=false;
            } else if isolated_free {
                let child = u32(joint_child[j]);
                fk_pair_rounding=true;
                let rotation=root_motion_rotation(j);
                let inverse_motion=root_motion_inverse(j,rotation);
                let arm=root_motion_arm(j,rotation);
                root_response_paired=true; root_response_slot=child;
                for (var c = 0u; c < 6u; c += 1u) {
                    var axis=vec3<f32>(0.0); axis[c%3u]=1.0;
                    var force=vec3<f32>(0.0); var torque=PositionPair(axis,vec3<f32>(0.0));
                    if c<3u {
                        force=axis;
                        let moment=paired_cross(child,arm,PositionPair(force,vec3<f32>(0.0)));
                        torque=PositionPair(-moment.high,-moment.low);
                    }
                    let angular=paired_matrix_pair_vector(child,inverse_motion,torque);
                    let moment=paired_cross(child,angular,arm);
                    let linear=position_add(child,PositionPair(force/body_mass[child],vec3<f32>(0.0)),PositionPair(-moment.high,-moment.low));
                    for (var r = 0u; r < 3u; r += 1u) {
                        d_dense_inv[k][r*6u+c] = linear.high[r];
                        root_inverse_low[r*6u+c] = linear.low[r];
                        d_dense_inv[k][(r+3u)*6u+c] = angular.high[r];
                        root_inverse_low[(r+3u)*6u+c] = angular.low[r];
                    }
                }
                fk_pair_rounding=false;
                var any_locked = false;
                for (var d = 0u; d < 6u; d++) { any_locked = any_locked || dof_active_arr[k][d] == 0u; }
                if any_locked {
                root_response_paired=false;
                var locked: array<f32,36>;
                let unrestricted = d_dense_inv[k];
                for (var r = 0u; r < 6u; r++) {
                    for (var c = 0u; c < 6u; c++) {
                        if dof_active_arr[k][r] == 0u && dof_active_arr[k][c] == 0u { locked[r*6u+c] = unrestricted[r*6u+c]; }
                    }
                }
                let inverse = joint_psd_inverse(locked,6u);
                if !inverse.valid { atomicOr(&stop_status,32u); return; }
                for (var r = 0u; r < 6u; r++) {
                    for (var c = 0u; c < 6u; c++) {
                        var correction = 0.0;
                        for (var a = 0u; a < 6u; a++) {
                            var column = 0.0;
                            for (var b = 0u; b < 6u; b++) { column = fma(inverse.matrix[a*6u+b],unrestricted[b*6u+c],column); }
                            correction = fma(unrestricted[r*6u+a],column,correction);
                        }
                        var value = unrestricted[r*6u+c]-correction;
                        if dof_active_arr[k][r] == 0u || dof_active_arr[k][c] == 0u { value = 0.0; }
                        if !(abs(value) <= 3.402823e38) { atomicOr(&stop_status,32u); return; }
                        d_dense_inv[k][r*6u+c] = value;
                    }
                }
                }
            } else {
            var factor: array<f32, 36>;
            var positive = true;
            for (var r = 0u; r < n_dof; r += 1u) {
                for (var c = 0u; c <= r; c += 1u) {
                    var value = 0.0;
                    if dof_active_arr[k][r] != 0u && dof_active_arr[k][c] != 0u {
                        value = dot(s_lin[k][r], u_lin[k][c]) + dot(s_ang[k][r], u_ang[k][c]);
                    } else if r == c { value = 1.0; }
                    for (var d = 0u; d < c; d += 1u) { value -= factor[r*6u+d] * factor[c*6u+d]; }
                    if r == c {
                        if !(value > 0.0 && value <= 3.402823e38) { positive = false; break; }
                        factor[r*6u+c] = sqrt(value);
                    }
                    else { factor[r*6u+c] = value / factor[c*6u+c]; }
                }
                if !positive { break; }
            }
            if !positive {
                var matrix: array<f32,36>;
                for (var r = 0u; r < n_dof; r++) {
                    for (var c = 0u; c < n_dof; c++) {
                        if dof_active_arr[k][r] != 0u && dof_active_arr[k][c] != 0u {
                            matrix[r*6u+c] = dot(s_lin[k][r],u_lin[k][c])+dot(s_ang[k][r],u_ang[k][c]);
                        }
                    }
                }
                let inverse = joint_psd_inverse(matrix,n_dof);
                if !inverse.valid { atomicOr(&stop_status,32u); return; }
                d_dense_inv[k] = inverse.matrix;
            } else {
            for (var c = 0u; c < n_dof; c += 1u) {
                var solution: array<f32, 6>;
                for (var r = 0u; r < n_dof; r += 1u) {
                    var value = select(0.0, 1.0, r == c);
                    for (var d = 0u; d < r; d += 1u) { value -= factor[r*6u+d] * solution[d]; }
                    solution[r] = value / factor[r*6u+r];
                }
                for (var back = 0u; back < n_dof; back += 1u) {
                    let r = n_dof - 1u - back;
                    var value = solution[r];
                    for (var d = r+1u; d < n_dof; d += 1u) { value -= factor[d*6u+r] * solution[d]; }
                    solution[r] = value / factor[r*6u+r];
                    d_dense_inv[k][r*6u+c] = solution[r];
                }
            }
            }
            }
            for (var c = 0u; c < n_dof; c += 1u) {
                var weighted_l = vec3<f32>(0.0);
                var weighted_a = vec3<f32>(0.0);
                for (var d = 0u; d < n_dof; d += 1u) {
                    weighted_l += u_lin[k][d] * d_dense_inv[k][d*6u+c];
                    weighted_a += u_ang[k][d] * d_dense_inv[k][d*6u+c];
                }
                let u = array<f32, 6>(u_lin[k][c].x, u_lin[k][c].y, u_lin[k][c].z,
                    u_ang[k][c].x, u_ang[k][c].y, u_ang[k][c].z);
                let v = array<f32, 6>(weighted_l.x, weighted_l.y, weighted_l.z,
                    weighted_a.x, weighted_a.y, weighted_a.z);
                for (var r = 0u; r < 6u; r += 1u) {
                    for (var d = 0u; d < 6u; d += 1u) { i_curr[r*6u+d] -= u[r] * v[d]; }
                }
            }
        }

        // p^a = p^A + i_curr · c + Σ U_k · (u_tau_k / D_k)
        let cv = smat6_mul_sv(i_curr, c_lin[k], c_ang[k]);
        var p_art_l = p_lin[k] + vec3<f32>(cv[0u], cv[1u], cv[2u]);
        var p_art_a = p_ang[k] + vec3<f32>(cv[3u], cv[4u], cv[5u]);
        for (var d = 0u; d < n_dof; d = d + 1u) {
            if (dof_active_arr[k][d] == 0u) { continue; }
            var scale_u = u_tau_arr[k][d] * d_inv_arr[k][d];
            if rigorous {
                scale_u = 0.0;
                for (var c = 0u; c < n_dof; c += 1u) { scale_u += d_dense_inv[k][d*6u+c] * u_tau_arr[k][c]; }
            }
            p_art_l = p_art_l + u_lin[k][d] * scale_u;
            p_art_a = p_art_a + u_ang[k][d] * scale_u;
        }

        if (parent >= 0 && (body_flags[u32(joint_child[j])] & 3u) == 0u) {
            var pj: i32 = -1;
            for (var kk = 0u; kk < k; kk = kk + 1u) {
                let jj = j_begin + kk;
                if (u32(joint_child[jj]) == u32(parent)) {
                    pj = i32(kk);
                    break;
                }
            }
            if (pj >= 0) {
                let pj_u = u32(pj);
                for (var rr = 0u; rr < 36u; rr = rr + 1u) {
                    i_a[pj_u][rr] = i_a[pj_u][rr] + i_curr[rr];
                }
                p_lin[pj_u] = p_lin[pj_u] + p_art_l;
                p_ang[pj_u] = p_ang[pj_u] + p_art_a;
            }
        }
    }

    // ── Pass 3 — forward acceleration solve + integrate ─────────
    for (var k = 0u; k < n; k = k + 1u) {
        let j = j_begin + k;
        let child = u32(joint_child[j]);
        if (body_flags[child] & 3u) != 0u {
            if (body_flags[child] & 2u) == 0u {
                let acceleration = boundary_acceleration[child];
                let center = body_q[child].position.xyz - spatial_origin + quat_rotate(body_q[child].rotation, body_com[child].xyz);
                a_lin[k] = acceleration.linear.xyz - cross(acceleration.angular.xyz, center)
                    - cross(body_qd[child].angular.xyz, body_qd[child].linear.xyz);
                a_ang[k] = acceleration.angular.xyz;
            }
            continue;
        }
        let parent = joint_parent[j];
        let jt = joint_type[j];
        let n_dof = n_dof_arr[k];
        let supported = (jt != JOINT_FIXED) && (n_dof > 0u);

        var ap_l = vec3<f32>(0.0);
        var ap_a = vec3<f32>(0.0);
        if (parent >= 0) {
            for (var kk = 0u; kk < k; kk = kk + 1u) {
                let jj = j_begin + kk;
                if (u32(joint_child[jj]) == u32(parent)) {
                    ap_l = a_lin[kk];
                    ap_a = a_ang[kk];
                    break;
                }
            }
        }

        ap_l += c_lin[k];
        ap_a += c_ang[k];
        var s_qdd_l = vec3<f32>(0.0);
        var s_qdd_a = vec3<f32>(0.0);
        if (supported) {
            for (var d = 0u; d < n_dof; d = d + 1u) {
                if (dof_active_arr[k][d] == 0u) {
                    q_dd_arr[k][d] = 0.0;
                    continue;
                }
                var q_dd_d = (u_tau_arr[k][d]
                              - (dot(u_lin[k][d], ap_l) + dot(u_ang[k][d], ap_a)))
                              * d_inv_arr[k][d];
                if rigorous {
                    q_dd_d = 0.0;
                    for (var c = 0u; c < n_dof; c += 1u) {
                        let rhs = u_tau_arr[k][c] - dot(u_lin[k][c], ap_l) - dot(u_ang[k][c], ap_a);
                        q_dd_d += d_dense_inv[k][d*6u+c] * rhs;
                    }
                }
                q_dd_arr[k][d] = q_dd_d;
                s_qdd_l = s_qdd_l + s_lin[k][d] * q_dd_d;
                s_qdd_a = s_qdd_a + s_ang[k][d] * q_dd_d;
            }
            a_lin[k] = ap_l + s_qdd_l;
            a_ang[k] = ap_a + s_qdd_a;
        } else {
            a_lin[k] = ap_l;
            a_ang[k] = ap_a;
        }
    }

        if uniforms.drift_only == 0u && rigorous && n == 1u
            && joint_parent[j_begin] < 0 && (n_dof_arr[0] == 6u || (joint_type[j_begin]==JOINT_BALL && n_dof_arr[0]==3u)) {
            let ball=joint_type[j_begin]==JOINT_BALL;
            var unrestricted = true;
            for (var d = 0u; d < n_dof_arr[0]; d += 1u) { unrestricted = unrestricted && dof_active_arr[0][d] != 0u; }
            if unrestricted {
                let child = u32(joint_child[j_begin]);
                fk_pair_rounding=true;
                let rotation=root_motion_rotation(j_begin);
                let arm=root_motion_arm(j_begin,rotation);
                let direction=select(1.0,-1.0,joint_reversed[j_begin]!=0u);
                var frame=QuaternionPair(joint_x_p[j_begin].rotation,joint_frame_references[j_begin].parent_rotation);
                if joint_reversed[j_begin]!=0u {
                    var body_rotation=QuaternionPair(body_q[child].rotation,vec4<f32>(0.0));
                    if uniforms.compensated_coordinates!=0u { body_rotation=fk_rotations[j_begin%MAX_JOINTS_PER_ART]; }
                    frame=multiply(child,body_rotation,QuaternionPair(joint_x_c[j_begin].rotation,joint_frame_references[j_begin].child_rotation));
                }
                frame=normalize_pair(child,frame);
                var load = body_f[child];
                if uniforms.control_body_wrenches != 0u {
                    load.linear += control_body_forces[child].linear;
                    load.angular += control_body_forces[child].angular;
                }
                let force_drive=PositionPair(select(direction*vec3<f32>(root_applied[0],root_applied[1],root_applied[2]),vec3<f32>(0.0),ball),vec3<f32>(0.0));
                let force=position_add(child,force_drive,position_pair_rotate(child,conjugate_pair(frame),PositionPair(load.linear.xyz,vec3<f32>(0.0))));
                let moment=paired_cross(child,arm,force_drive);
                let angular_start=select(3u,0u,ball);
                var torque=position_add(child,PositionPair(direction*vec3<f32>(root_applied[angular_start],root_applied[angular_start+1u],root_applied[angular_start+2u]),vec3<f32>(0.0)),PositionPair(-moment.high,-moment.low));
                torque=position_add(child,torque,position_pair_rotate(child,conjugate_pair(frame),PositionPair(load.angular.xyz,vec3<f32>(0.0))));
                let dof=joint_dof_offset[j_begin];
                let omega=PositionPair(direction*vec3<f32>(joint_qd[dof+angular_start],joint_qd[dof+angular_start+1u],joint_qd[dof+angular_start+2u]),vec3<f32>(0.0));
                let local_omega=position_pair_rotate(child,conjugate_pair(rotation),omega);
                let momentum=paired_matrix_vector(child,mat3_from_cols(body_inertia[child]),local_omega);
                let gyro=position_pair_rotate(child,rotation,paired_cross(child,local_omega,momentum));
                let centrifugal=paired_cross(child,omega,paired_cross(child,omega,arm));
                let g=position_pair_rotate(child,conjugate_pair(frame),PositionPair(gravity[u32(max(body_world[child],0))].xyz,vec3<f32>(0.0)));
                var inverse=root_motion_inverse(j_begin,rotation);
                if ball {
                    inverse=root_pivot_inverse(j_begin,rotation,arm);
                    let centripetal=PositionPair(body_mass[child]*centrifugal.high,body_mass[child]*centrifugal.low);
                    let weight=PositionPair(body_mass[child]*g.high,body_mass[child]*g.low);
                    let pivot_force=position_add(child,position_add(child,force,weight),PositionPair(-centripetal.high,-centripetal.low));
                    torque=position_add(child,torque,paired_cross(child,arm,pivot_force));
                }
                let alpha=paired_matrix_pair_vector(child,inverse,position_add(child,torque,PositionPair(-gyro.high,-gyro.low)));
                let angular_arm=paired_cross(child,alpha,arm);
                var acceleration=position_add(child,PositionPair(force.high/body_mass[child],force.low/body_mass[child]),g);
                acceleration=position_add(child,acceleration,PositionPair(-angular_arm.high,-angular_arm.low));
                acceleration=position_add(child,acceleration,PositionPair(-centrifugal.high,-centrifugal.low));
                for (var d = 0u; d < 3u; d += 1u) {
                    if !ball { q_dd_arr[0][d] = direction*(acceleration.high[d]+acceleration.low[d]); }
                    q_dd_arr[0][d+angular_start] = direction*(alpha.high[d]+alpha.low[d]);
                }
                fk_pair_rounding=false;
            }
        }

        if uniforms.drift_only==0u && !prepare_drive_locks(j_begin,n,&n_dof_arr,&s_lin,&s_ang,&u_lin,&u_ang,&d_dense_inv) {
            atomicOr(&stop_status,32u);
            return;
        }
        if uniforms.drift_only == 0u
            && !apply_motor_feedback(j_begin, n, &n_dof_arr, &s_lin, &s_ang, &u_lin, &u_ang, &d_dense_inv, &q_dd_arr) {
            atomicOr(&stop_status, 32u);
            return;
        }
        if uniforms.drift_only == 0u && has_compliance
            && !apply_joint_compliance(j_begin, n, &n_dof_arr, &s_lin, &s_ang, &u_lin, &u_ang, &d_dense_inv, &q_dd_arr) {
            atomicOr(&stop_status, 32u);
            return;
        }
        if uniforms.drift_only == 0u && uniforms.stable_motor_feedback != 0u
            && !react_velocity_feedback(j_begin, n, &n_dof_arr, &s_lin, &s_ang, &u_lin, &u_ang, &d_dense_inv, &q_dd_arr) {
            atomicOr(&stop_status, 32u);
            return;
        }
        if uniforms.drift_only != 0u {
            for (var k = 0u; k < n; k += 1u) {
                for (var d = 0u; d < n_dof_arr[k]; d += 1u) { q_dd_arr[k][d] = 0.0; }
            }
        }
        if mixed_stops && uniforms.external_scalar_bounds==0u && !position_only && !refresh_only
            && !react_mixed_stops(j_begin,n,&n_dof_arr,&s_lin,&s_ang,&u_lin,&u_ang,&d_dense_inv,&q_dd_arr) {
            atomicOr(&stop_status,32u); return;
        }
        if !mixed_stops && has_stops && !react_scalar_stops(j_begin, n, &n_dof_arr, &s_lin, &s_ang, &u_lin, &u_ang, &d_dense_inv, &q_dd_arr) {
            atomicOr(&stop_status, 32u);
            return;
        }
    }

    // ── Integrate q, qd ─────────────────────────────────────────
    if !refresh_only {
    for (var k = 0u; k < n; k = k + 1u) {
        let j = j_begin + k;
        let jt = joint_type[j];
        let coord = joint_coord_offset[j];
        let dof = joint_dof_offset[j];
        let n_dof = n_dof_arr[k];
        let damp = max(1.0 - uniforms.joint_damping * uniforms.dt, 0.0);
        var previous: array<f32,6>;
        var previous_pair: array<vec2<f32>,6>;
        var predicted_rate: array<f32,6>;
        rate_rounding_child=u32(joint_child[j]);
        for (var d = 0u; d < joint_dof_offset[j+1u]-dof; d++) {
            previous[d] = joint_qd[dof+d];
            if uniforms.paired_rates!=0u { previous_pair[d]=paired_rates[dof+d]; }
        }
        if (body_flags[u32(joint_child[j])] & 3u) != 0u || joint_enabled[j]==0u {
            if position_only { continue; }
            for (var d = dof; d < joint_dof_offset[j+1u]; d++) {
                if uniforms.drift_only == 0u { velocity_delta[d] = -joint_qd[d]; }
                joint_qd[d] = 0.0;
                rate_finish(d,previous_pair[d-dof],true);
            }
            continue;
        }
        if position_only {
            var moved = false;
            for (var d = 0u; d < joint_dof_offset[j+1u]-dof; d++) {
                moved = moved || velocity_delta[dof+d] != 0.0;
            }
            if !moved { continue; }
            for (var d = dof; d < joint_dof_offset[j+1u]; d++) {
                joint_qd[d] = velocity_delta[d];
            }
        }

        if (jt == JOINT_REVOLUTE || jt == JOINT_PRISMATIC) {
            let lo = joint_limit_lower[dof];
            let hi = joint_limit_upper[dof];
            if motion_axis_point(j,0u) {
                coordinate_limit_set(coord,dof,false);
                if uniforms.drift_only == 0u { velocity_delta[dof] = -joint_qd[dof]; }
                joint_qd[dof] = select(0.0, previous[0u], position_only);
                rate_finish(dof,previous_pair[0u],true);
                continue;
            }
            let qd_damped = rate_integrate(j,k,0u,q_dd_arr[k][0u],damp);
            predicted_rate[0u] = qd_damped;
            var qd_capped=qd_damped;
            if !position_only && !local_scalar_bounds { qd_capped=rate_cap(dof,qd_damped); }
            if position_only || local_scalar_bounds {
                joint_qd[dof] = qd_damped;
                joint_q[coord] = rate_coordinate_advance(coord,dof,qd_damped);
            } else {
                let q_pred = rate_coordinate_advance(coord,dof,qd_capped);
                joint_qd[dof] = qd_capped;
                joint_q[coord] = q_pred;
                if coordinate_limit_gap(coord,dof,false)>0.0 { coordinate_limit_set(coord,dof,false); }
                if coordinate_limit_gap(coord,dof,true)<0.0 { coordinate_limit_set(coord,dof,true); }
            }
        } else if (jt == JOINT_CABLE) {
            // Stage 39E — 2 scalar coords (q_l, q_b), 2 dofs. Symplectic
            // Euler per DoF, identical to two stacked 1-DoF integrators.
            for (var dd = 0u; dd < 2u; dd = dd + 1u) {
                let qd_damped = rate_integrate(j,k,dd,q_dd_arr[k][dd],damp);
                predicted_rate[dd] = qd_damped;
                var qd_capped=qd_damped;
                if !position_only && !local_scalar_bounds { qd_capped=rate_cap(dof+dd,qd_damped); }
                let q_pred = rate_coordinate_advance(coord+dd,dof+dd,qd_capped);
                let lo = joint_limit_lower[dof + dd];
                let hi = joint_limit_upper[dof + dd];
                var q_clamped = q_pred;
                var qd_after = qd_capped;
                if (!position_only && (lo == lo) && (q_clamped < lo)) {
                    q_clamped = lo;
                    if (qd_after < 0.0) { qd_after = 0.0; }
                }
                if (!position_only && (hi == hi) && (q_clamped > hi)) {
                    q_clamped = hi;
                    if (qd_after > 0.0) { qd_after = 0.0; }
                }
                joint_qd[dof + dd] = qd_after;
                joint_q[coord + dd] = q_clamped;
                if (!position_only && (q_pred <= lo || q_pred >= hi)) { coordinate_reset(coord + dd); }
            }
        } else if (jt == JOINT_BALL) {
            var omega = vec3<f32>(joint_qd[dof], joint_qd[dof+1u], joint_qd[dof+2u]);
            for (var d=0u; d<3u; d++) { omega[d]=rate_integrate(j,k,d,q_dd_arr[k][d],damp); }
            for (var d = 0u; d < 3u; d += 1u) { predicted_rate[d] = omega[d]; }
            if !position_only && !local_scalar_bounds {
                for (var d = 0u; d < 3u; d++) { omega[d] = rate_cap(dof+d,omega[d]); }
            }
            let q_old = vec4<f32>(joint_q[coord], joint_q[coord+1u], joint_q[coord+2u], joint_q[coord+3u]);
            let dq = vec4<f32>(omega.x * 0.5, omega.y * 0.5, omega.z * 0.5, 0.0);
            let q_dot = quat_mul(dq, q_old);
            var q_new = normalize(q_old + q_dot * uniforms.dt);
            if uniforms.compensated_coordinates!=0u { q_new=rate_quaternion_advance(coord,dof,omega); }
            joint_q[coord]    = q_new.x;
            joint_q[coord+1u] = q_new.y;
            joint_q[coord+2u] = q_new.z;
            joint_q[coord+3u] = q_new.w;
            joint_qd[dof]    = omega.x;
            joint_qd[dof+1u] = omega.y;
            joint_qd[dof+2u] = omega.z;
            if !position_only && motion_axis_point(j,0u) && motion_axis_point(j,1u) && motion_axis_point(j,2u) {
                coordinate_rotation_point_set(j,0u);
                for (var d=0u; d<3u; d++) { joint_qd[dof+d]=0.0; }
            }
        } else if (jt == JOINT_FREE || jt == JOINT_DISTANCE || jt == JOINT_D6) {
            var v = vec3<f32>(joint_qd[dof], joint_qd[dof+1u], joint_qd[dof+2u]);
            var omega = vec3<f32>(joint_qd[dof+3u], joint_qd[dof+4u], joint_qd[dof+5u]);
            for (var d=0u; d<3u; d++) {
                v[d]=rate_integrate(j,k,d,q_dd_arr[k][d],damp);
                omega[d]=rate_integrate(j,k,d+3u,q_dd_arr[k][d+3u],damp);
            }
            for (var d = 0u; d < 3u; d += 1u) {
                predicted_rate[d] = v[d];
                predicted_rate[d+3u] = omega[d];
            }
            if !position_only && !local_scalar_bounds {
                for (var d = 0u; d < 3u; d++) {
                    v[d] = rate_cap(dof+d,v[d]);
                    omega[d] = rate_cap(dof+3u+d,omega[d]);
                }
            }
            let p_new = vec3<f32>(rate_coordinate_advance(coord,dof,v.x),
                rate_coordinate_advance(coord+1u,dof+1u,v.y), rate_coordinate_advance(coord+2u,dof+2u,v.z));
            let r_old = vec4<f32>(joint_q[coord+3u], joint_q[coord+4u], joint_q[coord+5u], joint_q[coord+6u]);
            let dq = vec4<f32>(omega.x * 0.5, omega.y * 0.5, omega.z * 0.5, 0.0);
            let r_dot = quat_mul(dq, r_old);
            var r_new = normalize(r_old + r_dot * uniforms.dt);
            if uniforms.compensated_coordinates != 0u {
                r_new = rate_quaternion_advance(coord+3u,dof+3u,omega);
            }
            joint_q[coord]    = p_new.x;
            joint_q[coord+1u] = p_new.y;
            joint_q[coord+2u] = p_new.z;
            joint_q[coord+3u] = r_new.x;
            joint_q[coord+4u] = r_new.y;
            joint_q[coord+5u] = r_new.z;
            joint_q[coord+6u] = r_new.w;
            joint_qd[dof]    = v.x;
            joint_qd[dof+1u] = v.y;
            joint_qd[dof+2u] = v.z;
            joint_qd[dof+3u] = omega.x;
            joint_qd[dof+4u] = omega.y;
            joint_qd[dof+5u] = omega.z;

            // Project radial bounds in standalone position-clamp mode.
            if (jt == JOINT_DISTANCE && !position_only && !local_scalar_bounds && uniforms.rigorous_distance==0u) {
                let lo = joint_limit_lower[dof];
                let hi = joint_limit_upper[dof];
                let lo_finite = (lo == lo) && (abs(lo) < 1e30);
                let hi_finite = (hi == hi) && (abs(hi) < 1e30);

                var p_curr = vec3<f32>(joint_q[coord], joint_q[coord+1u], joint_q[coord+2u]);
                let dist = length(p_curr);
                var clamped = p_curr;
                var clip = false;
                if (lo_finite && (dist < lo) && (dist > 1e-12)) {
                    clamped = p_curr * (lo / dist);
                    clip = true;
                }
                if (hi_finite && (dist > hi) && (dist > 1e-12)) {
                    clamped = p_curr * (hi / dist);
                    clip = true;
                }
                if (clip) {
                    joint_q[coord]    = clamped.x;
                    joint_q[coord+1u] = clamped.y;
                    joint_q[coord+2u] = clamped.z;
                    coordinate_reset(coord); coordinate_reset(coord+1u); coordinate_reset(coord+2u);
                    let nrm = normalize(clamped);
                    let v_now = vec3<f32>(joint_qd[dof], joint_qd[dof+1u], joint_qd[dof+2u]);
                    let radial = nrm * dot(v_now, nrm);
                    let v_proj = v_now - radial;
                    joint_qd[dof]    = v_proj.x;
                    joint_qd[dof+1u] = v_proj.y;
                    joint_qd[dof+2u] = v_proj.z;
                }
            }

            // Project D6 translation and intrinsic XYZ angular bounds.
            if (jt == JOINT_D6 && !position_only) {
                for (var d = 0u; d < 3u; d = d + 1u) {
                    let lo = joint_limit_lower[dof + d];
                    let hi = joint_limit_upper[dof + d];
                    if motion_axis_point(j,d) {
                        coordinate_limit_set(coord+d,dof+d,false);
                        joint_qd[dof+d]=0.0;
                        continue;
                    }
                    if local_scalar_bounds && !motion_axis_point(j,d) { continue; }
                    var p_k = joint_q[coord + d];
                    var qd_k = joint_qd[dof + d];
                    if ((lo == lo) && (p_k < lo)) {
                        p_k = lo;
                        if (qd_k < 0.0) { qd_k = 0.0; }
                    }
                    if ((hi == hi) && (p_k > hi)) {
                        p_k = hi;
                        if (qd_k > 0.0) { qd_k = 0.0; }
                    }
                    if (p_k <= lo || p_k >= hi) { coordinate_reset(coord + d); }
                    joint_q[coord + d] = p_k;
                    joint_qd[dof + d] = qd_k;
                }

                let r_now = vec4<f32>(
                    joint_q[coord+3u],
                    joint_q[coord+4u],
                    joint_q[coord+5u],
                    joint_q[coord+6u],
                );
                var ang = bounded_euler_xyz(r_now,dof,local_scalar_bounds);
                var omega_arr = vec3<f32>(
                    joint_qd[dof+3u],
                    joint_qd[dof+4u],
                    joint_qd[dof+5u],
                );
                var clamped = false;
                for (var d = 0u; d < 3u; d = d + 1u) {
                    let angular_locks=d6_angular_lock_count(j);
                    if local_scalar_bounds && angular_locks>0u && angular_locks<3u { continue; }
                    let lo = joint_limit_lower[dof + 3u + d];
                    let hi = joint_limit_upper[dof + 3u + d];
                    if local_scalar_bounds && !motion_axis_point(j,3u+d) { continue; }
                    var ad = ang[d];
                    var od = omega_arr[d];
                    if ((lo == lo) && (ad < lo)) {
                        ad = lo;
                        if (od < 0.0) { od = 0.0; }
                        clamped = true;
                    }
                    if ((hi == hi) && (ad > hi)) {
                        ad = hi;
                        if (od > 0.0) { od = 0.0; }
                        clamped = true;
                    }
                    ang[d] = ad;
                    omega_arr[d] = od;
                }
                if d6_angular_lock_count(j)==3u {
                    coordinate_rotation_point_set(j,3u);
                    for (var d=0u; d<3u; d++) { joint_qd[dof+3u+d]=0.0; }
                } else if (clamped) {
                    let r_proj = quat_from_euler_xyz(ang);
                    joint_q[coord+3u] = r_proj.x;
                    joint_q[coord+4u] = r_proj.y;
                    joint_q[coord+5u] = r_proj.z;
                    joint_q[coord+6u] = r_proj.w;
                    for (var lane=0u; lane<4u; lane++) { coordinate_reset(coord+3u+lane); }
                    joint_qd[dof+3u] = omega_arr.x;
                    joint_qd[dof+4u] = omega_arr.y;
                    joint_qd[dof+5u] = omega_arr.z;
                }
            }
        }
        if position_only {
            for (var d = 0u; d < joint_dof_offset[j+1u]-dof; d++) {
                joint_qd[dof+d] = previous[d];
            }
        }
        if uniforms.drift_only == 0u {
            for (var d = 0u; d < joint_dof_offset[j+1u]-dof; d++) {
                let increment = q_dd_arr[k][d]*uniforms.dt;
                var delta = joint_qd[dof+d]-previous[d];
                if !mixed_velocity_enabled && joint_qd[dof+d] == predicted_rate[d] {
                    delta = increment*damp+previous[d]*(damp-1.0);
                }
                velocity_delta[dof+d] = delta;
            }
        }
        for (var d=0u; d<joint_dof_offset[j+1u]-dof; d++) {
            let locked=motion_axis_point(j,d) && (jt==JOINT_PRISMATIC || jt==JOINT_REVOLUTE
                || (jt==JOINT_D6 && (d<3u || d6_angular_lock_count(j)==3u))
                || (jt==JOINT_BALL && motion_axis_point(j,0u) && motion_axis_point(j,1u) && motion_axis_point(j,2u)));
            rate_finish(dof+d,previous_pair[d],locked);
        }
    }

    }

    // Reconstruct body poses and world COM velocities.
    for (var k = 0u; k < n; k = k + 1u) {
        let j = j_begin + k;
        let parent = joint_parent[j];
        let child  = u32(joint_child[j]);
        if (body_flags[child] & 2u) != 0u {
            body_qd[child] = SpatialVecGpu(vec4<f32>(0.0), vec4<f32>(0.0));
            if uniforms.paired_rates!=0u { body_velocity_low[child]=SpatialVecGpu(vec4<f32>(0.0),vec4<f32>(0.0)); }
            continue;
        }
        if (body_flags[child] & 1u) != 0u {
            if !position_only && !refresh_only && (body_flags[child] & 4u) != 0u { advance_kinematic(child); }
            continue;
        }
        let coord  = joint_coord_offset[j];
        let jt     = select(JOINT_FIXED,joint_type[j],joint_enabled[j]!=0u);
        let axis   = joint_axis[j].xyz;

        var parent_world: Transform;
        if (parent < 0) {
            parent_world.position = vec4<f32>(0.0);
            parent_world.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        } else {
            parent_world = body_q[u32(parent)];
        }

        let local = tree_local_transform(j);
        let xp = joint_x_p[j];
        let xc_inv = transform_inverse(joint_x_c[j]);
        let xpw = transform_compose(parent_world, xp);
        let after = transform_compose(xpw, local);
        let frame = motion_frame(j,xpw,after);
        let direction = select(1.0,-1.0,joint_reversed[j]!=0u);
        body_q[child] = transform_compose(after, xc_inv);
        if (uniforms.compensated_coordinates != 0u) { publish_compensated_position(j, parent_world); }
        let center = body_q[child].position.xyz + quat_rotate(body_q[child].rotation, body_com[child].xyz);
        if uniforms.paired_rates!=0u { publish_paired_body_velocity(j); continue; }
        var velocity = SpatialVecGpu(vec4<f32>(0.0), vec4<f32>(0.0));
        if parent >= 0 {
            let p = u32(parent);
            let parent_center = body_q[p].position.xyz + quat_rotate(body_q[p].rotation, body_com[p].xyz);
            velocity = body_qd[p];
            velocity.linear = vec4<f32>(velocity.linear.xyz + cross(velocity.angular.xyz, center - parent_center), 0.0);
        }
        let dof = joint_dof_offset[j];
        var linear = vec3<f32>(0.0);
        var angular = vec3<f32>(0.0);
        var origin = frame.position.xyz;
        if jt == JOINT_REVOLUTE { angular = axis * joint_qd[dof]; }
        if jt == JOINT_PRISMATIC { linear = axis * joint_qd[dof]; }
        if jt == JOINT_BALL {
            angular = vec3<f32>(joint_qd[dof], joint_qd[dof+1u], joint_qd[dof+2u]);
        }
        if jt == JOINT_FREE || jt == JOINT_DISTANCE || jt == JOINT_D6 {
            linear = vec3<f32>(joint_qd[dof], joint_qd[dof+1u], joint_qd[dof+2u]);
            angular = vec3<f32>(joint_qd[dof+3u], joint_qd[dof+4u], joint_qd[dof+5u]);
        }
        if jt == JOINT_CABLE {
            linear.z = joint_qd[dof];
            angular.x = joint_qd[dof+1u];
            origin += quat_rotate(frame.rotation, vec3<f32>(0.0, 0.0, joint_q[coord]));
        }
        for (var d = 0u; d < joint_dof_offset[j+1u] - dof; d += 1u) {
            let lo = joint_limit_lower[dof+d];
            let hi = joint_limit_upper[dof+d];
            if motion_axis_point(j,d) {
                if jt == JOINT_REVOLUTE { angular = vec3<f32>(0.0); }
                if jt == JOINT_PRISMATIC { linear = vec3<f32>(0.0); }
                if jt == JOINT_BALL && motion_axis_point(j,0u) && motion_axis_point(j,1u) && motion_axis_point(j,2u) { angular = vec3<f32>(0.0); }
                if jt == JOINT_D6 {
                    if d < 3u { linear[d] = 0.0; }
                    else if d6_angular_lock_count(j)==3u { angular[d-3u] = 0.0; }
                }
            }
        }
        angular = quat_rotate(frame.rotation, angular)*direction;
        velocity.linear = vec4<f32>(velocity.linear.xyz + quat_rotate(frame.rotation, linear)*direction + cross(angular, center-origin), 0.0);
        velocity.angular = vec4<f32>(velocity.angular.xyz + angular, 0.0);
        body_qd[child] = velocity;
    }
}
