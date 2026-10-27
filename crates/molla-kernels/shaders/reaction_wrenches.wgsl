struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Mat3Cols { col0: vec4<f32>, col1: vec4<f32>, col2: vec4<f32> }
struct Wrench { linear: vec4<f32>, angular: vec4<f32> }
struct Uniforms { articulations: u32, pad0: u32, pad1: u32, pad2: u32 }
struct Spatial { linear: vec3<f32>, angular: vec3<f32> }

@group(0) @binding(0) var<storage, read> joint_type: array<u32>;
@group(0) @binding(1) var<storage, read> joint_parent: array<i32>;
@group(0) @binding(2) var<storage, read> joint_child: array<i32>;
@group(0) @binding(3) var<storage, read> joint_x_p: array<Transform>;
@group(0) @binding(4) var<storage, read> joint_x_c: array<Transform>;
@group(0) @binding(5) var<storage, read> joint_axis: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> joint_dof_offset: array<u32>;
@group(0) @binding(7) var<storage, read> joint_reversed: array<u32>;
@group(0) @binding(8) var<storage, read> joint_enabled: array<u32>;
@group(0) @binding(9) var<storage, read> joint_limit_lower: array<f32>;
@group(0) @binding(10) var<storage, read> joint_limit_upper: array<f32>;
@group(0) @binding(11) var<storage, read> body_mass: array<f32>;
@group(0) @binding(12) var<storage, read> body_inertia: array<Mat3Cols>;
@group(0) @binding(13) var<storage, read> body_com: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read> body_flags: array<u32>;
@group(0) @binding(15) var<storage, read> body_world: array<i32>;
@group(0) @binding(16) var<storage, read> gravity: array<vec4<f32>>;
@group(0) @binding(17) var<storage, read> art_offsets: array<u32>;

@group(1) @binding(0) var<storage, read> body_q: array<Transform>;
@group(1) @binding(1) var<storage, read> joint_qd: array<f32>;
@group(1) @binding(2) var<storage, read> body_qd: array<Wrench>;
@group(1) @binding(3) var<storage, read> acceleration: array<f32>;
@group(1) @binding(4) var<storage, read> applied_loads: array<Wrench>;
@group(1) @binding(5) var<storage, read> selected_articulations: array<u32>;
@group(1) @binding(6) var<storage, read_write> efforts: array<f32>;
@group(1) @binding(7) var<storage, read_write> wrenches: array<Wrench>;
@group(1) @binding(8) var<storage, read_write> status: atomic<u32>;
@group(1) @binding(9) var<storage, read> boundary_acceleration: array<Wrench>;
@group(2) @binding(0) var<uniform> uniforms: Uniforms;

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w*b.xyz + b.w*a.xyz + cross(a.xyz, b.xyz), a.w*b.w - dot(a.xyz, b.xyz));
}
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w*v);
}
fn transform_compose(a: Transform, b: Transform) -> Transform {
    return Transform(vec4<f32>(a.position.xyz + quat_rotate(a.rotation, b.position.xyz), 0.0), normalize(quat_mul(a.rotation, b.rotation)));
}
fn spatial_inertia(mass: f32, center: vec3<f32>, inertia: mat3x3<f32>, motion: Spatial) -> Spatial {
    let linear = mass * (motion.linear + cross(motion.angular, center));
    return Spatial(linear, inertia * motion.angular + cross(center, linear));
}
fn finite(v: vec3<f32>) -> bool {
    return all(abs(v) <= vec3<f32>(3.402823e38));
}

@compute @workgroup_size(32)
fn reaction_wrenches(@builtin(global_invocation_id) gid: vec3<u32>) {
    let art = gid.x;
    if art >= uniforms.articulations { return; }
    let begin = art_offsets[art];
    let end = art_offsets[art + 1u];
    let count = end - begin;
    if count > 64u { atomicOr(&status, 2u); return; }
    if count == 0u { return; }
    for (var j = begin; j < end; j += 1u) {
        wrenches[j] = Wrench(vec4<f32>(0.0), vec4<f32>(0.0));
        for (var d = joint_dof_offset[j]; d < joint_dof_offset[j + 1u]; d += 1u) { efforts[d] = 0.0; }
    }
    if selected_articulations[art] == 0u { return; }
    for (var j = begin; j < end; j += 1u) {
        let child = u32(joint_child[j]);
        let prescribed = boundary_acceleration[child];
        if !finite(prescribed.linear.xyz) || !finite(prescribed.angular.xyz) {
            atomicOr(&status, 1u); return;
        }
        if (body_flags[child] & 3u) != 1u && (any(prescribed.linear.xyz != vec3<f32>(0.0)) || any(prescribed.angular.xyz != vec3<f32>(0.0))) {
            atomicOr(&status, 4u); return;
        }
    }
    let origin = body_q[u32(joint_child[begin])].position.xyz;
    var parents: array<i32, 64>;
    var frames: array<Transform, 64>;
    var velocity: array<Spatial, 64>;
    var motion_acceleration: array<Spatial, 64>;
    var force: array<Spatial, 64>;
    for (var k = 0u; k < count; k += 1u) {
        let j = begin + k;
        let child = u32(joint_child[j]);
        let pose = body_q[child];
        let center = (pose.position.xyz - origin) + quat_rotate(pose.rotation, body_com[child].xyz);
        let frame = response_frame(j, 0u);
        frames[k] = frame;
        parents[k] = -1;
        var vp = Spatial(vec3<f32>(0.0), vec3<f32>(0.0));
        let g = gravity[u32(max(body_world[child], 0))].xyz;
        var ap = Spatial(-g, vec3<f32>(0.0));
        if joint_parent[j] >= 0 {
            for (var p = 0u; p < k; p += 1u) {
                if joint_child[begin + p] == joint_parent[j] {
                    parents[k] = i32(p);
                    vp = velocity[p];
                    ap = motion_acceleration[p];
                    break;
                }
            }
        }
        var vj = Spatial(vec3<f32>(0.0), vec3<f32>(0.0));
        var aj = vj;
        let width = response_width(joint_type[j]);
        let base = joint_dof_offset[j];
        for (var d = 0u; d < width; d += 1u) {
            let column = response_column(j, d, frame, origin);
            vj.linear += column.linear * joint_qd[base + d];
            vj.angular += column.angular * joint_qd[base + d];
            aj.linear += column.linear * acceleration[base + d];
            aj.angular += column.angular * acceleration[base + d];
        }
        var vi = Spatial(vp.linear + vj.linear, vp.angular + vj.angular);
        var ai = Spatial(ap.linear + aj.linear + cross(vi.angular, vj.linear) + cross(vi.linear, vj.angular),
                         ap.angular + aj.angular + cross(vi.angular, vj.angular));
        if joint_type[j] >= 4u && joint_type[j] <= 6u {
            let translation = quat_rotate(frame.rotation, vec3<f32>(joint_qd[base], joint_qd[base+1u], joint_qd[base+2u]));
            ai.linear -= cross(vj.angular, translation);
        }
        if (body_flags[child] & 3u) != 0u {
            vi = Spatial(vec3<f32>(0.0), vec3<f32>(0.0));
            ai = Spatial(-g, vec3<f32>(0.0));
            if (body_flags[child] & 2u) == 0u {
                let prescribed = body_qd[child];
                vi = Spatial(prescribed.linear.xyz - cross(prescribed.angular.xyz, center), prescribed.angular.xyz);
                ai.linear -= cross(prescribed.angular.xyz, prescribed.linear.xyz);
                let derivative = boundary_acceleration[child];
                ai.linear += derivative.linear.xyz - cross(derivative.angular.xyz, center);
                ai.angular = derivative.angular.xyz;
            }
        }
        velocity[k] = vi;
        motion_acceleration[k] = ai;
        let rotation = mat3x3<f32>(quat_rotate(pose.rotation, vec3<f32>(1.0, 0.0, 0.0)),
                                  quat_rotate(pose.rotation, vec3<f32>(0.0, 1.0, 0.0)),
                                  quat_rotate(pose.rotation, vec3<f32>(0.0, 0.0, 1.0)));
        let local = body_inertia[child];
        let inertia = rotation * mat3x3<f32>(local.col0.xyz, local.col1.xyz, local.col2.xyz) * transpose(rotation);
        let momentum = spatial_inertia(body_mass[child], center, inertia, vi);
        let inertial = spatial_inertia(body_mass[child], center, inertia, ai);
        let load = applied_loads[child];
        force[k] = Spatial(inertial.linear + cross(vi.angular, momentum.linear) - load.linear.xyz,
            inertial.angular + cross(vi.angular, momentum.angular) + cross(vi.linear, momentum.linear)
            - load.angular.xyz - cross(center, load.linear.xyz));
    }
    for (var reverse = 0u; reverse < count; reverse += 1u) {
        let k = count - 1u - reverse;
        let j = begin + k;
        let child = body_q[u32(joint_child[j])];
        let anchor = (child.position.xyz - origin) + quat_rotate(child.rotation, joint_x_c[j].position.xyz);
        let torque = force[k].angular - cross(anchor, force[k].linear);
        if !finite(force[k].linear) || !finite(torque) { atomicOr(&status, 1u); }
        wrenches[j] = Wrench(vec4<f32>(force[k].linear, 0.0), vec4<f32>(torque, 0.0));
        for (var d = 0u; d < response_width(joint_type[j]); d += 1u) {
            let column = response_column(j, d, frames[k], origin);
            let effort = dot(column.linear, force[k].linear) + dot(column.angular, force[k].angular);
            efforts[joint_dof_offset[j] + d] = effort;
            if !(abs(effort) <= 3.402823e38) { atomicOr(&status, 1u); }
        }
        if parents[k] >= 0 {
            let p = u32(parents[k]);
            force[p].linear += force[k].linear;
            force[p].angular += force[k].angular;
        }
    }
}
