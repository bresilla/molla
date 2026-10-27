struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Wrench { linear: vec4<f32>, angular: vec4<f32> }
struct Joint { parent: i32, child: u32, compiled: i32, reversed: u32,
    parent_frame: Transform, child_frame: Transform, row_start: u32, row_count: u32, pad0: u32, pad1: u32 }
struct Mean { child: Wrench, parent: Wrench, peaks: vec2<f32>, count: u32, pad: u32 }
struct Sample { child: Wrench, parent: Wrench, parent_anchor: vec4<f32>, child_anchor: vec4<f32>, tag: vec4<u32> }
struct Params { dt: f32, weight: f32, substep: u32, joints: u32, bodies: u32, dofs: u32, capture: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> joints: array<Joint>;
@group(0) @binding(1) var<storage, read> row_indices: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> row_forces: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> poses: array<Transform>;
@group(0) @binding(4) var<storage, read> centers: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> body_loads: array<Wrench>;
@group(0) @binding(6) var<storage, read> solved_delta: array<f32>;
@group(0) @binding(7) var<storage, read_write> acceleration: array<f32>;
@group(0) @binding(8) var<storage, read_write> external_loads: array<Wrench>;
@group(0) @binding(9) var<storage, read> tree_wrenches: array<Wrench>;
@group(0) @binding(10) var<storage, read> selected_joints: array<u32>;
@group(0) @binding(11) var<storage, read_write> means: array<Mean>;
@group(0) @binding(12) var<storage, read_write> samples: array<Sample>;
@group(0) @binding(13) var<storage, read> reaction_status: array<u32>;
@group(0) @binding(14) var<storage, read_write> solver_status: atomic<u32>;
@group(0) @binding(15) var<uniform> params: Params;
@group(0) @binding(16) var<storage, read> body_offsets: array<u32>;
@group(0) @binding(17) var<storage, read> body_joints: array<u32>;
@group(0) @binding(18) var<storage, read> break_limits: array<vec2<f32>>;
@group(0) @binding(19) var<storage, read_write> break_header: array<atomic<u32>>;
@group(0) @binding(20) var<storage, read_write> break_events: array<Sample>;
@group(0) @binding(21) var<storage, read> control_loads: array<Wrench>;
@group(0) @binding(22) var<storage, read> compliant_loads: array<Wrench>;
@group(0) @binding(23) var<storage, read> compliant_parent_loads: array<Wrench>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w*v);
}
fn anchor(body: i32, frame: Transform) -> vec3<f32> {
    if body < 0 { return frame.position.xyz; }
    let pose = poses[u32(body)];
    return pose.position.xyz + rotate(pose.rotation, frame.position.xyz);
}
fn loop_wrench(joint: Joint) -> Wrench {
    var load = Wrench(vec4<f32>(0.0), vec4<f32>(0.0));
    for (var row = joint.row_start; row < joint.row_start + joint.row_count; row += 1u) {
        let record = row_indices[row];
        if record.y == 0u { load.linear += row_forces[record.x]; }
        else if record.y == 1u { load.angular += row_forces[record.x]; }
        else {
            let applied = compliant_loads[record.x];
            let pose = poses[joint.child];
            let arm = rotate(pose.rotation,joint.child_frame.position.xyz-centers[joint.child].xyz);
            load.linear += applied.linear;
            load.angular += vec4<f32>(applied.angular.xyz-cross(arm,applied.linear.xyz),0.0);
        }
    }
    return load;
}
fn compliant(joint: Joint) -> bool {
    for (var row = joint.row_start; row < joint.row_start+joint.row_count; row++) {
        if row_indices[row].y == 2u { return true; }
    }
    return false;
}
fn loop_parent_wrench(joint: Joint) -> Wrench {
    var load = Wrench(vec4<f32>(0.0),vec4<f32>(0.0));
    var arm = vec3<f32>(0.0);
    if joint.parent >= 0 {
        let parent = u32(joint.parent);
        arm = rotate(poses[parent].rotation,joint.parent_frame.position.xyz-centers[parent].xyz);
    }
    for (var row = joint.row_start; row < joint.row_start+joint.row_count; row++) {
        let record = row_indices[row];
        if record.y == 2u {
            let applied = compliant_parent_loads[record.x];
            load.linear += applied.linear;
            load.angular += vec4<f32>(applied.angular.xyz-cross(arm,applied.linear.xyz),0.0);
        }
    }
    return load;
}
fn opposite(load: Wrench, point_a: vec3<f32>, point_b: vec3<f32>) -> Wrench {
    return Wrench(-load.linear, vec4<f32>(-load.angular.xyz - cross(point_a-point_b, load.linear.xyz), 0.0));
}
fn finite(v: vec3<f32>) -> bool { return all(abs(v) <= vec3<f32>(3.402823e38)); }
fn magnitude(v: vec3<f32>) -> f32 {
    var scaled = v;
    var factor = 1.0;
    let largest = max(abs(v.x), max(abs(v.y), abs(v.z)));
    if largest > 1e18 { scaled *= 1e-20; factor = 1e20; }
    else if largest < 1e-18 { scaled *= 1e20; factor = 1e-20; }
    let scale = max(abs(scaled.x), max(abs(scaled.y), abs(scaled.z)));
    if scale == 0.0 { return 0.0; }
    let norm = length(scaled / scale);
    if factor > 1.0 && scale > (3.402823e38 / factor) / norm {
        atomicOr(&solver_status, 16384u);
        return 0.0;
    }
    return (scale * norm) * factor;
}

@compute @workgroup_size(64)
fn assemble(@builtin(global_invocation_id) id: vec3<u32>) {
    if atomicLoad(&solver_status) != 0u { return; }
    let index = id.x;
    if index < params.dofs { acceleration[index] = solved_delta[index] / params.dt; }
    if index >= params.bodies { return; }
    var load = body_loads[index];
    load.linear += control_loads[index].linear;
    load.angular += control_loads[index].angular;
    let center = poses[index].position.xyz + rotate(poses[index].rotation, centers[index].xyz);
    for (var entry = body_offsets[index]; entry < body_offsets[index + 1u]; entry += 1u) {
        let j = body_joints[entry];
        let joint = joints[j];
        if selected_joints[j] == 0u { continue; }
        if joint.child == index || joint.parent == i32(index) {
            let reaction = loop_wrench(joint);
            if joint.child == index {
                let point = anchor(i32(joint.child), joint.child_frame);
                var arm = point-center;
                if compliant(joint) { arm = rotate(poses[index].rotation,joint.child_frame.position.xyz-centers[index].xyz); }
                load.linear += reaction.linear;
                load.angular += vec4<f32>(reaction.angular.xyz + cross(arm, reaction.linear.xyz), 0.0);
            }
            if joint.parent == i32(index) {
                let point = anchor(joint.parent, joint.parent_frame);
                var parent = Wrench(-reaction.linear,-reaction.angular);
                if joint.pad0 != 0u {
                    parent = opposite(reaction,anchor(i32(joint.child),joint.child_frame),point);
                }
                var arm = point-center;
                if compliant(joint) {
                    parent = loop_parent_wrench(joint);
                    arm = rotate(poses[index].rotation,joint.parent_frame.position.xyz-centers[index].xyz);
                }
                load.linear += parent.linear;
                load.angular += vec4<f32>(parent.angular.xyz+cross(arm,parent.linear.xyz),0.0);
            }
        }
    }
    external_loads[index] = load;
}

@compute @workgroup_size(64)
fn store(@builtin(global_invocation_id) id: vec3<u32>) {
    if atomicLoad(&solver_status) != 0u { return; }
    if reaction_status[0] != 0u { atomicOr(&solver_status, 16384u); }
    if atomicLoad(&solver_status) != 0u || id.x >= params.joints { return; }
    let j = id.x;
    var mean = Mean(Wrench(vec4<f32>(0.0), vec4<f32>(0.0)), Wrench(vec4<f32>(0.0), vec4<f32>(0.0)), vec2<f32>(0.0), 0u, 0u);
    if params.substep != 0u { mean = means[j]; }
    let slot = params.substep * params.joints + j;
    var sample = Sample(Wrench(vec4<f32>(0.0), vec4<f32>(0.0)), Wrench(vec4<f32>(0.0), vec4<f32>(0.0)), vec4<f32>(0.0), vec4<f32>(0.0), vec4<u32>(0u));
    if selected_joints[j] != 0u {
        let joint = joints[j];
        let a = anchor(joint.parent, joint.parent_frame);
        let b = anchor(i32(joint.child), joint.child_frame);
        sample.parent_anchor = vec4<f32>(a, 0.0);
        sample.child_anchor = vec4<f32>(b, 0.0);
        sample.tag.x = 1u;
        if joint.compiled < 0 {
            sample.child = loop_wrench(joint);
            sample.parent = Wrench(-sample.child.linear, -sample.child.angular);
            if joint.pad0 != 0u { sample.parent = opposite(sample.child,b,a); }
            if compliant(joint) { sample.parent = loop_parent_wrench(joint); }
        } else if joint.reversed == 0u {
            sample.child = tree_wrenches[u32(joint.compiled)];
            sample.parent = opposite(sample.child, b, a);
        } else {
            sample.parent = tree_wrenches[u32(joint.compiled)];
            sample.child = opposite(sample.parent, a, b);
        }
        let peak = vec2<f32>(magnitude(sample.child.linear.xyz), magnitude(sample.child.angular.xyz));
        mean.child.linear += sample.child.linear * params.weight;
        mean.child.angular += sample.child.angular * params.weight;
        mean.parent.linear += sample.parent.linear * params.weight;
        mean.parent.angular += sample.parent.angular * params.weight;
        mean.peaks = max(mean.peaks, peak);
        mean.count += 1u;
        if !finite(a) || !finite(b) || !finite(sample.child.linear.xyz) || !finite(sample.child.angular.xyz)
            || !finite(sample.parent.linear.xyz) || !finite(sample.parent.angular.xyz)
            || !finite(mean.child.linear.xyz) || !finite(mean.child.angular.xyz)
            || !finite(mean.parent.linear.xyz) || !finite(mean.parent.angular.xyz)
            || !all(peak <= vec2<f32>(3.402823e38)) { atomicOr(&solver_status, 16384u); return; }
        let limits = break_limits[j];
        let trigger = select(0u, 1u, peak.x >= limits.x) | select(0u, 2u, peak.y >= limits.y);
        if trigger != 0u {
            var event = sample;
            event.tag = vec4<u32>(1u, trigger, j, params.substep);
            let slot = atomicAdd(&break_header[1], 1u);
            break_events[slot] = event;
            atomicStore(&break_header[0], params.substep + 1u);
        }
    }
    if params.capture != 0u { samples[slot] = sample; }
    means[j] = mean;
}

@compute @workgroup_size(1)
fn halt() {
    if atomicLoad(&break_header[0]) != 0u { atomicOr(&solver_status, 1073741824u); }
}
