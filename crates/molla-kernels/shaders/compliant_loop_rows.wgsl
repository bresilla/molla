struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Velocity { linear: vec4<f32>, angular: vec4<f32> }
struct Request {
    parent_frame: Transform, child_frame: Transform, metadata: vec4<i32>,
    range: vec4<f32>, feedback: vec4<f32>,
    parent_position_residue: vec4<f32>, child_position_residue: vec4<f32>,
    parent_rotation_residue: vec4<f32>, child_rotation_residue: vec4<f32>
}
struct Row {
    metadata: vec4<i32>, child_linear: vec4<f32>, child_angular: vec4<f32>,
    parent_linear: vec4<f32>, parent_angular: vec4<f32>, feedback: vec4<f32>
}
@group(0) @binding(0) var<storage, read> requests: array<Request>;
@group(0) @binding(1) var<storage, read> poses: array<Transform>;
@group(0) @binding(2) var<storage, read> velocities: array<Velocity>;
@group(0) @binding(3) var<storage, read> centers: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> position_residues: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> rows: array<Row>;
@group(0) @binding(6) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(7) var<storage, read_write> rounding: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> body_rotations: array<QuaternionPair>;

fn rounded(slot: u32, value: f32) -> f32 {
    atomicStore(&rounding[slot],bitcast<u32>(value));
    return bitcast<f32>(atomicLoad(&rounding[slot]));
}
fn two_sum(slot: u32, a: f32, b: f32) -> vec2<f32> {
    let sum = rounded(slot,a+b);
    let virtual_b = rounded(slot,sum-a);
    let virtual_a = rounded(slot,sum-virtual_b);
    return vec2<f32>(sum,(a-virtual_a)+(b-virtual_b));
}

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn finite(v: vec4<f32>) -> bool { return all(abs(v) <= vec4<f32>(3.402823e38)); }
fn rotation(slot: u32, body: i32, frame: Transform, residue: vec4<f32>) -> QuaternionPair {
    let packed = QuaternionPair(frame.rotation,residue);
    if body < 0 { return packed; }
    return multiply(slot,body_rotations[u32(body)],packed);
}
fn anchor(slot: u32, body: i32, frame: Transform, frame_residue: vec3<f32>) -> mat2x3<f32> {
    if body < 0 { return mat2x3<f32>(frame.position.xyz,frame_residue); }
    let pose = poses[u32(body)];
    let arm = rotate(pose.rotation,frame.position.xyz);
    var high = vec3<f32>(0.0);
    var low = position_residues[u32(body)].xyz + rotate(pose.rotation,frame_residue);
    for (var axis = 0u; axis < 3u; axis++) {
        let sum = two_sum(slot,pose.position[axis],arm[axis]);
        high[axis] = sum.x;
        low[axis] += sum.y;
    }
    return mat2x3<f32>(high,low);
}

@compute @workgroup_size(64)
fn evaluate_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x;
    if slot >= arrayLength(&requests) { return; }
    rows[slot] = Row(vec4<i32>(0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0));
    if atomicLoad(&status) != 0u { return; }
    let request = requests[slot];
    let parent = request.metadata.x;
    let child = request.metadata.y;
    let axis = request.metadata.z;
    if parent < -1 || child < 0 || parent == child || axis < 0 || axis >= 6
        || child >= i32(arrayLength(&poses)) || parent >= i32(arrayLength(&poses))
        || request.metadata.w < 0 || request.metadata.w > 3 {
        atomicOr(&status,32768u); return;
    }
    let a = anchor(slot,parent,request.parent_frame,request.parent_position_residue.xyz);
    let b = anchor(slot,child,request.child_frame,request.child_position_residue.xyz);
    var delta = vec3<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis++) {
        let difference = two_sum(slot,b[0][axis],-a[0][axis]);
        let low_difference = rounded(slot,b[1][axis]-a[1][axis]);
        let low = rounded(slot,difference.y+low_difference);
        delta[axis] = rounded(slot,difference.x+low);
    }
    let parent_pair = rotation(slot,parent,request.parent_frame,request.parent_rotation_residue);
    let child_pair = rotation(slot,child,request.child_frame,request.child_rotation_residue);
    let parent_rotation = normalize(parent_pair.high+parent_pair.low);
    let translation = rotate(vec4<f32>(-parent_rotation.xyz,parent_rotation.w),delta);
    let inverse_parent = QuaternionPair(vec4<f32>(-parent_pair.high.xyz,parent_pair.high.w),
        vec4<f32>(-parent_pair.low.xyz,parent_pair.low.w));
    let relative_pair = multiply(slot,inverse_parent,child_pair);
    var relative = relative_pair.high+relative_pair.low;
    if relative.w < 0.0 { relative = -relative; }
    relative = normalize(relative);
    let sine = length(relative.xyz);
    var log_rotation = 2.0*relative.xyz;
    if sine <= 0.1 {
        let squared = sine*sine;
        log_rotation *= 1.0+squared*(1.0/6.0+squared*(3.0/40.0+squared*(5.0/112.0+squared*35.0/1152.0)));
    } else { log_rotation = relative.xyz * (2.0*atan2(sine,relative.w)/sine); }
    var position = translation[u32(axis)%3u];
    var direction = vec3<f32>(0.0);
    direction[u32(axis)%3u] = 1.0;
    if axis >= 3 {
        position = log_rotation[u32(axis)-3u];
        let angle = length(log_rotation);
        var coefficient = 1.0/12.0 + angle*angle/720.0;
        if angle >= 1e-4 { coefficient = (1.0-0.5*angle/tan(0.5*angle))/(angle*angle); }
        direction += 0.5*cross(log_rotation,direction) + coefficient*cross(log_rotation,cross(log_rotation,direction));
    }
    direction = rotate(parent_rotation,direction);
    var error = position;
    var lower = -request.range.w;
    var upper = request.range.w;
    var engaged = false;
    if request.metadata.w == 0 {
        let lo = request.range.x;
        let hi = request.range.y;
        engaged = request.feedback.x > 0.0 && (lo == hi || position < lo || position > hi);
        error -= clamp(position,lo,hi);
        if lo != hi {
            if position < lo { lower = 0.0; }
            else { upper = 0.0; }
        }
    } else if request.metadata.w == 3 {
        error = 0.0;
        lower = clamp(request.range.z,-request.range.w,request.range.w);
        upper = lower;
        engaged = true;
    } else {
        error -= request.range.z;
        if request.metadata.w == 2 { error = 0.0; }
        engaged = request.feedback.x > 0.0 || request.feedback.y > 0.0;
    }
    if !engaged { return; }
    var child_linear = direction;
    var child_angular = cross(rotate(poses[u32(child)].rotation,
        request.child_frame.position.xyz + request.child_position_residue.xyz - centers[u32(child)].xyz),direction);
    var parent_linear = -direction;
    var parent_angular = -cross(delta,direction);
    if parent >= 0 {
        let arm = delta + rotate(poses[u32(parent)].rotation,
            request.parent_frame.position.xyz + request.parent_position_residue.xyz - centers[u32(parent)].xyz);
        parent_angular = -cross(arm,direction);
    }
    if axis >= 3 {
        child_linear = vec3<f32>(0.0);
        parent_linear = vec3<f32>(0.0);
        child_angular = direction;
        parent_angular = -direction;
    }
    var rate = dot(child_linear,velocities[u32(child)].linear.xyz) + dot(child_angular,velocities[u32(child)].angular.xyz);
    if parent >= 0 {
        rate += dot(parent_linear,velocities[u32(parent)].linear.xyz) + dot(parent_angular,velocities[u32(parent)].angular.xyz);
    }
    if !finite(vec4<f32>(child_linear,error)) || !finite(vec4<f32>(child_angular,rate))
        || !finite(vec4<f32>(parent_angular,0.0)) || !finite(request.feedback) {
        atomicOr(&status,32768u); return;
    }
    let mode = select(1,2,request.metadata.w == 3);
    rows[slot] = Row(vec4<i32>(parent,child,axis,mode),vec4<f32>(child_linear,error),vec4<f32>(child_angular,rate),
        vec4<f32>(parent_linear,lower),vec4<f32>(parent_angular,upper),request.feedback);
}
