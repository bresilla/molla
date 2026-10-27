struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Instance { lo: vec4<f32>, hi: vec4<f32>, pose: Transform, ids: vec4<i32> }
struct Params { start: u32, count: u32, bodies: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> instances: array<Instance>;
@group(0) @binding(1) var<storage, read> topology: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bodies: array<Transform>;
@group(0) @binding(3) var<storage, read_write> lower: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> upper: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> poses: array<Transform>;
@group(0) @binding(6) var<storage, read_write> status: array<u32>;
@group(1) @binding(0) var<uniform> params: Params;
const MAX: f32 = 0x1.fffffep127f;
const SAFE: f32 = 0x1p120f;

fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
fn finite3(v: vec3<f32>) -> bool { return finite(v.x) && finite(v.y) && finite(v.z); }
fn valid_pose(p: Transform) -> bool {
    if !finite3(p.position.xyz) || !finite3(p.rotation.xyz) || !finite(p.rotation.w) { return false; }
    if any(abs(p.rotation) > vec4<f32>(2.0)) { return false; }
    return abs(dot(p.rotation, p.rotation) - 1.0) <= 2e-5;
}
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
fn multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w*b.xyz + b.w*a.xyz + cross(a.xyz, b.xyz), a.w*b.w-dot(a.xyz,b.xyz));
}
fn invalid(i: u32, code: u32) {
    lower[i] = vec4<f32>(vec3<f32>(-MAX), f32(code));
    upper[i] = vec4<f32>(vec3<f32>(MAX), 0.0);
    poses[i] = Transform(vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    status[i] = code;
}

@compute @workgroup_size(64)
fn leaves(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count { return; }
    let instance = instances[i];
    var pose = instance.pose;
    if !valid_pose(pose) || !finite3(instance.lo.xyz) || !finite3(instance.hi.xyz) {
        invalid(i, 1u); return;
    }
    if any(abs(pose.position.xyz) > vec3<f32>(SAFE)) || any(abs(instance.lo.xyz) > vec3<f32>(SAFE))
        || any(abs(instance.hi.xyz) > vec3<f32>(SAFE)) { invalid(i, 2u); return; }
    var scale = abs(pose.position.xyz) + abs(instance.lo.xyz) + abs(instance.hi.xyz);
    if instance.ids.y >= 0 {
        if u32(instance.ids.y) >= params.bodies { invalid(i, 1u); return; }
        let body = bodies[u32(instance.ids.y)];
        if !valid_pose(body) { invalid(i, 1u); return; }
        if any(abs(body.position.xyz) > vec3<f32>(SAFE)) { invalid(i, 2u); return; }
        scale += abs(body.position.xyz);
        pose.position = vec4<f32>(body.position.xyz + rotate(body.rotation, pose.position.xyz), 0.0);
        pose.rotation = multiply(body.rotation, pose.rotation);
    }
    let center = instance.lo.xyz * 0.5 + instance.hi.xyz * 0.5;
    let extent = instance.hi.xyz * 0.5 - instance.lo.xyz * 0.5;
    let world = rotate(pose.rotation, center) + pose.position.xyz;
    let radius = abs(rotate(pose.rotation, vec3<f32>(extent.x,0.0,0.0)))
        + abs(rotate(pose.rotation, vec3<f32>(0.0,extent.y,0.0)))
        + abs(rotate(pose.rotation, vec3<f32>(0.0,0.0,extent.z)));
    let magnitude = max(max(scale.x, scale.y), scale.z);
    let pad = vec3<f32>((magnitude + 1.0) * 0.000003814697265625);
    lower[i] = vec4<f32>(world - radius - pad, 0.0);
    upper[i] = vec4<f32>(world + radius + pad, 0.0);
    poses[i] = pose;
    status[i] = 0u;
}

@compute @workgroup_size(64)
fn parents(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.count { return; }
    let i = params.start + gid.x;
    let children = topology[i].xy;
    let flags = u32(lower[children.x].w) | u32(lower[children.y].w);
    lower[i] = vec4<f32>(min(lower[children.x].xyz, lower[children.y].xyz), f32(flags));
    upper[i] = vec4<f32>(max(upper[children.x].xyz, upper[children.y].xyz), 0.0);
}
