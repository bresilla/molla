struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Shape {
    position: vec4<f32>, rotation: vec4<f32>, data: vec4<f32>, ids: vec4<u32>,
}
struct Contact { normal_gap: vec4<f32>, material: vec4<f32>, ids: vec4<u32> }
struct Params {
    node_count: u32, shape_count: u32, particle_count: u32, body_count: u32,
    margin: f32, pad0: u32, pad1: u32, pad2: u32,
}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> radii: array<f32>;
@group(0) @binding(2) var<storage, read> surface: array<u32>;
@group(0) @binding(3) var<storage, read> shapes: array<Shape>;
@group(0) @binding(4) var<storage, read> poses: array<Transform>;
@group(0) @binding(5) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(6) var<uniform> params: Params;

fn rotate(q: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    return p + 2.0 * cross(q.xyz, cross(q.xyz, p) + q.w * p);
}
fn multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w*b.xyz + b.w*a.xyz + cross(a.xyz,b.xyz), a.w*b.w-dot(a.xyz,b.xyz));
}
fn nonzero_sign(p: vec3<f32>) -> vec3<f32> {
    return select(vec3<f32>(-1.0), vec3<f32>(1.0), p >= vec3<f32>(0.0));
}
fn distance(kind: u32, data: vec3<f32>, p: vec3<f32>) -> vec4<f32> {
    if kind == 0u {
        let r = length(p);
        var n = vec3<f32>(0.0, 1.0, 0.0);
        if r > 1e-12 { n = p / r; }
        return vec4<f32>(n, r - data.x);
    }
    if kind == 1u {
        let q = abs(p) - data;
        let outside = max(q, vec3<f32>(0.0));
        let r = length(outside);
        let s = nonzero_sign(p);
        var n = vec3<f32>(0.0);
        if r > 1e-12 { n = s * outside / r; }
        else if q.x >= q.y && q.x >= q.z { n.x = s.x; }
        else if q.y >= q.z { n.y = s.y; }
        else { n.z = s.z; }
        return vec4<f32>(n, r + min(max(q.x, max(q.y, q.z)), 0.0));
    }
    let radial = vec3<f32>(p.x, 0.0, p.z);
    let r = length(radial);
    var side = vec3<f32>(1.0, 0.0, 0.0);
    if r > 1e-12 { side = radial / r; }
    let cap = vec3<f32>(0.0, nonzero_sign(p).y, 0.0);
    let dr = r - data.x;
    let dy = abs(p.y) - data.y;
    let outside = length(max(vec2<f32>(dr, dy), vec2<f32>(0.0)));
    var n = select(cap, side, dr >= dy);
    if outside > 1e-12 { n = (side * max(dr, 0.0) + cap * max(dy, 0.0)) / outside; }
    return vec4<f32>(n, outside + min(max(dr, dy), 0.0));
}

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.node_count * params.shape_count { return; }
    let shape_id = row / params.node_count;
    let node = surface[row % params.node_count];
    let shape = shapes[shape_id];
    var result: Contact;
    result.normal_gap = vec4<f32>(0.0);
    result.material = vec4<f32>(shape.data.w, 0.0, 0.0, 0.0);
    result.ids = vec4<u32>(node, shape.ids.y, shape_id, 0u);
    if node >= params.particle_count || shape.ids.x > 2u
        || (shape.ids.y != 0xffffffffu && shape.ids.y >= params.body_count) {
        result.ids.w = 2u;
        contacts[row] = result;
        return;
    }
    var center = shape.position.xyz;
    var rotation = shape.rotation;
    if shape.ids.y != 0xffffffffu {
        let pose = poses[shape.ids.y];
        center = pose.position.xyz + rotate(pose.rotation, center);
        rotation = multiply(pose.rotation, rotation);
    }
    let local = rotate(vec4<f32>(-rotation.xyz, rotation.w), positions[node].xyz - center);
    let sample = distance(shape.ids.x, shape.data.xyz, local);
    result.normal_gap = vec4<f32>(rotate(rotation, sample.xyz), sample.w - radii[node]);
    result.ids.w = select(0u, 1u, result.normal_gap.w <= params.margin);
    contacts[row] = result;
}
