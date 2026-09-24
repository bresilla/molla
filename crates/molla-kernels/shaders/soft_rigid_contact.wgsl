struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Shape {
    position: vec4<f32>, rotation: vec4<f32>, data: vec4<f32>, ids: vec4<u32>,
}
struct Contact {
    normal_gap: vec4<f32>, material: vec4<f32>, ids: vec4<u32>,
    support: vec4<u32>, weights: vec4<f32>,
}
struct Params {
    node_count: u32, shape_count: u32, particle_count: u32, body_count: u32,
    margin: f32, sampled: u32, capacity: u32, tiles: u32,
}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> radii: array<f32>;
@group(0) @binding(2) var<storage, read> surface: array<u32>;
@group(0) @binding(3) var<storage, read> shapes: array<Shape>;
@group(0) @binding(4) var<storage, read> poses: array<Transform>;
@group(0) @binding(5) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read_write> ranges: array<u32>;
var<workgroup> prefix: array<u32, 64>;

fn rotate(q: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    return p + 2.0 * cross(q.xyz, cross(q.xyz, p) + q.w * p);
}
fn multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w*b.xyz + b.w*a.xyz + cross(a.xyz,b.xyz), a.w*b.w-dot(a.xyz,b.xyz));
}
fn nonzero_sign(p: vec3<f32>) -> vec3<f32> {
    return select(vec3<f32>(-1.0), vec3<f32>(1.0), p >= vec3<f32>(0.0));
}
fn box_distance(q: vec3<f32>, s: vec3<f32>) -> vec4<f32> {
    let outside = max(q, vec3<f32>(0.0));
    let r = length(outside);
    var n = vec3<f32>(0.0);
    if r > 1e-12 { n = s * outside / r; }
    else if q.x >= q.y && q.x >= q.z { n.x = s.x; }
    else if q.y >= q.z { n.y = s.y; }
    else { n.z = s.z; }
    return vec4<f32>(n, r + min(max(q.x, max(q.y, q.z)), 0.0));
}
struct PrismSample { value: vec4<f32>, anchor: vec3<f32>, planar: bool }
fn prism_distance(p: vec3<f32>, data: vec3<f32>, top: f32) -> PrismSample {
    let corners = array<vec2<f32>, 4>(vec2<f32>(-data.y, -data.z),
        vec2<f32>(data.y, -data.z), vec2<f32>(top, data.z), vec2<f32>(-top, data.z));
    var squared = 3.402823e38;
    var offset = vec2<f32>(0.0);
    var edge_normal = vec2<f32>(0.0);
    var anchor = vec3<f32>(0.0);
    var on_edge = false;
    var inside = true;
    for (var i = 0u; i < 4u; i += 1u) {
        let edge = corners[(i + 1u) % 4u] - corners[i];
        let delta = p.yz - corners[i];
        inside = inside && edge.x * delta.y - edge.y * delta.x >= 0.0;
        let fraction = clamp(dot(delta, edge) / dot(edge, edge), 0.0, 1.0);
        let nearest = delta - edge * fraction;
        let candidate = dot(nearest, nearest);
        if candidate < squared {
            squared = candidate;
            offset = nearest;
            edge_normal = normalize(vec2<f32>(edge.y, -edge.x));
            anchor = vec3<f32>(0.0, corners[i]);
            on_edge = fraction > 0.0 && fraction < 1.0;
        }
    }
    let planar_length = sqrt(squared);
    let planar_gap = select(planar_length, -planar_length, inside);
    var planar_normal = edge_normal;
    if planar_length > 1e-12 {
        planar_normal = offset * select(1.0, -1.0, inside) / planar_length;
    }
    let cap_gap = abs(p.x) - data.x;
    let cap = vec3<f32>(nonzero_sign(p).x, 0.0, 0.0);
    let side = vec3<f32>(0.0, planar_normal);
    let outside = length(max(vec2<f32>(cap_gap, planar_gap), vec2<f32>(0.0)));
    var normal = select(side, cap, cap_gap >= planar_gap);
    if outside > 1e-12 {
        normal = (cap * max(cap_gap, 0.0) + side * max(planar_gap, 0.0)) / outside;
    }
    var planar = on_edge;
    if cap_gap >= planar_gap {
        anchor = cap * data.x;
        planar = true;
    }
    if cap_gap > 0.0 && planar_gap > 0.0 { planar = false; }
    if planar { normal = select(vec3<f32>(0.0, edge_normal), cap, cap_gap >= planar_gap); }
    return PrismSample(vec4<f32>(normal, outside + min(max(cap_gap, planar_gap), 0.0)), anchor, planar);
}
fn distance(kind: u32, data: vec3<f32>, p: vec3<f32>) -> vec4<f32> {
    if kind == 0u {
        let r = length(p);
        var n = vec3<f32>(0.0, 1.0, 0.0);
        if r > 1e-12 { n = p / r; }
        return vec4<f32>(n, r - data.x);
    }
    if kind == 1u {
        return box_distance(abs(p) - data, nonzero_sign(p));
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

fn contact_at(row: u32) -> Contact {
    let shape_id = row / params.node_count;
    let index = row % params.node_count;
    var nodes = vec4<u32>(surface[index], 0u, 0u, 1u);
    var weights = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    if params.sampled != 0u {
        let start = index * 8u;
        nodes = vec4<u32>(surface[start], surface[start+1u], surface[start+2u], surface[start+3u]);
        weights = bitcast<vec4<f32>>(vec4<u32>(surface[start+4u], surface[start+5u], surface[start+6u], surface[start+7u]));
    }
    let node = nodes.x;
    let shape = shapes[shape_id];
    var result: Contact;
    result.normal_gap = vec4<f32>(0.0);
    result.material = vec4<f32>(shape.data.w, 0.0, 0.0, 0.0);
    result.ids = vec4<u32>(node, shape.ids.y, shape_id, 0u);
    result.support = vec4<u32>(nodes.yz, nodes.w, 0u);
    result.weights = weights;
    if node >= params.particle_count || shape.ids.x > 3u
        || (shape.ids.y != 0xffffffffu && shape.ids.y >= params.body_count)
        || nodes.w == 0u || nodes.w > 3u {
        result.ids.w = 2u;
        return result;
    }
    if shape.ids.x == 3u && !(all(shape.data.xyz >= vec3<f32>(1e-15))
        && all(shape.data.xyz < vec3<f32>(1e15)) && shape.position.w >= 1e-15 && shape.position.w < 1e15) {
        result.ids.w = 2u;
        return result;
    }
    var point = vec3<f32>(0.0);
    var radius = 0.0;
    var weight_sum = 0.0;
    for (var i = 0u; i < nodes.w; i += 1u) {
        var valid = nodes[i] < params.particle_count && weights[i] >= 0.0 && weights[i] <= 1.0;
        for (var j = 0u; j < i; j += 1u) { valid = valid && nodes[i] != nodes[j]; }
        if !valid {
            result.ids.w = 2u; return result;
        }
        point += weights[i] * positions[nodes[i]].xyz;
        radius += weights[i] * radii[nodes[i]];
        weight_sum += weights[i];
    }
    if abs(weight_sum - 1.0) > 1e-5 || !(dot(point, point) <= 3.402823e38)
        || !(radius >= 0.0 && radius <= 3.402823e38) {
        result.ids.w = 2u; return result;
    }
    var center = shape.position.xyz;
    var rotation = shape.rotation;
    if shape.ids.y != 0xffffffffu {
        let pose = poses[shape.ids.y];
        center = pose.position.xyz + rotate(pose.rotation, center);
        rotation = multiply(pose.rotation, rotation);
    }
    let local = rotate(vec4<f32>(-rotation.xyz, rotation.w), point - center);
    var sample: vec4<f32>;
    if shape.ids.x == 1u {
        let signs = nonzero_sign(local);
        let nx = rotate(rotation, vec3<f32>(signs.x, 0.0, 0.0));
        let ny = rotate(rotation, vec3<f32>(0.0, signs.y, 0.0));
        let nz = rotate(rotation, vec3<f32>(0.0, 0.0, signs.z));
        let face_x = center + nx * shape.data.x;
        let face_y = center + ny * shape.data.y;
        let face_z = center + nz * shape.data.z;
        var gaps = vec3<f32>(0.0);
        for (var i = 0u; i < nodes.w; i += 1u) {
            let p = positions[nodes[i]].xyz;
            gaps += weights[i] * vec3<f32>(dot(nx, p - face_x), dot(ny, p - face_y), dot(nz, p - face_z));
        }
        sample = box_distance(gaps, signs);
    } else if shape.ids.x == 3u {
        let prism = prism_distance(local, shape.data.xyz, shape.position.w);
        sample = prism.value;
        if prism.planar {
            let normal = rotate(rotation, sample.xyz);
            let face = center + rotate(rotation, prism.anchor);
            sample.w = 0.0;
            for (var i = 0u; i < nodes.w; i += 1u) {
                sample.w += weights[i] * dot(normal, positions[nodes[i]].xyz - face);
            }
        }
    } else {
        sample = distance(shape.ids.x, shape.data.xyz, local);
    }
    result.normal_gap = vec4<f32>(rotate(rotation, sample.xyz), sample.w - radius);
    result.ids.w = select(0u, 1u, result.normal_gap.w <= params.margin);
    return result;
}

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < params.node_count * params.shape_count { contacts[gid.x] = contact_at(gid.x); }
}

fn scan(enabled: u32, lane: u32) {
    prefix[lane] = enabled;
    workgroupBarrier();
    for (var offset = 1u; offset < 64u; offset *= 2u) {
        var previous = 0u;
        if lane >= offset { previous = prefix[lane - offset]; }
        workgroupBarrier();
        prefix[lane] += previous;
        workgroupBarrier();
    }
}

@compute @workgroup_size(64)
fn count_contacts(@builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let tile = group.x + group.y * groups.x;
    if tile >= params.tiles { return; }
    let row = tile * 64u + lane;
    var enabled = 0u;
    if row < params.node_count * params.shape_count {
        enabled = select(0u, 1u, contact_at(row).ids.w != 0u);
    }
    scan(enabled, lane);
    if lane == 63u { ranges[tile] = prefix[63]; }
}

@compute @workgroup_size(1)
fn offset_contacts() {
    var total = 0u;
    for (var tile = 0u; tile < params.tiles; tile += 1u) {
        let count = ranges[tile];
        ranges[tile] = total;
        total += count;
    }
    ranges[params.tiles] = total;
    if total > params.capacity { contacts[0].ids.w = 4u; }
}

@compute @workgroup_size(64)
fn scatter_contacts(@builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let tile = group.x + group.y * groups.x;
    if tile >= params.tiles { return; }
    let row = tile * 64u + lane;
    var result: Contact;
    var enabled = 0u;
    if row < params.node_count * params.shape_count && ranges[params.tiles] <= params.capacity {
        result = contact_at(row);
        enabled = select(0u, 1u, result.ids.w != 0u);
    }
    scan(enabled, lane);
    if enabled != 0u { contacts[ranges[tile] + prefix[lane] - 1u] = result; }
}
