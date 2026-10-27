// Stackless instance and mesh traversal with separate infinite planes.

struct Params {
    count: u32, max_distance: f32, root: u32, plane_offset: u32,
    plane_count: u32, _pad0: u32, _pad1: u32, _pad2: u32,
}
struct Node { left: u32, right: u32, shape: u32, escape: u32 }
struct Hit { t: f32, shape: i32, mesh: u32, point: vec3<f32>, normal: vec3<f32> }

@group(0) @binding(0) var<storage, read> shapes: array<u32>;
@group(0) @binding(1) var<storage, read> meshes: array<u32>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> lower: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> upper: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> bodies: array<Transform>;
@group(0) @binding(6) var<storage, read> origins: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> directions: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> hit_t: array<f32>;
@group(0) @binding(9) var<storage, read_write> hit_shape: array<i32>;
@group(0) @binding(10) var<storage, read_write> hit_point: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> hit_normal: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> params: Params;

fn shape_vec(i: u32) -> vec4<f32> {
    return bitcast<vec4<f32>>(vec4<u32>(shapes[i], shapes[i+1u], shapes[i+2u], shapes[i+3u]));
}
fn mesh_vec(i: u32) -> vec3<f32> {
    return bitcast<vec3<f32>>(vec3<u32>(meshes[i], meshes[i+1u], meshes[i+2u]));
}
fn intersects_box(o: vec3<f32>, d: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, limit: f32) -> bool {
    var near = 0.0;
    var far = limit;
    for (var axis = 0u; axis < 3u; axis++) {
        if (d[axis] == 0.0) {
            if (o[axis] < lo[axis] || o[axis] > hi[axis]) { return false; }
        } else {
            let a = (lo[axis] - o[axis]) / d[axis];
            let b = (hi[axis] - o[axis]) / d[axis];
            near = max(near, min(a, b));
            far = min(far, max(a, b));
            if (far < near) { return false; }
        }
    }
    return true;
}
fn miss() -> Hit {
    return Hit(params.max_distance, -1, 1u, vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0));
}
fn valid_scene_pose(pose: Transform) -> bool {
    let bits = bitcast<vec4<u32>>(pose.rotation) & vec4<u32>(0x7f800000u);
    return all(abs(pose.position.xyz) <= vec3<f32>(0x1p120f))
        && all(bits != vec4<u32>(0x7f800000u))
        && all(abs(pose.rotation) <= vec4<f32>(2.0))
        && abs(dot(pose.rotation, pose.rotation) - 1.0) <= 2e-5;
}
fn valid_plane_pose(id: u32) -> bool {
    let base = id * 24u;
    if (!valid_scene_pose(Transform(shape_vec(base+8u), shape_vec(base+12u)))) { return false; }
    let body = bitcast<i32>(shapes[base+1u]);
    if (body < 0) { return true; }
    if (u32(body) >= arrayLength(&bodies)) { return false; }
    return valid_scene_pose(bodies[u32(body)]);
}
fn write_invalid(i: u32) {
    hit_t[i] = -2.0;
    hit_shape[i] = -2;
    hit_point[i] = vec4<f32>(0.0);
    hit_normal[i] = vec4<f32>(0.0);
}
fn wins(candidate: Hit, best: Hit) -> bool {
    if (candidate.shape < 0 || !(candidate.t > 0.0 && candidate.t < params.max_distance)) { return false; }
    if (best.shape < 0 || candidate.t < best.t) { return true; }
    if (candidate.t != best.t) { return false; }
    return candidate.mesh < best.mesh || (candidate.mesh == best.mesh && candidate.shape < best.shape);
}
fn shape_hit(id: u32, ro: vec3<f32>, rd: vec3<f32>, metric_scale: f32) -> Hit {
    let base = id * 24u;
    let kind = shapes[base];
    let body_id = bitcast<i32>(shapes[base+1u]);
    var pose = Transform(shape_vec(base+8u), shape_vec(base+12u));
    let is_mesh = shapes[base+20u] != 0u;
    if (body_id >= 0) {
        let parent = bodies[u32(body_id)];
        if (is_mesh) {
            pose.position = vec4<f32>(parent.position.xyz + quat_rotate(parent.rotation, pose.position.xyz), 0.0);
            let a = parent.rotation;
            let b = pose.rotation;
            pose.rotation = vec4<f32>(a.w*b.xyz + b.w*a.xyz + cross(a.xyz, b.xyz), a.w*b.w - dot(a.xyz, b.xyz));
        } else {
            pose = transform_compose(parent, pose);
        }
    }
    var hit = miss();
    if (!is_mesh) {
        let data = shape_vec(base+4u).xyz;
        let local_hit = ray_shape(kind, data, pose, ro, rd, params.max_distance / metric_scale);
        let t = local_hit.t * metric_scale;
        if (t > 0.0 && t < params.max_distance) {
            let p = local_hit.point;
            hit = Hit(t, i32(id), 0u, transform_point(pose, p), normalize(quat_rotate(pose.rotation, shape_normal(kind, data, p))));
        }
        return hit;
    }
    let inv = quat_conj(pose.rotation);
    let lo = quat_rotate(inv, ro-pose.position.xyz);
    let ld = quat_rotate(inv, rd);
    hit.t = params.max_distance / metric_scale;
    let node_offset = shapes[base+16u];
    let order_offset = shapes[base+17u];
    let index_offset = shapes[base+18u];
    let vertex_offset = shapes[base+19u];
    var index = 0u;
    var normal = vec3<f32>(0.0);
    while (index != 0xffffffffu) {
        let node = node_offset + index*12u;
        var next = meshes[node+7u];
        if (intersects_box(lo, ld, mesh_vec(node), mesh_vec(node+4u), hit.t)) {
            if (meshes[node+3u] == 1u) {
                let start = meshes[node+8u];
                let count = meshes[node+9u];
                for (var k = 0u; k < count; k++) {
                    let tri = meshes[order_offset+start+k];
                    let index = index_offset+tri*3u;
                    let a = mesh_vec(vertex_offset+meshes[index]*4u);
                    let b = mesh_vec(vertex_offset+meshes[index+1u]*4u);
                    let c = mesh_vec(vertex_offset+meshes[index+2u]*4u);
                    let triangle = mesh_triangle_hit(lo, ld, a, b, c);
                    let t = triangle.t;
                    if (t > 0.0 && t < hit.t) {
                        hit.t = t;
                        hit.shape = i32(id);
                        normal = triangle.normal;
                    }
                }
            } else {
                next = meshes[node+11u];
            }
        }
        index = next;
    }
    if (hit.shape >= 0) {
        let geometric = quat_rotate(pose.rotation, normalize(normal));
        let n2 = dot(geometric, geometric);
        if (n2 > 1e-24) {
            hit.normal = geometric * inverseSqrt(n2);
            if (dot(hit.normal, rd) > 0.0) { hit.normal = -hit.normal; }
        }
        hit.point = ro + rd*hit.t;
        hit.t *= metric_scale;
    }
    return hit;
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    if (params.root != 0xffffffffu) {
        if (lower[params.root].w != 0.0) { write_invalid(i); return; }
    }
    for (var p = 0u; p < params.plane_count; p++) {
        if (!valid_plane_pose(shapes[params.plane_offset+p])) { write_invalid(i); return; }
    }
    let ro = origins[i].xyz;
    let rd = ray_scaled_direction(directions[i].xyz);
    let len = length(rd);
    var best = miss();
    if (len >= 1e-12) {
        var index = params.root;
        while (index != 0xffffffffu) {
            let node = nodes[index];
            var next = node.escape;
            let margin = min(abs(best.t)*1e-5 + 1e-5, (3.4028234e38 - best.t)*0.5);
            let limit = (best.t + margin) / len;
            if (intersects_box(ro, rd, lower[index].xyz, upper[index].xyz, limit)) {
                if (node.left == 0xffffffffu) {
                    let candidate = shape_hit(node.shape, ro, rd, len);
                    if (wins(candidate, best)) { best = candidate; }
                } else {
                    next = node.right;
                }
            }
            index = next;
        }
        for (var p = 0u; p < params.plane_count; p++) {
            let candidate = shape_hit(shapes[params.plane_offset+p], ro, rd, len);
            if (wins(candidate, best)) { best = candidate; }
        }
    }
    hit_t[i] = select(NO_HIT_T, best.t, best.shape >= 0);
    hit_shape[i] = best.shape;
    hit_point[i] = vec4<f32>(best.point, 0.0);
    hit_normal[i] = vec4<f32>(best.normal, 0.0);
}
