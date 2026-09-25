// Local-space BVH traversal with rigid instance poses and world-space normals.
// Mesh upload validates tree depth against the per-ray stack capacity.

struct BvhNode {
    aabb_min: vec4<f32>,   // .xyz = min; .w = is_leaf (as f32 0 or 1)
    aabb_max: vec4<f32>,   // .xyz = max; .w = padding
    payload: vec4<u32>,    // (prim_start, tri_count, left, right)
};

struct Uniforms {
    ray_count:    u32,
    node_count:   u32,
    max_distance: f32,
    shape_id:     u32,
    body_id:      i32,
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
};

struct Transform { position: vec4<f32>, rotation: vec4<f32> }

@group(0) @binding(0) var<storage, read> bvh_nodes:    array<BvhNode>;
@group(0) @binding(1) var<storage, read> tri_order:    array<u32>;
@group(0) @binding(2) var<storage, read> mesh_indices: array<u32>;
@group(0) @binding(3) var<storage, read> mesh_vertices: array<vec4<f32>>;

@group(1) @binding(0) var<storage, read> ray_origin:    array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> ray_direction: array<vec4<f32>>;

@group(2) @binding(0) var<storage, read_write> hit_t:      array<f32>;
@group(2) @binding(1) var<storage, read_write> hit_tri:    array<i32>;
@group(2) @binding(2) var<storage, read_write> hit_normal: array<vec4<f32>>;

@group(3) @binding(0) var<uniform> u: Uniforms;
@group(3) @binding(1) var<storage, read> body_poses: array<Transform>;
@group(3) @binding(2) var<storage, read> shape_poses: array<Transform>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

const MAX_STACK: u32 = 64u;

fn ray_scaled_direction(direction: vec3<f32>) -> vec3<f32> {
    let magnitude = max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
    if (magnitude == 0.0 || (bitcast<u32>(magnitude) & 0x7f800000u) == 0x7f800000u) {
        return vec3<f32>(0.0);
    }
    let exponent = clamp((bitcast<u32>(magnitude) >> 23u) & 255u, 1u, 253u);
    return direction * bitcast<f32>((254u - exponent) << 23u);
}

fn ray_aabb(origin: vec3<f32>, direction: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, max_t: f32) -> bool {
    var near = 0.0;
    var far = max_t;
    for (var axis = 0u; axis < 3u; axis++) {
        if (direction[axis] == 0.0) {
            if (origin[axis] < lo[axis] || origin[axis] > hi[axis]) { return false; }
        } else {
            let a = (lo[axis] - origin[axis]) / direction[axis];
            let b = (hi[axis] - origin[axis]) / direction[axis];
            near = max(near, min(a, b));
            far = min(far, max(a, b));
            if (far < near) { return false; }
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn raycast_mesh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.ray_count) { return; }
    hit_t[i] = u.max_distance;
    hit_tri[i] = -1;
    hit_normal[i] = vec4<f32>(0.0);

    if (u.node_count == 0u) { return; }

    let direction = ray_scaled_direction(ray_direction[i].xyz);
    let metric_scale = length(direction);
    if (metric_scale < 1e-12) { return; }

    var pose = shape_poses[u.shape_id];
    if (u.body_id >= 0) {
        let body = body_poses[u32(u.body_id)];
        pose.position = vec4<f32>(body.position.xyz + rotate(body.rotation, pose.position.xyz), 0.0);
        pose.rotation = multiply(body.rotation, pose.rotation);
    }
    let inverse_rotation = vec4<f32>(-pose.rotation.xyz, pose.rotation.w);
    let origin = rotate(inverse_rotation, ray_origin[i].xyz - pose.position.xyz);
    let dir = rotate(inverse_rotation, direction);

    var best_t: f32 = u.max_distance / metric_scale;
    var best_tri: i32 = -1;
    var best_n: vec3<f32> = vec3<f32>(0.0);

    var stack: array<u32, MAX_STACK>;
    var sp: u32 = 0u;
    stack[sp] = 0u;
    sp = sp + 1u;

    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let idx = stack[sp];
        if (idx >= u.node_count) { continue; }
        let node = bvh_nodes[idx];
        if (!ray_aabb(origin, dir, node.aabb_min.xyz, node.aabb_max.xyz, best_t)) { continue; }
        let is_leaf = u32(node.aabb_min.w);
        if (is_leaf == 1u) {
            let prim_start = node.payload.x;
            let tri_count = node.payload.y;
            for (var k: u32 = 0u; k < tri_count; k = k + 1u) {
                let tri_idx = tri_order[prim_start + k];
                let i0 = mesh_indices[tri_idx * 3u + 0u];
                let i1 = mesh_indices[tri_idx * 3u + 1u];
                let i2 = mesh_indices[tri_idx * 3u + 2u];
                let v0 = mesh_vertices[i0].xyz;
                let v1 = mesh_vertices[i1].xyz;
                let v2 = mesh_vertices[i2].xyz;
                let triangle = mesh_triangle_hit(origin, dir, v0, v1, v2);
                let t = triangle.t;
                if (t > 0.0 && t < best_t) {
                    best_t = t;
                    best_tri = i32(tri_idx);
                    best_n = triangle.normal;
                }
            }
        } else {
            if (sp + 2u <= MAX_STACK) {
                stack[sp] = node.payload.z; sp = sp + 1u;
                stack[sp] = node.payload.w; sp = sp + 1u;
            }
        }
    }

    hit_tri[i] = best_tri;
    if (best_tri >= 0) {
        hit_t[i] = best_t * metric_scale;
        hit_normal[i] = vec4<f32>(rotate(pose.rotation, normalize(best_n)), 0.0);
    }
}
