// Per-mesh BVH ray-cast on the GPU. Mirrors `MeshBvh::ray_cast` from
// `crates/molla/src/geometry/mesh_bvh.rs`: iterative stack-based
// traversal of a flat `BvhNode` array, Möller–Trumbore for each
// triangle in a leaf.
//
// One thread per ray. Per-thread stack capped at MAX_STACK; deeper
// trees fall back to "no hit" (a future TODO; current trees stay
// well under 64 deep for ~1k-tri meshes).
//
// Outputs (per ray): `hit_t`, `hit_tri`, `hit_normal`. `hit_tri == -1`
// means no hit within `max_distance`.

struct BvhNode {
    aabb_min: vec4<f32>,   // .xyz = min; .w = is_leaf (as f32 0 or 1)
    aabb_max: vec4<f32>,   // .xyz = max; .w = padding
    payload: vec4<u32>,    // (prim_start, tri_count, left, right)
};

struct Uniforms {
    ray_count:    u32,
    node_count:   u32,
    max_distance: f32,
    _pad:         f32,
};

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

const MAX_STACK: u32 = 64u;

fn ray_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, max_t: f32) -> bool {
    let t1 = (lo - origin) * inv_dir;
    let t2 = (hi - origin) * inv_dir;
    let tmin_v = min(t1, t2);
    let tmax_v = max(t1, t2);
    let tmin = max(tmin_v.x, max(tmin_v.y, tmin_v.z));
    let tmax = min(tmax_v.x, min(tmax_v.y, tmax_v.z));
    return tmax >= max(tmin, 0.0) && tmin <= max_t;
}

// Möller–Trumbore. Returns `t > 0` on hit, `-1.0` on miss.
fn ray_triangle(origin: vec3<f32>, dir: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(dir, edge2);
    let a = dot(edge1, h);
    if (abs(a) < 1e-12) { return -1.0; }
    let f = 1.0 / a;
    let s = origin - v0;
    let uu = f * dot(s, h);
    if (uu < 0.0 || uu > 1.0) { return -1.0; }
    let q = cross(s, edge1);
    let vv = f * dot(dir, q);
    if (vv < 0.0 || uu + vv > 1.0) { return -1.0; }
    return f * dot(edge2, q);
}

@compute @workgroup_size(64)
fn raycast_mesh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.ray_count) { return; }
    hit_t[i] = u.max_distance;
    hit_tri[i] = -1;
    hit_normal[i] = vec4<f32>(0.0);

    if (u.node_count == 0u) { return; }

    let origin = ray_origin[i].xyz;
    let dir = ray_direction[i].xyz;
    let inv_dir = vec3<f32>(
        select(1e30, 1.0 / dir.x, abs(dir.x) > 1e-30),
        select(1e30, 1.0 / dir.y, abs(dir.y) > 1e-30),
        select(1e30, 1.0 / dir.z, abs(dir.z) > 1e-30),
    );

    var best_t: f32 = u.max_distance;
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
        if (!ray_aabb(origin, inv_dir, node.aabb_min.xyz, node.aabb_max.xyz, best_t)) { continue; }
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
                let t = ray_triangle(origin, dir, v0, v1, v2);
                if (t > 0.0 && t < best_t) {
                    best_t = t;
                    best_tri = i32(tri_idx);
                    best_n = cross(v1 - v0, v2 - v0);
                }
            }
        } else {
            if (sp + 2u <= MAX_STACK) {
                stack[sp] = node.payload.z; sp = sp + 1u;
                stack[sp] = node.payload.w; sp = sp + 1u;
            }
        }
    }

    hit_t[i] = best_t;
    hit_tri[i] = best_tri;
    if (best_tri >= 0) {
        hit_normal[i] = vec4<f32>(normalize(best_n), 0.0);
    }
}
