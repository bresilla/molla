// Mesh → voxel signed-distance bake on the GPU. Mirrors the CPU
// `bake_mesh_to_sdf` in `crates/molla-geometry/src/sdf_bake.rs`:
//   - unsigned distance = min over all triangles of the
//     closest-point-on-triangle distance (Ericson §5.1.5),
//   - sign = parity of +X ray crossings (Möller–Trumbore), with the
//     ray's y/z nudged off-axis exactly as the CPU does.
//
// One thread per voxel; the whole grid is independent so no BVH is
// needed for the parity-sized meshes this validates against. The ray
// starts at the grid origin (not 1e6 out like the f64 CPU path) so the
// crossing test stays well-conditioned in f32 while counting the same
// crossings.

struct Uniforms {
    origin: vec4<f32>,   // .xyz = grid origin, .w = cell_size
    dims:   vec4<u32>,   // (res, tri_count, total, _pad)
};

@group(0) @binding(0) var<storage, read> mesh_vertices: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> mesh_indices:  array<u32>;

@group(1) @binding(0) var<storage, read_write> out_dist: array<f32>;

@group(2) @binding(0) var<uniform> u: Uniforms;

// Closest point on triangle (a, b, c) to p — barycentric region
// classification, Ericson "Real-Time Collision Detection" §5.1.5.
fn closest_pt_tri(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return a; }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return b; }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return c; }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + ab * v + ac * w;
}

// +X ray vs triangle (Möller–Trumbore). Returns hit distance t > 0
// along +X, or -1 for a miss.
fn ray_x_tri(origin: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    let dir = vec3<f32>(1.0, 0.0, 0.0);
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(dir, edge2);
    let a = dot(edge1, h);
    if (abs(a) < 1.0e-14) { return -1.0; }
    let f = 1.0 / a;
    let s = origin - v0;
    let uu = f * dot(s, h);
    if (uu < 0.0 || uu > 1.0) { return -1.0; }
    let q = cross(s, edge1);
    let vv = f * dot(dir, q);
    if (vv < 0.0 || uu + vv > 1.0) { return -1.0; }
    let t = f * dot(edge2, q);
    if (t <= 0.0) { return -1.0; }
    return t;
}

@compute @workgroup_size(64)
fn bake_sdf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let res = u.dims.x;
    let tri_count = u.dims.y;
    let total = u.dims.z;
    if (i >= total) { return; }

    let x = i % res;
    let y = (i / res) % res;
    let z = i / (res * res);
    let cell = u.origin.w;
    let org = u.origin.xyz;
    let p = org + vec3<f32>(f32(x), f32(y), f32(z)) * cell;

    // Off-axis nudges match the CPU so the crossing set agrees.
    let nudge_y = cell * 1.0e-3 * 1.4142135623730951;
    let nudge_z = cell * 1.0e-3 * 0.7853981633974483;
    let ray_org = vec3<f32>(
        org.x,
        org.y + f32(y) * cell + nudge_y,
        org.z + f32(z) * cell + nudge_z,
    );
    let threshold = f32(x) * cell;

    var best2 = 3.4028235e38;
    var crossings = 0u;
    for (var t = 0u; t < tri_count; t = t + 1u) {
        let v0 = mesh_vertices[mesh_indices[3u * t + 0u]].xyz;
        let v1 = mesh_vertices[mesh_indices[3u * t + 1u]].xyz;
        let v2 = mesh_vertices[mesh_indices[3u * t + 2u]].xyz;
        let q = closest_pt_tri(p, v0, v1, v2);
        let dq = p - q;
        best2 = min(best2, dot(dq, dq));
        let th = ray_x_tri(ray_org, v0, v1, v2);
        if (th > 0.0 && th <= threshold) { crossings = crossings + 1u; }
    }

    let unsigned = sqrt(best2);
    var result = unsigned;
    if ((crossings & 1u) == 1u) { result = -unsigned; }
    out_dist[i] = result;
}
