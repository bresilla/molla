// GPU broad-phase query for the soft-triangle BVH. The CPU
// `SoftBvh::query_point_swept` returns exactly the tris whose AABB
// intersects the swept-point AABB (margin-inflated); the BVH is only an
// acceleration structure, so a flat AABB sweep over `tri_aabbs`
// reproduces the identical candidate set. One thread per (query, tri).

struct Uniforms {
    counts: vec4<u32>,   // (tri_count, query_count, pair_count, _)
};

// 2 vec4 per tri: [min.xyz, _], [max.xyz, _].
@group(0) @binding(0) var<storage, read> tri_aabbs: array<vec4<f32>>;
// 2 vec4 per query: [p_t0.xyz, margin], [p_t1.xyz, _].
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;

@group(1) @binding(0) var<storage, read_write> out_flags: array<u32>;

@group(2) @binding(0) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn query_swept(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let tri_count = u.counts.x;
    let pair_count = u.counts.z;
    if (i >= pair_count) { return; }

    let qi = i / tri_count;
    let ti = i % tri_count;

    let amin = tri_aabbs[2u * ti + 0u].xyz;
    let amax = tri_aabbs[2u * ti + 1u].xyz;

    let q0e = queries[2u * qi + 0u];
    let p0 = q0e.xyz;
    let margin = q0e.w;
    let p1 = queries[2u * qi + 1u].xyz;
    let q_min = min(p0, p1);
    let q_max = max(p0, p1);

    let overlap =
        amin.x - margin <= q_max.x && amax.x + margin >= q_min.x &&
        amin.y - margin <= q_max.y && amax.y + margin >= q_min.y &&
        amin.z - margin <= q_max.z && amax.z + margin >= q_min.z;

    out_flags[i] = select(0u, 1u, overlap);
}
