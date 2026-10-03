// Morton-ordered complete binary hierarchy: leaves [0,n), internals [n,2n-1).
struct Aabb { min: vec4<f32>, max: vec4<f32> };
struct LbvhNode {
    aabb_min: vec4<f32>, aabb_max: vec4<f32>,
    left: u32, right: u32, parent: u32, is_leaf: u32,
};
struct Uniforms {
    primitive_count: u32, first: u32, count: u32, pad: u32,
    scene_min: vec4<f32>, scene_max: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> primitive_aabb: array<Aabb>;
@group(0) @binding(1) var<storage, read> sorted_idx: array<u32>;
@group(0) @binding(2) var<storage, read> sorted_codes: array<u32>;
@group(0) @binding(3) var<storage, read_write> nodes: array<LbvhNode>;
@group(1) @binding(0) var<uniform> u: Uniforms;
@group(2) @binding(0) var<storage, read_write> pair_count: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> pair_a: array<u32>;
@group(2) @binding(2) var<storage, read_write> pair_b: array<u32>;

fn power(n: u32) -> u32 {
    return 1u << (32u - countLeadingZeros(n - 1u));
}
fn node_index(heap: u32, n: u32) -> u32 {
    if (heap < n - 1u) { return n + heap; }
    let p = power(n);
    if (heap >= p - 1u) { return heap - (p - 1u); }
    return heap - (n - 1u) + (2u * n - p);
}

@compute @workgroup_size(64)
fn build_hierarchy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u.primitive_count;
    let i = gid.x;
    if (i >= n) { return; }
    if (n == 1u) {
        nodes[0] = LbvhNode(vec4<f32>(0.0), vec4<f32>(0.0), 0u, 0u, 0xffffffffu, 1u);
        atomicStore(&pair_count[1], 0u);
        return;
    }
    let p = power(n);
    let bottom = 2u * n - p;
    var heap = p - 1u + i;
    if (i >= bottom) { heap = n - 1u + i - bottom; }
    let parent = node_index((heap - 1u) / 2u, n);
    nodes[i] = LbvhNode(vec4<f32>(0.0), vec4<f32>(0.0), 0u, 0u, parent, 1u);
    if (i < n - 1u) {
        var internal_parent = 0xffffffffu;
        if (i != 0u) { internal_parent = node_index((i - 1u) / 2u, n); }
        nodes[n + i] = LbvhNode(vec4<f32>(0.0), vec4<f32>(0.0),
            node_index(2u * i + 1u, n), node_index(2u * i + 2u, n), internal_parent, 0u);
    }
    if (i == 0u) { atomicStore(&pair_count[1], u.primitive_count); }
}

// Leaves first, then one dispatch per internal heap level, deepest to root.
@compute @workgroup_size(64)
fn refit_aabbs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.count) { return; }
    if (u.first == 0xffffffffu) {
        let aabb = primitive_aabb[sorted_idx[i]];
        nodes[i].aabb_min = aabb.min;
        nodes[i].aabb_max = aabb.max;
        return;
    }
    let index = u.primitive_count + u.first + i;
    let left = nodes[index].left;
    let right = nodes[index].right;
    nodes[index].aabb_min = min(nodes[left].aabb_min, nodes[right].aabb_min);
    nodes[index].aabb_max = max(nodes[left].aabb_max, nodes[right].aabb_max);
}

fn aabb_overlap(amin: vec3<f32>, amax: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    return all(amin <= bmax) && all(amax >= bmin);
}

@compute @workgroup_size(64)
fn find_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u.primitive_count;
    let leaf = gid.x;
    if (leaf >= n || n < 2u) { return; }
    let me = sorted_idx[leaf];
    let aabb = primitive_aabb[me];
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = atomicLoad(&pair_count[1]);
    loop {
        if (sp == 0u) { break; }
        sp -= 1u;
        let index = stack[sp];
        if (index < n) {
            let other = sorted_idx[index];
            if (me >= other) { continue; }
            let b = primitive_aabb[other];
            if (!aabb_overlap(aabb.min.xyz, aabb.max.xyz, b.min.xyz, b.max.xyz)) { continue; }
            let slot = atomicAdd(&pair_count[0], 1u);
            if (slot < arrayLength(&pair_a)) { pair_a[slot] = me; pair_b[slot] = other; }
        } else {
            let node = nodes[index];
            if (!aabb_overlap(aabb.min.xyz, aabb.max.xyz, node.aabb_min.xyz, node.aabb_max.xyz)) { continue; }
            stack[sp] = node.left;
            stack[sp + 1u] = node.right;
            sp += 2u;
        }
    }
}
