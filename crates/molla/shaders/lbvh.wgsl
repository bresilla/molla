// LBVH (Linear BVH) construction over an array of primitive AABBs,
// following Karras (2012). Three compute kernels run in sequence:
//
//   morton_codes — one thread per primitive: compute the Morton code
//                  of the centroid (mapped into the global AABB to
//                  [0, 2^10) per axis), and emit (code, primitive_idx).
//   build_internal — one thread per internal node: determine the
//                  child range via the longest-common-prefix split,
//                  store left/right child indices in `nodes`.
//   refit_aabbs   — one thread per leaf, walking up to the root with
//                  atomic min/max to union child AABBs.
//
// `primitive_aabb` is sorted-by-Morton-code on the host (radix sort
// is its own kernel; for v1 we let the CPU sort and just upload the
// sorted indices). Internal nodes live at indices `[0, n-1)`; leaves
// at `[n-1, 2n-1)` mirroring the standard Karras layout.
//
// Inputs / outputs match the host-side `LbvhBuild` helper. Used by
// the GPU broad phase as a per-frame replacement for SAP when the
// pair count grows past the SAP scratch budget.

struct Aabb {
    min: vec4<f32>,
    max: vec4<f32>,
};

struct LbvhNode {
    aabb_min:    vec4<f32>,
    aabb_max:    vec4<f32>,
    left:        u32,
    right:       u32,
    parent:      u32,
    is_leaf:     u32,
};

struct Uniforms {
    primitive_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    scene_min:       vec4<f32>,
    scene_max:       vec4<f32>,
};

@group(0) @binding(0) var<storage, read>      primitive_aabb:    array<Aabb>;
@group(0) @binding(1) var<storage, read>      sorted_idx:        array<u32>;
@group(0) @binding(2) var<storage, read>      sorted_codes:      array<u32>;
@group(0) @binding(3) var<storage, read_write> nodes:            array<LbvhNode>;
@group(0) @binding(4) var<storage, read_write> visited:          array<atomic<u32>>;
@group(1) @binding(0) var<uniform>             u: Uniforms;

// Pair-output bind group — populated by `find_pairs`. `pair_count` is
// an atomic counter; `pair_a` / `pair_b` are append-style arrays sized
// to the host capacity (host re-issues a larger dispatch if it
// overflowed). `pair_count[1]` is the high-water mark before clamp.
@group(2) @binding(0) var<storage, read_write> pair_count: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> pair_a:     array<u32>;
@group(2) @binding(2) var<storage, read_write> pair_b:     array<u32>;

// Spread the lower 10 bits of `v` across every 3rd bit slot.
fn spread_bits(v: u32) -> u32 {
    var x = v & 0x000003ffu;
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u))  & 0x0300f00fu;
    x = (x | (x << 4u))  & 0x030c30c3u;
    x = (x | (x << 2u))  & 0x09249249u;
    return x;
}

fn morton3(p: vec3<f32>) -> u32 {
    // p ∈ [0, 1) per component → 30-bit Morton code.
    let xi = u32(clamp(p.x, 0.0, 0.99999) * 1024.0);
    let yi = u32(clamp(p.y, 0.0, 0.99999) * 1024.0);
    let zi = u32(clamp(p.z, 0.0, 0.99999) * 1024.0);
    return spread_bits(xi) | (spread_bits(yi) << 1u) | (spread_bits(zi) << 2u);
}

@compute @workgroup_size(64)
fn morton_codes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.primitive_count) { return; }
    let aabb = primitive_aabb[i];
    let centroid = (aabb.min.xyz + aabb.max.xyz) * 0.5;
    let extent = max(u.scene_max.xyz - u.scene_min.xyz, vec3<f32>(1e-9));
    let normalized = (centroid - u.scene_min.xyz) / extent;
    let code = morton3(normalized);
    // Re-use the `nodes` storage's scratch slots for (code, idx)
    // pairs — host re-uploads after sorting, so we only need the
    // codes here; the idx is implicit (= i before sort).
    nodes[i].left = code;
    nodes[i].right = i;
}

// Karras-style child-range split using longest-common-prefix on
// successive Morton codes. Each thread owns one INTERNAL node
// (index `i ∈ [0, n-1)`) and computes its `left` / `right` children.
fn delta(i: i32, j: i32) -> i32 {
    let n = i32(u.primitive_count);
    if (j < 0 || j >= n) { return -1; }
    let ci = sorted_codes[i];
    let cj = sorted_codes[j];
    if (ci == cj) {
        // Tie-break by primitive index (sorted).
        return 32 + i32(countLeadingZeros(u32(i) ^ u32(j)));
    }
    return i32(countLeadingZeros(ci ^ cj));
}

@compute @workgroup_size(64)
fn build_internal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = i32(u.primitive_count);
    let idx = i32(gid.x);
    if (idx >= n - 1) { return; }
    // Determine direction of range (+1 or -1).
    let d_minus = delta(idx, idx - 1);
    let d_plus  = delta(idx, idx + 1);
    var dir: i32 = 1;
    if (d_minus > d_plus) { dir = -1; }
    // Compute upper bound for length of range.
    let d_min = min(d_minus, d_plus);
    var l_max: i32 = 2;
    while (delta(idx, idx + l_max * dir) > d_min) {
        l_max = l_max * 2;
        if (l_max > n) { break; }
    }
    // Binary search for the exact end.
    var l: i32 = 0;
    var t: i32 = l_max / 2;
    while (t >= 1) {
        if (delta(idx, idx + (l + t) * dir) > d_min) {
            l = l + t;
        }
        t = t / 2;
    }
    let j = idx + l * dir;
    let d_node = delta(idx, j);

    // Find split position with binary search.
    var s: i32 = 0;
    var div: i32 = 2;
    var t2: i32 = (l + (div - 1)) / div;
    while (t2 >= 1) {
        if (delta(idx, idx + (s + t2) * dir) > d_node) {
            s = s + t2;
        }
        div = div * 2;
        t2 = (l + (div - 1)) / div;
        if (div > l + 1) { break; }
    }
    let split = idx + s * dir + min(dir, 0);

    let left_idx  = u32(min(idx, j));
    let right_idx = u32(max(idx, j));
    let left_child  = select(u32(split) + u32(n) - 1u, u32(split),       u32(split) == left_idx);
    let right_child = select(u32(split + 1) + u32(n) - 1u, u32(split + 1), u32(split + 1) == right_idx);

    let me = u32(idx);
    nodes[me].left  = left_child;
    nodes[me].right = right_child;
    nodes[me].is_leaf = 0u;
    // Children record this as their parent.
    nodes[left_child].parent  = me;
    nodes[right_child].parent = me;
}

// Per-leaf traversal: walk the LBVH from the root and emit any pair
// (this leaf, other leaf) whose AABBs overlap. We only emit pairs
// where `me_prim < other_prim` to keep the output deduplicated.
//
// `pair_a` / `pair_b` hold the result; `pair_count[0]` is an atomic
// counter that doubles as a clamp index — appends past the host
// capacity are dropped (host can detect via `pair_count[0] > cap` and
// re-dispatch with a bigger buffer next frame).
//
// Iterative stack-based traversal — WGSL has no recursion. Stack
// depth is bounded by `2*ceil(log2(n)) + 1` for the Karras layout;
// 64 is enough for n up to ~2^31. Most adapters allow 32 KB local
// memory per workgroup, but we keep the stack in registers (per
// thread) to dodge cross-thread coordination.
fn aabb_overlap(amin: vec3<f32>, amax: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    return all(amin <= bmax) && all(amax >= bmin);
}

@compute @workgroup_size(64)
fn find_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u.primitive_count;
    let leaf = gid.x;
    if (leaf >= n) { return; }
    let me_prim = sorted_idx[leaf];
    let me_aabb = primitive_aabb[me_prim];
    let me_min = me_aabb.min.xyz;
    let me_max = me_aabb.max.xyz;

    // No pairs in a 1-element scene.
    if (n < 2u) { return; }

    var stack: array<u32, 64>;
    var sp: i32 = 0;
    stack[0] = 0u; // root = internal node 0
    sp = 1;

    loop {
        if (sp <= 0) { break; }
        sp = sp - 1;
        let node_idx = stack[sp];
        let node = nodes[node_idx];
        // Leaf?
        if (node.is_leaf == 1u) {
            // The internal-node `parent` field of leaves is set, but
            // we identify "leaf" via `is_leaf`. The leaf's primitive
            // is `sorted_idx[leaf_local]` where `leaf_local = node_idx - (n-1)`.
            let leaf_local = node_idx - (n - 1u);
            if (leaf_local == leaf) { continue; } // skip self
            let other_prim = sorted_idx[leaf_local];
            // Dedup: emit only when me < other.
            if (me_prim >= other_prim) { continue; }
            if (!aabb_overlap(me_min, me_max,
                              primitive_aabb[other_prim].min.xyz,
                              primitive_aabb[other_prim].max.xyz)) {
                continue;
            }
            let slot = atomicAdd(&pair_count[0], 1u);
            // Always bump the high-water mark; clamp the write.
            let cap = arrayLength(&pair_a);
            if (slot < cap) {
                pair_a[slot] = me_prim;
                pair_b[slot] = other_prim;
            }
            continue;
        }
        // Internal — test against the node's stored AABB and recurse.
        if (!aabb_overlap(me_min, me_max, node.aabb_min.xyz, node.aabb_max.xyz)) {
            continue;
        }
        // Push children.
        if (sp + 2 <= 64) {
            stack[sp] = node.left;
            sp = sp + 1;
            stack[sp] = node.right;
            sp = sp + 1;
        }
    }
}

// Bottom-up AABB refit using `visited` atomic counters: each
// internal node waits until both children have been processed.
@compute @workgroup_size(64)
fn refit_aabbs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u.primitive_count;
    let leaf = gid.x;
    if (leaf >= n) { return; }
    // Initialise the leaf node from the primitive AABB.
    let leaf_idx = (n - 1u) + leaf;
    let prim = sorted_idx[leaf];
    let aabb = primitive_aabb[prim];
    nodes[leaf_idx].aabb_min = aabb.min;
    nodes[leaf_idx].aabb_max = aabb.max;
    nodes[leaf_idx].is_leaf = 1u;
    // Walk to root, atomically incrementing each ancestor's visited
    // counter. The thread that bumps it from 1 → 2 unions the
    // children and continues; the other thread bails.
    var cur = nodes[leaf_idx].parent;
    loop {
        if (cur == 0xffffffffu) { break; }
        let count = atomicAdd(&visited[cur], 1u);
        if (count == 0u) {
            break; // Other child hasn't arrived yet.
        }
        let l = nodes[cur].left;
        let r = nodes[cur].right;
        nodes[cur].aabb_min = min(nodes[l].aabb_min, nodes[r].aabb_min);
        nodes[cur].aabb_max = max(nodes[l].aabb_max, nodes[r].aabb_max);
        if (cur == 0u) { break; }
        cur = nodes[cur].parent;
    }
}
