// LBVH (Linear BVH) construction over an array of primitive AABBs,
// using NVIDIA Warp's bottom-up build (`xtra/warp/warp/native/bvh.cu :
// build_hierarchy`). Three compute kernels:
//
//   morton_codes — one thread per primitive: compute the Morton code
//                  of the centroid (mapped into the global AABB to
//                  [0, 2^10) per axis), and emit (code, primitive_idx).
//                  Optional — host can compute on the CPU and upload
//                  the sorted (code, idx) arrays directly.
//   build_hierarchy — one thread per primitive (leaf): walks bottom-
//                  up choosing parent direction by which neighbour key
//                  delta is greater; writes parent link, child slot,
//                  range, and refit AABBs in the same pass via the
//                  per-internal-node atomic `visited` counter.
//   find_pairs   — one thread per leaf: traverse the tree from the
//                  root; emit overlapping pairs into the per-pair
//                  output buffers.
//
// Layout: leaves [0, n), internal nodes [n, 2n-1). Total nodes 2n-1.
// `visited` is sized `n - 1` (one counter per internal node, index
// `internal_idx - n`). Root is computed during the build pass and
// stored in `root_out[0]`.

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

// Pair-output bind group:
//   pair_count[0] — atomic pair-emit counter (high-water mark; if
//                   > arrayLength(&pair_a) the output overflowed).
//   pair_count[1] — root node index (written by build_hierarchy).
//   pair_count[2..] — reserved.
@group(2) @binding(0) var<storage, read_write> pair_count: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> pair_a:     array<u32>;
@group(2) @binding(2) var<storage, read_write> pair_b:     array<u32>;

// Morton codes are computed CPU-side at `molla-geometry/src/lbvh.rs::dispatch`
// — host-side codegen + AABB collection dominate over wgpu submit
// overhead at v1 scene sizes, so the on-device `morton_codes` /
// `morton3` / `spread_bits` entry points were never wired up and have
// been removed (see git log for the deleted scaffold).

// `delta(i)` = longest common prefix between `sorted_codes[i]` and
// `sorted_codes[i+1]`, with index-based tie-break for duplicate codes.
// Returns -1 if `i+1 >= n` so callers can compare across boundaries.
fn key_delta(i: i32) -> i32 {
    let n = i32(u.primitive_count);
    if (i < 0 || i + 1 >= n) { return -1; }
    let ca = sorted_codes[i];
    let cb = sorted_codes[i + 1];
    if (ca == cb) {
        // Degenerate codes: tie-break by primitive index.
        return 32 + i32(countLeadingZeros(u32(i) ^ u32(i + 1)));
    }
    return i32(countLeadingZeros(ca ^ cb));
}

// Bottom-up Karras build (port of warp's `build_hierarchy`).
// **Topology only** — writes parent/child/range fields. AABBs are
// written by `refit_aabbs` in a separate kernel pass so we get a
// global memory barrier between the two phases (WGSL has no
// cross-workgroup memory fence beyond the queue / kernel boundary).
//
// Layout: leaves at [0, n), internal nodes at [n, 2n-1). `visited`
// is sized `n - 1` and indexed `internal_idx - n`. Per-internal-node
// `range_left` / `range_right` is stashed in `aabb_min.w` /
// `aabb_max.w` lanes during the build — refit overwrites those
// lanes' xyz channels but preserves the .w stashes via explicit
// reads.
@compute @workgroup_size(64)
fn build_hierarchy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = i32(u.primitive_count);
    if (n < 2) {
        if (gid.x == 0u) {
            // Trivial 1-leaf tree: root is leaf 0.
            atomicStore(&pair_count[1], 0u);
            if (n == 1) {
                nodes[0].is_leaf = 1u;
                nodes[0].parent = 0xffffffffu;
                nodes[0].aabb_min.w = bitcast<f32>(0u);
                nodes[0].aabb_max.w = bitcast<f32>(0u);
            }
        }
        return;
    }
    let idx = i32(gid.x);
    if (idx >= n) { return; }

    // Per-leaf ascending range starts as [idx, idx]. Stash in `.w`
    // lanes; AABB xyz lanes are filled by refit_aabbs afterwards.
    nodes[idx].is_leaf = 1u;
    nodes[idx].aabb_min.w = bitcast<f32>(u32(idx));
    nodes[idx].aabb_max.w = bitcast<f32>(u32(idx));

    var cur: i32 = idx;
    loop {
        let left_range  = bitcast<i32>(nodes[cur].aabb_min.w);
        let right_range = bitcast<i32>(nodes[cur].aabb_max.w);

        // Are we the root?
        if (left_range == 0 && right_range == n - 1) {
            nodes[cur].parent = 0xffffffffu;
            atomicStore(&pair_count[1], u32(cur));
            return;
        }

        // Pick parent direction.
        var parent_right: bool = false;
        if (left_range == 0) {
            parent_right = true;
        } else if (right_range == n - 1) {
            parent_right = false;
        } else {
            let dl = key_delta(left_range - 1);
            let dr = key_delta(right_range);
            parent_right = dr >= dl;
        }

        var parent_idx: i32;
        var i_am_left: bool;
        if (parent_right) {
            parent_idx = right_range + n;
            i_am_left = true;
        } else {
            parent_idx = left_range - 1 + n;
            i_am_left = false;
        }

        nodes[cur].parent = u32(parent_idx);
        if (i_am_left) {
            nodes[parent_idx].left = u32(cur);
            nodes[parent_idx].aabb_min.w = bitcast<f32>(left_range);
        } else {
            nodes[parent_idx].right = u32(cur);
            nodes[parent_idx].aabb_max.w = bitcast<f32>(right_range);
        }

        // Rendezvous on the parent's arrival counter: the FIRST child
        // to arrive returns; the SECOND continues climbing. NOTE: the
        // second arrival then reads `nodes[parent]` range fields the
        // first arrival wrote — possibly from another workgroup.
        // WebGPU's memory model does NOT guarantee cross-workgroup
        // visibility of those non-atomic writes via this atomic alone;
        // correctness relies on the coherent memory that real GPU
        // drivers provide (verified by `make test-broadphase-lbvh-gpu`).
        // A strictly spec-compliant build would make the handed-off
        // Node range/AABB fields `atomic<u32>`.
        let visited_idx = parent_idx - n;
        let prior = atomicAdd(&visited[visited_idx], 1u);
        if (prior == 0u) {
            return;
        }
        nodes[parent_idx].is_leaf = 0u;
        cur = parent_idx;
    }
}

// AABB union pass. Run AFTER build_hierarchy (or any time you want
// to refresh AABBs without changing topology). Each leaf writes its
// primitive's AABB into `nodes[idx].aabb_xyz`, then walks up via
// `nodes[cur].parent` atomically counting arrivals.
//
// **Caller responsibility**: zero out the `visited` counters before
// dispatching this kernel — the same counters are reused from the
// build pass, where each one has already been bumped to 2.
@compute @workgroup_size(64)
fn refit_aabbs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = i32(u.primitive_count);
    if (n < 1) { return; }
    let idx = i32(gid.x);
    if (idx >= n) { return; }

    let prim = sorted_idx[u32(idx)];
    let aabb = primitive_aabb[prim];
    // Preserve the per-leaf range stash (.w lanes) so the next refit
    // can climb the same parent pointers.
    let lr = nodes[idx].aabb_min.w;
    let rr = nodes[idx].aabb_max.w;
    nodes[idx].aabb_min = vec4<f32>(aabb.min.xyz, lr);
    nodes[idx].aabb_max = vec4<f32>(aabb.max.xyz, rr);

    var cur: i32 = idx;
    loop {
        let parent = i32(nodes[cur].parent);
        if (parent < 0 || u32(parent) == 0xffffffffu) {
            return;
        }
        let visited_idx = parent - n;
        let prior = atomicAdd(&visited[visited_idx], 1u);
        if (prior == 0u) { return; }
        let l_child = nodes[parent].left;
        let r_child = nodes[parent].right;
        let lo = min(nodes[l_child].aabb_min.xyz, nodes[r_child].aabb_min.xyz);
        let hi = max(nodes[l_child].aabb_max.xyz, nodes[r_child].aabb_max.xyz);
        let lr2 = nodes[parent].aabb_min.w;
        let rr2 = nodes[parent].aabb_max.w;
        nodes[parent].aabb_min = vec4<f32>(lo, lr2);
        nodes[parent].aabb_max = vec4<f32>(hi, rr2);
        cur = parent;
    }
}

// Per-leaf traversal: walk the LBVH from the root and emit any pair
// (this leaf's primitive, other leaf's primitive) whose AABBs
// overlap. Only emit pairs where `me_prim < other_prim` to dedup.
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
    if (n < 2u) { return; }

    let root = atomicLoad(&pair_count[1]);

    var stack: array<u32, 64>;
    var sp: i32 = 0;
    stack[0] = root;
    sp = 1;

    loop {
        if (sp <= 0) { break; }
        sp = sp - 1;
        let node_idx = stack[sp];
        // Leaf? Layout puts leaves at [0, n). Equivalently check
        // `is_leaf == 1u` for safety.
        if (node_idx < n) {
            // Leaf primitive
            if (node_idx == leaf) { continue; }
            let other_prim = sorted_idx[node_idx];
            // Dedup
            if (me_prim >= other_prim) { continue; }
            if (!aabb_overlap(me_min, me_max,
                              primitive_aabb[other_prim].min.xyz,
                              primitive_aabb[other_prim].max.xyz)) {
                continue;
            }
            let slot = atomicAdd(&pair_count[0], 1u);
            let cap = arrayLength(&pair_a);
            if (slot < cap) {
                pair_a[slot] = me_prim;
                pair_b[slot] = other_prim;
            }
            continue;
        }
        // Internal — test against the node's stored AABB and recurse.
        let node = nodes[node_idx];
        if (!aabb_overlap(me_min, me_max, node.aabb_min.xyz, node.aabb_max.xyz)) {
            continue;
        }
        if (sp + 2 <= 64) {
            stack[sp] = node.left;
            sp = sp + 1;
            stack[sp] = node.right;
            sp = sp + 1;
        }
    }
}

// Old Karras top-down `build_internal` entry point removed: the
// stub had a long-standing parent-pointer race with our layout that
// the bottom-up `build_hierarchy` sidesteps entirely, and nothing on
// the host dispatched it. See git log if you need to resurrect the
// top-down side-by-side.
