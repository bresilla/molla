// Stage 24e — NanoVDB-equivalent sparse-grid traversal.
//
// Three-level Root → Internal → Leaf lookup, no recursion (WGSL
// disallows it), no dynamic dispatch — every load is a flat index
// into one of five storage buffers. The Rust host packs the matching
// `SparseGrid` into the buffers via `SparseGridGpu::from_grid`; the
// WGSL only does read-only traversal.
//
// Bindings:
//   group(0) binding(0) : root_entries     i32[ , , , ]   linear table
//                         (4 entries per row: key_x, key_y, key_z,
//                          internal_idx)
//   group(0) binding(1) : internal_children u32[]         flat 32³ per
//                         internal node; u32::MAX = empty
//   group(0) binding(2) : internal_masks    u32[]         1024 u32s
//                         per internal node = 32³ bits
//   group(0) binding(3) : leaf_values       f32[]         512 f32s per leaf
//   group(0) binding(4) : leaf_masks        u32[]         16 u32s per
//                         leaf = 8³ bits
//   group(1) binding(0) : query_coords      vec4<i32>[]   one query
//                         per thread; .w unused (vec4 gets 16-byte
//                         alignment to dodge wgpu padding rules)
//   group(1) binding(1) : query_results     f32[]         one f32 per
//                         query; background returned for misses
//   group(2) binding(0) : uniforms          { query_count, root_entry_count,
//                                             background, _pad }
//
// The kernel is `lookup_kernel`, one thread per query. WGSL has no
// recursion so root-level lookup is a *linear scan* over root_entries
// — fine for v1 (root entry counts are usually < 100, even for 1024³
// volumes). If the root ever blows up to thousands of entries we'd
// switch to a GPU-side hash; for now linear-search is the simplest
// thing that works portably across every wgpu backend.
//
// Constants below MUST match `crates/molla/src/geometry/sparse_grid.rs`.

const LEAF_SHIFT_GPU:     u32 = 3u;
const INTERNAL_SHIFT_GPU: u32 = 8u;
const INTERNAL_DIM_GPU:   u32 = 32u;
const LEAF_DIM_GPU:       u32 = 8u;
const INVALID_CHILD:      u32 = 0xFFFFFFFFu;
const INTERNAL_NODE_VOXELS: u32 = 32768u; // 32³
const INTERNAL_NODE_MASK_U32S: u32 = 1024u; // 32³ / 32
const LEAF_NODE_VOXELS:   u32 = 512u; // 8³
const LEAF_NODE_MASK_U32S: u32 = 16u; // 8³ / 32

struct Uniforms {
    query_count:       u32,
    root_entry_count:  u32,
    background:        f32,
    _pad:              u32,
};

@group(0) @binding(0) var<storage, read> root_entries:      array<i32>;
@group(0) @binding(1) var<storage, read> internal_children: array<u32>;
@group(0) @binding(2) var<storage, read> internal_masks:    array<u32>;
@group(0) @binding(3) var<storage, read> leaf_values:       array<f32>;
@group(0) @binding(4) var<storage, read> leaf_masks:        array<u32>;

@group(1) @binding(0) var<storage, read>       query_coords:  array<vec4<i32>>;
@group(1) @binding(1) var<storage, read_write> query_results: array<f32>;

@group(2) @binding(0) var<uniform> uniforms: Uniforms;

// ── Helpers ────────────────────────────────────────────────────────

fn root_key_of(p: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        p.x >> INTERNAL_SHIFT_GPU,
        p.y >> INTERNAL_SHIFT_GPU,
        p.z >> INTERNAL_SHIFT_GPU,
    );
}

fn internal_local_of(p: vec3<i32>) -> u32 {
    let ix = u32((p.x >> LEAF_SHIFT_GPU) & 31);
    let iy = u32((p.y >> LEAF_SHIFT_GPU) & 31);
    let iz = u32((p.z >> LEAF_SHIFT_GPU) & 31);
    return ix + 32u * (iy + 32u * iz);
}

fn leaf_local_of(p: vec3<i32>) -> u32 {
    let lx = u32(p.x & 7);
    let ly = u32(p.y & 7);
    let lz = u32(p.z & 7);
    return lx + 8u * (ly + 8u * lz);
}

// Linear scan for the matching root entry. Returns either the
// internal-node index or `INVALID_CHILD` when the key is absent.
//
// Bounded loop (`uniforms.root_entry_count` is set by the host) so
// WGSL's "uniform control flow" rules are satisfied — no early return
// inside non-uniform branches, no `break` from a loop with a
// non-uniform condition; we *do* break early when found, but that's
// a uniform condition because every thread runs the same scan over
// the same root table.
fn root_lookup(key: vec3<i32>) -> u32 {
    let n = uniforms.root_entry_count;
    var hit: u32 = INVALID_CHILD;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let base = i * 4u;
        let kx = root_entries[base + 0u];
        let ky = root_entries[base + 1u];
        let kz = root_entries[base + 2u];
        if (kx == key.x && ky == key.y && kz == key.z) {
            hit = u32(root_entries[base + 3u]);
            // Don't `break` — keep the loop bounded so all threads
            // execute the same number of iterations. Cheap (≤ 100
            // root entries in v1 use cases).
        }
    }
    return hit;
}

fn lookup(p: vec3<i32>) -> f32 {
    let key = root_key_of(p);
    let internal_idx = root_lookup(key);
    if (internal_idx == INVALID_CHILD) {
        return uniforms.background;
    }

    // Bounds-check the internal_idx against the buffer extent. The
    // host should never produce an OOB index, but a corrupt upload
    // shouldn't tank the GPU: returning background is safe and
    // matches the "miss" path semantically.
    let internal_mask_base = internal_idx * INTERNAL_NODE_MASK_U32S;
    if (internal_mask_base >= arrayLength(&internal_masks)) {
        return uniforms.background;
    }

    let local = internal_local_of(p);
    let mask_word = internal_masks[internal_mask_base + (local >> 5u)];
    let bit = 1u << (local & 31u);
    if ((mask_word & bit) == 0u) {
        return uniforms.background;
    }

    let internal_child_base = internal_idx * INTERNAL_NODE_VOXELS;
    let leaf_idx = internal_children[internal_child_base + local];

    let leaf_mask_base = leaf_idx * LEAF_NODE_MASK_U32S;
    if (leaf_mask_base >= arrayLength(&leaf_masks)) {
        return uniforms.background;
    }

    let leaf_local = leaf_local_of(p);
    let leaf_mask_word = leaf_masks[leaf_mask_base + (leaf_local >> 5u)];
    let leaf_bit = 1u << (leaf_local & 31u);
    if ((leaf_mask_word & leaf_bit) == 0u) {
        return uniforms.background;
    }

    let leaf_value_base = leaf_idx * LEAF_NODE_VOXELS;
    return leaf_values[leaf_value_base + leaf_local];
}

// ── Entry point ────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn lookup_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.query_count) {
        return;
    }
    let coord = query_coords[i].xyz;
    query_results[i] = lookup(coord);
}
