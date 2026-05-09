// Stage 24e.1 — GPU-side mutable insert into a sparse grid.
//
// Three entry points, all `@compute @workgroup_size(64)`:
//
//   • allocate_internal_atomic — for each (coord) in `query_coords`,
//     ensure a root-table slot exists for the (root_key) of that coord
//     and that an internal node is wired into it. Internal nodes are
//     drawn from a pre-allocated pool of size `internal_capacity`
//     (host-sized at construction); when the pool is exhausted we set
//     a status bit in `error_flags` and stop allocating (the write
//     becomes a no-op).
//
//   • allocate_leaf_atomic — for each coord, ensure the addressed
//     leaf is allocated under its parent internal node. Leaves come
//     from a pool of size `leaf_capacity`; same exhaustion semantics.
//
//   • set_value — does both allocations as needed, then writes
//     `query_values[i]` into `leaf_values` at the matching slot.
//
// **Atomic semantics** — all three kernels use the
// `atomicCompareExchangeWeak` pattern on the per-bit activity masks
// to win the right to allocate (only one thread per slot wins; the
// others observe the bit set and use the existing index). The pool
// counters use `atomicAdd` for index allocation. WGSL guarantees these
// are sequentially consistent within the dispatch.
//
// **Templated value types** — this WGSL is the **f32** instantiation.
// `vec3<f32>` and `f16` versions are built by the host via simple
// string-replacement of the `VALUE_TYPE` placeholder in the inline
// `set_value_one` block. The atomic-allocation kernels are
// type-agnostic; only `set_value` differs across types.
//
// Bindings (mirror `sparse_grid.wgsl` for read-only buffers, plus
// atomic write access on the topology buffers + counters):
//
//   group(0) binding(0) : root_table       atomic<i32>[]   open-
//                         addressing hash table; 4 i32s per slot
//                         (key.xyz, internal_idx). `internal_idx ==
//                         INVALID_CHILD` marks empty.
//   group(0) binding(1) : internal_children atomic<u32>[]  flat 32³
//                         per internal-node slot in the pool.
//   group(0) binding(2) : internal_masks    atomic<u32>[]  1024 u32s
//                         per internal-node slot.
//   group(0) binding(3) : leaf_values       VALUE_TYPE[]    512 values
//                         per leaf slot (f32 / vec3<f32> / f16).
//   group(0) binding(4) : leaf_masks        atomic<u32>[]  16 u32s
//                         per leaf slot.
//   group(1) binding(0) : counters          atomic<u32>[3] — [0] =
//                         next-internal-index, [1] = next-leaf-index,
//                         [2] = error_flags (bit 0 = internal pool
//                         exhausted, bit 1 = leaf pool exhausted, bit
//                         2 = root table full).
//   group(1) binding(1) : query_coords      vec4<i32>[]    one query
//                         per thread; .w unused.
//   group(1) binding(2) : query_values      VALUE_TYPE[]   one value
//                         per thread (set_value only; ignored for
//                         allocate_*).
//   group(2) binding(0) : uniforms          { count, root_table_capacity,
//                                             internal_capacity,
//                                             leaf_capacity }
//
// The root_table_capacity MUST be a power of two (matches the host
// `SparseGridGpu::root_table_capacity` invariant).

const LEAF_SHIFT_GPU:     u32 = 3u;
const INTERNAL_SHIFT_GPU: u32 = 8u;
const INTERNAL_NODE_VOXELS:    u32 = 32768u; // 32³
const INTERNAL_NODE_MASK_U32S: u32 = 1024u;  // 32³ / 32
const LEAF_NODE_VOXELS:        u32 = 512u;   // 8³
const LEAF_NODE_MASK_U32S:     u32 = 16u;    // 8³ / 32
const INVALID_CHILD:           u32 = 0xFFFFFFFFu;

// Cap on root-hash linear-probe attempts before we declare the table
// full. The host sizes the table for ≤ 0.7 load factor so probes
// stay short, but a runaway insert (table near saturation) needs a
// hard ceiling so the kernel terminates.
const MAX_ROOT_PROBES: u32 = 64u;

struct Uniforms {
    count:                u32,
    root_table_capacity:  u32, // power of two
    internal_capacity:    u32,
    leaf_capacity:        u32,
};

@group(0) @binding(0) var<storage, read_write> root_table:        array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> internal_children: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> internal_masks:    array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> leaf_values:       array<VALUE_TYPE>;
@group(0) @binding(4) var<storage, read_write> leaf_masks:        array<atomic<u32>>;

@group(1) @binding(0) var<storage, read_write> counters:     array<atomic<u32>>;
@group(1) @binding(1) var<storage, read>       query_coords: array<vec4<i32>>;
@group(1) @binding(2) var<storage, read>       query_values: array<VALUE_TYPE>;

@group(2) @binding(0) var<uniform> uniforms: Uniforms;

const COUNTER_NEXT_INTERNAL: u32 = 0u;
const COUNTER_NEXT_LEAF:     u32 = 1u;
const COUNTER_ERRORS:        u32 = 2u;

const ERR_INTERNAL_FULL: u32 = 1u;
const ERR_LEAF_FULL:     u32 = 2u;
const ERR_ROOT_FULL:     u32 = 4u;

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

fn root_hash3(key: vec3<i32>) -> u32 {
    var h: u32 = 0x9E3779B9u;
    var ks = array<i32, 3>(key.x, key.y, key.z);
    for (var i: i32 = 0; i < 3; i = i + 1) {
        var k: u32 = u32(ks[i]);
        k = k * 0xCC9E2D51u;
        k = (k << 15u) | (k >> 17u);
        k = k * 0x1B873593u;
        h = h ^ k;
        h = (h << 13u) | (h >> 19u);
        h = h * 5u + 0xE6546B64u;
    }
    h = h ^ (h >> 16u);
    h = h * 0x85EBCA6Bu;
    h = h ^ (h >> 13u);
    h = h * 0xC2B2AE35u;
    h = h ^ (h >> 16u);
    return h;
}

// Atomically claim a fresh slot from a counter pool. Returns
// `INVALID_CHILD` when the pool is exhausted (and sets the matching
// error bit). The atomic increment happens regardless — that's safe
// because the counter only ever grows, and any "stale" allocation
// past the pool end is benign as long as the caller checks the
// returned index.
fn allocate_pool_slot(counter: u32, capacity: u32, err_bit: u32) -> u32 {
    let raw = atomicAdd(&counters[counter], 1u);
    if (raw >= capacity) {
        atomicOr(&counters[COUNTER_ERRORS], err_bit);
        return INVALID_CHILD;
    }
    return raw;
}

// Get-or-create the internal-node index for `key`. Open-addressed
// linear probing on `root_table`; Murmur3-mixed start slot.
//
// **Concurrency design — eager allocation, single CAS install:**
// pre-allocate a tentative internal-node index up front with one
// `atomicAdd` on the pool counter. Walk the probe sequence; on the
// first INVALID slot, CAS the internal_idx field from INVALID to
// our tentative index in **one** atomic operation. If we win we
// publish the key cells; if we lose we re-read the slot and either
// match-the-key (returning the winner's index — our tentative
// becomes a leaked pool slot) or advance.
//
// The cost: one wasted pool slot per losing thread. The host pool
// sizer accounts for this by adding `expected_collisions × 2`
// headroom. The benefit: no RESERVED sentinel, no inner spin loop,
// no warp-divergence storm — every CAS-loss thread just falls
// through to the read-back path on the very next iteration.
//
// Note on key publication ordering: we write the key cells **after**
// the CAS-win on the index field. A reader observing "valid index
// but stale key" must therefore spin briefly on the key cells until
// they read a key consistent with the index slot it's checking. We
// detect "stale key" as `(0, 0, 0)` because the host pre-init writes
// zeros into every key cell — which means valid key (0, 0, 0) is
// indistinguishable from "uninitialised". To paper over this we
// reserve key (0, 0, 0) as a special-case "wait until consistent"
// marker; the host caller is documented to never insert exactly that
// root key (rare in practice — root key (0, 0, 0) maps to the first
// 256³ block of voxel space, not a typical asset placement).
fn get_or_create_internal(key: vec3<i32>) -> u32 {
    let cap = uniforms.root_table_capacity;
    let mask = cap - 1u;
    let invalid_i = i32(INVALID_CHILD);
    var slot: u32 = root_hash3(key) & mask;

    // Pre-allocate. May fail (pool full) — return INVALID then.
    let new_internal = allocate_pool_slot(
        COUNTER_NEXT_INTERNAL,
        uniforms.internal_capacity,
        ERR_INTERNAL_FULL,
    );
    if (new_internal == INVALID_CHILD) {
        return INVALID_CHILD;
    }

    for (var probe: u32 = 0u; probe < MAX_ROOT_PROBES; probe = probe + 1u) {
        let base = slot * 4u;

        // Try to claim a fresh slot in one CAS.
        let cas = atomicCompareExchangeWeak(
            &root_table[base + 3u],
            invalid_i,
            i32(new_internal),
        );
        if (cas.exchanged) {
            // Won — publish key cells. The winning index is already
            // visible to other threads; they'll spin on the key
            // cells until they observe the publish (see below).
            atomicStore(&root_table[base + 0u], key.x);
            atomicStore(&root_table[base + 1u], key.y);
            atomicStore(&root_table[base + 2u], key.z);
            return new_internal;
        }
        // CAS lost. Two cases:
        //   (a) cas.old_value is some other thread's published index
        //       for a DIFFERENT key → advance.
        //   (b) cas.old_value is some other thread's published index
        //       for OUR key → return it.
        //   (c) Spurious Weak failure → cas.old_value == invalid_i
        //       → re-loop on the same slot (don't advance).
        let observed = cas.old_value;
        if (observed == invalid_i) {
            // Spurious failure — retry same slot.
            continue;
        }
        // Read key cells; spin briefly if they're still zero
        // (publication race: winner did the CAS but hasn't
        // published the key yet).
        var kx: i32 = atomicLoad(&root_table[base + 0u]);
        var ky: i32 = atomicLoad(&root_table[base + 1u]);
        var kz: i32 = atomicLoad(&root_table[base + 2u]);
        var spin_budget: u32 = 4096u;
        loop {
            if (kx != 0 || ky != 0 || kz != 0) { break; }
            if (spin_budget == 0u) { break; }
            spin_budget = spin_budget - 1u;
            kx = atomicLoad(&root_table[base + 0u]);
            ky = atomicLoad(&root_table[base + 1u]);
            kz = atomicLoad(&root_table[base + 2u]);
        }
        if (kx == key.x && ky == key.y && kz == key.z) {
            return u32(observed);
        }
        slot = (slot + 1u) & mask;
    }
    atomicOr(&counters[COUNTER_ERRORS], ERR_ROOT_FULL);
    return INVALID_CHILD;
}

// Get-or-create the leaf index for the given coord under
// `internal_idx`. Same CAS pattern: claim the activity bit; the
// winner allocates a leaf and writes its index into
// `internal_children[base + local]`, then atomicOrs the bit. Losers
// observe the bit set and read back the index.
fn get_or_create_leaf(internal_idx: u32, p: vec3<i32>) -> u32 {
    let local = internal_local_of(p);
    let mask_base = internal_idx * INTERNAL_NODE_MASK_U32S;
    let bit = 1u << (local & 31u);
    let mask_word = mask_base + (local >> 5u);

    let prev = atomicOr(&internal_masks[mask_word], bit);
    let already_set = (prev & bit) != 0u;
    let child_slot = internal_idx * INTERNAL_NODE_VOXELS + local;

    if (already_set) {
        // Spin until the winning thread publishes the child index.
        for (var spin: u32 = 0u; spin < 256u; spin = spin + 1u) {
            let v = atomicLoad(&internal_children[child_slot]);
            if (v != INVALID_CHILD) {
                return v;
            }
        }
        return INVALID_CHILD;
    }

    // We won the race — allocate a fresh leaf and publish.
    let new_leaf = allocate_pool_slot(
        COUNTER_NEXT_LEAF,
        uniforms.leaf_capacity,
        ERR_LEAF_FULL,
    );
    atomicStore(&internal_children[child_slot], new_leaf);
    return new_leaf;
}

// ── Entry points ───────────────────────────────────────────────────

@compute @workgroup_size(64)
fn allocate_internal_atomic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.count) { return; }
    let p = query_coords[i].xyz;
    let key = root_key_of(p);
    _ = get_or_create_internal(key);
}

@compute @workgroup_size(64)
fn allocate_leaf_atomic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.count) { return; }
    let p = query_coords[i].xyz;
    let key = root_key_of(p);
    let internal_idx = get_or_create_internal(key);
    if (internal_idx == INVALID_CHILD) { return; }
    _ = get_or_create_leaf(internal_idx, p);
}

@compute @workgroup_size(64)
fn set_value(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.count) { return; }
    let p = query_coords[i].xyz;

    let key = root_key_of(p);
    let internal_idx = get_or_create_internal(key);
    if (internal_idx == INVALID_CHILD) { return; }

    let leaf_idx = get_or_create_leaf(internal_idx, p);
    if (leaf_idx == INVALID_CHILD) { return; }

    let local = leaf_local_of(p);
    let value_slot = leaf_idx * LEAF_NODE_VOXELS + local;
    leaf_values[value_slot] = query_values[i];

    let leaf_mask_base = leaf_idx * LEAF_NODE_MASK_U32S;
    let bit = 1u << (local & 31u);
    _ = atomicOr(&leaf_masks[leaf_mask_base + (local >> 5u)], bit);
}
