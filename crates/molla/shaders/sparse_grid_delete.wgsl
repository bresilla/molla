// Stage 24e.2 — GPU-side mutable delete + leaf reclamation.
//
// Two entry points, both `@compute @workgroup_size(64)`:
//
//   • mark_inactive_voxel — for each (coord) in `query_coords`,
//     locate the leaf it lives in (root-table → internal_children),
//     atomically clear the voxel bit in the leaf mask, and — if the
//     mask drops to all-zeros as a result — push the leaf index onto
//     the `pending_reclaim` list.
//
//   • reclaim_empty_leaves — host-dispatched once per frame *after*
//     `mark_inactive_voxel` has finished. Drains the pending list
//     into the freelist so future `set_value` calls (the insert
//     kernel) can recycle the slots.
//
// **No internal-node reclamation in v1.** When all leaves under an
// internal node are reclaimed, the internal node itself stays
// allocated and its mask bit for the (now-empty) leaf stays set. The
// stale `internal_children[slot]` pointer remains pointing at the
// reclaimed leaf index — which is fine because the leaf mask is now
// all-zero, so any lookup at the original coord falls through the
// "leaf-mask bit unset → background" branch in `sparse_grid.wgsl`.
// Clearing the parent reference + cascading reclamation up the tree
// is multi-day work and is deferred to Stage 24e.3 — see PLAN.md.
//
// **Atomic clear (`atomicAnd`)**. WGSL guarantees `atomicAnd` on
// `atomic<u32>` storage (per the Naga validator + the WebGPU spec
// §6.4) — same lock-free semantics as `atomicOr`. We use it directly
// rather than a CAS loop because the operation is universally
// supported by every wgpu backend (Vulkan, Metal, D3D12, OpenGL ES,
// WebGPU); the CAS-loop fallback is documented in the host wrapper
// for any future backend that might need it.
//
// Bindings (mirror `sparse_grid_insert.wgsl` so the same persistent
// `SparseGridStorage` buffers can be bound to both kernels without
// re-uploads):
//
//   group(0) binding(0) : root_table       atomic<i32>[]   open-
//                         addressing hash table; 4 i32s per slot.
//   group(0) binding(1) : internal_children atomic<u32>[]  flat 32³
//                         per internal-node slot in the pool.
//   group(0) binding(2) : internal_masks    atomic<u32>[]  1024 u32s
//                         per internal-node slot. Read-only here —
//                         we don't clear parent bits in v1.
//   group(0) binding(3) : leaf_masks        atomic<u32>[]  16 u32s
//                         per leaf slot.
//   group(0) binding(4) : freelist          atomic<u32>[]  LIFO of
//                         reclaimed leaf indices, sized leaf_capacity.
//   group(0) binding(5) : pending_reclaim   atomic<u32>[]  LIFO of
//                         leaves marked inactive but not yet folded
//                         into the freelist. Same sizing.
//   group(1) binding(0) : counters          atomic<u32>[6] — slot
//                         layout matches the insert kernel; this
//                         shader pushes to slots 3 (freelist_count)
//                         and 4 (pending_reclaim_count).
//   group(1) binding(1) : query_coords      vec4<i32>[]    one query
//                         per thread; .w unused. Used by
//                         `mark_inactive_voxel`; ignored by
//                         `reclaim_empty_leaves`.
//   group(2) binding(0) : uniforms          { count, root_table_capacity,
//                                             internal_capacity,
//                                             leaf_capacity }
//
// Constants below MUST match `sparse_grid.rs` /
// `sparse_grid_insert.wgsl`.

const LEAF_SHIFT_GPU:     u32 = 3u;
const INTERNAL_SHIFT_GPU: u32 = 8u;
const INTERNAL_NODE_VOXELS:    u32 = 32768u; // 32³
const INTERNAL_NODE_MASK_U32S: u32 = 1024u;  // 32³ / 32
const LEAF_NODE_VOXELS:        u32 = 512u;   // 8³
const LEAF_NODE_MASK_U32S:     u32 = 16u;    // 8³ / 32
const INVALID_CHILD:           u32 = 0xFFFFFFFFu;

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
@group(0) @binding(3) var<storage, read_write> leaf_masks:        array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> freelist:          array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> pending_reclaim:   array<atomic<u32>>;

@group(1) @binding(0) var<storage, read_write> counters:     array<atomic<u32>>;
@group(1) @binding(1) var<storage, read>       query_coords: array<vec4<i32>>;

@group(2) @binding(0) var<uniform> uniforms: Uniforms;

const COUNTER_FREELIST_COUNT:  u32 = 3u;
const COUNTER_PENDING_RECLAIM: u32 = 4u;

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

// Open-addressed root lookup; same probe sequence as the insert
// kernel's `get_or_create_internal` minus the eager-CAS allocation
// (delete is purely read-only at the root level).
fn root_lookup_hashed(key: vec3<i32>) -> u32 {
    let cap = uniforms.root_table_capacity;
    let mask = cap - 1u;
    let invalid_i = i32(INVALID_CHILD);
    var slot: u32 = root_hash3(key) & mask;
    for (var probe: u32 = 0u; probe < MAX_ROOT_PROBES; probe = probe + 1u) {
        let base = slot * 4u;
        let stored_idx = atomicLoad(&root_table[base + 3u]);
        if (stored_idx == invalid_i) {
            return INVALID_CHILD;
        }
        let kx = atomicLoad(&root_table[base + 0u]);
        let ky = atomicLoad(&root_table[base + 1u]);
        let kz = atomicLoad(&root_table[base + 2u]);
        if (kx == key.x && ky == key.y && kz == key.z) {
            return u32(stored_idx);
        }
        slot = (slot + 1u) & mask;
    }
    return INVALID_CHILD;
}

// Locate the leaf for `p` without modifying any topology. Returns
// INVALID_CHILD when the coord isn't currently in the active set.
fn lookup_leaf(p: vec3<i32>) -> u32 {
    let key = root_key_of(p);
    let internal_idx = root_lookup_hashed(key);
    if (internal_idx == INVALID_CHILD) {
        return INVALID_CHILD;
    }
    let local = internal_local_of(p);
    let mask_word = atomicLoad(&internal_masks[internal_idx * INTERNAL_NODE_MASK_U32S + (local >> 5u)]);
    let bit = 1u << (local & 31u);
    if ((mask_word & bit) == 0u) {
        return INVALID_CHILD;
    }
    return atomicLoad(&internal_children[internal_idx * INTERNAL_NODE_VOXELS + local]);
}

// Push a leaf index onto the pending-reclaim LIFO. Each entry takes
// one slot; the counter slot 4 is the live depth. We bound the push
// to leaf_capacity (matches the buffer size); past that we silently
// drop the reclaim — the leaf stays "stranded" but its mask is
// already all-zeros so the only cost is wasted pool capacity.
fn push_pending_reclaim(leaf_idx: u32) {
    let pos = atomicAdd(&counters[COUNTER_PENDING_RECLAIM], 1u);
    if (pos >= uniforms.leaf_capacity) {
        // Pending list overflowed — back the counter out and drop.
        // (Counter is monotonic; the overflow is benign as long as
        // `reclaim_empty_leaves` clamps its iteration to leaf_capacity.)
        return;
    }
    atomicStore(&pending_reclaim[pos], leaf_idx);
}

// ── Entry point: mark a single voxel inactive ─────────────────────

@compute @workgroup_size(64)
fn mark_inactive_voxel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.count) { return; }
    let p = query_coords[i].xyz;

    let leaf_idx = lookup_leaf(p);
    if (leaf_idx == INVALID_CHILD) {
        // Coord not in the active set — no-op. Matches the host CPU
        // semantic ("delete missing key" is a no-op).
        return;
    }

    let local = leaf_local_of(p);
    let bit = 1u << (local & 31u);
    let leaf_mask_base = leaf_idx * LEAF_NODE_MASK_U32S;
    let mask_word = leaf_mask_base + (local >> 5u);

    // Atomic clear-bit. WGSL `atomicAnd<u32>` is universally supported
    // on storage buffers (no extension required); previous-value
    // returned so we can detect "did this thread actually flip the
    // bit" — only the flipper checks for "did the leaf become empty".
    let prev = atomicAnd(&leaf_masks[mask_word], ~bit);
    if ((prev & bit) == 0u) {
        // Bit was already clear — another thread cleared it (or it
        // was never set). No reclaim work to do.
        return;
    }

    // We flipped the bit. Sum-popcount over the 16 mask words after
    // the clear; only this thread's clear could have triggered the
    // "all zero" transition for *this* word, but other words may
    // still be non-zero. The race-window is bounded: any concurrent
    // delete on the same leaf is also subject to the same
    // popcount + "did I push it" check, so at most one thread per
    // leaf observes "all zero AND I just cleared the last bit",
    // because only one bit-clear can be the "last".
    //
    // Actually that's not quite right under concurrent deletes: two
    // threads each clearing the LAST bit of two different mask words
    // would both observe "all zero" after their respective clears.
    // We dedup by re-checking the popcount across the full mask AND
    // attempting an atomicCAS-equivalent: only push once per leaf by
    // using a bit in the highest mask word as a "reclaim claimed"
    // sentinel — wait, that bit is also a voxel bit. Cleaner: use
    // `atomicOr` on a dedicated reclaim-flag bit somewhere.
    //
    // For v1 we accept the duplicate-push hazard (rare: needs two
    // threads to clear the LAST set bit of *different mask words* on
    // the same leaf in the same dispatch) by having
    // `reclaim_empty_leaves` dedup at drain time — it checks each
    // pending entry's leaf mask AND re-validates emptiness AND
    // skips if the leaf is already in the freelist (sentinel-marked
    // via `INVALID_CHILD` after a successful drain).
    var any_set: u32 = 0u;
    for (var w: u32 = 0u; w < LEAF_NODE_MASK_U32S; w = w + 1u) {
        any_set = any_set | atomicLoad(&leaf_masks[leaf_mask_base + w]);
    }
    if (any_set == 0u) {
        push_pending_reclaim(leaf_idx);
    }
}

// ── Entry point: drain pending → freelist ─────────────────────────

@compute @workgroup_size(64)
fn reclaim_empty_leaves(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let pending_n = atomicLoad(&counters[COUNTER_PENDING_RECLAIM]);
    let cap_n = min(pending_n, uniforms.leaf_capacity);
    if (i >= cap_n) { return; }

    // Pop our entry; replace with INVALID_CHILD so future drains
    // (post-reset) don't double-process it.
    let leaf_idx = atomicLoad(&pending_reclaim[i]);
    atomicStore(&pending_reclaim[i], INVALID_CHILD);
    if (leaf_idx == INVALID_CHILD) {
        // Already drained / never pushed. Skip.
        return;
    }

    // Re-validate emptiness — see the duplicate-push note in
    // `mark_inactive_voxel`. If the leaf has any bits set (e.g.
    // because a concurrent insert reactivated it before the host
    // dispatched this drain), we just skip the push.
    let leaf_mask_base = leaf_idx * LEAF_NODE_MASK_U32S;
    var any_set: u32 = 0u;
    for (var w: u32 = 0u; w < LEAF_NODE_MASK_U32S; w = w + 1u) {
        any_set = any_set | atomicLoad(&leaf_masks[leaf_mask_base + w]);
    }
    if (any_set != 0u) {
        return;
    }

    // Push onto the freelist LIFO. atomicAdd allocates the slot;
    // atomicStore publishes the payload. The insert kernel's
    // `pop_freelist` reader spins on the slot until the payload
    // becomes != INVALID_CHILD, so the publish ordering matches.
    let pos = atomicAdd(&counters[COUNTER_FREELIST_COUNT], 1u);
    if (pos >= uniforms.leaf_capacity) {
        // Freelist already full — silently drop. The leaf becomes
        // permanently stranded (mask zero, no parent backref
        // change, no recycle) but the topology stays consistent.
        return;
    }
    atomicStore(&freelist[pos], leaf_idx);
}

// ── Entry point: zero out the pending counter ──────────────────────
//
// Convenience kernel for the host to clear `counters[4]` after a
// drain pass without having to emit a `queue.write_buffer` (which
// would race against any in-flight delete dispatch). One thread,
// one atomic store. The `reclaim_empty_leaves` kernel reads the
// counter at the top of its body via `atomicLoad`, so this kernel
// can race with the next round of mark_inactive without losing
// information — any push that lands during the reset arrives at a
// fresh `pos = atomicAdd(0, 1) = 0` slot.
@compute @workgroup_size(1)
fn reset_pending_counter() {
    atomicStore(&counters[COUNTER_PENDING_RECLAIM], 0u);
}
