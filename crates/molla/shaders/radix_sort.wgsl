// Multi-pass 8-bit-radix sort over (key=u32, value=u32) pairs.
//
// === Algorithm choice ===
//
// Why radix (not bitonic / merge):
//
//   * Radix sort over 32-bit keys is the textbook GPU choice for LBVH
//     Morton codes — Karras (2012), Lauterbach (2009), and every
//     production GPU BVH builder (Embree, OptiX, Warp / cuBQL) all use
//     it. O(n * k) time where k = 4 passes for a 32-bit key, dominated
//     by one global histogram + one block-bin transpose per pass.
//
//   * Bitonic sort is O(n log² n) — at n = 100 000 that's ~16x more
//     work than radix's O(n * 4). It also can't be stable.
//
//   * GPU merge sort hits memory-bandwidth limits earlier than radix
//     because each merge pass touches all elements, and the implementation
//     surface is much larger (per-block sort + multi-level merge).
//
// We pay the radix-sort cost (4 passes, 3 dispatches per pass) up
// front and the resulting kernel is ~150 LOC, stable, and trivially
// scalable to 8M elements before anyone notices.
//
// === Per-pass pipeline ===
//
// Each pass extracts an 8-bit digit `(key >> shift) & 0xFF`, where
// `shift` is one of {0, 8, 16, 24}. Three dispatches per pass:
//
//   1. count_block_bins
//        One workgroup per BLOCK_SIZE-element block. Each thread reads
//        one input key, atomically increments a local 256-bin histogram
//        in workgroup memory, then writes the 256 local bin counts to
//        global memory at `block_bins[bin * num_blocks + block]`. The
//        bin-major layout means an exclusive prefix-sum across the
//        entire `block_bins` array places all elements with the same
//        digit into a contiguous output segment, with each block's
//        slot correctly offset.
//
//   2. scan_global_bins
//        Single-workgroup serial-tile exclusive scan over the entire
//        `block_bins` array (256 * num_blocks entries). Tiny on
//        wallclock — for 100k inputs that's ~100k adds done serially
//        by 256 threads collaboratively. Output overwrites the input
//        buffer with exclusive-scan values; the running total persists
//        between tiles via workgroup memory.
//
//   3. scatter
//        One workgroup per BLOCK_SIZE-element block. Each thread reads
//        its (key, value) pair, computes its digit, then atomically
//        increments a per-block local counter for that digit to get
//        the in-block rank, and writes the pair to
//          `out[block_bins[digit * num_blocks + block] + local_rank]`.
//        Stable: in-block rank is decided by atomicAdd order which is
//        non-deterministic, but we use a serial pass over each digit
//        in a separate compaction pre-pass to enforce stability — see
//        `scatter` below.
//
// === Stability note ===
//
// True radix-sort stability requires that for two equal-digit keys
// the earlier one (lower input index) ends up at the lower output
// index. The CUB / Onesweep implementation enforces this via a
// per-block warp-cooperative scan. Here we use a simpler approach:
// scatter does a serial pre-pass per digit per block which guarantees
// in-block order is preserved (each block reads its inputs in index
// order, and increments a per-digit counter as it goes, so the i-th
// occurrence of digit d in block b ends up at offset
// `block_bins[d * num_blocks + b] + i`). Across blocks, the earlier
// block writes lower offsets because the global scan is done in
// (digit, block) order. → Stable.

const BLOCK_SIZE: u32 = 256u;
const NUM_BINS:   u32 = 256u;
// One u32-entry per (bin, block); bin-major layout.

struct Uniforms {
    n:           u32,
    shift:       u32,
    num_blocks:  u32,
    _pad:        u32,
};

@group(0) @binding(0) var<storage, read>       keys_in:    array<u32>;
@group(0) @binding(1) var<storage, read>       values_in:  array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out:   array<u32>;
@group(0) @binding(3) var<storage, read_write> values_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> block_bins: array<u32>;
@group(1) @binding(0) var<uniform>             u:          Uniforms;

// ---------------------------------------------------------------------
// Kernel 1 — per-block 256-bin histogram.
//
// Each workgroup handles BLOCK_SIZE elements. We use an atomic local
// histogram in workgroup memory (one u32 per bin), then flush to
// global `block_bins[bin * num_blocks + block]`.
// ---------------------------------------------------------------------

var<workgroup> local_bins: array<atomic<u32>, NUM_BINS>;

@compute @workgroup_size(256)
fn count_block_bins(
    @builtin(global_invocation_id)   gid:  vec3<u32>,
    @builtin(local_invocation_id)    lid:  vec3<u32>,
    @builtin(workgroup_id)           wid:  vec3<u32>,
) {
    // Zero the local histogram.
    atomicStore(&local_bins[lid.x], 0u);
    workgroupBarrier();

    let i = gid.x;
    if (i < u.n) {
        let key   = keys_in[i];
        let digit = (key >> u.shift) & 0xFFu;
        atomicAdd(&local_bins[digit], 1u);
    }
    workgroupBarrier();

    // Flush local histogram to global. Bin-major layout:
    //   block_bins[bin * num_blocks + block]
    let bin   = lid.x;
    let block = wid.x;
    let count = atomicLoad(&local_bins[bin]);
    block_bins[bin * u.num_blocks + block] = count;
}

// ---------------------------------------------------------------------
// Kernel 2 — single-workgroup exclusive scan over block_bins.
//
// Length = NUM_BINS * num_blocks. We process it in tiles of
// BLOCK_SIZE; running_sum carries between tiles.
//
// Within each tile we use a Hillis-Steele scan in workgroup memory
// (inclusive → shift right for exclusive). The tile's last inclusive
// value becomes the next tile's running offset.
// ---------------------------------------------------------------------

var<workgroup> tile_buf: array<u32, 256>;
var<workgroup> tile_buf2: array<u32, 256>;
var<workgroup> running_sum: u32;

@compute @workgroup_size(256)
fn scan_global_bins(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let total = NUM_BINS * u.num_blocks;
    if (lid.x == 0u) {
        running_sum = 0u;
    }
    workgroupBarrier();

    let num_tiles = (total + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let i = t * BLOCK_SIZE + lid.x;
        var v: u32 = 0u;
        if (i < total) {
            v = block_bins[i];
        }
        tile_buf[lid.x] = v;
        workgroupBarrier();

        // Hillis-Steele inclusive scan, ping-pong between tile_buf
        // and tile_buf2 to avoid in-place hazards.
        var src_is_a: bool = true;
        var step: u32 = 1u;
        loop {
            if (step >= BLOCK_SIZE) { break; }
            // src → dst.
            var s: u32 = 0u;
            if (src_is_a) {
                s = tile_buf[lid.x];
                if (lid.x >= step) {
                    s = s + tile_buf[lid.x - step];
                }
                workgroupBarrier();
                tile_buf2[lid.x] = s;
            } else {
                s = tile_buf2[lid.x];
                if (lid.x >= step) {
                    s = s + tile_buf2[lid.x - step];
                }
                workgroupBarrier();
                tile_buf[lid.x] = s;
            }
            src_is_a = !src_is_a;
            workgroupBarrier();
            step = step * 2u;
        }
        // Inclusive result is now in whichever buffer src_is_a points at
        // (after the last swap). Pull it into tile_buf for output.
        if (!src_is_a) {
            tile_buf[lid.x] = tile_buf2[lid.x];
            workgroupBarrier();
        }

        // Convert inclusive → exclusive: each lane writes its
        // predecessor's inclusive sum, plus the running offset.
        var excl: u32 = running_sum;
        if (lid.x > 0u) {
            excl = excl + tile_buf[lid.x - 1u];
        }
        if (i < total) {
            block_bins[i] = excl;
        }
        // Update running_sum for the next tile: take the tile's
        // total (last lane's inclusive value).
        workgroupBarrier();
        if (lid.x == BLOCK_SIZE - 1u) {
            running_sum = running_sum + tile_buf[BLOCK_SIZE - 1u];
        }
        workgroupBarrier();
    }
}

// ---------------------------------------------------------------------
// Kernel 3 — scatter.
//
// Stable per-block scatter: thread 0 of each workgroup walks the
// block's BLOCK_SIZE inputs in order, computes each digit, looks up
// the global offset `block_bins[digit * num_blocks + block]`, and
// writes the (key, value) pair. A per-digit local counter tracks how
// many of that digit have already been placed in this block.
//
// Yes, this is serial within each block — but BLOCK_SIZE is 256 and
// each step is 2-3 reads + 2 writes. For 100k inputs that's 391 blocks
// × 256 serial steps in parallel = ~100k effective work units, and
// the GPU launches all 391 blocks in parallel. Wallclock is dominated
// by memory bandwidth, not by the serial loop. A future Onesweep-style
// warp-cooperative scatter could shave the constant factor — left as
// follow-up.
// ---------------------------------------------------------------------

var<workgroup> local_offsets: array<u32, NUM_BINS>;

@compute @workgroup_size(256)
fn scatter(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    // Each thread loads its block's per-digit global offset into
    // workgroup memory (one entry per bin).
    let block = wid.x;
    let bin   = lid.x;
    local_offsets[bin] = block_bins[bin * u.num_blocks + block];
    workgroupBarrier();

    // Thread 0 does a serial scatter for the whole block. Maintains
    // a private counter per digit by mutating local_offsets[digit].
    if (lid.x == 0u) {
        let block_start = block * BLOCK_SIZE;
        let block_end   = min(block_start + BLOCK_SIZE, u.n);
        for (var i: u32 = block_start; i < block_end; i = i + 1u) {
            let key   = keys_in[i];
            let value = values_in[i];
            let digit = (key >> u.shift) & 0xFFu;
            let dst   = local_offsets[digit];
            keys_out[dst]   = key;
            values_out[dst] = value;
            local_offsets[digit] = dst + 1u;
        }
    }
}
