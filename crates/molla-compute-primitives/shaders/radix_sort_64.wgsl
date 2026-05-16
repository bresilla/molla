// Stage 38e — 64-bit radix sort over (key=u64, value=u32) pairs.
//
// Same structure as `radix_sort.wgsl` (per-block 256-bin histogram +
// single-WG global scan + per-block stable scatter) but with 8 passes
// of 8 bits each (instead of 4) to cover the full 64-bit key. The key
// is stored as two parallel `u32` arrays — `keys_lo`, `keys_hi` — so
// WGSL's lack of `u64` is irrelevant: each pass picks the right
// word via `shift`.
//
// Shift mapping per pass:
//   pass 0..3 → low word, shifts 0/8/16/24
//   pass 4..7 → high word, shifts 0/8/16/24
//
// The host sets `u.use_high = 1u` for passes 4..7 so the GPU picks the
// high word for digit extraction.
//
// Segmented sort: the caller pre-tags keys with segment ID in the top
// bits of `keys_hi` before invoking, so sorting in normal 64-bit order
// keeps segments contiguous. Untagging happens host-side after the
// sort. See `kernels/radix_sort.rs::segmented_sort_pairs_u64` for the
// host-side wiring.

const BLOCK_SIZE: u32 = 256u;
const NUM_BINS:   u32 = 256u;

struct Uniforms {
    n:           u32,
    shift:       u32,
    num_blocks:  u32,
    use_high:    u32,
};

@group(0) @binding(0) var<storage, read>       keys_lo_in:    array<u32>;
@group(0) @binding(1) var<storage, read>       keys_hi_in:    array<u32>;
@group(0) @binding(2) var<storage, read>       values_in:     array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_lo_out:   array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_hi_out:   array<u32>;
@group(0) @binding(5) var<storage, read_write> values_out:    array<u32>;
@group(0) @binding(6) var<storage, read_write> block_bins:    array<u32>;
@group(1) @binding(0) var<uniform>             u:             Uniforms;

fn digit_at(i: u32) -> u32 {
    var word: u32;
    if (u.use_high == 1u) {
        word = keys_hi_in[i];
    } else {
        word = keys_lo_in[i];
    }
    return (word >> u.shift) & 0xFFu;
}

// ---------------------------------------------------------------------
// Kernel 1 — per-block 256-bin histogram.
// ---------------------------------------------------------------------

var<workgroup> local_bins: array<atomic<u32>, NUM_BINS>;

@compute @workgroup_size(256)
fn count_block_bins(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    atomicStore(&local_bins[lid.x], 0u);
    workgroupBarrier();

    let i = gid.x;
    if (i < u.n) {
        let digit = digit_at(i);
        atomicAdd(&local_bins[digit], 1u);
    }
    workgroupBarrier();

    let bin   = lid.x;
    let block = wid.x;
    let count = atomicLoad(&local_bins[bin]);
    block_bins[bin * u.num_blocks + block] = count;
}

// ---------------------------------------------------------------------
// Kernel 2 — single-WG exclusive scan over block_bins (length =
// NUM_BINS * num_blocks). Identical to the 32-bit variant.
// ---------------------------------------------------------------------

var<workgroup> tile_buf:  array<u32, 256>;
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

        var src_is_a: bool = true;
        var step: u32 = 1u;
        loop {
            if (step >= BLOCK_SIZE) { break; }
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
        if (!src_is_a) {
            tile_buf[lid.x] = tile_buf2[lid.x];
            workgroupBarrier();
        }

        var excl: u32 = running_sum;
        if (lid.x > 0u) {
            excl = excl + tile_buf[lid.x - 1u];
        }
        if (i < total) {
            block_bins[i] = excl;
        }
        workgroupBarrier();
        if (lid.x == BLOCK_SIZE - 1u) {
            running_sum = running_sum + tile_buf[BLOCK_SIZE - 1u];
        }
        workgroupBarrier();
    }
}

// ---------------------------------------------------------------------
// Kernel 3 — stable scatter. Thread 0 serially walks the block,
// computing each digit and writing to the global offset for that digit
// + per-block running counter. Same pattern as the 32-bit kernel.
// ---------------------------------------------------------------------

var<workgroup> local_offsets: array<u32, NUM_BINS>;

@compute @workgroup_size(256)
fn scatter(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let block = wid.x;
    let bin   = lid.x;
    local_offsets[bin] = block_bins[bin * u.num_blocks + block];
    workgroupBarrier();

    if (lid.x == 0u) {
        let block_start = block * BLOCK_SIZE;
        let block_end   = min(block_start + BLOCK_SIZE, u.n);
        for (var i: u32 = block_start; i < block_end; i = i + 1u) {
            let klo   = keys_lo_in[i];
            let khi   = keys_hi_in[i];
            let value = values_in[i];
            let digit = digit_at(i);
            let dst   = local_offsets[digit];
            keys_lo_out[dst] = klo;
            keys_hi_out[dst] = khi;
            values_out[dst]  = value;
            local_offsets[digit] = dst + 1u;
        }
    }
}
