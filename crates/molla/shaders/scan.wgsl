// Stage 38e — device-wide u32 exclusive / inclusive scan.
//
// Two-pass scan-then-fixup pattern (Blelloch / Harris):
//
//   Pass 1 (scan_local)
//     Per-workgroup exclusive scan over WG_SIZE = 256 elements. Each
//     workgroup also writes its total sum (inclusive over the WG slice)
//     into `group_sums[wg_id]`.
//
//   Pass 2 (scan_groups)
//     Single-workgroup exclusive scan over `group_sums`. Handles
//     `total > WG_SIZE` by tiling: a running offset persists between
//     tiles. Same pattern as `radix_sort.wgsl::scan_global_bins`.
//
//   Pass 3 (scan_fixup)
//     Each workgroup adds `group_sums[wg_id]` to all its output
//     elements. If `u.inclusive == 1u`, fixup also adds `input[i]` to
//     turn the exclusive result into inclusive.
//
// Per Warp's `_src/native/scan.cu`, the scan-then-fixup form is the
// standard two-level approach for arbitrary-length input; it scales to
// at least WG_SIZE * WG_SIZE = 65 536 elements without a second
// hierarchy. For larger N we add a multi-tile pass-2 loop (here) so a
// single scan can handle ~16 M elements before pass 2 itself would
// need to be tiled hierarchically.

const WG_SIZE: u32 = 256u;

struct Uniforms {
    n:          u32,
    inclusive:  u32,
    num_groups: u32,
    _pad:       u32,
};

@group(0) @binding(0) var<storage, read>       input_buf:   array<u32>;
@group(0) @binding(1) var<storage, read_write> output_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> group_sums:  array<u32>;
@group(1) @binding(0) var<uniform>             u:           Uniforms;

// ---------------------------------------------------------------------
// Pass 1 — per-workgroup exclusive scan.
// ---------------------------------------------------------------------

var<workgroup> tile_a: array<u32, 256>;
var<workgroup> tile_b: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_local(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    let i = gid.x;
    let lane = lid.x;

    // Load input (zero-pad past n).
    var v: u32 = 0u;
    if (i < u.n) {
        v = input_buf[i];
    }
    tile_a[lane] = v;
    workgroupBarrier();

    // Hillis-Steele inclusive scan, ping-ponging between tile_a/tile_b.
    var src_is_a: bool = true;
    var step: u32 = 1u;
    loop {
        if (step >= WG_SIZE) { break; }
        if (src_is_a) {
            var s = tile_a[lane];
            if (lane >= step) {
                s = s + tile_a[lane - step];
            }
            workgroupBarrier();
            tile_b[lane] = s;
        } else {
            var s = tile_b[lane];
            if (lane >= step) {
                s = s + tile_b[lane - step];
            }
            workgroupBarrier();
            tile_a[lane] = s;
        }
        src_is_a = !src_is_a;
        workgroupBarrier();
        step = step * 2u;
    }

    // After the loop, the inclusive result lives in tile_a if src_is_a,
    // else in tile_b. Normalise into tile_a so the rest of the kernel
    // doesn't branch.
    if (!src_is_a) {
        tile_a[lane] = tile_b[lane];
    }
    workgroupBarrier();

    // Exclusive[i] = Inclusive[i] - input[i] = Inclusive[i-1] (for i > 0).
    // Lane 0 writes 0; others write the previous lane's inclusive value.
    var excl: u32 = 0u;
    if (lane > 0u) {
        excl = tile_a[lane - 1u];
    }
    if (i < u.n) {
        output_buf[i] = excl;
    }

    // Last lane writes the workgroup total (= inclusive[255]) to
    // group_sums[wg]. If the WG is partial, tile_a[lane] for lane >= n
    // is zero, so the inclusive at lane 255 is still the slice sum.
    if (lane == WG_SIZE - 1u) {
        group_sums[wid.x] = tile_a[WG_SIZE - 1u];
    }
}

// ---------------------------------------------------------------------
// Pass 2 — single-workgroup exclusive scan over `group_sums`.
//
// Length = num_groups. Tile-loop if num_groups > WG_SIZE.
// ---------------------------------------------------------------------

var<workgroup> sum_tile_a: array<u32, 256>;
var<workgroup> sum_tile_b: array<u32, 256>;
var<workgroup> running_offset: u32;

@compute @workgroup_size(256)
fn scan_groups(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n_groups = u.num_groups;
    if (lid.x == 0u) {
        running_offset = 0u;
    }
    workgroupBarrier();

    let n_tiles = (n_groups + WG_SIZE - 1u) / WG_SIZE;
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let i = t * WG_SIZE + lid.x;
        var v: u32 = 0u;
        if (i < n_groups) {
            v = group_sums[i];
        }
        sum_tile_a[lid.x] = v;
        workgroupBarrier();

        // Hillis-Steele inclusive scan.
        var src_is_a: bool = true;
        var step: u32 = 1u;
        loop {
            if (step >= WG_SIZE) { break; }
            if (src_is_a) {
                var s = sum_tile_a[lid.x];
                if (lid.x >= step) {
                    s = s + sum_tile_a[lid.x - step];
                }
                workgroupBarrier();
                sum_tile_b[lid.x] = s;
            } else {
                var s = sum_tile_b[lid.x];
                if (lid.x >= step) {
                    s = s + sum_tile_b[lid.x - step];
                }
                workgroupBarrier();
                sum_tile_a[lid.x] = s;
            }
            src_is_a = !src_is_a;
            workgroupBarrier();
            step = step * 2u;
        }
        if (!src_is_a) {
            sum_tile_a[lid.x] = sum_tile_b[lid.x];
        }
        workgroupBarrier();

        // Inclusive → exclusive, add tile-running offset.
        var excl: u32 = running_offset;
        if (lid.x > 0u) {
            excl = excl + sum_tile_a[lid.x - 1u];
        }
        if (i < n_groups) {
            group_sums[i] = excl;
        }

        // Update running_offset for the next tile.
        workgroupBarrier();
        if (lid.x == WG_SIZE - 1u) {
            running_offset = running_offset + sum_tile_a[WG_SIZE - 1u];
        }
        workgroupBarrier();
    }
}

// ---------------------------------------------------------------------
// Pass 3 — fixup. Add group_sums[wg] (and optionally input[i] for the
// inclusive variant) to output[i].
// ---------------------------------------------------------------------

@compute @workgroup_size(256)
fn scan_fixup(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    let i = gid.x;
    if (i >= u.n) { return; }
    var v = output_buf[i] + group_sums[wid.x];
    if (u.inclusive == 1u) {
        v = v + input_buf[i];
    }
    output_buf[i] = v;
}
