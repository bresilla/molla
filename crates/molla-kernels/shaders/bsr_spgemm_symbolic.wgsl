// Stage 39F — GPU SpGEMM symbolic phase.
//
// Three entry points cover the on-device symbolic pipeline for
// `C = A · B`:
//
//   1. `symbolic_count`  — per A-row workgroup. Builds a per-row dense
//      bitmap of which B-columns are touched (atomicOr into a global
//      buffer `row_bitmaps[row * BITMAP_WORDS + word]`), then writes the
//      number of unique columns into `count[row]`.
//
//   2. (host) — runs `ScanKernel.exclusive_scan_u32(count)` to convert
//      `count[]` → `row_offsets[]`.
//
//   3. `symbolic_write` — per A-row workgroup. Reads back the saved
//      bitmap, enumerates set bits in ascending column order, and
//      writes them into `col_indices[row_offsets[row] + k]`. Output is
//      sorted-by-column per row, matching the CPU `BsrMatrix` invariant.
//
//   4. `numeric_atomic_add` — given C's `row_offsets` + `col_indices`,
//      walks every (a_slot, b_slot) contribution pair and CAS-adds the
//      product into `c_values[c_slot]`. `c_slot` is found by binary
//      search in `col_indices[row_offsets[i] .. row_offsets[i+1]]`.
//
// ## Why dense bitmaps instead of a hashtable?
//
// The Stage 39F spec sketches a 16-bucket workgroup-local hashtable
// with global overflow. For the problem sizes that motivate stage 39F
// (FEM Stokes Schur, 16³ tet = 5k DoFs) a typical C-row has ≫ 16
// unique cols (banded A × banded B → ~b² unique cols, b ≈ 7 → 49).
// The overflow path would dominate, the workgroup hashtable would buy
// us nothing, and we'd pay for two probes per insert.
//
// Dense bitmaps trade memory (`nrows · ⌈ncols/32⌉ · 4` bytes — 2 MB at
// 4096² and 50 MB at 10000²) for:
//   - O(1) atomicOr per insert (no probing, no overflow path).
//   - Implicit per-row dedup: setting an already-set bit is a no-op.
//   - **Sorted output for free** — enumerating set bits in ascending
//     index order produces sorted `col_indices`, matching the CPU
//     `BsrMatrix` invariant.
//   - The same bitmap is reused for the write pass — no re-walk needed.
//
// The atomicOr is on a per-row buffer slice with no contention across
// rows (single WG per row); within a row, contention is on the same
// bitmap word only if two threads happen to set bits in the same
// 32-wide window, which is rare and short.

// Module-level constant: must match `BITMAP_WORDS_PER_ROW_MAX` in the
// dispatcher. Currently sized at 1024 u32s → 32 768 cols per row max.
// Override via specialization constants if larger problems are needed.

const WG_SIZE: u32 = 256u;

// ---------------------------------------------------------------------
// Pass 1 — `symbolic_count`
// ---------------------------------------------------------------------

struct SymUniforms {
    a_nrows:        u32,
    b_nrows:        u32,
    b_ncols:        u32,
    bitmap_words:   u32, // ⌈b_ncols / 32⌉, padded to a u32 stride.
};

@group(0) @binding(0) var<storage, read>       a_row_off:    array<u32>;
@group(0) @binding(1) var<storage, read>       a_col_idx:    array<u32>;
@group(0) @binding(2) var<storage, read>       b_row_off:    array<u32>;
@group(0) @binding(3) var<storage, read>       b_col_idx:    array<u32>;
@group(0) @binding(4) var<storage, read_write> row_bitmaps:  array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> count:        array<u32>;
@group(1) @binding(0) var<uniform>             u_sym:        SymUniforms;

// Workgroup scratch: per-thread popcount accumulator, then reduced.
var<workgroup> wg_pop: array<u32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn symbolic_count(
    @builtin(workgroup_id)         wid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= u_sym.a_nrows) { return; }
    let lane = lid.x;
    let bm_words = u_sym.bitmap_words;
    let bm_base = row * bm_words;

    // ── Step 1: zero this row's bitmap slice. ────────────────────────
    var w: u32 = lane;
    loop {
        if (w >= bm_words) { break; }
        atomicStore(&row_bitmaps[bm_base + w], 0u);
        w = w + WG_SIZE;
    }
    workgroupBarrier();

    // ── Step 2: for each A[row, k], walk B[k, :] and atomicOr its
    //          columns into this row's bitmap.
    //
    // The A-row's nonzeros are flat-distributed across the workgroup;
    // each thread takes A-slots `lane, lane + WG, lane + 2*WG, ...`.
    let a_lo = a_row_off[row];
    let a_hi = a_row_off[row + 1u];
    var sa: u32 = a_lo + lane;
    loop {
        if (sa >= a_hi) { break; }
        let k = a_col_idx[sa];
        if (k < u_sym.b_nrows) {
            let b_lo = b_row_off[k];
            let b_hi = b_row_off[k + 1u];
            var sb: u32 = b_lo;
            loop {
                if (sb >= b_hi) { break; }
                let j = b_col_idx[sb];
                if (j < u_sym.b_ncols) {
                    let word = j >> 5u;
                    let bit  = j & 31u;
                    atomicOr(&row_bitmaps[bm_base + word], 1u << bit);
                }
                sb = sb + 1u;
            }
        }
        sa = sa + WG_SIZE;
    }
    workgroupBarrier();

    // ── Step 3: per-thread popcount over a stride of words, reduce. ──
    var acc: u32 = 0u;
    var ww: u32 = lane;
    loop {
        if (ww >= bm_words) { break; }
        acc = acc + countOneBits(atomicLoad(&row_bitmaps[bm_base + ww]));
        ww = ww + WG_SIZE;
    }
    wg_pop[lane] = acc;
    workgroupBarrier();

    // Tree-reduce over the workgroup.
    var s: u32 = WG_SIZE / 2u;
    loop {
        if (s == 0u) { break; }
        if (lane < s) {
            wg_pop[lane] = wg_pop[lane] + wg_pop[lane + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    if (lane == 0u) {
        count[row] = wg_pop[0];
    }
}

// ---------------------------------------------------------------------
// Pass 3 — `symbolic_write` (output sorted col indices)
// ---------------------------------------------------------------------

struct WriteUniforms {
    a_nrows:        u32,
    bitmap_words:   u32,
    _pad0:          u32,
    _pad1:          u32,
};

@group(0) @binding(0) var<storage, read>       w_row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read>       w_row_bitmaps: array<u32>;
@group(0) @binding(2) var<storage, read_write> w_col_indices: array<u32>;
@group(1) @binding(0) var<uniform>             u_write:       WriteUniforms;

// Per-thread "prefix-position-within-word" scratch. We use a serial
// scan over words on lane 0 because the per-row bitmap is small
// (typically ≤ 128 words for ncols ≤ 4096) — and we already know the
// total count is bounded by ~ncols, so this is fast enough that
// a parallel scan over words wouldn't change wall-time.
var<workgroup> wg_scan_cursor: u32;

@compute @workgroup_size(WG_SIZE)
fn symbolic_write(
    @builtin(workgroup_id)         wid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= u_write.a_nrows) { return; }
    let lane = lid.x;
    let bm_words = u_write.bitmap_words;
    let bm_base = row * bm_words;
    let out_base = w_row_offsets[row];

    if (lane == 0u) {
        wg_scan_cursor = 0u;
    }
    workgroupBarrier();

    // Walk words sequentially on lane 0 — for each word, broadcast it
    // and have the workgroup collectively enumerate set bits in order.
    // Because we want sorted output, the words are processed in
    // ascending order; within a word, bit 0 < bit 1 < ... .
    //
    // Implementation: lane 0 reads the word, then we use a Hillis-Steele
    // -style scan over `bit_set` flags to find each set bit's rank
    // within the word, and write at `out_base + cursor + rank`.

    // Use a workgroup shared array of 32 entries to hold bit indices
    // for the current word, then write them in parallel.
    // 32 ≤ WG_SIZE, so threads 0..32 each look at one bit.

    // Process one word at a time. (Simple and produces sorted output;
    // bitmap_words ≤ 128 for 4096 cols → ≤ 128 iterations.)
    for (var w: u32 = 0u; w < bm_words; w = w + 1u) {
        let word_val = w_row_bitmaps[bm_base + w];
        // For each set bit in `word_val`, compute its rank within the
        // word (i.e. count-of-set-bits below its position). Then write
        // `c_col = w * 32 + bit_pos` at `out_base + cursor + rank`.
        if (lane < 32u) {
            let bit = lane;
            let is_set = (word_val >> bit) & 1u;
            if (is_set == 1u) {
                // Rank = number of set bits in positions 0..bit.
                let mask = (1u << bit) - 1u; // bits 0..bit-1
                let rank = countOneBits(word_val & mask);
                let c_col = w * 32u + bit;
                w_col_indices[out_base + wg_scan_cursor + rank] = c_col;
            }
        }
        workgroupBarrier();
        if (lane == 0u) {
            wg_scan_cursor = wg_scan_cursor + countOneBits(word_val);
        }
        workgroupBarrier();
    }
}

// ---------------------------------------------------------------------
// Pass 4 — `numeric_atomic_add`
//
// One workgroup per A-row. Each thread takes a stride of A-row slots,
// for each A-slot walks the matching B-row, for each B-slot binary-
// searches C's row to find the output slot, then CAS-adds the product
// into c_values.
//
// f32 atomic add: WGSL exposes only `atomic<u32>` / `atomic<i32>`. We
// CAS-loop bitcasting to/from u32. Native f32 atomics would be faster;
// this is the portable path.
// ---------------------------------------------------------------------

struct NumUniforms {
    a_nrows:        u32,
    b_nrows:        u32,
    _pad0:          u32,
    _pad1:          u32,
};

@group(0) @binding(0) var<storage, read>       n_a_row_off:   array<u32>;
@group(0) @binding(1) var<storage, read>       n_a_col_idx:   array<u32>;
@group(0) @binding(2) var<storage, read>       n_a_values:    array<f32>;
@group(0) @binding(3) var<storage, read>       n_b_row_off:   array<u32>;
@group(0) @binding(4) var<storage, read>       n_b_col_idx:   array<u32>;
@group(0) @binding(5) var<storage, read>       n_b_values:    array<f32>;
@group(0) @binding(6) var<storage, read>       n_c_row_off:   array<u32>;
@group(0) @binding(7) var<storage, read>       n_c_col_idx:   array<u32>;
@group(0) @binding(8) var<storage, read_write> n_c_values:    array<atomic<u32>>;
@group(1) @binding(0) var<uniform>             u_num:         NumUniforms;

fn binary_search_col(c_lo: u32, c_hi: u32, key: u32) -> u32 {
    // Returns the index in [c_lo, c_hi) whose `n_c_col_idx[idx] == key`.
    // Caller guarantees the col is present (symbolic phase invariant).
    var lo = c_lo;
    var hi = c_hi;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) >> 1u;
        let v = n_c_col_idx[mid];
        if (v == key) {
            return mid;
        } else if (v < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    // Should be unreachable if symbolic was correct.
    return c_lo;
}

fn atomic_add_f32(idx: u32, addend: f32) {
    var old_u = atomicLoad(&n_c_values[idx]);
    loop {
        let old_f = bitcast<f32>(old_u);
        let new_f = old_f + addend;
        let new_u = bitcast<u32>(new_f);
        let res = atomicCompareExchangeWeak(&n_c_values[idx], old_u, new_u);
        if (res.exchanged) {
            break;
        }
        old_u = res.old_value;
    }
}

@compute @workgroup_size(WG_SIZE)
fn numeric_atomic_add(
    @builtin(workgroup_id)         wid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= u_num.a_nrows) { return; }
    let lane = lid.x;

    let a_lo = n_a_row_off[row];
    let a_hi = n_a_row_off[row + 1u];
    let c_lo = n_c_row_off[row];
    let c_hi = n_c_row_off[row + 1u];

    var sa: u32 = a_lo + lane;
    loop {
        if (sa >= a_hi) { break; }
        let k = n_a_col_idx[sa];
        let av = n_a_values[sa];
        if (k < u_num.b_nrows) {
            let b_lo = n_b_row_off[k];
            let b_hi = n_b_row_off[k + 1u];
            var sb: u32 = b_lo;
            loop {
                if (sb >= b_hi) { break; }
                let j = n_b_col_idx[sb];
                let bv = n_b_values[sb];
                let c_slot = binary_search_col(c_lo, c_hi, j);
                atomic_add_f32(c_slot, av * bv);
                sb = sb + 1u;
            }
        }
        sa = sa + WG_SIZE;
    }
}
