// Stage 37.8 — single-workgroup stream compaction (gather).
//
// Given an input array of `f32` and a parallel `u32` mask (1 = keep,
// 0 = drop), produce the dense array `output[k] = input[i]` where the
// `i`'s are the indices at which `mask[i] == 1`. Final dense count is
// written into `count_buf[0]`.
//
// Implementation: in-shared-memory exclusive prefix sum (Hillis-Steele)
// over the mask, then scatter-write. Single workgroup with `WG`
// threads; supports `n ≤ WG · STRIDE` items where each thread handles
// `STRIDE` slots in a strided fashion. For v1 we use `WG = 256`
// `STRIDE = 4` → up to 1024 items, matching the spec.
//
// Out-of-band slots (`i >= n`) are treated as `mask = 0`.

const WG: u32 = 256u;
const STRIDE: u32 = 4u;
const N_MAX: u32 = 1024u;

struct Uniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read> mask_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> count_buf: array<u32>;
@group(1) @binding(0) var<uniform> u: Uniforms;

var<workgroup> scratch_a: array<u32, 1024>;
var<workgroup> scratch_b: array<u32, 1024>;

@compute @workgroup_size(256, 1, 1)
fn tile_stream_compaction(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = u.n;
    let tid = lid.x;

    // Step 1: load mask values into shared memory (`scratch_a`),
    // padding out-of-bounds with 0.
    for (var s: u32 = 0u; s < STRIDE; s = s + 1u) {
        let idx = tid * STRIDE + s;
        if (idx < N_MAX) {
            if (idx < n) {
                scratch_a[idx] = mask_buf[idx];
            } else {
                scratch_a[idx] = 0u;
            }
        }
    }
    workgroupBarrier();

    // Step 2: exclusive Hillis-Steele scan. Operates on `scratch_a`
    // ping-ponging into `scratch_b`. For simplicity we run an inclusive
    // scan first then convert to exclusive at scatter time.
    var step: u32 = 1u;
    var src_in_a: bool = true;
    while (step < N_MAX) {
        if (src_in_a) {
            for (var s: u32 = 0u; s < STRIDE; s = s + 1u) {
                let idx = tid * STRIDE + s;
                if (idx < N_MAX) {
                    if (idx >= step) {
                        scratch_b[idx] = scratch_a[idx] + scratch_a[idx - step];
                    } else {
                        scratch_b[idx] = scratch_a[idx];
                    }
                }
            }
        } else {
            for (var s: u32 = 0u; s < STRIDE; s = s + 1u) {
                let idx = tid * STRIDE + s;
                if (idx < N_MAX) {
                    if (idx >= step) {
                        scratch_a[idx] = scratch_b[idx] + scratch_b[idx - step];
                    } else {
                        scratch_a[idx] = scratch_b[idx];
                    }
                }
            }
        }
        src_in_a = !src_in_a;
        step = step * 2u;
        workgroupBarrier();
    }

    // After the loop, `src_in_a` is true if the *next* write would have
    // gone to `scratch_b`, meaning the most recent values live in
    // `scratch_a`. Inclusive scan results = scratch_a if src_in_a else
    // scratch_b.

    // Step 3: scatter. For each i with mask[i]==1, write
    // output[scan_excl[i]] = input[i]. Exclusive index = inclusive_scan
    // - mask[i] (since the inclusive scan counts mask[i] itself).
    for (var s: u32 = 0u; s < STRIDE; s = s + 1u) {
        let i = tid * STRIDE + s;
        if (i < n) {
            let mi = mask_buf[i];
            if (mi != 0u) {
                let inc = select(scratch_b[i], scratch_a[i], src_in_a);
                let dst = inc - 1u; // inclusive → exclusive minus 1
                output_buf[dst] = input_buf[i];
            }
        }
    }

    // Step 4: thread 0 writes the total count = inclusive_scan[n-1].
    if (tid == 0u) {
        if (n == 0u) {
            count_buf[0] = 0u;
        } else {
            let last = n - 1u;
            let inc = select(scratch_b[last], scratch_a[last], src_in_a);
            count_buf[0] = inc;
        }
    }
}
