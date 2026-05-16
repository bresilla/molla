// Stage 38a — sparse matrix-matrix product `C = A · B`, NUMERIC pass.
//
// The SpGEMM splits into two phases:
//   1. SYMBOLIC (host) — walk A and B's CSR patterns, build C's
//      `row_offsets` and `col_indices`, and for every output slot
//      `c[c_slot]` compute the list of contribution pairs
//      `(slot_a, slot_b)` such that
//         C.values[c_slot] = Σ A.values[slot_a] · B.values[slot_b]
//      Flatten the per-output-slot lists into a single
//      `contrib_pairs[contrib_offsets[c_slot] .. contrib_offsets[c_slot+1]]`
//      run, where each pair is packed as two u32s.
//   2. NUMERIC (this shader) — one thread per output slot, walks its
//      contribution list and accumulates products.
//
// `c_nnz` is the total number of output slots (length of `c.values`).
//
// Why not full GPU SpGEMM? A proper GPU symbolic pass needs a parallel
// hash + bitonic-sort + segmented reduction. v1 is "make it correct,
// keep numeric on the GPU so it scales with nnz(C)"; v2 will move the
// symbolic pass on-device once we have the radix-sort + hash building
// blocks composed.

struct Uniforms {
    c_nnz: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> a_values : array<f32>;
@group(0) @binding(1) var<storage, read> b_values : array<f32>;
@group(0) @binding(2) var<storage, read> contrib_offsets : array<u32>;
@group(0) @binding(3) var<storage, read> contrib_pairs : array<u32>; // packed (slot_a, slot_b)
@group(0) @binding(4) var<storage, read_write> c_values : array<f32>;

@group(1) @binding(0) var<uniform> u : Uniforms;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn bsr_spgemm_numeric(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= u.c_nnz) {
        return;
    }
    let lo = contrib_offsets[slot];
    let hi = contrib_offsets[slot + 1u];
    var acc: f32 = 0.0;
    // `contrib_pairs` is packed: pair `k` lives at indices `2*k` and
    // `2*k + 1`. The host writer guarantees this layout.
    var k: u32 = lo;
    loop {
        if (k >= hi) { break; }
        let slot_a = contrib_pairs[2u * k];
        let slot_b = contrib_pairs[2u * k + 1u];
        acc = acc + a_values[slot_a] * b_values[slot_b];
        k = k + 1u;
    }
    c_values[slot] = acc;
}
