# SpGEMM design

Sparse-matrix-times-sparse-matrix is the most structurally awkward
kernel in the BSR algebra suite. The output's pattern is itself the
result of a computation, so a single-pass scheme can't pre-allocate
the right layout. molla ships two paths and picks at dispatch time;
this page explains both.

The entry point is `BsrAlgebraKernel::spgemm(a, b)` in
`crates/molla/src/math/sparse_gpu.rs`. The cutover threshold:

```rust
/// Below this matrix dimension (in scalar B = 1 units) `spgemm`
/// falls back to the host-symbolic v1 path.
pub const SPGEMM_GPU_SYMBOLIC_CUTOVER: u32 = 64;
```

For `max(A.nrows, B.ncols) < 64` the host-symbolic path wins on
kernel-launch overhead; for larger problems the full-GPU path takes
over. Empirically the cross-over sits between 32 and 64 on every GPU
the project has tested.

## Path 1 — host-symbolic + GPU-numeric (Stage 38a)

The Stage 38a baseline. Host reads back both patterns, walks them in
the obvious nested-loop fashion to build `C`'s `row_offsets` and
`col_idx`, plus a per-output-slot **contribution list**
(`(slot_a, slot_b)` pairs that sum into the slot). GPU then runs a
single numeric kernel that, for each output slot, iterates the
contribution list and accumulates `a_value · b_value` into the slot.

```rust
pub fn spgemm_host_symbolic(&self, a: &BsrGpu, b: &BsrGpu) -> Result<BsrGpu>;
```

Pros:

- Single numeric kernel, no atomics, easy to reason about.
- Trivially correct — the host walk is the same code you'd write for
  a CPU SpGEMM.

Cons:

- Two pattern readbacks per call (each ~20-100 µs of submit-and-wait
  latency).
- O(C-nnz) host memory for the contribution-pair flat array.
- Doesn't scale past ~64 since the host walk is single-threaded.

## Path 2 — full-GPU SpGEMM with bitmap symbolic (Stage 39F)

The production path for everything ≥ 64. Four dispatches, all
on-device:

1. **Symbolic count.** One workgroup per A-row. Walks `A[row, :]`,
   for each nonzero `(row, k)` walks `B[k, :]`, and `atomicOr`s the
   B-col indices into a **per-row dense bitmap** of width
   `ceil(B.ncols / 32)` `u32`s. After the row is processed, the
   workgroup popcounts the bitmap to produce `count[row]` — the
   number of unique columns that will appear in `C[row, :]`.

2. **Scan.** Exclusive scan of `count[]` produces `c_row_offsets[]`.
   Uses [`ScanKernel`](./scan_sort.md) — the same Stage 38e scan
   dispatcher that's reused across the project.

3. **Symbolic write.** Walks each row's saved bitmap and emits the
   set bit positions in **ascending order** to `c_col_indices[]`.
   The "set-bits ascending" order is what gives the output the
   sorted-column-indices invariant that [`BsrMatrix`](./bsr_cpu.md)
   relies on, for free.

4. **Numeric.** One workgroup per A-row. For each `(a_slot, b_slot)`
   pair, **binary-searches** `c_col_indices[c_row_offsets[row]..]` to
   find the output slot and CAS-adds `a · b` into `c_values`. The
   CAS-add idiom is documented in [Kernel
   infrastructure](./../compute/kernels.md) — WGSL has no
   `atomic<f32>`, so float atomics are spelled with `bitcast` +
   `atomicCompareExchangeWeak`.

```rust
pub fn spgemm_gpu_symbolic(&self, a: &BsrGpu, b: &BsrGpu) -> Result<BsrGpu>;
```

The four pipelines (`symbolic_count`, `symbolic_write`,
`numeric_atomic_add`, plus the older `bsr_spgemm_numeric` for the
host-symbolic path) are all compiled and cached up front by
`BsrAlgebraKernel::new`.

## Why dense bitmaps and not a parallel hashtable?

Warp's `bsr_mm` uses a workgroup-local hashtable with global
overflow. For the **FEM/Schur problems that motivate Stage 39F** the
hashtable overflow path always fires:

- FEM mass / stiffness matrices are roughly 7-banded (each vertex
  touches ~7 neighbours through the element graph).
- A 7-banded × 7-banded SpGEMM produces ~49 unique columns per C-row.
- The 16-bucket workgroup hashtable in the Warp design holds at most
  a quarter of the working set, so the overflow path takes over and
  the workgroup-local cache buys nothing.

Dense bitmaps are the cleaner answer for this shape:

- **Storage.** `nrows · ceil(ncols / 32) · 4` bytes — ≈ 2 MB at
  4096². Comfortable for any FEM mesh that fits in device memory.
- **Insertion.** O(1) `atomicOr` per insert; natural dedup falls out
  of bit-set semantics.
- **Output order.** Walking set bits in ascending order produces
  sorted `col_indices` for free; no separate sort dispatch needed.

The downside — wasted bits in sparse rows — is irrelevant at typical
FEM densities. For dramatically larger `ncols` (millions) the bitmap
storage would dominate and a sparse hashtable would win; that case
doesn't exist in molla's current workload, but the dispatcher is
structured so a third path could slot in below `SPGEMM_GPU_SYMBOLIC_CUTOVER`
without churning the public API.

## Atomics for the numeric phase

Multiple `(a_slot, b_slot)` pairs can contribute to the same output
slot — even within a single A-row when both inputs have repeated
column ranges that line up. Each contribution does the standard
`atomicCompareExchangeWeak` float-add:

```wgsl
fn atomic_add_csr(slot: u32, value: f32) {
    if (value == 0.0) { return; }
    var old_bits = atomicLoad(&csr_values[slot]);
    loop {
        let old_f = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_f + value);
        let r = atomicCompareExchangeWeak(&csr_values[slot], old_bits, new_bits);
        if (r.exchanged) { return; }
        old_bits = r.old_value;
    }
}
```

(From `crates/molla-fem/shaders/fem_assembly.wgsl`; the SpGEMM kernel
uses the same pattern.)

## Cross-references

- [BsrGpu (GPU)](./bsr_gpu.md) — the dispatcher entry point.
- [Scan & 64-bit radix sort](./scan_sort.md) — the scan that turns
  `count[]` into `c_row_offsets[]`.
- [Kernel infrastructure](./../compute/kernels.md) — the float-atomic
  CAS-add idiom.
