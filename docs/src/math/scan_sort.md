# Scan & 64-bit radix sort

molla's GPU pipelines lean on two device-wide primitives:

- **`ScanKernel`** (`crates/molla/src/kernels/scan.rs`) — exclusive
  or inclusive `u32` prefix scan. Three passes, scales to ≈ 16 M
  elements.
- **`RadixSortKernel64`** (`crates/molla/src/kernels/radix_sort.rs`) —
  8 passes of 8-bit radix on `(u64 key, u32 value)` pairs. Plus
  segmented sort via segment-tag prefix.

Both are designed for reuse: build once per device, dispatch many
times. Most callers don't construct them directly — they're owned by
higher-level kernels (`BsrAlgebraKernel`, `LbvhKernel`, broad-phase
spatial-hash builder, …) and exposed only through that surface.

## `ScanKernel` — 3-pass device-wide scan

```rust
pub struct ScanKernel { /* … */ }

impl ScanKernel {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self;

    pub fn exclusive_scan_u32(
        &self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: u32,
    ) -> Result<()>;

    pub fn inclusive_scan_u32(
        &self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: u32,
    ) -> Result<()>;

    pub fn total_sum_u32(&self, input: &wgpu::Buffer, n: u32) -> Result<u64>;
}
```

Stage 38e — replaces the single-workgroup Hillis-Steele scan that
`tile_stream_compaction.wgsl` shipped, for the cases that need
`n ≫ workgroup_size` (sparse-grid insert, hash-grid buckets, contact
pair compaction, SpGEMM).

### Algorithm — scan-then-fixup

Two-level Blelloch / Harris pattern, matching Warp's `scan.cu` and
NVIDIA's 2007 GPU Gems chapter:

```
Pass 1 (scan_local)
  Per-workgroup exclusive Hillis-Steele scan over WG_SIZE = 256
  elements. Each workgroup writes its inclusive total into
  group_sums[wg_id].

Pass 2 (scan_groups)
  Single-workgroup exclusive scan over group_sums. Internally
  tiled so a single dispatch covers up to WG_SIZE² = 65 536
  groups → 16 M elements before a hierarchical fallback is
  needed.

Pass 3 (scan_fixup)
  Each workgroup adds group_sums[wg_id] to all its elements;
  inclusive variant also adds input[i] to convert exclusive →
  inclusive.
```

The single-WG pass-2 tile is what bounds the maximum input size:

```rust
const WG_SIZE: u32 = 256;
const MAX_GROUPS: u32 = WG_SIZE * WG_SIZE;   // 65 536 groups → 16 M elements
```

The 256 KiB `group_sums` scratch buffer is allocated up front at
`ScanKernel::new` so dispatch never has to grow it. The uniform
buffer is similarly preallocated and just written each call.

### Concurrency contract

Internal `group_sums` + `uniforms` buffers are shared across calls on
a single instance. Issuing a second scan before the first's GPU
submission has drained will race on those buffers — pass-1 of the
second scan may read stale `group_sums`. For pipelined usage
(multiple scans in flight) construct one `ScanKernel` per logical
stream, or `device.poll(...)` between calls. Single-threaded
"submit, poll, submit" usage is safe and what every current consumer
does.

### `total_sum_u32` — small extra readback

Convenience wrapper: runs an inclusive scan into a scratch buffer and
reads back the last `u32` as a host `u64`. Adds one mapped-readback
worth of latency, so use sparingly — the typical pattern is "scan to
get offsets, then exclusively read the last element of an
already-existing scan output" rather than calling this.

## `RadixSortKernel64` — 8-pass 64-bit sort

```rust
pub struct RadixSortKernel64 { /* … */ }

impl RadixSortKernel64 {
    pub const MAX_SEGMENTS: u32 = 1u32 << 16;  // 65 536

    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        initial_capacity: usize,
    ) -> Result<Self>;

    pub fn sort_pairs_u64(
        &mut self,
        keys_lo: &wgpu::Buffer,
        keys_hi: &wgpu::Buffer,
        values: &wgpu::Buffer,
        n: u32,
    ) -> Result<()>;

    pub fn segmented_sort_pairs_u64(
        &mut self,
        keys_lo: &wgpu::Buffer,
        keys_hi: &wgpu::Buffer,
        values: &wgpu::Buffer,
        segment_offsets: &wgpu::Buffer,
        n_segments: u32,
        n_total: u32,
    ) -> Result<()>;
}
```

8 passes of 8-bit radix. The 64-bit key is stored as two parallel
`u32` buffers (`keys_lo`, `keys_hi`); each pass dispatches three
shader entry points (`count_block_bins`, `scan_global_bins`, `scatter`)
and ping-pongs between the caller's buffers and a private scratch
pair. After 8 passes the final sorted result lands back in the
caller's buffers.

### Stability

The scatter kernel processes each block's elements in input-index
order from a single thread, so the i-th occurrence of digit `d` in
block `b` always lands at `block_bins[d, b] + i`. Across blocks the
global scan visits `(digit, block)` in row-major order so earlier
blocks always get lower output offsets. **Stability is load-bearing**
for LBVH construction: ties on Morton codes need to break
deterministically so the WGSL `delta` function in `lbvh.wgsl` sees
the same ordering the build pass expects.

### Why radix and not bitonic / merge

Discussed in the WGSL header `shaders/radix_sort.wgsl`. TL;DR:

- 4/8-pass 8-bit radix is the textbook GPU LBVH choice — Karras 2012,
  OptiX, Embree, NVIDIA Warp / cuBQL all do this.
- Bitonic sort is `O(n log² n)` and loses past `n ≈ 10k`.
- Merge sort hits memory bandwidth before radix does.

Radix wins because the work per pass is `O(n)` with very high
arithmetic intensity in the count + scatter steps, and 64-bit keys
only need 8 passes.

### Segmented sort via segment-tag prefix

Segmented sort is the awkward one — pre-existing GPU radix kernels
typically don't ship it. molla's approach is a **prefix-tag** trick:

```text
1. Host-XOR each key's keys_hi top 16 bits with (segment_id << 16).
2. Run a single 64-bit global sort. The tag dominates, so segments
   stay contiguous AND sort within each segment.
3. Mask out the tag bits.
```

The cost is that **keys must not use the top 16 bits of `keys_hi`**
when using segmented sort. With `n_segments ≤ 65 536` (the assertion
in `MAX_SEGMENTS`) this is safe by construction. The plain
`sort_pairs_u64` has no such restriction.

The tag step needs a single readback of `segment_offsets` and one
mapped read+write of `keys_hi` — both happen host-side. The actual
sort is then a single 8-pass dispatch.

### Validation guard

Before tagging, the code validates `segment_offsets`:

```rust
if offsets_host[seg + 1] < offsets_host[seg] {
    return Err(/* non-monotonic */);
}
if offsets_host[n_segments] != n_total {
    return Err(/* coverage mismatch */);
}
```

Hoisted ahead of the host slicing so malformed offsets produce a real
error instead of panicking deep inside a `&mut hi_host[start..end]`.

## Perf

Both kernels were tuned to be one-shot fast on modern desktop GPUs.
On an RTX 4070, the perf gate in `radix_sort_perf.rs` keeps a 1 M-key
64-bit sort under ~5 ms; scan does 16 M elements in ~3 ms. Numbers
shift by ±50% between vendors but the asymptotic shape is consistent
because both kernels are bandwidth-bound at large `n`.

## Cross-references

- [SpGEMM design](./spgemm.md) consumes the scan for the symbolic
  phase's `count → c_row_offsets` step.
- [BVH](./../geometry/bvh.md) uses the radix sort for Morton-code
  ordering of leaves before the LBVH build.
- [Sparse grids (NanoVDB-equivalent)](./../geometry/sparse_grid.md)
  uses scan for bucket prefix-sums during insert.
