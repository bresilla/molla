# Storage abstractions

`crates/molla/src/storage/` defines `Buffer<T>` — a homogeneous array
that mirrors itself across host and device memory. Both halves are
functional under the `gpu` feature; without it only the host half
exists, so all tests against the math + sim layer can drive the
host-only path directly.

The host is `f64`, the device is `f32`. The conversion happens
**exactly once at the buffer boundary** through the `BufferElement`
trait, which keeps every physics module from sprinkling `as f32` /
`as f64` everywhere.

## `BufferElement` — the host↔gpu bridge trait

```rust
pub trait BufferElement: Copy + 'static {
    type Gpu: bytemuck::Pod + bytemuck::Zeroable + Copy;
    fn to_gpu(&self) -> Self::Gpu;
    fn from_gpu(gpu: &Self::Gpu) -> Self;
}
```

Implementations cover every math type that ever lives in a buffer:

| Host type        | `Gpu` mirror             | Size (bytes) |
|------------------|--------------------------|--------------|
| `Real` (`f64`)   | `f32`                    | 4            |
| `Vec3` (`DVec3`) | `[f32; 4]` (vec3 padded) | 16           |
| `Mat3` (`DMat3`) | `GpuMat3`                | 48           |
| `Transform`      | `GpuTransform`           | 32           |
| `Quat` (`DQuat`) | `[f32; 4]`               | 16           |
| `SpatialVector`  | `GpuSpatialVector`       | 32           |
| `u32`            | `u32`                    | 4            |
| `i32`            | `i32`                    | 4            |

The `Gpu` mirrors live in `crates/molla/src/math/gpu.rs`. `GpuMat3`
stores three `vec4`-padded columns to match WGSL's `mat3x3<f32>`
column alignment of 16. `GpuTransform` and `GpuSpatialVector` each
hold two padded `vec3`s. The `gpu_struct_sizes_match_wgsl_expectations`
test pins those sizes:

```rust
assert_eq!(core::mem::size_of::<GpuTransform>(), 32);
assert_eq!(core::mem::size_of::<GpuSpatialVector>(), 32);
assert_eq!(core::mem::size_of::<GpuMat3>(), 48);
```

## `BufferUsage`

```rust
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BufferUsage {
    Storage,            // read-only on device (e.g. Model data)
    StorageReadWrite,   // read-write on device (e.g. State arrays)
    Uniform,            // small constants
}
```

Picked at allocation time to set the correct `wgpu::BufferUsages` bit
mask. `Storage` and `StorageReadWrite` differ only conceptually —
both translate to `STORAGE | COPY_SRC | COPY_DST`; the
read-only-ness is enforced by the binding type in the bind-group
layout, not by the buffer.

## `Buffer<T>`

```rust
pub struct Buffer<T: BufferElement> {
    host: Vec<T>,
    usage: BufferUsage,
    state: SyncState,
    #[cfg(feature = "gpu")]
    device: Option<wgpu::Buffer>,
}
```

Construction:

```rust
impl<T: BufferElement> Buffer<T> {
    pub fn from_vec(host: Vec<T>, usage: BufferUsage) -> Self;
    pub fn zeros(len: usize, usage: BufferUsage) -> Self where T: Default;
    pub fn from_slice(slice: &[T], usage: BufferUsage) -> Self;
}
```

`from_vec` is the most efficient — moves the existing `Vec<T>` in.
`zeros` is convenient when you just need allocated state. None of
these allocate a GPU buffer; that's deferred.

## Sync-state machine

```rust
enum SyncState {
    HostOnly,      // no GPU yet allocated
    InSync,        // host and device hold equal data
    HostDirty,     // host was mutated since last upload
    DeviceDirty,   // device was mutated since last download
}
```

The state machine lets the buffer track which side is authoritative
without round-tripping. `host()` errors when the state is
`DeviceDirty` — the caller has to `download()` first. `host_mut()`
similarly errors and additionally marks the new state `HostDirty`.
A kernel that writes the buffer in place is expected to call
`mark_device_dirty()` afterwards.

```rust
pub fn host(&self) -> Result<&[T]>;
pub fn host_mut(&mut self) -> Result<&mut [T]>;
pub fn mark_device_dirty(&mut self);  // gpu only
```

The pattern protects against a class of stale-read bugs that would
otherwise be invisible until the next CPU/GPU parity test failed.

## GPU half

```rust
#[cfg(feature = "gpu")] impl<T: BufferElement> Buffer<T> {
    pub fn ensure_device(&mut self, device: &wgpu::Device) -> Result<()>;
    pub fn upload(&mut self, queue: &wgpu::Queue) -> Result<()>;
    pub fn download(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()>;
    pub fn device_buffer(&self) -> Option<&wgpu::Buffer>;
}
```

- `ensure_device` lazily allocates the `wgpu::Buffer` sized for
  `host.len() * size_of::<T::Gpu>()`. No-op if already allocated.
  Empty buffers stay un-allocated (wgpu requires non-zero size).
- `upload` converts `f64 → f32` per element via `BufferElement::to_gpu`
  and writes through `Queue::write_buffer`. Marks state `InSync`.
- `download` issues a staging-buffer round-trip with `MAP_READ`,
  `pollster::block_on`s the map fence, then converts `f32 → f64` back
  into the host vec. Marks state `InSync`.

`download` blocks the calling thread. It's the simplest possible
implementation — there's no async API in molla today; persistent-buffer
paths avoid `download` entirely (see [Persistent-buffer
pattern](./persistent.md)).

## Why `f64` host / `f32` device

Long-running articulated integration accumulates drift at the `f32`
epsilon scale within ~10⁴ steps. Featherstone, RNEA, and CRBA all
need `f64` to stay stable across the kinds of multi-second time
windows that physics-for-robotics demands. `f32` on the GPU is the
WGSL ground truth — there is no `f64` in WGSL today (a small subset
of GPUs expose it through Vulkan extensions, but that's a forked
codebase).

The boundary is **exactly the `BufferElement::{to_gpu, from_gpu}`
pair**. No other code in molla converts between the two. This
constraint is what makes the CPU paths the precision-of-record for
every solver — the GPU paths converge to within `f32` tolerance after
conversion-aware comparison.

## Test coverage

The buffer module ships a small test set in
`crates/molla/src/storage/buffer.rs` that pins the round-trip
behaviour without needing a GPU. End-to-end GPU tests run against
specific kernel pipelines (`fem_gpu_parity`, `sparse_gpu_parity`,
`mpm_gpu_parity`) and exercise the upload/download path indirectly.

## Cross-references

- [Vectors, matrices, transforms](./../math/types.md) — the host
  types that show up in `BufferElement` impls.
- [GPU runtime](./runtime.md) — the `Engine` that hands out
  `Device` / `Queue` references.
- [Persistent-buffer pattern](./persistent.md) — when `Buffer<T>`
  isn't the right abstraction (resident solver state, repeated
  dispatches).
