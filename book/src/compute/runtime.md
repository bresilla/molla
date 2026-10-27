# GPU runtime

`crates/molla/src/compute/engine.rs` owns the `wgpu` runtime. One
`Engine` per process holds the `Device` + `Queue` references that
every kernel dispatcher borrows.

## `Backend` enum

```rust
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Backend {
    /// Pure CPU — no `wgpu` resources allocated. Useful for tests
    /// and for CPU-fallback paths every solver must provide.
    Cpu,
    /// Pick whatever GPU backend wgpu finds best on this host.
    AutoGpu,
    /// Force a specific wgpu backend (Vulkan, Metal, Dx12, …).
    Gpu(wgpu::Backends),
}
```

`Cpu` is not a fake adapter — there are no `wgpu` resources at all.
Every solver and sensor that supports a CPU path checks
`Engine::device()` and falls back to the host implementation when
`None`. This is what makes the test suite runnable on CI machines
without GPUs.

## `Engine`

```rust
pub struct Engine { /* … */ }

impl Engine {
    pub fn new_cpu() -> Self;
    pub fn new_gpu() -> Result<Self>;
    pub fn new(backend: Backend) -> Result<Self>;
    pub fn from_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        adapter_info: Option<AdapterInfo>,
    ) -> Self;

    pub fn backend(&self) -> Backend;
    pub fn adapter_info(&self) -> Option<&AdapterInfo>;
    pub fn device(&self) -> Option<&Arc<wgpu::Device>>;
    pub fn queue(&self) -> Option<&Arc<wgpu::Queue>>;
}
```

`new_cpu` allocates nothing. `new_gpu` (= `new(Backend::AutoGpu)`)
walks the `wgpu::Instance`, picks the highest-performance adapter,
and requests a device with that adapter's native limits. `new(backend)`
is the general constructor; `Backend::Gpu(wgpu::Backends::VULKAN)`
forces a specific API.

`from_device` adopts a device and queue that another runtime owns, such
as a renderer that already holds the GPU; its backend reports
`Backend::External` and `Engine::new(Backend::External)` is an error.
The owner keeps its own device-lost callback, so this engine's
`DeviceHealth` only knows what the owner forwards through
`DeviceHealth::record`; every Molla resource created through the engine
belongs to the adopted device and must be dropped before it. Gearbox uses
this to run the GPU sensors on Bevy's render device.

`device()` / `queue()` return `None` on CPU backends. Callers that
require GPU should bail loudly:

```rust
let device = engine.device()
    .ok_or_else(|| Error::Gpu("GPU backend required".into()))?;
```

## Adapter selection — the limits decision

```rust
let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
    label: Some("molla::compute::Engine"),
    required_features: wgpu::Features::empty(),
    required_limits: adapter.limits(),
    experimental_features: wgpu::ExperimentalFeatures::default(),
    memory_hints: wgpu::MemoryHints::Performance,
    trace: wgpu::Trace::Off,
}))
.map_err(|e| Error::Gpu(format!("device request failed: {e}")))?;
```

The key choice is `required_limits: adapter.limits()` — molla
requests **whatever this adapter natively supports**, not
`wgpu::Limits::downlevel_defaults()`. The reason is documented
inline:

> `downlevel_defaults` caps storage buffers at 4 per stage, which we
> exceed in the rigid-body integrator (7 model + 5 state + 1 control
> = 13). Modern desktop GPUs typically allow 16+, mobile / web much
> less; if we ever target the browser we'll need to consolidate
> buffers.

The downside is that molla won't run on WebGPU today. The Stage 41
rigid-body kernel and the FEM elasticity stack both bind more than
the WebGPU minimum; targeting browsers means consolidating into
fewer, fatter storage buffers.

## `AdapterInfo`

```rust
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    pub vendor: u32,         // PCI id when reported
    pub name: String,        // device name
    pub backend: String,     // "Vulkan", "Metal", "Dx12", "Gl", "BrowserWebGpu"
    pub is_software: bool,   // true for software rasterisers (e.g. llvmpipe)
    pub driver: String,
}
```

Surfaces in `--version` output of `molla_viewer` and as a sanity
check at startup. Software adapters (llvmpipe, SwiftShader) are
allowed — they run but at a fraction of native speed; they exist
mainly so the GPU paths can be exercised on CI without real
hardware.

## Feature gating

The `gpu` Cargo feature gates everything in this module. With
`--no-default-features`:

- `Engine::new` becomes `Engine::new_cpu`-only.
- `wgpu` doesn't link.
- `BsrGpu`, `BsrAlgebraKernel`, `ScanKernel`, `RadixSortKernel*`,
  every `Tile*Kernel`, and all `crate::kernels::*` GPU dispatchers
  disappear.
- The CPU paths (host `BsrMatrix`, all `*_cpu` reference functions,
  the iterative solvers in `crate::math::iterative`, host
  Featherstone, etc.) all still compile.

Default features include `gpu`. CI tests both with and without the
feature to keep CPU-only configurations honest.

## Cross-references

- [Storage abstractions](./storage.md) — `Buffer<T>` is the upload /
  download primitive that bridges host vectors to the `Device`
  exposed here.
- [Kernel infrastructure](./kernels.md) — the dispatcher pattern every
  kernel module uses.
- [Persistent-buffer pattern](./persistent.md) — when to keep GPU
  buffers resident across many dispatches.
