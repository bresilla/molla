# Kernel infrastructure

Every WGSL kernel in molla follows the same dispatcher pattern. This
page documents the shape — how pipelines are built, how bind groups
are laid out, and the float-atomic CAS-add idiom that WGSL's missing
`atomic<f32>` forces.

## Dispatcher pattern

A typical kernel module exposes one type that owns the compiled
pipelines and bind-group layouts. Build it once per device, dispatch
it many times. The canonical shape:

```rust
pub struct FooKernel {
    device: Arc<wgpu::Device>,
    queue:  Arc<wgpu::Queue>,

    pipeline: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl FooKernel {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::foo"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/foo.wgsl").into()),
        });
        let layout_io = device.create_bind_group_layout(/* … */);
        let layout_uniforms = device.create_bind_group_layout(/* … */);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::foo layout"),
            bind_group_layouts: &[&layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::foo pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("foo_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { device, queue, pipeline, layout_io, layout_uniforms }
    }

    pub fn dispatch(&self, /* args */) -> Result<()> {
        let bg_io = self.device.create_bind_group(/* … */);
        let bg_u  = self.device.create_bind_group(/* … */);
        let mut encoder = self.device.create_command_encoder(/* … */);
        {
            let mut pass = encoder.begin_compute_pass(/* … */);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, groups_z);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}
```

`ScanKernel`, `RadixSortKernel{,64}`, `BsrAlgebraKernel`,
`Dense3Kernel`, `TileMatmulKernel`, `TileConvolutionKernel`, all the
`fem_*` and `mpm_*` dispatchers — they all wear this skin.

## Bind-group conventions

Every kernel splits resources into two groups:

- **Group 0 — IO.** Storage buffers (read-only inputs, read-write
  outputs). Read-only inputs are bound at slot indices `0..k`,
  read-write outputs at slots `k..n`. Convention varies per kernel
  but is consistent within a kernel.
- **Group 1 — Uniforms.** A single struct of compile-time-known
  layout, padded to 16-byte alignment, holding shape parameters
  (dimensions, scalar coefficients, step counters). The uniform
  struct is `#[repr(C)]` and derives `bytemuck::Pod + Zeroable`.

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n: u32,
    inclusive: u32,
    num_groups: u32,
    _pad: u32,        // 16-byte align
}
```

Padding fields are explicit. WGSL's uniform layout rules require
`vec3`-aligned-to-16; the host-side `#[repr(C)]` struct must match
or `bytemuck::bytes_of` will write the wrong layout into the
uniform buffer.

## Read-only vs read-write bind entries

The helper functions `ro_storage(binding)` and `rw_storage(binding)`
appear across every kernel module and abbreviate the layout entries:

```rust
fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
```

`rw_storage` is identical except `read_only: false`. The WGSL side
matches with `var<storage, read>` or `var<storage, read_write>`.

## Workgroup size — the 256 convention

Most molla kernels use `@workgroup_size(256)`. The pick matches Warp's
default and is the sweet spot for AMD/NVIDIA/Intel desktop adapters —
big enough for warp-level reductions to amortise, small enough to
keep occupancy reasonable on small inputs.

```wgsl
const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn kernel_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n) { return; }   // bounds check; let the tail bail out
    // …
}
```

The trailing bounds check is **mandatory** because dispatch grids
are sized to the nearest workgroup multiple. Skipping the check
produces out-of-bounds reads on the last workgroup.

## Float atomics — CAS-add idiom

WGSL has no `atomic<f32>`. Float reductions that need to commute
across thread groups (FEM assembly, MPM P2G, SpGEMM numeric) bit-cast
through `atomic<u32>` and `atomicCompareExchangeWeak` until the swap
succeeds:

```wgsl
fn atomic_add_csr(slot: u32, value: f32) {
    if (value == 0.0) { return; }
    var old_bits = atomicLoad(&csr_values[slot]);
    loop {
        let old_f = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_f + value);
        let r = atomicCompareExchangeWeak(&csr_values[slot], old_bits, new_bits);
        if (r.exchanged) {
            return;
        }
        old_bits = r.old_value;
    }
}
```

The host-side buffer is declared `array<atomic<u32>>` in the WGSL
binding; the `bytemuck`/`f32` slice on the host is the same memory
viewed without the atomic wrapper. The pattern shows up in:

- `shaders/fem_assembly.wgsl` (FEM CSR scatter)
- `shaders/fem_elasticity_assemble.wgsl`
- `shaders/fem_ns_advect.wgsl` (advection scatter)
- `shaders/fem_stokes_assemble.wgsl`
- `shaders/bsr_spgemm_symbolic.wgsl` (SpGEMM numeric)
- `shaders/apic_locate_tet.wgsl` (APIC P2G locate-and-add)
- `shaders/splat_raytrace_sh_backward.wgsl` (Gaussian splat adjoint)

The `if (value == 0.0) return;` early-exit matters: contributors with
zero-valued blocks (off-pattern slots) shouldn't busy-wait on the CAS
loop.

## Workgroup-scoped reductions

For per-workgroup reductions (scan, sort, prefix sum) the standard
pattern is a `var<workgroup>` shared array, an explicit
`workgroupBarrier()` between read and write phases, and the
Hillis-Steele or Blelloch ping-pong:

```wgsl
var<workgroup> tile_a: array<u32, 256>;
var<workgroup> tile_b: array<u32, 256>;

tile_a[lane] = v;
workgroupBarrier();

var step: u32 = 1u;
loop {
    if (step >= WG_SIZE) { break; }
    // … read tile_a, write tile_b, swap …
    workgroupBarrier();
    step = step * 2u;
}
```

See `shaders/scan.wgsl` for a complete example.

## Why `Arc<wgpu::Device>` everywhere

Kernel dispatchers hold `Arc<wgpu::Device>` and `Arc<wgpu::Queue>`
because the wgpu types are `Clone` (cheap reference counting under the
hood) but Rust's borrow checker doesn't propagate that across struct
fields. `Arc` makes the kernels self-contained and lets one `Engine`
hand its device out to a dozen kernel dispatchers without lifetime
juggling. The cost is one atomic increment at construction time —
negligible compared to the pipeline compile.

## Cross-references

- [GPU runtime](./runtime.md) — the `Engine` that produces
  `Arc<wgpu::Device>`.
- [Storage abstractions](./storage.md) — the buffer type that
  bind groups consume.
- [Persistent-buffer pattern](./persistent.md) — how to avoid
  per-dispatch bind-group allocation.
