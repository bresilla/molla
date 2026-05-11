# Feature flags

molla and `molla_viewer` together expose five Cargo features. They
are all additive (turning one on never breaks a build that had it
off), and none of them is required for a working CPU build of the
engine.

## Engine-crate features (`molla`)

### `gpu` — the wgpu backend *(default-on)*

Pulls in:

- `wgpu = "27"` with the `wgsl`, `vulkan`, `metal`, `dx12`, `std`
  features (every desktop backend; WebGPU support comes free).
- `pollster = "0.4"` for blocking on `wgpu` futures in the sync
  API.

Without `gpu`, the engine still compiles. Every solver's `new_cpu()`
path works; only `new_gpu(&engine)` constructors and the `compute::Engine`
struct itself are conditionally compiled out. The CPU paths are not
fallbacks — they are the oracle every parity test runs against.

The pinned version (`wgpu 27`) is chosen to match what Bevy 0.18
ships, so `molla` + `molla_viewer` share one `wgpu` in the linked
binary. Multiple linked `wgpu`s cause adapter-mismatch errors at
runtime.

### `mujoco` — the MuJoCo wrapper solver

Pulls in:

- `mujoco-rust = "0.0.6"` with `default-features = false` (skips
  the upstream `mj-render` OpenGL feature).

With `mujoco` on, `solvers::MuJoCoSolver` is available. The wrapper
serialises a `molla::Model` to an MJCF document via the in-crate
`MjcfWriter`, hands it to `libmujoco` for stepping, and reads body
poses back into a `molla::State`. See
[`importers/mjcf.md`](./../importers/mjcf.md) and
[`solvers/mujoco.md`](./../solvers/mujoco.md).

Off by default because `mujoco-rust` requires `libmujoco` to be
system-installed (the build script searches `~/.local/mujoco` and
the standard system paths). The molla nix devShell does **not**
carry `libmujoco` because bindgen needs a `libstdc++.so.6` that Nix
doesn't drop into the wgpu shell. The MuJoCo wrapper lane is run on
CI runners that have the `libmujoco` `.deb` installed (Ubuntu
22.04+).

With `mujoco` off, `solvers::mujoco` compiles to an empty module so
users without MuJoCo still build the rest of the project.

### `russell` — direct sparse solver *(Stage 36.7)*

Currently a reserved feature flag. The original Stage 36.7 spec
called for `russell_sparse` (SuiteSparse + LAPACKE), but those
dependencies require hard-coded include paths
(`/usr/include/suitesparse`) that aren't available in molla's nix
devShell. The flag presently ships a **pure-Rust dense LU**
implementation under `fem::DirectSolver` / `fem::solve_steady_state_direct`
so the API surface is in place; the swap-out story to a real
SuiteSparse-backed solver lives in `crate::fem::russell`.

With `russell` off, the module compiles to nothing and the public
API is unavailable. Iterative solvers (CG / PCG / BiCGStab / GMRES)
in `math::sparse` work without the feature.

## Viewer-crate features (`molla_viewer`)

### `profile` — Tracy instrumentation

Pulls `bevy/trace_tracy` through. Every Bevy schedule, system, and
render pass emits timeline events to the Tracy protocol. To use:

```bash
make run-profile RELEASE=1
# in another terminal:
tracy   # connect immediately, or the trace buffer grows unbounded
```

If no Tracy client is connected when `profile` is on, the trace
buffer grows without bound and causes frame stalls. Use `log_diag`
instead for plain text-mode perf data.

### `log_diag` — text-mode diagnostics

Prints every Bevy diagnostic to stdout once per second. No Tracy
connection, no memory growth — just FPS, frame time, entity counts,
and the molla-side `SimStats` resource. Useful on headless CI or
when Tracy isn't available.

### `mujoco` — passthrough to engine

Forwards the engine's `mujoco` feature. When on, the viewer's
`Scene::MuJoCo` runs through the real MuJoCo backend. When off, the
scene is still listed (for discoverability) but its update system
logs once and skips — the viewer stays usable.

## The matrix

| Feature      | Crate         | Default | Pulls                        | Gates                                            |
|--------------|---------------|---------|------------------------------|--------------------------------------------------|
| `gpu`        | molla         | yes     | `wgpu`, `pollster`           | `compute::Engine`, every `_gpu` kernel + solver  |
| `mujoco`     | molla         | no      | `mujoco-rust`                | `solvers::MuJoCoSolver`, `solvers::mujoco::MjcfWriter` |
| `russell`    | molla         | no      | —                            | `fem::DirectSolver`, `fem::solve_steady_state_direct` |
| `profile`    | molla_viewer  | no      | `bevy/trace_tracy`           | Tracy timeline events                            |
| `log_diag`   | molla_viewer  | no      | —                            | One-second text-mode diagnostics                 |
| `mujoco`     | molla_viewer  | no      | `molla/mujoco` passthrough   | `Scene::MuJoCo` real backend                     |

## Common combos

### Stock dev build (Linux + Vulkan)

```bash
nixVulkan cargo build --workspace --all-targets
```

`gpu` is default-on; the nix shell provides the Vulkan loader.
Everything works.

### Headless CI lane (CPU only, no wgpu adapter needed)

```bash
cargo build --workspace --no-default-features
cargo test  --workspace --no-default-features
```

Skips `wgpu`; every `_gpu` test is `#[ignore]`'d so the default
`cargo test` lane runs CPU-only tests + the oracle tests that do
not touch the adapter.

### Full perf lane (Tracy + real MuJoCo)

```bash
make run-profile RELEASE=1 ARGS=""
# or:
cargo run --release -p molla_viewer --features "profile,mujoco" -- $ARGS
```

You need `libmujoco` installed system-side for `--features mujoco`
to link.

## What's NOT a feature flag

The following are intentionally compile-time constants, not feature
flags:

- **BSR block size `B = 1`**. Every sparse matrix in molla is CSR-shaped.
  See [design_rules.md](./design_rules.md) for the rationale.
- **Host scalar = `f64`**. `math::Real = f64` is a `pub type`, not a
  feature. f32 host would require parallel implementations of every
  solver and is not in scope.
- **Device scalar = `f32`**. WGSL has no `f64`; we don't pretend it
  does.

If those need to change in the future they will be runtime knobs,
not features.
