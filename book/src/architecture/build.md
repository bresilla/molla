# Build & test commands

The Makefile at the repository root is the canonical entry point for
every build, test, run, and bench command. Every target prepends
`nixVulkan` so the `wgpu` Vulkan loader is reachable from the nix
devShell; if you've installed the engine outside the devShell, swap
`nixVulkan cargo` for plain `cargo`.

## Quick reference

| Target               | What it does                                                  |
|----------------------|---------------------------------------------------------------|
| `make build` / `b`   | `cargo build --workspace --all-targets`                       |
| `make compile` / `c` | `cargo clean` then build                                      |
| `make run` / `r`     | Launch `molla_viewer` (set `RELEASE=1` for optimised build)   |
| `make run-profile`   | Launch with Tracy instrumentation (Bevy `trace_tracy`)        |
| `make test` / `t`    | `cargo test --workspace` (skips `#[ignore]`'d GPU tests)      |
| `make test-gpu`      | The flagship CPU/GPU parity test                              |
| `make test-*-gpu`    | ~25 dedicated GPU lanes, one per major kernel family          |
| `make bench-cpu`     | CPU semi-implicit step bench                                  |
| `make bench-gpu`     | GPU semi-implicit step bench                                  |
| `make bench-featherstone` | CPU-vs-GPU Featherstone parity bench                     |
| `make check`         | `cargo check --workspace --all-targets`                       |
| `make fmt`           | `cargo fmt --all`                                             |
| `make clean`         | `cargo clean`                                                 |

## The two cargo modes

molla has two effective build modes, gated by Cargo features.

### Default mode — `gpu` on

```bash
nixVulkan cargo build --workspace --all-targets
nixVulkan cargo test  --workspace
```

Compiles everything. The `#[ignore]`-gated GPU parity tests don't
run; the CPU paths + CPU oracle tests do. This is the everyday
lane.

### Headless / CI mode — `gpu` off

```bash
cargo build --workspace --no-default-features
cargo test  --workspace --no-default-features
```

Skips `wgpu` entirely. Every `_gpu` test compiles out (they're
`#[cfg(feature = "gpu")]`-gated). Useful when you don't have a
wgpu adapter (cloud CI runners, docker images without GPU
passthrough).

## Running GPU parity tests

GPU tests are gated behind `#[ignore]` because they need a real
`wgpu::Adapter`. The Makefile has one target per major kernel family;
all run via `nixVulkan` to give wgpu the Vulkan loader:

```bash
make test-gpu                       # CPU/GPU parity (the canonical lane)
make test-featherstone-gpu          # Featherstone ABA parity
make test-featherstone-multidof-gpu # multi-DOF Featherstone
make test-autograd-featherstone-gpu # Stage 23.1 — diff Featherstone
make test-xpbd-gpu                  # XPBD rigid parity
make test-raycast-gpu               # Stage 14 raycaster
make test-imu-gpu                   # Stage 14 IMU
make test-contact-gpu               # Stage 14 contact sensor
make test-frame-transform-gpu       # Stage 14 frame xform
make test-tiled-camera-gpu          # Stage 14 5-channel camera
make test-actuator-gpu              # Stage 18 actuator pipeline
make test-actuator-pipeline-gpu     # full pipeline
make test-equality-gpu              # XPBD equality projection
make test-equality-joint-mimic-gpu  # Stage 32 joint mimic
make test-particles-gpu             # particle XPBD
make test-hydroelastic-gpu          # Stage 22.1 hydroelastic
make test-sparse-grid-gpu           # Stage 24e NanoVDB
make test-mpm-gpu                   # Stage 24d v1 explicit MPM
make test-mpm-v3-gpu                # Stage 24d.1 semi-implicit MPM + Snow
make test-implicit-mpm-gpu           # implicit MPM GPU kernels/parity
make test-vbd-cloth-gpu             # VBD cloth GPU suite
make test-vbd-tets-gpu              # VBD tetrahedral GPU suite
make test-vbd-tets-v3-gpu           # VBD tet large-strain/plasticity GPU suite
make test-vbd-tets-v4-gpu           # VBD tet damping/contact GPU suite
make test-vbd-extended-gpu          # extended VBD mode/contact GPU suite
make test-soft-vt-gpu               # Stage 33.2 soft VT contact
make test-soft-friction-gpu         # Stage 33.2.1 soft friction + restitution
make test-fem-gpu                   # Stage 36.5 GPU FEM v1
make test-fem-elasticity-gpu        # Stage 41.1 elasticity + resident CG
make test-fem-stokes-gpu            # Stage 41 v2 Stokes parity
```

Each line is just a `cargo test --features gpu --test <name> -- --ignored`
invocation. You can run any of them by hand if you're not using the
Makefile.

## Benches

Benches use `criterion` and live with their owning functionality crates. The
Makefile exposes three convenience targets:

```bash
make bench-cpu              # CPU step bench
make bench-gpu              # GPU step bench (needs wgpu adapter)
make bench-featherstone     # CPU-vs-GPU Featherstone in one criterion group
```

To run a specific bench:

```bash
nixVulkan cargo bench -p molla-solvers --bench bench_xpbd_pile_gpu --features gpu
```

The bench inventory is in each owning crate's Cargo.toml under its `[[bench]]`
sections. For example solver/XPBD/VBD/MPM benches are in
`crates/molla-solvers`, geometry benches in `crates/molla-geometry`, kernel
benches in `crates/molla-kernels`, sensor benches in `crates/molla-sensors`,
math benches in `crates/molla-math`, and facade step benches in `crates/molla`.

Paired CPU-vs-GPU benches (suffix `_cpu_vs_gpu`) print side-by-side
reports in one criterion group so you can eyeball the wall-clock
ratio at matched problem sizes.

## Running the viewer

```bash
make run RELEASE=1
```

`RELEASE=1` is recommended — debug builds are noticeably laggy under
Bevy. The Makefile sets `DISPLAY=:1 WAYLAND_DISPLAY=` to pin winit
to X11 (Bevy's Wayland path has had stability issues in the molla
nix shell).

To pass CLI args:

```bash
make run RELEASE=1 ARGS="--some-flag"
```

The viewer also recognises hotkeys for the marquee scenes — see
[`viewer/overview.md`](./../viewer/overview.md).

## Common gotchas

- **wgpu init fails with "no adapter found"** — you forgot
  `nixVulkan`. Or you're in `--no-default-features` mode (which is
  fine, but then GPU tests don't apply).
- **Tracy frame stalls** — `--features profile` is on but no Tracy
  client is connected; the trace buffer grows unbounded. Connect
  immediately on launch, or use `--features log_diag` instead.
- **Long initial build** — first compile pulls ~400 Mb of
  dependencies (Bevy, wgpu, egui, glam, criterion, ...). Subsequent
  incremental builds are fast.
