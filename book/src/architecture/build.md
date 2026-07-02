# Build & test commands

The Makefile at the repository root is the canonical entry point for
building, testing, and running. GPU-touching targets go through the
`RUN_WITH` wrapper (default `nixVulkan`) so the `wgpu` Vulkan loader is
reachable from the nix devShell; outside the devShell, override it with
`RUN_WITH=` (empty) to run plain `cargo`.

## Quick reference

| Target                 | What it does                                                        |
|------------------------|---------------------------------------------------------------------|
| `make build` / `b`     | `cargo build -p molla_viewer` (the viewer pulls in every crate)     |
| `make compile` / `c`   | `cargo clean` then build                                            |
| `make run` / `r`       | `nixVulkan cargo run -p molla_viewer`                               |
| `make test` / `t`      | `cargo test -p molla_viewer` (viewer suite; skips `#[ignore]` GPU)  |
| `make test-gpu`        | One GPU test lane: `P=<crate> T=<test-file> A=<extra args>`         |
| `make test-all`        | `cargo test --workspace --all-targets`                              |
| `make check`           | `cargo check -p molla_viewer`                                       |
| `make check-all`       | `cargo check --workspace --all-targets`                             |
| `make harden`          | The full gate: whitespace check, `fmt --check`, no-default-features build+test, `clippy --all-features -D warnings`, all-feature tests |
| `make bench`           | `nixVulkan cargo bench`                                             |
| `make docs`            | Build the mdBook into `docs/`                                       |
| `make clean`           | `cargo clean`                                                       |

There is no `make fmt` target — run `cargo fmt --all` directly (or rely
on `make harden`, which checks formatting).

Useful variables: `RUN_WITH=` (drop the nixVulkan wrapper), `RUN_WITH=nixGL`
(OpenGL instead of Vulkan), `APP_PKG=<pkg>` (build/run/test a different
package), `BACKEND=x11` (display backend; `.envrc` auto-detects).

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
`wgpu::Adapter`. They all run through one generic Makefile lane —
pick the package with `P=`, the integration-test file with `T=`,
optional extra test-binary args with `A=`, and `RELEASE=1` for the
perf-gated lanes:

```bash
make test-gpu                                        # default: molla-solvers cpu_gpu_parity
make test-gpu P=molla-solvers T=xpbd_gpu_parity      # XPBD rigid parity
make test-gpu P=molla-solvers T=vbd_cloth_gpu        # VBD cloth suite
make test-gpu P=molla-solvers T=joint_muscle_gpu     # Hill muscle parity
make test-gpu P=molla-fem     T=fem_transient_gpu    # GPU FEM heat stepper
make test-gpu P=molla-fem     T=fem_wave_gpu         # GPU FEM Newmark wave
make test-gpu P=molla-geometry T=gjk_gpu A=--nocapture
make test-gpu P=molla-kernels T=tire_fem_gpu A=tire_fem_cpu_vs_gpu_rolling_contact_hub_x_parity
make test-gpu P=molla-fem T=fem_ns_gpu RELEASE=1 A='--skip gpu_ns_perf'
```

Each invocation expands to
`nixVulkan cargo test -p $(P) --features gpu --test $(T) -- --ignored $(A)`,
so any lane can also be run by hand. To discover the available lanes,
list the GPU test files:

```bash
ls crates/*/tests | grep gpu
```

## Benches

Benches use `criterion` and live with their owning functionality
crates. `make bench` runs the whole inventory; to run a specific bench:

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
make run
```

For an optimised build, run cargo directly:

```bash
nixVulkan cargo run -p molla_viewer --release
```

The viewer recognises hotkeys for the marquee scenes — see
[`viewer/overview.md`](./../viewer/overview.md).

## Common gotchas

- **wgpu init fails with "no adapter found"** — you forgot the
  `nixVulkan` wrapper (`RUN_WITH` was overridden to empty). Or you're
  in `--no-default-features` mode (which is fine, but then GPU tests
  don't apply).
- **`cargo test --features gpu` run directly hangs or panics** — GPU
  tests must go through `make test-gpu` (or a manual `nixVulkan`
  prefix) so a real adapter is reachable.
- **Long initial build** — first compile pulls ~400 Mb of
  dependencies (Bevy, wgpu, egui, glam, criterion, ...). Subsequent
  incremental builds are fast.
