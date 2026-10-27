# Feature flags

molla keeps feature flags small and physics-focused. Importers, external solver
wrappers, MJCF/USD/URDF pipelines, and FFI bindings are intentionally not feature
flags in this roadmap.

## Engine-crate features (`molla`)

### `gpu` — the wgpu backend *(default-on)*

Pulls in `wgpu 27` and `pollster`, enabling GPU dispatchers and GPU solver
constructors. CPU paths still compile and remain the correctness oracle.

### `russell` — direct solver shim

Enables `fem::DirectSolver` / `fem::solve_steady_state_direct`. The current
implementation is pure Rust; no SuiteSparse/LAPACKE dependency is required.

## Viewer-crate features (`molla_viewer`)

### `profile`

Forwards `bevy/trace_tracy` for Tracy instrumentation.

### `log_diag`

Prints Bevy/molla diagnostics once per second without Tracy.

## Matrix

| Feature | Crate | Default | Pulls | Gates |
|---------|-------|---------|-------|-------|
| `gpu` | molla | yes | `wgpu`, `pollster` | GPU kernels, GPU solvers/sensors |
| `russell` | molla | no | — | FEM direct solve shim |
| `profile` | molla_viewer | no | `bevy/trace_tracy` | Tracy spans |
| `log_diag` | molla_viewer | no | — | text diagnostics |

There is no MuJoCo/MJCF/URDF/USD/Gmsh feature in the physics-only plan.
