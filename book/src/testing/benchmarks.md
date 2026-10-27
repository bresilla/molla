# Benchmarks

molla ships [`criterion`](https://github.com/bheisler/criterion.rs) benches in
the crate that owns the measured functionality. Most physics benches now live
under `crates/molla-solvers/benches/`, geometry benches under
`crates/molla-geometry/benches/`, kernel benches under
`crates/molla-kernels/benches/`, sensor benches under
`crates/molla-sensors/benches/`, and facade/product step benches under
`crates/molla/benches/`. They serve two purposes:

1. **Tracking perf regressions** — criterion saves baseline times
   and reports relative changes per run.
2. **Comparing CPU vs GPU** — paired `_cpu_vs_gpu` benches run both
   backends in one criterion group so the reports print
   side-by-side.

This chapter documents what each bench family measures and how to
read the output.

## Running

The Makefile exposes three convenience targets:

```bash
make bench-cpu              # CPU step bench
make bench-gpu              # GPU step bench (needs wgpu adapter)
make bench-featherstone     # CPU-vs-GPU Featherstone in one criterion group
```

To run a specific bench:

```bash
nixVulkan cargo bench -p molla-solvers --bench bench_xpbd_pile_gpu --features gpu
```

Each bench's name maps 1:1 to a `benches/<name>.rs` file in the owning crate.
The owning crate's Cargo.toml `[[bench]]` sections set
`required-features = ["gpu"]` for GPU benches so they only attempt to build when
the feature is on.

## Bench inventory by family

### Stage A — rigid XPBD

| Bench                       | Measures                                          |
|-----------------------------|---------------------------------------------------|
| `bench_xpbd_pile`           | CPU XPBD step on a 1000-sphere pile               |
| `bench_xpbd_pile_gpu`       | Same scene, GPU                                   |
| `bench_xpbd_pile_cpu_vs_gpu`| Paired side-by-side at matched N                  |
| `bench_xpbd_stacking`       | Stable box-stack convergence                      |
| `bench_xpbd_chain`          | Distance-constraint chain                         |

The GPU XPBD pile pushes to **1 M bodies** in the perf gate
(`bench_xpbd_pile_gpu.rs::test_one_million`). The bench fixture
verifies it.

### Stage B — Featherstone (articulated)

| Bench                          | Measures                                       |
|--------------------------------|------------------------------------------------|
| `bench_featherstone_chain`     | N-link serial chain ABA step                   |
| `bench_featherstone_tree`      | Branching tree ABA step                        |
| `featherstone_cpu_vs_gpu`      | Paired CPU/GPU at matched articulation depth   |

### Stage C — cloth (XPBD + VBD)

| Bench                        | Measures                                     |
|------------------------------|----------------------------------------------|
| `bench_xpbd_cloth`           | XPBD cloth at multiple resolutions           |
| `bench_vbd_cloth`            | VBD cloth CPU                                |
| `bench_vbd_cloth_gpu`        | VBD cloth GPU                                |
| `bench_vbd_cloth_cpu_vs_gpu` | Paired                                       |

### Stage D — VBD tets (soft body)

| Bench                         | Measures                                    |
|-------------------------------|---------------------------------------------|
| `bench_vbd_tets`              | CPU                                         |
| `bench_vbd_tets_gpu`          | GPU                                         |
| `bench_vbd_tets_cpu_vs_gpu`   | Paired                                      |

### Stage E — cable

| Bench               | Measures                                            |
|---------------------|-----------------------------------------------------|
| `bench_cable_drop`  | Cable polyline draping over a fixed cylinder        |

### Stage F — MPM

| Bench                  | Measures                                         |
|------------------------|--------------------------------------------------|
| `bench_mpm_explicit`   | Stage 24d v1 explicit MPM                        |
| `bench_mpm_implicit`   | Stage 39A implicit Bingham MPM                   |
| `bench_mpm_diff`       | Stage 38.1 differentiable-MPM forward + backward |

### Stage G — soft contact (Stage 33 family)

| Bench                          | Measures                                     |
|--------------------------------|----------------------------------------------|
| `bench_soft_vt_contact_cpu`    | VT contact CPU                               |
| `bench_soft_vt_contact_gpu`    | VT contact GPU                               |
| `bench_soft_self_collision`    | Stage 33.3 cloth-collar self-collision       |
| `bench_soft_ee_contact`        | Edge-edge contact                            |
| `bench_soft_friction`          | Stage 33.2.1 Coulomb friction                |

### Stage H — broadphase

| Bench                       | Measures                                        |
|-----------------------------|-------------------------------------------------|
| `bench_lbvh_build`          | GPU LBVH build                                  |
| `bench_lbvh_refit`          | GPU LBVH refit (per-frame after build)          |
| `bench_broadphase_pairs`    | Pair generation throughput                      |

### Stage I — narrow phase / contact pipeline

| Bench                        | Measures                                       |
|------------------------------|------------------------------------------------|
| `bench_narrow_phase_pairs`   | CPU narrow phase on mixed primitives           |
| `bench_mpr_vs_gjk`           | MPR vs GJK head-to-head                        |
| `bench_contact_reduction`    | Contact reduction kernel (Stage 28.1)          |

### Stage J — IK + sensors

| Bench                  | Measures                                             |
|------------------------|------------------------------------------------------|
| `bench_ik`             | GPU IK on the Franka                                 |
| `bench_sensor_raycast` | Stage 14 raycaster throughput                        |
| `bench_sensor_camera`  | Stage 14 5-channel TiledCamera                       |

### Stage K — iterative solvers

| Bench              | Measures                                                  |
|--------------------|-----------------------------------------------------------|
| `bench_iterative`  | CG / BiCGStab / GMRES / PCG on 1D Laplacian               |

### Stage L — autograd

| Bench                       | Measures                                          |
|-----------------------------|---------------------------------------------------|
| `bench_autograd_featherstone` | Featherstone reverse adjoint cost              |
| `bench_autograd_tape`       | Tape replay forward + backward                    |

### Stage M — sparse grid

| Bench                       | Measures                                          |
|-----------------------------|---------------------------------------------------|
| `bench_sparse_grid_insert`  | Mutable insert throughput                         |
| `bench_sparse_grid_query`   | Lookup latency                                    |

### Stage N — math + FFT

| Bench               | Measures                                                  |
|---------------------|-----------------------------------------------------------|
| `bench_math`        | Small-matrix decompositions, BSR matvec, etc.             |
| `bench_fft`         | GPU FFT                                                   |
| `bench_spgemm_gpu`  | Stage 39F full-GPU SpGEMM vs host-symbolic baseline        |

## Reading criterion output

A typical bench output:

```
xpbd_pile/cpu_1000      time:   [3.4231 ms 3.4567 ms 3.4923 ms]
xpbd_pile/gpu_1000      time:   [0.4123 ms 0.4256 ms 0.4401 ms]
                        change: [-2.1% -1.4% -0.7%]   (p = 0.03)
                        Performance has improved.
```

- The first row reports the CPU bench at N=1000.
- The second row reports the GPU bench at N=1000.
- `change` is relative to the saved baseline (criterion stores
  baselines under `target/criterion/`).
- `p = 0.03` is the t-test confidence — values below 0.05 are
  considered significant.

For the paired `_cpu_vs_gpu` benches, criterion groups CPU and
GPU samples on the same plot so the wall-clock ratio is one chart
read.

## Async / readback caveats

The GPU benches measure **end-to-end** wall time including `wgpu`
submit + readback. This matters because:

- Many GPU kernels are sub-millisecond on the device itself, but
  the `pollster::block_on(device.poll(MaintainBase::Wait))` round
  trip can dwarf the kernel cost.
- Persistent-buffer dispatchers (Stage 24d.2 MPM, Stage 41 v6 FEM)
  amortise the submit cost across substeps; the bench numbers
  reflect that.

If a bench number looks suspiciously slow, the most common cause is
per-substep readback. The fix is the persistent-buffer pattern
documented in [design rules](../architecture/design_rules.md).

## See also

- [GPU port status matrix](./gpu_status.md) — which solvers have
  benches, which don't.
- [Test strategy](./strategy.md) — perf gates as a test family.
- [Build & test commands](../architecture/build.md) — Makefile
  targets for benches.
