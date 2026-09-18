# Scout B — independent audit: crates/molla-geometry, crates/molla-sensors

Read-only. Every claim is anchored to `file:line`. Method: `git ls-files|xargs cat` for loc,
`grep -rn` for edges. No prior audit file was read (see §7 limits).

## 1. Per-crate purpose

**molla-geometry** (27240 loc / 74 files) — collision detection *and* mesh authoring.
- Pipeline: `narrow_phase.rs` (2399) hand-rolled sphere/box/plane + SAT box-box, falling through to
  `gjk.rs` (603) / `epa.rs` (558) / `mpr.rs` (502); `pipeline.rs` (1368) is the stateful
  `CollisionPipeline` plus a stateless `collide()`.
- Broad phase ships **three** implementations: `broad_phase.rs` (293, SAP, the default),
  `bvh.rs` (446, SAH BVH), `lbvh.rs` (1265, GPU Karras LBVH).
- Acceleration structures: `mesh_bvh.rs`/`mesh_bvh_raycast_gpu.rs`, `tet_bvh.rs` (479),
  `soft_bvh.rs`/`soft_bvh_gpu.rs`, `spatial_hash.rs` (266).
- Physics models: `hydroelastic.rs` (1355) patch contact, `contact_reduction.rs`/`contact_extras.rs`,
  `soft_ccd.rs`/`vbd_ccd_gpu.rs` (Stage-33 soft contact family).
- Mesh authoring: `decimate.rs` (QEM), `hull.rs` (QuickHull), `sdf_bake.rs`, `marching_cubes.rs`,
  `sparse_grid.rs` (1179, NanoVDB-style Root/Internal/Leaf), `gaussian_splat.rs` (1267).
- GPU twins for nearly all of the above, all `#[cfg(feature = "gpu")]` (`lib.rs:14-63`).

**molla-sensors** (7182 loc / 26 files) — GPU-first sensor orchestrators over `molla-kernels`
dispatches, plus one CPU adjoint.
- `lidar.rs` (1038), `tiled_camera.rs` (733), `pointcloud.rs` (721), `raycast.rs` (468),
  `dataset.rs` (407), `raycast_adjoint.rs` (402), `camera_post.rs` (336), `imu.rs` (221),
  `frame_transform.rs` (190), `contact.rs` (154).
- Critical: `Cargo.toml:8` sets `default = []` and `lib.rs:4-17` gates **nine of ten** modules behind
  `#[cfg(feature = "gpu")]`. Only `raycast_adjoint` (`lib.rs:19`) is unconditional, and it imports
  nothing but `molla_math` (`raycast_adjoint.rs:29`).

## 2. Real dependency edges vs Cargo claims

| edge | Cargo claim | source evidence | verdict |
|---|---|---|---|
| kernels → geometry | `molla-kernels/Cargo.toml:21,32` | `molla-kernels/src/lib.rs:203-211` re-exports Epa/Gjk/Mpr/Lbvh/MeshBvh; `hydroelastic.rs:66`; `sparse_grid.rs:20` | **REAL, load-bearing** |
| solvers → geometry | `molla-solvers/Cargo.toml:35` | `vbd_soft_contact.rs:26,901` uses `SoftBvh`, `vt_ccd` | **REAL** (`CollisionPipeline` only in benches/tests) |
| fem → geometry | `molla-fem/Cargo.toml:26` | `nonconforming.rs:94` `use molla_geometry::{TetAabb, TetBvh}` | **REAL, narrow** |
| sensors → geometry | `molla-sensors/Cargo.toml:17-20,30` | `raycast.rs:37` `MeshBvh/MeshBvhGpu/MeshBvhRaycastKernel` | **REAL, one import site** |
| molla → both | `molla/Cargo.toml:28-29,54-55` | `molla/src/geometry/mod.rs:56-113`; 30 shims e.g. `geometry/aabb.rs:3` | **REAL but pure forwarding** |
| viewer → both | no direct dep | `main.rs:53,56-59` via `molla::` facade | **REAL — the only non-test sensor consumer** |
| geometry → sim/math/core/storage | non-optional (`:31-34`) | `broad_phase.rs:14`, `epa.rs:11`, `contact_extras.rs:17`, `pipeline.rs:9` | **REAL, deep** |
| geometry → compute(-primitives) | optional, gpu only | `pipeline.rs:194`, `lbvh.rs:51` | **REAL, correctly optional** |
| sensors → kernels/compute/sim/storage/core | optional, gpu only | `contact.rs:28-33`, `raycast.rs:38`, `tiled_camera.rs:21` | **REAL but all inside gpu-gated modules** |

## 3. Dead / dormant

- **`molla-sensors::pointcloud` (721 loc)**: zero external callers. All 20 exports appear only in
  `lib.rs:40-43` and `molla/src/sensors/mod.rs:64-68`; the 17 `#[test]` fns at `pointcloud.rs:467-698`
  are the only callers. No viewer scene, bench, or integration test.
- **`FrameTransformSensor`/`FrameTarget`/`FrameReading`**: only in
  `molla-sensors/tests/sensor_frame_transform_gpu.rs:9,37,41,...` plus re-exports. Not in the viewer
  import list (`main.rs:56-59`).
- **GPU kernels with no consumer**: `SdfBakeKernel` (`lib.rs:105`), `SelfContactKernel` (`lib.rs:108`),
  `SoftCcdKernel`+`vt_ccd_batch_gpu` (`lib.rs:117`), `vbd_ccd_earliest_toi_gpu` (`lib.rs:132`), and the
  geometry-side `HydroelasticKernel` (`lib.rs:90`). Each is referenced only inside
  `crates/molla-geometry/tests/`. The CPU `bake_mesh_to_sdf` and `vt_ccd`, by contrast, ARE used in
  production (`main.rs:10679`; `vbd_soft_contact.rs:901`).
- **`SpatialHashGrid`** (`lib.rs:122`): no caller outside the two re-export lines.
- **`NarrowPhaseBackend::MprCpu`/`MprGpu`** (`pipeline.rs:34-53`): zero non-test callers; the viewer
  uses only the older GJK toggles (`main.rs:4657,4671`).
- **`bvh::Bvh`** (`lib.rs:69`): no caller outside the re-export.

## 4. Redundancy

- **`HydroelasticKernel` exists twice, same name, two crates.** geometry's (`lib.rs:90-91`, inline in
  `hydroelastic.rs:957`) is dead; kernels' (`molla-kernels/src/hydroelastic.rs:114`, re-exported
  `lib.rs:178`) is live and itself consumes geometry's CPU helpers (`:66-67`). A stack-trace hazard.
- **CPU/GPU twin families** — `gjk`/`gjk_gpu`, `epa`/`epa_gpu`, `mpr`/`mpr_gpu`,
  `marching_cubes`/`_gpu`, `sdf_bake`/`_gpu`, `self_contact`/`_gpu`, `soft_bvh`/`_gpu`,
  `soft_ccd`/`_gpu`. Deliberate parity design, but three of the GPU halves have no consumer, so the
  "mirrors the CPU path" property is now maintained entirely at test cost.
- **Three broad phases** (`broad_phase.rs:1-6` calls brute force "kept for tests / parity"; `bvh.rs:4-6`
  is self-described as "an alternative").
- `gaussian_splat.rs`'s `composite_*_cpu` family is a CPU oracle for `molla-kernels/src/splat_raytrace.rs`,
  reached only by six `molla-kernels/tests/splat_*.rs` files.

## 5. Connection health

Both crates connect widely at *compile* time and narrowly at *runtime*. molla-geometry is the real
dependency — kernels, solvers, fem and sensors all consume it in production. molla-sensors is nearly a
leaf: the viewer, reached through the `molla` facade, is its only non-test consumer, and most of its
modules have exactly one such consumer. The facade layer in `crates/molla/src/{geometry,sensors}` is
~30 files whose whole body is `pub use molla_geometry::<mod>::*`; it adds a rename and nothing else.
The weak seam inside geometry is the unused GPU mirrors plus the two-stage pipeline wiring
(`enable_full_gpu_narrow_phase`, `enable_mpr_gpu_backend`) that only tests and two viewer demos pull.

## 6. Removal candidates

| candidate | evidence | risk | decision needed |
|---|---|---|---|
| `pointcloud.rs` (721 loc) | 0 external refs | LOW tech, MEDIUM intent | product feature or spec leftover? |
| 5 unconsumed GPU kernels | `lib.rs:90,105,108,117,132`; refs only in geometry tests | MEDIUM — deleting drops the only GPU parity coverage | keep as parity lanes, or delete? |
| `bvh::Bvh` (446 loc) | no caller outside `lib.rs:69` | LOW-MED | real alternative or superseded? |
| `SpatialHashGrid` (266 loc) | 2 re-export lines only | LOW | same |
| `MprCpu`/`MprGpu` pipeline wiring | `pipeline.rs:169-192`, tests only | LOW (algorithm tested at `tests/mpr.rs`) | wire into a scene, or bench-only? |
| `molla-sensors/Cargo.toml:45` dev-dep | already provided via gpu at `:30` | LOW | none — safe delete |
| `crates/molla/src/{geometry,sensors}` facade | 30 pure `pub use` files | MEDIUM — viewer rewrite | deliberate boundary or accidental? |

## 7. Verdict

molla-geometry is the genuine article: a large, mostly-coherent collision and geometry library whose
CPU core (narrow phase, pipeline, broad phase, GJK/EPA/MPR, hydroelastic, LBVH, sparse grid) is properly
consumed by molla-kernels, molla-solvers and molla-fem at runtime. What it should *be* is smaller at the
edges: the GPU "mirror" files whose only caller is a test, the second `HydroelasticKernel`, the third
broad phase, and the two file-level utilities (`bvh.rs`, `spatial_hash.rs`) with no consumer at all.
molla-sensors is a different animal — it is a viewer-scene support library wearing a sensor-library's
name. Nine of its ten modules are unreachable without the `gpu` feature, its only non-test consumer is
`bin/molla_viewer`, and one entire module (`pointcloud`, 721 loc) has no consumer anywhere. The region's
real problem is not that things are broken but that parity experiments, authoring utilities and
production kernels are all shipped as equally-public peers, so "what is this for" is unanswerable from
the API surface alone. The right shape is one small always-on core (narrow phase + pipeline + support +
the BVH family) with everything else explicitly labelled as either a GPU backend, a test oracle, or an
authoring tool — and the unused GPU mirrors deleted or demoted to `#[cfg(test)]`.

## 8. Build probe (verbatim)

```
$ cargo metadata --no-deps --format-version 1
exit 0
108764 bytes JSON on stdout, empty stderr.

$ timeout 600 cargo check -p molla-geometry --offline
exit 101
warning: spurious network error (3 tries remaining): failed to make directory '/pkg/cargo/git/checkouts/bevy_frost-7cd5eeeaf3285665/001fc84': Read-only file system; class=Os (2)
warning: spurious network error (2 tries remaining): failed to make directory '/pkg/cargo/git/checkouts/bevy_frost-7cd5eeeaf3285665/001fc84': Read-only file system; class=Os (2)
warning: spurious network error (1 try remaining): failed to make directory '/pkg/cargo/git/checkouts/bevy_frost-7cd5eeeaf3285665/001fc84': Read-only file system; class=Os (2)
error: failed to get `bevy_frost` as a dependency of package `molla_viewer v0.0.1 (/home/bresilla/data/code/OUSD/molla/bin/molla_viewer)`

Caused by:
  failed to load source for dependency `bevy_frost`

Caused by:
  Unable to update https://github.com/bresilla/bevy_frost?branch=develop#001fc84a

Caused by:
  failed to make directory '/pkg/cargo/git/checkouts/bevy_frost-7cd5eeeaf3285665/001fc84': Read-only file system; class=Os (2)
```

Resolution failure, not compilation: `CARGO_HOME=/pkg/cargo` and the pinned `bevy_frost` checkout is on
a read-only filesystem. Because workspace resolution is all-or-nothing, `-p molla-geometry` never
reaches the compiler despite having no bevy dependency. **No type-check was obtained; every finding
above is from reading source.**

## 9. Limits

- **No compile.** All "unused"/"dead" findings are static-reference findings, not compiler or
  dead-code-analysis output. Trait-object-only reachability could misjudge them.
- The five prior-audit files (`plans/MAP.md`, `REPORT.md`, `plans/_prior_execution_record.md`,
  `GUI_SPLIT.md`, `plans/STATUS.md`) were **not read**; I ran one `ls` to confirm they exist, so I saw
  their names only.
- loc figures are the pre-established ones (27240/74, 7182/26); my `src/*.rs` sums are 19768 and 4722 —
  the established numbers include `tests/`, `benches/`, `shaders/`.
- `external_refs` counts come from `grep -rn ... crates bin --include=*.rs`; they exclude `book/`,
  `xtra/` (workspace-excluded), and non-Rust files.
- No test was executed; the 105 `#[ignore]` attributes are taken at face value as GPU-adapter gates.
- Shader duplication was checked with `find . -name '*.wgsl'` plus `include_str!` greps: no wgsl file is
  shipped twice. The one name collision is `sparse_grid.wgsl`, referenced only by
  `molla-kernels/src/sparse_grid.rs:50`; `molla-geometry/src/sparse_grid.rs:52` merely documents it.
- I did not verify that the viewer's `gpu_proof` strings (`main.rs:2800-3300`) name tests that exist.
