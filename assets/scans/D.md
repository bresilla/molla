# Scout D — blind audit of molla-kernels, molla-vehicle, molla-actuators

Read-only, independent of the prior audit. No compilation was achieved (see §7). Numbers are
physical lines via `git ls-files | xargs cat`.

## 1. Per-crate purpose

**molla-kernels** — 90975 loc / 167 files. A WGSL compute-kernel *catalog* plus one Rust
dispatch wrapper per shader (bind groups, uniforms, upload/readback): a GPU-primitive library,
not a simulator. **Rust src 48167, Rust tests+benches 19881, WGSL 22866** (71 `.wgsl`), toml 61
(~25% shader text, ~22% test code). `src/lib.rs:101-300` re-exports 128 public names, clustered
as `TireFem*` (~50), `TireDem*` (~35), `TireVehicle*` (~30). Nearly every module is
`#[cfg(feature = "gpu")]`; only `dahl_oracle` (100 loc) and `featherstone_adjoint` (288 loc) are
ungated (`lib.rs:43,48`).

**molla-vehicle** — 41793 loc / 53 files, all Rust. CPU wheeled-vehicle/tire physics: terrain
(`terrain.rs`, 1571), tire models (`tire.rs`), suspension, shaft, powertrain, `Vehicle`, sysid,
soft soil (`deformable.rs`, `granular.rs`). **`real_tire/` is 37306 of 41780 src lines (89%)**;
inside it `reports/` is ~6.4k loc and `rig/` ~5.6k, and `real_tire/tests.rs` alone is 9124 lines
(`#[cfg(test)]`-gated, `real_tire/mod.rs:224`). `src/lib.rs` exports ~250 names, mostly
`Tire*Report`.

**molla-actuators** — 4348 loc / 18 files. The *host-side* actuator layer that builds and drives
the kernel wrappers, plus a CPU-only muscle model (`actuator.rs`, six `controller_*`,
`clamping/`, `delay.rs`, `muscle.rs`). Only `muscle` compiles without `gpu` (`lib.rs:24`).

## 2. Real edges vs Cargo claims

Correct: `molla → kernels` (`crates/molla/Cargo.toml:53`, `crates/molla/src/kernels/mod.rs:118`),
`molla → vehicle` (`:59`, `src/vehicle.rs:4`), `molla → actuators` (`:52`,
`src/actuators/mod.rs:41`), `actuators → kernels` (`crates/molla-actuators/Cargo.toml:27` + `:19`;
`controller_pd.rs:19`, same in all six controllers, `delay.rs`, `clamping/`; each owns a kernel,
`controller_pd.rs:38`), `solvers → kernels` (`molla-solvers/Cargo.toml:36`;
`src/semi_implicit.rs:38`, `barrier_solver.rs:59`), `sensors → kernels` (5 files),
`fem → kernels` (3 files).

**The inverted edge.**

```
crates/molla-kernels/Cargo.toml:24    "dep:molla-vehicle",
crates/molla-kernels/Cargo.toml:37    molla-vehicle = { path = "../molla-vehicle", optional = true }
crates/molla-kernels/src/tire_thermal.rs:505   snapshot: &molla_vehicle::TireRigSnapshot,
crates/molla-kernels/src/tire_thermal.rs:539   snapshot: &molla_vehicle::TireRigSnapshot,
crates/molla-kernels/src/tire_thermal.rs:579   snapshots: &[molla_vehicle::TireRigSnapshot],
crates/molla-kernels/src/tire_fem.rs:1404      tire: &molla_vehicle::TetraTireFem,
```

**Verdict: inverted, but not a cycle.** kernels is the low-level GPU-primitive crate (5138
`wgpu::` tokens vs 45 vehicle tokens, mostly doc-comments); vehicle sits far higher and per Cargo
depends on **nothing**. So the arrow points primitive→application. No Cargo cycle exists — but
the direction is backwards. All four uses are **snapshot adapters** (they pack a vehicle struct's
fields into a kernels GPU input), i.e. vehicle-boundary code living in the kernel crate.

**The hidden edge Cargo cannot see:** `crates/molla-vehicle/src/real_tire/observability.rs:124-127`
`include_str!("../../../molla-kernels/shaders/tire_{brush,fem,thermal,transient}.wgsl")` — vehicle
reaches into kernels' source tree although its `[dependencies]` is empty (`Cargo.toml:19-20`).
This is **unique in the workspace** (all other `include_str!` stay within their crate). Together
the two crates are mutually entangled: kernels→vehicle by API, vehicle→kernels by file.

**The missing edge:** `molla-vehicle` depends on nothing — not `molla-core`, `molla-math`,
`molla-sim`. `cargo metadata --no-deps` confirms `molla-vehicle deps: []`. It reimplements math on
raw arrays: `real_tire/internal.rs:47` (`dot`), `:51` (`cross`), plus `tire.rs:290,299`,
`granular.rs:453`. An island with no shared type is *why* it could be depended on without a cycle.

## 3. Dead / dormant

- **Only 388 lines compile without `gpu`** (`dahl_oracle`, `featherstone_adjoint`; `lib.rs:43,48`).
  `dahl_oracle` is called from one test (`molla-solvers/tests/vbd_extended_gpu.rs:29,188,222`) and
  one viewer line (`main.rs:10403`).
- **kernels' production use is mostly facade forwarding.** Of ~56 `molla_kernels::` tokens outside
  the crate, 24 are 3-line `pub use` shims in `crates/molla/src/kernels/*.rs`. Only solvers (2
  files), sensors (5), fem (3), actuators (10) dispatch a kernel. `grep -rn "molla_kernels::" bin/`
  returns **zero** — the viewer goes through `molla::kernels`.
- **vehicle's reports/+rig/ (~12k loc, ~29%)** is consumed by the viewer (`main.rs:6067-7676`) and
  vehicle's own tests only. Validation scaffolding, not simulation physics.

## 4. Redundancy

`kernels/actuator_pd.rs` (201 loc, `ActuatorPdKernel`) vs `actuators/controller_pd.rs` (141 loc,
`use molla_kernels::ActuatorPdKernel`) is a clean two-layer split, **not** duplication — worth
saying because the names invite a wrong merge. The facade's `crates/molla/src/kernels/*.rs` are 24
three-to-six-line shims; the local ones forward to `molla_fem`. The real redundancy is structural:
**every tire kernel has a CPU twin in vehicle** (`tire_brush.rs:3` mirrors `evaluate_brush_tire`;
`tire_fem.rs:3,1388` mirrors `TetraTireFem`; `tire_bekker.rs:3` mirrors `DeformableTerrain`) —
which is exactly what produced the inversion.

## 5. Connection health and correct layering

One inversion, one undeclared edge, one missing edge; every other edge is correct.

```
L0 molla-core, molla-math, molla-storage      L3 molla-actuators -> kernels   (correct)
L1 molla-compute, -compute-primitives         L4 sensors/fem/solvers -> kernels+actuators (correct)
L2 molla-kernels (deps ONLY L0/L1, never       L5 molla-vehicle -> core/math/sim (MISSING) + kernels
   vehicle)                                    L6 molla -> all, kernels and vehicle as siblings (correct)
```

Do: (a) delete `molla-kernels/Cargo.toml:37` and `:24`; (b) move
`tire_thermal_input_from_rig_snapshot`/`_state_from_rig_snapshot`/`_build_rig_replay_batch`
(`tire_thermal.rs:504,538,578`) and `tire_fem_gpu_input_from_tetra_fem` (`tire_fem.rs:1403`) into
`molla-vehicle::real_tire`, which then depends on kernels; (c) give vehicle a real
`molla-core`/`molla-math` dep; (d) replace the four `include_str!` reaches with a declared
mechanism or move those assertions into kernels' tests.

**Who depends on vehicle:** exactly three artifacts — `molla` (`src/vehicle.rs:4`), `molla_viewer`
(300+ sites), and wrongly `molla-kernels`. **Is actuators generic?** Yes: grep for
`vehicle|terrain|tire` in `crates/molla-actuators/src` returns 2 hits, both the word "entirely" in
doc comments (`controller_neural_lstm.rs:26`, `controller_neural_mlp.rs:4`).

## 6. Removal candidates

| candidate | evidence | risk |
|---|---|---|
| kernels' dep on vehicle | `Cargo.toml:37,24`; 4 fns `tire_thermal.rs:504,538,578`, `tire_fem.rs:1403` | MEDIUM — moves public paths; **top-value fix** |
| vehicle's `include_str!` of kernels shaders | `observability.rs:124-127` | LOW struct., MED behaviourally (viewer renders it, `main.rs:6199`) |
| vehicle's `reports/`+`rig/` (~12k loc) | 7+6 files, viewer-only consumers | HIGH — viewer diagnostics depend on it |
| kernels' `rayon` dep | `Cargo.toml:39`; 3 `rayon::` tokens | LOW |
| vehicle's local `dot`/`cross` | `internal.rs:47,51`; `tire.rs:290,299` | LOW-MED — only with (c) above |

## 7. Build probe (verbatim)

```
$ cargo metadata --no-deps --format-version 1
exit 0. 108764 bytes of JSON, empty stderr. Confirms molla-kernels deps include
('molla-vehicle','*',None); molla-actuators include ('molla-kernels','*',None); molla-vehicle deps == [].

$ cargo check -p molla-kernels
exit 101
warning: spurious network error (3 tries remaining): failed to make directory
'/pkg/cargo/git/checkouts/bevy_frost-7cd5eeeaf3285665/001fc84': Read-only file system; class=Os (2)
...(2 and 1 tries remaining, same)...
error: failed to get `bevy_frost` as a dependency of package `molla_viewer v0.0.1
(/home/bresilla/data/code/OUSD/molla/bin/molla_viewer)`
Caused by: Unable to update https://github.com/bresilla/bevy_frost?branch=develop#001fc84a
Caused by: failed to make directory '/pkg/cargo/git/checkouts/bevy_frost-7cd5eeeaf3285665/001fc84':
Read-only file system; class=Os (2)

$ CARGO_HOME=$HOME/.cargo_probe cargo fetch     # writable mirror of /pkg/cargo
exit 101
warning: spurious network error: failed to resolve address for github.com:
Temporary failure in name resolution; class=Net (12)
Caused by: failed to clone into: /home/bresilla/.cargo_probe/git/db/bevy_frost-7cd5eeeaf3285665
Caused by: revision 001fc84aedcad2d9eebf4f4c1b093d7aa8acb9f1 not found

$ CARGO_HOME=$HOME/.cargo_probe cargo check -p molla-kernels --offline
exit 101
Caused by: can't checkout from 'https://github.com/bresilla/bevy_frost': you are in the offline mode (--offline)
```

**Conclusion: no compilation was achieved.** Exit 101 is a *resolution* failure, not a compile
error: cargo resolves the whole workspace before building any package, and `bin/molla_viewer`
depends on git crate `bevy_frost` at rev `001fc84aedcad2d9eebf4f4c1b093d7aa8acb9f1`, which cannot
be checked out because `/pkg/cargo/git/checkouts` is read-only; a writable-HOME retry reached the
network and failed DNS. `cargo metadata --no-deps` did succeed. **Every dead-code claim here is
grep-derived, not compiler-derived.**

## 8. Verdict

This region is one large GPU-primitive catalog (`molla-kernels`, 91k physical lines, a quarter of
it shader text), one self-contained vehicle/tire domain crate (`molla-vehicle`, 41k, 89% of it
`real_tire/` and ~29% validation reporting), and one small honest host layer (`molla-actuators`,
4.3k, genuinely generic). The layering is *almost* right — facade, actuator split, and
sensor/FEM/solver edges all point correctly — but it is spoiled by one inversion: `molla-kernels`
declares a dependency on `molla-vehicle` to get four snapshot-conversion helpers, and vehicle
reciprocates through an undeclared `include_str!` into kernels' shader directory, while vehicle
itself is an island of zero dependencies that reimplements `dot`/`cross`. The fix is small: move
four functions down, delete one Cargo line, add vehicle's missing math dep — after which the
region is cleanly layered, primitives at the bottom, orchestration above, application on top.
