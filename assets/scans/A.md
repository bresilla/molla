# Scout A — region scan (core, math, storage, compute, compute-primitives, optim, autograd)

Independent, read-only audit. Paths relative to repo root. LOC = established
physical-line counts (include docs/tests), not code-only.

## 1. Per-crate purpose

**molla-core** (155/4) — vocabulary only: five u32 handle newtypes (`src/ids.rs`),
`Error`/`Result` (`src/error.rs:9,6`), `Axis` + `MAXVAL` (`src/lib.rs:13,27`).
`src/ids.rs:4` doc-links `crate::sim::ModelBuilder::build`, impossible here — stale.

**molla-math** (5857/17) — the workspace's f64 numerics: `Transform` + spatial 6D
algebra, `dense3` (3×3 SVD/QR/eig/polar + adjoints), `dense_solve` Cholesky,
`sparse` `BsrMatrix`, `iterative` CG/PCG/BiCGSTAB, `coloring`, `robust_loss`,
`solar` (NOAA sun position), `gpu` f32 WGSL mirrors. The one unambiguously
load-bearing crate in the region (581 `molla_math::` tokens; molla-fem alone hits
`sparse`/`iterative` 55/49/31/30/26 times per file).

**molla-storage** (438/3) — one type, `Buffer<T>`, in one 403-line file.
`BufferElement` (`src/buffer.rs:29-120`) is the single f64→f32 boundary; device
half `#[cfg(feature="gpu")]` at 7 sites (`buffer.rs:149…362`).

**molla-compute** (201/3) — `Engine` owning `Arc<Device>`/`Arc<Queue>` + `Backend`,
`AdapterInfo` (`src/engine.rs`). With `gpu` off (`Cargo.toml:13`) `src/lib.rs:3-7`
exports nothing.

**molla-compute-primitives** (4690/19) — seven self-contained algorithms, CPU
oracle + gated GPU path: `radix_sort` (906 loc) and six `tile_*` modules.

**molla-optim** (540/5) — `AdamConfig`/`AdamState`, `SgdConfig`/`SgdState`, f64
state structs mirroring Warp's `wp.optim`. `src/lib.rs:11-19` claims GPU dispatch
lives in `crate::kernels::optim_{adam,sgd}` — false from in here; that is molla-kernels.

**molla-autograd** (8087/19) — CPU reverse-mode differentiation: `Tape`/`TapeOp`
(5 variants) plus five hand-written closed-form adjoints (Featherstone ABA 2416,
MPM 890+774, XPBD 627, Stam 394, arm 194). Explicitly the CPU exception to the
GPU-first rule (`crates/molla/src/autograd/mod.rs`).

## 2. Real dependency edges vs Cargo claims

| from → to | Cargo | source | verdict |
|---|---|---|---|
| math → core | `math/Cargo.toml:19` | 1 token, `dense_solve.rs:9` (`Result`) | real, near-empty |
| storage → core | `storage/Cargo.toml:16` | `buffer.rs:9` | real, load-bearing |
| storage → math (+glam) | `storage/Cargo.toml:17,19` | `buffer.rs:10-12,57,67-70` | real, that IS the crate |
| compute → core | `compute/Cargo.toml:17` | `engine.rs:6` | real, one import |
| compute-primitives → core | `…/Cargo.toml:18` | 7 files | real, error plumbing |
| compute-primitives → math | `…/Cargo.toml:19` | **1 token ever**: `tile_mcgp.rs:27` `use molla_math::Real` | barely real |
| optim → math | `optim/Cargo.toml:16` | **3 tokens**, all `use molla_math::Real` (`adam.rs:7`, `sgd.rs:6`) | barely real |
| autograd → math | `autograd/Cargo.toml:16` | all 7 src modules; `lib.rs:7` `pub use molla_math as math` | real |
| autograd → core | *not declared* | 0 tokens | absent by design |

**Reexport/alias edges that hide real use:** `kernels/src/lib.rs:3-7` and
`solvers/src/lib.rs:7-18` re-export the region under `as` aliases (`crate::math::`,
`crate::core::`, `crate::autograd::`). The 10 files under `crates/molla/src/{autograd,compute}/`
plus `optim.rs:3`, `storage.rs:3`, `math/mod.rs:3` are 1–2-line `pub use` shims.
**Incoming edges**: kernels→autograd (`kernels/Cargo.toml:31`, used in
`featherstone_adjoint.rs`); solvers→autograd (feature-gated, `Cargo.toml:15,30`,
OFF by default, used `solvers/src/featherstone.rs:265`); kernels→optim
(`Cargo.toml:34`, config structs at `optim_adam.rs:14`/`optim_sgd.rs:10`);
kernels→compute-primitives (`Cargo.toml:20-21`, radix sort only).

**Weakest Cargo entries (none strictly zero):** `optim → molla-math` and
`compute-primitives → molla-math` — a whole crate edge for `type Real = f64`.
`molla-core`'s `gpu = []` (`Cargo.toml:16-17`) is an empty marker forwarded by 7
manifests whose only effect is enabling `Error::Gpu` (`error.rs:27`).

## 3. Dead / dormant

- **`tile_*` primitives are demo-only**: every external reference outside the
  crate + facade wrappers is under `bin/molla_viewer/src/examples/` (`63_`, `65_`,
  `66_`, `81_`, `82_`, `90_`, `91_`, `92_*`). No solver or kernel calls them.
- **`math::solar`** (386 loc): one call site, `examples/107_solar_sky.rs:71`.
- **`autograd::stam_diff`** (394 loc), **`TapeCheckpoint`**: 0 external refs.
- **`xpbd_adjoint::{distance_chain_grad_initial_pos, smooth_plane_contact_step_with_grad}`**: 0 external refs.
- **`autograd::{mpm_cpu_diff, mpm_adjoint}`** (1664 loc): reachable only through
  facade shims (`crates/molla/src/compute/mod.rs:17`, `.../autograd/mod.rs:45`).

## 4. Redundancy

- 10 facade shims add zero code and document every edge twice.
- `solvers/lib.rs:13` re-exports kernels, `kernels/lib.rs:5` re-exports compute —
  Cargo is acyclic but the namespace layering reads as a tangle.
- `dense3::eig3_sym` vs `dense_solve::dense_eig_sym` overlap (1 vs 0 external uses).
- molla-math owns generic Krylov solvers while molla-fem re-implements FEM-specific
  ones — worth a check, not a defect.

## 5. Connection health

Low-level spine is healthy and acyclic: core → math → storage →
compute/compute-primitives → optim/autograd. math and storage are central. The top
is frayed in one specific way: crates are correct and tested, but their
interesting behaviour has almost no production caller.

- **optim**: `AdamState::step`/`SgdState::step` are called only from tests
  (molla-fem ×3, molla-kernels ×2, molla-autograd ×2, molla-solvers ×1). Production
  callers read only its *config structs* as data inside molla-kernels' dispatchers.
- **autograd**: genuinely used = `DiffArm` (viewer + `crates/molla/examples/diff_arm_reach.rs`
  + `crates/molla/tests/scene_smoke.rs`), `FeatherstoneRevoluteChain` (kernels+solvers),
  `JacobianScratch`/`JacobianStrategy` (kernels), `Tape` (3 files). MPM/Stam: zero.
- **compute-primitives**: `radix_sort` is real (`geometry/src/lbvh.rs:51`,
  `kernels/src/tire_dem.rs` ×6); the six `tile_*` are viewer material.

**Grep caveat for any prune:** because of the `as` aliases, `grep molla_<name>::`
undercounts; follow `crate::math::`, `crate::core::`, `crate::autograd::` too.
Zero is a candidate, never proof.

## 6. Removal candidates

| candidate | evidence | risk | decision needed |
|---|---|---|---|
| Fold molla-optim into molla-kernels (only real consumer) | 540 loc; imports only `Real` (`adam.rs:7`); production use = config structs (`optim_adam.rs:14`) | low tech, medium churn (5 manifests) | Is `molla::optim` a promised public API? |
| Move/retire `math::solar` | 386 loc, one demo call site (`107_solar_sky.rs:71`) | low | Does a sun model belong in the engine? |
| Extract or drop the six `tile_*` (~1800 loc) | all external refs are viewer examples | medium — 8 scenes depend on them | Primitive library or demo material? |
| Delete the 10 facade shims | 2-line `pub use` each | low code, API break | Is external compat still promised? |
| Drop `molla-core/gpu` empty feature | `Cargo.toml:16-17`; effect is `error.rs:27` | low | Should `Error::Gpu` just always exist? |
| Retire `stam_diff`/`TapeCheckpoint`/2 xpbd fns if diff-MPM is abandoned | 0 external refs | low | Shipped capability or exploration? |

## 7. Verdict

This region is the engine's numerical foundation and is in good shape underneath:
a small vocabulary crate, a genuinely strong math crate that half the workspace
reaches into, a single-purpose storage boundary, a thin GPU runtime, and two large
well-documented computational libraries. Nothing is duplicated spaghetti and no
crate cycle exists. What it should be is narrower and more honestly wired: ~2000
loc of compute-primitives (`tile_*`), ~2000 loc of autograd (stam + MPM) and 386
loc of solar have no consumer outside the demo viewer; the 540-loc optim crate
exists to be depended on for four config structs; and the only production path
into autograd runs through an alias re-export in molla-kernels. The fix is not
deletion for its own sake but a decision, per capability, between "promote it and
give it a real caller or an external example" and "demote it to the viewer /
delete it" — then let the crate graph say that out loud instead of hiding it
behind ten empty shims.

## Build probe (verbatim)

```
$ cargo metadata --no-deps --format-version 1
exit 0, no stderr, 108764 bytes of JSON on stdout. Workspace manifest itself is consistent.

$ cargo check -p molla-storage --no-default-features --features gpu
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
exit 101

$ cargo check -p molla-storage --no-default-features --features gpu --offline
identical failure — the checkout dir is unwritable, not the network.
```

Environment: `ls /pkg/cargo/git/checkouts/bevy_frost-7cd5eeeaf3285665/` is **empty**
(`total 0`); `touch` there → `Read-only file system`; `mount` shows
`/dev/sdb1 on /pkg/cargo/git type btrfs (ro,…)`. `CARGO_HOME=/pkg/cargo`.

**Conclusion: no crate in this region was compiled or type-checked.** Resolution
dies during package-graph construction, before rustc, on `bin/molla_viewer`'s git
dependency. Environmental and outside my region; every claim above is source- and
grep-derived. A probe excluding `bin/molla_viewer` (or a writable `CARGO_HOME`) is
needed for server truth.

## Periphery (inventory + tracked size)

| path | files | bytes | verdict |
|---|---|---|---|
| `xtra/` | 0 | 0 | **stale/untracked reference material**. In Cargo `exclude` (`Cargo.toml:21`), never committed (`git log -- xtra` empty), not in `.gitignore` either. ~3.3 GB on disk (chrono 2.2G, helios 940M, warp 99M, newton 73M). Zero tracked weight. |
| `book/` | 268 | 11 877 094 | **source live, build output accidentally tracked**: `book/src` = 119 files / 863 947 B (SUMMARY.md, 187 entries); `book/book` = 149 files / **11 012 365 B** of rendered HTML+fonts+JS. Makefile `docs` and CI both emit to `docs/`, not `book/book/` → stale; belongs in `.gitignore`. 93% of book weight. |
| `Makefile` | 1 | 3 920 | **live**. `build/run/test/check` target `APP_PKG=molla_viewer`; workspace lanes are `test-all` and `harden` (`Makefile:71-77`: fmt, no-default-features check+test, clippy `-D warnings`, all-features tests). |
| `.github/` | 2 | 6 326 | **live**: `book.yml` (mdBook → gh-pages), `rust.yml` (fmt/clippy/build/test + continue-on-error `--ignored` GPU lane, `RUSTFLAGS=-D warnings`). |
| `flake.nix`, `flake.lock`, `.envrc` | 3 | 7 367 | **live**: `.envrc` runs `use nvidia; use display; use flake --impure` — the devShell `make run RUN_WITH=nixVulkan` needs. |
| `Cargo.toml`, `Cargo.lock`, `.gitignore`, `PROJECT`, `LICENSE` | 5 | 149 905 | **live**. 17-member workspace, `exclude = ["xtra"]`, curated lint block (`Cargo.toml:29-56`). Cargo.lock = largest tracked text file (146 956 B). `.gitignore` does NOT ignore `book/book`. |
| `assets/` | 0 | 0 | **live but untracked**; holds only `assets/scans/`. |

## Limits

No successful build — the dominant limit. I did not read `plans/MAP.md`,
`REPORT.md`, `plans/_prior_execution_record.md`, `GUI_SPLIT.md` or
`plans/STATUS.md`; I incidentally saw those filenames in a top-level `ls` /
`git ls-files`, none of their contents. LOC are physical lines — in this
unusually comment-dense repo (optim `src/lib.rs`: 52 lines mostly doc) that
overstates executable code substantially. Edge counts are grep-derived lower
bounds. No feature-matrix analysis (impossible without a build). The "demo-only"
verdicts assume no consumer outside this repository. I audited no other workspace
crate beyond establishing edges into my region.