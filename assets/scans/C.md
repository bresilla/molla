# Scout C — blind audit of `molla-solvers` and `molla-fem`

Region: `crates/molla-solvers` (50049 loc / 118 files), `crates/molla-fem` (56747 loc / 133 files).
All claims are source-derived; **no build completed** (see §7).

## 1. Per-crate purpose, from the code

**`molla-solvers` — time-domain dynamics.** A library of integrators that advance a `State` by
`dt`. Rigid: `featherstone.rs:1-14` (reduced-coordinate ABA), `crba.rs`, `rnea.rs`, `jacobian.rs`,
`xpbd.rs:1-10`, `semi_implicit.rs:1-5`, `ik.rs:1-8`. Soft: `vbd.rs:1-8` (vertex block descent),
`vbd_soft_contact.rs:1-10` (VT/EE via `crate::geometry::SoftBvh`), `barrier_contact.rs:1-28`
(IPC cubic barrier), `barrier_solver.rs:1-8` (the `Solver`-trait wrapper). Particles/fluids:
`mpm.rs:1-8` (GPU granular, APIC affine `C`), `implicit_mpm.rs:1-8` (Bingham), `sph.rs:1-7` (WCSPH),
`sph_oracle.rs:1-8`, `sph_diff.rs`, `mpm_oracle.rs:1-6`. The crate-wide contract is the `Solver`
trait at `lib.rs:74-88`.

**`molla-fem` — spatial discretisation.** `lib.rs:1-17` calls it "Finite-element method framework",
and the module list is a full toolbox: geometry and spaces (`mesh.rs`, `function_space.rs`,
`quadrature.rs`, `p2_element.rs`, `nedelec_element.rs`, `dg_element.rs`), local kernels
(`element.rs:1-8`), assembly (`assembly.rs`), BCs (`bc.rs`), then the PDE solvers built on it —
Poisson/heat/wave (`solver.rs:1-6`), Taylor-Hood Stokes/NS (`stokes.rs:1-6`), time-harmonic
Maxwell (`eddy_current.rs:1-6`), magnetostatics (`magneto.rs`), Signorini contact
(`signorini.rs:1-4`), phase field, AMR (`amr_mesh.rs`), adjoint (`adjoint.rs:1-6`), FFT/Stam fluid,
and the GPU ports in `gpu_kernels/` and `*_gpu.rs`.

## 2. Real dependency edges vs Cargo claims

| from → to | Cargo claims | source evidence | verdict |
|---|---|---|---|
| `molla` → `molla-solvers` | `crates/molla/Cargo.toml:56` | `crates/molla/src/solvers/*.rs:3` re-exports; `solvers/mod.rs:37,44-46`; `compute/mpm.rs:3` | real, consumed |
| `molla` → `molla-fem` | `crates/molla/Cargo.toml:58` | `fem/mod.rs:6` `pub use molla_fem::*`; 8 `kernels/*.rs:3`; `compute/stam_fluid.rs:6` | real, consumed |
| `molla-solvers` → `molla-fem` | **absent** | `grep molla_fem crates/molla-solvers/` = 0 hits | no edge |
| `molla-fem` → `molla-solvers` | **absent** | only a doc comment, `crates/molla-fem/tests/fem_surface_contact.rs:3` | no edge |
| viewer → both | `bin/molla_viewer/Cargo.toml:33` (only `molla-vehicle`) | `main.rs:61` (solvers), `main.rs:53` + 19 examples (fem) | both arrive **through the facade** |

The two crates are **siblings, not a stack**: zero mutual edges. Both depend on the same six
internal crates (`molla-core`, `-math`, `-sim`, `-geometry`, `-storage`, `-kernels`/`-compute`).

**Unused / over-declared Cargo deps.** `molla-solvers`: `molla-storage` (1 token, `lib.rs:18`),
`molla-actuators` (alias only, `lib.rs:7`), `tracing` (one call, `vbd.rs:2176`), `glam` (only
`EulerRot`, `featherstone.rs:1392,1431`). `molla-fem`: `molla-storage` and `molla-compute` are
declared **non-optional** (`Cargo.toml:24`) yet every source reference sits inside
`cfg(feature = "gpu")` modules (`lib.rs:53,55,69-113`) — both belong behind `dep:`.

**Likely compile break (unconfirmed).** `crates/molla-fem/src/stam_fluid.rs:51` and `:53` are
ungated `use`s (`crate::fem::{PoissonStencil, solve_pressure_poisson_fft_with_stencil}`,
`crate::kernels::FftKernel`) but their targets are gpu-gated (`lib.rs:88`, `lib.rs:55-56`), and
`stam_fluid` itself is ungated (`lib.rs:115`). A `--no-default-features` build should fail. I could
not reproduce it by compiling.

## 3. Dead / dormant

- **`MpmConfig::friction_angle`** (`molla-solvers/src/mpm.rs:254`) — never read. Every site is a
  write *from* `rheology` (`:1031-1032`); runtime reads `rheology` (`:1420`). Doc at `:250-253`
  already says "Deprecated".
- **`StokesSimpleConfig::alpha_p`** (`molla-fem/src/stokes.rs:795`) — comment at `:793` says
  "currently unused"; only writes remain (`:810`, `tests/fem_stokes.rs:298`,
  `viewer/examples/51_fem_diff_beam.rs:149`).
- **`SphFsiOracle`** (`sph_oracle.rs:363`, exported `lib.rs:62`) — **zero** external references
  workspace-wide; one in-file test at `:684`.
- **`molla-fem::russell`** — gated `lib.rs:113`, off by default, reached only by
  `tests/fem_russell.rs`; `Cargo.toml:35-42` admits the upstream crate never shipped.
- **`amg.rs` + `multigrid.rs`** (1564 loc) — sole in-crate consumer is `eddy_current.rs`
  (`:87,93,996,1014,1699,1780`); zero external users.
- **Test-only public API in fem**: `solve_stokes_gpu`, `solve_mixed_elasticity_gpu`,
  `solve_eddy_current_steady`, `solve_active_set_signorini`, `solve_phase_field_coupled_step`,
  `assemble_global_amr`, `MeshHierarchy`, `AmgHierarchy` — viewer/facade refs **0**, tests only.
  Contrast `solve_pressure_poisson_fft` (6 viewer refs) and `solve_magnetostatics_steady` (3).

## 4. Redundancy — the head-to-head

| concept | solvers evidence | fem evidence | verdict |
|---|---|---|---|
| Neo-Hookean tet | `vbd.rs:1897` **analytic** `P = μF + s·cof(F)` (`:1970-1973`) + analytic Hessian (`:2030-2040`) + plastic `F=Fe·Fp` (`:1922-1954`) | `element.rs:765/780` force & tangent = **finite differences** of the energy at `:708` (same `½μ(I₁−3) − μlnJ + ½λln²J`) | **Same law, different model. Keep solvers separate — but dedupe the formula.** FEM pays ~9 extra energy evals/element/iteration for a tangent VBD has in closed form. |
| Newton | `barrier_contact.rs:1-28` damped Newton + CCD line search, matrix-free PCG | `solver.rs:1656,1825`, `signorini.rs:1308` PDAS, Uzawa-CG (`stokes.rs`), GMRES (`eddy_current.rs`) | **Different problems sharing a name.** Particle-potential minimisation vs DoF saddle-point/residual. Keep separate; rename for clarity. |
| Contact / BC | barrier (`barrier_contact.rs:1-28`), VT/EE (`vbd_soft_contact.rs:1-10`), XPBD (`xpbd.rs:1-10`) | Signorini Lagrange + Coulomb cone (`signorini.rs:1-60`), Dirichlet/Nitsche (`bc.rs`), penalty (`nonconforming.rs`) | **Three distinct formulations — keep separate.** Both correctly share the broad phase via `molla-geometry` (`vbd_soft_contact.rs:26`; `signorini.rs:124`). |
| Krylov / LA | CPU side delegates to `molla-math` (`barrier_contact.rs:29-30`); GPU in `molla-kernels` (`bsr_solve.rs:27,138`) | CPU: `solver.rs:29` imports `math::iterative::{cg_solve, pcg_solve}` (`:177,208,291,332`). **GPU: own 3873-loc stack**, `gpu_kernels/fem_cg.rs` (`:614,794,950,1152,1325`), explicitly claiming parity with `math::iterative` (`:613,1323`) over the same `kernels::BsrGpu` (`:1155,2845`) | **Real GPU duplication — mergeable.** Two CG/PCG implementations over one BSR type in one workspace. |
| Fluids | Lagrangian: `mpm.rs:63,186` (APIC affine), `sph.rs:1-7`, `implicit_mpm.rs:1-8` | Eulerian: `stokes.rs:1-6`, `fft_ns.rs:4-6` (self-declared "complementary"), `stam_fluid.rs`, `multiphase.rs`; `apic_coupling.rs:1-40` is a particle→**FEM mesh** bridge, not a second MPM | **Not redundant.** Particle vs grid discretisations. Rename `apic_coupling` → `mpm_fem_hybrid`. |

**Shared by design** (keep): CPU Krylov via `molla-math/src/iterative.rs:155,247,361`; contact
candidates via `molla-geometry`.

## 5. Connection health

Neither crate is orphaned; both are live in the viewer (solver engines constructed at
`main.rs:3534-3557,3863,4007,11199`; 19 fem example scenes). The health problem is not
disconnection but **over-export**: `fem/mod.rs:6` is a blanket `pub use molla_fem::*` on top of
`lib.rs:120-245`'s ~125-line re-export wall, so every internal experiment becomes a public API
commitment — which is exactly why so many fem entry points have zero non-test callers.

## 6. Removal candidates

| candidate | evidence | risk | decision |
|---|---|---|---|
| fem's private GPU CG (`gpu_kernels/fem_cg.rs`, 3873 loc) | overlaps `bsr_cg_gpu`/`bsr_gmres_gpu` (`bsr_solve.rs:27,138`) on the same `BsrGpu` | **high** — benchmark first | are the lazy/fused paths measurably faster, or predate `bsr_cg_gpu`? |
| `MpmConfig::friction_angle` | never read; `mpm.rs:1031-1032,1420` | low | keep for compat or delete? |
| `StokesSimpleConfig::alpha_p` | `stokes.rs:793` says unused | low | — |
| `SphFsiOracle` | 0 external refs | low | still on the roadmap? |
| `molla-fem::russell` | off by default, `Cargo.toml:35-42` | low | keep escape hatch or drop? |
| FEM's FD hyperelastic tangent | duplicates analytic `vbd.rs:1970-2040` | medium | is the FD accuracy known-good? |
| make `molla-compute`/`-storage` optional in fem | all uses gpu-gated | low | is CPU-only fem supported at all? |

## 7. Build probe (verbatim) and verdict

```
$ cargo metadata --no-deps --format-version 1
EXIT=0        # succeeded; confirmed declared dep kinds/optionality for both crates

$ cargo check -p molla-solvers
EXIT=101
warning: spurious network error (3 tries remaining): failed to make directory
'/pkg/cargo/git/checkouts/bevy_frost-7cd5eeeaf3285665/001fc84': Read-only file system; class=Os (2)
error: failed to get `bevy_frost` as a dependency of package `molla_viewer v0.0.1`
Caused by: failed to make directory '/pkg/cargo/git/checkouts/bevy_frost-...': Read-only file system

$ cargo check --offline -p molla-solvers        # same failure; the checkout dir exists but is EMPTY
$ cargo tree -p molla-fem --no-default-features --offline   # same failure (one resolution graph)
```

`--offline` does not help: `/pkg/cargo/git/checkouts/bevy_frost-7cd5eeeaf3285665` is an empty
directory on a read-only filesystem. An isolated probe outside the workspace got further — it
resolved and began compiling — then died with `No space left on device (os error 28)`; `df -h /tmp`
shows a 5.9G tmpfs at **100%**, 0 available. **No build completed.** Every other claim here is from
reading source and grep.

**Verdict.** *Should `solvers` and `fem` be merged?* **No.** They are siblings with zero mutual
dependency edges, and the code shows they solve disjoint problems: `solvers` is time-domain
dynamics over particles and joints (MPM/SPH/XPBD/VBD/barrier), `fem` is spatial discretisation over
a mesh (elements, assembly, PDE solvers). The name collisions — Newton, contact, elasticity, CG —
are genuine conceptual overlaps but not duplicated implementations; the one place where the same
algorithm really is written twice is the GPU Krylov stack, where `molla-fem/src/gpu_kernels/fem_cg.rs`
(3873 loc) reimplements what `molla-kernels/src/bsr_solve.rs` already provides over the same `BsrGpu`
type. The second real duplication is the Neo-Hookean constitutive law, analytic in `solvers/vbd.rs`
and finite-differenced in `fem/element.rs`. So: **keep the two crates separate, extract the shared
constitutive formula into `molla-math`, and consolidate the GPU Krylov stack** — after benchmarking,
since fem's lazy/fused variants may be load-bearing. The larger structural issue is not redundancy
between the crates but fem's blanket re-export, which has promoted a large body of test-only code to
public API.
