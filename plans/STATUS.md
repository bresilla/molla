# molla — project STATUS map

_Last regenerated: 2026-08-11 (full multi-agent audit of every crate + WGSL shader)._

This is the "where are we, actually" map: what is done, what is half-built, what is
named-but-empty, and how the crates wire together. It is descriptive of the code as
it stands, cross-checked against source (not against the old planning logs).

- Forward work / roadmap → [`ROADMAP.md`](ROADMAP.md)
- The real-tire program specifically → [`TIRE.md`](TIRE.md)

### Recently landed (2026-08-11)
Work against the "three recurring patterns" below:
- **`SuspensionKind` is no longer label-only** — each kind now has a distinct
  vertical-plane linkage character (rising wheel rate, damping, bump-stop travel) in
  `Suspension::force`. Verified: `molla-vehicle` 226/0.
- **The two analytic "FEM port" scenes are now honestly labeled** "(analytic field)"
  (WarpFemTaylorGreen, WarpFemStreamlines) — they still run no solver, but the UI no
  longer claims a FEM solve.
- **Scalar FEM Neumann flux now integrates** its P1 surface load into the RHS
  (`bc.rs`), matching the vector-traction path — was a documented no-op. Verified via
  `bc` tests.
- **Map correction:** the barrier solver's GPU path (`BarrierNewtonGpu`) **is** wired
  and selectable via `BarrierSolver::new_gpu` — the earlier "GPU kernel not wired"
  claim was stale.
- **Pattern 1 (GPU-first): the entire `molla-geometry` CPU-only set is now ported +
  hardware-validated** (each an exact mirror of its CPU oracle, own parity lane):
  - `sdf_bake` → `bake_mesh_to_sdf_gpu` — `T=sdf_bake_gpu`
  - `self_contact` → `detect_vertex_triangle_self_contacts_gpu` — `T=self_contact_gpu`
  - `soft_ccd` (VT-CCD) → `vt_ccd_batch_gpu` — `T=soft_ccd_gpu`
  - `marching_cubes` → `marching_cubes_gpu` (uploads the Lorensen tables; exact index
    topology) — `T=marching_cubes_gpu`
  - `soft_bvh` (swept broad-phase query) → `soft_bvh_query_points_swept_gpu` — `T=soft_bvh_gpu`

  **FEM band in progress** (each hardware-validated against its f64 oracle):
  - `adjoint` Young's-modulus gradient → `compute_grad_youngs_gpu` (per-cell `λᵀKu`) — `T=adjoint_gpu`
  - `phase_field` damage Jacobi solve → `solve_phase_field_fracture_jacobi_gpu` (per-row
    bounded-irreversible sweeps, ping-ponged) — `T=phase_field_gpu`
  Remaining `molla-fem` units (Nédélec/magneto, signorini, AMR, eddy-current) + VBD CCD are
  larger linear solves / active-set iterations driving the GPU CG/PCG (tracked in ROADMAP).
- **Pattern 2 investigated → not a defect pattern** (see the gap list below): all three
  "not-the-default" items are intentional, reachable, tested opt-ins; nothing to fix.

## How to read the status labels

| Label | Meaning |
|---|---|
| **complete** | Implemented and tested; trust it. |
| **partial** | Works, but a significant sub-case / backend / feature is missing (documented). |
| **experimental** | Implemented but unvalidated in-tree, or validated only via manual GPU lanes. |
| **label-only** | Named as if it does something; the physics is identical to a simpler variant, or no solver runs. |
| **stub / missing** | Skeleton, placeholder, or absent. **molla has almost none of these** (see below). |

Backend column: `cpu` = host f64 only, `gpu` = device f32 only, `gpu+cpu` = both paths
exist (usually with a parity test), `n/a` = data/infra, no compute.

### Headline: there is very little literal "stub" debt
A workspace-wide sweep found **~23 files** touching `todo!/unimplemented!/FIXME/TODO`, and on
inspection almost all are legitimate `unreachable!` match guards or prose doc-notes. The
**only hard `unimplemented!`** in the engine is `molla-fem/src/element.rs:829` (nonlinear
deformation gradient is Tet-only, panics on Tri). The viewer's `PhysicsAuditStatus::Placeholder`
variant is deliberately **empty** (test-enforced).

**So the real gaps are not stubs — they are `partial` / `label-only` / `cpu-only-vs-GPU-first`.**
That is where this document points the flashlight.

---

## Crate dependency graph (how it connects)

Strict bottom-up DAG, split by **function**, not by CPU-vs-GPU. Every physics primitive
has a runtime CPU/GPU toggle (`new_cpu()` / `new_gpu()`) behind one trait — the backend is
not a compile-time feature of the algorithm.

```
                                   ┌─────────────┐
  LAYER 0  handles/errors          │ molla-core  │
                                   └──────┬──────┘
                                          │
  LAYER 1  host f64 math            ┌─────▼──────┐
                                    │ molla-math │
                                    └──┬──────┬──┘
                     ┌─────────────────┤      ├──────────────┐
  LAYER 2            ▼                 ▼      ▼               ▼
              ┌────────────┐   ┌────────────┐   ┌──────────────┐
              │molla-storage│  │molla-compute│  │ molla-optim  │
              │ f64⇄f32     │  │ wgpu runtime│  │ Adam / SGD   │
              │ Buffer<T>   │  └──────┬──────┘  └──────┬───────┘
              └──────┬──────┘         │                │
                     │        ┌───────┴────────┐       │
  LAYER 3            │        ▼                ▼        ▼
                     │ ┌───────────────────┐ ┌──────────────────┐
              ┌──────▼─▼──┐ │molla-compute-  │ │ molla-autograd   │
              │ molla-sim │ │  primitives    │ │ CPU reverse-AD   │
              │Model/State│ │ radix/tile/... │ │ (tape+adjoints)  │
              └──────┬────┘ └───────┬────────┘ └────────┬─────────┘
                     │              │                   │
  LAYER 4            └──────┬───────┴───────────────────┤
                           ▼                            │
                   ┌──────────────┐                     │
                   │molla-geometry│  collision: BVH/LBVH,│
                   │ broad+narrow │  GJK/EPA/MPR, SDF,   │
                   └──────┬───────┘  hydroelastic, splat │
                          │                              │
  LAYER 5                 ▼                              │
                   ┌──────────────┐◄────────────────────┘
                   │molla-kernels │  71 WGSL dispatch wrappers
                   │ THE shaders  │  + CPU oracles
                   └──┬────┬───┬──┘
        ┌────────────┘    │   └──────────────┐        ⚠ molla-kernels also
        ▼                 ▼                  ▼          optionally depends on
 ┌─────────────┐  ┌──────────────┐   ┌──────────────┐   molla-vehicle (gpu feat)
 │molla-actuat.│  │  molla-fem   │   │molla-sensors │   — a LAYERING INVERSION
 │(gpu-gated)  │  │ FEM/PDE zoo  │   │(gpu-gated)   │   (see ROADMAP A6/kernels).
 └──────┬──────┘  └──────────────┘   └──────────────┘
        │
  LAYER 6                                        ┌──────────────────────┐
        ▼                                        │   molla-vehicle      │
 ┌──────────────┐                                │  ZERO deps (island)  │
 │molla-solvers │  Solver trait: semi-implicit,  │  tire/terrain/soil   │
 │              │  XPBD, Featherstone, VBD,       │  its GPU kernels live │
 │              │  barrier, MPM, SPH, IK          │  in molla-kernels ───►│ (that's the
 └──────────────┘                                └──────────────────────┘  inversion)

  FACADE   molla ── re-exports all of the above (+ still HOSTS geometry/solvers/
                    kernels/compute internally); prelude = Model/State/Control/Contacts
  APP      bin/molla_viewer ── molla + molla-vehicle + Bevy 0.18 / wgpu 27 / egui
```

Key structural notes:
- **`molla-vehicle` is a dependency island** (not even `molla-math`; it re-implements
  `[f64;3]` vec math inline). Yet **`molla-kernels` optionally depends back on it** for the
  tire dispatchers — the one inverted edge in the graph.
- **`molla` (facade)** is two things: a re-export shim for the extracted crates *and* the
  actual home of `geometry` / `solvers` / `kernels` / `compute` code that was never extracted.
- The viewer splices **135 scene files via `include!`** (not modules) into one 11k-line
  `main.rs` → one flat namespace, one compilation unit.

---

## Status by cluster

### Foundation — core / math / storage / compute / compute-primitives / sim
Mature and well-tested. The f64→f32 narrowing all funnels through `molla-storage::Buffer<T>`.

| Module | Status | Backend | Note |
|---|---|---|---|
| core (ids/error/Axis) | complete | n/a | trivial handle/error types |
| math (transform, spatial, dense3 SVD/QR/eig, CG/PCG/BiCGSTAB/GMRES, BSR sparse, coloring, robust-loss, solar) | complete | cpu | GPU only via POD mirror structs |
| storage `Buffer<T>` | complete | gpu+cpu | the single f64⇄f32 chokepoint |
| compute `Engine`/`Backend` | complete | gpu+cpu | wgpu runtime; `new_cpu`/`new_gpu` |
| compute-primitives: matmul, convolution, filtering, radix-sort | complete | gpu+cpu | radix-sort is gpu-only by design |
| compute-primitives: **block-Cholesky** | partial | gpu+cpu | GPU single-workgroup, **N>1024 returns garbage** (host rejects); multi-WG = v2 |
| compute-primitives: **stream-compaction** | partial | gpu+cpu | GPU capped n≤1024; decoupled-look-back = v2 |
| compute-primitives: **tile-mcgp (SPSA)** | partial | cpu | **no GPU kernel at all** despite being a "tiled primitive" |
| sim: Model/State/Control/Contacts, builder, FK, joints | complete | n/a / cpu | data-model layer; all 8 joint types in FK |
| sim: **analytic SDF shapes** | partial | n/a | ships torus+sphere; user-callback SDFs = TODO |

### Geometry — collision
Broad, mature **CPU** stack; GPU coverage is real but selective and mostly fallback-guarded.

| Module | Status | Backend | Note |
|---|---|---|---|
| narrow_phase (GJK/EPA + SAT dispatch), pipeline, broad_phase (SAP), bvh, mesh_bvh | complete | cpu (pipeline gpu+cpu) | exhaustive shape dispatch; well tested |
| gjk / epa / mpr (CPU) | complete | cpu | the authoritative narrow phase |
| support, hull (QuickHull), decimate (QEM), spatial_hash, contact_reduction, contact_extras, aabb | complete | cpu | |
| marching_cubes | complete | gpu+cpu | GPU port landed 2026-08-11 (`marching_cubes_gpu`, exact-topology parity test) |
| sdf_bake (mesh→voxel SDF) | complete | gpu+cpu | GPU bake landed 2026-08-11 (`bake_mesh_to_sdf_gpu`, parity-tested); closed-manifold only |
| **lbvh** broadphase | experimental | gpu | validated via GPU test; top-down variant was deleted (race) |
| **gjk_gpu / epa_gpu / mpr_gpu** | experimental | gpu | emit `STATUS_FALLBACK` → re-run on CPU; not authoritative alone |
| **mesh_bvh_raycast_gpu** | partial | gpu | stack depth capped 64; deeper trees silently → "no hit" |
| **sparse_grid** (NanoVDB-style) | partial | gpu+cpu | CPU + GPU-upload; device traversal maturity unverified |
| **hydroelastic** pressure-field contact | partial | gpu+cpu | works with documented approximations (Fibonacci sampling, coarse bucketing) |
| **gaussian_splat** | partial | cpu | forward + FD-gradient oracle only here; real adjoint/GPU lives in molla-kernels |
| self_contact | partial | gpu+cpu | GPU port landed 2026-08-11 (`detect_vertex_triangle_self_contacts_gpu`, parity-tested); still O(N²) brute force, VT candidates only, no edge-edge |
| soft_ccd | partial | gpu+cpu | GPU batch port landed 2026-08-11 (`vt_ccd_batch_gpu`, parity-tested); vertex-triangle only, no edge-edge CCD |
| soft_bvh | partial | gpu+cpu | GPU broad-phase query landed 2026-08-11 (`soft_bvh_query_points_swept_gpu`, parity-tested); flat AABB sweep — LBVH refit for scale still future |
| **tet_bvh** | partial | gpu+cpu | named "BVH" but is flat-AABB brute force + top-level cull (honest in docs) |

### Kernels — 71 WGSL dispatch wrappers (the engine room)
Mostly complete-and-tested families; CPU oracles live in *sibling* crates. Biggest issue is
**in-crate test coverage gaps**: several families have no `tests/` here (parity is asserted
only in the solver crate, or not at all).

| Family | Status | Backend | Note |
|---|---|---|---|
| splat_raytrace (fwd + full adjoint + SH) | complete | gpu | strong differentiable coverage incl. pose-recovery |
| barrier_newton (C0 f32 port) | complete | gpu+cpu | on-device Newton; oracle in molla-solvers |
| fft (Stockham), scan, dense3 (+adjoint), robust_loss, optim (adam/sgd), hydroelastic, contact_reduction, sparse BSR (+SpGEMM), dahl_friction, frame_transform | complete | gpu+cpu | clean, tested |
| **articulated dynamics** (featherstone/crba/rnea/jacobian) | partial | gpu+cpu | **CRBA revolute+prismatic only** (Fixed/multi-DOF unsupported); no in-crate parity test |
| **featherstone adjoint** | partial | gpu+cpu | GPU only wins at chain_count≳8; `GpuFallbackCpu` falls back for single articulation; MAX_LINKS=8 |
| **integration / xpbd / xpbd_particles / xpbd_equality** | partial | gpu | full pipelines but **no `tests/` in-crate** — parity rests on solver crate |
| **implicit_mpm** | partial | gpu+cpu | Bingham return-mapping still runs **CPU-side**; no in-crate test |
| **actuators** (pd/pid/mlp/lstm) + **clamping** + **muscle** | partial | gpu | dispatch wrappers, **zero in-crate parity tests** |
| **sensors** (imu/raycast/tiled_camera/contact) | partial | gpu | **zero in-crate tests**; IMU finite-differences accel |
| **tire** (brush/bekker/dem/fem/mpm/thermal/transient/vehicle) | partial | gpu+cpu | best-covered family; but tire_mpm/tire_vehicle are compact single-lane couplings, not full solvers |
| **cloth_diff / vbd_cloth** | partial | gpu+cpu | cloth_diff tested; vbd_cloth untested here, stubs tet/bending for cloth-only |
| **soft_vt_contact / soft_bvh_gpu** | experimental | gpu | refit-only BVH w/ documented Morton-drift limit; untested in-crate |
| **sparse_grid insert/delete** | partial | gpu+cpu | leaf reclamation marks-inactive but **not folded into topology** |

### Solvers — the Solver trait + continuum sims
Rigid-body pillars are complete with GPU parity. Soft/continuum are real but carry honest caveats.

| Module | Status | Backend | Note |
|---|---|---|---|
| semi_implicit, xpbd, featherstone (ABA, all joints), rnea, crba, jacobian | complete | gpu+cpu (rnea/crba/jac cpu) | the trustworthy core |
| barrier_contact (IPC oracle) | complete | cpu | f64 oracle; κ is a mass-consistent **placeholder** (elasticity-inclusive deferred) |
| sph_oracle, mpm_oracle | complete | cpu | verification-only helpers, not full CPU sims |
| **ik** | partial | gpu+cpu | GPU path: **SphereAvoid unsupported**; JointLimit is soft-penalty only |
| **vbd** (cloth + soft-on-rigid, +AVBD) | partial | gpu+cpu | CCD/tet-vs-tet/friction are **CPU-only**; sparse SDF backing not GPU-wired |
| **vbd_soft_contact** (VT/EE) | partial | cpu | quasi-rigid impact-arrest approx; self-collision is separate stage |
| barrier_solver engine | partial | gpu+cpu | GPU `BarrierNewtonGpu` **is** wired (`new_gpu`/`step` dispatch, static obstacles); v1 gaps: doesn't consume Model tets, rigid coupling weak/explicit |
| **mpm** (granular) | partial | gpu | "Implicit" = fixed-iter **Jacobi, not CG**; rheologies real |
| **implicit_mpm** (Bingham) | partial | gpu+cpu | single rheology; default 27-pt Hessian (125-pt opt-in); wall-time target deferred |
| **sph** (WCSPH) | partial | gpu | FSI coupling quasi-static within a batch |
| **sph_diff** | experimental | cpu | centroid-of-fluid losses only; full adjoint + GPU deferred |

### FEM — the PDE zoo (55k LOC, unusually complete for its size)
Genuinely implemented + tested CPU physics. GPU acceleration is **selective**, and several
solvers ship documented approximations.

| Module | Status | Backend | Note |
|---|---|---|---|
| P1 elements, solver drivers (steady/transient/wave/nonlinear), apic_coupling, stam_fluid, mesh/quadrature/assembly/bc, dg_element, magneto, signorini (hard contact), nonconforming, amg | complete | mixed | see backends below |
| p2_element (Taylor-Hood), nedelec (edge/H(curl)) | complete | cpu | no GPU path; no P3 |
| adjoint (linear-elasticity) | complete | gpu+cpu | Young's-modulus gradient GPU-ported 2026-08-11 (`compute_grad_youngs_gpu`, tet-only, parity-tested `T=adjoint_gpu`); solve reuses GPU CG |
| gpu_kernels (resident CG/PCG, elasticity, stokes, NS, wave/transient, DG-SWE) | complete | gpu | f32 (~1e-4 vs f64); coverage matches which CPU modules were ported |
| **element.rs nonlinear (Neo-Hookean/MR/SVK)** | partial | gpu+cpu | **Tet-only; Tri panics** (`unimplemented!` at :829) |
| **stokes / stokes_gpu** | partial | gpu+cpu | simplified Laplacian (exact only all-Dirichlet) + diagonal-Schur SIMPLE |
| **eddy_current** | partial | cpu | AMS preconditioner approximates diagonal block only (documented) |
| **multigrid** (geometric) | partial | cpu | v3 stepping-stone, superseded by AMG; non-recursive |
| **amr_mesh** | partial | cpu | refinement complete; **coarsening is single-level** |
| phase_field (fracture) | partial | gpu+cpu | damage Jacobi solve GPU-ported 2026-08-11 (`solve_phase_field_fracture_jacobi_gpu`, parity-tested `T=phase_field_gpu`); assembly host-side; 2D-tri, quasi-static |
| **multiphase** (level-set CSF) | partial | cpu | curvature/surface-tension **primitives only, not a full solver** |
| **russell / DirectSolver** | partial | cpu | named a sparse bridge but is a **dense LU**, capped ~1024 dim, off by default |
| bc: Neumann flux + traction | complete | cpu | scalar flux and vector traction both integrate the P1 area/3 surface load into the RHS |

### Vehicle + tire — the flagship (41k LOC, zero deps, unusually honest self-audit)
Classical laws + terrain + powertrain are complete. The advanced physics all works but
carries self-admitted realtime approximations. **The entire report/ledger machinery is
explicitly NON-CALIBRATION** — it validates internal sign/convergence/parity, never a match
to measured tire data. Full detail in [`TIRE.md`](TIRE.md).

| Module | Status | Backend | Note |
|---|---|---|---|
| classical tires (Rigid/TmSimple/Pac89 Magic Formula + friction ellipse), terrain (flat/heightfield/trimesh/sdf), powertrain, shaft, sysid | complete | cpu | force-element tires (no GJK); solid + tested |
| SuspensionKind (RigidAxle/DoubleWishbone/MacPherson) | partial | cpu | now distinct per-kind wheel-rate / damping / bump-stop; camber/toe kinematics + solid-axle beam coupling still need a chassis roll DOF |
| vehicle 4-DOF chassis | partial | gpu+cpu | heave/long/lat/yaw only; **quasi-static massless unsprung** (no roll/pitch/wheel mass) |
| brush/bristle tire (+transient) | partial | gpu+cpu | validated internally (sign/camber/saturation) but non-calibration |
| legacy vehicle→brush adapter | experimental | cpu | evaluates on a **synthetic flat patch**; bridge for old scenes |
| carcass (shell-FEM ring) | partial | cpu | compact real dynamic model, CPU only |
| **TetraTireFem** (volumetric Neo-Hookean + Holzapfel fibers + cavity gas) | partial | gpu+cpu | real deforming FEM but **explicit/semi-implicit by default**; coarse viewer res |
| **matrix-free implicit Newton FEM** | experimental | gpu+cpu | validated CPU+GPU but **opt-in debug, not the runtime default**; no spin DOF |
| sidewall laminate / ply | partial | gpu+cpu | XPBD proxy, not a true laminated-shell stress solve |
| cavity / bead-rim seating | partial | gpu+cpu | polytropic proxy; rim-seat contact not validated |
| thermal + wear | experimental | gpu+cpu | hand-tuned layered multipliers, uncalibrated |
| Bekker deformable soil | partial | gpu+cpu | pragmatic 2.5-D; works for ruts/drawbar |
| DEM granular soil | partial | gpu+cpu | genuine DEM (angle of repose); coupling is first-pass |
| MPM particle-soil feedback | partial | gpu+cpu | tire→MPM bridge exists; broad scene ownership incomplete |
| rig/sweeps + parity ledger + governance/observability/realism + ~50 Report families | complete | cpu | huge internal-consistency apparatus; validates contracts, **not physical realism** |

### ML / diff / perception — autograd / actuators / sensors / optim
Functional, not stubbed. Two structural facts dominate.

| Module | Status | Backend | Note |
|---|---|---|---|
| autograd tape, diff_arm | complete | cpu | CPU-only **by design** (dynamic tape can't map to wgpu pipelines) |
| autograd featherstone_adjoint | partial | cpu | serial revolute chains only, box-inertia approx |
| autograd mpm/stam/xpbd adjoints | partial | cpu | each scoped (no plasticity / 2D-periodic / drops non-smooth constraints) |
| optim Adam / SGD | complete | cpu | GPU siblings live in `molla::kernels` |
| actuators: pd/pid/delay/clamping/mlp/lstm/orchestrator | complete–partial | **gpu-only** | ⚠ **nearly the whole crate is `#[cfg(gpu)]`** → CPU-only build exposes only `muscle`; MLP/LSTM are inference-only, fixed topology |
| sensors: lidar/frame_transform/dataset/camera_post/pointcloud | complete | gpu / cpu | ⚠ **all gpu-feature-gated** → CPU-only build exposes only `raycast_adjoint` |
| sensors: imu/raycast/contact/tiled_camera | partial | gpu | dynamic TriMesh skipped; camera does primitives only; contact reports **no forces** |
| sensors raycast_adjoint | partial | cpu | FD-only, sphere primitives only |

### Viewer — 135 scenes, test-enforced self-audit registry
Every scene is force-classified by an exhaustive `physics_audit()` match (`main.rs:2770`) with
two enforcement tests. The registry is accurate (agents cross-checked it). Distribution:
**~45 Complete, ~21 Partial, ~11 CpuOnly, ~9 ManualGpu, 0 Placeholder.**

| Group | Status | Note |
|---|---|---|
| core rigid/XPBD, sensors, articulation/actuator/IK, VBD cloth+soft, barrier PPF **cloth** gallery, working FEM PDE scenes, complete diffsim (DiffArm/Ball/Spring, SplatPoseRecovery) | complete | genuine GPU-backed demos with parity lanes |
| MPM family, MpmMultiMaterial, tire GPU FEM/brush/DEM (TireDynamics/Traction/Realistic), sensor H-series | experimental (ManualGpu) | real physics, validated **only via manual `make test-gpu`** — not in default `make test` |
| barrier PPF **rigid/rod** gallery, kinematic vehicle scenes, CPU-only FEM (AMR/magnetostatics/phase-field), tile N-body/MCGP, SolarSky | partial (CpuOnly) | real f64 CPU physics, **no GPU path** (against the GPU-first goal) |
| 21 "parity" scenes (diffsim GPU adjoints, NutBoltHydro, mixed/nonconforming FEM, IkCubeStacking, MpmBeamTwist…) | partial | reduced procedural stand-in for the named Newton/Warp reference |
| WarpFemTaylorGreen, WarpFemStreamlines | partial (honestly labeled) | run **NO solver** (closed-form field visualizations) — now UI-labeled "(analytic field)" so they no longer claim a FEM solve |
| **NutBoltHydro** | partial | headline hydroelastic physics **not yet coupled** into the live sim (reuses SDF contact) |

---

## The honest gap list (consolidated "what we don't have / half-have")

**GPU-first violations (CPU-only where the project wants GPU):**
- FEM: Nédélec/EM, signorini contact, AMR, phase-field, multiphase, adjoint, eddy-current — all CPU-only.
- Solvers: VBD CCD/tet-vs-tet/friction; sph_diff.
- Geometry: gaussian-splat adjoint (partly elsewhere). _(sdf_bake, self_contact, soft_ccd, marching_cubes, soft_bvh query all ported to GPU 2026-08-11 — the geometry CPU-only set is cleared.)_
- Viewer: the entire PPF rigid/rod gallery + CPU-only FEM band + tile N-body/MCGP.

**Named-but-inert (label-only):**
- ~~`SuspensionKind`~~ — **fixed 2026-08-11**: kinds now have distinct wheel-rate/damping/bump-stop physics.
- ~~`WarpFemTaylorGreen` / `WarpFemStreamlines` mislabel~~ — **fixed 2026-08-11**: relabeled "(analytic field)". They still run no solver (that would need wiring the real FEM Stokes path — see ROADMAP A5).

**Implemented-but-not-the-default — investigated 2026-08-11, NOT a defect pattern.**
All three instances are intentional, documented, publicly-reachable opt-in options; the
defaults are deliberate perf/stability choices and flipping them would be a regression:
- Tire matrix-free implicit Newton — reachable via `TetraTireFemSolverMode::MatrixFreeNewtonDebug`
  + `TetraTireFemParams::high_pressure_implicit()`; has dispatch + default-flip tests.
- `implicit_mpm` Full125 Hessian — reachable via the public `stencil` config field; tested in `tests/implicit_mpm.rs`.
- Barrier GPU Newton — already wired and selectable via `BarrierSolver::new_gpu`.
There is nothing broken/unreachable here to fix; the earlier "gap" framing was inaccurate.

**Scope-capped / approximate (documented):**
- compute-primitives GPU: block-Cholesky (N≤1024), stream-compaction (n≤1024), no GPU mcgp.
- CRBA/RNEA: revolute+prismatic only. FEM nonlinear: tet-only (Tri panics).
- `russell` "sparse" solver is a dense LU capped ~1024 dim.
- mesh_bvh_raycast_gpu: silently misses hits below stack depth 64.
- Hydroelastic, tire thermal/wear, sidewall, cavity-bead: real but uncalibrated approximations.

**Test-coverage gaps (code exists, in-crate parity does not):**
- molla-kernels: actuators, clamping, muscle, sensors, xpbd/integration, implicit_mpm,
  vbd_cloth, soft_vt, barrier — **no `tests/` in the kernels crate itself** (covered, if at
  all, only from the solver/actuator/sensor crates).

**Coupling / architecture debt:**
- `molla-kernels → molla-vehicle` inverted dependency (low-level depends on high-level).
- `molla-vehicle` is an island: re-implements `[f64;3]` math, not in the facade, no path into a molla-sim scene.
- `molla-vehicle` public surface ≈7× actual external usage; `real_tire/tests.rs` is one 9089-line file.
- Newest solver physics (Stages 39-51) has **zero viewer scenes** — invisible.

---

## Verification (how to check any of the above)

```
make check                 # fmt + clippy (viewer-scoped)
make test                  # CPU test suite
make test-gpu P=<pkg> T=<test> [A=<fn>]   # one GPU lane, via nixVulkan (real adapter)
make harden                # full gate: fmt + no-default-features + clippy -D warnings + all-feature tests
```

GPU tests are `#[ignore]`-gated (~245 of them) and **only** run through `make test-gpu` on
Vulkan hardware — never `cargo test --features gpu` (no adapter). This is why so much GPU
physics reads as "experimental / ManualGpu" here: it is real and passing, but its only guard
is a human running the lane.
