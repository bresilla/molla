# GAP analysis v2 — molla vs Warp + Newton (post-FEM-GPU push)

**Refreshed:** 2026-05-11 (after Stage 36.5–37.8 family completion).
**Method:** 5 independent parallel read-only agent scans:
  - Warp examples (2 agents): `xtra/warp/warp/examples/` files vs molla viewer scenes
  - Warp library (1 agent): `xtra/warp/warp/{fem,sparse,_src,native,optim}` vs molla compute/kernels
  - Newton solvers (1 agent): `xtra/newton/newton/_src/{solvers,sim,actuators,sensors,ik,geometry}` vs molla physics
  - Newton examples (1 agent): Newton demo tree vs molla viewer scenes
  - molla baseline (1 agent): full capability inventory + coverage matrix
**Predecessor:** `GAP.md` (dated 2026-05-09) drove Stages 14–24e.2. **This document supersedes it.**

---

## 0. Executive summary

molla today is **substantially feature-complete relative to Newton on the core engine surface** and **leading on FEM + soft contact**. The recent FEM-GPU push (Stages 36.5–37.8) added persistent GPU steppers, Jacobi-preconditioned resident CG, a direct-solver bridge, and 35 new viewer scenes covering Warp's FEM/MPM/tile primitive demos.

Three takeaways:

1. **Engine parity is high.** All Newton core solvers ship (Featherstone, XPBD, VBD, MPM); all 5 Newton sensor types ship (raycast, IMU, contact, frame_transform, tiled_camera); all 7 actuator pipeline pieces ship (PD/PID/Neural MLP/LSTM + 3 clamping models + delay). Several molla areas (FEM, soft CCD, contact pipeline) **exceed** Newton.

2. **The real remaining gaps are infrastructure primitives, not user-facing physics.** GPU sparse algebra (`bsr_mm`/`bsr_axpy`/`bsr_transpose`), differentiable SVD3 adjoint, an `optim/` module (Adam/SGD), graph coloring, and a device-wide scan are the bottlenecks blocking the next layer of capability.

3. **Demo-side, the highest-ROI work is mechanical.** ~10 viewer scenes can be wired with primitives molla already ships, including a long-overdue showcase for the new GPU FEM steppers.

---

## 1. Where molla leads

Three axes where molla has more than either upstream:

### 1.1 FEM stack
Newton ships no first-class FEM examples; Warp's `wp.fem` is a Python-level DSL atop manual integrators. molla has:
- P1 + P2 shape functions on triangles + tets, with quadrature rules up to 11-pt
- Symmetric Dirichlet reduction (column-elim, not just row-pin)
- Mass + stiffness + wave + convection + SUPG + distortion + elasticity + St. Venant–Kirchhoff stress element matrices
- Time-stepping: backward Euler (heat), Newmark-β (wave), implicit-Euler IMEX (NS) with SUPG-stabilized convection
- Mixed Taylor-Hood + Uzawa-CG Stokes / NS
- FFT-Poisson NS coupling (Stage 37.5) with wide-stencil correctness
- APIC FEM-MPM hybrid (Stage 37.6)
- Adjoint via stiffness-transpose
- **GPU port (Stage 36.5–37.8)**: P1 tet+tri stiffness/mass kernels, resident CG with cached bind groups (3.1× speedup), PCG (~30% iter cut), persistent steppers for wave + transient, direct-solver bridge under `russell` feature

Coverage: 14 viewer scenes (47–51 native + 67–80 Warp-style ports + 78 WarpWave with CPU/GPU toggle).

### 1.2 Soft contact
Stage 33.x family shipped a complete cross-mesh soft-contact pipeline that Newton does not have:
- Vertex–triangle (VT) + edge–edge (EE) contact
- Self-collision (single-mesh) + cross-mesh
- Continuous collision detection (CCD)
- Friction + restitution on GPU
- Dedicated `SoftBvhGpu` for soft-mesh broadphase

5 viewer scenes (43–46) exercise the full pipeline.

### 1.3 Articulation kernel split
Newton bundles CRBA/RNEA/Jacobian into helper functions. molla exposes each as a standalone WGSL kernel (`kernels::crba`, `kernels::rnea`, `kernels::jacobian`) — cleaner for caller composition + GPU IK + adjoint paths.

---

## 2. Coverage by phenomenon

| Phenomenon | CPU | GPU | Tests | Viewer scene(s) | Status vs Newton/Warp |
|------------|-----|-----|-------|-----------------|------------------------|
| Rigid bodies (semi-implicit) | ✓ | ✓ | ✓ | 01 | parity |
| XPBD rigid contact | ✓ | ✓ | ✓ | 01–03, 30, 42 | parity |
| Articulated (Featherstone ABA) | ✓ | ✓ | ✓ | 08, 14, 26–27, 37–38 | parity |
| CRBA / RNEA / Jacobian | ✓ | ✓ | ✓ | — | **ahead** (cleaner split) |
| Multi-DOF joints (Ball/D6/Distance) | ✓ | ✓ | ✓ | 14, 27 | parity |
| Joint mimic (quartic coupling) | ✓ | ✓ | ✓ | 34, 39 | partial (data wired, full coupler tbd) |
| CABLE joint type | — | — | — | 23, 33, 34 use Distance | **missing** (have DAHL kernel) |
| IK (LM + L-BFGS) | ✓ | ✓ | ✓ | 10 | partial (no Roberts seeding) |
| Cloth XPBD | ✓ | ✓ | ✓ | 04, 31–32 | parity |
| Cloth VBD + AVBD | ✓ | ✓ | ✓ | 16, 43–44 | parity |
| Style3D cloth (projective) | — | — | — | — | **missing** (explicit choice per PLAN.md) |
| Cable VBD + Dahl friction | ✓ | ✓ | ✓ | 02, 23, 33–34 | parity |
| Soft body (VBD tets) | ✓ | ✓ | ✓ | 25, 43 | parity |
| Soft-vs-soft (VT+EE+CCD+self) | ✓ | ✓ | ✓ | 43–46 | **ahead** |
| MPM (sand/snow/two-way) — explicit | ✓ | ✓ | ✓ | 24, 35–36 | parity (explicit) |
| MPM implicit + rheology | — | — | — | — | **missing** (Newton's `implicit_mpm`) |
| FEM Laplace/heat/wave | ✓ | ✓ | ✓ | 47, 49, 78 | **ahead** (Newton has none) |
| FEM elasticity + distortion | ✓ | partial | ✓ | 48, 51, 74–75, 77 | **ahead** |
| FEM Stokes / NS | ✓ | partial | ✓ | 50, 69–72, 76 | **ahead** |
| FEM–MPM APIC | ✓ | partial | ✓ | 79 | **ahead** |
| Hydroelastic contact | ✓ | — | ✓ | — | partial (basic shipped, GPU + anchor-contact missing) |
| Sparse-grid (NanoVDB-like) | ✓ | ✓ | ✓ | 58 | partial (no `volume_sample_grad`) |
| Marching cubes / SDF bake | ✓ | — | ✓ | 57 | partial (CPU only; Warp has GPU MC) |
| Mesh decimation / hull | ✓ | — | ✓ | 20, 54 | parity |
| FFT (Stockham radix-2) | ✓ | ✓ | ✓ | 62, 76 | parity |
| Featherstone reverse adjoint | ✓ | ✓ | ✓ | 19 | parity |
| XPBD / VBD autograd | ✓ | partial | ✓ | 18, 40–41 | partial (Newton has frozen-narrow-phase pattern) |
| Raycast adjoint | ✓ FD | — | ✓ | 80 | partial (analytic adjoint deferred) |
| Tiled camera (5-ch) | — | ✓ | ✓ | 12 | parity |
| Differentiable Gaussian raytrace | — | — | — | — | **missing** (Newton has it; no molla equivalent) |
| Sensors (raycast, IMU, contact, frame, camera) | — | ✓ | ✓ | 05, 06, 07, 12 | parity |
| MuJoCo external solver | ✓ (FFI) | — | ✓ | 17 | parity |
| Kamino PA-ADMM (closed-loop) | — | — | — | — | **missing** (explicit OOS per PLAN.md) |
| Actuator pipeline (PD/PID/MLP/LSTM/DC motor/delay) | ✓ | ✓ | ✓ | 13, 21–22 | parity |
| Tile primitives (matmul/Cholesky/conv/compact/MCGP/filter) | — | ✓ | ✓ | 63–65, 81–82 | parity |

**Tally:** 13 ✓ parity, 8 partial, 6 missing (4 explicit OOS), 4 **ahead**.

---

## 3. The Tier-1 gaps — high-impact, contained

These five primitives are each ~1-3 days of focused work and individually unblock multiple downstream demos / capabilities.

### 3.1 GPU sparse BSR algebra
**What's missing:** `bsr_mm` (SpGEMM), `bsr_axpy` (sparse `A += α B` with pattern union), `bsr_transpose`, WGSL `bsr_mv`.
**Why it matters:** Currently molla assembles K on GPU then pulls it host-side for any non-matvec operation. Blocks: scaling FEM beyond what fits on one host, implicit MPM, optimization pipelines that need `Aᵀ A`.
**Suggested module:** `crates/molla/src/math/sparse_gpu.rs` + `shaders/bsr_mm.wgsl`, `shaders/bsr_axpy.wgsl`, `shaders/bsr_transpose.wgsl`.
**Reference:** `xtra/warp/warp/_src/sparse.py` — Warp ships these end-to-end.

### 3.2 Adjoint SVD3 + polar decomposition
**What's missing:** `adj_svd3` (reverse-mode through 3×3 SVD), polar-decomposition adjoint.
**Why it matters:** Forward `svd3` is already in `crates/molla/src/math/dense3.rs` and `crates/molla/src/kernels/dense3.rs`. The adjoint is **the** missing piece for differentiable MPM (co-rotated elasticity), differentiable FEM hyperelasticity, neural material learning.
**Suggested module:** add `adj_svd3` to `math/dense3.rs` + WGSL adjoint kernel. Pair with #3.3.
**Reference:** `xtra/warp/warp/native/svd.h::adj_svd3`.

### 3.3 First-order optimizers (`optim/` module)
**What's missing:** molla has **no** `optim/` directory. No Adam, no SGD, no momentum/weight-decay helpers.
**Why it matters:** Blocks differentiable-physics training, parameter ID, policy learning examples. Warp's `optim/sgd.py` + `optim/adam.py` total ~150 LOC each.
**Suggested module:** new top-level `crates/molla/src/optim/` directory; `adam.rs`, `sgd.rs`, with WGSL fused-update kernels for `fp32`/`vec3` weight tensors.
**Reference:** `xtra/warp/warp/optim/{adam,sgd}.py`.

### 3.4 Graph coloring utility
**What's missing:** Centralized greedy + MCS coloring helper.
**Why it matters:** `solvers::vbd` and XPBD distance-constraint kernels each reinvent it ad-hoc. Centralizing unlocks colored Gauss-Seidel for parallel constraint relaxation (cleanup) and scales VBD beyond current limits.
**Suggested module:** `crates/molla/src/geometry/graph_coloring.rs` (host-side, ~150 LOC).
**Reference:** `xtra/warp/warp/_src/fem/utils.py::graph_coloring_assign`.

### 3.5 Device-wide scan + 64-bit segmented radix sort
**What's missing:** General GPU inclusive/exclusive scan; 64-bit segmented radix-sort variant.
**Why it matters:** Building blocks under everything else — sparse-grid insertion, hash-grid bucket counts, contact pair compaction, future SpGEMM. Today molla has only tile-local Hillis-Steele scan (`tile_stream_compaction`) and 32-bit pair-only radix sort.
**Suggested module:** `crates/molla/src/kernels/scan.rs` + WGSL; extend `kernels/radix_sort.rs` to 64-bit + segmented mode.

**Path:** spinning 5 parallel agents (one per Tier-1 gap) would ship the whole cluster in ~1 work-week. Same execution pattern as the recent FEM-GPU push.

---

## 4. Tier-2 gaps — substantial new capability (multi-day each)

### 4.1 2D Stam stable fluid
**File:** `xtra/warp/warp/examples/core/example_fluid.py`.
**Why:** Unblocks 3 demos in one shot: `fft_poisson_navier_stokes_2d`, `optim/fluid_checkpoint`, `optim/navier_stokes_perturbation`.
**Effort:** ~1 day. molla has FFT-Poisson + advection helpers; this is wiring + a viewer scene.
**New scene number:** #83 WarpStableFluid.

### 4.2 DG (Discontinuous Galerkin) basis
**Files:** `example_burgers.py`, `example_convection_diffusion_dg.py`, `example_shallow_water.py`.
**Why:** All three share the same missing primitive — a DG basis in molla FEM. molla has CG (continuous Galerkin) only.
**Effort:** ~3 days (basis + flux assembly + slope limiter for Burgers).
**Suggested module:** extend `crates/molla/src/fem/function_space.rs` with `FunctionSpace::DGNodal { degree }` + DG-specific element matrices in `fem/dg_element.rs`.

### 4.3 Hydroelastic contacts (Drake-style)
**Files:** `xtra/newton/newton/examples/robot/example_robot_panda_hydro.py`, `contacts/example_nut_bolt_hydro.py`.
**Why:** molla has basic hydroelastic in `crates/molla/src/geometry/hydroelastic.rs` (Stage 22.1) but lacks anchor-contact for moment balance + aggregate-stiffness reduction.
**Effort:** ~3 days.
**Status:** marked PARTIAL in critic finding from Stage 33 review.

### 4.4 CABLE joint type + tendons/muscles
**Files:** Newton's `JointType.CABLE` (1 linear stretch + 1 isotropic bend/twist DOF), `muscle_*` arrays in `sim/model.py`, `kernels_muscle.py`.
**Why:** molla has DAHL friction kernel + cable distance constraints but no first-class cable joint type or muscle actuation. Blocks tendon-driven robotics + biomechanics demos.
**Effort:** ~2 days for joint type + integration; ~3 more for muscles.

### 4.5 PCG SpGEMM-on-GPU
**Why:** Builds on Tier-1 #1. Future-proofs Stokes/NS Schur complement + future implicit MPM. Required for any Krylov-style preconditioner that needs `Aᵀ A` (e.g. CGNR for non-symmetric systems).
**Effort:** ~2 days once Tier-1 #1 ships.

---

## 5. Tier-3 gaps — genuinely new paradigms (weeks each)

These are flagged for completeness; some are explicitly out-of-scope per `PLAN.md`.

| Gap | Source | Effort | PLAN.md status |
|-----|--------|--------|----------------|
| **Kamino PA-ADMM** (closed-loop kinematics + hard contact + spatial friction) | Newton `solvers/kamino/` | weeks | **explicit OOS** (Stage 25+) |
| **Style3D cloth** (anisotropic bending, panel sewing, dedicated BVH) | Newton `solvers/style3d/` | weeks | **explicit OOS** (Stage 20 picked VBD) |
| **Implicit MPM with rheology** (color-block GS, yield-parameter integration, rasterized rigid coupling) | Newton `solvers/implicit_mpm/` | 1-2 weeks | partial (Stage 24d.4 deferred) |
| **Differentiable Gaussian raytrace renderer** (3DGS in-the-loop) | Newton `sensors/warp_raytrace/` | 2-3 weeks | not on roadmap |
| **Adaptive grid FEM** (AMR refinement near collider) | Warp `example_adaptive_grid.py` | 1-2 weeks | not on roadmap |
| **H(curl) / Nedelec vector FEM elements** (magnetostatics) | Warp `example_magnetostatics.py` | 1 week | not on roadmap |
| **NonconformingField** (elastic contact between non-matching meshes) | Warp `example_nonconforming_contact.py` | 1-2 weeks | not on roadmap |

**Recommendation:** Kamino + Style3D should stay out-of-scope per PLAN.md unless a concrete user need surfaces. Implicit MPM is the most useful to revisit (Stage 24d.4) since the explicit base shipped. Diff Gaussian raytrace is the gateway to learned-perception losses but is a substantial multi-week project.

---

## 6. Quick wins — mechanical viewer scenes (~100 LOC each)

10 demos using primitives molla already ships. Combined effort: ~1 day if done as a focused mini-pass.

### 6.1 Newton demos with existing molla primitives

1. **`cloth_poker_cards`** — 52 stiff "cards" stacking and getting knocked off. Pure VBD cloth stress test, zero new primitives, spectacular visual.
2. **`softbody_hanging`** — 4 tet grids with damping sweep. **Best showcase for the new GPU FEM steppers** that has nothing in Newton or molla viewer yet.
3. **`cable_twist`** — spin one cable endpoint, watch twist propagate across 90° bends. Exercises molla's CONNECT topology + bend stiffness.
4. **`ik_custom`** — Franka IK with collision-avoidance sphere objective on existing `GpuIkArm` scene.
5. **`diffsim_drone`** — MPC over existing `Quadrotor` scene + autograd. Headline-grade demo connecting molla's autograd + multi-DOF Featherstone.

### 6.2 Warp demos with existing molla primitives

6. **`tile_matmul` standalone scene** — base GEMM primitive, currently buried inside `WarpTileMlp`.
7. **`tile_block_cholesky` variant** — different blocking pattern than current `WarpTileCholesky`.
8. **`tile_convolution` standalone** — separable convolution; currently buried inside `WarpTileFilter`.
9. **`tile_mcgp` (walk-on-spheres)** — Monte Carlo Laplace via `wp.Mesh`. molla has both primitives.
10. **`fft_poisson_navier_stokes_2d`** — reuse tile FFT (#62) + 2D scalar grid viz; new compute kernel only.

---

## 7. What shipped since the original `GAP.md`

`GAP.md` was dated 2026-05-09 and drove Stages 14–24e.2. Since then, the following major work has landed:

### 7.1 Stage 33 family (soft contact v2)
Cross-mesh VT + EE + self-collision + CCD + friction + GPU port.

### 7.2 Stage 36 family (FEM framework)
- v1 P1 nodal CPU
- 36.1 elasticity (linear + SVK)
- 36.2 Stokes + NS (Taylor-Hood, Uzawa-CG)
- 36.3 P2 (higher-order elements)
- 36.4 adjoint (reverse-mode through FEM)
- 36.5 GPU port (resident CG, persistent steppers, PCG)
- 36.6 Gmsh import via `mshio`
- 36.7 direct solver bridge (`russell` feature, dense LU fallback)

### 7.3 Stage 37 family (Phase 2 Warp engine work)
- 37.1 Wave PDE element matrix
- 37.2 Convection element + upwinding
- 37.3 NS time-stepping
- 37.4 Distortion-energy Newton minimization
- 37.5 FFT pressure-Poisson NS coupling
- 37.6 APIC FEM-MPM hybrid
- 37.7 Raycast adjoint (FD)
- 37.8 Tile compute primitives bundle

### 7.4 Hardening + perf
- Symmetric Dirichlet reduction in `apply_bcs`
- SUPG stabilization wired into NS
- Newton divergence detection
- Russell singularity tolerance (relative pivot)
- GPU CG bind-group caching (3.1× speedup)
- Jacobi PCG everywhere (~30% iter cut)
- Wide-stencil Poisson symbol for fractional-step NS
- 2D matvec dispatch grid (>65K rows)
- `BsrMatrix::from_scalar_triplets` (~2.2× faster assembly at 16³)
- 35 new viewer scenes (47–82)

### 7.5 Test + bench expansion
- 274 lib tests (was ~200)
- 100 integration test files
- 41 bench files including CPU-vs-GPU paired benches
- Full Stage 36.5–37.8 critic round + fix wave

---

## 8. Recommended next-stage plan — Stage 38

If picking a single coherent next push: **Stage 38 — GPU sparse algebra + optimization toolkit.**

Scope (Tier-1 cluster, ~1 work-week):
1. GPU sparse BSR algebra (`bsr_mm`, `bsr_axpy`, `bsr_transpose`, WGSL `bsr_mv`)
2. Adjoint SVD3 (`adj_svd3` + polar adjoint)
3. `optim/` module (Adam + SGD + WGSL fused-update kernels)
4. Graph coloring utility
5. Device-wide scan + 64-bit segmented radix sort

**Deliverable signature:**
- New top-level `crates/molla/src/optim/` directory
- 4 new shaders: `bsr_mm.wgsl`, `bsr_axpy.wgsl`, `bsr_transpose.wgsl`, `scan.wgsl`
- Extended `crates/molla/src/math/sparse.rs` + `kernels/{radix_sort,scan}.rs`
- Adjoint SVD3 in `math/dense3.rs` + matching WGSL kernel
- `crates/molla/src/geometry/graph_coloring.rs`
- Parity tests against CPU oracle for each new primitive
- Bench files showing speedup vs CPU equivalents

**Unlocks:**
- Differentiable MPM (Tier-1 #2 closes the chain)
- Implicit MPM with rheology (Tier-3 #3 becomes tractable)
- Sub-stage 38.1+ family: differentiable FEM via Adam, neural-material learning, parameter ID

**Execution pattern:** spin 5 parallel agents in worktrees, one per Tier-1 gap. Files are non-overlapping. Same model as the Stage 37 family that just shipped.

**Concurrent:** the 10 Tier-1.5 viewer scenes (§6) as a parallel mini-pass — mechanical work, ~1 day total, mostly viewer plumbing.

---

## 9. Out-of-scope (explicit)

Per `PLAN.md` Stage 25+ section, these are deliberately NOT on the roadmap regardless of effort:

- **Python bindings** (`pyo3`) — Newton serves Python users; molla serves Rust
- **TensorCore tile matmul** — wgpu doesn't expose `cooperative_matrix` portably
- **CUDA graphs** for command batching — no wgpu equivalent
- **Multi-GPU peer access** — wgpu is single-device
- **Kamino PA-ADMM** — overlaps MuJoCo wrapper for the constrained-dynamics niche

Both **Style3D cloth** and **Kamino** are out-of-scope choices, not gaps that need closing. Implicit MPM and adaptive-grid FEM are deferred (not OOS) and can be revisited if a concrete user need surfaces.

---

## 10. Cross-reference index

- **Source agent reports:** see `~/.claude/projects/.../tasks/{ac8c4a89c564f28b0,ae5be8a52549139f1,a962ce9cd82fa430b,a79edaff1920966e0,a41c1097431b92691}.output`
- **molla baseline files:**
  - Solvers: `crates/molla/src/solvers/mod.rs`
  - Kernels: `crates/molla/src/kernels/mod.rs`
  - FEM: `crates/molla/src/fem/{solver.rs,mod.rs}`
  - Geometry: `crates/molla/src/geometry/mod.rs`
  - Viewer scenes: `crates/molla_viewer/src/main.rs` (`Scene::ALL` @ L1175; labels @ L1295)
  - Roadmap: `PLAN.md` (stage list @ L1124+)
- **Predecessor:** `GAP.md` (2026-05-09; drove Stages 14–24e.2)
- **Recent commits driving this update:** `8dd1fe5` (Slice E), `8b27d24` (Slice F), `479387d` (Slice G), `c509427` (Slice A), `c0d737e` (Slice D), `65f90ec` (combined Slice H + russell + assembly perf), `ddf017f` (PCG), `3fc3d9b` (steppers use PCG), `0af1727` (CPU PCG paths), `9048c7f` (russell singularity), `50c3923` + `70377d0` (tile filter + compaction scenes)
