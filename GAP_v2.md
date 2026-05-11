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

molla today is **substantially feature-complete relative to Newton on the core engine surface** and **leading on FEM + soft contact + differentiable continuum mechanics**. Since the v2 doc was first written (the day Stage 36.5–37.8 closed), the **Tier-1 cluster shipped in full**: GPU sparse BSR algebra, adjoint SVD3, an `optim/` module (Adam + SGD), graph coloring, device-wide scan + 64-bit segmented radix sort. Plus a capstone: **Stage 38.1 — differentiable MPM** that chains `adj_svd3` + Adam to learn Young's modulus from a target state (FD oracle to 4.14e-6, 99.95% loss reduction in 75 Adam iters).

Three current takeaways:

1. **Tier-1 infrastructure is complete.** Every primitive flagged as "high-impact contained" in v2-v1 of this doc has shipped, hardened by a critic round (2 critical fixes), and is exercised by either parity tests or a viewer demo. molla now has the machinery for differentiable physics, parameter ID, and gradient-based learning **inside** its own engine.

2. **External-crate evaluation: `burn` rejected.** The user asked whether adopting `burn` would shortcut autodiff/optim work. A read-only agent evaluation concluded **no** — wgpu version mismatch (burn 29 vs molla 27), type-system shift (burn's `Tensor<B,D>` vs molla's `&[Real]`/`BsrMatrix`/`Vec<Mat3>`), and structural-vs-algebraic adjoint mismatch all argue against adoption. The hand-rolled stack is the right tool for "physics-shaped" parameters.

3. **Demo-side, the highest-ROI work is still mechanical.** Most of §6's quick-win viewer scenes are still unported. The most impactful unported demo is `softbody_hanging` — Newton's missing FEM demo, perfectly matched by molla's new GPU FEM steppers + Stage 38.1's diff-MPM forward.

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

## 3. The Tier-1 gaps — ✅ ALL SHIPPED (Stage 38)

The five primitives below shipped via parallel-agent push between 2026-05-11 ~16:00 and ~20:00. Plus one capstone (38.1) that chains 38b + 38c into a working demo.

### 3.1 GPU sparse BSR algebra — ✅ `e99d93f`
`crates/molla/src/math/sparse_gpu.rs` (1060 LOC) — `BsrGpu` + `BsrAlgebraKernel` with `matvec / axpy_same_pattern_unchecked / transpose / spgemm`. 4 parity tests vs CPU oracle pass at f32 round-off. **SpGEMM symbolic phase runs host-side; numeric phase on GPU** (full GPU SpGEMM with parallel hash deferred to v2).

### 3.2 Adjoint SVD3 + polar decomposition — ✅ `f3325b5`
`adj_svd3` + `adj_polar` in `crates/molla/src/math/dense3.rs` + WGSL adjoint kernel. FD oracle agreement: **1.5e-9 relative**, machine precision. Degenerate-σ Tikhonov regularization with `ε_cpu = 1e-12`, `ε_gpu = 1e-6`. GPU/CPU parity 3.2e-4 (f32 vs f64 dominates).

### 3.3 First-order optimizers (`optim/` module) — ✅ `0eaaabd`
New top-level `crates/molla/src/optim/` directory: `adam.rs` (AdamW-style decoupled WD) + `sgd.rs` (momentum + Nesterov + L2 WD) + 2 WGSL fused-update kernels. Convergence tests: Rosenbrock J=1.4e-14 in 5000 Adam steps; GPU/CPU parity 7.3e-6.

### 3.4 Graph coloring utility — ✅ `bba46e7` + wired `dc70cd2` + `4a559e2`
`crates/molla/src/geometry/graph_coloring.rs` — `greedy_color`, `mcs_color`, `color_constraints`, `group_by_color`. 4×4 cloth → optimal 4 colors via MCS. **38d.1 follow-up wired into `sim/builder.rs`**: replaced the ad-hoc `color_springs` + `color_particles_with_tets_and_bending` greedy with calls to the new utility. Net **−66 LOC**, single source of truth, all VBD/XPBD tests still green.

### 3.5 Device-wide scan + 64-bit segmented radix sort — ✅ `bb17a41`
`crates/molla/src/kernels/scan.rs` (3-pass scan-then-fixup) + `RadixSortKernel64` in extended `radix_sort.rs`. 1M-key sort: **2.7× faster than CPU** (146 ms vs 390 ms). 8M-element scan: 29.7 ms vs CPU 36.6 ms. Segmented sort via segment-tag prefix encoding, max 65,536 segments.

### 3.6 Capstone — Differentiable MPM — ✅ `6934075` (Stage 38.1)
`crates/molla/src/compute/mpm_cpu_diff.rs` (764 LOC) + `crates/molla/src/autograd/mpm_adjoint.rs` (880 LOC). Chains `adj_svd3` (Stage 38b) through P2G + grid update + G2P + co-rotated Saint-Venant Kirchhoff stress; Adam (Stage 38c) optimizes Young's modulus to match a target final state. **Result: E=6e4 → 1.6108e4 (target 1.5e4) in 75 Adam iters, 7.4% rel err, 99.95% loss reduction. FD oracle agrees to 4.14e-6.** Wall: 18 ms/iter.

### 3.7 Critic round on Stage 38 — ✅ `90f3708`
Background agent audited all 5 slices. **Zero critical math bugs.** 2 critical safety/contract issues fixed:
- `BsrAlgebraKernel::axpy` renamed `axpy_same_pattern_unchecked` with explicit safety doc (previous doc claimed pattern verification that wasn't happening).
- `segmented_sort_pairs_u64` validates `segment_offsets` monotonicity + total before host-side indexing.
- Plus: Adam `step==0` assertion, `BsrGpu::upload` shape consistency check, `ScanKernel` concurrency doc.

### 3.8 Adam showcase scene #83 — ✅ `08e7b04`
`AdamLandscape` viewer scene — 2D Rosenbrock landscape rendered as height-field, Adam trajectory drawn as yellow line + markers, green target marker at `(1, 1)`, pane exposes lr/β₁/β₂/wd/init.

### 3.9 External-crate evaluation: `burn` — ❌ REJECTED
Critical background agent evaluated using the `burn` crate (Rust ML framework with wgpu backend) for autodiff + optim. **Verdict: do not adopt.** Hard block: burn pins wgpu 29 via cubecl, molla pins wgpu 27 (Bevy coupling). Even unblocked: burn-autodiff operates on its own `Tensor<B, D>` type, not on molla's `&[Real]` / `BsrMatrix` / `Vec<Mat3>` — would add a *second* autograd system, not replace the existing one. Dep tree growth ~5-10×. Hand-rolled stack (Adam + SGD + structural adjoints) is the right tool for molla's "physics-shaped" parameters.

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

**Recommendation:** Kamino + Style3D should stay out-of-scope per PLAN.md unless a concrete user need surfaces. **Implicit MPM is now the most attractive Tier-3 item** because Stage 38 shipped its three building blocks: SpGEMM (38a), graph coloring (38d), device-wide scan + 64-bit sort (38e). Newton's `implicit_mpm/` is essentially "build the rheology Schur system + color-block Gauss-Seidel + project" — most of which is one-step-away from molla's existing MPM + the new primitives. Diff Gaussian raytrace remains the long-term gateway to learned-perception losses but is a 2-3 week project.

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

## 8. Stage 38 — ✅ SHIPPED. Next-stage candidates.

Stage 38 (the GPU sparse algebra + optimization toolkit + diff MPM capstone) is **fully closed out**. See §3 for the slice-by-slice landing record.

### What to push next (ranked)

**Option A — Stage 39: Implicit MPM with rheology.** Now genuinely tractable because Stage 38 shipped the three primitives that block it (SpGEMM 38a, graph coloring 38d, scan + 64-bit segmented sort 38e). Plus, Stage 38.1 wrote a differentiable forward MPM that can serve as the host-side scaffold. Newton's `implicit_mpm/` is the reference: Schur-system assembly + color-block Gauss-Seidel + projection. Estimated 1-2 weeks. **Highest impact**: gives molla stable plasticity at large dt, which the explicit base can't deliver.

**Option B — Quick-win viewer scenes (§6).** 10 demos worth ~1 day of mechanical work as a focused mini-pass. Highest single payoff is `softbody_hanging` (Newton's missing FEM demo — and now mostly an MPM-like elasticity case that pairs naturally with the new `mpm_cpu_diff`).

**Option C — Stage 39 — 2D Stam stable fluid + DG basis.** Unblocks **6** missing Warp demos (1 Stam → 3 demos; 1 DG basis → 3 demos). Tier-2 work. Estimated 1-3 days each. Lower compute / research interest than implicit MPM but big surface-area win on the demo gap.

**Option D — Differentiable FEM via Adam + adjoint chain.** Mirror Stage 38.1 (diff MPM) on the FEM side: chain `fem::adjoint::solve_adjoint` through Adam to learn material parameters in a tet-mesh elasticity problem. molla already has `fem/adjoint.rs`; missing is the Adam loop wrapper + a viewer demo. Estimated 1-2 days.

### Recommended ordering

1. **Now:** ship Option D (~1 day) — closes the FEM differentiability story mirroring Stage 38.1's MPM win.
2. **Next:** Option B mini-pass (~1 day) — clears the viewer gap left over from GAP_v2 §6.
3. **Then:** Option A — implicit MPM as a focused multi-day push.
4. **Last:** Option C — the 2D Stam fluid + DG basis (Tier-2). Lower priority than the others.

The user pattern from the recent session is "spin parallel agents in worktrees for non-overlapping work." Option A is the natural next parallel-agent target (subdivides nicely: Schur assembly, color-block GS kernel, projection step, integration into existing MPM forward, viewer scene).

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
- **Stage 38 commits (post-GAP_v2 v1):** `bba46e7` (graph coloring), `e99d93f` (BSR algebra), `f3325b5` (adj_svd3), `bb17a41` (scan + radix-64), `0eaaabd` (Adam + SGD), `08e7b04` (AdamLandscape scene #83), `90f3708` (critic fixes), `dc70cd2` + `4a559e2` (wire coloring into builder, Stage 38d.1), `6934075` (Stage 38.1 — differentiable MPM)
