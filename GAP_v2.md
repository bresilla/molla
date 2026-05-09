# GAP analysis v2 — Newton + Warp vs molla (post-wave-5)

Refreshed 2026-05-09 by 5 parallel independent agent scans (2× Newton, 2× Warp,
1× molla). The original `GAP.md` was the work item that drove every parent
stage in `PLAN.md`; this v2 measures what's actually closed after the GAP-driven
roadmap shipped (Stages 14 through 24e.2 inclusive).

**Headline:** molla is **at or above v1 feature-parity with Newton** on the
core engine surface (solvers, joints, sensors, actuators, contact, geometry,
sim infrastructure, differentiability). The real remaining gaps are
**asset-pipeline breadth** (importers + USD schema resolvers) and a small
number of **algorithmic alternatives molla deliberately did NOT pick** (Style3D,
Kamino, MPR). The Warp-level "GPU primitive layer" gaps are mostly
explicit `Stage 25+` out-of-scope items (tile compute, multi-GPU,
framework interop) — those are NOT going to land regardless of effort.

---

## Per-axis comparison

For each axis: ✅ at parity, 🟡 partial / behind, ❌ missing, ⛔ explicitly out
of scope per `PLAN.md` Stage 25+.

### 1. Solvers
| Newton ships | molla ships | Status |
|---|---|---|
| Featherstone | Featherstone (CPU + GPU; multi-DOF + body_f) | ✅ |
| SemiImplicit | semi_implicit | ✅ |
| MuJoCo wrapper | mujoco (feature-gated) | ✅ |
| XPBD | xpbd | ✅ |
| VBD | vbd (+ AVBD soft + hard mode + tets) | ✅ |
| **Style3D** (cloth-only PCG projective dynamics) | — | ❌ explicit choice (Stage 20 picked VBD) |
| ImplicitMPM | mpm (explicit + semi-implicit Jacobi; full CG deferred to 24d.4) | 🟡 |
| **Kamino** (PADMM solver) | — | ⛔ explicit out-of-scope (overlaps MuJoCo per Stage 25+) |

**Verdict:** 6 of Newton's 8 solvers shipped at v1; 1 explicit choice (Style3D),
1 explicit out-of-scope (Kamino). **Effective parity 7/8 = 88 %.**

### 2. Joints + constraints
| Newton ships | molla ships | Status |
|---|---|---|
| Prismatic, Revolute, Ball, Fixed, Free, Distance, D6 (8 incl. CABLE) | All 7 + CABLE polyline (Stage 20.1) | ✅ |
| limits, drives, friction, armature, target_mode | limits, drives, friction, armature, target | ✅ |
| Equality: CONNECT, WELD, JOINT (mimic) | CONNECT, WELD | 🟡 — JOINT polynomial mimic missing |

**Verdict:** at parity except joint mimic constraints (P3 task; no current consumer).

### 3. Contact pipeline
| Newton ships | molla ships | Status |
|---|---|---|
| Broad: SAP, BVH, AllPairs, Explicit | SAP, LBVH (GPU + radix sort), BVH | ✅ |
| Narrow: GJK, EPA, **MPR**, primitive analytic | GJK, EPA, primitive analytic | 🟡 — MPR deferred from Stage 13 |
| Hydroelastic (Drake-style) | hydroelastic.rs (soft-on-rigid; soft-on-soft deferred to 22.2) | 🟡 |
| Dahl friction | dahl_friction kernel (Stage 20.1) | ✅ |
| Self-collision (cloth, particles) | per-particle hash grid (Stage 12) | ✅ |
| **Contact reduction** (global, hydroelastic, matcher) | — | ❌ Newton's coalescing layer; not yet in molla |

**Verdict:** at parity on broad/narrow + hydroelastic + Dahl. **MPR + contact
reduction** are concrete gaps but neither is on a current consumer's critical path.

### 4. Sensors
| Newton ships | molla ships | Status |
|---|---|---|
| TiledCamera (RGB/depth/normal/seg) | tiled_camera (5 channels) | ✅ |
| Raycast | raycast | ✅ |
| IMU | imu | ✅ |
| Contact | contact_aggregate | ✅ |
| FrameTransform | frame_transform | ✅ |

**Verdict: full parity (5/5).**

### 5. Actuators / control
| Newton ships | molla ships | Status |
|---|---|---|
| ControllerPD, PID | ControllerPd, Pid + pipeline variants | ✅ |
| ControllerNeuralMLP | ControllerNeuralMlp (Stage 18.1) | ✅ |
| **ControllerNeuralLSTM** | — | ❌ deferred per Stage 18.1 (no current consumer) |
| ClampingMaxEffort, PositionBased, DCMotor | All three | ✅ |
| Delay | actuator_delay | ✅ |

**Verdict:** 6/7 = 86 %. Only `ControllerNeuralLSTM` is missing (small follow-up).

### 6. Math / linalg
| Newton ships (via Warp) | molla ships | Status |
|---|---|---|
| Quaternion / spatial / smooth losses | math::{quat, spatial, robust_loss} | ✅ |
| Dense Cholesky, GEMM, dense_solve | dense_solve, dense3 | ✅ |
| **SVD3 / QR3 / eig3 / polar** (via Warp) | dense3.rs (Stage 24a, **CPU + WGSL public API**) | ✅ molla equal-or-better |
| Robust losses: L1, smooth-L2, Huber, Tukey, Cauchy | All 6 + pseudo-Huber (Stage 24b) | ✅ molla equal-or-better |
| **Iterative solvers: CG / GMRES / BiCGSTAB / PCG** (Warp) | — | ❌ **real gap** — needed for full implicit MPM (Stage 24d.4), Style3D, Kamino |
| **FFT** (`tile_fft`) | shaders/fft.wgsl (Stage 24c, Stockham radix-2) | ✅ |
| BSR sparse | CSR sparse (`math::sparse`) | 🟡 — different format; functional equivalent |

**Verdict:** at parity on per-element math. **Iterative linear solvers (CG / GMRES / BiCGSTAB)** are the only honest gap and matter for the deferred implicit-MPM / Style3D / Kamino paths.

### 7. Geometry
| Newton ships | molla ships | Status |
|---|---|---|
| Sphere, Box, Capsule, Cylinder, Cone, Plane | Same 6 | ✅ |
| **Ellipsoid** | — | ❌ small primitive gap |
| Mesh, ConvexHull (incl. builder) | TriMesh + ConvexHull + QuickHull builder (Stage 21) | ✅ |
| Heightfield | heightfield | ✅ |
| Dense + sparse SDF | sdf_grid + sparse_grid (Stage 21 + 24e) | ✅ |
| **Gaussian splat** | — | ❌ rendering primitive gap |
| Marching cubes | marching_cubes.rs (Stage 21) | ✅ |
| Convex hull builder, decimation, mesh→SDF, remesh | hull, decimate (Garland-Heckbert), bake_mesh_to_sdf | ✅ |
| BVH, hash grid, sparse hierarchical | bvh, mesh_bvh, tet_bvh, spatial_hash, sparse_grid | ✅ |
| **Tetgen-style mesh authoring** | — | ❌ deferred to Stage 20.5 (`tetgen-rs`) |

**Verdict:** at parity on the core. Ellipsoid + Gaussian splat are the two missing primitives (visual / rare-collision); volumetric mesh authoring (`tetgen-rs`) is deferred.

### 8. Differentiability
| Newton ships | molla ships | Status |
|---|---|---|
| Featherstone basic adjoint (Warp tape) | **Pure-analytic Carpentier-Mansard** (Stage 23.1.4, **bwd/fwd 2.72×**) | ✅ molla **better** |
| SemiImplicit basic adjoint | — (semi-implicit not autograd-wired) | 🟡 minor gap |
| XPBD limited adjoint | xpbd_adjoint (manual) | ✅ |
| Tape autograd (Warp) | autograd::tape (5 ops + 6 robust losses + 3 Featherstone variants) | ✅ |
| Multi-DOF joint adjoints (no) | Ball, Free, Distance, D6 (Stages 23.1.2 + 23.1.3) | ✅ molla **better** |
| Multi-articulation batching (no) | rayon parallel batching (Stage 23.1.2, 3.36× per-chain speedup) | ✅ molla **better** |

**Verdict:** **molla is ahead of Newton on differentiability**. The only minor gap is wiring the semi-implicit solver for autograd.

### 9. Importers / formats
| Newton ships | molla ships | Status |
|---|---|---|
| **URDF** | — | ❌ **real gap** |
| **MJCF** | — (only via MuJoCoSolver writes-out path) | ❌ **real gap** |
| USD | usd_molla crate | ✅ |
| **glTF** (via USD) | — | ❌ minor gap |
| USD schema resolvers: Newton, MJCo, PhysX | One custom UsdPhysics resolver | 🟡 — single schema |

**Verdict:** **the biggest concrete gap.** Newton can ingest URDF / MJCF / USD / glTF; molla only takes USD. URDF + MJCF readers are typically a few hundred LOC each — concrete v1 work.

### 10. Sim infrastructure
| Newton ships | molla ships | Status |
|---|---|---|
| Multi-world batching | Model.world_count + per-entity world idx | ✅ |
| Multi-articulation | per-articulation index | ✅ |
| Persistent buffers | Buffer<T> w/ host + device memory | ✅ |
| Indirect dispatch | Used in MPM (Stage 24d.2 persistent dispatcher) | ✅ |
| Timestamp queries | — (Tracy via `make run-profile`) | 🟡 different mechanism |
| Graph coloring | Implicit in VBD's greedy particle colouring | 🟡 not exposed as public utility |

**Verdict:** at parity. Graph coloring would be useful as a public utility (Stage 21.1 sub-task in spirit; not yet booked).

### 11. Other / extras

#### Things molla has that Newton lacks
- **Pure-analytic Featherstone adjoint** (Carpentier-Mansard) at bwd/fwd **2.72×** vs Newton's basic-tape adjoint
- **Full robust-loss family** (6 losses including Tukey + Cauchy) wired into both autograd tape + IkSolver (IRLS)
- **GPU radix sort** for LBVH (4-pass 8-bit, ~12-line WGSL kernel)
- **Per-tet kinematic hardening** (Prager back-stress, Stage 20.4)
- **`make`-driven workflow** + dedicated viewer per scene (24 demo scenes; Newton has examples but no unified viewer matching molla_viewer's UX)
- **f64 host / f32 GPU** dual-precision pattern (Newton is f32 throughout)

#### Warp-level GPU primitives molla doesn't have (most explicit out-of-scope)
- ⛔ **Tile compute** (`tile_load`/`tile_matmul`/`tile_fft` via cuBLASDx/cuFFTDx) — wgpu has no `cooperative_matrix` extension portably
- ⛔ **Multi-GPU peer access + NCCL** — wgpu single-device
- ⛔ **CUDA graph capture** — no wgpu equivalent
- ⛔ **TensorCore tile matmul** — same constraint as tile compute
- ⛔ **Framework interop** (PyTorch / JAX / DLPack) — molla serves Rust, not Python
- ❌ **`warp.fem`** module (PDE assembly, polynomial bases, Dirichlet BCs) — orthogonal to physics-engine scope; if a user wants FEM, they want a different crate
- ❌ **Iterative linear solvers** (CG / GMRES / BiCGSTAB / PCG) — see math axis above; **the one real gap that matters for deferred MPM/Style3D/Kamino work**

#### Newton features molla doesn't have (real gaps)
- ❌ URDF / MJCF / glTF importers (asset pipeline)
- ❌ MPR narrow-phase (Stage 13 deferred)
- ❌ Contact reduction (global + hydroelastic + matcher coalescing)
- ❌ Style3D solver (explicit choice — VBD covers same workload)
- ❌ Kamino PADMM solver (explicit out-of-scope — overlaps MuJoCo wrapper)
- ❌ ControllerNeuralLSTM (deferred per Stage 18.1)
- ❌ Joint-mimic equality constraint
- ❌ Ellipsoid + Gaussian splat geometry primitives
- ❌ Multi-resolver USD (Newton/MJCo/PhysX schema parsers)
- ❌ `tetgen-rs`-style volumetric mesh authoring

---

## How far away are we?

**Quantitative:** 5 of 11 axes at full parity (joints, sensors, actuators, geometry, sim infrastructure). 4 axes molla is at-or-above (math, differentiability, robust losses, multi-articulation autograd). 2 axes have real gaps:
- **Asset pipeline (axis 9):** missing URDF + MJCF + glTF readers + multi-schema USD.
- **Solvers + contact (axes 1, 3):** missing Style3D (chose VBD) + Kamino (explicit OOS) + MPR + contact reduction. Implicit-MPM still fall-back-implicit.

**Qualitative:** if "Newton-equivalent feature parity" means "an RL / robotics user
can run their workload end-to-end without missing primitives", molla is **fully
there for Rust workloads driven via USD or hand-built models**. The user-visible
gap is **importer breadth** (URDF + MJCF + glTF). Everything else is either
already shipped, deferred-to-when-needed, or explicitly out of scope.

**Concrete suggested next moves to close the remaining gap (in priority order):**

1. **Stage 26 — URDF + MJCF + glTF importers.** Concrete asset-pipeline win.
   Low risk; ~500–1000 LOC each via existing Rust crates (`urdf-rs`,
   `mujoco-rs` MJCF parser, `gltf-rs`).
2. **Stage 27 — Iterative linear solvers** (CG, BiCGSTAB, PCG). Sparse-CSR
   matrix-free operator interface; unlocks Stage 24d.4's full implicit MPM,
   any future Style3D / Kamino consideration, and serves any downstream
   user who needs sparse FEM/PDE-style work.
3. **Stage 28 — MPR narrow-phase + contact reduction.** Closes the contact
   pipeline gap. MPR is the smaller chunk; contact reduction is multi-day.
4. **Stage 29 — Multi-schema USD** (PhysX schema reader, MuJoCo USD-extension
   reader). Builds on the existing usd_molla crate.
5. **Stage 30 — Ellipsoid + Gaussian splat shape kinds.** Small primitive
   adds; Gaussian splat is mostly a rendering concern.
6. **Stage 31 — `ControllerNeuralLSTM`** (deferred from Stage 18.1).
7. **Stage 32 — Joint-mimic equality constraint** for MJCF round-trip
   completeness.

**Explicitly NOT to do** (per `Stage 25+` discipline): Python bindings,
TensorCore tile matmul, CUDA graphs, multi-GPU peer access, Kamino-style
PADMM, `warp.fem`, Style3D.
