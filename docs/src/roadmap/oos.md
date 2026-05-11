# Out of scope

This page is the **definitive list of things molla deliberately does not
do**. Every entry is a *design decision*, not a gap. The list is curated
from `PLAN.md` §1 (non-goals), §4 (Newton reference inventory),
§20 Stage 25+ (explicit OOS), and the decision log.

The structure for each entry:

1. **What** is being skipped.
2. **Why** — the design reason it's out of scope. This is the load-bearing
   part: the same reasoning should re-apply on every re-evaluation.
3. **When we'd reconsider** — the explicit signal that would re-open the
   decision. If that signal arrives, this page gets edited; otherwise the
   item stays out.
4. **Source** — the commit / stage / PLAN section that originally locked
   it in.

The point of writing these down is that an item being missing from molla
is **almost always one of two things**: it's in
[`./open.md`](./open.md) (an open item, planned), or it's here (OOS,
deliberate). If it's in neither, the gap analysis missed it.

---

## OOS-1. Python bindings (`pyo3`)

**What.** A first-class Python API exposing `Model`, `State`, the
solvers, and the autograd surface. Newton ships exactly this.

**Why.** molla is a **Rust-first** engine. Two reasons:

- The user base for molla is Rust apps, including `bevy_glacial` /
  `usd_molla` consumers and embedded Rust robotics stacks. None of
  these need Python.
- Newton already covers the "Python-first physics engine on GPU" niche
  better than molla ever could — it has the Warp JIT, the NumPy
  ecosystem, and the existing Newton user base. Building a worse copy
  is not the value-add.

A Python binding would also infect the surface area: every public type
in molla would need a wrapper, every callback would need GIL handling,
and the autograd story would have to bridge with `torch.Tensor` /
`numpy.ndarray`. Implementation cost is large; maintenance cost is
permanent.

**When we'd reconsider.** A concrete user explicitly asks for the
bindings *and* commits to the maintenance burden. "It would be nice to
have" doesn't qualify.

**Source.** `PLAN.md` §1 ("non-goals"), `GAP.md` §9 ("explicitly out of
scope"), `GAP_v2.md` §9. Decision-log entry 2026-05-09.

---

## OOS-2. TensorCore tile matmul

**What.** GEMM primitives that compile to NVIDIA TensorCore mma
instructions, the way Warp's `tile_matmul` does via libmathdx.

**Why.** **wgpu does not expose the `cooperative_matrix` WGSL extension
portably.** The extension is Vulkan-spec and exists in
`VK_KHR_cooperative_matrix`, but wgpu's WGSL surface doesn't yet bind
it consistently across the three backends (Vulkan / Metal / DX12). Even
when it does land, it'll be vendor-specific: TensorCore on NVIDIA,
`AMX` on Apple, `XMX` on Intel. molla is committed to *portable* GPU
compute; a TensorCore-only path is a different engine.

The pragmatic position from `GAP.md` §8: "Don't try to match Warp on
tensor-core / multi-GPU / CUDA-graph features. Those are CUDA-specific
and out of scope for a portable wgpu engine."

The Stage 37.8 tile primitives shipped on stock WGSL (no cooperative
matrix). They're slower than Warp's TensorCore path on hardware that has
TensorCore, and faster than a no-tile baseline on hardware that doesn't.
That's the right tradeoff for a portable engine.

**When we'd reconsider.** wgpu lands stable WGSL `cooperative_matrix`
support across at least Vulkan + Metal backends, *and* a user surfaces a
matmul-bound workload (a neural controller too big for the existing
`ControllerNeuralMLP`, an FEM problem whose stiffness assembly is
GEMM-shaped). Both conditions, not either.

**Source.** `GAP.md` §8, `PLAN.md` §20 Stage 25+, `GAP_v2.md` §9.

---

## OOS-3. CUDA Graphs for command batching

**What.** A capture-replay pattern that records a sequence of kernel
launches once and replays them with one host submit. Warp uses this to
amortize Python launch overhead.

**Why.** **wgpu has no equivalent capture API.** Each `CommandEncoder`
is one-shot — record, submit, drop. wgpu does coalesce dispatches into
fewer submits when the encoder is reused, but there's no
record-once-replay-many primitive.

Building a wgpu-side equivalent would mean either:

- Pre-recording into a "command buffer template" that wgpu doesn't
  support, or
- Building a JIT that fuses the kernels into a single mega-kernel — which
  is what Stage 41 v6 (`2bde182`) actually does for the inner CG loop,
  manually.

The mega-kernel approach is the right pattern for wgpu and is the one
the Stage 41 v3–v6 sequence has been progressively adopting. CUDA
Graphs is a different mechanism solving the same problem in a
different way.

**When we'd reconsider.** wgpu adds a command-buffer-template
abstraction in a future spec revision *and* the host-dispatch overhead
becomes a measurable bottleneck. The current Stage 41 v6 work has
*already* eliminated the per-iteration host round-trip via a fused
kernel, so the bottleneck is moving.

**Source.** `GAP.md` §8 table row, `PLAN.md` Stage 25+, `GAP_v2.md` §9.

---

## OOS-4. Multi-GPU peer access

**What.** Splitting a single physics scene across multiple GPUs with
peer-to-peer memory copies, the way Warp's multi-device API does on
NVLink hardware.

**Why.** **wgpu is single-device** by construction. Each `Instance` /
`Adapter` / `Device` triplet binds one physical GPU; cross-device buffer
sharing is not in the spec.

The single-device model is the right one for portable consumer hardware
(laptops, single-GPU workstations, dev machines) where molla actually
runs. Multi-GPU is a niche for HPC clusters and trading desks. molla is
not a cluster physics engine and isn't trying to be one.

A multi-process / multi-instance model that runs N independent molla
worlds in parallel (one per GPU) on the host side is a different thing
— that's already possible today, no engine changes needed. Real
multi-GPU peer access (single scene across N GPUs, shared state buffer)
is the one that's OOS.

**When we'd reconsider.** wgpu adds multi-device buffer sharing
*and* the user base shifts to HPC / cluster workloads. Both are
unlikely.

**Source.** `GAP.md` §8, `PLAN.md` Stage 25+, `GAP_v2.md` §9.

---

## OOS-5. Kamino-style PADMM solver

**What.** A constrained-dynamics solver in the style of Newton's
`kamino/` — closed-loop kinematics, hard contact via NCP, spatial
friction, restitution, all wrapped in a PADMM (proximal ADMM) outer
loop.

**Why.** **It overlaps with the MuJoCo wrapper for the
constrained-dynamics niche, and we picked the wrapper.** PLAN.md §20
Stage 22 added the MuJoCo wrapper as molla's answer to "I need a
mature, battle-tested constrained-dynamics solver." MuJoCo is the
industry standard; Kamino is a research solver. Shipping both means
maintaining two contact codepaths with overlapping responsibilities and
diverging APIs.

A secondary reason: PADMM-on-NCP is research-grade and Kamino itself is
still beta in Newton (CPU-only, "use at your own risk"). The cost of
reimplementing it for molla is high; the upside is "we have a research
solver no one asked for." For users that need the algorithm, Newton +
mujoco_warp is right there.

**When we'd reconsider.** A user explicitly wants Kamino-the-algorithm
(not "a constrained-dynamics solver") *and* MuJoCo wrapper is
insufficient for their use case. Both conditions. The MuJoCo wrapper is
the default answer; Kamino is the escape hatch we're not building.

**Source.** `GAP.md` §9, `PLAN.md` §4 (Newton reference: `kamino`
listed as "Research solver — Out of scope"), `PLAN.md` Stage 25+,
`GAP_v2.md` §9.

---

## OOS-6. Style3D cloth (projective dynamics for cloth)

**What.** Newton's `style3d/` — a production cloth solver with
anisotropic bending, panel sewing, dedicated BVH, and a global PCG
linear solve. Used in fashion / garment simulation.

**Why.** **Stage 20 picked VBD over Style3D, and the decision still
stands.** The reasoning (recorded in PLAN.md decision-log entry
2026-05-09):

- VBD ports more cleanly onto WGSL — per-vertex 3×3 Newton block,
  graph-coloured passes mirror the existing XPBD particles kernel.
- Style3D needs a global PCG linear solve. wgpu has no good primitive
  for that *at the granularity Style3D wants* (a single global solve
  per cloth body, low-iteration, on a Hessian-structured matrix).
  Stage 38a (BSR algebra) and Stage 39F (GPU SpGEMM) get us there
  *now*, but two years late vs the Stage 20 ship date.
- Stage 20 + Stage 33 + Stage 33.1–33.4 collectively cover the use
  case Style3D solves — including self-collision, CCD, and friction.
  The cloth that needs Style3D's anisotropic-bending models is a
  garment-design workflow that molla isn't targeting.

**When we'd reconsider.** A garment-design / cloth-authoring user
surfaces who needs anisotropic-bending materials *and* finds VBD's
isotropic-bending model insufficient. The current VBD pipeline can
already do panel sewing via CONNECT constraints; the missing piece is
specifically anisotropic bending energy.

**Source.** `GAP.md` §2 (solver inventory), `PLAN.md` §4 (Newton
reference: `style3d` listed as "Production cloth — Out of scope"),
`PLAN.md` Stage 25+, `PLAN.md` decision-log 2026-05-09 (VBD-vs-Style3D
choice), `GAP_v2.md` §2 + §9.

---

## OOS-7. NanoVDB / Warp Volume parity beyond what Stage 24e shipped

**What.** Full Warp `Volume` API parity, in particular
`volume_sample_grad` (analytic gradient of a tri-linear sample) and the
full set of value-type templates (`Vec3f`, `Mat3f`, packed types).

**Why.** **Stage 24e + 24e.1 + 24e.2 already ship a working
NanoVDB-equivalent.** The remaining items (`volume_sample_grad`, broader
templating) are not blocking any current workload. Adding them is
template-engineering work in WGSL, which is painful — WGSL doesn't have
templates, so each value type is a kernel-family copy.

This is not a permanent OOS; it's a "we ship what users ask for"
deferral. The sparse-grid surface in molla today is enough for sand /
snow / fluid-with-volume workflows; if a workflow appears that needs
analytic gradients on volume samples or a Mat3 value type, we add it.

**When we'd reconsider.** A user needs analytic volume-sample
gradients (for differentiable rendering through a volume, e.g.) or a
non-scalar value type (vector fields stored sparsely). Reactive
expansion, not proactive.

**Source.** `GAP_v2.md` §2 ("sparse-grid partial — no
`volume_sample_grad`"), Stage 24e history.

---

## OOS-8. Differentiable raycast with analytic adjoint

**What.** A closed-form adjoint of the raycast operator (`∂t / ∂scene_params`
where `t` is the hit distance), as opposed to the finite-difference adjoint
that Stage 37.7 shipped.

**Why.** **FD adjoint is enough for the current diff-Gaussian use case
(Stage 44).** Closed-form raycast adjoint requires:

- Triangle / primitive intersection equations differentiated symbolically.
- A consistent treatment of silhouette discontinuities (the hit point
  jumps when the ray transitions across a triangle edge).
- A per-primitive type implementation in WGSL.

The silhouette-discontinuity handling is the genuinely hard part; the
solution (edge-aware differentiable rendering) is an active research
area. molla's diff-Gaussian path (Stage 44 v1–v4) bypasses this by using
*α-composited Gaussians* which are smooth — no silhouettes, no
discontinuities. The FD raycast adjoint is for the path-tracing /
hard-surface case that we're not targeting.

**When we'd reconsider.** A workload appears that needs hard-surface
differentiable rendering with backprop through opaque mesh
intersections. At that point we'd evaluate either a closed-form
implementation or the established libraries (Mitsuba 3, NVIDIA SLANG /
Falcor).

**Source.** `GAP_v2.md` §2 ("raycast adjoint partial — analytic adjoint
deferred"), Stage 37.7 history.

---

## OOS-9. `@wp.kernel` JIT from Python

**What.** Compile Python source to GPU kernels at runtime, the way
Warp's `@wp.kernel` decorator does. Lets users author kernels in Python
without recompiling the engine.

**Why.** **molla is Rust.** Users author kernels by editing WGSL files
in `crates/molla/shaders/`. Kernels compile at engine startup
(`include_str!` + naga validation); subsequent runs are zero-cost.

A Rust equivalent would require either:

- A Rust → WGSL transpiler (`rust-gpu` is the closest thing, and isn't
  there for compute kernels yet), or
- A runtime DSL embedded in Rust macros — which loses static-type
  checking and IDE integration, the things Rust users actually want.

The pragmatic stance from `GAP.md` §8: "molla compiles WGSL at startup;
comparable cost" to Warp's JIT. The JIT path is *more* flexible (kernel
specialization per call); the AOT path is *more* predictable (one
compile, deterministic execution).

**When we'd reconsider.** `rust-gpu` (or a successor) becomes
production-ready for compute kernels with full WGSL feature parity. At
that point we'd evaluate it. Until then, hand-written WGSL is the
right call.

**Source.** `GAP.md` §8 ("comparable cost" line).

---

## OOS-10. Kinematic constraint handling beyond what's shipped

**What.** Newton's MuJoCo wrapper exposes the full quartic-polynomial
mimic constraint form `q_a = c0 + c1·q_b + c2·q_b² + c3·q_b³ + c4·q_b⁴`.
molla's Stage 32 mimic constraint supports only the linear form
`q_a = c0 + c1·q_b`.

**Why.** **No user has asked for higher-order mimic, and the
constraint-handling story in molla is intentionally minimal.** PLAN.md
§1 explicitly lists "complex constraint handling" as a non-goal —
"complex" meaning anything beyond CONNECT / WELD / linear-mimic +
joint limits + simple equality.

The full quartic mimic is for one very specific Newton-via-MuJoCo
workflow (cam-follower kinematics with non-linear gear trains). It's a
day or two of work but it adds a permanent surface to maintain.

**When we'd reconsider.** A user with non-linear gear-train kinematics
or cam-follower modeling needs it. Until then, linear mimic + multiple
joints chained together covers every kinematics case we've seen.

**Source.** `PLAN.md` §1 ("non-goals — complex constraint handling"),
`GAP.md` §2.2 ("Mimic — molla supports linear; Newton supports full
quartic").

---

## Items that are NOT on this page (and why)

If you came looking for one of these, you want
[`./open.md`](./open.md), not this page:

- **NonconformingField / nonconforming-mesh contact** — open item; v4
  shipped Newton-Raphson penalty, see open A6 + B6.
- **Self-contact in the new contact stack** — open item, B6.
- **Phase-field fracture, multi-phase flow, compressible Euler,
  Mooney-Rivlin tangents, rate-and-state friction** — open items, see
  Tier-C of `open.md`. They're called "not on this page" because there's
  no design reason they can't ship; they're deferred for lack of demand,
  not by deliberate scope reduction.
- **GPU Bingham / NS time integrator on-device / Π-projection AMS / DG
  limiter GPU port / AMR coarsening / DG-AMR level recursion / Krylov
  warm restart / active-set anti-cycling / muscle SH gradient GPU** —
  all open items in `open.md` Tiers A/B.

The rule of thumb: if Newton ships it and we don't, look here first
(deliberate scope choice). If we don't and Newton doesn't either, look
in `open.md` Tier-C (genuinely new physics).

---

## Decision log

Every entry above is anchored to one or more of:

- **`PLAN.md` §1 (Goals and non-goals).** The originating non-goals
  list. OOS-1 (Python), OOS-2 (TensorCore), OOS-3 (CUDA graphs),
  OOS-4 (multi-GPU), OOS-10 (complex constraints).
- **`PLAN.md` §4 (Newton reference inventory).** Lists `mujoco`,
  `style3d`, `implicit_mpm`, `kamino` as Newton modules with explicit
  "Out of scope" annotations (the first three were OOS-by-design; the
  last is open or has shipped variants).
- **`PLAN.md` §20 Stage 25+ ("Out of scope / explicitly deferred").**
  The roadmap-side OOS list. Mirrors and extends §1.
- **`PLAN.md` §22 (Decision log).** Each major OOS decision has a
  dated entry. Notably 2026-05-09 (Stages 14–25 + the explicit OOS
  list) and the Stage 20 VBD-vs-Style3D pick.
- **`GAP.md` §8 (wgpu/WGSL limitations) and §9 (OOS).** The
  cross-validated agent-scan version of the OOS list (2026-05-09).
- **`GAP_v2.md` §9 (Out-of-scope).** The refreshed list as of
  2026-05-11 evening, post-Stages 39–46.

If an item on this page is reconsidered and reversed, the reversal
should be recorded in `PLAN.md` §22 decision log with date, signal
that triggered the change, and the resulting roadmap entry.
