# APIC FEM-MPM coupling

The APIC (**A**ffine **P**article-**I**n-**C**ell) hybrid couples a
Material-Point-Method (MPM) particle representation with an FEM
mesh-side velocity update. It lives in
`crates/molla/src/fem/apic_coupling.rs` (CPU, Stage 37.6) and
`apic_coupling_gpu.rs` (GPU, Stage 41 v2). Particles transfer
momentum to mesh nodes via APIC weights, the mesh solves the
velocity-field step, and particles read the updated node velocities
back to update their own velocity and affine matrix.

## Why APIC?

Three transfer schemes coexist in the literature:

| Scheme  | Pros                       | Cons                                |
|---------|----------------------------|-------------------------------------|
| **PIC** | Simple, robust             | Loses angular momentum (one pass)   |
| **FLIP**| Cheap, preserves momentum  | Noise injection ⇒ jittery particles |
| **APIC**| Preserves angular momentum cleanly | One extra `Mat3` per particle |

Molla picks APIC because (a) the angular-momentum loss in PIC kills
rotational scenes, (b) the FLIP noise is hard to control, and (c)
APIC's extra storage is a single `Mat3` per particle — cheap.

## Data structures

```rust
pub struct ApicParticle {
    pub x: Vec3,        // position
    pub v: Vec3,        // velocity
    pub c: Mat3,        // affine velocity matrix
    pub mass: Real,
}

pub struct ApicHybrid {
    pub mesh: FemMesh,
    pub particles: Vec<ApicParticle>,
    pub gravity: Vec3,
}
```

The affine matrix `c` captures the local velocity gradient at the
particle's location. PIC sets `c = 0`; APIC carries it forward to
restore angular momentum across the P2G / G2P round-trip.

## The per-step pipeline

For each substep `Δt`:

1. **`locate_particle_tet`** — brute-force barycentric scan: for each
   particle, find the tet that contains it and return
   `(tet_verts, P1_weights, vertex_positions)`. v1 does an O(P · T)
   scan; the
   [`apic_locate_tet.wgsl`](#gpu-port--stage-41-v2) shader does the
   same on GPU.

2. **P2G** (particle → grid) — each particle deposits momentum at
   the four vertices of its enclosing tet:

   \\[
   \text{node\_momentum}[i] \mathrel{+}= w_i \, m_p \, (v_p + C_p \cdot (x_i - x_p))
   \\]

   where \\( w_i \\) is the P1 barycentric weight of \\( x_p \\) at
   tet vertex \\( i \\). Mass is also scattered proportionally so the
   caller can recover \\( v_{\text{node}} = \text{momentum} / \text{mass} \\).

3. **Mesh solve** — recover `node_velocity`, add `dt · gravity`,
   apply Dirichlet BCs (no-slip walls pin to zero), and run **one
   viscous Laplacian smoothing pass** \\( v \leftarrow v - dt \cdot \nu \cdot L v \\)
   to dissipate sub-grid energy. This is a v1 stand-in for a full P1
   Stokes pressure-Poisson projection — molla's Stokes solver lives
   on Taylor-Hood (P2 velocity), so plumbing the APIC P1 deposit
   through P2 needs either a P2 deposit (rebuilds the contract) or an
   L2 projection between spaces (Stage 37.6.1).

4. **G2P** (grid → particle) — gather mesh velocities back:

   \\[
   v_p^{\text{new}} = \sum_i w_i v_i, \quad
   C_p^{\text{new}} = \sum_i w_i v_i \, (x_i - x_p)^T \cdot D^{-1}
   \\]

   where \\( D = \sum_i w_i (x_i - x_p)(x_i - x_p)^T \\) is the
   per-tet second-moment matrix. For P1 tets this has a clean closed
   form — `apic_coupling.rs` evaluates it inline.

5. **Advect:** \\( x_p \mathrel{+}= dt \cdot v_p^{\text{new}} \\).

## Contract tests

The CPU implementation ships three round-trip tests in
`crates/molla/tests/`:

| Test                                | What it verifies                            |
|-------------------------------------|---------------------------------------------|
| `apic_zero_dt_momentum_roundtrip`   | `Σ m v` exact through P2G → G2P (`dt = 0`). |
| `apic_uniform_motion_no_advection`  | Uniform velocity field unchanged after one substep. |
| `apic_falling_particles_settle`     | Particles in a box with gravity settle to box floor; KE stays bounded. |

The "uniform motion" check is what `c = 0` particles would fail at —
APIC keeps it exact because the affine matrix captures the (zero)
velocity gradient.

## GPU port — Stage 41 v2

```rust
#[cfg(feature = "gpu")]
pub struct ApicGpuState { /* persistent buffers + pipelines */ }

impl ApicHybrid {
    #[cfg(feature = "gpu")]
    pub fn step_gpu(&mut self, state: &mut ApicGpuState, dt: Real,
                    bcs: &[BoundaryCondition], viscosity: Real) -> Result<()>;
}
```

`ApicGpuState::new(engine, hybrid)` uploads:

- Mesh: `vertices`, `cell_offsets`, `cell_indices`.
- Particles: `x`, `v`, `c`, `mass`, plus a per-particle `loc` slot
  that the locate kernel writes into.
- Node graph: a precomputed adjacency CSR for the viscous-smoothing
  Jacobi sweep, built once on CPU at construction.

Per step the pipeline dispatches seven WGSL kernels in order:

```
locate_tet  →  reset_grid  →  p2g  →  grid_update
                 ↓
            viscous_smoothing (one Jacobi sweep)
                 ↓
              apply_bcs  →  g2p
```

The locate kernel is the only one with `O(P · T)` cost; for the
contract-test sizes (≤ 200 particles, ≤ 500 tets) it's fine. A
follow-up version with a tet-BVH (`crates/molla/src/geometry/tet_bvh.rs`)
would drop locate to `O(P · log T)` without changing the public API.

The Stage 41 v2 sub-3 measurement on the falling-particles fixture
showed a **14.5× speedup** over CPU at 200 particles in a 16³ tet
mesh — locate-dominated workload, so the speedup grows further as the
particle count rises.

## Caveats

- **No real Stokes projection in v1.** The viscous-smoothing
  placeholder is incompressible only in the limit \\( \nu \cdot dt \to \infty \\).
  A true pressure-Poisson on the P1 mesh is Stage 37.6.1.
- **Particles can drift out of bounds.** Explicit advection
  occasionally overshoots a wall; the contract test clamps to the
  bounding box AABB to keep the demo honest.
- **Brute-force locate.** O(P · T) on CPU and GPU; not a problem at
  v1 sizes but doesn't scale to 10⁴+ particles or 10⁴+ tets.

## Cross-references

- [APIC compute](../compute/apic.md) (when documented) — the raw
  particle-grid transfer primitive that the FEM hybrid wraps.
- [MPM solvers](../solvers/mpm.md) (when documented) — the pure-MPM
  variant for the Stokes-free case.
- [Stokes / NS](./stokes_ns.md) — the planned full pressure-Poisson
  replacement for the viscous-smoothing placeholder.
- [Mesh](./mesh.md) — `FemMesh` is what `ApicHybrid` owns.
