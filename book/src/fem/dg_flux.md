# DG numerical fluxes

The numerical flux \\( \hat F(u^-, u^+, \mathbf{n}) \\) at each face is
what couples the otherwise-disjoint per-cell DoFs of a DG
discretisation. Molla exposes a small library of fluxes in
`dg_element.rs` via the `NumericalFlux` enum and dedicated
per-face block evaluators.

## The `NumericalFlux` enum

```rust
pub enum NumericalFlux {
    Central,                  // ½(F_L + F_R)·n
    Upwind { wind: Vec3 },    // (wind·n) · u_upwind
    LaxFriedrichs { alpha: Real }, // ½(F_L+F_R)·n − ½α(u_R−u_L)
    Roe,                      // = Upwind for scalar; placeholder for systems
    Hll,                      // (Stage 43 v3) HLL — system flux, SWE only
    Hllc,                     // (Stage 43 v3) HLLC — system flux, SWE only
    RoeSwe,                   // (Stage 43 v3) Roe — system flux, SWE only
}
```

The first four are scalar fluxes that work for any conservation law of
the form \\( u_t + \nabla \cdot (\mathbf{a} u) = 0 \\). The last three
are **system fluxes** specialised for shallow-water; the scalar block
assembler panics if you pass them in.

## Scalar flux selection guide

| Flux                 | Use when                                                   |
|----------------------|------------------------------------------------------------|
| `Central`            | Diffusion-dominated; no upwinding (paired with SIP block) |
| `Upwind { wind }`    | Linear advection with constant velocity — exact Riemann   |
| `LaxFriedrichs { α }`| Non-linear scalar (Burgers'); robust + diffusive          |
| `Roe`                | Same as `Upwind` for scalar problems (kept for generic code) |

The `α` parameter for Lax-Friedrichs is the **global** maximum wave
speed: for Burgers' use \\( \max_u |u| \\), for SWE use
\\( \max_u (|u| + \sqrt{gh}) \\).

## Scalar per-face block

```rust
pub fn compute_dg_numerical_flux_block(
    cell_left: &FemCell,
    cell_right: &FemCell,
    face: &InteriorFace,
    vertices: &[Vec3],
    flux: NumericalFlux,
    degree: usize,
) -> Vec<Vec<Real>>;
```

Returns an `(nl + nr) × (nl + nr)` row-major block that scatters into
the global DG system. Rows / cols `[0..nl]` hold the left-cell DoFs,
rows / cols `[nl..]` hold the right-cell DoFs. The block is
**non-symmetric** in general — advection is not self-adjoint.

The weak-form contribution of one interior face \\( F \\) is

\\[
-\int_F \hat F(u_L, u_R, \mathbf{n}) \, \llbracket v \rrbracket \, dS
\\]

with \\( \llbracket v \rrbracket = v_L \mathbf{n}_L + v_R \mathbf{n}_R \\) the
vector jump. The integrator uses

- 2-point Gauss-Legendre quadrature on a 2D edge (degree-3 exact),
- 3-point mid-edge rule on a 3D triangular face (degree-2 exact),

unless `degree == 2` in which case it switches to the higher-order
companions (3-pt Gauss / 6-pt Dunavant on the face).

There's also a vector-valued companion for SWE component-wise fluxes:

```rust
pub fn compute_dg_numerical_flux_block_vector(...);
```

## SIP flux for elliptic diffusion

```rust
pub fn compute_dg_flux_jump_block(
    cell_left: &FemCell,
    cell_right: &FemCell,
    face: &InteriorFace,
    vertices: &[Vec3],
    coeff: Real,
    degree: usize,
    eta: Real,
) -> Vec<Vec<Real>>;
```

The **Symmetric Interior Penalty** Laplacian flux — standard,
symmetric, and provably coercive for sufficient penalty:

\\[
a_F(u, v) = -\int_F \{\alpha \nabla u \cdot \mathbf{n}\} \, \llbracket v \rrbracket \, dS
            -\int_F \llbracket u \rrbracket \, \{\alpha \nabla v \cdot \mathbf{n}\} \, dS
            +\frac{\eta \alpha}{h_F} \int_F \llbracket u \rrbracket \, \llbracket v \rrbracket \, dS
\\]

with the average and scalar-jump operators

\\[
\{w\} = \tfrac{1}{2}(w^- + w^+), \quad
\llbracket w \rrbracket = w^- - w^+.
\\]

`eta` is the penalty parameter; typical values are
\\( \eta = \mathrm{degree} \cdot (\mathrm{degree}+1) \cdot c \\) with
\\( c \approx 10 \\) (the Arnold-Brezzi-Cockburn-Marini choice; matches
[`default_nitsche_sigma`](./bc.md#dg-nitsche-weak-dirichlet)).

## Nitsche boundary block

```rust
pub fn compute_nitsche_boundary_block(
    cell: &FemCell,
    face_verts: &[u32],
    vertices: &[Vec3],
    alpha: Real,
    degree: usize,
    sigma: Real,
    bc_values: &[Real],
) -> (Vec<Vec<Real>>, Vec<Real>);
```

Returns the per-cell matrix block + RHS for the
Nitsche weak-Dirichlet formulation — see
[boundary conditions](./bc.md#dg-nitsche-weak-dirichlet) for the
applier and the math.

## SWE-specific fluxes — Stage 43 v3

Shallow-water equations (SWE) are a system of three conservation laws
in 2D:

\\[
\partial_t \begin{pmatrix} h \\ hu \\ hv \end{pmatrix}
+ \nabla \cdot \begin{pmatrix} hu, hv \\ hu^2 + \tfrac{1}{2} g h^2, \, h u v \\ huv, \, h v^2 + \tfrac{1}{2} g h^2 \end{pmatrix} = 0.
\\]

The eigenstructure (wave speeds \\( u \pm c \\) with \\( c = \sqrt{gh} \\)
plus the contact \\( u \\)) makes shock-capturing tricky — generic
scalar fluxes (Lax-Friedrichs etc.) work but smear contacts. SWE-
specific Riemann solvers do better.

### HLL — two-wave Harten-Lax-van-Leer

```rust
pub fn compute_swe_numerical_flux_hll(
    cell_left: &FemCell,
    cell_right: &FemCell,
    face: &InteriorFace,
    vertices: &[Vec3],
    u_left: &[Real],     // 3 components: (h, hu, hv) at face quad point
    u_right: &[Real],
    g: Real,
) -> Vec<Real>;
```

Two-wave Riemann solver: pick a single star-region state satisfying
conservation across the slowest and fastest wave speeds. Davis-style
wave-speed estimates:

\\[
s_l = \min(u_L \cdot \mathbf{n} - \sqrt{g h_L}, \, \tilde u - \tilde c), \quad
s_r = \max(u_R \cdot \mathbf{n} + \sqrt{g h_R}, \, \tilde u + \tilde c)
\\]

where \\( (\tilde u, \tilde c) \\) are the Roe-average wave speeds.
The HLL flux is then

\\[
\hat F_{\text{HLL}} =
\begin{cases}
F_L & \text{if } s_l \ge 0 \\
\dfrac{s_r F_L - s_l F_R + s_l s_r (U_R - U_L)}{s_r - s_l} & \text{if } s_l < 0 < s_r \\
F_R & \text{if } s_r \le 0
\end{cases}
\\]

Robust, diffusive in the contact. Good first choice for "any SWE
problem".

### HLLC — three-wave Toro 1994

Adds a contact-discontinuity star-region between the two HLL waves:

```rust
pub fn compute_swe_numerical_flux_hllc(...);
```

The contact speed
\\( s_* = (s_l h_L (u_L - s_l) - s_r h_R (u_R - s_r)) / \ldots \\) splits
the star region into a left and right side, each carrying the
upstream tangential velocity. The result: **sharper contact
resolution** than HLL — important for dam-break / wet-bed problems
where the post-shock height has a slowly-moving contact between the
left and right star states.

### RoeSwe — linearised Riemann

```rust
pub fn compute_swe_numerical_flux_roe(...);
```

Builds the Roe-average state

\\[
\hat h = \tfrac{1}{2}(h_L + h_R), \quad
\hat u = \frac{u_L \sqrt{h_L} + u_R \sqrt{h_R}}{\sqrt{h_L} + \sqrt{h_R}}, \quad
\hat c = \sqrt{g \hat h}
\\]

(the SWE Roe average — the density-weighted form from gas dynamics
collapses to this), then **upwinds each characteristic field
independently** with eigenvalues
\\( (\hat u \cdot \mathbf{n} - \hat c, \, \hat u \cdot \mathbf{n}, \, \hat u \cdot \mathbf{n} + \hat c) \\).

An **entropy fix** (Harten-Hyman) replaces eigenvalues smaller than
\\( \delta = \max(0, \hat\lambda - \lambda_L, \lambda_R - \hat\lambda) \\)
with the smooth
\\( (\lambda^2 + \delta^2) / (2\delta) \\) to prevent transonic-
rarefaction "anti-entropy" sonic glitches.

Sharpest contact + shock resolution of the three for smooth flows;
needs the entropy fix to stay robust at transonic rarefactions.

### Selection guide for SWE

| Flux       | Resolution | Robustness | Use case                                  |
|------------|------------|------------|-------------------------------------------|
| `Hll`      | OK         | Excellent  | Anywhere — default fallback              |
| `Hllc`     | Better     | Very good  | Dam-break / wet-bed contact resolution   |
| `RoeSwe`   | Best       | Good (with entropy fix) | Smooth flow + isolated shocks |

The `DgShallowWaterDamBreak` viewer scene (#97) exercises all three on
the same dam-break geometry — comparing the height profile shows the
expected contact-sharpening progression.

## Face enumeration helpers

```rust
pub struct InteriorFace {
    pub cell_left: u32,
    pub cell_right: u32,
    pub face_verts: Vec<u32>,  // sorted ascending
}

pub fn compute_interior_faces(mesh: &FemMesh) -> Vec<InteriorFace>;
pub fn periodic_face_pairs(mesh: &FemMesh, axis: usize) -> Vec<PeriodicPair>;
```

`compute_interior_faces` returns each face shared by exactly **two**
cells (boundary faces get one owner; manifold meshes never have three).
`periodic_face_pairs` pairs up opposite boundary faces along the
chosen axis — the Stage 43 trick that lets the SSP-RK time stepper
roll a periodic Burgers shock around forever.

## Cross-references

- [DG basis](./dg_basis.md) — per-cell DoF layout that these blocks
  scatter into.
- [DG limiters](./dg_limiters.md) — the post-stage clip applied after
  each SSP-RK substep.
- [DG time](./dg_time.md) — SSP-RK steppers that pair with these
  fluxes.
- [DG BC](./dg_bc.md) — Nitsche boundary blocks.
- [DG GPU](./dg_gpu.md) — Stage 43 v4 GPU port of HLL / HLLC / Roe.
