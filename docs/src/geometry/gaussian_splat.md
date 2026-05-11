# Gaussian splats

`crates/molla/src/geometry/gaussian_splat.rs` defines the 3D
Gaussian splat primitives that feed the differentiable raytrace
renderer (`crates/molla/src/sensors/splat_raytracer.rs`) and the
forward / backward WGSL kernels in `crates/molla/shaders/splat_raytrace*.wgsl`.

Stage 44 evolved through four revisions, each adding shading
fidelity while keeping the core geometry pipeline byte-identical:

| Rev | Type | Adds | Coefficients |
|-----|------|------|--------------|
| v1  | `GaussianSplat3d`              | forward-only diffuse RGB | 3 |
| v2  | (same type)                    | reverse-mode adjoint via `SplatParam` | — |
| v3  | `GaussianSplatSh3d`            | spherical-harmonic colour, bands 0..=2 | 9 × 3 = 27 |
| v4  | `GaussianSplatShBand3d`        | adds band 3 (Kerbl 2023) | 16 × 3 = 48 |
| v4  | `GaussianSplatShDiffuseView3d` | diffuse / view-dependent split | 1 + 15 |

All four share the same geometric kernel (anisotropic 3D Gaussian
density along a ray) — they differ only in how per-splat colour
is parameterised.

## v1 — `GaussianSplat3d` (diffuse RGB)

```rust
pub struct GaussianSplat3d {
    pub position: Vec3,    // world-space mean μ
    pub scale:    Vec3,    // per-axis stddev (must be > 0)
    pub rotation: Quat,    // covariance is R · diag(s²) · Rᵀ
    pub color:    [Real; 4], // diffuse RGB + reserved alpha
    pub opacity:  Real,    // bare opacity ∈ [0, 1]
}

impl GaussianSplat3d {
    pub fn new(
        position: Vec3, scale: Vec3, rotation: Quat,
        color_rgb: Vec3, opacity: Real,
    ) -> Self;
    pub fn color_rgb(&self) -> Vec3;
}
```

## Ray-Gaussian density integral

A 3D Gaussian splat is an anisotropic Gaussian with mean `μ`,
rotation `R`, scale `s`, opacity `o`. For a ray `(p, d)`, the
forward path transforms the ray into the splat's **canonical**
(unit-isotropic) frame:

```text
o_local = R⁻¹ · (p - μ) / s          (component-wise division)
d_local = R⁻¹ · d        / s
```

The maximum density along the canonical ray is reached at the
point of closest approach to the origin:

```text
ρ_max = exp( -½ · |d_local × o_local|² / |d_local|² )
```

(because the squared distance from origin to the line
`o_local + t · d_local` is `|d × o|² / |d|²`). The per-splat α is
`α = min(1, ρ_max · o)`. This is the standard analytic
approximation used in Newton's
`warp_raytrace/gaussians.py::canonical_ray_max_kernel_response`
and most 3DGS renderers — it captures the dominant integrand of
the line-integral over the Gaussian density without resorting to a
numerical erf evaluation.

The hit distance along the **world** ray for sorting purposes is
the projection of `(μ - p)` onto `d` — i.e. the parametric `t`
where the ray reaches the splat centre's projection. This is the
camera-distance sort matching Newton's default.

```rust
pub fn map_ray_to_canonical(
    splat: &GaussianSplat3d, ray_origin: Vec3, ray_dir: Vec3,
) -> (Vec3, Vec3); // (o_local, d_local) — d not normalised

pub fn canonical_ray_max_response(o_local: Vec3, d_local: Vec3) -> Real;

pub fn gaussian_response(
    ray_origin: Vec3, ray_dir: Vec3, splat: &GaussianSplat3d,
) -> Real;

pub fn splat_camera_distance(
    ray_origin: Vec3, ray_dir: Vec3, splat: &GaussianSplat3d,
) -> Real;
```

`map_ray_to_canonical` is careful with quaternion conventions
(`gaussian_splat.rs:140-164`): it uses the **BAC-CAB** expansion
`v + 2 q_w (q_v × v) + 2 (q_v × (q_v × v))` rather than `glam`'s
canonical `q v q*` formula. For unit `q` the two agree, but the
FD adjoint oracle perturbs `q` off the unit sphere; the WGSL
`quat_rotate` uses BAC-CAB, so the CPU forward must match exactly
or the FD gradient check fails.

## Depth sort + α-composite

The forward composite (`composite_forward_cpu`):

```text
C_out = Σᵢ Cᵢ · αᵢ · ∏_{j<i}(1 - αⱼ),    T_out = ∏ᵢ (1 - αᵢ)
```

Front-to-back α-compositing. The host sorts splats by
`splat_camera_distance`, skips splats with `t ≤ 0` or
`t > max_distance`, and clamps tiny α below `min_alpha` for
performance. Termination short-circuits when transmittance drops
below `1e-4` — invisible splats further down the ray don't
contribute.

```rust
pub fn composite_forward_cpu(
    ray_origin: Vec3, ray_dir: Vec3, splats: &[GaussianSplat3d],
    max_distance: Real, min_alpha: Real,
) -> (Vec3, Real); // (rgb, transmittance)

pub fn composite_rgb_cpu(
    ray_origin: Vec3, ray_dir: Vec3, splats: &[GaussianSplat3d],
    max_distance: Real, min_alpha: Real,
) -> Vec3;
```

## v2 — Reverse-mode adjoint (`SplatParam`)

`SplatParam` (`gaussian_splat.rs:287-333`) lists every
per-splat learnable parameter as an enum variant — `Position`,
`Scale`, `Rotation`, `Color`, `Opacity`. The reverse-mode
backward kernel (`shaders/splat_raytrace_backward.wgsl`) writes
gradients into the same enum-indexed slots; `fd_grad_one_splat`
is the CPU finite-difference oracle used by the parity test:

```rust
pub fn fd_grad_one_splat(
    ray_origin: Vec3, ray_dir: Vec3, splats: &[GaussianSplat3d],
    splat_index: usize, param: SplatParam,
    max_distance: Real, min_alpha: Real, h: Real,
) -> Vec3;
```

The FD step `h` is the standard symmetric `± h` perturbation; on
the unit quaternion this is the off-sphere perturbation the BAC-CAB
discussion above hinges on.

## v3 — Spherical harmonics, bands 0..=2

```rust
pub const SH_BAND_2_COUNT: usize = 9; // bands 0, 1, 2

pub const SH_C0: Real = 0.28209479177387814;    // 1 / (2 √π)
pub const SH_C1: Real = 0.4886025119029199;    // √(3 / 4π)
pub const SH_C2A: Real = 1.0925484305920792;   //  ½ √(15/π)
pub const SH_C2B: Real = -1.0925484305920792;
pub const SH_C2C: Real = 0.31539156525252005;  //  ¼ √(5/π)
pub const SH_C2D: Real = -1.0925484305920792;
pub const SH_C2E: Real = 0.5462742152960396;   //  ¼ √(15/π)

pub struct GaussianSplatSh3d {
    pub position: Vec3, pub scale: Vec3, pub rotation: Quat,
    pub sh:       [Vec3; SH_BAND_2_COUNT], // 9 coefficient triples
    pub opacity:  Real,
}

impl GaussianSplatSh3d {
    pub fn diffuse(position: Vec3, scale: Vec3, rotation: Quat, rgb: Vec3, opacity: Real) -> Self;
    pub fn from_sh (position: Vec3, scale: Vec3, rotation: Quat, sh:  [Vec3; 9], opacity: Real) -> Self;
    pub fn eval_color(&self, dir: Vec3) -> Vec3;
}

pub fn eval_sh(coeffs: &[Vec3; SH_BAND_2_COUNT], dir: Vec3) -> Vec3;
pub fn sh_basis(dir: Vec3) -> [Real; SH_BAND_2_COUNT];
```

`diffuse(rgb)` sets `sh[0] = rgb / SH_C0` so `eval_sh` returns
`rgb` for any view direction — the v1 back-compat sanity check.

`sh_basis(dir)` evaluates the 9 basis functions at the normalised
direction. Order is band-major: `[Y_00, Y_{1-1}, Y_{10}, Y_{11},
Y_{2-2}, Y_{2-1}, Y_{20}, Y_{21}, Y_{22}]`. The bands 1 / 2 entries
match the standard Sloan-2008 convention (the same one used by
Kerbl 2023 "3D Gaussian Splatting" and the gaussian-splatting
reference renderer).

## v4 — Band 3 + diffuse/view split

Band 3 adds 7 more harmonics for finer angular detail:

```rust
pub const SH_BAND_3_COUNT:   usize = 16; // bands 0..=3
pub const SH_VIEW_DEP_COUNT: usize = 15; // band 1 + 2 + 3 (no diffuse)

pub struct GaussianSplatShBand3d {
    pub position: Vec3, pub scale: Vec3, pub rotation: Quat,
    pub sh:       [Vec3; SH_BAND_3_COUNT],
    pub opacity:  Real,
}

pub struct GaussianSplatShDiffuseView3d {
    pub position: Vec3, pub scale: Vec3, pub rotation: Quat,
    pub diffuse:  Vec3,                       // band-0 contribution
    pub view:     [Vec3; SH_VIEW_DEP_COUNT],  // bands 1..=3
    pub opacity:  Real,
}
```

`GaussianSplatShDiffuseView3d` separates the band-0 ("diffuse")
coefficient from the higher bands so callers can train the
direction-independent component independently from the
view-dependent ones — Kerbl 2023 finds this gives faster
convergence and avoids the diffuse term getting absorbed into
spurious specular highlights.

Sibling WGSL forward + backward kernels:
`splat_raytrace_sh.wgsl` / `splat_raytrace_sh_backward.wgsl`
(band 0..=2), `splat_raytrace_sh_band3.wgsl` /
`splat_raytrace_sh_band3_backward.wgsl` (band 0..=3), and a
diffuse/view variant pair. All four share the same α-composite
inner loop — the only differences are the SH coefficient count
and the view-direction evaluation formula.

## CPU oracle coverage

The CPU functions (`composite_*`, `eval_sh*`, `fd_grad_*_one_splat`)
exist primarily as oracles for the GPU forward + backward kernels.
The parity tests
(`tests/splat_raytrace_*.rs`) check that:

* GPU forward agrees with CPU forward to within `1e-5` for diffuse
  splats and `1e-4` for SH splats (single-precision `f32` causes
  the slightly looser SH bound).
* GPU backward agrees with CPU finite-difference gradient to within
  `1e-3` on random splat configurations.

Production rendering goes through the GPU kernels — the CPU paths
are not intended for production-size scene rendering (a 10k-splat
scene through `composite_forward_cpu` is seconds, not milliseconds).
