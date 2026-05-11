# Differentiable Gaussian splat renderer

Stage 44 is molla's differentiable 3D Gaussian Splat (3DGS) raytrace
renderer — an end-to-end on-GPU rendering pipeline whose reverse
pass scatters per-splat gradients (position, scale, rotation,
colour, opacity, spherical-harmonic coefficients) from a pixel loss.
It is the only fully GPU-resident differentiable path in molla today
and the one that drove the per-splat atomic-CAS scatter idiom now
reused by the FEM and contact pipelines.

Source: `crates/molla/src/sensors/` (Rust orchestrators reuse the
camera sensor scaffolding) and the WGSL shaders:

| Stage | Shader                                          | Purpose                          |
|-------|-------------------------------------------------|----------------------------------|
| v1    | `splat_raytrace.wgsl`                           | Forward α-composited raytrace    |
| v2    | `splat_raytrace_backward.wgsl`                  | Reverse adjoint (position/scale/rot/color/opacity) |
| v3    | `splat_raytrace_sh.wgsl` + `*_sh_backward.wgsl` | SH bands 0–2 (9 coeffs/channel)  |
| v4    | `splat_raytrace_sh_band3.wgsl` + backward       | SH band 3 (16 coeffs/channel)    |

The numbered commits are:

- **v1 forward** (`d4583a7`) — top-K depth-sorted α-composite.
- **v2 reverse** (`237caa2`) — pose-recovery Adam loop ships.
- **v3 SH** (`00d8083`) — diffuse + ambient view-dependence,
  bands 0–2.
- **v4 band 3** (`121c9d3`) — directional specular highlights via
  the cubic basis, splitting diffuse + view-dependent terms
  (Kerbl 2023).

## Forward — α-composited Gaussian raytrace

One thread per pixel. Each thread:

1. Walks the full splat array.
2. For each splat, transforms the ray into the splat's canonical
   (unit-isotropic) frame using the inverse rotation + scale, then
   evaluates the response in that frame:

   ```text
   m² = ‖d × o‖² / ‖d‖²        # closest-approach distance², canonical frame
   ρ  = exp(−½ · m²)
   α  = opacity · ρ
   ```

3. Keeps a top-K mini-sort of the closest hits in front-to-back
   order (default `MAX_HITS = 16`, matching Newton's
   `gaussians_max_num_hits`).
4. α-composites them:

   ```text
   T_0 = 1
   for i in 0..K:
       C    += T_i · α_i · c_i
       T_{i+1} = T_i · (1 − α_i)
       if T_{i+1} < MIN_TRANSMITTANCE: break
   alpha_out = 1 − T_K
   ```

`MIN_TRANSMITTANCE = 0.003` (Newton's
`gaussians_min_transmittance`). Per-splat α below `min_alpha = 1/255`
is discarded — also matching Newton.

## Reverse — gradient chain

The reverse kernel runs **one thread per pixel** and re-runs the
exact same deterministic top-K insertion sort as the forward (the
front-to-back order is uniquely determined by the splat array + ray,
so no extra storage is needed). Then it walks the hit list
**back-to-front** to accumulate per-splat gradients via the standard
3DGS α-composite recurrence.

### α-composite adjoint

Let `c_i` be the (possibly view-dependent) RGB of hit `i`, `α_i`
its alpha, `T_i = ∏_{j<i}(1 − α_j)` the front-prefix transmittance,
and `S_after = Σ_{j>i} c_j α_j T_j` the RGB that piles up behind hit
`i`. The reverse pass uses:

```text
dL/dc_i = α_i · T_i · dL/dC_out
dL/dα_i = T_i · c_i · dL/dC_out  −  S_after · dL/dC_out / (1 − α_i)
```

The `−S_after / (1 − α_i)` term is the "what the splats behind me
would have contributed if I hadn't blocked them" correction — the
recurrence is *closed-form* over the back-to-front walk because we
can accumulate `S_after` in O(1) per step.

The forward writes `alpha_out = 1 − T_final`. v2 ignores
`dL/dalpha_out` because the pose-recovery loss only uses RGB;
extending to alpha-channel gradients is a one-line addition (an
extra `T_i · T_final / ((1 − α_i) · T_i)` term in `dL/dα_i`).

### α → opacity, ρ, m²

```text
α = opacity · ρ,    ρ = exp(−½ m²)
dL/dopacity = ρ · dL/dα
dL/dρ       = opacity · dL/dα
dL/d(m²)    = −½ · ρ · dL/dρ
```

### m² → canonical frame → (scale, rotation, position)

In the canonical frame, `o_local = R⁻¹(o − c) ⊘ s` and
`d_local = R⁻¹ d ⊘ s` (where `R⁻¹ = R*` is the inverse quaternion,
`⊘` is elementwise divide, `s` is the per-splat anisotropic scale,
`c` is the centre).

```text
crd = d_local × o_local
m²  = ‖crd‖² / ‖d_local‖²
```

The chain rule through `crd`, `d_local`, `o_local` lands gradients
in `(s, R, c)`. The full per-component chain is in the shader; the
geometric form of each step is:

```text
dL/dcrd      = (2 / ‖d_local‖²) · crd · dL/d(m²) − (2 · m² / ‖d_local‖²) · d_local · dL/d(m²)
dL/d(d_local) = (BAC-CAB: dL/d(d × o) = o × dL/d(d × o)ᵀ … see below)
dL/d(o_local) = symmetric counterpart
```

### BAC-CAB quaternion derivative subtlety

The gradient of `R⁻¹ v` with respect to the rotation quaternion `R`
needs care. Using the BAC-CAB identity (a × (b × c) = b (a · c) − c (a · b))
gives:

```text
∂(R⁻¹ v) / ∂R = 2 (v × q̂ − q̂_w v − q̂ × (q̂ × v))
```

where `q̂` is the vector part of `R`. The shader unrolls this using
the explicit quaternion-rotate formula
`R⁻¹ v = v − 2 q̂ × (q̂ × v) + 2 q̂_w (v × q̂)` (matches
`raycast.wgsl`'s `quat_rotate(quat_conj(q), v)`) and differentiates
it term-by-term. The key subtlety: each `cross` invocation contributes
**two** non-commuting terms via BAC-CAB — one from the inner cross
and one from the outer — and the signs must be tracked carefully.
Newton's `gaussians.py` derives the same expression with the same
sign conventions.

### Atomic-CAS f32 scatter

WGSL has no native `atomic<f32>`. Per-splat gradients are scattered
via the standard `atomicCompareExchangeWeak` CAS loop on bit-cast
`u32` slots — the same idiom that `shaders/fem_assembly.wgsl` uses:

```text
atomic_add(slot, v):
  old = atomicLoad(&buf[slot])
  loop:
    new_bits = bitcast<u32>(bitcast<f32>(old) + v)
    r = atomicCompareExchangeWeak(&buf[slot], old, new_bits)
    if r.exchanged: return
    old = r.old_value
```

One thread per pixel ⇒ contention is proportional to the number of
pixels lit by any single splat. For the demo scenes this is
typically `< 100` ⇒ contention is mild and the CAS loops rarely
spin more than 2–3 times.

## v3 / v4 — Spherical harmonics

Bands 0–2 give 9 coefficients per channel × 3 channels = 27 SH
coefficients per splat. Band 3 adds 7 cubic-basis coefficients per
channel ⇒ 16 total × 3 = 48 SH coefficients per splat.

The view-dependent colour is evaluated against the unit view
direction `v̂` (camera → splat):

```text
c(v̂) = Σ_l Σ_m c_{l,m} · Y_{l,m}(v̂)
```

with `Y_{l,m}` the real spherical-harmonic basis. The basis
constants are the standard ones; v4 uses:

```text
SH_C0  =  0.28209479177387814
SH_C1  =  0.4886025119029199
SH_C2A =  1.0925484305920792
SH_C2B = -1.0925484305920792
SH_C2C =  0.31539156525252005
SH_C2D = -1.0925484305920792
SH_C2E =  0.5462742152960396
SH_C3A = -0.5900435899266435
SH_C3B =  2.890611442640554
SH_C3C = -0.4570457994644658
SH_C3D =  0.3731763325901154
SH_C3E = -0.4570457994644658
SH_C3F =  1.445305721320277
SH_C3G = -0.5900435899266435
```

Per Kerbl 2023, v4 splits the **diffuse** band-0 coefficient from
the **view-dependent** (bands 1+) coefficients so a static
ambient + dynamic specular can be edited independently. The reverse
kernel chain is identical to v2 for the geometric quantities;
only the colour assembly changes (the per-band `dL/dc_{l,m}` now
involves the SH basis terms evaluated at `v̂`).

## Pose-recovery Adam loop

The v2 ship-test runs an Adam optimisation that recovers a
3-DoF translation of a Gaussian splat cloud from a target RGB
image:

```text
for step in 0..n_iters:
    forward: render(splats translated by p) → pred_rgb
    loss:    L = ‖pred_rgb − target‖²  (per-pixel L2)
    reverse: dL/dp = scatter splat gradients, average position gradient over splats
    update:  p ← Adam(p, dL/dp)
```

Converges in ~200 iterations for typical scenes. Gradient
verification against finite differences passes at `1e−3` relative
tolerance for all geometric channels; SH gradients tolerate `1e−2`
because the basis-coefficient sums introduce more round-off.

## Cross-references

- [`Geometry / Gaussian splats`](../geometry/gaussian_splat.md) —
  splat-data model + forward rendering details.
- [`Sensors / TiledCamera`](../sensors/tiled_camera.md) — same ray-
  generation infrastructure feeds this kernel.
- [`Tape-based autograd`](./tape.md) — *not* used here: the splat
  reverse is fully hand-written and GPU-resident, the canonical
  `wp.func_grad`-style example for molla.
- [`Math / Optimizers`](../math/optim.md) — Adam shared with the
  rest of the autograd demos.
- References: Kerbl et al. 2023 ("3D Gaussian Splatting for
  Real-Time Radiance Field Rendering"), Newton `gaussians.py`.
