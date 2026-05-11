//! 3D Gaussian splat primitive — Stage 44 v1 (forward only).
//!
//! A [`GaussianSplat3d`] is an anisotropic 3D Gaussian with diffuse
//! RGB + per-splat opacity. Used by the differentiable raytrace
//! renderer ([`crate::sensors::SplatRaytracer`]) and the forward
//! WGSL kernel in `shaders/splat_raytrace.wgsl`.
//!
//! # Math (forward path, no adjoint yet)
//!
//! Newton-style 3DGS raycasting. For a splat with mean `μ`, rotation
//! `R`, anisotropic scale `s = (sx, sy, sz)`, opacity `o`, and a ray
//! `(p, d)`, we transform the ray into the splat's **canonical**
//! frame (the unit isotropic Gaussian):
//!
//! ```text
//! o_local = R⁻¹ · (p - μ) / s          (component-wise division)
//! d_local = R⁻¹ · d        / s
//! ```
//!
//! The maximum density along the canonical ray is attained at the
//! point of closest approach to the origin and equals
//!
//! ```text
//! ρ_max = exp(-½ · |d_local × o_local|² / |d_local|²)
//! ```
//!
//! (because the squared distance from the origin to the line
//! parameterised by `o_local + t·d_local` is `|d_local × o_local|² /
//! |d_local|²`). The per-splat α is `α = min(1, ρ_max · o)`. This is
//! the standard analytic approximation used in Newton's
//! `warp_raytrace/gaussians.py::canonical_ray_max_kernel_response`
//! and most 3DGS renderers — it captures the dominant integrand of
//! the line-integral over the Gaussian density without resorting to
//! a numerical erf evaluation.
//!
//! The hit distance along the **world** ray is the projection of
//! `(μ - p)` onto `d`, i.e. the parametric `t` where the ray reaches
//! the splat centre's projection (camera-distance sort, matching
//! Newton's default).
//!
//! Composition is front-to-back α-compositing:
//!
//! ```text
//! C_out = Σᵢ Cᵢ · αᵢ · ∏_{j<i}(1 - αⱼ),    T_out = ∏ᵢ (1 - αᵢ)
//! ```
//!
//! See `shaders/splat_raytrace.wgsl` for the GPU implementation.

use crate::math::{Quat, Real, Vec3};

/// A single anisotropic 3D Gaussian splat. Diffuse RGB only in v1;
/// view-dependent spherical-harmonic shading is deferred to v2.
///
/// Fields follow the prevailing 3DGS convention (`position`, `scale`,
/// `rotation`, per-channel colour, separate opacity). `color.w` is
/// reserved (read-only) for future use; the alpha that participates
/// in the α-composite is computed analytically from the ray geometry
/// times [`Self::opacity`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianSplat3d {
    /// World-space mean μ (m).
    pub position: Vec3,
    /// Per-axis stddev (m). Anisotropic; must be > 0 on every axis.
    pub scale: Vec3,
    /// Local-to-world rotation. The covariance is `R · diag(s²) · Rᵀ`.
    pub rotation: Quat,
    /// Diffuse RGB + reserved alpha slot. `.w` is currently unused by
    /// the renderer (kept for symmetry with v2 SH or pre-multiplied
    /// pipelines).
    pub color: [Real; 4],
    /// Bare opacity in `[0, 1]`. The composited alpha is `o · ρ_max`
    /// (see module docs).
    pub opacity: Real,
}

impl GaussianSplat3d {
    /// Convenience constructor. `color_rgb` is the diffuse colour;
    /// the reserved alpha slot is set to 1.0.
    #[must_use]
    pub fn new(
        position: Vec3,
        scale: Vec3,
        rotation: Quat,
        color_rgb: Vec3,
        opacity: Real,
    ) -> Self {
        Self {
            position,
            scale,
            rotation,
            color: [color_rgb.x, color_rgb.y, color_rgb.z, 1.0],
            opacity,
        }
    }

    /// Diffuse colour as a [`Vec3`].
    #[must_use]
    pub fn color_rgb(&self) -> Vec3 {
        Vec3::new(self.color[0], self.color[1], self.color[2])
    }
}

/// Conjugate (xyz negated, w preserved) — packed as a 4-component
/// tuple so we can hand it to [`quat_rotate_bac`] without round-trip
/// through `glam::DQuat`.
#[inline]
fn quat_conj_components(q: Quat) -> (Real, Real, Real, Real) {
    (-q.x, -q.y, -q.z, q.w)
}

/// `v + 2 qw (qv × v) + 2 (qv × (qv × v))` — exactly the WGSL
/// `quat_rotate` formula. Differs from `glam::DQuat * DVec3` when q
/// is **non-unit**; agrees for unit q. We need this exact form so the
/// CPU oracle and GPU kernel produce bit-identical results on an
/// FD-perturbed (off-sphere) quaternion — see the comment in
/// [`map_ray_to_canonical`].
#[inline]
fn quat_rotate_bac(q: (Real, Real, Real, Real), v: Vec3) -> Vec3 {
    let qv = Vec3::new(q.0, q.1, q.2);
    let qw = q.3;
    let t = qv.cross(v) * 2.0;
    v + t * qw + qv.cross(t)
}

/// Bring a ray into the splat's **canonical** (unit-isotropic) frame.
/// `(o_local, d_local)` such that the Gaussian becomes `N(0, I)`.
///
/// The returned direction is **not** normalised — callers using the
/// closest-approach formula need the raw length, see
/// [`canonical_ray_max_response`].
#[must_use]
pub fn map_ray_to_canonical(
    splat: &GaussianSplat3d,
    ray_origin: Vec3,
    ray_dir: Vec3,
) -> (Vec3, Vec3) {
    // Mirror the WGSL `quat_rotate` exactly so the CPU oracle and
    // GPU kernel agree even when the FD oracle perturbs `q` off the
    // unit sphere. Two subtleties:
    //   - WGSL uses `quat_conj` (NOT a true `Quat::inverse`), which
    //     for non-unit q has a different gradient than `conj/|q|²`.
    //   - glam's `Quat * Vec3` uses the canonical `q v q*` formula
    //     which equals `|q|² · R(q̂) v` for non-unit q. The WGSL
    //     `quat_rotate` instead evaluates the BAC-CAB expansion
    //     `v + 2 qw (qv × v) + 2 (qv × (qv × v))` which is **not**
    //     `q v q*` for non-unit q.
    // For the FD oracle to agree with the GPU adjoint, the CPU
    // forward must use the same BAC-CAB formula. The two collapse to
    // the same result for unit q, so this changes nothing in
    // production — it just gives the FD oracle a consistent
    // off-sphere extension.
    let centered = ray_origin - splat.position;
    let o_rot = quat_rotate_bac(quat_conj_components(splat.rotation), centered);
    let d_rot = quat_rotate_bac(quat_conj_components(splat.rotation), ray_dir);
    // Inverse anisotropic scale.
    let inv_s = Vec3::new(
        1.0 / splat.scale.x,
        1.0 / splat.scale.y,
        1.0 / splat.scale.z,
    );
    let o_local = Vec3::new(o_rot.x * inv_s.x, o_rot.y * inv_s.y, o_rot.z * inv_s.z);
    let d_local = Vec3::new(d_rot.x * inv_s.x, d_rot.y * inv_s.y, d_rot.z * inv_s.z);
    (o_local, d_local)
}

/// Maximum Gaussian response along the (already-canonical) ray.
///
/// Returns `exp(-½ · m²)` where `m²` is the squared distance from the
/// canonical-frame origin to the ray's closest point. The ray
/// direction does **not** need to be unit; the formula uses `|d × o|²
/// / |d|²` directly.
#[must_use]
pub fn canonical_ray_max_response(o_local: Vec3, d_local: Vec3) -> Real {
    let d2 = d_local.length_squared();
    if d2 <= 0.0 {
        // Degenerate ray — pretend we are at the origin.
        return (-0.5 * o_local.length_squared()).exp();
    }
    let crd = d_local.cross(o_local);
    let m2 = crd.length_squared() / d2;
    (-0.5 * m2).exp()
}

/// Convenience helper: the **per-splat α contribution** for a given
/// world-space ray. `α = min(1, o · ρ_max)`.
///
/// This is the analytic Gaussian density integral approximation used
/// by both the GPU shader and the CPU oracle. See module docs.
#[must_use]
pub fn gaussian_response(
    ray_origin: Vec3,
    ray_dir: Vec3,
    splat: &GaussianSplat3d,
) -> Real {
    let (o_l, d_l) = map_ray_to_canonical(splat, ray_origin, ray_dir);
    let rho = canonical_ray_max_response(o_l, d_l);
    (splat.opacity * rho).min(1.0).max(0.0)
}

/// Parametric `t` along the world-space ray at which the splat's
/// centre is closest to the ray. Used as the depth-sort key (matches
/// Newton's `CAMERA_DISTANCE` sorting mode).
#[must_use]
pub fn splat_camera_distance(
    ray_origin: Vec3,
    ray_dir: Vec3,
    splat: &GaussianSplat3d,
) -> Real {
    let d2 = ray_dir.length_squared();
    if d2 <= 0.0 {
        return 0.0;
    }
    let to_center = splat.position - ray_origin;
    to_center.dot(ray_dir) / d2
}

/// CPU oracle for the forward α-composite. Walks all splats, sorts
/// them front-to-back by [`splat_camera_distance`], and composites
/// their `α · color_rgb` contributions. Returns
/// `(rgb_out, transmittance_out)`. Used by tests and as the GPU
/// parity check.
///
/// Splats with `t ≤ 0` (behind the camera) are skipped. Splats with
/// `α < min_alpha` are also skipped — matches the 1/255 threshold
/// used in Newton's reference implementation.
#[must_use]
pub fn composite_forward_cpu(
    ray_origin: Vec3,
    ray_dir: Vec3,
    splats: &[GaussianSplat3d],
    max_distance: Real,
    min_alpha: Real,
) -> (Vec3, Real) {
    let mut hits: Vec<(Real, Real, Vec3)> = Vec::with_capacity(splats.len());
    for s in splats {
        let t = splat_camera_distance(ray_origin, ray_dir, s);
        if t <= 0.0 || t > max_distance {
            continue;
        }
        let alpha = gaussian_response(ray_origin, ray_dir, s);
        if alpha < min_alpha {
            continue;
        }
        hits.push((t, alpha, s.color_rgb()));
    }
    // Front-to-back: ascending t.
    hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

    let mut rgb = Vec3::ZERO;
    let mut transmittance: Real = 1.0;
    for (_t, alpha, color) in hits {
        rgb += color * alpha * transmittance;
        transmittance *= 1.0 - alpha;
        if transmittance < 1.0e-4 {
            break;
        }
    }
    (rgb, transmittance)
}

// ─────────────────────────────────────────────────────────────────
// Stage 44 v2 — finite-difference oracle for the per-splat gradient.
// Used by tests as the ground truth against which the analytic GPU
// backward kernel is validated.
// ─────────────────────────────────────────────────────────────────

/// Composited rgb for a single pixel given a slice of splats. Pure
/// CPU mirror of the forward kernel, restricted to a single ray.
/// Used by the FD oracle below.
#[must_use]
pub fn composite_rgb_cpu(
    ray_origin: Vec3,
    ray_dir: Vec3,
    splats: &[GaussianSplat3d],
    max_distance: Real,
    min_alpha: Real,
) -> Vec3 {
    let (rgb, _t) = composite_forward_cpu(ray_origin, ray_dir, splats, max_distance, min_alpha);
    rgb
}

/// Which parameter is being perturbed in [`fd_grad_one_splat`].
/// Encodes the natural splat parameterisation; for the quaternion
/// we just perturb one of the four components (no normalisation —
/// caller projects onto the tangent of the unit sphere if needed).
#[derive(Copy, Clone, Debug)]
pub enum SplatParam {
    PosX, PosY, PosZ,
    ScaleX, ScaleY, ScaleZ,
    RotX, RotY, RotZ, RotW,
    ColorR, ColorG, ColorB,
    Opacity,
}

/// Apply a perturbation `delta` to the chosen parameter of `s`.
fn perturb_splat(s: &GaussianSplat3d, p: SplatParam, delta: Real) -> GaussianSplat3d {
    let mut out = *s;
    match p {
        SplatParam::PosX => out.position.x += delta,
        SplatParam::PosY => out.position.y += delta,
        SplatParam::PosZ => out.position.z += delta,
        SplatParam::ScaleX => out.scale.x += delta,
        SplatParam::ScaleY => out.scale.y += delta,
        SplatParam::ScaleZ => out.scale.z += delta,
        SplatParam::RotX => {
            out.rotation = Quat::from_xyzw(s.rotation.x + delta, s.rotation.y, s.rotation.z, s.rotation.w)
        }
        SplatParam::RotY => {
            out.rotation = Quat::from_xyzw(s.rotation.x, s.rotation.y + delta, s.rotation.z, s.rotation.w)
        }
        SplatParam::RotZ => {
            out.rotation = Quat::from_xyzw(s.rotation.x, s.rotation.y, s.rotation.z + delta, s.rotation.w)
        }
        SplatParam::RotW => {
            out.rotation = Quat::from_xyzw(s.rotation.x, s.rotation.y, s.rotation.z, s.rotation.w + delta)
        }
        SplatParam::ColorR => out.color[0] += delta,
        SplatParam::ColorG => out.color[1] += delta,
        SplatParam::ColorB => out.color[2] += delta,
        SplatParam::Opacity => out.opacity += delta,
    }
    out
}

/// Central finite-difference `dL/d param` for a single splat at index
/// `splat_idx`. The loss is `L = dl_dc · rgb_out` (a linear functional
/// of the composited rgb — equivalent to having an upstream
/// `dL/dC_out = dl_dc` for this single-pixel example).
///
/// All other splats are held fixed. Returns one scalar gradient per
/// `SplatParam` variant supplied. Step size `h` should be `~1e-4`
/// for f64 — much smaller and round-off dominates; much larger and
/// second-order error wins.
#[must_use]
pub fn fd_grad_one_splat(
    splats: &[GaussianSplat3d],
    splat_idx: usize,
    params: &[SplatParam],
    ray_origin: Vec3,
    ray_dir: Vec3,
    dl_dc: Vec3,
    max_distance: Real,
    min_alpha: Real,
    h: Real,
) -> Vec<Real> {
    let mut out = Vec::with_capacity(params.len());
    let mut probe = splats.to_vec();
    let orig = splats[splat_idx];
    for &p in params {
        probe[splat_idx] = perturb_splat(&orig, p, h);
        let rgb_plus = composite_rgb_cpu(ray_origin, ray_dir, &probe, max_distance, min_alpha);
        probe[splat_idx] = perturb_splat(&orig, p, -h);
        let rgb_minus = composite_rgb_cpu(ray_origin, ray_dir, &probe, max_distance, min_alpha);
        probe[splat_idx] = orig;
        let dc = (rgb_plus - rgb_minus) * (1.0 / (2.0 * h));
        out.push(dl_dc.dot(dc));
    }
    out
}

// ─────────────────────────────────────────────────────────────────
// Stage 44 v3 — Spherical-harmonics (view-dependent) splats.
//
// A `GaussianSplatSh3d` keeps the same anisotropic Gaussian as
// `GaussianSplat3d`, but replaces the per-splat diffuse RGB with
// `SH_BAND_2_COUNT = 9` per-band-coefficient RGB triples (band 0, 1,
// 2 — total 9 vec3 coefficients ⇒ 27 floats per splat). Band 3 is
// deferred to v3+.
//
// Layout choice (interleaved per-coefficient): the per-splat SoA
// buffer is laid out as `array<vec3<f32>>` of length
// `splat_count * 9`, with the 9 coefficients for splat `i` occupying
// indices `[i*9 .. (i+1)*9]` in band-major order:
//
//   c[0]  = c_{0,0}    (band 0, scalar — the "diffuse" coefficient)
//   c[1]  = c_{1,-1}   (band 1, m = -1)
//   c[2]  = c_{1, 0}
//   c[3]  = c_{1,+1}
//   c[4]  = c_{2,-2}   (band 2, m = -2)
//   c[5]  = c_{2,-1}
//   c[6]  = c_{2, 0}
//   c[7]  = c_{2,+1}
//   c[8]  = c_{2,+2}
//
// Each coefficient is a `Vec3` (R, G, B); the channels share the SH
// basis but otherwise evaluate independently. The WGSL buffer mirrors
// this exactly (one `vec4<f32>` per coefficient, `.w` padding).
//
// Basis convention (Kerbl 2023, gsplat, Newton): the basis functions
// `Y_{l,m}(d)` for a unit view direction `d = (x, y, z)` are
//
//   Y_{0, 0} =  C0
//   Y_{1,-1} = -C1 · y
//   Y_{1, 0} =  C1 · z
//   Y_{1,+1} = -C1 · x
//   Y_{2,-2} =  C2a · x · y
//   Y_{2,-1} =  C2b · y · z
//   Y_{2, 0} =  C2c · (3z² − 1)         (assuming |d| = 1)
//   Y_{2,+1) =  C2d · x · z
//   Y_{2,+2} =  C2e · (x² − y²)
//
// where
//   C0  = 1/(2√π)                ≈ 0.28209479177387814
//   C1  = √(3/(4π))              ≈ 0.4886025119029199
//   C2a =  √(15/π) / 2           ≈ 1.0925484305920792
//   C2b = −√(15/π) / 2           ≈ −1.0925484305920792
//   C2c =  √(5/π) / 4            ≈ 0.31539156525252005
//   C2d = −√(15/π) / 2           ≈ −1.0925484305920792
//   C2e =  √(15/π) / 4           ≈ 0.5462742152960396
//
// Setting only `c[0] = (r, g, b) / C0` recovers diffuse RGB `(r, g, b)`
// — matches the band-0 sanity check in the parity tests.

/// Number of SH coefficients per channel for bands 0..=2 (= 9).
pub const SH_BAND_2_COUNT: usize = 9;

/// Band-0 normalisation `1 / (2·√π)`.
pub const SH_C0: Real = 0.28209479177387814;
/// Band-1 normalisation `√(3 / (4·π))`.
pub const SH_C1: Real = 0.4886025119029199;
/// Band-2 coefficient `½·√(15/π)`.
pub const SH_C2A: Real = 1.0925484305920792;
/// Band-2 coefficient `−½·√(15/π)`.
pub const SH_C2B: Real = -1.0925484305920792;
/// Band-2 coefficient `¼·√(5/π)`.
pub const SH_C2C: Real = 0.31539156525252005;
/// Band-2 coefficient `−½·√(15/π)`.
pub const SH_C2D: Real = -1.0925484305920792;
/// Band-2 coefficient `¼·√(15/π)`.
pub const SH_C2E: Real = 0.5462742152960396;

/// Anisotropic 3D Gaussian with view-dependent **spherical-harmonic**
/// colour (Stage 44 v3). Geometry (mean / scale / rotation / opacity)
/// is identical to [`GaussianSplat3d`]; the diffuse RGB is replaced by
/// 9 per-band coefficient triples (band 0 / 1 / 2). See the module
/// docs for the basis convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianSplatSh3d {
    /// World-space mean μ (m).
    pub position: Vec3,
    /// Per-axis stddev (m). Anisotropic; must be > 0 on every axis.
    pub scale: Vec3,
    /// Local-to-world rotation. Same convention as
    /// [`GaussianSplat3d::rotation`].
    pub rotation: Quat,
    /// 9 SH coefficient triples (one per (band, m) up to band 2,
    /// band-major order — see module docs). `sh[0]` is the band-0
    /// "diffuse" coefficient; `sh[1..=3]` is band 1; `sh[4..=8]` is
    /// band 2.
    pub sh: [Vec3; SH_BAND_2_COUNT],
    /// Bare opacity in `[0, 1]`. Composited α is `o · ρ_max`.
    pub opacity: Real,
}

impl GaussianSplatSh3d {
    /// Construct a band-0-only ("diffuse") SH splat from a plain RGB
    /// colour. Sets `sh[0] = rgb / SH_C0` so that
    /// [`eval_sh`] returns `rgb` for any view direction; all other
    /// coefficients are zero. Mirrors the v1 `GaussianSplat3d::new`
    /// behaviour.
    #[must_use]
    pub fn diffuse(
        position: Vec3,
        scale: Vec3,
        rotation: Quat,
        color_rgb: Vec3,
        opacity: Real,
    ) -> Self {
        let mut sh = [Vec3::ZERO; SH_BAND_2_COUNT];
        sh[0] = color_rgb * (1.0 / SH_C0);
        Self { position, scale, rotation, sh, opacity }
    }

    /// Construct from explicit SH coefficients.
    #[must_use]
    pub fn from_sh(
        position: Vec3,
        scale: Vec3,
        rotation: Quat,
        sh: [Vec3; SH_BAND_2_COUNT],
        opacity: Real,
    ) -> Self {
        Self { position, scale, rotation, sh, opacity }
    }

    /// Evaluate the per-channel colour at a view direction `dir`. The
    /// direction need not be unit length — [`eval_sh`] normalises it
    /// internally (matches the WGSL kernel).
    #[must_use]
    pub fn eval_color(&self, dir: Vec3) -> Vec3 {
        eval_sh(&self.sh, dir)
    }

    /// "Drop SH → diffuse" projection. Returns the band-0 contribution
    /// `c_{0,0} · Y_{0,0} = SH_C0 · sh[0]` evaluated at any direction
    /// (band 0 is direction-invariant). Used for the v1 back-compat
    /// sanity check.
    #[must_use]
    pub fn diffuse_component(&self) -> Vec3 {
        self.sh[0] * SH_C0
    }
}

/// Evaluate the SH basis at view direction `dir` and dot it with the
/// per-band-coefficient triples. Returns the per-channel colour. If
/// `dir` has zero length, returns the band-0 component only (the
/// limit of `eval_sh` as |dir| → 0 with constant first-order terms is
/// undefined, but we keep this branch consistent with the WGSL).
#[must_use]
pub fn eval_sh(coeffs: &[Vec3; SH_BAND_2_COUNT], dir: Vec3) -> Vec3 {
    let basis = sh_basis(dir);
    let mut out = Vec3::ZERO;
    for i in 0..SH_BAND_2_COUNT {
        out += coeffs[i] * basis[i];
    }
    out
}

/// Evaluate the 9 SH basis functions at the *normalised* `dir`. If
/// `dir` is zero-length, returns `[C0, 0, 0, 0, ...]` (band-0 only).
#[must_use]
pub fn sh_basis(dir: Vec3) -> [Real; SH_BAND_2_COUNT] {
    let len_sq = dir.length_squared();
    if len_sq <= 0.0 {
        let mut out = [0.0; SH_BAND_2_COUNT];
        out[0] = SH_C0;
        return out;
    }
    let inv_len = 1.0 / len_sq.sqrt();
    let x = dir.x * inv_len;
    let y = dir.y * inv_len;
    let z = dir.z * inv_len;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    [
        // band 0
        SH_C0,
        // band 1 (Y_1-1, Y_10, Y_11)
        -SH_C1 * y,
        SH_C1 * z,
        -SH_C1 * x,
        // band 2 (Y_2-2 .. Y_22)
        SH_C2A * (x * y),
        SH_C2B * (y * z),
        SH_C2C * (3.0 * zz - 1.0),
        SH_C2D * (x * z),
        SH_C2E * (xx - yy),
    ]
}

/// CPU mirror of the SH α-composite forward (mirrors
/// [`composite_forward_cpu`] for diffuse splats). The view direction
/// used to evaluate each splat's SH colour is the *ray's direction*
/// — matches the WGSL kernel (one direction per pixel, not one per
/// splat hit).
#[must_use]
pub fn composite_sh_forward_cpu(
    ray_origin: Vec3,
    ray_dir: Vec3,
    splats: &[GaussianSplatSh3d],
    max_distance: Real,
    min_alpha: Real,
) -> (Vec3, Real) {
    let mut hits: Vec<(Real, Real, Vec3)> = Vec::with_capacity(splats.len());
    for s in splats {
        let geom = GaussianSplat3d {
            position: s.position,
            scale: s.scale,
            rotation: s.rotation,
            color: [0.0, 0.0, 0.0, 1.0],
            opacity: s.opacity,
        };
        let t = splat_camera_distance(ray_origin, ray_dir, &geom);
        if t <= 0.0 || t > max_distance {
            continue;
        }
        let alpha = gaussian_response(ray_origin, ray_dir, &geom);
        if alpha < min_alpha {
            continue;
        }
        let color = eval_sh(&s.sh, ray_dir);
        hits.push((t, alpha, color));
    }
    hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

    let mut rgb = Vec3::ZERO;
    let mut transmittance: Real = 1.0;
    for (_t, alpha, color) in hits {
        rgb += color * alpha * transmittance;
        transmittance *= 1.0 - alpha;
        if transmittance < 1.0e-4 {
            break;
        }
    }
    (rgb, transmittance)
}

/// Convenience wrapper: returns only the composited rgb (used by the
/// FD oracle for SH gradients).
#[must_use]
pub fn composite_sh_rgb_cpu(
    ray_origin: Vec3,
    ray_dir: Vec3,
    splats: &[GaussianSplatSh3d],
    max_distance: Real,
    min_alpha: Real,
) -> Vec3 {
    composite_sh_forward_cpu(ray_origin, ray_dir, splats, max_distance, min_alpha).0
}

/// Identifies a single SH-coefficient scalar — used by the FD oracle
/// for gradient checks. `(coeff_idx, channel)` with `coeff_idx ∈
/// 0..9` and `channel ∈ 0..3`.
#[derive(Copy, Clone, Debug)]
pub struct ShCoeffParam {
    /// Coefficient index in `0..SH_BAND_2_COUNT`.
    pub coeff: usize,
    /// Channel `0 = R, 1 = G, 2 = B`.
    pub channel: usize,
}

fn perturb_sh(s: &GaussianSplatSh3d, p: ShCoeffParam, delta: Real) -> GaussianSplatSh3d {
    let mut out = *s;
    let v = match p.channel {
        0 => &mut out.sh[p.coeff].x,
        1 => &mut out.sh[p.coeff].y,
        2 => &mut out.sh[p.coeff].z,
        _ => unreachable!("invalid channel"),
    };
    *v += delta;
    out
}

/// Central finite-difference `dL/d sh_coeff[coeff].channel` for a
/// single splat. Mirrors [`fd_grad_one_splat`] but ranges over SH
/// coefficient slots instead of `SplatParam`.
#[must_use]
pub fn fd_grad_sh_one_splat(
    splats: &[GaussianSplatSh3d],
    splat_idx: usize,
    params: &[ShCoeffParam],
    ray_origin: Vec3,
    ray_dir: Vec3,
    dl_dc: Vec3,
    max_distance: Real,
    min_alpha: Real,
    h: Real,
) -> Vec<Real> {
    let mut out = Vec::with_capacity(params.len());
    let mut probe = splats.to_vec();
    let orig = splats[splat_idx];
    for &p in params {
        probe[splat_idx] = perturb_sh(&orig, p, h);
        let rgb_plus = composite_sh_rgb_cpu(ray_origin, ray_dir, &probe, max_distance, min_alpha);
        probe[splat_idx] = perturb_sh(&orig, p, -h);
        let rgb_minus = composite_sh_rgb_cpu(ray_origin, ray_dir, &probe, max_distance, min_alpha);
        probe[splat_idx] = orig;
        let dc = (rgb_plus - rgb_minus) * (1.0 / (2.0 * h));
        out.push(dl_dc.dot(dc));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_splat(centre: Vec3, opacity: Real) -> GaussianSplat3d {
        GaussianSplat3d::new(centre, Vec3::ONE, Quat::IDENTITY, Vec3::new(1.0, 0.0, 0.0), opacity)
    }

    #[test]
    fn response_peaks_through_center() {
        let s = unit_splat(Vec3::ZERO, 1.0);
        // Ray through origin → max response should be exp(0) = 1.0.
        let r = gaussian_response(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, &s);
        assert!((r - 1.0).abs() < 1e-9, "got {r}");
    }

    #[test]
    fn response_falls_off_at_one_sigma() {
        let s = unit_splat(Vec3::ZERO, 1.0);
        // Ray offset 1 unit (= 1σ) sideways. Closest approach = 1,
        // max response = exp(-0.5) ≈ 0.6065.
        let r = gaussian_response(Vec3::new(1.0, 0.0, -5.0), Vec3::Z, &s);
        assert!((r - (-0.5_f64).exp()).abs() < 1e-9, "got {r}");
    }

    #[test]
    fn anisotropic_scaling_affects_response() {
        // Splat is wide along x (σx = 4) but narrow along y (σy = 0.5).
        // A ray offset 1 unit along x sees ~exp(-0.5·(1/4)²)·1.
        let s = GaussianSplat3d::new(
            Vec3::ZERO,
            Vec3::new(4.0, 0.5, 1.0),
            Quat::IDENTITY,
            Vec3::ONE,
            1.0,
        );
        let r = gaussian_response(Vec3::new(1.0, 0.0, -5.0), Vec3::Z, &s);
        let expected = (-0.5_f64 * (1.0 / 4.0_f64).powi(2)).exp();
        assert!((r - expected).abs() < 1e-9, "got {r}, expected {expected}");
    }

    #[test]
    fn sh_band0_diffuse_parity() {
        // Build a band-0-only SH splat with `sh[0] = rgb / SH_C0` and
        // verify its evaluated colour matches the diffuse RGB for an
        // arbitrary direction. Then compare against the v1 diffuse
        // splat composite.
        let rgb = Vec3::new(0.4, 0.6, 0.2);
        let diffuse = GaussianSplat3d::new(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, rgb, 0.6);
        let sh = GaussianSplatSh3d::diffuse(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, rgb, 0.6);
        let dir = Vec3::new(0.3, -0.5, 1.0).normalize();
        let c_sh = sh.eval_color(dir);
        assert!((c_sh - rgb).length() < 1e-12, "sh = {c_sh:?}, rgb = {rgb:?}");

        let ro = Vec3::new(0.0, 0.0, -5.0);
        let rd = Vec3::Z;
        let (rgb_v1, _) = composite_forward_cpu(ro, rd, &[diffuse], 100.0, 0.0);
        let (rgb_v3, _) = composite_sh_forward_cpu(ro, rd, &[sh], 100.0, 0.0);
        let drgb = (rgb_v1 - rgb_v3).length();
        assert!(drgb < 1e-9, "v1 {rgb_v1:?} vs sh band-0 {rgb_v3:?}, Δ = {drgb}");
    }

    #[test]
    fn sh_band1_view_dependent() {
        // Pure band-1 (m=0) coefficient on the red channel — colour
        // should be `c · SH_C1 · z` where z is the normalised view
        // direction's z-component. Check this varies with direction.
        let mut sh = [Vec3::ZERO; SH_BAND_2_COUNT];
        sh[2] = Vec3::new(1.0, 0.0, 0.0); // Y_{1,0}
        let dir_up = Vec3::new(0.0, 0.0, 1.0);
        let dir_side = Vec3::new(1.0, 0.0, 0.0);
        let c_up = eval_sh(&sh, dir_up);
        let c_side = eval_sh(&sh, dir_side);
        // c_up.x should equal SH_C1 ≈ 0.4886; c_side.x should be 0.
        assert!((c_up.x - SH_C1).abs() < 1e-9, "c_up = {c_up:?}");
        assert!(c_side.x.abs() < 1e-9, "c_side = {c_side:?}");
        // Sanity: the difference signals view dependence.
        assert!((c_up.x - c_side.x).abs() > 1e-3);
    }

    #[test]
    fn front_to_back_composite_two_splats() {
        // Two unit splats at (0,0,0) and (0,0,5) — both peak through.
        // Front splat α=0.5, red; back splat α=0.7, green.
        let s0 = GaussianSplat3d::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::ONE,
            Quat::IDENTITY,
            Vec3::new(1.0, 0.0, 0.0),
            0.5,
        );
        let s1 = GaussianSplat3d::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ONE,
            Quat::IDENTITY,
            Vec3::new(0.0, 1.0, 0.0),
            0.7,
        );
        let (rgb, t) =
            composite_forward_cpu(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, &[s0, s1], 100.0, 0.0);
        // C = (1,0,0)·0.5 + (0,1,0)·0.7·(1-0.5) = (0.5, 0.35, 0).
        assert!((rgb.x - 0.5).abs() < 1e-9, "rgb.x = {}", rgb.x);
        assert!((rgb.y - 0.35).abs() < 1e-9, "rgb.y = {}", rgb.y);
        // T = (1-0.5)*(1-0.7) = 0.15.
        assert!((t - 0.15).abs() < 1e-9, "t = {t}");
    }
}
