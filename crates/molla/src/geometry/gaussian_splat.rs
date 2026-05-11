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
    let inv_rot = splat.rotation.inverse();
    let centered = ray_origin - splat.position;
    let o_rot = inv_rot * centered;
    let d_rot = inv_rot * ray_dir;
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
