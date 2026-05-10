//! Stage 37.7 — finite-difference adjoint for the raycast pipeline.
//!
//! Forward stays the same as the GPU [`crate::sensors::Raycaster`]
//! (sphere-vs-ray closest-hit), but here we run a compact CPU
//! implementation parameterised by a flat `Vec<Real>` so we can call
//! `f(params + ε) − f(params − ε)` cheaply for central FD.
//!
//! v1 ships **finite-difference** adjoints only. Analytic adjoints
//! (closed-form `∂t_hit / ∂param` per primitive) are deferred to
//! Stage 37.7.1; the FD path serves as the oracle for that follow-up.
//!
//! Mirrors `xtra/warp/warp/examples/optim/example_diffray.py` —
//! gradient-descent on a shape's pose to match a target depth image.
//! Where the Warp example uses `wp.Tape` over a triangle mesh, our v1
//! uses analytic sphere primitives + FD: triangle BVH adjoints are
//! significantly more book-keeping than the `Stage 37.7` contract
//! requires.
//!
//! # Conventions
//!
//! - **Image** = flat `Vec<Real>` of per-ray hit distances. Misses
//!   carry [`Real::INFINITY`]. The L2 loss skips any pixel where
//!   either side is `∞` — so silhouettes don't blow up the residual.
//! - **Params** are unstructured: each [`SphereParam`] consumes 3
//!   slots (xyz). Layout = `[s0.x, s0.y, s0.z, s1.x, s1.y, s1.z, ...]`.
//! - **Determinism** — forward depends only on `(params, rays)`; FD
//!   gradient depends only on `(params, target, rays, ε)`.

use crate::math::{Real, Vec3};

/// World-space ray. Direction is **not** required to be unit-length;
/// the trace returns `t` along whatever `direction` you pass in.
/// Mirroring `Raycaster::cast`'s convention here keeps interop with
/// the GPU sensor straightforward — the same `(origin, direction)`
/// pair gives the same hit distance.
#[derive(Copy, Clone, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }
}

/// Differentiable shape primitives supported in v1. Sphere is the
/// only one needed for the contract test ("optimise sphere position
/// to match target depth image"); plane is handy as a backdrop in
/// the viewer scene so rays past the sphere terminate visibly.
#[derive(Copy, Clone, Debug)]
pub enum SdfShape {
    /// Sphere — 3 differentiable params (centre xyz). Radius is held
    /// fixed (non-differentiable) so a single param vector indexes
    /// only the optimisation degrees of freedom.
    Sphere {
        /// Index of the centre's `x` slot in the flat `params` vector.
        /// `y` lives at `param_offset + 1`, `z` at `param_offset + 2`.
        param_offset: usize,
        /// Sphere radius (m). Held constant in v1 — extending to
        /// `radius` as a 4th differentiable param is a one-line edit
        /// once the test contract calls for it.
        radius: Real,
    },
    /// Infinite plane through the origin with outward normal `+Y`,
    /// offset by `height` along that normal. Non-differentiable in
    /// v1 — included as a backdrop so the depth image doesn't go to
    /// `∞` everywhere when the optimised sphere drifts off-screen.
    PlaneY { height: Real },
    /// Infinite plane perpendicular to `+Z` at `z = depth`. Used as a
    /// backdrop for orthographic-down-Z camera setups (the contract
    /// tests + the viewer scene); without it every silhouette pixel
    /// would degenerate to `∞` and the L2 loss would skip them all.
    PlaneZ { depth: Real },
}

/// Scene description for the differentiable raycast — a flat list of
/// shape primitives. Build once, call [`RayScene::trace`] every time
/// the parameters change.
#[derive(Clone, Debug, Default)]
pub struct RayScene {
    pub shapes: Vec<SdfShape>,
}

impl RayScene {
    #[must_use]
    pub fn new() -> Self {
        Self { shapes: Vec::new() }
    }

    /// Add a sphere with its centre stored at `params[param_offset..param_offset+3]`.
    pub fn add_sphere(&mut self, param_offset: usize, radius: Real) {
        self.shapes.push(SdfShape::Sphere { param_offset, radius });
    }

    /// Add a `+Y`-axis ground plane at the given height.
    pub fn add_plane_y(&mut self, height: Real) {
        self.shapes.push(SdfShape::PlaneY { height });
    }

    /// Add a `+Z`-axis backdrop plane at `z = depth`.
    pub fn add_plane_z(&mut self, depth: Real) {
        self.shapes.push(SdfShape::PlaneZ { depth });
    }

    /// Forward: trace `rays` through the scene; return per-ray hit
    /// distance (or [`Real::INFINITY`] for misses). The result vector
    /// has the same length as `rays` — element `i` corresponds to
    /// ray `i`. Hits at `t < 0` are treated as misses (intersections
    /// behind the ray origin).
    pub fn trace(&self, params: &[Real], rays: &[Ray]) -> Vec<Real> {
        rays.iter()
            .map(|ray| self.closest_hit(params, ray))
            .collect()
    }

    fn closest_hit(&self, params: &[Real], ray: &Ray) -> Real {
        let mut best = Real::INFINITY;
        for shape in &self.shapes {
            let t = match *shape {
                SdfShape::Sphere { param_offset, radius } => {
                    let centre = Vec3::new(
                        params[param_offset],
                        params[param_offset + 1],
                        params[param_offset + 2],
                    );
                    ray_sphere_t(ray, centre, radius)
                }
                SdfShape::PlaneY { height } => ray_plane_y_t(ray, height),
                SdfShape::PlaneZ { depth } => ray_plane_z_t(ray, depth),
            };
            if t > 0.0 && t < best {
                best = t;
            }
        }
        best
    }
}

/// Closest-hit distance for ray vs sphere. Returns `+∞` on miss.
/// Standard quadratic formula with the smaller positive root; if both
/// roots are negative (sphere is behind the ray) the result is `+∞`.
#[inline]
fn ray_sphere_t(ray: &Ray, centre: Vec3, radius: Real) -> Real {
    let oc = ray.origin - centre;
    let a = ray.direction.dot(ray.direction);
    if a <= 1.0e-30 {
        return Real::INFINITY;
    }
    let b = oc.dot(ray.direction);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - a * c;
    if disc < 0.0 {
        return Real::INFINITY;
    }
    let sqrt_d = disc.sqrt();
    let t0 = (-b - sqrt_d) / a;
    if t0 > 0.0 {
        return t0;
    }
    let t1 = (-b + sqrt_d) / a;
    if t1 > 0.0 {
        return t1;
    }
    Real::INFINITY
}

/// Closest-hit `t` for ray vs the plane `y = height` with normal `+Y`.
/// Returns `+∞` for parallel / behind-the-origin cases.
#[inline]
fn ray_plane_y_t(ray: &Ray, height: Real) -> Real {
    let denom = ray.direction.y;
    if denom.abs() <= 1.0e-30 {
        return Real::INFINITY;
    }
    let t = (height - ray.origin.y) / denom;
    if t > 0.0 {
        t
    } else {
        Real::INFINITY
    }
}

/// Closest-hit `t` for ray vs the plane `z = depth` with normal `+Z`.
#[inline]
fn ray_plane_z_t(ray: &Ray, depth: Real) -> Real {
    let denom = ray.direction.z;
    if denom.abs() <= 1.0e-30 {
        return Real::INFINITY;
    }
    let t = (depth - ray.origin.z) / denom;
    if t > 0.0 {
        t
    } else {
        Real::INFINITY
    }
}

/// L2 loss `J = ½ Σ (current_i − target_i)²` skipping any pixel where
/// either side is `±∞`. The `½` factor matches Newton's optimisation
/// utilities (cancels the 2 in the gradient) and is consistent with
/// the example_diffray reference.
#[must_use]
pub fn compute_loss_l2(current: &[Real], target: &[Real]) -> Real {
    debug_assert_eq!(current.len(), target.len(), "loss: image size mismatch");
    let mut sum = 0.0;
    for (c, t) in current.iter().zip(target.iter()) {
        if c.is_finite() && t.is_finite() {
            let d = c - t;
            sum += d * d;
        }
    }
    0.5 * sum
}

/// Pseudo-Huber loss `J = Σ δ²(√(1 + (d/δ)²) − 1)` skipping any
/// pixel where either side is `±∞`. Smooth everywhere (no kink at
/// `d=0`), with the gradient saturating to `±δ` for large residuals
/// — useful when silhouette mismatches produce large outlier
/// residuals that would otherwise dominate the SGD step. Mirrors the
/// pseudo-Huber kernel in `example_diffray.py`.
#[must_use]
pub fn compute_loss_pseudo_huber(current: &[Real], target: &[Real], delta: Real) -> Real {
    debug_assert_eq!(current.len(), target.len(), "loss: image size mismatch");
    let d2 = delta * delta;
    let mut sum = 0.0;
    for (c, t) in current.iter().zip(target.iter()) {
        if c.is_finite() && t.is_finite() {
            let r = (c - t) / delta;
            sum += d2 * ((1.0 + r * r).sqrt() - 1.0);
        }
    }
    sum
}

/// Central finite-difference gradient of [`compute_loss_l2`] with
/// respect to each scene parameter.
///
/// For each parameter `k`:
///
/// ```text
/// J_plus  = loss(trace(params with params[k] += eps), target)
/// J_minus = loss(trace(params with params[k] -= eps), target)
/// grad[k] = (J_plus - J_minus) / (2 eps)
/// ```
///
/// Cost is `O(P · N_rays)` — fine for a handful of params, slow for
/// hundreds. Acceptable for the v1 contract (sphere = 3 params).
///
/// `params` is restored to its input value before this function returns.
pub fn compute_grad_raycast(
    scene: &RayScene,
    params: &mut [Real],
    target: &[Real],
    rays: &[Ray],
    eps: Real,
) -> Vec<Real> {
    let mut grad = vec![0.0; params.len()];
    for k in 0..params.len() {
        let original = params[k];

        params[k] = original + eps;
        let img_plus = scene.trace(params, rays);
        let j_plus = compute_loss_l2(&img_plus, target);

        params[k] = original - eps;
        let img_minus = scene.trace(params, rays);
        let j_minus = compute_loss_l2(&img_minus, target);

        params[k] = original;
        grad[k] = (j_plus - j_minus) / (2.0 * eps);
    }
    grad
}

/// Convenience: forward + L2 loss in a single call. Returns
/// `(image, loss)`; useful in viewer / training loops where you want
/// both the rendered image and the scalar metric.
pub fn forward_with_loss(
    scene: &RayScene,
    params: &[Real],
    target: &[Real],
    rays: &[Ray],
) -> (Vec<Real>, Real) {
    let image = scene.trace(params, rays);
    let loss = compute_loss_l2(&image, target);
    (image, loss)
}

// ── Tests ─────────────────────────────────────────────────────────
// Smoke-check the forward + a few invariants of the loss helper.
// The contract tests for the FD gradient + gradient descent live in
// `crates/molla/tests/raycast_adjoint.rs` (per the Stage 37.7 spec).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_trace_hits_along_minus_z() {
        let mut scene = RayScene::new();
        scene.add_sphere(0, 0.5);
        let params = vec![0.0, 0.0, -2.0];
        let rays = vec![Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0))];
        let image = scene.trace(&params, &rays);
        // Hit on the near side of a unit-radius/2 sphere centred at z=-2.
        assert!((image[0] - 1.5).abs() < 1.0e-9);
    }

    #[test]
    fn sphere_miss_yields_infinity() {
        let mut scene = RayScene::new();
        scene.add_sphere(0, 0.5);
        let params = vec![0.0, 0.0, -2.0];
        let rays = vec![Ray::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0))];
        let image = scene.trace(&params, &rays);
        assert!(image[0].is_infinite());
    }

    #[test]
    fn loss_skips_infinities() {
        let cur = vec![1.0, Real::INFINITY, 3.0];
        let tgt = vec![1.0, 5.0, 3.5];
        // Only the (3.0 − 3.5)² term should count.
        let l = compute_loss_l2(&cur, &tgt);
        assert!((l - 0.5 * 0.25).abs() < 1.0e-12);
    }
}
