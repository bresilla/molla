//! H4 + H5 — Terrestrial LiDAR scan model: `Lidar` wraps a
//! [`Raycaster`] and lays out rays according to a (θ, φ) angular grid
//! plus an optional beam-divergence cone with `subray_count` sub-rays.
//!
//! The split is deliberate: `Raycaster` remains the low-level "fire
//! these rays, return these hits" primitive; `Lidar` is the
//! sensor-model layer on top — scan metadata in, range-image-shaped
//! scan out. The scan back-pointer lets downstream callers project
//! the cloud to a `(row, col)` index image (H6).
//!
//! Conventions (mirroring helios's `ScanMetadata`):
//!
//!   * `θ` (zenithal) is measured from the local "up" axis. `θ = 0`
//!     points along `up`; `θ = π/2` lies on the plane perpendicular
//!     to `up` (the equator of the scan sphere).
//!   * `φ` (azimuthal) is measured around `up`. `φ = 0` points along
//!     `forward`; `φ = π/2` along `forward × up`.
//!   * The scan is laid out **row-major over θ**: cell index
//!     `k = row * n_phi + col` with `row = i_θ`, `col = i_φ`. Each
//!     cell carries `subray_count` sub-rays back-to-back: the k-th
//!     sub-ray of cell `(row, col)` is at flat index
//!     `(row * n_phi + col) * subray_count + k`.
//!
//! Beam divergence is a property of the pulse cone (a real laser is
//! not a line). H5 ships cone sampling: for `subray_count > 1`, K-1
//! extra rays are sampled with a Fibonacci-spiral pattern inside the
//! cone of half-angle `beam_divergence`. The `Scan` exposes per-cell
//! `first_hit_at` / `last_hit_at` / `sub_hits_at` selectors so the
//! host filters first/last/all. The full per-pulse hit-stack along a
//! single sub-ray (canopy-edge + ground returns through a translucent
//! surface) is deferred to a kernel-level follow-up.

#![cfg(feature = "gpu")]

use crate::raycast::{RayHit, Raycaster};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_sim::{Model, State};

/// Parameters describing one terrestrial laser scan. Mirrors helios's
/// `ScanMetadata` (see `xtra/helios/plugins/lidar/include/LiDAR.h`).
///
/// Construct with [`ScanMetadata::new`] or by setting fields directly;
/// the field shape is public so callers can serialise / deserialise
/// scan recipes.
#[derive(Debug, Clone, Copy)]
pub struct ScanMetadata {
    /// World-space scanner position (m).
    pub origin: Vec3,
    /// Forward axis of the scanner. `φ = 0` points here. Need not be
    /// unit; will be re-orthonormalised internally.
    pub forward: Vec3,
    /// "Up" axis of the scanner — the scan's zenith. `θ = 0` points
    /// here. Need not be unit; will be re-orthonormalised internally.
    pub up: Vec3,
    /// Number of θ (zenithal) samples — the row count.
    pub n_theta: u32,
    /// Number of φ (azimuthal) samples — the column count.
    pub n_phi: u32,
    /// `(θ_min, θ_max)` in radians. Inclusive both ends; if `n_theta`
    /// is 1 the scan fires at `θ_min`.
    pub theta_range: (Real, Real),
    /// `(φ_min, φ_max)` in radians.
    pub phi_range: (Real, Real),
    /// Diameter of the laser beam at the exit aperture (m). Stored
    /// for downstream physics (e.g. range-dependent footprint); not
    /// yet sampled by H4.
    pub exit_diameter: Real,
    /// Half-angle of the beam-divergence cone (rad). Combined with
    /// `subray_count > 1`, this widens the per-pulse footprint via
    /// the H5 Fibonacci-spiral cone sampler.
    pub beam_divergence: Real,
    /// Number of sub-rays fired per pulse. `1` (default) = central
    /// ray only; `K > 1` adds `K - 1` Fibonacci-spiral cone samples
    /// inside the `beam_divergence` cone.
    pub subray_count: u32,
    /// Maximum ray distance (m). Hits past this are reported as
    /// misses.
    pub max_distance: Real,
}

impl ScanMetadata {
    /// Construct a scan recipe with sensible defaults: forward = `+Z`,
    /// up = `+Y`, zero beam divergence, zero exit diameter,
    /// `max_distance = 100` m. The caller must still set the angular
    /// grid.
    pub fn new(origin: Vec3, n_theta: u32, n_phi: u32) -> Self {
        Self {
            origin,
            forward: Vec3::new(0.0, 0.0, 1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            n_theta,
            n_phi,
            theta_range: (0.0, std::f64::consts::PI),
            phi_range: (0.0, 2.0 * std::f64::consts::PI),
            exit_diameter: 0.0,
            beam_divergence: 0.0,
            subray_count: 1,
            max_distance: 100.0,
        }
    }

    /// Sub-ray count clamped to at least 1.
    pub fn effective_subray_count(&self) -> u32 {
        self.subray_count.max(1)
    }

    /// Total ray budget, saturating for invalid metadata; use `validate` before allocation.
    pub fn ray_count(&self) -> u32 {
        self.cell_count()
            .saturating_mul(self.effective_subray_count())
    }

    /// Checked total ray budget for allocation and dispatch.
    pub fn checked_ray_count(&self) -> Result<u32> {
        self.n_theta
            .checked_mul(self.n_phi)
            .and_then(|cells| cells.checked_mul(self.effective_subray_count()))
            .ok_or_else(|| Error::Solver("ScanMetadata: ray count exceeds u32".into()))
    }

    /// Total cell count (`n_theta * n_phi`). The cell index `k` for
    /// `(row, col)` is `row * n_phi + col`. Saturates for invalid metadata.
    pub fn cell_count(&self) -> u32 {
        self.n_theta.saturating_mul(self.n_phi)
    }

    /// Validate the metadata against the invariants the scan layout
    /// assumes: `n_theta` / `n_phi` non-zero, theta/phi ranges
    /// well-ordered and finite, max-distance positive,
    /// beam-divergence in `[0, π/2)`. Returns an `Error::Solver` with
    /// a precise message on the first violated invariant.
    pub fn validate(&self) -> Result<()> {
        self.checked_ray_count()?;
        if self.n_theta == 0 || self.n_phi == 0 {
            return Err(Error::Solver(format!(
                "ScanMetadata: n_theta and n_phi must be non-zero (got {}, {})",
                self.n_theta, self.n_phi
            )));
        }
        for (name, range) in [
            ("theta_range", self.theta_range),
            ("phi_range", self.phi_range),
        ] {
            if !range.0.is_finite() || !range.1.is_finite() {
                return Err(Error::Solver(format!(
                    "ScanMetadata: {name} must be finite (got {:?})",
                    range
                )));
            }
            if range.0 > range.1 {
                return Err(Error::Solver(format!(
                    "ScanMetadata: {name} min > max ({} > {})",
                    range.0, range.1
                )));
            }
        }
        if !self.max_distance.is_finite() || self.max_distance <= 0.0 {
            return Err(Error::Solver(format!(
                "ScanMetadata: max_distance must be positive and finite (got {})",
                self.max_distance
            )));
        }
        if !self.beam_divergence.is_finite()
            || self.beam_divergence < 0.0
            || self.beam_divergence >= std::f64::consts::FRAC_PI_2
        {
            return Err(Error::Solver(format!(
                "ScanMetadata: beam_divergence must be in [0, π/2) (got {})",
                self.beam_divergence
            )));
        }
        if !self.exit_diameter.is_finite() || self.exit_diameter < 0.0 {
            return Err(Error::Solver(format!(
                "ScanMetadata: exit_diameter must be non-negative + finite (got {})",
                self.exit_diameter
            )));
        }
        // Forward / up must be finite and not parallel.
        let fmag = self.forward.x * self.forward.x
            + self.forward.y * self.forward.y
            + self.forward.z * self.forward.z;
        let umag = self.up.x * self.up.x + self.up.y * self.up.y + self.up.z * self.up.z;
        if !fmag.is_finite() || fmag < 1.0e-24 {
            return Err(Error::Solver(format!(
                "ScanMetadata: forward is zero / non-finite ({:?})",
                self.forward
            )));
        }
        if !umag.is_finite() || umag < 1.0e-24 {
            return Err(Error::Solver(format!(
                "ScanMetadata: up is zero / non-finite ({:?})",
                self.up
            )));
        }
        if (self.forward / fmag.sqrt())
            .cross(self.up / umag.sqrt())
            .length_squared()
            <= 1.0e-24
        {
            return Err(Error::Solver(
                "ScanMetadata: forward and up must not be parallel".into(),
            ));
        }
        Ok(())
    }

    /// Convert `(row, col)` to the central ray direction for that
    /// cell. The direction is returned **un-normalised** (it is a
    /// unit vector regardless, since it's a spherical-coord sample).
    pub fn rc_to_direction(&self, row: u32, col: u32) -> Vec3 {
        let (f, u, r) = self.local_frame();
        let theta = sample_axis(self.theta_range, self.n_theta, row);
        let phi = sample_axis(self.phi_range, self.n_phi, col);
        let st = theta.sin();
        let ct = theta.cos();
        let sp = phi.sin();
        let cp = phi.cos();
        Vec3::new(
            f.x * st * cp + u.x * ct + r.x * st * sp,
            f.y * st * cp + u.y * ct + r.y * st * sp,
            f.z * st * cp + u.z * ct + r.z * st * sp,
        )
    }

    /// Build an orthonormal `(forward, up, right)` frame from
    /// `self.forward` and `self.up`, with `right = up × forward`.
    pub(crate) fn local_frame(&self) -> (Vec3, Vec3, Vec3) {
        let f = normalise(self.forward, Vec3::new(0.0, 0.0, 1.0));
        let u_in = normalise(self.up, Vec3::new(0.0, 1.0, 0.0));
        // Project `up` perpendicular to `forward` so the frame is
        // orthogonal even if the caller gave a slightly-misaligned up.
        let dot = f.x * u_in.x + f.y * u_in.y + f.z * u_in.z;
        let u_ortho = normalise(
            Vec3::new(u_in.x - f.x * dot, u_in.y - f.y * dot, u_in.z - f.z * dot),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let r = normalise(cross(u_ortho, f), Vec3::new(1.0, 0.0, 0.0));
        (f, u_ortho, r)
    }
}

fn normalise(v: Vec3, fallback: Vec3) -> Vec3 {
    let m2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if m2 > 1.0e-24 {
        let inv = 1.0 / m2.sqrt();
        Vec3::new(v.x * inv, v.y * inv, v.z * inv)
    } else {
        fallback
    }
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn sample_axis(range: (Real, Real), n: u32, i: u32) -> Real {
    if n <= 1 {
        return range.0;
    }
    range.0 + (range.1 - range.0) * (i as Real / (n - 1) as Real)
}

/// A finished lidar scan. Hits are stored row-major over θ with
/// `subray_count` consecutive sub-rays per cell; index with
/// [`Scan::hit_at`] / [`Scan::sub_hits_at`] for the `(row, col)`
/// lookups.
#[derive(Debug, Clone)]
pub struct Scan {
    /// The metadata used to lay out the scan — kept so callers can
    /// project the cloud back into a `(row, col)` range image.
    pub metadata: ScanMetadata,
    /// `n_theta * n_phi * subray_count` hits, laid out
    /// `(row * n_phi + col) * subray_count + k`. The k-th entry for
    /// each cell is the k-th sub-ray (k = 0 is the central ray).
    pub hits: Vec<RayHit>,
}

impl Scan {
    /// Total number of stored hits (cells × sub-rays). For the cell
    /// count, see `metadata.cell_count()`.
    pub fn len(&self) -> usize {
        self.hits.len()
    }

    /// Whether the scan carries no cells.
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }

    /// Flat index of the first sub-ray for cell `(row, col)`.
    fn cell_base(&self, row: u32, col: u32) -> usize {
        let k = (row * self.metadata.n_phi + col) as usize;
        k * self.metadata.effective_subray_count() as usize
    }

    /// All `subray_count` sub-ray hits for cell `(row, col)`.
    pub fn sub_hits_at(&self, row: u32, col: u32) -> &[RayHit] {
        let base = self.cell_base(row, col);
        let k = self.metadata.effective_subray_count() as usize;
        &self.hits[base..base + k]
    }

    /// Closest real hit (smallest `t`) among a cell's sub-rays. `None`
    /// when every sub-ray missed.
    pub fn first_hit_at(&self, row: u32, col: u32) -> Option<&RayHit> {
        self.sub_hits_at(row, col)
            .iter()
            .filter(|h| h.shape_id.is_some())
            .min_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Furthest real hit (largest finite `t`) among a cell's sub-rays.
    /// `None` when every sub-ray missed.
    pub fn last_hit_at(&self, row: u32, col: u32) -> Option<&RayHit> {
        self.sub_hits_at(row, col)
            .iter()
            .filter(|h| h.shape_id.is_some() && h.t.is_finite())
            .max_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Compatibility shim: behaves like H4's single-return `hit_at` —
    /// returns the first (closest) hit for the cell, or a synthesized
    /// miss `RayHit` when every sub-ray missed. Prefer
    /// [`Scan::first_hit_at`] in new code.
    pub fn hit_at(&self, row: u32, col: u32) -> RayHit {
        if let Some(h) = self.first_hit_at(row, col) {
            *h
        } else {
            // Synthesize the same miss sentinel `Raycaster` returns.
            RayHit {
                shape_id: None,
                body_id: molla_core::BodyId::NONE,
                t: Real::INFINITY,
                point: Vec3::ZERO,
                normal: Vec3::Y,
            }
        }
    }

    /// Number of *cells* with at least one real hit among their
    /// sub-rays. The right summary for "scan coverage."
    pub fn hit_count(&self) -> usize {
        let k = self.metadata.effective_subray_count() as usize;
        let cells = self.metadata.cell_count() as usize;
        (0..cells)
            .filter(|c| {
                self.hits[c * k..(c + 1) * k]
                    .iter()
                    .any(|h| h.shape_id.is_some())
            })
            .count()
    }

    /// Total number of *sub-rays* that recorded a real hit. Strictly
    /// `≥ hit_count()` when `subray_count > 1`.
    pub fn subray_hit_count(&self) -> usize {
        self.hits.iter().filter(|h| h.shape_id.is_some()).count()
    }
}

/// Terrestrial-LiDAR sensor: a [`Raycaster`] driven by a scan-grid
/// recipe. One `Lidar` is cheap to construct and reusable across
/// many scans up to its ray budget (= max cells × max sub-rays).
pub struct Lidar {
    raycaster: Raycaster,
    max_rays: u32,
}

// `Lidar` and the metadata types must be Send + Sync so they can sit
// in a Bevy `Resource` or be passed across threads. Compile-time
// asserts: any future change that introduces a non-Send/Sync field
// (e.g. an `Rc`) breaks the build instead of silently regressing.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Lidar>();
    assert_send_sync::<ScanMetadata>();
    assert_send_sync::<Scan>();
};

impl Lidar {
    /// Build a lidar with a ray budget of `max_rays`. A single scan
    /// uses `n_theta * n_phi * subray_count` rays, so size the budget
    /// to the peak cone-sampled load.
    pub fn new(engine: &Engine, max_rays: u32) -> Result<Self> {
        Ok(Self {
            raycaster: Raycaster::new(engine, max_rays.max(1))?,
            max_rays: max_rays.max(1),
        })
    }

    /// Ray budget passed at construction.
    pub fn max_rays(&self) -> u32 {
        self.max_rays
    }

    /// Cast one full scan against `model` + `state`. Returns a `Scan`
    /// whose `hits` vector has `cells × subray_count` entries
    /// (`subray_count = 1` → one entry per cell). The metadata is
    /// validated up-front via [`ScanMetadata::validate`] — malformed
    /// inputs are rejected before any GPU work.
    pub fn cast(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        scan: &ScanMetadata,
    ) -> Result<Scan> {
        scan.validate()?;
        let n_rays = scan.checked_ray_count()?;
        if n_rays > self.max_rays {
            return Err(Error::Solver(format!(
                "lidar::cast: {n_rays} rays exceeds max_rays {} ({} cells × {} sub-rays)",
                self.max_rays,
                scan.cell_count(),
                scan.effective_subray_count(),
            )));
        }
        let (origins, directions) = self.lay_out_rays(scan);
        let hits = self.raycaster.cast(
            engine,
            model,
            state,
            &origins,
            &directions,
            scan.max_distance,
        )?;
        Ok(Scan {
            metadata: *scan,
            hits,
        })
    }

    /// Build the `(origins, directions)` lists from the scan recipe.
    /// For each cell, the central ray is fired first and `K - 1`
    /// extra rays are sampled in a Fibonacci spiral inside the
    /// `beam_divergence` cone (H5).
    fn lay_out_rays(&self, scan: &ScanMetadata) -> (Vec<Vec3>, Vec<Vec3>) {
        let n = scan.ray_count() as usize;
        let mut origins = Vec::with_capacity(n);
        let mut dirs = Vec::with_capacity(n);
        let (f, u, r) = scan.local_frame();
        for i_theta in 0..scan.n_theta {
            let theta = sample_axis(scan.theta_range, scan.n_theta, i_theta);
            let st = theta.sin();
            let ct = theta.cos();
            for i_phi in 0..scan.n_phi {
                let phi = sample_axis(scan.phi_range, scan.n_phi, i_phi);
                let sp = phi.sin();
                let cp = phi.cos();
                let central = Vec3::new(
                    f.x * st * cp + u.x * ct + r.x * st * sp,
                    f.y * st * cp + u.y * ct + r.y * st * sp,
                    f.z * st * cp + u.z * ct + r.z * st * sp,
                );
                // The central ray plus K - 1 cone samples per cell —
                // even at K = 1 we still push a single (central) ray
                // so the layout is always `cells × subray_count`.
                cone_sample_into(
                    central,
                    scan.beam_divergence,
                    scan.effective_subray_count(),
                    scan.origin,
                    &mut origins,
                    &mut dirs,
                );
            }
        }
        (origins, dirs)
    }
}

/// Fibonacci-spiral cone sampler. Pushes `K` rays (`origin`,
/// `direction`) into the output vectors with `axis` as the central
/// direction and `cone_half_angle` (rad) as the cone half-angle.
/// `K = 1` collapses to the central ray.
///
/// The radial schedule is `α_k = cone_half_angle · sqrt(k / (K - 1))`
/// so `k = 0` is the centre and `k = K - 1` sits on the cone surface;
/// the azimuthal schedule advances by the golden angle
/// (`π · (3 − √5) ≈ 2.39996` rad) so the sub-rays distribute evenly
/// without rational-fraction stripes.
fn cone_sample_into(
    axis: Vec3,
    cone_half_angle: Real,
    k: u32,
    origin: Vec3,
    origins: &mut Vec<Vec3>,
    dirs: &mut Vec<Vec3>,
) {
    let k = k.max(1);
    let axis = normalise(axis, Vec3::new(0.0, 0.0, 1.0));
    if k == 1 || cone_half_angle <= 0.0 {
        for _ in 0..k {
            origins.push(origin);
            dirs.push(axis);
        }
        return;
    }
    // Build any orthonormal frame perpendicular to `axis`.
    let helper = if axis.y.abs() < 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let e1 = normalise(cross(axis, helper), Vec3::new(1.0, 0.0, 0.0));
    let e2 = normalise(cross(axis, e1), Vec3::new(0.0, 1.0, 0.0));
    let golden_angle: Real = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let k_minus_1 = (k - 1) as Real;
    for i in 0..k {
        if i == 0 {
            origins.push(origin);
            dirs.push(axis);
            continue;
        }
        let alpha = cone_half_angle * (i as Real / k_minus_1).sqrt();
        let beta = (i as Real) * golden_angle;
        let sa = alpha.sin();
        let ca = alpha.cos();
        let cb = beta.cos();
        let sb = beta.sin();
        let d = Vec3::new(
            ca * axis.x + sa * (cb * e1.x + sb * e2.x),
            ca * axis.y + sa * (cb * e1.y + sb * e2.y),
            ca * axis.z + sa * (cb * e1.z + sb * e2.z),
        );
        origins.push(origin);
        dirs.push(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_metadata_grid_size_matches_axes() {
        let s = ScanMetadata::new(Vec3::ZERO, 64, 128);
        assert_eq!(s.cell_count(), 64 * 128);
    }

    #[test]
    fn validate_rejects_overflowed_grid_or_subray_count() {
        let grid = ScanMetadata::new(Vec3::ZERO, u32::MAX, 2);
        assert!(grid.validate().is_err());
        let mut subrays = ScanMetadata::new(Vec3::ZERO, 65_536, 1);
        subrays.subray_count = 65_536;
        assert!(subrays.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_axes() {
        let mut s = ScanMetadata::new(Vec3::ZERO, 0, 8);
        assert!(s.validate().is_err());
        s = ScanMetadata::new(Vec3::ZERO, 8, 0);
        assert!(s.validate().is_err());
    }

    #[test]
    fn validate_rejects_inverted_angular_range() {
        let mut s = ScanMetadata::new(Vec3::ZERO, 4, 4);
        s.theta_range = (1.0, 0.5);
        assert!(s.validate().is_err());
    }

    #[test]
    fn validate_rejects_nonfinite_inputs() {
        let mut s = ScanMetadata::new(Vec3::ZERO, 4, 4);
        s.theta_range = (0.0, f64::NAN);
        assert!(s.validate().is_err());
        s = ScanMetadata::new(Vec3::ZERO, 4, 4);
        s.max_distance = f64::INFINITY;
        assert!(s.validate().is_err());
    }

    #[test]
    fn validate_rejects_out_of_range_beam_divergence() {
        let mut s = ScanMetadata::new(Vec3::ZERO, 4, 4);
        s.beam_divergence = -0.1;
        assert!(s.validate().is_err());
        s.beam_divergence = 2.0;
        assert!(s.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_forward_or_up() {
        let mut s = ScanMetadata::new(Vec3::ZERO, 4, 4);
        s.forward = Vec3::ZERO;
        assert!(s.validate().is_err());
        s = ScanMetadata::new(Vec3::ZERO, 4, 4);
        s.up = Vec3::ZERO;
        assert!(s.validate().is_err());
    }

    #[test]
    fn validate_accepts_default_metadata() {
        let s = ScanMetadata::new(Vec3::ZERO, 32, 64);
        assert!(s.validate().is_ok());
    }

    #[test]
    fn validate_rejects_parallel_frame_axes() {
        for direction in [Vec3::X, Vec3::Y, Vec3::new(1.0, 2.0, 3.0)] {
            let mut s = ScanMetadata::new(Vec3::ZERO, 1, 1);
            s.forward = direction;
            s.up = direction * 2.0;
            assert!(s.validate().is_err());
            s.up = -direction;
            assert!(s.validate().is_err());
        }
    }

    #[test]
    fn rc_to_direction_at_grid_endpoints_lines_up_with_frame_axes() {
        // θ = 0 picks the up axis exactly; φ doesn't matter there.
        let mut s = ScanMetadata::new(Vec3::ZERO, 2, 4);
        s.theta_range = (0.0, std::f64::consts::PI);
        s.phi_range = (0.0, 2.0 * std::f64::consts::PI);
        let d = s.rc_to_direction(0, 0);
        assert!(
            (d - Vec3::Y).length() < 1.0e-6,
            "θ=0 should point along up, got {d:?}"
        );
        // θ = π picks -up.
        let d_bottom = s.rc_to_direction(1, 0);
        assert!(
            (d_bottom + Vec3::Y).length() < 1.0e-6,
            "θ=π should point along -up, got {d_bottom:?}"
        );
    }

    #[test]
    fn cone_sample_property_subrays_inside_cone() {
        // Every Fibonacci-cone sub-ray lies within `cone_half_angle`
        // of the central direction. This is the property the H5
        // multi-return acceptance leans on: cone widening implies
        // sub-rays scatter inside a controlled solid angle.
        let axis = Vec3::new(0.3, -0.9, 0.2).normalize();
        let half_angle: Real = 0.15;
        let k: u32 = 16;
        let mut origins = Vec::new();
        let mut dirs = Vec::new();
        cone_sample_into(axis, half_angle, k, Vec3::ZERO, &mut origins, &mut dirs);
        assert_eq!(dirs.len(), k as usize);
        for d in dirs.iter().skip(1) {
            // Each sub-ray's direction makes an angle ≤ cone_half_angle
            // with the axis. dot ≥ cos(half_angle).
            let cos = d.normalize().dot(axis);
            assert!(
                cos >= half_angle.cos() - 1.0e-9,
                "sub-ray angle {} > cone half-angle {}, dot={}",
                cos.acos(),
                half_angle,
                cos
            );
        }
    }

    #[test]
    fn cone_sample_property_central_ray_is_axis() {
        // K = 1 (or sub-ray index 0) returns the axis unchanged.
        let axis = Vec3::new(0.0, -1.0, 0.0);
        let mut origins = Vec::new();
        let mut dirs = Vec::new();
        cone_sample_into(axis, 0.5, 1, Vec3::ZERO, &mut origins, &mut dirs);
        assert_eq!(dirs.len(), 1);
        assert!((dirs[0] - axis).length() < 1.0e-9);

        let mut origins = Vec::new();
        let mut dirs = Vec::new();
        cone_sample_into(axis, 0.5, 8, Vec3::ZERO, &mut origins, &mut dirs);
        assert!((dirs[0] - axis).length() < 1.0e-9);
    }

    #[test]
    fn cone_sample_with_zero_half_angle_collapses_all_to_central() {
        let axis = Vec3::new(1.0, 0.0, 0.0);
        let mut origins = Vec::new();
        let mut dirs = Vec::new();
        cone_sample_into(axis, 0.0, 5, Vec3::ZERO, &mut origins, &mut dirs);
        assert_eq!(dirs.len(), 5);
        for d in &dirs {
            assert!((d - axis).length() < 1.0e-9);
        }
    }
}

// ── H7 — AerialLidar ──────────────────────────────────────────────

/// Parameters for a single aerial-LiDAR pass. Mirrors helios's
/// `AerialScanMetadata` (`xtra/helios/plugins/aeriallidar/include/AerialLiDAR.h`):
/// the scan is described by a footprint AABB (pulse origins on its
/// upper face), a per-area pulse density, and an optional downward
/// cone-of-fire half-angle.
#[derive(Debug, Clone, Copy)]
pub struct AerialScanMetadata {
    /// Footprint AABB minimum corner (world-space). Pulse origins are
    /// drawn on the AABB's **upper face** (`y == max.y`).
    pub footprint_min: Vec3,
    /// Footprint AABB maximum corner.
    pub footprint_max: Vec3,
    /// Pulse density on the upper face, in pulses per m².
    pub scan_density: Real,
    /// Full scan-cone half-angle (rad). `0` (default) = straight
    /// nadir; `> 0` tilts each pulse by a Halton-distributed offset
    /// inside the cone.
    pub cone_angle: Real,
    /// Beam exit diameter at the pulse origin (m). Stored for
    /// downstream physics; not yet sampled.
    pub exit_diameter: Real,
    /// Per-pulse beam-divergence cone half-angle (rad). Used by the
    /// H5 sub-ray sampler.
    pub beam_divergence: Real,
    /// Sub-rays per pulse (`1` = central only).
    pub subray_count: u32,
    /// Maximum ray distance (m).
    pub max_distance: Real,
}

impl AerialScanMetadata {
    /// A nadir-only scan with `1` sub-ray per pulse.
    pub fn new(footprint_min: Vec3, footprint_max: Vec3, scan_density: Real) -> Self {
        Self {
            footprint_min,
            footprint_max,
            scan_density,
            cone_angle: 0.0,
            exit_diameter: 0.0,
            beam_divergence: 0.0,
            subray_count: 1,
            max_distance: 200.0,
        }
    }

    /// Footprint area on the upper face (m²).
    pub fn footprint_area(&self) -> Real {
        let dx = (self.footprint_max.x - self.footprint_min.x).abs();
        let dz = (self.footprint_max.z - self.footprint_min.z).abs();
        dx * dz
    }

    /// Number of pulses laid down at the configured density (rounded
    /// to the nearest integer, at least 1).
    pub fn pulse_count(&self) -> u32 {
        (self.footprint_area() * self.scan_density).round().max(1.0) as u32
    }

    /// Sub-ray count clamped to at least 1.
    pub fn effective_subray_count(&self) -> u32 {
        self.subray_count.max(1)
    }

    /// Total ray budget, saturating for invalid metadata; use `validate` before allocation.
    pub fn ray_count(&self) -> u32 {
        self.pulse_count()
            .saturating_mul(self.effective_subray_count())
    }

    /// Checked pulse count before conversion to the GPU index type.
    pub fn checked_pulse_count(&self) -> Result<u32> {
        let pulses = self.footprint_area() * self.scan_density;
        if !pulses.is_finite() || pulses <= 0.0 || pulses.round() > u32::MAX as Real {
            return Err(Error::Solver(
                "AerialScanMetadata: pulse count outside u32 budget".into(),
            ));
        }
        Ok(pulses.round().max(1.0) as u32)
    }

    /// Checked total ray budget for allocation and dispatch.
    pub fn checked_ray_count(&self) -> Result<u32> {
        self.checked_pulse_count()?
            .checked_mul(self.effective_subray_count())
            .ok_or_else(|| Error::Solver("AerialScanMetadata: ray count exceeds u32".into()))
    }

    /// Validate the metadata before any GPU work. Returns an
    /// `Error::Solver` with a precise message on the first violated
    /// invariant: finite + non-degenerate footprint, positive
    /// scan-density and max-distance, cone-angle and beam-divergence
    /// in `[0, π/2)`.
    pub fn validate(&self) -> Result<()> {
        self.checked_ray_count()?;
        for (name, v) in [
            ("footprint_min", self.footprint_min),
            ("footprint_max", self.footprint_max),
        ] {
            if !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite() {
                return Err(Error::Solver(format!(
                    "AerialScanMetadata: {name} must be finite ({:?})",
                    v
                )));
            }
        }
        if self.footprint_max.x < self.footprint_min.x
            || self.footprint_max.y < self.footprint_min.y
            || self.footprint_max.z < self.footprint_min.z
        {
            return Err(Error::Solver(format!(
                "AerialScanMetadata: footprint_max < footprint_min ({:?} < {:?})",
                self.footprint_max, self.footprint_min
            )));
        }
        if self.footprint_area() <= 0.0 {
            return Err(Error::Solver(format!(
                "AerialScanMetadata: footprint area is zero ({} × {} m)",
                self.footprint_max.x - self.footprint_min.x,
                self.footprint_max.z - self.footprint_min.z
            )));
        }
        if !self.scan_density.is_finite() || self.scan_density <= 0.0 {
            return Err(Error::Solver(format!(
                "AerialScanMetadata: scan_density must be positive + finite (got {})",
                self.scan_density
            )));
        }
        if !self.max_distance.is_finite() || self.max_distance <= 0.0 {
            return Err(Error::Solver(format!(
                "AerialScanMetadata: max_distance must be positive + finite (got {})",
                self.max_distance
            )));
        }
        for (name, a) in [
            ("cone_angle", self.cone_angle),
            ("beam_divergence", self.beam_divergence),
        ] {
            if !a.is_finite() || !(0.0..std::f64::consts::FRAC_PI_2).contains(&a) {
                return Err(Error::Solver(format!(
                    "AerialScanMetadata: {name} must be in [0, π/2) (got {a})"
                )));
            }
        }
        if !self.exit_diameter.is_finite() || self.exit_diameter < 0.0 {
            return Err(Error::Solver(format!(
                "AerialScanMetadata: exit_diameter must be non-negative + finite (got {})",
                self.exit_diameter
            )));
        }
        Ok(())
    }
}

/// One finished aerial-LiDAR pass. The hits are interleaved
/// per-pulse: pulse `p`'s sub-rays occupy
/// `hits[p * subray_count .. (p + 1) * subray_count]`.
#[derive(Debug, Clone)]
pub struct AerialScan {
    pub metadata: AerialScanMetadata,
    /// Pulse origins (one per pulse, not per sub-ray).
    pub origins: Vec<Vec3>,
    /// Pulse central directions (one per pulse).
    pub directions: Vec<Vec3>,
    /// `pulse_count × subray_count` hits.
    pub hits: Vec<RayHit>,
}

impl AerialScan {
    pub fn pulse_count(&self) -> u32 {
        self.origins.len() as u32
    }

    pub fn subray_count(&self) -> u32 {
        self.metadata.effective_subray_count()
    }

    /// Sub-rays of pulse `p`.
    pub fn sub_hits(&self, pulse: u32) -> &[RayHit] {
        let k = self.subray_count() as usize;
        let base = pulse as usize * k;
        &self.hits[base..base + k]
    }

    /// Closest hit of pulse `p` (`None` if every sub-ray missed).
    pub fn first_hit(&self, pulse: u32) -> Option<&RayHit> {
        self.sub_hits(pulse)
            .iter()
            .filter(|h| h.shape_id.is_some())
            .min_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Number of pulses with at least one real sub-ray hit.
    pub fn hit_count(&self) -> usize {
        let k = self.subray_count() as usize;
        let n = self.pulse_count() as usize;
        (0..n)
            .filter(|p| {
                self.hits[p * k..(p + 1) * k]
                    .iter()
                    .any(|h| h.shape_id.is_some())
            })
            .count()
    }
}

/// Aerial-LiDAR sensor: pulse origins low-discrepancy-distributed
/// over a footprint AABB upper face, directions nadir or downward
/// cone, dispatched through a shared [`Raycaster`].
pub struct AerialLidar {
    raycaster: Raycaster,
    max_rays: u32,
}

impl AerialLidar {
    /// Build a sensor with `max_rays` ray budget. Size for the peak
    /// `pulse_count * subray_count`.
    pub fn new(engine: &Engine, max_rays: u32) -> Result<Self> {
        Ok(Self {
            raycaster: Raycaster::new(engine, max_rays.max(1))?,
            max_rays: max_rays.max(1),
        })
    }

    pub fn max_rays(&self) -> u32 {
        self.max_rays
    }

    /// Run one aerial pass against `model` + `state`. The metadata
    /// is validated up-front via [`AerialScanMetadata::validate`].
    pub fn cast(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        scan: &AerialScanMetadata,
    ) -> Result<AerialScan> {
        scan.validate()?;
        let n_rays = scan.checked_ray_count()?;
        if n_rays > self.max_rays {
            return Err(Error::Solver(format!(
                "aerial_lidar::cast: {n_rays} rays exceeds max_rays {} ({} pulses × {} sub-rays)",
                self.max_rays,
                scan.pulse_count(),
                scan.effective_subray_count()
            )));
        }
        let (origins, directions, pulse_origins, pulse_dirs) = aerial_layout(scan);
        let hits = self.raycaster.cast(
            engine,
            model,
            state,
            &origins,
            &directions,
            scan.max_distance,
        )?;
        Ok(AerialScan {
            metadata: *scan,
            origins: pulse_origins,
            directions: pulse_dirs,
            hits,
        })
    }
}

/// Build the (per-sub-ray-origin, per-sub-ray-direction) arrays plus
/// the per-pulse origin / direction summary used in the returned
/// `AerialScan`. Pulse origins are Halton-distributed on the
/// footprint's upper face; pulse directions are nadir tilted by a
/// Halton-distributed cone offset; sub-rays inside each pulse are
/// the H5 Fibonacci-cone sampler.
fn aerial_layout(scan: &AerialScanMetadata) -> (Vec<Vec3>, Vec<Vec3>, Vec<Vec3>, Vec<Vec3>) {
    let n_pulses = scan.pulse_count();
    let k = scan.effective_subray_count();
    let total = (n_pulses * k) as usize;
    let mut origins = Vec::with_capacity(total);
    let mut dirs = Vec::with_capacity(total);
    let mut pulse_origins = Vec::with_capacity(n_pulses as usize);
    let mut pulse_dirs = Vec::with_capacity(n_pulses as usize);
    let xs = scan.footprint_max.x.max(scan.footprint_min.x);
    let xn = scan.footprint_max.x.min(scan.footprint_min.x);
    let zs = scan.footprint_max.z.max(scan.footprint_min.z);
    let zn = scan.footprint_max.z.min(scan.footprint_min.z);
    let y_top = scan.footprint_max.y.max(scan.footprint_min.y);
    for p in 0..n_pulses {
        let u = halton(p + 1, 2);
        let v = halton(p + 1, 3);
        let ox = xn + (xs - xn) * u;
        let oz = zn + (zs - zn) * v;
        let origin = Vec3::new(ox, y_top, oz);
        // Pulse central direction — nadir tilted by a Halton-2D cone
        // offset for cone_angle > 0.
        let dir = if scan.cone_angle <= 0.0 {
            Vec3::new(0.0, -1.0, 0.0)
        } else {
            let r = scan.cone_angle * halton(p + 1, 5).sqrt();
            let theta = halton(p + 1, 7) * 2.0 * std::f64::consts::PI;
            let sin_r = r.sin();
            let cos_r = r.cos();
            Vec3::new(sin_r * theta.cos(), -cos_r, sin_r * theta.sin())
        };
        pulse_origins.push(origin);
        pulse_dirs.push(dir);
        // K sub-rays per pulse inside the beam-divergence cone.
        cone_sample_into(
            dir,
            scan.beam_divergence,
            k,
            origin,
            &mut origins,
            &mut dirs,
        );
    }
    (origins, dirs, pulse_origins, pulse_dirs)
}

/// `i`-th value (`i ≥ 1`) of the radical-inverse Halton sequence in
/// the given prime base. Returns a deterministic low-discrepancy
/// quasi-random sample in `(0, 1)`. The classic 2-D Halton uses
/// bases `(2, 3)`; 4-D adds `(5, 7)`.
fn halton(mut i: u32, base: u32) -> Real {
    let mut f = 1.0;
    let mut r = 0.0;
    while i > 0 {
        f /= base as Real;
        r += f * ((i % base) as Real);
        i /= base;
    }
    r
}

#[cfg(test)]
mod aerial_tests {
    use super::*;

    #[test]
    fn pulse_count_scales_with_density_and_area() {
        let mut s =
            AerialScanMetadata::new(Vec3::new(-5.0, 10.0, -5.0), Vec3::new(5.0, 10.0, 5.0), 1.0);
        // 10 × 10 m = 100 m², density 1 → 100 pulses.
        assert_eq!(s.pulse_count(), 100);
        s.scan_density = 4.0;
        assert_eq!(s.pulse_count(), 400);
    }

    #[test]
    fn aerial_validate_rejects_overflowed_pulse_and_ray_counts() {
        let mut scan = AerialScanMetadata::new(Vec3::ZERO, Vec3::ONE, f64::MAX);
        assert!(scan.validate().is_err());
        scan.scan_density = u32::MAX as f64;
        scan.subray_count = 2;
        assert!(scan.validate().is_err());
        scan.footprint_max = Vec3::new(f64::MAX, 1.0, f64::MAX);
        scan.scan_density = 1.0;
        assert!(scan.validate().is_err());
    }

    #[test]
    fn aerial_validate_rejects_inverted_footprint() {
        let s = AerialScanMetadata::new(
            Vec3::new(5.0, 10.0, 5.0),
            Vec3::new(-5.0, 10.0, -5.0), // max < min
            1.0,
        );
        assert!(s.validate().is_err());
    }

    #[test]
    fn aerial_validate_rejects_zero_area_footprint() {
        let s = AerialScanMetadata::new(
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0), // identical → area 0
            1.0,
        );
        assert!(s.validate().is_err());
    }

    #[test]
    fn aerial_validate_rejects_nonpositive_density_or_distance() {
        let mut s =
            AerialScanMetadata::new(Vec3::new(-1.0, 10.0, -1.0), Vec3::new(1.0, 10.0, 1.0), 0.0);
        assert!(s.validate().is_err());
        s.scan_density = 1.0;
        s.max_distance = -1.0;
        assert!(s.validate().is_err());
    }

    #[test]
    fn aerial_validate_rejects_out_of_range_cone_or_divergence() {
        let mut s =
            AerialScanMetadata::new(Vec3::new(-1.0, 10.0, -1.0), Vec3::new(1.0, 10.0, 1.0), 1.0);
        s.cone_angle = 2.0;
        assert!(s.validate().is_err());
        s.cone_angle = 0.5;
        s.beam_divergence = -0.1;
        assert!(s.validate().is_err());
    }

    #[test]
    fn aerial_validate_accepts_default_metadata() {
        let s =
            AerialScanMetadata::new(Vec3::new(-5.0, 50.0, -5.0), Vec3::new(5.0, 50.0, 5.0), 1.0);
        assert!(s.validate().is_ok());
    }

    #[test]
    fn halton_2d_distributes_inside_unit_square() {
        // Spot check: the first 10 (base-2, base-3) Halton pairs all
        // lie in [0, 1).
        for i in 1..=10u32 {
            let u = halton(i, 2);
            let v = halton(i, 3);
            assert!((0.0..1.0).contains(&u), "u {u} out of [0,1)");
            assert!((0.0..1.0).contains(&v), "v {v} out of [0,1)");
        }
    }
}
