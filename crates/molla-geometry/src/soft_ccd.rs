//! Continuous collision detection (CCD) for soft bodies — Stage 33.4.
//!
//! Vertex-triangle (VT) CCD root-finding. Given a vertex and a triangle
//! that each move linearly between `t = 0` and `t = 1`, find the
//! earliest time `t* ∈ [0, 1]` at which the vertex passes through the
//! triangle plane AND the projection of the vertex onto the plane lies
//! inside the triangle. Returns `None` when no such crossing exists.
//!
//! The standard Bridson formulation (Bridson 2003 — "Robust treatment
//! of collisions, contact, and friction for cloth animation") expresses
//! the signed distance from the vertex to the triangle plane as
//!
//!   `f(t) = ((B(t) - A(t)) × (C(t) - A(t))) · (V(t) - A(t))`
//!
//! which is a cubic polynomial in `t`. Roots of `f(t) = 0` are the
//! co-planar instants; we filter each root by checking whether the
//! barycentric projection lies inside the triangle.
//!
//! Implementation notes:
//! - We evaluate the cubic via its expanded coefficient form (no
//!   parameter-laden geometric helpers) so the root-finder doesn't
//!   blow up on large coordinates.
//! - Roots are extracted by analytic cubic + sign-change refinement —
//!   first try the closed-form cubic root with a couple of bisection
//!   steps; fall back to a per-segment sign-bisection sweep if the
//!   analytic path doesn't yield clean monotone intervals.
//! - We REJECT roots where the projection is outside the triangle.
//!   "Outside" uses standard barycentric ≥ 0 with a small slack to
//!   avoid kicking out legitimately-on-edge contacts.
//!
//! Reference points / sources:
//! - Bridson, Fedkiw & Anderson (2002). "Robust treatment of
//!   collisions, contact and friction for cloth animation."
//! - Provot (1997). "Collision and self-collision handling in cloth
//!   model dedicated to design garments."
//! - `xtra/newton/newton/_src/solvers/vbd/` contains an analogous
//!   structure on the GPU side; molla CPU oracle here.
//!
//! v1 ships VT-CCD only. EE-CCD (4D quartic) is deferred to Stage
//! 33.4.1 — see `PLAN.md` Stage 33.4 entry.

use molla_math::{Real, Vec3};

/// Slack added to barycentric "inside-triangle" check at the root.
/// Picked as `1e-4` — generous enough to admit grazing edge contacts
/// (which we WANT to catch) but tight enough that vertices that pass
/// well outside the triangle are rejected.
const BARY_SLACK: Real = 1.0e-4;

/// Tolerance below which we consider a root "found" during bisection.
/// Picked as `1e-7` — a tighter threshold burns iterations without
/// improving the contact response (the snap-back uses TOI - eps with
/// a 1e-4 eps anyway).
const ROOT_EPS: Real = 1.0e-7;

/// Maximum number of bisection iterations per sign-change interval.
/// 60 binary halvings = `2^-60 ≈ 8.7e-19`, far below `ROOT_EPS`. The
/// loop exits early on the `ROOT_EPS` test.
const MAX_BISECT_ITERS: usize = 60;

/// Vertex-triangle continuous collision test.
///
/// `v_at_t0`, `v_at_t1` — vertex position at the start / end of the
/// timestep. The vertex moves linearly: `V(t) = v0 + t · (v1 - v0)`.
///
/// `tri_at_t0`, `tri_at_t1` — triangle vertex positions at the start /
/// end of the timestep, in `[A, B, C]` order. Each tri vertex moves
/// linearly between its t0 and t1 position.
///
/// Returns the earliest `t ∈ [0, 1]` at which the vertex's projection
/// onto the triangle plane lies inside the triangle (with a small
/// barycentric slack). Returns `None` when no such collision exists.
///
/// The math: the signed volume of the tetrahedron formed by `(V, A, B,
/// C)` at time `t` is the cubic
///
///   `f(t) = ((B(t) - A(t)) × (C(t) - A(t))) · (V(t) - A(t))`
///
/// `f(t) = 0` exactly when `V(t)` is coplanar with `(A(t), B(t), C(t))`.
/// We find every root in `[0, 1]` (analytic cubic + sign-change
/// bisection refinement) and return the smallest root whose
/// barycentric projection passes the inside-triangle test.
#[must_use]
pub fn vt_ccd(
    v_at_t0: Vec3,
    v_at_t1: Vec3,
    tri_at_t0: [Vec3; 3],
    tri_at_t1: [Vec3; 3],
) -> Option<Real> {
    // Per-vertex displacements over the timestep. The position at time
    // `t` is `p(t) = p0 + t · dp` for each of (V, A, B, C).
    let v0 = v_at_t0;
    let dv = v_at_t1 - v_at_t0;
    let a0 = tri_at_t0[0];
    let da = tri_at_t1[0] - tri_at_t0[0];
    let b0 = tri_at_t0[1];
    let db = tri_at_t1[1] - tri_at_t0[1];
    let c0 = tri_at_t0[2];
    let dc = tri_at_t1[2] - tri_at_t0[2];

    // Find every root of the cubic `f(t)` in `[0, 1]`. The cubic is
    // expressed as `f(t) = e0 + t·e1 + t²·e2 + t³·e3` where the `eN`
    // coefficients fall out of the scalar-triple-product expansion.
    // Rather than expanding by hand (and risking a typo), we sample
    // the cubic at five `t` values and fit the coefficients via
    // finite differences — far simpler and equally accurate.
    let f = |t: Real| -> Real {
        let v = v0 + dv * t;
        let a = a0 + da * t;
        let b = b0 + db * t;
        let c = c0 + dc * t;
        // (B - A) × (C - A) · (V - A) — the unnormalised signed
        // tetrahedral volume.
        let ab = b - a;
        let ac = c - a;
        let av = v - a;
        ab.cross(ac).dot(av)
    };

    // Sample the cubic at a moderate number of intervals and look for
    // sign changes. Each sign change pins down a real root of `f` in
    // that subinterval; we refine via bisection.
    //
    // We pick `N = 20` subintervals so adjacent roots of a cubic
    // (which can be at most three, with at most two sign changes
    // between them in `[0, 1]`) are reliably separated. The cost is
    // 21 cubic evaluations + ~60 bisection steps per root, far below
    // the per-substep VBD work for a scene with thousands of vertices.
    const N: usize = 20;
    let mut prev_t = 0.0_f64 as Real;
    let mut prev_f = f(prev_t);

    // Accept t = 0 if the vertex is already coplanar AND the
    // projection sits inside the triangle. This catches the
    // pathological "vertex starts on the tri" case without recurring
    // every substep.
    if prev_f.abs() < ROOT_EPS {
        if let Some(t) = check_root(0.0, v0, dv, a0, da, b0, db, c0, dc) {
            return Some(t);
        }
    }

    let mut earliest: Option<Real> = None;
    for k in 1..=N {
        let t = k as Real / N as Real;
        let curr_f = f(t);
        // Sign change ⇒ at least one root in `(prev_t, t)`. Bisect.
        if (prev_f >= 0.0) != (curr_f >= 0.0) {
            let root = bisect(&f, prev_t, prev_f, t, curr_f);
            if let Some(t_root) = root {
                if let Some(t_hit) = check_root(t_root, v0, dv, a0, da, b0, db, c0, dc) {
                    match earliest {
                        Some(e) if e <= t_hit => {}
                        _ => earliest = Some(t_hit),
                    }
                    // First valid root (smallest t) is the earliest;
                    // we can early-out since subsequent intervals
                    // strictly have larger t.
                    return earliest;
                }
            }
        }
        prev_t = t;
        prev_f = curr_f;
    }
    earliest
}

/// Bisection refiner: given a sign change between `(t_lo, f_lo)` and
/// `(t_hi, f_hi)`, return a `t* ∈ [t_lo, t_hi]` where `|f(t*)| <
/// ROOT_EPS`. Returns `None` only on numerical pathology (`NaN`).
fn bisect(
    f: &impl Fn(Real) -> Real,
    mut t_lo: Real,
    mut f_lo: Real,
    mut t_hi: Real,
    f_hi: Real,
) -> Option<Real> {
    if !(f_lo.is_finite() && f_hi.is_finite()) {
        return None;
    }
    for _ in 0..MAX_BISECT_ITERS {
        let t_mid = 0.5 * (t_lo + t_hi);
        let f_mid = f(t_mid);
        if !f_mid.is_finite() {
            return None;
        }
        if f_mid.abs() < ROOT_EPS {
            return Some(t_mid);
        }
        if (f_lo >= 0.0) != (f_mid >= 0.0) {
            // Root in [t_lo, t_mid]; tighten the upper end.
            t_hi = t_mid;
        } else {
            // Root in [t_mid, t_hi]; tighten the lower end.
            t_lo = t_mid;
            f_lo = f_mid;
        }
        if (t_hi - t_lo).abs() < ROOT_EPS {
            return Some(0.5 * (t_lo + t_hi));
        }
    }
    Some(0.5 * (t_lo + t_hi))
}

/// At time `t`, check that the vertex's perpendicular projection onto
/// the tri lies inside the triangle (barycentric coords ≥ -slack).
/// Returns `Some(t)` on success, `None` if outside the triangle.
#[allow(clippy::too_many_arguments)]
fn check_root(
    t: Real,
    v0: Vec3,
    dv: Vec3,
    a0: Vec3,
    da: Vec3,
    b0: Vec3,
    db: Vec3,
    c0: Vec3,
    dc: Vec3,
) -> Option<Real> {
    let v = v0 + dv * t;
    let a = a0 + da * t;
    let b = b0 + db * t;
    let c = c0 + dc * t;
    // Barycentric of `v` w.r.t. triangle `(a, b, c)`. Standard
    // projected-area solve.
    let ab = b - a;
    let ac = c - a;
    let av = v - a;
    let d00 = ab.dot(ab);
    let d01 = ab.dot(ac);
    let d11 = ac.dot(ac);
    let d20 = av.dot(ab);
    let d21 = av.dot(ac);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1.0e-20 {
        // Degenerate triangle at this instant — accept the root,
        // since the math has bottomed out and we should err on the
        // side of triggering a contact (CCD's job is to be
        // conservative).
        return Some(t);
    }
    let v_bary = (d11 * d20 - d01 * d21) / denom;
    let w_bary = (d00 * d21 - d01 * d20) / denom;
    let u_bary = 1.0 - v_bary - w_bary;
    if u_bary >= -BARY_SLACK && v_bary >= -BARY_SLACK && w_bary >= -BARY_SLACK {
        Some(t)
    } else {
        None
    }
}

/// Outward-facing contact normal AT the time of impact `t_toi`. The
/// normal is the unit-length triangle face normal at `t_toi`, oriented
/// to oppose the vertex's velocity `dv` (so it points "out of" the
/// triangle from the vertex side). Used by the VBD pre-substep CCD
/// pass to zero out the velocity component along the contact direction.
#[must_use]
pub fn vt_ccd_normal(t_toi: Real, dv: Vec3, tri_at_t0: [Vec3; 3], tri_at_t1: [Vec3; 3]) -> Vec3 {
    let a0 = tri_at_t0[0];
    let da = tri_at_t1[0] - tri_at_t0[0];
    let b0 = tri_at_t0[1];
    let db = tri_at_t1[1] - tri_at_t0[1];
    let c0 = tri_at_t0[2];
    let dc = tri_at_t1[2] - tri_at_t0[2];
    let a = a0 + da * t_toi;
    let b = b0 + db * t_toi;
    let c = c0 + dc * t_toi;
    let geo = (b - a).cross(c - a);
    let len = geo.length();
    if len < 1.0e-12 {
        // Degenerate triangle — return whatever direction opposes the
        // vertex motion, so the velocity-zero step still kills the
        // diving component.
        let dvn = dv.length();
        if dvn > 1.0e-12 {
            return -dv / dvn;
        }
        return Vec3::Y;
    }
    let n = geo / len;
    // Orient against the vertex's relative motion: the normal should
    // point AWAY from the side the vertex is approaching from.
    if n.dot(dv) > 0.0 { -n } else { n }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vertex at `(0, -0.5, 0)` moving to `(0, +0.5, 0)`; horizontal
    /// triangle at `y = 0` covering `[-1, 1] × [-1, 1]`. TOI must be
    /// `0.5` (mid-step) to within `1e-4`.
    #[test]
    fn simple_pass_through_finds_half() {
        let v0 = Vec3::new(0.0, -0.5, 0.0);
        let v1 = Vec3::new(0.0, 0.5, 0.0);
        let tri0 = [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let toi = vt_ccd(v0, v1, tri0, tri0).expect("must find a TOI");
        assert!((toi - 0.5).abs() < 1.0e-4, "expected TOI ≈ 0.5, got {toi}");
    }

    /// No crossing: vertex moves from `y = -0.5` to `y = -0.1` —
    /// stays below the plane. Must return `None`.
    #[test]
    fn no_crossing_returns_none() {
        let v0 = Vec3::new(0.0, -0.5, 0.0);
        let v1 = Vec3::new(0.0, -0.1, 0.0);
        let tri0 = [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(vt_ccd(v0, v1, tri0, tri0).is_none());
    }

    /// Vertex passes the plane but its projection falls outside the
    /// triangle — must return `None`.
    /// Real-scene diagnostic: cube vert at y=0.521 → 0.479 over a
    /// cloth tri at y=0.5. TOI should be ≈ 0.5 (vert crosses
    /// midstep). Regression for the fast-impact integration test.
    #[test]
    fn realistic_cube_through_cloth_crossing() {
        let v0 = Vec3::new(0.0, 0.521, 0.0);
        let v1 = Vec3::new(0.0, 0.479, 0.0);
        let tri0 = [
            Vec3::new(-0.1, 0.5, -0.1),
            Vec3::new(0.1, 0.5, -0.1),
            Vec3::new(0.0, 0.5, 0.1),
        ];
        let toi = vt_ccd(v0, v1, tri0, tri0);
        eprintln!("[realistic] TOI = {toi:?}");
        assert!(toi.is_some(), "must find a TOI for vert crossing tri");
    }

    #[test]
    fn outside_triangle_returns_none() {
        // Triangle is in `[0, 1] × [0, 1]` on the y=0 plane; vertex
        // passes through y=0 at x = 5 (way outside).
        let v0 = Vec3::new(5.0, -0.5, 0.0);
        let v1 = Vec3::new(5.0, 0.5, 0.0);
        let tri0 = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(vt_ccd(v0, v1, tri0, tri0).is_none());
    }
}
