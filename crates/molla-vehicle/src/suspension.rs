//! V3 — suspension templates.
//!
//! A suspension corner is a vertical spring-damper between a chassis
//! mount and the wheel. This is the "library on top" layer — it
//! composes parts molla already provides rather than adding engine
//! physics.
//!
//! **Per-kind physics (2026-08, was INSPECTION.md F9 "label only"):**
//! each [`SuspensionKind`] now carries a distinct *vertical-plane
//! linkage character* — a rising wheel rate (progressivity), a damping
//! character, and the usable travel before the bump stop. These are the
//! effects that do **not** need a chassis roll DOF, so they live inside
//! [`Suspension::force`] directly. What still needs the roll DOF —
//! kind-driven camber/toe kinematics and true solid-axle left↔right beam
//! coupling — remains out of the 3-DOF `Vehicle` and is the seam where
//! the kinds would diverge further once a fuller chassis lands.

/// Suspension geometry template. Each variant maps to a distinct
/// vertical-plane linkage character (see [`SuspensionKind::geometry`]);
/// the wheel-rate shape, damping character, and bump-stop travel differ
/// per kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuspensionKind {
    /// Solid axle — both wheels on a rigid beam.
    RigidAxle,
    /// Independent double-wishbone (two A-arms).
    DoubleWishbone,
    /// MacPherson strut.
    MacPherson,
}

/// Vertical-plane linkage character derived from the geometry kind.
#[derive(Clone, Copy, Debug)]
struct SuspensionGeometry {
    /// Rising-rate coefficient: extra wheel rate blended in toward the
    /// bump stop (`0` = perfectly linear).
    progressive: f64,
    /// Damping character multiplier (valving / motion-ratio effect).
    damping_scale: f64,
    /// Usable compression before the hard bump stop, as a fraction of
    /// the rest length.
    bump_stop_frac: f64,
}

impl SuspensionKind {
    /// Linkage character for this geometry. These are the vertical-plane
    /// effects that do not need a chassis roll DOF; camber/toe kinematics
    /// and solid-axle beam coupling are not modelled here (see module docs).
    fn geometry(self) -> SuspensionGeometry {
        match self {
            // Solid beam: near-linear wheel rate, firmly damped, shorter
            // usable travel.
            SuspensionKind::RigidAxle => SuspensionGeometry {
                progressive: 0.05,
                damping_scale: 1.15,
                bump_stop_frac: 0.85,
            },
            // Inboard spring on an A-arm: a tunable rising rate.
            SuspensionKind::DoubleWishbone => SuspensionGeometry {
                progressive: 0.30,
                damping_scale: 1.00,
                bump_stop_frac: 0.90,
            },
            // Strut nearly in line with wheel travel: mild rise, lighter
            // valving.
            SuspensionKind::MacPherson => SuspensionGeometry {
                progressive: 0.18,
                damping_scale: 0.90,
                bump_stop_frac: 0.92,
            },
        }
    }
}

/// A vertical spring-damper suspension corner.
#[derive(Clone, Copy, Debug)]
pub struct Suspension {
    /// Geometry template — selects the linkage character (rising rate,
    /// damping, bump-stop travel) used by [`Suspension::force`].
    pub kind: SuspensionKind,
    /// Free (uncompressed) length (m).
    pub rest_length: f64,
    /// Spring stiffness (N/m).
    pub stiffness: f64,
    /// Damping coefficient (N·s/m).
    pub damping: f64,
    /// Bump-stop: maximum compression (m).
    pub max_compression: f64,
    /// Current compression (m) — telemetry, updated by the vehicle step.
    pub travel: f64,
    /// Current compression rate (m/s) — telemetry.
    pub travel_rate: f64,
}

impl Suspension {
    /// A suspension corner with the given template and parameters. The
    /// bump-stop travel is taken from the kind's linkage character.
    pub fn new(kind: SuspensionKind, rest_length: f64, stiffness: f64, damping: f64) -> Self {
        Self {
            kind,
            rest_length,
            stiffness,
            damping,
            max_compression: rest_length * kind.geometry().bump_stop_frac,
            travel: 0.0,
            travel_rate: 0.0,
        }
    }

    /// Spring + damper force (N, ≥ 0 — a suspension only pushes apart)
    /// for the given `compression` and compression `rate`. The kind's
    /// linkage character adds a rising wheel rate toward the bump stop
    /// and scales the damping.
    pub fn force(&self, compression: f64, rate: f64) -> f64 {
        let g = self.kind.geometry();
        let c = compression.clamp(0.0, self.max_compression);
        let frac = if self.max_compression > 0.0 {
            c / self.max_compression
        } else {
            0.0
        };
        let spring = self.stiffness * c * (1.0 + g.progressive * frac);
        let damp = self.damping * g.damping_scale * rate;
        (spring + damp).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suspension_pushes_apart_and_respects_the_bump_stop() {
        let s = Suspension::new(SuspensionKind::DoubleWishbone, 0.30, 4.0e4, 3.0e3);
        // No compression → no force.
        assert_eq!(s.force(0.0, 0.0), 0.0);
        // Compressed → positive (upward) force.
        assert!(s.force(0.05, 0.0) > 0.0);
        // A suspension never pulls, even when "extended".
        assert_eq!(s.force(-0.1, 0.0), 0.0);
        // Past the bump stop the force saturates at the clamp.
        let at_stop = s.force(s.max_compression, 0.0);
        let past_stop = s.force(s.max_compression * 2.0, 0.0);
        assert!((at_stop - past_stop).abs() < 1e-9);
    }

    #[test]
    fn kinds_produce_distinct_force_signatures() {
        let mk = |k| Suspension::new(k, 0.35, 4.0e4, 3.0e3);
        let rigid = mk(SuspensionKind::RigidAxle);
        let wishbone = mk(SuspensionKind::DoubleWishbone);
        let strut = mk(SuspensionKind::MacPherson);

        // Bump-stop travel differs per kind (solid axle is shortest).
        assert!(rigid.max_compression < strut.max_compression);
        assert!(wishbone.max_compression < strut.max_compression);

        // Rising-rate signature: at the same compression the softer,
        // more progressive wishbone carries more spring force than the
        // near-linear solid axle, with the strut in between.
        let (fr, fw, fs) = (
            rigid.force(0.2, 0.0),
            wishbone.force(0.2, 0.0),
            strut.force(0.2, 0.0),
        );
        assert!(fr < fs && fs < fw, "rigid={fr} strut={fs} wishbone={fw}");

        // Damping character differs: solid axle is the firmest, strut the
        // lightest, at the same compression rate.
        let (dr, dw, ds) = (
            rigid.force(0.0, 1.0),
            wishbone.force(0.0, 1.0),
            strut.force(0.0, 1.0),
        );
        assert!(ds < dw && dw < dr, "rigid={dr} wishbone={dw} strut={ds}");
    }
}
