//! V3 — suspension templates.
//!
//! A suspension corner is a vertical spring-damper between a chassis
//! mount and the wheel. The [`SuspensionKind`] templates (rigid axle,
//! double wishbone, MacPherson) share that spring-damper here; in a
//! fuller model the kind would also drive the camber/toe kinematics.
//! This is the "library on top" layer — it composes parts molla
//! already provides rather than adding engine physics.

/// Suspension geometry template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuspensionKind {
    /// Solid axle — both wheels on a rigid beam.
    RigidAxle,
    /// Independent double-wishbone (two A-arms).
    DoubleWishbone,
    /// MacPherson strut.
    MacPherson,
}

/// A vertical spring-damper suspension corner.
#[derive(Clone, Copy, Debug)]
pub struct Suspension {
    /// Geometry template.
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
    /// A suspension corner with the given template and parameters.
    pub fn new(kind: SuspensionKind, rest_length: f64, stiffness: f64, damping: f64) -> Self {
        Self {
            kind,
            rest_length,
            stiffness,
            damping,
            max_compression: rest_length * 0.9,
            travel: 0.0,
            travel_rate: 0.0,
        }
    }

    /// Spring + damper force (N, ≥ 0 — a suspension only pushes apart)
    /// for the given `compression` and compression `rate`.
    pub fn force(&self, compression: f64, rate: f64) -> f64 {
        let c = compression.clamp(0.0, self.max_compression);
        (self.stiffness * c + self.damping * rate).max(0.0)
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
}
