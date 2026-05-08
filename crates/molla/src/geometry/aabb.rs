//! Axis-aligned bounding box, world-frame `f64`.

use crate::math::{Real, Vec3};

/// Axis-aligned bounding box. `min` <= `max` componentwise.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// AABB of a single point.
    #[must_use]
    pub fn point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }

    /// AABB centred at `p` with the given half-extents.
    #[must_use]
    pub fn centred(p: Vec3, half: Vec3) -> Self {
        Self {
            min: p - half,
            max: p + half,
        }
    }

    /// "Infinite" AABB used for half-space planes — they're trivially
    /// touched by every other AABB, so the broad phase always emits a
    /// pair against them.
    #[must_use]
    pub fn infinite() -> Self {
        Self {
            min: Vec3::splat(-Real::INFINITY),
            max: Vec3::splat(Real::INFINITY),
        }
    }

    /// `true` if the two AABBs share any volume.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Inflate by `margin` on every axis. Useful for adding a slack
    /// envelope to avoid missing contacts on the margin.
    #[must_use]
    pub fn expanded(&self, margin: Real) -> Self {
        Self {
            min: self.min - Vec3::splat(margin),
            max: self.max + Vec3::splat(margin),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_intersects_itself() {
        let a = Aabb::point(Vec3::new(1.0, 2.0, 3.0));
        assert!(a.intersects(&a));
    }

    #[test]
    fn disjoint_does_not_intersect() {
        let a = Aabb::centred(Vec3::ZERO, Vec3::splat(1.0));
        let b = Aabb::centred(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(1.0));
        assert!(!a.intersects(&b));
    }

    #[test]
    fn touching_intersects() {
        let a = Aabb::centred(Vec3::ZERO, Vec3::splat(1.0));
        let b = Aabb::centred(Vec3::new(2.0, 0.0, 0.0), Vec3::splat(1.0));
        assert!(a.intersects(&b));
    }

    #[test]
    fn infinite_intersects_anything() {
        let inf = Aabb::infinite();
        let p = Aabb::centred(Vec3::new(100.0, 0.0, 0.0), Vec3::splat(0.1));
        assert!(inf.intersects(&p));
    }
}
