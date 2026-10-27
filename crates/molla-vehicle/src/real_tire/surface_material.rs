/// Ground/tread contact material.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceMaterial {
    /// Static Coulomb coefficient. Trial bristle forces below this remain
    /// stuck to the ground.
    pub mu_static: f64,
    /// Dynamic/sliding Coulomb coefficient once bristles saturate.
    pub mu_dynamic: f64,
    /// Extra damping on bristle slip speed (N per (m/s) per m² contact area).
    pub bristle_damping: f64,
    /// Rolling resistance coefficient. This is intentionally separate from
    /// Coulomb friction: soft sand can have moderate grip but high rolling
    /// losses; ice can have low grip but low rolling loss.
    pub rolling_resistance: f64,
    /// Optional cohesive shear stress (Pa) for soil-like surfaces.
    pub cohesion: f64,
}

impl SurfaceMaterial {
    pub const fn new(mu_static: f64, mu_dynamic: f64) -> Self {
        Self {
            mu_static,
            mu_dynamic,
            bristle_damping: 0.0,
            rolling_resistance: 0.015,
            cohesion: 0.0,
        }
    }

    pub fn from_terrain_friction(mu: f64) -> Self {
        Self {
            mu_static: mu,
            mu_dynamic: 0.82 * mu,
            bristle_damping: 3.0e4,
            rolling_resistance: 0.015,
            cohesion: 0.0,
        }
    }

    pub const fn dry_asphalt() -> Self {
        Self {
            mu_static: 1.05,
            mu_dynamic: 0.88,
            bristle_damping: 5.0e4,
            rolling_resistance: 0.012,
            cohesion: 0.0,
        }
    }

    pub const fn loose_dirt() -> Self {
        Self {
            mu_static: 0.55,
            mu_dynamic: 0.42,
            bristle_damping: 7.0e4,
            rolling_resistance: 0.055,
            cohesion: 800.0,
        }
    }

    pub const fn soft_soil() -> Self {
        Self {
            mu_static: 0.38,
            mu_dynamic: 0.28,
            bristle_damping: 9.0e4,
            rolling_resistance: 0.090,
            cohesion: 1_600.0,
        }
    }

    pub const fn ice() -> Self {
        Self {
            mu_static: 0.08,
            mu_dynamic: 0.045,
            bristle_damping: 5.0e3,
            rolling_resistance: 0.006,
            cohesion: 0.0,
        }
    }
}

pub(crate) fn sanitized_surface_material(material: &SurfaceMaterial) -> SurfaceMaterial {
    use super::internal::finite_nonnegative;
    let mu_static = finite_nonnegative(material.mu_static);
    let mu_dynamic = finite_nonnegative(material.mu_dynamic).min(mu_static);
    SurfaceMaterial {
        mu_static,
        mu_dynamic,
        bristle_damping: finite_nonnegative(material.bristle_damping),
        rolling_resistance: finite_nonnegative(material.rolling_resistance),
        cohesion: finite_nonnegative(material.cohesion),
    }
}
