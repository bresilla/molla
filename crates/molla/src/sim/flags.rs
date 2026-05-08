//! Bitflags for `Model.body_flags`, `Model.particle_flags`, `Model.shape_flags`.

/// Per-body flags.
#[allow(non_snake_case)]
pub mod BodyFlags {
    /// Body responds to forces; this is the default.
    pub const DYNAMIC: u32 = 0;
    /// Body's pose is user-prescribed; integrators pass it through unchanged.
    pub const KINEMATIC: u32 = 1 << 0;
}

/// Per-particle flags. Populated in stage 4.
#[allow(non_snake_case)]
pub mod ParticleFlags {
    /// Particle is integrated each step.
    pub const ACTIVE: u32 = 1 << 0;
}

/// Per-shape flags. Populated in stage 4.
#[allow(non_snake_case)]
pub mod ShapeFlags {
    /// Shape participates in collision detection.
    pub const COLLIDES: u32 = 1 << 0;
}
