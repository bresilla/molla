//! Bitflags for `Model.body_flags`, `Model.particle_flags`, `Model.shape_flags`.

/// Per-body flags.
#[allow(non_snake_case)]
pub mod BodyFlags {
    /// Body responds to forces; this is the default.
    pub const DYNAMIC: u32 = 0;
    /// User-prescribed body; pose is held unless KINEMATIC_DRIVEN is also set.
    pub const KINEMATIC: u32 = 1 << 0;
    /// Body remains at its world pose with zero velocity.
    pub const FIXED: u32 = 1 << 1;
    /// Featherstone advances kinematic COM motion from prescribed world acceleration.
    pub const KINEMATIC_DRIVEN: u32 = 1 << 2;
    /// Bodies whose motion cannot be changed by forces or constraints.
    pub const IMMOVABLE: u32 = KINEMATIC | FIXED;
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
    /// Shape generates overlap readings without contact forces.
    pub const SENSOR: u32 = 1 << 1;
}
