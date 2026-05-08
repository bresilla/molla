//! Stable handle types used during model construction and (in some cases)
//! at simulation time.
//!
//! IDs are simple `u32` newtypes. After [`crate::sim::ModelBuilder::build`]
//! they are guaranteed to equal the index into the corresponding `Model`
//! array — e.g. `model.body_mass.host()[body_id.index()]` is the mass of
//! `body_id`.

macro_rules! id_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name(pub u32);

        impl $name {
            /// Index into the corresponding `Model` array.
            #[must_use]
            pub const fn index(self) -> usize {
                self.0 as usize
            }

            /// Sentinel meaning "no such entity / static / world frame".
            pub const NONE: Self = Self(u32::MAX);

            /// Construct from a `usize` index. Panics in debug builds if
            /// the index doesn't fit a `u32`.
            #[must_use]
            pub fn from_index(i: usize) -> Self {
                debug_assert!(
                    i < u32::MAX as usize,
                    concat!(stringify!($name), " index overflow")
                );
                Self(i as u32)
            }
        }
    };
}

id_newtype! {
    /// Identifies a rigid body within a `Model`.
    BodyId
}

id_newtype! {
    /// Identifies a collision shape within a `Model`.
    ShapeId
}

id_newtype! {
    /// Identifies a joint within a `Model`.
    JointId
}

id_newtype! {
    /// Identifies a particle within a `Model`.
    ParticleId
}

id_newtype! {
    /// Identifies a simulation world (multi-world / RL grouping).
    WorldId
}
