//! Time-integration solvers.

pub mod crba;
mod featherstone;
pub mod rnea;
mod semi_implicit;
mod xpbd;

pub use crba::mass_matrix;
pub use featherstone::FeatherstoneSolver;
pub use rnea::inverse_dynamics;
pub use semi_implicit::SemiImplicitSolver;
pub use xpbd::XpbdSolver;

use crate::core::Result;
use crate::math::Real;
use crate::sim::{Contacts, Control, Model, State};

/// Common interface for every time-integration scheme.
///
/// A solver advances a [`State`] by one timestep `dt`, possibly using the
/// current [`Control`] inputs and [`Contacts`] from the collision pipeline.
pub trait Solver: Send + Sync {
    /// Advance `state_in` by `dt` and write the result into `state_out`.
    ///
    /// `state_in` and `state_out` must be distinct buffers — solvers may
    /// hold borrows of both for the duration of the call.
    fn step(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
    ) -> Result<()>;
}
