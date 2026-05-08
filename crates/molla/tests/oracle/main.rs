//! Oracle test crate: compares molla's solvers against a venv'd Newton.
//!
//! All tests in this crate are `#[ignore]` by default — they require a
//! Python interpreter with `newton-physics` installed, pointed at via
//! `MOLLA_NEWTON_PYTHON`. To run:
//!
//! ```bash
//! MOLLA_NEWTON_PYTHON=$(pwd)/xtra/newton/.venv/bin/python \
//!   cargo test -p molla --test oracle -- --ignored
//! ```

mod free_fall;
mod free_fall_many;
mod harness;
