//! Buffer abstraction that owns either CPU memory, GPU memory, or both.
//!
//! See PLAN.md §8 for the design contract.

mod buffer;

pub use buffer::{Buffer, BufferElement, BufferUsage};
