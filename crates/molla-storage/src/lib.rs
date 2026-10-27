//! Buffer abstraction that owns either CPU memory, GPU memory, or both.
//!
//! See PLAN.md §8 for the design contract.

mod buffer;

#[cfg(feature = "gpu")]
pub use buffer::BufferDevice;
pub use buffer::{Buffer, BufferElement, BufferUsage};
