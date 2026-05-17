//! Facade compatibility wrapper for sparse-grid insert GPU helpers.

pub use molla_kernels::{
    SparseGridInsertKernel, SparseGridStorage, SparseGridStorageStats, Value, ValueF16, ValueVec3,
    read_storage_to_pairs,
};
