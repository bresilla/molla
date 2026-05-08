//! `Contacts` — collision pipeline output, consumed by solvers.

use crate::math::{Real, Vec3};
use crate::storage::Buffer;

/// Active contact set produced by the collision pipeline. Reset each step.
///
/// `count` is the number of valid entries; the per-contact arrays are
/// sized to `capacity` and only entries `0..count` are meaningful.
#[derive(Debug)]
pub struct Contacts {
    /// Number of currently active contacts.
    pub count: u32,
    /// Allocated capacity. `count <= capacity`.
    pub capacity: usize,
    /// First body/shape index of each pair.
    pub a_index: Buffer<u32>,
    /// Second body/shape index of each pair.
    pub b_index: Buffer<u32>,
    /// World-space contact point.
    pub point: Buffer<Vec3>,
    /// Contact normal pointing from `a` to `b`.
    pub normal: Buffer<Vec3>,
    /// Penetration depth (m).
    pub depth: Buffer<Real>,
    /// Friction coefficient at this contact.
    pub friction: Buffer<Real>,
    /// Restitution coefficient at this contact.
    pub restitution: Buffer<Real>,
}

impl Contacts {
    /// Allocate a `Contacts` set with the given capacity. All entries are
    /// initialised to zero; `count` starts at 0.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        use crate::storage::BufferUsage::StorageReadWrite;
        Self {
            count: 0,
            capacity,
            a_index: Buffer::from_vec(vec![0; capacity], StorageReadWrite),
            b_index: Buffer::from_vec(vec![0; capacity], StorageReadWrite),
            point: Buffer::from_vec(vec![Vec3::ZERO; capacity], StorageReadWrite),
            normal: Buffer::from_vec(vec![Vec3::ZERO; capacity], StorageReadWrite),
            depth: Buffer::from_vec(vec![0.0; capacity], StorageReadWrite),
            friction: Buffer::from_vec(vec![0.0; capacity], StorageReadWrite),
            restitution: Buffer::from_vec(vec![0.0; capacity], StorageReadWrite),
        }
    }

    /// Mark all contacts as inactive (sets `count = 0`). Per-entry buffers
    /// are not zeroed — the solver should only consume `0..count`.
    pub fn clear(&mut self) {
        self.count = 0;
    }
}
