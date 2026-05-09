//! `Contacts` — collision pipeline output, consumed by solvers.

use crate::math::{Real, Vec3};
use crate::storage::Buffer;

/// Active contact set produced by the collision pipeline. Reset each step.
///
/// `count` is the number of valid entries; the per-contact arrays are
/// sized to `capacity` and only entries `0..count` are meaningful.
///
/// **Per-body indexing**: the GPU XPBD solver iterates contacts
/// per-body — each thread processes the contacts that involve its
/// body without writing to anyone else's. The CSR layout
/// `body_contact_offsets` (length `body_count + 1`) +
/// `body_contact_indices` (length `2 * count` because every contact
/// shows up under both endpoints) is built by the collision pipeline
/// after each broad+narrow pass. Static-world peers (`STATIC_BODY`)
/// are deliberately excluded from the index — there's no body slot
/// to update for them.
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
    /// Rolling friction (μ_rolling) at this contact — opposes the
    /// angular velocity component perpendicular to the normal.
    pub rolling_friction: Buffer<Real>,
    /// Torsional friction (μ_torsional) at this contact — opposes
    /// the angular velocity component ALONG the normal (spinning).
    pub torsional_friction: Buffer<Real>,
    /// Restitution coefficient at this contact.
    pub restitution: Buffer<Real>,

    /// CSR offsets into `body_contact_indices`, length `body_count + 1`.
    /// Body `i` owns entries `body_contact_indices[offsets[i]..offsets[i+1]]`.
    pub body_contact_offsets: Buffer<u32>,
    /// Packed contact indices, with the **sign bit reused** to encode
    /// which side of the contact the body is on:
    /// - top bit clear (`< 0x8000_0000`): body is the `a` side
    /// - top bit set: body is the `b` side, real index =
    ///   `value & 0x7fff_ffff`
    ///
    /// Decode helper: see [`Contacts::decode_index`].
    pub body_contact_indices: Buffer<u32>,
}

impl Contacts {
    /// Bit set in [`Contacts::body_contact_indices`] when the body is
    /// the `b` side of the contact.
    pub const B_SIDE_BIT: u32 = 0x8000_0000;

    /// Decode an entry from `body_contact_indices` into `(contact_idx, is_b_side)`.
    #[must_use]
    pub fn decode_index(packed: u32) -> (u32, bool) {
        (packed & !Self::B_SIDE_BIT, packed & Self::B_SIDE_BIT != 0)
    }

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
            rolling_friction: Buffer::from_vec(vec![0.0; capacity], StorageReadWrite),
            torsional_friction: Buffer::from_vec(vec![0.0; capacity], StorageReadWrite),
            restitution: Buffer::from_vec(vec![0.0; capacity], StorageReadWrite),
            body_contact_offsets: Buffer::from_vec(Vec::new(), StorageReadWrite),
            body_contact_indices: Buffer::from_vec(Vec::new(), StorageReadWrite),
        }
    }

    /// Mark all contacts as inactive (sets `count = 0`). Per-entry buffers
    /// are not zeroed — the solver should only consume `0..count`.
    pub fn clear(&mut self) {
        self.count = 0;
    }

    /// Allocate device buffers and upload host data. Re-uploads every
    /// frame (the contact set turns over completely each step).
    #[cfg(feature = "gpu")]
    pub fn upload_to_device(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> crate::core::Result<()> {
        macro_rules! up {
            ($field:ident) => {{
                self.$field.ensure_device(device)?;
                self.$field.upload(queue)?;
            }};
        }
        up!(a_index);
        up!(b_index);
        up!(point);
        up!(normal);
        up!(depth);
        up!(friction);
        up!(rolling_friction);
        up!(torsional_friction);
        up!(restitution);
        up!(body_contact_offsets);
        up!(body_contact_indices);
        Ok(())
    }
}
