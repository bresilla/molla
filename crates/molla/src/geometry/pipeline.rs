//! End-to-end CPU collision pipeline. `collide` runs broad + narrow
//! phase against the current state and writes the active contact set
//! into `Contacts`.

use crate::core::Result;
use crate::sim::{Contacts, Model, State};
use crate::storage::BufferUsage;

use super::broad_phase::{brute_force_pairs, ShapePair};
use super::narrow_phase::{pair_contact, ContactManifold};

/// Top-level CPU collision pipeline. Mutates `contacts` in place;
/// returns the number of contacts found.
///
/// `pair_buf` is a scratch vector reused across calls — pass an
/// empty `Vec` once and re-pass the same one each frame to avoid
/// reallocating.
pub fn collide(
    model: &Model,
    state: &State,
    contacts: &mut Contacts,
    pair_buf: &mut Vec<ShapePair>,
) -> Result<usize> {
    contacts.clear();

    if model.shape_count < 2 {
        return Ok(0);
    }

    // Generate AABB-overlap pairs.
    brute_force_pairs(model, state, model.contact_margin, pair_buf);

    // Narrow phase + push into Contacts.
    let mut manifolds: Vec<ContactManifold> = Vec::with_capacity(pair_buf.len());
    for &pair in pair_buf.iter() {
        if let Some(m) = pair_contact(
            model,
            state,
            pair,
            model.default_friction,
            model.default_restitution,
        ) {
            manifolds.push(m);
        }
    }

    push_contacts_into(contacts, &manifolds, model)?;
    Ok(manifolds.len())
}

fn push_contacts_into(
    contacts: &mut Contacts,
    manifolds: &[ContactManifold],
    model: &Model,
) -> Result<()> {
    let n = manifolds.len();

    // If we'd overflow the existing capacity, regrow the buffers.
    if n > contacts.capacity {
        let cap = (n.max(64)).next_power_of_two();
        contacts.capacity = cap;
        contacts.a_index = crate::storage::Buffer::from_vec(vec![0u32; cap], BufferUsage::StorageReadWrite);
        contacts.b_index = crate::storage::Buffer::from_vec(vec![0u32; cap], BufferUsage::StorageReadWrite);
        contacts.point = crate::storage::Buffer::from_vec(
            vec![crate::math::Vec3::ZERO; cap],
            BufferUsage::StorageReadWrite,
        );
        contacts.normal = crate::storage::Buffer::from_vec(
            vec![crate::math::Vec3::ZERO; cap],
            BufferUsage::StorageReadWrite,
        );
        contacts.depth = crate::storage::Buffer::from_vec(vec![0.0; cap], BufferUsage::StorageReadWrite);
        contacts.friction = crate::storage::Buffer::from_vec(vec![0.0; cap], BufferUsage::StorageReadWrite);
        contacts.restitution = crate::storage::Buffer::from_vec(vec![0.0; cap], BufferUsage::StorageReadWrite);
    }

    let shape_body = model.shape_body.host()?;

    let a_index = contacts.a_index.host_mut()?;
    let b_index = contacts.b_index.host_mut()?;
    for (i, m) in manifolds.iter().enumerate() {
        // Translate shape→body so solvers can apply impulses to the
        // owning body. -1 (static) stays as a sentinel.
        a_index[i] = body_or_static(shape_body[m.a_shape as usize]);
        b_index[i] = body_or_static(shape_body[m.b_shape as usize]);
    }

    let point = contacts.point.host_mut()?;
    for (i, m) in manifolds.iter().enumerate() {
        point[i] = m.point;
    }
    let normal = contacts.normal.host_mut()?;
    for (i, m) in manifolds.iter().enumerate() {
        normal[i] = m.normal;
    }
    let depth = contacts.depth.host_mut()?;
    for (i, m) in manifolds.iter().enumerate() {
        depth[i] = m.depth;
    }
    let fric = contacts.friction.host_mut()?;
    for (i, m) in manifolds.iter().enumerate() {
        fric[i] = m.friction;
    }
    let rest = contacts.restitution.host_mut()?;
    for (i, m) in manifolds.iter().enumerate() {
        rest[i] = m.restitution;
    }

    contacts.count = n as u32;
    Ok(())
}

fn body_or_static(idx: i32) -> u32 {
    if idx < 0 {
        u32::MAX
    } else {
        idx as u32
    }
}

/// Sentinel value written into `Contacts.a_index` / `b_index` when
/// the shape has no owning body (static-world). Solvers check this
/// to know they should skip updating that side.
pub const STATIC_BODY: u32 = u32::MAX;

#[allow(unused_imports)]
use crate::storage::Buffer; // referenced via fully-qualified paths above
