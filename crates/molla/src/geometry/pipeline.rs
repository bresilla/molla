//! End-to-end CPU collision pipeline. `collide` runs broad + narrow
//! phase against the current state and writes the active contact set
//! into `Contacts`.

use crate::core::Result;
use crate::sim::{Contacts, Model, State};
use crate::storage::BufferUsage;

use super::broad_phase::{brute_force_pairs, BroadPhase, ShapePair};
use super::narrow_phase::{pair_contact, ContactManifold};

/// Stateful CPU collision pipeline: caches the SAP scratch buffers
/// across frames. Use this in real simulations; [`collide`] still
/// exists as a stateless brute-force entry-point for tests.
#[derive(Default)]
pub struct CollisionPipeline {
    broad: BroadPhase,
    pair_buf: Vec<ShapePair>,
    manifolds: Vec<ContactManifold>,
    #[cfg(feature = "gpu")]
    gpu_prefilter: Option<GpuPrefilterState>,
}

#[cfg(feature = "gpu")]
struct GpuPrefilterState {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    kernel: crate::kernels::GjkKernel,
    /// Reusable on-device buffers for the GJK pre-filter dispatch.
    body_q: crate::storage::Buffer<crate::math::Transform>,
    pair_a: crate::storage::Buffer<u32>,
    pair_b: crate::storage::Buffer<u32>,
    overlap: crate::storage::Buffer<u32>,
    capacity: usize,
}

impl CollisionPipeline {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable the GPU GJK pre-filter for the broad-phase candidate
    /// list. After SAP produces candidate pairs, the kernel runs in
    /// parallel and reports definite-overlap / definite-disjoint /
    /// fall-back. Pairs the kernel marks definite-disjoint skip the
    /// CPU narrow-phase entirely. `model` must already have its
    /// shape data uploaded to the device.
    #[cfg(feature = "gpu")]
    pub fn enable_gpu_prefilter(
        &mut self,
        engine: &crate::compute::Engine,
        initial_capacity: usize,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| crate::core::Error::Gpu("no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| crate::core::Error::Gpu("no GPU queue".into()))?
            .clone();
        let kernel = crate::kernels::GjkKernel::new(&device);
        let cap = initial_capacity.max(1);
        let (mut pair_a, mut pair_b, mut overlap) =
            crate::kernels::GjkKernel::alloc_io(&device, &queue, cap)?;
        let _ = (&mut pair_a, &mut pair_b, &mut overlap);
        // body_q buffer — sized by the caller's first dispatch.
        let mut body_q = crate::storage::Buffer::from_vec(
            vec![crate::math::Transform::IDENTITY; 1],
            BufferUsage::Storage,
        );
        body_q.ensure_device(&device)?;
        body_q.upload(&queue)?;
        self.gpu_prefilter = Some(GpuPrefilterState {
            device,
            queue,
            kernel,
            body_q,
            pair_a,
            pair_b,
            overlap,
            capacity: cap,
        });
        Ok(())
    }

    /// Run the SAP broad phase + narrow phase, write into `contacts`.
    /// Returns the contact count.
    pub fn collide(
        &mut self,
        model: &Model,
        state: &State,
        contacts: &mut Contacts,
    ) -> Result<usize> {
        contacts.clear();
        if model.shape_count < 2 {
            // No pairs possible; still write an empty body-contact
            // CSR so `body_contact_offsets` / `body_contact_indices`
            // get sized for `body_count`. Solvers + sensors bind
            // these unconditionally and would otherwise hit "no
            // device mirror" on first dispatch.
            push_contacts_into(contacts, &[], model)?;
            return Ok(0);
        }

        self.broad
            .sap_pairs(model, state, model.contact_margin, &mut self.pair_buf);

        // Optional GPU GJK pre-filter — kicks pairs the kernel can
        // definitely resolve (Sphere/Box/Capsule/Cylinder/Cone) out
        // of the candidate list. Pairs flagged "fall back" (Plane,
        // TriMesh, ConvexHull, iter-cap) stay in for CPU narrow phase.
        #[cfg(feature = "gpu")]
        let prefilter_mask: Option<Vec<u32>> = self.run_gpu_prefilter(model, state)?;
        #[cfg(not(feature = "gpu"))]
        let prefilter_mask: Option<Vec<u32>> = None;

        self.manifolds.clear();
        for (i, &pair) in self.pair_buf.iter().enumerate() {
            if let Some(mask) = &prefilter_mask {
                // 0 = definite-disjoint (skip), 1 = overlap (run NP),
                // 2 = fall back to CPU (run NP).
                if mask.get(i).copied().unwrap_or(2) == 0 {
                    continue;
                }
            }
            pair_contact(
                model,
                state,
                pair,
                model.default_friction,
                model.default_restitution,
                &mut self.manifolds,
            );
        }

        push_contacts_into(contacts, &self.manifolds, model)?;
        Ok(self.manifolds.len())
    }

    #[cfg(feature = "gpu")]
    fn run_gpu_prefilter(
        &mut self,
        model: &Model,
        state: &State,
    ) -> Result<Option<Vec<u32>>> {
        if self.pair_buf.is_empty() {
            return Ok(None);
        }
        let pf = match self.gpu_prefilter.as_mut() {
            Some(p) => p,
            None => return Ok(None),
        };
        let n = self.pair_buf.len();
        // Grow the IO buffers if the candidate list got bigger.
        if n > pf.capacity {
            let cap = (n * 2).max(64);
            let (pa, pb, ov) =
                crate::kernels::GjkKernel::alloc_io(&pf.device, &pf.queue, cap)?;
            pf.pair_a = pa;
            pf.pair_b = pb;
            pf.overlap = ov;
            pf.capacity = cap;
        }
        // Push the SAP pair list into the device buffers.
        {
            let pa = pf.pair_a.host_mut()?;
            let pb = pf.pair_b.host_mut()?;
            for (i, p) in self.pair_buf.iter().enumerate() {
                pa[i] = p.a;
                pb[i] = p.b;
            }
        }
        pf.pair_a.upload(&pf.queue)?;
        pf.pair_b.upload(&pf.queue)?;

        // Mirror state.body_q into the prefilter's own buffer (the
        // pipeline doesn't own the simulation state, so we can't
        // re-use the solver's). For body-light scenes this is a tiny
        // copy.
        let body_count = state.body_q.host()?.len();
        if pf.body_q.host()?.len() < body_count {
            pf.body_q = crate::storage::Buffer::from_vec(
                vec![crate::math::Transform::IDENTITY; body_count.max(1)],
                BufferUsage::Storage,
            );
            pf.body_q.ensure_device(&pf.device)?;
        }
        {
            let dst = pf.body_q.host_mut()?;
            let src = state.body_q.host()?;
            dst[..src.len()].copy_from_slice(src);
        }
        pf.body_q.upload(&pf.queue)?;

        let body_q_buf = pf
            .body_q
            .device_buffer()
            .ok_or_else(|| crate::core::Error::Gpu("body_q device_buffer".into()))?;
        pf.kernel.dispatch(
            &pf.device,
            &pf.queue,
            model,
            body_q_buf,
            &pf.pair_a,
            &pf.pair_b,
            &mut pf.overlap,
            n as u32,
        )?;
        pf.overlap.download(&pf.device, &pf.queue)?;
        let mask = pf.overlap.host()?[..n].to_vec();
        Ok(Some(mask))
    }
}

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
        // No pairs possible; still build an **empty** body-contact CSR
        // so `body_contact_offsets` / `body_contact_indices` get sized
        // for `body_count` (and gain a GPU device buffer on the next
        // `upload_to_device`). Solvers / sensors that bind these
        // unconditionally would otherwise hit "no device mirror".
        push_contacts_into(contacts, &[], model)?;
        return Ok(0);
    }

    // Generate AABB-overlap pairs.
    brute_force_pairs(model, state, model.contact_margin, pair_buf);

    // Narrow phase + push into Contacts. Each pair can yield 0..N
    // manifolds (boxes can give up to 4).
    let mut manifolds: Vec<ContactManifold> = Vec::with_capacity(pair_buf.len());
    for &pair in pair_buf.iter() {
        pair_contact(
            model,
            state,
            pair,
            model.default_friction,
            model.default_restitution,
            &mut manifolds,
        );
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
        contacts.rolling_friction = crate::storage::Buffer::from_vec(vec![0.0; cap], BufferUsage::StorageReadWrite);
        contacts.torsional_friction = crate::storage::Buffer::from_vec(vec![0.0; cap], BufferUsage::StorageReadWrite);
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
    let roll = contacts.rolling_friction.host_mut()?;
    for (i, m) in manifolds.iter().enumerate() {
        roll[i] = m.rolling_friction;
    }
    let tors = contacts.torsional_friction.host_mut()?;
    for (i, m) in manifolds.iter().enumerate() {
        tors[i] = m.torsional_friction;
    }
    let rest = contacts.restitution.host_mut()?;
    for (i, m) in manifolds.iter().enumerate() {
        rest[i] = m.restitution;
    }

    contacts.count = n as u32;

    // Build the per-body CSR that the GPU XPBD solver consumes.
    // Two passes: count, then scatter.
    build_body_contact_csr(contacts, model)?;
    Ok(())
}

fn build_body_contact_csr(contacts: &mut Contacts, model: &Model) -> Result<()> {
    let body_count = model.body_count;
    let n = contacts.count as usize;

    // Count per-body contact participation. Skip STATIC_BODY peers —
    // there's no body slot to update.
    let mut counts = vec![0u32; body_count];
    {
        let a = contacts.a_index.host()?;
        let b = contacts.b_index.host()?;
        for c in 0..n {
            let ai = a[c];
            let bi = b[c];
            if ai != STATIC_BODY && (ai as usize) < body_count {
                counts[ai as usize] += 1;
            }
            if bi != STATIC_BODY && (bi as usize) < body_count {
                counts[bi as usize] += 1;
            }
        }
    }

    // Prefix-sum into offsets [body_count + 1].
    let mut offsets = Vec::with_capacity(body_count + 1);
    offsets.push(0u32);
    let mut acc = 0u32;
    for &c in &counts {
        acc += c;
        offsets.push(acc);
    }
    let total = acc as usize;

    // Scatter — use a per-body cursor to fill `indices`.
    let mut indices = vec![0u32; total];
    let mut cursor = vec![0u32; body_count];
    {
        let a = contacts.a_index.host()?;
        let b = contacts.b_index.host()?;
        for c in 0..n {
            let ai = a[c];
            let bi = b[c];
            let c32 = c as u32;
            if ai != STATIC_BODY && (ai as usize) < body_count {
                let ix = offsets[ai as usize] + cursor[ai as usize];
                indices[ix as usize] = c32; // top bit clear → a-side
                cursor[ai as usize] += 1;
            }
            if bi != STATIC_BODY && (bi as usize) < body_count {
                let ix = offsets[bi as usize] + cursor[bi as usize];
                indices[ix as usize] = c32 | Contacts::B_SIDE_BIT;
                cursor[bi as usize] += 1;
            }
        }
    }

    // Resize buffers if needed and write.
    let need_offsets = body_count + 1;
    if contacts.body_contact_offsets.len() != need_offsets {
        contacts.body_contact_offsets =
            Buffer::from_vec(offsets, BufferUsage::StorageReadWrite);
    } else {
        contacts.body_contact_offsets.host_mut()?.copy_from_slice(&offsets);
    }
    if contacts.body_contact_indices.len() < total.max(1) {
        // Round up so we don't reallocate every frame on minor jitter.
        let cap = total.max(1).next_power_of_two();
        let mut buf = vec![0u32; cap];
        buf[..total].copy_from_slice(&indices);
        contacts.body_contact_indices = Buffer::from_vec(buf, BufferUsage::StorageReadWrite);
    } else {
        let dst = contacts.body_contact_indices.host_mut()?;
        dst[..total].copy_from_slice(&indices);
    }

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
