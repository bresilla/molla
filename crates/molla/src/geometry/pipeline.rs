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
    #[cfg(feature = "gpu")]
    gpu_full_np: Option<GpuFullNarrowPhase>,
    /// Stage-19 GPU LBVH broad phase. When set, replaces the CPU SAP
    /// pair generator. Output flows into `pair_buf` exactly the same
    /// way SAP would, and the rest of the pipeline (prefilter / full
    /// narrow phase / CPU NP fallback) is unchanged.
    #[cfg(feature = "gpu")]
    gpu_broad: Option<GpuBroadPhaseState>,
    /// Number of times the CPU narrow-phase `pair_contact` was
    /// invoked on the most recent `collide` call. Useful for
    /// asserting that a fully-GPU pipeline doesn't fall back to CPU
    /// in steady state.
    last_cpu_np_calls: usize,
    /// Number of pairs returned by the broad phase on the most
    /// recent `collide` call (post-filter). Surfaces in the GPU
    /// broad-phase demo so the user can compare CPU SAP vs GPU LBVH
    /// pair counts side-by-side.
    last_broad_phase_pairs: usize,
}

#[cfg(feature = "gpu")]
struct GpuBroadPhaseState {
    lbvh: crate::kernels::LbvhBroadPhase,
    /// If `true`, reuse the previous frame's tree topology and only
    /// re-run refit + pair-finder. Useful for cloth / soft-body
    /// scenes where shape topology is fixed across frames.
    refit_only: bool,
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

#[cfg(feature = "gpu")]
struct GpuFullNarrowPhase {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    gjk: crate::kernels::GjkKernel,
    epa: crate::kernels::EpaKernel,
    body_q: crate::storage::Buffer<crate::math::Transform>,
    pair_a: crate::storage::Buffer<u32>,
    pair_b: crate::storage::Buffer<u32>,
    overlap: crate::storage::Buffer<u32>,
    simplex: crate::storage::Buffer<crate::math::Vec3>,
    epa_out: crate::storage::Buffer<crate::math::Real>,
    capacity: usize,
}

impl CollisionPipeline {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of pairs that fell back to the CPU narrow phase on the
    /// most recent `collide` call. With `enable_full_gpu_narrow_phase`
    /// active and a fully GPU-supported scene, this should be 0
    /// (Plane / TriMesh / Heightfield / SDF and any iter-cap or
    /// polytope-cap fallbacks bump it).
    #[must_use]
    pub fn last_cpu_narrow_phase_calls(&self) -> usize {
        self.last_cpu_np_calls
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

    /// Enable the **full GPU narrow phase**: GJK + EPA on the device
    /// produce contact manifolds for every Sphere/Box/Capsule/Cylinder/
    /// Cone pair. Plane, TriMesh, Heightfield, SDF, ConvexHull, and
    /// Triangle pairs still flow through the CPU narrow phase (the
    /// device-side support functions don't cover them). When this is
    /// active, `enable_gpu_prefilter` is irrelevant — the GJK pass run
    /// here doubles as the prefilter.
    #[cfg(feature = "gpu")]
    pub fn enable_full_gpu_narrow_phase(
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
        let gjk = crate::kernels::GjkKernel::new(&device);
        let epa = crate::kernels::EpaKernel::new(&device);
        let cap = initial_capacity.max(1);
        let (pair_a, pair_b, overlap) =
            crate::kernels::GjkKernel::alloc_io(&device, &queue, cap)?;
        let simplex = crate::kernels::GjkKernel::alloc_simplex(&device, &queue, cap)?;
        let epa_out = crate::kernels::EpaKernel::alloc_out(&device, &queue, cap)?;
        let mut body_q = crate::storage::Buffer::from_vec(
            vec![crate::math::Transform::IDENTITY; 1],
            BufferUsage::Storage,
        );
        body_q.ensure_device(&device)?;
        body_q.upload(&queue)?;
        self.gpu_full_np = Some(GpuFullNarrowPhase {
            device,
            queue,
            gjk,
            epa,
            body_q,
            pair_a,
            pair_b,
            overlap,
            simplex,
            epa_out,
            capacity: cap,
        });
        Ok(())
    }

    /// Enable the **full GPU broad phase** (Stage 19): replace the
    /// CPU SAP pair generator with a GPU LBVH (Linear BVH) build +
    /// pair-finder. The output (`pair_buf`) is identical in content
    /// to SAP — same set of overlapping AABB pairs, just sorted in a
    /// different order. The narrow phase / GPU prefilter / full GPU
    /// narrow phase all consume the result transparently.
    ///
    /// Set `refit_only` to `true` for scenes where shape topology is
    /// fixed but per-shape AABBs change every frame (cloth, soft
    /// bodies, deformable particles) — saves the Morton-sort + tree
    /// build cost.
    #[cfg(feature = "gpu")]
    pub fn enable_full_gpu_broad_phase(
        &mut self,
        engine: &crate::compute::Engine,
        initial_prim_capacity: usize,
        initial_pair_capacity: usize,
        refit_only: bool,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| crate::core::Error::Gpu("no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| crate::core::Error::Gpu("no GPU queue".into()))?
            .clone();
        let lbvh = crate::kernels::LbvhBroadPhase::new(
            device,
            queue,
            initial_prim_capacity,
            initial_pair_capacity,
        )?;
        self.gpu_broad = Some(GpuBroadPhaseState { lbvh, refit_only });
        Ok(())
    }

    /// Toggle the cached LBVH refit-only flag at runtime. No-op when
    /// the GPU broad phase isn't enabled.
    #[cfg(feature = "gpu")]
    pub fn set_lbvh_refit_only(&mut self, refit_only: bool) {
        if let Some(state) = self.gpu_broad.as_mut() {
            state.refit_only = refit_only;
        }
    }

    /// Number of pairs returned by the broad phase on the most
    /// recent `collide` call (post pair-filter dedup, excluding the
    /// plane-vs-everything implicit set when GPU broad phase is on,
    /// which contributes to this count via the LBVH path's natural
    /// pair output).
    #[must_use]
    pub fn last_broad_phase_pair_count(&self) -> usize {
        self.last_broad_phase_pairs
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
        self.last_cpu_np_calls = 0;
        self.last_broad_phase_pairs = 0;
        if model.shape_count < 2 {
            // No pairs possible; still write an empty body-contact
            // CSR so `body_contact_offsets` / `body_contact_indices`
            // get sized for `body_count`. Solvers + sensors bind
            // these unconditionally and would otherwise hit "no
            // device mirror" on first dispatch.
            push_contacts_into(contacts, &[], model)?;
            return Ok(0);
        }

        // Pick GPU LBVH broad phase if enabled, otherwise CPU SAP.
        #[cfg(feature = "gpu")]
        let used_gpu_broad = self.run_gpu_broad_phase(model, state)?;
        #[cfg(not(feature = "gpu"))]
        let used_gpu_broad = false;
        if !used_gpu_broad {
            self.broad
                .sap_pairs(model, state, model.contact_margin, &mut self.pair_buf);
        }
        self.last_broad_phase_pairs = self.pair_buf.len();

        // Full GPU narrow phase takes precedence over the prefilter.
        #[cfg(feature = "gpu")]
        if self.gpu_full_np.is_some() {
            self.manifolds.clear();
            self.run_gpu_full_narrow_phase(model, state)?;
            push_contacts_into(contacts, &self.manifolds, model)?;
            return Ok(self.manifolds.len());
        }

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
            self.last_cpu_np_calls += 1;
        }

        push_contacts_into(contacts, &self.manifolds, model)?;
        Ok(self.manifolds.len())
    }

    /// Run the full GPU narrow phase. Splits pairs into:
    ///   * **GPU-eligible** — both shapes are Sphere/Box/Capsule/
    ///     Cylinder/Cone. GJK + EPA on device produces manifolds
    ///     directly. EPA fallbacks (status==2) are re-run on CPU.
    ///   * **CPU-only** — at least one shape is Plane / TriMesh /
    ///     Heightfield / SDF / ConvexHull / Triangle. Routed to the
    ///     existing CPU `pair_contact`.
    #[cfg(feature = "gpu")]
    fn run_gpu_full_narrow_phase(
        &mut self,
        model: &Model,
        state: &State,
    ) -> Result<()> {
        // Partition pair indices into GPU vs CPU. Borrow the buffer
        // before taking a mutable reference into `gpu_full_np`.
        let mut gpu_indices: Vec<usize> = Vec::with_capacity(self.pair_buf.len());
        let mut cpu_indices: Vec<usize> = Vec::with_capacity(self.pair_buf.len());
        for (i, p) in self.pair_buf.iter().enumerate() {
            let ka = model.shape_geometry[p.a as usize];
            let kb = model.shape_geometry[p.b as usize];
            if is_gpu_eligible(ka) && is_gpu_eligible(kb) {
                gpu_indices.push(i);
            } else {
                cpu_indices.push(i);
            }
        }

        // CPU-only pairs run first (cheap; many bound to fall through
        // sphere/box fast paths). Tracks the call counter.
        for &i in &cpu_indices {
            let pair = self.pair_buf[i];
            pair_contact(
                model,
                state,
                pair,
                model.default_friction,
                model.default_restitution,
                &mut self.manifolds,
            );
            self.last_cpu_np_calls += 1;
        }

        // GPU pairs: dispatch GJK with simplex + EPA, then translate
        // each device contact back into a `ContactManifold` and push.
        if gpu_indices.is_empty() {
            return Ok(());
        }
        let n = gpu_indices.len();
        let pf = self.gpu_full_np.as_mut().expect("gpu_full_np");
        if n > pf.capacity {
            let cap = (n * 2).max(64);
            let (pa, pb, ov) =
                crate::kernels::GjkKernel::alloc_io(&pf.device, &pf.queue, cap)?;
            pf.pair_a = pa;
            pf.pair_b = pb;
            pf.overlap = ov;
            pf.simplex =
                crate::kernels::GjkKernel::alloc_simplex(&pf.device, &pf.queue, cap)?;
            pf.epa_out = crate::kernels::EpaKernel::alloc_out(&pf.device, &pf.queue, cap)?;
            pf.capacity = cap;
        }
        {
            let pa_h = pf.pair_a.host_mut()?;
            let pb_h = pf.pair_b.host_mut()?;
            for (slot, &i) in gpu_indices.iter().enumerate() {
                let p = self.pair_buf[i];
                pa_h[slot] = p.a;
                pb_h[slot] = p.b;
            }
        }
        pf.pair_a.upload(&pf.queue)?;
        pf.pair_b.upload(&pf.queue)?;

        // Mirror state.body_q.
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

        pf.gjk.dispatch_with_simplex(
            &pf.device,
            &pf.queue,
            model,
            body_q_buf,
            &pf.pair_a,
            &pf.pair_b,
            &mut pf.overlap,
            &mut pf.simplex,
            n as u32,
        )?;
        pf.epa.dispatch(
            &pf.device,
            &pf.queue,
            model,
            body_q_buf,
            &pf.pair_a,
            &pf.pair_b,
            &pf.overlap,
            &pf.simplex,
            &mut pf.epa_out,
            n as u32,
        )?;
        pf.overlap.download(&pf.device, &pf.queue)?;
        pf.epa_out.download(&pf.device, &pf.queue)?;

        let overlap = pf.overlap.host()?[..n].to_vec();
        let epa_host = pf.epa_out.host()?;
        let results = crate::kernels::decode_epa_out(epa_host, n);

        for (slot, &i) in gpu_indices.iter().enumerate() {
            let pair = self.pair_buf[i];
            let ov = overlap[slot];
            let res = results[slot];
            if ov == 0 {
                continue; // disjoint
            }
            // Either ov == 2 (GJK fell back) or status == FALLBACK
            // (EPA polytope cap / iter cap). Re-run on CPU.
            if ov == 2
                || res.status == crate::kernels::EPA_STATUS_FALLBACK
                || res.status == crate::kernels::EPA_STATUS_NONE
            {
                pair_contact(
                    model,
                    state,
                    pair,
                    model.default_friction,
                    model.default_restitution,
                    &mut self.manifolds,
                );
                self.last_cpu_np_calls += 1;
                continue;
            }
            // Build a `ContactManifold` from the GPU EPA result. The
            // contact-point convention matches CPU EPA: midpoint
            // between point_a and point_b.
            let pair_friction = pair_material(
                model.shape_friction.host()?,
                pair.a as usize,
                pair.b as usize,
                model.default_friction,
            );
            let pair_restitution = pair_material(
                model.shape_restitution.host()?,
                pair.a as usize,
                pair.b as usize,
                model.default_restitution,
            );
            let pair_rolling = pair_material(
                model.shape_rolling_friction.host()?,
                pair.a as usize,
                pair.b as usize,
                0.0,
            );
            let pair_torsional = pair_material(
                model.shape_torsional_friction.host()?,
                pair.a as usize,
                pair.b as usize,
                0.0,
            );
            let normal_len_sq = res.normal.length_squared();
            if normal_len_sq < 1e-12 {
                continue;
            }
            let normal = res.normal.normalize();
            let point = (res.point_a + res.point_b) * 0.5;
            self.manifolds.push(ContactManifold {
                a_shape: pair.a,
                b_shape: pair.b,
                point,
                normal,
                depth: res.depth as crate::math::Real,
                friction: pair_friction,
                rolling_friction: pair_rolling,
                torsional_friction: pair_torsional,
                restitution: pair_restitution,
            });
        }
        Ok(())
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

    /// Stage-19 GPU LBVH broad phase. Runs the LBVH build (or refit)
    /// + pair-finder, then post-filters the raw AABB pairs through
    /// `pair_passes_filters` (collision groups, self-collision,
    /// world-id partitioning) so the downstream narrow phase sees the
    /// same set the CPU SAP would. Returns `true` if the GPU broad
    /// phase ran (so the caller can skip CPU SAP).
    #[cfg(feature = "gpu")]
    fn run_gpu_broad_phase(&mut self, model: &Model, state: &State) -> Result<bool> {
        let bp = match self.gpu_broad.as_mut() {
            Some(b) => b,
            None => return Ok(false),
        };
        let pairs = if bp.refit_only {
            bp.lbvh
                .refit_from_model(model, state, model.contact_margin)?
        } else {
            bp.lbvh
                .dispatch_from_model(model, state, model.contact_margin)?
        };

        // Apply the same pair-level filters CPU SAP applies (collision
        // groups, world-id partitioning, explicit exclusions, static-
        // pair drop). Order pairs as `(min, max)` to match SAP output.
        self.pair_buf.clear();
        let shape_body = model.shape_body.host()?;
        let body_world = model.body_world.host()?;
        for k in 0..pairs.a.len() {
            let i = pairs.a[k];
            let j = pairs.b[k];
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            if !super::broad_phase::pair_passes_filters(
                model,
                shape_body,
                body_world,
                a as usize,
                b as usize,
            ) {
                continue;
            }
            self.pair_buf.push(super::ShapePair { a, b });
        }
        Ok(true)
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

#[cfg(feature = "gpu")]
fn is_gpu_eligible(g: crate::sim::ShapeGeometry) -> bool {
    use crate::sim::ShapeGeometry as G;
    matches!(
        g,
        G::Sphere { .. }
            | G::Box { .. }
            | G::Capsule { .. }
            | G::Cylinder { .. }
            | G::Cone { .. }
    )
}

#[cfg(feature = "gpu")]
fn pair_material(
    arr: &[crate::math::Real],
    a: usize,
    b: usize,
    default: crate::math::Real,
) -> crate::math::Real {
    // Per-shape material arrays are guaranteed finite by `ModelBuilder`
    // (NaN sentinels are substituted with defaults at finalize). A NaN
    // here would corrupt the pair friction/restitution silently — let
    // debug builds catch it.
    let ma = arr.get(a).copied().unwrap_or(default);
    let mb = arr.get(b).copied().unwrap_or(default);
    debug_assert!(ma.is_finite(), "pair_material: shape {a} material is not finite ({ma})");
    debug_assert!(mb.is_finite(), "pair_material: shape {b} material is not finite ({mb})");
    ma.max(mb)
}
