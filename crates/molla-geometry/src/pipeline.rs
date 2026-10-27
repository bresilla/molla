//! End-to-end CPU collision pipeline. `collide` runs broad + narrow
//! phase against the current state and writes the active contact set
//! into `Contacts`.

use molla_core::Result;
#[cfg(feature = "gpu")]
use molla_math::Real;
use molla_sim::{Contacts, Model, ShapeGeometry, State};
use molla_storage::BufferUsage;

use super::broad_phase::{BroadPhase, ShapePair, brute_force_pairs};
use super::narrow_phase::{ContactManifold, pair_contact, pair_contact_cached};
use super::shape_world::shape_world_transform;

/// Selectable narrow-phase backend used by [`CollisionPipeline`].
///
/// The default is [`NarrowPhaseBackend::Gjk`] — the legacy CPU GJK +
/// EPA path. Two opt-in alternatives are available since Stage 28.1:
///
/// * [`NarrowPhaseBackend::MprCpu`] — runs the CPU MPR portal
///   refinement ([`super::mpr_collide`]) on convex pairs. Prefer
///   for deep-penetration scenes (cloth reset states, MPM
///   couplings) where EPA's polytope expansion can stall.
/// * [`NarrowPhaseBackend::MprGpu`] — batches all eligible convex
///   pairs into a single dispatch on the [`molla_geometry::MprKernel`].
///   Prefer for high pair counts (≥ 256) on a `gpu`-feature build
///   where the per-pair work amortizes the launch + readback.
///
/// Pairs the chosen backend cannot resolve (Plane / TriMesh /
/// Heightfield / SDF / Triangle / GPU-status `FALLBACK`) transparently
/// fall through to the CPU GJK + EPA path so callers always get a
/// complete manifold set.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum NarrowPhaseBackend {
    /// CPU GJK + EPA (default). Robust, no GPU dependency, exhaustive
    /// shape-type support via the existing `pair_contact` dispatch.
    #[default]
    Gjk,
    /// CPU MPR portal refinement. One algorithm for both overlap and
    /// penetration depth — preferred when EPA's polytope grows
    /// numerically unstable. Falls back to GJK + EPA on shape pairs
    /// MPR's support map can't handle (Plane / TriMesh / Heightfield /
    /// SDF / Triangle on either side).
    MprCpu,
    /// GPU MPR portal refinement ([`molla_geometry::MprKernel`]).
    /// Batches all convex pairs through one dispatch and only re-runs
    /// the FALLBACK / unsupported pairs through CPU. Requires the
    /// `gpu` feature.
    MprGpu,
}

/// Stateful CPU collision pipeline: caches the SAP scratch buffers
/// across frames. Use this in real simulations; [`collide`] still
/// exists as a stateless brute-force entry-point for tests.
///
/// # Backend selection
///
/// The narrow-phase backend is selectable at runtime via
/// [`Self::set_narrow_phase_backend`] / [`Self::with_narrow_phase_backend`]:
///
/// * [`NarrowPhaseBackend::Gjk`] (default) — CPU GJK + EPA via
///   [`pair_contact`]. Best general-purpose choice; no GPU dependency
///   and exhaustive shape-type support.
/// * [`NarrowPhaseBackend::MprCpu`] — CPU MPR portal refinement.
///   Prefer for deep-penetration scenes where EPA polytope expansion
///   stalls; not generally faster than GJK on shallow contact.
/// * [`NarrowPhaseBackend::MprGpu`] — GPU MPR batch dispatch. Wins
///   on large pair counts (≥ 256) where the per-dispatch overhead is
///   amortized; otherwise the CPU paths are cheaper.
///
/// Selecting `MprCpu` or `MprGpu` is independent of the existing GPU
/// prefilter / full GPU GJK+EPA narrow phase. When the full GPU
/// narrow phase is enabled it takes precedence over the MPR backend
/// choice (both run on the GPU, but only one at a time).
#[derive(Default)]
pub struct CollisionPipeline {
    extra_shape_exclusions: std::collections::BTreeSet<(u32, u32)>,
    include_sensor_overlaps: bool,
    broad: BroadPhase,
    pair_buf: Vec<ShapePair>,
    /// The pairs the CPU narrow phase takes, kept for its storage.
    narrow_pairs: Vec<ShapePair>,
    manifolds: Vec<ContactManifold>,
    mesh_cache: crate::mesh_cache::MeshBvhCache,
    /// Narrow-phase algorithm used for each pair. Defaults to
    /// [`NarrowPhaseBackend::Gjk`] so existing behaviour is unchanged.
    narrow_phase_backend: NarrowPhaseBackend,
    /// Lazily-constructed MPR GPU state. Created on the first
    /// `collide` call when `narrow_phase_backend == MprGpu` and the
    /// pipeline has been configured via `enable_mpr_gpu_backend`.
    #[cfg(feature = "gpu")]
    mpr_gpu_state: Option<MprGpuState>,
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
struct MprGpuState {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    kernel: crate::MprKernel,
    body_q: molla_storage::Buffer<molla_math::Transform>,
    pair_a: molla_storage::Buffer<u32>,
    pair_b: molla_storage::Buffer<u32>,
    mpr_out: molla_storage::Buffer<Real>,
    capacity: usize,
    tolerance: f32,
}

#[cfg(feature = "gpu")]
struct GpuBroadPhaseState {
    lbvh: crate::LbvhBroadPhase,
    /// If `true`, reuse the previous frame's tree topology and only
    /// re-run refit + pair-finder. Useful for cloth / soft-body
    /// scenes where shape topology is fixed across frames.
    refit_only: bool,
}

#[cfg(feature = "gpu")]
struct GpuPrefilterState {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    kernel: crate::GjkKernel,
    /// Reusable on-device buffers for the GJK pre-filter dispatch.
    body_q: molla_storage::Buffer<molla_math::Transform>,
    pair_a: molla_storage::Buffer<u32>,
    pair_b: molla_storage::Buffer<u32>,
    overlap: molla_storage::Buffer<u32>,
    capacity: usize,
}

#[cfg(feature = "gpu")]
struct GpuFullNarrowPhase {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    gjk: crate::GjkKernel,
    epa: crate::EpaKernel,
    body_q: molla_storage::Buffer<molla_math::Transform>,
    pair_a: molla_storage::Buffer<u32>,
    pair_b: molla_storage::Buffer<u32>,
    overlap: molla_storage::Buffer<u32>,
    simplex: molla_storage::Buffer<molla_math::Vec3>,
    epa_out: molla_storage::Buffer<molla_math::Real>,
    capacity: usize,
}

impl CollisionPipeline {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of triangle-mesh BVHs built by this pipeline since construction.
    pub fn mesh_bvh_build_count(&self) -> usize {
        self.mesh_cache.builds()
    }

    /// Include sensor geometry for overlap queries or solvers that suppress sensor forces.
    pub fn with_sensor_overlaps(mut self, include: bool) -> Self {
        self.include_sensor_overlaps = include;
        self
    }

    /// Select the narrow-phase backend. Pairs the backend can't
    /// resolve (Plane / TriMesh / Heightfield / SDF / Triangle on
    /// either side, or GPU `FALLBACK`) fall through to CPU GJK + EPA.
    pub fn set_narrow_phase_backend(&mut self, backend: NarrowPhaseBackend) {
        self.narrow_phase_backend = backend;
    }

    /// Fluent variant of [`Self::set_narrow_phase_backend`] for use at
    /// construction.
    #[must_use]
    pub fn with_narrow_phase_backend(mut self, backend: NarrowPhaseBackend) -> Self {
        self.narrow_phase_backend = backend;
        self
    }

    /// Currently-selected narrow-phase backend.
    #[must_use]
    pub fn narrow_phase_backend(&self) -> NarrowPhaseBackend {
        self.narrow_phase_backend
    }

    /// Pre-allocate the GPU MPR resources so the first `collide` call
    /// with `narrow_phase_backend == MprGpu` doesn't pay the
    /// initialization cost. Optional — `collide` will lazily
    /// initialize on first use if this hasn't been called.
    #[cfg(feature = "gpu")]
    pub fn enable_mpr_gpu_backend(
        &mut self,
        engine: &molla_compute::Engine,
        initial_capacity: usize,
        tolerance: f32,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| molla_core::Error::Gpu("no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| molla_core::Error::Gpu("no GPU queue".into()))?
            .clone();
        let kernel = crate::MprKernel::new(&device);
        let cap = initial_capacity.max(1);
        let (pair_a, pair_b) = crate::MprKernel::alloc_pairs(&device, &queue, cap)?;
        let mpr_out = crate::MprKernel::alloc_out(&device, &queue, cap)?;
        let mut body_q = molla_storage::Buffer::from_vec(
            vec![molla_math::Transform::IDENTITY; 1],
            BufferUsage::Storage,
        );
        body_q.ensure_device(&device)?;
        body_q.upload(&queue)?;
        self.mpr_gpu_state = Some(MprGpuState {
            device,
            queue,
            kernel,
            body_q,
            pair_a,
            pair_b,
            mpr_out,
            capacity: cap,
            tolerance,
        });
        Ok(())
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
        engine: &molla_compute::Engine,
        initial_capacity: usize,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| molla_core::Error::Gpu("no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| molla_core::Error::Gpu("no GPU queue".into()))?
            .clone();
        let kernel = crate::GjkKernel::new(&device);
        let cap = initial_capacity.max(1);
        let (mut pair_a, mut pair_b, mut overlap) =
            crate::GjkKernel::alloc_io(&device, &queue, cap)?;
        let _ = (&mut pair_a, &mut pair_b, &mut overlap);
        // body_q buffer — sized by the caller's first dispatch.
        let mut body_q = molla_storage::Buffer::from_vec(
            vec![molla_math::Transform::IDENTITY; 1],
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
        engine: &molla_compute::Engine,
        initial_capacity: usize,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| molla_core::Error::Gpu("no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| molla_core::Error::Gpu("no GPU queue".into()))?
            .clone();
        let gjk = crate::GjkKernel::new(&device);
        let epa = crate::EpaKernel::new(&device);
        let cap = initial_capacity.max(1);
        let (pair_a, pair_b, overlap) = crate::GjkKernel::alloc_io(&device, &queue, cap)?;
        let simplex = crate::GjkKernel::alloc_simplex(&device, &queue, cap)?;
        let epa_out = crate::EpaKernel::alloc_out(&device, &queue, cap)?;
        let mut body_q = molla_storage::Buffer::from_vec(
            vec![molla_math::Transform::IDENTITY; 1],
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
        engine: &molla_compute::Engine,
        initial_prim_capacity: usize,
        initial_pair_capacity: usize,
        refit_only: bool,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| molla_core::Error::Gpu("no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| molla_core::Error::Gpu("no GPU queue".into()))?
            .clone();
        let lbvh = crate::LbvhBroadPhase::new(
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
        self.collide_filtered(model, state, contacts, |_, _| false)
    }

    /// Replace supplemental dense shape exclusions; rebuild these after topology edits.
    pub fn set_extra_shape_exclusions(
        &mut self,
        pairs: impl IntoIterator<Item = (molla_core::ShapeId, molla_core::ShapeId)>,
    ) {
        self.extra_shape_exclusions = pairs
            .into_iter()
            .map(|(a, b)| (a.0.min(b.0), a.0.max(b.0)))
            .collect();
    }

    /// Drop all shape pairs whose body pair is excluded; `None` denotes the world.
    pub fn collide_filtered(
        &mut self,
        model: &Model,
        state: &State,
        contacts: &mut Contacts,
        excluded: impl FnMut(Option<molla_core::BodyId>, Option<molla_core::BodyId>) -> bool,
    ) -> Result<usize> {
        self.collide_filtered_inner(model, state, contacts, excluded, None)
    }

    /// Refresh contacts using bounds prepared for this immutably borrowed model.
    pub fn collide_prepared(
        &mut self,
        bounds: &crate::shape_world::PreparedShapeBounds<'_>,
        state: &State,
        contacts: &mut Contacts,
        excluded: impl FnMut(Option<molla_core::BodyId>, Option<molla_core::BodyId>) -> bool,
    ) -> Result<usize> {
        self.collide_filtered_inner(bounds.model(), state, contacts, excluded, Some(bounds))
    }

    fn collide_filtered_inner(
        &mut self,
        model: &Model,
        state: &State,
        contacts: &mut Contacts,
        mut excluded: impl FnMut(Option<molla_core::BodyId>, Option<molla_core::BodyId>) -> bool,
        bounds: Option<&crate::shape_world::PreparedShapeBounds<'_>>,
    ) -> Result<usize> {
        contacts.clear();
        self.last_cpu_np_calls = 0;
        self.last_broad_phase_pairs = 0;
        self.mesh_cache.begin_query(model.meshes.len(), bounds.and_then(|bounds| bounds.stamp()));
        if model.shape_count < 2 {
            // No pairs possible; still write an empty body-contact
            // CSR so `body_contact_offsets` / `body_contact_indices`
            // get sized for `body_count`. Solvers + sensors bind
            // these unconditionally and would otherwise hit "no
            // device mirror" on first dispatch.
            push_contacts_into(contacts, &[], model)?;
            return Ok(0);
        }

        let broad_span = tracing::debug_span!("collision_broad_phase").entered();
        // Pick GPU LBVH broad phase if enabled, otherwise CPU SAP.
        #[cfg(feature = "gpu")]
        let used_gpu_broad = self.run_gpu_broad_phase(model, state)?;
        #[cfg(not(feature = "gpu"))]
        let used_gpu_broad = false;
        if !used_gpu_broad {
            if let Some(bounds) = bounds {
                self.broad.sap_pairs_prepared(
                    bounds,
                    state,
                    model.contact_margin,
                    &mut self.pair_buf,
                );
            } else {
                self.broad
                    .sap_pairs(model, state, model.contact_margin, &mut self.pair_buf);
            }
        }
        drop(broad_span);
        let filter_span = tracing::debug_span!("collision_pair_filter").entered();
        let bodies = model.shape_body.host()?;
        let body_id = |s: u32| {
            let b = bodies[s as usize];
            (b >= 0).then_some(molla_core::BodyId(b as u32))
        };
        let flags = model.shape_flags.host()?;
        self.pair_buf.retain(|p| {
            (self.include_sensor_overlaps
                || (flags[p.a as usize] | flags[p.b as usize]) & molla_sim::ShapeFlags::SENSOR == 0)
                && !excluded(body_id(p.a), body_id(p.b))
                && !self
                    .extra_shape_exclusions
                    .contains(&(p.a.min(p.b), p.a.max(p.b)))
        });
        self.last_broad_phase_pairs = self.pair_buf.len();
        drop(filter_span);
        for pair in &self.pair_buf {
            model.combined_shape_friction_coefficients(pair.a as usize, pair.b as usize)?;
        }

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

        // GPU MPR backend: batch all MPR-eligible pairs through the
        // kernel in a single dispatch. Ineligible pairs (and FALLBACK
        // status from the kernel) drop to the CPU GJK + EPA path.
        #[cfg(feature = "gpu")]
        if self.narrow_phase_backend == NarrowPhaseBackend::MprGpu {
            self.run_mpr_gpu_backend(model, state, prefilter_mask.as_ref())?;
            push_contacts_into(contacts, &self.manifolds, model)?;
            return Ok(self.manifolds.len());
        }

        // Pairs that survive the optional prefilter mask and the
        // heightfield reach test, then their meshes, then the pairs
        // themselves across the pool, gathered back in order.
        let mut pairs = std::mem::take(&mut self.narrow_pairs);
        pairs.clear();
        for (i, &pair) in self.pair_buf.iter().enumerate() {
            if let Some(mask) = &prefilter_mask
                && mask.get(i).copied().unwrap_or(2) == 0
            {
                continue;
            }
            if !used_gpu_broad
                && bounds.is_some_and(|bounds| {
                    !bounds.heightfield_may_contact(
                        state,
                        pair.a as usize,
                        self.broad.shape_aabb(pair.b as usize),
                    ) || !bounds.heightfield_may_contact(
                        state,
                        pair.b as usize,
                        self.broad.shape_aabb(pair.a as usize),
                    )
                })
            {
                continue;
            }
            pairs.push(pair);
        }
        prepare_meshes(&mut self.mesh_cache, model, &pairs);
        let (backend, cache) = (self.narrow_phase_backend, &self.mesh_cache);
        let parts = crate::parallel::map_chunks(&pairs, |_, chunk| {
            let mut out = Vec::new();
            let calls = chunk
                .iter()
                .filter(|&&pair| dispatch_pair_cpu(backend, model, state, pair, cache, &mut out))
                .count();
            (out, calls)
        });
        for (out, calls) in parts {
            self.manifolds.extend(out);
            self.last_cpu_np_calls += calls;
        }
        self.narrow_pairs = pairs;

        let _span = tracing::debug_span!("collision_contact_output").entered();
        push_contacts_into(contacts, &self.manifolds, model)?;
        Ok(self.manifolds.len())
    }

    /// Run the GPU MPR backend: batch all MPR-eligible pairs through
    /// one kernel dispatch, then translate per-pair results back into
    /// `ContactManifold`s. Pairs flagged FALLBACK / NONE (or that the
    /// kernel can't handle) drop to CPU.
    ///
    /// Requires [`Self::enable_mpr_gpu_backend`] to have been called.
    /// When the GPU state isn't initialised this transparently falls
    /// back to the CPU MPR backend so callers never get a silent
    /// no-op contact list.
    #[cfg(feature = "gpu")]
    fn run_mpr_gpu_backend(
        &mut self,
        model: &Model,
        state: &State,
        prefilter_mask: Option<&Vec<u32>>,
    ) -> Result<()> {
        prepare_meshes(&mut self.mesh_cache, model, &self.pair_buf);
        if self.mpr_gpu_state.is_none() {
            // No GPU state — degrade to CPU MPR semantics. The caller
            // saw the failure via `enable_mpr_gpu_backend` (or didn't
            // call it). Equivalent to `MprCpu` for this frame.
            for (i, &pair) in self.pair_buf.iter().enumerate() {
                if let Some(mask) = prefilter_mask
                    && mask.get(i).copied().unwrap_or(2) == 0
                {
                    continue;
                }
                let mpr = NarrowPhaseBackend::MprCpu;
                if dispatch_pair_cpu(mpr, model, state, pair, &self.mesh_cache, &mut self.manifolds) {
                    self.last_cpu_np_calls += 1;
                }
            }
            return Ok(());
        }

        // Partition pairs into MPR-GPU-eligible and CPU-fallback.
        let mut gpu_pairs: Vec<usize> = Vec::with_capacity(self.pair_buf.len());
        let mut cpu_pairs: Vec<usize> = Vec::with_capacity(self.pair_buf.len());
        for (i, p) in self.pair_buf.iter().enumerate() {
            if let Some(mask) = prefilter_mask
                && mask.get(i).copied().unwrap_or(2) == 0
            {
                continue;
            }
            let ka = model.shape_geometry[p.a as usize];
            let kb = model.shape_geometry[p.b as usize];
            if mpr_gpu_eligible(ka) && mpr_gpu_eligible(kb) {
                gpu_pairs.push(i);
            } else {
                cpu_pairs.push(i);
            }
        }

        // CPU-fallback pairs first — these stay on the original
        // pair_contact path (Plane / TriMesh / Heightfield / SDF /
        // Triangle / ConvexHull / Ellipsoid).
        for &i in &cpu_pairs {
            let pair = self.pair_buf[i];
            pair_contact_cached(
                model,
                state,
                pair,
                model.default_friction,
                model.default_restitution,
                &mut self.manifolds,
                Some(&self.mesh_cache),
            );
            self.last_cpu_np_calls += 1;
        }

        if gpu_pairs.is_empty() {
            return Ok(());
        }

        let n = gpu_pairs.len();
        let state_ref = self.mpr_gpu_state.as_mut().expect("mpr_gpu_state");

        if n > state_ref.capacity {
            let cap = (n * 2).max(64);
            let (pa, pb) = crate::MprKernel::alloc_pairs(&state_ref.device, &state_ref.queue, cap)?;
            state_ref.pair_a = pa;
            state_ref.pair_b = pb;
            state_ref.mpr_out =
                crate::MprKernel::alloc_out(&state_ref.device, &state_ref.queue, cap)?;
            state_ref.capacity = cap;
        }

        {
            let pa_h = state_ref.pair_a.host_mut()?;
            let pb_h = state_ref.pair_b.host_mut()?;
            for (slot, &i) in gpu_pairs.iter().enumerate() {
                let p = self.pair_buf[i];
                pa_h[slot] = p.a;
                pb_h[slot] = p.b;
            }
        }
        state_ref.pair_a.upload(&state_ref.queue)?;
        state_ref.pair_b.upload(&state_ref.queue)?;

        // Mirror state.body_q.
        let body_count = state.body_q.host()?.len();
        if state_ref.body_q.host()?.len() < body_count {
            state_ref.body_q = molla_storage::Buffer::from_vec(
                vec![molla_math::Transform::IDENTITY; body_count.max(1)],
                BufferUsage::Storage,
            );
            state_ref.body_q.ensure_device(&state_ref.device)?;
        }
        {
            let dst = state_ref.body_q.host_mut()?;
            let src = state.body_q.host()?;
            dst[..src.len()].copy_from_slice(src);
        }
        state_ref.body_q.upload(&state_ref.queue)?;

        let body_q_buf = state_ref
            .body_q
            .device_buffer()
            .ok_or_else(|| molla_core::Error::Gpu("body_q device_buffer".into()))?;

        state_ref.kernel.dispatch(
            &state_ref.device,
            &state_ref.queue,
            model,
            body_q_buf,
            &state_ref.pair_a,
            &state_ref.pair_b,
            &mut state_ref.mpr_out,
            n as u32,
            state_ref.tolerance,
        )?;
        state_ref
            .mpr_out
            .download(&state_ref.device, &state_ref.queue)?;
        let host = state_ref.mpr_out.host()?;
        let results = crate::decode_mpr_out(host, n);

        for (slot, &i) in gpu_pairs.iter().enumerate() {
            let pair = self.pair_buf[i];
            let res = results[slot];

            // Anything but a clean OK goes back to CPU GJK + EPA so we
            // never silently drop a contact the kernel struggled with.
            if res.status != crate::MPR_STATUS_OK
                || !res.normal.length_squared().is_finite()
                || res.normal.length_squared() < 1e-12
                || res.depth <= 0.0
            {
                if res.status == crate::MPR_STATUS_NONE {
                    // Disjoint — no contact.
                    continue;
                }
                // FALLBACK / NaN / non-positive depth: re-run on CPU.
                pair_contact_cached(
                    model,
                    state,
                    pair,
                    model.default_friction,
                    model.default_restitution,
                    &mut self.manifolds,
                    Some(&self.mesh_cache),
                );
                self.last_cpu_np_calls += 1;
                continue;
            }

            let normal = res.normal.normalize();
            let point = (res.point_a + res.point_b) * 0.5;
            let pair_friction = model.combined_shape_friction(pair.a as usize, pair.b as usize)?;
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

            self.manifolds.push(ContactManifold {
                a_shape: pair.a,
                b_shape: pair.b,
                point,
                normal,
                depth: res.depth as Real,
                friction: pair_friction,
                rolling_friction: pair_rolling,
                torsional_friction: pair_torsional,
                restitution: pair_restitution,
            });
        }

        Ok(())
    }

    /// Run the full GPU narrow phase. Splits pairs into:
    ///   * **GPU-eligible** — both shapes are Sphere/Box/Capsule/
    ///     Cylinder/Cone. GJK + EPA on device produces manifolds
    ///     directly. EPA fallbacks (status==2) are re-run on CPU.
    ///   * **CPU-only** — at least one shape is Plane / TriMesh /
    ///     Heightfield / SDF / ConvexHull / Triangle. Routed to the
    ///     existing CPU `pair_contact`.
    #[cfg(feature = "gpu")]
    fn run_gpu_full_narrow_phase(&mut self, model: &Model, state: &State) -> Result<()> {
        prepare_meshes(&mut self.mesh_cache, model, &self.pair_buf);
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
            pair_contact_cached(
                model,
                state,
                pair,
                model.default_friction,
                model.default_restitution,
                &mut self.manifolds,
                Some(&self.mesh_cache),
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
            let (pa, pb, ov) = crate::GjkKernel::alloc_io(&pf.device, &pf.queue, cap)?;
            pf.pair_a = pa;
            pf.pair_b = pb;
            pf.overlap = ov;
            pf.simplex = crate::GjkKernel::alloc_simplex(&pf.device, &pf.queue, cap)?;
            pf.epa_out = crate::EpaKernel::alloc_out(&pf.device, &pf.queue, cap)?;
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
            pf.body_q = molla_storage::Buffer::from_vec(
                vec![molla_math::Transform::IDENTITY; body_count.max(1)],
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
            .ok_or_else(|| molla_core::Error::Gpu("body_q device_buffer".into()))?;

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
        let results = crate::decode_epa_out(epa_host, n);

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
                || res.status == crate::EPA_STATUS_FALLBACK
                || res.status == crate::EPA_STATUS_NONE
            {
                pair_contact_cached(
                    model,
                    state,
                    pair,
                    model.default_friction,
                    model.default_restitution,
                    &mut self.manifolds,
                    Some(&self.mesh_cache),
                );
                self.last_cpu_np_calls += 1;
                continue;
            }
            // Build a `ContactManifold` from the GPU EPA result. The
            // contact-point convention matches CPU EPA: midpoint
            // between point_a and point_b.
            let pair_friction = model.combined_shape_friction(pair.a as usize, pair.b as usize)?;
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
                depth: res.depth as molla_math::Real,
                friction: pair_friction,
                rolling_friction: pair_rolling,
                torsional_friction: pair_torsional,
                restitution: pair_restitution,
            });
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn run_gpu_prefilter(&mut self, model: &Model, state: &State) -> Result<Option<Vec<u32>>> {
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
            let (pa, pb, ov) = crate::GjkKernel::alloc_io(&pf.device, &pf.queue, cap)?;
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
            pf.body_q = molla_storage::Buffer::from_vec(
                vec![molla_math::Transform::IDENTITY; body_count.max(1)],
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
            .ok_or_else(|| molla_core::Error::Gpu("body_q device_buffer".into()))?;
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
    ///   `pair_passes_filters` (collision groups, self-collision,
    ///   world-id partitioning) so the downstream narrow phase sees the
    ///   same set the CPU SAP would. Returns `true` if the GPU broad
    ///   phase ran (so the caller can skip CPU SAP).
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
                model, shape_body, body_world, a as usize, b as usize,
            ) {
                continue;
            }
            self.pair_buf.push(ShapePair { a, b });
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
    let flags = model.shape_flags.host()?;
    pair_buf.retain(|p| {
        (flags[p.a as usize] | flags[p.b as usize]) & molla_sim::ShapeFlags::SENSOR == 0
    });

    // Narrow phase + push into Contacts. Each pair can yield 0..N
    // manifolds (boxes can give up to 4).
    let mut manifolds: Vec<ContactManifold> = Vec::with_capacity(pair_buf.len());
    for &pair in pair_buf.iter() {
        model.combined_shape_friction_coefficients(pair.a as usize, pair.b as usize)?;
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
        contacts.a_index =
            molla_storage::Buffer::from_vec(vec![0u32; cap], BufferUsage::StorageReadWrite);
        contacts.b_index =
            molla_storage::Buffer::from_vec(vec![0u32; cap], BufferUsage::StorageReadWrite);
        contacts.a_shape =
            molla_storage::Buffer::from_vec(vec![u32::MAX; cap], BufferUsage::StorageReadWrite);
        contacts.b_shape =
            molla_storage::Buffer::from_vec(vec![u32::MAX; cap], BufferUsage::StorageReadWrite);
        contacts.point = molla_storage::Buffer::from_vec(
            vec![molla_math::Vec3::ZERO; cap],
            BufferUsage::StorageReadWrite,
        );
        contacts.normal = molla_storage::Buffer::from_vec(
            vec![molla_math::Vec3::ZERO; cap],
            BufferUsage::StorageReadWrite,
        );
        contacts.depth =
            molla_storage::Buffer::from_vec(vec![0.0; cap], BufferUsage::StorageReadWrite);
        contacts.friction =
            molla_storage::Buffer::from_vec(vec![0.0; cap], BufferUsage::StorageReadWrite);
        contacts.static_friction = molla_storage::Buffer::from_vec(
            vec![molla_math::Real::NAN; cap],
            BufferUsage::StorageReadWrite,
        );
        contacts.rolling_friction =
            molla_storage::Buffer::from_vec(vec![0.0; cap], BufferUsage::StorageReadWrite);
        contacts.torsional_friction =
            molla_storage::Buffer::from_vec(vec![0.0; cap], BufferUsage::StorageReadWrite);
        contacts.restitution =
            molla_storage::Buffer::from_vec(vec![0.0; cap], BufferUsage::StorageReadWrite);
    }

    let shape_body = model.shape_body.host()?;

    let a_index = contacts.a_index.host_mut()?;
    let b_index = contacts.b_index.host_mut()?;
    let a_shape = contacts.a_shape.host_mut()?;
    let b_shape = contacts.b_shape.host_mut()?;
    for (i, m) in manifolds.iter().enumerate() {
        // Translate shape→body so solvers can apply impulses to the
        // owning body. -1 (static) stays as a sentinel.
        a_index[i] = body_or_static(shape_body[m.a_shape as usize]);
        b_index[i] = body_or_static(shape_body[m.b_shape as usize]);
        a_shape[i] = m.a_shape;
        b_shape[i] = m.b_shape;
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
        fric[i] = model.combined_shape_friction(m.a_shape as usize, m.b_shape as usize)?;
    }
    let stat = contacts.static_friction.host_mut()?;
    for (i, m) in manifolds.iter().enumerate() {
        stat[i] = model
            .combined_shape_friction_coefficients(m.a_shape as usize, m.b_shape as usize)?
            .mu_static;
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
        contacts.body_contact_offsets = Buffer::from_vec(offsets, BufferUsage::StorageReadWrite);
    } else {
        contacts
            .body_contact_offsets
            .host_mut()?
            .copy_from_slice(&offsets);
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
    if idx < 0 { u32::MAX } else { idx as u32 }
}

/// Sentinel value written into `Contacts.a_index` / `b_index` when
/// the shape has no owning body (static-world). Solvers check this
/// to know they should skip updating that side.
pub const STATIC_BODY: u32 = u32::MAX;

#[allow(unused_imports)]
use molla_storage::Buffer; // referenced via fully-qualified paths above

#[cfg(feature = "gpu")]
fn is_gpu_eligible(g: molla_sim::ShapeGeometry) -> bool {
    use molla_sim::ShapeGeometry as G;
    matches!(
        g,
        G::Sphere { .. } | G::Box { .. } | G::Capsule { .. } | G::Cylinder { .. } | G::Cone { .. }
    )
}

/// Shapes the CPU MPR portal-loop can handle via `ShapeSupport` /
/// `support_with_hulls`. Plane / TriMesh / Heightfield / SDF have no
/// usable support function for portal refinement and must fall back to
/// the existing CPU GJK + EPA path (which has bespoke handling for
/// each). Triangle is technically supported by `support_with_hulls`
/// but degenerate in volume — keep it on the CPU mesh-tri pipeline so
/// triangle-vs-mesh paths stay consistent.
fn mpr_eligible(g: ShapeGeometry) -> bool {
    matches!(
        g,
        ShapeGeometry::Sphere { .. }
            | ShapeGeometry::Box { .. }
            | ShapeGeometry::Capsule { .. }
            | ShapeGeometry::Cylinder { .. }
            | ShapeGeometry::Cone { .. }
            | ShapeGeometry::Ellipsoid { .. }
            | ShapeGeometry::ConvexHull { .. }
    )
}

/// Subset of [`mpr_eligible`] supported by the GPU MPR kernel. The
/// kernel only ships analytic support maps for the 5 primitives
/// (Sphere / Box / Capsule / Cylinder / Cone) — Ellipsoid and
/// ConvexHull stay on CPU even with the GPU backend active.
#[cfg(feature = "gpu")]
fn mpr_gpu_eligible(g: ShapeGeometry) -> bool {
    matches!(
        g,
        ShapeGeometry::Sphere { .. }
            | ShapeGeometry::Box { .. }
            | ShapeGeometry::Capsule { .. }
            | ShapeGeometry::Cylinder { .. }
            | ShapeGeometry::Cone { .. }
    )
}

/// Bring the BVH of each mesh the narrow phase will search for `pairs` up to
/// date for this query; a mesh against a plane is tested without one.
fn prepare_meshes(cache: &mut crate::mesh_cache::MeshBvhCache, model: &Model, pairs: &[ShapePair]) {
    for pair in pairs {
        let mesh = match (model.shape_geometry[pair.a as usize], model.shape_geometry[pair.b as usize]) {
            (ShapeGeometry::TriMesh { .. }, ShapeGeometry::Plane { .. })
            | (ShapeGeometry::Plane { .. }, ShapeGeometry::TriMesh { .. }) => None,
            (ShapeGeometry::TriMesh { mesh_id }, _) | (_, ShapeGeometry::TriMesh { mesh_id }) => Some(mesh_id),
            _ => None,
        };
        if let Some(mesh) = mesh.and_then(|id| model.meshes.get(id as usize).map(|mesh| (id, mesh))) {
            cache.prepare(mesh.0 as usize, mesh.1);
        }
    }
}

/// Dispatch a single pair through the configured CPU narrow-phase
/// backend into `out`. Routes MPR-eligible convex pairs to
/// [`super::mpr_collide`] when [`NarrowPhaseBackend::MprCpu`] is
/// selected; everything else (Gjk backend, MPR-ineligible pairs,
/// MPR disjoint result) goes through the existing CPU
/// [`pair_contact`] dispatch. True when it took the CPU GJK + EPA path.
fn dispatch_pair_cpu(
    backend: NarrowPhaseBackend,
    model: &Model,
    state: &State,
    pair: ShapePair,
    cache: &crate::mesh_cache::MeshBvhCache,
    out: &mut Vec<ContactManifold>,
) -> bool {
    let _span = tracing::debug_span!("collision_pair", a = pair.a, b = pair.b).entered();
    let geom_a = model.shape_geometry[pair.a as usize];
    let geom_b = model.shape_geometry[pair.b as usize];
    if backend == NarrowPhaseBackend::MprCpu && mpr_eligible(geom_a) && mpr_eligible(geom_b) {
        // On overlap, a manifold; on disjoint, nothing (as CPU GJK+EPA).
        out.extend(cpu_mpr_manifold(model, state, pair, geom_a, geom_b));
        return false;
    }
    pair_contact_cached(
        model,
        state,
        pair,
        model.default_friction,
        model.default_restitution,
        out,
        Some(cache),
    );
    true
}

/// Run CPU MPR on a pair and translate the result into a
/// [`ContactManifold`] with the standard pair-material rules applied.
/// Returns `None` on a disjoint pair (no contact emitted). MPR's
/// normal convention is "A → B", same as CPU GJK + EPA, so no flip.
fn cpu_mpr_manifold(
    model: &Model,
    state: &State,
    pair: ShapePair,
    geom_a: ShapeGeometry,
    geom_b: ShapeGeometry,
) -> Option<ContactManifold> {
    let xf_a = shape_world_transform(model, state, pair.a as usize);
    let xf_b = shape_world_transform(model, state, pair.b as usize);
    let sa = super::mpr::ShapeSupport::with_hulls(geom_a, &xf_a, &model.hulls);
    let sb = super::mpr::ShapeSupport::with_hulls(geom_b, &xf_b, &model.hulls);
    let m = super::mpr::mpr_collide(&sa, &sb, 1e-8, 64)?;
    if m.depth <= 0.0 || !m.normal.length_squared().is_finite() {
        return None;
    }
    let normal_len_sq = m.normal.length_squared();
    if normal_len_sq < 1e-12 {
        return None;
    }
    let normal = m.normal / normal_len_sq.sqrt();
    let point = (m.point_a + m.point_b) * 0.5;
    let pair_friction = model
        .combined_shape_friction(pair.a as usize, pair.b as usize)
        .expect("shape friction material");
    let pair_restitution = pair_material(
        model.shape_restitution.host().expect("shape_restitution"),
        pair.a as usize,
        pair.b as usize,
        model.default_restitution,
    );
    let pair_rolling = pair_material(
        model
            .shape_rolling_friction
            .host()
            .expect("shape_rolling_friction"),
        pair.a as usize,
        pair.b as usize,
        0.0,
    );
    let pair_torsional = pair_material(
        model
            .shape_torsional_friction
            .host()
            .expect("shape_torsional_friction"),
        pair.a as usize,
        pair.b as usize,
        0.0,
    );
    Some(ContactManifold {
        a_shape: pair.a,
        b_shape: pair.b,
        point,
        normal,
        depth: m.depth,
        friction: pair_friction,
        rolling_friction: pair_rolling,
        torsional_friction: pair_torsional,
        restitution: pair_restitution,
    })
}

fn pair_material(
    arr: &[molla_math::Real],
    a: usize,
    b: usize,
    default: molla_math::Real,
) -> molla_math::Real {
    // Per-shape material arrays are guaranteed finite by `ModelBuilder`
    // (NaN sentinels are substituted with defaults at finalize). A NaN
    // here would corrupt the pair friction/restitution silently — let
    // debug builds catch it.
    let ma = arr.get(a).copied().unwrap_or(default);
    let mb = arr.get(b).copied().unwrap_or(default);
    debug_assert!(
        ma.is_finite(),
        "pair_material: shape {a} material is not finite ({ma})"
    );
    debug_assert!(
        mb.is_finite(),
        "pair_material: shape {b} material is not finite ({mb})"
    );
    ma.max(mb)
}
