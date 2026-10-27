//! GPU soil ownership with bounded selected disk residency.

use super::*;
use molla_vehicle::{SoilParticleId, SoilResidencyPlan, SoilTileLayout};
use std::sync::Mutex;

#[path = "mpm_soil_disk_storage.rs"]
mod storage;
use storage::{Read, ReadTask, SelectionAdmission, Store, WriteTask};

#[path = "mpm_soil_disk_recenter.rs"]
mod recentering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpmSoilDiskWorldStatus {
    Ready,
    Capturing,
    Writing,
    Loading,
    Faulted,
}

enum Pending {
    Recenter(recentering::Recenter),
    Capture {
        regions: Vec<(SoilRegionId, Vec<SoilTileCoord>)>,
        ticket: MpmSoilCheckpointTicket,
        checkpoint: Option<MpmSoilCheckpoint>,
    },
    Write {
        task: Mutex<WriteTask>,
        plan: SoilResidencyPlan,
        revision: u64,
        activity: Option<MpmSoilActivityReport>,
    },
    Load {
        task: Mutex<ReadTask>,
        plan: SoilResidencyPlan,
        revision: u64,
        activity: Option<MpmSoilActivityReport>,
    },
}

/// Owns an empty-initialized GPU solver and an exclusive committed tile store.
/// Only selected particle payloads and bounded manifest/cache metadata are retained.
/// Every active particle is pinned, not classified asleep.
/// Caller requests must include sufficient interaction halos before stepping.
/// Region transitions freeze stepping through capture, durable write and load.
/// Storage and GPU commits are separate: a post-publication failure faults this
/// owner rather than claiming rollback. Dropping does not flush active physics or
/// cancel a pending write. Rigid attachment and physical sleep are not provided.
/// Previously retained raw GPU handles must not mutate the owned solver state.
pub struct MpmSoilDiskWorld {
    health: Arc<molla_compute::DeviceHealth>,
    simulation: MpmSimulation,
    store: Option<Store>,
    read: Option<Read>,
    captured: Option<MpmSoilCheckpoint>,
    layout: SoilTileLayout,
    world_origin: Option<[f64; 3]>,
    total: u32,
    residency: SoilTileResidency,
    readback: MpmSoilCheckpointReadback,
    gpu_pins: Option<pins::SoilPins>,
    transfer: Option<transfer::SoilTransfer>,
    io_limits: MpmSoilStoreReadLimits,
    pending: Option<Pending>,
    failure: Option<String>,
    next_revision: u64,
    committed_revision: Option<u64>,
    horizon: regions::Horizon,
    pub(super) activity_clock: ActivityClock,
    activity_enabled: bool,
    last_activity: Option<MpmSoilActivityReport>,
    last_transfer: MpmSoilTransferStats,
    last_write: Option<(u64, MpmSoilStoreStats)>,
    committed_read: MpmSoilStoreReadStats,
}

impl MpmSoilDiskWorld {
    /// Installs committed material metadata without loading field particle blobs.
    /// Requires an empty detached solver and a store with a committed manifest.
    pub fn new(
        engine: &Engine,
        simulation: MpmSimulation,
        store: MpmSoilStore,
        limits: SoilResidencyLimits,
        io_limits: MpmSoilStoreReadLimits,
        max_staging_bytes: u64,
    ) -> Result<Self> {
        Self::new_in_frame(
            engine,
            simulation,
            Store::Raw(store),
            limits,
            io_limits,
            max_staging_bytes,
            None,
        )
    }

    /// Owns a framed committed store with a fixed shared world offset. Solver
    /// positions/grid and swept-envelope bounds stay local; residency keys and
    /// GPU pin domains are world tiles. This does not rebase machines or the grid.
    pub fn new_framed(
        engine: &Engine,
        simulation: MpmSimulation,
        store: MpmSoilFrameStore,
        limits: SoilResidencyLimits,
        io_limits: MpmSoilStoreReadLimits,
        max_staging_bytes: u64,
    ) -> Result<Self> {
        let (store, origin) = store.into_world_parts()?;
        Self::new_in_frame(
            engine,
            simulation,
            Store::Raw(store),
            limits,
            io_limits,
            max_staging_bytes,
            Some(origin),
        )
    }

    /// Owns canonical storage in an explicit solver frame. Only committed
    /// metadata is read during construction. Background region loads assemble
    /// selected tiles in this frame; unrelated canonical tiles are not rebased.
    /// This does not move physical grid boundaries, machines or renderers.
    pub fn new_canonical(
        engine: &Engine,
        simulation: MpmSimulation,
        store: MpmSoilCanonicalStore,
        world_origin: [f64; 3],
        limits: SoilResidencyLimits,
        io_limits: MpmSoilStoreReadLimits,
        max_staging_bytes: u64,
    ) -> Result<Self> {
        let grid = simulation.cfg.grid_origin;
        let cells = simulation.grid_window_cells;
        Self::new_in_frame(
            engine,
            simulation,
            Store::Canonical {
                store,
                origin: world_origin,
                grid,
                cells,
            },
            limits,
            io_limits,
            max_staging_bytes,
            Some(world_origin),
        )
    }

    fn new_in_frame(
        engine: &Engine,
        mut simulation: MpmSimulation,
        store: Store,
        limits: SoilResidencyLimits,
        io_limits: MpmSoilStoreReadLimits,
        max_staging_bytes: u64,
        world_origin: Option<[f64; 3]>,
    ) -> Result<Self> {
        if simulation.particle_count() != 0
            || (matches!(&store, Store::Raw(_)) && simulation.grid_window_cells != [0; 3])
        {
            return Err(invalid(
                "disk world requires empty soil; raw storage is origin-only",
            ));
        }
        let layout = store
            .layout()?
            .ok_or_else(|| invalid("disk world requires committed soil"))?;
        let total = store
            .total_particle_count()?
            .ok_or_else(|| invalid("disk world requires committed soil"))?;
        let read = store
            .read_tiles(
                &[],
                MpmSoilStoreReadLimits {
                    tiles: 0,
                    particles: 0,
                    encoded_bytes: 0,
                },
            )?
            .ok_or_else(|| invalid("disk world requires committed soil"))?;
        let residency =
            SoilTileResidency::new(limits).map_err(|error| invalid(&error.to_string()))?;
        simulation.validate_soil_restore(read.selection().checkpoint())?;
        let readback = simulation.prepare_soil_checkpoint_readback(
            engine,
            simulation.cfg.max_particles,
            1,
            max_staging_bytes,
        )?;
        simulation.restore_soil_selection(engine, read.selection())?;
        let committed_read = read.stats();
        Ok(Self {
            health: engine.device_health().unwrap().clone(),
            simulation,
            store: Some(store),
            read: Some(read),
            captured: None,
            layout,
            world_origin,
            total,
            residency,
            readback,
            gpu_pins: None,
            transfer: None,
            io_limits,
            pending: None,
            failure: None,
            next_revision: 0,
            committed_revision: None,
            horizon: regions::Horizon::default(),
            activity_clock: ActivityClock::default(),
            activity_enabled: false,
            last_activity: None,
            last_transfer: MpmSoilTransferStats::default(),
            last_write: None,
            committed_read,
        })
    }

    pub fn status(&self) -> MpmSoilDiskWorldStatus {
        if self.failure.is_some() || self.health.check().is_err() {
            return MpmSoilDiskWorldStatus::Faulted;
        }
        match self.pending {
            None => MpmSoilDiskWorldStatus::Ready,
            Some(Pending::Recenter(_)) => MpmSoilDiskWorldStatus::Capturing,
            Some(Pending::Capture { .. }) => MpmSoilDiskWorldStatus::Capturing,
            Some(Pending::Write { .. }) => MpmSoilDiskWorldStatus::Writing,
            Some(Pending::Load { .. }) => MpmSoilDiskWorldStatus::Loading,
        }
    }
    pub fn active_particle_count(&self) -> u32 {
        self.simulation.particle_count()
    }

    /// Current shared world offset; None denotes the unframed API.
    pub fn world_origin(&self) -> Option<[f64; 3]> {
        self.world_origin
    }

    /// Enable population/tile CPU endpoint measurements over mandatory captures. No
    /// extra GPU copies/readbacks or automatic sleeping. Disabled by default.
    pub fn set_activity_monitoring(&mut self, enabled: bool) -> Result<()> {
        self.ready()?;
        self.activity_enabled = enabled;
        self.last_activity = None;
        Ok(())
    }

    /// Historical pre-transfer population at the last committed boundary; this
    /// does not include particles imported at that boundary or subsequent steps.
    pub fn last_activity_report(&self) -> Option<&MpmSoilActivityReport> {
        self.last_activity.as_ref()
    }
    pub fn total_particle_count(&self) -> u32 {
        self.total
    }
    pub fn staging_bytes(&self) -> u64 {
        self.readback.staging_bytes()
    }
    /// Retained swept G2P proposals and uniforms, separate from capture staging.
    pub fn swept_motion_guard_bytes(&self) -> u64 {
        self.simulation.soil_motion_guard_bytes()
    }
    /// Preallocates transactional swept G2P within a caller-supplied byte budget.
    /// Requires 16 + 128 * max(max_particles, 1) bytes; no particle state changes.
    pub fn prepare_swept_motion_guard(&mut self, max_bytes: u64) -> Result<()> {
        self.ready()?;
        self.simulation
            .prepare_soil_motion_guard_with_budget(max_bytes)
    }
    pub fn readback_stats(&self) -> MpmSoilReadbackStats {
        self.readback.stats()
    }

    /// Enables per-substep exact-rest tracking on an already prepared swept guard.
    /// Additional retained records/readback cost 16 + 16 * max(max_particles, 1) bytes.
    /// Does not classify physical sleep or release particles.
    pub fn prepare_swept_quiescence(&mut self, max_bytes: u64) -> Result<()> {
        self.ready()?;
        self.simulation.prepare_soil_quiescence(max_bytes)
    }
    pub fn swept_quiescence_bytes(&self) -> u64 {
        self.simulation.soil_quiescence_bytes()
    }
    /// Prepares an explicit transfer-support query; requires swept quiescence.
    /// Retains 80 + 4 * grid_size³ + 8 * max(particle_capacity, 1) bytes.
    /// This dense logical-node scratch is separate from sparse solver storage.
    pub fn prepare_swept_support(&mut self, max_bytes: u64) -> Result<()> {
        self.ready()?;
        self.simulation.prepare_soil_support(max_bytes)
    }
    pub fn swept_support_bytes(&self) -> u64 {
        self.simulation.soil_support_bytes()
    }
    pub fn disable_swept_support(&mut self) -> Result<()> {
        self.ready()?;
        self.simulation.disable_soil_support();
        Ok(())
    }
    /// Blocking two-dispatch occupancy query at a current exact-rest boundary.
    /// Includes all 27 transfer nodes; clipped stencils reject the whole query.
    /// No sleep, force equilibrium, support removal or eviction is authorized.
    pub fn read_swept_support(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
    ) -> Result<MpmSoilSupportSnapshot> {
        self.ready()?;
        self.simulation
            .read_soil_support(engine, snapshot, minimum_quiet_steps, max_particles)
    }

    /// Queues a direct support observation without waiting; false means the slot
    /// is occupied. Retains one bounded snapshot and the existing staging buffer.
    /// No sleep or eviction is authorized.
    pub fn try_request_swept_support(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
    ) -> Result<bool> {
        self.ready()?;
        self.simulation.try_request_soil_support(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
        )
    }

    /// Nonblocking poll; each completion is returned once. Validate with
    /// validate_swept_support before using a historical observation as current.
    /// Map/poll faults require disabling and preparing support again.
    /// A different query kind errors without consuming its pending observation.
    pub fn poll_swept_support(
        &mut self,
        engine: &Engine,
    ) -> Result<Option<MpmSoilSupportSnapshot>> {
        self.ready()?;
        self.simulation.poll_soil_support(engine)
    }

    /// Discards any support kind without waiting. Any support poll can drain map
    /// completion before slot reuse; scratch stays allocated.
    pub fn cancel_swept_support(&mut self) -> Result<bool> {
        self.ready()?;
        Ok(self.simulation.cancel_soil_support())
    }
    /// Blocking transitive stencil query with 1..=64 propagation rounds.
    /// Uses prepared support storage and 2 * rounds + 2 dispatches for nonempty soil.
    /// An incomplete fixed point returns an error, never a partial sleep decision.
    pub fn read_swept_support_closure(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
    ) -> Result<MpmSoilSupportClosureSnapshot> {
        self.ready()?;
        self.simulation.read_soil_support_closure(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            rounds,
        )
    }
    pub fn validate_swept_support_closure(
        &self,
        snapshot: &MpmSoilSupportClosureSnapshot,
    ) -> Result<()> {
        self.ready()?;
        self.simulation
            .validate_soil_quiescence(snapshot.quiescence())
    }
    /// Blocking closure seeded by nonquiet particles and explicit current global IDs.
    /// Nonempty requests upload a four-byte mask per live particle into retained scratch.
    /// Unknown/duplicate IDs and incomplete propagation reject; no particles are evicted.
    pub fn read_swept_retained_support(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
        requested: &[molla_vehicle::SoilParticleId],
    ) -> Result<MpmSoilRetainedSupportSnapshot> {
        self.ready()?;
        self.simulation.read_soil_retained_support(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            rounds,
            requested,
        )
    }
    pub fn validate_swept_retained_support(
        &self,
        snapshot: &MpmSoilRetainedSupportSnapshot,
    ) -> Result<()> {
        self.ready()?;
        self.simulation
            .validate_soil_quiescence(snapshot.quiescence())
    }
    /// Prepares 1..=64 region boxes at 32 bytes each, separately from support storage.
    pub fn prepare_swept_support_regions(&mut self, capacity: u32, max_bytes: u64) -> Result<()> {
        self.ready()?;
        self.simulation
            .prepare_soil_support_regions(capacity, max_bytes)
    }
    pub fn swept_support_region_bytes(&self) -> u64 {
        self.simulation.soil_support_region_bytes()
    }
    /// Seeds GPU retention from outward-rounded expanded archive-tile boxes.
    /// Uses current GPU positions; closed boundaries may conservatively overretain.
    /// Does not load missing tiles, publish region claims or authorize deactivation.
    pub fn read_swept_region_support(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
        envelopes: &[MpmSoilRegionEnvelope],
    ) -> Result<MpmSoilRetainedSupportSnapshot> {
        self.ready()?;
        self.simulation.validate_soil_quiescence(snapshot)?;
        let boxes = regions::support_boxes(
            envelopes,
            self.layout,
            &self.simulation.cfg,
            self.residency.limits(),
            self.world_origin,
            self.simulation.soil_support_region_capacity(),
        )?;
        self.simulation.read_soil_region_support(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            rounds,
            &boxes,
        )
    }
    /// Queues a bounded closure without waiting; false means any support query is pending.
    /// Shares direct-support storage, cancellation and map-fault recovery.
    pub fn try_request_swept_support_closure(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
    ) -> Result<bool> {
        self.ready()?;
        self.simulation.try_request_soil_support_closure(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            rounds,
        )
    }

    /// Nonblocking closure poll; another query kind errors without consuming it.
    /// Validate returned provenance before current use; no sleep is authorized.
    pub fn poll_swept_support_closure(
        &mut self,
        engine: &Engine,
    ) -> Result<Option<MpmSoilSupportClosureSnapshot>> {
        self.ready()?;
        self.simulation.poll_soil_support_closure(engine)
    }

    /// Queues stable-ID retention without waiting; IDs are copied at request time.
    /// Shares the bounded support slot and rejects unknown or duplicate IDs.
    pub fn try_request_swept_retained_support(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
        requested: &[molla_vehicle::SoilParticleId],
    ) -> Result<bool> {
        self.ready()?;
        self.simulation.try_request_soil_retained_support(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            rounds,
            requested,
        )
    }

    /// Nonblocking ID-retention poll; another query kind remains pending on error.
    /// Validate returned provenance before current use; no eviction is authorized.
    pub fn poll_swept_retained_support(
        &mut self,
        engine: &Engine,
    ) -> Result<Option<MpmSoilRetainedSupportSnapshot>> {
        self.ready()?;
        self.simulation.poll_soil_retained_support(engine)
    }

    /// Queues GPU region seeds and closure without waiting; shares the support slot.
    /// Uses the blocking query's outward-rounded boxes and current-state validation.
    pub fn try_request_swept_region_support(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
        envelopes: &[MpmSoilRegionEnvelope],
    ) -> Result<bool> {
        self.ready()?;
        self.simulation.validate_soil_quiescence(snapshot)?;
        let boxes = regions::support_boxes(
            envelopes,
            self.layout,
            &self.simulation.cfg,
            self.residency.limits(),
            self.world_origin,
            self.simulation.soil_support_region_capacity(),
        )?;
        self.simulation.try_request_soil_region_support(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            rounds,
            &boxes,
        )
    }

    /// Nonblocking region-retention poll; another query kind remains pending on error.
    /// Validate returned provenance before current use; missing soil is not certified.
    pub fn poll_swept_region_support(
        &mut self,
        engine: &Engine,
    ) -> Result<Option<MpmSoilRetainedSupportSnapshot>> {
        self.ready()?;
        self.simulation.poll_soil_region_support(engine)
    }

    pub fn validate_swept_support(&self, snapshot: &MpmSoilSupportSnapshot) -> Result<()> {
        self.ready()?;
        self.simulation
            .validate_soil_quiescence(snapshot.quiescence())
    }
    /// Opt-in exact-input constitutive memoization; requires swept quiescence.
    /// Retains 112 bytes per capacity slot, separately from observation storage.
    /// All particles still scatter mass/stress; no support is removed.
    pub fn prepare_swept_projection_cache(&mut self, max_bytes: u64) -> Result<()> {
        self.ready()?;
        self.simulation.prepare_soil_projection_cache(max_bytes)
    }
    pub fn swept_projection_cache_bytes(&self) -> u64 {
        self.simulation.soil_projection_cache_bytes()
    }
    pub fn disable_swept_projection_cache(&mut self) -> Result<()> {
        self.ready()?;
        self.simulation.disable_soil_projection_cache();
        Ok(())
    }
    pub fn disable_swept_quiescence(&mut self) -> Result<()> {
        self.ready()?;
        self.simulation.disable_soil_quiescence();
        Ok(())
    }
    /// Blocking bounded observation of guarded substeps; rejects stale/faulted state.
    pub fn read_swept_quiescence(
        &mut self,
        engine: &Engine,
        max_particles: u32,
    ) -> Result<MpmSoilQuiescenceSnapshot> {
        self.ready()?;
        self.simulation.read_soil_quiescence(engine, max_particles)
    }

    /// Queues one bounded exact-rest readback without waiting; false means busy.
    /// Shares staging with blocking reads and captures request-time provenance.
    pub fn try_request_swept_quiescence(
        &mut self,
        engine: &Engine,
        max_particles: u32,
    ) -> Result<bool> {
        self.ready()?;
        self.simulation
            .try_request_soil_quiescence(engine, max_particles)
    }

    /// Nonblocking one-shot poll. Stepping or restoring may make the result historical.
    /// Validate before current use. Map/poll faults require disable/reprepare.
    pub fn poll_swept_quiescence(
        &mut self,
        engine: &Engine,
    ) -> Result<Option<MpmSoilQuiescenceSnapshot>> {
        self.ready()?;
        self.simulation.poll_soil_quiescence(engine)
    }

    /// Discards the observation without waiting; polling drains completion before reuse.
    pub fn cancel_swept_quiescence(&mut self) -> Result<bool> {
        self.ready()?;
        Ok(self.simulation.cancel_soil_quiescence())
    }

    /// Validates immutable observations against this ready owner's current state.
    /// Encoding work (even if cancelled), restore and configuration changes stale
    /// snapshots. No GPU work is issued; this is not sleep/support authorization.
    pub fn validate_swept_quiescence(&self, snapshot: &MpmSoilQuiescenceSnapshot) -> Result<()> {
        self.ready()?;
        self.simulation.validate_soil_quiescence(snapshot)
    }

    /// Enables bounded GPU occupancy pins with optional integer-tile dilation.
    /// Pins supplement swept support bounds; active checkpoint capture remains.
    pub fn prepare_gpu_pins(
        &mut self,
        engine: &Engine,
        config: MpmSoilPinConfig,
        max_bytes: u64,
    ) -> Result<()> {
        self.ready()?;
        self.check_engine(engine)?;
        let pins = pins::SoilPins::new_in_frame(
            engine,
            &self.simulation,
            self.layout,
            config,
            max_bytes,
            self.world_origin,
        )?;
        self.gpu_pins = Some(pins);
        Ok(())
    }

    /// Returns to host occupancy pins over mandatory checkpoint captures.
    pub fn disable_gpu_pins(&mut self) -> Result<()> {
        self.ready()?;
        self.gpu_pins = None;
        Ok(())
    }

    /// Logical pin query storage including its retained readback slot.
    pub fn gpu_pin_buffer_bytes(&self) -> u64 {
        self.gpu_pins.as_ref().map_or(0, |pins| pins.bytes())
    }
    /// Remaining committed swept-envelope substeps; None for raw tile requests.
    pub fn remaining_region_substeps(&self) -> Option<u32> {
        self.horizon.remaining
    }
    /// Last observed successful store write, paired with its request revision.
    /// A later load failure does not undo this durable publication.
    pub fn last_store_write(&self) -> Option<(u64, MpmSoilStoreStats)> {
        self.last_write
    }
    /// Disk read statistics for the last device-published selection.
    /// Initially describes the metadata-only empty selection.
    pub fn committed_read_stats(&self) -> MpmSoilStoreReadStats {
        self.committed_read
    }
    /// Current retained archive metadata; only available while the owner is ready.
    /// Excludes selected/captured payloads, solver/GPU memory and OS/worker overhead.
    pub fn store_storage_stats(&self) -> Result<MpmSoilStoreStorageStats> {
        self.ready()?;
        self.store
            .as_ref()
            .expect("ready store owner")
            .storage_stats()
    }
    /// Retained transfer buffers, excluding solver storage and capture slots.
    pub fn transfer_scratch_bytes(&self) -> u64 {
        self.transfer
            .as_ref()
            .map_or(0, |transfer| transfer.bytes())
    }
    /// Enables bounded experimental device gather/import for changed selections.
    /// Retained particle histories stay on-device; unchanged selections skip work.
    /// This does not eliminate active checkpoint capture or disk payload loading.
    pub fn prepare_device_transfers(
        &mut self,
        engine: &Engine,
        max_scratch_bytes: u64,
    ) -> Result<()> {
        self.ready()?;
        self.check_engine(engine)?;
        if transfer::SoilTransfer::required_bytes(&self.simulation)? > max_scratch_bytes {
            return Err(invalid("disk transfer scratch budget exceeded"));
        }
        if self.transfer.is_none() {
            self.transfer = Some(transfer::SoilTransfer::new(engine, &self.simulation)?);
        }
        Ok(())
    }
    /// Restores full uploads for changed selections and releases transfer scratch.
    pub fn disable_device_transfers(&mut self) -> Result<()> {
        self.ready()?;
        self.transfer = None;
        Ok(())
    }
    pub fn committed_revision(&self) -> Option<u64> {
        self.committed_revision
    }
    pub fn resident_tiles(&self) -> impl ExactSizeIterator<Item = SoilTileCoord> + '_ {
        self.residency.tiles()
    }
    pub fn claims(&self, tile: SoilTileCoord) -> Option<&SoilTileClaims> {
        self.residency.claims(tile)
    }
    pub fn identity(&self) -> &MpmSoilIdentitySnapshot {
        self.simulation
            .soil_identity_snapshot()
            .expect("disk world identity")
    }
    pub fn last_transfer_stats(&self) -> MpmSoilTransferStats {
        self.last_transfer
    }

    /// Retained capture and global IDs after a write completion, including faults.
    /// None while the worker owns the capture; this is not a current solver view.
    /// In framed mode, coordinates are local to world_origin(), not absolute.
    pub fn captured_state(&self) -> Option<(&MpmSoilCheckpoint, &[SoilParticleId])> {
        Some((
            self.captured.as_ref()?,
            self.read.as_ref()?.selection().particle_ids(),
        ))
    }

    pub(super) fn ready(&self) -> Result<()> {
        self.health.check()?;
        if let Some(error) = &self.failure {
            return Err(invalid(error));
        }
        if self.pending.is_some() {
            return Err(invalid("disk transition freezes stepping and updates"));
        }
        Ok(())
    }
    fn check_engine(&self, engine: &Engine) -> Result<()> {
        engine.check_device()?;
        molla_kernels::validate_soil_device(engine.device().unwrap(), &self.simulation.device)
    }
    fn fault(&mut self, error: Error) -> Error {
        self.failure = Some(error.to_string());
        error
    }

    pub fn step_n(&mut self, engine: &Engine, dt: f32, substeps: usize) -> Result<()> {
        self.ready()?;
        self.check_engine(engine)?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(invalid("invalid disk world time step"));
        }
        let remaining = self.horizon.after_steps(substeps)?;
        let activity_clock = self.activity_clock.after_steps(dt, substeps)?;
        self.simulation.step_n(engine, dt, substeps)?;
        self.horizon.remaining = remaining;
        self.activity_clock = activity_clock;
        Ok(())
    }

    /// Expands swept contact bounds using the solver's local stencil and supplied
    /// motion horizon. Captured active particles receive stencil/travel pins even
    /// outside machine bounds. No particle blobs are read before capture. This does
    /// not sleep soil. Stepping enforces the shortest envelope's substep horizon;
    /// actual displacement is conservatively checked on the GPU each substep.
    /// A violation publishes no G2P particle changes in that substep and latches
    /// a fault rejected at capture. Prior substeps and grid scratch are not rolled back.
    /// Lazy preparation retains 16 + 128 * max(max_particles, 1) bytes separately.
    /// Allow rounding slack in bounds; changing subnormal coordinates rejects.
    /// Empty envelopes have a zero
    /// horizon. A committed raw tile update removes the horizon guard.
    pub fn begin_swept_region_update(&mut self, envelopes: &[MpmSoilRegionEnvelope]) -> Result<()> {
        self.ready()?;
        let owned = regions::expand_in_frame(
            envelopes,
            self.layout,
            &self.simulation.cfg,
            self.residency.limits(),
            self.world_origin,
        )?;
        let requests: Vec<_> = owned
            .iter()
            .map(|(id, tiles)| SoilRegionRequest { id: *id, tiles })
            .collect();
        if !envelopes.is_empty() {
            self.simulation.prepare_soil_motion_guard()?;
        }
        self.begin_region_update(&requests)?;
        self.horizon.request_swept(envelopes);
        Ok(())
    }

    pub fn begin_region_update(&mut self, regions: &[SoilRegionRequest<'_>]) -> Result<()> {
        self.ready()?;
        self.residency
            .plan(regions, &[])
            .map_err(|error| invalid(&error.to_string()))?;
        let revision = self.next_revision;
        let next = revision
            .checked_add(1)
            .ok_or_else(|| invalid("disk world revision exhausted"))?;
        let ticket = self
            .readback
            .try_capture(&self.simulation, revision)?
            .ok_or_else(|| invalid("disk capture slot unavailable"))?;
        self.pending = Some(Pending::Capture {
            regions: regions
                .iter()
                .map(|region| (region.id, region.tiles.to_vec()))
                .collect(),
            ticket,
            checkpoint: None,
        });
        self.next_revision = next;
        self.horizon.request(None);
        if let Some(pins) = &mut self.gpu_pins {
            if let Err(error) = pins.submit(&self.simulation, revision) {
                return Err(self.fault(error));
            }
        }
        Ok(())
    }

    fn plan_capture(
        &self,
        regions: &[(SoilRegionId, Vec<SoilTileCoord>)],
        checkpoint: &MpmSoilCheckpoint,
        gpu_tiles: Option<Vec<SoilTileCoord>>,
    ) -> Result<(SoilResidencyPlan, SelectionAdmission)> {
        let pins = if gpu_tiles.is_some() && !self.horizon.has_swept_pins() {
            gpu_tiles.unwrap()
        } else {
            let mut pins: BTreeSet<_> = self
                .horizon
                .particle_pins_in_frame(
                    checkpoint.stream(0),
                    self.layout,
                    &self.simulation.cfg,
                    self.residency.limits().tiles,
                    self.world_origin,
                )?
                .into_iter()
                .collect();
            if let Some(tiles) = gpu_tiles {
                pins.extend(tiles);
            }
            pins.into_iter().collect()
        };
        let requests: Vec<_> = regions
            .iter()
            .map(|(id, tiles)| SoilRegionRequest { id: *id, tiles })
            .collect();
        let plan = self
            .residency
            .plan(&requests, &pins)
            .map_err(|error| invalid(&error.to_string()))?;
        let old: BTreeSet<_> = self.read.as_ref().unwrap().tiles().collect();
        let mut counts = std::collections::BTreeMap::new();
        for tile in plan.tiles() {
            counts.insert(tile, 0);
        }
        for position in checkpoint.stream(0).chunks_exact(4) {
            let point = std::array::from_fn(|axis| f32::from_bits(position[axis]) as f64);
            let tile = match self.world_origin {
                Some(origin) => self.layout.tile_at_local(point, origin),
                None => self.layout.tile_at(point),
            }
            .map_err(|error| invalid(&error.to_string()))?;
            *counts
                .get_mut(&tile)
                .ok_or_else(|| invalid("disk pins omit an active particle tile"))? += 1;
        }
        let admission = SelectionAdmission {
            entries: counts
                .into_iter()
                .map(|(tile, count)| (tile, count, old.contains(&tile)))
                .collect(),
            max_particles: self.simulation.cfg.max_particles,
            limits: self.io_limits,
        };
        Ok((plan, admission))
    }

    /// Advances at most one stage without waiting for GPU capture or disk I/O.
    /// True means region or recenter device publication, not merely durable storage.
    /// Pre-write capacity/validation errors allow retry. Post-write load/device
    /// failures fault the owner; captured inputs remain available when recovered.
    pub fn poll_region_update(&mut self, engine: &Engine) -> Result<bool> {
        self.check_engine(engine)?;
        if let Some(error) = &self.failure {
            return Err(invalid(error));
        }
        let Some(pending) = self.pending.take() else {
            return Ok(false);
        };
        match pending {
            Pending::Recenter(request) => return self.poll_recenter(engine, request),
            Pending::Capture {
                regions,
                ticket,
                mut checkpoint,
            } => {
                if checkpoint.is_none() {
                    let completion = match self.readback.poll_next() {
                        Ok(Some(completion)) => completion,
                        Ok(None) => {
                            self.pending = Some(Pending::Capture {
                                regions,
                                ticket,
                                checkpoint,
                            });
                            return Ok(false);
                        }
                        Err(error) => return Err(self.fault(error)),
                    };
                    if completion.ticket().revision() != ticket.revision()
                        || !self
                            .readback
                            .is_compatible(&self.simulation, completion.ticket())
                        || completion.identity().is_none_or(|identity| {
                            !self.simulation.is_soil_identity_current(identity)
                        })
                    {
                        return Err(self.fault(invalid("disk capture identity mismatch")));
                    }
                    checkpoint = Some(
                        completion
                            .into_result()
                            .map_err(|error| self.fault(error))?,
                    );
                }
                let gpu_tiles = if let Some(pins) = &mut self.gpu_pins {
                    let snapshot = match pins.poll() {
                        Ok(Some(snapshot)) => snapshot,
                        Ok(None) => {
                            self.pending = Some(Pending::Capture {
                                regions,
                                ticket,
                                checkpoint,
                            });
                            return Ok(false);
                        }
                        Err(error) => {
                            return Err(if matches!(&error, Error::Gpu(_)) {
                                self.fault(error)
                            } else {
                                error
                            });
                        }
                    };
                    if snapshot.revision != ticket.revision() {
                        return Err(self.fault(invalid("disk GPU pin revision mismatch")));
                    }
                    Some(snapshot.tiles)
                } else {
                    None
                };
                let checkpoint = checkpoint.unwrap();
                let (plan, admission) = self.plan_capture(&regions, &checkpoint, gpu_tiles)?;
                let activity = self.activity_clock.report_in_frame(
                    self.activity_enabled,
                    self.read.as_ref().unwrap().selection(),
                    &checkpoint,
                    ticket.revision(),
                    self.layout,
                    self.world_origin,
                )?;
                let task = self.store.take().unwrap().begin_writeback(
                    self.read.take().unwrap(),
                    checkpoint,
                    self.io_limits,
                    admission,
                );
                self.pending = Some(Pending::Write {
                    task: Mutex::new(task),
                    plan,
                    revision: ticket.revision(),
                    activity,
                });
            }
            Pending::Write {
                mut task,
                plan,
                revision,
                activity,
            } => {
                let outcome = task.get_mut().expect("exclusive task").poll();
                let completion = match outcome {
                    Ok(Some(completion)) => completion,
                    Ok(None) => {
                        self.pending = Some(Pending::Write {
                            task,
                            plan,
                            revision,
                            activity,
                        });
                        return Ok(false);
                    }
                    Err(error) => return Err(self.fault(error)),
                };
                self.store = Some(completion.store);
                self.read = Some(completion.read);
                self.captured = Some(completion.updated);
                let write_stats = match completion.result {
                    Ok(stats) => stats,
                    Err(error) => {
                        if self
                            .store
                            .as_ref()
                            .unwrap()
                            .validate_read(self.read.as_ref().unwrap())
                            .is_err()
                        {
                            return Err(self.fault(error));
                        }
                        return Err(error);
                    }
                };
                self.last_write = Some((revision, write_stats));
                let keys: Vec<_> = plan.tiles().collect();
                let request_tiles = self.residency.limits().tiles.min(u32::MAX as usize) as u32;
                let task =
                    self.store
                        .take()
                        .unwrap()
                        .begin_read(&keys, request_tiles, self.io_limits);
                self.pending = Some(Pending::Load {
                    task: Mutex::new(task),
                    plan,
                    revision,
                    activity,
                });
            }
            Pending::Load {
                mut task,
                plan,
                revision,
                activity,
            } => {
                let completion = match task.get_mut().expect("exclusive task").poll() {
                    Ok(Some(completion)) => completion,
                    Ok(None) => {
                        self.pending = Some(Pending::Load {
                            task,
                            plan,
                            revision,
                            activity,
                        });
                        return Ok(false);
                    }
                    Err(error) => return Err(self.fault(error)),
                };
                self.store = Some(completion.store);
                let read = completion
                    .result
                    .map_err(|error| self.fault(error))?
                    .ok_or_else(|| self.fault(invalid("committed disk world lost manifest")))?;
                let selected = read.selection();
                if self
                    .identity()
                    .particle_ids()
                    .iter()
                    .any(|id| selected.particle_ids().binary_search(id).is_err())
                {
                    return Err(
                        self.fault(invalid("disk transition would discard pinned particles"))
                    );
                }
                self.store
                    .as_ref()
                    .unwrap()
                    .validate_read(&read)
                    .map_err(|error| self.fault(error))?;
                let transfer = if let Some(transfer) = &self.transfer {
                    transfer.apply(engine, &mut self.simulation, selected)
                } else {
                    transfer::restore_default(engine, &mut self.simulation, selected)
                }
                .map_err(|error| self.fault(error))?;
                self.residency
                    .commit(plan)
                    .expect("exclusive disk residency plan");
                self.committed_read = read.stats();
                self.read = Some(read);
                self.captured = None;
                self.last_transfer = transfer;
                self.committed_revision = Some(revision);
                self.horizon.commit();
                self.simulation.set_soil_motion_bound(self.horizon.travel);
                self.activity_clock = ActivityClock::default();
                self.last_activity = activity;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
#[path = "mpm_soil_disk_world_tests.rs"]
mod tests;
