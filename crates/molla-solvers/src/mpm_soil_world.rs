//! Exclusive archive/solver ownership at asynchronous region-change boundaries.

use super::activity::ActivityClock;
use super::*;
use molla_vehicle::{
    SoilRegionId, SoilRegionRequest, SoilResidencyLimits, SoilTileClaims, SoilTileCoord,
    SoilTileResidency,
};
use std::collections::BTreeSet;

#[path = "mpm_soil_settling.rs"]
mod settling;
pub use settling::{
    MpmSoilSettling, MpmSoilSettlingConfig, MpmSoilSettlingLimits, MpmSoilSettlingProgress,
    MpmSoilSettlingStatus,
};

#[path = "mpm_soil_transfer.rs"]
mod transfer;
pub use transfer::{MpmSoilTransferPath, MpmSoilTransferStats};

#[path = "mpm_soil_pins.rs"]
mod pins;
pub use pins::MpmSoilPinConfig;

#[path = "mpm_soil_regions.rs"]
mod regions;
pub use regions::MpmSoilRegionEnvelope;

#[path = "mpm_soil_disk_world.rs"]
mod disk;
pub use disk::{MpmSoilDiskWorld, MpmSoilDiskWorldStatus};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpmSoilWorldStatus {
    Ready,
    Capturing,
    Faulted,
}

struct RegionUpdate {
    regions: Vec<(SoilRegionId, Vec<SoilTileCoord>)>,
    ticket: MpmSoilCheckpointTicket,
    checkpoint: Option<MpmSoilCheckpoint>,
}

/// Owns one archive and its sole active solver. Region overlaps share physical
/// particles. Every already-active particle remains pinned at its captured current
/// tile; this owner does not infer sleep or evict moving material to meet capacity.
/// Raw tile requests require caller-supplied interaction halos; swept envelopes
/// expand local stencil/motion coverage with a guarded substep horizon. Optional
/// GPU pins expand existing particle tiles, not new requests or future motion.
/// Physical sleeping and rigid coupling are not provided. Explicit synchronized
/// saves support incremental tile storage outside stepping.
/// Captures still read active checkpoints. Unchanged selections skip
/// restoration; changed selections use full upload unless device gather is enabled.
/// Independent archive reloads create independent worlds; there is no global
/// world-ID registry. Dropping the owner does not implicitly flush or save it.
/// Previously retained raw GPU handles must not mutate the owned solver state.
pub struct MpmSoilWorld {
    health: Arc<molla_compute::DeviceHealth>,
    simulation: MpmSimulation,
    archive: MpmSoilTileArchive,
    selection: MpmSoilSelection,
    residency: SoilTileResidency,
    readback: MpmSoilCheckpointReadback,
    transfer: Option<transfer::SoilTransfer>,
    gpu_pins: Option<pins::SoilPins>,
    last_transfer: MpmSoilTransferStats,
    pending: Option<RegionUpdate>,
    failure: Option<String>,
    next_revision: u64,
    committed_revision: Option<u64>,
    synchronized: bool,
    horizon: regions::Horizon,
    activity_clock: ActivityClock,
    activity_enabled: bool,
    last_activity: Option<MpmSoilActivityReport>,
}

impl MpmSoilWorld {
    /// Takes exclusive ownership of an empty, detached solver and an archive.
    /// Allocates one fixed-capacity async slot and installs the archive's material
    /// table/global IDs. Existing solver particles or attached rigids reject.
    pub fn new(
        engine: &Engine,
        mut simulation: MpmSimulation,
        archive: MpmSoilTileArchive,
        limits: SoilResidencyLimits,
        max_staging_bytes: u64,
    ) -> Result<Self> {
        if simulation.particle_count() != 0 || simulation.grid_window_cells != [0; 3] {
            return Err(invalid("world requires an empty origin-only solver"));
        }
        let residency = SoilTileResidency::new(limits).map_err(|e| invalid(&e.to_string()))?;
        let selection = archive.select(&[], simulation.cfg.max_particles)?;
        simulation.validate_soil_restore(selection.checkpoint())?;
        let readback = simulation.prepare_soil_checkpoint_readback(
            engine,
            simulation.cfg.max_particles,
            1,
            max_staging_bytes,
        )?;
        simulation.restore_soil_selection(engine, &selection)?;
        Ok(Self {
            health: engine.device_health().unwrap().clone(),
            simulation,
            archive,
            selection,
            residency,
            readback,
            transfer: None,
            gpu_pins: None,
            last_transfer: MpmSoilTransferStats::default(),
            pending: None,
            failure: None,
            next_revision: 0,
            committed_revision: None,
            synchronized: true,
            horizon: regions::Horizon::default(),
            activity_clock: ActivityClock::default(),
            activity_enabled: false,
            last_activity: None,
        })
    }

    pub fn status(&self) -> MpmSoilWorldStatus {
        if self.failure.is_some() || self.health.check().is_err() {
            MpmSoilWorldStatus::Faulted
        } else if self.pending.is_some() {
            MpmSoilWorldStatus::Capturing
        } else {
            MpmSoilWorldStatus::Ready
        }
    }

    pub fn active_particle_count(&self) -> u32 {
        self.simulation.particle_count()
    }
    pub fn total_particle_count(&self) -> u32 {
        self.archive.particle_count()
    }
    pub fn resident_tiles(&self) -> impl ExactSizeIterator<Item = SoilTileCoord> + '_ {
        self.residency.tiles()
    }
    pub fn claims(&self, tile: SoilTileCoord) -> Option<&SoilTileClaims> {
        self.residency.claims(tile)
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
    /// is occupied. Uses the prepared staging buffer and retains one bounded
    /// quiescence snapshot. No sleep or eviction is authorized.
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

    /// Nonblocking poll; each completed observation is returned once. A returned
    /// observation may be historical: validate_swept_support checks current state.
    /// Map/poll faults require disabling and preparing support again.
    /// A different query kind errors without consuming its pending observation.
    pub fn poll_swept_support(
        &mut self,
        engine: &Engine,
    ) -> Result<Option<MpmSoilSupportSnapshot>> {
        self.ready()?;
        self.simulation.poll_soil_support(engine)
    }

    /// Discards the pending observation without waiting. The slot remains busy
    /// until any support poll drains map completion; scratch stays allocated.
    /// Applies to direct, closure, stable-ID and region requests.
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
            self.archive.layout(),
            &self.simulation.cfg,
            self.residency.limits(),
            None,
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
            self.archive.layout(),
            &self.simulation.cfg,
            self.residency.limits(),
            None,
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
    /// Remaining committed swept-envelope substeps; None for raw tile requests.
    pub fn remaining_region_substeps(&self) -> Option<u32> {
        self.horizon.remaining
    }
    /// Three retained transfer buffers; excludes particles, IDs and readback slots.
    pub fn transfer_scratch_bytes(&self) -> u64 {
        self.transfer
            .as_ref()
            .map_or(0, |transfer| transfer.bytes())
    }
    /// Most recent committed region transfer, excluding initial construction.
    pub fn last_transfer_stats(&self) -> MpmSoilTransferStats {
        self.last_transfer
    }

    /// Enable exact bounded GPU occupancy pins with optional integer-tile dilation.
    /// Existing particles all contribute; this is not physical sleep classification.
    /// New machine requests and future motion still require caller-supplied halos.
    /// Unsupported layouts and insufficient memory fail without changing pin mode.
    pub fn prepare_gpu_pins(
        &mut self,
        engine: &Engine,
        config: MpmSoilPinConfig,
        max_bytes: u64,
    ) -> Result<()> {
        self.ready()?;
        self.check_engine(engine)?;
        let pins = pins::SoilPins::new(
            engine,
            &self.simulation,
            self.archive.layout(),
            config,
            max_bytes,
        )?;
        self.gpu_pins = Some(pins);
        Ok(())
    }

    /// Return to exact host membership over the mandatory archive capture.
    pub fn disable_gpu_pins(&mut self) -> Result<()> {
        self.ready()?;
        self.gpu_pins = None;
        Ok(())
    }

    /// Logical pin query buffers including its single readback slot.
    pub fn gpu_pin_buffer_bytes(&self) -> u64 {
        self.gpu_pins.as_ref().map_or(0, |pins| pins.bytes())
    }

    /// Enable experimental device compaction/import with an explicit logical
    /// scratch budget. No universal speedup is established; default changes use
    /// full uploads. Unchanged selections skip transfers in either mode.
    pub fn prepare_device_transfers(
        &mut self,
        engine: &Engine,
        max_scratch_bytes: u64,
    ) -> Result<()> {
        self.ready()?;
        self.check_engine(engine)?;
        if transfer::SoilTransfer::required_bytes(&self.simulation)? > max_scratch_bytes {
            return Err(invalid("device transfer scratch budget exceeded"));
        }
        if self.transfer.is_none() {
            self.transfer = Some(transfer::SoilTransfer::new(engine, &self.simulation)?);
        }
        Ok(())
    }

    /// Return to full uploads for changed selections and release retained scratch.
    pub fn disable_device_transfers(&mut self) -> Result<()> {
        self.ready()?;
        self.transfer = None;
        Ok(())
    }
    /// Last committed request ordinal, not elapsed/completed physical time.
    /// None denotes the original archive before the first successful update.
    pub fn committed_revision(&self) -> Option<u64> {
        self.committed_revision
    }
    pub fn identity(&self) -> &MpmSoilIdentitySnapshot {
        self.simulation
            .soil_identity_snapshot()
            .expect("world-owned identity")
    }

    fn ready(&self) -> Result<()> {
        self.health.check()?;
        if let Some(error) = &self.failure {
            return Err(invalid(error));
        }
        if self.pending.is_some() {
            return Err(invalid("region capture freezes stepping and updates"));
        }
        Ok(())
    }

    fn check_engine(&self, engine: &Engine) -> Result<()> {
        engine.check_device()?;
        molla_kernels::validate_soil_device(engine.device().unwrap(), &self.simulation.device)
    }

    /// Submit physical steps only while no residency transition is pending.
    /// The owner forwards every requested substep. Compact-grid overflow can stop
    /// solver advancement and is rejected by the next mandatory archive capture.
    pub fn step_n(&mut self, engine: &Engine, dt: f32, substeps: usize) -> Result<()> {
        self.ready()?;
        self.check_engine(engine)?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(invalid("invalid world time step"));
        }
        let remaining = self.horizon.after_steps(substeps)?;
        let activity_clock = self.activity_clock.after_steps(dt, substeps)?;
        self.simulation.step_n(engine, dt, substeps)?;
        self.horizon.remaining = remaining;
        self.activity_clock = activity_clock;
        if substeps != 0 && self.active_particle_count() != 0 {
            self.synchronized = false;
        }
        Ok(())
    }

    /// Expands swept contact bounds with local MPM stencil/motion padding, then
    /// starts the same frozen transition as `begin_region_update`. Captured active
    /// particles also receive stencil/travel pins outside machine bounds. Does not
    /// sleep particles. Stepping enforces the shortest envelope's substep horizon;
    /// actual displacement is conservatively checked on the GPU each substep.
    /// A violation publishes no G2P particle changes in that substep and latches
    /// a fault rejected at capture. Prior substeps and grid scratch are not rolled back.
    /// Lazy preparation retains 16 + 128 * max(max_particles, 1) bytes separately.
    /// Allow rounding slack in bounds; changing subnormal coordinates rejects.
    /// Empty envelopes have a zero
    /// horizon. A committed raw tile update removes the horizon guard.
    pub fn begin_swept_region_update(&mut self, envelopes: &[MpmSoilRegionEnvelope]) -> Result<()> {
        self.ready()?;
        let owned = regions::expand(
            envelopes,
            self.archive.layout(),
            &self.simulation.cfg,
            self.residency.limits(),
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

    /// Begin a mandatory boundary capture and freeze stepping until polling finishes.
    /// Requests replace region claims; absent archive tiles are empty space. Active
    /// particles are pinned from the completed capture, independently of requests.
    pub fn begin_region_update(&mut self, regions: &[SoilRegionRequest<'_>]) -> Result<()> {
        self.ready()?;
        self.residency
            .plan(regions, &[])
            .map_err(|e| invalid(&e.to_string()))?;
        let revision = self.next_revision;
        let next = revision
            .checked_add(1)
            .ok_or_else(|| invalid("world revision exhausted"))?;
        let owned = regions.iter().map(|r| (r.id, r.tiles.to_vec())).collect();
        let ticket = self
            .readback
            .try_capture(&self.simulation, revision)?
            .ok_or_else(|| invalid("world capture slot unavailable"))?;
        self.pending = Some(RegionUpdate {
            regions: owned,
            ticket,
            checkpoint: None,
        });
        self.next_revision = next;
        self.horizon.request(None);
        if let Some(pins) = &mut self.gpu_pins {
            if let Err(error) = pins.submit(&self.simulation, revision) {
                self.failure = Some(error.to_string());
                return Err(error);
            }
        }
        Ok(())
    }

    /// Poll without a GPU wait. True means the region/archive/device transition
    /// committed. Capacity/validation failures leave the old GPU and archive state
    /// intact and permit a new request. Capture/device faults stop this owner.
    /// Ready host snapshot processing still copies and validates active state.
    /// A committed restore is queued, not GPU-completed; later steps use queue
    /// ordering. Device loss is not transactional rollback.
    pub fn poll_region_update(&mut self, engine: &Engine) -> Result<bool> {
        self.check_engine(engine)?;
        if let Some(error) = &self.failure {
            return Err(invalid(error));
        }
        if self.pending.is_none() {
            return Ok(false);
        }
        if self.pending.as_ref().unwrap().checkpoint.is_none() {
            let completion = match self.readback.poll_next() {
                Ok(Some(completion)) => completion,
                Ok(None) => return Ok(false),
                Err(error) => {
                    self.failure = Some(error.to_string());
                    return Err(error);
                }
            };
            let pending = self.pending.as_ref().expect("pending region update");
            if completion.ticket().revision() != pending.ticket.revision()
                || !self
                    .readback
                    .is_compatible(&self.simulation, completion.ticket())
                || completion
                    .identity()
                    .is_none_or(|ids| !self.simulation.is_soil_identity_current(ids))
            {
                self.failure = Some("world capture identity mismatch".into());
                return Err(invalid("world capture identity mismatch"));
            }
            let checkpoint = match completion.into_result() {
                Ok(checkpoint) => checkpoint,
                Err(error) => {
                    self.failure = Some(error.to_string());
                    return Err(error);
                }
            };
            self.pending.as_mut().unwrap().checkpoint = Some(checkpoint);
        }
        let gpu_tiles = if let Some(pins) = &mut self.gpu_pins {
            let snapshot = match pins.poll() {
                Ok(Some(snapshot)) => snapshot,
                Ok(None) => return Ok(false),
                Err(error) => {
                    if matches!(&error, Error::Gpu(_)) {
                        self.failure = Some(error.to_string());
                    }
                    self.pending = None;
                    return Err(error);
                }
            };
            if snapshot.revision != self.pending.as_ref().unwrap().ticket.revision() {
                self.failure = Some("GPU pin revision mismatch".into());
                return Err(invalid("GPU pin revision mismatch"));
            }
            Some(snapshot.tiles)
        } else {
            None
        };
        let pending = self.pending.take().unwrap();
        let checkpoint = pending.checkpoint.unwrap();
        let activity = self.activity_clock.report(
            self.activity_enabled,
            &self.selection,
            &checkpoint,
            pending.ticket.revision(),
            self.archive.layout(),
        )?;
        let pins = if gpu_tiles.is_some() && !self.horizon.has_swept_pins() {
            gpu_tiles.unwrap()
        } else {
            let mut pins: BTreeSet<_> = self
                .horizon
                .particle_pins(
                    checkpoint.stream(0),
                    self.archive.layout(),
                    &self.simulation.cfg,
                    self.residency.limits().tiles,
                )?
                .into_iter()
                .collect();
            if let Some(tiles) = gpu_tiles {
                pins.extend(tiles);
            }
            pins.into_iter().collect()
        };
        let regions: Vec<_> = pending
            .regions
            .iter()
            .map(|(id, tiles)| SoilRegionRequest { id: *id, tiles })
            .collect();
        let plan = self
            .residency
            .plan(&regions, &pins)
            .map_err(|e| invalid(&e.to_string()))?;
        let tiles: Vec<_> = plan.tiles().collect();
        let transaction = self
            .archive
            .prepare_writeback(&self.selection, &checkpoint)?;
        let selection = transaction.select_existing(&tiles, self.simulation.cfg.max_particles)?;
        let transfer = if let Some(transfer) = &self.transfer {
            transfer.apply(engine, &mut self.simulation, &selection)?
        } else {
            transfer::restore_default(engine, &mut self.simulation, &selection)?
        };
        transaction.commit();
        self.residency
            .commit(plan)
            .expect("exclusive residency plan");
        self.selection = selection;
        self.last_transfer = transfer;
        self.committed_revision = Some(pending.ticket.revision());
        self.horizon.commit();
        self.simulation.set_soil_motion_bound(self.horizon.travel);
        self.activity_clock = ActivityClock::default();
        self.last_activity = activity;
        self.synchronized = true;
        Ok(true)
    }

    /// Serialize only a synchronized complete world archive. After stepping, finish
    /// a region update first. This allocates/copies the whole archive on the host;
    /// durable storage and incremental tile writes remain caller-owned.
    pub fn synchronized_archive_bytes(&self) -> Result<Vec<u8>> {
        self.ready()?;
        if !self.synchronized {
            return Err(invalid("world archive is older than active physics"));
        }
        Ok(self.archive.to_bytes())
    }

    /// Persist current synchronized soil using incremental blocking tile I/O.
    /// Unsynchronized physics, pending capture and faulted owners reject before I/O.
    /// Storage failure does not roll back or fault the independent GPU simulation.
    pub fn save_synchronized_archive(&self, store: &mut MpmSoilStore) -> Result<MpmSoilStoreStats> {
        self.ready()?;
        if !self.synchronized {
            return Err(invalid("world archive is older than active physics"));
        }
        store.save(&self.archive)
    }

    /// Export the last committed archive even during capture or after a fault.
    /// Active particle state can be older than the current GPU; use only as an
    /// explicitly historical recovery point, paired with `committed_revision`.
    pub fn last_committed_archive_bytes(&self) -> Vec<u8> {
        self.archive.to_bytes()
    }
}

#[cfg(test)]
#[path = "mpm_soil_world_tests.rs"]
mod tests;
