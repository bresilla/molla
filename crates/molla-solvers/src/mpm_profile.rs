use super::*;

#[cfg(test)]
#[path = "mpm_profile_budget_tests.rs"]
mod budget_tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpmProfileStage {
    Initialization,
    GridReset,
    ActiveTiles,
    ParticleOrdering,
    RigidReset,
    ParticleToGrid,
    RigidImprint,
    ContactProjection,
    ContactPreparation,
    ContactSolve,
    ContactPublication,
    PreparationCandidates,
    PreparationReset,
    PreparationGeometry,
    PreparationContacts,
    PreparationFactor,
    PreparationResponse,
    PreparationCleanup,
    PreparationHistory,
    GridUpdate,
    Implicit,
    RigidForce,
    GridToParticle,
    Plasticity,
    GuardedGatherProjection,
    GuardedParticleCommit,
    #[cfg(test)]
    GatherProjection,
}

impl MpmProfileStage {
    pub const PREPARATION_STAGES: [Self; 8] = [
        Self::PreparationCandidates,
        Self::PreparationReset,
        Self::PreparationGeometry,
        Self::PreparationContacts,
        Self::PreparationFactor,
        Self::PreparationResponse,
        Self::PreparationCleanup,
        Self::PreparationHistory,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Initialization => "initialization",
            Self::GridReset => "grid-reset",
            Self::ActiveTiles => "active-tiles",
            Self::ParticleOrdering => "particle-ordering",
            Self::RigidReset => "rigid-reset",
            Self::ParticleToGrid => "p2g",
            Self::RigidImprint => "rigid-imprint",
            Self::ContactProjection => "contact-projection",
            Self::ContactPreparation => "contact-preparation",
            Self::ContactSolve => "contact-solve",
            Self::ContactPublication => "contact-publication",
            Self::PreparationCandidates => "prepare-candidates",
            Self::PreparationReset => "prepare-reset",
            Self::PreparationGeometry => "prepare-geometry",
            Self::PreparationContacts => "prepare-contacts",
            Self::PreparationFactor => "prepare-factor",
            Self::PreparationResponse => "prepare-response",
            Self::PreparationCleanup => "prepare-cleanup",
            Self::PreparationHistory => "prepare-history",
            Self::GridUpdate => "grid-update",
            Self::Implicit => "implicit",
            Self::RigidForce => "rigid-force",
            Self::GridToParticle => "g2p",
            Self::Plasticity => "plasticity",
            Self::GuardedGatherProjection => "guarded-g2p-plasticity",
            Self::GuardedParticleCommit => "guarded-particle-commit",
            #[cfg(test)]
            Self::GatherProjection => "g2p-plasticity",
        }
    }
}

#[derive(Clone, Debug)]
pub struct MpmStageTiming {
    pub stage: MpmProfileStage,
    pub substep: Option<usize>,
    /// GPU timestamp offset from the first captured stage, including intervening queue work.
    pub start_ms: f64,
    pub elapsed_ms: f64,
}

/// GPU stage timings for one submitted batch, not a physical completion or overflow verdict.
#[derive(Clone, Debug)]
pub struct MpmStepProfile {
    pub requested_substeps: usize,
    pub requested_simulated_seconds: f64,
    pub particles: u32,
    pub active_grid: bool,
    pub compact_grid: bool,
    pub native_float_atomics: bool,
    pub particle_tile_ordering: bool,
    pub parallel_particle_sort: bool,
    #[cfg(test)]
    pub fused_gather_projection: bool,
    pub timestamp_period_ns: f32,
    pub readback_bytes: u64,
    pub stages: Vec<MpmStageTiming>,
    /// Nested contact timings, already included in the top-level contact-projection stage.
    pub contact_stages: Vec<MpmStageTiming>,
    pub preparation_details_enabled: bool,
    /// Nested inside contact-preparation, never additive to contact or MPM totals.
    pub preparation_stages: Vec<MpmStageTiming>,
}

fn preparation_profile_valid(profile: &MpmStepProfile) -> bool {
    if !profile.preparation_details_enabled {
        return profile.preparation_stages.is_empty();
    }
    let mut parents = profile
        .contact_stages
        .iter()
        .filter(|s| s.stage == MpmProfileStage::ContactPreparation);
    if profile.preparation_stages.len() % 8 != 0 {
        return false;
    }
    let epsilon = f64::from(profile.timestamp_period_ns) * 1e-6;
    for children in profile.preparation_stages.chunks_exact(8) {
        let Some(parent) = parents.next() else {
            return false;
        };
        for (child, kind) in children.iter().zip(MpmProfileStage::PREPARATION_STAGES) {
            if child.stage != kind
                || child.substep != parent.substep
                || child.substep.is_none()
                || !child.start_ms.is_finite()
                || !child.elapsed_ms.is_finite()
                || child.elapsed_ms < 0.0
                || !(child.start_ms + child.elapsed_ms).is_finite()
                || child.start_ms < parent.start_ms
                || child.start_ms + child.elapsed_ms > parent.start_ms + parent.elapsed_ms + epsilon
            {
                return false;
            }
        }
        if children
            .windows(2)
            .any(|s| s[0].start_ms + s[0].elapsed_ms > s[1].start_ms + epsilon)
        {
            return false;
        }
    }
    parents.next().is_none()
}

fn contact_profile_valid(profile: &MpmStepProfile) -> bool {
    if !preparation_profile_valid(profile) {
        return false;
    }
    if profile.contact_stages.is_empty() {
        return true;
    }
    if profile.contact_stages.len() % 3 != 0 {
        return false;
    }
    let mut parents = profile
        .stages
        .iter()
        .filter(|s| s.stage == MpmProfileStage::ContactProjection);
    let epsilon = f64::from(profile.timestamp_period_ns) * 1e-6;
    for children in profile.contact_stages.chunks_exact(3) {
        let Some(parent) = parents.next() else {
            return false;
        };
        for (child, kind) in children.iter().zip([
            MpmProfileStage::ContactPreparation,
            MpmProfileStage::ContactSolve,
            MpmProfileStage::ContactPublication,
        ]) {
            if child.stage != kind
                || child.substep != parent.substep
                || child.substep.is_none()
                || !child.start_ms.is_finite()
                || !child.elapsed_ms.is_finite()
                || child.elapsed_ms < 0.0
                || child.start_ms < parent.start_ms
                || child.start_ms + child.elapsed_ms > parent.start_ms + parent.elapsed_ms + epsilon
            {
                return false;
            }
        }
        if children
            .windows(2)
            .any(|s| s[0].start_ms + s[0].elapsed_ms > s[1].start_ms + epsilon)
        {
            return false;
        }
    }
    parents.next().is_none()
}

pub(super) struct StepProfiler {
    queries: Vec<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
    capacity: usize,
    entries: Vec<(MpmProfileStage, Option<usize>)>,
    pending: bool,
    context: Option<MpmStepProfile>,
    window: Option<ProfileWindow>,
    preparation_details: bool,
}

struct ProfileWindow {
    remaining: usize,
    offset: usize,
    dt_bits: u32,
    epoch: Arc<()>,
}

fn preparation_query_count(capacity: usize) -> Result<usize> {
    capacity
        .checked_mul(40)
        .and_then(|n| n.checked_add(2))
        .filter(|&n| capacity > 0 && n <= wgpu::QUERY_SET_MAX_QUERIES as usize)
        .ok_or_else(|| {
            Error::Solver(
                "detailed preparation profiling capacity must be 1 through 102 substeps".into(),
            )
        })
}

fn profile_query_count(capacity: usize, preparation_details: bool) -> Result<usize> {
    if preparation_details {
        return preparation_query_count(capacity);
    }
    if capacity == 0 || capacity > (wgpu::QUERY_SET_MAX_QUERIES as usize - 2) / 20 {
        return Err(Error::Solver(
            "MPM profiling capacity must be 1 through 204 substeps".into(),
        ));
    }
    Ok(capacity * 24 + 2)
}

#[test]
fn preparation_profile_query_budget() {
    assert_eq!(preparation_query_count(1).unwrap(), 42);
    assert_eq!(preparation_query_count(34).unwrap(), 1362);
    assert_eq!(preparation_query_count(102).unwrap(), 4082);
    for n in [0, 103, usize::MAX] {
        assert!(preparation_query_count(n).is_err());
    }
}

impl StepProfiler {
    fn new(device: &wgpu::Device, capacity: usize) -> Result<Self> {
        Self::new_with_preparation(device, capacity, false)
    }
    fn new_with_preparation(
        device: &wgpu::Device,
        capacity: usize,
        preparation_details: bool,
    ) -> Result<Self> {
        let queries = profile_query_count(capacity, preparation_details)?;
        let features = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        if !device.features().contains(features) {
            return Err(Error::Gpu(
                "MPM profiling requires timestamp queries inside encoders and passes".into(),
            ));
        }
        let query_sets = (0..queries as u32)
            .step_by(wgpu::QUERY_SET_MAX_QUERIES as usize)
            .map(|offset| {
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("MPM stage profile"),
                    ty: wgpu::QueryType::Timestamp,
                    count: (queries as u32 - offset).min(wgpu::QUERY_SET_MAX_QUERIES),
                })
            })
            .collect();
        let allocate = |label, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: queries as u64 * 8,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        Ok(Self {
            queries: query_sets,
            resolve: allocate(
                "MPM profile resolve",
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            staging: allocate(
                "MPM profile readback",
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            capacity,
            entries: Vec::with_capacity(queries / 2),
            pending: false,
            context: None,
            window: None,
            preparation_details,
        })
    }

    pub fn begin(
        &mut self,
        substeps: usize,
        dt: f32,
        particles: u32,
        active_grid: bool,
        compact_grid: bool,
        native_float_atomics: bool,
        particle_tile_ordering: bool,
        parallel_particle_sort: bool,
        #[cfg(test)] fused_gather_projection: bool,
        period: f32,
    ) -> Result<()> {
        if self.pending || self.window.is_some() || substeps == 0 || substeps > self.capacity {
            return Err(Error::Solver(
                "MPM profile unread or batch exceeds prepared capacity".into(),
            ));
        }
        if !dt.is_finite() || dt <= 0. || !period.is_finite() || period <= 0. {
            return Err(Error::Solver(
                "MPM profile timestep or timestamp period is invalid".into(),
            ));
        }
        self.entries.clear();
        self.context = Some(MpmStepProfile {
            requested_substeps: substeps,
            requested_simulated_seconds: f64::from(dt) * substeps as f64,
            particles,
            active_grid,
            compact_grid,
            native_float_atomics,
            particle_tile_ordering,
            parallel_particle_sort,
            #[cfg(test)]
            fused_gather_projection,
            timestamp_period_ns: period,
            readback_bytes: 0,
            stages: Vec::new(),
            contact_stages: Vec::new(),
            preparation_details_enabled: self.preparation_details,
            preparation_stages: Vec::new(),
        });
        Ok(())
    }

    fn start(&mut self, stage: MpmProfileStage, substep: Option<usize>) -> u32 {
        let index = self.entries.len() as u32 * 2;
        let substep = substep.map(|step| step + self.window.as_ref().map_or(0, |w| w.offset));
        self.entries.push((stage, substep));
        index
    }

    fn end(&self) -> u32 {
        self.entries.len() as u32 * 2 - 1
    }

    fn query(&self, index: u32) -> (&wgpu::QuerySet, u32) {
        (
            &self.queries[(index / wgpu::QUERY_SET_MAX_QUERIES) as usize],
            index % wgpu::QUERY_SET_MAX_QUERIES,
        )
    }

    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let count = self.entries.len() as u32 * 2;
        for offset in (0..count).step_by(wgpu::QUERY_SET_MAX_QUERIES as usize) {
            let (queries, _) = self.query(offset);
            let end = (count - offset).min(wgpu::QUERY_SET_MAX_QUERIES);
            encoder.resolve_query_set(queries, 0..end, &self.resolve, u64::from(offset) * 8);
        }
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.staging, 0, u64::from(count) * 8);
        self.pending = true;
    }

    pub fn finish_batch(&mut self, encoder: &mut wgpu::CommandEncoder, substeps: usize) {
        if let Some(window) = &mut self.window {
            window.remaining -= substeps;
            window.offset += substeps;
            if window.remaining != 0 {
                return;
            }
            self.window = None;
        }
        self.resolve(encoder);
    }

    fn read(&mut self, device: &wgpu::Device) -> Result<Option<MpmStepProfile>> {
        if !self.pending {
            return Ok(None);
        }
        let bytes = self.entries.len() as u64 * 16;
        let slice = self.staging.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let result = (|| {
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(5)),
                })
                .map_err(|e| Error::Gpu(e.to_string()))?;
            rx.recv_timeout(std::time::Duration::from_secs(5))
                .map_err(|e| Error::Gpu(e.to_string()))?
                .map_err(|e| Error::Gpu(e.to_string()))?;
            let data = molla_core::gpu_readback!(slice);
            let ticks: &[u64] = bytemuck::cast_slice(&data);
            let mut profile = self.context.as_ref().unwrap().clone();
            profile.readback_bytes = bytes;
            let origin = ticks[0];
            for ((stage, substep), pair) in self.entries.iter().zip(ticks.chunks_exact(2)) {
                let offset = pair[0]
                    .checked_sub(origin)
                    .ok_or_else(|| Error::Gpu("MPM timestamp precedes capture origin".into()))?;
                let delta = pair[1]
                    .checked_sub(pair[0])
                    .ok_or_else(|| Error::Gpu("MPM timestamp end precedes start".into()))?;
                let timing = MpmStageTiming {
                    stage: *stage,
                    substep: *substep,
                    start_ms: offset as f64 * f64::from(profile.timestamp_period_ns) * 1e-6,
                    elapsed_ms: delta as f64 * f64::from(profile.timestamp_period_ns) * 1e-6,
                };
                if MpmProfileStage::PREPARATION_STAGES.contains(stage) {
                    profile.preparation_stages.push(timing);
                } else if matches!(
                    stage,
                    MpmProfileStage::ContactPreparation
                        | MpmProfileStage::ContactSolve
                        | MpmProfileStage::ContactPublication
                ) {
                    profile.contact_stages.push(timing);
                } else {
                    profile.stages.push(timing);
                }
            }
            if !contact_profile_valid(&profile) {
                return Err(Error::Gpu(
                    "MPM contact timestamp nesting is invalid".into(),
                ));
            }
            Ok(Some(profile))
        })();
        self.staging.unmap();
        self.pending = false;
        self.context = None;
        result
    }
}

#[cfg(test)]
#[test]
fn newton_contact_profile_contract() {
    let timing = |stage, start_ms, elapsed_ms| MpmStageTiming {
        stage,
        substep: Some(0),
        start_ms,
        elapsed_ms,
    };
    let mut profile = MpmStepProfile {
        requested_substeps: 1,
        requested_simulated_seconds: 0.001,
        particles: 1,
        active_grid: false,
        compact_grid: false,
        native_float_atomics: true,
        particle_tile_ordering: false,
        parallel_particle_sort: false,
        fused_gather_projection: false,
        timestamp_period_ns: 1.0,
        readback_bytes: 64,
        stages: vec![timing(MpmProfileStage::ContactProjection, 1.0, 6.0)],
        contact_stages: vec![
            timing(MpmProfileStage::ContactPreparation, 1.1, 1.0),
            timing(MpmProfileStage::ContactSolve, 2.2, 3.0),
            timing(MpmProfileStage::ContactPublication, 5.3, 1.0),
        ],
        preparation_details_enabled: false,
        preparation_stages: Vec::new(),
    };
    assert!(contact_profile_valid(&profile));
    let control = profile.clone();
    profile.preparation_details_enabled = true;
    profile.preparation_stages = MpmProfileStage::PREPARATION_STAGES
        .into_iter()
        .enumerate()
        .map(|(i, stage)| timing(stage, 1.1 + 0.1 * i as f64, 0.05))
        .collect();
    assert!(contact_profile_valid(&profile));
    let detailed = profile.clone();
    for change in 0..9 {
        profile = detailed.clone();
        match change {
            0 => {
                profile.preparation_stages.pop();
            }
            1 => profile.preparation_details_enabled = false,
            2 => profile.preparation_stages[0].stage = MpmProfileStage::ContactSolve,
            3 => profile.preparation_stages[0].substep = Some(1),
            4 => profile.preparation_stages[1].start_ms = 1.1,
            5 => profile.preparation_stages[0].start_ms = 1.0,
            6 => profile.preparation_stages[7].elapsed_ms = 2.0,
            7 => profile.preparation_stages[0].elapsed_ms = f64::NAN,
            _ => profile.preparation_stages[0].elapsed_ms = -1.0,
        }
        assert!(!contact_profile_valid(&profile), "detail mutation {change}");
    }
    for change in 0..6 {
        profile = control.clone();
        match change {
            0 => {
                profile.contact_stages.pop();
            }
            1 => profile.contact_stages[1].start_ms = 2.0,
            2 => profile.contact_stages[2].elapsed_ms = 2.0,
            3 => profile.contact_stages[0].substep = Some(1),
            4 => profile.contact_stages[0].stage = MpmProfileStage::ContactSolve,
            _ => profile.contact_stages[1].elapsed_ms = f64::NAN,
        }
        assert!(!contact_profile_valid(&profile));
    }
    profile = control;
    profile.contact_stages.clear();
    assert!(contact_profile_valid(&profile));
}

impl MpmSimulation {
    /// Allocate retained timestamp resources outside stepping; requires explicitly enabled device features.
    pub fn prepare_step_profiling(&mut self, max_substeps: usize) -> Result<()> {
        self.prepare_step_profiling_mode(max_substeps, false)
    }

    /// Retain nested Newton preparation timestamps for up to 102 substeps.
    pub fn prepare_step_profiling_with_preparation(&mut self, max_substeps: usize) -> Result<()> {
        self.prepare_step_profiling_mode(max_substeps, true)
    }

    fn prepare_step_profiling_mode(&mut self, max_substeps: usize, details: bool) -> Result<()> {
        if self
            .profiler
            .as_ref()
            .is_some_and(|p| p.pending || p.window.is_some())
        {
            return Err(Error::Solver(
                "MPM profile must be read before replacing its resources".into(),
            ));
        }
        self.profiler = Some(if details {
            StepProfiler::new_with_preparation(&self.device, max_substeps, true)?
        } else {
            StepProfiler::new(&self.device, max_substeps)?
        });
        Ok(())
    }

    pub(super) fn preparation_profile_enabled(&self) -> bool {
        self.profiler
            .as_ref()
            .is_some_and(|p| p.preparation_details)
    }

    pub(crate) fn begin_frame_profile(
        &mut self,
        queue: &wgpu::Queue,
        dt: f32,
        substeps: usize,
    ) -> bool {
        if self.particle_count == 0 || self.profiler.is_none() {
            return false;
        }
        if self.begin_step_profile(queue, dt, substeps).is_err() {
            return false;
        }
        self.profiler.as_mut().unwrap().window = Some(ProfileWindow {
            remaining: substeps,
            offset: 0,
            dt_bits: dt.to_bits(),
            epoch: self.step_epoch.clone(),
        });
        true
    }

    pub(crate) fn discard_step_profile(&mut self) {
        if let Some(profile) = &mut self.profiler {
            profile.window = None;
            profile.pending = false;
            profile.context = None;
            profile.entries.clear();
        }
    }

    pub(crate) fn step_profile_pending(&self) -> bool {
        self.profiler.as_ref().is_some_and(|p| p.pending)
    }

    pub(super) fn profile_window_batch(&mut self, dt: f32, substeps: usize) -> bool {
        let Some(window) = self.profiler.as_ref().and_then(|p| p.window.as_ref()) else {
            return false;
        };
        if window.dt_bits != dt.to_bits()
            || substeps > window.remaining
            || !Arc::ptr_eq(&window.epoch, &self.step_epoch)
        {
            self.discard_step_profile();
            return false;
        }
        true
    }

    pub(super) fn begin_step_profile(
        &mut self,
        queue: &wgpu::Queue,
        dt: f32,
        substeps: usize,
    ) -> Result<()> {
        let parallel = self.use_particle_order && self.particle_tile_ordering_uses_parallel_sort();
        self.profiler
            .as_mut()
            .ok_or_else(|| Error::Solver("MPM profiling resources not prepared".into()))?
            .begin(
                substeps,
                dt,
                self.particle_count,
                self.use_active_grid,
                self.sparse_grid,
                self.native_float_atomics,
                self.use_particle_order,
                parallel,
                #[cfg(test)]
                self.use_fused_gather,
                queue.get_timestamp_period(),
            )
    }

    /// Submit a profiled batch without mapping or waiting; an unread capture rejects another profiled batch.
    pub fn step_n_profiled_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
        substeps: usize,
    ) -> Result<()> {
        self.step_n_impl(device, queue, dt, substeps, true)
    }

    /// Block for the pending timestamp snapshot; ordinary unprofiled stepping does not overwrite it.
    pub fn read_step_profile(&mut self) -> Result<Option<MpmStepProfile>> {
        match self.profiler.as_mut() {
            Some(profile) => profile.read(&self.device),
            None => Ok(None),
        }
    }

    pub(super) fn profile_begin_pass(
        &mut self,
        enabled: bool,
        pass: &mut wgpu::ComputePass<'_>,
        stage: MpmProfileStage,
        step: usize,
    ) {
        if enabled {
            let profile = self.profiler.as_mut().unwrap();
            let index = profile.start(stage, Some(step));
            let (queries, index) = profile.query(index);
            pass.write_timestamp(queries, index);
        }
    }

    pub(super) fn profile_end_pass(&self, enabled: bool, pass: &mut wgpu::ComputePass<'_>) {
        if enabled {
            let profile = self.profiler.as_ref().unwrap();
            let (queries, index) = profile.query(profile.end());
            pass.write_timestamp(queries, index);
        }
    }

    pub(super) fn profile_begin_encoder(
        &mut self,
        enabled: bool,
        encoder: &mut wgpu::CommandEncoder,
        stage: MpmProfileStage,
        step: Option<usize>,
    ) {
        let _ = self.profile_begin_scope(enabled, encoder, stage, step);
    }

    pub(super) fn profile_begin_scope(
        &mut self,
        enabled: bool,
        encoder: &mut wgpu::CommandEncoder,
        stage: MpmProfileStage,
        step: Option<usize>,
    ) -> Option<u32> {
        if enabled {
            let profile = self.profiler.as_mut().unwrap();
            let index = profile.start(stage, step);
            let (queries, local) = profile.query(index);
            encoder.write_timestamp(queries, local);
            Some(index + 1)
        } else {
            None
        }
    }

    pub(super) fn profile_end_scope(&self, encoder: &mut wgpu::CommandEncoder, end: Option<u32>) {
        if let Some(index) = end {
            let (queries, index) = self.profiler.as_ref().unwrap().query(index);
            encoder.write_timestamp(queries, index);
        }
    }

    pub(super) fn profile_end_encoder(&self, enabled: bool, encoder: &mut wgpu::CommandEncoder) {
        if enabled {
            let profile = self.profiler.as_ref().unwrap();
            let (queries, index) = profile.query(profile.end());
            encoder.write_timestamp(queries, index);
        }
    }
}
