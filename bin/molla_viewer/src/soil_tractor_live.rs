use molla::compute::Engine;
use molla_solvers::rigid_world::MpmSoilTransferInfo;
use molla_solvers::soil_tractor::{SoilTractor, SoilTractorConfig, SoilTractorTransfer};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::Instant;

#[path = "soil_tractor_profile.rs"]
pub mod profiling;

pub const DT: f64 = 1.0 / 60.0;
pub const TORQUE: [f64; 4] = [300.0; 4];
pub const LOAD_BYTES: u64 = 5 * 48;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Prediction {
    #[default]
    Featherstone,
    Native,
}

impl Prediction {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "featherstone" => Ok(Self::Featherstone),
            "native" => Ok(Self::Native),
            _ => Err(format!(
                "unknown machine prediction {name}; expected featherstone or native"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Featherstone => "featherstone",
            Self::Native => "native",
        }
    }

    fn features(self) -> wgpu::Features {
        let base = wgpu::Features::SHADER_FLOAT32_ATOMIC | wgpu::Features::PASSTHROUGH_SHADERS;
        if self == Self::Native {
            base | wgpu::Features::SHADER_F64
        } else {
            base
        }
    }

    fn check(self, native: bool) -> Result<(), String> {
        if native != (self == Self::Native) {
            return Err(format!(
                "requested {} prediction, live native={native}",
                self.name()
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Transfer {
    #[default]
    Atomic,
    FixedOrder,
    GridReuse,
}

impl Transfer {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "atomic" => Ok(Self::Atomic),
            "fixed-order" => Ok(Self::FixedOrder),
            "grid-reuse" => Ok(Self::GridReuse),
            _ => Err(format!(
                "unknown soil transfer {name}; expected atomic, fixed-order or grid-reuse"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Atomic => "atomic",
            Self::FixedOrder => "fixed-order",
            Self::GridReuse => "grid-reuse",
        }
    }

    fn config(self) -> SoilTractorTransfer {
        match self {
            Self::Atomic => SoilTractorTransfer::Atomic,
            Self::FixedOrder => SoilTractorTransfer::FixedOrder {
                max_gather_bytes: 3_200_000,
            },
            Self::GridReuse => SoilTractorTransfer::GridReuse {
                max_gather_bytes: 14_075_712,
            },
        }
    }

    fn check(self, info: Option<MpmSoilTransferInfo>) -> Result<MpmSoilTransferInfo, String> {
        let info = info.ok_or("missing live soil transfer information")?;
        let fixed = self != Self::Atomic;
        let cfg = SoilTractorConfig::default();
        let bytes = self
            .config()
            .required_gather_bytes(
                cfg.particle_count().map_err(|e| e.to_string())?,
                cfg.grid_size,
            )
            .map_err(|e| e.to_string())?;
        if info.fixed_order != fixed
            || info.shared_chunks
            || info.grid_reuse != (self == Self::GridReuse)
            || info.particle_ordering != fixed
            || info.parallel_sort != fixed
            || info.gather_bytes != bytes
            || (info.ordering_bytes > 0) != fixed
        {
            return Err(format!("requested {} transfer, got {info:?}", self.name()));
        }
        Ok(info)
    }
}

pub fn solver_profile() -> &'static str {
    if cfg!(feature = "soil-live-soa") {
        "soa"
    } else {
        "volatile"
    }
}

pub fn preparation_profile() -> &'static str {
    if cfg!(feature = "soil-live-row-preparation") {
        "cooperative"
    } else {
        "default"
    }
}

pub fn submission_profile() -> &'static str {
    if cfg!(feature = "soil-live-batched-projection") {
        "composed"
    } else {
        "separate"
    }
}

pub fn check_submission_backend(composed: Option<bool>) -> Result<(), String> {
    if composed != Some(cfg!(feature = "soil-live-batched-projection")) {
        return Err(format!(
            "compiled {} submissions do not match runtime {composed:?}",
            submission_profile()
        ));
    }
    Ok(())
}

pub fn drift_profile() -> &'static str {
    if cfg!(feature = "soil-live-deferred-drift") {
        "deferred"
    } else {
        "full"
    }
}

pub fn geometry_profile() -> &'static str {
    if cfg!(feature = "soil-live-native-lazy-drift") {
        "native-lazy"
    } else {
        "eager"
    }
}

pub fn geometry_features() -> wgpu::Features {
    if cfg!(feature = "soil-live-native-lazy-drift") {
        wgpu::Features::SHADER_F64
    } else {
        wgpu::Features::empty()
    }
}

pub fn check_geometry_backend(backend: Option<&str>) -> Result<(), String> {
    let expected = if cfg!(feature = "soil-live-native-lazy-drift") {
        "native-geometry-lazy-anchors"
    } else {
        "eager-anchors"
    };
    if backend != Some(expected) {
        return Err(format!(
            "compiled {} geometry requires prepared {expected}, got {backend:?}",
            geometry_profile()
        ));
    }
    Ok(())
}

pub fn check_drift_backend(prediction: Prediction, deferred: bool) -> Result<(), String> {
    let expected = cfg!(feature = "soil-live-deferred-drift") && prediction == Prediction::Native;
    if deferred != expected {
        return Err(format!(
            "compiled {} native drift with {} prediction differs from runtime deferred={deferred}",
            drift_profile(),
            prediction.name()
        ));
    }
    Ok(())
}

pub fn compiled_preparation_backend() -> &'static str {
    molla_solvers::soil_newton_prepare_gpu::ExperimentalSoilNewtonPrepareGpu::backend_label()
}

pub fn check_preparation_backend(backend: &str) -> Result<(), String> {
    let expected = if cfg!(feature = "soil-live-row-preparation") {
        "compensated-row-parallel-cooperative-factor"
    } else {
        "compensated-dense-terms"
    };
    if backend != expected {
        return Err(format!(
            "compiled {} preparation requires {expected}, got {backend}",
            preparation_profile()
        ));
    }
    Ok(())
}

pub fn expected_newton_backend() -> &'static str {
    if cfg!(feature = "soil-live-soa") {
        "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-parallel-indices-soa-scratch-256-pairs"
    } else {
        "fischer-conditioned-protected-stalled-volatile-256-pairs"
    }
}

fn check_newton_backend(backend: Option<&str>) -> Result<(), String> {
    if backend != Some(expected_newton_backend()) {
        return Err(format!(
            "compiled {} viewer requires {}, got {backend:?}",
            solver_profile(),
            expected_newton_backend()
        ));
    }
    Ok(())
}
pub const SURFACE: molla_solvers::kernels::SoilSurfaceConfig =
    molla_solvers::kernels::SoilSurfaceConfig {
        origin: [-3.2, -2.0],
        spacing: 0.1,
        nx: 64,
        nz: 40,
        floor_height: -0.5,
    };

#[derive(Clone, Copy)]
pub struct ObservationPolicy {
    pub particles: bool,
    pub capture_end: Option<u64>,
}

impl ObservationPolicy {
    fn wants_exact_surface(self, frame: u64) -> bool {
        frame == 0 || self.capture_end == Some(frame)
    }

    fn wants_particles(self, frame: u64) -> bool {
        self.particles
            || self
                .capture_end
                .is_some_and(|end| frame == 0 || frame == end)
    }
}

#[derive(Clone, Debug)]
pub struct Pose {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
}

pub struct Snapshot {
    pub frame: u64,
    pub seconds: f64,
    pub poses: [Pose; 6],
    pub particles: Vec<[f32; 3]>,
    pub surface: molla_solvers::kernels::SoilSurfaceSnapshot,
    pub surface_frame: u64,
    pub surface_generation: u64,
    pub surface_readback_bytes: u64,
    pub surface_waited: bool,
    pub surface_ms: f64,
    pub slip: [f64; 4],
    pub load: Option<ForceSample>,
    pub load_readback_bytes: u64,
    pub load_ms: f64,
    pub load_waited: bool,
    pub physics_ms: f64,
    pub observation_ms: f64,
}

pub struct SurfaceUpdate {
    pub surface: molla_solvers::kernels::SoilSurfaceSnapshot,
    pub frame: u64,
    pub generation: u64,
    pub readback_bytes: u64,
    pub poll_ms: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForceSample {
    pub frame: u64,
    pub seconds: f64,
    pub force: [f64; 3],
}

impl ForceSample {
    fn from_impulse(frame: u64, seconds: f64, impulse: [f32; 3]) -> Result<Self, String> {
        let sample = Self {
            frame,
            seconds,
            force: impulse.map(|v| f64::from(v) / seconds),
        };
        if !sample.valid(frame) {
            return Err("invalid captured force or interval".into());
        }
        Ok(sample)
    }

    fn valid(&self, frame: u64) -> bool {
        self.frame > 0
            && self.frame <= frame
            && self.seconds.is_finite()
            && self.seconds > 0.0
            && self.force.iter().all(|v| v.is_finite())
    }
}

pub struct LoadUpdate {
    pub sample: ForceSample,
    pub readback_bytes: u64,
    pub poll_ms: f64,
}

#[derive(Default)]
struct LoadCache {
    sample: Option<molla_solvers::rigid_world::MpmSoilFrameLoadObservation>,
    requested: Option<u64>,
}

impl LoadCache {
    fn pending(&self) -> bool {
        self.requested.is_some_and(|frame| {
            self.sample
                .as_ref()
                .is_none_or(|s| s.loads().committed_frame != frame)
        })
    }

    fn force(&self, tractor: &SoilTractor, frame: u64) -> Result<Option<ForceSample>, String> {
        self.sample
            .as_ref()
            .map(|sample| {
                if !tractor.is_current_soil_frame_load_observation(sample) {
                    return Err("stale load owner or generation".into());
                }
                let loads = sample.loads();
                let handles = tractor.handles();
                if loads.bodies.len() != 5
                    || loads
                        .bodies
                        .iter()
                        .zip(handles.wheels.into_iter().chain([handles.tine]))
                        .any(|(load, body)| load.body != body)
                {
                    return Err("expected four wheel and one tine load bindings".into());
                }
                let tine = loads
                    .bodies
                    .iter()
                    .find(|b| b.body == tractor.handles().tine)
                    .ok_or("missing tine load")?;
                let force = ForceSample::from_impulse(
                    loads.committed_frame,
                    loads.simulated_seconds,
                    tine.linear,
                )?;
                if !force.valid(frame) {
                    return Err("invalid delayed force sample".into());
                }
                Ok(force)
            })
            .transpose()
    }
}

fn valid_surface(surface: &molla_solvers::kernels::SoilSurfaceSnapshot) -> bool {
    surface.config == SURFACE
        && surface.heights.len() == (SURFACE.nx * SURFACE.nz) as usize
        && surface.counts.len() == surface.heights.len()
        && surface.counts.iter().map(|&n| u64::from(n)).sum::<u64>()
            + u64::from(surface.outside_particles)
            == 50_000
        && surface
            .heights
            .iter()
            .zip(&surface.counts)
            .all(|(&height, &count)| {
                height.is_finite()
                    && height >= SURFACE.floor_height
                    && (count != 0 || height == SURFACE.floor_height)
            })
}

impl Snapshot {
    pub fn apply_load(&mut self, update: LoadUpdate) -> Result<(), String> {
        if !update.sample.valid(self.frame)
            || self
                .load
                .as_ref()
                .is_some_and(|old| old.frame >= update.sample.frame)
            || update.readback_bytes != LOAD_BYTES
            || !update.poll_ms.is_finite()
            || update.poll_ms < 0.0
        {
            return Err("invalid idle force update; committed snapshot retained".into());
        }
        self.load = Some(update.sample);
        self.load_readback_bytes = update.readback_bytes;
        self.load_ms = update.poll_ms;
        self.load_waited = false;
        Ok(())
    }

    pub fn apply_surface(&mut self, update: SurfaceUpdate) -> Result<(), String> {
        if update.frame <= self.surface_frame
            || update.frame > self.frame
            || update.generation != self.surface_generation
            || update.readback_bytes != u64::from(SURFACE.nx) * u64::from(SURFACE.nz) * 24
            || !update.poll_ms.is_finite()
            || update.poll_ms < 0.0
            || !valid_surface(&update.surface)
        {
            return Err("invalid idle surface update; committed snapshot retained".into());
        }
        self.surface = update.surface;
        self.surface_frame = update.frame;
        self.surface_generation = update.generation;
        self.surface_readback_bytes = update.readback_bytes;
        self.surface_ms = update.poll_ms;
        self.surface_waited = false;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        let expected = self.frame as f64 * DT;
        if !self.seconds.is_finite()
            || (self.seconds - expected).abs() > 1e-8 * expected.max(1.0)
            || (!self.particles.is_empty() && self.particles.len() != 50_000)
            || self.particles.iter().flatten().any(|v| !v.is_finite())
            || self.surface_frame > self.frame
            || (self.surface_waited && self.surface_frame != self.frame)
            || self.surface_readback_bytes % (u64::from(SURFACE.nx) * u64::from(SURFACE.nz) * 24)
                != 0
            || !valid_surface(&self.surface)
            || self.poses.iter().any(|p| {
                p.position.iter().chain(&p.rotation).any(|v| !v.is_finite())
                    || (p.rotation.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() > 1e-3
            })
            || self.slip.iter().any(|v| !v.is_finite())
            || self
                .load
                .as_ref()
                .is_some_and(|sample| !sample.valid(self.frame))
            || (self.load_waited && self.load.as_ref().is_none_or(|s| s.frame != self.frame))
            || self.load_readback_bytes % LOAD_BYTES != 0
            || (self.load.is_none() && self.load_readback_bytes != 0)
            || [
                self.physics_ms,
                self.observation_ms,
                self.surface_ms,
                self.load_ms,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(
                "invalid or temporally inconsistent observation; snapshot not published".into(),
            );
        }
        Ok(())
    }
}

pub enum Event {
    Ready {
        adapter: String,
        construction_ms: f64,
        newton_backend: &'static str,
        transfer: Transfer,
        transfer_info: MpmSoilTransferInfo,
        prediction: Prediction,
        native_prediction: bool,
        compiled_preparation_backend: &'static str,
        composed_submissions: bool,
        profile_stages: bool,
        native_drift_deferred: bool,
        unrestricted_drift_backend: &'static str,
    },
    Profile(profiling::FrameProfile),
    Frame(Snapshot),
    Surface(SurfaceUpdate),
    Loads(LoadUpdate),
    Failed(String),
}

#[derive(Default)]
pub struct Schedule {
    pub ready: bool,
    pub busy: bool,
    pub paused: bool,
    pub frame: Option<u64>,
    pub failed: bool,
}

impl Schedule {
    pub fn accept(&mut self, frame: u64) -> Result<(), String> {
        let expected = self.frame.map_or(0, |n| n + 1);
        if self.failed || frame != expected || (self.frame.is_some() && !self.busy) {
            return Err(format!(
                "unexpected committed frame {frame}, expected {expected}"
            ));
        }
        self.frame = Some(frame);
        self.ready = true;
        self.busy = false;
        Ok(())
    }

    pub fn request(&mut self, once: bool, limit: Option<u64>) -> bool {
        if !self.ready
            || self.busy
            || self.failed
            || (self.paused && !once)
            || limit.is_some_and(|n| self.frame.is_some_and(|f| f >= n))
        {
            return false;
        }
        self.busy = true;
        true
    }
}

pub struct Worker {
    command: Option<mpsc::SyncSender<()>>,
    pub events: Mutex<mpsc::Receiver<Event>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Worker {
    pub fn spawn(
        policy: ObservationPolicy,
        transfer: Transfer,
        prediction: Prediction,
        profile_stages: bool,
    ) -> Result<Self, String> {
        if profile_stages && policy.capture_end.is_none() {
            return Err("stage profiling requires bounded capture".into());
        }
        let started = Instant::now();
        let (engine, adapter) = prepare_engine(prediction, profile_stages)?;
        let (command, requests) = mpsc::sync_channel(1);
        let (events, received) = mpsc::sync_channel(2);
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let thread = std::thread::Builder::new()
            .name("soil-tractor-physics".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(
                        requests,
                        &events,
                        &stopped,
                        engine,
                        adapter,
                        started,
                        policy,
                        transfer,
                        prediction,
                        profile_stages,
                    )
                }));
                let failure = match result {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => Some(error),
                    Err(_) => Some("physics worker panicked; last committed frame retained".into()),
                };
                if let Some(error) = failure {
                    let _ = events.send(Event::Failed(error));
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            command: Some(command),
            events: Mutex::new(received),
            stop,
            thread: Some(thread),
        })
    }

    pub fn step(&self) -> Result<(), String> {
        self.command
            .as_ref()
            .ok_or("worker stopped")?
            .try_send(())
            .map_err(|e| e.to_string())
    }

    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.command.take();
        let (_, disconnected) = mpsc::channel();
        *self.events.get_mut().unwrap_or_else(|e| e.into_inner()) = disconnected;
        if let Some(thread) = self.thread.take() {
            eprintln!("soil-live physics worker shutdown begin");
            let _ = thread.join();
            eprintln!("soil-live physics worker shutdown complete");
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn prepare_engine(
    prediction: Prediction,
    profile_stages: bool,
) -> Result<(Arc<Engine>, String), String> {
    if molla_solvers::soil_newton_gpu::diagnostic_inner_limit()
        .map_err(|e| e.to_string())?
        .is_some()
    {
        return Err("live viewer does not accept a diagnostic solver-work override".into());
    }
    let features =
        prediction.features() | geometry_features() | profiling::features(profile_stages);
    let engine = Arc::new(Engine::new_gpu_with_features(features).map_err(|e| e.to_string())?);
    let adapter = engine.adapter_info().ok_or("missing physics adapter")?;
    if adapter.is_software || !molla_solvers::soil_newton_gpu::protected_enabled(&engine) {
        return Err("hardware native-atomic protected Newton adapter required; no fallback".into());
    }
    let adapter = format!(
        "{} / {} {}",
        adapter.name, adapter.driver, adapter.driver_info
    );
    Ok((engine, adapter))
}

fn run(
    requests: mpsc::Receiver<()>,
    events: &mpsc::SyncSender<Event>,
    stop: &AtomicBool,
    engine: Arc<Engine>,
    adapter: String,
    started: Instant,
    policy: ObservationPolicy,
    transfer: Transfer,
    prediction: Prediction,
    profile_stages: bool,
) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        return Ok(());
    }
    let coupling = molla_solvers::mpm::MpmUnilateralConfig {
        node_capacity: 4096,
        search_distance: 0.05,
        max_iterations: 64,
    };
    let mut tractor = match prediction {
        Prediction::Featherstone => SoilTractor::new_native_explicit_with_paired_newton_transfer(
            engine,
            SoilTractorConfig::default(),
            coupling,
            0.3,
            molla_solvers::kernels::FeatherstoneExecution::Generic,
            molla_solvers::mpm::MpmGridStorage::Dense,
            transfer.config(),
        ),
        Prediction::Native => SoilTractor::new_native_prediction_with_newton_transfer(
            engine,
            SoilTractorConfig::default(),
            coupling,
            0.3,
            molla_solvers::mpm::MpmGridStorage::Dense,
            DT,
            transfer.config(),
        ),
    }
    .map_err(|e| e.to_string())?;
    let newton_backend = tractor.world().mpm_soil_newton_backend();
    check_newton_backend(newton_backend)?;
    let newton_backend = newton_backend.unwrap();
    let transfer_info = transfer.check(tractor.world().mpm_soil_transfer_info())?;
    let native_prediction = tractor.world().gpu_uses_native_prediction();
    prediction.check(native_prediction)?;
    let compiled_preparation_backend = compiled_preparation_backend();
    check_preparation_backend(compiled_preparation_backend)?;
    let composed_submissions = tractor.world().mpm_soil_composes_submissions();
    check_submission_backend(composed_submissions)?;
    let composed_submissions = composed_submissions.unwrap();
    let native_drift_deferred = tractor.world().mpm_soil_defers_native_prediction_drift();
    check_drift_backend(prediction, native_drift_deferred)?;
    let unrestricted_drift_backend = tractor.world().solver.gpu_unrestricted_drift_backend();
    check_geometry_backend(unrestricted_drift_backend)?;
    let unrestricted_drift_backend = unrestricted_drift_backend.unwrap();
    if profile_stages {
        tractor
            .prepare_soil_frame_profiling_with_preparation(profiling::substeps())
            .map_err(|e| e.to_string())?;
    }
    tractor
        .prepare_soil_frame_loads()
        .map_err(|e| e.to_string())?;
    let mut surface = tractor
        .prepare_soil_surface(SURFACE, 16 * 1024 * 1024)
        .map_err(|e| e.to_string())?;
    let mut surface_cache = None;
    let mut load_cache = LoadCache::default();
    if stop.load(Ordering::Acquire) {
        return Ok(());
    }
    events
        .send(Event::Ready {
            adapter,
            construction_ms: started.elapsed().as_secs_f64() * 1000.0,
            newton_backend,
            transfer,
            transfer_info,
            prediction,
            native_prediction,
            compiled_preparation_backend,
            composed_submissions,
            profile_stages,
            native_drift_deferred,
            unrestricted_drift_backend,
        })
        .map_err(|e| e.to_string())?;
    events
        .send(Event::Frame(snapshot(
            &mut tractor,
            &mut surface,
            &mut surface_cache,
            &mut load_cache,
            0,
            0.0,
            policy,
            stop,
        )?))
        .map_err(|e| e.to_string())?;
    let mut frame = 0;
    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let request = if surface.snapshot_pending() || load_cache.pending() {
            requests.recv_timeout(std::time::Duration::from_millis(5))
        } else {
            requests
                .recv()
                .map_err(|_| mpsc::RecvTimeoutError::Disconnected)
        };
        match request {
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let started = Instant::now();
                let bytes = observe_surface(
                    &tractor,
                    &mut surface,
                    &mut surface_cache,
                    frame,
                    false,
                    stop,
                )?;
                if bytes > 0 {
                    let sample = surface_cache.as_ref().ok_or("missing idle surface")?;
                    events
                        .send(Event::Surface(SurfaceUpdate {
                            surface: sample.surface().clone(),
                            frame: sample.revision(),
                            generation: sample.generation(),
                            readback_bytes: bytes,
                            poll_ms: started.elapsed().as_secs_f64() * 1000.0,
                        }))
                        .map_err(|e| e.to_string())?;
                }
                let started = Instant::now();
                let bytes = observe_loads(&mut tractor, &mut load_cache, frame, false, stop)?;
                if bytes > 0 {
                    events
                        .send(Event::Loads(LoadUpdate {
                            sample: load_cache
                                .force(&tractor, frame)?
                                .ok_or("missing idle loads")?,
                            readback_bytes: bytes,
                            poll_ms: started.elapsed().as_secs_f64() * 1000.0,
                        }))
                        .map_err(|e| e.to_string())?;
                }
                continue;
            }
            Ok(()) => {}
        }
        if stop.load(Ordering::Acquire) {
            break;
        }
        let started = Instant::now();
        tractor.step(DT, TORQUE).map_err(|e| e.to_string())?;
        let physics_ms = started.elapsed().as_secs_f64() * 1000.0;
        frame += 1;
        if stop.load(Ordering::Acquire) {
            break;
        }
        if profile_stages {
            let started = Instant::now();
            let profile = tractor
                .read_soil_frame_profile()
                .map_err(|e| e.to_string())?
                .ok_or("missing requested committed-frame profile")?;
            let sample = profiling::FrameProfile::new(
                profile,
                frame,
                started.elapsed().as_secs_f64() * 1000.0,
            )?;
            events
                .send(Event::Profile(sample))
                .map_err(|e| e.to_string())?;
        }
        events
            .send(Event::Frame(snapshot(
                &mut tractor,
                &mut surface,
                &mut surface_cache,
                &mut load_cache,
                frame,
                physics_ms,
                policy,
                stop,
            )?))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn snapshot(
    tractor: &mut SoilTractor,
    surface: &mut molla_solvers::kernels::PreparedSoilSurface,
    surface_cache: &mut Option<molla_solvers::kernels::SoilSurfaceObservation>,
    load_cache: &mut LoadCache,
    frame: u64,
    physics_ms: f64,
    policy: ObservationPolicy,
    stop: &AtomicBool,
) -> Result<Snapshot, String> {
    let started = Instant::now();
    let handles = tractor.handles();
    let bodies = [
        handles.chassis,
        handles.wheels[0],
        handles.wheels[1],
        handles.wheels[2],
        handles.wheels[3],
        handles.tine,
    ];
    let poses: Vec<_> = bodies
        .into_iter()
        .map(|body| {
            tractor
                .world()
                .scene
                .body_pose(body)
                .map(|pose| Pose {
                    position: pose.position.to_array().map(|v| v as f32),
                    rotation: pose.rotation.to_array().map(|v| v as f32),
                })
                .map_err(|e| e.to_string())
        })
        .collect::<Result<_, _>>()?;
    let telemetry = tractor.telemetry().map_err(|e| e.to_string())?;
    let want_particles = policy.wants_particles(frame);
    let particles = if want_particles {
        tractor.read_soil_positions().map_err(|e| e.to_string())?
    } else {
        Vec::new()
    };
    if (want_particles && particles.len() != 50_000)
        || particles.iter().flatten().any(|v| !v.is_finite())
    {
        return Err("invalid particle observation; snapshot not published".into());
    }
    let load_started = Instant::now();
    let load_waited = frame > 0 && policy.capture_end == Some(frame);
    let load_readback_bytes = observe_loads(tractor, load_cache, frame, load_waited, stop)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let load = load_cache.force(tractor, frame)?;
    let surface_started = Instant::now();
    let exact_surface = policy.wants_exact_surface(frame);
    let surface_readback_bytes =
        observe_surface(tractor, surface, surface_cache, frame, exact_surface, stop)?;
    let surface_ms = surface_started.elapsed().as_secs_f64() * 1000.0;
    let observed = surface_cache
        .as_ref()
        .ok_or("missing initial geometric surface")?;
    let snapshot = Snapshot {
        frame,
        seconds: telemetry.simulated_seconds,
        poses: poses.try_into().map_err(|_| "six body poses required")?,
        particles,
        surface: observed.surface().clone(),
        surface_frame: observed.revision(),
        surface_generation: observed.generation(),
        surface_readback_bytes,
        surface_waited: exact_surface,
        surface_ms,
        slip: telemetry.nominal_slip,
        load,
        load_readback_bytes,
        load_ms,
        load_waited,
        physics_ms,
        observation_ms: started.elapsed().as_secs_f64() * 1000.0,
    };
    snapshot.validate()?;
    Ok(snapshot)
}

fn observe_loads(
    tractor: &mut SoilTractor,
    cache: &mut LoadCache,
    frame: u64,
    exact: bool,
    stop: &AtomicBool,
) -> Result<u64, String> {
    if frame == 0 {
        return Ok(0);
    }
    let deadline = Instant::now() + std::time::Duration::from_secs(15);
    let mut bytes = 0;
    loop {
        if stop.load(Ordering::Acquire) {
            return Err("load observation cancelled".into());
        }
        if let Some(sample) = tractor.poll_soil_frame_loads().map_err(|e| e.to_string())? {
            let revision = sample.loads().committed_frame;
            if !tractor.is_current_soil_frame_load_observation(&sample)
                || Some(revision) != cache.requested
                || revision > frame
                || cache
                    .sample
                    .as_ref()
                    .is_some_and(|old| old.loads().committed_frame >= revision)
                || sample.loads().bodies.len() != 5
            {
                return Err("invalid delayed load revision or generation".into());
            }
            bytes += LOAD_BYTES;
            cache.sample = Some(sample);
        }
        if cache
            .sample
            .as_ref()
            .is_some_and(|s| s.loads().committed_frame == frame)
        {
            break;
        }
        if !cache.pending() {
            if !tractor
                .try_request_soil_frame_loads()
                .map_err(|e| e.to_string())?
            {
                return Err("unexpected occupied load stage".into());
            }
            cache.requested = Some(frame);
        }
        if !exact {
            break;
        }
        if Instant::now() >= deadline {
            return Err("exact load observation timed out".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    Ok(bytes)
}

fn observe_surface(
    tractor: &SoilTractor,
    surface: &mut molla_solvers::kernels::PreparedSoilSurface,
    cache: &mut Option<molla_solvers::kernels::SoilSurfaceObservation>,
    frame: u64,
    exact: bool,
    stop: &AtomicBool,
) -> Result<u64, String> {
    let deadline = Instant::now() + std::time::Duration::from_secs(15);
    let mut bytes = 0;
    let mut requested = false;
    loop {
        if stop.load(Ordering::Acquire) {
            return Err("surface observation cancelled".into());
        }
        if let Some(sample) = tractor
            .poll_soil_surface(surface)
            .map_err(|e| e.to_string())?
        {
            if !surface.is_current_snapshot(&sample)
                || sample.revision() > frame
                || cache.as_ref().is_some_and(|old| {
                    old.generation() == sample.generation() && old.revision() >= sample.revision()
                })
            {
                return Err("invalid delayed surface revision or generation".into());
            }
            bytes += surface.readback_bytes();
            *cache = Some(sample);
        }
        if cache
            .as_ref()
            .is_some_and(|sample| sample.revision() == frame && surface.is_current_snapshot(sample))
        {
            break;
        }
        if !requested && !surface.snapshot_pending() {
            requested = tractor
                .try_request_soil_surface(surface)
                .map_err(|e| e.to_string())?;
        }
        if !exact {
            break;
        }
        if Instant::now() >= deadline {
            return Err("exact surface observation timed out".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    if cache
        .as_ref()
        .is_none_or(|sample| !surface.is_current_snapshot(sample))
    {
        return Err("no current-generation surface available".into());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prediction_selection_requires_live_identity_and_device_features() {
        assert_eq!(Prediction::default(), Prediction::Featherstone);
        for prediction in [Prediction::Featherstone, Prediction::Native] {
            assert_eq!(Prediction::parse(prediction.name()).unwrap(), prediction);
            let native = prediction == Prediction::Native;
            prediction.check(native).unwrap();
            assert!(prediction.check(!native).is_err());
            assert_eq!(
                prediction.features().contains(wgpu::Features::SHADER_F64),
                native
            );
            assert!(prediction.features().contains(
                wgpu::Features::SHADER_FLOAT32_ATOMIC | wgpu::Features::PASSTHROUGH_SHADERS
            ));
        }
        assert!(Prediction::parse("unknown").is_err());
    }

    #[test]
    fn transfer_selection_requires_live_mode_and_storage() {
        assert_eq!(Transfer::default(), Transfer::Atomic);
        assert!(matches!(
            Transfer::Atomic.config(),
            SoilTractorTransfer::Atomic
        ));
        assert!(matches!(
            Transfer::FixedOrder.config(),
            SoilTractorTransfer::FixedOrder {
                max_gather_bytes: 3_200_000
            }
        ));
        let atomic = MpmSoilTransferInfo::default();
        let fixed = MpmSoilTransferInfo {
            fixed_order: true,
            particle_ordering: true,
            parallel_sort: true,
            gather_bytes: 3_200_000,
            ordering_bytes: 1_201_840,
            ..atomic
        };
        Transfer::Atomic.check(Some(atomic)).unwrap();
        Transfer::FixedOrder.check(Some(fixed)).unwrap();
        let reuse = MpmSoilTransferInfo {
            grid_reuse: true,
            gather_bytes: 14_075_712,
            ..fixed
        };
        Transfer::GridReuse.check(Some(reuse)).unwrap();
        Transfer::GridReuse
            .config()
            .validate_with_grid(50_000, 64)
            .unwrap();
        for (mode, valid) in [
            (Transfer::Atomic, atomic),
            (Transfer::FixedOrder, fixed),
            (Transfer::GridReuse, reuse),
        ] {
            assert_eq!(Transfer::parse(mode.name()).unwrap(), mode);
            assert!(mode.check(None).is_err());
            for info in [atomic, fixed, reuse] {
                assert_eq!(mode.check(Some(info)).is_ok(), info == valid);
            }
        }
        for info in [
            MpmSoilTransferInfo {
                grid_reuse: false,
                ..reuse
            },
            MpmSoilTransferInfo {
                shared_chunks: true,
                ..reuse
            },
            MpmSoilTransferInfo {
                particle_ordering: false,
                ..reuse
            },
            MpmSoilTransferInfo {
                parallel_sort: false,
                ..reuse
            },
            MpmSoilTransferInfo {
                fixed_order: false,
                ..reuse
            },
            MpmSoilTransferInfo {
                gather_bytes: 14_075_711,
                ..reuse
            },
            MpmSoilTransferInfo {
                ordering_bytes: 0,
                ..reuse
            },
        ] {
            assert!(Transfer::GridReuse.check(Some(info)).is_err());
        }
        assert!(Transfer::Atomic.check(Some(fixed)).is_err());
        assert!(Transfer::FixedOrder.check(Some(atomic)).is_err());
        assert!(Transfer::FixedOrder.check(None).is_err());
        for invalid in [
            MpmSoilTransferInfo {
                fixed_order: false,
                ..fixed
            },
            MpmSoilTransferInfo {
                shared_chunks: true,
                ..fixed
            },
            MpmSoilTransferInfo {
                particle_ordering: false,
                ..fixed
            },
            MpmSoilTransferInfo {
                parallel_sort: false,
                ..fixed
            },
            MpmSoilTransferInfo {
                gather_bytes: 3_199_999,
                ..fixed
            },
            MpmSoilTransferInfo {
                ordering_bytes: 0,
                ..fixed
            },
        ] {
            assert!(Transfer::FixedOrder.check(Some(invalid)).is_err());
        }
    }
    #[test]
    fn schedule_bounds_work_and_publishes_only_committed_frames() {
        let mut state = Schedule::default();
        assert!(!state.request(false, None));
        state.accept(0).unwrap();
        assert!(state.request(false, None));
        assert!(!state.request(true, None));
        assert!(state.accept(2).is_err());
        state.accept(1).unwrap();
        assert!(state.accept(1).is_err());
        assert!(!state.request(false, Some(1)));
        state.paused = true;
        assert!(!state.request(false, None));
        assert!(state.request(true, None));
        state.accept(2).unwrap();
        state.failed = true;
        assert!(!state.request(true, None));
        assert!(state.accept(3).is_err());
    }

    #[test]
    fn particle_observation_is_explicit_or_capture_endpoint_only() {
        let expected = if cfg!(feature = "soil-live-native-lazy-drift") {
            "native-geometry-lazy-anchors"
        } else {
            "eager-anchors"
        };
        check_geometry_backend(Some(expected)).unwrap();
        assert!(check_geometry_backend(None).is_err());
        assert!(check_geometry_backend(Some("unknown")).is_err());
        assert_eq!(
            geometry_features().contains(wgpu::Features::SHADER_F64),
            cfg!(feature = "soil-live-native-lazy-drift")
        );
        assert!(matches!(drift_profile(), "full" | "deferred"));
        for prediction in [Prediction::Native, Prediction::Featherstone] {
            let expected =
                cfg!(feature = "soil-live-deferred-drift") && prediction == Prediction::Native;
            check_drift_backend(prediction, expected).unwrap();
            assert!(check_drift_backend(prediction, !expected).is_err());
        }
        check_preparation_backend(compiled_preparation_backend()).unwrap();
        assert!(check_preparation_backend("unknown").is_err());
        assert!(matches!(preparation_profile(), "default" | "cooperative"));
        check_submission_backend(Some(cfg!(feature = "soil-live-batched-projection"))).unwrap();
        assert!(check_submission_backend(None).is_err());
        assert!(
            check_submission_backend(Some(!cfg!(feature = "soil-live-batched-projection")))
                .is_err()
        );
        assert!(matches!(submission_profile(), "separate" | "composed"));
        check_newton_backend(Some(expected_newton_backend())).unwrap();
        assert!(check_newton_backend(None).is_err());
        assert!(check_newton_backend(Some("unrecognized-backend")).is_err());
        assert!(matches!(solver_profile(), "soa" | "volatile"));
        let default = ObservationPolicy {
            particles: false,
            capture_end: None,
        };
        assert!(!(0..34).any(|n| default.wants_particles(n)));
        assert!(default.wants_exact_surface(0));
        assert!(!(1..34).any(|n| default.wants_exact_surface(n)));
        let capture = ObservationPolicy {
            capture_end: Some(33),
            ..default
        };
        assert_eq!(
            (0..34)
                .filter(|&n| capture.wants_particles(n))
                .collect::<Vec<_>>(),
            [0, 33]
        );
        let diagnostic = ObservationPolicy {
            particles: true,
            ..default
        };
        assert!((0..34).all(|n| diagnostic.wants_particles(n)));
        assert_eq!(
            (0..34)
                .filter(|&n| capture.wants_exact_surface(n))
                .collect::<Vec<_>>(),
            [0, 33]
        );
    }

    fn fixture_snapshot() -> Snapshot {
        Snapshot {
            frame: 0,
            seconds: 0.0,
            poses: std::array::from_fn(|_| Pose {
                position: [0.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
            }),
            particles: vec![[0.0; 3]; 50_000],
            surface: molla_solvers::kernels::SoilSurfaceSnapshot {
                config: SURFACE,
                heights: vec![SURFACE.floor_height; 2560],
                counts: {
                    let mut counts = vec![0; 2560];
                    counts[0] = 50_000;
                    counts
                },
                outside_particles: 0,
            },
            surface_ms: 0.0,
            surface_frame: 0,
            surface_generation: 0,
            surface_readback_bytes: 0,
            surface_waited: true,
            slip: [0.0; 4],
            load: None,
            load_readback_bytes: 0,
            load_ms: 0.0,
            load_waited: false,
            physics_ms: 0.0,
            observation_ms: 0.0,
        }
    }

    #[test]
    fn snapshot_rejects_invalid_or_inconsistent_observations() {
        let mut snapshot = fixture_snapshot();
        snapshot.validate().unwrap();
        snapshot.frame = 1;
        assert!(snapshot.validate().is_err());
        snapshot.seconds = DT;
        assert!(snapshot.validate().is_err());
        snapshot.load = Some(ForceSample {
            frame: 1,
            seconds: DT,
            force: [0.0; 3],
        });
        assert!(snapshot.validate().is_err());
        snapshot.surface_waited = false;
        snapshot.validate().unwrap();
        snapshot.surface_frame = 2;
        assert!(snapshot.validate().is_err());
        snapshot.surface_frame = 0;
        snapshot.surface_readback_bytes = 1;
        assert!(snapshot.validate().is_err());
        snapshot.surface_readback_bytes = 61440;
        snapshot.validate().unwrap();
        snapshot.particles[123][1] = f32::NAN;
        assert!(snapshot.validate().is_err());
        snapshot.particles[123][1] = 0.0;
        snapshot.poses[2].rotation = [0.0; 4];
        assert!(snapshot.validate().is_err());
        snapshot.poses[2].rotation[3] = 1.0;
        snapshot.observation_ms = f64::INFINITY;
        assert!(snapshot.validate().is_err());
    }

    #[test]
    fn idle_surface_updates_only_geometry_and_rejects_invalid_updates_atomically() {
        let mut snapshot = fixture_snapshot();
        snapshot.frame = 2;
        snapshot.seconds = 2.0 * DT;
        snapshot.load = Some(ForceSample {
            frame: 1,
            seconds: DT,
            force: [1.0; 3],
        });
        snapshot.surface_waited = false;
        snapshot.physics_ms = 123.0;
        snapshot.observation_ms = 45.0;
        let update = || SurfaceUpdate {
            surface: fixture_snapshot().surface,
            frame: 1,
            generation: 0,
            readback_bytes: 61440,
            poll_ms: 0.1,
        };
        for invalid in 0..8 {
            let mut bad = update();
            match invalid {
                0 => bad.frame = 0,
                1 => bad.frame = 3,
                2 => bad.generation = 1,
                3 => bad.readback_bytes = 0,
                4 => bad.poll_ms = f64::NAN,
                5 => bad.poll_ms = -1.0,
                6 => bad.surface.heights[0] = f32::INFINITY,
                _ => bad.surface.counts[0] -= 1,
            }
            assert!(snapshot.apply_surface(bad).is_err());
            assert_eq!(snapshot.surface_frame, 0);
            assert_eq!(snapshot.surface.counts[0], 50_000);
            snapshot.validate().unwrap();
        }
        snapshot.apply_surface(update()).unwrap();
        assert_eq!(snapshot.surface_frame, 1);
        assert!(snapshot.apply_surface(update()).is_err());
        let mut latest = update();
        latest.frame = 2;
        snapshot.apply_surface(latest).unwrap();
        snapshot.validate().unwrap();
        assert_eq!(snapshot.frame, 2);
        assert_eq!(snapshot.seconds, 2.0 * DT);
        assert_eq!(snapshot.load.as_ref().unwrap().force, [1.0; 3]);
        assert_eq!(snapshot.physics_ms, 123.0);
        assert_eq!(snapshot.observation_ms, 45.0);
        assert_eq!(snapshot.particles.len(), 50_000);
        assert!(!snapshot.surface_waited);
    }

    #[test]
    #[ignore = "actual 50k tractor GPU worker; bounded dedicated acceptance target"]
    fn idle_surface_worker_gpu() {
        let deadline = Instant::now() + std::time::Duration::from_secs(170);
        let mut worker = Worker::spawn(
            ObservationPolicy {
                particles: true,
                capture_end: None,
            },
            Transfer::Atomic,
            Prediction::Featherstone,
            false,
        )
        .unwrap();
        let receive = |idle: bool| {
            let remaining = deadline.saturating_duration_since(Instant::now());
            worker
                .events
                .lock()
                .unwrap()
                .recv_timeout(if idle {
                    remaining.min(std::time::Duration::from_secs(15))
                } else {
                    remaining
                })
                .expect("worker event deadline")
        };
        match receive(false) {
            Event::Ready {
                adapter,
                construction_ms,
                newton_backend,
                transfer,
                transfer_info,
                prediction,
                native_prediction,
                compiled_preparation_backend,
                composed_submissions,
                profile_stages,
                native_drift_deferred,
                unrestricted_drift_backend,
            } => {
                check_geometry_backend(Some(unrestricted_drift_backend)).unwrap();
                check_drift_backend(prediction, native_drift_deferred).unwrap();
                assert!(!profile_stages);
                assert_eq!(transfer, Transfer::Atomic);
                assert_eq!(prediction, Prediction::Featherstone);
                prediction.check(native_prediction).unwrap();
                check_preparation_backend(compiled_preparation_backend).unwrap();
                check_submission_backend(Some(composed_submissions)).unwrap();
                transfer.check(Some(transfer_info)).unwrap();
                eprintln!(
                    "idle-surface adapter={adapter} construction_ms={construction_ms} newton_backend={newton_backend}"
                );
            }
            _ => panic!("missing ready event"),
        }
        let mut schedule = Schedule::default();
        let Event::Frame(initial) = receive(false) else {
            panic!("missing initial frame")
        };
        initial.validate().unwrap();
        schedule.accept(initial.frame).unwrap();
        for frame in 1..=2 {
            schedule.paused = true;
            assert!(schedule.request(true, None));
            worker.step().unwrap();
            let Event::Frame(mut snapshot) = receive(false) else {
                panic!("missing stepped frame")
            };
            snapshot.validate().unwrap();
            schedule.accept(snapshot.frame).unwrap();
            assert_eq!(snapshot.frame, frame);
            assert_eq!(snapshot.surface_frame, frame - 1);
            assert!(!schedule.request(false, None));
            assert!(snapshot.load.as_ref().is_none_or(|s| s.frame < frame));
            assert!(!snapshot.load_waited);
            while snapshot.surface_frame != frame
                || snapshot.load.as_ref().is_none_or(|s| s.frame != frame)
            {
                match receive(true) {
                    Event::Surface(update) => snapshot.apply_surface(update).unwrap(),
                    Event::Loads(update) => snapshot.apply_load(update).unwrap(),
                    Event::Failed(error) => panic!("worker: {error}"),
                    _ => panic!("unexpected physics progress while paused"),
                }
            }
            let load = snapshot.load.as_ref().unwrap();
            assert!((load.seconds - DT).abs() < 1e-8);
            assert_eq!(snapshot.load_readback_bytes, LOAD_BYTES);
            assert!(!snapshot.load_waited);
            eprintln!(
                "idle-force frame={} interval={} force={:?} bytes={} poll_ms={}",
                load.frame,
                load.seconds,
                load.force,
                snapshot.load_readback_bytes,
                snapshot.load_ms
            );
            snapshot.validate().unwrap();
            assert_eq!(schedule.frame, Some(frame));
            assert!(!schedule.busy);
            assert!(schedule.paused);
            let mut counts = vec![0_u32; 2560];
            let mut heights = vec![SURFACE.floor_height; 2560];
            let mut outside = 0;
            let interval = |v: f32, origin: f32, size: u32| {
                (0..size).find(|&i| {
                    let low =
                        (f64::from(origin) + f64::from(i) * f64::from(SURFACE.spacing)) as f32;
                    let high =
                        (f64::from(origin) + f64::from(i + 1) * f64::from(SURFACE.spacing)) as f32;
                    v >= low && v < high
                })
            };
            for &[x, y, z] in &snapshot.particles {
                match (
                    interval(x, SURFACE.origin[0], SURFACE.nx),
                    interval(z, SURFACE.origin[1], SURFACE.nz),
                ) {
                    (Some(x), Some(z)) => {
                        let index = (z * SURFACE.nx + x) as usize;
                        counts[index] += 1;
                        heights[index] = heights[index].max(y);
                    }
                    _ => outside += 1,
                }
            }
            assert_eq!(counts, snapshot.surface.counts);
            assert_eq!(outside, snapshot.surface.outside_particles);
            let max_error = heights
                .iter()
                .zip(&snapshot.surface.heights)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_error <= 1e-6, "height error {max_error}");
            eprintln!(
                "idle-surface frame={frame} particles={} bytes={} poll_ms={} height_error={max_error} physics_ms={}",
                snapshot.particles.len(),
                snapshot.surface_readback_bytes,
                snapshot.surface_ms,
                snapshot.physics_ms
            );
            assert!(matches!(
                worker
                    .events
                    .lock()
                    .unwrap()
                    .recv_timeout(std::time::Duration::from_millis(30)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ));
        }
        worker.shutdown();
        assert!(worker.thread.is_none());
    }

    #[test]
    fn force_updates_preserve_physics_and_reject_invalid_or_backwards_samples() {
        let force = ForceSample::from_impulse(7, 0.25, [1.0, -2.0, 0.5]).unwrap();
        assert_eq!(force.force, [4.0, -8.0, 2.0]);
        for seconds in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(ForceSample::from_impulse(7, seconds, [1.0; 3]).is_err());
        }
        let mut snapshot = fixture_snapshot();
        snapshot.frame = 2;
        snapshot.seconds = 2.0 * DT;
        snapshot.surface_waited = false;
        snapshot.validate().unwrap();
        let update = || LoadUpdate {
            sample: ForceSample {
                frame: 1,
                seconds: 0.01,
                force: [1.0, 2.0, 3.0],
            },
            readback_bytes: LOAD_BYTES,
            poll_ms: 0.1,
        };
        for invalid in 0..8 {
            let mut bad = update();
            match invalid {
                0 => bad.sample.frame = 0,
                1 => bad.sample.frame = 3,
                2 => bad.sample.seconds = 0.0,
                3 => bad.sample.seconds = f64::NAN,
                4 => bad.sample.force[0] = f64::INFINITY,
                5 => bad.readback_bytes = 0,
                6 => bad.poll_ms = -1.0,
                _ => bad.poll_ms = f64::NAN,
            }
            assert!(snapshot.apply_load(bad).is_err());
            assert!(snapshot.load.is_none());
            snapshot.validate().unwrap();
        }
        snapshot.apply_load(update()).unwrap();
        assert!(snapshot.apply_load(update()).is_err());
        snapshot.load_waited = true;
        assert!(snapshot.validate().is_err());
        snapshot.load_waited = false;
        let mut latest = update();
        latest.sample.frame = 2;
        snapshot.apply_load(latest).unwrap();
        snapshot.validate().unwrap();
        assert_eq!(snapshot.frame, 2);
        assert_eq!(snapshot.seconds, 2.0 * DT);
        assert_eq!(snapshot.surface_frame, 0);
        assert_eq!(snapshot.physics_ms, 0.0);
        assert_eq!(snapshot.load.as_ref().unwrap().seconds, 0.01);
        assert_eq!(snapshot.load.as_ref().unwrap().force, [1.0, 2.0, 3.0]);
        assert!(!snapshot.load_waited);
    }

    #[test]
    fn shutdown_joins_idle_and_backpressured_workers() {
        for blocked_send in [false, true] {
            let (command, requests) = mpsc::sync_channel(1);
            let (events, received) = mpsc::sync_channel(0);
            let (started, ready) = mpsc::channel();
            let thread = std::thread::spawn(move || {
                started.send(()).unwrap();
                if blocked_send {
                    let _ = events.send(Event::Failed("test".into()));
                } else {
                    assert!(requests.recv().is_err());
                }
            });
            ready
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap();
            let mut worker = Worker {
                command: Some(command),
                events: Mutex::new(received),
                stop: Arc::new(AtomicBool::new(false)),
                thread: Some(thread),
            };
            worker.shutdown();
            assert!(worker.thread.is_none());
            assert!(worker.stop.load(Ordering::Acquire));
            assert!(worker.step().is_err());
            worker.shutdown();
        }
    }
}
