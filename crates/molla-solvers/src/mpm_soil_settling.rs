//! Bounded full-population preparation through existing archive boundaries.

use super::*;
use std::time::{Duration, Instant};

/// Explicit numerical activity cutoffs, not calibrated material parameters.
/// Units match `MpmSoilActivity`. All limits must be finite and nonnegative.
#[derive(Clone, Copy, Debug)]
pub struct MpmSoilSettlingLimits {
    pub speed: f64,
    pub affine_speed: f64,
    pub displacement_rate: f64,
    pub acceleration: f64,
    pub elastic_rate: f64,
    pub plastic_rate: f64,
    pub tire_history_rates: [f64; 4],
}

impl MpmSoilSettlingLimits {
    fn values(self) -> [f64; 10] {
        [
            self.speed,
            self.affine_speed,
            self.displacement_rate,
            self.acceleration,
            self.elastic_rate,
            self.plastic_rate,
            self.tire_history_rates[0],
            self.tire_history_rates[1],
            self.tire_history_rates[2],
            self.tire_history_rates[3],
        ]
    }

    fn accepts(self, a: &MpmSoilActivity) -> bool {
        let values = [
            a.max_speed,
            a.max_affine_speed,
            a.max_displacement_rate,
            a.max_acceleration,
            a.max_elastic_rate,
            a.max_plastic_rate,
            a.max_tire_history_rates[0],
            a.max_tire_history_rates[1],
            a.max_tire_history_rates[2],
            a.max_tire_history_rates[3],
        ];
        a.particles != 0
            && values
                .into_iter()
                .zip(self.values())
                .all(|(v, limit)| v.is_finite() && v >= 0.0 && v <= limit)
    }
}

/// Preparation limits. Each poll submits at most one batch of 1..=1024 substeps
/// or advances one capture/I/O stage. The wall limit stops new submissions; an
/// already-submitted boundary must drain before a terminal wall-limit result.
#[derive(Clone, Copy, Debug)]
pub struct MpmSoilSettlingConfig {
    pub dt: f32,
    pub substeps_per_observation: u32,
    pub max_substeps: u64,
    pub quiet_seconds: f64,
    pub quiet_observations: u32,
    pub wall_limit: Duration,
    pub limits: MpmSoilSettlingLimits,
}

impl MpmSoilSettlingConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.dt.is_finite()
            || self.dt <= 0.0
            || !(1..=1024).contains(&self.substeps_per_observation)
            || self.max_substeps == 0
            || !self.quiet_seconds.is_finite()
            || self.quiet_seconds <= 0.0
            || self.quiet_observations < 2
            || self.wall_limit.is_zero()
            || self
                .limits
                .values()
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(invalid("invalid soil settling policy"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpmSoilSettlingStatus {
    Running,
    DrainingAfterWallLimit,
    ActivityConverged,
    StepLimit,
    WallLimit,
    Failed,
}

impl MpmSoilSettlingStatus {
    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Running | Self::DrainingAfterWallLimit)
    }
}

/// Historical preparation progress. ActivityConverged means the configured
/// endpoint cutoffs held over the required observations/time; it does not certify
/// force equilibrium, physical calibration or permission to sleep/evict soil.
#[derive(Clone, Debug)]
pub struct MpmSoilSettlingProgress {
    pub status: MpmSoilSettlingStatus,
    pub submitted_substeps: u64,
    pub observed_seconds: f64,
    pub quiet_seconds: f64,
    pub quiet_observations: u32,
    pub observations: u64,
    pub wall_seconds: f64,
    pub latest: Option<MpmSoilActivityReport>,
    pub failure: Option<String>,
}

enum World<'a> {
    Memory(&'a mut MpmSoilWorld),
    Disk(&'a mut MpmSoilDiskWorld),
}

/// Exclusively borrows a ready full-population world at a committed boundary.
/// Sparse swept-horizon requests reject. Preparation replaces region claims with
/// occupancy pins and leaves activity monitoring enabled. Dropping releases the
/// borrow without canceling pending capture/I/O; finish that boundary before
/// ordinary stepping. It never discards active material or silently saves on drop.
pub struct MpmSoilSettling<'a> {
    world: World<'a>,
    config: MpmSoilSettlingConfig,
    progress: MpmSoilSettlingProgress,
    started: Instant,
    pending_steps: u32,
    particles: u32,
    ids_hash: [u8; 32],
}

impl<'a> MpmSoilSettling<'a> {
    pub fn memory(world: &'a mut MpmSoilWorld, config: MpmSoilSettlingConfig) -> Result<Self> {
        config.validate()?;
        world.ready()?;
        Self::validate_world(
            world.active_particle_count(),
            world.total_particle_count(),
            world.remaining_region_substeps(),
            world.activity_clock.is_empty(),
        )?;
        world.set_activity_monitoring(true)?;
        Ok(Self::new(World::Memory(world), config))
    }

    pub fn disk(world: &'a mut MpmSoilDiskWorld, config: MpmSoilSettlingConfig) -> Result<Self> {
        config.validate()?;
        world.ready()?;
        Self::validate_world(
            world.active_particle_count(),
            world.total_particle_count(),
            world.remaining_region_substeps(),
            world.activity_clock.is_empty(),
        )?;
        world.set_activity_monitoring(true)?;
        Ok(Self::new(World::Disk(world), config))
    }

    fn validate_world(
        active: u32,
        total: u32,
        horizon: Option<u32>,
        at_boundary: bool,
    ) -> Result<()> {
        if active == 0 || active != total || horizon.is_some() || !at_boundary {
            return Err(invalid(
                "settling requires nonempty full soil residency at a raw committed boundary",
            ));
        }
        Ok(())
    }

    fn new(world: World<'a>, config: MpmSoilSettlingConfig) -> Self {
        let ids = match &world {
            World::Memory(w) => w.identity().particle_ids(),
            World::Disk(w) => w.identity().particle_ids(),
        };
        let mut hash = blake3::Hasher::new();
        for id in ids {
            hash.update(&id.0.to_le_bytes());
        }
        let particles = ids.len() as u32;
        Self {
            world,
            config,
            started: Instant::now(),
            pending_steps: 0,
            particles,
            ids_hash: *hash.finalize().as_bytes(),
            progress: MpmSoilSettlingProgress {
                status: MpmSoilSettlingStatus::Running,
                submitted_substeps: 0,
                observed_seconds: 0.0,
                quiet_seconds: 0.0,
                quiet_observations: 0,
                observations: 0,
                wall_seconds: 0.0,
                latest: None,
                failure: None,
            },
        }
    }

    pub fn progress(&self) -> &MpmSoilSettlingProgress {
        &self.progress
    }
    pub fn config(&self) -> MpmSoilSettlingConfig {
        self.config
    }

    /// Advances without a GPU or worker wait. A timeout never reports convergence.
    /// On error the world remains borrowed and recoverable; progress records Failed.
    pub fn poll(&mut self, engine: &Engine) -> Result<MpmSoilSettlingStatus> {
        if self.progress.status.is_terminal() {
            return Ok(self.progress.status);
        }
        let result = self.advance(engine);
        self.progress.wall_seconds = self.started.elapsed().as_secs_f64();
        if let Err(error) = &result {
            self.progress.status = MpmSoilSettlingStatus::Failed;
            self.progress.failure = Some(error.to_string());
        }
        result.map(|()| self.progress.status)
    }

    fn advance(&mut self, engine: &Engine) -> Result<()> {
        if self.pending_steps != 0 {
            let complete = match &mut self.world {
                World::Memory(w) => w.poll_region_update(engine)?,
                World::Disk(w) => w.poll_region_update(engine)?,
            };
            if !complete {
                if self.started.elapsed() >= self.config.wall_limit {
                    self.progress.status = MpmSoilSettlingStatus::DrainingAfterWallLimit;
                }
                return Ok(());
            }
            let report = match &self.world {
                World::Memory(w) => w.last_activity_report(),
                World::Disk(w) => w.last_activity_report(),
            }
            .ok_or_else(|| invalid("settling boundary lost its activity report"))?
            .clone();
            if report.activity.particles != self.particles
                || report.activity.particle_ids_hash != self.ids_hash
                || report.activity.substeps != u64::from(self.pending_steps)
                || report.activity.seconds
                    != f64::from(self.config.dt) * f64::from(self.pending_steps)
            {
                return Err(invalid("settling population or physical interval changed"));
            }
            self.pending_steps = 0;
            self.observe(report);
            return Ok(());
        }
        if self.started.elapsed() >= self.config.wall_limit {
            self.progress.status = MpmSoilSettlingStatus::WallLimit;
            return Ok(());
        }
        let count = u64::from(self.config.substeps_per_observation)
            .min(self.config.max_substeps - self.progress.submitted_substeps)
            as u32;
        match &mut self.world {
            World::Memory(w) => w.step_n(engine, self.config.dt, count as usize)?,
            World::Disk(w) => w.step_n(engine, self.config.dt, count as usize)?,
        }
        self.progress.submitted_substeps += u64::from(count);
        match &mut self.world {
            World::Memory(w) => w.begin_region_update(&[])?,
            World::Disk(w) => w.begin_region_update(&[])?,
        }
        self.pending_steps = count;
        Ok(())
    }

    fn observe(&mut self, report: MpmSoilActivityReport) {
        let progress = &mut self.progress;
        progress.observations += 1;
        progress.observed_seconds += report.activity.seconds;
        if self.config.limits.accepts(&report.activity) {
            progress.quiet_seconds += report.activity.seconds;
            progress.quiet_observations = progress.quiet_observations.saturating_add(1);
        } else {
            progress.quiet_seconds = 0.0;
            progress.quiet_observations = 0;
        }
        progress.latest = Some(report);
        progress.status = if self.started.elapsed() >= self.config.wall_limit {
            MpmSoilSettlingStatus::WallLimit
        } else if progress.quiet_seconds >= self.config.quiet_seconds
            && progress.quiet_observations >= self.config.quiet_observations
        {
            MpmSoilSettlingStatus::ActivityConverged
        } else if progress.submitted_substeps >= self.config.max_substeps {
            MpmSoilSettlingStatus::StepLimit
        } else {
            MpmSoilSettlingStatus::Running
        };
    }
}

#[cfg(test)]
#[path = "mpm_soil_settling_tests.rs"]
mod tests;
