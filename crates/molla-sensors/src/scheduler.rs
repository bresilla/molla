//! Bounded sensor scheduling from caller-supplied simulation timestamps.

use molla_core::{Error, Result};
use std::{sync::Arc, time::Duration};

const MAX_SENSORS: usize = 4096;

/// Nanosecond-resolution cadence; the first deadline is epoch + phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SensorSchedule {
    pub sensor_id: u64,
    pub period: Duration,
    pub phase: Duration,
}

/// One current-state sample request. Earlier missed deadlines are not replayed.
#[derive(Debug, Clone)]
pub struct ScheduledSample {
    pub sensor_id: u64,
    pub sample_id: u64,
    pub generation: u64,
    pub scene_revision: u64,
    pub period: Duration,
    pub scheduled_time: Duration,
    pub simulation_time: Duration,
    /// Time since the previous issued request, not a dispatch acknowledgement.
    pub elapsed: Option<Duration>,
    /// Missed cadence deadlines; excludes caller-side dispatch/readback failures.
    pub skipped_samples: u64,
    owner: Arc<()>,
    slot: usize,
    next_index: u64,
}

impl ScheduledSample {
    /// Collection metadata; exact integer timestamps remain in this sample.
    #[cfg(feature = "gpu")]
    pub fn readback_stamp(&self) -> crate::readback::ReadbackStamp {
        crate::readback::ReadbackStamp {
            sensor_id: self.sensor_id,
            sample_id: self.sample_id,
            scene_revision: self.scene_revision,
            simulation_time: Some(self.simulation_time.as_secs_f64()),
        }
    }

    #[cfg(feature = "gpu")]
    pub fn matches_readback(&self, frame: &crate::readback::ReadbackFrame) -> bool {
        frame.stamp == self.readback_stamp()
    }
}

struct Entry {
    schedule: SensorSchedule,
    anchor: Duration,
    next_index: u64,
    last_sample: Option<Duration>,
}

/// Call after every physics snapshot, not only at render time. Each call emits
/// at most one request per sensor, in configuration order, using retained storage.
/// Requests advance cadence when issued; dispatch/collection failures are not retried.
pub struct SensorScheduler {
    entries: Vec<Entry>,
    due: Vec<ScheduledSample>,
    owner: Arc<()>,
    generation: u64,
    scene_revision: u64,
    sample_id: u64,
    last_time: Duration,
    paused: bool,
}

impl SensorScheduler {
    pub fn new(epoch: Duration, scene_revision: u64, schedules: &[SensorSchedule]) -> Result<Self> {
        let entries = build_entries(epoch, schedules)?;
        Ok(Self {
            entries,
            due: Vec::with_capacity(schedules.len()),
            owner: Arc::new(()),
            generation: 0,
            scene_revision,
            sample_id: 0,
            last_time: epoch,
            paused: false,
        })
    }

    /// A late call samples now, reports missed deadlines and preserves cadence.
    /// Repeated timestamps emit nothing; backward time requires an explicit reset.
    pub fn advance(&mut self, simulation_time: Duration) -> Result<&[ScheduledSample]> {
        if simulation_time < self.last_time {
            return Err(Error::Solver(
                "sensor scheduler: time moved backward; reset required".into(),
            ));
        }
        self.due.clear();
        if !self.paused {
            for (slot, entry) in self.entries.iter().enumerate() {
                let Some(since_anchor) = simulation_time.checked_sub(entry.anchor) else {
                    continue;
                };
                let period = entry.schedule.period.as_nanos();
                let index = u64::try_from(since_anchor.as_nanos() / period).map_err(|_| {
                    Error::Solver("sensor scheduler: cadence index exhausted".into())
                })?;
                if index < entry.next_index {
                    continue;
                }
                let next_index = index.checked_add(1).ok_or_else(|| {
                    Error::Solver("sensor scheduler: cadence index exhausted".into())
                })?;
                let sample_id = self
                    .sample_id
                    .checked_add(self.due.len() as u64 + 1)
                    .ok_or_else(|| {
                        Error::Solver("sensor scheduler: sample identity exhausted".into())
                    })?;
                let late = duration_from_nanos(since_anchor.as_nanos() % period);
                self.due.push(ScheduledSample {
                    sensor_id: entry.schedule.sensor_id,
                    sample_id,
                    generation: self.generation,
                    scene_revision: self.scene_revision,
                    period: entry.schedule.period,
                    scheduled_time: simulation_time - late,
                    simulation_time,
                    elapsed: entry.last_sample.map(|previous| simulation_time - previous),
                    skipped_samples: index - entry.next_index,
                    owner: self.owner.clone(),
                    slot,
                    next_index,
                });
            }
        }
        for sample in &self.due {
            let entry = &mut self.entries[sample.slot];
            entry.next_index = sample.next_index;
            entry.last_sample = Some(simulation_time);
        }
        self.sample_id += self.due.len() as u64;
        self.last_time = simulation_time;
        Ok(&self.due)
    }

    /// Suppress delivery without shifting deadlines. Resuming reports any skips.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Reset cadence and elapsed times; invalidate previous-generation samples.
    /// Sample IDs remain monotonic and the paused state is preserved.
    pub fn reset(&mut self, epoch: Duration, scene_revision: u64) -> Result<()> {
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::Solver("sensor scheduler: generation exhausted".into()))?;
        for entry in &self.entries {
            check_anchor(epoch, entry.schedule.phase)?;
        }
        for entry in &mut self.entries {
            entry.anchor = epoch + entry.schedule.phase;
            entry.next_index = 0;
            entry.last_sample = None;
        }
        self.generation = generation;
        self.scene_revision = scene_revision;
        self.last_time = epoch;
        self.due.clear();
        Ok(())
    }

    /// Replace the fixed schedule set and reset its epoch atomically on success.
    pub fn replace(
        &mut self,
        epoch: Duration,
        scene_revision: u64,
        schedules: &[SensorSchedule],
    ) -> Result<()> {
        let entries = build_entries(epoch, schedules)?;
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::Solver("sensor scheduler: generation exhausted".into()))?;
        self.entries = entries;
        self.due = Vec::with_capacity(schedules.len());
        self.generation = generation;
        self.scene_revision = scene_revision;
        self.last_time = epoch;
        Ok(())
    }

    /// Reject pending samples from other schedulers or pre-reset/replaced scenes.
    pub fn is_current(&self, sample: &ScheduledSample) -> bool {
        Arc::ptr_eq(&self.owner, &sample.owner)
            && sample.generation == self.generation
            && sample.scene_revision == self.scene_revision
            && self
                .entries
                .get(sample.slot)
                .is_some_and(|entry| entry.schedule.sensor_id == sample.sensor_id)
    }
}

fn build_entries(epoch: Duration, schedules: &[SensorSchedule]) -> Result<Vec<Entry>> {
    if schedules.len() > MAX_SENSORS {
        return Err(Error::Build(
            "sensor scheduler: at most 4096 sensors supported".into(),
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(schedules.len());
    let mut entries = Vec::with_capacity(schedules.len());
    for &schedule in schedules {
        if schedule.period.is_zero() || !seen.insert(schedule.sensor_id) {
            return Err(Error::Build(
                "sensor scheduler: zero period or duplicate sensor ID".into(),
            ));
        }
        entries.push(Entry {
            schedule,
            anchor: check_anchor(epoch, schedule.phase)?,
            next_index: 0,
            last_sample: None,
        });
    }
    Ok(entries)
}

fn check_anchor(epoch: Duration, phase: Duration) -> Result<Duration> {
    epoch
        .checked_add(phase)
        .ok_or_else(|| Error::Build("sensor scheduler: epoch + phase overflow".into()))
}

fn duration_from_nanos(nanos: u128) -> Duration {
    Duration::new(
        (nanos / 1_000_000_000) as u64,
        (nanos % 1_000_000_000) as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule(id: u64, period_ms: u64, phase_ms: u64) -> SensorSchedule {
        SensorSchedule {
            sensor_id: id,
            period: Duration::from_millis(period_ms),
            phase: Duration::from_millis(phase_ms),
        }
    }

    #[test]
    fn independent_rates_phase_and_integer_boundaries() {
        let mut scheduler =
            SensorScheduler::new(Duration::ZERO, 7, &[schedule(8, 10, 0), schedule(2, 25, 5)])
                .unwrap();
        let first = scheduler.advance(Duration::ZERO).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].sensor_id, 8);
        assert_eq!(first[0].elapsed, None);
        assert!(scheduler.advance(Duration::ZERO).unwrap().is_empty());
        assert!(
            scheduler
                .advance(Duration::from_nanos(4_999_999))
                .unwrap()
                .is_empty()
        );
        let second = scheduler.advance(Duration::from_millis(5)).unwrap();
        assert_eq!(second[0].sensor_id, 2);
        let both = scheduler.advance(Duration::from_millis(30)).unwrap();
        assert_eq!(both.len(), 2);
        assert_eq!(both[0].sensor_id, 8);
        assert_eq!(both[0].skipped_samples, 2);
        assert_eq!(both[0].elapsed, Some(Duration::from_millis(30)));
        assert_eq!(both[1].skipped_samples, 0);
        assert_eq!(both[1].elapsed, Some(Duration::from_millis(25)));
        for sample in both {
            assert_eq!(sample.scheduled_time, Duration::from_millis(30));
        }
    }

    #[test]
    fn late_samples_are_current_state_not_fabricated_history() {
        let mut scheduler = SensorScheduler::new(Duration::ZERO, 7, &[schedule(0, 10, 0)]).unwrap();
        let first = scheduler.advance(Duration::from_millis(103)).unwrap()[0].clone();
        assert_eq!(first.scheduled_time, Duration::from_millis(100));
        assert_eq!(first.simulation_time, Duration::from_millis(103));
        assert_eq!(first.skipped_samples, 10);
        assert_eq!(first.elapsed, None);
        assert!(
            scheduler
                .advance(Duration::from_millis(109))
                .unwrap()
                .is_empty()
        );
        let next = &scheduler.advance(Duration::from_millis(110)).unwrap()[0];
        assert_eq!(next.elapsed, Some(Duration::from_millis(7)));
        assert_eq!(next.skipped_samples, 0);
    }

    #[test]
    fn pause_reset_replacement_and_foreign_samples() {
        let config = [schedule(0, 10, 0)];
        let mut scheduler = SensorScheduler::new(Duration::ZERO, 7, &config).unwrap();
        let old = scheduler.advance(Duration::ZERO).unwrap()[0].clone();
        scheduler.set_paused(true);
        assert!(
            scheduler
                .advance(Duration::from_millis(50))
                .unwrap()
                .is_empty()
        );
        scheduler.set_paused(false);
        let resumed = scheduler.advance(Duration::from_millis(50)).unwrap()[0].clone();
        assert_eq!(resumed.skipped_samples, 4);
        assert_eq!(resumed.elapsed, Some(Duration::from_millis(50)));
        assert!(scheduler.advance(Duration::ZERO).is_err());
        scheduler.reset(Duration::ZERO, 8).unwrap();
        assert!(!scheduler.is_current(&old));
        let new = scheduler.advance(Duration::ZERO).unwrap()[0].clone();
        assert!(scheduler.is_current(&new));
        assert!(new.sample_id > resumed.sample_id);
        assert_eq!(new.elapsed, None);
        let other = SensorScheduler::new(Duration::ZERO, 8, &config).unwrap();
        assert!(!other.is_current(&new));
        scheduler
            .replace(Duration::ZERO, 8, &[schedule(2, 3, 1)])
            .unwrap();
        assert!(!scheduler.is_current(&new));
        assert_eq!(
            scheduler.advance(Duration::from_millis(1)).unwrap()[0].sensor_id,
            2
        );
        scheduler.set_paused(true);
        scheduler.reset(Duration::ZERO, 8).unwrap();
        assert!(scheduler.is_paused());
    }

    #[test]
    fn render_frequency_does_not_change_physics_sample_requests() {
        fn run(render_every: u64) -> Vec<(u64, Duration, Duration)> {
            let mut scheduler =
                SensorScheduler::new(Duration::ZERO, 0, &[schedule(0, 10, 0), schedule(1, 25, 5)])
                    .unwrap();
            let mut samples = Vec::new();
            for tick in 0..=100 {
                let time = Duration::from_millis(tick);
                samples.extend(
                    scheduler
                        .advance(time)
                        .unwrap()
                        .iter()
                        .map(|s| (s.sensor_id, s.simulation_time, s.scheduled_time)),
                );
                if tick % render_every == 0 {
                    assert!(scheduler.advance(time).unwrap().is_empty());
                }
            }
            samples
        }
        assert_eq!(run(1), run(7));
        assert_eq!(run(7), run(33));
    }

    #[test]
    fn empty_invalid_overflow_and_atomic_errors() {
        let mut empty = SensorScheduler::new(Duration::ZERO, 0, &[]).unwrap();
        assert!(empty.advance(Duration::MAX).unwrap().is_empty());
        assert!(SensorScheduler::new(Duration::ZERO, 0, &[schedule(0, 0, 0)]).is_err());
        assert!(SensorScheduler::new(Duration::ZERO, 0, &[schedule(0, 1, 0); 2]).is_err());
        assert!(SensorScheduler::new(Duration::MAX, 0, &[schedule(0, 1, 1)]).is_err());
        assert!(
            SensorScheduler::new(Duration::ZERO, 0, &vec![schedule(0, 1, 0); MAX_SENSORS + 1])
                .is_err()
        );
        let mut scheduler = SensorScheduler::new(
            Duration::ZERO,
            0,
            &[
                SensorSchedule {
                    period: Duration::from_secs(1 << 40),
                    ..schedule(0, 1, 1)
                },
                SensorSchedule {
                    period: Duration::from_nanos(1),
                    ..schedule(1, 1, 0)
                },
            ],
        )
        .unwrap();
        assert!(scheduler.advance(Duration::MAX).is_err());
        assert!(scheduler.reset(Duration::MAX, 1).is_err());
        assert!(
            scheduler
                .replace(Duration::ZERO, 1, &[schedule(0, 0, 0)])
                .is_err()
        );
        assert_eq!(scheduler.generation(), 0);
        let first = scheduler.advance(Duration::from_millis(1)).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].sample_id, 1);
        assert_eq!(first[0].elapsed, None);
        assert_eq!(first[1].sample_id, 2);
    }

    #[test]
    fn large_epoch_preserves_nanosecond_deadlines_and_retained_batch() {
        let epoch = Duration::from_secs(1 << 40);
        let mut scheduler = SensorScheduler::new(
            epoch,
            0,
            &[SensorSchedule {
                sensor_id: 0,
                period: Duration::from_nanos(3),
                phase: Duration::from_nanos(1),
            }],
        )
        .unwrap();
        let pointer = scheduler.due.as_ptr();
        assert!(scheduler.advance(epoch).unwrap().is_empty());
        for tick in 0..100 {
            let time = epoch + Duration::from_nanos(1 + tick * 3);
            let due = scheduler.advance(time).unwrap();
            assert_eq!(due[0].scheduled_time, time);
            assert_eq!(due[0].skipped_samples, 0);
            assert_eq!(due.as_ptr(), pointer);
        }
    }

    #[test]
    fn sample_and_generation_exhaustion_leave_cursors_unchanged() {
        let mut scheduler =
            SensorScheduler::new(Duration::ZERO, 0, &[schedule(0, 10, 0), schedule(1, 10, 0)])
                .unwrap();
        scheduler.sample_id = u64::MAX - 1;
        assert!(scheduler.advance(Duration::ZERO).is_err());
        assert_eq!(scheduler.sample_id, u64::MAX - 1);
        assert!(
            scheduler
                .entries
                .iter()
                .all(|entry| entry.next_index == 0 && entry.last_sample.is_none())
        );
        scheduler
            .replace(Duration::ZERO, 0, &[schedule(0, 10, 0)])
            .unwrap();
        assert_eq!(
            scheduler.advance(Duration::ZERO).unwrap()[0].sample_id,
            u64::MAX
        );
        assert!(scheduler.advance(Duration::from_millis(10)).is_err());
        assert_eq!(scheduler.entries[0].next_index, 1);
        scheduler.generation = u64::MAX;
        assert!(scheduler.reset(Duration::ZERO, 9).is_err());
        assert!(scheduler.replace(Duration::ZERO, 9, &[]).is_err());
        assert_eq!(scheduler.entries.len(), 1);
        assert_eq!(scheduler.scene_revision, 0);
    }
}
