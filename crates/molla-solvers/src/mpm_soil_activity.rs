//! Endpoint activity measurements at completed soil checkpoint boundaries.

use super::*;
use molla_vehicle::{SoilParticleId, SoilTileCoord, SoilTileLayout};
use std::collections::BTreeMap;

/// Endpoint measurements over an explicitly identified particle population.
/// These are not force residuals or a sleep certificate. Interior transients can
/// be missed when several substeps separate observations. Maxima retain isolated
/// moving particles; RMS speed and energy describe the current population.
#[derive(Clone, Debug, PartialEq)]
pub struct MpmSoilActivity {
    pub particles: u32,
    pub substeps: u64,
    pub seconds: f64,
    pub particle_ids_hash: [u8; 32],
    /// Maximum endpoint speed, in m/s, across both snapshots.
    pub max_speed: f64,
    /// Maximum endpoint Frobenius norm of APIC C times cell size, in m/s.
    pub max_affine_speed: f64,
    /// Current mass-weighted RMS particle speed, in m/s.
    pub rms_speed: f64,
    /// Translational particle kinetic energy in J; excludes APIC affine energy.
    pub kinetic_energy: f64,
    /// Maximum endpoint displacement divided by elapsed seconds, in m/s.
    pub max_displacement_rate: f64,
    /// Maximum endpoint velocity change divided by elapsed seconds, in m/s².
    pub max_acceleration: f64,
    /// Maximum Frobenius norm of elastic F change per second.
    pub max_elastic_rate: f64,
    /// Maximum plastic-history change per second: sand flow or snow log(det Fp).
    pub max_plastic_rate: f64,
    /// Componentwise absolute rates: plastic compaction, normal impulse,
    /// shear impulse and active weight, in 1/s, N, N and 1/s.
    pub max_tire_history_rates: [f64; 4],
}

/// Historical measurement ending at a committed region request ordinal.
/// Imported particles at that same transition are not part of the measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct MpmSoilActivityReport {
    pub revision: u64,
    pub activity: MpmSoilActivity,
    pub tile_layout: SoilTileLayout,
    pub tiles: Vec<MpmSoilTileActivity>,
}

/// Activity of the union of IDs occupying a tile at either endpoint, sorted by
/// tile coordinate in reports. A migrating ID contributes once to each endpoint
/// tile; energy and particle counts are therefore not additive across tiles.
/// Intermediate crossed tiles and interior transients are not observed.
/// Neither unchanged membership nor quiet endpoints certify physical sleep.
#[derive(Clone, Debug, PartialEq)]
pub struct MpmSoilTileActivity {
    pub tile: SoilTileCoord,
    pub previous_particles: u32,
    pub current_particles: u32,
    pub entered_particles: u32,
    pub exited_particles: u32,
    pub activity: MpmSoilActivity,
}

impl MpmSoilCheckpoint {
    /// Measures endpoints with the same strictly increasing global particle IDs,
    /// configuration, material table, material assignments, mass and volume.
    /// Elapsed physical time and substep count are caller-provided provenance.
    /// Allocates no per-particle scratch and performs no GPU work or readback.
    pub fn activity_since(
        &self,
        previous: &Self,
        previous_ids: &[SoilParticleId],
        current_ids: &[SoilParticleId],
        seconds: f64,
        substeps: u64,
    ) -> Result<MpmSoilActivity> {
        self.measure_activity(
            previous,
            previous_ids,
            current_ids,
            seconds,
            substeps,
            None,
            None,
        )
        .map(|(activity, _)| activity)
    }

    /// Measures population and endpoint-tile unions in one particle pass using
    /// the same validation and provenance as `activity_since`. Uses O(T) scratch
    /// and O(N log T) work, with at most 2N occupied endpoint tiles. No GPU work,
    /// checkpoint copies, swept-path coverage or sleep classification occurs.
    pub fn activity_by_tile_since(
        &self,
        previous: &Self,
        previous_ids: &[SoilParticleId],
        current_ids: &[SoilParticleId],
        seconds: f64,
        substeps: u64,
        layout: SoilTileLayout,
    ) -> Result<(MpmSoilActivity, Vec<MpmSoilTileActivity>)> {
        self.measure_activity(
            previous,
            previous_ids,
            current_ids,
            seconds,
            substeps,
            Some(layout),
            None,
        )
    }

    fn measure_activity(
        &self,
        previous: &Self,
        previous_ids: &[SoilParticleId],
        current_ids: &[SoilParticleId],
        seconds: f64,
        substeps: u64,
        layout: Option<SoilTileLayout>,
        origin: Option<[f64; 3]>,
    ) -> Result<(MpmSoilActivity, Vec<MpmSoilTileActivity>)> {
        self.validate()?;
        previous.validate()?;
        let mut compatible = self.cfg;
        compatible.max_particles = previous.cfg.max_particles;
        compatible.max_rigid = previous.cfg.max_rigid;
        if !seconds.is_finite()
            || seconds <= 0.0
            || substeps == 0
            || self.count != previous.count
            || self.grid_window_cells != previous.grid_window_cells
            || previous_ids.len() != self.count as usize
            || previous_ids != current_ids
            || previous_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || config_words(compatible) != config_words(previous.cfg)
            || self.materials != previous.materials
            || [4, 5, 7]
                .iter()
                .any(|&s| self.stream(s) != previous.stream(s))
        {
            return Err(invalid(
                "incompatible soil activity endpoints or time interval",
            ));
        }
        let mut total = ActivityAccumulator::default();
        let mut tiles = BTreeMap::<SoilTileCoord, TileAccumulator>::new();
        let before: [&[u32]; 9] = std::array::from_fn(|s| previous.stream(s));
        let after: [&[u32]; 9] = std::array::from_fn(|s| self.stream(s));
        for (i, &id) in current_ids.iter().enumerate() {
            let value = |streams: &[&[u32]; 9], stream: usize, component: usize| {
                f64::from(f32::from_bits(
                    streams[stream][i * STRIDES[stream] + component],
                ))
            };
            let norm = |stream: usize, delta: bool, old: bool| {
                let components = if stream < 2 { 3 } else { 9 };
                (0..components)
                    .map(|k| {
                        let k = if stream < 2 { k } else { k / 3 * 4 + k % 3 };
                        let x = if delta {
                            value(&after, stream, k) - value(&before, stream, k)
                        } else {
                            value(if old { &before } else { &after }, stream, k)
                        };
                        x * x
                    })
                    .sum::<f64>()
                    .sqrt()
            };
            let speed = norm(1, false, false);
            let measurement = ParticleActivity {
                maxima: [
                    speed.max(norm(1, false, true)),
                    norm(2, false, false).max(norm(2, false, true)) * f64::from(self.cfg.cell_size),
                    norm(0, true, false) / seconds,
                    norm(1, true, false) / seconds,
                    norm(3, true, false) / seconds,
                    (value(&after, 6, 0) - value(&before, 6, 0)).abs() / seconds,
                    (value(&after, 8, 0) - value(&before, 8, 0)).abs() / seconds,
                    (value(&after, 8, 1) - value(&before, 8, 1)).abs() / seconds,
                    (value(&after, 8, 2) - value(&before, 8, 2)).abs() / seconds,
                    (value(&after, 8, 3) - value(&before, 8, 3)).abs() / seconds,
                ],
                mass: value(&after, 5, 0),
                speed_squared_mass: value(&after, 5, 0) * speed * speed,
            };
            total.add(id, &measurement);
            if let Some(layout) = layout {
                let tile_at = |streams| {
                    let point = std::array::from_fn(|axis| value(streams, 0, axis));
                    match origin {
                        Some(origin) => layout.tile_at_local(point, origin),
                        None => layout.tile_at(point),
                    }
                    .map_err(|e| invalid(&e.to_string()))
                };
                let old = tile_at(&before)?;
                let new = tile_at(&after)?;
                let source = tiles.entry(old).or_default();
                source.activity.add(id, &measurement);
                source.previous += 1;
                if old == new {
                    source.current += 1;
                } else {
                    source.exited += 1;
                    let destination = tiles.entry(new).or_default();
                    destination.activity.add(id, &measurement);
                    destination.current += 1;
                    destination.entered += 1;
                }
            }
        }
        let activity = total.finish(seconds, substeps)?;
        let tiles = tiles
            .into_iter()
            .map(|(tile, entry)| {
                Ok(MpmSoilTileActivity {
                    tile,
                    previous_particles: entry.previous,
                    current_particles: entry.current,
                    entered_particles: entry.entered,
                    exited_particles: entry.exited,
                    activity: entry.activity.finish(seconds, substeps)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok((activity, tiles))
    }
}

struct ParticleActivity {
    maxima: [f64; 10],
    mass: f64,
    speed_squared_mass: f64,
}

#[derive(Default)]
struct ActivityAccumulator {
    maxima: [f64; 10],
    mass: f64,
    speed_squared_sum: f64,
    particles: u32,
    hash: blake3::Hasher,
}

impl ActivityAccumulator {
    fn add(&mut self, id: SoilParticleId, particle: &ParticleActivity) {
        self.hash.update(&id.0.to_le_bytes());
        self.particles += 1;
        self.mass += particle.mass;
        self.speed_squared_sum += particle.speed_squared_mass;
        for (max, value) in self.maxima.iter_mut().zip(particle.maxima) {
            *max = max.max(value);
        }
    }

    fn finish(self, seconds: f64, substeps: u64) -> Result<MpmSoilActivity> {
        let result = MpmSoilActivity {
            particles: self.particles,
            substeps,
            seconds,
            particle_ids_hash: *self.hash.finalize().as_bytes(),
            max_speed: self.maxima[0],
            max_affine_speed: self.maxima[1],
            rms_speed: if self.mass == 0.0 {
                0.0
            } else {
                (self.speed_squared_sum / self.mass).sqrt()
            },
            kinetic_energy: self.speed_squared_sum * 0.5,
            max_displacement_rate: self.maxima[2],
            max_acceleration: self.maxima[3],
            max_elastic_rate: self.maxima[4],
            max_plastic_rate: self.maxima[5],
            max_tire_history_rates: self.maxima[6..].try_into().unwrap(),
        };
        if [
            result.max_speed,
            result.max_affine_speed,
            result.rms_speed,
            result.kinetic_energy,
            result.max_displacement_rate,
            result.max_acceleration,
            result.max_elastic_rate,
            result.max_plastic_rate,
        ]
        .into_iter()
        .chain(result.max_tire_history_rates)
        .any(|v| !v.is_finite())
        {
            return Err(invalid("soil activity measurement overflow"));
        }
        Ok(result)
    }
}

#[derive(Default)]
struct TileAccumulator {
    activity: ActivityAccumulator,
    previous: u32,
    current: u32,
    entered: u32,
    exited: u32,
}

#[derive(Clone, Copy, Default)]
pub(super) struct ActivityClock {
    seconds: f64,
    substeps: u64,
}

impl ActivityClock {
    pub(super) fn is_empty(&self) -> bool {
        self.substeps == 0
    }

    pub(super) fn after_steps(&self, dt: f32, substeps: usize) -> Result<Self> {
        let count = u64::try_from(substeps).map_err(|_| invalid("activity substep overflow"))?;
        let substeps = self
            .substeps
            .checked_add(count)
            .ok_or_else(|| invalid("activity substep overflow"))?;
        let seconds = self.seconds + f64::from(dt) * count as f64;
        if !seconds.is_finite() || (count != 0 && seconds <= self.seconds) {
            return Err(invalid(
                "activity time overflow or loss of clock resolution",
            ));
        }
        Ok(Self { seconds, substeps })
    }

    pub(super) fn report(
        &self,
        enabled: bool,
        previous: &MpmSoilSelection,
        current: &MpmSoilCheckpoint,
        revision: u64,
        tile_layout: SoilTileLayout,
    ) -> Result<Option<MpmSoilActivityReport>> {
        self.report_in_frame(enabled, previous, current, revision, tile_layout, None)
    }

    pub(super) fn report_in_frame(
        &self,
        enabled: bool,
        previous: &MpmSoilSelection,
        current: &MpmSoilCheckpoint,
        revision: u64,
        tile_layout: SoilTileLayout,
        origin: Option<[f64; 3]>,
    ) -> Result<Option<MpmSoilActivityReport>> {
        if !enabled || self.substeps == 0 {
            return Ok(None);
        }
        let ids = previous.particle_ids();
        let (activity, tiles) = current.measure_activity(
            previous.checkpoint(),
            ids,
            ids,
            self.seconds,
            self.substeps,
            Some(tile_layout),
            origin,
        )?;
        Ok(Some(MpmSoilActivityReport {
            revision,
            activity,
            tile_layout,
            tiles,
        }))
    }
}

#[cfg(test)]
#[path = "mpm_soil_activity_tests.rs"]
mod tests;
