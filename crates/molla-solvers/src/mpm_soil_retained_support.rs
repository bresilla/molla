//! Stable-ID retention seeds for conservative stencil connectivity.

use super::*;
use molla_vehicle::SoilParticleId;
use std::collections::BTreeSet;

/// Physical-slot membership in a component containing nonquiet or requested soil.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmSoilRetainedSupportParticle {
    pub quiet: bool,
    pub requested: bool,
    pub retain: bool,
}

/// Immutable converged dependencies for explicit stable-ID retention requests.
/// This is not force-equilibrium, wake-coverage, sleep or eviction authorization.
#[derive(Clone, Debug)]
pub struct MpmSoilRetainedSupportSnapshot {
    quiescence: MpmSoilQuiescenceSnapshot,
    minimum_quiet_steps: u32,
    rounds: u32,
    requested_particles: Vec<SoilParticleId>,
    particles: Vec<MpmSoilRetainedSupportParticle>,
}

impl MpmSoilRetainedSupportSnapshot {
    pub fn quiescence(&self) -> &MpmSoilQuiescenceSnapshot {
        &self.quiescence
    }
    pub fn minimum_quiet_steps(&self) -> u32 {
        self.minimum_quiet_steps
    }
    pub fn rounds(&self) -> u32 {
        self.rounds
    }
    pub fn requested_particles(&self) -> &[SoilParticleId] {
        &self.requested_particles
    }
    pub fn particles(&self) -> &[MpmSoilRetainedSupportParticle] {
        &self.particles
    }
}

fn seed_mask(ids: &[SoilParticleId], requested: &[SoilParticleId]) -> Result<Vec<u32>> {
    if requested.len() > ids.len() {
        return Err(Error::Solver(
            "soil support seed count exceeds live particles".into(),
        ));
    }
    let unique: BTreeSet<_> = requested.iter().copied().collect();
    if unique.len() != requested.len() {
        return Err(Error::Solver(
            "duplicate soil support retention seed".into(),
        ));
    }
    let mut mask = Vec::new();
    mask.try_reserve_exact(ids.len())
        .map_err(|_| Error::Solver("soil support seed allocation failed".into()))?;
    mask.extend(ids.iter().map(|id| u32::from(unique.contains(id))));
    if mask.iter().filter(|&&v| v != 0).count() != unique.len() {
        return Err(Error::Solver("unknown soil support retention seed".into()));
    }
    Ok(mask)
}

fn decode(words: &[u32]) -> Result<Vec<MpmSoilRetainedSupportParticle>> {
    if words.iter().any(|&word| word & 128 != 0) {
        return Err(Error::Solver(
            "soil retained support did not converge within round budget".into(),
        ));
    }
    words
        .iter()
        .map(|&word| {
            if !matches!(word, 1 | 65 | 66 | 97 | 98) {
                return Err(Error::Solver(
                    "soil retained support has invalid or clipped transfer stencil".into(),
                ));
            }
            Ok(MpmSoilRetainedSupportParticle {
                quiet: word & 1 != 0,
                requested: word & 32 != 0,
                retain: word & 64 != 0,
            })
        })
        .collect()
}

impl MpmSimulation {
    fn retained_support_seeds(
        &self,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
        requested: &[SoilParticleId],
    ) -> Result<Option<Vec<u32>>> {
        self.validate_soil_quiescence(snapshot)?;
        if !(1..=64).contains(&rounds)
            || minimum_quiet_steps == 0
            || self.particle_count > max_particles
        {
            return Err(Error::Solver(
                "unsupported soil retained support request".into(),
            ));
        }
        let ids = snapshot.identity().ok_or_else(|| {
            Error::Solver("soil retention seeds require current global identities".into())
        })?;
        Ok(if requested.is_empty() {
            None
        } else {
            Some(seed_mask(ids.particle_ids(), requested)?)
        })
    }

    pub(in crate::mpm) fn read_soil_retained_support(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
        requested: &[SoilParticleId],
    ) -> Result<MpmSoilRetainedSupportSnapshot> {
        let seeds = self.retained_support_seeds(
            snapshot,
            minimum_quiet_steps,
            max_particles,
            rounds,
            requested,
        )?;
        let words = self.read_soil_support_words(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            Some(rounds),
            seeds.as_deref(),
            None,
        )?;
        Ok(MpmSoilRetainedSupportSnapshot {
            quiescence: snapshot.clone(),
            minimum_quiet_steps,
            rounds,
            requested_particles: requested.to_vec(),
            particles: decode(&words)?,
        })
    }

    pub(in crate::mpm) fn try_request_soil_retained_support(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
        requested: &[SoilParticleId],
    ) -> Result<bool> {
        if !self.soil_support_request_available(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
        )? {
            return Ok(false);
        }
        let seeds = self.retained_support_seeds(
            snapshot,
            minimum_quiet_steps,
            max_particles,
            rounds,
            requested,
        )?;
        self.request_soil_support_words(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            Some(rounds),
            seeds.as_deref(),
            None,
            Some(SupportRequest::Retained(rounds, requested.to_vec())),
        )
    }

    pub(in crate::mpm) fn poll_soil_retained_support(
        &mut self,
        engine: &Engine,
    ) -> Result<Option<MpmSoilRetainedSupportSnapshot>> {
        self.poll_soil_support_words(engine, false, Some(SupportKind::Retained))?
            .map(|(pending, words)| {
                let Some(SupportRequest::Retained(rounds, requested_particles)) = pending.request
                else {
                    unreachable!("checked support request kind");
                };
                Ok(MpmSoilRetainedSupportSnapshot {
                    quiescence: pending.snapshot.expect("asynchronous support provenance"),
                    minimum_quiet_steps: pending.minimum_quiet_steps,
                    rounds,
                    requested_particles,
                    particles: decode(&words)?,
                })
            })
            .transpose()
    }
}

pub(super) fn from_region_words(
    snapshot: &MpmSoilQuiescenceSnapshot,
    minimum_quiet_steps: u32,
    rounds: u32,
    words: &[u32],
) -> Result<MpmSoilRetainedSupportSnapshot> {
    let particles = decode(words)?;
    let ids = snapshot
        .identity()
        .ok_or_else(|| Error::Solver("soil region support requires identities".into()))?;
    if particles.len() != ids.particle_ids().len() {
        return Err(Error::Solver(
            "soil region support identity count mismatch".into(),
        ));
    }
    let requested_particles = ids
        .particle_ids()
        .iter()
        .zip(&particles)
        .filter_map(|(&id, p)| p.requested.then_some(id))
        .collect();
    Ok(MpmSoilRetainedSupportSnapshot {
        quiescence: snapshot.clone(),
        minimum_quiet_steps,
        rounds,
        requested_particles,
        particles,
    })
}

#[cfg(test)]
#[path = "mpm_soil_retained_support_tests.rs"]
pub(super) mod tests;
