use molla_solvers::rigid_world::MpmSoilFrameProfile;

pub fn features(enabled: bool) -> wgpu::Features {
    if enabled {
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
    } else {
        wgpu::Features::empty()
    }
}

pub fn substeps() -> usize {
    (super::DT / molla_solvers::soil_tractor::SoilTractorConfig::default().max_substep_s).ceil()
        as usize
}

pub struct FrameProfile {
    profile: MpmSoilFrameProfile,
    read_ms: f64,
}

impl FrameProfile {
    pub fn new(profile: MpmSoilFrameProfile, frame: u64, read_ms: f64) -> Result<Self, String> {
        let p = &profile.profile;
        let valid = |n: f64| n.is_finite() && n >= 0.0;
        let timing = |step: Option<usize>, start, elapsed| {
            step.is_none_or(|s| s < p.requested_substeps) && valid(start) && valid(elapsed)
        };
        if frame == 0
            || profile.committed_frame != frame
            || p.requested_substeps != substeps()
            || !p.requested_simulated_seconds.is_finite()
            || (p.requested_simulated_seconds - super::DT).abs() > 1e-8
            || p.particles != 50_000
            || !p.preparation_details_enabled
            || !p.timestamp_period_ns.is_finite()
            || p.timestamp_period_ns <= 0.0
            || !valid(read_ms)
            || p.stages.is_empty()
            || p.contact_stages.is_empty()
            || p.preparation_stages.is_empty()
            || profile.machine_stages.is_empty()
            || profile.newton_work.len() != substeps()
            || profile
                .newton_work
                .iter()
                .enumerate()
                .any(|(i, w)| w.substep != i || w.faults != [0; 4])
            || profile.machine_profile_readback_bytes
                < (profile.machine_stages.len() * 16 + profile.newton_work.len() * 32) as u64
            || profile.host_stage_ms.iter().any(|(_, ms)| !valid(*ms))
            || p.stages
                .iter()
                .chain(&p.contact_stages)
                .chain(&p.preparation_stages)
                .any(|s| !timing(s.substep, s.start_ms, s.elapsed_ms))
            || profile
                .machine_stages
                .iter()
                .any(|s| !timing(s.substep, s.start_ms, s.elapsed_ms))
        {
            return Err("invalid or missing committed-frame stage profile".into());
        }
        Ok(Self { profile, read_ms })
    }

    pub fn frame(&self) -> u64 {
        self.profile.committed_frame
    }

    pub fn record(&self) -> serde_json::Value {
        let p = &self.profile.profile;
        let stages = |stages: &[molla_solvers::mpm::MpmStageTiming]| {
            stages
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "stage":s.stage.name(), "substep":s.substep,
                        "start_ms":s.start_ms, "elapsed_ms":s.elapsed_ms,
                    })
                })
                .collect::<Vec<_>>()
        };
        serde_json::json!({
            "event":"stage_profile", "version":1, "committed_frame":self.frame(),
            "diagnostic":true, "acceptance":false,
            "requested_substeps":p.requested_substeps,
            "requested_simulated_seconds":p.requested_simulated_seconds,
            "particles":p.particles, "timestamp_period_ns":p.timestamp_period_ns,
            "active_grid":p.active_grid, "compact_grid":p.compact_grid,
            "native_float_atomics":p.native_float_atomics,
            "particle_tile_ordering":p.particle_tile_ordering,
            "parallel_particle_sort":p.parallel_particle_sort,
            "profile_read_ms":self.read_ms,
            "profile_read_scope":"blocking diagnostic read outside physics_ms and observation_ms; includes both timestamp readers",
            "mpm_readback_bytes":p.readback_bytes,
            "machine_readback_bytes":self.profile.machine_profile_readback_bytes,
            "newton_work_bytes_within_machine_readback":self.profile.newton_work.len() * 32,
            "newton_work_scope":"per-substep recovery reports; linear_iterations combines Krylov iterations and condensed solves, not exclusively Krylov work",
            "newton_work":self.profile.newton_work.iter().map(|w|serde_json::json!({
                "substep":w.substep,"faults":w.faults,"steps":w.steps,"linear_iterations":w.linear_iterations,"trials":w.trials,
            })).collect::<Vec<_>>(),
            "nesting":"contact_stages are inside stages; preparation_stages are inside contact-preparation; do not add nested times to parent totals",
            "clock_origins":"MPM stages share one origin; machine stages have an independent origin; neither is a host clock",
            "stages":stages(&p.stages),
            "contact_stages":stages(&p.contact_stages),
            "preparation_stages":stages(&p.preparation_stages),
            "machine_stages":self.profile.machine_stages.iter().map(|s|serde_json::json!({
                "stage":s.stage,"substep":s.substep,"start_ms":s.start_ms,"elapsed_ms":s.elapsed_ms,
            })).collect::<Vec<_>>(),
            "host_stage_ms":self.profile.host_stage_ms.iter().map(|(stage, ms)|serde_json::json!({
                "stage":stage,"elapsed_ms":ms,
            })).collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_solvers::mpm::{MpmProfileStage, MpmStageTiming, MpmStepProfile};
    use molla_solvers::rigid_world::MpmSoilMachineTiming;

    fn fixture() -> MpmSoilFrameProfile {
        let stages = |stage| {
            (0..substeps())
                .map(|i| MpmStageTiming {
                    stage,
                    substep: Some(i),
                    start_ms: i as f64,
                    elapsed_ms: 0.5,
                })
                .collect()
        };
        MpmSoilFrameProfile {
            committed_frame: 1,
            profile: MpmStepProfile {
                requested_substeps: substeps(),
                requested_simulated_seconds: super::super::DT,
                particles: 50_000,
                active_grid: false,
                compact_grid: false,
                native_float_atomics: true,
                particle_tile_ordering: true,
                parallel_particle_sort: true,
                timestamp_period_ns: 1.0,
                readback_bytes: 1632,
                stages: stages(MpmProfileStage::ContactProjection),
                contact_stages: stages(MpmProfileStage::ContactPreparation),
                preparation_details_enabled: true,
                preparation_stages: stages(MpmProfileStage::PreparationFactor),
            },
            host_stage_ms: [("fixture", 1.0); 6],
            machine_stages: vec![MpmSoilMachineTiming {
                stage: "fixture",
                substep: None,
                start_ms: 0.0,
                elapsed_ms: 1.0,
            }],
            newton_work: (0..substeps())
                .map(|substep| molla_solvers::rigid_world::MpmSoilNewtonWork {
                    substep,
                    faults: [0; 4],
                    steps: 8,
                    linear_iterations: 24,
                    trials: 8,
                })
                .collect(),
            machine_profile_readback_bytes: 16 + substeps() as u64 * 32,
        }
    }

    #[test]
    fn stage_profile_keeps_nested_clocks_and_read_cost_separate() {
        assert!(features(false).is_empty());
        assert!(features(true).contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS));
        assert!(features(true).contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES));
        assert_eq!(substeps(), 34);
        let record = FrameProfile::new(fixture(), 1, 3.0).unwrap().record();
        assert_eq!(record["profile_read_ms"], 3.0);
        assert_eq!(record["machine_readback_bytes"], 16 + substeps() * 32);
        assert_eq!(record["newton_work"].as_array().unwrap().len(), substeps());
        assert_eq!(record["stages"].as_array().unwrap().len(), 34);
        assert_eq!(record["contact_stages"].as_array().unwrap().len(), 34);
        assert_eq!(record["preparation_stages"].as_array().unwrap().len(), 34);
        assert!(record.get("total_ms").is_none());
        assert!(record.get("physics_ms").is_none());
    }

    #[test]
    fn stage_profile_rejects_invalid_or_wrong_frame_data() {
        assert!(FrameProfile::new(fixture(), 0, 0.0).is_err());
        assert!(FrameProfile::new(fixture(), 2, 0.0).is_err());
        assert!(FrameProfile::new(fixture(), 1, f64::NAN).is_err());
        for mutate in [
            |p: &mut MpmSoilFrameProfile| {
                p.newton_work.clear();
            },
            |p: &mut MpmSoilFrameProfile| p.newton_work[0].substep = 1,
            |p: &mut MpmSoilFrameProfile| p.newton_work[0].faults[0] = 8192,
            |p: &mut MpmSoilFrameProfile| p.profile.requested_substeps = 33,
            |p: &mut MpmSoilFrameProfile| p.profile.requested_simulated_seconds = f64::NAN,
            |p: &mut MpmSoilFrameProfile| p.profile.preparation_stages.clear(),
            |p: &mut MpmSoilFrameProfile| p.profile.stages[0].substep = Some(34),
            |p: &mut MpmSoilFrameProfile| p.profile.stages[0].elapsed_ms = -1.0,
            |p: &mut MpmSoilFrameProfile| p.machine_stages[0].start_ms = f64::INFINITY,
            |p: &mut MpmSoilFrameProfile| p.host_stage_ms[0].1 = f64::NAN,
        ] {
            let mut p = fixture();
            mutate(&mut p);
            assert!(FrameProfile::new(p, 1, 0.0).is_err());
        }
    }
}
