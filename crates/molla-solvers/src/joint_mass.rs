use molla_core::Result;
use molla_math::Real;
use molla_sim::{Model, State};

use crate::rigid_impulse::MassFactor;

#[path = "joint_mass/seed.rs"]
mod seed;
pub(crate) use seed::MassSeed;

#[path = "joint_mass/drive_locks.rs"]
mod drive_locks;

struct Entry {
    matrix: Vec<Real>,
    factor: Option<MassFactor>,
    drive_locks: Option<drive_locks::Projection>,
}

/// Lazy articulation mass responses at one immutable post-FK state.
pub(crate) struct JointMass<'a> {
    model: &'a Model,
    state: &'a State,
    entries: Vec<Option<Entry>>,
}

impl<'a> JointMass<'a> {
    pub(crate) fn new(model: &'a Model, state: &'a State) -> Self {
        Self {
            model,
            state,
            entries: (0..model.articulation_count.max(1)).map(|_| None).collect(),
        }
    }

    pub(crate) fn with_seed(
        model: &'a Model,
        state: &'a State,
        seed: Option<MassSeed<'_>>,
    ) -> Result<Self> {
        let mut result = Self::new(model, state);
        if let Some(seed) = seed {
            for (art, (matrix, factor)) in seed.into_matrices(model, state)? {
                result.entries[art] = Some(Entry {
                    matrix,
                    factor,
                    drive_locks: None,
                });
            }
        }
        Ok(result)
    }

    /// Records every matrix built here, with its full-rank factor, into `seed`.
    pub(crate) fn export(self, seed: &mut MassSeed<'_>) -> Result<()> {
        let matrices = self
            .entries
            .into_iter()
            .enumerate()
            .filter_map(|(art, entry)| {
                entry.map(|entry| {
                    (
                        art,
                        (
                            entry.matrix,
                            entry.factor.and_then(MassFactor::into_full_rank),
                        ),
                    )
                })
            })
            .collect();
        seed.record(self.model, self.state, matrices)
    }

    fn entry(&mut self, articulation: usize) -> Result<&mut Entry> {
        if self.entries[articulation].is_none() {
            self.entries[articulation] = Some(Entry {
                matrix: crate::response_mass_matrix(self.model, self.state, articulation)?,
                factor: None,
                drive_locks: None,
            });
        }
        Ok(self.entries[articulation].as_mut().unwrap())
    }

    pub(crate) fn matrix(&mut self, articulation: usize) -> Result<&[Real]> {
        Ok(&self.entry(articulation)?.matrix)
    }

    pub(crate) fn cached_cholesky(&self, articulation: usize, n: usize) -> Option<(Real, &[Real])> {
        self.entries
            .get(articulation)?
            .as_ref()?
            .factor
            .as_ref()?
            .full_rank_cholesky(n)
    }

    pub(crate) fn retain_cholesky(
        &mut self,
        articulation: usize,
        factor: Vec<Real>,
        n: usize,
        scale: Real,
    ) -> Result<()> {
        self.entry(articulation)?.factor = MassFactor::from_cholesky(factor, n, scale);
        Ok(())
    }

    pub(crate) fn factor(&mut self, articulation: usize, n: usize) -> Result<&MassFactor> {
        let entry = self.entry(articulation)?;
        if entry.factor.is_none() {
            entry.factor = Some(MassFactor::new(&entry.matrix, n)?);
        }
        Ok(entry.factor.as_ref().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::{Mat3, Vec3};
    use molla_sim::MassProperties;
    use molla_sim::runtime::{BodyDesc, RigidScene};

    #[test]
    fn lazy_matrices_and_factors_are_scoped_to_one_state() {
        let mut scene = RigidScene::new().unwrap();
        let mut bodies = Vec::new();
        for mass in [2.0, 5.0] {
            bodies.push(
                scene
                    .insert_body(BodyDesc {
                        additional_mass: Some(MassProperties {
                            mass,
                            local_com: Vec3::ZERO,
                            inertia: Mat3::IDENTITY * mass,
                        }),
                        ..Default::default()
                    })
                    .unwrap(),
            );
        }
        {
            let mut cache = JointMass::new(scene.model(), scene.state());
            assert!(cache.entries.iter().all(Option::is_none));
            for articulation in 0..2 {
                let matrix = cache.matrix(articulation).unwrap();
                let pointer = matrix.as_ptr();
                let fresh = MassFactor::new(matrix, 6).unwrap();
                assert!(
                    cache.entries[articulation]
                        .as_ref()
                        .unwrap()
                        .factor
                        .is_none()
                );
                for axis in 0..6 {
                    let mut unit = [0.0; 6];
                    unit[axis] = 1.0;
                    assert_eq!(
                        cache.factor(articulation, 6).unwrap().solve(&unit).unwrap(),
                        fresh.solve(&unit).unwrap()
                    );
                }
                assert_eq!(cache.matrix(articulation).unwrap().as_ptr(), pointer);
                let factor = cache.factor(articulation, 6).unwrap() as *const MassFactor;
                assert_eq!(
                    cache.factor(articulation, 6).unwrap() as *const MassFactor,
                    factor
                );
                assert_eq!(
                    cache.entries.iter().filter(|v| v.is_some()).count(),
                    articulation + 1
                );
            }
        }
        scene.remove_body(bodies[0]).unwrap();
        let mut cache = JointMass::new(scene.model(), scene.state());
        assert!(cache.entries.iter().all(Option::is_none));
        let mut unit = [0.0; 6];
        unit[0] = 1.0;
        let response = cache.factor(0, 6).unwrap().solve(&unit).unwrap();
        assert!((response[0] - 0.2).abs() < 1e-14);
    }
}
