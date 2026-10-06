use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{Real, Transform};
use molla_sim::{Model, State};

#[derive(Clone)]
pub(crate) struct MassSeed<'a> {
    model: &'a Model,
    coordinates: Vec<Real>,
    poses: Vec<Transform>,
    matrices: BTreeMap<usize, (Vec<Real>, Option<crate::rigid_impulse::MassFactor>)>,
}

impl<'a> MassSeed<'a> {
    pub(crate) fn new(model: &'a Model) -> Self {
        Self {
            model,
            coordinates: Vec::new(),
            poses: Vec::new(),
            matrices: BTreeMap::new(),
        }
    }

    pub(crate) fn record(
        &mut self,
        model: &Model,
        state: &State,
        matrices: BTreeMap<usize, (Vec<Real>, Option<crate::rigid_impulse::MassFactor>)>,
    ) -> Result<()> {
        if !std::ptr::eq(self.model, model) {
            return Err(Error::Solver("mass seed model mismatch".into()));
        }
        if !matrices.is_empty() {
            self.coordinates = state.joint_q.host()?.to_vec();
            self.poses = state.body_q.host()?.to_vec();
        }
        self.matrices = matrices;
        Ok(())
    }

    pub(super) fn into_matrices(
        self,
        model: &Model,
        state: &State,
    ) -> Result<BTreeMap<usize, (Vec<Real>, Option<crate::rigid_impulse::MassFactor>)>> {
        Ok(if self.recorded_at(model, state)? {
            self.matrices
        } else {
            BTreeMap::new()
        })
    }

    /// A copy of the matrices when `state` is the configuration they were recorded at.
    pub(crate) fn matrices_at(
        &self,
        model: &Model,
        state: &State,
    ) -> Result<BTreeMap<usize, (Vec<Real>, Option<crate::rigid_impulse::MassFactor>)>> {
        Ok(if self.recorded_at(model, state)? {
            self.matrices.clone()
        } else {
            BTreeMap::new()
        })
    }

    fn recorded_at(&self, model: &Model, state: &State) -> Result<bool> {
        if self.matrices.is_empty() || !std::ptr::eq(self.model, model) {
            return Ok(false);
        }
        let same = |a: Real, b: Real| a.is_finite() && a.to_bits() == b.to_bits();
        let q = state.joint_q.host()?;
        let poses = state.body_q.host()?;
        Ok(self.coordinates.len() == q.len()
            && self.coordinates.iter().zip(q).all(|(&a, &b)| same(a, b))
            && self.poses.len() == poses.len()
            && self.poses.iter().zip(poses).all(|(a, b)| {
                a.position
                    .to_array()
                    .into_iter()
                    .chain(a.rotation.to_array())
                    .zip(
                        b.position
                            .to_array()
                            .into_iter()
                            .chain(b.rotation.to_array()),
                    )
                    .all(|(a, b)| same(a, b))
            }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::{Mat3, SpatialVector, Vec3};
    use molla_sim::MassProperties;
    use molla_sim::runtime::{BodyDesc, RigidScene};

    #[test]
    fn seed_reuses_allocation_only_for_exact_model_and_configuration() {
        let mut scene = RigidScene::new().unwrap();
        scene
            .insert_body(BodyDesc {
                additional_mass: Some(MassProperties {
                    mass: 2.0,
                    local_com: Vec3::new(0.1, 0.2, -0.1),
                    inertia: Mat3::IDENTITY,
                }),
                ..Default::default()
            })
            .unwrap();
        for edit in 0..8 {
            let model = scene.model();
            let state = scene.state();
            let matrix = crate::mass_matrix(model, state, 0).unwrap();
            let allocation = matrix.as_ptr();
            let mut seed = MassSeed::new(model);
            seed.record(model, state, BTreeMap::from([(0, (matrix, None))]))
                .unwrap();
            let mut next = state.clone_host().unwrap();
            match edit {
                0 => {}
                1 => {
                    next.joint_qd.host_mut().unwrap()[0] = 2.0;
                    next.body_qd.host_mut().unwrap()[0] = SpatialVector::new(Vec3::X, Vec3::Y);
                    next.body_f.host_mut().unwrap()[0] = SpatialVector::new(Vec3::Z, Vec3::X);
                }
                2 => next.joint_q.host_mut().unwrap()[0] += 0.125,
                3 => next.body_q.host_mut().unwrap()[0].position.x = 0.1,
                4 => {
                    assert_eq!(next.joint_q.host().unwrap()[3].to_bits(), 0.0_f64.to_bits());
                    next.joint_q.host_mut().unwrap()[3] = -0.0;
                }
                5 => next.body_q.host_mut().unwrap()[0].rotation.w = -1.0,
                6 => next.joint_q.host_mut().unwrap()[0] = Real::NAN,
                _ => next.body_q.host_mut().unwrap()[0].position.y = Real::INFINITY,
            }
            let mut cache = super::super::JointMass::with_seed(model, &next, Some(seed)).unwrap();
            assert_eq!(cache.entries[0].is_some(), edit < 2, "edit={edit}");
            if edit < 2 {
                assert_eq!(cache.matrix(0).unwrap().as_ptr(), allocation);
                assert!(cache.entries[0].as_ref().unwrap().factor.is_none());
                assert_eq!(
                    cache.matrix(0).unwrap(),
                    crate::mass_matrix(model, &next, 0).unwrap()
                );
            }
        }
        let other = RigidScene::new().unwrap();
        let mut seed = MassSeed::new(scene.model());
        assert!(
            seed.record(other.model(), other.state(), BTreeMap::new())
                .is_err()
        );
        seed.record(
            scene.model(),
            scene.state(),
            BTreeMap::from([(0, (vec![1.0; 36], None))]),
        )
        .unwrap();
        assert!(
            seed.into_matrices(other.model(), other.state())
                .unwrap()
                .is_empty()
        );
    }
}
