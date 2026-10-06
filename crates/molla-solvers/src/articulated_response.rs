use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{Real, SpatialVector};
use molla_sim::{Model, State};

pub(crate) struct ArticulatedResponse<'a> {
    model: &'a Model,
    state: &'a State,
    masses: BTreeMap<usize, (Vec<Real>, crate::rigid_impulse::MassFactor)>,
    seeded: BTreeMap<usize, (Vec<Real>, Option<crate::rigid_impulse::MassFactor>)>,
    jacobians: BTreeMap<(usize, usize), Vec<Real>>,
}

impl<'a> ArticulatedResponse<'a> {
    pub(crate) fn new(model: &'a Model, state: &'a State) -> Self {
        Self {
            model,
            state,
            masses: BTreeMap::new(),
            seeded: BTreeMap::new(),
            jacobians: BTreeMap::new(),
        }
    }

    /// Starts from the matrices `seed` recorded, when it was recorded at `state`.
    pub(crate) fn seeded(
        model: &'a Model,
        state: &'a State,
        seed: &crate::joint_mass::MassSeed<'_>,
    ) -> Result<Self> {
        let mut response = Self::new(model, state);
        response.seeded = seed.matrices_at(model, state)?;
        Ok(response)
    }

    pub(crate) fn state(&self) -> &'a State {
        self.state
    }

    fn factor(&mut self, art: usize, n: usize) -> Result<&crate::rigid_impulse::MassFactor> {
        if let std::collections::btree_map::Entry::Vacant(entry) = self.masses.entry(art) {
            let (matrix, factor) = match self.seeded.remove(&art) {
                Some((matrix, factor)) if matrix.len() == n * n => (matrix, factor),
                _ => (
                    crate::response_mass_matrix(self.model, self.state, art)?,
                    None,
                ),
            };
            let factor = match factor.filter(|f| f.full_rank_cholesky(n).is_some()) {
                Some(factor) => factor,
                None => {
                    let _span = tracing::debug_span!("response_mass_factor", dofs = n).entered();
                    crate::rigid_impulse::MassFactor::new(&matrix, n)?
                }
            };
            entry.insert((matrix, factor));
        }
        Ok(&self.masses[&art].1)
    }

    pub(crate) fn export_mass(self, seed: &mut crate::joint_mass::MassSeed<'_>) -> Result<()> {
        seed.record(
            self.model,
            self.state,
            self.masses
                .into_iter()
                .map(|(art, (matrix, factor))| (art, (matrix, factor.into_full_rank())))
                .collect(),
        )
    }

    pub(crate) fn jacobian(&mut self, art: usize, body: usize) -> Result<&[Real]> {
        if let std::collections::btree_map::Entry::Vacant(entry) = self.jacobians.entry((art, body))
        {
            let _span = tracing::debug_span!("response_jacobian", art, body).entered();
            let point = *self
                .model
                .body_com
                .host()?
                .get(body)
                .ok_or_else(|| Error::Solver("missing response body".into()))?;
            entry.insert(crate::jacobian(self.model, self.state, art, body, point)?);
        }
        Ok(&self.jacobians[&(art, body)])
    }

    pub(crate) fn add_gram(
        &mut self,
        art: usize,
        n: usize,
        projected: &[Vec<Real>],
        response: &mut [Real],
    ) -> Result<()> {
        self.factor(art, n)?.add_gram(projected, response)
    }

    pub(crate) fn solve(&mut self, art: usize, rhs: &[Real]) -> Result<Vec<Real>> {
        self.factor(art, rhs.len())?.solve(rhs)
    }

    pub(crate) fn add_gram_compact(
        &mut self,
        art: usize,
        n: usize,
        projected: Vec<Vec<Real>>,
        response: &mut [Real],
    ) -> Result<()> {
        self.factor(art, n)?.add_gram_compact(projected, response)
    }

    /// COM wrench quadratic response `wᵀ J M⁻¹ Jᵀ w` at the current substep state.
    pub(crate) fn quadratic(&mut self, body: usize, wrench: SpatialVector) -> Result<Real> {
        if !wrench.linear.is_finite() || !wrench.angular.is_finite() {
            return Err(Error::Solver("non-finite articulated wrench".into()));
        }
        if wrench == SpatialVector::ZERO {
            return Ok(0.0);
        }
        let model = self.model;
        let children = model.joint_child.host()?;
        let starts = model.articulation_start.host()?;
        let Some(joint) = children.iter().position(|&child| child == body as i32) else {
            return Ok(0.0);
        };
        let (art, begin, end) = if model.articulation_count == 0 {
            (0, 0, model.joint_count)
        } else {
            let (art, range) = starts
                .windows(2)
                .enumerate()
                .find(|(_, range)| (range[0] as usize..range[1] as usize).contains(&joint))
                .ok_or_else(|| Error::Solver("missing wrench articulation".into()))?;
            (art, range[0] as usize, range[1] as usize)
        };
        let offsets = model.joint_dof_offset.host()?;
        let n = (offsets[end] - offsets[begin]) as usize;
        if n == 0 {
            return Ok(0.0);
        }
        let j = self.jacobian(art, body)?;
        let projected: Vec<_> = (0..n)
            .map(|i| {
                wrench.linear.x * j[i]
                    + wrench.linear.y * j[n + i]
                    + wrench.linear.z * j[2 * n + i]
                    + wrench.angular.x * j[3 * n + i]
                    + wrench.angular.y * j[4 * n + i]
                    + wrench.angular.z * j[5 * n + i]
            })
            .collect();
        self.factor(art, n)?.quadratic(&projected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::{Mat3, Quat, Transform, Vec3};
    use molla_sim::runtime::{BodyDesc, RigidScene};
    use molla_sim::{BodyParams, MassProperties};

    #[test]
    fn shared_responses_reuse_each_articulation_and_refresh_with_state() {
        let mut scene = RigidScene::new().unwrap();
        let mut bodies = Vec::new();
        for mass in [2.0, 5.0] {
            bodies.push(
                scene
                    .insert_body(BodyDesc {
                        params: BodyParams::new(),
                        additional_mass: Some(MassProperties {
                            mass,
                            local_com: Vec3::ZERO,
                            inertia: Mat3::from_diagonal(Vec3::new(1.0, 2.0, 3.0)),
                        }),
                        ..Default::default()
                    })
                    .unwrap(),
            );
        }
        {
            let mut response = ArticulatedResponse::new(scene.model(), scene.state());
            for (art, (&body, mass)) in bodies.iter().zip([2.0, 5.0]).enumerate() {
                let body = scene.body_index(body).unwrap().index();
                let wrench = SpatialVector::new(Vec3::X, Vec3::ZERO);
                let quadratic = response.quadratic(body, wrench).unwrap();
                assert!((quadratic - 1.0 / mass).abs() < 1e-14);
                let jacobian = response.jacobian(art, body).unwrap();
                let pointer = jacobian.as_ptr();
                let n = jacobian.len() / 6;
                let projected = vec![jacobian[..n].to_vec(), jacobian[n..2 * n].to_vec()];
                let mut gram = vec![0.0; 4];
                response.add_gram(art, n, &projected, &mut gram).unwrap();
                assert!((gram[0] - quadratic).abs() < 1e-14);
                assert!((gram[3] - quadratic).abs() < 1e-14);
                assert_eq!(gram[1], 0.0);
                assert_eq!(gram[2], 0.0);
                assert_eq!(response.jacobian(art, body).unwrap().as_ptr(), pointer);
                assert_eq!(response.masses.len(), art + 1);
                assert_eq!(response.jacobians.len(), art + 1);
            }
            let body = scene.body_index(bodies[0]).unwrap().index();
            let angular = response
                .quadratic(body, SpatialVector::new(Vec3::ZERO, Vec3::X))
                .unwrap();
            assert!((angular - 1.0).abs() < 1e-14);
        }
        scene
            .set_body_pose(
                bodies[0],
                Transform {
                    position: Vec3::new(30.0, -4.0, 6.0),
                    rotation: Quat::from_rotation_y(std::f64::consts::FRAC_PI_2),
                },
            )
            .unwrap();
        let mut response = ArticulatedResponse::new(scene.model(), scene.state());
        assert!(response.masses.is_empty() && response.jacobians.is_empty());
        let body = scene.body_index(bodies[0]).unwrap().index();
        let angular = response
            .quadratic(body, SpatialVector::new(Vec3::ZERO, Vec3::X))
            .unwrap();
        assert!((angular - 1.0 / 3.0).abs() < 1e-14);
        assert_eq!(response.masses.len(), 1);
    }
}
