use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::dense_solve::{back_sub, dense_chol, forward_sub};
use molla_math::{Real, SpatialVector};
use molla_sim::{Model, State};

struct MassSolve {
    size: usize,
    singular: Option<crate::rigid_impulse::MassFactor>,
    factor: Option<Vec<Real>>,
}

#[derive(Default)]
pub(crate) struct ArticulatedResponse {
    masses: BTreeMap<usize, MassSolve>,
}

impl ArticulatedResponse {
    /// COM wrench quadratic response `wᵀ J M⁻¹ Jᵀ w` at the current substep state.
    pub(crate) fn quadratic(
        &mut self,
        model: &Model,
        state: &State,
        body: usize,
        wrench: SpatialVector,
    ) -> Result<Real> {
        if !wrench.linear.is_finite() || !wrench.angular.is_finite() {
            return Err(Error::Solver("non-finite articulated wrench".into()));
        }
        if wrench == SpatialVector::ZERO {
            return Ok(0.0);
        }
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
        if let std::collections::btree_map::Entry::Vacant(entry) = self.masses.entry(art) {
            let matrix = crate::mass_matrix(model, state, art)?;
            if matrix.iter().any(|v| !v.is_finite()) {
                return Err(Error::Solver("non-finite articulated mass".into()));
            }
            let scale = matrix.iter().copied().map(Real::abs).fold(0.0, Real::max);
            let mut factor = matrix.clone();
            let factored = dense_chol(&mut factor, n).is_ok()
                && (0..n).all(|i| factor[i * n + i].powi(2) > scale * 1e-12);
            entry.insert(MassSolve {
                size: n,
                singular: if factored {
                    None
                } else {
                    Some(crate::rigid_impulse::MassFactor::new(&matrix, n)?)
                },
                factor: factored.then_some(factor),
            });
        }
        let j = crate::jacobian(model, state, art, body, model.body_com.host()?[body])?;
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
        let mass = &self.masses[&art];
        let solved = if let Some(factor) = &mass.factor {
            back_sub(
                factor,
                &forward_sub(factor, &projected, mass.size)?,
                mass.size,
            )?
        } else {
            mass.singular.as_ref().unwrap().solve(&projected)?
        };
        let response = projected
            .iter()
            .zip(solved)
            .map(|(a, b)| a * b)
            .sum::<Real>();
        if !response.is_finite() {
            return Err(Error::Solver("non-finite articulated response".into()));
        }
        Ok(response.max(0.0))
    }
}
