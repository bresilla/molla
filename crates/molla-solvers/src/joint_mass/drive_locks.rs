use super::{JointMass, MassFactor};
use molla_core::Result;
use molla_math::{Quat, Real, Vec3};
use molla_sim::JointType;

pub(super) struct Projection {
    rows: Vec<Vec<Real>>,
    errors: Vec<Real>,
    responses: Vec<Vec<Real>>,
    factor: Option<MassFactor>,
}

impl JointMass<'_> {
    fn prepare_drive_locks(&mut self, articulation: usize, count: usize) -> Result<()> {
        if self.entry(articulation)?.drive_locks.is_some() {
            return Ok(());
        }
        let model = self.model;
        let starts = model.articulation_start.host()?;
        let (begin, end) = if model.articulation_count == 0 {
            (0, model.joint_count)
        } else {
            (
                starts[articulation] as usize,
                starts[articulation + 1] as usize,
            )
        };
        let offsets = model.joint_dof_offset.host()?;
        let first = offsets[begin] as usize;
        let coords = model.joint_coord_offset.host()?;
        let kinds = model.joint_type.host()?;
        let enabled = model.joint_enabled.host()?;
        let lower = model.joint_limit_lower.host()?;
        let upper = model.joint_limit_upper.host()?;
        let q = self.state.joint_q.host()?;
        let mut rows = Vec::new();
        let mut errors = Vec::new();
        for j in begin..end {
            let angular = if kinds[j] == JointType::Ball as u32 {
                0
            } else if kinds[j] == JointType::D6 as u32 {
                3
            } else {
                continue;
            };
            if enabled[j] == 0 {
                continue;
            }
            let start = offsets[j] as usize + angular;
            let lo = Vec3::from_slice(&lower[start..start + 3]);
            let hi = Vec3::from_slice(&upper[start..start + 3]);
            if !(0..3)
                .any(|d| lo[d].is_finite() && hi[d].is_finite() && (hi[d] - lo[d]).abs() < 1e-12)
            {
                continue;
            }
            let c = coords[j] as usize + angular;
            for row in crate::joint_angles::stops(
                Quat::from_xyzw(q[c], q[c + 1], q[c + 2], q[c + 3]),
                lo,
                hi,
            )? {
                if !row.bilateral {
                    continue;
                }
                let mut unit = vec![0.; count];
                unit[start - first..start - first + 3].copy_from_slice(&row.gradient.to_array());
                rows.push(unit);
                errors.push(row.error);
            }
        }
        let responses = if rows.is_empty() {
            Vec::new()
        } else {
            self.factor(articulation, count)?.solve_batch(&rows)?
        };
        let n = rows.len();
        let mut gram = vec![0.; n * n];
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] = rows[i].iter().zip(&responses[j]).map(|(a, b)| a * b).sum();
            }
        }
        let factor = if n == 0 {
            None
        } else {
            Some(MassFactor::new(&gram, n)?)
        };
        self.entry(articulation)?.drive_locks = Some(Projection {
            rows,
            errors,
            responses,
            factor,
        });
        Ok(())
    }

    pub(crate) fn has_drive_locks(&mut self, articulation: usize, count: usize) -> Result<bool> {
        self.prepare_drive_locks(articulation, count)?;
        Ok(self
            .entry(articulation)?
            .drive_locks
            .as_ref()
            .unwrap()
            .factor
            .is_some())
    }

    pub(crate) fn drive_responses(
        &mut self,
        articulation: usize,
        count: usize,
        units: &[Vec<Real>],
    ) -> Result<Vec<Vec<Real>>> {
        self.prepare_drive_locks(articulation, count)?;
        let mut responses = self.factor(articulation, count)?.solve_batch(units)?;
        let projection = self.entry(articulation)?.drive_locks.as_ref().unwrap();
        for response in &mut responses {
            projection.remove_normal(response, None)?;
        }
        Ok(responses)
    }

    pub(crate) fn project_drive_acceleration(
        &mut self,
        articulation: usize,
        qd: &[Real],
        acceleration: &mut [Real],
        dt: Real,
    ) -> Result<()> {
        self.prepare_drive_locks(articulation, acceleration.len())?;
        let projection = self.entry(articulation)?.drive_locks.as_ref().unwrap();
        if projection.factor.is_none() {
            return Ok(());
        }
        let mut velocity: Vec<_> = qd
            .iter()
            .zip(acceleration.iter())
            .map(|(v, a)| v + dt * a)
            .collect();
        let target: Vec<_> = projection.errors.iter().map(|error| -error / dt).collect();
        projection.remove_normal(&mut velocity, Some(&target))?;
        for ((a, v), old) in acceleration.iter_mut().zip(velocity).zip(qd) {
            *a = (v - old) / dt;
        }
        Ok(())
    }
}

impl Projection {
    fn remove_normal(&self, value: &mut [Real], target: Option<&[Real]>) -> Result<()> {
        let Some(factor) = &self.factor else {
            return Ok(());
        };
        let rhs: Vec<_> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .zip(value.iter())
                    .map(|(a, b)| a * b)
                    .sum::<Real>()
                    - target.map_or(0., |target| target[i])
            })
            .collect();
        let multipliers = factor.solve(&rhs)?;
        for (response, multiplier) in self.responses.iter().zip(multipliers) {
            for (value, response) in value.iter_mut().zip(response) {
                *value -= multiplier * response;
            }
        }
        Ok(())
    }
}
