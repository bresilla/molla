use super::{Bond, Fiber, StrandModel, invalid};
use molla_core::Result;
use molla_math::{Mat3, Quat, Real, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct ConstraintRow {
    pub value: Real,
    pub linear: [Vec3; 2],
    pub angular: [Vec3; 2],
    pub compliance: Real,
}

fn rotation_vector(rotation: Quat) -> Vec3 {
    let mut q = rotation.normalize();
    if q.w < 0.0 {
        q = -q;
    }
    let v = Vec3::new(q.x, q.y, q.z);
    let s2 = v.length_squared();
    if s2 < 1e-10 {
        return v * (2.0 + s2 / 3.0 + 3.0 * s2 * s2 / 20.0);
    }
    let s = s2.sqrt();
    v * (2.0 * s.atan2(q.w) / s)
}

/// Objective translation and principal rotation-vector residuals with exact Jacobians.
pub fn bond_rows(bond: &Bond, a: &Fiber, b: &Fiber) -> [ConstraintRow; 6] {
    let qa = a.motion.pose.rotation;
    let ra = qa * (bond.anchors[0] - a.mass.local_com);
    let rb = b.motion.pose.rotation * (bond.anchors[1] - b.mass.local_com);
    let delta = b.center() + rb - a.center() - ra;
    let translation = qa.conjugate() * delta - bond.rest_offset;
    let error = qa.conjugate() * b.motion.pose.rotation * bond.rest_rotation.conjugate();
    let vector = rotation_vector(error);
    let theta2 = vector.length_squared();
    let jacobian_term = if theta2 < 1e-6 {
        1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0
    } else {
        let half = 0.5 * theta2.sqrt();
        (1.0 - half / half.tan()) / theta2
    };
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let weakening = (1.0 - bond.damage).max(1e-6);
    std::array::from_fn(|i| {
        let axis = axes[i % 3];
        if i < 3 {
            let n = qa * axis;
            ConstraintRow {
                value: translation[i],
                linear: [-n, n],
                angular: [n.cross(delta) - ra.cross(n), rb.cross(n)],
                compliance: bond.material.translation[i] / weakening,
            }
        } else {
            let n = qa
                * (axis
                    + 0.5 * vector.cross(axis)
                    + jacobian_term * vector.cross(vector.cross(axis)));
            ConstraintRow {
                value: vector[i - 3],
                linear: [Vec3::ZERO; 2],
                angular: [-n, n],
                compliance: bond.material.rotation[i - 3] / weakening,
            }
        }
    })
}

#[derive(Clone, Copy, Debug)]
pub struct PointGrab {
    pub fiber: u64,
    pub local_point: Vec3,
    pub target: Vec3,
    /// Point-constraint compliance in m/N.
    pub compliance: Real,
    /// Target velocity in m/s.
    pub target_velocity: Vec3,
    /// Viscous point damping in N s/m.
    pub damping: Real,
    /// Optional point-force limit in N.
    pub max_force: Option<Real>,
}

impl PointGrab {
    pub(super) fn compliance_at(self, point: Vec3) -> Real {
        self.max_force.map_or(self.compliance, |limit| {
            self.compliance.max(point.distance(self.target) / limit)
        })
    }
    pub(super) fn damped(self, fiber: &Fiber, dt: Real) -> Self {
        let ratio = self.damping * self.compliance / dt;
        let previous = fiber.motion.pose.transform_point(self.local_point);
        Self {
            target: (self.target + ratio * (previous + dt * self.target_velocity)) / (1.0 + ratio),
            compliance: self.compliance / (1.0 + ratio),
            ..self
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StepConfig {
    pub dt: Real,
    pub iterations: usize,
    pub gravity: Vec3,
    pub air_damping: Real,
    pub ground_height: Option<Real>,
    pub friction: Real,
    /// Opt-in discrete cell contact with this detection margin; no shape inflation or CCD.
    pub contact_margin: Option<Real>,
    /// Exclude contact across this many intact bonds: one for direct neighbors, two for junction neighborhoods.
    pub contact_exclusion_hops: u32,
    /// Restitution for opt-in convex cell pairs, in [0, 1].
    pub pair_restitution: Real,
    /// Coupled body-space solve, or color-batched local projections when absent.
    pub coupled: Option<super::CoupledConfig>,
    pub grab: Option<PointGrab>,
}

impl Default for StepConfig {
    fn default() -> Self {
        Self {
            dt: 1.0 / 240.0,
            iterations: 32,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            air_damping: 0.05,
            ground_height: Some(0.0),
            friction: 0.5,
            contact_margin: None,
            contact_exclusion_hops: 1,
            pair_restitution: 0.0,
            coupled: None,
            grab: None,
        }
    }
}

impl StepConfig {
    pub fn validate(self) -> Result<()> {
        if self.grab.is_some_and(|g| {
            !g.local_point.is_finite()
                || !g.target.is_finite()
                || !g.compliance.is_finite()
                || g.compliance <= 0.0
                || !g.target_velocity.is_finite()
                || !g.damping.is_finite()
                || g.damping < 0.0
                || g.max_force.is_some_and(|f| !f.is_finite() || f <= 0.0)
        }) {
            return Err(invalid("invalid point grab"));
        }
        if let Some(coupled) = self.coupled {
            coupled.validate()?;
        }
        if !self.dt.is_finite()
            || self.dt <= 0.0
            || !self.dt.powi(2).is_normal()
            || !(1..=4096).contains(&self.iterations)
            || !self.gravity.is_finite()
            || !self.air_damping.is_finite()
            || self.air_damping < 0.0
            || self.ground_height.is_some_and(|h| !h.is_finite())
            || !self.friction.is_finite()
            || self.friction < 0.0
            || self
                .contact_margin
                .is_some_and(|m| !m.is_finite() || m < 0.0)
            || !self.pair_restitution.is_finite()
            || !(0.0..=1.0).contains(&self.pair_restitution)
            || !(1..=2).contains(&self.contact_exclusion_hops)
        {
            return Err(invalid("invalid step parameters"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct StepReport {
    pub broken_bonds: Vec<u64>,
    pub bond_loads: Vec<(u64, Vec3, Vec3)>,
    pub elastic_energy: Real,
}

pub(super) fn rotate(fiber: &mut Fiber, delta: Vec3, center: Vec3) {
    fiber.motion.pose.rotation =
        (Quat::from_scaled_axis(delta) * fiber.motion.pose.rotation).normalize();
    fiber.motion.pose.position = center - fiber.motion.pose.rotation * fiber.mass.local_com;
}

pub(super) fn pair_mut(fibers: &mut [Fiber], a: usize, b: usize) -> (&mut Fiber, &mut Fiber) {
    if a < b {
        let (lo, hi) = fibers.split_at_mut(b);
        (&mut lo[a], &mut hi[0])
    } else {
        let (lo, hi) = fibers.split_at_mut(a);
        (&mut hi[0], &mut lo[b])
    }
}

pub(super) fn weight(row: ConstraintRow, a: &Fiber, b: &Fiber) -> Result<(Real, [Mat3; 2])> {
    let inertia = [a.world_inverse_inertia()?, b.world_inverse_inertia()?];
    let w = a.inverse_mass() * row.linear[0].length_squared()
        + b.inverse_mass() * row.linear[1].length_squared()
        + row.angular[0].dot(inertia[0] * row.angular[0])
        + row.angular[1].dot(inertia[1] * row.angular[1]);
    Ok((w, inertia))
}

fn project(
    row: ConstraintRow,
    a: &mut Fiber,
    b: &mut Fiber,
    lambda: &mut Real,
    dt: Real,
) -> Result<()> {
    let (w, inertia) = weight(row, a, b)?;
    if w == 0.0 {
        return Ok(());
    }
    let alpha = row.compliance / (dt * dt);
    let dl = (-row.value - alpha * *lambda) / (w + alpha);
    if !dl.is_finite() {
        return Err(invalid("constraint projection overflow"));
    }
    *lambda += dl;
    for (i, fiber) in [a, b].into_iter().enumerate() {
        if fiber.pin.is_some() {
            continue;
        }
        let center = fiber.center() + row.linear[i] * (fiber.inverse_mass() * dl);
        rotate(fiber, inertia[i] * row.angular[i] * dl, center);
    }
    Ok(())
}

fn predict(fiber: &mut Fiber, config: StepConfig) -> Result<()> {
    if fiber.pin.is_some() {
        return Ok(());
    }
    let attenuation = (-config.air_damping * config.dt).exp();
    fiber.motion.linear_velocity =
        (fiber.motion.linear_velocity + config.gravity * config.dt) * attenuation;
    let q = fiber.motion.pose.rotation;
    let r = Mat3::from_quat(q);
    let angular_momentum =
        r * fiber.mass.inertia * r.transpose() * fiber.motion.angular_velocity * attenuation;
    let local_inverse = fiber.mass.inverse_inertia()?;
    let center = fiber.center() + fiber.motion.linear_velocity * config.dt;
    let initial = q.conjugate() * fiber.motion.angular_velocity * attenuation;
    let mut omega = initial;
    let half = 0.5 * config.dt;
    for _ in 0..12 {
        let momentum = fiber.mass.inertia * omega;
        let residual = omega - initial + half * (local_inverse * omega.cross(momentum));
        if residual.length() <= 1e-12 * (1.0 + initial.length()) {
            break;
        }
        let columns = [Vec3::X, Vec3::Y, Vec3::Z].map(|axis| {
            axis + half
                * (local_inverse * (axis.cross(momentum) + omega.cross(fiber.mass.inertia * axis)))
        });
        omega -= Mat3::from_cols(columns[0], columns[1], columns[2]).inverse() * residual;
    }
    let residual =
        omega - initial + half * (local_inverse * omega.cross(fiber.mass.inertia * omega));
    if !omega.is_finite() || residual.length() > 1e-10 * (1.0 + initial.length()) {
        return Err(invalid("free rotation midpoint did not converge"));
    }
    let delta = Quat::from_xyzw(half * omega.x, half * omega.y, half * omega.z, 1.0).normalize();
    fiber.motion.pose.rotation = (q * delta).normalize();
    fiber.motion.pose.position = center - fiber.motion.pose.rotation * fiber.mass.local_com;
    fiber.motion.angular_velocity = fiber.world_inverse_inertia()? * angular_momentum;
    Ok(())
}

fn project_ground(fiber: &mut Fiber, height: Real, lambdas: &mut [Real]) -> Result<()> {
    if fiber.pin.is_some() {
        return Ok(());
    }
    for (index, lambda) in lambdas.iter_mut().enumerate() {
        let point = fiber
            .motion
            .pose
            .transform_point(fiber.shape.vertices()[index]);
        let error = point.y - height;
        if error >= 0.0 && *lambda == 0.0 {
            continue;
        }
        let r = point - fiber.center();
        let angular = r.cross(Vec3::Y);
        let inertia = fiber.world_inverse_inertia()?;
        let w = fiber.inverse_mass() + angular.dot(inertia * angular);
        let new_lambda = (*lambda - error / w).max(0.0);
        let mut dl = new_lambda - *lambda;
        let angle = (inertia * angular).length() * dl.abs();
        if angle > 0.2 {
            dl *= 0.2 / angle;
        }
        *lambda += dl;
        let center = fiber.center() + Vec3::Y * (fiber.inverse_mass() * dl);
        rotate(fiber, inertia * angular * dl, center);
    }
    Ok(())
}

fn damp(row: ConstraintRow, a: &mut Fiber, b: &mut Fiber, factor: Real) -> Result<()> {
    let (w, inertia) = weight(row, a, b)?;
    if w == 0.0 {
        return Ok(());
    }
    let rate = row.linear[0].dot(a.motion.linear_velocity)
        + row.linear[1].dot(b.motion.linear_velocity)
        + row.angular[0].dot(a.motion.angular_velocity)
        + row.angular[1].dot(b.motion.angular_velocity);
    let impulse = -factor * rate / w;
    for (i, fiber) in [a, b].into_iter().enumerate() {
        fiber.motion.linear_velocity += row.linear[i] * (fiber.inverse_mass() * impulse);
        fiber.motion.angular_velocity += inertia[i] * row.angular[i] * impulse;
    }
    Ok(())
}

/// CPU reference step; errors leave the input graph and all motion unchanged.
pub fn step_cpu(model: &mut StrandModel, config: StepConfig) -> Result<StepReport> {
    config.validate()?;
    let grab_index = config
        .grab
        .map(|g| {
            model
                .indices
                .get(&g.fiber)
                .copied()
                .filter(|&i| {
                    model.fibers[i].pin.is_none() && model.fibers[i].contains_local(g.local_point)
                })
                .ok_or_else(|| invalid("invalid grabbed fiber"))
        })
        .transpose()?;
    let mut grab_lambda = Vec3::ZERO;
    let grab = config
        .grab
        .zip(grab_index)
        .map(|(g, i)| g.damped(&model.fibers[i], config.dt));
    let mut next = model.clone();
    let mut lambdas = vec![[0.0; 6]; next.bonds.len()];
    let mut ground_lambdas: Vec<_> = next
        .fibers
        .iter()
        .map(|f| vec![0.0; f.shape.vertices().len()])
        .collect();
    for fiber in &mut next.fibers {
        predict(fiber, config)?;
    }
    let predicted: Vec<_> = next.fibers.iter().map(|f| (f.center(), f.motion)).collect();
    let mut contacts = config
        .contact_margin
        .map(|margin| {
            super::contact::Contacts::detect(&next, margin, config.contact_exclusion_hops)
        })
        .transpose()?;
    let iterations = config
        .coupled
        .map_or(config.iterations, |c| c.nonlinear_iterations);
    for iteration in 0..iterations {
        if config.coupled.is_some() {
            super::coupled::project(&mut next, &mut lambdas, config.dt, grab, &mut grab_lambda)?;
        } else {
            for group in &next.colors {
                for &j in group {
                    let bond = &next.bonds[j];
                    if !bond.intact() {
                        continue;
                    }
                    let [ai, bi] = bond.fibers.map(|id| next.indices[&id]);
                    let (a, b) = pair_mut(&mut next.fibers, ai, bi);
                    for k in 0..6 {
                        let axis = if iteration % 2 == 0 { k } else { 5 - k };
                        let row = bond_rows(bond, a, b)[axis];
                        project(row, a, b, &mut lambdas[j][axis], config.dt)?;
                    }
                }
            }
        }
        for _ in 0..config.iterations.div_ceil(iterations) {
            if let Some(contacts) = &mut contacts {
                contacts.project(&mut next)?;
            }
            if let Some(height) = config.ground_height {
                for (fiber, lambda) in next.fibers.iter_mut().zip(&mut ground_lambdas) {
                    project_ground(fiber, height, lambda)?;
                }
            }
            if config.coupled.is_none()
                && let (Some(grab), Some(index)) = (grab, grab_index)
            {
                let f = &mut next.fibers[index];
                let alpha = grab.compliance_at(f.motion.pose.transform_point(grab.local_point))
                    / (config.dt * config.dt);
                for (k, axis) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
                    let r = f.motion.pose.rotation * (grab.local_point - f.mass.local_com);
                    let error = (f.center() + r - grab.target).dot(axis);
                    let angular = r.cross(axis);
                    let inverse = f.world_inverse_inertia()?;
                    let w = f.inverse_mass() + angular.dot(inverse * angular);
                    let mut dl = (-error - alpha * grab_lambda[k]) / (w + alpha);
                    let angle = (inverse * angular).length() * dl.abs();
                    if angle > 0.2 {
                        dl *= 0.2 / angle;
                    }
                    grab_lambda[k] += dl;
                    let center = f.center() + axis * (f.inverse_mass() * dl);
                    rotate(f, inverse * angular * dl, center);
                }
            }
        }
    }
    if contacts.is_some()
        && let Some(height) = config.ground_height
    {
        for _ in 0..8 {
            for (fiber, lambda) in next.fibers.iter_mut().zip(&mut ground_lambdas) {
                project_ground(fiber, height, lambda)?;
            }
        }
    }
    for (fiber, (center, motion)) in next.fibers.iter_mut().zip(predicted) {
        if fiber.pin.is_some() {
            continue;
        }
        fiber.motion.linear_velocity += (fiber.center() - center) / config.dt;
        let delta = fiber.motion.pose.rotation * motion.pose.rotation.conjugate();
        fiber.motion.angular_velocity += rotation_vector(delta) / config.dt;
    }
    let mut report = StepReport::default();
    for (j, bond) in next.bonds.iter_mut().enumerate() {
        if !bond.intact() {
            continue;
        }
        let [ai, bi] = bond.fibers.map(|id| next.indices[&id]);
        let (a, b) = pair_mut(&mut next.fibers, ai, bi);
        let rows = bond_rows(bond, a, b);
        report.elastic_energy += rows
            .iter()
            .map(|r| 0.5 * r.value * r.value / r.compliance)
            .sum::<Real>();
        let force =
            Vec3::new(lambdas[j][0], lambdas[j][1], lambdas[j][2]) / (config.dt * config.dt);
        let torque =
            Vec3::new(lambdas[j][3], lambdas[j][4], lambdas[j][5]) / (config.dt * config.dt);
        report.bond_loads.push((bond.id, force, torque));
        let section = (1.0 - bond.damage).max(1e-6);
        let utilization = (force / (bond.material.force_limit * section))
            .length()
            .max((torque / (bond.material.torque_limit * section)).length());
        if !force.is_finite() || !torque.is_finite() || !utilization.is_finite() {
            return Err(invalid("bond loading overflow"));
        }
        bond.damage = (bond.damage
            + config.dt * bond.material.damage_rate * (utilization - 1.0).max(0.0))
        .min(1.0);
        if !bond.intact() {
            report.broken_bonds.push(bond.id);
            continue;
        }
    }
    for group in &next.colors {
        for &j in group {
            let bond = &next.bonds[j];
            if !bond.intact() {
                continue;
            }
            let [ai, bi] = bond.fibers.map(|id| next.indices[&id]);
            let (a, b) = pair_mut(&mut next.fibers, ai, bi);
            let factor = 1.0 - (-bond.material.damping * config.dt).exp();
            for row in bond_rows(bond, a, b) {
                damp(row, a, b, factor)?;
            }
        }
    }
    for (fiber, lambdas) in next.fibers.iter_mut().zip(ground_lambdas) {
        if fiber.pin.is_some() || config.friction == 0.0 {
            continue;
        }
        for (index, lambda) in lambdas.into_iter().enumerate() {
            if lambda == 0.0 {
                continue;
            }
            let r = fiber
                .motion
                .pose
                .transform_point(fiber.shape.vertices()[index])
                - fiber.center();
            let velocity = fiber.motion.linear_velocity + fiber.motion.angular_velocity.cross(r);
            let tangent = Vec3::new(velocity.x, 0.0, velocity.z);
            if tangent.length_squared() == 0.0 {
                continue;
            }
            let n = tangent.normalize();
            let inertia = fiber.world_inverse_inertia()?;
            let w = fiber.inverse_mass() + r.cross(n).dot(inertia * r.cross(n));
            let impulse = -n * (tangent.length() / w).min(config.friction * lambda / config.dt);
            fiber.motion.linear_velocity += impulse * fiber.inverse_mass();
            fiber.motion.angular_velocity += inertia * r.cross(impulse);
        }
    }
    if let Some(contacts) = &contacts {
        contacts.velocities(
            &mut next,
            config.friction,
            config.dt,
            config.pair_restitution,
        )?;
    }
    if !report.elastic_energy.is_finite() {
        return Err(invalid("nonfinite elastic energy"));
    }
    for fiber in &next.fibers {
        fiber.validate()?;
    }
    next.advance_revision()?;
    *model = next;
    Ok(report)
}
