//! Substep Coulomb impulses using articulated tangential effective mass.

use crate::{
    articulated_response::ArticulatedResponse,
    penalty_contact::{ContactForce, FrictionRegime},
};
use molla_core::{Error, Result};
use molla_math::{Real, SpatialVector, Vec3};
use molla_sim::{BodyFlags, Model, State};
use std::collections::{BTreeMap, BTreeSet};

const CAPTURE_SPEED: Real = 1e-4;

pub(crate) struct Contact {
    pub reading: usize,
    pub directions: [Vec3; 2],
    pub ends: [Vec<(usize, SpatialVector)>; 2],
    pub rates: [Real; 2],
    pub moving: bool,
}

fn root(parents: &[usize], mut i: usize) -> usize {
    while parents[i] != i {
        i = parents[i];
    }
    i
}

fn solve_block(a: [Real; 3], v: [Real; 2], shift: Real) -> [Real; 2] {
    let scale = a[0].max(a[2]).max(shift);
    if scale == 0.0 {
        return [0.0; 2];
    }
    let x = (a[0] + shift) / scale;
    let z = (a[2] + shift) / scale;
    let y = a[1] / scale;
    let determinant = x * z - y * y;
    if determinant <= 0.0 {
        return [Real::INFINITY; 2];
    }
    [
        (-z * v[0] + y * v[1]) / scale / determinant,
        (y * v[0] - x * v[1]) / scale / determinant,
    ]
}

pub(crate) fn impulse(
    a: [Real; 3],
    v: [Real; 2],
    static_cap: Real,
    dynamic_cap: Real,
) -> ([Real; 2], bool) {
    let scale = a[0].max(a[2]);
    let trial = solve_block(a, v, scale * 1e-12);
    let residual = [
        v[0] + a[0] * trial[0] + a[1] * trial[1],
        v[1] + a[1] * trial[0] + a[2] * trial[1],
    ];
    if trial[0].hypot(trial[1]) <= static_cap && residual[0].hypot(residual[1]) <= 1e-9 {
        return (trial, true);
    }
    if dynamic_cap == 0.0 {
        return ([0.0; 2], false);
    }
    let mut low = 0.0;
    let mut high = (v[0].hypot(v[1]) / dynamic_cap).max(scale * 1e-12);
    for _ in 0..64 {
        let value = solve_block(a, v, high);
        if value[0].hypot(value[1]) <= dynamic_cap {
            break;
        }
        high *= 2.0;
    }
    for _ in 0..64 {
        let middle = 0.5 * low + 0.5 * high;
        let value = solve_block(a, v, middle);
        if value[0].hypot(value[1]) > dynamic_cap {
            low = middle;
        } else {
            high = middle;
        }
    }
    (solve_block(a, v, high), false)
}

pub(crate) fn disks(gram: &[Real], free: &[Real], caps: &[Real]) -> Result<(Vec<Real>, Vec<bool>)> {
    let n = free.len();
    let scales: Vec<_> = caps.iter().flat_map(|&c| [c, c]).collect();
    let mut matrix = vec![0.0; n * n];
    let mut force = vec![0.0; n];
    for i in 0..n {
        force[i] = free[i] * scales[i];
        for j in 0..n {
            matrix[i * n + j] = gram[i * n + j] * scales[i] * scales[j];
        }
    }
    let scale = matrix
        .iter()
        .chain(&force)
        .copied()
        .map(Real::abs)
        .fold(0.0, Real::max);
    if scale == 0.0 {
        return Ok((
            vec![0.0; n],
            free.chunks_exact(2)
                .map(|v| v[0].hypot(v[1]) <= 1e-7)
                .collect(),
        ));
    }
    for v in &mut matrix {
        *v /= scale;
    }
    for v in &mut force {
        *v /= scale;
    }
    let objective = |x: &[Real], barrier: Real| {
        let mut value = 0.0;
        for i in 0..n {
            value +=
                x[i] * (force[i] + 0.5 * (0..n).map(|j| matrix[i * n + j] * x[j]).sum::<Real>());
        }
        for c in 0..caps.len() {
            let slack = 1.0 - x[2 * c] * x[2 * c] - x[2 * c + 1] * x[2 * c + 1];
            if slack <= 0.0 {
                return Real::INFINITY;
            }
            value -= barrier * slack.ln();
        }
        value
    };
    let mut x = vec![0.0; n];
    let mut barrier = 0.01;
    for _ in 0..11 {
        for _ in 0..60 {
            let mut gradient: Vec<_> = (0..n)
                .map(|i| force[i] + (0..n).map(|j| matrix[i * n + j] * x[j]).sum::<Real>())
                .collect();
            let mut hessian = matrix.clone();
            for c in 0..caps.len() {
                let r = 2 * c;
                let slack = 1.0 - x[r] * x[r] - x[r + 1] * x[r + 1];
                for i in 0..2 {
                    gradient[r + i] += 2.0 * barrier * x[r + i] / slack;
                    hessian[(r + i) * n + r + i] += 2.0 * barrier / slack;
                    for j in 0..2 {
                        hessian[(r + i) * n + r + j] +=
                            4.0 * barrier * x[r + i] * x[r + j] / (slack * slack);
                    }
                }
            }
            let delta = crate::rigid_impulse::solve_mass(
                &hessian,
                &gradient.iter().map(|v| -v).collect::<Vec<_>>(),
                n,
            )?;
            let descent: Real = gradient.iter().zip(&delta).map(|(a, b)| a * b).sum();
            if -descent < barrier * 1e-3 {
                break;
            }
            let old = objective(&x, barrier);
            let mut step = 1.0;
            let mut accepted = false;
            for _ in 0..60 {
                let next: Vec<_> = x.iter().zip(&delta).map(|(x, d)| x + step * d).collect();
                if objective(&next, barrier) <= old + 1e-4 * step * descent {
                    x = next;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                break;
            }
        }
        barrier *= 0.1;
    }
    let impulses: Vec<_> = x.iter().zip(&scales).map(|(x, s)| x * s).collect();
    let mut sticking = Vec::new();
    let mut error: Real = 0.0;
    for c in 0..caps.len() {
        let r = 2 * c;
        let v = [
            free[r] + (0..n).map(|j| gram[r * n + j] * impulses[j]).sum::<Real>(),
            free[r + 1]
                + (0..n)
                    .map(|j| gram[(r + 1) * n + j] * impulses[j])
                    .sum::<Real>(),
        ];
        let speed = v[0].hypot(v[1]);
        sticking.push(speed <= 1e-7);
        if speed <= 1e-7 || caps[c] == 0.0 {
            continue;
        }
        let norm = impulses[r].hypot(impulses[r + 1]);
        if norm == 0.0 {
            error = error.max(speed);
            continue;
        }
        let direction = [impulses[r] / norm, impulses[r + 1] / norm];
        error = error.max((v[0] + speed * direction[0]).hypot(v[1] + speed * direction[1]));
        error = error.max(speed * (1.0 - norm / caps[c]).abs());
        let curvature = (gram[r * n + r].abs() + gram[r * n + r + 1].abs())
            .max(gram[(r + 1) * n + r + 1].abs() + gram[r * n + r + 1].abs());
        error = error.max(speed.min(curvature * (caps[c] - norm).abs()));
    }
    if error > 1e-7 {
        return Err(Error::Solver(format!(
            "friction disk KKT residual {error:e} exceeds 1e-7"
        )));
    }
    Ok((impulses, sticking))
}

pub(crate) fn prepare(
    model: &Model,
    input: &State,
    readings: &[ContactForce],
    dt: Real,
) -> Result<(Vec<Option<usize>>, BTreeMap<usize, Vec<Contact>>)> {
    let boundary_delta = crate::kinematic_motion::velocity_delta(model, input, dt)?;
    let flags = model.body_flags.host()?;
    let children = model.joint_child.host()?;
    let starts = if model.articulation_count == 0 {
        vec![0, model.joint_count as u32]
    } else {
        model.articulation_start.host()?.to_vec()
    };
    let mut body_art = vec![None; model.body_count];
    for art in 0..starts.len() - 1 {
        for &body in &children[starts[art] as usize..starts[art + 1] as usize] {
            if flags[body as usize] & BodyFlags::IMMOVABLE == 0 {
                body_art[body as usize] = Some(art);
            }
        }
    }
    let mut parents: Vec<_> = (0..starts.len() - 1).collect();
    let poses = input.body_q.host()?;
    let velocities = input.body_qd.host()?;
    let centers = model.body_com.host()?;
    let mut contacts = Vec::new();
    for (index, reading) in readings.iter().enumerate() {
        if reading.normal_force <= 0.0 || reading.static_friction == 0.0 {
            continue;
        }
        let normal = reading.normal;
        let basis = if normal.x.abs() < 0.6 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let tangent = normal.cross(basis).normalize();
        let directions = [tangent, normal.cross(tangent)];
        let mut contact = Contact {
            reading: index,
            directions,
            ends: [Vec::new(), Vec::new()],
            rates: [0.0; 2],
            moving: false,
        };
        let mut arts = Vec::new();
        for side in 0..2 {
            let sign = if side == 0 { -1.0 } else { 1.0 };
            let body = reading.bodies[side].map(|b| b.index());
            let mut belt = reading.colliders[side]
                .and_then(|s| model.shape_surface_velocity.get(s.index()).copied())
                .unwrap_or(Vec3::ZERO);
            if let Some(body) = body {
                belt = poses[body].rotation * belt;
                let arm =
                    reading.point - (poses[body].position + poses[body].rotation * centers[body]);
                for row in 0..2 {
                    let wrench = SpatialVector::new(
                        directions[row] * sign,
                        arm.cross(directions[row] * sign),
                    );
                    if flags[body] & BodyFlags::FIXED == 0 {
                        contact.rates[row] += wrench.linear.dot(velocities[body].linear)
                            + wrench.angular.dot(velocities[body].angular);
                    }
                    contact.ends[row].push((body, wrench));
                }
                if let Some(art) = body_art[body] {
                    arts.push(art);
                }
            }
            for row in 0..2 {
                contact.rates[row] += sign * directions[row].dot(belt);
            }
        }
        contact.moving = contact.rates[0].hypot(contact.rates[1]) > CAPTURE_SPEED;
        for row in 0..2 {
            contact.rates[row] +=
                crate::kinematic_motion::project(&boundary_delta, &contact.ends[row]);
        }
        if let Some(&first) = arts.first() {
            for &art in &arts[1..] {
                let a = root(&parents, first);
                let b = root(&parents, art);
                parents[b] = a;
            }
            contacts.push((first, contact));
        }
    }
    let mut groups = BTreeMap::<usize, Vec<Contact>>::new();
    for (art, contact) in contacts {
        groups.entry(root(&parents, art)).or_default().push(contact);
    }
    Ok((body_art, groups))
}

pub(crate) fn apply(
    model: &Model,
    input: &State,
    rates: &mut [Real],
    dt: Real,
    readings: &mut [ContactForce],
) -> Result<bool> {
    if !readings
        .iter()
        .any(|r| r.normal_force > 0.0 && r.static_friction > 0.0)
    {
        return Ok(false);
    }
    let (body_art, groups) = prepare(model, input, readings, dt)?;
    let offsets = model.joint_dof_offset.host()?;
    let starts = if model.articulation_count == 0 {
        vec![0, model.joint_count as u32]
    } else {
        model.articulation_start.host()?.to_vec()
    };
    let changed = !groups.is_empty();
    let mut response = ArticulatedResponse::new(model, input);
    for contacts in groups.into_values() {
        let count = 2 * contacts.len();
        let mut gram = vec![
            0.0;
            count.checked_mul(count).ok_or_else(|| Error::Solver(
                "friction row capacity overflow".into()
            ))?
        ];
        let mut free: Vec<_> = contacts.iter().flat_map(|c| c.rates).collect();
        let arts: BTreeSet<_> = contacts
            .iter()
            .flat_map(|c| &c.ends[0])
            .filter_map(|(body, _)| body_art[*body])
            .collect();
        let mut projections = Vec::new();
        for art in arts {
            let first = offsets[starts[art] as usize] as usize;
            let n = offsets[starts[art + 1] as usize] as usize - first;
            if n == 0 {
                continue;
            }
            let mut rows = vec![vec![0.0; n]; count];
            for (c, contact) in contacts.iter().enumerate() {
                for row in 0..2 {
                    for &(body, wrench) in &contact.ends[row] {
                        if body_art[body] != Some(art) {
                            continue;
                        }
                        let j = response.jacobian(art, body)?;
                        for k in 0..n {
                            rows[2 * c + row][k] += wrench.linear.x * j[k]
                                + wrench.linear.y * j[n + k]
                                + wrench.linear.z * j[2 * n + k]
                                + wrench.angular.x * j[3 * n + k]
                                + wrench.angular.y * j[4 * n + k]
                                + wrench.angular.z * j[5 * n + k];
                        }
                    }
                    for k in 0..n {
                        free[2 * c + row] += rows[2 * c + row][k]
                            * (rates[first + k] - input.joint_qd.host()?[first + k]);
                    }
                }
            }
            response.add_gram(art, n, &rows, &mut gram)?;
            projections.push((art, first, rows));
        }
        let mut impulses = crate::rigid_impulse::solve_mass(
            &gram,
            &free.iter().map(|v| -v).collect::<Vec<_>>(),
            count,
        )?;
        let mut sticking = vec![false; contacts.len()];
        let mut dynamic: Vec<_> = contacts.iter().map(|c| c.moving).collect();
        let cap = |c: usize, sliding: bool| {
            let r = &readings[contacts[c].reading];
            r.normal_force
                * dt
                * if sliding {
                    r.friction
                } else {
                    r.static_friction
                }
        };
        let mut converged = false;
        let slip = |c: usize, impulses: &[Real]| {
            let r = 2 * c;
            (free[r]
                + (0..count)
                    .map(|k| gram[r * count + k] * impulses[k])
                    .sum::<Real>())
            .hypot(
                free[r + 1]
                    + (0..count)
                        .map(|k| gram[(r + 1) * count + k] * impulses[k])
                        .sum::<Real>(),
            )
        };
        let exact = (0..contacts.len())
            .all(|c| impulses[2 * c].hypot(impulses[2 * c + 1]) <= cap(c, dynamic[c]))
            && (0..count).all(|r| {
                (free[r]
                    + (0..count)
                        .map(|k| gram[r * count + k] * impulses[k])
                        .sum::<Real>())
                .abs()
                    <= 1e-8
            });
        if exact {
            sticking.fill(true);
            converged = true;
        } else {
            impulses.fill(0.0);
        }
        for _ in 0..200 {
            if converged {
                break;
            }
            for c in 0..contacts.len() {
                let row = 2 * c;
                let a = [
                    gram[row * count + row],
                    0.5 * (gram[row * count + row + 1] + gram[(row + 1) * count + row]),
                    gram[(row + 1) * count + row + 1],
                ];
                let mut v = [free[row], free[row + 1]];
                for k in 0..count {
                    if k == row || k == row + 1 {
                        continue;
                    }
                    v[0] += gram[row * count + k] * impulses[k];
                    v[1] += gram[(row + 1) * count + k] * impulses[k];
                }
                let limit = cap(c, dynamic[c]);
                let (next, stuck) = impulse(a, v, limit, limit);
                if next.iter().any(|v| !v.is_finite()) {
                    return Err(Error::Solver("nonfinite friction impulse".into()));
                }
                sticking[c] = stuck;
                impulses[row..row + 2].copy_from_slice(&next);
            }
            let mut residual: Real = 0.0;
            for c in 0..contacts.len() {
                let r = 2 * c;
                let v = [
                    free[r]
                        + (0..count)
                            .map(|k| gram[r * count + k] * impulses[k])
                            .sum::<Real>(),
                    free[r + 1]
                        + (0..count)
                            .map(|k| gram[(r + 1) * count + k] * impulses[k])
                            .sum::<Real>(),
                ];
                if sticking[c] {
                    residual = residual.max(v[0].hypot(v[1]));
                } else if cap(c, dynamic[c]) > 0.0 {
                    let length = impulses[r].hypot(impulses[r + 1]);
                    if length == 0.0 {
                        residual = residual.max(v[0].hypot(v[1]));
                        continue;
                    }
                    let direction = [impulses[r] / length, impulses[r + 1] / length];
                    let projection = (-(v[0] * direction[0] + v[1] * direction[1])).max(0.0);
                    residual = residual.max(
                        (v[0] + projection * direction[0]).hypot(v[1] + projection * direction[1]),
                    );
                }
            }
            if residual <= 1e-7 {
                let mut transition = false;
                for c in 0..contacts.len() {
                    if !dynamic[c] && slip(c, &impulses) > CAPTURE_SPEED {
                        dynamic[c] = true;
                        transition = true;
                    }
                }
                if !transition {
                    converged = true;
                    break;
                }
            }
        }
        if !converged {
            for _ in 0..=contacts.len() {
                let caps: Vec<_> = (0..contacts.len()).map(|c| cap(c, dynamic[c])).collect();
                (impulses, sticking) = disks(&gram, &free, &caps)?;
                let mut transition = false;
                for c in 0..contacts.len() {
                    if !dynamic[c] && slip(c, &impulses) > CAPTURE_SPEED {
                        dynamic[c] = true;
                        transition = true;
                    }
                }
                if !transition {
                    converged = true;
                    break;
                }
            }
        }
        if !converged {
            return Err(Error::Solver(
                "friction regime transition did not converge".into(),
            ));
        }
        for (art, first, rows) in projections {
            let rhs: Vec<_> = (0..rows[0].len())
                .map(|k| (0..count).map(|r| rows[r][k] * impulses[r]).sum())
                .collect();
            let delta = response.solve(art, &rhs)?;
            for (k, value) in delta.into_iter().enumerate() {
                rates[first + k] += value;
            }
        }
        for (c, contact) in contacts.iter().enumerate() {
            let reading = &mut readings[contact.reading];
            reading.tangent_impulse = contact.directions[0] * impulses[2 * c]
                + contact.directions[1] * impulses[2 * c + 1];
            reading.force_on_b += reading.tangent_impulse / dt;
            reading.friction_regime = if !dynamic[c] || sticking[c] {
                FrictionRegime::Sticking
            } else {
                FrictionRegime::Sliding
            };
            reading.applied_friction = if dynamic[c] {
                reading.friction
            } else {
                reading.static_friction
            };
            reading.acted = reading.force_on_b.length_squared() > 0.0;
        }
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearly_loaded_static_disks_cancel_slip_across_scales_and_directions() {
        for scale in [1e-6, 1.0, 1e6] {
            let gram: Vec<_> = [
                1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            ]
            .into_iter()
            .map(|g| g / scale)
            .collect();
            for angle in [0.0_f64, 0.37, 1.2] {
                for load in [0.8, 0.99, 0.999, 0.9999] {
                    let free: Vec<_> = [0.005 * load * angle.cos(), 0.005 * load * angle.sin()]
                        .into_iter()
                        .cycle()
                        .take(4)
                        .collect();
                    let (impulses, _) =
                        disks(&gram, &free, &[0.001 * scale, 0.004 * scale]).unwrap();
                    for c in 0..2 {
                        let slip = [2 * c, 2 * c + 1].map(|r| {
                            free[r] + (0..4).map(|k| gram[r * 4 + k] * impulses[k]).sum::<f64>()
                        });
                        assert!(
                            slip[0].hypot(slip[1]) <= 1e-7,
                            "scale={scale} angle={angle} load={load} slip={slip:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn disk_solver_respects_circular_limits_and_dissipates_predicted_slip_energy() {
        for (matrix, velocity, caps) in [
            (vec![1.0, 0.2, 0.2, 10.0], vec![1.0, -0.6], vec![0.1]),
            (
                vec![
                    1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                ],
                vec![0.3, 0.4, 0.3, 0.4],
                vec![0.3, 0.1],
            ),
            (vec![0.0; 4], vec![1.0, 2.0], vec![0.2]),
        ] {
            let (impulses, _) = disks(&matrix, &velocity, &caps).unwrap();
            for c in 0..caps.len() {
                assert!(impulses[2 * c].hypot(impulses[2 * c + 1]) <= caps[c]);
            }
            let n = velocity.len();
            let energy: Real = (0..n)
                .map(|i| {
                    impulses[i]
                        * (velocity[i]
                            + 0.5
                                * (0..n)
                                    .map(|j| matrix[i * n + j] * impulses[j])
                                    .sum::<Real>())
                })
                .sum();
            assert!(energy <= 1e-12, "energy={energy:e}");
        }
    }
}
