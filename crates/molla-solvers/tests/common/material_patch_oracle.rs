use super::readback;
use molla_compute::Engine;
use molla_kernels::soft_rigid_contact::SoftRigidContactGpu;
use molla_math::{Mat3, Quat, Vec3};
use molla_sim::Model;
use molla_solvers::fem_rigid_gpu::FemRigidGpuSystem;

fn projected(v: Vec3) -> Vec3 {
    Vec3::from_array(v.to_array().map(|x| x as f32 as f64))
}
fn vector(v: [f32; 4]) -> Vec3 {
    Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
}
fn scalar(x: f64) -> f64 {
    x as f32 as f64
}

pub struct Reference {
    rest: Vec<Vec3>,
    inverse: Mat3,
    volume: f64,
    mu: f64,
    lambda: f64,
    shear: f64,
    bulk: f64,
}

impl Reference {
    pub fn new(model: &Model) -> Self {
        assert_eq!(model.tet_count, 1);
        assert_eq!(model.particle_count, 4);
        assert_eq!(model.spring_count, 0);
        assert_eq!(model.tet_indices.host().unwrap(), [0, 1, 2, 3]);
        let inverse = model.tet_dm_inv.host().unwrap()[0];
        let viscosity = model.tet_viscosity.host().unwrap()[0];
        Self {
            rest: model
                .initial_particle_q
                .host()
                .unwrap()
                .iter()
                .copied()
                .map(projected)
                .collect(),
            inverse: Mat3::from_cols(
                projected(inverse.x_axis),
                projected(inverse.y_axis),
                projected(inverse.z_axis),
            ),
            volume: scalar(model.tet_rest_volume.host().unwrap()[0]),
            mu: scalar(model.tet_mu.host().unwrap()[0]),
            lambda: scalar(model.tet_lambda.host().unwrap()[0]),
            shear: scalar(viscosity.shear),
            bulk: scalar(viscosity.bulk),
        }
    }

    pub fn check(&self, engine: &Engine, system: &FemRigidGpuSystem, prior_velocity: &[[f32; 4]]) {
        let snapshot = system.contact_snapshot().unwrap();
        let settings = snapshot.settings().unwrap();
        let h = settings.dt as f64;
        let buffer = |name| {
            snapshot
                .buffers()
                .chain(snapshot.solved_buffers())
                .find(|(key, _)| *key == name)
                .unwrap()
                .1
        };
        let vectors = |name, count| {
            readback::read::<[f32; 4]>(engine, buffer(name), count)
                .into_iter()
                .map(vector)
                .collect::<Vec<_>>()
        };
        let q = vectors("positions", 4);
        let velocity = vectors("solved_velocities", 4);
        let inverse_mass = readback::read::<f32>(engine, buffer("inverse_mass"), 4);
        let contacts = readback::read::<SoftRigidContactGpu>(
            engine,
            buffer("contacts"),
            system.contact_row_count() as usize,
        );
        let rows =
            readback::read::<[f32; 8]>(engine, system.contact_row_diagnostics(), contacts.len());
        let jacobian = readback::read::<f32>(engine, buffer("jacobians"), 6);
        let linear = Vec3::new(jacobian[0] as f64, jacobian[1] as f64, jacobian[2] as f64);
        let angular = Vec3::new(jacobian[3] as f64, jacobian[4] as f64, jacobian[5] as f64);
        let pose = readback::read::<[f32; 4]>(engine, buffer("body_poses"), 2);
        let rotation = Quat::from_array(pose[1].map(f64::from));
        let center = vector(pose[0]) + rotation * vectors("body_com", 1)[0];
        let mass = readback::read::<f32>(engine, buffer("mass_matrix"), 1)[0] as f64;
        let rigid_before = readback::read::<f32>(engine, buffer("rigid_velocities"), 1)[0] as f64;
        let rigid = readback::read::<f32>(engine, buffer("solved_rigid_velocities"), 1)[0] as f64;
        let mut impulses = [Vec3::ZERO; 4];
        let mut reaction = 0.0;
        let mut worst_contact = 0.0_f64;
        let mut active = 0;
        for (contact, row) in contacts.iter().zip(&rows) {
            if contact.ids[3] == 0 {
                continue;
            }
            assert_eq!(contact.ids[3], 1);
            assert_eq!(contact.ids[1], 0);
            active += 1;
            let normal = vector(contact.normal_gap);
            let tangent = normal
                .cross(if normal.x.abs() < 0.8 {
                    Vec3::X
                } else {
                    Vec3::Y
                })
                .normalize();
            let basis = Mat3::from_cols(normal, tangent, normal.cross(tangent));
            let impulse = Vec3::new(row[4] as f64, row[5] as f64, row[6] as f64);
            assert!(
                impulse.x >= -1e-12
                    && impulse.y.hypot(impulse.z) <= contact.material[0] as f64 * impulse.x + 1e-9
            );
            let world = basis * impulse;
            let nodes = [contact.ids[0], contact.support[0], contact.support[1]];
            let mut point = Vec3::ZERO;
            let mut relative = Vec3::ZERO;
            let mut soft_mass = 0.0;
            for j in 0..contact.support[2] as usize {
                let node = nodes[j] as usize;
                let weight = contact.weights[j] as f64;
                point += weight * q[node];
                relative += weight * velocity[node];
                soft_mass += weight * weight * inverse_mass[node] as f64;
                impulses[node] += weight * world;
            }
            let point_jacobian = linear + angular.cross(point - center);
            reaction -= point_jacobian.dot(world);
            relative = basis.transpose() * (relative - point_jacobian * rigid);
            let g = basis.transpose() * point_jacobian;
            let normal_mass = soft_mass + g.x * g.x / mass;
            let tangent_mass = soft_mass + (g.y * g.y + g.z * g.z) / mass;
            assert!((normal_mass - row[0] as f64).abs() < 1e-5 * normal_mass);
            assert!((tangent_mass - row[1] as f64).abs() < 1e-5 * tangent_mass);
            let gap = contact.normal_gap[3] as f64;
            let target = if gap < 0.0 {
                (-gap * settings.penetration_recovery as f64 / h)
                    .min(settings.max_recovery_speed as f64)
            } else {
                -gap / h
            };
            let normal_error =
                ((impulse.x + (target - relative.x) / normal_mass).max(0.0) - impulse.x).abs()
                    * normal_mass;
            let trial = Vec3::new(
                0.0,
                impulse.y - relative.y / tangent_mass,
                impulse.z - relative.z / tangent_mass,
            );
            let trial = trial
                * (contact.material[0] as f64 * impulse.x / trial.length().max(1e-30)).min(1.0);
            let tangent_error =
                (trial - Vec3::new(0.0, impulse.y, impulse.z)).length() * tangent_mass;
            worst_contact = worst_contact.max(normal_error).max(tangent_error);
        }
        assert_eq!(active, 3);
        let rigid_error = mass * (rigid - rigid_before) - reaction;
        assert!(rigid_error.abs() < 1e-6 && (rigid_error / mass).abs() < 1e-4);
        let targets =
            readback::read::<[f32; 4]>(engine, system.material_force_samples().unwrap(), 4);
        let delta: Vec<_> = velocity.iter().map(|v| projected(h * *v)).collect();
        let edges = |p: &[Vec3]| Mat3::from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0]);
        let displacement: Vec<_> = q.iter().zip(&self.rest).map(|(q, r)| *q - *r).collect();
        let previous = Mat3::IDENTITY + edges(&displacement) * self.inverse;
        let increment = edges(&delta) * self.inverse;
        let current = Mat3::IDENTITY + (edges(&displacement) + edges(&delta)) * self.inverse;
        let cofactor = Mat3::from_cols(
            current.y_axis.cross(current.z_axis),
            current.z_axis.cross(current.x_axis),
            current.x_axis.cross(current.y_axis),
        );
        let j = current.determinant();
        assert!(j > 0.5);
        let strain = (previous.transpose() * increment
            + increment.transpose() * previous
            + increment.transpose() * increment)
            * 0.5;
        let trace = strain.x_axis.x + strain.y_axis.y + strain.z_axis.z;
        let viscous = (previous + increment)
            * ((strain - Mat3::IDENTITY * (trace / 3.0)) * (2.0 * self.shear)
                + Mat3::IDENTITY * (self.bulk * trace))
            / h;
        let stress = (current - cofactor) * self.mu
            + cofactor * ((self.lambda + self.mu) * (j - 1.0))
            + viscous;
        let gradients = [
            -self.inverse.row(0) - self.inverse.row(1) - self.inverse.row(2),
            self.inverse.row(0),
            self.inverse.row(1),
            self.inverse.row(2),
        ];
        let mut worst_material = 0.0_f64;
        let mut worst_target = 0.0_f64;
        for i in 0..4 {
            let target = vector(targets[i]);
            let inv = inverse_mass[i] as f64;
            worst_target = worst_target
                .max((target / h - vector(prior_velocity[i]) - inv * impulses[i]).length());
            let force = -(stress * gradients[i]) * self.volume;
            worst_material =
                worst_material.max((h * inv * force + (target - delta[i]) / h).length());
        }
        println!(
            "ORACLE material={worst_material} contact={worst_contact} target={worst_target} rigid_impulse={rigid_error} J={j}"
        );
        assert!(worst_material <= 1e-4 && worst_contact <= 1e-4 && worst_target <= 1e-4);
    }
}
