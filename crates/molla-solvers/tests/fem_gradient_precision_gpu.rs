#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_math::{Mat3, Vec3};
use molla_sim::{ModelBuilder, TetViscosity};
use molla_solvers::VbdSolver;

#[path = "common/fem_gradient_fixtures.rs"]
mod fixtures;
#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

fn scalar(bits: u32) -> f64 {
    f32::from_bits(bits) as f64
}

fn vectors(bits: &[u32]) -> Vec<Vec3> {
    bits.chunks_exact(4)
        .map(|v| Vec3::new(scalar(v[0]), scalar(v[1]), scalar(v[2])))
        .collect()
}

fn projected(v: Vec3) -> Vec3 {
    Vec3::from_array(v.to_array().map(|x| x as f32 as f64))
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn cancelling_tet_gradients_match_f64_force() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (case, fixture) in fixtures::CASES.iter().enumerate() {
        let rest = vectors(fixture.rest);
        let q = vectors(fixture.positions);
        let velocity = vectors(fixture.velocities);
        let target = vectors(fixture.targets);
        let inverse_mass: Vec<_> = fixture.inverse_mass.iter().map(|v| scalar(*v)).collect();
        let tets: Vec<[u32; 4]> = fixture
            .tets
            .chunks_exact(4)
            .map(|v| v.try_into().unwrap())
            .collect();
        let columns = vectors(fixture.inverse_rest);
        let inverse: Vec<_> = columns
            .chunks_exact(3)
            .map(|c| Mat3::from_cols(c[0], c[1], c[2]))
            .collect();
        let h = fixture.dt as f64;
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        builder.set_linear_damping(0.0);
        for (&p, &w) in rest.iter().zip(&inverse_mass) {
            builder.add_particle(p, 1.0 / w, 0.0, WorldId(0));
        }
        builder.add_tet_mesh(&tets, 2e6, 0.45, None).unwrap();
        let mut model = builder.build().unwrap();
        model
            .tet_dm_inv
            .host_mut()
            .unwrap()
            .copy_from_slice(&inverse);
        for t in 0..tets.len() {
            model.tet_rest_volume.host_mut().unwrap()[t] = scalar(fixture.volume[t]);
            model.tet_mu.host_mut().unwrap()[t] = scalar(fixture.mu[t]);
            model.tet_lambda.host_mut().unwrap()[t] = scalar(fixture.lambda[t]);
            model.tet_viscosity.host_mut().unwrap()[t] = TetViscosity {
                shear: scalar(fixture.viscosity[t * 2]),
                bulk: scalar(fixture.viscosity[t * 2 + 1]),
            };
        }
        let mut input = model.state().unwrap();
        let mut output = model.state().unwrap();
        input.particle_q.host_mut().unwrap().copy_from_slice(&q);
        output
            .particle_qd
            .host_mut()
            .unwrap()
            .copy_from_slice(&velocity);
        let incoming: Vec<_> = target.iter().map(|y| projected(*y / h)).collect();
        input
            .particle_qd
            .host_mut()
            .unwrap()
            .copy_from_slice(&incoming);
        let delta: Vec<_> = velocity.iter().map(|v| projected(*v * h)).collect();
        let target: Vec<_> = incoming.iter().map(|v| projected(*v * h)).collect();
        let node = fixture.node;
        let mut force = (target[node] - delta[node]) / (inverse_mass[node] * h * h);
        for (t, tet) in tets.iter().enumerate() {
            let Some(role) = tet.iter().position(|i| *i as usize == node) else {
                continue;
            };
            let origin = tet[0] as usize;
            let edge = |i: usize| {
                (q[tet[i] as usize] - q[origin]) - (rest[tet[i] as usize] - rest[origin])
            };
            let previous = Mat3::IDENTITY + Mat3::from_cols(edge(1), edge(2), edge(3)) * inverse[t];
            let increment = Mat3::from_cols(
                delta[tet[1] as usize] - delta[origin],
                delta[tet[2] as usize] - delta[origin],
                delta[tet[3] as usize] - delta[origin],
            ) * inverse[t];
            let f = previous + increment;
            let cof = Mat3::from_cols(
                f.y_axis.cross(f.z_axis),
                f.z_axis.cross(f.x_axis),
                f.x_axis.cross(f.y_axis),
            );
            let mu = scalar(fixture.mu[t]);
            let lam = scalar(fixture.lambda[t]);
            let mut stress = (f - cof) * mu + cof * ((lam + mu) * (f.determinant() - 1.0));
            let strain = (previous.transpose() * increment
                + increment.transpose() * previous
                + increment.transpose() * increment)
                * 0.5;
            let trace = strain.x_axis.x + strain.y_axis.y + strain.z_axis.z;
            let viscous_stress = (strain - Mat3::IDENTITY * (trace / 3.0))
                * (2.0 * scalar(fixture.viscosity[2 * t]))
                + Mat3::IDENTITY * (trace * scalar(fixture.viscosity[2 * t + 1]));
            stress += f * viscous_stress / h;
            let rows = inverse[t].transpose();
            let gradients = [
                -(rows.x_axis + rows.y_axis + rows.z_axis),
                rows.x_axis,
                rows.y_axis,
                rows.z_axis,
            ];
            force -= stress * gradients[role] * scalar(fixture.volume[t]);
        }
        model.upload_to_device(device, queue).unwrap();
        input.upload_to_device(device, queue).unwrap();
        output.upload_to_device(device, queue).unwrap();
        let mut solver = VbdSolver::new_gpu(&engine, &model).unwrap();
        let buffer = solver
            .material_force_residual_gpu(&model, &input, &mut output, h)
            .unwrap();
        let samples = readback::read::<[f32; 4]>(&engine, buffer, q.len());
        assert_eq!(
            samples[node][..3],
            target[node].to_array().map(|v| v as f32)
        );
        let expected = force.length() * h * inverse_mass[node];
        let error = (samples[node][3] as f64 - expected).abs();
        eprintln!(
            "case={case} GPU={} oracle={expected} error={error}",
            samples[node][3]
        );
        assert!(
            error < 3e-5,
            "case={case}: GPU/oracle force mismatch {error}"
        );
    }
}
