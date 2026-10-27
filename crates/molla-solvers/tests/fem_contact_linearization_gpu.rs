#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_contact::{
    SoftRigidBallJoint, SoftRigidContactGpu, SoftRigidDofBound,
};
use molla_kernels::soft_rigid_solve::{
    ContactLinearizationBindings, ContactLinearizationKernel, SoftRigidSolveBindings,
    SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use molla_math::{Mat3, Vec3};
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, values: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}
fn v(x: [f32; 4]) -> Vec3 {
    Vec3::new(x[0] as f64, x[1] as f64, x[2] as f64)
}
fn storage_write<T: bytemuck::Pod>(engine: &Engine, target: &wgpu::Buffer, values: &[T]) {
    let device = engine.device().unwrap();
    let source = buffer(device, values);
    assert_eq!(source.size(), target.size());
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test storage initialization"),
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read> source: array<u32>;
             @group(0) @binding(1) var<storage, read_write> destination: array<u32>;
             @compute @workgroup_size(64)
             fn seed(@builtin(global_invocation_id) id: vec3<u32>) {
                 if id.x < arrayLength(&source) { destination[id.x] = source[id.x]; }
             }"
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("seed"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: target.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((source.size() / 4).div_ceil(64) as u32, 1, 1);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
}
fn basis(normal: Vec3) -> Mat3 {
    let tangent = normal
        .cross(if normal.x.abs() < 0.8 {
            Vec3::X
        } else {
            Vec3::Y
        })
        .normalize();
    Mat3::from_cols(normal, tangent, normal.cross(tangent))
}

struct Fixture {
    contacts: Vec<SoftRigidContactGpu>,
    initial: Vec<[f32; 4]>,
    direction: Vec<[f32; 4]>,
    impulses: Vec<[f32; 4]>,
    impulse_direction: Vec<[f32; 4]>,
    state: Vec<wgpu::Buffer>,
    reference: wgpu::Buffer,
    directions: [wgpu::Buffer; 4],
    settings: SoftRigidSolveSettings,
}

impl Fixture {
    fn new(engine: &Engine) -> Self {
        let device = engine.device().unwrap();
        let point = SoftRigidContactGpu {
            normal_gap: [0.6, 0.8, 0.0, -0.001],
            material: [0.6, 0.0, 0.0, 0.0],
            ids: [0, u32::MAX, 0, 1],
            support: [0, 0, 1, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        let contacts = vec![
            point,
            SoftRigidContactGpu {
                ids: [1, 0, 0, 1],
                ..point
            },
            SoftRigidContactGpu {
                ids: [2, 1, 0, 1],
                normal_gap: [0.6, 0.8, 0.0, 0.5],
                ..point
            },
            SoftRigidContactGpu {
                ids: [0, 0, 0, 1],
                support: [1, 0, 2, 0],
                weights: [0.3, 0.7, 0.0, 0.0],
                ..point
            },
            SoftRigidContactGpu {
                ids: [0, 1, 0, 1],
                support: [1, 2, 3, 0],
                weights: [0.2, 0.3, 0.5, 0.0],
                material: [1.2, 0.0, 0.0, 0.0],
                ..point
            },
            SoftRigidBallJoint {
                body_a: 0,
                body_b: 1,
                anchor_a: [0.0; 3],
                anchor_b: [0.0; 3],
            }
            .row(2)
            .unwrap(),
            SoftRigidDofBound {
                dof: 1,
                sign: 1.0,
                speed: 0.05,
            }
            .row(3)
            .unwrap(),
            SoftRigidContactGpu {
                ids: [0, 0, 0, 0],
                ..point
            },
        ];
        let initial = vec![
            [0.3, -0.2, 0.1, 0.0],
            [-0.1, 0.4, 0.2, 0.0],
            [0.2, 0.1, -0.3, 0.0],
            [0.0; 4],
        ];
        let direction = vec![
            [0.1, 0.02, -0.03, 0.0],
            [-0.07, 0.09, 0.06, 0.0],
            [0.03, -0.02, 0.08, 0.0],
            [0.0; 4],
        ];
        let impulses = vec![
            [0.8, 0.0, 0.0, 0.0],
            [0.05, 0.3, -0.2, 0.0],
            [-0.02, 0.1, 0.0, 0.0],
            [0.1, 0.2, 0.05, 0.0],
            [0.3, 0.01, 0.02, 0.0],
            [0.07, -0.03, 0.01, 0.0],
            [0.05, 0.0, 0.0, 0.0],
            [0.03, -0.04, 0.02, 0.0],
        ];
        let impulse_direction: Vec<_> = (0..contacts.len())
            .map(|i| [0.011 * (i as f32 + 1.0), -0.007, 0.009, 0.0])
            .collect();
        let mut jacobians = vec![0.0_f32; 36];
        for d in 0..3 {
            jacobians[d * 3 + d] = 1.0;
            jacobians[18 + d * 3 + d] = [0.3, 0.4, 0.5][d];
        }
        let state = vec![
            buffer(device, &[[0.0_f32; 4]; 4]),
            buffer(device, &initial),
            buffer(device, &[2.0_f32, 4.0, 3.0, 0.0]),
            buffer(device, &contacts),
            buffer(
                device,
                &[
                    [0.0_f32, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [-0.003, 0.002, -0.001, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ),
            buffer(device, &[[0.0_f32; 4]; 2]),
            buffer(device, &jacobians),
            buffer(device, &[2.0_f32, 0.3, 0.1, 0.3, 1.5, 0.2, 0.1, 0.2, 1.1]),
            buffer(device, &[-0.1_f32, 0.03, 0.1]),
            buffer(device, &[[0.0_f32; 4]; 4]),
        ];
        let reference = buffer(device, &[0.1_f32, -0.05, 0.2]);
        let directions = [
            buffer(device, &direction),
            buffer(device, &[0.04_f32, -0.02, 0.03]),
            buffer(device, &impulses),
            buffer(device, &impulse_direction),
        ];
        Self {
            contacts,
            initial,
            direction,
            impulses,
            impulse_direction,
            state,
            reference,
            directions,
            settings: SoftRigidSolveSettings {
                iterations: 1,
                dt: 0.01,
                penetration_recovery: 0.1,
                max_recovery_speed: 1.0,
            },
        }
    }
    fn state(&self) -> SoftRigidSolveBindings<'_> {
        SoftRigidSolveBindings {
            positions: &self.state[0],
            velocities: &self.state[1],
            inverse_mass: &self.state[2],
            contacts: &self.state[3],
            body_poses: &self.state[4],
            body_com: &self.state[5],
            jacobians: &self.state[6],
            mass_matrix: &self.state[7],
            rigid_velocities: &self.state[8],
            body_velocities: &self.state[9],
            particle_count: 4,
            body_count: 2,
        }
    }
    fn bindings(&self) -> ContactLinearizationBindings<'_> {
        ContactLinearizationBindings {
            state: self.state(),
            rigid_reference: &self.reference,
            soft_direction: &self.directions[0],
            rigid_direction: &self.directions[1],
            impulses: &self.directions[2],
            impulse_direction: &self.directions[3],
        }
    }
    fn snapshot(&self, engine: &Engine, solve: &SoftRigidSolveKernel) -> Vec<Vec<u8>> {
        self.state
            .iter()
            .chain(self.directions.iter())
            .chain([
                &self.reference,
                solve.row_state(),
                solve.projected_rows(),
                solve.row_responses(),
            ])
            .map(|b| readback::read::<u8>(engine, b, b.size() as usize))
            .collect()
    }
    fn cpu(&self, rows: &[[f32; 8]], epsilon: f64) -> (Vec<Vec3>, Vec3, Vec<Vec3>) {
        let soft: Vec<_> = self
            .initial
            .iter()
            .zip(&self.direction)
            .map(|(&x, &d)| v(x) + epsilon * v(d))
            .collect();
        let rigid = v([-0.1, 0.03, 0.1, 0.0]) + epsilon * v([0.04, -0.02, 0.03, 0.0]);
        let reference = v([0.1, -0.05, 0.2, 0.0]);
        let matrix = Mat3::from_cols_array(
            &[2.0_f32, 0.3, 0.1, 0.3, 1.5, 0.2, 0.1, 0.2, 1.1].map(f64::from),
        );
        let inverse = matrix.inverse();
        let body = [Vec3::ONE, v([0.3, 0.4, 0.5, 0.0])];
        let mut transpose = vec![Vec3::ZERO; 4];
        let mut reaction = Vec3::ZERO;
        let mut residual = vec![];
        for (r, c) in self.contacts.iter().enumerate() {
            let impulse = v(self.impulses[r]) + epsilon * v(self.impulse_direction[r]);
            let change =
                impulse - Vec3::new(rows[r][4] as f64, rows[r][5] as f64, rows[r][6] as f64);
            let axes = basis(v(c.normal_gap));
            let mut projected = Mat3::ZERO;
            let mut relative = Vec3::ZERO;
            if c.ids[3] == 1 {
                for i in 0..c.support[2] as usize {
                    let node = if i == 0 { c.ids[0] } else { c.support[i - 1] } as usize;
                    transpose[node] += c.weights[i] as f64 * (axes * change);
                    relative += c.weights[i] as f64 * (axes.transpose() * soft[node]);
                }
                if c.ids[1] != u32::MAX {
                    projected = axes.transpose() * Mat3::from_diagonal(body[c.ids[1] as usize]);
                }
            } else if c.ids[3] == 3 {
                projected = Mat3::from_diagonal(body[1] - body[0]);
            } else if c.ids[3] == 5 {
                projected = Mat3::from_cols(Vec3::ZERO, Vec3::NEG_X, Vec3::ZERO);
            }
            relative -= projected * rigid;
            reaction += projected.transpose() * change;
            let value = if c.ids[3] == 3 {
                relative
                    - v([-0.003, 0.002, -0.001, 0.0])
                        * (self.settings.penetration_recovery as f64 / self.settings.dt as f64)
            } else if c.ids[3] == 1 || c.ids[3] == 5 {
                let normal = rows[r][0] as f64;
                let mut value = impulse;
                value.x = (impulse.x
                    - (impulse.x + (rows[r][2] as f64 - relative.x) / normal).max(0.0))
                    * normal;
                if c.ids[3] == 1 {
                    let tangent = rows[r][1] as f64;
                    let trial = Vec3::new(
                        0.0,
                        impulse.y - relative.y / tangent,
                        impulse.z - relative.z / tangent,
                    );
                    let radius = c.material[0] as f64 * impulse.x.max(0.0);
                    let projected = trial * (radius / trial.length().max(1e-30)).min(1.0);
                    value.y = (impulse.y - projected.y) * tangent;
                    value.z = (impulse.z - projected.z) * tangent;
                }
                value
            } else {
                impulse
            };
            residual.push(value);
        }
        for (node, weight) in transpose.iter_mut().zip([2.0, 4.0, 3.0, 0.0]) {
            *node *= weight;
        }
        (transpose, rigid - reference + inverse * reaction, residual)
    }
}

fn assert_close(actual: f32, expected: f64) {
    assert!(
        (actual as f64 - expected).abs() <= 2e-6 + 2e-5 * expected.abs(),
        "GPU={actual} oracle={expected}"
    );
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn contact_linearization_matches_finite_differences_and_impulse_balance() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let fixture = Fixture::new(&engine);
    let solve = SoftRigidSolveKernel::new(device, 3, fixture.contacts.len() as u32).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    solve
        .encode_preparation(device, &mut encoder, fixture.state(), fixture.settings)
        .unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(readback::read::<u32>(&engine, solve.status(), 1), [0]);
    let mut rows = readback::read::<[f32; 8]>(&engine, solve.row_state(), fixture.contacts.len());
    for (row, c) in rows.iter_mut().zip(&fixture.contacts) {
        if [1, 3, 5].contains(&c.ids[3]) {
            row[4] = 0.02;
            row[5] = 0.0;
            row[6] = 0.0;
        }
    }
    storage_write(&engine, solve.row_state(), &rows);
    let mut operator = ContactLinearizationKernel::new(device, 4, rows.len() as u32, 3, 2).unwrap();
    assert!(
        operator
            .encode(&mut device.create_command_encoder(&Default::default()))
            .is_err()
    );
    let before = fixture.snapshot(&engine, &solve);
    let expected = fixture.cpu(&rows, 0.0);
    let epsilon = 1e-4;
    let plus = fixture.cpu(&rows, epsilon);
    let minus = fixture.cpu(&rows, -epsilon);
    let mut prior = None;
    for _ in 0..3 {
        let mut encoder = device.create_command_encoder(&Default::default());
        operator
            .prepare(
                device,
                &mut encoder,
                &solve,
                fixture.bindings(),
                fixture.settings,
            )
            .unwrap();
        operator.encode(&mut encoder).unwrap();
        queue.submit([encoder.finish()]);
        assert_eq!(readback::read::<u32>(&engine, operator.status(), 2), [0, 0]);
        let soft = readback::read::<[f32; 4]>(&engine, operator.soft_terms(), 8);
        let rigid = readback::read::<[f32; 4]>(&engine, operator.rigid_terms(), 3);
        let contact = readback::read::<[f32; 4]>(&engine, operator.contact_terms(), 2 * rows.len());
        for i in 0..4 {
            for k in 0..3 {
                assert_close(soft[i][k], expected.0[i][k]);
                assert_close(
                    soft[4 + i][k],
                    (plus.0[i][k] - minus.0[i][k]) / (2.0 * epsilon),
                );
            }
        }
        for k in 0..3 {
            assert_close(rigid[k][0], expected.1[k]);
            assert_close(rigid[k][1], (plus.1[k] - minus.1[k]) / (2.0 * epsilon));
        }
        for i in 0..rows.len() {
            for k in 0..3 {
                assert_close(contact[i][k], expected.2[i][k]);
                assert_close(
                    contact[rows.len() + i][k],
                    (plus.2[i][k] - minus.2[i][k]) / (2.0 * epsilon),
                );
            }
        }
        let bits: Vec<_> = soft
            .iter()
            .chain(rigid.iter())
            .chain(contact.iter())
            .flatten()
            .map(|v| v.to_bits())
            .collect();
        if let Some(prior) = &prior {
            assert_eq!(&bits, prior);
        } else {
            prior = Some(bits);
        }
        assert_eq!(before, fixture.snapshot(&engine, &solve));
    }
    queue.write_buffer(&fixture.directions[2], 0, bytemuck::bytes_of(&f32::NAN));
    let mut encoder = device.create_command_encoder(&Default::default());
    operator.encode(&mut encoder).unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(readback::read::<u32>(&engine, operator.status(), 2), [0, 4]);
    queue.write_buffer(
        &fixture.directions[2],
        0,
        bytemuck::cast_slice(&fixture.impulses),
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    operator.encode(&mut encoder).unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(readback::read::<u32>(&engine, operator.status(), 2), [0, 0]);
    assert_eq!(before, fixture.snapshot(&engine, &solve));
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn contact_linearization_handles_zero_friction_and_unobservable_loops() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut fixture = Fixture::new(&engine);
    fixture.contacts[0].material[0] = 0.0;
    queue.write_buffer(
        &fixture.state[3],
        0,
        bytemuck::cast_slice(&fixture.contacts),
    );
    queue.write_buffer(
        &fixture.state[1],
        0,
        bytemuck::cast_slice(&[[0.0_f32; 4]; 4]),
    );
    let mut jacobians = vec![0.0_f32; 36];
    for body in 0..2 {
        for d in 0..3 {
            jacobians[body * 18 + d * 3 + d] = 1.0;
        }
    }
    queue.write_buffer(&fixture.state[6], 0, bytemuck::cast_slice(&jacobians));
    let mut poses = [
        [0.0_f32; 4],
        [0.0, 0.0, 0.0, 1.0],
        [0.0; 4],
        [0.0, 0.0, 0.0, 1.0],
    ];
    queue.write_buffer(&fixture.state[4], 0, bytemuck::cast_slice(&poses));
    let solve = SoftRigidSolveKernel::new(device, 3, fixture.contacts.len() as u32).unwrap();
    let mut operator =
        ContactLinearizationKernel::new(device, 4, fixture.contacts.len() as u32, 3, 2).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    solve
        .encode_preparation(device, &mut encoder, fixture.state(), fixture.settings)
        .unwrap();
    operator
        .prepare(
            device,
            &mut encoder,
            &solve,
            fixture.bindings(),
            fixture.settings,
        )
        .unwrap();
    operator.encode(&mut encoder).unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(readback::read::<u32>(&engine, operator.status(), 2), [0, 0]);
    let rows = readback::read::<[f32; 8]>(&engine, solve.row_state(), fixture.contacts.len());
    let terms = readback::read::<[f32; 4]>(
        &engine,
        operator.contact_terms(),
        fixture.contacts.len() * 2,
    );
    for k in 1..3 {
        assert_eq!(terms[0][k], 0.0);
        assert_close(
            terms[fixture.contacts.len()][k],
            rows[0][1] as f64 * fixture.impulse_direction[0][k] as f64,
        );
    }
    for k in 0..3 {
        assert_eq!(rows[5][k], 0.0);
        assert_eq!(terms[5][k], fixture.impulses[5][k]);
        assert_eq!(
            terms[fixture.contacts.len() + 5][k],
            fixture.impulse_direction[5][k]
        );
    }
    poses[2][0] = 0.1;
    queue.write_buffer(&fixture.state[4], 0, bytemuck::cast_slice(&poses));
    let mut encoder = device.create_command_encoder(&Default::default());
    solve
        .encode_preparation(device, &mut encoder, fixture.state(), fixture.settings)
        .unwrap();
    operator
        .prepare(
            device,
            &mut encoder,
            &solve,
            fixture.bindings(),
            fixture.settings,
        )
        .unwrap();
    operator.encode(&mut encoder).unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(readback::read::<u32>(&engine, operator.status(), 2), [0, 2]);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn contact_linearization_rejects_bad_inputs_and_propagates_status() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let fixture = Fixture::new(&engine);
    let solve = SoftRigidSolveKernel::new(device, 3, fixture.contacts.len() as u32).unwrap();
    for dimensions in [
        (0, 8, 3, 2),
        (4, 0, 3, 2),
        (4, 8, 0, 2),
        (4, 8, 65, 2),
        (4, 8, 3, 0),
        (u32::MAX, 8, 3, 2),
    ] {
        assert!(
            ContactLinearizationKernel::new(
                device,
                dimensions.0,
                dimensions.1,
                dimensions.2,
                dimensions.3
            )
            .is_err()
        );
    }
    let mut operator = ContactLinearizationKernel::new(device, 4, 8, 3, 2).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    solve
        .encode_preparation(device, &mut encoder, fixture.state(), fixture.settings)
        .unwrap();
    queue.submit([encoder.finish()]);
    for flag in [0_u32, 1, 2, 4] {
        storage_write(&engine, solve.status(), &[flag]);
        let mut encoder = device.create_command_encoder(&Default::default());
        operator
            .prepare(
                device,
                &mut encoder,
                &solve,
                fixture.bindings(),
                fixture.settings,
            )
            .unwrap();
        operator.encode(&mut encoder).unwrap();
        queue.submit([encoder.finish()]);
        assert_eq!(
            readback::read::<u32>(&engine, operator.status(), 2),
            [flag, flag]
        );
    }
    let small = buffer(device, &[0.0_f32]);
    let invalid_usage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut settings = fixture.settings;
    settings.dt = f32::NAN;
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(
        operator
            .prepare(device, &mut encoder, &solve, fixture.bindings(), settings)
            .is_err()
    );
    assert!(operator.encode(&mut encoder).is_err());
    for source in [&small, &invalid_usage, &operator.soft_terms().clone()] {
        let mut bindings = fixture.bindings();
        bindings.soft_direction = source;
        assert!(
            operator
                .prepare(device, &mut encoder, &solve, bindings, fixture.settings)
                .is_err()
        );
        assert!(operator.encode(&mut encoder).is_err());
    }
    storage_write(&engine, solve.status(), &[0_u32]);
    let mut contacts = fixture.contacts.clone();
    contacts[4].weights[0] = f32::NAN;
    queue.write_buffer(&fixture.state[3], 0, bytemuck::cast_slice(&contacts));
    operator
        .prepare(
            device,
            &mut encoder,
            &solve,
            fixture.bindings(),
            fixture.settings,
        )
        .unwrap();
    operator.encode(&mut encoder).unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(readback::read::<u32>(&engine, operator.status(), 2), [2, 2]);
    queue.write_buffer(
        &fixture.state[3],
        0,
        bytemuck::cast_slice(&fixture.contacts),
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    operator.encode(&mut encoder).unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(readback::read::<u32>(&engine, operator.status(), 2), [2, 2]);
    let mut encoder = device.create_command_encoder(&Default::default());
    operator
        .prepare(
            device,
            &mut encoder,
            &solve,
            fixture.bindings(),
            fixture.settings,
        )
        .unwrap();
    operator.encode(&mut encoder).unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(readback::read::<u32>(&engine, operator.status(), 2), [0, 0]);
}
