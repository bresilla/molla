use molla_core::WorldId;
use molla_math::{Mat3, Real, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene};
use molla_sim::{BodyParams, JointType, MassProperties, Model, ModelBuilder, State};
use molla_solvers::soft_rigid_contact::{SoftRigidContact, SoftRigidContactConfig, solve_cpu};

fn rigid(anchored: bool) -> (RigidScene, usize) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new(),
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(Vec3::new(0.2, 0.25, 0.3)),
            }),
            ..Default::default()
        })
        .unwrap();
    if anchored {
        scene
            .insert_joint(JointDesc {
                parent: None,
                child: body,
                kind: JointType::Revolute,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
                axis: Vec3::X,
            })
            .unwrap();
    }
    let index = scene.body_index(body).unwrap().index();
    (scene, index)
}

fn soft(position: Vec3, velocity: Vec3) -> (Model, State) {
    let mut builder = ModelBuilder::new();
    builder.add_particle(position, 0.2, 0.0, WorldId(0));
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state.particle_qd.host_mut().unwrap()[0] = velocity;
    (model, state)
}

fn momentum(model: &Model, state: &State, soft: &State) -> (Vec3, Vec3, Real) {
    let v = state.body_qd.host().unwrap()[0];
    let pose = state.body_q.host().unwrap()[0];
    let mass = model.body_mass.host().unwrap()[0];
    let rotation = Mat3::from_quat(pose.rotation);
    let inertia = rotation * model.body_inertia.host().unwrap()[0] * rotation.transpose();
    let p = soft.particle_q.host().unwrap()[0];
    let sv = soft.particle_qd.host().unwrap()[0];
    (
        v.linear * mass + sv * 0.2,
        pose.position.cross(v.linear * mass) + inertia * v.angular + p.cross(sv * 0.2),
        0.5 * mass * v.linear.length_squared()
            + 0.5 * v.angular.dot(inertia * v.angular)
            + 0.1 * sv.length_squared(),
    )
}

#[test]
fn dynamic_contact_conserves_momenta_and_dissipates_energy() {
    for friction in [0.0, 0.3, 2.0] {
        let (scene, body) = rigid(false);
        let mut rigid = scene.state().clone_host().unwrap();
        let (model, mut state) = soft(Vec3::new(0.3, 0.55, -0.1), Vec3::new(0.8, -1.2, 0.6));
        let before = momentum(scene.model(), &rigid, &state);
        let report = solve_cpu(
            &model,
            &mut state,
            scene.model(),
            &mut rigid,
            &[SoftRigidContact {
                particle: 0,
                body: Some(body),
                normal: Vec3::Y,
                separation: 0.0,
                friction,
            }],
            1.0 / 480.0,
            SoftRigidContactConfig {
                iterations: 60,
                ..Default::default()
            },
        )
        .unwrap();
        let after = momentum(scene.model(), &rigid, &state);
        assert!((after.0 - before.0).length() < 1e-12);
        assert!((after.1 - before.1).length() < 1e-12);
        assert!(after.2 <= before.2 + 1e-12);
        let impulse = report.impulses[0];
        assert!(impulse.y > 0.0);
        assert!(Vec3::new(impulse.x, 0.0, impulse.z).length() <= friction * impulse.y + 1e-12);
        let v = rigid.body_qd.host().unwrap()[body];
        let rel = state.particle_qd.host().unwrap()[0]
            - v.linear
            - v.angular.cross(state.particle_q.host().unwrap()[0]);
        assert!(rel.y.abs() < 1e-9, "{rel:?}");
    }
}

#[test]
fn contact_uses_constrained_sprocket_inertia() {
    let (scene, body) = rigid(true);
    let mut rotor = scene.state().clone_host().unwrap();
    let (model, mut state) = soft(Vec3::Y * 0.5, -Vec3::Z);
    solve_cpu(
        &model,
        &mut state,
        scene.model(),
        &mut rotor,
        &[SoftRigidContact {
            particle: 0,
            body: Some(body),
            normal: Vec3::Z,
            separation: 0.0,
            friction: 0.0,
        }],
        1.0 / 480.0,
        SoftRigidContactConfig::default(),
    )
    .unwrap();
    let expected = 1.0 / (1.0 / 0.2 + 0.5 * 0.5 / 0.2);
    assert!((state.particle_qd.host().unwrap()[0].z - (-1.0 + expected / 0.2)).abs() < 1e-12);
    let rotor_velocity = rotor.body_qd.host().unwrap()[body];
    assert!(rotor_velocity.linear.length() < 1e-12);
    assert!((rotor_velocity.angular.x + expected * 0.5 / 0.2).abs() < 1e-12);
}

#[test]
fn separating_and_distant_nodes_receive_no_adhesion() {
    let (scene, body) = rigid(false);
    for (velocity, gap) in [(Vec3::Y, 0.0), (-Vec3::Y, 0.1)] {
        let mut rigid = scene.state().clone_host().unwrap();
        let (model, mut state) = soft(Vec3::Y, velocity + Vec3::X);
        let report = solve_cpu(
            &model,
            &mut state,
            scene.model(),
            &mut rigid,
            &[SoftRigidContact {
                particle: 0,
                body: Some(body),
                normal: Vec3::Y,
                separation: gap,
                friction: 1.0,
            }],
            1.0 / 480.0,
            SoftRigidContactConfig::default(),
        )
        .unwrap();
        assert_eq!(report.impulses[0], Vec3::ZERO);
        assert_eq!(state.particle_qd.host().unwrap()[0], velocity + Vec3::X);
        assert_eq!(rigid.body_qd.host().unwrap()[0], SpatialVector::ZERO);
    }
}

#[test]
fn invalid_late_contact_does_not_mutate_either_state() {
    let (scene, body) = rigid(false);
    let mut rigid = scene.state().clone_host().unwrap();
    let (model, mut state) = soft(Vec3::Y, -Vec3::Y);
    let contact = SoftRigidContact {
        particle: 0,
        body: Some(body),
        normal: Vec3::Y,
        separation: 0.0,
        friction: 0.5,
    };
    let invalid = SoftRigidContact {
        particle: 999,
        ..contact
    };
    assert!(
        solve_cpu(
            &model,
            &mut state,
            scene.model(),
            &mut rigid,
            &[contact, invalid],
            1.0 / 480.0,
            SoftRigidContactConfig::default()
        )
        .is_err()
    );
    assert_eq!(state.particle_qd.host().unwrap()[0], -Vec3::Y);
    assert_eq!(rigid.body_qd.host().unwrap()[0], SpatialVector::ZERO);
}

#[test]
fn powered_tooth_transfers_torque_into_volumetric_rubber() {
    use molla_sim::JointTargetMode;
    use molla_sim::joint_control::{JointAxis, JointMotor};
    use molla_sim::runtime::{ColliderDesc, ColliderGeometry};
    use molla_sim::soft_belt::{RubberBeltMesh, RubberBeltParams};
    use molla_solvers::fem_rigid::{FemRigidConfig, FemRigidSystem};
    use molla_solvers::rigid_world::RigidWorld;

    for sign in [-1.0, 1.0] {
        let mut outcomes = Vec::new();
        for coupled in [false, true] {
            let (mut scene, body) = rigid(true);
            let (rotor, _) = scene.bodies().next().unwrap();
            let (joint, _) = scene.joints().next().unwrap();
            scene
                .set_joint_motor(
                    joint,
                    JointAxis::AngX,
                    JointMotor {
                        mode: JointTargetMode::Velocity,
                        target_velocity: sign * 2.0,
                        damping: 5.0,
                        max_force: 5.0,
                        ..Default::default()
                    },
                )
                .unwrap();
            let mut tooth = ColliderDesc::new(ColliderGeometry::Cuboid {
                half_extents: Vec3::new(0.11, 0.04, 0.015),
            });
            tooth.parent = Some(rotor);
            tooth.pose = Transform::from_translation(Vec3::new(0.0, 0.48, -sign * 0.016));
            tooth.friction = 0.0;
            let tooth = scene.insert_collider(tooth).unwrap();
            let mut world = RigidWorld::from_scene(scene).unwrap();
            world.solver = world.solver.with_substeps(1);
            let mesh = RubberBeltMesh::circular(RubberBeltParams {
                segments: 32,
                width_cells: 2,
                thickness_cells: 2,
                ..Default::default()
            })
            .unwrap();
            let mut builder = ModelBuilder::new();
            builder.set_world_gravity(WorldId(0), Vec3::ZERO);
            builder.set_linear_damping(0.0);
            mesh.append_to(&mut builder, WorldId(0), Vec3::ZERO)
                .unwrap();
            let model = builder.build().unwrap();
            let mut fem = FemRigidSystem::new_cpu(model, FemRigidConfig::default()).unwrap();
            let dt = 1.0 / 2400.0;
            let mut total_impulse = 0.0;
            let mut max_penetration: Real = 0.0;
            for _ in 0..240 {
                let colliders = if coupled { vec![tooth] } else { vec![] };
                let report = fem.step_cpu(&mut world, &colliders, dt).unwrap();
                max_penetration = max_penetration.max(report.max_penetration);
                total_impulse += report.impulse_magnitude_sum;
                assert!(
                    mesh.minimum_volume_ratio(fem.state().particle_q.host().unwrap())
                        .unwrap()
                        > 0.75
                );
            }
            let state = fem.state();
            let angular: Vec3 = state
                .particle_q
                .host()
                .unwrap()
                .iter()
                .zip(state.particle_qd.host().unwrap())
                .zip(&mesh.masses)
                .map(|((p, v), m)| p.cross(*v * *m))
                .sum();
            let omega = world.scene.state().body_qd.host().unwrap()[body].angular.x;
            let displacement = state
                .particle_q
                .host()
                .unwrap()
                .iter()
                .zip(&mesh.positions)
                .map(|(p, q)| (*p - *q).length())
                .fold(0.0, Real::max);
            eprintln!(
                "tooth sign={sign} coupled={coupled}: omega={omega}, belt Lx={}, impulse={total_impulse}, penetration={max_penetration}, displacement={displacement}",
                angular.x
            );
            if coupled {
                assert!(sign * angular.x > 0.001, "no tooth-to-rubber torque");
                assert!(total_impulse > 0.01);
                assert!(displacement > 0.001);
                assert!(max_penetration < 0.002);
            } else {
                assert!(angular.length() < 1e-10);
                assert!(displacement < 1e-10);
            }
            outcomes.push(sign * omega);
        }
        assert!(
            outcomes[1] < outcomes[0] * 0.9,
            "belt must load the motor: {outcomes:?}"
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_gathers_contacts_from_resident_fem_state() {
    use molla_kernels::soft_rigid_contact::{
        SoftRigidContactBindings, SoftRigidContactGpu, SoftRigidContactKernel, SoftRigidShapeGpu,
    };
    use molla_math::Quat;
    use molla_sim::Contacts;
    use molla_sim::runtime::{ColliderDesc, ColliderGeometry};
    use molla_sim::soft_belt::{RubberBeltMesh, RubberBeltParams};
    use molla_solvers::soft_rigid_contact::gather_primitive_contacts;
    use molla_solvers::{Solver, VbdSolver};
    use molla_storage::{Buffer, BufferUsage};
    use wgpu::util::DeviceExt;

    let engine = molla_compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mesh = RubberBeltMesh::circular(RubberBeltParams {
        segments: 32,
        width_cells: 2,
        thickness_cells: 2,
        ..Default::default()
    })
    .unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    mesh.append_to(&mut builder, WorldId(0), Vec3::ZERO)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut state = model.state().unwrap();
    for p in state.particle_q.host_mut().unwrap() {
        p.y *= 1.02;
    }
    let mut next = model.state().unwrap();
    state.upload_to_device(device, queue).unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut fem = VbdSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_iterations(12)
        .with_contact_stiffness(0.0);
    for _ in 0..8 {
        fem.step(
            &model,
            &state,
            &model.control(),
            &Contacts::with_capacity(0),
            1.0 / 2400.0,
            &mut next,
        )
        .unwrap();
        std::mem::swap(&mut state, &mut next);
    }
    assert!(
        state.particle_q.host().is_err(),
        "FEM positions must remain device-authoritative"
    );
    let mut surface: Vec<_> = molla_sim::compute_tet_surface_triangles(&mesh.tets)
        .into_iter()
        .flatten()
        .collect();
    surface.sort_unstable();
    surface.dedup();
    let nodes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test FEM surface"),
        contents: bytemuck::cast_slice(&surface),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let (mut scene, body_index) = rigid(false);
    let (body, _) = scene.bodies().next().unwrap();
    scene
        .set_body_pose(
            body,
            Transform {
                position: Vec3::new(0.01, -0.02, 0.03),
                rotation: Quat::from_rotation_z(0.08),
            },
        )
        .unwrap();
    let geometry = [
        (
            ColliderGeometry::Ball { radius: 0.13 },
            Vec3::Z * 0.5,
            [0.13, 0.0, 0.0, 0.4],
        ),
        (
            ColliderGeometry::Cuboid {
                half_extents: Vec3::new(0.11, 0.06, 0.08),
            },
            Vec3::Y * 0.5,
            [0.11, 0.06, 0.08, 0.4],
        ),
        (
            ColliderGeometry::Cylinder {
                radius: 0.13,
                half_height: 0.1,
            },
            -Vec3::Y * 0.5,
            [0.13, 0.1, 0.0, 0.4],
        ),
    ];
    let mut colliders = Vec::new();
    let mut shapes = Vec::new();
    for (kind, (geometry, position, data)) in geometry.into_iter().enumerate() {
        let rotation = Quat::from_rotation_z(0.4);
        let mut desc = ColliderDesc::new(geometry);
        desc.parent = Some(body);
        desc.pose = Transform { position, rotation };
        desc.friction = 0.4;
        colliders.push(scene.insert_collider(desc).unwrap());
        shapes.push(SoftRigidShapeGpu {
            position: [position.x as f32, position.y as f32, position.z as f32, 0.0],
            rotation: rotation.to_array().map(|v| v as f32),
            data,
            ids: [kind as u32, body_index as u32, 0, 0],
        });
    }
    let shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test FEM shapes"),
        contents: bytemuck::cast_slice(&shapes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut poses = Buffer::from_vec(
        scene.state().body_q.host().unwrap().to_vec(),
        BufferUsage::Storage,
    );
    poses.ensure_device(device.as_ref()).unwrap();
    poses.upload(queue).unwrap();
    let kernel =
        SoftRigidContactKernel::new(device, surface.len() as u32, shapes.len() as u32).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    kernel
        .encode(
            device,
            &mut encoder,
            SoftRigidContactBindings {
                positions: state.particle_q.device_buffer().unwrap(),
                radii: model.particle_radius.device_buffer().unwrap(),
                surface: &nodes,
                shapes: &shape_buffer,
                body_poses: poses.device_buffer().unwrap(),
                particle_count: model.particle_count as u32,
                body_count: scene.model().body_count as u32,
            },
            0.001,
        )
        .unwrap();
    let bytes = kernel.row_count() as u64 * std::mem::size_of::<SoftRigidContactGpu>() as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test contact readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(kernel.rows(), 0, &staging, 0, bytes);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapping = staging.slice(..).get_mapped_range();
    let rows: &[SoftRigidContactGpu] = bytemuck::cast_slice(&mapping);
    state.particle_q.download(device, queue).unwrap();
    let cpu =
        gather_primitive_contacts(&model, &state, &surface, &scene, &colliders, 0.001).unwrap();
    let active: Vec<_> = rows.iter().filter(|r| r.ids[3] == 1).collect();
    assert!(rows.iter().all(|r| r.ids[3] <= 1));
    assert_eq!(active.len(), cpu.len());
    assert!(active.len() > 12);
    for (gpu, cpu) in active.iter().zip(&cpu) {
        assert_eq!(gpu.ids[0] as usize, cpu.particle);
        assert_eq!(gpu.ids[1] as usize, cpu.body.unwrap());
        let normal = Vec3::new(
            gpu.normal_gap[0] as Real,
            gpu.normal_gap[1] as Real,
            gpu.normal_gap[2] as Real,
        );
        assert!((normal - cpu.normal).length() < 5e-6);
        assert!((gpu.normal_gap[3] as Real - cpu.separation).abs() < 5e-6);
    }
    eprintln!(
        "GPU FEM contact gather: {} resident candidates, {} matching active CPU contacts",
        rows.len(),
        active.len()
    );
}
