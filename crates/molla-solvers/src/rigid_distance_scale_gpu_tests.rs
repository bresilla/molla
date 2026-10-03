use super::*;
use std::time::Duration;

#[path = "../benches/primitives/fixtures.rs"]
mod fixtures;

fn fence(engine: &Engine, stage: &str) {
    eprintln!("distance_scale_stage_wait stage={stage}");
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap_or_else(|error| panic!("distance scale stage {stage}: {error}"));
    engine.check_device().unwrap();
    eprintln!("distance_scale_stage_complete stage={stage}");
}

fn read_words(engine: &Engine, source: &wgpu::Buffer, stage: &str) -> Vec<u32> {
    let device = engine.device().unwrap();
    let buffer = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("distance scale diagnostic readback"),
            size: source.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    fence(engine, stage);
    map_read(device, &buffer).unwrap();
    let mapped = molla_core::gpu_readback!(buffer.slice(..));
    let words = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    words
}

fn check_status(engine: &Engine, status: &wgpu::Buffer, stage: &str) {
    assert_eq!(read_words(engine, status, stage)[0], 0, "stage={stage}");
}

#[test]
#[ignore = "requires a hardware GPU; paired FK chains and frame upload edits"]
fn paired_fk_chains_retain_frames_after_model_upload_edits() {
    use molla_core::{BodyId, WorldId};
    use molla_kernels::{QuaternionFkBindings, QuaternionFkKernel};
    use molla_math::{Mat3, Quat, Transform, Vec3};
    use molla_sim::{BodyParams, JointType, ModelBuilder};

    let engine = Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software);
    println!("adapter={adapter:?} paired_fk_frame_mutation=true");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for fixed_only in [false, true] {
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        let mut authored = Vec::new();
        for art in 0..if fixed_only { 2 } else { 4 } {
            builder.begin_articulation();
            let mut parent = BodyId::NONE;
            for kind in [
                JointType::Fixed,
                JointType::Prismatic,
                JointType::Revolute,
                JointType::Ball,
                JointType::Free,
                JointType::Distance,
                JointType::D6,
            ] {
                let kind = if fixed_only { JointType::Fixed } else { kind };
                let xp = Transform::new(
                    Vec3::new(0.2, -0.3, 0.4 + f64::from(art)),
                    Quat::from_euler(glam::EulerRot::XYZ, 0.12, -0.18, 0.24),
                );
                let xc = Transform::new(
                    Vec3::new(-0.3, 0.2, -0.4),
                    Quat::from_euler(glam::EulerRot::XYZ, -0.1, 0.2, -0.15),
                );
                let body =
                    builder.add_body(BodyParams::new().with_mass(2.).with_inertia(Mat3::IDENTITY));
                let axis = Vec3::new(0.2, 0.3, 0.4).normalize();
                let joint = match kind {
                    JointType::Fixed => builder.add_joint_fixed(parent, body, xp, xc),
                    JointType::Prismatic => builder.add_joint_prismatic(parent, body, xp, xc, axis),
                    JointType::Revolute => builder.add_joint_revolute(parent, body, xp, xc, axis),
                    JointType::Ball => builder.add_joint_ball(parent, body, xp, xc),
                    JointType::Free => builder.add_joint_free(parent, body, xp, xc),
                    JointType::Distance => builder.add_joint_distance(parent, body, xp, xc),
                    _ => builder.add_joint_d6(parent, body, xp, xc),
                }
                .unwrap();
                let reversed = authored.len() % 2 == 1;
                builder.set_joint_reversed(joint, reversed).unwrap();
                builder
                    .set_joint_enabled(joint, authored.len() % 3 != 1)
                    .unwrap();
                authored.push((kind, xp, xc));
                parent = body;
            }
        }
        let mut model = builder.build().unwrap();
        let mut input = model.state().unwrap();
        for (j, &(kind, _, _)) in authored.iter().enumerate() {
            let coord = model.joint_coord_offset.host().unwrap()[j] as usize;
            let q = input.joint_q.host_mut().unwrap();
            match kind {
                JointType::Prismatic => q[coord] = 0.15 + 1e-9,
                JointType::Revolute => q[coord] = [0.15, 3.1, 6.4, -4.7][j / 7] + 1e-9,
                JointType::Ball | JointType::Free | JointType::Distance | JointType::D6 => {
                    let angular = if kind == JointType::Ball { 0 } else { 3 };
                    if angular == 3 {
                        q[coord..coord + 3].copy_from_slice(&[0.3 + 1e-9, -0.2, 0.1]);
                    }
                    let rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.13, -0.17, 0.21)
                        * if j % 2 == 0 { 1. } else { -1. };
                    q[coord + angular..coord + angular + 4].copy_from_slice(&rotation.to_array());
                }
                _ => {}
            }
        }
        molla_sim::eval_fk(&model, &mut input).unwrap();
        model.upload_to_device(device, queue).unwrap();
        input.upload_to_device(device, queue).unwrap();
        let mut output = model.state().unwrap();
        output.upload_to_device(device, queue).unwrap();
        let mut control = model.control();
        control.upload_to_device(device, queue).unwrap();
        let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
        solver
            .prepare_gpu_coordinate_compensation(&[&input, &output])
            .unwrap();
        solver.seed_gpu_coordinate_compensation(&input).unwrap();
        let allocate = |label, size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }
            )
        };
        let rotations = allocate("paired FK chain rotations", model.body_count as u64 * 32);
        let rounding = allocate("paired FK chain rounding", model.body_count as u64 * 4);
        for edit in 0..3 {
            for (j, &(_, parent, child)) in authored.iter().enumerate() {
                let offset = Vec3::new(0.1, -0.2, 0.3) * (f64::from(edit) * 1e-3);
                model.joint_x_p.host_mut().unwrap()[j] = Transform::new(
                    parent.position + offset,
                    parent.rotation * Quat::from_rotation_z(f64::from(edit) * 1e-3),
                );
                model.joint_x_c.host_mut().unwrap()[j] = Transform::new(
                    child.position - offset,
                    child.rotation * Quat::from_rotation_x(-f64::from(edit) * 1e-3),
                );
            }
            model.upload_to_device(device, queue).unwrap();
            let mut expected = input.clone_host().unwrap();
            molla_sim::eval_fk(&model, &mut expected).unwrap();
            solver
                .refresh_gpu(&model, &input, &control, &mut output)
                .unwrap();
            let kernel = QuaternionFkKernel::new_with_coordinate_residues(&engine, &model).unwrap();
            let dispatch = kernel
                .prepare(
                    device,
                    QuaternionFkBindings {
                        coordinates: if model.joint_coord_total == 0 {
                            None
                        } else {
                            Some(output.joint_q.device_buffer_current().unwrap())
                        },
                        coordinate_residues: Some(
                            solver.gpu_coordinate_compensation_buffer(&output).unwrap(),
                        ),
                        poses: output.body_q.device_buffer_current().unwrap(),
                        rotations: &rotations,
                        rounding: &rounding,
                        status: solver.gpu_status_buffer().unwrap(),
                    },
                )
                .unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            dispatch.encode(&mut encoder);
            queue.submit([encoder.finish()]);
            check_status(
                &engine,
                solver.gpu_status_buffer().unwrap(),
                "paired-chain-refresh",
            );
            let poses = read_words(
                &engine,
                output.body_q.device_buffer_current().unwrap(),
                "paired-chain-poses",
            );
            let tails = read_words(
                &engine,
                solver.gpu_position_compensation_buffer(&output).unwrap(),
                "paired-chain-position-tails",
            );
            let paired = read_words(&engine, &rotations, "paired-chain-quaternions");
            let mut position_error: f64 = 0.;
            let mut rotation_error: f64 = 0.;
            for (body, pose) in expected.body_q.host().unwrap().iter().enumerate() {
                for axis in 0..3 {
                    let actual = f64::from(f32::from_bits(poses[8 * body + axis]))
                        + f64::from(f32::from_bits(tails[4 * body + axis]));
                    position_error = position_error.max((actual - pose.position[axis]).abs());
                }
                let mut same: f64 = 0.;
                let mut flipped: f64 = 0.;
                for (lane, value) in pose.rotation.normalize().to_array().into_iter().enumerate() {
                    let actual = f64::from(f32::from_bits(paired[8 * body + lane]))
                        + f64::from(f32::from_bits(paired[8 * body + 4 + lane]));
                    same = same.max((actual - value).abs());
                    flipped = flipped.max((actual + value).abs());
                }
                rotation_error = rotation_error.max(same.min(flipped));
            }
            println!(
                "paired_chain fixed_only={fixed_only} edit={edit} bodies={} position_error={position_error:.9e} rotation_error={rotation_error:.9e}",
                model.body_count
            );
            assert!(position_error <= 1e-12);
            assert!(rotation_error <= 1e-12);
        }
    }
}

#[test]
#[ignore = "requires a hardware GPU; bounded radial/cap stage check"]
fn radial_point_zero_cap_first_step_stages() {
    use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
    use molla_sim::joint_control::JointAxis;
    use molla_sim::runtime::{BodyDesc, DistanceLimits, JointDesc, RigidScene};
    use molla_sim::{BodyParams, JointType, MassProperties};

    let engine = Arc::new(Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software);
    println!("adapter={adapter:?} isolated_radial_cap=true");
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let marker = scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 1.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let marker = scene
        .insert_joint(JointDesc {
            parent: None,
            child: marker,
            kind: JointType::Ball,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_joint_limits(marker, JointAxis::AngX, Some([0., 0.]))
        .unwrap();
    let parent = Transform::from_translation(Vec3::new(1.2, -0.4, 0.8));
    let child = Transform::new(
        Vec3::new(-0.3, 0.5, -0.9),
        Quat::from_euler(glam::EulerRot::XYZ, -0.4, 0.3, -0.2),
    );
    let pose = parent
        .compose(&Transform::from_translation(Vec3::X))
        .compose(&child.inverse());
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(pose),
            additional_mass: Some(MassProperties {
                mass: 1.,
                local_com: Vec3::new(0.4, -0.3, 0.2),
                inertia: Mat3::from_cols(
                    Vec3::new(3., 0.2, 0.3),
                    Vec3::new(0.2, 4., 0.4),
                    Vec3::new(0.3, 0.4, 5.),
                ),
            }),
            ..Default::default()
        })
        .unwrap();
    let desc = JointDesc {
        parent: None,
        child: body,
        kind: JointType::D6,
        axis: Vec3::X,
        frame_parent: parent,
        frame_child: child,
    };
    let tree = scene.insert_joint(desc.clone()).unwrap();
    scene
        .set_joint_velocity_limit(tree, JointAxis::LinX, Some(0.))
        .unwrap();
    let distance = scene
        .insert_joint(JointDesc {
            kind: JointType::Distance,
            ..desc
        })
        .unwrap();
    scene
        .set_distance_limits(
            distance,
            DistanceLimits {
                min: Some(1.),
                max: Some(1.),
            },
        )
        .unwrap();
    scene
        .add_body_wrench(
            body,
            SpatialVector::new(Vec3::new(0.4, -0.3, 0.2), Vec3::new(3., 2., -1.)),
        )
        .unwrap();
    assert!(scene.model().requires_angular_lock_rows().unwrap());
    let [a, b] =
        crate::distance_constraints::anchors(&scene, scene.state(), scene.joint(distance).unwrap())
            .unwrap();
    println!(
        "radial_host a={a:?} b={b:?} delta={:?} q={:?}",
        b - a,
        scene.state().joint_q.host().unwrap()
    );
    let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
    world.solver.substeps = 1;
    let gpu = world.gpu.as_mut().unwrap();
    gpu.compile(&mut world.scene, &mut world.solver, false, false)
        .unwrap();
    let compiled = gpu.compiled.as_mut().unwrap();
    let active = vec![true; world.scene.model().articulation_count];
    world.solver.set_gpu_active_articulations(&active).unwrap();
    compiled
        .distance
        .as_mut()
        .unwrap()
        .set_active(engine.queue().unwrap(), &active)
        .unwrap();
    compiled
        .motion
        .as_mut()
        .unwrap()
        .prepare_frame(
            &engine,
            &world.solver,
            &compiled.states,
            world.scene.state(),
        )
        .unwrap();
    let position = read_words(
        &engine,
        world
            .solver
            .gpu_position_compensation_buffer(&compiled.states[0])
            .unwrap(),
        "initial-position-tail",
    );
    println!(
        "radial_position_tails={:?}",
        position
            .iter()
            .map(|&v| f32::from_bits(v))
            .collect::<Vec<_>>()
    );
    let [input, output] = &mut compiled.states;
    world
        .solver
        .refresh_gpu(world.scene.model(), input, world.scene.control(), output)
        .unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    for (from, to) in [
        (
            output.body_q.device_buffer_current().unwrap(),
            input.body_q.device_buffer_current().unwrap(),
        ),
        (
            output.body_qd.device_buffer_current().unwrap(),
            input.body_qd.device_buffer_current().unwrap(),
        ),
        (
            world
                .solver
                .gpu_position_compensation_buffer(output)
                .unwrap(),
            world
                .solver
                .gpu_position_compensation_buffer(input)
                .unwrap(),
        ),
    ] {
        encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    check_status(&engine, &compiled.solver_status, "isolated-refresh");
    let refreshed = read_words(
        &engine,
        input.body_q.device_buffer_current().unwrap(),
        "refreshed-pose",
    );
    let tails = read_words(
        &engine,
        world
            .solver
            .gpu_position_compensation_buffer(input)
            .unwrap(),
        "refreshed-position-tail",
    );
    let index = world.scene.body_index(body).unwrap().index();
    let mut position_error: f64 = 0.;
    for axis in 0..3 {
        let high = f64::from(f32::from_bits(refreshed[8 * index + axis]));
        let low = f64::from(f32::from_bits(tails[4 * index + axis]));
        position_error = position_error.max((high + low - pose.position[axis]).abs());
    }
    println!("refreshed_position_error={position_error:.9e}");
    assert!(position_error <= 1e-12);
    world
        .solver
        .step(
            world.scene.model(),
            input,
            world.scene.control(),
            &Contacts::with_capacity(0),
            1e-5,
            output,
        )
        .unwrap();
    check_status(&engine, &compiled.solver_status, "isolated-predictor");
    let rows = compiled.distance.as_ref().unwrap();
    rows.dispatch(&engine, 0, 1e-5, 1, 0).unwrap();
    let status = read_words(&engine, &compiled.solver_status, "isolated-radial-status")[0];
    let id = rows
        .owners
        .iter()
        .position(|&owner| owner == Some(distance))
        .unwrap();
    let reaction = read_words(&engine, &rows.samples, "isolated-radial-reaction");
    let response = read_words(&engine, &rows.row_forces, "isolated-radial-response");
    println!(
        "radial_device status={status} row={id} reaction={:?} response={:?}",
        reaction[8 * id..8 * id + 8]
            .iter()
            .map(|&v| f32::from_bits(v))
            .collect::<Vec<_>>(),
        response[4 * id..4 * id + 4]
            .iter()
            .map(|&v| f32::from_bits(v))
            .collect::<Vec<_>>()
    );
    assert_eq!(status, 0);
}

#[test]
#[ignore = "requires GPU; isolated 1000-instance first-frame stage diagnostic"]
fn distance_thousand_first_frame_stages() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (scene, _) = fixtures::build("distance", 1000).unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
    world.solver.substeps = 4;
    let gpu = world.gpu.as_mut().unwrap();
    gpu.compile(&mut world.scene, &mut world.solver, true, false)
        .unwrap();
    fence(&engine, "compile");
    let compiled = gpu.compiled.as_mut().unwrap();
    assert!(compiled.contacts.is_none());
    let active = vec![true; world.scene.model().articulation_count];
    world.solver.set_gpu_active_articulations(&active).unwrap();
    compiled
        .distance
        .as_mut()
        .unwrap()
        .set_active(engine.queue().unwrap(), &active)
        .unwrap();
    compiled
        .reactions
        .as_mut()
        .unwrap()
        .set_active(
            engine.queue().unwrap(),
            &world.scene,
            &vec![true; world.scene.model().body_count],
            &active,
        )
        .unwrap();
    if let Some(motion) = &mut compiled.motion {
        motion
            .prepare_frame(
                &engine,
                &world.solver,
                &compiled.states,
                world.scene.state(),
            )
            .unwrap();
    }
    fence(&engine, "prepare");
    world.solver.substeps = 1;
    let dt = 1. / 240. / 4.;
    for substep in 0..4 {
        eprintln!("distance_scale_substep index={substep}");
        let [a, b] = &mut compiled.states;
        let (input, output) = if substep % 2 == 0 { (a, b) } else { (b, a) };
        world
            .solver
            .step(
                world.scene.model(),
                input,
                world.scene.control(),
                &Contacts::with_capacity(0),
                dt,
                output,
            )
            .unwrap();
        fence(&engine, "predictor");
        check_status(&engine, &compiled.solver_status, "predictor-status");
        let rates = read_words(
            &engine,
            output.joint_qd.device_buffer().unwrap(),
            "predictor-rates",
        );
        let mut peak = 0.0f32;
        for (index, bits) in rates.into_iter().enumerate() {
            let rate = f32::from_bits(bits);
            assert!(
                rate.is_finite(),
                "substep={substep} dof={index} rate={rate}"
            );
            peak = peak.max(rate.abs());
        }
        eprintln!("distance_scale_predictor_peak substep={substep} rate={peak}");
        compiled
            .distance
            .as_ref()
            .unwrap()
            .dispatch(&engine, substep % 2, dt, 4, substep)
            .unwrap();
        fence(&engine, "distance");
        check_status(&engine, &compiled.solver_status, "distance-status");
        world
            .solver
            .drift_gpu(
                world.scene.model(),
                input,
                world.scene.control(),
                dt,
                output,
            )
            .unwrap();
        fence(&engine, "drift");
        check_status(&engine, &compiled.solver_status, "drift-status");
        compiled
            .reactions
            .as_ref()
            .unwrap()
            .dispatch(&engine, substep % 2, dt, substep);
        fence(&engine, "reaction");
        check_status(&engine, &compiled.solver_status, "reaction-status");
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-distance-scale-gpu"]
fn distance_thousand_runtime_one_second() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (cpu_scene, _) = fixtures::build("distance", 1000).unwrap();
    let (gpu_scene, _) = fixtures::build("distance", 1000).unwrap();
    let bodies: Vec<_> = cpu_scene
        .bodies()
        .zip(gpu_scene.bodies())
        .map(|((cpu, desc), (gpu, _))| (cpu, gpu, desc.params.pose))
        .collect();
    let joints: Vec<_> = cpu_scene
        .joints()
        .zip(gpu_scene.joints())
        .map(|((cpu, _), (gpu, _))| (cpu, gpu))
        .collect();
    assert_eq!(bodies.len(), 1000);
    assert_eq!(joints.len(), 1000);
    let mut cpu = RigidWorld::from_scene(cpu_scene).unwrap();
    let mut gpu = RigidWorld::from_scene_gpu(gpu_scene, engine).unwrap();
    for world in [&mut cpu, &mut gpu] {
        world.solver.substeps = 4;
        world.set_reaction_readout(true).unwrap();
    }
    let mut peak_pose = 0.0f64;
    let mut peak_rate = 0.0f64;
    let mut peak_force = 0.0f64;
    for frame in 0..240 {
        cpu.step(1. / 240.).unwrap();
        gpu.step(1. / 240.).unwrap();
        for &(c, g, initial) in &bodies {
            for (world, body) in [(&cpu, c), (&gpu, g)] {
                let pose = world.scene.body_pose(body).unwrap();
                let rate = world.scene.body_velocity(body).unwrap();
                assert!(pose.position.is_finite() && pose.rotation.is_finite());
                assert!(rate.linear.is_finite() && rate.angular.is_finite());
                peak_pose = peak_pose.max((pose.position - initial.position).length());
                peak_pose = peak_pose.max(
                    2. * pose
                        .rotation
                        .normalize()
                        .dot(initial.rotation.normalize())
                        .abs()
                        .min(1.)
                        .acos(),
                );
                peak_rate = peak_rate
                    .max(rate.linear.length())
                    .max(rate.angular.length());
            }
        }
        for &(c, g) in &joints {
            let cr = cpu.distance_reading(c).unwrap();
            let gr = gpu.distance_reading(g).unwrap();
            peak_force = peak_force.max((cr.force - gr.force).length());
            for (world, joint) in [(&cpu, c), (&gpu, g)] {
                let reaction = world.joint_reaction(joint).unwrap();
                assert!(reaction.child.linear.is_finite() && reaction.child.angular.is_finite());
                peak_force =
                    peak_force.max((reaction.child.linear - molla_math::Vec3::Y * 10.).length());
                peak_force = peak_force.max(reaction.child.angular.length());
                assert!((reaction.child.linear + reaction.parent.linear).length() < 0.02);
            }
        }
        assert!(peak_pose < 1e-3, "frame={frame} peak_pose={peak_pose}");
        assert!(peak_rate < 1e-3, "frame={frame} peak_rate={peak_rate}");
        assert!(peak_force < 0.02, "frame={frame} peak_force={peak_force}");
    }
    assert_eq!(gpu.gpu_diagnostics().unwrap().model_compilations, 2);
    assert!(gpu.joint_break_events().is_empty());
    eprintln!(
        "distance_scale_parity instances=1000 frames=240 peak_pose={peak_pose} peak_rate={peak_rate} peak_force={peak_force}"
    );
}
