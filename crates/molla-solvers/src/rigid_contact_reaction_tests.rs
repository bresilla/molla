use super::*;
use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyKind, ColliderDesc, ColliderGeometry};
use molla_sim::{BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, MassProperties};

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-reactions-gpu"]
fn failed_impact_history_is_not_inherited_when_only_capacity_changes() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::Y * 0.499),
            velocity: SpatialVector::new(-Vec3::Y, Vec3::ZERO),
            additional_mass: Some(MassProperties {
                mass: 2.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let mut shape = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    shape.parent = Some(body);
    shape.mass = ColliderMass::Mass(0.);
    shape.restitution = 0.5;
    shape.contact_material.friction = Some(FrictionCoefficients::uniform(0.));
    scene.insert_collider(shape.clone()).unwrap();
    scene.insert_collider(shape).unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(5., 0.1, 5.),
    });
    ground.pose = Transform::from_translation(-Vec3::Y * 0.1);
    ground.restitution = 0.5;
    ground.contact_material.friction = Some(FrictionCoefficients::uniform(0.));
    scene.insert_collider(ground).unwrap();
    let mut world = RigidWorld::from_scene_gpu_with_contact_capacity(
        scene,
        engine,
        GpuContactCapacity {
            pairs: 8,
            contacts_per_island: 1,
        },
    )
    .unwrap();
    world.solver.substeps = 4;
    let model = world.scene.model_stamp();
    let state = world.scene.body_state_stamp(body);
    let structure = world.scene.body_structure_stamp(body);
    let error = world.step(0.004).unwrap_err();
    assert!(
        error.to_string().ends_with("solver fault 2097152"),
        "{error}"
    );
    let gpu = world.gpu.as_mut().unwrap();
    gpu.contact_capacity.contacts_per_island = 2;
    gpu.compile(&mut world.scene, &mut world.solver, false, false)
        .unwrap();
    assert_eq!(world.scene.model_stamp(), model);
    assert_eq!(world.scene.body_state_stamp(body), state);
    assert_eq!(world.scene.body_structure_stamp(body), structure);
    world.step(0.004).unwrap();
    assert!((world.scene.body_velocity(body).unwrap().linear.y - 0.5).abs() < 2e-5);
    let impulse: f64 = world
        .contact_pairs()
        .iter()
        .flat_map(|p| &p.points)
        .map(|p| p.normal_impulse)
        .sum();
    assert!((impulse - 3.).abs() < 2e-5, "impulse={impulse}");
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-reactions-gpu"]
fn immovable_contact_endpoint_receives_reaction_load_without_predictor_force() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::new(2.0, -10.0, 0.0)).unwrap();
    let ground = scene
        .insert_body(BodyDesc {
            kind: BodyKind::Fixed,
            params: BodyParams::new().with_position(-Vec3::Y * 0.1),
            additional_mass: Some(MassProperties {
                mass: 3.0,
                local_com: Vec3::new(0.2, 0.0, 0.1),
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let block = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::Y * 0.4998),
            velocity: SpatialVector::new(Vec3::X, Vec3::ZERO),
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let material = ContactMaterial {
        stiffness: 1e5,
        damping: 300.0,
        friction: Some(FrictionCoefficients {
            mu_static: 0.6,
            mu_dynamic: 0.2,
        }),
        ..Default::default()
    };
    for (body, geometry) in [
        (
            ground,
            ColliderGeometry::Cuboid {
                half_extents: Vec3::new(5.0, 0.1, 5.0),
            },
        ),
        (block, ColliderGeometry::Ball { radius: 0.5 }),
    ] {
        let mut collider = ColliderDesc::new(geometry);
        collider.parent = Some(body);
        collider.mass = ColliderMass::Mass(0.0);
        collider.contact_material = material;
        scene.insert_collider(collider).unwrap();
    }
    let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
    let authored_position = world.scene.body_pose(ground).unwrap().position;
    let published_position = Vec3::from_array(authored_position.to_array().map(|value| {
        let high = value as f32 as f64;
        high + (value - high) as f32 as f64
    }));
    world.solver.substeps = 2;
    world.set_reaction_readout(true).unwrap();
    world.step(0.002).unwrap();
    let pairs = world.contact_pairs();
    assert_eq!(pairs.len(), 1);
    let pair = &pairs[0];
    let point = pair.points.last().unwrap();
    let force = point.force_on_b
        * if pair.bodies[1] == Some(ground) {
            1.0
        } else {
            -1.0
        };
    assert!(force.x.abs() > 1.0 && force.y.abs() > 10.0);
    let expected = SpatialVector::new(
        force,
        (point.point - world.scene.body_world_com(ground).unwrap()).cross(force),
    );
    let compiled = world.gpu.as_ref().unwrap().compiled.as_ref().unwrap();
    let (loads, _) = compiled
        .contacts
        .as_ref()
        .unwrap()
        .reaction_buffers()
        .unwrap();
    let index = world.scene.body_index(ground).unwrap().index() as u64;
    let output = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("immovable contact reaction fixture"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(loads, index * 32, &output, 0, 32);
    encoder.copy_buffer_to_buffer(
        compiled.states[0].body_f.device_buffer().unwrap(),
        index * 32,
        &output,
        32,
        32,
    );
    queue.submit([encoder.finish()]);
    map_read(device, &output).unwrap();
    let bytes = molla_core::gpu_readback!(output.slice(..));
    let actual: SpatialVector =
        bytemuck::pod_read_unaligned::<GpuSpatialVector>(&bytes[..32]).into();
    let predictor: SpatialVector =
        bytemuck::pod_read_unaligned::<GpuSpatialVector>(&bytes[32..]).into();
    assert!(
        (actual.linear - expected.linear).length() < 1e-4,
        "{actual:?} != {expected:?}"
    );
    assert!(
        (actual.angular - expected.angular).length() < 1e-4,
        "{actual:?} != {expected:?}"
    );
    assert_eq!(predictor, SpatialVector::ZERO);
    drop(bytes);
    output.unmap();
    assert_eq!(
        world.scene.body_pose(ground).unwrap(),
        Transform::from_translation(published_position)
    );
    for substeps in [1, 3, 8] {
        world.solver.substeps = substeps;
        world.step(0.002).unwrap();
        assert_eq!(
            world.scene.body_pose(ground).unwrap(),
            Transform::from_translation(published_position)
        );
        assert_eq!(
            world.scene.body_velocity(ground).unwrap(),
            SpatialVector::ZERO
        );
    }
}
