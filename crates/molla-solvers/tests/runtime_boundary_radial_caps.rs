use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::JointAxis;
use molla_sim::runtime::{
    BodyDesc, BodyHandle, BodyKind, DistanceLimits, JointDesc, JointHandle, RigidScene,
};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

const DT: f64 = 1e-5;
const FORCE: Vec3 = Vec3::new(0.4, -0.3, 0.2);
const TORQUE: Vec3 = Vec3::new(3., 2., -1.);

struct Case {
    parent: BodyHandle,
    body: BodyHandle,
    tree: JointHandle,
    axis: usize,
    angular_acceleration: Vec3,
    com_acceleration: Vec3,
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    for kind in [BodyKind::Fixed, BodyKind::Kinematic] {
        for sign in [-1., 1.] {
            for (axis, local) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
                let mass = [1., 17., 83.][axis];
                let parent_pose = Transform::new(
                    Vec3::new(2.3, -1.7, 0.9),
                    Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.3, 0.4) * sign,
                );
                let xp = Transform::new(
                    Vec3::new(0.3, -0.2, 0.4),
                    Quat::from_euler(glam::EulerRot::XYZ, -0.1, 0.2, 0.3),
                );
                let xc = Transform::new(
                    Vec3::new(-0.3, 0.5, -0.9),
                    Quat::from_euler(glam::EulerRot::XYZ, -0.4, 0.3, -0.2),
                );
                let pose = parent_pose
                    .compose(&xp)
                    .compose(&Transform::from_translation(local))
                    .compose(&xc.inverse());
                let properties = MassProperties {
                    mass,
                    local_com: Vec3::new(0.4, -0.3, 0.2),
                    inertia: Mat3::from_cols(
                        Vec3::new(3., 0.2, 0.3),
                        Vec3::new(0.2, 4., 0.4),
                        Vec3::new(0.3, 0.4, 5.),
                    ) * mass,
                };
                let parent = scene
                    .insert_body(BodyDesc {
                        kind,
                        params: BodyParams::new().with_pose(parent_pose),
                        additional_mass: Some(MassProperties {
                            mass: 1.,
                            local_com: Vec3::ZERO,
                            inertia: Mat3::IDENTITY,
                        }),
                        ..Default::default()
                    })
                    .unwrap();
                let body = scene
                    .insert_body(BodyDesc {
                        params: BodyParams::new().with_pose(pose),
                        additional_mass: Some(properties),
                        ..Default::default()
                    })
                    .unwrap();
                let desc = JointDesc {
                    parent: Some(parent),
                    child: body,
                    kind: JointType::D6,
                    axis: Vec3::X,
                    frame_parent: xp,
                    frame_child: xc,
                };
                let tree = scene.insert_joint(desc.clone()).unwrap();
                scene
                    .set_joint_velocity_limit(tree, JointAxis::ALL[axis], Some(0.))
                    .unwrap();
                let radial = scene
                    .insert_joint(JointDesc {
                        kind: JointType::Distance,
                        ..desc
                    })
                    .unwrap();
                scene
                    .set_distance_limits(
                        radial,
                        DistanceLimits {
                            min: Some(1.),
                            max: Some(1.),
                        },
                    )
                    .unwrap();
                let normal = parent_pose.rotation * xp.rotation * local;
                let arm = pose.rotation * (properties.local_com - xc.position);
                let rotation = Mat3::from_quat(pose.rotation);
                let inverse = (rotation * properties.inertia * rotation.transpose()).inverse();
                let angular_direction = -arm.cross(normal);
                let free_angular = inverse * TORQUE;
                let free_com = FORCE / mass;
                let mobility = 1. / mass + angular_direction.dot(inverse * angular_direction);
                let reaction = -normal.dot(free_com - free_angular.cross(arm)) / mobility;
                cases.push(Case {
                    parent,
                    body,
                    tree,
                    axis,
                    angular_acceleration: free_angular + inverse * angular_direction * reaction,
                    com_acceleration: free_com + normal * (reaction / mass),
                });
            }
        }
    }
    assert_eq!(cases.len(), 12);
    assert!(!scene.model().requires_angular_lock_rows().unwrap());
    (scene, cases)
}

fn verify(mut world: RigidWorld, cases: &[Case]) {
    world.solver.substeps = 1;
    let parent_poses: Vec<_> = cases
        .iter()
        .map(|c| world.scene.body_pose(c.parent).unwrap())
        .collect();
    for case in cases {
        world
            .scene
            .add_body_wrench(case.body, SpatialVector::new(FORCE, TORQUE))
            .unwrap();
    }
    world.step(DT).unwrap();
    assert!(world.quarantined_bodies().is_empty());
    assert!(world.joint_break_events().is_empty());
    assert_eq!(world.completed_steps(), 1);
    let mut peak_angular: f64 = 0.;
    let mut peak_com: f64 = 0.;
    for (index, (case, initial)) in cases.iter().zip(parent_poses).enumerate() {
        let velocity = world.scene.body_velocity(case.body).unwrap();
        let angular = (velocity.angular / DT - case.angular_acceleration).length()
            / case.angular_acceleration.length().max(1e-10);
        let com = (velocity.linear / DT - case.com_acceleration).length()
            / case.com_acceleration.length().max(1e-10);
        peak_angular = peak_angular.max(angular);
        peak_com = peak_com.max(com);
        assert!(
            angular.is_finite() && angular <= 0.01,
            "case={index} angular_error={angular:e}"
        );
        assert!(
            com.is_finite() && com <= 0.01,
            "case={index} com_error={com:e}"
        );
        let joint = world.scene.joint_index(case.tree).unwrap().index();
        let dof = world.scene.model().joint_dof_offset.host().unwrap()[joint] as usize;
        assert_eq!(
            world.scene.state().joint_qd.host().unwrap()[dof + case.axis],
            0.
        );
        let pose = world.scene.body_pose(case.parent).unwrap();
        assert!((pose.position - initial.position).length() <= 1e-6);
        assert!(
            pose.rotation
                .normalize()
                .angle_between(initial.rotation.normalize())
                .abs()
                <= 1e-6
        );
        let boundary = world.scene.body_velocity(case.parent).unwrap();
        assert_eq!(boundary.linear, Vec3::ZERO);
        assert_eq!(boundary.angular, Vec3::ZERO);
    }
    println!(
        "boundary_radial_cases={} fixed_kinematic=true signs=both axes=XYZ dt={DT} peak_angular_error={peak_angular:.9e} peak_com_error={peak_com:.9e}",
        cases.len()
    );
}

#[test]
fn cpu_fixed_kinematic_radial_caps_preserve_permitted_motion() {
    let (scene, cases) = fixture();
    verify(RigidWorld::from_scene(scene).unwrap(), &cases);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_fixed_kinematic_radial_caps_preserve_permitted_motion() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software);
    println!("adapter={adapter:?} boundary_radial_caps=true");
    let (scene, cases) = fixture();
    verify(RigidWorld::from_scene_gpu(scene, engine).unwrap(), &cases);
}
