use molla_math::{Mat3, Transform, Vec3};
use molla_sim::joint_control::JointSoftness;
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, JointHandle, RigidScene};
use molla_sim::{JointType, MassProperties};

fn records(scene: &RigidScene) -> Vec<(u32, u32, f64, f64, f64, f64)> {
    scene
        .model()
        .joint_compliance
        .host()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c.enabled,
                c.padding,
                c.lower,
                c.upper,
                c.natural_frequency,
                c.damping_ratio,
            )
        })
        .collect()
}

fn fixture(external: bool, two_body: bool) -> (RigidScene, [BodyHandle; 3], JointHandle) {
    let mut scene = RigidScene::new().unwrap();
    let bodies = std::array::from_fn(|_| {
        scene
            .insert_body(BodyDesc {
                additional_mass: Some(MassProperties {
                    mass: 1.,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY,
                }),
                ..Default::default()
            })
            .unwrap()
    });
    let desc = JointDesc {
        parent: two_body.then_some(bodies[0]),
        child: bodies[1],
        kind: JointType::Fixed,
        axis: Vec3::X,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
    };
    let softness = JointSoftness {
        natural_frequency: 3.,
        damping_ratio: 0.5,
    };
    let joint = if external {
        scene.insert_loop_joint(desc, softness).unwrap()
    } else {
        let joint = scene.insert_joint(desc).unwrap();
        scene.set_joint_softness(joint, Some(softness)).unwrap();
        joint
    };
    (scene, bodies, joint)
}

#[test]
fn hot_gain_edits_activate_only_authored_endpoints_without_rebuilding() {
    for external in [false, true] {
        for two_body in [false, true] {
            let (mut scene, bodies, joint) = fixture(external, two_body);
            let activation = bodies.map(|body| scene.activation_stamp(body));
            let poses = bodies.map(|body| scene.body_pose(body).unwrap());
            let velocities = bodies.map(|body| scene.body_velocity(body).unwrap());
            let stamp = scene.stamp();
            let model_stamp = scene.model_stamp();
            let dynamics = scene.dynamics_stamp();
            let revision = scene.revision();
            let updated = JointSoftness {
                natural_frequency: 6.,
                damping_ratio: 1.,
            };
            scene.set_joint_softness(joint, Some(updated)).unwrap();
            assert_eq!(scene.joint_softness(joint).unwrap(), Some(updated));
            assert_eq!(scene.stamp(), stamp);
            assert_eq!(scene.model_stamp(), model_stamp);
            assert_eq!(scene.revision(), revision);
            assert_ne!(scene.dynamics_stamp(), dynamics);
            for (i, body) in bodies.into_iter().enumerate() {
                assert_eq!(
                    scene.activation_stamp(body) != activation[i],
                    i == 1 || (i == 0 && two_body)
                );
                assert_eq!(scene.body_pose(body).unwrap(), poses[i]);
                assert_eq!(scene.body_velocity(body).unwrap(), velocities[i]);
            }
            if let Some(index) = scene.joint_index(joint) {
                let offsets = scene.model().joint_dof_offset.host().unwrap();
                let range = offsets[index.index()] as usize..offsets[index.index() + 1] as usize;
                assert!(
                    scene.model().joint_compliance.host().unwrap()[range]
                        .iter()
                        .all(|c| {
                            c.natural_frequency == updated.natural_frequency
                                && c.damping_ratio == updated.damping_ratio
                        })
                );
            }
        }
    }
}

#[test]
fn identical_and_rejected_gains_preserve_activation_and_parameter_records() {
    for external in [false, true] {
        let (mut scene, bodies, joint) = fixture(external, true);
        let activation = bodies.map(|body| scene.activation_stamp(body));
        let stamp = scene.stamp();
        let model_stamp = scene.model_stamp();
        let dynamics = scene.dynamics_stamp();
        let revision = scene.revision();
        let original = scene.joint_softness(joint).unwrap();
        let before = records(&scene);
        scene.set_joint_softness(joint, original).unwrap();
        for invalid in [
            JointSoftness {
                natural_frequency: f64::NAN,
                damping_ratio: 1.,
            },
            JointSoftness {
                natural_frequency: -1.,
                damping_ratio: 1.,
            },
            JointSoftness {
                natural_frequency: f64::MAX,
                damping_ratio: 1.,
            },
            JointSoftness {
                natural_frequency: 1.,
                damping_ratio: -1.,
            },
            JointSoftness {
                natural_frequency: 1.,
                damping_ratio: f64::INFINITY,
            },
        ] {
            assert!(scene.set_joint_softness(joint, Some(invalid)).is_err());
            assert_eq!(scene.joint_softness(joint).unwrap(), original);
            assert_eq!(scene.stamp(), stamp);
            assert_eq!(scene.model_stamp(), model_stamp);
            assert_eq!(scene.dynamics_stamp(), dynamics);
            assert_eq!(scene.revision(), revision);
            assert_eq!(bodies.map(|body| scene.activation_stamp(body)), activation);
            assert_eq!(records(&scene), before);
        }
    }
}
