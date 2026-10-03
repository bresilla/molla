use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor};
use molla_sim::runtime::{
    BodyDesc, ColliderDesc, ColliderGeometry, DistanceLimits, JointDesc, JointHandle, RigidScene,
    RigidSceneBuilder, SphericalConeLimits,
};
use molla_sim::{
    BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, JointTargetMode, JointType,
    MassProperties,
};

pub const CASES: &[&str] = &[
    "distance", "cone", "d6", "reaction", "break", "friction", "mixed",
];

pub fn build(kind: &str, count: usize) -> molla_core::Result<(RigidScene, Vec<JointHandle>)> {
    let mut scene = RigidSceneBuilder::new();
    scene.set_gravity(-Vec3::Y * 10.)?;
    let mut breaks = Vec::new();
    for instance in 0..count {
        let kinds: &[&str] = if kind == "mixed" {
            &CASES[..6]
        } else {
            std::slice::from_ref(&kind)
        };
        for (lane, &primitive) in kinds.iter().enumerate() {
            let origin = Vec3::new(instance as f64 * 16., 2., lane as f64 * 4.);
            let position = origin - Vec3::Y * f64::from(primitive == "distance");
            let body = scene.insert_body(BodyDesc {
                params: BodyParams::new().with_position(position),
                velocity: if primitive == "friction" {
                    SpatialVector::new(Vec3::X * 0.1, Vec3::ZERO)
                } else {
                    SpatialVector::ZERO
                },
                additional_mass: Some(MassProperties {
                    mass: 1.,
                    local_com: if primitive == "cone" {
                        Vec3::X * 0.2
                    } else {
                        Vec3::ZERO
                    },
                    inertia: Mat3::IDENTITY,
                }),
                ..Default::default()
            })?;
            if primitive == "friction" || (kind == "mixed" && primitive == "d6") {
                let material = ContactMaterial {
                    stiffness: 1e4,
                    damping: 100.,
                    friction: Some(FrictionCoefficients {
                        mu_static: 0.8,
                        mu_dynamic: 0.4,
                    }),
                    ..Default::default()
                };
                let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
                ball.parent = Some(body);
                ball.mass = ColliderMass::Mass(0.);
                ball.contact_material = material;
                scene.insert_collider(ball)?;
                let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
                    half_extents: Vec3::new(2., 0.125, 1.),
                });
                let depth = if primitive == "d6" { 0.001 } else { 0. };
                ground.pose = Transform::from_translation(position - Vec3::Y * (0.625 - depth));
                ground.contact_material = material;
                scene.insert_collider(ground)?;
                if primitive == "friction" {
                    continue;
                }
            }
            let joint = scene.insert_joint(JointDesc {
                parent: None,
                child: body,
                kind: match primitive {
                    "distance" => JointType::Distance,
                    "cone" => JointType::Ball,
                    "d6" => JointType::D6,
                    _ => JointType::Fixed,
                },
                axis: Vec3::X,
                frame_parent: Transform::from_translation(origin),
                frame_child: Transform::IDENTITY,
            })?;
            match primitive {
                "distance" => scene.set_distance_limits(
                    joint,
                    DistanceLimits {
                        min: Some(1.),
                        max: Some(1.),
                    },
                )?,
                "cone" => scene.set_cone_limits(
                    joint,
                    SphericalConeLimits {
                        angle0: Some(0.25),
                        angle1: Some(0.4),
                    },
                )?,
                "d6" => {
                    for axis in [
                        JointAxis::LinY,
                        JointAxis::LinZ,
                        JointAxis::AngX,
                        JointAxis::AngY,
                        JointAxis::AngZ,
                    ] {
                        scene.set_joint_limits(joint, axis, Some([0., 0.]))?;
                    }
                    scene.set_joint_motor(
                        joint,
                        JointAxis::LinX,
                        JointMotor {
                            mode: JointTargetMode::Velocity,
                            target_velocity: 0.1,
                            damping: 10.,
                            ..Default::default()
                        },
                    )?;
                }
                "break" if kind != "mixed" || instance == 0 => breaks.push(joint),
                _ => {}
            }
        }
    }
    Ok((scene.build()?, breaks))
}
