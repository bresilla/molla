use super::{AirForces, RotorDisk, invalid};
use crate::devices::{
    CopterCommand, CopterId, CopterLimits, CopterSpec, DragId, DragSpec, MotorLimits, PropellerId,
    PropellerSpec,
};
use crate::rigid_world::RigidWorld;
use molla_core::Result;
use molla_math::{Mat3, Quat, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry};
use molla_sim::{BodyParams, ColliderMass, MassProperties};

#[derive(Clone, Copy, Debug)]
pub struct QuadrotorConfig {
    pub mass: f64,
    pub arm: f64,
    pub rotor_radius: f64,
    pub thrust_coefficient: f64,
    pub torque_coefficient: f64,
    pub maximum_speed: f64,
    pub motor_acceleration: f64,
    pub flight_limits: CopterLimits,
}
impl Default for QuadrotorConfig {
    fn default() -> Self {
        Self {
            mass: 8.0,
            arm: 0.42,
            rotor_radius: 0.28,
            thrust_coefficient: 3.5e-5,
            torque_coefficient: 5.0e-7,
            maximum_speed: 1100.0,
            motor_acceleration: 5000.0,
            flight_limits: CopterLimits {
                max_speed: 2.0,
                max_climb: 1.2,
                max_tilt: 0.35,
                velocity_gain: 2.8,
                position_gain: 1.8,
                attitude_frequency: 9.0,
                ..Default::default()
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RotorAirLoad {
    pub disk: RotorDisk,
    pub force: Vec3,
    pub torque: Vec3,
}
impl RotorAirLoad {
    pub fn deposit(self, forces: &mut AirForces, weight: f64) -> Result<()> {
        if !weight.is_finite() || weight < 0.0 {
            return Err(invalid("invalid rotor load weight"));
        }
        forces.add_disk_load(self.disk, self.force * weight, self.torque * weight)
    }
}

/// Y-up quadrotor using RigidWorld dynamics and its existing cascaded copter controller.
pub struct Quadrotor {
    pub world: RigidWorld,
    pub body: BodyHandle,
    pub controller: CopterId,
    pub propellers: [PropellerId; 4],
    pub config: QuadrotorConfig,
    drag: DragId,
}

impl Quadrotor {
    pub fn new(position: Vec3, config: QuadrotorConfig) -> Result<Self> {
        if !position.is_finite()
            || [
                config.mass,
                config.arm,
                config.rotor_radius,
                config.thrust_coefficient,
                config.torque_coefficient,
                config.maximum_speed,
                config.motor_acceleration,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || config.rotor_radius >= config.arm
            || 4.0 * config.thrust_coefficient * config.maximum_speed.powi(2) <= config.mass * 9.81
        {
            return Err(invalid(
                "invalid quadrotor configuration or insufficient hover thrust",
            ));
        }
        let mut world = RigidWorld::new()?;
        world.scene.set_gravity(-Vec3::Y * 9.81)?;
        let inertia = config.mass * config.arm.powi(2) * Vec3::new(0.55, 0.55, 1.0);
        let body = world.scene.insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::new(
                position,
                Quat::from_rotation_x(-std::f64::consts::FRAC_PI_2),
            )),
            additional_mass: Some(MassProperties {
                mass: config.mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(inertia),
            }),
            ..Default::default()
        })?;
        let mut hull = ColliderDesc::new(ColliderGeometry::Cuboid {
            half_extents: Vec3::new(0.22, 0.22, 0.2),
        });
        hull.parent = Some(body);
        hull.mass = ColliderMass::Mass(0.0);
        world.scene.insert_collider(hull)?;
        let mut rotors = Vec::new();
        for (x, y, spin) in [
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, 1.0),
        ] {
            rotors.push(world.devices.propellers.insert(
                &world.scene,
                PropellerSpec {
                    body,
                    frame: Transform::new(
                        Vec3::new(x * config.arm, y * config.arm, 0.0),
                        Quat::from_rotation_y(-std::f64::consts::FRAC_PI_2),
                    ),
                    thrust: [config.thrust_coefficient * spin, 0.0],
                    torque: [config.torque_coefficient, 0.0],
                    limits: MotorLimits {
                        max_velocity: config.maximum_speed,
                        acceleration: Some(config.motor_acceleration),
                        ..Default::default()
                    },
                    max_torque: config.torque_coefficient * config.maximum_speed.powi(2),
                },
            )?);
        }
        let propellers: [PropellerId; 4] = rotors.try_into().unwrap();
        let controller = world.devices.copters.insert(
            &world.scene,
            &world.devices.propellers,
            CopterSpec {
                body,
                frame: Transform::IDENTITY,
                propellers: propellers.to_vec(),
                limits: config.flight_limits,
            },
        )?;
        let drag = world.devices.aero.insert(
            &world.scene,
            DragSpec {
                body,
                frame: Transform::IDENTITY,
                area: Vec3::new(0.07, 0.07, 0.14),
                density: 1.225,
            },
        )?;
        Ok(Self {
            world,
            body,
            controller,
            propellers,
            config,
            drag,
        })
    }

    pub fn pose(&self) -> Result<Transform> {
        self.world.scene.body_pose(self.body)
    }
    pub fn velocity(&self) -> Result<Vec3> {
        Ok(self.world.scene.body_velocity(self.body)?.linear)
    }

    pub fn command_velocity(&mut self, velocity: Vec3, yaw_rate: f64) -> Result<()> {
        if !velocity.is_finite() || !yaw_rate.is_finite() {
            return Err(invalid("nonfinite flight command"));
        }
        let front = self.pose()?.rotation * Vec3::X;
        let front = Vec3::new(front.x, 0.0, front.z).normalize_or_zero();
        let left = Vec3::Y.cross(front);
        self.world.devices.copters.command(
            self.controller,
            CopterCommand::Velocity {
                forward: velocity.dot(front),
                left: velocity.dot(left),
                up: velocity.y,
                yaw_rate,
            },
        )
    }

    pub fn hold(&mut self, target: Vec3, maximum_velocity: f64) -> Result<()> {
        if !target.is_finite() || !maximum_velocity.is_finite() || maximum_velocity <= 0.0 {
            return Err(invalid("invalid position target"));
        }
        let velocity = ((target - self.pose()?.position) * 1.2).clamp_length_max(maximum_velocity);
        self.command_velocity(velocity, 0.0)
    }

    /// Tracks a world-space flight reference using rotor forces and moments.
    pub fn track(&mut self, reference: super::FlightReference, heading: Vec3) -> Result<()> {
        self.world.devices.copters.command(
            self.controller,
            CopterCommand::Trajectory {
                position: reference.position,
                velocity: reference.velocity,
                acceleration: reference.acceleration,
                heading,
            },
        )
    }

    pub fn stop_motors(&mut self) -> Result<()> {
        self.world
            .devices
            .copters
            .command(self.controller, CopterCommand::Off)
    }

    pub fn step(&mut self, dt: f64, wind: Vec3) -> Result<()> {
        if !dt.is_finite() || dt <= 0.0 || !wind.is_finite() {
            return Err(invalid("invalid drone step"));
        }
        self.world
            .devices
            .aero
            .set_body_wind(self.body, Some(wind))?;
        self.world.step(dt)
    }

    pub fn rotor_frames(&self) -> Result<[Transform; 4]> {
        let pose = self.pose()?;
        Ok(self
            .propellers
            .map(|id| pose.compose(&self.world.devices.propellers.spec(id).unwrap().frame)))
    }

    pub fn air_loads(&self) -> Result<[RotorAirLoad; 4]> {
        let frames = self.rotor_frames()?;
        Ok(std::array::from_fn(|i| {
            let reading = self
                .world
                .devices
                .propellers
                .reading(self.propellers[i])
                .unwrap();
            let force = -reading.world_force;
            let disk = RotorDisk {
                center: frames[i].position,
                axis: frames[i].rotation * Vec3::X,
                radius: self.config.rotor_radius,
                thrust: reading.thrust,
                air_torque: reading.torque,
            };
            RotorAirLoad {
                disk,
                force,
                torque: -reading.world_moment - disk.center.cross(force),
            }
        }))
    }

    pub fn hull_air_load(&self) -> Result<(Vec3, Vec3)> {
        Ok((
            self.pose()?.position,
            -self.world.devices.aero.reading(self.drag).unwrap().force,
        ))
    }
}
