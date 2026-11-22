use super::{Attachment, Config, Particle, invalid};
use molla_core::Result;
use molla_math::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BladeFrame {
    pub direction: Vec3,
    pub normal: Vec3,
    pub angular_velocity: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeafBody {
    pub frame: BladeFrame,
    pub length: f64,
    pub width: f64,
    /// Effective aerodynamic area in square metres, including sample weight.
    pub area: f64,
    /// Curvature height in metres; zero gives a planar blade.
    pub camber: f64,
}

impl LeafBody {
    pub(super) fn validate(self) -> Result<()> {
        if [self.length, self.width, self.area]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
            || !self.camber.is_finite()
            || self.camber < 0.0
            || self.camber > 0.2 * self.length.min(self.width)
            || !self.frame.direction.is_finite()
            || !self.frame.normal.is_finite()
            || !self.frame.angular_velocity.is_finite()
            || (self.frame.direction.length_squared() - 1.0).abs() > 1e-4
            || (self.frame.normal.length_squared() - 1.0).abs() > 1e-4
            || self.frame.direction.dot(self.frame.normal).abs() > 1e-4
        {
            return Err(invalid());
        }
        Ok(())
    }

    /// Six rim points and the center, in world-oriented coordinates about the mass center.
    pub fn contact_offsets(self) -> [Vec3; 7] {
        let d = self.frame.direction;
        let n = self.frame.normal;
        let s = d.cross(n);
        [
            (-0.5, 0.0),
            (-1.0 / 6.0, 0.5),
            (1.0 / 6.0, 0.5),
            (0.5, 0.0),
            (1.0 / 6.0, -0.5),
            (-1.0 / 6.0, -0.5),
            (0.0, 0.0),
        ]
        .map(|(x, z)| {
            d * (x * self.length)
                + s * (z * self.width)
                + n * (self.camber * (4.0 * x * x + 4.0 * z * z - 0.5))
        })
    }

    pub fn inverse_inertia(self, mass: f64, torque: Vec3) -> Vec3 {
        let d = self.frame.direction;
        let n = self.frame.normal;
        let s = d.cross(n);
        let ix = mass * self.width * self.width / 16.0;
        let iz = mass * self.length * self.length / 16.0;
        d * (torque.dot(d) / ix) + s * (torque.dot(s) / iz) + n * (torque.dot(n) / (ix + iz))
    }

    pub fn angular_momentum(self, mass: f64) -> Vec3 {
        let d = self.frame.direction;
        let n = self.frame.normal;
        let s = d.cross(n);
        let w = self.frame.angular_velocity;
        let ix = mass * self.width * self.width / 16.0;
        let iz = mass * self.length * self.length / 16.0;
        d * (w.dot(d) * ix) + s * (w.dot(s) * iz) + n * (w.dot(n) * (ix + iz))
    }

    /// Kinetic energy of the blade and its unresolved added air in uniform flow.
    pub fn kinetic_energy(self, mass: f64, relative_velocity: Vec3) -> f64 {
        let added = std::f64::consts::PI * 1.225 * self.area * self.length / 4.0;
        let n = self.frame.normal;
        let d = self.frame.direction;
        let s = d.cross(n);
        let w = self.frame.angular_velocity;
        0.5 * (mass * relative_velocity.length_squared()
            + added * relative_velocity.dot(n).powi(2)
            + w.dot(self.angular_momentum(mass))
            + added / 32.0
                * (self.width.powi(2) * w.dot(d).powi(2) + self.length.powi(2) * w.dot(s).powi(2)))
    }
}

#[derive(Clone, Copy)]
struct Flight {
    velocity: Vec3,
    omega: Vec3,
    direction: Vec3,
    normal: Vec3,
}

impl Flight {
    fn offset(self, rate: Self, dt: f64) -> Self {
        Self {
            velocity: self.velocity + rate.velocity * dt,
            omega: self.omega + rate.omega * dt,
            direction: self.direction + rate.direction * dt,
            normal: self.normal + rate.normal * dt,
        }
    }
}

// APW (2005b), equations 3.1–3.9, with a three-dimensional circulation vector.
fn derivative(f: Flight, leaf: LeafBody, mass: f64, c: Config) -> Flight {
    let d = f.direction.normalize();
    let n = (f.normal - d * f.normal.dot(d)).normalize();
    let s = d.cross(n);
    let u = f.velocity - c.wind;
    let un = u.dot(n);
    let speed = u.length().max(1e-12);
    let rho_area = 1.225 * leaf.area;
    let added = std::f64::consts::PI * rho_area * leaf.length / 4.0;
    let circulation = -1.2 * rho_area * un / speed * u.cross(n)
        + 0.5
            * std::f64::consts::PI
            * rho_area
            * (s * (leaf.length * f.omega.dot(s)) + d * (leaf.width * f.omega.dot(d)));
    let pressure_arm = n * (0.5 * leaf.camber);
    let surface_velocity = u + f.omega.cross(pressure_arm);
    let surface_speed = surface_velocity.length().max(1e-12);
    let surface_normal = surface_velocity.dot(n);
    let drag = 0.5
        * rho_area
        * (0.4 + 2.0 * surface_normal.powi(2) / surface_speed.powi(2))
        * surface_speed;
    let drag_force = -drag * surface_velocity;
    let an = n * (added * un);
    let force = c.gravity * mass + circulation.cross(u) + drag_force - f.omega.cross(an)
        + n * (added * f.omega.cross(u).dot(n));
    let acceleration = (force - n * force.dot(n)) / mass + n * (force.dot(n) / (mass + added));
    let ix = mass * leaf.width.powi(2) / 16.0;
    let iz = mass * leaf.length.powi(2) / 16.0;
    let jx = ix + added * leaf.width.powi(2) / 32.0;
    let jz = iz + added * leaf.length.powi(2) / 32.0;
    let jn = ix + iz;
    let momentum =
        d * (jx * f.omega.dot(d)) + s * (jz * f.omega.dot(s)) + n * (jn * f.omega.dot(n));
    let terminal = (2.0 * mass * 9.81 / (std::f64::consts::PI * rho_area)).sqrt();
    let mut damping = Vec3::ZERO;
    for (axis, chord) in [(d, leaf.width), (s, leaf.length), (n, leaf.width)] {
        let spin = f.omega.dot(axis);
        let coefficient = added * chord * chord * 0.05 * (2.0 * terminal / chord + spin.abs());
        damping += axis * (coefficient * spin);
    }
    let torque = an.cross(u) - damping - f.omega.cross(momentum) + pressure_arm.cross(drag_force);
    Flight {
        velocity: acceleration,
        omega: d * (torque.dot(d) / jx) + s * (torque.dot(s) / jz) + n * (torque.dot(n) / jn),
        direction: f.omega.cross(f.direction),
        normal: f.omega.cross(f.normal),
    }
}

pub(super) fn advance(a: Attachment, p: &mut Particle, c: Config) {
    let mut leaf = p.leaf.unwrap();
    let rate = (32.0 * (p.velocity - c.wind).length() / leaf.length.min(leaf.width)
        + 16.0 * leaf.frame.angular_velocity.length())
    .max(480.0);
    let steps = (c.dt * rate).ceil().max(1.0) as usize;
    let dt = c.dt / steps as f64;
    let mut air_impulse = Vec3::ZERO;
    let mut air_moment = Vec3::ZERO;
    for _ in 0..steps {
        let old_velocity = p.velocity;
        let old_momentum = leaf.angular_momentum(a.mass);
        let f = Flight {
            velocity: p.velocity,
            omega: leaf.frame.angular_velocity,
            direction: leaf.frame.direction,
            normal: leaf.frame.normal,
        };
        let k1 = derivative(f, leaf, a.mass, c);
        let f2 = f.offset(k1, dt * 0.5);
        let k2 = derivative(f2, leaf, a.mass, c);
        let f3 = f.offset(k2, dt * 0.5);
        let k3 = derivative(f3, leaf, a.mass, c);
        let f4 = f.offset(k3, dt);
        let k4 = derivative(f4, leaf, a.mass, c);
        let next = f
            .offset(k1, dt / 6.0)
            .offset(k2, dt / 3.0)
            .offset(k3, dt / 3.0)
            .offset(k4, dt / 6.0);
        p.position +=
            (f.velocity + 2.0 * f2.velocity + 2.0 * f3.velocity + f4.velocity) * (dt / 6.0);
        p.velocity = next.velocity;
        leaf.frame.direction = next.direction.normalize();
        leaf.frame.normal = (next.normal
            - leaf.frame.direction * next.normal.dot(leaf.frame.direction))
        .normalize();
        leaf.frame.angular_velocity = next.omega;
        air_impulse += a.mass * (p.velocity - old_velocity - c.gravity * dt);
        air_moment += leaf.angular_momentum(a.mass) - old_momentum;
        let d = leaf.frame.direction;
        let side = d.cross(leaf.frame.normal);
        let mut corners = leaf.contact_offsets();
        let count = if leaf.camber > 0.0 { 7 } else { 4 };
        if leaf.camber == 0.0 {
            for (i, r) in corners[..4].iter_mut().enumerate() {
                *r = d * (leaf.length * if i & 1 == 0 { -0.5 } else { 0.5 })
                    + side * (leaf.width * if i & 2 == 0 { -0.5 } else { 0.5 });
            }
        }
        let depth = c.ground + a.radius
            - p.position.y
            - corners[..count]
                .iter()
                .map(|r| r.y)
                .fold(f64::INFINITY, f64::min);
        if depth > -0.0005 {
            p.position.y += depth.max(0.0);
            let mut normal_impulses = [0.0; 7];
            let mut friction_impulses = [Vec3::ZERO; 7];
            let bounce = corners.map(|r| {
                let v = (p.velocity + leaf.frame.angular_velocity.cross(r)).y;
                if v < -0.5 { -0.02 * v } else { 0.0 }
            });
            for _ in 0..4 {
                for (i, r) in corners[..count].iter().copied().enumerate() {
                    if p.position.y + r.y > c.ground + a.radius + 0.0005 {
                        continue;
                    }
                    let velocity = p.velocity + leaf.frame.angular_velocity.cross(r);
                    let inverse = a.mass.recip()
                        + Vec3::Y.dot(leaf.inverse_inertia(a.mass, r.cross(Vec3::Y)).cross(r));
                    let next = (normal_impulses[i] + (bounce[i] - velocity.y) / inverse).max(0.0);
                    let j = Vec3::Y * (next - normal_impulses[i]);
                    normal_impulses[i] = next;
                    p.velocity += j / a.mass;
                    leaf.frame.angular_velocity += leaf.inverse_inertia(a.mass, r.cross(j));
                    let velocity = p.velocity + leaf.frame.angular_velocity.cross(r);
                    let tangent = Vec3::new(velocity.x, 0.0, velocity.z).normalize_or_zero();
                    if tangent.length_squared() > 0.0 {
                        let inverse = a.mass.recip()
                            + tangent.dot(leaf.inverse_inertia(a.mass, r.cross(tangent)).cross(r));
                        let next = (friction_impulses[i]
                            - tangent * (velocity.dot(tangent) / inverse))
                            .clamp_length_max(0.3 * normal_impulses[i]);
                        let j = next - friction_impulses[i];
                        friction_impulses[i] = next;
                        p.velocity += j / a.mass;
                        leaf.frame.angular_velocity += leaf.inverse_inertia(a.mass, r.cross(j));
                    }
                }
            }
        }
    }
    p.leaf = Some(leaf);
    p.air_force = air_impulse / c.dt;
    p.air_torque = air_moment / c.dt;
}
