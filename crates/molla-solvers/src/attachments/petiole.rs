//! Rigid leaf blade held to its twig by an elastic petiole.
//!
//! The blade is a six-degree-of-freedom rigid body whose centroid lies
//! `arm = petiole.length + leaf.length / 2` beyond the petiole joint along the blade
//! direction. The joint is tied to the twig anchor by the attachment's translational
//! spring-damper and resists bending and torsion relative to the twig-carried rest
//! frame. Four blade-element plate forces (the free-flight coefficients) with a
//! centre of pressure that moves with incidence drive the motion. The joint spring
//! force is the abscission load. Plate forces are scaled by the canopy sheltering fraction.

use super::{Attachment, BladeFrame, Config, LeafBody, Particle};
use molla_math::Vec3;
use std::f64::consts::{PI, TAU};

const RHO: f64 = 1.225;
/// Blade-element centres as fractions of blade length (along) and width (across).
const ELEMENTS: [(f64, f64); 4] = [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)];
const MAX_SPIN: f64 = 400.0;

/// Elastic petiole joining an attached leaf blade to its twig.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Petiole {
    /// Distance from the twig joint to the blade base, in metres.
    pub length: f64,
    /// Bending stiffness about axes across the petiole, in N·m/rad.
    pub bending: f64,
    /// Torsional stiffness about the petiole axis, in N·m/rad.
    pub torsion: f64,
    /// Fraction of critical structural damping.
    pub damping_ratio: f64,
    /// Fraction of the isolated-plate air force felt by the blade inside its canopy.
    pub sheltering: f64,
}

impl Petiole {
    /// Unsheltered petiole whose bending mode about the joint has frequency `bending_hz`;
    /// torsional stiffness is bending stiffness divided by `twist_to_bend` (EI/GJ).
    pub fn from_frequency(
        mass: f64,
        leaf: LeafBody,
        length: f64,
        bending_hz: f64,
        twist_to_bend: f64,
        damping_ratio: f64,
    ) -> Self {
        let bending = joint_inertia(mass, leaf, length + 0.5 * leaf.length)[1]
            * (TAU * bending_hz).powi(2);
        Self {
            length,
            bending,
            torsion: bending / twist_to_bend,
            damping_ratio,
            sheltering: 1.0,
        }
    }

    pub(super) fn valid(self) -> bool {
        [
            self.length,
            self.bending,
            self.torsion,
            self.damping_ratio,
            self.sheltering,
        ]
        .iter()
        .all(|x| x.is_finite())
            && self.length >= 0.0
            && self.bending > 0.0
            && self.torsion > 0.0
            && self.damping_ratio >= 0.0
            && self.sheltering > 0.0
            && self.sheltering <= 1.0
    }
}

fn added_mass(leaf: LeafBody) -> f64 {
    PI * RHO * leaf.area * leaf.length / 4.0
}

/// Moments of inertia about the centroid along blade direction, span and normal,
/// with the free-flight added air inertia.
fn centroid_inertia(mass: f64, leaf: LeafBody) -> [f64; 3] {
    let added = added_mass(leaf);
    let (l2, w2) = (leaf.length.powi(2), leaf.width.powi(2));
    [
        mass * w2 / 16.0 + added * w2 / 32.0,
        mass * l2 / 16.0 + added * l2 / 32.0,
        mass * (l2 + w2) / 16.0,
    ]
}

/// Moments of inertia about the petiole joint, `arm` from the centroid.
fn joint_inertia(mass: f64, leaf: LeafBody, arm: f64) -> [f64; 3] {
    let [d, s, n] = centroid_inertia(mass, leaf);
    let added = added_mass(leaf);
    [d, s + (mass + added) * arm * arm, n + mass * arm * arm]
}

/// Rotation vector taking the orthonormal frame `rest` onto `current`.
fn rotation_vector(rest: [Vec3; 3], current: [Vec3; 3]) -> Vec3 {
    let mut sine = Vec3::ZERO;
    let mut trace = 0.0;
    for (a, b) in rest.iter().zip(&current) {
        sine += a.cross(*b);
        trace += a.dot(*b);
    }
    let sine = sine * 0.5;
    let s = sine.length();
    if s < 1e-12 {
        sine
    } else {
        sine * (s.atan2(0.5 * (trace - 1.0)) / s)
    }
}

/// Plate force on one element moving with `u` relative to the air.
fn element_force(u: Vec3, n: Vec3, rho_area: f64) -> Vec3 {
    let speed = u.length();
    if speed < 1e-9 {
        return Vec3::ZERO;
    }
    let un = u.dot(n);
    let circulation = -1.2 * rho_area * un / speed * u.cross(n);
    let drag = 0.5 * rho_area * (0.4 + 2.0 * (un / speed).powi(2)) * speed;
    circulation.cross(u) - u * drag
}

/// Centre-of-pressure shift towards the leading edge: a quarter chord at grazing
/// incidence, the element centre at normal incidence.
fn pressure_shift(u: Vec3, d: Vec3, s: Vec3, n: Vec3, leaf: LeafBody) -> Vec3 {
    let tangential = u - n * u.dot(n);
    let (t, speed) = (tangential.length(), u.length());
    if t < 1e-9 || speed < 1e-9 {
        return Vec3::ZERO;
    }
    let e = tangential / t;
    let chord = 1.0 / ((e.dot(d) / leaf.length).powi(2) + (e.dot(s) / leaf.width).powi(2)).sqrt();
    e * (0.25 * chord * t / speed)
}

fn rotate(v: Vec3, axis: Vec3, angle: f64) -> Vec3 {
    let (sin, cos) = angle.sin_cos();
    v * cos + axis.cross(v) * sin + axis * (axis.dot(v) * (1.0 - cos))
}

/// Result of one attached-blade step: joint spring force on the blade and mean air force.
pub(super) struct Held {
    pub joint: Vec3,
    pub air: Vec3,
}

/// Advance an attached blade one step; `pivot` is the anchor's end-of-step position and velocity.
pub(super) fn advance(
    a: Attachment,
    petiole: Petiole,
    p: &mut Particle,
    pivot: (Vec3, Vec3),
    rest: BladeFrame,
    wind: Vec3,
    c: Config,
) -> Held {
    let mut leaf = p.leaf.unwrap();
    let m = a.mass;
    let arm = petiole.length + 0.5 * leaf.length;
    let added = added_mass(leaf);
    let [jd, js, jn] = centroid_inertia(m, leaf);
    let [joint_d, joint_s, _] = joint_inertia(m, leaf, arm);
    let damping_bend = 2.0 * petiole.damping_ratio * (petiole.bending * joint_s).sqrt();
    let damping_twist = 2.0 * petiole.damping_ratio * (petiole.torsion * joint_d).sqrt();
    let (d0, n0) = (rest.direction, rest.normal);
    let rest_axes = [d0, d0.cross(n0), n0];
    let rho_area = RHO * leaf.area * petiole.sheltering / ELEMENTS.len() as f64;
    let preload = (d0 * arm).cross(c.gravity * m);
    let mut position = p.position;
    let mut velocity = p.velocity;
    let mut omega = leaf.frame.angular_velocity;
    let mut d = leaf.frame.direction;
    let mut n = leaf.frame.normal;
    let rate = (60.0 * (wind - velocity).length() + 8.0 * omega.length()).max(480.0);
    let steps = (c.dt * rate).ceil().max(1.0) as usize;
    let h = c.dt / steps as f64;
    let spring = |position: Vec3, velocity: Vec3, omega: Vec3, d: Vec3, time: f64| {
        let joint = -d * arm;
        let anchor = pivot.0 + pivot.1 * (time - c.dt);
        (anchor - position - joint) * a.stiffness
            + (pivot.1 - velocity - omega.cross(joint)) * a.damping
    };
    let mut impulse = Vec3::ZERO;
    for step in 0..steps {
        let s = d.cross(n);
        let mut force = c.gravity * m;
        let mut torque = Vec3::ZERO;
        for (x, z) in ELEMENTS {
            let r = d * (x * leaf.length) + s * (z * leaf.width);
            let u = velocity + omega.cross(r) - wind;
            let f = element_force(u, n, rho_area);
            impulse += f * h;
            force += f;
            torque += (r + pressure_shift(u, d, s, n, leaf) + n * (0.5 * leaf.camber)).cross(f);
        }
        let held = spring(position, velocity, omega, d, h * step as f64);
        force += held;
        let theta = rotation_vector(rest_axes, [d, s, n]);
        let axis = (d + d0).normalize_or(d);
        let twist = axis * theta.dot(axis);
        let relative = omega - rest.angular_velocity;
        let relative_twist = axis * relative.dot(axis);
        torque += (-d * arm).cross(held) - preload
            - (theta - twist) * petiole.bending
            - twist * petiole.torsion
            - (relative - relative_twist) * damping_bend
            - relative_twist * damping_twist;
        velocity += ((force - n * force.dot(n)) / m + n * (force.dot(n) / (m + added))) * h;
        position += velocity * h;
        let momentum = d * (jd * omega.dot(d)) + s * (js * omega.dot(s)) + n * (jn * omega.dot(n));
        let net = torque - omega.cross(momentum);
        omega += (d * (net.dot(d) / jd) + s * (net.dot(s) / js) + n * (net.dot(n) / jn)) * h;
        omega = omega.clamp_length_max(MAX_SPIN);
        let angle = omega.length() * h;
        if angle > 1e-12 {
            let spin = omega.normalize();
            d = rotate(d, spin, angle);
            n = rotate(n, spin, angle);
        }
        d = d.normalize();
        n = (n - d * n.dot(d)).normalize();
    }
    leaf.frame = BladeFrame {
        direction: d,
        normal: n,
        angular_velocity: omega,
    };
    p.leaf = Some(leaf);
    p.position = position;
    p.velocity = velocity;
    Held {
        joint: spring(position, velocity, omega, d, c.dt),
        air: impulse / c.dt,
    }
}
