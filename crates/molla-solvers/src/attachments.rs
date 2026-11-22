//! Breakable spring attachments, optional aerodynamic leaf bodies, and plane contact.

use molla_core::{Error, Result};
use molla_math::Vec3;

mod leaf;
mod petiole;
pub use leaf::{BladeFrame, LeafBody};
pub use petiole::Petiole;

#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::GpuAttachments;

#[derive(Clone, Copy, Debug)]
pub struct Attachment {
    pub mass: f64,
    pub stiffness: f64,
    pub damping: f64,
    pub strength: f64,
    pub drag: f64,
    pub radius: f64,
    /// Hinges an attached leaf body to its anchor; `None` keeps the point spring.
    pub petiole: Option<Petiole>,
}

#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    pub position: Vec3,
    pub velocity: Vec3,
    pub valid: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub attached: bool,
    pub age: f64,
    pub force: Vec3,
    pub peak_force: f64,
    /// Air force on the particle before ground contact, in world-space newtons.
    pub air_force: Vec3,
    pub air_torque: Vec3,
    pub leaf: Option<LeafBody>,
    /// Air speed at the particle for its last step, in metres per second.
    pub wind_speed: f64,
}

impl Particle {
    pub fn attached(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            attached: true,
            age: 0.0,
            force: Vec3::ZERO,
            peak_force: 0.0,
            air_force: Vec3::ZERO,
            air_torque: Vec3::ZERO,
            leaf: None,
            wind_speed: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub dt: f64,
    pub gravity: Vec3,
    pub wind: Vec3,
    pub ground: f64,
    pub strength_scale: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dt: 1.0 / 240.0,
            gravity: -Vec3::Y * 9.81,
            wind: Vec3::ZERO,
            ground: 0.0,
            strength_scale: 1.0,
        }
    }
}

fn invalid() -> Error {
    Error::Solver("invalid breakable attachment input".into())
}

fn validate(a: Attachment, p: Particle, anchor: Anchor, c: Config) -> Result<()> {
    let positive = [a.mass, a.stiffness, a.strength, c.dt, c.strength_scale];
    let nonnegative = [a.damping, a.drag, a.radius, p.age, p.peak_force, p.wind_speed];
    if positive.iter().any(|x| !x.is_finite() || *x <= 0.0)
        || nonnegative.iter().any(|x| !x.is_finite() || *x < 0.0)
        || !c.ground.is_finite()
        || [
            p.position,
            p.velocity,
            p.force,
            p.air_force,
            p.air_torque,
            anchor.position,
            anchor.velocity,
            c.gravity,
            c.wind,
        ]
        .iter()
        .any(|v| !v.is_finite())
        || a.petiole.is_some_and(|petiole| !petiole.valid())
    {
        return Err(invalid());
    }
    if let Some(leaf) = p.leaf {
        leaf.validate()?;
        if c.dt > 0.1 {
            return Err(invalid());
        }
    }
    Ok(())
}

/// Advance like the GPU blade path: an attached leaf body with a petiole swings about its
/// anchor from the twig-carried `rest` frame; other attached leaf bodies adopt `rest`.
pub fn step_cpu_blade(
    a: Attachment,
    p: &mut Particle,
    anchor: Anchor,
    rest: BladeFrame,
    c: Config,
) -> Result<()> {
    validate(a, *p, anchor, c)?;
    LeafBody {
        frame: rest,
        length: 1.0,
        width: 1.0,
        area: 1.0,
        camber: 0.0,
    }
    .validate()?;
    let Some(leaf) = p.leaf.filter(|_| p.attached && anchor.valid) else {
        return step_cpu(a, p, anchor, c);
    };
    let Some(petiole) = a.petiole else {
        p.leaf = Some(LeafBody { frame: rest, ..leaf });
        return step_cpu(a, p, anchor, c);
    };
    let mut next = *p;
    let held = petiole::advance(
        a,
        petiole,
        &mut next,
        (anchor.position, anchor.velocity),
        rest,
        c.wind,
        c,
    );
    let load = held.joint.length();
    next.peak_force = next.peak_force.max(load);
    next.attached = load <= a.strength * c.strength_scale;
    next.force = if next.attached { held.joint } else { Vec3::ZERO };
    if !next.attached {
        next.age += c.dt;
    }
    next.air_force = held.air;
    next.wind_speed = c.wind.length();
    next.air_torque = Vec3::ZERO;
    validate(a, next, anchor, c)?;
    *p = next;
    Ok(())
}

/// Advance an attached spring or a detached point/leaf body with local airflow.
pub fn step_cpu(a: Attachment, p: &mut Particle, anchor: Anchor, c: Config) -> Result<()> {
    validate(a, *p, anchor, c)?;
    let mut next = *p;
    let drag = a.drag * (p.velocity - c.wind).length();
    let momentum = p.velocity * a.mass + (c.gravity * a.mass + c.wind * drag) * c.dt;
    let free = momentum / (a.mass + c.dt * drag);
    next.force = Vec3::ZERO;
    if p.attached && anchor.valid {
        next.velocity = (momentum
            + ((anchor.position - p.position) * a.stiffness + anchor.velocity * a.damping) * c.dt)
            / (a.mass + c.dt * (drag + a.damping) + c.dt * c.dt * a.stiffness);
        let x = p.position + next.velocity * c.dt;
        next.force =
            (anchor.position - x) * a.stiffness + (anchor.velocity - next.velocity) * a.damping;
        let load = next.force.length();
        next.peak_force = next.peak_force.max(load);
        next.attached = load <= a.strength * c.strength_scale;
    } else {
        next.attached = false;
    }
    if !next.attached {
        next.velocity = free;
        next.force = Vec3::ZERO;
        next.age += c.dt;
    }
    next.air_force = (c.wind - next.velocity) * drag;
    next.wind_speed = c.wind.length();
    next.air_torque = Vec3::ZERO;
    if !next.attached && p.leaf.is_some() {
        next.position = p.position;
        next.velocity = p.velocity;
        leaf::advance(a, &mut next, c);
        validate(a, next, anchor, c)?;
        *p = next;
        return Ok(());
    }
    next.position = p.position + next.velocity * c.dt;
    if next.position.y < c.ground + a.radius {
        next.position.y = c.ground + a.radius;
        if next.velocity.y < 0.0 {
            next.velocity.y *= -0.08;
        }
        let friction = (-18.0 * c.dt).exp();
        next.velocity.x *= friction;
        next.velocity.z *= friction;
    }
    validate(a, next, anchor, c)?;
    *p = next;
    Ok(())
}
