//! V1/V2 — tire force-element models.
//!
//! A tire is a **force element, not a collider**. Each step the wheel
//! queries the [`Terrain`](crate::Terrain) under its hub, derives the
//! contact patch, computes the longitudinal slip ratio `κ` and the
//! lateral slip angle `α` from the wheel kinematics, and a
//! [`TireModel`] maps `(κ, α, Fz)` to a ground force. The result is a
//! world-space wrench applied to the wheel body — no GJK runs against
//! the ground.
//!
//! - **V1** — [`RigidTire`] and [`TmSimpleTire`].
//! - **V2** — [`Pac89Tire`] (Pacejka Magic Formula).

use crate::terrain::Terrain;

/// Contact-frame force law for a tire. Implementors return the
/// longitudinal force `Fx`, lateral force `Fy`, and aligning moment
/// `Mz`, already kept inside the friction ellipse `‖(Fx,Fy)‖ ≤ μ·Fz`.
pub trait TireModel: Send + Sync {
    /// Evaluate the tire at longitudinal slip ratio `slip_ratio`,
    /// lateral `slip_angle` (rad), normal load `fz`, surface friction
    /// `friction`. Returns `(fx, fy, mz)` in the contact frame.
    fn evaluate(&self, slip_ratio: f64, slip_angle: f64, fz: f64, friction: f64)
    -> (f64, f64, f64);
}

/// Clamp a `(fx, fy)` pair to the friction-circle radius `limit`.
pub fn friction_ellipse(fx: f64, fy: f64, limit: f64) -> (f64, f64) {
    let mag = (fx * fx + fy * fy).sqrt();
    if mag > limit && mag > 1e-12 {
        let s = limit / mag;
        (fx * s, fy * s)
    } else {
        (fx, fy)
    }
}

/// A stiff linear tire — force rises steeply with slip, then saturates
/// hard at the friction limit. The simplest [`TireModel`].
#[derive(Clone, Copy, Debug)]
pub struct RigidTire {
    /// Longitudinal slip stiffness (N per unit slip ratio).
    pub c_kappa: f64,
    /// Cornering stiffness (N per rad of slip angle).
    pub c_alpha: f64,
}

impl Default for RigidTire {
    fn default() -> Self {
        Self {
            c_kappa: 1.2e5,
            c_alpha: 8.0e4,
        }
    }
}

impl TireModel for RigidTire {
    fn evaluate(
        &self,
        slip_ratio: f64,
        slip_angle: f64,
        fz: f64,
        friction: f64,
    ) -> (f64, f64, f64) {
        let limit = friction * fz;
        let fx = self.c_kappa * slip_ratio;
        let fy = -self.c_alpha * slip_angle;
        let (fx, fy) = friction_ellipse(fx, fy, limit);
        (fx, fy, 0.0)
    }
}

/// A simple smooth empirical tire ("TMsimple"-style): the force
/// saturates smoothly via `tanh`, so it is differentiable and has no
/// hard corner at the friction limit.
#[derive(Clone, Copy, Debug)]
pub struct TmSimpleTire {
    /// Longitudinal slip stiffness (N per unit slip ratio).
    pub c_kappa: f64,
    /// Cornering stiffness (N per rad).
    pub c_alpha: f64,
    /// Pneumatic trail (m) — sets the aligning moment `Mz = -trail·Fy`.
    pub pneumatic_trail: f64,
}

impl Default for TmSimpleTire {
    fn default() -> Self {
        Self {
            c_kappa: 1.0e5,
            c_alpha: 6.0e4,
            pneumatic_trail: 0.03,
        }
    }
}

impl TireModel for TmSimpleTire {
    fn evaluate(
        &self,
        slip_ratio: f64,
        slip_angle: f64,
        fz: f64,
        friction: f64,
    ) -> (f64, f64, f64) {
        let limit = (friction * fz).max(1e-9);
        let fx = limit * (self.c_kappa * slip_ratio / limit).tanh();
        let fy = -limit * (self.c_alpha * slip_angle / limit).tanh();
        let (fx, fy) = friction_ellipse(fx, fy, limit);
        (fx, fy, -self.pneumatic_trail * fy)
    }
}

/// Pacejka 1989 "Magic Formula" tire (V2). Longitudinal and lateral
/// forces follow `D·sin(C·atan(B·s − E·(B·s − atan(B·s))))`, scaled to
/// the friction limit and combined via the friction ellipse. The
/// formula is smooth, so this tire is autograd-friendly.
#[derive(Clone, Copy, Debug)]
pub struct Pac89Tire {
    /// Longitudinal stiffness factor `B`.
    pub b_long: f64,
    /// Longitudinal shape factor `C`.
    pub c_long: f64,
    /// Longitudinal curvature factor `E`.
    pub e_long: f64,
    /// Lateral stiffness factor `B`.
    pub b_lat: f64,
    /// Lateral shape factor `C`.
    pub c_lat: f64,
    /// Lateral curvature factor `E`.
    pub e_lat: f64,
    /// Pneumatic trail (m).
    pub pneumatic_trail: f64,
}

impl Default for Pac89Tire {
    fn default() -> Self {
        Self {
            b_long: 10.0,
            c_long: 1.65,
            e_long: 0.97,
            b_lat: 8.0,
            c_lat: 1.30,
            e_lat: 0.97,
            pneumatic_trail: 0.03,
        }
    }
}

/// Magic-Formula core: `sin(C·atan(B·s − E·(B·s − atan(B·s))))`, in
/// `[-1, 1]`. Smooth in `s`.
pub fn magic_formula(s: f64, b: f64, c: f64, e: f64) -> f64 {
    let bs = b * s;
    (c * (bs - e * (bs - bs.atan())).atan()).sin()
}

/// Analytic derivative of [`magic_formula`] — returns `(∂/∂s, ∂/∂b)`.
/// The Magic Formula is smooth, so a Pacejka tire is differentiable;
/// this is what makes gradient-based tire-parameter identification (V6)
/// possible — something Chrono's tires cannot do.
pub fn magic_formula_deriv(s: f64, b: f64, c: f64, e: f64) -> (f64, f64) {
    let bs = b * s;
    let inner = bs - e * (bs - bs.atan());
    let phi = c * inner.atan();
    let cos_phi = phi.cos();
    let d_outer = c / (1.0 + inner * inner);
    // ∂inner/∂bs = 1 − e·(1 − 1/(1+bs²)); chain in ∂bs/∂s = b, ∂bs/∂b = s.
    let d_inner_d_bs = 1.0 - e * (1.0 - 1.0 / (1.0 + bs * bs));
    let d_ds = cos_phi * d_outer * d_inner_d_bs * b;
    let d_db = cos_phi * d_outer * d_inner_d_bs * s;
    (d_ds, d_db)
}

impl TireModel for Pac89Tire {
    fn evaluate(
        &self,
        slip_ratio: f64,
        slip_angle: f64,
        fz: f64,
        friction: f64,
    ) -> (f64, f64, f64) {
        let limit = friction * fz;
        // The Magic Formula returns a normalized [-1,1] shape; scale to
        // the friction limit. Lateral slip uses tan(alpha) so the input
        // is the lateral-to-longitudinal velocity ratio.
        let fx = limit * magic_formula(slip_ratio, self.b_long, self.c_long, self.e_long);
        let fy = -limit * magic_formula(slip_angle.tan(), self.b_lat, self.c_lat, self.e_lat);
        let (fx, fy) = friction_ellipse(fx, fy, limit);
        (fx, fy, -self.pneumatic_trail * fy)
    }
}

/// The per-step kinematic state of one wheel.
#[derive(Clone, Copy, Debug)]
pub struct WheelContact {
    /// World position of the wheel hub.
    pub hub_pos: [f64; 3],
    /// World linear velocity of the hub.
    pub hub_vel: [f64; 3],
    /// Unit forward (rolling) direction — already includes steer.
    pub heading: [f64; 3],
    /// Wheel spin rate about its axle (rad/s); `spin·radius` is the
    /// circumferential speed.
    pub spin: f64,
    /// Wheel radius (m).
    pub radius: f64,
    /// Normal-load stiffness `Fz = k·penetration` (N/m).
    pub normal_stiffness: f64,
    /// Normal-load damping (N·s/m).
    pub normal_damping: f64,
}

/// The ground wrench a tire produces this step.
#[derive(Clone, Copy, Debug)]
pub struct TireOutput {
    /// Whether the wheel is touching the ground.
    pub in_contact: bool,
    /// World-space ground force on the wheel.
    pub force: [f64; 3],
    /// World contact point (on the tire, below the hub).
    pub contact_point: [f64; 3],
    /// Aligning moment about the surface normal (N·m).
    pub aligning_moment: f64,
    /// Normal load (N).
    pub fz: f64,
    /// Longitudinal slip ratio.
    pub slip_ratio: f64,
    /// Lateral slip angle (rad).
    pub slip_angle: f64,
}

impl TireOutput {
    fn none() -> Self {
        Self {
            in_contact: false,
            force: [0.0; 3],
            contact_point: [0.0; 3],
            aligning_moment: 0.0,
            fz: 0.0,
            slip_ratio: 0.0,
            slip_angle: 0.0,
        }
    }
}

/// Compute the ground wrench on a wheel: query the terrain, derive the
/// contact patch and slip, evaluate the tire model. This is the V1
/// force-element pipeline — no collision detection.
pub fn wheel_ground_wrench(
    w: &WheelContact,
    terrain: &dyn Terrain,
    tire: &dyn TireModel,
) -> TireOutput {
    let (distance, normal) = terrain.signed_distance_and_normal(w.hub_pos);
    let n = normalize(normal);
    let penetration = w.radius - distance;
    if penetration <= 0.0 {
        return TireOutput::none();
    }
    // Normal load — penalty spring + damping on the closing speed.
    let velocity = sub(
        w.hub_vel,
        terrain.surface_velocity(w.hub_pos[0], w.hub_pos[2]),
    );
    let approach = -dot(velocity, n);
    let fz = (w.normal_stiffness * penetration + w.normal_damping * approach).max(0.0);

    // Contact frame: longitudinal = heading projected onto the ground
    // plane; lateral completes a right-handed frame with the normal.
    let long = normalize(sub(w.heading, scale(n, dot(w.heading, n))));
    let lat = cross(n, long);
    let v_long = dot(velocity, long);
    let v_lat = dot(velocity, lat);
    let v_eps = 0.3; // standstill regularisation (m/s)
    let denom = v_long.abs().max(v_eps);
    let slip_ratio = (w.spin * w.radius - v_long) / denom;
    let slip_angle = (v_lat / denom).atan();

    let (fx, fy, mz) = tire.evaluate(
        slip_ratio,
        slip_angle,
        fz,
        terrain.friction_at(w.hub_pos[0], w.hub_pos[2]),
    );
    let force = [
        fx * long[0] + fy * lat[0] + fz * n[0],
        fx * long[1] + fy * lat[1] + fz * n[1],
        fx * long[2] + fy * lat[2] + fz * n[2],
    ];
    TireOutput {
        in_contact: true,
        force,
        contact_point: sub(w.hub_pos, scale(n, w.radius)),
        aligning_moment: mz,
        fz,
        slip_ratio,
        slip_angle,
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalize(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt();
    if l > 1e-12 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        [1.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::FlatTerrain;

    #[test]
    fn wheel_wrench_uses_local_friction_and_ground_relative_velocity() {
        struct MovingPatch;
        impl Terrain for MovingPatch {
            fn height(&self, _: f64, _: f64) -> f64 {
                0.0
            }
            fn normal(&self, _: f64, _: f64) -> [f64; 3] {
                [0.0, 1.0, 0.0]
            }
            fn friction(&self) -> f64 {
                1.0
            }
            fn friction_at(&self, _: f64, _: f64) -> f64 {
                0.2
            }
            fn surface_velocity(&self, _: f64, _: f64) -> [f64; 3] {
                [2.0, 0.0, 0.0]
            }
        }
        let mut wheel = WheelContact {
            hub_pos: [0.0, 0.49, 0.0],
            hub_vel: [2.0, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            spin: 0.0,
            radius: 0.5,
            normal_stiffness: 1e5,
            normal_damping: 100.0,
        };
        let output = wheel_ground_wrench(&wheel, &MovingPatch, &RigidTire::default());
        assert_eq!(output.slip_ratio, 0.0);
        assert_eq!(output.force[0], 0.0);
        wheel.spin = 100.0;
        let output = wheel_ground_wrench(&wheel, &MovingPatch, &RigidTire::default());
        assert!((output.force[0] - 0.2 * output.fz).abs() < 1e-8);
    }

    #[test]
    fn wheel_penetration_is_normal_distance_on_a_slope() {
        struct Slope;
        impl Terrain for Slope {
            fn height(&self, x: f64, _: f64) -> f64 {
                -0.75 * x
            }
            fn normal(&self, _: f64, _: f64) -> [f64; 3] {
                [0.6, 0.8, 0.0]
            }
            fn friction(&self) -> f64 {
                0.0
            }
        }
        let wheel = WheelContact {
            hub_pos: [0.0, 0.49 / 0.8, 0.0],
            hub_vel: [0.0; 3],
            heading: [0.0, 0.0, 1.0],
            spin: 0.0,
            radius: 0.5,
            normal_stiffness: 1e5,
            normal_damping: 0.0,
        };
        let output = wheel_ground_wrench(&wheel, &Slope, &RigidTire::default());
        assert!((output.fz - 1000.0).abs() < 1e-8);
        assert!((output.force[0] - 600.0).abs() < 1e-8);
        assert!((output.force[1] - 800.0).abs() < 1e-8);
    }

    #[test]
    fn force_signs_are_physical() {
        // Drive (kappa > 0) → forward force; brake (kappa < 0) → backward.
        for tire in tire_zoo() {
            let (fx_drive, _, _) = tire.evaluate(0.1, 0.0, 3000.0, 0.9);
            let (fx_brake, _, _) = tire.evaluate(-0.1, 0.0, 3000.0, 0.9);
            assert!(fx_drive > 0.0, "drive slip must push forward");
            assert!(fx_brake < 0.0, "brake slip must push backward");
            // Positive slip angle → lateral force opposing it (negative).
            let (_, fy, _) = tire.evaluate(0.0, 0.08, 3000.0, 0.9);
            assert!(
                fy < 0.0,
                "positive slip angle must give a restoring lateral force"
            );
        }
    }

    #[test]
    fn forces_respect_the_friction_ellipse() {
        for tire in tire_zoo() {
            let fz = 3000.0;
            let mu = 0.9;
            // Huge combined slip — force must stay within mu*Fz.
            let (fx, fy, _) = tire.evaluate(5.0, 1.2, fz, mu);
            let mag = (fx * fx + fy * fy).sqrt();
            assert!(
                mag <= mu * fz + 1.0,
                "force {mag} exceeded friction limit {}",
                mu * fz
            );
        }
    }

    #[test]
    fn magic_formula_derivative_matches_finite_difference() {
        let (b, c, e) = (10.0, 1.65, 0.97);
        let eps = 1.0e-6;
        for &s in &[-0.3, -0.05, 0.05, 0.2, 0.6] {
            let (d_ds, d_db) = magic_formula_deriv(s, b, c, e);
            let fd_s =
                (magic_formula(s + eps, b, c, e) - magic_formula(s - eps, b, c, e)) / (2.0 * eps);
            let fd_b =
                (magic_formula(s, b + eps, c, e) - magic_formula(s, b - eps, c, e)) / (2.0 * eps);
            assert!(
                (d_ds - fd_s).abs() < 1e-4,
                "∂/∂s {d_ds} vs FD {fd_s} at s={s}"
            );
            assert!(
                (d_db - fd_b).abs() < 1e-4,
                "∂/∂b {d_db} vs FD {fd_b} at s={s}"
            );
        }
    }

    fn tire_zoo() -> Vec<Box<dyn TireModel>> {
        vec![
            Box::new(RigidTire::default()),
            Box::new(TmSimpleTire::default()),
            Box::new(Pac89Tire::default()),
        ]
    }

    /// Minimal single-wheel integrator for the rolling test.
    struct WheelBody {
        pos: [f64; 3],
        vel: [f64; 3],
        spin: f64,
        mass: f64,
        spin_inertia: f64,
        radius: f64,
    }

    impl WheelBody {
        fn step(&mut self, terrain: &dyn Terrain, tire: &dyn TireModel, dt: f64) -> TireOutput {
            let wc = WheelContact {
                hub_pos: self.pos,
                hub_vel: self.vel,
                heading: [1.0, 0.0, 0.0],
                spin: self.spin,
                radius: self.radius,
                normal_stiffness: 2.0e5,
                normal_damping: 4.0e3,
            };
            let out = wheel_ground_wrench(&wc, terrain, tire);
            for k in 0..3 {
                self.vel[k] += out.force[k] / self.mass * dt;
            }
            self.vel[1] -= 9.81 * dt;
            // Longitudinal ground force fx reacts on the spin as a
            // torque -fx·R: a braking force (fx < 0) spins the wheel up.
            let fx = out.force[0];
            self.spin += (-fx * self.radius / self.spin_inertia) * dt;
            for k in 0..3 {
                self.pos[k] += self.vel[k] * dt;
            }
            out
        }
    }

    #[test]
    fn skidding_wheel_spins_up_to_steady_rolling() {
        let terrain = FlatTerrain::new(0.0, 0.9);
        let tire = TmSimpleTire::default();
        let radius = 0.3;
        let mut wheel = WheelBody {
            // Resting at its equilibrium penetration (mass·g / k).
            pos: [0.0, radius - 80.0 * 9.81 / 2.0e5, 0.0],
            vel: [3.0, 0.0, 0.0], // moving forward, not spinning → skid
            spin: 0.0,
            mass: 80.0,
            spin_inertia: 80.0 * radius * radius * 0.5,
            radius,
        };
        let mut last = wheel.step(&terrain, &tire, 1.0e-3);
        for _ in 0..4000 {
            last = wheel.step(&terrain, &tire, 1.0e-3);
        }
        assert!(last.in_contact, "wheel left the ground");
        assert!(
            wheel.pos.iter().all(|c| c.is_finite()),
            "wheel state went non-finite"
        );
        // The longitudinal tire force has driven the slip toward zero —
        // the wheel now rolls (spin·R ≈ forward speed).
        assert!(
            last.slip_ratio.abs() < 0.1,
            "wheel never reached steady rolling: slip_ratio = {}",
            last.slip_ratio
        );
        assert!(wheel.spin > 0.0, "wheel did not spin up");
    }
}
