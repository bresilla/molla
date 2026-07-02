//! Dynamic tire-carcass shell-FEM model.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::internal::dot;

/// Parameters for a dynamic tire carcass model.
///
/// This is a reduced shell-FEM carcass: two sidewall/tread rings with
/// circumferential membrane edges, widthwise belt edges, diagonal shear plies,
/// bending regularizers, radial sidewall coupling to a one-DOF hub, pressure
/// loading, and floor contact. It is intentionally small and deterministic for
/// the viewer, but it is a real dynamic system, not a pre-baked shape.
#[derive(Clone, Copy, Debug)]
pub struct DynamicTireCarcassParams {
    pub radius: f64,
    pub width: f64,
    pub segments: usize,
    pub pressure_pa: f64,
    pub node_mass: f64,
    pub hub_mass: f64,
    pub downforce: f64,
    pub sidewall_stiffness: f64,
    pub tread_stiffness: f64,
    pub belt_stiffness: f64,
    pub shear_stiffness: f64,
    pub bending_stiffness: f64,
    pub damping: f64,
    pub hub_damping: f64,
    pub floor_stiffness: f64,
    pub floor_damping: f64,
}

impl Default for DynamicTireCarcassParams {
    fn default() -> Self {
        Self {
            radius: 0.32,
            width: 0.185,
            segments: 48,
            pressure_pa: 180_000.0,
            node_mass: 0.28,
            hub_mass: 18.0,
            downforce: 2_600.0,
            sidewall_stiffness: 1.8e4,
            tread_stiffness: 3.5e4,
            belt_stiffness: 2.5e4,
            shear_stiffness: 1.6e4,
            bending_stiffness: 3.5e3,
            damping: 35.0,
            hub_damping: 180.0,
            floor_stiffness: 2.5e5,
            floor_damping: 900.0,
        }
    }
}

/// Runtime state for the dynamic carcass.
#[derive(Clone, Debug)]
pub struct DynamicTireCarcass {
    params: DynamicTireCarcassParams,
    hub: [f64; 3],
    hub_v_y: f64,
    pos: Vec<[f64; 3]>,
    vel: Vec<[f64; 3]>,
}

impl DynamicTireCarcass {
    pub fn new(hub_x: f64, pressure_pa: f64, mut params: DynamicTireCarcassParams) -> Self {
        params.pressure_pa = pressure_pa;
        params.segments = params.segments.max(12);
        let n = params.segments;
        let mut pos = Vec::with_capacity(2 * n);
        let mut vel = Vec::with_capacity(2 * n);
        let hub_y = params.radius * 1.05;
        for side in 0..2 {
            let z = if side == 0 {
                -params.width * 0.5
            } else {
                params.width * 0.5
            };
            for i in 0..n {
                let theta = i as f64 / n as f64 * std::f64::consts::TAU;
                let radial = [theta.cos(), theta.sin()];
                pos.push([
                    hub_x + params.radius * radial[0],
                    hub_y + params.radius * radial[1],
                    z,
                ]);
                vel.push([0.0; 3]);
            }
        }
        Self {
            params,
            hub: [hub_x, hub_y, 0.0],
            hub_v_y: 0.0,
            pos,
            vel,
        }
    }

    pub fn set_pressure(&mut self, pressure_pa: f64) {
        self.params.pressure_pa = pressure_pa.max(0.0);
    }

    pub fn pressure(&self) -> f64 {
        self.params.pressure_pa
    }

    pub fn hub(&self) -> [f64; 3] {
        self.hub
    }

    pub fn nodes(&self) -> &[[f64; 3]] {
        &self.pos
    }

    pub fn segments(&self) -> usize {
        self.params.segments
    }

    pub fn width(&self) -> f64 {
        self.params.width
    }

    pub fn radius(&self) -> f64 {
        self.params.radius
    }

    pub fn step(&mut self, dt: f64) {
        let max_dt = 7.5e-5;
        let substeps = (dt / max_dt).ceil().max(1.0) as usize;
        let h = dt / substeps as f64;
        for _ in 0..substeps {
            self.step_inner(h);
        }
    }

    pub fn contact_patch_width(&self) -> f64 {
        let xs: Vec<f64> = self
            .pos
            .iter()
            .filter(|p| p[1] <= 1.0e-2)
            .map(|p| p[0])
            .collect();
        if xs.len() < 2 {
            return 0.0;
        }
        let mn = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        mx - mn
    }

    fn step_inner(&mut self, dt: f64) {
        let n = self.params.segments;
        let mut force = vec![[0.0; 3]; self.pos.len()];
        let mut hub_force_y = -self.params.downforce - 9.81 * self.params.hub_mass;

        for i in 0..self.pos.len() {
            force[i][1] -= 9.81 * self.params.node_mass;
            force[i][0] -= self.params.damping * self.vel[i][0];
            force[i][1] -= self.params.damping * self.vel[i][1];
            force[i][2] -= self.params.damping * self.vel[i][2];

            // Radial sidewall spring to the hub/side plane.
            let rx = self.pos[i][0] - self.hub[0];
            let ry = self.pos[i][1] - self.hub[1];
            let r = (rx * rx + ry * ry).sqrt().max(1.0e-9);
            let nx = rx / r;
            let ny = ry / r;
            let extension = r - self.params.radius;
            let pressure_stiffness = 0.09 * self.params.pressure_pa.max(0.0);
            let f = -(self.params.sidewall_stiffness + pressure_stiffness) * extension;
            force[i][0] += f * nx;
            force[i][1] += f * ny;
            hub_force_y -= f * ny;

            // Pneumatic pressure approximation: outward load over each
            // circumferential patch. It is balanced by sidewall/belt springs
            // and contact, so pressure changes the dynamic equilibrium.
            let arc = self.params.radius * std::f64::consts::TAU / n as f64;
            let pressure_force = self.params.pressure_pa * self.params.width * arc / n as f64;
            force[i][0] += pressure_force * nx;
            force[i][1] += pressure_force * ny;

            // Floor contact.
            if self.pos[i][1] < 0.0 {
                let pen = -self.pos[i][1];
                let normal = self.params.floor_stiffness * pen
                    - self.params.floor_damping * self.vel[i][1].min(0.0);
                force[i][1] += normal.max(0.0);
                force[i][0] -= 220.0 * self.vel[i][0];
                force[i][2] -= 220.0 * self.vel[i][2];
            }
        }

        // Circumferential membrane springs on each side ring.
        let rest_tread = 2.0 * self.params.radius * (std::f64::consts::PI / n as f64).sin();
        for side in 0..2 {
            let base = side * n;
            for i in 0..n {
                self.apply_spring(
                    base + i,
                    base + (i + 1) % n,
                    rest_tread,
                    self.params.tread_stiffness,
                    &mut force,
                );
            }
        }

        // Belt/width springs connecting the two sidewalls.
        let rest_shear = (self.params.width * self.params.width + rest_tread * rest_tread).sqrt();
        let rest_bend = 2.0 * self.params.radius * (2.0 * std::f64::consts::PI / n as f64).sin();
        for i in 0..n {
            self.apply_spring(
                i,
                n + i,
                self.params.width,
                self.params.belt_stiffness,
                &mut force,
            );
            // Bias-ply/shear diagonals keep the carcass from behaving like
            // two independent hula hoops connected only by width bars.
            self.apply_spring(
                i,
                n + (i + 1) % n,
                rest_shear,
                self.params.shear_stiffness,
                &mut force,
            );
            self.apply_spring(
                i,
                n + (i + n - 1) % n,
                rest_shear,
                self.params.shear_stiffness,
                &mut force,
            );
            // A light second-neighbour spring approximates shell bending
            // resistance so low-pressure tires wrinkle/flatten instead of
            // numerically crumpling into alternating nodes.
            self.apply_spring(
                i,
                (i + 2) % n,
                rest_bend,
                self.params.bending_stiffness,
                &mut force,
            );
            self.apply_spring(
                n + i,
                n + (i + 2) % n,
                rest_bend,
                self.params.bending_stiffness,
                &mut force,
            );
        }

        for i in 0..self.pos.len() {
            for k in 0..3 {
                self.vel[i][k] += force[i][k] / self.params.node_mass * dt;
                self.vel[i][k] = self.vel[i][k].clamp(-8.0, 8.0);
                self.pos[i][k] += self.vel[i][k] * dt;
            }
            if self.pos[i][1] < 0.0 {
                self.pos[i][1] = 0.0;
                if self.vel[i][1] < 0.0 {
                    self.vel[i][1] = 0.0;
                }
            }
        }

        hub_force_y -= self.params.hub_damping * self.hub_v_y;
        self.hub_v_y += hub_force_y / self.params.hub_mass * dt;
        self.hub_v_y = self.hub_v_y.clamp(-3.0, 3.0);
        self.hub[1] += self.hub_v_y * dt;
        let hub_floor = self.params.radius * 0.22;
        if self.hub[1] < hub_floor {
            self.hub[1] = hub_floor;
            if self.hub_v_y < 0.0 {
                self.hub_v_y = 0.0;
            }
        }
    }

    fn apply_spring(&self, a: usize, b: usize, rest: f64, k: f64, force: &mut [[f64; 3]]) {
        let dx = [
            self.pos[b][0] - self.pos[a][0],
            self.pos[b][1] - self.pos[a][1],
            self.pos[b][2] - self.pos[a][2],
        ];
        let len = dot(dx, dx).sqrt().max(1.0e-9);
        let f = k * (len - rest);
        let n = [dx[0] / len, dx[1] / len, dx[2] / len];
        for c in 0..3 {
            force[a][c] += f * n[c];
            force[b][c] -= f * n[c];
        }
    }
}
