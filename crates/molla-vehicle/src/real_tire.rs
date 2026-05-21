//! Real tire contact primitives.
//!
//! This module is the first implementation step from `PLAN_REALTIRE.md`.
//! It models the tire/ground interface as a pressure-dependent contact patch
//! plus a longitudinal brush/bristle law. It is still compact enough for the
//! viewer, but unlike the old viewer patch multiplier, forces come from:
//!
//! - tire pressure and carcass compliance determining patch length;
//! - per-contact-cell normal load;
//! - tread shear stiffness;
//! - material static/dynamic friction caps.

use crate::terrain::Terrain;

/// Ground/tread contact material.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceMaterial {
    /// Static Coulomb coefficient. Trial bristle forces below this remain
    /// stuck to the ground.
    pub mu_static: f64,
    /// Dynamic/sliding Coulomb coefficient once bristles saturate.
    pub mu_dynamic: f64,
    /// Extra damping on bristle slip speed (N per (m/s) per m² contact area).
    pub bristle_damping: f64,
    /// Rolling resistance coefficient. This is intentionally separate from
    /// Coulomb friction: soft sand can have moderate grip but high rolling
    /// losses; ice can have low grip but low rolling loss.
    pub rolling_resistance: f64,
    /// Optional cohesive shear stress (Pa) for soil-like surfaces.
    pub cohesion: f64,
}

impl SurfaceMaterial {
    pub const fn new(mu_static: f64, mu_dynamic: f64) -> Self {
        Self {
            mu_static,
            mu_dynamic,
            bristle_damping: 0.0,
            rolling_resistance: 0.015,
            cohesion: 0.0,
        }
    }

    pub fn from_terrain_friction(mu: f64) -> Self {
        Self {
            mu_static: mu,
            mu_dynamic: 0.82 * mu,
            bristle_damping: 3.0e4,
            rolling_resistance: 0.015,
            cohesion: 0.0,
        }
    }

    pub const fn dry_asphalt() -> Self {
        Self {
            mu_static: 1.05,
            mu_dynamic: 0.88,
            bristle_damping: 5.0e4,
            rolling_resistance: 0.012,
            cohesion: 0.0,
        }
    }

    pub const fn loose_dirt() -> Self {
        Self {
            mu_static: 0.55,
            mu_dynamic: 0.42,
            bristle_damping: 7.0e4,
            rolling_resistance: 0.055,
            cohesion: 800.0,
        }
    }

    pub const fn ice() -> Self {
        Self {
            mu_static: 0.08,
            mu_dynamic: 0.045,
            bristle_damping: 5.0e3,
            rolling_resistance: 0.006,
            cohesion: 0.0,
        }
    }
}

/// Parameters for one deformable tire contact patch.
#[derive(Clone, Copy, Debug)]
pub struct RealTireParams {
    pub radius: f64,
    pub width: f64,
    pub pressure_pa: f64,
    /// Number of circumferential samples used for the contact patch.
    pub segments: usize,
    /// Carcass normal stiffness per contact area (Pa/m). This is the part
    /// that still supports load at zero pressure.
    pub carcass_normal_stiffness: f64,
    /// Tread longitudinal shear stiffness per contact area (N/m³).
    pub tread_shear_stiffness: f64,
    /// Slip speed used to regularize bristle displacement near standstill.
    pub relaxation_speed: f64,
}

impl Default for RealTireParams {
    fn default() -> Self {
        Self {
            radius: 0.32,
            width: 0.185,
            pressure_pa: 180_000.0,
            segments: 48,
            carcass_normal_stiffness: 4.0e5,
            tread_shear_stiffness: 2.6e6,
            relaxation_speed: 0.35,
        }
    }
}

/// Kinematic state of one wheel.
#[derive(Clone, Copy, Debug)]
pub struct RealWheelState {
    pub hub_pos: [f64; 3],
    pub hub_vel: [f64; 3],
    pub heading: [f64; 3],
    /// Wheel angular velocity (rad/s). Positive means rolling forward along
    /// `heading`.
    pub omega: f64,
}

/// A single contact-patch sample.
#[derive(Clone, Copy, Debug)]
pub struct TireContactCell {
    pub local_x: f64,
    pub penetration: f64,
    pub area: f64,
    pub normal_load: f64,
    pub tangential_force: f64,
    pub sliding: bool,
}

/// Total contact output for one tire.
#[derive(Clone, Debug)]
pub struct RealTireContact {
    pub in_contact: bool,
    pub force: [f64; 3],
    pub normal_load: f64,
    pub longitudinal_force: f64,
    pub rolling_resistance_force: f64,
    pub slip_ratio: f64,
    pub slip_speed: f64,
    pub patch_length: f64,
    pub contact_count: usize,
    pub sliding_count: usize,
    pub cells: Vec<TireContactCell>,
}

impl RealTireContact {
    fn empty() -> Self {
        Self {
            in_contact: false,
            force: [0.0; 3],
            normal_load: 0.0,
            longitudinal_force: 0.0,
            rolling_resistance_force: 0.0,
            slip_ratio: 0.0,
            slip_speed: 0.0,
            patch_length: 0.0,
            contact_count: 0,
            sliding_count: 0,
            cells: Vec::new(),
        }
    }
}

/// Solve the hub height that gives `target_load` on flat or heightfield-like
/// terrain at the current `(x,z)` location.
pub fn solve_loaded_tire_hub_height(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    x: f64,
    z: f64,
    target_load: f64,
) -> f64 {
    let gh = terrain.height(x, z);
    let mut lo = gh + params.radius * 0.18;
    let mut hi = gh + params.radius * 1.08;
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        let state = RealWheelState {
            hub_pos: [x, mid, z],
            hub_vel: [0.0; 3],
            heading: [1.0, 0.0, 0.0],
            omega: 0.0,
        };
        let out = evaluate_brush_tire(
            params,
            &state,
            terrain,
            &SurfaceMaterial::from_terrain_friction(terrain.friction()),
        );
        if out.normal_load > target_load {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Evaluate the pressure-dependent contact patch and longitudinal brush force.
pub fn evaluate_brush_tire(
    params: &RealTireParams,
    state: &RealWheelState,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
) -> RealTireContact {
    let segments = params.segments.max(8);
    let heading = normalize(project_to_ground(
        state.heading,
        terrain.normal(state.hub_pos[0], state.hub_pos[2]),
    ));
    let v_long = dot(state.hub_vel, heading);
    let wheel_speed = state.omega * params.radius;
    let slip_speed = wheel_speed - v_long;
    let denom = v_long.abs().max(params.relaxation_speed.max(1.0e-4));
    let slip_ratio = slip_speed / denom;
    let dtheta = std::f64::consts::TAU / segments as f64;
    let arc_len = params.radius * dtheta;
    let cell_area = params.width * arc_len;
    let normal_per_area_per_m = params.carcass_normal_stiffness
        + 3.0 * params.pressure_pa.max(0.0) / params.radius.max(1.0e-6);

    let mut cells = Vec::new();
    let mut normal_load = 0.0;
    let mut fx = 0.0;
    let mut sliding_count = 0usize;
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;

    for i in 0..segments {
        // theta = 0 is the bottom of the wheel. local_x spans the rolling
        // direction through the patch.
        let theta = (i as f64) * dtheta;
        let local_x = params.radius * theta.sin();
        let local_y = -params.radius * theta.cos();
        let world_x = state.hub_pos[0] + heading[0] * local_x;
        let world_z = state.hub_pos[2] + heading[2] * local_x;
        let ground_y = terrain.height(world_x, world_z);
        let point_y = state.hub_pos[1] + local_y;
        let penetration = ground_y - point_y;
        if penetration <= 0.0 {
            continue;
        }

        let fn_i = normal_per_area_per_m * penetration * cell_area;
        let bristle_disp = slip_ratio * params.radius;
        let trial = params.tread_shear_stiffness * cell_area * bristle_disp
            + material.bristle_damping * cell_area * slip_speed;
        let static_cap = material.mu_static * fn_i + material.cohesion * cell_area;
        let dynamic_cap = material.mu_dynamic * fn_i + material.cohesion * cell_area;
        let (ft_i, sliding) = if trial.abs() <= static_cap {
            (trial, false)
        } else {
            (dynamic_cap * trial.signum(), true)
        };

        normal_load += fn_i;
        fx += ft_i;
        if sliding {
            sliding_count += 1;
        }
        min_x = min_x.min(local_x);
        max_x = max_x.max(local_x);
        cells.push(TireContactCell {
            local_x,
            penetration,
            area: cell_area,
            normal_load: fn_i,
            tangential_force: ft_i,
            sliding,
        });
    }

    if cells.is_empty() {
        return RealTireContact::empty();
    }

    let rolling = material.rolling_resistance * normal_load * v_long.signum();
    let longitudinal_force = fx - rolling;
    RealTireContact {
        in_contact: true,
        force: [
            longitudinal_force * heading[0],
            normal_load,
            longitudinal_force * heading[2],
        ],
        normal_load,
        longitudinal_force,
        rolling_resistance_force: rolling,
        slip_ratio,
        slip_speed,
        patch_length: (max_x - min_x).max(0.0),
        contact_count: cells.len(),
        sliding_count,
        cells,
    }
}

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

/// Parameters for a true volumetric tetrahedral FEM tire.
///
/// The mesh is an annular tire volume: circumferential segments × radial
/// layers × width layers. Every parametric hexahedral cell is split into
/// tetrahedra and solved with a finite-deformation compressible Neo-Hookean
/// constitutive law. The inner radius is bead/rim-coupled to a one-DOF hub;
/// the outer tread nodes receive floor contact. Inflation pressure is modeled
/// primarily as membrane pre-stress / tangent stiffening; optional direct
/// boundary traction exists but defaults to zero on this coarse exterior mesh.
#[derive(Clone, Copy, Debug)]
pub struct TetraTireFemParams {
    pub outer_radius: f64,
    pub inner_radius: f64,
    pub width: f64,
    pub segments: usize,
    pub radial_layers: usize,
    pub width_layers: usize,
    pub pressure_pa: f64,
    pub density: f64,
    /// Fallback/body Young's modulus for carcass rubber (Pa).
    pub young_modulus: f64,
    /// Softer sidewall material (Pa).
    pub sidewall_young_modulus: f64,
    /// Stiffer tread/belt material (Pa).
    pub tread_young_modulus: f64,
    /// Stiff bead/rim-adjacent material (Pa).
    pub bead_young_modulus: f64,
    pub poisson_ratio: f64,
    pub node_damping: f64,
    pub hub_mass: f64,
    pub downforce: f64,
    pub bead_stiffness: f64,
    pub bead_damping: f64,
    pub floor_stiffness: f64,
    pub floor_damping: f64,
    /// Direct traction scale for pressure on the coarse exterior surface.
    /// Defaults to zero because inflation is represented as membrane
    /// pre-stress/tangent stiffening; applying pressure directly to the
    /// exterior tread would push the bottom of the tire into the ground.
    pub pressure_load_scale: f64,
}

impl Default for TetraTireFemParams {
    fn default() -> Self {
        Self {
            outer_radius: 0.32,
            inner_radius: 0.12,
            width: 0.185,
            segments: 32,
            radial_layers: 3,
            width_layers: 3,
            pressure_pa: 180_000.0,
            density: 1_100.0,
            young_modulus: 4.0e6,
            sidewall_young_modulus: 2.0e6,
            tread_young_modulus: 8.0e6,
            bead_young_modulus: 2.5e7,
            poisson_ratio: 0.46,
            node_damping: 18.0,
            hub_mass: 22.0,
            downforce: 2_600.0,
            bead_stiffness: 5.5e4,
            bead_damping: 650.0,
            floor_stiffness: 2.8e5,
            floor_damping: 1_200.0,
            pressure_load_scale: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
struct TetraElement {
    ids: [usize; 4],
    grads: [[f64; 3]; 4],
    volume: f64,
    mu: f64,
    lambda: f64,
}

/// Runtime state for the volumetric tetrahedral tire FEM.
#[derive(Clone, Debug)]
pub struct TetraTireFem {
    params: TetraTireFemParams,
    hub: [f64; 3],
    hub_v_y: f64,
    rest: Vec<[f64; 3]>,
    pos: Vec<[f64; 3]>,
    vel: Vec<[f64; 3]>,
    mass: Vec<f64>,
    tets: Vec<TetraElement>,
    surface_edges: Vec<(usize, usize)>,
    tread_nodes: Vec<usize>,
    bead_offsets: Vec<(usize, [f64; 3])>,
}

/// Raw f32 mesh/state payload for GPU tire FEM kernels.
#[derive(Clone, Debug)]
pub struct TetraTireFemGpuSnapshot {
    pub rest_positions: Vec<[f32; 4]>,
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub masses: Vec<f32>,
    pub tet_indices: Vec<u32>,
    pub tet_grads: Vec<[f32; 4]>,
    pub tet_volumes: Vec<f32>,
    pub tet_mu: Vec<f32>,
    pub tet_lambda: Vec<f32>,
    pub bead_ids: Vec<u32>,
    pub bead_offsets: Vec<[f32; 4]>,
    pub tread_ids: Vec<u32>,
    pub pressure_area_per_tread_node: f32,
}

impl TetraTireFem {
    pub fn new(hub_x: f64, pressure_pa: f64, mut params: TetraTireFemParams) -> Self {
        params.pressure_pa = pressure_pa.max(0.0);
        params.segments = params.segments.max(8);
        params.radial_layers = params.radial_layers.max(2);
        params.width_layers = params.width_layers.max(2);
        params.inner_radius = params.inner_radius.clamp(0.02, params.outer_radius * 0.92);
        params.poisson_ratio = params.poisson_ratio.clamp(-0.9, 0.49);

        let n = params.segments;
        let nr = params.radial_layers;
        let nw = params.width_layers;
        let hub_y = params.outer_radius * 1.05;
        let hub = [hub_x, hub_y, 0.0];
        let node_count = n * nr * nw;
        let mut rest = Vec::with_capacity(node_count);
        let mut pos = Vec::with_capacity(node_count);
        let mut vel = Vec::with_capacity(node_count);
        let idx = |i: usize, r: usize, w: usize| -> usize { ((i % n) * nr + r) * nw + w };

        for i in 0..n {
            let theta = i as f64 / n as f64 * std::f64::consts::TAU;
            let dir = [theta.cos(), theta.sin()];
            for r in 0..nr {
                let q = if nr > 1 {
                    r as f64 / (nr - 1) as f64
                } else {
                    0.0
                };
                let radius = params.inner_radius + (params.outer_radius - params.inner_radius) * q;
                for w in 0..nw {
                    let wz = if nw > 1 {
                        w as f64 / (nw - 1) as f64
                    } else {
                        0.5
                    };
                    let p = [
                        hub_x + radius * dir[0],
                        hub_y + radius * dir[1],
                        -0.5 * params.width + params.width * wz,
                    ];
                    rest.push(p);
                    pos.push(p);
                    vel.push([0.0; 3]);
                }
            }
        }

        let mut tets = Vec::new();
        let mut mass = vec![0.0; node_count];
        for i in 0..n {
            let ni = (i + 1) % n;
            for r in 0..(nr - 1) {
                for w in 0..(nw - 1) {
                    let young = if r == 0 {
                        params.bead_young_modulus
                    } else if r + 1 >= nr - 1 {
                        params.tread_young_modulus
                    } else if w == 0 || w + 1 >= nw - 1 {
                        params.sidewall_young_modulus
                    } else {
                        params.young_modulus
                    };
                    let (mu, lambda) = lame_from_young_poisson(young, params.poisson_ratio);
                    let a = idx(i, r, w);
                    let b = idx(ni, r, w);
                    let c = idx(ni, r + 1, w);
                    let d = idx(i, r + 1, w);
                    let e = idx(i, r, w + 1);
                    let f = idx(ni, r, w + 1);
                    let g = idx(ni, r + 1, w + 1);
                    let h = idx(i, r + 1, w + 1);
                    let cell_tets = [
                        [a, b, d, e],
                        [b, c, d, g],
                        [b, e, f, g],
                        [d, e, g, h],
                        [b, d, e, g],
                    ];
                    for ids in cell_tets {
                        if let Some(tet) = make_tetra_element(ids, &rest, mu, lambda) {
                            let nodal_mass = params.density * tet.volume / 4.0;
                            for &id in &tet.ids {
                                mass[id] += nodal_mass;
                            }
                            tets.push(tet);
                        }
                    }
                }
            }
        }
        for m in &mut mass {
            *m = m.max(0.02);
        }

        let mut surface_edges = Vec::new();
        for i in 0..n {
            let ni = (i + 1) % n;
            // Outer tread circumferential and width lines.
            for w in 0..nw {
                surface_edges.push((idx(i, nr - 1, w), idx(ni, nr - 1, w)));
            }
            for w in 0..(nw - 1) {
                surface_edges.push((idx(i, nr - 1, w), idx(i, nr - 1, w + 1)));
            }
            // Sidewall radial lines on both side faces.
            for r in 0..(nr - 1) {
                surface_edges.push((idx(i, r, 0), idx(i, r + 1, 0)));
                surface_edges.push((idx(i, r, nw - 1), idx(i, r + 1, nw - 1)));
            }
            // Inner bead/rim circumferential hints.
            surface_edges.push((idx(i, 0, 0), idx(ni, 0, 0)));
            surface_edges.push((idx(i, 0, nw - 1), idx(ni, 0, nw - 1)));
        }

        let mut tread_nodes = Vec::with_capacity(n * nw);
        for i in 0..n {
            for w in 0..nw {
                tread_nodes.push(idx(i, nr - 1, w));
            }
        }
        let mut bead_offsets = Vec::with_capacity(n * nw);
        for i in 0..n {
            for w in 0..nw {
                let id = idx(i, 0, w);
                bead_offsets.push((id, sub(rest[id], hub)));
            }
        }

        Self {
            params,
            hub,
            hub_v_y: 0.0,
            rest,
            pos,
            vel,
            mass,
            tets,
            surface_edges,
            tread_nodes,
            bead_offsets,
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

    pub fn surface_edges(&self) -> &[(usize, usize)] {
        &self.surface_edges
    }

    pub fn tread_nodes(&self) -> &[usize] {
        &self.tread_nodes
    }

    pub fn tetra_count(&self) -> usize {
        self.tets.len()
    }

    pub fn segments(&self) -> usize {
        self.params.segments
    }

    pub fn width(&self) -> f64 {
        self.params.width
    }

    pub fn outer_radius(&self) -> f64 {
        self.params.outer_radius
    }

    pub fn gpu_snapshot(&self) -> TetraTireFemGpuSnapshot {
        let vec4 = |p: [f64; 3]| -> [f32; 4] { [p[0] as f32, p[1] as f32, p[2] as f32, 0.0] };
        let mut tet_indices = Vec::with_capacity(self.tets.len() * 4);
        let mut tet_grads = Vec::with_capacity(self.tets.len() * 4);
        let mut tet_volumes = Vec::with_capacity(self.tets.len());
        let mut tet_mu = Vec::with_capacity(self.tets.len());
        let mut tet_lambda = Vec::with_capacity(self.tets.len());
        for tet in &self.tets {
            for &id in &tet.ids {
                tet_indices.push(id as u32);
            }
            for &g in &tet.grads {
                tet_grads.push(vec4(g));
            }
            tet_volumes.push(tet.volume as f32);
            let p_mu = 2.0 * self.params.pressure_pa.max(0.0);
            tet_mu.push((tet.mu + p_mu) as f32);
            tet_lambda.push((tet.lambda + 2.0 * p_mu) as f32);
        }
        let arc = self.params.outer_radius * std::f64::consts::TAU / self.params.segments as f64;
        let pressure_area_per_tread_node = arc * self.params.width
            / self.params.width_layers as f64
            * self.params.pressure_load_scale;
        TetraTireFemGpuSnapshot {
            rest_positions: self.rest.iter().copied().map(vec4).collect(),
            positions: self.pos.iter().copied().map(vec4).collect(),
            velocities: self.vel.iter().copied().map(vec4).collect(),
            masses: self.mass.iter().map(|&m| m as f32).collect(),
            tet_indices,
            tet_grads,
            tet_volumes,
            tet_mu,
            tet_lambda,
            bead_ids: self.bead_offsets.iter().map(|&(id, _)| id as u32).collect(),
            bead_offsets: self
                .bead_offsets
                .iter()
                .map(|&(_, offset)| vec4(offset))
                .collect(),
            tread_ids: self.tread_nodes.iter().map(|&id| id as u32).collect(),
            pressure_area_per_tread_node: pressure_area_per_tread_node as f32,
        }
    }

    pub fn apply_gpu_node_step(
        &mut self,
        positions: &[[f32; 4]],
        velocities: &[[f32; 4]],
        dt: f64,
    ) -> Result<(), &'static str> {
        if positions.len() != self.pos.len() || velocities.len() != self.vel.len() {
            return Err("gpu tire FEM output length mismatch");
        }
        for (dst, src) in self.pos.iter_mut().zip(positions.iter()) {
            *dst = [src[0] as f64, src[1] as f64, src[2] as f64];
        }
        for (dst, src) in self.vel.iter_mut().zip(velocities.iter()) {
            *dst = [src[0] as f64, src[1] as f64, src[2] as f64];
        }

        // The first GPU kernel updates nodes. Keep the one-DOF hub integration
        // on CPU by recomputing the equal/opposite bead reaction from the
        // updated bead-node positions. This keeps the viewer fully using the
        // GPU node FEM while preserving the loaded hub motion until the GPU
        // reduction path lands.
        let mut hub_force_y = -self.params.downforce - 9.81 * self.params.hub_mass;
        for &(id, offset) in &self.bead_offsets {
            let target = add(self.hub, offset);
            let delta = sub(target, self.pos[id]);
            let rel_v_y = self.hub_v_y - self.vel[id][1];
            let f_y = self.params.bead_stiffness * delta[1] + self.params.bead_damping * rel_v_y;
            hub_force_y -= f_y;
        }
        hub_force_y -= 260.0 * self.hub_v_y;
        self.hub_v_y += hub_force_y / self.params.hub_mass * dt;
        self.hub_v_y = self.hub_v_y.clamp(-2.5, 2.5);
        self.hub[1] += self.hub_v_y * dt;
        let hub_floor = self.params.inner_radius * 0.75;
        if self.hub[1] < hub_floor {
            self.hub[1] = hub_floor;
            if self.hub_v_y < 0.0 {
                self.hub_v_y = 0.0;
            }
        }
        Ok(())
    }

    pub fn step(&mut self, dt: f64) {
        let max_dt = 1.25e-4;
        let substeps = (dt / max_dt).ceil().max(1.0) as usize;
        let h = dt / substeps as f64;
        for _ in 0..substeps {
            self.step_inner(h);
        }
    }

    pub fn contact_patch_width(&self) -> f64 {
        let xs: Vec<f64> = self
            .tread_nodes
            .iter()
            .map(|&id| self.pos[id])
            .filter(|p| p[1] <= 3.0e-2)
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
        let mut force = vec![[0.0; 3]; self.pos.len()];
        let mut hub_force_y = -self.params.downforce - 9.81 * self.params.hub_mass;

        for i in 0..self.pos.len() {
            force[i][1] -= 9.81 * self.mass[i];
            force[i] = sub(force[i], scale(self.vel[i], self.params.node_damping));
        }

        for tet in &self.tets {
            // Compressible Neo-Hookean tetra: P = μ(F - F^-T) + λ ln(J) F^-T.
            // This handles finite rotations/deformations much better than the
            // earlier small-strain linear model.
            let mut f_def = [[0.0; 3]; 3];
            for a in 0..4 {
                let id = tet.ids[a];
                for row in 0..3 {
                    for col in 0..3 {
                        f_def[row][col] += self.pos[id][row] * tet.grads[a][col];
                    }
                }
            }
            let Some(inv_f) = inv3(f_def) else {
                continue;
            };
            let f_inv_t = transpose3(inv_f);
            let j = det3(f_def).max(0.20);
            let ln_j = j.ln();
            let p_mu = 2.0 * self.params.pressure_pa.max(0.0);
            let mu = tet.mu + p_mu;
            let lambda = tet.lambda + 2.0 * p_mu;
            let mut p_stress = [[0.0; 3]; 3];
            for r in 0..3 {
                for c in 0..3 {
                    p_stress[r][c] =
                        mu * (f_def[r][c] - f_inv_t[r][c]) + lambda * ln_j * f_inv_t[r][c];
                }
            }

            for a in 0..4 {
                let f = scale(mat_vec(p_stress, tet.grads[a]), -tet.volume);
                force[tet.ids[a]] = add(force[tet.ids[a]], f);
            }
        }

        // Bead/rim coupling: the innermost volumetric layer is attached to
        // the wheel hub, and equal/opposite Y force loads the hub.
        for &(id, offset) in &self.bead_offsets {
            let target = add(self.hub, offset);
            let delta = sub(target, self.pos[id]);
            let rel_v_y = self.hub_v_y - self.vel[id][1];
            let mut f = scale(delta, self.params.bead_stiffness);
            f[1] += self.params.bead_damping * rel_v_y;
            force[id] = add(force[id], f);
            hub_force_y -= f[1];
        }

        // Optional pressure loading on the outer tread nodes, along the rest
        // radial normal. The default scale is zero because the coarse exterior
        // surface would otherwise push the lower tread into the floor. The
        // main pressure effect is the pre-stress/tangent stiffening above.
        let arc = self.params.outer_radius * std::f64::consts::TAU / self.params.segments as f64;
        let area_per_tread_node = arc * self.params.width / self.params.width_layers as f64
            * self.params.pressure_load_scale;
        for &id in &self.tread_nodes {
            let radial = normalize([
                self.pos[id][0] - self.hub[0],
                self.pos[id][1] - self.hub[1],
                0.0,
            ]);
            force[id] = add(
                force[id],
                scale(radial, self.params.pressure_pa * area_per_tread_node),
            );
        }

        for i in 0..self.pos.len() {
            if self.pos[i][1] < 0.0 {
                let pen = -self.pos[i][1];
                let normal = self.params.floor_stiffness * pen
                    - self.params.floor_damping * self.vel[i][1].min(0.0);
                force[i][1] += normal.max(0.0);
                force[i][0] -= 320.0 * self.vel[i][0];
                force[i][2] -= 320.0 * self.vel[i][2];
            }
        }

        for i in 0..self.pos.len() {
            for c in 0..3 {
                self.vel[i][c] += force[i][c] / self.mass[i] * dt;
                self.vel[i][c] = self.vel[i][c].clamp(-6.0, 6.0);
                self.pos[i][c] += self.vel[i][c] * dt;
            }
            if self.pos[i][1] < 0.0 {
                self.pos[i][1] = 0.0;
                if self.vel[i][1] < 0.0 {
                    self.vel[i][1] = 0.0;
                }
            }
        }

        hub_force_y -= 260.0 * self.hub_v_y;
        self.hub_v_y += hub_force_y / self.params.hub_mass * dt;
        self.hub_v_y = self.hub_v_y.clamp(-2.5, 2.5);
        self.hub[1] += self.hub_v_y * dt;
        let hub_floor = self.params.inner_radius * 0.75;
        if self.hub[1] < hub_floor {
            self.hub[1] = hub_floor;
            if self.hub_v_y < 0.0 {
                self.hub_v_y = 0.0;
            }
        }
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn mat_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inv3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let d = det3(m);
    if d.abs() < 1.0e-12 {
        return None;
    }
    let inv_d = 1.0 / d;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_d,
        ],
    ])
}

fn transpose3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn lame_from_young_poisson(young: f64, poisson: f64) -> (f64, f64) {
    let nu = poisson.clamp(-0.9, 0.49);
    let mu = young / (2.0 * (1.0 + nu));
    let lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    (mu, lambda)
}

fn make_tetra_element(
    mut ids: [usize; 4],
    rest: &[[f64; 3]],
    mu: f64,
    lambda: f64,
) -> Option<TetraElement> {
    let mut dm = tetra_dm(ids, rest);
    let mut det = det3(dm);
    if det.abs() < 1.0e-12 {
        return None;
    }
    if det < 0.0 {
        ids.swap(2, 3);
        dm = tetra_dm(ids, rest);
        det = det3(dm);
    }
    let inv = inv3(dm)?;
    let g1 = [inv[0][0], inv[0][1], inv[0][2]];
    let g2 = [inv[1][0], inv[1][1], inv[1][2]];
    let g3 = [inv[2][0], inv[2][1], inv[2][2]];
    let g0 = scale(add(add(g1, g2), g3), -1.0);
    Some(TetraElement {
        ids,
        grads: [g0, g1, g2, g3],
        volume: det.abs() / 6.0,
        mu,
        lambda,
    })
}

fn tetra_dm(ids: [usize; 4], rest: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let x0 = rest[ids[0]];
    let x1 = rest[ids[1]];
    let x2 = rest[ids[2]];
    let x3 = rest[ids[3]];
    [
        [x1[0] - x0[0], x2[0] - x0[0], x3[0] - x0[0]],
        [x1[1] - x0[1], x2[1] - x0[1], x3[1] - x0[1]],
        [x1[2] - x0[2], x2[2] - x0[2], x3[2] - x0[2]],
    ]
}

fn project_to_ground(v: [f64; 3], n: [f64; 3]) -> [f64; 3] {
    let d = dot(v, n);
    [v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2]]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let l = dot(v, v).sqrt();
    if l > 1.0e-12 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [1.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::FlatTerrain;

    fn loaded_output(pressure: f64, material: SurfaceMaterial, slip_speed: f64) -> RealTireContact {
        let terrain = FlatTerrain::new(0.0, material.mu_static);
        let params = RealTireParams {
            pressure_pa: pressure,
            ..Default::default()
        };
        let load = 4_500.0;
        let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load);
        evaluate_brush_tire(
            &params,
            &RealWheelState {
                hub_pos: [0.0, hub_y, 0.0],
                hub_vel: [1.0, 0.0, 0.0],
                heading: [1.0, 0.0, 0.0],
                omega: (1.0 + slip_speed) / params.radius,
            },
            &terrain,
            &material,
        )
    }

    #[test]
    fn lower_pressure_makes_a_longer_contact_patch() {
        let material = SurfaceMaterial::dry_asphalt();
        let high = loaded_output(360_000.0, material, 0.0);
        let low = loaded_output(90_000.0, material, 0.0);
        assert!(
            low.patch_length > high.patch_length,
            "low pressure should make a longer patch: low={} high={}",
            low.patch_length,
            high.patch_length
        );
        assert!(
            low.contact_count >= high.contact_count,
            "low pressure should touch at least as many cells"
        );
    }

    #[test]
    fn surface_material_dominates_ultimate_grip() {
        let asphalt = loaded_output(320_000.0, SurfaceMaterial::dry_asphalt(), 3.0);
        let ice = loaded_output(320_000.0, SurfaceMaterial::ice(), 3.0);
        assert!(
            asphalt.longitudinal_force > ice.longitudinal_force * 6.0,
            "asphalt should transmit far more force than ice: asphalt={} ice={}",
            asphalt.longitudinal_force,
            ice.longitudinal_force
        );
    }

    #[test]
    fn lower_pressure_improves_low_mu_tractive_stiffness() {
        let material = SurfaceMaterial::new(0.22, 0.17);
        let high = loaded_output(360_000.0, material, 0.02);
        let low = loaded_output(90_000.0, material, 0.02);
        assert!(
            low.longitudinal_force > high.longitudinal_force * 1.10,
            "low pressure should improve brush force before full sliding: low={} high={}",
            low.longitudinal_force,
            high.longitudinal_force
        );
    }

    #[test]
    fn dynamic_carcass_stays_above_floor_and_forms_patch() {
        let mut tire = DynamicTireCarcass::new(
            0.0,
            120_000.0,
            DynamicTireCarcassParams {
                segments: 32,
                downforce: 2_000.0,
                ..Default::default()
            },
        );
        for _ in 0..300 {
            tire.step(1.0 / 240.0);
        }
        assert!(
            tire.nodes()
                .iter()
                .all(|p| p[1] >= -1.0e-12 && p[1].is_finite()),
            "dynamic carcass node went below floor or non-finite"
        );
        assert!(
            tire.contact_patch_width() > 0.02,
            "loaded dynamic tire should form a visible contact patch"
        );
    }

    #[test]
    fn dynamic_carcass_pressure_reduces_patch_width() {
        let settle = |pressure| {
            let mut tire = DynamicTireCarcass::new(
                0.0,
                pressure,
                DynamicTireCarcassParams {
                    segments: 32,
                    downforce: 2_200.0,
                    ..Default::default()
                },
            );
            for _ in 0..300 {
                tire.step(1.0 / 240.0);
            }
            tire.contact_patch_width()
        };
        let low = settle(70_000.0);
        let high = settle(320_000.0);
        assert!(
            low > high,
            "lower pressure should have wider dynamic patch: low={low} high={high}"
        );
    }

    #[test]
    fn tetra_fem_builds_a_volumetric_tire_mesh() {
        let tire = TetraTireFem::new(
            0.0,
            160_000.0,
            TetraTireFemParams {
                segments: 16,
                radial_layers: 3,
                width_layers: 3,
                ..Default::default()
            },
        );
        assert_eq!(tire.nodes().len(), 16 * 3 * 3);
        assert!(
            tire.tetra_count() > 16 * 4,
            "tetra mesh should be volumetric"
        );
        assert!(
            tire.surface_edges().len() > tire.segments(),
            "viewer should have surface edges to draw"
        );
        let snap = tire.gpu_snapshot();
        assert_eq!(snap.positions.len(), tire.nodes().len());
        assert_eq!(snap.tet_indices.len(), tire.tetra_count() * 4);
        assert_eq!(snap.tet_grads.len(), tire.tetra_count() * 4);
        assert_eq!(snap.tet_volumes.len(), tire.tetra_count());
        assert_eq!(snap.tet_mu.len(), tire.tetra_count());
        assert_eq!(snap.tet_lambda.len(), tire.tetra_count());
        assert!(snap.pressure_area_per_tread_node >= 0.0);
    }

    #[test]
    fn tetra_fem_stays_above_floor_and_forms_patch() {
        let mut tire = TetraTireFem::new(
            0.0,
            130_000.0,
            TetraTireFemParams {
                segments: 16,
                radial_layers: 3,
                width_layers: 3,
                downforce: 3_600.0,
                ..Default::default()
            },
        );
        for _ in 0..360 {
            tire.step(1.0 / 240.0);
        }
        assert!(
            tire.nodes()
                .iter()
                .all(|p| p[1] >= -1.0e-12 && p.iter().all(|x| x.is_finite())),
            "tetra FEM tire node went below floor or non-finite"
        );
        let patch = tire.contact_patch_width();
        assert!(
            patch > 0.005,
            "loaded tetra FEM tire should form a visible contact patch, got {patch}"
        );
    }

    #[test]
    fn tetra_fem_pressure_reduces_patch_width() {
        let settle = |pressure| {
            let mut tire = TetraTireFem::new(
                0.0,
                pressure,
                TetraTireFemParams {
                    segments: 16,
                    radial_layers: 3,
                    width_layers: 3,
                    downforce: 2_400.0,
                    ..Default::default()
                },
            );
            for _ in 0..180 {
                tire.step(1.0 / 240.0);
            }
            tire.contact_patch_width()
        };
        let low = settle(70_000.0);
        let high = settle(320_000.0);
        assert!(
            low > high,
            "lower pressure should have wider tetra FEM patch: low={low} high={high}"
        );
    }
}
