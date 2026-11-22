use super::{AirConfig, AirForces, Grid, invalid};
use molla_core::Result;
use molla_math::Vec3;

#[derive(Clone, Debug)]
pub struct AirState {
    pub(super) config: AirConfig,
    pub(super) velocity: Vec<Vec3>,
    pub(super) pressure: Vec<f64>,
    /// Uniform velocity increment the next step adds to the whole air mass.
    pub(super) drive: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct Diagnostics {
    pub divergence_rms: f64,
    pub divergence_max: f64,
    /// Maximum absolute staggered face-velocity component, in metres per second.
    pub speed_max: f64,
    pub ground_flux_max: f64,
}

pub(super) fn sample(config: AirConfig, velocity: &[Vec3], point: Vec3) -> Vec3 {
    let g = config.grid;
    let p = (point - g.origin) / g.spacing;
    let dims = Vec3::new(g.cells[0] as f64, g.cells[1] as f64, g.cells[2] as f64);
    let outside = (p - dims)
        .max(Vec3::new(-p.x, 0.0, -p.z))
        .max_element()
        .clamp(0.0, 1.0);
    if outside >= 1.0 {
        return config.background(point);
    }
    let interior = Vec3::from_array(std::array::from_fn(|axis| {
        config
            .grid
            .stencil(point, axis)
            .iter()
            .map(|&(i, w)| velocity[i][axis] * w)
            .sum()
    }));
    interior.lerp(config.background(point), outside)
}

fn sample_exterior(config: AirConfig, velocity: &[Vec3], point: Vec3) -> Vec3 {
    let edge = point.clamp(config.grid.origin, config.grid.upper());
    if config.exterior_decay_length == 0.0 || point == edge {
        return sample(config, velocity, point);
    }
    let distance = (point - edge).length() / config.exterior_decay_length;
    config.background(point)
        + (sample(config, velocity, edge) - config.background(edge)) * (-distance * distance).exp()
}

impl AirState {
    pub fn new(config: AirConfig) -> Result<Self> {
        config.validate(1e-6)?;
        let mut state = Self {
            config,
            velocity: vec![Vec3::ZERO; config.grid.len],
            pressure: vec![0.0; config.grid.len],
            drive: Vec3::ZERO,
        };
        state.seed_velocity(|p| config.background(p))?;
        Ok(state)
    }

    /// Background wind and gust amplitude from the next step on; the next step also moves
    /// the whole air mass by the change in background wind.
    pub fn set_background(&mut self, ambient: Vec3, gust_strength: f64) -> Result<()> {
        if !ambient.is_finite() || !gust_strength.is_finite() || gust_strength < 0.0 {
            return Err(invalid("background wind must be finite with non-negative gusts"));
        }
        self.drive += ambient - self.config.ambient;
        self.config.ambient = ambient;
        self.config.gust_strength = gust_strength;
        Ok(())
    }
    pub fn config(&self) -> AirConfig {
        self.config
    }
    pub fn face_velocities(&self) -> &[Vec3] {
        &self.velocity
    }
    pub fn pressures(&self) -> &[f64] {
        &self.pressure
    }
    pub fn velocity_at(&self, point: Vec3) -> Result<Vec3> {
        if !point.is_finite() {
            return Err(invalid("nonfinite sample position"));
        }
        Ok(sample_exterior(self.config, &self.velocity, point) + self.config.turbulence(point))
    }

    pub fn seed_velocity(&mut self, field: impl Fn(Vec3) -> Vec3) -> Result<()> {
        let g = self.config.grid;
        let mut next = vec![Vec3::ZERO; g.len];
        for (i, v) in next.iter_mut().enumerate() {
            let p = g.coordinates(i);
            for axis in 0..3 {
                if g.face_valid(p, axis) && !(axis == 1 && p[1] == 0) {
                    v[axis] = field(g.face_position(p, axis))[axis];
                }
            }
            if !v.is_finite() {
                return Err(invalid("nonfinite velocity field"));
            }
        }
        self.velocity = next;
        self.pressure.fill(0.0);
        Ok(())
    }

    pub fn step(&mut self, dt: f64, forces: &AirForces) -> Result<()> {
        let c = self.config;
        c.validate(dt)?;
        forces.validate(c.grid)?;
        let g = c.grid;
        let mut predicted = vec![Vec3::ZERO; g.len];
        for (i, value) in predicted.iter_mut().enumerate() {
            let p = g.coordinates(i);
            for axis in 0..3 {
                if !g.face_valid(p, axis) || (axis == 1 && p[1] == 0) {
                    continue;
                }
                let x = g.face_position(p, axis);
                let mid = x - sample(c, &self.velocity, x) * (0.5 * dt);
                let back = x - sample(c, &self.velocity, mid) * dt;
                let advected = sample(c, &self.velocity, back)[axis];
                let mut laplacian = 0.0;
                for a in 0..3 {
                    for delta in [-1isize, 1] {
                        let mut q = p;
                        let bound = g.cells[a] - usize::from(a != axis);
                        q[a] = (p[a] as isize + delta).clamp(0, bound as isize) as usize;
                        laplacian += self.velocity[g.index(q)][axis] - self.velocity[i][axis];
                    }
                }
                value[axis] = advected
                    + self.drive[axis]
                    + c.viscosity * dt / g.spacing.powi(2) * laplacian
                    + dt * forces.values[i][axis] / (c.density * g.spacing.powi(3));
            }
        }
        let (velocity, pressure) = project(c, dt, &predicted);
        if velocity.iter().any(|v| !v.is_finite()) || pressure.iter().any(|v| !v.is_finite()) {
            return Err(invalid("air step overflow"));
        }
        self.velocity = velocity;
        self.pressure = pressure;
        self.config.time += dt;
        self.drive = Vec3::ZERO;
        Ok(())
    }

    pub fn project(&mut self, dt: f64) -> Result<()> {
        self.config.validate(dt)?;
        let (v, p) = project(self.config, dt, &self.velocity);
        if v.iter().any(|v| !v.is_finite()) || p.iter().any(|v| !v.is_finite()) {
            return Err(invalid("projection overflow"));
        }
        self.velocity = v;
        self.pressure = p;
        Ok(())
    }

    pub fn diagnostics(&self) -> Diagnostics {
        let g = self.config.grid;
        let mut d = Diagnostics {
            divergence_rms: 0.0,
            divergence_max: 0.0,
            speed_max: 0.0,
            ground_flux_max: 0.0,
        };
        for (i, v) in self.velocity.iter().enumerate() {
            let p = g.coordinates(i);
            d.speed_max = d.speed_max.max(v.abs().max_element());
            if p[1] == 0 {
                d.ground_flux_max = d.ground_flux_max.max(v.y.abs());
            }
            if (0..3).any(|a| p[a] >= g.cells[a]) {
                continue;
            }
            let div = divergence(g, &self.velocity, p);
            d.divergence_rms += div * div;
            d.divergence_max = d.divergence_max.max(div.abs());
        }
        d.divergence_rms = (d.divergence_rms / g.cells.iter().product::<usize>() as f64).sqrt();
        d
    }
}

fn divergence(g: Grid, velocity: &[Vec3], p: [usize; 3]) -> f64 {
    let mut div = 0.0;
    for a in 0..3 {
        let mut q = p;
        q[a] += 1;
        div += velocity[g.index(q)][a] - velocity[g.index(p)][a];
    }
    div / g.spacing
}

fn project(c: AirConfig, dt: f64, predicted: &[Vec3]) -> (Vec<Vec3>, Vec<f64>) {
    let g = c.grid;
    let mut pressure = vec![0.0; g.len];
    let mut rhs = vec![0.0; g.len];
    for (i, r) in rhs.iter_mut().enumerate() {
        let p = g.coordinates(i);
        if (0..3).all(|a| p[a] < g.cells[a]) {
            *r = -c.density * g.spacing.powi(2) / dt * divergence(g, predicted, p);
        }
    }
    for _ in 0..c.pressure_sweeps {
        for color in 0..2 {
            for z in 0..g.cells[2] {
                for y in 0..g.cells[1] {
                    for x in 0..g.cells[0] {
                        if (x + y + z) % 2 != color {
                            continue;
                        }
                        let p = [x, y, z];
                        let i = g.index(p);
                        let mut sum = rhs[i];
                        let mut diagonal = 0.0;
                        for a in 0..3 {
                            for delta in [-1isize, 1] {
                                let neighbor = p[a] as isize + delta;
                                if neighbor >= 0 && neighbor < g.cells[a] as isize {
                                    let mut q = p;
                                    q[a] = neighbor as usize;
                                    sum += pressure[g.index(q)];
                                    diagonal += 1.0;
                                } else if !(a == 1 && neighbor < 0) {
                                    diagonal += 2.0;
                                }
                            }
                        }
                        pressure[i] += c.relaxation * (sum / diagonal - pressure[i]);
                    }
                }
            }
        }
    }
    let mut velocity = vec![Vec3::ZERO; g.len];
    for (i, v) in velocity.iter_mut().enumerate() {
        let p = g.coordinates(i);
        for a in 0..3 {
            if !g.face_valid(p, a) || (a == 1 && p[1] == 0) {
                continue;
            }
            let mut left = p;
            left[a] = p[a].saturating_sub(1);
            let mut right = p;
            right[a] = p[a].min(g.cells[a] - 1);
            let l = pressure[g.index(left)] * if p[a] == 0 { -1.0 } else { 1.0 };
            let r = pressure[g.index(right)] * if p[a] == g.cells[a] { -1.0 } else { 1.0 };
            v[a] = predicted[i][a] - dt / (c.density * g.spacing) * (r - l);
        }
    }
    (velocity, pressure)
}
