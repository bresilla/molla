//! Three-dimensional incompressible air on a staggered grid, with rotor disk forcing.

mod cpu;
mod drag;
mod droplet;
#[cfg(feature = "gpu")]
mod gpu;
mod rotorcraft;
mod trajectory;
mod turbulence;
pub use cpu::{AirState, Diagnostics};
pub use drag::CylinderDrag;
pub use droplet::{TurbulentDispersion, WaterDroplet};
#[cfg(feature = "gpu")]
pub use gpu::{AirCoupling, GpuAir};
pub use rotorcraft::{Quadrotor, QuadrotorConfig, RotorAirLoad};
pub use trajectory::{FlightReference, FlightSegment, MotionLimits};

use molla_core::{Error, Result};
use molla_math::Vec3;

fn invalid(message: &str) -> Error {
    Error::Build(format!("airflow: {message}"))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    cells: [usize; 3],
    spacing: f64,
    origin: Vec3,
    len: usize,
}

impl Grid {
    pub fn new(cells: [usize; 3], spacing: f64, origin: Vec3) -> Result<Self> {
        let len = cells.iter().try_fold(1usize, |n, d| {
            d.checked_add(1).and_then(|d| n.checked_mul(d))
        });
        if cells.iter().any(|d| *d < 4)
            || len.is_none_or(|n| n > u32::MAX as usize)
            || !spacing.is_finite()
            || spacing <= 0.0
            || !origin.is_finite()
        {
            return Err(invalid("invalid grid dimensions or geometry"));
        }
        let grid = Self {
            cells,
            spacing,
            origin,
            len: len.unwrap(),
        };
        if !grid.upper().is_finite() {
            return Err(invalid("grid extent overflow"));
        }
        Ok(grid)
    }

    pub fn cells(self) -> [usize; 3] {
        self.cells
    }
    pub fn spacing(self) -> f64 {
        self.spacing
    }
    pub fn origin(self) -> Vec3 {
        self.origin
    }
    pub fn upper(self) -> Vec3 {
        self.origin
            + Vec3::new(
                self.cells[0] as f64,
                self.cells[1] as f64,
                self.cells[2] as f64,
            ) * self.spacing
    }
    pub fn storage_len(self) -> usize {
        self.len
    }
    pub(super) fn index(self, p: [usize; 3]) -> usize {
        (p[2] * (self.cells[1] + 1) + p[1]) * (self.cells[0] + 1) + p[0]
    }
    pub(super) fn coordinates(self, n: usize) -> [usize; 3] {
        let x = self.cells[0] + 1;
        let y = self.cells[1] + 1;
        [n % x, n / x % y, n / (x * y)]
    }
    pub(super) fn face_valid(self, p: [usize; 3], axis: usize) -> bool {
        (0..3).all(|a| p[a] < self.cells[a] + usize::from(a == axis))
    }
    pub(super) fn offset(axis: usize) -> Vec3 {
        let mut offset = Vec3::splat(0.5);
        offset[axis] = 0.0;
        offset
    }
    pub fn face_position(self, p: [usize; 3], axis: usize) -> Vec3 {
        self.origin
            + (Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64) + Self::offset(axis)) * self.spacing
    }
    pub(super) fn weights(self, point: Vec3, axis: usize) -> ([usize; 3], Vec3) {
        let limit = Vec3::new(
            self.cells[0] as f64,
            self.cells[1] as f64,
            self.cells[2] as f64,
        ) - Vec3::ONE
            + Vec3::from_array(std::array::from_fn(|a| if a == axis { 1.0 } else { 0.0 }));
        let s =
            ((point - self.origin) / self.spacing - Self::offset(axis)).clamp(Vec3::ZERO, limit);
        let base = s.floor().min(limit - Vec3::ONE);
        (
            [base.x as usize, base.y as usize, base.z as usize],
            s - base,
        )
    }
    pub(super) fn stencil(self, point: Vec3, axis: usize) -> [(usize, f64); 8] {
        let (base, f) = self.weights(point, axis);
        std::array::from_fn(|corner| {
            let mut p = base;
            let mut weight = 1.0;
            for a in 0..3 {
                let high = (corner >> a) & 1;
                p[a] += high;
                weight *= if high == 1 { f[a] } else { 1.0 - f[a] };
            }
            (self.index(p), weight)
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AirConfig {
    pub grid: Grid,
    pub density: f64,
    pub viscosity: f64,
    pub ambient: Vec3,
    pub gust_strength: f64,
    /// Exterior wake decay length for particle queries; zero uses the one-cell blend.
    pub exterior_decay_length: f64,
    /// Amplitude of the boundary-aware, multiscale curl field in particle queries.
    pub turbulence_strength: f64,
    pub time: f64,
    pub pressure_sweeps: usize,
    pub relaxation: f64,
}

impl AirConfig {
    pub fn new(grid: Grid) -> Self {
        Self {
            grid,
            density: 1.225,
            viscosity: 1.5e-5,
            ambient: Vec3::ZERO,
            gust_strength: 0.0,
            exterior_decay_length: 0.0,
            turbulence_strength: 0.0,
            time: 0.0,
            pressure_sweeps: 120,
            relaxation: 1.7,
        }
    }
    /// Spatially coherent, divergence-free background gusts, defined beyond the grid.
    pub fn background(self, point: Vec3) -> Vec3 {
        if self.gust_strength == 0.0 {
            return self.ambient;
        }
        let phase = 0.9 * point.x - 0.55 * self.time;
        self.ambient
            + Vec3::new(
                0.55 * (0.7 * point.z - 0.4 * self.time).sin()
                    + 0.54 * (1.2 * point.y).cos() * phase.sin(),
                -0.405 * (1.2 * point.y).sin() * phase.cos(),
                0.45 * (0.8 * point.x + 0.31 * self.time).sin(),
            ) * self.gust_strength
    }
    pub(super) fn validate(self, dt: f64) -> Result<()> {
        if !self.density.is_finite()
            || self.density <= 0.0
            || !self.viscosity.is_finite()
            || self.viscosity < 0.0
            || !self.ambient.is_finite()
            || !self.gust_strength.is_finite()
            || self.gust_strength < 0.0
            || !self.exterior_decay_length.is_finite()
            || self.exterior_decay_length < 0.0
            || !self.turbulence_strength.is_finite()
            || self.turbulence_strength < 0.0
            || !self.time.is_finite()
            || self.time < 0.0
            || !dt.is_finite()
            || dt <= 0.0
            || self.pressure_sweeps == 0
            || !self.relaxation.is_finite()
            || self.relaxation <= 0.0
            || self.relaxation >= 2.0
            || self.viscosity * dt / self.grid.spacing.powi(2) > 1.0 / 6.0
        {
            return Err(invalid(
                "invalid air settings or unstable explicit diffusion step",
            ));
        }
        Ok(())
    }
}

/// Rotor loads on air: axial force is opposite the aircraft's thrust.
#[derive(Clone, Copy, Debug)]
pub struct RotorDisk {
    pub center: Vec3,
    pub axis: Vec3,
    pub radius: f64,
    pub thrust: f64,
    pub air_torque: f64,
}

#[derive(Clone, Debug)]
pub struct AirForces {
    grid: Grid,
    values: Vec<Vec3>,
}

impl AirForces {
    pub fn new(grid: Grid) -> Self {
        Self {
            grid,
            values: vec![Vec3::ZERO; grid.len],
        }
    }
    pub fn clear(&mut self) {
        self.values.fill(Vec3::ZERO);
    }

    /// Deposit a pure couple using symmetric force pairs on three axes.
    pub fn add_torque(&mut self, point: Vec3, torque: Vec3) -> Result<()> {
        let h = self.grid.spacing;
        if !point.is_finite()
            || !torque.is_finite()
            || !point.cmpge(self.grid.origin + Vec3::splat(2.0 * h)).all()
            || !point.cmple(self.grid.upper() - Vec3::splat(2.0 * h)).all()
        {
            return Err(invalid("torque footprint outside grid"));
        }
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let force = torque.cross(axis) / (4.0 * h);
            self.add(point + axis * h, force)?;
            self.add(point - axis * h, -force)?;
        }
        Ok(())
    }

    /// Deposit force in newtons; the complete interpolation footprint must be interior.
    pub fn add(&mut self, point: Vec3, force: Vec3) -> Result<()> {
        if !force.is_finite()
            || !point.is_finite()
            || !point
                .cmpge(self.grid.origin + Vec3::splat(self.grid.spacing))
                .all()
            || !point
                .cmple(self.grid.upper() - Vec3::splat(self.grid.spacing))
                .all()
        {
            return Err(invalid(
                "force must be finite and at least one cell inside the domain",
            ));
        }
        for axis in 0..3 {
            for (index, weight) in self.grid.stencil(point, axis) {
                self.values[index][axis] += weight * force[axis];
            }
        }
        Ok(())
    }

    pub fn add_rotor(&mut self, rotor: RotorDisk) -> Result<()> {
        self.add_disk_load(
            rotor,
            -rotor.axis * rotor.thrust,
            rotor.axis * rotor.air_torque,
        )
    }

    /// Deposit a disk's air force and torque about its center, in world coordinates.
    pub fn add_disk_load(&mut self, rotor: RotorDisk, force: Vec3, torque: Vec3) -> Result<()> {
        if !rotor.center.is_finite()
            || !rotor.axis.is_finite()
            || (rotor.axis.length_squared() - 1.0).abs() > 1e-6
            || !rotor.radius.is_finite()
            || rotor.radius <= 0.0
            || !rotor.thrust.is_finite()
            || !rotor.air_torque.is_finite()
            || !force.is_finite()
            || !torque.is_finite()
        {
            return Err(invalid("invalid rotor disk"));
        }
        let reference = if rotor.axis.x.abs() < 0.8 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let e = rotor.axis.cross(reference).normalize();
        let f = rotor.axis.cross(e);
        let mut samples = Vec::with_capacity(64);
        for ring in 0..4 {
            let r = rotor.radius * ((ring as f64 + 0.5) / 4.0).sqrt();
            for spoke in 0..16 {
                let a = std::f64::consts::TAU * (spoke as f64 + 0.5 * (ring % 2) as f64) / 16.0;
                samples.push((e * a.cos() + f * a.sin()) * r);
            }
        }
        let moment: f64 = samples.iter().map(|r| r.length_squared()).sum();
        let axial = force / samples.len() as f64;
        let rotation = (torque * 2.0 - rotor.axis * torque.dot(rotor.axis)) / moment;
        let loads: Vec<_> = samples
            .iter()
            .map(|r| (rotor.center + *r, axial + rotation.cross(*r)))
            .collect();
        for (p, force) in &loads {
            if !force.is_finite()
                || !p
                    .cmpge(self.grid.origin + Vec3::splat(self.grid.spacing))
                    .all()
                || !p
                    .cmple(self.grid.upper() - Vec3::splat(self.grid.spacing))
                    .all()
            {
                return Err(invalid("rotor forcing footprint leaves the domain"));
            }
        }
        for (point, force) in loads {
            self.add(point, force)?;
        }
        Ok(())
    }

    pub fn wrench(&self, about: Vec3) -> (Vec3, Vec3) {
        let mut force = Vec3::ZERO;
        let mut torque = Vec3::ZERO;
        for (i, value) in self.values.iter().enumerate() {
            let p = self.grid.coordinates(i);
            for axis in 0..3 {
                let mut f = Vec3::ZERO;
                f[axis] = value[axis];
                force += f;
                torque += (self.grid.face_position(p, axis) - about).cross(f);
            }
        }
        (force, torque)
    }

    pub(super) fn validate(&self, grid: Grid) -> Result<()> {
        if self.grid != grid || self.values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("force field does not match grid or is nonfinite"));
        }
        Ok(())
    }
}
