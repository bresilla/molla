//! V0 — terrain query abstraction.
//!
//! A tire model is a force element: each step it queries the ground
//! under the wheel rather than running collision detection. [`Terrain`]
//! is that query — height, surface normal, and friction at a world
//! `(x, z)`. A flat plane and a regular-grid heightfield are provided;
//! an MPM/SPH-surface terrain arrives with V5.

/// A drivable ground surface.
pub trait Terrain {
    /// Ground height (world `y`) at `(x, z)`.
    fn height(&self, x: f64, z: f64) -> f64;
    /// Unit surface normal at `(x, z)`.
    fn normal(&self, x: f64, z: f64) -> [f64; 3];
    /// Coulomb friction coefficient of the surface.
    fn friction(&self) -> f64;
}

/// An infinite flat plane at a fixed height.
#[derive(Clone, Copy, Debug)]
pub struct FlatTerrain {
    /// World `y` of the plane.
    pub height: f64,
    /// Surface friction coefficient.
    pub friction: f64,
}

impl FlatTerrain {
    /// A flat plane at `height` with friction `friction`.
    pub fn new(height: f64, friction: f64) -> Self {
        Self { height, friction }
    }
}

impl Terrain for FlatTerrain {
    fn height(&self, _x: f64, _z: f64) -> f64 {
        self.height
    }
    fn normal(&self, _x: f64, _z: f64) -> [f64; 3] {
        [0.0, 1.0, 0.0]
    }
    fn friction(&self) -> f64 {
        self.friction
    }
}

/// A regular-grid heightfield. Heights are sampled on an `nx × nz`
/// grid; `height` bilinearly interpolates and `normal` is taken from
/// the finite-difference gradient of the interpolated surface.
#[derive(Clone, Debug)]
pub struct HeightfieldTerrain {
    /// World `(x, z)` of sample `(0, 0)`.
    pub origin: [f64; 2],
    /// Grid spacing.
    pub spacing: f64,
    /// Samples per row (`x`).
    pub nx: usize,
    /// Number of rows (`z`).
    pub nz: usize,
    /// Row-major heights, `heights[iz * nx + ix]`.
    pub heights: Vec<f64>,
    /// Surface friction coefficient.
    pub friction: f64,
}

impl HeightfieldTerrain {
    /// Build a heightfield. `heights.len()` must equal `nx * nz`.
    pub fn new(
        origin: [f64; 2],
        spacing: f64,
        nx: usize,
        nz: usize,
        heights: Vec<f64>,
        friction: f64,
    ) -> Self {
        assert_eq!(heights.len(), nx * nz, "heights length must be nx * nz");
        Self {
            origin,
            spacing,
            nx,
            nz,
            heights,
            friction,
        }
    }

    /// Clamped grid sample.
    fn sample(&self, ix: i64, iz: i64) -> f64 {
        let ix = ix.clamp(0, self.nx as i64 - 1) as usize;
        let iz = iz.clamp(0, self.nz as i64 - 1) as usize;
        self.heights[iz * self.nx + ix]
    }
}

impl Terrain for HeightfieldTerrain {
    fn height(&self, x: f64, z: f64) -> f64 {
        let fx = (x - self.origin[0]) / self.spacing;
        let fz = (z - self.origin[1]) / self.spacing;
        let ix = fx.floor() as i64;
        let iz = fz.floor() as i64;
        let tx = fx - ix as f64;
        let tz = fz - iz as f64;
        let h00 = self.sample(ix, iz);
        let h10 = self.sample(ix + 1, iz);
        let h01 = self.sample(ix, iz + 1);
        let h11 = self.sample(ix + 1, iz + 1);
        let h0 = h00 * (1.0 - tx) + h10 * tx;
        let h1 = h01 * (1.0 - tx) + h11 * tx;
        h0 * (1.0 - tz) + h1 * tz
    }

    fn normal(&self, x: f64, z: f64) -> [f64; 3] {
        let d = self.spacing * 0.5;
        let dhdx = (self.height(x + d, z) - self.height(x - d, z)) / (2.0 * d);
        let dhdz = (self.height(x, z + d) - self.height(x, z - d)) / (2.0 * d);
        let n = [-dhdx, 1.0, -dhdz];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        [n[0] / len, n[1] / len, n[2] / len]
    }

    fn friction(&self) -> f64 {
        self.friction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_terrain_is_level() {
        let t = FlatTerrain::new(0.5, 0.9);
        assert_eq!(t.height(3.0, -2.0), 0.5);
        assert_eq!(t.normal(3.0, -2.0), [0.0, 1.0, 0.0]);
        assert_eq!(t.friction(), 0.9);
    }

    #[test]
    fn heightfield_interpolates_and_clamps() {
        // A 3×3 grid that ramps in x: h = 2·x (spacing 1).
        let heights = vec![0.0, 2.0, 4.0, 0.0, 2.0, 4.0, 0.0, 2.0, 4.0];
        let t = HeightfieldTerrain::new([0.0, 0.0], 1.0, 3, 3, heights, 0.8);
        // Exact on a sample, bilinear between samples.
        assert!((t.height(1.0, 1.0) - 2.0).abs() < 1e-9);
        assert!((t.height(0.5, 0.5) - 1.0).abs() < 1e-9);
        // Out-of-range clamps to the edge sample.
        assert!((t.height(-5.0, 0.0) - 0.0).abs() < 1e-9);
        assert!((t.height(99.0, 0.0) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn heightfield_normal_tilts_with_the_slope() {
        // A ramp h = 0.5·x → ∂h/∂x = 0.5 → normal ∝ (-0.5, 1, 0).
        let mut heights = Vec::new();
        for _iz in 0..5 {
            for ix in 0..5 {
                heights.push(0.5 * ix as f64);
            }
        }
        let t = HeightfieldTerrain::new([0.0, 0.0], 1.0, 5, 5, heights, 0.8);
        let n = t.normal(2.0, 2.0);
        let expect = {
            let v = [-0.5_f64, 1.0, 0.0];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        for k in 0..3 {
            assert!(
                (n[k] - expect[k]).abs() < 1e-9,
                "normal[{k}] {} vs {}",
                n[k],
                expect[k]
            );
        }
    }
}
