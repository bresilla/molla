use molla_math::Real;
use molla_sim::Heightfield;

use crate::aabb::Aabb;

const TILE: usize = 8;

#[derive(Clone)]
pub(super) struct HeightfieldTiles {
    cols: usize,
    maxima: Vec<Real>,
}

impl HeightfieldTiles {
    pub(super) fn new(field: &Heightfield) -> Option<Self> {
        field.validate().ok()?;
        let cols = (field.nx as usize - 1).div_ceil(TILE);
        let rows = (field.nz as usize - 1).div_ceil(TILE);
        let mut maxima = vec![Real::NEG_INFINITY; cols * rows];
        for z in 0..rows {
            for x in 0..cols {
                let end_x = ((x + 1) * TILE).min(field.nx as usize - 1);
                let end_z = ((z + 1) * TILE).min(field.nz as usize - 1);
                for j in z * TILE..=end_z {
                    for i in x * TILE..=end_x {
                        maxima[z * cols + x] = maxima[z * cols + x]
                            .max(field.origin.y + field.sample(i as u32, j as u32));
                    }
                }
            }
        }
        Some(Self { cols, maxima })
    }

    pub(super) fn may_contact(&self, field: &Heightfield, bounds: Aabb) -> bool {
        if !bounds.min.is_finite() || !bounds.max.is_finite() {
            return true;
        }
        let tile = |value: Real, origin: Real, cell: Real, vertices: u32| {
            (((value - origin) / cell)
                .floor()
                .clamp(0.0, (vertices - 2) as Real) as usize)
                / TILE
        };
        let x0 = tile(bounds.min.x, field.origin.x, field.cell_size, field.nx);
        let x1 = tile(bounds.max.x, field.origin.x, field.cell_size, field.nx);
        let z0 = tile(bounds.min.z, field.origin.z, field.cell_size_z, field.nz);
        let z1 = tile(bounds.max.z, field.origin.z, field.cell_size_z, field.nz);
        for z in z0..=z1 {
            for x in x0..=x1 {
                let upper = self.maxima[z * self.cols + x];
                let tolerance = 1e-9 * (1.0 + upper.abs() + bounds.min.y.abs());
                if bounds.min.y <= upper + tolerance {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::Vec3;

    #[test]
    fn tile_edges_partial_tiles_and_triangle_samples_are_conservative() {
        let mut field =
            Heightfield::from_scaled(11, 19, vec![0.0; 11 * 19], Vec3::new(9.0, 1.0, 15.0))
                .unwrap();
        field.origin += Vec3::new(7.0, -2.0, -5.0);
        field.heights[8 * 19 + 8] = 4.0;
        field.heights[10 * 19 + 18] = 9.0;
        let tiles = HeightfieldTiles::new(&field).unwrap();
        for z in 0..10 {
            for x in 0..18 {
                for t in [0.0, 0.2, 0.8, 1.0] {
                    let px = field.origin.x + (x as Real + t) * field.cell_size;
                    let pz = field.origin.z + (z as Real + 1.0 - t) * field.cell_size_z;
                    let height = field.height_and_normal(px, pz).unwrap().0;
                    let point = Vec3::new(px, height, pz);
                    assert!(tiles.may_contact(
                        &field,
                        Aabb {
                            min: point,
                            max: point
                        }
                    ));
                }
            }
        }
        let point = field.origin + Vec3::new(8.1, 1.0, 0.1);
        assert!(!tiles.may_contact(
            &field,
            Aabb {
                min: point,
                max: point
            }
        ));
        assert!(tiles.may_contact(&field, Aabb::infinite()));
        field.heights[0] = Real::NAN;
        assert!(HeightfieldTiles::new(&field).is_none());
    }
}
