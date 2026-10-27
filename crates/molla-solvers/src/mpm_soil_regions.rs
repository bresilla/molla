//! Bounded machine-region requests with MPM transfer and motion padding.

use super::*;
use molla_vehicle::SoilTileLayout;

/// A closed swept contact AABB over a caller-controlled activation horizon.
/// Include the entire machine/tool sweep, including rotation and contact margins.
/// This expands requests only: it does not establish equilibrium, freeze boundary
/// soil, account for nonlocal rigid coupling, or authorize particle deactivation.
#[derive(Clone, Copy, Debug)]
pub struct MpmSoilRegionEnvelope {
    pub id: SoilRegionId,
    pub minimum: [f64; 3],
    pub maximum: [f64; 3],
    /// Componentwise upper bound for any soil particle's displacement in each
    /// substep, including acceleration; not merely its current velocity times dt.
    pub particle_travel_per_substep: [f64; 3],
    /// Maximum number of solver substeps before refreshing the request.
    pub substeps: u32,
}

impl MpmSoilRegionEnvelope {
    /// Enumerates sorted archive tiles, retaining empty-space claims. Padding is
    /// `(3 + Jacobi sweeps) * cell_size + particle_travel` per substep. Three
    /// cells cover two particles sharing a quadratic-transfer node; each Jacobi
    /// sweep adds one face-neighbor cell. An outward f32 roundoff allowance covers
    /// grid-coordinate arithmetic. Unsupported precision ranges reject. This covers
    /// local stencil paths,
    /// not arbitrary position clamps or other nonlocal operations. Callers must
    /// refresh before the horizon expires or any supplied motion bound is exceeded.
    /// Capacity/coordinate errors reject before tile allocation or enumeration.
    pub fn tiles(
        &self,
        layout: SoilTileLayout,
        config: &MpmConfig,
        max_tiles: usize,
    ) -> Result<Vec<SoilTileCoord>> {
        self.tiles_in_frame(layout, config, max_tiles, None)
    }

    pub(super) fn tiles_in_frame(
        &self,
        layout: SoilTileLayout,
        config: &MpmConfig,
        max_tiles: usize,
        origin: Option<[f64; 3]>,
    ) -> Result<Vec<SoilTileCoord>> {
        let (minimum, maximum) = self.bounds(config)?;
        let at = |point| {
            match origin {
                Some(origin) => layout.tile_at_local(point, origin),
                None => layout.tile_at(point),
            }
            .map_err(|e| invalid(&e.to_string()))
        };
        let lo = at(minimum)?.0;
        let hi = at(maximum)?.0;
        let mut count = 1_usize;
        for axis in 0..3 {
            let length = usize::try_from(hi[axis] - lo[axis] + 1)
                .map_err(|_| invalid("soil activation range overflow"))?;
            count = count
                .checked_mul(length)
                .filter(|&n| n <= max_tiles)
                .ok_or_else(|| invalid("soil activation tile capacity exceeded"))?;
        }
        let mut tiles = Vec::new();
        tiles
            .try_reserve_exact(count)
            .map_err(|_| invalid("soil activation allocation failed"))?;
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    tiles.push(SoilTileCoord([x, y, z]));
                }
            }
        }
        Ok(tiles)
    }

    fn bounds(&self, config: &MpmConfig) -> Result<([f64; 3], [f64; 3])> {
        let inverse = 1.0_f32 / config.cell_size;
        let steps = f64::from(self.substeps);
        let error_factor = 32.0 * f64::from(f32::EPSILON) * steps;
        if self.substeps == 0
            || !config.cell_size.is_normal()
            || config.cell_size <= 0.0
            || !inverse.is_normal()
            || inverse <= 0.0
            || config.grid_origin.iter().any(|v| !v.is_finite())
            || error_factor >= 0.5
        {
            return Err(invalid(
                "unsupported soil activation horizon or f32 grid precision",
            ));
        }
        let sweeps = match config.time_step {
            MpmTimeStep::Explicit => 0,
            MpmTimeStep::Implicit { jacobi_iters } => jacobi_iters,
        };
        let h = f64::from(config.cell_size);
        let stencil = ((3.0 + f64::from(sweeps)) * h).next_up();
        let mut minimum = [0.0; 3];
        let mut maximum = [0.0; 3];
        for axis in 0..3 {
            let travel = self.particle_travel_per_substep[axis];
            if !self.minimum[axis].is_finite()
                || !self.maximum[axis].is_finite()
                || self.minimum[axis] > self.maximum[axis]
                || !travel.is_finite()
                || travel < 0.0
            {
                return Err(invalid("invalid soil swept bounds or particle travel"));
            }
            let scale = (self.minimum[axis].abs().max(self.maximum[axis].abs())
                + f64::from(config.grid_origin[axis]).abs())
            .next_up();
            let geometric = ((stencil + travel).next_up() * steps).next_up();
            let roundoff = (error_factor * (scale + h).next_up()).next_up();
            let flush = 32.0 * steps * f64::from(f32::MIN_POSITIVE);
            let numerator = ((geometric + roundoff).next_up() + flush).next_up();
            let padding = (numerator / (1.0 - error_factor).next_down()).next_up();
            let expanded_scale = (scale + padding).next_up();
            if !padding.is_finite()
                || expanded_scale > f64::from(f32::MAX) / 16.0
                || expanded_scale * f64::from(inverse) >= (1_u64 << 28) as f64
            {
                return Err(invalid(
                    "soil activation exceeds supported f32 coordinate range",
                ));
            }
            minimum[axis] = (self.minimum[axis] - padding).next_down();
            maximum[axis] = (self.maximum[axis] + padding).next_up();
        }
        Ok((minimum, maximum))
    }
}

pub(super) fn expand(
    regions: &[MpmSoilRegionEnvelope],
    layout: SoilTileLayout,
    config: &MpmConfig,
    limits: SoilResidencyLimits,
) -> Result<Vec<(SoilRegionId, Vec<SoilTileCoord>)>> {
    expand_in_frame(regions, layout, config, limits, None)
}

pub(super) fn expand_in_frame(
    regions: &[MpmSoilRegionEnvelope],
    layout: SoilTileLayout,
    config: &MpmConfig,
    limits: SoilResidencyLimits,
    origin: Option<[f64; 3]>,
) -> Result<Vec<(SoilRegionId, Vec<SoilTileCoord>)>> {
    if regions.len() > limits.regions {
        return Err(invalid("soil activation region capacity exceeded"));
    }
    let mut seen = BTreeSet::new();
    let mut remaining = limits.claims;
    let mut owned = Vec::new();
    for region in regions {
        if !seen.insert(region.id) {
            return Err(invalid("duplicate soil activation region"));
        }
        let tiles = region.tiles_in_frame(layout, config, limits.tiles.min(remaining), origin)?;
        remaining -= tiles.len();
        owned.push((region.id, tiles));
    }
    Ok(owned)
}

fn support_box(
    layout: SoilTileLayout,
    minimum: SoilTileCoord,
    maximum: SoilTileCoord,
    origin: Option<[f64; 3]>,
) -> Result<[f32; 8]> {
    let mut result = [0.; 8];
    for axis in 0..3 {
        let end = maximum.0[axis]
            .checked_add(1)
            .ok_or_else(|| invalid("soil support tile endpoint overflow"))?;
        let lo = ((((minimum.0[axis] as f64).next_down() * layout.edge()).next_down()
            + layout.origin()[axis])
            .next_down()
            - origin.unwrap_or([0.; 3])[axis])
            .next_down();
        let hi = ((((end as f64).next_up() * layout.edge()).next_up() + layout.origin()[axis])
            .next_up()
            - origin.unwrap_or([0.; 3])[axis])
            .next_up();
        let mut lower = lo as f32;
        let mut upper = hi as f32;
        if f64::from(lower) > lo {
            lower = lower.next_down();
        }
        if f64::from(upper) < hi {
            upper = upper.next_up();
        }
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(invalid(
                "soil support tile box exceeds finite local f32 range",
            ));
        }
        result[axis] = lower;
        result[4 + axis] = upper;
    }
    Ok(result)
}

pub(super) fn support_boxes(
    envelopes: &[MpmSoilRegionEnvelope],
    layout: SoilTileLayout,
    config: &MpmConfig,
    limits: SoilResidencyLimits,
    origin: Option<[f64; 3]>,
    capacity: u32,
) -> Result<Vec<[f32; 8]>> {
    if capacity == 0 || envelopes.len() > capacity as usize {
        return Err(invalid(
            "soil support region seeds are unprepared or exceed capacity",
        ));
    }
    expand_in_frame(envelopes, layout, config, limits, origin)?
        .into_iter()
        .map(|(_, tiles)| {
            let first = *tiles
                .first()
                .ok_or_else(|| invalid("empty expanded soil support region"))?;
            let mut minimum = first;
            let mut maximum = first;
            for tile in tiles {
                for axis in 0..3 {
                    minimum.0[axis] = minimum.0[axis].min(tile.0[axis]);
                    maximum.0[axis] = maximum.0[axis].max(tile.0[axis]);
                }
            }
            support_box(layout, minimum, maximum, origin)
        })
        .collect()
}

#[derive(Default)]
pub(super) struct Horizon {
    pub(super) remaining: Option<u32>,
    pub(super) travel: Option<[f64; 3]>,
    requested: Option<u32>,
    requested_travel: Option<[f64; 3]>,
    requested_guard: Option<[f64; 3]>,
}

impl Horizon {
    pub(super) fn request(&mut self, substeps: Option<u32>) {
        self.requested = substeps;
        self.requested_travel = None;
        self.requested_guard = None;
    }

    pub(super) fn request_swept(&mut self, envelopes: &[MpmSoilRegionEnvelope]) {
        self.requested = Some(envelopes.iter().map(|e| e.substeps).min().unwrap_or(0));
        self.requested_guard = (!envelopes.is_empty()).then(|| {
            std::array::from_fn(|axis| {
                envelopes
                    .iter()
                    .map(|e| e.particle_travel_per_substep[axis])
                    .fold(f64::INFINITY, f64::min)
            })
        });
        self.requested_travel = (!envelopes.is_empty()).then(|| {
            std::array::from_fn(|axis| {
                envelopes
                    .iter()
                    .map(|e| e.particle_travel_per_substep[axis])
                    .fold(0.0_f64, f64::max)
            })
        });
    }

    pub(super) fn has_swept_pins(&self) -> bool {
        self.requested_travel.is_some()
    }

    /// Pins per-occupied-tile particle AABBs over the shortest requested horizon.
    /// Componentwise maximum travel is doubled to cover relative particle motion.
    /// Raw requests and empty swept requests retain occupancy pins only.
    pub(super) fn particle_pins(
        &self,
        positions: &[u32],
        layout: SoilTileLayout,
        config: &MpmConfig,
        max_tiles: usize,
    ) -> Result<Vec<SoilTileCoord>> {
        self.particle_pins_in_frame(positions, layout, config, max_tiles, None)
    }

    pub(super) fn particle_pins_in_frame(
        &self,
        positions: &[u32],
        layout: SoilTileLayout,
        config: &MpmConfig,
        max_tiles: usize,
        origin: Option<[f64; 3]>,
    ) -> Result<Vec<SoilTileCoord>> {
        let mut occupied = std::collections::BTreeMap::new();
        for position in positions.chunks_exact(4) {
            let point = std::array::from_fn(|axis| f64::from(f32::from_bits(position[axis])));
            let tile = match origin {
                Some(origin) => layout.tile_at_local(point, origin),
                None => layout.tile_at(point),
            }
            .map_err(|e| invalid(&e.to_string()))?;
            let bounds = occupied.entry(tile).or_insert((point, point));
            for axis in 0..3 {
                bounds.0[axis] = bounds.0[axis].min(point[axis]);
                bounds.1[axis] = bounds.1[axis].max(point[axis]);
            }
            if occupied.len() > max_tiles {
                return Err(invalid("active soil pin capacity exceeded"));
            }
        }
        let Some(travel) = self.requested_travel else {
            return Ok(occupied.into_keys().collect());
        };
        let mut pins = BTreeSet::new();
        for (_, (minimum, maximum)) in occupied {
            let envelope = MpmSoilRegionEnvelope {
                id: SoilRegionId(0),
                minimum,
                maximum,
                particle_travel_per_substep: travel.map(|v| 2.0 * v),
                substeps: self.requested.expect("swept horizon"),
            };
            for tile in envelope.tiles_in_frame(layout, config, max_tiles, origin)? {
                pins.insert(tile);
                if pins.len() > max_tiles {
                    return Err(invalid("active soil swept pin capacity exceeded"));
                }
            }
        }
        Ok(pins.into_iter().collect())
    }

    pub(super) fn commit(&mut self) {
        self.remaining = self.requested.take();
        self.travel = self.requested_guard.take();
        self.requested_travel = None;
    }

    pub(super) fn after_steps(&self, substeps: usize) -> Result<Option<u32>> {
        self.remaining
            .map(|remaining| {
                u32::try_from(substeps)
                    .ok()
                    .and_then(|n| remaining.checked_sub(n))
                    .ok_or_else(|| {
                        invalid(
                            "soil activation horizon exhausted; refresh regions before stepping",
                        )
                    })
            })
            .transpose()
    }
}

#[cfg(test)]
#[path = "mpm_soil_regions_tests.rs"]
mod tests;
