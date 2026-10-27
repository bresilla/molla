//! World-tile coordinates and bounded shared soil residency metadata.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[path = "soil_state_coordinates.rs"]
mod coordinates;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SoilTileCoord(pub [i64; 3]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SoilParticleId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SoilMaterialId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SoilRegionId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoilStateError {
    InvalidLayout,
    UnrepresentablePosition,
    InvalidLimits,
    CapacityExceeded,
    DuplicateRegion,
    StalePlan,
}

impl std::fmt::Display for SoilStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "soil state: {self:?}")
    }
}
impl std::error::Error for SoilStateError {}

/// Cubic physical archive tiles; independent of solver cells and rendering LOD.
/// Membership is floor((world - origin) / edge) in f64 arithmetic, including
/// negative coordinates. Coordinates at magnitude >= 2^52 are rejected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoilTileLayout {
    origin: [f64; 3],
    edge: f64,
}

impl SoilTileLayout {
    pub fn new(origin: [f64; 3], edge: f64) -> Result<Self, SoilStateError> {
        if origin.iter().any(|v| !v.is_finite()) || !edge.is_finite() || edge <= 0.0 {
            return Err(SoilStateError::InvalidLayout);
        }
        Ok(Self { origin, edge })
    }

    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }
    pub fn edge(&self) -> f64 {
        self.edge
    }

    pub fn tile_at(&self, world: [f64; 3]) -> Result<SoilTileCoord, SoilStateError> {
        let mut tile = [0; 3];
        for axis in 0..3 {
            let coordinate = ((world[axis] - self.origin[axis]) / self.edge).floor();
            if !coordinate.is_finite() || coordinate.abs() >= (1_u64 << 52) as f64 {
                return Err(SoilStateError::UnrepresentablePosition);
            }
            tile[axis] = coordinate as i64;
        }
        Ok(SoilTileCoord(tile))
    }

    /// Exact floor membership of represented `local + world_origin - origin`
    /// divided by the represented edge, without forming an absolute position.
    /// Nonfinite inputs/intermediates and tile magnitudes >= 2^52 reject.
    /// This differs from `tile_at`'s rounded f64 subtraction/division contract.
    pub fn tile_at_local(
        &self,
        local: [f64; 3],
        world_origin: [f64; 3],
    ) -> Result<SoilTileCoord, SoilStateError> {
        let mut tile = [0; 3];
        for axis in 0..3 {
            tile[axis] = coordinates::coordinate(
                local[axis],
                world_origin[axis],
                self.origin[axis],
                self.edge,
            )?;
        }
        Ok(SoilTileCoord(tile))
    }
}

/// Maximum live metadata, not total visited field area or archive storage.
#[derive(Clone, Copy, Debug)]
pub struct SoilResidencyLimits {
    pub regions: usize,
    pub tiles: usize,
    pub claims: usize,
}

pub struct SoilRegionRequest<'a> {
    pub id: SoilRegionId,
    pub tiles: &'a [SoilTileCoord],
}

/// One physical tile shared by all requesting regions. The lowest region ID is
/// its deterministic coordinator, not an exclusive physical contact owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoilTileClaims {
    regions: BTreeSet<SoilRegionId>,
    pinned: bool,
}

impl SoilTileClaims {
    pub fn regions(&self) -> impl ExactSizeIterator<Item = SoilRegionId> + '_ {
        self.regions.iter().copied()
    }
    pub fn coordinator(&self) -> Option<SoilRegionId> {
        self.regions.first().copied()
    }
    pub fn pinned(&self) -> bool {
        self.pinned
    }
}

/// Validated desired residency and its physical load/unload delta.
/// Dropping a plan changes nothing; commit only after required transfers succeed.
pub struct SoilResidencyPlan {
    epoch: Arc<()>,
    tiles: BTreeMap<SoilTileCoord, SoilTileClaims>,
    activate: Vec<SoilTileCoord>,
    deactivate: Vec<SoilTileCoord>,
}

impl SoilResidencyPlan {
    pub fn activate(&self) -> &[SoilTileCoord] {
        &self.activate
    }
    pub fn deactivate(&self) -> &[SoilTileCoord] {
        &self.deactivate
    }
    pub fn tiles(&self) -> impl ExactSizeIterator<Item = SoilTileCoord> + '_ {
        self.tiles.keys().copied()
    }
    pub fn claims(&self, tile: SoilTileCoord) -> Option<&SoilTileClaims> {
        self.tiles.get(&tile)
    }
}

/// CPU lifecycle metadata only; this type does not solve, snapshot or sleep soil.
/// Physics owners must pin every tile with moving/otherwise awake soil, including
/// tiles outside machine regions. Capacity failure never evicts another tile.
pub struct SoilTileResidency {
    limits: SoilResidencyLimits,
    epoch: Arc<()>,
    tiles: BTreeMap<SoilTileCoord, SoilTileClaims>,
}

impl SoilTileResidency {
    pub fn new(limits: SoilResidencyLimits) -> Result<Self, SoilStateError> {
        if limits.regions == 0 || limits.tiles == 0 || limits.claims == 0 {
            return Err(SoilStateError::InvalidLimits);
        }
        Ok(Self {
            limits,
            epoch: Arc::new(()),
            tiles: BTreeMap::new(),
        })
    }

    pub fn tiles(&self) -> impl ExactSizeIterator<Item = SoilTileCoord> + '_ {
        self.tiles.keys().copied()
    }

    pub fn limits(&self) -> SoilResidencyLimits {
        self.limits
    }

    pub fn claims(&self, tile: SoilTileCoord) -> Option<&SoilTileClaims> {
        self.tiles.get(&tile)
    }

    /// Pins and region requests replace the previous complete desired set.
    /// Duplicate tiles are shared; duplicate region IDs are rejected. Input
    /// claims, including duplicates, count against the work/allocation bound.
    pub fn plan(
        &self,
        regions: &[SoilRegionRequest<'_>],
        pins: &[SoilTileCoord],
    ) -> Result<SoilResidencyPlan, SoilStateError> {
        if regions.len() > self.limits.regions || pins.len() > self.limits.claims {
            return Err(SoilStateError::CapacityExceeded);
        }
        let claims = regions
            .iter()
            .try_fold(pins.len(), |n, region| n.checked_add(region.tiles.len()));
        if claims.is_none_or(|n| n > self.limits.claims) {
            return Err(SoilStateError::CapacityExceeded);
        }
        let mut seen = BTreeSet::new();
        let mut tiles = BTreeMap::<_, SoilTileClaims>::new();
        for region in regions {
            if !seen.insert(region.id) {
                return Err(SoilStateError::DuplicateRegion);
            }
            for &tile in region.tiles {
                tiles
                    .entry(tile)
                    .or_insert_with(|| SoilTileClaims {
                        regions: BTreeSet::new(),
                        pinned: false,
                    })
                    .regions
                    .insert(region.id);
            }
        }
        for &tile in pins {
            tiles
                .entry(tile)
                .or_insert_with(|| SoilTileClaims {
                    regions: BTreeSet::new(),
                    pinned: false,
                })
                .pinned = true;
        }
        if tiles.len() > self.limits.tiles {
            return Err(SoilStateError::CapacityExceeded);
        }
        let activate = tiles
            .keys()
            .filter(|key| !self.tiles.contains_key(key))
            .copied()
            .collect();
        let deactivate = self
            .tiles
            .keys()
            .filter(|key| !tiles.contains_key(key))
            .copied()
            .collect();
        Ok(SoilResidencyPlan {
            epoch: self.epoch.clone(),
            tiles,
            activate,
            deactivate,
        })
    }

    pub fn commit(&mut self, plan: SoilResidencyPlan) -> Result<(), SoilStateError> {
        if !Arc::ptr_eq(&self.epoch, &plan.epoch) {
            return Err(SoilStateError::StalePlan);
        }
        self.tiles = plan.tiles;
        self.epoch = Arc::new(());
        Ok(())
    }
}

#[cfg(test)]
#[path = "soil_state_tests.rs"]
mod tests;
