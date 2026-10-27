use super::*;

const MAGIC: &[u8; 8] = b"MOLLFTIL";
const FRAMED_HEADER: usize = HEADER_BYTES + 24;

/// Identified host soil archive with exact world-tile keys and local positions.
/// The shared world offset is retained separately; no absolute f32 positions are
/// formed. Live GPU/store ownership and per-tile coordinate frames are absent.
pub struct MpmSoilFrameTileArchive {
    pub(super) archive: MpmSoilTileArchive,
    pub(super) origin: [f64; 3],
}

/// Global-ID-sorted local soil selection with its immutable world offset.
/// This is an archive snapshot, not a residency lease or device identity buffer.
pub struct MpmSoilFrameSelection {
    pub(super) selection: MpmSoilSelection,
    pub(super) origin: [f64; 3],
}

impl MpmSoilFrameSelection {
    pub(in crate::mpm) fn local_selection(&self) -> &MpmSoilSelection {
        &self.selection
    }

    pub fn world_origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Local-coordinate particle state; interpret it together with world_origin.
    pub fn soil(&self) -> &MpmSoilCheckpoint {
        self.selection.checkpoint()
    }

    pub fn particle_ids(&self) -> &[SoilParticleId] {
        self.selection.particle_ids()
    }

    pub fn material_ids(&self) -> &[SoilMaterialId] {
        self.selection.material_ids()
    }

    /// Copy the selected physical/history payload into an owned framed checkpoint.
    pub fn to_frame_checkpoint(&self) -> Result<MpmSoilFrameCheckpoint> {
        MpmSoilFrameCheckpoint::new(self.selection.checkpoint.clone(), self.origin)
    }
}

impl MpmSoilFrameCheckpoint {
    /// Partition by exact split-coordinate world membership, retaining every
    /// local particle/history word and all caller-assigned global IDs.
    pub fn partition_tiles(
        &self,
        layout: SoilTileLayout,
        ids: &[SoilParticleId],
        material_ids: &[SoilMaterialId],
    ) -> Result<MpmSoilFrameTileArchive> {
        Ok(MpmSoilFrameTileArchive {
            archive: self.soil().partition_tiles_in_frame(
                layout,
                ids,
                material_ids,
                Some(self.world_origin()),
            )?,
            origin: self.world_origin(),
        })
    }
}

impl MpmSoilFrameTileArchive {
    pub fn world_origin(&self) -> [f64; 3] {
        self.origin
    }

    pub fn layout(&self) -> SoilTileLayout {
        self.archive.layout()
    }

    pub fn particle_count(&self) -> u32 {
        self.archive.particle_count()
    }

    pub fn material_ids(&self) -> &[SoilMaterialId] {
        self.archive.material_ids()
    }

    pub fn tiles(&self) -> impl ExactSizeIterator<Item = SoilTileCoord> + '_ {
        self.archive.tiles()
    }

    pub fn tile_particle_count(&self, tile: SoilTileCoord) -> Option<usize> {
        self.archive.tile_particle_count(tile)
    }

    /// Select deduplicated world tiles; unknown tiles and capacity excess reject.
    pub fn select(
        &self,
        tiles: &[SoilTileCoord],
        max_particles: u32,
    ) -> Result<MpmSoilFrameSelection> {
        Ok(MpmSoilFrameSelection {
            selection: self.archive.select(tiles, max_particles)?,
            origin: self.origin,
        })
    }

    /// Atomically repartition observed local state in the selected slot order.
    /// Frame, configuration, mass, reference volume and material identity must
    /// match. Stale/foreign selections reject before any archive mutation.
    pub fn writeback(
        &mut self,
        selection: &MpmSoilFrameSelection,
        updated: &MpmSoilFrameCheckpoint,
    ) -> Result<()> {
        if selection.origin.map(f64::to_bits) != self.origin.map(f64::to_bits)
            || updated.world_origin().map(f64::to_bits) != self.origin.map(f64::to_bits)
        {
            return Err(invalid("framed tile writeback changes world origin"));
        }
        self.archive
            .prepare_writeback_in_frame(&selection.selection, updated.soil(), Some(self.origin))?
            .commit();
        Ok(())
    }

    /// Return a new archive/selection epoch after an exact local/grid translation.
    /// World-tile membership and global IDs must remain unchanged. Failure leaves
    /// this archive intact; no live solver, machine or persisted store is rebased.
    pub fn recentered(&self, grid_origin: [f32; 3]) -> Result<Self> {
        let keys: Vec<_> = self.tiles().collect();
        let selected = self.select(&keys, self.particle_count())?;
        let shifted = selected.to_frame_checkpoint()?.recentered(grid_origin)?;
        let result =
            shifted.partition_tiles(self.layout(), selected.particle_ids(), self.material_ids())?;
        if self.archive.tiles.len() != result.archive.tiles.len()
            || self.archive.tiles.iter().any(|(key, tile)| {
                result
                    .archive
                    .tiles
                    .get(key)
                    .is_none_or(|other| other.ids != tile.ids)
            })
        {
            return Err(invalid(
                "soil archive recenter changes world-tile membership",
            ));
        }
        Ok(result)
    }

    /// Versioned frame/layout/ID/history envelope. Filesystem durability and
    /// atomic file replacement remain caller-owned.
    pub fn to_bytes(&self) -> Vec<u8> {
        encode_payload(self.archive.to_bytes(), self.origin)
    }

    pub fn from_bytes(bytes: &[u8], max_particles: u32) -> Result<Self> {
        if bytes.len() < FRAMED_HEADER
            || bytes.len() as u64 > FRAMED_HEADER as u64 + max_tile_encoded_bytes(max_particles)
            || &bytes[..8] != MAGIC
            || bytes[8..12] != 1u32.to_le_bytes()
            || bytes[12..HEADER_BYTES] != *blake3::hash(&bytes[HEADER_BYTES..]).as_bytes()
        {
            return Err(invalid(
                "invalid framed tile archive header, digest or allocation bound",
            ));
        }
        let mut reader = Bytes(&bytes[HEADER_BYTES..]);
        let origin = [reader.float()?, reader.float()?, reader.float()?];
        if origin.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite framed tile archive origin"));
        }
        Ok(Self {
            archive: MpmSoilTileArchive::from_bytes_in_frame(
                reader.0,
                max_particles,
                Some(origin),
            )?,
            origin,
        })
    }
}

pub(super) fn encode_payload(payload: Vec<u8>, origin: [f64; 3]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(FRAMED_HEADER + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&[0; 32]);
    for value in origin {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&payload);
    let digest = blake3::hash(&bytes[HEADER_BYTES..]);
    bytes[12..HEADER_BYTES].copy_from_slice(digest.as_bytes());
    bytes
}

#[cfg(test)]
#[path = "mpm_soil_framed_tiles_tests.rs"]
mod tests;
