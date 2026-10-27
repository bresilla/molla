use super::*;

#[path = "mpm_soil_framed_store_tasks.rs"]
mod tasks;
pub use tasks::{
    MpmSoilFrameStoreReadCompletion, MpmSoilFrameStoreReadTask,
    MpmSoilFrameStoreWritebackCompletion, MpmSoilFrameStoreWritebackTask,
};

/// Exclusive durable framed tile store using synchronized blobs and atomic HEAD.
/// Direct operations block; begin_read/begin_writeback move I/O to a worker.
/// Limits, poisoning and filesystem requirements
/// match MpmSoilStore; framed and unframed committed stores reject cross-opening.
pub struct MpmSoilFrameStore {
    inner: MpmSoilStore,
}

/// Bounded selected payloads and their request-time frame/generation identity.
/// This is not a complete replacement archive or a live residency lease.
pub struct MpmSoilFrameStoreRead {
    read: MpmSoilStoreRead,
    origin: [f64; 3],
}

impl MpmSoilFrameStoreRead {
    pub fn world_origin(&self) -> [f64; 3] {
        self.origin
    }
    pub fn soil(&self) -> &MpmSoilCheckpoint {
        self.read.selection().checkpoint()
    }
    pub fn particle_ids(&self) -> &[SoilParticleId] {
        self.read.selection().particle_ids()
    }
    pub fn material_ids(&self) -> &[SoilMaterialId] {
        self.read.selection().material_ids()
    }
    pub fn tiles(&self) -> impl ExactSizeIterator<Item = SoilTileCoord> + '_ {
        self.read.selection.tiles.iter().copied()
    }
    pub fn stats(&self) -> MpmSoilStoreReadStats {
        self.read.stats()
    }

    /// Copy selected physical/history and identity data for detached frame restore.
    /// The original read remains the store writeback/provenance token.
    pub fn selection_snapshot(&self) -> MpmSoilFrameSelection {
        let selected = &self.read.selection;
        MpmSoilFrameSelection {
            selection: MpmSoilSelection {
                epoch: selected.epoch.clone(),
                tiles: selected.tiles.clone(),
                checkpoint: selected.checkpoint.clone(),
                ids: selected.ids.clone(),
                material_ids: selected.material_ids.clone(),
            },
            origin: self.origin,
        }
    }
}

impl MpmSoilFrameStore {
    pub(in crate::mpm::checkpoint) fn into_world_parts(self) -> Result<(MpmSoilStore, [f64; 3])> {
        let origin = self
            .world_origin()?
            .ok_or_else(|| invalid("disk world requires committed soil"))?;
        Ok((self.inner, origin))
    }

    pub fn open(path: impl AsRef<Path>, limits: MpmSoilStoreLimits) -> Result<Self> {
        Ok(Self {
            inner: MpmSoilStore::open_mode(path, limits, true)?,
        })
    }
    pub fn generation(&self) -> Option<u64> {
        self.inner.generation()
    }
    /// Explicit blocking v1-to-v2 index upgrade; changed generations stale prior reads.
    pub fn upgrade_index(&mut self) -> Result<MpmSoilStoreStats> {
        self.inner.upgrade_index()
    }
    /// Retained index/cache accounting without payload or filesystem access.
    pub fn storage_stats(&self) -> Result<MpmSoilStoreStorageStats> {
        self.inner.storage_stats()
    }
    pub fn world_origin(&self) -> Result<Option<[f64; 3]>> {
        self.inner.healthy()?;
        self.inner
            .manifest
            .as_ref()
            .map(|manifest| {
                decode_archive(&manifest.metadata, 0, true)?
                    .1
                    .ok_or_else(|| invalid("framed store lost its frame metadata"))
            })
            .transpose()
    }
    pub fn layout(&self) -> Result<Option<SoilTileLayout>> {
        self.inner.layout()
    }
    pub fn tiles(&self) -> Result<impl Iterator<Item = SoilTileCoord> + '_> {
        self.inner.tiles()
    }
    pub fn tile_particle_count(&self, tile: SoilTileCoord) -> Result<Option<u32>> {
        self.inner.tile_particle_count(tile)
    }
    /// Cached committed particle count without field enumeration or disk I/O.
    pub fn total_particle_count(&self) -> Result<Option<u32>> {
        self.inner.total_particle_count()
    }

    /// Atomically replace the full archive, including its shared frame.
    /// Unchanged blobs/metadata are reused; changed generations stale old reads.
    pub fn save(&mut self, archive: &MpmSoilFrameTileArchive) -> Result<MpmSoilStoreStats> {
        self.inner
            .save_with_frame(&archive.archive, Some(archive.origin), |_| Ok(()))
    }
    pub fn load(&mut self) -> Result<Option<MpmSoilFrameTileArchive>> {
        let origin = self.world_origin()?;
        self.inner
            .load()?
            .map(|archive| {
                Ok(MpmSoilFrameTileArchive {
                    archive,
                    origin: origin
                        .ok_or_else(|| invalid("framed store lost its frame metadata"))?,
                })
            })
            .transpose()
    }

    /// Read only bounded requested tiles, preserving the committed world offset.
    /// Integrity/ID uniqueness checks cover selected payloads, not unloaded tiles.
    pub fn read_tiles(
        &self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
    ) -> Result<Option<MpmSoilFrameStoreRead>> {
        let origin = self.world_origin()?;
        self.inner
            .read_tiles(tiles, limits)?
            .map(|read| {
                Ok(MpmSoilFrameStoreRead {
                    read,
                    origin: origin
                        .ok_or_else(|| invalid("framed store lost its frame metadata"))?,
                })
            })
            .transpose()
    }
    pub fn validate_read(&self, read: &MpmSoilFrameStoreRead) -> Result<()> {
        self.inner.validate_read(&read.read)?;
        if self.world_origin()?.map(|v| v.map(f64::to_bits)) != Some(read.origin.map(f64::to_bits))
        {
            return Err(invalid("store read frame differs from committed frame"));
        }
        Ok(())
    }

    /// Commit same-frame selected updates and merge checked destinations.
    /// Bounds cover affected tiles; publication errors require reopening.
    pub fn writeback(
        &mut self,
        read: &MpmSoilFrameStoreRead,
        updated: &MpmSoilFrameCheckpoint,
        limits: MpmSoilStoreReadLimits,
    ) -> Result<MpmSoilStoreStats> {
        self.validate_read(read)?;
        if updated.world_origin().map(f64::to_bits) != read.origin.map(f64::to_bits) {
            return Err(invalid("framed store writeback changes world origin"));
        }
        self.inner.writeback(&read.read, updated.soil(), limits)
    }
    pub fn collect_unreferenced(&mut self) -> Result<u64> {
        self.inner.collect_unreferenced()
    }
}

#[cfg(test)]
#[path = "mpm_soil_framed_store_tests.rs"]
mod tests;
