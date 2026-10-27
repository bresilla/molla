//! Incremental canonical tile blobs sharing the exclusive publication protocol.

use super::super::canonical::CanonicalTile;
use super::*;

const TILE_BYTES: u64 = MAX_METADATA + 28;
const PARTICLE_BYTES: u64 = 181;

#[path = "mpm_soil_canonical_store_tasks.rs"]
mod tasks;
pub use tasks::{
    MpmSoilCanonicalStoreReadCompletion, MpmSoilCanonicalStoreReadTask,
    MpmSoilCanonicalStoreWritebackCompletion, MpmSoilCanonicalStoreWritebackTask,
};

/// Durable canonical storage with blocking direct APIs and explicit worker tasks.
/// Bounds apply to selected/affected payloads. Manifests have fixed retained buffers;
/// upgraded v2 indexes publish changed paths. No GPU membership/frame is owned.
/// Locking, synchronization, filesystem and poisoning contracts match MpmSoilStore.
pub struct MpmSoilCanonicalStore {
    inner: MpmSoilStore,
}

/// Bounded canonical payloads tagged with the exclusive store/generation identity.
/// Selection assembles an explicit solver frame without changing stored coordinates.
pub struct MpmSoilCanonicalStoreRead {
    store: Arc<()>,
    archive: MpmSoilCanonicalTileArchive,
    stats: MpmSoilStoreReadStats,
}

impl MpmSoilCanonicalStoreRead {
    pub fn stats(&self) -> MpmSoilStoreReadStats {
        self.stats
    }
    pub fn tiles(&self) -> impl ExactSizeIterator<Item = SoilTileCoord> + '_ {
        self.archive.tiles()
    }
    /// Assemble the entire bounded read; keep both read and selection for writeback.
    /// Conversion and physical grid-boundary contracts match canonical archive selection.
    pub fn select(
        &self,
        world_origin: [f64; 3],
        grid_origin: [f32; 3],
    ) -> Result<MpmSoilCanonicalSelection> {
        self.select_window(world_origin, grid_origin, [0; 3])
    }

    /// Assembles the bounded read in an explicit client lattice window.
    pub fn select_window(
        &self,
        world_origin: [f64; 3],
        grid_origin: [f32; 3],
        cells: [i32; 3],
    ) -> Result<MpmSoilCanonicalSelection> {
        self.archive.select_window(
            &self.archive.tiles().collect::<Vec<_>>(),
            self.archive.count,
            world_origin,
            grid_origin,
            cells,
        )
    }
}

impl MpmSoilCanonicalStore {
    pub fn open(path: impl AsRef<Path>, limits: MpmSoilStoreLimits) -> Result<Self> {
        Ok(Self {
            inner: MpmSoilStore::open_family(path, limits, false, true)?,
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
    pub fn layout(&self) -> Result<Option<SoilTileLayout>> {
        self.inner.healthy()?;
        self.inner
            .manifest
            .as_ref()
            .map(|manifest| {
                MpmSoilCanonicalTileArchive::from_bytes(&manifest.metadata, 0)
                    .map(|archive| archive.layout)
            })
            .transpose()
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
    pub fn validate_read(&self, read: &MpmSoilCanonicalStoreRead) -> Result<()> {
        self.inner.healthy()?;
        if !Arc::ptr_eq(&self.inner.epoch, &read.store)
            || self.generation() != Some(read.stats.generation)
        {
            return Err(invalid("stale or foreign canonical store read"));
        }
        Ok(())
    }

    /// Read only requested canonical tiles after deduplicated count/byte preflight.
    /// Integrity/ID checks cover selected payloads, not unloaded field data.
    pub fn read_tiles(
        &self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
    ) -> Result<Option<MpmSoilCanonicalStoreRead>> {
        let keys = self
            .inner
            .selected_keys_sized(tiles, limits, TILE_BYTES, PARTICLE_BYTES)?;
        let Some(manifest) = &self.inner.manifest else {
            return Ok(None);
        };
        let mut archive = MpmSoilCanonicalTileArchive::from_bytes(&manifest.metadata, 0)?;
        let mut ids = BTreeSet::new();
        let mut read_bytes = 0;
        for key in keys {
            let entry = manifest
                .get(key)?
                .ok_or_else(|| invalid("unknown canonical stored tile"))?;
            let (tile, bytes) = self.read_tile(manifest, key, &entry)?;
            if tile.tile.ids.iter().any(|&id| !ids.insert(id)) {
                return Err(invalid("duplicate canonical ID across selected blobs"));
            }
            archive.count += entry.count;
            archive.tiles.insert(key, tile);
            read_bytes += bytes;
        }
        let stats = MpmSoilStoreReadStats {
            generation: manifest.generation,
            tiles: archive.tiles.len() as u32,
            particles: archive.count,
            read_bytes,
        };
        Ok(Some(MpmSoilCanonicalStoreRead {
            store: self.inner.epoch.clone(),
            archive,
            stats,
        }))
    }

    fn read_tile(
        &self,
        manifest: &manifest::ManifestFile,
        key: SoilTileCoord,
        entry: &Entry,
    ) -> Result<(CanonicalTile, u64)> {
        let bytes = read_bounded(
            &self.inner.blob_path(&entry.hash),
            TILE_BYTES + u64::from(entry.count) * PARTICLE_BYTES,
        )?;
        if blake3::hash(&bytes).as_bytes() != &entry.hash {
            return Err(invalid("canonical tile blob hash mismatch"));
        }
        let mut archive = MpmSoilCanonicalTileArchive::from_bytes(&bytes, entry.count)?;
        if archive.count != entry.count
            || archive.tiles.len() != 1
            || archive.encoded_tiles(&[])? != manifest.metadata
        {
            return Err(invalid("canonical tile metadata or count mismatch"));
        }
        let tile = archive
            .tiles
            .remove(&key)
            .ok_or_else(|| invalid("canonical tile coordinate mismatch"))?;
        Ok((tile, bytes.len() as u64))
    }

    /// Load and validate the full field; explicitly unbounded by active-region size.
    pub fn load(&mut self) -> Result<Option<MpmSoilCanonicalTileArchive>> {
        let keys: Vec<_> = self.tiles()?.collect();
        let read = self.read_tiles(
            &keys,
            MpmSoilStoreReadLimits {
                tiles: self.inner.limits.tiles,
                particles: self.inner.limits.particles,
                encoded_bytes: u64::MAX,
            },
        )?;
        if let Some(read) = read {
            self.inner.epochs = read
                .archive
                .tiles
                .iter()
                .take(EPOCH_CACHE_TILES)
                .map(|(&key, tile)| (key, tile.tile.epoch.clone()))
                .collect();
            Ok(Some(read.archive))
        } else {
            Ok(None)
        }
    }

    /// Incrementally save a complete canonical archive; unchanged epochs reuse blobs.
    /// Publication errors poison the store, requiring reopen to resolve HEAD.
    pub fn save(&mut self, archive: &MpmSoilCanonicalTileArchive) -> Result<MpmSoilStoreStats> {
        self.save_with(archive, |_| Ok(()))
    }

    fn save_with(
        &mut self,
        archive: &MpmSoilCanonicalTileArchive,
        mut barrier: impl FnMut(Stage) -> Result<()>,
    ) -> Result<MpmSoilStoreStats> {
        self.inner.healthy()?;
        if archive.count > self.inner.limits.particles
            || archive.tiles.len() as u64 > u64::from(self.inner.limits.tiles)
        {
            return Err(invalid("canonical archive exceeds store limits"));
        }
        let result = self.save_inner(archive, None, &mut barrier);
        if result.is_err() {
            self.inner.failed = true;
        }
        result
    }

    fn save_inner(
        &mut self,
        archive: &MpmSoilCanonicalTileArchive,
        replace: Option<&BTreeSet<SoilTileCoord>>,
        barrier: &mut impl FnMut(Stage) -> Result<()>,
    ) -> Result<MpmSoilStoreStats> {
        let metadata = archive.encoded_tiles(&[])?;
        let same_metadata = self
            .inner
            .manifest
            .as_ref()
            .is_some_and(|manifest| manifest.metadata == metadata);
        let mut entries = BTreeMap::new();
        let mut epochs = BTreeMap::new();
        let mut stats = MpmSoilStoreStats::default();
        if let Some(replace) = replace {
            let manifest = self
                .inner
                .manifest
                .as_ref()
                .expect("validated canonical patch");
            if !same_metadata {
                return Err(invalid("canonical patch changes metadata"));
            }
            stats.reused_tiles = manifest.retained(replace)?.0;
        }
        for (&key, tile) in &archive.tiles {
            if epochs.len() < EPOCH_CACHE_TILES {
                epochs.insert(key, tile.tile.epoch.clone());
            }
            if same_metadata
                && self
                    .inner
                    .epochs
                    .get(&key)
                    .is_some_and(|epoch| Arc::ptr_eq(epoch, &tile.tile.epoch))
            {
                entries.insert(
                    key,
                    self.inner
                        .manifest
                        .as_ref()
                        .unwrap()
                        .get(key)?
                        .ok_or_else(|| invalid("cached canonical tile missing from manifest"))?,
                );
                stats.reused_tiles += 1;
            } else {
                let bytes = archive.encoded_tiles(&[key])?;
                let hash = self.inner.install_blob(&bytes, &mut stats)?;
                entries.insert(
                    key,
                    Entry {
                        count: tile.tile.ids.len() as u32,
                        hash,
                    },
                );
            }
        }
        self.inner
            .publish(metadata, entries, replace, epochs, stats, barrier)
    }

    /// Publish an entire bounded read's selected-frame update and merge destinations.
    /// Limits cover affected source/destination tiles. Frames may differ between
    /// reads, but an update must match its selection; stored coordinates stay canonical.
    pub fn writeback(
        &mut self,
        read: &MpmSoilCanonicalStoreRead,
        selection: &MpmSoilCanonicalSelection,
        updated: &MpmSoilFrameCheckpoint,
        limits: MpmSoilStoreReadLimits,
    ) -> Result<MpmSoilStoreStats> {
        self.writeback_with(read, selection, updated, limits, |_| Ok(()))
    }

    fn writeback_with(
        &mut self,
        read: &MpmSoilCanonicalStoreRead,
        selection: &MpmSoilCanonicalSelection,
        updated: &MpmSoilFrameCheckpoint,
        limits: MpmSoilStoreReadLimits,
        mut barrier: impl FnMut(Stage) -> Result<()>,
    ) -> Result<MpmSoilStoreStats> {
        self.validate_writeback_request(read, selection, updated, limits)?;
        read.archive.validate_writeback(selection, updated)?;
        let manifest = self.inner.manifest.as_ref().unwrap();
        let source: BTreeSet<_> = read.tiles().collect();
        let mut affected = source.clone();
        for position in updated.soil().stream(0).chunks_exact(4) {
            affected.insert(
                read.archive
                    .layout
                    .tile_at_local(
                        std::array::from_fn(|a| f64::from(f32::from_bits(position[a]))),
                        updated.world_origin(),
                    )
                    .map_err(|e| invalid(&e.to_string()))?,
            );
            if affected.len() as u64 > u64::from(limits.tiles) {
                return Err(invalid("canonical affected tiles exceed bound"));
            }
        }
        let particles =
            affected
                .difference(&source)
                .try_fold(u64::from(read.archive.count), |sum, key| {
                    Ok::<_, Error>(
                        sum + u64::from(manifest.get(*key)?.map_or(0, |entry| entry.count)),
                    )
                })?;
        if affected.len() as u64 > u64::from(limits.tiles)
            || particles > u64::from(limits.particles)
            || affected.len() as u64 * TILE_BYTES + particles * PARTICLE_BYTES
                > limits.encoded_bytes
        {
            return Err(invalid("canonical affected payloads exceed bounds"));
        }
        let mut archive = read.archive.copy_for_patch();
        let mut ids: BTreeSet<_> = selection.framed.particle_ids().iter().copied().collect();
        for key in affected.difference(&source) {
            if let Some(entry) = manifest.get(*key)? {
                let (tile, _) = self.read_tile(manifest, *key, &entry)?;
                if tile.tile.ids.iter().any(|&id| !ids.insert(id)) {
                    return Err(invalid("duplicate canonical destination ID"));
                }
                archive.count += entry.count;
                archive.tiles.insert(*key, tile);
            }
        }
        archive.writeback(selection, updated)?;
        let retained = manifest.retained(&affected)?.0;
        if retained as u64 + archive.tiles.len() as u64 > u64::from(self.inner.limits.tiles) {
            return Err(invalid("canonical writeback exceeds total tile bound"));
        }
        let result = self.save_inner(&archive, Some(&affected), &mut barrier);
        if result.is_err() {
            self.inner.failed = true;
        }
        result
    }

    pub fn collect_unreferenced(&mut self) -> Result<u64> {
        if self.load()?.is_none() {
            return Err(invalid("canonical collection requires committed soil"));
        }
        self.inner.collect_validated()
    }

    fn validate_writeback_request(
        &self,
        read: &MpmSoilCanonicalStoreRead,
        selection: &MpmSoilCanonicalSelection,
        updated: &MpmSoilFrameCheckpoint,
        limits: MpmSoilStoreReadLimits,
    ) -> Result<()> {
        self.validate_read(read)?;
        if !Arc::ptr_eq(&read.archive.epoch, &selection.epoch)
            || !selection.tiles.iter().copied().eq(read.archive.tiles())
            || updated.world_origin().map(f64::to_bits)
                != selection.framed.world_origin().map(f64::to_bits)
            || updated.soil().count != read.archive.count
            || selection.framed.soil().count != read.archive.count
            || read.archive.count > limits.particles
            || selection.tiles.len() as u64 > u64::from(limits.tiles)
            || selection.tiles.len() as u64 * TILE_BYTES
                + u64::from(read.archive.count) * PARTICLE_BYTES
                > limits.encoded_bytes
        {
            return Err(invalid(
                "canonical write request provenance/frame/count/bounds mismatch",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "mpm_soil_canonical_store_tests.rs"]
mod tests;
