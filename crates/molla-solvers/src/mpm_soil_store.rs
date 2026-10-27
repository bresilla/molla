//! Incremental tile blobs with a synchronized, atomically replaced manifest.

use super::*;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[path = "mpm_soil_framed_store.rs"]
mod framed_store;
pub use framed_store::{
    MpmSoilFrameStore, MpmSoilFrameStoreRead, MpmSoilFrameStoreReadCompletion,
    MpmSoilFrameStoreReadTask, MpmSoilFrameStoreWritebackCompletion,
    MpmSoilFrameStoreWritebackTask,
};

#[path = "mpm_soil_canonical_store.rs"]
mod canonical_store;
pub use canonical_store::{
    MpmSoilCanonicalStore, MpmSoilCanonicalStoreRead, MpmSoilCanonicalStoreReadCompletion,
    MpmSoilCanonicalStoreReadTask, MpmSoilCanonicalStoreWritebackCompletion,
    MpmSoilCanonicalStoreWritebackTask,
};

#[path = "mpm_soil_store_read.rs"]
mod read;
pub use read::{MpmSoilStoreReadCompletion, MpmSoilStoreReadTask};

#[path = "mpm_soil_store_writeback.rs"]
mod writeback;

#[path = "mpm_soil_store_write_task.rs"]
mod write_task;
pub use write_task::{MpmSoilStoreWritebackCompletion, MpmSoilStoreWritebackTask};

const STORE_MAGIC: &[u8; 8] = b"MOLLSTOR";
const MAX_METADATA: u64 = 1024;
const ENTRY_BYTES: u64 = 60;
const EPOCH_CACHE_TILES: usize = 1024;

#[path = "mpm_soil_store_manifest.rs"]
mod manifest;

#[path = "mpm_soil_store_tree_manifest.rs"]
mod tree_manifest;

/// Bounds apply before manifest/chunk allocation and full archive loading.
#[derive(Clone, Copy, Debug)]
pub struct MpmSoilStoreLimits {
    pub particles: u32,
    pub tiles: u32,
}

/// Logical newly written bytes include manifests/index pages, not filesystem overhead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MpmSoilStoreStats {
    pub generation: u64,
    pub written_tiles: u32,
    pub reused_tiles: u32,
    pub written_bytes: u64,
}

/// Committed archive dimensions and retained index/cache accounting, without I/O.
/// Encoded manifest bytes describe HEAD, excluding separate index-page files.
/// Buffer bytes exclude allocator/container/path overhead, file descriptors, OS page
/// cache, worker stacks, selected payloads and transient publication buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmSoilStoreStorageStats {
    pub generation: Option<u64>,
    pub archived_tiles: u32,
    pub archived_particles: u32,
    pub manifest_encoded_bytes: u64,
    pub manifest_buffer_bytes: u64,
    pub cached_tiles: u32,
    pub cache_tile_limit: u32,
}

/// Selected-read bounds; encoded bytes reserve 1024 per tile plus 168 per particle.
/// Bounds exclude the already-open manifest and host allocator/container overhead.
#[derive(Clone, Copy, Debug)]
pub struct MpmSoilStoreReadLimits {
    pub tiles: u32,
    pub particles: u32,
    pub encoded_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmSoilStoreReadStats {
    pub generation: u64,
    pub tiles: u32,
    pub particles: u32,
    pub read_bytes: u64,
}

/// Read-only selected payloads, not a complete archive for replacement saving.
/// Integrity and particle-ID uniqueness cover selected blobs only.
pub struct MpmSoilStoreRead {
    store: Arc<()>,
    selection: MpmSoilSelection,
    stats: MpmSoilStoreReadStats,
}

impl MpmSoilStoreRead {
    pub fn tiles(&self) -> impl ExactSizeIterator<Item = SoilTileCoord> + '_ {
        self.selection.tiles.iter().copied()
    }
    pub fn selection(&self) -> &MpmSoilSelection {
        &self.selection
    }
    pub fn stats(&self) -> MpmSoilStoreReadStats {
        self.stats
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    count: u32,
    hash: [u8; 32],
}

#[cfg(test)]
struct Manifest {
    generation: u64,
    particles: u32,
    metadata: Vec<u8>,
    entries: BTreeMap<SoilTileCoord, Entry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    ChunksSynced,
    ManifestSynced,
    Published,
    DirectorySynced,
}

/// Exclusive store in an existing private directory on a local filesystem.
/// Requires advisory file locks, atomic same-directory rename and directory fsync.
/// The caller owns directory creation/durability and must not alter its contents
/// while open. Failed writes poison the handle; reopen to inspect HEAD.
/// A failure after publication may leave the new generation visible. Old blobs
/// remain until explicit collection; full loading materializes all tiles.
/// Disk I/O is explicit, never part of a solver substep. `begin_read` moves the
/// exclusive handle into a background read; `begin_writeback` does so for partial
/// writes. Other I/O APIs are synchronous.
/// Manifests retain fixed read buffers. Open validation and v1 publication perform
/// field-sized I/O; v2 selected publication replaces changed paths. Reuse epochs
/// retain at most 1024 tiles; misses verify blob bytes.
pub struct MpmSoilStore {
    epoch: Arc<()>,
    root: PathBuf,
    _lock: StoreLock,
    directory: File,
    limits: MpmSoilStoreLimits,
    manifest: Option<manifest::ManifestFile>,
    epochs: BTreeMap<SoilTileCoord, Arc<()>>,
    failed: bool,
    #[cfg(test)]
    serial_index_updates: bool,
    framed: bool,
    canonical: bool,
}

struct StoreLock(File);

impl StoreLock {
    fn acquire(file: File) -> Result<Self> {
        file.try_lock().map_err(io_error)?;
        Ok(Self(file))
    }
}

impl Drop for StoreLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn io_error(error: impl std::fmt::Display) -> Error {
    invalid(&format!("tile store I/O: {error}"))
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    if !fs::symlink_metadata(path).map_err(io_error)?.is_file() {
        return Err(invalid("store entry is not a regular file"));
    }
    let file = File::open(path).map_err(io_error)?;
    if file.metadata().map_err(io_error)?.len() > limit {
        return Err(invalid("store file exceeds declared allocation bound"));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    if bytes.len() as u64 > limit {
        return Err(invalid("store file grew beyond declared bound"));
    }
    Ok(bytes)
}

fn metadata(archive: &MpmSoilTileArchive) -> Vec<u8> {
    archive.encode_selection(archive.select(&[], 0).expect("empty archive selection"))
}

fn metadata_in_frame(archive: &MpmSoilTileArchive, origin: Option<[f64; 3]>) -> Vec<u8> {
    let bytes = metadata(archive);
    match origin {
        Some(origin) => framed::encode_payload(bytes, origin),
        None => bytes,
    }
}

fn decode_archive(
    bytes: &[u8],
    max_particles: u32,
    framed: bool,
) -> Result<(MpmSoilTileArchive, Option<[f64; 3]>)> {
    if framed {
        let frame = MpmSoilFrameTileArchive::from_bytes(bytes, max_particles)?;
        Ok((frame.archive, Some(frame.origin)))
    } else {
        Ok((MpmSoilTileArchive::from_bytes(bytes, max_particles)?, None))
    }
}

#[cfg(test)]
impl Manifest {
    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(STORE_MAGIC);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&[0; 32]);
        bytes.extend_from_slice(&self.generation.to_le_bytes());
        bytes.extend_from_slice(&(self.metadata.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.metadata);
        for (key, entry) in &self.entries {
            for coordinate in key.0 {
                bytes.extend_from_slice(&coordinate.to_le_bytes());
            }
            bytes.extend_from_slice(&entry.count.to_le_bytes());
            bytes.extend_from_slice(&entry.hash);
        }
        let digest = blake3::hash(&bytes[HEADER_BYTES..]);
        bytes[12..HEADER_BYTES].copy_from_slice(digest.as_bytes());
        bytes
    }

    fn decode(bytes: &[u8], limits: MpmSoilStoreLimits) -> Result<Self> {
        Self::decode_mode(bytes, limits, false)
    }

    fn decode_mode(bytes: &[u8], limits: MpmSoilStoreLimits, framed: bool) -> Result<Self> {
        Self::decode_family(bytes, limits, framed, false)
    }

    fn decode_family(
        bytes: &[u8],
        limits: MpmSoilStoreLimits,
        framed: bool,
        canonical: bool,
    ) -> Result<Self> {
        if bytes.len() < HEADER_BYTES + 16
            || &bytes[..8] != STORE_MAGIC
            || bytes[8..12] != 1_u32.to_le_bytes()
            || bytes[12..HEADER_BYTES] != *blake3::hash(&bytes[HEADER_BYTES..]).as_bytes()
        {
            return Err(invalid("invalid store manifest header/digest"));
        }
        let mut reader = Bytes(&bytes[HEADER_BYTES..]);
        let generation = u64::from_le_bytes(reader.take()?);
        let size = u32::from_le_bytes(reader.take()?) as usize;
        let tiles = u32::from_le_bytes(reader.take()?);
        if generation == 0
            || size as u64 > MAX_METADATA
            || tiles > limits.tiles
            || reader.0.len() as u64 != size as u64 + u64::from(tiles) * ENTRY_BYTES
        {
            return Err(invalid("store manifest dimensions exceed bounds"));
        }
        let (metadata, tail) = reader.0.split_at(size);
        if canonical {
            MpmSoilCanonicalTileArchive::from_bytes(metadata, 0)?;
        } else {
            decode_archive(metadata, 0, framed)?;
        }
        reader.0 = tail;
        let mut entries = BTreeMap::new();
        let mut count = 0_u64;
        for _ in 0..tiles {
            let key = SoilTileCoord([
                i64::from_le_bytes(reader.take()?),
                i64::from_le_bytes(reader.take()?),
                i64::from_le_bytes(reader.take()?),
            ]);
            let entry = Entry {
                count: u32::from_le_bytes(reader.take()?),
                hash: reader.take()?,
            };
            count += u64::from(entry.count);
            if entry.count == 0
                || count > u64::from(limits.particles)
                || entries
                    .last_key_value()
                    .is_some_and(|(&last, _)| last >= key)
            {
                return Err(invalid("store tile counts/order exceed bounds"));
            }
            entries.insert(key, entry);
        }
        Ok(Self {
            generation,
            particles: count as u32,
            metadata: metadata.to_vec(),
            entries,
        })
    }
}

impl MpmSoilStore {
    /// Opens or initializes store metadata in an existing directory. Another open
    /// handle/process rejects immediately. A missing HEAD is an uncommitted store.
    pub fn open(path: impl AsRef<Path>, limits: MpmSoilStoreLimits) -> Result<Self> {
        Self::open_mode(path, limits, false)
    }

    fn open_mode(path: impl AsRef<Path>, limits: MpmSoilStoreLimits, framed: bool) -> Result<Self> {
        Self::open_family(path, limits, framed, false)
    }

    fn open_family(
        path: impl AsRef<Path>,
        limits: MpmSoilStoreLimits,
        framed: bool,
        canonical: bool,
    ) -> Result<Self> {
        let root = fs::canonicalize(path).map_err(io_error)?;
        let directory = File::open(&root).map_err(io_error)?;
        if !directory.metadata().map_err(io_error)?.is_dir() {
            return Err(invalid("store root must be an existing directory"));
        }
        let lock_path = root.join("LOCK");
        if let Ok(meta) = fs::symlink_metadata(&lock_path) {
            if !meta.is_file() {
                return Err(invalid("store lock is not a regular file"));
            }
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)
            .map_err(io_error)?;
        let lock = StoreLock::acquire(lock)?;
        directory.sync_all().map_err(io_error)?;
        let head = root.join("HEAD");
        let manifest = match fs::symlink_metadata(&head) {
            Ok(_) => Some(manifest::ManifestFile::open(
                &head, limits, framed, canonical,
            )?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(io_error(e)),
        };
        let pending = root.join("pending");
        match fs::symlink_metadata(&pending) {
            Ok(meta) if meta.is_file() => fs::remove_file(pending).map_err(io_error)?,
            Ok(_) => return Err(invalid("store pending entry is not a regular file")),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => return Err(io_error(e)),
        }
        Ok(Self {
            epoch: Arc::new(()),
            root,
            _lock: lock,
            directory,
            limits,
            manifest,
            epochs: BTreeMap::new(),
            failed: false,
            #[cfg(test)]
            serial_index_updates: false,
            framed,
            canonical,
        })
    }

    /// Last successfully opened/published generation; not authoritative after a
    /// failed write that may already have replaced HEAD.
    pub fn generation(&self) -> Option<u64> {
        self.manifest.as_ref().map(|m| m.generation)
    }

    /// Cached committed particle count, without entry enumeration or disk I/O.
    /// None means uncommitted; failed handles reject until reopened.
    pub fn total_particle_count(&self) -> Result<Option<u32>> {
        self.healthy()?;
        Ok(self.manifest.as_ref().map(|m| m.particles))
    }

    /// Reports retained archive metadata, not process RSS or solver memory.
    /// Failed stores reject accounting until reopened; no file/cursor access occurs.
    pub fn storage_stats(&self) -> Result<MpmSoilStoreStorageStats> {
        self.healthy()?;
        let (archived_tiles, archived_particles, manifest_encoded_bytes, manifest_buffer_bytes) =
            self.manifest.as_ref().map_or((0, 0, 0, 0), |manifest| {
                let (encoded, retained) = manifest.storage_bytes();
                (manifest.tiles, manifest.particles, encoded, retained)
            });
        Ok(MpmSoilStoreStorageStats {
            generation: self.generation(),
            archived_tiles,
            archived_particles,
            manifest_encoded_bytes,
            manifest_buffer_bytes,
            cached_tiles: self.epochs.len() as u32,
            cache_tile_limit: EPOCH_CACHE_TILES as u32,
        })
    }

    fn healthy(&self) -> Result<()> {
        if self.failed {
            return Err(invalid(
                "tile store failed; drop and reopen to resolve HEAD",
            ));
        }
        Ok(())
    }

    fn blob_path(&self, hash: &[u8; 32]) -> PathBuf {
        self.root
            .join(format!("{}.tile", blake3::Hash::from_bytes(*hash).to_hex()))
    }

    /// Synchronously materializes stored coordinates without reading tile payloads.
    /// Enumeration uses field-sized key storage and propagates manifest I/O errors.
    pub fn tiles(&self) -> Result<impl Iterator<Item = SoilTileCoord> + '_> {
        self.healthy()?;
        let keys = self
            .manifest
            .iter()
            .flat_map(|m| m.iter())
            .map(|entry| entry.map(|(key, _)| key))
            .collect::<Result<Vec<_>>>()?;
        Ok(keys.into_iter())
    }

    /// Committed layout from manifest metadata, without particle payload I/O.
    pub fn layout(&self) -> Result<Option<SoilTileLayout>> {
        self.healthy()?;
        self.manifest
            .as_ref()
            .map(|manifest| {
                decode_archive(&manifest.metadata, 0, self.framed)
                    .map(|(archive, _)| archive.layout)
            })
            .transpose()
    }

    /// Committed count for a tile; None means absent, not an unloaded payload.
    pub fn tile_particle_count(&self, tile: SoilTileCoord) -> Result<Option<u32>> {
        self.healthy()?;
        match &self.manifest {
            Some(manifest) => Ok(manifest.get(tile)?.map(|entry| entry.count)),
            None => Ok(None),
        }
    }

    /// Rejects a read from another open handle or an older committed generation.
    pub fn validate_read(&self, read: &MpmSoilStoreRead) -> Result<()> {
        self.healthy()?;
        if !Arc::ptr_eq(&self.epoch, &read.store)
            || self.generation() != Some(read.stats.generation)
        {
            return Err(invalid("stale or foreign store read"));
        }
        Ok(())
    }

    fn selected_keys(
        &self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
    ) -> Result<BTreeSet<SoilTileCoord>> {
        self.selected_keys_sized(tiles, limits, MAX_METADATA, 168)
    }

    fn selected_keys_sized(
        &self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
        tile_bytes: u64,
        particle_bytes: u64,
    ) -> Result<BTreeSet<SoilTileCoord>> {
        let unique = self.bounded_read_keys(tiles, limits)?;
        let Some(manifest) = &self.manifest else {
            return Ok(unique);
        };
        let mut count = 0u64;
        let mut bound = 0u64;
        for key in &unique {
            let entry = manifest
                .get(*key)?
                .ok_or_else(|| invalid("unknown stored tile"))?;
            count = count
                .checked_add(u64::from(entry.count))
                .ok_or_else(|| invalid("selected particle count overflow"))?;
            bound = bound
                .checked_add(tile_bytes + u64::from(entry.count) * particle_bytes)
                .ok_or_else(|| invalid("selected byte bound overflow"))?;
            if count > u64::from(limits.particles) || bound > limits.encoded_bytes {
                return Err(invalid(
                    "selected particles or encoded bytes exceed read bound",
                ));
            }
        }
        Ok(unique)
    }

    fn bounded_read_keys(
        &self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
    ) -> Result<BTreeSet<SoilTileCoord>> {
        self.healthy()?;
        let mut unique = BTreeSet::new();
        if self.manifest.is_none() {
            return Ok(unique);
        }
        for key in tiles {
            if unique.contains(key) {
                continue;
            }
            if unique.len() as u64 >= u64::from(limits.tiles) {
                return Err(invalid("selected tile count exceeds read bound"));
            }
            unique.insert(*key);
        }
        Ok(unique)
    }

    fn existing_read_keys(
        &self,
        keys: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
        tile_bytes: u64,
        particle_bytes: u64,
    ) -> Result<Vec<SoilTileCoord>> {
        self.healthy()?;
        let mut existing = BTreeSet::new();
        for &key in keys {
            if self.tile_particle_count(key)?.is_none() || existing.contains(&key) {
                continue;
            }
            if existing.len() as u64 >= u64::from(limits.tiles) {
                return Err(invalid("selected tile count exceeds read bound"));
            }
            existing.insert(key);
        }
        let keys: Vec<_> = existing.into_iter().collect();
        self.selected_keys_sized(&keys, limits, tile_bytes, particle_bytes)?;
        Ok(keys)
    }

    /// Blocking read of unique requested tiles. All bounds and keys are checked
    /// before reading blobs; unselected payloads and save-reuse epochs are untouched.
    /// Returns None for an uncommitted store. This does not activate live GPU soil.
    pub fn read_tiles(
        &self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
    ) -> Result<Option<MpmSoilStoreRead>> {
        let unique = self.selected_keys(tiles, limits)?;
        let Some(manifest) = &self.manifest else {
            return Ok(None);
        };
        let (mut archive, _) = decode_archive(&manifest.metadata, 0, self.framed)?;
        let mut ids = BTreeSet::new();
        let mut read_bytes = 0;
        for &key in &unique {
            let entry = manifest
                .get(key)?
                .ok_or_else(|| invalid("unknown stored tile"))?;
            let (tile, bytes) = self.read_tile(manifest, key, &entry)?;
            if tile.ids.iter().any(|&id| !ids.insert(id)) {
                return Err(invalid(
                    "duplicate global particle ID across selected tile blobs",
                ));
            }
            read_bytes += bytes;
            archive.count += entry.count;
            archive.tiles.insert(key, tile);
        }
        let selection =
            archive.select(&unique.into_iter().collect::<Vec<_>>(), limits.particles)?;
        Ok(Some(MpmSoilStoreRead {
            store: self.epoch.clone(),
            selection,
            stats: MpmSoilStoreReadStats {
                generation: manifest.generation,
                tiles: archive.tiles.len() as u32,
                particles: archive.count,
                read_bytes,
            },
        }))
    }

    fn read_tile(
        &self,
        manifest: &manifest::ManifestFile,
        key: SoilTileCoord,
        entry: &Entry,
    ) -> Result<(Tile, u64)> {
        let bytes = read_bounded(
            &self.blob_path(&entry.hash),
            MAX_METADATA + u64::from(entry.count) * 168,
        )?;
        if blake3::hash(&bytes).as_bytes() != &entry.hash {
            return Err(invalid("tile blob content hash mismatch"));
        }
        let (mut archive, origin) = decode_archive(&bytes, entry.count, self.framed)?;
        if archive.count != entry.count
            || archive.tiles.len() != 1
            || metadata_in_frame(&archive, origin) != manifest.metadata
        {
            return Err(invalid("tile blob metadata/count mismatch"));
        }
        let tile = archive
            .tiles
            .remove(&key)
            .ok_or_else(|| invalid("tile blob coordinate mismatch"))?;
        Ok((tile, bytes.len() as u64))
    }

    /// Loads every referenced tile with hash, count, metadata and global-ID checks.
    /// Unknown unreferenced blobs are ignored; missing/corrupt referenced data fails.
    pub fn load(&mut self) -> Result<Option<MpmSoilTileArchive>> {
        self.healthy()?;
        let Some(manifest) = &self.manifest else {
            return Ok(None);
        };
        let (mut archive, _) = decode_archive(&manifest.metadata, 0, self.framed)?;
        let mut ids = BTreeSet::new();
        let mut epochs = BTreeMap::new();
        for item in manifest.iter() {
            let (key, entry) = item?;
            let (tile, _) = self.read_tile(manifest, key, &entry)?;
            if tile.ids.iter().any(|&id| !ids.insert(id)) {
                return Err(invalid("duplicate global particle ID across tile blobs"));
            }
            archive.count += entry.count;
            if epochs.len() < EPOCH_CACHE_TILES {
                epochs.insert(key, tile.epoch.clone());
            }
            archive.tiles.insert(key, tile);
        }
        self.epochs = epochs;
        Ok(Some(archive))
    }

    /// Saves only changed tile payloads, then fsyncs and atomically publishes HEAD.
    /// An unchanged archive performs no writes. Full replacement processes all
    /// stored tiles; unchanged cached payloads are not reread to detect disk damage.
    pub fn save(&mut self, archive: &MpmSoilTileArchive) -> Result<MpmSoilStoreStats> {
        self.save_with(archive, |_| Ok(()))
    }

    /// Explicitly upgrades a committed v1 index to immutable pages and a v2 HEAD.
    /// Blocks on migration I/O and invalidates old reads when the generation changes.
    /// Existing v2 indexes are unchanged. Publication errors poison the handle;
    /// reopen to resolve HEAD. Uncommitted-store rejection leaves the handle healthy.
    pub fn upgrade_index(&mut self) -> Result<MpmSoilStoreStats> {
        self.upgrade_index_with(|_| Ok(()))
    }

    fn upgrade_index_with(
        &mut self,
        mut barrier: impl FnMut(Stage) -> Result<()>,
    ) -> Result<MpmSoilStoreStats> {
        self.healthy()?;
        let manifest = self
            .manifest
            .as_ref()
            .ok_or_else(|| invalid("index upgrade requires committed soil"))?;
        let metadata = manifest.metadata.clone();
        let stats = MpmSoilStoreStats {
            reused_tiles: manifest.tiles,
            ..Default::default()
        };
        let result = self.publish_with_index(
            metadata,
            BTreeMap::new(),
            Some(&BTreeSet::new()),
            self.epochs.clone(),
            stats,
            &mut barrier,
            true,
        );
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    /// Validate HEAD's complete archive, then delete unreferenced regular tile/index
    /// blobs and sync the directory. Returns logical removed bytes. This explicit
    /// maintenance discards older generations, not arbitrary files or symlinks.
    pub fn collect_unreferenced(&mut self) -> Result<u64> {
        self.healthy()?;
        if self.load()?.is_none() {
            return Err(invalid("collection requires a committed manifest"));
        }
        self.collect_validated()
    }

    fn collect_validated(&mut self) -> Result<u64> {
        let mut retained: BTreeSet<_> = self
            .manifest
            .as_ref()
            .unwrap()
            .iter()
            .map(|entry| entry.map(|(_, entry)| self.blob_path(&entry.hash)))
            .collect::<Result<_>>()?;
        retained.extend(self.manifest.as_ref().unwrap().index_files()?);
        let result = (|| {
            let mut removed = 0_u64;
            for entry in fs::read_dir(&self.root).map_err(io_error)? {
                let entry = entry.map_err(io_error)?;
                let name = entry.file_name();
                let Some(name) = name.to_str() else {
                    continue;
                };
                if !((name.len() == 69 && name.ends_with(".tile"))
                    || (name.len() == 70 && name.ends_with(".index")))
                    || !name.as_bytes()[..64]
                        .iter()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
                    || retained.contains(&entry.path())
                    || !entry.file_type().map_err(io_error)?.is_file()
                {
                    continue;
                }
                let bytes = entry.metadata().map_err(io_error)?.len();
                fs::remove_file(entry.path()).map_err(io_error)?;
                removed += bytes;
            }
            self.directory.sync_all().map_err(io_error)?;
            Ok(removed)
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn save_with(
        &mut self,
        archive: &MpmSoilTileArchive,
        barrier: impl FnMut(Stage) -> Result<()>,
    ) -> Result<MpmSoilStoreStats> {
        self.save_with_frame(archive, None, barrier)
    }

    fn save_with_frame(
        &mut self,
        archive: &MpmSoilTileArchive,
        origin: Option<[f64; 3]>,
        mut barrier: impl FnMut(Stage) -> Result<()>,
    ) -> Result<MpmSoilStoreStats> {
        self.healthy()?;
        if self.canonical || self.framed != origin.is_some() {
            return Err(invalid("store/archive frame family mismatch"));
        }
        if archive.count > self.limits.particles
            || archive.tiles.len() as u64 > u64::from(self.limits.tiles)
        {
            return Err(invalid("archive exceeds store limits"));
        }
        let result = self.save_inner(archive, None, origin, &mut barrier);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn stage(&self, bytes: &[u8]) -> Result<()> {
        let mut file = File::create_new(self.root.join("pending")).map_err(io_error)?;
        file.write_all(bytes).map_err(io_error)?;
        file.sync_all().map_err(io_error)
    }

    fn save_inner(
        &mut self,
        archive: &MpmSoilTileArchive,
        replace: Option<&BTreeSet<SoilTileCoord>>,
        origin: Option<[f64; 3]>,
        barrier: &mut impl FnMut(Stage) -> Result<()>,
    ) -> Result<MpmSoilStoreStats> {
        let metadata = metadata_in_frame(archive, origin);
        let same_metadata = self
            .manifest
            .as_ref()
            .is_some_and(|m| m.metadata == metadata);
        let mut entries = BTreeMap::new();
        let mut epochs = BTreeMap::new();
        let mut stats = MpmSoilStoreStats::default();
        if let Some(replace) = replace {
            let manifest = self.manifest.as_ref().expect("validated patch generation");
            stats.reused_tiles = manifest.retained(replace)?.0;
        }
        for (&key, tile) in &archive.tiles {
            if epochs.len() < EPOCH_CACHE_TILES {
                epochs.insert(key, tile.epoch.clone());
            }
            if same_metadata
                && self
                    .epochs
                    .get(&key)
                    .is_some_and(|epoch| Arc::ptr_eq(epoch, &tile.epoch))
            {
                entries.insert(
                    key,
                    self.manifest
                        .as_ref()
                        .unwrap()
                        .get(key)?
                        .ok_or_else(|| invalid("cached tile missing from manifest"))?,
                );
                stats.reused_tiles += 1;
                continue;
            }
            let bytes = archive.encode_selection(archive.select(&[key], archive.count)?);
            let bytes = match origin {
                Some(origin) => framed::encode_payload(bytes, origin),
                None => bytes,
            };
            let hash = self.install_blob(&bytes, &mut stats)?;
            entries.insert(
                key,
                Entry {
                    count: tile.ids.len() as u32,
                    hash,
                },
            );
        }
        self.publish(metadata, entries, replace, epochs, stats, barrier)
    }

    fn install_blob(&self, bytes: &[u8], stats: &mut MpmSoilStoreStats) -> Result<[u8; 32]> {
        let hash = *blake3::hash(bytes).as_bytes();
        let path = self.blob_path(&hash);
        match fs::symlink_metadata(&path) {
            Ok(_) => {
                if read_bounded(&path, bytes.len() as u64)? != bytes {
                    return Err(invalid("existing tile blob does not match its hash"));
                }
                stats.reused_tiles += 1;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                self.stage(bytes)?;
                fs::rename(self.root.join("pending"), path).map_err(io_error)?;
                stats.written_tiles += 1;
                stats.written_bytes += bytes.len() as u64;
            }
            Err(e) => return Err(io_error(e)),
        }
        Ok(hash)
    }

    fn edit_index(
        &self,
        pages: &mut impl index::Pages,
        root: Option<&index::Link>,
        edits: &[(SoilTileCoord, Option<Entry>)],
    ) -> Result<Option<index::Link>> {
        #[cfg(test)]
        if self.serial_index_updates {
            let mut root = root.cloned();
            for (key, entry) in edits {
                root = index::update(pages, root.as_ref(), *key, entry.clone())?;
            }
            return Ok(root);
        }
        index::update_many(pages, root, edits)
    }

    fn publish(
        &mut self,
        metadata: Vec<u8>,
        entries: BTreeMap<SoilTileCoord, Entry>,
        replace: Option<&BTreeSet<SoilTileCoord>>,
        epochs: BTreeMap<SoilTileCoord, Arc<()>>,
        stats: MpmSoilStoreStats,
        barrier: &mut impl FnMut(Stage) -> Result<()>,
    ) -> Result<MpmSoilStoreStats> {
        let incremental = self
            .manifest
            .as_ref()
            .is_some_and(|manifest| manifest.tree.is_some());
        self.publish_with_index(
            metadata,
            entries,
            replace,
            epochs,
            stats,
            barrier,
            incremental,
        )
    }

    fn publish_with_index(
        &mut self,
        metadata: Vec<u8>,
        entries: BTreeMap<SoilTileCoord, Entry>,
        replace: Option<&BTreeSet<SoilTileCoord>>,
        epochs: BTreeMap<SoilTileCoord, Arc<()>>,
        mut stats: MpmSoilStoreStats,
        barrier: &mut impl FnMut(Stage) -> Result<()>,
        incremental: bool,
    ) -> Result<MpmSoilStoreStats> {
        let base = self.manifest.as_ref();
        let (retained_tiles, retained_particles) = match (base, replace) {
            (Some(base), Some(replace)) => base.retained(replace)?,
            (_, None) => (0, 0),
            _ => return Err(invalid("patch requires committed manifest")),
        };
        if replace.is_some_and(|set| entries.keys().any(|key| !set.contains(key))) {
            return Err(invalid("manifest patch escapes affected tiles"));
        }
        let tiles = u32::try_from(entries.len())
            .ok()
            .and_then(|count| retained_tiles.checked_add(count))
            .ok_or_else(|| invalid("stored tile count overflow"))?;
        let particles = entries
            .values()
            .try_fold(retained_particles, |sum, entry| {
                sum.checked_add(entry.count)
                    .ok_or_else(|| invalid("stored particle count overflow"))
            })?;
        let mut unchanged = base.is_some_and(|base| {
            base.metadata == metadata && base.tiles == tiles && base.tree.is_some() == incremental
        });
        if unchanged {
            let base = base.unwrap();
            for (&key, entry) in &entries {
                if base.get(key)?.as_ref() != Some(entry) {
                    unchanged = false;
                    break;
                }
            }
        }
        if unchanged {
            stats.generation = self.manifest.as_ref().unwrap().generation;
            self.epochs = epochs;
            return Ok(stats);
        }
        let generation = self
            .generation()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| invalid("tile store generation exhausted"))?;
        let tree_root = if incremental {
            if tiles > self.limits.tiles || particles > self.limits.particles {
                return Err(invalid("incremental publication exceeds store bounds"));
            }
            let mut pages = tree_manifest::Pages {
                directory: &self.root,
                written_bytes: 0,
            };
            let root = if let (Some(tree), Some(replace)) =
                (base.and_then(|base| base.tree.as_ref()), replace)
            {
                let edits: Vec<_> = replace
                    .iter()
                    .map(|&key| (key, entries.get(&key).cloned()))
                    .collect();
                self.edit_index(&mut pages, tree.root.as_ref(), &edits)?
            } else {
                index::build_sorted(
                    &mut pages,
                    manifest::merged(base.filter(|_| replace.is_some()), replace, &entries),
                )?
            };
            if root
                .as_ref()
                .map_or((0, 0), |root| (root.tiles, root.particles))
                != (tiles, particles)
            {
                return Err(invalid("incremental candidate totals mismatch"));
            }
            stats.written_bytes += pages.written_bytes;
            Some(root)
        } else {
            None
        };
        self.directory.sync_all().map_err(io_error)?;
        barrier(Stage::ChunksSynced)?;
        let bytes = if let Some(root) = &tree_root {
            let encoded = tree_manifest::encode(generation, &metadata, root.as_ref(), self.limits)?;
            self.stage(&encoded)?;
            encoded.len() as u64
        } else {
            let mut file = File::create_new(self.root.join("pending")).map_err(io_error)?;
            let bytes = manifest::write_entries(
                &mut file,
                generation,
                &metadata,
                tiles,
                particles,
                manifest::merged(base.filter(|_| replace.is_some()), replace, &entries),
                self.limits,
            )?;
            file.sync_all().map_err(io_error)?;
            bytes
        };
        let manifest = if let Some(root) = &tree_root {
            manifest::ManifestFile::open_created_tree(
                &self.root.join("pending"),
                self.limits,
                self.framed,
                self.canonical,
                root.as_ref(),
            )?
        } else {
            manifest::ManifestFile::open(
                &self.root.join("pending"),
                self.limits,
                self.framed,
                self.canonical,
            )?
        };
        barrier(Stage::ManifestSynced)?;
        fs::rename(self.root.join("pending"), self.root.join("HEAD")).map_err(io_error)?;
        barrier(Stage::Published)?;
        self.directory.sync_all().map_err(io_error)?;
        barrier(Stage::DirectorySynced)?;
        stats.generation = generation;
        stats.written_bytes += bytes;
        self.manifest = Some(manifest);
        self.epochs = epochs;
        Ok(stats)
    }
}

#[cfg(test)]
#[path = "mpm_soil_store_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "mpm_soil_store_stream_tests.rs"]
mod stream_tests;

#[cfg(test)]
#[path = "mpm_soil_store_incremental_tests.rs"]
mod incremental_tests;

#[path = "mpm_soil_store_index.rs"]
mod index;
