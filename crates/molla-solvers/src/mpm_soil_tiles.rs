//! Lossless world-tile partitioning of explicit host soil archives.

use super::*;
use molla_vehicle::{SoilMaterialId, SoilParticleId, SoilTileCoord, SoilTileLayout};
use std::collections::{BTreeMap, BTreeSet};

#[path = "mpm_soil_framed_tiles.rs"]
mod framed;
pub use framed::{MpmSoilFrameSelection, MpmSoilFrameTileArchive};

#[path = "mpm_soil_canonical_tiles.rs"]
mod canonical;
pub use canonical::{MpmSoilCanonicalSelection, MpmSoilCanonicalTileArchive};

#[path = "mpm_soil_store.rs"]
mod store;
pub use store::{
    MpmSoilCanonicalStore, MpmSoilCanonicalStoreRead, MpmSoilCanonicalStoreReadCompletion,
    MpmSoilCanonicalStoreReadTask, MpmSoilCanonicalStoreWritebackCompletion,
    MpmSoilCanonicalStoreWritebackTask, MpmSoilFrameStore, MpmSoilFrameStoreRead,
    MpmSoilFrameStoreReadCompletion, MpmSoilFrameStoreReadTask,
    MpmSoilFrameStoreWritebackCompletion, MpmSoilFrameStoreWritebackTask, MpmSoilStore,
    MpmSoilStoreLimits, MpmSoilStoreRead, MpmSoilStoreReadCompletion, MpmSoilStoreReadLimits,
    MpmSoilStoreReadStats, MpmSoilStoreReadTask, MpmSoilStoreStats, MpmSoilStoreStorageStats,
    MpmSoilStoreWritebackCompletion, MpmSoilStoreWritebackTask,
};

const TILE_MAGIC: &[u8; 8] = b"MOLLTILE";

#[derive(Clone)]
struct Tile {
    epoch: Arc<()>,
    ids: Vec<SoilParticleId>,
    words: Vec<u32>,
}

impl Tile {
    fn particle_stream(&self, stream: usize, particle: usize) -> &[u32] {
        let start =
            STRIDES[..stream].iter().sum::<usize>() * self.ids.len() + STRIDES[stream] * particle;
        &self.words[start..start + STRIDES[stream]]
    }
}

/// Archived volumetric soil with stable caller-assigned global IDs.
/// Selection is a snapshot, not a live residency lease or physical activation.
/// Partitioning/assembly run on explicit host checkpoints, never in GPU stepping.
pub struct MpmSoilTileArchive {
    epoch: Arc<()>,
    cfg: MpmConfig,
    materials: Vec<MpmRheology>,
    material_ids: Vec<SoilMaterialId>,
    layout: SoilTileLayout,
    count: u32,
    tiles: BTreeMap<SoilTileCoord, Tile>,
}

/// Unique selected particles sorted by global ID, with unchanged per-particle bits.
/// Material-table slots map to `material_ids()`. Restoring this checkpoint may
/// reorder solver slots; it does not promise bitwise-identical later reductions.
pub struct MpmSoilSelection {
    epoch: Arc<()>,
    tiles: Vec<SoilTileCoord>,
    checkpoint: MpmSoilCheckpoint,
    ids: Vec<SoilParticleId>,
    material_ids: Vec<SoilMaterialId>,
}

impl MpmSoilSelection {
    pub fn checkpoint(&self) -> &MpmSoilCheckpoint {
        &self.checkpoint
    }
    pub fn particle_ids(&self) -> &[SoilParticleId] {
        &self.ids
    }
    pub fn material_ids(&self) -> &[SoilMaterialId] {
        &self.material_ids
    }
}

impl MpmSoilCheckpoint {
    /// Partition all particles by current world position without changing mass,
    /// reference volume, resolution, material slots or any history word. IDs must
    /// be unique and cover every particle/material row, including the default row.
    pub fn partition_tiles(
        &self,
        layout: SoilTileLayout,
        ids: &[SoilParticleId],
        material_ids: &[SoilMaterialId],
    ) -> Result<MpmSoilTileArchive> {
        self.partition_tiles_in_frame(layout, ids, material_ids, None)
    }

    fn partition_tiles_in_frame(
        &self,
        layout: SoilTileLayout,
        ids: &[SoilParticleId],
        material_ids: &[SoilMaterialId],
        origin: Option<[f64; 3]>,
    ) -> Result<MpmSoilTileArchive> {
        self.require_origin_only_grid()?;
        self.partition_particle_tiles_in_frame(layout, ids, material_ids, origin)
    }

    fn partition_particle_tiles_in_frame(
        &self,
        layout: SoilTileLayout,
        ids: &[SoilParticleId],
        material_ids: &[SoilMaterialId],
        origin: Option<[f64; 3]>,
    ) -> Result<MpmSoilTileArchive> {
        self.validate()?;
        if ids.len() != self.count as usize
            || material_ids.len() != self.materials.len().max(1)
            || ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len()
            || material_ids.iter().copied().collect::<BTreeSet<_>>().len() != material_ids.len()
        {
            return Err(invalid(
                "tile archive requires complete unique particle/material IDs",
            ));
        }
        let mut groups = BTreeMap::<SoilTileCoord, Vec<(SoilParticleId, usize)>>::new();
        for (index, &id) in ids.iter().enumerate() {
            let position =
                std::array::from_fn(|axis| f32::from_bits(self.stream(0)[index * 4 + axis]) as f64);
            let tile = match origin {
                Some(origin) => layout.tile_at_local(position, origin),
                None => layout.tile_at(position),
            }
            .map_err(|e| invalid(&e.to_string()))?;
            groups.entry(tile).or_default().push((id, index));
        }
        let mut tiles = BTreeMap::new();
        for (coordinate, mut particles) in groups {
            particles.sort_unstable_by_key(|&(id, _)| id);
            let mut words = Vec::with_capacity(particles.len() * WORDS_PER_PARTICLE);
            for (stream, stride) in STRIDES.into_iter().enumerate() {
                for &(_, index) in &particles {
                    words.extend_from_slice(
                        &self.stream(stream)[index * stride..(index + 1) * stride],
                    );
                }
            }
            tiles.insert(
                coordinate,
                Tile {
                    epoch: Arc::new(()),
                    ids: particles.into_iter().map(|(id, _)| id).collect(),
                    words,
                },
            );
        }
        Ok(MpmSoilTileArchive {
            epoch: Arc::new(()),
            cfg: self.cfg,
            materials: self.materials.clone(),
            material_ids: material_ids.to_vec(),
            layout,
            count: self.count,
            tiles,
        })
    }
}

impl MpmSoilTileArchive {
    pub fn layout(&self) -> SoilTileLayout {
        self.layout
    }
    pub fn particle_count(&self) -> u32 {
        self.count
    }
    pub fn material_ids(&self) -> &[SoilMaterialId] {
        &self.material_ids
    }
    pub fn tiles(&self) -> impl ExactSizeIterator<Item = SoilTileCoord> + '_ {
        self.tiles.keys().copied()
    }
    pub fn tile_particle_count(&self, tile: SoilTileCoord) -> Option<usize> {
        self.tiles.get(&tile).map(|tile| tile.ids.len())
    }

    /// Deduplicates overlapping tile requests. Unknown tiles and excessive total
    /// particle count fail before allocating particle output; no tile is omitted.
    pub fn select(&self, tiles: &[SoilTileCoord], max_particles: u32) -> Result<MpmSoilSelection> {
        self.select_from(
            tiles,
            max_particles,
            self.epoch.clone(),
            |key| self.tiles.get(key),
            false,
        )
    }

    fn select_from<'a>(
        &self,
        tiles: &[SoilTileCoord],
        max_particles: u32,
        epoch: Arc<()>,
        lookup: impl Fn(&SoilTileCoord) -> Option<&'a Tile>,
        allow_empty: bool,
    ) -> Result<MpmSoilSelection> {
        let unique: BTreeSet<_> = tiles.iter().copied().collect();
        let mut count = 0_u64;
        let mut present = Vec::new();
        for key in &unique {
            let Some(tile) = lookup(key) else {
                if allow_empty {
                    continue;
                }
                return Err(invalid("unknown archive tile"));
            };
            present.push(*key);
            count += tile.ids.len() as u64;
            if count > u64::from(max_particles) {
                return Err(invalid("selected particles exceed capacity"));
            }
        }
        let mut particles = Vec::with_capacity(count as usize);
        for key in &present {
            let tile = lookup(key).expect("validated archive tile");
            particles.extend(
                tile.ids
                    .iter()
                    .enumerate()
                    .map(|(index, &id)| (id, tile, index)),
            );
        }
        particles.sort_unstable_by_key(|&(id, _, _)| id);
        let mut words = Vec::with_capacity(count as usize * WORDS_PER_PARTICLE);
        for stream in 0..STRIDES.len() {
            for &(_, tile, index) in &particles {
                words.extend_from_slice(tile.particle_stream(stream, index));
            }
        }
        Ok(MpmSoilSelection {
            epoch,
            tiles: present,
            checkpoint: MpmSoilCheckpoint {
                cfg: self.cfg,
                grid_window_cells: [0; 3],
                materials: self.materials.clone(),
                count: count as u32,
                words,
            },
            ids: particles.into_iter().map(|(id, _, _)| id).collect(),
            material_ids: self.material_ids.clone(),
        })
    }

    /// Replace a selected snapshot with observed GPU state in the same slot order.
    /// Repartitions moved particles and merges destination tiles without changing
    /// global IDs or unselected records. Stale/foreign snapshots, count/mass/reference
    /// volume changes and material/configuration changes fail before mutation. All selections
    /// become stale after success; this does not commit live GPU residency.
    pub fn writeback(
        &mut self,
        selection: &MpmSoilSelection,
        updated: &MpmSoilCheckpoint,
    ) -> Result<()> {
        self.prepare_writeback(selection, updated)?.commit();
        Ok(())
    }

    pub(super) fn prepare_writeback<'a>(
        &'a mut self,
        selection: &MpmSoilSelection,
        updated: &MpmSoilCheckpoint,
    ) -> Result<PreparedWriteback<'a>> {
        self.prepare_writeback_in_frame(selection, updated, None)
    }

    fn prepare_writeback_in_frame<'a>(
        &'a mut self,
        selection: &MpmSoilSelection,
        updated: &MpmSoilCheckpoint,
        origin: Option<[f64; 3]>,
    ) -> Result<PreparedWriteback<'a>> {
        self.validate_writeback(selection, updated)?;
        let replacement = updated.partition_tiles_in_frame(
            self.layout,
            &selection.ids,
            &self.material_ids,
            origin,
        )?;
        let mut patches = replacement.tiles;
        for (key, tile) in &mut patches {
            if selection.tiles.binary_search(key).is_ok() {
                continue;
            }
            if let Some(untouched) = self.tiles.get(key) {
                *tile = merge_tiles(untouched, tile)?;
            }
        }
        Ok(PreparedWriteback {
            archive: self,
            patches,
            removed: selection.tiles.clone(),
            epoch: Arc::new(()),
        })
    }

    fn validate_writeback(
        &self,
        selection: &MpmSoilSelection,
        updated: &MpmSoilCheckpoint,
    ) -> Result<()> {
        updated.require_origin_only_grid()?;
        let mut compatible = updated.cfg;
        compatible.max_particles = self.cfg.max_particles;
        compatible.max_rigid = self.cfg.max_rigid;
        if !Arc::ptr_eq(&self.epoch, &selection.epoch)
            || updated.count as usize != selection.ids.len()
            || config_words(compatible) != config_words(self.cfg)
            || updated.materials != self.materials
        {
            return Err(invalid(
                "stale/foreign tile selection or changed particle/material configuration",
            ));
        }
        updated.validate()?;
        if [4, 5, 7]
            .into_iter()
            .any(|stream| updated.stream(stream) != selection.checkpoint.stream(stream))
        {
            return Err(invalid(
                "writeback changes mass, reference volume or particle material identity",
            ));
        }
        Ok(())
    }

    /// Versioned container preserving tile layout, global IDs and the full soil
    /// checkpoint. Archive I/O, durability and live ownership are caller-owned.
    pub fn to_bytes(&self) -> Vec<u8> {
        let keys: Vec<_> = self.tiles().collect();
        let selection = self
            .select(&keys, self.count)
            .expect("validated archive count");
        self.encode_selection(selection)
    }

    fn encode_selection(&self, selection: MpmSoilSelection) -> Vec<u8> {
        let checkpoint = selection.checkpoint.to_bytes();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(TILE_MAGIC);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&[0; 32]);
        for coordinate in self.layout.origin().into_iter().chain([self.layout.edge()]) {
            bytes.extend_from_slice(&coordinate.to_le_bytes());
        }
        bytes.extend_from_slice(&selection.checkpoint.count.to_le_bytes());
        bytes.extend_from_slice(&(self.material_ids.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(checkpoint.len() as u64).to_le_bytes());
        for id in &self.material_ids {
            bytes.extend_from_slice(&id.0.to_le_bytes());
        }
        for id in selection.ids {
            bytes.extend_from_slice(&id.0.to_le_bytes());
        }
        bytes.extend_from_slice(&checkpoint);
        let digest = blake3::hash(&bytes[HEADER_BYTES..]);
        bytes[12..HEADER_BYTES].copy_from_slice(digest.as_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8], max_particles: u32) -> Result<Self> {
        Self::from_bytes_in_frame(bytes, max_particles, None)
    }

    fn from_bytes_in_frame(
        bytes: &[u8],
        max_particles: u32,
        origin: Option<[f64; 3]>,
    ) -> Result<Self> {
        let max = max_tile_encoded_bytes(max_particles);
        if bytes.len() < HEADER_BYTES + 48
            || bytes.len() as u64 > max
            || &bytes[..8] != TILE_MAGIC
            || bytes[8..12] != 1_u32.to_le_bytes()
            || bytes[12..HEADER_BYTES] != *blake3::hash(&bytes[HEADER_BYTES..]).as_bytes()
        {
            return Err(invalid(
                "invalid tile archive header, digest or payload bound",
            ));
        }
        let mut reader = Bytes(&bytes[HEADER_BYTES..]);
        let layout_origin = [reader.float()?, reader.float()?, reader.float()?];
        let layout = SoilTileLayout::new(layout_origin, reader.float()?)
            .map_err(|e| invalid(&e.to_string()))?;
        let count = u32::from_le_bytes(reader.take()?);
        let rows = u32::from_le_bytes(reader.take()?);
        let checkpoint_bytes = u64::from_le_bytes(reader.take()?);
        if count > max_particles
            || rows == 0
            || rows as usize > MAX_MPM_MATERIALS
            || ((u64::from(count) + u64::from(rows)) * 8).checked_add(checkpoint_bytes)
                != Some(reader.0.len() as u64)
        {
            return Err(invalid("tile archive count or length mismatch"));
        }
        let materials = (0..rows)
            .map(|_| Ok(SoilMaterialId(u64::from_le_bytes(reader.take()?))))
            .collect::<Result<Vec<_>>>()?;
        let ids = (0..count)
            .map(|_| Ok(SoilParticleId(u64::from_le_bytes(reader.take()?))))
            .collect::<Result<Vec<_>>>()?;
        let checkpoint = MpmSoilCheckpoint::from_bytes(reader.0, max_particles)?;
        checkpoint.partition_tiles_in_frame(layout, &ids, &materials, origin)
    }
}

fn max_tile_encoded_bytes(max_particles: u32) -> u64 {
    2 * HEADER_BYTES as u64
        + 48
        + 23 * 4
        + MAX_MPM_MATERIALS as u64 * 24
        + u64::from(max_particles) * (WORDS_PER_PARTICLE as u64 * 4 + 8)
}

pub(super) struct PreparedWriteback<'a> {
    archive: &'a mut MpmSoilTileArchive,
    patches: BTreeMap<SoilTileCoord, Tile>,
    removed: Vec<SoilTileCoord>,
    epoch: Arc<()>,
}

impl PreparedWriteback<'_> {
    pub(super) fn select_existing(
        &self,
        tiles: &[SoilTileCoord],
        max_particles: u32,
    ) -> Result<MpmSoilSelection> {
        self.archive.select_from(
            tiles,
            max_particles,
            self.epoch.clone(),
            |key| {
                self.patches.get(key).or_else(|| {
                    if self.removed.binary_search(key).is_ok() {
                        None
                    } else {
                        self.archive.tiles.get(key)
                    }
                })
            },
            true,
        )
    }

    pub(super) fn commit(self) {
        for key in &self.removed {
            self.archive.tiles.remove(key);
        }
        self.archive.tiles.extend(self.patches);
        self.archive.epoch = self.epoch;
    }
}

fn merge_tiles(a: &Tile, b: &Tile) -> Result<Tile> {
    let mut particles: Vec<_> = [a, b]
        .into_iter()
        .flat_map(|tile| {
            tile.ids
                .iter()
                .enumerate()
                .map(move |(index, &id)| (id, tile, index))
        })
        .collect();
    particles.sort_unstable_by_key(|&(id, _, _)| id);
    if particles.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(invalid("writeback would duplicate a global particle ID"));
    }
    let mut words = Vec::with_capacity(particles.len() * WORDS_PER_PARTICLE);
    for stream in 0..STRIDES.len() {
        for &(_, tile, index) in &particles {
            words.extend_from_slice(tile.particle_stream(stream, index));
        }
    }
    Ok(Tile {
        epoch: Arc::new(()),
        ids: particles.into_iter().map(|(id, _, _)| id).collect(),
        words,
    })
}

struct Bytes<'a>(&'a [u8]);
impl Bytes<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        if self.0.len() < N {
            return Err(invalid("truncated tile archive metadata"));
        }
        let (head, tail) = self.0.split_at(N);
        self.0 = tail;
        Ok(head.try_into().unwrap())
    }
    fn float(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take()?))
    }
}

#[cfg(test)]
#[path = "mpm_soil_tiles_tests.rs"]
mod tests;
