//! Per-tile f64 position offsets with unchanged f32 soil history.

use super::super::recenter::exact_sum;
use super::*;

const MAGIC: &[u8; 8] = b"MOLLCTIL";
const RECORD_BYTES: u64 = 181;

#[derive(Clone)]
pub(super) struct CanonicalTile {
    pub(super) tile: Tile,
    positions: Vec<[f64; 3]>,
}

/// Host archive with tile-local f64 positions and exact particle/material IDs.
/// Grid origin is supplied at selection, not a physical boundary guarantee.
/// Requires exactly representable tile anchors and coordinate translations;
/// unsupported ranges reject instead of rounding. No live/durable owner is implied.
pub struct MpmSoilCanonicalTileArchive {
    pub(super) epoch: Arc<()>,
    pub(super) metadata: MpmSoilCheckpoint,
    pub(super) material_ids: Vec<SoilMaterialId>,
    pub(super) layout: SoilTileLayout,
    pub(super) count: u32,
    pub(super) tiles: BTreeMap<SoilTileCoord, CanonicalTile>,
}

/// Bounded archive snapshot assembled in an explicit solver frame/grid origin.
/// Retain this token for writeback; its framed view can restore a detached solver.
pub struct MpmSoilCanonicalSelection {
    pub(super) epoch: Arc<()>,
    pub(super) tiles: Vec<SoilTileCoord>,
    pub(super) framed: MpmSoilFrameSelection,
}

impl MpmSoilCanonicalSelection {
    pub fn framed_selection(&self) -> &MpmSoilFrameSelection {
        &self.framed
    }
    pub fn tiles(&self) -> impl ExactSizeIterator<Item = SoilTileCoord> + '_ {
        self.tiles.iter().copied()
    }
}

fn anchor(layout: SoilTileLayout, key: SoilTileCoord) -> Result<[f64; 3]> {
    let mut result = [0.0; 3];
    for axis in 0..3 {
        let coordinate = key.0[axis] as f64;
        let product = coordinate * layout.edge();
        if key.0[axis].unsigned_abs() >= 1u64 << 52
            || !product.is_finite()
            || coordinate.mul_add(layout.edge(), -product) != 0.0
        {
            return Err(invalid(
                "canonical tile anchor is not exactly representable",
            ));
        }
        result[axis] = exact_sum(layout.origin()[axis], product)?;
    }
    Ok(result)
}

fn translated(value: f64, delta: f64) -> Result<f64> {
    if delta == 0.0 {
        Ok(value)
    } else {
        exact_sum(value, delta)
    }
}

impl MpmSoilFrameCheckpoint {
    /// Canonicalize each occupied tile without forming absolute particle positions.
    /// All non-position streams and position padding retain their original bits.
    /// Archives exclude the client's frame/window; select_window supplies them on read.
    pub fn partition_canonical_tiles(
        &self,
        layout: SoilTileLayout,
        ids: &[SoilParticleId],
        material_ids: &[SoilMaterialId],
    ) -> Result<MpmSoilCanonicalTileArchive> {
        let partition = self.soil().partition_particle_tiles_in_frame(
            layout,
            ids,
            material_ids,
            Some(self.world_origin()),
        )?;
        let mut metadata = MpmSoilCheckpoint {
            cfg: self.soil().cfg,
            grid_window_cells: [0; 3],
            materials: self.soil().materials.clone(),
            count: 0,
            words: Vec::new(),
        };
        metadata.cfg.grid_origin = [0.0; 3];
        let mut tiles = BTreeMap::new();
        for (key, mut tile) in partition.tiles {
            let base = anchor(layout, key)?;
            let mut delta = [0.0; 3];
            for axis in 0..3 {
                delta[axis] = exact_sum(self.world_origin()[axis], -base[axis])?;
            }
            let mut positions = Vec::with_capacity(tile.ids.len());
            for particle in tile.words[..tile.ids.len() * 4].chunks_exact_mut(4) {
                let mut position = [0.0; 3];
                for axis in 0..3 {
                    let word = particle[axis];
                    position[axis] = translated(f64::from(f32::from_bits(word)), delta[axis])?;
                    if position[axis] < 0.0 || position[axis] >= layout.edge() {
                        return Err(invalid("canonical position is outside its tile"));
                    }
                    particle[axis] = if word == (-0.0f32).to_bits() { word } else { 0 };
                }
                positions.push(position);
            }
            tiles.insert(key, CanonicalTile { tile, positions });
        }
        Ok(MpmSoilCanonicalTileArchive {
            epoch: Arc::new(()),
            metadata,
            material_ids: material_ids.to_vec(),
            layout,
            count: self.soil().count,
            tiles,
        })
    }
}

impl MpmSoilCanonicalTileArchive {
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
        self.tiles.get(&tile).map(|t| t.tile.ids.len())
    }

    /// Assemble requested world tiles in a chosen frame, sorted by global ID.
    /// Rejects unknown tiles, capacity excess, nonfinite frames and lossy f32
    /// conversion. This does not validate changed grid boundaries or future physics.
    pub fn select(
        &self,
        tiles: &[SoilTileCoord],
        max_particles: u32,
        world_origin: [f64; 3],
        grid_origin: [f32; 3],
    ) -> Result<MpmSoilCanonicalSelection> {
        self.select_window(tiles, max_particles, world_origin, grid_origin, [0; 3])
    }

    /// Assembles a client frame and absolute cell window without changing global storage.
    /// Checks grid ranges; live stencil support and changed-boundary admission are separate.
    pub fn select_window(
        &self,
        tiles: &[SoilTileCoord],
        max_particles: u32,
        world_origin: [f64; 3],
        grid_origin: [f32; 3],
        cells: [i32; 3],
    ) -> Result<MpmSoilCanonicalSelection> {
        if cells != [0; 3] {
            crate::soil_candidates_gpu::validate_node_window(
                grid_origin,
                self.metadata.cfg.cell_size,
                self.metadata.cfg.grid_size,
                cells,
            )?;
        }
        if world_origin.iter().any(|v| !v.is_finite()) || grid_origin.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("nonfinite canonical selection frame"));
        }
        let keys: Vec<_> = tiles
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut count = 0u64;
        for key in &keys {
            count += self
                .tiles
                .get(key)
                .ok_or_else(|| invalid("unknown canonical tile"))?
                .tile
                .ids
                .len() as u64;
            if count > u64::from(max_particles) {
                return Err(invalid("canonical selection exceeds capacity"));
            }
        }
        let mut particles = Vec::with_capacity(count as usize);
        for key in &keys {
            let tile = &self.tiles[key];
            let base = anchor(self.layout, *key)?;
            let mut delta = [0.0; 3];
            for axis in 0..3 {
                delta[axis] = exact_sum(base[axis], -world_origin[axis])?;
            }
            for (index, &id) in tile.tile.ids.iter().enumerate() {
                particles.push((id, tile, index, delta));
            }
        }
        particles.sort_unstable_by_key(|&(id, _, _, _)| id);
        let mut checkpoint = self.metadata.clone();
        checkpoint.cfg.grid_origin = grid_origin;
        checkpoint.grid_window_cells = cells;
        checkpoint.count = count as u32;
        checkpoint
            .words
            .reserve(count as usize * WORDS_PER_PARTICLE);
        for stream in 0..STRIDES.len() {
            for &(_, tile, index, delta) in &particles {
                let record = tile.tile.particle_stream(stream, index);
                if stream == 0 {
                    for axis in 0..3 {
                        let local = translated(tile.positions[index][axis], delta[axis])?;
                        let rounded = local as f32;
                        if !rounded.is_finite() || f64::from(rounded) != local {
                            return Err(invalid(
                                "canonical selection would round a particle coordinate",
                            ));
                        }
                        checkpoint.words.push(if local == 0.0 {
                            record[axis]
                        } else {
                            rounded.to_bits()
                        });
                    }
                    checkpoint.words.push(record[3]);
                } else {
                    checkpoint.words.extend_from_slice(record);
                }
            }
        }
        checkpoint.validate()?;
        let framed = MpmSoilFrameSelection {
            origin: world_origin,
            selection: MpmSoilSelection {
                epoch: self.epoch.clone(),
                tiles: keys.clone(),
                checkpoint,
                ids: particles.into_iter().map(|(id, _, _, _)| id).collect(),
                material_ids: self.material_ids.clone(),
            },
        };
        Ok(MpmSoilCanonicalSelection {
            epoch: self.epoch.clone(),
            tiles: keys,
            framed,
        })
    }

    /// Atomically canonicalize a same-frame selected update and merge destinations.
    /// All earlier selections become stale. Mass, reference volume, material IDs,
    /// configuration and selected count remain invariant; unloaded tiles are retained.
    pub fn writeback(
        &mut self,
        selection: &MpmSoilCanonicalSelection,
        updated: &MpmSoilFrameCheckpoint,
    ) -> Result<()> {
        self.validate_writeback(selection, updated)?;
        self.commit_writeback(selection, updated)
    }

    pub(super) fn validate_writeback(
        &self,
        selection: &MpmSoilCanonicalSelection,
        updated: &MpmSoilFrameCheckpoint,
    ) -> Result<()> {
        let before = selection.framed.soil();
        let mut cfg = updated.soil().cfg;
        cfg.max_particles = before.cfg.max_particles;
        cfg.max_rigid = before.cfg.max_rigid;
        if !Arc::ptr_eq(&selection.epoch, &self.epoch)
            || updated.world_origin().map(f64::to_bits)
                != selection.framed.world_origin().map(f64::to_bits)
            || updated.soil().count != before.count
            || updated.soil().grid_window_cells != before.grid_window_cells
            || config_words(cfg) != config_words(before.cfg)
            || updated.soil().materials != before.materials
        {
            return Err(invalid(
                "stale/foreign canonical selection or changed frame/configuration",
            ));
        }
        updated.soil().validate()?;
        if [4, 5, 7]
            .iter()
            .any(|&s| updated.soil().stream(s) != before.stream(s))
        {
            return Err(invalid(
                "canonical writeback changes mass, reference volume or material identity",
            ));
        }
        Ok(())
    }

    fn commit_writeback(
        &mut self,
        selection: &MpmSoilCanonicalSelection,
        updated: &MpmSoilFrameCheckpoint,
    ) -> Result<()> {
        let mut replacement = updated
            .partition_canonical_tiles(
                self.layout,
                selection.framed.particle_ids(),
                &self.material_ids,
            )?
            .tiles;
        for (key, tile) in &mut replacement {
            if selection.tiles.binary_search(key).is_ok() {
                continue;
            }
            if let Some(untouched) = self.tiles.get(key) {
                let merged = merge_tiles(&untouched.tile, &tile.tile)?;
                let positions = merged
                    .ids
                    .iter()
                    .map(|id| {
                        if let Ok(index) = untouched.tile.ids.binary_search(id) {
                            untouched.positions[index]
                        } else {
                            tile.positions[tile
                                .tile
                                .ids
                                .binary_search(id)
                                .expect("merged canonical ID")]
                        }
                    })
                    .collect();
                *tile = CanonicalTile {
                    tile: merged,
                    positions,
                };
            }
        }
        for key in &selection.tiles {
            self.tiles.remove(key);
        }
        self.tiles.extend(replacement);
        self.epoch = Arc::new(());
        Ok(())
    }

    /// Versioned canonical positions/history container, independent of a live frame.
    /// Filesystem durability and live-world publication remain caller-owned.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.encoded_tiles(&self.tiles().collect::<Vec<_>>())
            .expect("validated canonical tile keys")
    }

    pub(super) fn copy_for_patch(&self) -> Self {
        Self {
            epoch: self.epoch.clone(),
            metadata: self.metadata.clone(),
            material_ids: self.material_ids.clone(),
            layout: self.layout,
            count: self.count,
            tiles: self.tiles.clone(),
        }
    }

    pub(super) fn encoded_tiles(&self, keys: &[SoilTileCoord]) -> Result<Vec<u8>> {
        if keys.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid("canonical encoding requires sorted unique keys"));
        }
        let count = keys.iter().try_fold(0u32, |count, key| {
            let tile = self
                .tiles
                .get(key)
                .ok_or_else(|| invalid("unknown canonical encoding tile"))?;
            count
                .checked_add(tile.tile.ids.len() as u32)
                .ok_or_else(|| invalid("canonical encoding count overflow"))
        })?;
        let metadata = self.metadata.to_bytes();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&[0; 32]);
        for value in self.layout.origin().into_iter().chain([self.layout.edge()]) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [
            count,
            keys.len() as u32,
            self.material_ids.len() as u32,
            metadata.len() as u32,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&metadata);
        for id in &self.material_ids {
            bytes.extend_from_slice(&id.0.to_le_bytes());
        }
        for key in keys {
            let tile = &self.tiles[key];
            for axis in key.0 {
                bytes.extend_from_slice(&axis.to_le_bytes());
            }
            bytes.extend_from_slice(&(tile.tile.ids.len() as u32).to_le_bytes());
            for (index, id) in tile.tile.ids.iter().enumerate() {
                bytes.extend_from_slice(&id.0.to_le_bytes());
                for position in tile.positions[index] {
                    bytes.extend_from_slice(&position.to_le_bytes());
                }
                let flags = (0..3).fold(0u8, |flags, a| {
                    flags | (((tile.tile.particle_stream(0, index)[a] >> 31) as u8) << a)
                });
                bytes.push(flags);
                for stream in 0..STRIDES.len() {
                    let start = if stream == 0 { 3 } else { 0 };
                    for word in &tile.tile.particle_stream(stream, index)[start..] {
                        bytes.extend_from_slice(&word.to_le_bytes());
                    }
                }
            }
        }
        let hash = blake3::hash(&bytes[HEADER_BYTES..]);
        bytes[12..HEADER_BYTES].copy_from_slice(hash.as_bytes());
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8], max_particles: u32) -> Result<Self> {
        if bytes.len() < HEADER_BYTES + 48
            || bytes.len() as u64 > 1024 + u64::from(max_particles) * (RECORD_BYTES + 28)
            || &bytes[..8] != MAGIC
            || bytes[8..12] != 1u32.to_le_bytes()
            || bytes[12..HEADER_BYTES] != *blake3::hash(&bytes[HEADER_BYTES..]).as_bytes()
        {
            return Err(invalid(
                "invalid canonical archive header, digest or allocation bound",
            ));
        }
        let mut reader = Bytes(&bytes[HEADER_BYTES..]);
        let origin = [reader.float()?, reader.float()?, reader.float()?];
        let layout =
            SoilTileLayout::new(origin, reader.float()?).map_err(|e| invalid(&e.to_string()))?;
        let count = u32::from_le_bytes(reader.take()?);
        let tile_count = u32::from_le_bytes(reader.take()?);
        let rows = u32::from_le_bytes(reader.take()?);
        let length = u32::from_le_bytes(reader.take()?) as usize;
        if count > max_particles
            || tile_count > count
            || rows == 0
            || rows as usize > MAX_MPM_MATERIALS
            || length > reader.0.len()
        {
            return Err(invalid("invalid canonical archive counts"));
        }
        let metadata = MpmSoilCheckpoint::from_bytes(&reader.0[..length], 0)?;
        metadata.require_origin_only_grid()?;
        reader.0 = &reader.0[length..];
        if metadata.count != 0
            || metadata.cfg.grid_origin.map(f32::to_bits) != [0; 3]
            || rows as usize != metadata.materials.len().max(1)
        {
            return Err(invalid("invalid canonical archive metadata"));
        }
        let material_ids = (0..rows)
            .map(|_| Ok(SoilMaterialId(u64::from_le_bytes(reader.take()?))))
            .collect::<Result<Vec<_>>>()?;
        if material_ids.iter().collect::<BTreeSet<_>>().len() != material_ids.len() {
            return Err(invalid("duplicate canonical material ID"));
        }
        let mut ids = BTreeSet::new();
        let mut tiles = BTreeMap::new();
        let mut total = 0u64;
        for _ in 0..tile_count {
            let key = SoilTileCoord([
                i64::from_le_bytes(reader.take()?),
                i64::from_le_bytes(reader.take()?),
                i64::from_le_bytes(reader.take()?),
            ]);
            anchor(layout, key)?;
            if tiles.last_key_value().is_some_and(|(last, _)| *last >= key) {
                return Err(invalid("unsorted canonical tiles"));
            }
            let n = u32::from_le_bytes(reader.take()?);
            total += u64::from(n);
            if n == 0
                || total > u64::from(count)
                || u64::from(n) * RECORD_BYTES > reader.0.len() as u64
            {
                return Err(invalid("canonical tile count or length mismatch"));
            }
            let mut tile_ids = Vec::with_capacity(n as usize);
            let mut positions = Vec::with_capacity(n as usize);
            let mut records = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let id = SoilParticleId(u64::from_le_bytes(reader.take()?));
                if !ids.insert(id) || tile_ids.last().is_some_and(|last| *last >= id) {
                    return Err(invalid("duplicate or unsorted canonical ID"));
                }
                tile_ids.push(id);
                let position = [reader.float()?, reader.float()?, reader.float()?];
                if position
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0.0 || *v >= layout.edge())
                {
                    return Err(invalid("invalid canonical tile-local position"));
                }
                positions.push(position);
                let flags = reader.take::<1>()?[0];
                if flags & !7 != 0 {
                    return Err(invalid("invalid canonical zero flags"));
                }
                let mut words = [0u32; WORDS_PER_PARTICLE];
                for (a, word) in words[..3].iter_mut().enumerate() {
                    *word = u32::from((flags >> a) & 1) << 31;
                }
                for word in &mut words[3..] {
                    *word = u32::from_le_bytes(reader.take()?);
                }
                records.push(words);
            }
            let mut checkpoint = metadata.clone();
            checkpoint.count = n;
            let mut start = 0;
            for stride in STRIDES {
                for record in &records {
                    checkpoint
                        .words
                        .extend_from_slice(&record[start..start + stride]);
                }
                start += stride;
            }
            checkpoint.validate()?;
            tiles.insert(
                key,
                CanonicalTile {
                    tile: Tile {
                        epoch: Arc::new(()),
                        ids: tile_ids,
                        words: checkpoint.words,
                    },
                    positions,
                },
            );
        }
        if total != u64::from(count) || !reader.0.is_empty() {
            return Err(invalid("canonical archive count or trailing bytes"));
        }
        Ok(Self {
            epoch: Arc::new(()),
            metadata,
            material_ids,
            layout,
            count,
            tiles,
        })
    }
}

#[cfg(test)]
#[path = "mpm_soil_canonical_tiles_tests.rs"]
mod tests;
