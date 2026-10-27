//! Bounded selected writeback preserving unloaded manifest entries.

use super::*;

impl MpmSoilStore {
    /// Blocking, atomic writeback of a current read in its global-ID slot order.
    /// Preserves count, mass, reference volume, material identity and unloaded tiles.
    /// Movement merges existing destination payloads. Limits cover the source and
    /// destination tile union and its particles/encoded bound, excluding manifest
    /// and allocator overhead. Integrity checks cover affected payloads only.
    /// Preflight/read errors leave the handle healthy; publication errors poison
    /// it and require reopen. Changed generations invalidate every prior read.
    pub fn writeback(
        &mut self,
        read: &MpmSoilStoreRead,
        updated: &MpmSoilCheckpoint,
        limits: MpmSoilStoreReadLimits,
    ) -> Result<MpmSoilStoreStats> {
        self.writeback_with(read, updated, limits, |_| Ok(()))
    }

    pub(super) fn writeback_with(
        &mut self,
        read: &MpmSoilStoreRead,
        updated: &MpmSoilCheckpoint,
        limits: MpmSoilStoreReadLimits,
        mut barrier: impl FnMut(Stage) -> Result<()>,
    ) -> Result<MpmSoilStoreStats> {
        self.validate_writeback_request(read, updated, limits)?;
        let manifest = self.manifest.as_ref().expect("validated read generation");
        let source = read.selection();
        let mut affected: BTreeSet<_> = source.tiles.iter().copied().collect();
        let (metadata_archive, origin) = decode_archive(&manifest.metadata, 0, self.framed)?;
        let layout = metadata_archive.layout;
        updated.validate()?;
        for position in updated.stream(0).chunks_exact(4) {
            let world = std::array::from_fn(|axis| f32::from_bits(position[axis]) as f64);
            let tile = match origin {
                Some(origin) => layout.tile_at_local(world, origin),
                None => layout.tile_at(world),
            }
            .map_err(|e| invalid(&e.to_string()))?;
            affected.insert(tile);
            if affected.len() as u64 > u64::from(limits.tiles) {
                return Err(invalid("writeback affected tiles exceed bound"));
            }
        }
        let mut particles = u64::from(source.checkpoint.count);
        for key in &affected {
            if source.tiles.binary_search(key).is_err() {
                particles += manifest
                    .get(*key)?
                    .map_or(0, |entry| u64::from(entry.count));
            }
        }
        let encoded = (affected.len() as u64)
            .checked_mul(MAX_METADATA)
            .and_then(|bytes| bytes.checked_add(particles * 168))
            .ok_or_else(|| invalid("writeback encoded bound overflow"))?;
        if affected.len() as u64 > u64::from(limits.tiles)
            || particles > u64::from(limits.particles)
            || encoded > limits.encoded_bytes
        {
            return Err(invalid(
                "writeback affected particles or encoded bytes exceed bound",
            ));
        }
        let mut archive = source.checkpoint.partition_tiles_in_frame(
            layout,
            &source.ids,
            &source.material_ids,
            origin,
        )?;
        let selection = archive.select(&source.tiles, limits.particles)?;
        archive.validate_writeback(&selection, updated)?;
        if metadata_in_frame(&archive, origin) != manifest.metadata {
            return Err(invalid("writeback metadata differs from committed store"));
        }
        let mut ids: BTreeSet<_> = source.ids.iter().copied().collect();
        for key in &affected {
            if source.tiles.binary_search(key).is_ok() {
                continue;
            }
            if let Some(entry) = manifest.get(*key)? {
                let (tile, _) = self.read_tile(manifest, *key, &entry)?;
                if tile.ids.iter().any(|&id| !ids.insert(id)) {
                    return Err(invalid(
                        "duplicate global particle ID in writeback destination",
                    ));
                }
                archive.count += entry.count;
                archive.tiles.insert(*key, tile);
            }
        }
        archive
            .prepare_writeback_in_frame(&selection, updated, origin)?
            .commit();
        let retained = manifest.retained(&affected)?.0;
        if (retained as u64) + (archive.tiles.len() as u64) > u64::from(self.limits.tiles) {
            return Err(invalid("writeback exceeds total store tile limit"));
        }
        let result = self.save_inner(&archive, Some(&affected), origin, &mut barrier);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    pub(super) fn validate_writeback_request(
        &self,
        read: &MpmSoilStoreRead,
        updated: &MpmSoilCheckpoint,
        limits: MpmSoilStoreReadLimits,
    ) -> Result<()> {
        self.validate_read(read)?;
        let source = read.selection();
        if source.checkpoint.count > limits.particles
            || source.tiles.len() as u64 > u64::from(limits.tiles)
            || updated.count != source.checkpoint.count
            || source.tiles.len() as u64 * MAX_METADATA + u64::from(source.checkpoint.count) * 168
                > limits.encoded_bytes
        {
            return Err(invalid(
                "writeback selected particle count exceeds bound or changed",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "mpm_soil_store_writeback_tests.rs"]
mod tests;
