//! Disk-owner storage dispatch retaining family and selected-frame provenance.

use super::*;

pub(super) struct SelectionAdmission {
    pub entries: Vec<(SoilTileCoord, u32, bool)>,
    pub max_particles: u32,
    pub limits: MpmSoilStoreReadLimits,
}

impl SelectionAdmission {
    fn validate(
        &self,
        mut stored_count: impl FnMut(SoilTileCoord) -> Result<Option<u32>>,
        tile_bytes: u64,
        particle_bytes: u64,
    ) -> Result<()> {
        let mut particles = 0u64;
        let mut tiles = 0u64;
        for &(tile, active, replaces) in &self.entries {
            let count = u64::from(active)
                + if replaces {
                    0
                } else {
                    u64::from(stored_count(tile)?.unwrap_or(0))
                };
            particles += count;
            tiles += u64::from(count != 0);
            if particles > u64::from(self.max_particles.min(self.limits.particles))
                || tiles > u64::from(self.limits.tiles)
                || tiles * tile_bytes + particles * particle_bytes > self.limits.encoded_bytes
            {
                return Err(invalid(
                    "next disk selection exceeds solver or I/O capacity",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_selection_admission_counts_replacements_migration_and_family_bytes() {
        let old = SoilTileCoord([0; 3]);
        let merged = SoilTileCoord([1; 3]);
        let absent = SoilTileCoord([2; 3]);
        let mut admission = SelectionAdmission {
            entries: vec![(old, 2, true), (merged, 3, false), (absent, 0, false)],
            max_particles: 9,
            limits: MpmSoilStoreReadLimits {
                tiles: 2,
                particles: 9,
                encoded_bytes: 2 * 1024 + 9 * 168,
            },
        };
        let count = |tile| {
            assert_ne!(tile, old, "replaced tile must not read stale counts");
            Ok((tile == merged).then_some(4))
        };
        admission.validate(count, 1024, 168).unwrap();
        assert!(admission.validate(count, 1052, 181).is_err());
        admission.limits.encoded_bytes = 2 * 1052 + 9 * 181;
        admission.validate(count, 1052, 181).unwrap();
        admission.max_particles = 8;
        assert!(admission.validate(count, 1052, 181).is_err());
        admission.max_particles = 9;
        admission.limits.particles = 8;
        assert!(admission.validate(count, 1052, 181).is_err());
        admission.limits.particles = 9;
        admission.limits.tiles = 1;
        assert!(admission.validate(count, 1052, 181).is_err());
        admission.limits.tiles = 2;
        assert!(
            admission
                .validate(|_| Err(invalid("manifest read failed")), 1052, 181)
                .is_err()
        );
    }
}

pub(super) enum Store {
    Raw(MpmSoilStore),
    Canonical {
        store: MpmSoilCanonicalStore,
        origin: [f64; 3],
        grid: [f32; 3],
        cells: [i32; 3],
    },
}

pub(super) enum Read {
    Raw(MpmSoilStoreRead),
    Canonical {
        read: MpmSoilCanonicalStoreRead,
        selection: MpmSoilCanonicalSelection,
    },
}

impl Read {
    pub(super) fn selection(&self) -> &MpmSoilSelection {
        match self {
            Self::Raw(read) => read.selection(),
            Self::Canonical { selection, .. } => selection.framed_selection().local_selection(),
        }
    }
    pub(super) fn stats(&self) -> MpmSoilStoreReadStats {
        match self {
            Self::Raw(read) => read.stats(),
            Self::Canonical { read, .. } => read.stats(),
        }
    }
    pub(super) fn tiles(&self) -> impl Iterator<Item = SoilTileCoord> + '_ {
        let (raw, canonical) = match self {
            Self::Raw(read) => (Some(read.tiles()), None),
            Self::Canonical { selection, .. } => (None, Some(selection.tiles())),
        };
        raw.into_iter()
            .flatten()
            .chain(canonical.into_iter().flatten())
    }
}

impl Store {
    pub(super) fn storage_stats(&self) -> Result<MpmSoilStoreStorageStats> {
        match self {
            Self::Raw(store) => store.storage_stats(),
            Self::Canonical { store, .. } => store.storage_stats(),
        }
    }
    pub(super) fn layout(&self) -> Result<Option<SoilTileLayout>> {
        match self {
            Self::Raw(store) => store.layout(),
            Self::Canonical { store, .. } => store.layout(),
        }
    }
    pub(super) fn total_particle_count(&self) -> Result<Option<u32>> {
        match self {
            Self::Raw(store) => store.total_particle_count(),
            Self::Canonical { store, .. } => store.total_particle_count(),
        }
    }
    pub(super) fn validate_read(&self, read: &Read) -> Result<()> {
        match (self, read) {
            (Self::Raw(store), Read::Raw(read)) => store.validate_read(read),
            (
                Self::Canonical {
                    store,
                    origin,
                    grid,
                    cells,
                },
                Read::Canonical { read, selection },
            ) => {
                store.validate_read(read)?;
                if selection
                    .framed_selection()
                    .world_origin()
                    .map(f64::to_bits)
                    != origin.map(f64::to_bits)
                    || selection
                        .framed_selection()
                        .soil()
                        .config()
                        .grid_origin
                        .map(f32::to_bits)
                        != grid.map(f32::to_bits)
                    || selection.framed_selection().soil().grid_window_cells() != *cells
                {
                    return Err(invalid("canonical disk selection frame mismatch"));
                }
                Ok(())
            }
            _ => Err(invalid("disk storage/read family mismatch")),
        }
    }
    pub(super) fn read_tiles(
        &self,
        keys: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
    ) -> Result<Option<Read>> {
        match self {
            Self::Raw(store) => store
                .read_tiles(keys, limits)
                .map(|read| read.map(Read::Raw)),
            Self::Canonical {
                store,
                origin,
                grid,
                cells,
            } => store
                .read_tiles(keys, limits)?
                .map(|read| {
                    let selection = read.select_window(*origin, *grid, *cells)?;
                    Ok(Read::Canonical { read, selection })
                })
                .transpose(),
        }
    }
    pub(super) fn begin_read(
        self,
        keys: &[SoilTileCoord],
        request_tiles: u32,
        limits: MpmSoilStoreReadLimits,
    ) -> ReadTask {
        match self {
            Self::Raw(store) => {
                ReadTask::Raw(store.begin_read_existing(keys, request_tiles, limits))
            }
            Self::Canonical {
                store,
                origin,
                grid,
                cells,
            } => ReadTask::Canonical {
                task: store.begin_read_existing_window(
                    keys,
                    request_tiles,
                    limits,
                    origin,
                    grid,
                    cells,
                ),
                origin,
                grid,
                cells,
            },
        }
    }
    pub(super) fn begin_writeback(
        self,
        read: Read,
        updated: MpmSoilCheckpoint,
        limits: MpmSoilStoreReadLimits,
        admission: SelectionAdmission,
    ) -> WriteTask {
        match (self, read) {
            (Self::Raw(store), Read::Raw(read)) => WriteTask::Raw(store.begin_writeback_checked(
                read,
                updated,
                limits,
                move |store| admission.validate(|tile| store.tile_particle_count(tile), 1024, 168),
            )),
            (
                Self::Canonical {
                    store,
                    origin,
                    grid,
                    cells,
                },
                Read::Canonical { read, selection },
            ) => {
                let updated = MpmSoilFrameCheckpoint {
                    origin,
                    soil: updated,
                };
                WriteTask::Canonical {
                    task: store.begin_writeback_checked(
                        read,
                        selection,
                        updated,
                        limits,
                        move |store| {
                            admission.validate(|tile| store.tile_particle_count(tile), 1052, 181)
                        },
                    ),
                    origin,
                    grid,
                    cells,
                }
            }
            (store, read) => WriteTask::Ready(Some(WriteCompletion {
                store,
                read,
                updated,
                result: Err(invalid("disk write family mismatch")),
            })),
        }
    }
    #[cfg(test)]
    pub(super) fn tile_particle_count(&self, tile: SoilTileCoord) -> Result<Option<u32>> {
        match self {
            Self::Raw(store) => store.tile_particle_count(tile),
            Self::Canonical { store, .. } => store.tile_particle_count(tile),
        }
    }
    #[cfg(test)]
    pub(super) fn generation(&self) -> Option<u64> {
        match self {
            Self::Raw(store) => store.generation(),
            Self::Canonical { store, .. } => store.generation(),
        }
    }
    #[cfg(test)]
    pub(super) fn raw_mut(&mut self) -> &mut MpmSoilStore {
        match self {
            Self::Raw(store) => store,
            _ => panic!("expected raw store test fixture"),
        }
    }
}

pub(super) struct ReadCompletion {
    pub store: Store,
    pub result: Result<Option<Read>>,
}
pub(super) enum ReadTask {
    Raw(MpmSoilStoreReadTask),
    Canonical {
        task: MpmSoilCanonicalStoreReadTask,
        origin: [f64; 3],
        grid: [f32; 3],
        cells: [i32; 3],
    },
}
impl ReadTask {
    pub(super) fn poll(&mut self) -> Result<Option<ReadCompletion>> {
        match self {
            Self::Raw(task) => task.poll().map(|completion| {
                completion.map(|c| ReadCompletion {
                    store: Store::Raw(c.store),
                    result: c.result.map(|read| read.map(Read::Raw)),
                })
            }),
            Self::Canonical {
                task,
                origin,
                grid,
                cells,
            } => task.poll().map(|completion| {
                completion.map(|c| ReadCompletion {
                    store: Store::Canonical {
                        store: c.store,
                        origin: *origin,
                        grid: *grid,
                        cells: *cells,
                    },
                    result: c.result.map(|read| {
                        read.map(|(read, selection)| Read::Canonical { read, selection })
                    }),
                })
            }),
        }
    }
}

pub(super) struct WriteCompletion {
    pub store: Store,
    pub read: Read,
    pub updated: MpmSoilCheckpoint,
    pub result: Result<MpmSoilStoreStats>,
}
pub(super) enum WriteTask {
    Raw(MpmSoilStoreWritebackTask),
    Canonical {
        task: MpmSoilCanonicalStoreWritebackTask,
        origin: [f64; 3],
        grid: [f32; 3],
        cells: [i32; 3],
    },
    Ready(Option<WriteCompletion>),
}
impl WriteTask {
    pub(super) fn poll(&mut self) -> Result<Option<WriteCompletion>> {
        match self {
            Self::Raw(task) => task.poll().map(|completion| {
                completion.map(|c| WriteCompletion {
                    store: Store::Raw(c.store),
                    read: Read::Raw(c.read),
                    updated: c.updated,
                    result: c.result,
                })
            }),
            Self::Canonical {
                task,
                origin,
                grid,
                cells,
            } => task.poll().map(|completion| {
                completion.map(|c| WriteCompletion {
                    store: Store::Canonical {
                        store: c.store,
                        origin: *origin,
                        grid: *grid,
                        cells: *cells,
                    },
                    read: Read::Canonical {
                        read: c.read,
                        selection: c.selection,
                    },
                    updated: c.updated.soil,
                    result: c.result,
                })
            }),
            Self::Ready(completion) => completion
                .take()
                .map(Some)
                .ok_or_else(|| invalid("disk write task already consumed")),
        }
    }
}
