//! Frame-preserving wrappers over exclusive single-flight store workers.

use super::*;

/// Returns the framed owner on success, startup failure or caught worker failure.
/// An unwinding worker panic poisons the store; reopen before further use.
pub struct MpmSoilFrameStoreReadCompletion {
    pub store: MpmSoilFrameStore,
    pub result: Result<Option<MpmSoilFrameStoreRead>>,
}

/// One bounded read. Drop neither joins nor cancels the worker; its lock remains
/// held until I/O completes. Bounds constrain payloads, not filesystem latency.
#[must_use]
pub struct MpmSoilFrameStoreReadTask {
    inner: MpmSoilStoreReadTask,
    origin: Option<[f64; 3]>,
}

impl MpmSoilFrameStoreReadTask {
    /// Nonblocking, single-consumption poll without payload copies or validation.
    /// A disconnected worker is terminal and cannot reconstruct the store owner.
    pub fn poll(&mut self) -> Result<Option<MpmSoilFrameStoreReadCompletion>> {
        self.inner.poll().map(|completion| {
            completion.map(|completion| MpmSoilFrameStoreReadCompletion {
                store: MpmSoilFrameStore {
                    inner: completion.store,
                },
                result: completion.result.and_then(|read| {
                    read.map(|read| {
                        Ok(MpmSoilFrameStoreRead {
                            read,
                            origin: self
                                .origin
                                .ok_or_else(|| invalid("framed read lost frame"))?,
                        })
                    })
                    .transpose()
                }),
            })
        })
    }
}

/// Returns original framed inputs even on failure. Successful changed publication
/// stales the read; publication errors or caught panics require reopening the store.
pub struct MpmSoilFrameStoreWritebackCompletion {
    pub store: MpmSoilFrameStore,
    pub read: MpmSoilFrameStoreRead,
    pub updated: MpmSoilFrameCheckpoint,
    pub result: Result<MpmSoilStoreStats>,
}

/// One owned write. Drop does not cancel publication or release the worker's lock
/// early; HEAD can change without the caller observing completion.
#[must_use]
pub struct MpmSoilFrameStoreWritebackTask {
    inner: MpmSoilStoreWritebackTask,
    read_origin: [f64; 3],
    updated_origin: [f64; 3],
}

impl MpmSoilFrameStoreWritebackTask {
    /// Nonblocking, single-consumption poll; returned payload buffers are moved.
    /// Disconnection does not establish whether publication took place.
    pub fn poll(&mut self) -> Result<Option<MpmSoilFrameStoreWritebackCompletion>> {
        self.inner.poll().map(|completion| {
            completion.map(|completion| MpmSoilFrameStoreWritebackCompletion {
                store: MpmSoilFrameStore {
                    inner: completion.store,
                },
                read: MpmSoilFrameStoreRead {
                    read: completion.read,
                    origin: self.read_origin,
                },
                updated: MpmSoilFrameCheckpoint {
                    soil: completion.updated,
                    origin: self.updated_origin,
                },
                result: completion.result,
            })
        })
    }
}

impl MpmSoilFrameStore {
    /// Consume the exclusive owner into a bounded selected read. Caller-side
    /// frame/manifest checks do not read tile payloads or activate GPU membership.
    pub fn begin_read(
        self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
    ) -> MpmSoilFrameStoreReadTask {
        self.start_read(|store| store.begin_read(tiles, limits))
    }

    fn start_read(
        self,
        start: impl FnOnce(MpmSoilStore) -> MpmSoilStoreReadTask,
    ) -> MpmSoilFrameStoreReadTask {
        match self.world_origin() {
            Ok(origin) => MpmSoilFrameStoreReadTask {
                inner: start(self.inner),
                origin,
            },
            Err(error) => MpmSoilFrameStoreReadTask {
                inner: MpmSoilStoreReadTask::ready(self.inner, Err(error)),
                origin: None,
            },
        }
    }

    /// Consume owner and framed inputs into same-frame selected writeback.
    /// Provenance/frame/count/bound checks precede spawning. Payload validation,
    /// destination reads and atomic publication run on the worker, with no fallback.
    pub fn begin_writeback(
        self,
        read: MpmSoilFrameStoreRead,
        updated: MpmSoilFrameCheckpoint,
        limits: MpmSoilStoreReadLimits,
    ) -> MpmSoilFrameStoreWritebackTask {
        self.start_writeback(read, updated, |store, read, updated| {
            store.begin_writeback(read, updated, limits)
        })
    }

    fn start_writeback(
        self,
        read: MpmSoilFrameStoreRead,
        updated: MpmSoilFrameCheckpoint,
        start: impl FnOnce(
            MpmSoilStore,
            MpmSoilStoreRead,
            MpmSoilCheckpoint,
        ) -> MpmSoilStoreWritebackTask,
    ) -> MpmSoilFrameStoreWritebackTask {
        let validation = self.validate_read(&read).and_then(|()| {
            if updated.origin.map(f64::to_bits) != read.origin.map(f64::to_bits) {
                return Err(invalid("framed store writeback changes world origin"));
            }
            Ok(())
        });
        MpmSoilFrameStoreWritebackTask {
            read_origin: read.origin,
            updated_origin: updated.origin,
            inner: match validation {
                Ok(()) => start(self.inner, read.read, updated.soil),
                Err(error) => MpmSoilStoreWritebackTask::ready(
                    (self.inner, read.read, updated.soil),
                    Err(error),
                ),
            },
        }
    }
}

#[cfg(test)]
#[path = "mpm_soil_framed_store_tasks_tests.rs"]
pub(super) mod tests;
