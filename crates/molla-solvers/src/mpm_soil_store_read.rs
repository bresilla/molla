//! Single-flight selected disk reads with exclusive store ownership.

use super::*;
use std::sync::mpsc::{self, Receiver, TryRecvError};

/// Returns the exclusive store even when preparation, spawning or reading fails.
/// An unwinding worker panic returns a poisoned store, requiring reopen.
pub struct MpmSoilStoreReadCompletion {
    pub store: MpmSoilStore,
    pub result: Result<Option<MpmSoilStoreRead>>,
}

enum State {
    Pending(Receiver<MpmSoilStoreReadCompletion>),
    Ready(Option<MpmSoilStoreReadCompletion>),
    Consumed,
}

/// One read and one result slot; no unbounded request queue or synchronous fallback.
/// Dropping does not join or interrupt disk I/O. The worker retains the file lock
/// until its bounded read ends; unread results and their store are then dropped.
/// Bounds limit requested data, not filesystem latency or worker stack size.
#[must_use]
pub struct MpmSoilStoreReadTask {
    state: State,
}

impl MpmSoilStoreReadTask {
    pub(super) fn ready(store: MpmSoilStore, result: Result<Option<MpmSoilStoreRead>>) -> Self {
        Self {
            state: State::Ready(Some(MpmSoilStoreReadCompletion { store, result })),
        }
    }

    /// Nonblocking; None means pending. A completion is returned once only.
    /// A disconnected worker faults the task; no valid store is reconstructed.
    pub fn poll(&mut self) -> Result<Option<MpmSoilStoreReadCompletion>> {
        let result = match &mut self.state {
            State::Ready(completion) => Ok(completion.take()),
            State::Pending(receiver) => match receiver.try_recv() {
                Ok(completion) => Ok(Some(completion)),
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => {
                    Err(invalid("soil store read worker disconnected"))
                }
            },
            State::Consumed => return Err(invalid("soil store read task already consumed")),
        };
        self.state = State::Consumed;
        result
    }
}

impl MpmSoilStore {
    /// Moves this store into one background selected read, retaining its file lock.
    /// Caller preflight bounds unique keys without manifest lookup or disk I/O.
    /// Membership, particle and encoded-byte checks run on the worker before blobs.
    /// Errors return the original store; completion does not activate GPU soil.
    pub fn begin_read(
        self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
    ) -> MpmSoilStoreReadTask {
        self.begin_read_with(
            tiles,
            limits,
            |worker| {
                std::thread::Builder::new()
                    .name("molla-soil-read".into())
                    .spawn(worker)
                    .map(drop)
            },
            MpmSoilStore::read_tiles,
        )
    }

    pub(super) fn begin_read_with<S, R>(
        self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
        spawn: S,
        read: R,
    ) -> MpmSoilStoreReadTask
    where
        S: FnOnce(Box<dyn FnOnce() + Send>) -> std::io::Result<()>,
        R: FnOnce(
                &MpmSoilStore,
                &[SoilTileCoord],
                MpmSoilStoreReadLimits,
            ) -> Result<Option<MpmSoilStoreRead>>
            + Send
            + 'static,
    {
        self.begin_read_mode_with(tiles, limits, None, spawn, read)
    }

    pub(in crate::mpm) fn begin_read_existing(
        self,
        tiles: &[SoilTileCoord],
        request_tiles: u32,
        limits: MpmSoilStoreReadLimits,
    ) -> MpmSoilStoreReadTask {
        self.begin_read_mode_with(
            tiles,
            limits,
            Some(request_tiles),
            |worker| {
                std::thread::Builder::new()
                    .name("molla-soil-read".into())
                    .spawn(worker)
                    .map(drop)
            },
            MpmSoilStore::read_tiles,
        )
    }

    fn begin_read_mode_with<S, R>(
        self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
        request_tiles: Option<u32>,
        spawn: S,
        read: R,
    ) -> MpmSoilStoreReadTask
    where
        S: FnOnce(Box<dyn FnOnce() + Send>) -> std::io::Result<()>,
        R: FnOnce(
                &MpmSoilStore,
                &[SoilTileCoord],
                MpmSoilStoreReadLimits,
            ) -> Result<Option<MpmSoilStoreRead>>
            + Send
            + 'static,
    {
        let request_limits = MpmSoilStoreReadLimits {
            tiles: request_tiles.unwrap_or(limits.tiles),
            ..limits
        };
        let keys: Vec<_> = match self.bounded_read_keys(tiles, request_limits) {
            Ok(keys) => keys.into_iter().collect(),
            Err(error) => return MpmSoilStoreReadTask::ready(self, Err(error)),
        };
        if self.manifest.is_none() {
            return MpmSoilStoreReadTask::ready(self, Ok(None));
        }
        let (input, request) = mpsc::sync_channel::<MpmSoilStore>(1);
        let (output, result) = mpsc::sync_channel(1);
        let worker = Box::new(move || {
            let Ok(mut store) = request.recv() else {
                return;
            };
            let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let keys = if request_tiles.is_some() {
                    store.existing_read_keys(&keys, limits, MAX_METADATA, 168)?
                } else {
                    store.selected_keys(&keys, limits)?;
                    keys
                };
                read(&store, &keys, limits)
            }));
            let result = read.unwrap_or_else(|_| {
                store.failed = true;
                Err(invalid("soil store read worker panicked; reopen store"))
            });
            let _ = output.send(MpmSoilStoreReadCompletion { store, result });
        });
        if let Err(error) = spawn(worker) {
            return MpmSoilStoreReadTask::ready(self, Err(io_error(error)));
        }
        if let Err(error) = input.send(self) {
            return MpmSoilStoreReadTask::ready(
                error.0,
                Err(invalid("soil store read worker exited before request")),
            );
        }
        MpmSoilStoreReadTask {
            state: State::Pending(result),
        }
    }
}

#[cfg(test)]
#[path = "mpm_soil_store_read_tests.rs"]
mod tests;
