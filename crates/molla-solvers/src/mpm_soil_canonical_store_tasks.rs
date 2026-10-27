//! Single-flight canonical I/O and frame assembly with retained request ownership.

use super::*;
use std::sync::mpsc::{self, Receiver, TryRecvError};

type Outcome<I, O> = (I, Result<O>);
enum State<I, O> {
    Pending(Receiver<Outcome<I, O>>),
    Ready(Option<Outcome<I, O>>),
    Consumed,
}
struct OwnedTask<I, O> {
    state: State<I, O>,
}

impl<I, O> OwnedTask<I, O> {
    fn ready(input: I, result: Result<O>) -> Self {
        Self {
            state: State::Ready(Some((input, result))),
        }
    }
    fn poll(&mut self) -> Result<Option<Outcome<I, O>>> {
        let result = match &mut self.state {
            State::Ready(value) => Ok(value.take()),
            State::Pending(receiver) => match receiver.try_recv() {
                Ok(value) => Ok(Some(value)),
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => {
                    Err(invalid("canonical store worker disconnected"))
                }
            },
            State::Consumed => return Err(invalid("canonical store task already consumed")),
        };
        self.state = State::Consumed;
        result
    }
}

impl<I: Send + 'static, O: Send + 'static> OwnedTask<I, O> {
    fn start(
        input: I,
        spawn: impl FnOnce(Box<dyn FnOnce() + Send>) -> std::io::Result<()>,
        work: impl FnOnce(&mut I) -> Result<O> + Send + 'static,
        poison: fn(&mut I),
    ) -> Self {
        let (input_tx, request) = mpsc::sync_channel::<I>(1);
        let (output, result) = mpsc::sync_channel(1);
        let worker = Box::new(move || {
            let Ok(mut input) = request.recv() else {
                return;
            };
            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&mut input)))
                    .unwrap_or_else(|_| {
                        poison(&mut input);
                        Err(invalid("canonical store worker panicked; reopen store"))
                    });
            let _ = output.send((input, outcome));
        });
        if let Err(error) = spawn(worker) {
            return Self::ready(input, Err(io_error(error)));
        }
        if let Err(error) = input_tx.send(input) {
            return Self::ready(
                error.0,
                Err(invalid("canonical store worker exited before request")),
            );
        }
        Self {
            state: State::Pending(result),
        }
    }
}

fn spawn(worker: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("molla-canonical-io".into())
        .spawn(worker)
        .map(drop)
}

/// Read result and exact selected-frame assembly with the returned exclusive owner.
/// Read/conversion failures return a healthy store; caught worker panics poison it.
pub struct MpmSoilCanonicalStoreReadCompletion {
    pub store: MpmSoilCanonicalStore,
    pub result: Result<Option<(MpmSoilCanonicalStoreRead, MpmSoilCanonicalSelection)>>,
}

/// One bounded read and frame assembly. Drop neither joins nor interrupts I/O;
/// the worker keeps the lock until completion. No filesystem latency bound applies.
#[must_use]
pub struct MpmSoilCanonicalStoreReadTask {
    inner: OwnedTask<
        MpmSoilCanonicalStore,
        Option<(MpmSoilCanonicalStoreRead, MpmSoilCanonicalSelection)>,
    >,
}

impl MpmSoilCanonicalStoreReadTask {
    /// Nonblocking, single-consumption polling without payload assembly/copies.
    /// Disconnection is terminal; a lost owner is not reconstructed.
    pub fn poll(&mut self) -> Result<Option<MpmSoilCanonicalStoreReadCompletion>> {
        self.inner.poll().map(|outcome| {
            outcome.map(|(store, result)| MpmSoilCanonicalStoreReadCompletion { store, result })
        })
    }
}

type WriteInput = (
    MpmSoilCanonicalStore,
    MpmSoilCanonicalStoreRead,
    MpmSoilCanonicalSelection,
    MpmSoilFrameCheckpoint,
);

/// Returns the original write inputs even on startup/validation/publication error.
/// Changed publication stales the read. Publication errors or caught worker panics
/// poison the store; reopen before deciding which generation committed.
pub struct MpmSoilCanonicalStoreWritebackCompletion {
    pub store: MpmSoilCanonicalStore,
    pub read: MpmSoilCanonicalStoreRead,
    pub selection: MpmSoilCanonicalSelection,
    pub updated: MpmSoilFrameCheckpoint,
    pub result: Result<MpmSoilStoreStats>,
}

/// One owned write request. Drop does not cancel publication or release its lock
/// early; HEAD can change without an observed completion.
#[must_use]
pub struct MpmSoilCanonicalStoreWritebackTask {
    inner: OwnedTask<WriteInput, MpmSoilStoreStats>,
}

impl MpmSoilCanonicalStoreWritebackTask {
    /// Nonblocking, single-consumption poll; original buffers are moved back.
    /// Disconnection does not establish whether publication occurred.
    pub fn poll(&mut self) -> Result<Option<MpmSoilCanonicalStoreWritebackCompletion>> {
        self.inner.poll().map(|outcome| {
            outcome.map(|((store, read, selection, updated), result)| {
                MpmSoilCanonicalStoreWritebackCompletion {
                    store,
                    read,
                    selection,
                    updated,
                    result,
                }
            })
        })
    }
}

impl MpmSoilCanonicalStore {
    /// Move manifest lookup, bounded tile I/O and explicit-frame assembly to a worker.
    /// Unique-key bounds and finite origins are checked before spawn without disk I/O.
    /// Thread creation and request deduplication remain caller work, not a hard deadline.
    pub fn begin_read(
        self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
        world_origin: [f64; 3],
        grid_origin: [f32; 3],
    ) -> MpmSoilCanonicalStoreReadTask {
        self.begin_read_window(tiles, limits, world_origin, grid_origin, [0; 3])
    }

    /// Reads bounded canonical particles into an explicit client lattice window.
    /// Invalid window ranges return through the task without publishing a selection.
    pub fn begin_read_window(
        self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
        world_origin: [f64; 3],
        grid_origin: [f32; 3],
        cells: [i32; 3],
    ) -> MpmSoilCanonicalStoreReadTask {
        self.begin_read_with(
            tiles,
            limits,
            world_origin,
            grid_origin,
            spawn,
            move |store, keys, limits, origin, grid| {
                store
                    .read_tiles(keys, limits)?
                    .map(|read| {
                        let selection = read.select_window(origin, grid, cells)?;
                        Ok((read, selection))
                    })
                    .transpose()
            },
        )
    }

    fn begin_read_with<S, R>(
        self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
        world_origin: [f64; 3],
        grid_origin: [f32; 3],
        spawn: S,
        read: R,
    ) -> MpmSoilCanonicalStoreReadTask
    where
        S: FnOnce(Box<dyn FnOnce() + Send>) -> std::io::Result<()>,
        R: FnOnce(
                &MpmSoilCanonicalStore,
                &[SoilTileCoord],
                MpmSoilStoreReadLimits,
                [f64; 3],
                [f32; 3],
            )
                -> Result<Option<(MpmSoilCanonicalStoreRead, MpmSoilCanonicalSelection)>>
            + Send
            + 'static,
    {
        self.begin_read_mode_with(tiles, limits, None, world_origin, grid_origin, spawn, read)
    }

    pub(in crate::mpm) fn begin_read_existing_window(
        self,
        tiles: &[SoilTileCoord],
        request_tiles: u32,
        limits: MpmSoilStoreReadLimits,
        world_origin: [f64; 3],
        grid_origin: [f32; 3],
        cells: [i32; 3],
    ) -> MpmSoilCanonicalStoreReadTask {
        self.begin_read_mode_with(
            tiles,
            limits,
            Some(request_tiles),
            world_origin,
            grid_origin,
            spawn,
            move |store, keys, limits, origin, grid| {
                store
                    .read_tiles(keys, limits)?
                    .map(|read| {
                        let selection = read.select_window(origin, grid, cells)?;
                        Ok((read, selection))
                    })
                    .transpose()
            },
        )
    }

    fn begin_read_mode_with<S, R>(
        self,
        tiles: &[SoilTileCoord],
        limits: MpmSoilStoreReadLimits,
        request_tiles: Option<u32>,
        world_origin: [f64; 3],
        grid_origin: [f32; 3],
        spawn: S,
        read: R,
    ) -> MpmSoilCanonicalStoreReadTask
    where
        S: FnOnce(Box<dyn FnOnce() + Send>) -> std::io::Result<()>,
        R: FnOnce(
                &MpmSoilCanonicalStore,
                &[SoilTileCoord],
                MpmSoilStoreReadLimits,
                [f64; 3],
                [f32; 3],
            )
                -> Result<Option<(MpmSoilCanonicalStoreRead, MpmSoilCanonicalSelection)>>
            + Send
            + 'static,
    {
        let prepared = (|| {
            if world_origin.iter().any(|v| !v.is_finite())
                || grid_origin.iter().any(|v| !v.is_finite())
            {
                return Err(invalid("nonfinite canonical read frame"));
            }
            self.inner.bounded_read_keys(
                tiles,
                MpmSoilStoreReadLimits {
                    tiles: request_tiles.unwrap_or(limits.tiles),
                    ..limits
                },
            )
        })();
        let keys: Vec<_> = match prepared {
            Ok(keys) => keys.into_iter().collect(),
            Err(error) => {
                return MpmSoilCanonicalStoreReadTask {
                    inner: OwnedTask::ready(self, Err(error)),
                };
            }
        };
        if self.inner.manifest.is_none() {
            return MpmSoilCanonicalStoreReadTask {
                inner: OwnedTask::ready(self, Ok(None)),
            };
        }
        MpmSoilCanonicalStoreReadTask {
            inner: OwnedTask::start(
                self,
                spawn,
                move |store| {
                    let keys = if request_tiles.is_some() {
                        store
                            .inner
                            .existing_read_keys(&keys, limits, TILE_BYTES, PARTICLE_BYTES)?
                    } else {
                        store.inner.selected_keys_sized(
                            &keys,
                            limits,
                            TILE_BYTES,
                            PARTICLE_BYTES,
                        )?;
                        keys
                    };
                    read(store, &keys, limits, world_origin, grid_origin)
                },
                |store| store.inner.failed = true,
            ),
        }
    }

    /// Move store, canonical read, selected-frame token and captured update into
    /// one bounded worker. Provenance/count/frame checks run before spawn without
    /// scanning particles; validation, canonicalization and I/O run on the worker.
    pub fn begin_writeback(
        self,
        read: MpmSoilCanonicalStoreRead,
        selection: MpmSoilCanonicalSelection,
        updated: MpmSoilFrameCheckpoint,
        limits: MpmSoilStoreReadLimits,
    ) -> MpmSoilCanonicalStoreWritebackTask {
        self.begin_writeback_checked(read, selection, updated, limits, |_| Ok(()))
    }

    pub(in crate::mpm) fn begin_writeback_checked(
        self,
        read: MpmSoilCanonicalStoreRead,
        selection: MpmSoilCanonicalSelection,
        updated: MpmSoilFrameCheckpoint,
        limits: MpmSoilStoreReadLimits,
        check: impl FnOnce(&MpmSoilCanonicalStore) -> Result<()> + Send + 'static,
    ) -> MpmSoilCanonicalStoreWritebackTask {
        self.begin_writeback_with(
            read,
            selection,
            updated,
            limits,
            spawn,
            move |store, read, selection, updated, limits| {
                check(store)?;
                store.writeback(read, selection, updated, limits)
            },
        )
    }

    fn begin_writeback_with<S, W>(
        self,
        read: MpmSoilCanonicalStoreRead,
        selection: MpmSoilCanonicalSelection,
        updated: MpmSoilFrameCheckpoint,
        limits: MpmSoilStoreReadLimits,
        spawn: S,
        write: W,
    ) -> MpmSoilCanonicalStoreWritebackTask
    where
        S: FnOnce(Box<dyn FnOnce() + Send>) -> std::io::Result<()>,
        W: FnOnce(
                &mut MpmSoilCanonicalStore,
                &MpmSoilCanonicalStoreRead,
                &MpmSoilCanonicalSelection,
                &MpmSoilFrameCheckpoint,
                MpmSoilStoreReadLimits,
            ) -> Result<MpmSoilStoreStats>
            + Send
            + 'static,
    {
        let validation = self.validate_writeback_request(&read, &selection, &updated, limits);
        let input = (self, read, selection, updated);
        let inner = match validation {
            Err(error) => OwnedTask::ready(input, Err(error)),
            Ok(()) => OwnedTask::start(
                input,
                spawn,
                move |(store, read, selection, updated)| {
                    write(store, read, selection, updated, limits)
                },
                |input| input.0.inner.failed = true,
            ),
        };
        MpmSoilCanonicalStoreWritebackTask { inner }
    }
}

#[cfg(test)]
#[path = "mpm_soil_canonical_store_tasks_tests.rs"]
pub(super) mod tests;
