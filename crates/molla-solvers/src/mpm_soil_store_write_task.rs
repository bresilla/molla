//! Single-flight selected writes retaining inputs through completion.

use super::*;
use std::sync::mpsc::{self, Receiver, TryRecvError};

/// Returns the store and original inputs on success or recoverable failure.
/// A changed generation makes `read` stale. Publication errors or unwinding
/// worker panics poison the store; reopen before deciding which state committed.
pub struct MpmSoilStoreWritebackCompletion {
    pub store: MpmSoilStore,
    pub read: MpmSoilStoreRead,
    pub updated: MpmSoilCheckpoint,
    pub result: Result<MpmSoilStoreStats>,
}

type Request = (MpmSoilStore, MpmSoilStoreRead, MpmSoilCheckpoint);

enum State {
    Pending(Receiver<MpmSoilStoreWritebackCompletion>),
    Ready(Option<MpmSoilStoreWritebackCompletion>),
    Consumed,
}

/// One owned write request and one completion slot. Dropping does not join or
/// cancel a submitted write: HEAD may change without an observed completion.
/// The worker retains the lock until it finishes and drops an undeliverable result.
/// Byte bounds do not bound filesystem latency or manifest/allocator overhead.
#[must_use]
pub struct MpmSoilStoreWritebackTask {
    state: State,
}

impl MpmSoilStoreWritebackTask {
    pub(super) fn ready(
        (store, read, updated): Request,
        result: Result<MpmSoilStoreStats>,
    ) -> Self {
        Self {
            state: State::Ready(Some(MpmSoilStoreWritebackCompletion {
                store,
                read,
                updated,
                result,
            })),
        }
    }

    /// Nonblocking; None means pending. Completion is consumed once only.
    /// Disconnection is terminal; it does not establish whether HEAD changed.
    pub fn poll(&mut self) -> Result<Option<MpmSoilStoreWritebackCompletion>> {
        let result = match &mut self.state {
            State::Ready(completion) => Ok(completion.take()),
            State::Pending(receiver) => match receiver.try_recv() {
                Ok(completion) => Ok(Some(completion)),
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => {
                    Err(invalid("soil store write worker disconnected"))
                }
            },
            State::Consumed => return Err(invalid("soil store write task already consumed")),
        };
        self.state = State::Consumed;
        result
    }
}

impl MpmSoilStore {
    /// Moves the store, current read and captured update into one background write.
    /// Caller-side preflight checks provenance/count/bounds without payload I/O.
    /// Full validation, destination reads and publication run on the worker.
    /// Failures return original inputs; there is no synchronous write fallback.
    pub fn begin_writeback(
        self,
        read: MpmSoilStoreRead,
        updated: MpmSoilCheckpoint,
        limits: MpmSoilStoreReadLimits,
    ) -> MpmSoilStoreWritebackTask {
        self.begin_writeback_checked(read, updated, limits, |_| Ok(()))
    }

    pub(in crate::mpm) fn begin_writeback_checked(
        self,
        read: MpmSoilStoreRead,
        updated: MpmSoilCheckpoint,
        limits: MpmSoilStoreReadLimits,
        check: impl FnOnce(&MpmSoilStore) -> Result<()> + Send + 'static,
    ) -> MpmSoilStoreWritebackTask {
        self.begin_writeback_with(
            read,
            updated,
            limits,
            |worker| {
                std::thread::Builder::new()
                    .name("molla-soil-write".into())
                    .spawn(worker)
                    .map(drop)
            },
            move |store, read, updated, limits| {
                check(store)?;
                store.writeback(read, updated, limits)
            },
        )
    }

    pub(super) fn begin_writeback_with<S, W>(
        self,
        read: MpmSoilStoreRead,
        updated: MpmSoilCheckpoint,
        limits: MpmSoilStoreReadLimits,
        spawn: S,
        write: W,
    ) -> MpmSoilStoreWritebackTask
    where
        S: FnOnce(Box<dyn FnOnce() + Send>) -> std::io::Result<()>,
        W: FnOnce(
                &mut MpmSoilStore,
                &MpmSoilStoreRead,
                &MpmSoilCheckpoint,
                MpmSoilStoreReadLimits,
            ) -> Result<MpmSoilStoreStats>
            + Send
            + 'static,
    {
        if let Err(error) = self.validate_writeback_request(&read, &updated, limits) {
            return MpmSoilStoreWritebackTask::ready((self, read, updated), Err(error));
        }
        let (input, request) = mpsc::sync_channel::<Request>(1);
        let (output, result) = mpsc::sync_channel(1);
        let worker = Box::new(move || {
            let Ok((mut store, read, updated)) = request.recv() else {
                return;
            };
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                write(&mut store, &read, &updated, limits)
            }));
            let result = outcome.unwrap_or_else(|_| {
                store.failed = true;
                Err(invalid("soil store write worker panicked; reopen store"))
            });
            let _ = output.send(MpmSoilStoreWritebackCompletion {
                store,
                read,
                updated,
                result,
            });
        });
        if let Err(error) = spawn(worker) {
            return MpmSoilStoreWritebackTask::ready((self, read, updated), Err(io_error(error)));
        }
        if let Err(error) = input.send((self, read, updated)) {
            return MpmSoilStoreWritebackTask::ready(
                error.0,
                Err(invalid("soil store write worker exited before request")),
            );
        }
        MpmSoilStoreWritebackTask {
            state: State::Pending(result),
        }
    }
}

#[cfg(test)]
#[path = "mpm_soil_store_write_task_tests.rs"]
mod tests;
