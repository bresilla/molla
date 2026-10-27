use super::super::super::tests::{Directory, limits};
use super::super::tests::{archive, moved, read_limits};
use super::*;
use std::sync::mpsc;
use std::time::{Duration, Instant};

fn spawn(worker: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
    std::thread::Builder::new().spawn(worker).map(drop)
}

fn wait<T>(mut poll: impl FnMut() -> Result<Option<T>>) -> T {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(value) = poll().unwrap() {
            return value;
        }
        assert!(Instant::now() < deadline, "framed worker did not finish");
        std::thread::sleep(Duration::from_millis(1));
    }
}

pub(crate) fn finish_read(task: &mut MpmSoilFrameStoreReadTask) -> MpmSoilFrameStoreReadCompletion {
    wait(|| task.poll())
}

pub(crate) fn finish_write(
    task: &mut MpmSoilFrameStoreWritebackTask,
) -> MpmSoilFrameStoreWritebackCompletion {
    wait(|| task.poll())
}

fn fixture(
    dir: &Directory,
) -> (
    MpmSoilFrameStore,
    MpmSoilFrameStoreRead,
    MpmSoilFrameCheckpoint,
) {
    let mut store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    store.save(&archive(129)).unwrap();
    let key = store.tiles().unwrap().next().unwrap();
    let read = store.read_tiles(&[key], read_limits(1)).unwrap().unwrap();
    let update = moved(&read, 1.0);
    (store, read, update)
}

#[test]
fn framed_background_read_preserves_frame_identity_buffers_and_lock() {
    let dir = Directory::new();
    let (store, read, _) = fixture(&dir);
    let keys: Vec<_> = read.tiles().collect();
    let expected = read
        .selection_snapshot()
        .to_frame_checkpoint()
        .unwrap()
        .to_bytes();
    let caller = std::thread::current().id();
    let (entered, ready) = mpsc::sync_channel(1);
    let (release, resume) = mpsc::sync_channel(1);
    let mut task = store.start_read(|store| {
        store.begin_read_with(&keys, read_limits(1), spawn, move |store, keys, bounds| {
            assert_ne!(caller, std::thread::current().id());
            let result = store.read_tiles(keys, bounds)?;
            let pointer = result
                .as_ref()
                .unwrap()
                .selection()
                .checkpoint()
                .words
                .as_ptr() as usize;
            entered.send(pointer).unwrap();
            resume.recv().unwrap();
            Ok(result)
        })
    });
    let pointer = ready.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(task.poll().unwrap().is_none());
    assert!(MpmSoilFrameStore::open(&dir.0, limits()).is_err());
    release.send(()).unwrap();
    let completion = finish_read(&mut task);
    let loaded = completion.result.unwrap().unwrap();
    assert_eq!(loaded.soil().words.as_ptr() as usize, pointer);
    assert_eq!(
        loaded
            .selection_snapshot()
            .to_frame_checkpoint()
            .unwrap()
            .to_bytes(),
        expected
    );
    assert_eq!(loaded.particle_ids(), read.particle_ids());
    completion.store.validate_read(&read).unwrap();
    completion.store.validate_read(&loaded).unwrap();
    assert!(task.poll().is_err());
}

#[test]
fn framed_background_write_moves_inputs_and_preserves_unloaded_tiles() {
    let dir = Directory::new();
    let (store, read, updated) = fixture(&dir);
    let pointer = updated.soil().words.as_ptr();
    let read_pointer = read.soil().words.as_ptr();
    let bytes = updated.to_bytes();
    let mut expected = archive(129);
    let selection = expected
        .select(&read.tiles().collect::<Vec<_>>(), 129)
        .unwrap();
    expected.writeback(&selection, &updated).unwrap();
    let mut task = store.begin_writeback(read, updated, read_limits(2));
    let mut completion = finish_write(&mut task);
    assert_eq!(completion.result.unwrap().generation, 2);
    assert_eq!(completion.updated.soil().words.as_ptr(), pointer);
    assert_eq!(completion.read.soil().words.as_ptr(), read_pointer);
    assert_eq!(completion.updated.to_bytes(), bytes);
    assert!(completion.store.validate_read(&completion.read).is_err());
    assert_eq!(
        completion.store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
    assert!(task.poll().is_err());
}

#[test]
fn framed_background_preflight_and_start_errors_return_exact_retry_inputs() {
    let dir = Directory::new();
    let (store, read, updated) = fixture(&dir);
    let wrong = MpmSoilFrameCheckpoint::new(updated.soil().clone(), [0.0; 3]).unwrap();
    let wrong_bytes = wrong.to_bytes();
    let mut task = store.start_writeback(read, wrong, |_, _, _| panic!("changed frame spawned"));
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    assert_eq!(completion.updated.to_bytes(), wrong_bytes);
    completion.store.validate_read(&completion.read).unwrap();
    let mut task =
        completion
            .store
            .start_writeback(completion.read, updated, |store, read, updated| {
                store.begin_writeback_with(
                    read,
                    updated,
                    read_limits(2),
                    |_| Err(std::io::Error::other("spawn failed")),
                    MpmSoilStore::writeback,
                )
            });
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    let bytes = completion.updated.to_bytes();
    let mut task =
        completion
            .store
            .begin_writeback(completion.read, completion.updated, read_limits(1));
    let completion = finish_write(&mut task);
    assert!(completion.result.is_err());
    assert_eq!(completion.updated.to_bytes(), bytes);
    completion.store.validate_read(&completion.read).unwrap();
    let mut task =
        completion
            .store
            .begin_writeback(completion.read, completion.updated, read_limits(2));
    assert_eq!(finish_write(&mut task).result.unwrap().generation, 2);
}

#[test]
fn framed_background_empty_reads_bounds_and_spawn_errors_preserve_owner() {
    let dir = Directory::new();
    let store = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
    let mut task = store.begin_read(&[], read_limits(0));
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.unwrap().is_none());
    let mut store = completion.store;
    store.save(&archive(129)).unwrap();
    let keys: Vec<_> = store.tiles().unwrap().collect();
    let mut task = store.begin_read(&keys, read_limits(0));
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    let mut task = completion.store.start_read(|store| {
        store.begin_read_with(
            &keys,
            read_limits(4),
            |_| Err(std::io::Error::other("spawn failed")),
            MpmSoilStore::read_tiles,
        )
    });
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    assert_eq!(completion.store.world_origin().unwrap(), Some([1e12; 3]));
    let mut task = completion.store.begin_read(&keys, read_limits(4));
    let completion = finish_read(&mut task);
    assert_eq!(completion.result.unwrap().unwrap().soil().count, 129);
}

#[test]
fn framed_background_read_errors_and_panics_preserve_health_contract() {
    for panics in [false, true] {
        let dir = Directory::new();
        let (store, read, _) = fixture(&dir);
        let keys: Vec<_> = read.tiles().collect();
        let mut task = store.start_read(|store| {
            store.begin_read_with(&keys, read_limits(1), spawn, move |_, _, _| {
                if panics {
                    panic!("injected framed read panic");
                }
                Err(invalid("injected framed read error"))
            })
        });
        let completion = finish_read(&mut task);
        assert!(completion.result.is_err());
        assert_eq!(completion.store.world_origin().is_err(), panics);
        drop(completion.store);
        let mut reopened = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
        assert_eq!(
            reopened.load().unwrap().unwrap().to_bytes(),
            archive(129).to_bytes()
        );
    }
}

#[test]
fn framed_background_publication_failure_and_panic_reopen_exact_published_frame() {
    for panics in [false, true] {
        let dir = Directory::new();
        let (store, read, updated) = fixture(&dir);
        let bytes = updated.to_bytes();
        let mut expected = archive(129);
        let selection = expected
            .select(&read.tiles().collect::<Vec<_>>(), 129)
            .unwrap();
        expected.writeback(&selection, &updated).unwrap();
        let mut task = store.start_writeback(read, updated, |store, read, updated| {
            store.begin_writeback_with(
                read,
                updated,
                read_limits(2),
                spawn,
                move |store, read, updated, bounds| {
                    store.writeback_with(read, updated, bounds, |stage| {
                        if stage == Stage::Published {
                            if panics {
                                panic!("injected framed publication panic");
                            }
                            return Err(invalid("injected framed publication error"));
                        }
                        Ok(())
                    })
                },
            )
        });
        let completion = finish_write(&mut task);
        assert!(completion.result.is_err());
        assert!(completion.store.world_origin().is_err());
        assert_eq!(completion.updated.to_bytes(), bytes);
        drop(completion.store);
        let mut reopened = MpmSoilFrameStore::open(&dir.0, limits()).unwrap();
        assert_eq!(reopened.generation(), Some(2));
        assert_eq!(
            reopened.load().unwrap().unwrap().to_bytes(),
            expected.to_bytes()
        );
    }
}

#[test]
fn framed_background_dropped_read_and_write_retain_lock_until_worker_ends() {
    for write in [false, true] {
        let dir = Directory::new();
        let (store, read, updated) = fixture(&dir);
        let keys: Vec<_> = read.tiles().collect();
        let mut expected = archive(129);
        if write {
            let selection = expected.select(&keys, 129).unwrap();
            expected.writeback(&selection, &updated).unwrap();
        }
        let (entered, ready) = mpsc::sync_channel(1);
        let (release, resume) = mpsc::sync_channel(1);
        if write {
            let task = store.start_writeback(read, updated, |store, read, updated| {
                store.begin_writeback_with(
                    read,
                    updated,
                    read_limits(2),
                    spawn,
                    move |store, read, updated, bounds| {
                        entered.send(()).unwrap();
                        resume.recv().unwrap();
                        store.writeback(read, updated, bounds)
                    },
                )
            });
            ready.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(MpmSoilFrameStore::open(&dir.0, limits()).is_err());
            drop(task);
        } else {
            let task = store.start_read(|store| {
                store.begin_read_with(&keys, read_limits(1), spawn, move |store, keys, bounds| {
                    entered.send(()).unwrap();
                    resume.recv().unwrap();
                    store.read_tiles(keys, bounds)
                })
            });
            ready.recv_timeout(Duration::from_secs(5)).unwrap();
            drop(task);
        }
        assert!(MpmSoilFrameStore::open(&dir.0, limits()).is_err());
        release.send(()).unwrap();
        let mut reopened = wait(|| Ok(MpmSoilFrameStore::open(&dir.0, limits()).ok()));
        assert_eq!(
            reopened.load().unwrap().unwrap().to_bytes(),
            expected.to_bytes()
        );
    }
}
