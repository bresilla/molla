use super::super::tests::{Directory, archive, limits, read_limits};
use super::*;
use std::time::{Duration, Instant};

fn spawn(worker: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
    std::thread::Builder::new().spawn(worker).map(drop)
}

fn finish(task: &mut MpmSoilStoreReadTask) -> MpmSoilStoreReadCompletion {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(completion) = task.poll().unwrap() {
            return completion;
        }
        assert!(Instant::now() < deadline, "read did not finish");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn background_read_retains_lock_identity_and_exact_history() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let archive = archive();
    store.save(&archive).unwrap();
    let key = store.tiles().unwrap().next().unwrap();
    let expected = archive.select(&[key], 129).unwrap();
    let generation = store.generation();
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    let epoch = store.epoch.clone();
    let (entered_tx, entered_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let caller = std::thread::current().id();
    let mut task = store.begin_read_with(
        &[key, key],
        read_limits(129, 1),
        spawn,
        move |store, keys, bounds| {
            assert_ne!(caller, std::thread::current().id());
            assert_eq!(keys, &[key]);
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            store.read_tiles(keys, bounds)
        },
    );
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(task.poll().unwrap().is_none());
    assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
    release_tx.send(()).unwrap();
    let completion = finish(&mut task);
    assert!(task.poll().is_err());
    let mut store = completion.store;
    let read = completion.result.unwrap().unwrap();
    assert!(Arc::ptr_eq(&store.epoch, &epoch));
    assert_eq!(store.generation(), generation);
    store.validate_read(&read).unwrap();
    assert_eq!(read.stats().tiles, 1);
    assert_eq!(read.selection().particle_ids(), expected.particle_ids());
    assert_eq!(
        read.selection().checkpoint().words,
        expected.checkpoint().words
    );
    assert_eq!(read.selection().material_ids(), expected.material_ids());
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    assert_eq!(store.save(&archive).unwrap().written_bytes, 0);
    store.validate_read(&read).unwrap();
    drop(store);
    let reopened = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert!(reopened.validate_read(&read).is_err());
}

#[test]
fn background_read_preflight_and_spawn_failures_return_owner() {
    let directory = Directory::new();
    let store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let epoch = store.epoch.clone();
    let mut empty = store.begin_read(&[], read_limits(0, 0));
    let completion = empty.poll().unwrap().unwrap();
    assert!(completion.result.unwrap().is_none());
    let mut store = completion.store;
    store.save(&archive()).unwrap();
    let key = store.tiles().unwrap().next().unwrap();
    let mut invalid = store.begin_read_with(
        &[key],
        read_limits(0, 0),
        |_| panic!("invalid request spawned a worker"),
        |_, _, _| panic!("invalid request read payloads"),
    );
    let completion = invalid.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    let store = completion.store;
    let mut failed = store.begin_read_with(
        &[key],
        read_limits(129, 1),
        |_| Err(std::io::Error::other("injected spawn failure")),
        |_, _, _| panic!("spawn failure read payloads"),
    );
    let completion = failed.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    assert!(Arc::ptr_eq(&completion.store.epoch, &epoch));
    completion.store.healthy().unwrap();
    assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
    let mut vanished = completion.store.begin_read_with(
        &[key],
        read_limits(129, 1),
        |worker| {
            drop(worker);
            Ok(())
        },
        MpmSoilStore::read_tiles,
    );
    let completion = vanished.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    completion.store.healthy().unwrap();
    assert!(Arc::ptr_eq(&completion.store.epoch, &epoch));
}

#[test]
fn background_read_io_error_and_panic_have_distinct_health() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    let archive = archive();
    store.save(&archive).unwrap();
    let key = store.tiles().unwrap().next().unwrap();
    let path = store.blob_path(
        &store
            .manifest
            .as_ref()
            .unwrap()
            .get(key)
            .unwrap()
            .unwrap()
            .hash,
    );
    let bytes = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    let mut task = store.begin_read(&[key], read_limits(129, 1));
    let completion = finish(&mut task);
    assert!(completion.result.is_err());
    completion.store.healthy().unwrap();
    fs::write(path, bytes).unwrap();
    let mut task = completion.store.begin_read(&[key], read_limits(129, 1));
    let completion = finish(&mut task);
    completion
        .store
        .validate_read(&completion.result.unwrap().unwrap())
        .unwrap();
    let mut task =
        completion
            .store
            .begin_read_with(&[key], read_limits(129, 1), spawn, |_, _, _| {
                panic!("injected read panic")
            });
    let mut completion = finish(&mut task);
    assert!(completion.result.is_err());
    assert!(completion.store.healthy().is_err());
    assert!(completion.store.save(&archive).is_err());
    assert!(completion.store.read_tiles(&[], read_limits(0, 0)).is_err());
    drop(completion);
    let mut reopened = MpmSoilStore::open(&directory.0, limits()).unwrap();
    assert_eq!(reopened.load().unwrap().unwrap().particle_count(), 129);
}

#[test]
fn dropping_pending_read_does_not_wait_or_release_worker_lock() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    store.save(&archive()).unwrap();
    let key = store.tiles().unwrap().next().unwrap();
    let (entered_tx, entered_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let task = store.begin_read_with(
        &[key],
        read_limits(129, 1),
        spawn,
        move |store, keys, bounds| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            store.read_tiles(keys, bounds)
        },
    );
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let (dropped_tx, dropped_rx) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        drop(task);
        dropped_tx.send(()).unwrap();
    });
    dropped_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
    release_tx.send(()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(mut store) = MpmSoilStore::open(&directory.0, limits()) {
            assert_eq!(store.load().unwrap().unwrap().particle_count(), 129);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "worker retained the store after completion"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn disconnected_result_is_terminal() {
    let (sender, receiver) = mpsc::sync_channel(1);
    drop(sender);
    let mut task = MpmSoilStoreReadTask {
        state: State::Pending(receiver),
    };
    assert!(task.poll().is_err());
    assert!(task.poll().is_err());
}

#[test]
fn background_manifest_preflight_runs_after_worker_admission() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    store.save(&archive()).unwrap();
    let key = store.tiles().unwrap().next().unwrap();
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    for (requested, bounds) in [
        (SoilTileCoord([i64::MAX; 3]), read_limits(129, 1)),
        (key, read_limits(0, 1)),
        (
            key,
            MpmSoilStoreReadLimits {
                encoded_bytes: 0,
                ..read_limits(129, 1)
            },
        ),
    ] {
        let (entered, entry) = mpsc::sync_channel(1);
        let (release, permit) = mpsc::sync_channel(1);
        let mut task = store.begin_read_with(
            &[requested, requested],
            bounds,
            move |worker| {
                spawn(Box::new(move || {
                    entered.send(()).unwrap();
                    permit.recv().unwrap();
                    worker();
                }))
            },
            |_, _, _| panic!("invalid manifest request reached payload reader"),
        );
        entry.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(task.poll().unwrap().is_none());
        assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
        release.send(()).unwrap();
        let completion = finish(&mut task);
        assert!(completion.result.is_err());
        completion.store.healthy().unwrap();
        assert_eq!(completion.store.total_particle_count().unwrap(), Some(129));
        assert!(task.poll().is_err());
        assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
        store = completion.store;
    }
}

#[test]
fn background_existing_read_filters_absent_keys_with_separate_request_bound() {
    let directory = Directory::new();
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    store.save(&archive()).unwrap();
    let key = store.tiles().unwrap().next().unwrap();
    let absent = SoilTileCoord([i64::MAX; 3]);
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    let caller = std::thread::current().id();
    let (entered, entry) = mpsc::sync_channel(1);
    let (release, permit) = mpsc::sync_channel(1);
    let mut task = store.begin_read_mode_with(
        &[absent, key, key],
        read_limits(129, 1),
        Some(2),
        move |worker| {
            spawn(Box::new(move || {
                entered.send(()).unwrap();
                permit.recv().unwrap();
                worker();
            }))
        },
        move |store, keys, limits| {
            assert_ne!(caller, std::thread::current().id());
            assert_eq!(keys, &[key]);
            store.read_tiles(keys, limits)
        },
    );
    entry.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(task.poll().unwrap().is_none());
    assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
    release.send(()).unwrap();
    let completion = finish(&mut task);
    assert_eq!(completion.result.unwrap().unwrap().stats().tiles, 1);
    let mut task = completion
        .store
        .begin_read_existing(&[absent], 1, read_limits(0, 0));
    let completion = finish(&mut task);
    assert_eq!(completion.result.unwrap().unwrap().stats().particles, 0);
    let mut task = completion
        .store
        .begin_read_existing(&[key, absent], 1, read_limits(129, 1));
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    let mut task = completion
        .store
        .begin_read_existing(&[key, absent], 2, read_limits(129, 0));
    let completion = finish(&mut task);
    assert!(completion.result.is_err());
    completion.store.healthy().unwrap();
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
}
