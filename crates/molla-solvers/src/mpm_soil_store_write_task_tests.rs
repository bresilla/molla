use super::super::tests::{Directory, archive, limits, read_limits};
use super::*;
use std::time::{Duration, Instant};

fn spawn(worker: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
    std::thread::Builder::new().spawn(worker).map(drop)
}

fn fixture(directory: &Directory) -> Request {
    let mut store = MpmSoilStore::open(&directory.0, limits()).unwrap();
    store.save(&archive()).unwrap();
    let key = store.tiles().unwrap().next().unwrap();
    let read = store
        .read_tiles(&[key], read_limits(129, 1))
        .unwrap()
        .unwrap();
    let mut updated = read.selection().checkpoint().clone();
    updated.words[updated.count as usize * 4] = 0.125_f32.to_bits();
    (store, read, updated)
}

fn finish(task: &mut MpmSoilStoreWritebackTask) -> MpmSoilStoreWritebackCompletion {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(completion) = task.poll().unwrap() {
            return completion;
        }
        assert!(Instant::now() < deadline, "write did not finish");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn background_checked_write_rejects_on_worker_before_publication_and_retries() {
    let directory = Directory::new();
    let (store, read, updated) = fixture(&directory);
    let expected = updated.to_bytes();
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    let caller = std::thread::current().id();
    let (entered, entry) = mpsc::sync_channel(1);
    let (release, permit) = mpsc::sync_channel(1);
    let mut task = store.begin_writeback_checked(read, updated, read_limits(129, 1), move |_| {
        assert_ne!(caller, std::thread::current().id());
        entered.send(()).unwrap();
        permit.recv().unwrap();
        Err(invalid("selection capacity rejected"))
    });
    entry.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(task.poll().unwrap().is_none());
    assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    release.send(()).unwrap();
    let completion = finish(&mut task);
    assert!(completion.result.is_err());
    completion.store.validate_read(&completion.read).unwrap();
    assert_eq!(completion.updated.to_bytes(), expected);
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    let mut retry =
        completion
            .store
            .begin_writeback(completion.read, completion.updated, read_limits(129, 1));
    assert_eq!(finish(&mut retry).result.unwrap().generation, 2);
}

#[test]
fn background_write_preserves_inputs_lock_and_complete_archive() {
    let directory = Directory::new();
    let (store, read, updated) = fixture(&directory);
    let input_bytes = updated.to_bytes();
    let epoch = store.epoch.clone();
    let mut expected = archive();
    let keys = read.selection().tiles.clone();
    let selection = expected.select(&keys, 129).unwrap();
    expected.writeback(&selection, &updated).unwrap();
    let (entered_tx, entered_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let caller = std::thread::current().id();
    let mut task = store.begin_writeback_with(
        read,
        updated,
        read_limits(129, 1),
        spawn,
        move |store, read, updated, bounds| {
            assert_ne!(caller, std::thread::current().id());
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            store.writeback(read, updated, bounds)
        },
    );
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(task.poll().unwrap().is_none());
    assert!(MpmSoilStore::open(&directory.0, limits()).is_err());
    release_tx.send(()).unwrap();
    let mut completion = finish(&mut task);
    assert!(task.poll().is_err());
    assert!(Arc::ptr_eq(&completion.store.epoch, &epoch));
    assert_eq!(completion.result.unwrap().generation, 2);
    assert_eq!(completion.updated.to_bytes(), input_bytes);
    assert!(completion.store.validate_read(&completion.read).is_err());
    assert_eq!(
        completion.store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
}

#[test]
fn background_write_start_failures_return_owner_and_retry_inputs() {
    let directory = Directory::new();
    let (store, read, updated) = fixture(&directory);
    let input = updated.to_bytes();
    let epoch = store.epoch.clone();
    let mut task = store.begin_writeback_with(
        read,
        updated,
        read_limits(0, 0),
        |_| panic!("invalid request spawned"),
        MpmSoilStore::writeback,
    );
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    let mut task = completion.store.begin_writeback_with(
        completion.read,
        completion.updated,
        read_limits(129, 1),
        |_| Err(std::io::Error::other("injected spawn failure")),
        MpmSoilStore::writeback,
    );
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    let mut task = completion.store.begin_writeback_with(
        completion.read,
        completion.updated,
        read_limits(129, 1),
        |worker| {
            drop(worker);
            Ok(())
        },
        MpmSoilStore::writeback,
    );
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    assert!(Arc::ptr_eq(&completion.store.epoch, &epoch));
    assert_eq!(completion.updated.to_bytes(), input);
    completion.store.validate_read(&completion.read).unwrap();
    let mut task =
        completion
            .store
            .begin_writeback(completion.read, completion.updated, read_limits(129, 1));
    assert_eq!(finish(&mut task).result.unwrap().generation, 2);
}

#[test]
fn background_write_validation_failure_is_retryable_without_publication() {
    let directory = Directory::new();
    let (store, read, mut updated) = fixture(&directory);
    let saved_word = updated.words[0];
    updated.words[0] = f32::NAN.to_bits();
    let head = fs::read(directory.0.join("HEAD")).unwrap();
    let mut task = store.begin_writeback(read, updated, read_limits(129, 1));
    let mut completion = finish(&mut task);
    assert!(completion.result.is_err());
    completion.store.validate_read(&completion.read).unwrap();
    assert_eq!(completion.updated.words[0], f32::NAN.to_bits());
    assert_eq!(fs::read(directory.0.join("HEAD")).unwrap(), head);
    completion.updated.words[0] = saved_word;
    let mut task =
        completion
            .store
            .begin_writeback(completion.read, completion.updated, read_limits(129, 1));
    assert_eq!(finish(&mut task).result.unwrap().generation, 2);
}

#[test]
fn background_write_publication_errors_and_panics_preserve_recovery_state() {
    for panic_after_publish in [false, true] {
        let directory = Directory::new();
        let (store, read, updated) = fixture(&directory);
        let input = updated.to_bytes();
        let mut task = store.begin_writeback_with(
            read,
            updated,
            read_limits(129, 1),
            spawn,
            move |store, read, updated, bounds| {
                store.writeback_with(read, updated, bounds, |stage| {
                    if stage != Stage::Published {
                        return Ok(());
                    }
                    if panic_after_publish {
                        panic!("injected panic after HEAD publication");
                    }
                    Err(invalid("injected failure after HEAD publication"))
                })
            },
        );
        let completion = finish(&mut task);
        assert!(completion.result.is_err());
        assert!(completion.store.healthy().is_err());
        assert_eq!(completion.updated.to_bytes(), input);
        let keys = completion.read.selection().tiles.clone();
        drop(completion.store);
        let store = MpmSoilStore::open(&directory.0, limits()).unwrap();
        assert_eq!(store.generation(), Some(2));
        let published = store
            .read_tiles(&keys, read_limits(129, 1))
            .unwrap()
            .unwrap();
        assert_eq!(published.selection().checkpoint().to_bytes(), input);
    }
}

#[test]
fn dropping_pending_write_does_not_cancel_publication_or_release_lock_early() {
    let directory = Directory::new();
    let (store, read, updated) = fixture(&directory);
    let keys = read.selection().tiles.clone();
    let input = updated.to_bytes();
    let (entered_tx, entered_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let task = store.begin_writeback_with(
        read,
        updated,
        read_limits(129, 1),
        spawn,
        move |store, read, updated, bounds| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            store.writeback(read, updated, bounds)
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
        if let Ok(store) = MpmSoilStore::open(&directory.0, limits()) {
            assert_eq!(store.generation(), Some(2));
            let published = store
                .read_tiles(&keys, read_limits(129, 1))
                .unwrap()
                .unwrap();
            assert_eq!(published.selection().checkpoint().to_bytes(), input);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "worker retained lock after write"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn disconnected_write_completion_is_terminal() {
    let (sender, receiver) = mpsc::sync_channel(1);
    drop(sender);
    let mut task = MpmSoilStoreWritebackTask {
        state: State::Pending(receiver),
    };
    assert!(task.poll().is_err());
    assert!(task.poll().is_err());
}
