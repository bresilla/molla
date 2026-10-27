use super::super::super::tests::{Directory, limits};
use super::super::tests::{archive, bounds, moved, shifted, state};
use super::*;
use std::time::{Duration, Instant};

fn wait<T>(mut poll: impl FnMut() -> Result<Option<T>>) -> T {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(value) = poll().unwrap() {
            return value;
        }
        assert!(Instant::now() < deadline, "canonical worker deadline");
        std::thread::sleep(Duration::from_millis(1));
    }
}
pub(crate) fn finish_read(
    task: &mut MpmSoilCanonicalStoreReadTask,
) -> MpmSoilCanonicalStoreReadCompletion {
    wait(|| task.poll())
}
pub(crate) fn finish_write(
    task: &mut MpmSoilCanonicalStoreWritebackTask,
) -> MpmSoilCanonicalStoreWritebackCompletion {
    wait(|| task.poll())
}
fn fixture(dir: &Directory) -> WriteInput {
    let mut store = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    store.save(&archive(&state(129))).unwrap();
    let key = store.tiles().unwrap().next().unwrap();
    let read = store.read_tiles(&[key], bounds(1)).unwrap().unwrap();
    let selection = shifted(&read);
    let updated = moved(&selection);
    (store, read, selection, updated)
}

#[test]
fn canonical_existing_read_filters_absent_keys_and_retains_frame() {
    let dir = Directory::new();
    let (store, read, _, _) = fixture(&dir);
    let key = read.tiles().next().unwrap();
    let absent = SoilTileCoord([i64::MAX; 3]);
    let head = fs::read(dir.0.join("HEAD")).unwrap();
    let origin = [1e12; 3];
    let grid = [-4.; 3];
    let mut task =
        store.begin_read_existing_window(&[absent, key, key], 2, bounds(1), origin, grid, [0; 3]);
    let completion = finish_read(&mut task);
    let (read, selection) = completion.result.unwrap().unwrap();
    assert_eq!(read.tiles().collect::<Vec<_>>(), vec![key]);
    assert_eq!(selection.framed_selection().world_origin(), origin);
    assert_eq!(
        selection.framed_selection().soil().config().grid_origin,
        grid
    );
    let mut task =
        completion
            .store
            .begin_read_existing_window(&[absent], 1, bounds(0), origin, grid, [0; 3]);
    let completion = finish_read(&mut task);
    assert_eq!(completion.result.unwrap().unwrap().0.stats().particles, 0);
    let mut task = completion.store.begin_read_existing_window(
        &[key, absent],
        1,
        bounds(1),
        origin,
        grid,
        [0; 3],
    );
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    let mut task = completion.store.begin_read_existing_window(
        &[key, absent],
        2,
        bounds(0),
        origin,
        grid,
        [0; 3],
    );
    let completion = finish_read(&mut task);
    assert!(completion.result.is_err());
    completion.store.inner.healthy().unwrap();
    assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
}

#[test]
fn canonical_checked_write_rejects_on_worker_before_publication_and_retries() {
    let dir = Directory::new();
    let (store, read, selection, updated) = fixture(&dir);
    let expected = updated.soil.words.clone();
    let head = fs::read(dir.0.join("HEAD")).unwrap();
    let caller = std::thread::current().id();
    let (entered, entry) = mpsc::sync_channel(1);
    let (release, permit) = mpsc::sync_channel(1);
    let mut task = store.begin_writeback_checked(read, selection, updated, bounds(2), move |_| {
        assert_ne!(caller, std::thread::current().id());
        entered.send(()).unwrap();
        permit.recv().unwrap();
        Err(invalid("selection capacity rejected"))
    });
    entry.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(task.poll().unwrap().is_none());
    assert!(MpmSoilCanonicalStore::open(&dir.0, limits()).is_err());
    assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
    release.send(()).unwrap();
    let completion = finish_write(&mut task);
    assert!(completion.result.is_err());
    completion.store.validate_read(&completion.read).unwrap();
    assert_eq!(completion.updated.soil.words, expected);
    assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
    let mut retry = completion.store.begin_writeback(
        completion.read,
        completion.selection,
        completion.updated,
        bounds(2),
    );
    assert_eq!(finish_write(&mut retry).result.unwrap().generation, 2);
}

#[test]
fn canonical_manifest_preflight_runs_on_worker_before_payloads() {
    let dir = Directory::new();
    let (mut store, read, _, _) = fixture(&dir);
    let key = read.tiles().next().unwrap();
    let head = fs::read(dir.0.join("HEAD")).unwrap();
    for (requested, limits) in [
        (SoilTileCoord([i64::MAX; 3]), bounds(1)),
        (
            key,
            MpmSoilStoreReadLimits {
                particles: 0,
                ..bounds(1)
            },
        ),
        (
            key,
            MpmSoilStoreReadLimits {
                encoded_bytes: 0,
                ..bounds(1)
            },
        ),
    ] {
        let (entered, entry) = mpsc::sync_channel(1);
        let (release, permit) = mpsc::sync_channel(1);
        let mut task = store.begin_read_with(
            &[requested, requested],
            limits,
            [1e12; 3],
            [-4.; 3],
            move |worker| {
                spawn(Box::new(move || {
                    entered.send(()).unwrap();
                    permit.recv().unwrap();
                    worker();
                }))
            },
            |_, _, _, _, _| panic!("invalid canonical request reached payload reader"),
        );
        entry.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(task.poll().unwrap().is_none());
        assert!(MpmSoilCanonicalStore::open(&dir.0, super::super::super::tests::limits()).is_err());
        release.send(()).unwrap();
        let completion = finish_read(&mut task);
        assert!(completion.result.is_err());
        assert_eq!(completion.store.total_particle_count().unwrap(), Some(129));
        assert!(task.poll().is_err());
        assert_eq!(fs::read(dir.0.join("HEAD")).unwrap(), head);
        store = completion.store;
    }
}

#[test]
fn canonical_integer_window_background_roundtrip_and_mismatch_rejection() {
    let dir = Directory::new();
    let initial = state(129);
    let mut windowed = initial.clone();
    windowed.soil.grid_window_cells = [-1, 0, 1];
    let global = archive(&initial);
    assert_eq!(archive(&windowed).to_bytes(), global.to_bytes());
    let keys: Vec<_> = global.tiles().collect();
    let mut store = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    store.save(&global).unwrap();
    let head = std::fs::read(dir.0.join("HEAD")).unwrap();
    let mut task = store.begin_read_window(
        &keys,
        bounds(4),
        initial.origin,
        initial.soil.cfg.grid_origin,
        [-1, 0, 1],
    );
    let mut completion = finish_read(&mut task);
    let (read, selection) = completion.result.unwrap().unwrap();
    let mut update = selection.framed.to_frame_checkpoint().unwrap();
    assert_eq!(update.to_bytes(), windowed.to_bytes());
    update.soil.grid_window_cells = [0; 3];
    assert!(
        completion
            .store
            .writeback(&read, &selection, &update, bounds(4))
            .is_err()
    );
    assert_eq!(std::fs::read(dir.0.join("HEAD")).unwrap(), head);
    completion.store.validate_read(&read).unwrap();
    update.soil.grid_window_cells = [-1, 0, 1];
    update.soil.words[129 * 4] = 0.25f32.to_bits();
    let expected = update.to_bytes();
    let mut task = completion
        .store
        .begin_writeback(read, selection, update, bounds(4));
    let completion = finish_write(&mut task);
    assert_eq!(completion.result.as_ref().unwrap().generation, 2);
    assert!(completion.store.validate_read(&completion.read).is_err());
    drop(completion);
    let mut reopened = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    let loaded = reopened.load().unwrap().unwrap();
    assert_eq!(loaded.metadata.grid_window_cells(), [0; 3]);
    assert_eq!(loaded.metadata.cfg.grid_origin, [0.0; 3]);
    let read = reopened.read_tiles(&keys, bounds(4)).unwrap().unwrap();
    assert_eq!(
        read.select_window(initial.origin, initial.soil.cfg.grid_origin, [-1, 0, 1])
            .unwrap()
            .framed
            .to_frame_checkpoint()
            .unwrap()
            .to_bytes(),
        expected
    );
    assert_eq!(
        read.select(initial.origin, initial.soil.cfg.grid_origin)
            .unwrap()
            .framed
            .soil()
            .grid_window_cells(),
        [0; 3]
    );
    let mut task = reopened.begin_read_window(
        &keys,
        bounds(4),
        initial.origin,
        initial.soil.cfg.grid_origin,
        [i32::MAX, 0, 0],
    );
    let completion = finish_read(&mut task);
    assert!(completion.result.is_err());
    completion.store.validate_read(&read).unwrap();
}
fn read_selected(
    store: &MpmSoilCanonicalStore,
    keys: &[SoilTileCoord],
    bounds: MpmSoilStoreReadLimits,
    origin: [f64; 3],
    grid: [f32; 3],
) -> Result<Option<(MpmSoilCanonicalStoreRead, MpmSoilCanonicalSelection)>> {
    store
        .read_tiles(keys, bounds)?
        .map(|read| {
            let selected = read.select(origin, grid)?;
            Ok((read, selected))
        })
        .transpose()
}

#[test]
fn canonical_background_read_assembles_frame_off_thread_and_returns_same_buffers() {
    let dir = Directory::new();
    let (store, read, selected, _) = fixture(&dir);
    let keys: Vec<_> = read.tiles().collect();
    let expected = selected
        .framed_selection()
        .to_frame_checkpoint()
        .unwrap()
        .to_bytes();
    let caller = std::thread::current().id();
    let (entered, ready) = mpsc::sync_channel(1);
    let (release, resume) = mpsc::sync_channel(1);
    let mut task = store.begin_read_with(
        &keys,
        bounds(1),
        selected.framed.world_origin(),
        selected.framed.soil().cfg.grid_origin,
        spawn,
        move |store, keys, bounds, origin, grid| {
            assert_ne!(caller, std::thread::current().id());
            let result = read_selected(store, keys, bounds, origin, grid)?;
            entered
                .send(result.as_ref().unwrap().1.framed.soil().words.as_ptr() as usize)
                .unwrap();
            resume.recv().unwrap();
            Ok(result)
        },
    );
    let pointer = ready.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(task.poll().unwrap().is_none());
    assert!(MpmSoilCanonicalStore::open(&dir.0, limits()).is_err());
    release.send(()).unwrap();
    let completion = finish_read(&mut task);
    let (loaded, selection) = completion.result.unwrap().unwrap();
    assert_eq!(selection.framed.soil().words.as_ptr() as usize, pointer);
    assert_eq!(
        selection.framed.to_frame_checkpoint().unwrap().to_bytes(),
        expected
    );
    completion.store.validate_read(&loaded).unwrap();
    completion.store.validate_read(&read).unwrap();
    assert!(task.poll().is_err());
}

#[test]
fn canonical_background_write_moves_all_inputs_and_preserves_unloaded_records() {
    let dir = Directory::new();
    let (store, read, selected, updated) = fixture(&dir);
    let expected_bytes = updated.to_bytes();
    let pointer = updated.soil().words.as_ptr();
    let selected_pointer = selected.framed.soil().words.as_ptr();
    let mut expected = archive(&state(129));
    let token = expected
        .select(
            &read.tiles().collect::<Vec<_>>(),
            129,
            selected.framed.world_origin(),
            selected.framed.soil().cfg.grid_origin,
        )
        .unwrap();
    expected.writeback(&token, &updated).unwrap();
    let mut task = store.begin_writeback(read, selected, updated, bounds(2));
    let mut completion = finish_write(&mut task);
    assert_eq!(completion.result.unwrap().generation, 2);
    assert_eq!(completion.updated.to_bytes(), expected_bytes);
    assert_eq!(completion.updated.soil().words.as_ptr(), pointer);
    assert_eq!(
        completion.selection.framed.soil().words.as_ptr(),
        selected_pointer
    );
    assert!(completion.store.validate_read(&completion.read).is_err());
    assert_eq!(
        completion.store.load().unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
    assert!(task.poll().is_err());
}

#[test]
fn canonical_background_preflight_spawn_and_worker_errors_return_retry_inputs() {
    let dir = Directory::new();
    let (store, read, selected, mut updated) = fixture(&dir);
    let frame = updated.origin;
    updated.origin[0] += 1.0;
    let mut task = store.begin_writeback_with(
        read,
        selected,
        updated,
        bounds(2),
        |_| panic!("invalid frame spawned"),
        MpmSoilCanonicalStore::writeback,
    );
    let mut completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    assert_eq!(completion.updated.origin[0], frame[0] + 1.0);
    completion.updated.origin = frame;
    let mut task = completion.store.begin_writeback_with(
        completion.read,
        completion.selection,
        completion.updated,
        bounds(2),
        |_| Err(std::io::Error::other("spawn failed")),
        MpmSoilCanonicalStore::writeback,
    );
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    let mut task = completion.store.begin_writeback_with(
        completion.read,
        completion.selection,
        completion.updated,
        bounds(2),
        |worker| {
            drop(worker);
            Ok(())
        },
        MpmSoilCanonicalStore::writeback,
    );
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    let mut task = completion.store.begin_writeback(
        completion.read,
        completion.selection,
        completion.updated,
        bounds(1),
    );
    let completion = finish_write(&mut task);
    assert!(completion.result.is_err());
    completion.store.validate_read(&completion.read).unwrap();
    let mut task = completion.store.begin_writeback(
        completion.read,
        completion.selection,
        completion.updated,
        bounds(2),
    );
    assert_eq!(finish_write(&mut task).result.unwrap().generation, 2);
}

#[test]
fn canonical_background_read_bounds_conversion_and_spawn_failures_preserve_store() {
    let dir = Directory::new();
    let (store, read, _, _) = fixture(&dir);
    let keys: Vec<_> = read.tiles().collect();
    let mut task = store.begin_read_with(
        &keys,
        bounds(0),
        [1e12; 3],
        [-4.0; 3],
        |_| panic!("invalid bounds spawned"),
        read_selected,
    );
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    let mut task = completion.store.begin_read_with(
        &keys,
        bounds(1),
        [1e12; 3],
        [-4.0; 3],
        |_| Err(std::io::Error::other("spawn failed")),
        read_selected,
    );
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.is_err());
    let mut task = completion
        .store
        .begin_read(&keys, bounds(1), [0.0; 3], [-4.0; 3]);
    let completion = finish_read(&mut task);
    assert!(completion.result.is_err());
    completion.store.validate_read(&read).unwrap();
    let mut task = completion
        .store
        .begin_read(&keys, bounds(1), [1e12; 3], [-4.0; 3]);
    let completion = finish_read(&mut task);
    assert_eq!(completion.result.unwrap().unwrap().0.stats().particles, 33);
}

#[test]
fn canonical_background_worker_panics_and_publication_errors_preserve_recovery_contract() {
    for write in [false, true] {
        for panics in [false, true] {
            let dir = Directory::new();
            let (store, read, selected, updated) = fixture(&dir);
            let expected_bytes = updated.to_bytes();
            let mut expected = archive(&state(129));
            let token = expected
                .select(
                    &read.tiles().collect::<Vec<_>>(),
                    129,
                    selected.framed.world_origin(),
                    selected.framed.soil().cfg.grid_origin,
                )
                .unwrap();
            if write {
                expected.writeback(&token, &updated).unwrap();
            }
            let returned = if write {
                let mut task = store.begin_writeback_with(
                    read,
                    selected,
                    updated,
                    bounds(2),
                    spawn,
                    move |store, read, selected, updated, bounds| {
                        store.writeback_with(read, selected, updated, bounds, |stage| {
                            if stage == Stage::Published {
                                if panics {
                                    panic!("injected canonical publish panic");
                                }
                                return Err(invalid("injected canonical publish error"));
                            }
                            Ok(())
                        })
                    },
                );
                let completion = finish_write(&mut task);
                assert!(completion.result.is_err());
                assert_eq!(completion.updated.to_bytes(), expected_bytes);
                completion.store
            } else {
                let keys: Vec<_> = read.tiles().collect();
                let mut task = store.begin_read_with(
                    &keys,
                    bounds(1),
                    [1e12; 3],
                    [-4.0; 3],
                    spawn,
                    move |_, _, _, _, _| {
                        if panics {
                            panic!("injected canonical read panic");
                        }
                        Err(invalid("injected canonical read error"))
                    },
                );
                let completion = finish_read(&mut task);
                assert!(completion.result.is_err());
                completion.store
            };
            assert_eq!(returned.layout().is_err(), write || panics);
            drop(returned);
            let mut reopened = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
            assert_eq!(
                reopened.load().unwrap().unwrap().to_bytes(),
                expected.to_bytes()
            );
        }
    }
}

#[test]
fn canonical_background_dropped_tasks_keep_lock_and_do_not_cancel_writes() {
    for write in [false, true] {
        let dir = Directory::new();
        let (store, read, selected, updated) = fixture(&dir);
        let keys: Vec<_> = read.tiles().collect();
        let (entered, ready) = mpsc::sync_channel(1);
        let (release, resume) = mpsc::sync_channel(1);
        let (dropped, observed) = mpsc::sync_channel(1);
        if write {
            let task = store.begin_writeback_with(
                read,
                selected,
                updated,
                bounds(2),
                spawn,
                move |store, read, selected, updated, bounds| {
                    entered.send(()).unwrap();
                    resume.recv().unwrap();
                    store.writeback(read, selected, updated, bounds)
                },
            );
            ready.recv_timeout(Duration::from_secs(5)).unwrap();
            std::thread::spawn(move || {
                drop(task);
                dropped.send(()).unwrap();
            });
        } else {
            let task = store.begin_read_with(
                &keys,
                bounds(1),
                [1e12; 3],
                [-4.0; 3],
                spawn,
                move |store, keys, bounds, origin, grid| {
                    entered.send(()).unwrap();
                    resume.recv().unwrap();
                    read_selected(store, keys, bounds, origin, grid)
                },
            );
            ready.recv_timeout(Duration::from_secs(5)).unwrap();
            std::thread::spawn(move || {
                drop(task);
                dropped.send(()).unwrap();
            });
        }
        observed.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(MpmSoilCanonicalStore::open(&dir.0, limits()).is_err());
        release.send(()).unwrap();
        let reopened = wait(|| Ok(MpmSoilCanonicalStore::open(&dir.0, limits()).ok()));
        assert_eq!(reopened.generation(), Some(if write { 2 } else { 1 }));
        assert_eq!(
            reopened.tile_particle_count(keys[0]).unwrap(),
            if write { None } else { Some(33) }
        );
    }
}

#[test]
fn canonical_background_empty_reads_and_disconnected_tasks_are_terminal() {
    fn send<T: Send>() {}
    send::<MpmSoilCanonicalStoreReadTask>();
    send::<MpmSoilCanonicalStoreWritebackTask>();
    let dir = Directory::new();
    let store = MpmSoilCanonicalStore::open(&dir.0, limits()).unwrap();
    let mut task = store.begin_read(&[], bounds(0), [0.0; 3], [0.0; 3]);
    let completion = task.poll().unwrap().unwrap();
    assert!(completion.result.unwrap().is_none());
    assert!(task.poll().is_err());
    drop(completion.store);
    let (sender, receiver) = mpsc::sync_channel::<Outcome<(), ()>>(1);
    drop(sender);
    let mut task = OwnedTask {
        state: State::Pending(receiver),
    };
    assert!(task.poll().is_err());
    assert!(task.poll().is_err());
}
