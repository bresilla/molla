use super::*;
use std::{sync::mpsc, time::Duration};

struct PanickingDrop;
impl Drop for PanickingDrop {
    fn drop(&mut self) {
        panic!("retired program destructor");
    }
}

#[test]
fn eviction_panic_leaves_ready_entry_and_releases_admission() {
    let cache = ProgramCache::new(1);
    drop(
        cache
            .get_or_create("retired", || Ok(PanickingDrop))
            .unwrap(),
    );
    assert!(std::panic::catch_unwind(|| cache.get_or_create("live", || Ok(23u32))).is_err());
    assert!(!cache.state.is_poisoned());
    assert!(cache.state.lock().unwrap().builds.is_empty());
    assert_eq!(
        *cache
            .get_or_create::<u32>("live", || panic!("published entry lost"))
            .unwrap(),
        23
    );
    assert_eq!(*cache.get_or_create("next", || Ok(29u32)).unwrap(), 29);
}

#[test]
fn clear_panic_does_not_poison_or_retain_cache_state() {
    let cache = ProgramCache::new(1);
    drop(
        cache
            .get_or_create("retired", || Ok(PanickingDrop))
            .unwrap(),
    );
    assert!(std::panic::catch_unwind(|| cache.clear()).is_err());
    assert!(!cache.state.is_poisoned());
    assert!(cache.state.lock().unwrap().entries.is_empty());
    assert_eq!(*cache.get_or_create("next", || Ok(31u32)).unwrap(), 31);
}

#[test]
fn unrelated_builds_overlap_and_ready_hits_remain_available() {
    let cache = Arc::new(ProgramCache::new(3));
    cache.get_or_create("ready", || Ok(7u32)).unwrap();
    let (started, observed) = mpsc::channel();
    let mut releases = Vec::new();
    let mut threads = Vec::new();
    for key in ["a", "b"] {
        let (release, wait) = mpsc::channel();
        releases.push(release);
        let (cache, started) = (cache.clone(), started.clone());
        threads.push(std::thread::spawn(move || {
            cache
                .get_or_create(key, || {
                    started.send(key).unwrap();
                    wait.recv().unwrap();
                    Ok(11u32)
                })
                .unwrap()
        }));
    }
    let a = observed.recv_timeout(Duration::from_secs(5));
    let b = observed.recv_timeout(Duration::from_secs(5));
    let hit = cache.get_or_create::<u32>("ready", || panic!("ready cache miss"));
    for release in releases {
        release.send(()).unwrap();
    }
    for thread in threads {
        thread.join().unwrap();
    }
    assert!(a.is_ok() && b.is_ok());
    assert_ne!(a.unwrap(), b.unwrap());
    assert_eq!(*hit.unwrap(), 7);
    assert!(cache.state.lock().unwrap().builds.is_empty());
}

#[test]
fn panic_and_error_release_single_flight_reservations() {
    let cache = ProgramCache::new(1);
    assert!(
        std::panic::catch_unwind(|| cache.get_or_create::<u32>("a", || panic!("factory panic")))
            .is_err()
    );
    assert!(cache.state.lock().unwrap().builds.is_empty());
    assert!(
        cache
            .get_or_create::<u32>("a", || Err(Error::Gpu("factory error".into())))
            .is_err()
    );
    assert!(cache.state.lock().unwrap().builds.is_empty());
    assert_eq!(*cache.get_or_create("a", || Ok(13u32)).unwrap(), 13);
}

#[test]
fn clear_during_build_does_not_retain_the_old_generation() {
    let cache = Arc::new(ProgramCache::new(2));
    let (started, observed) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let worker = {
        let cache = cache.clone();
        std::thread::spawn(move || {
            cache
                .get_or_create("a", || {
                    started.send(()).unwrap();
                    wait.recv().unwrap();
                    Ok(17u32)
                })
                .unwrap()
        })
    };
    observed.recv_timeout(Duration::from_secs(5)).unwrap();
    cache.clear();
    release.send(()).unwrap();
    let old = worker.join().unwrap();
    assert_eq!(*old, 17);
    assert!(cache.state.lock().unwrap().entries.is_empty());
    let fresh = cache.get_or_create("a", || Ok(19u32)).unwrap();
    assert_eq!(*fresh, 19);
    assert!(!Arc::ptr_eq(&old, &fresh));
}

#[test]
fn cold_build_admission_is_bounded() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let cache = Arc::new(ProgramCache::new(64));
    let active = Arc::new(AtomicUsize::new(0));
    let maximum = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(std::sync::Barrier::new(16));
    let threads: Vec<_> = (0..16)
        .map(|i| {
            let (cache, active, maximum, barrier) = (
                cache.clone(),
                active.clone(),
                maximum.clone(),
                barrier.clone(),
            );
            std::thread::spawn(move || {
                barrier.wait();
                cache
                    .get_or_create(&i.to_string(), || {
                        let n = active.fetch_add(1, Ordering::SeqCst) + 1;
                        maximum.fetch_max(n, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(2));
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok(i)
                    })
                    .unwrap()
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    assert!(maximum.load(Ordering::SeqCst) <= 2);
    assert_eq!(active.load(Ordering::SeqCst), 0);
    let state = cache.state.lock().unwrap();
    assert!(state.builds.is_empty());
    assert_eq!(state.entries.len(), 16);
}
