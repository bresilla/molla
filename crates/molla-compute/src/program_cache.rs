use molla_core::{Error, Result};
use std::any::{Any, TypeId};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

struct Entry {
    kind: TypeId,
    variant: String,
    program: Arc<dyn Any + Send + Sync>,
}

pub(crate) struct ProgramCache {
    capacity: usize,
    state: Mutex<CacheState>,
    changed: Condvar,
}

struct Build {
    kind: TypeId,
    variant: String,
    identity: Arc<()>,
}

#[derive(Default)]
struct CacheState {
    entries: VecDeque<Entry>,
    builds: Vec<Build>,
    epoch: Arc<()>,
}

struct Admission<'a> {
    cache: &'a ProgramCache,
    identity: Arc<()>,
    epoch: Arc<()>,
    released: bool,
}

impl Admission<'_> {
    fn release(&mut self, program: Option<Arc<dyn Any + Send + Sync>>) {
        let mut state = self.cache.state.lock().unwrap_or_else(|e| e.into_inner());
        let index = state
            .builds
            .iter()
            .position(|b| Arc::ptr_eq(&b.identity, &self.identity))
            .expect("live cache admission");
        let build = state.builds.swap_remove(index);
        let mut retired = None;
        if let Some(program) = program {
            if Arc::ptr_eq(&state.epoch, &self.epoch) {
                state.entries.push_back(Entry {
                    kind: build.kind,
                    variant: build.variant,
                    program,
                });
                if state.entries.len() > self.cache.capacity {
                    retired = state.entries.pop_front();
                }
            }
        }
        self.released = true;
        drop(state);
        self.cache.changed.notify_all();
        drop(retired);
    }
}

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        if !self.released {
            self.release(None);
        }
    }
}

impl Default for ProgramCache {
    fn default() -> Self {
        Self::new(64)
    }
}

impl ProgramCache {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            capacity,
            state: Mutex::new(CacheState::default()),
            changed: Condvar::new(),
        }
    }

    pub(crate) fn get_or_create<T: Any + Send + Sync>(
        &self,
        variant: &str,
        create: impl FnOnce() -> Result<T>,
    ) -> Result<Arc<T>> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let mut admission = loop {
            if let Some(index) = state
                .entries
                .iter()
                .position(|entry| entry.kind == TypeId::of::<T>() && entry.variant == variant)
            {
                let entry = state.entries.remove(index).expect("matched program entry");
                let program = Arc::clone(&entry.program)
                    .downcast::<T>()
                    .map_err(|_| Error::Gpu("GPU program cache type mismatch".into()))?;
                state.entries.push_back(entry);
                return Ok(program);
            }
            if state.builds.len() < self.capacity.min(2)
                && !state
                    .builds
                    .iter()
                    .any(|b| b.kind == TypeId::of::<T>() && b.variant == variant)
            {
                let identity = Arc::new(());
                state.builds.push(Build {
                    kind: TypeId::of::<T>(),
                    variant: variant.to_owned(),
                    identity: identity.clone(),
                });
                break Admission {
                    cache: self,
                    identity,
                    epoch: state.epoch.clone(),
                    released: false,
                };
            }
            state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        };
        drop(state);
        let program = Arc::new(create()?);
        admission.release(Some(program.clone()));
        Ok(program)
    }

    pub(crate) fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let retired = std::mem::take(&mut state.entries);
        state.epoch = Arc::new(());
        drop(state);
        drop(retired);
    }
}

#[cfg(test)]
#[path = "program_cache_concurrency_tests.rs"]
mod concurrency_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn cache_retains_programs_and_separates_type_and_variant() {
        let cache = ProgramCache::new(4);
        let first = cache.get_or_create("a", || Ok(17_u32)).unwrap();
        let weak = Arc::downgrade(&first);
        drop(first);
        let retained = cache
            .get_or_create("a", || panic!("cached variant recompiled"))
            .unwrap();
        assert!(Arc::ptr_eq(&weak.upgrade().unwrap(), &retained));
        assert_eq!(*cache.get_or_create("a", || Ok(23_u64)).unwrap(), 23);
        assert_eq!(*cache.get_or_create("b", || Ok(29_u32)).unwrap(), 29);
        drop(retained);
        cache.clear();
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn failed_initialization_can_be_retried() {
        let cache = ProgramCache::default();
        assert!(
            cache
                .get_or_create::<u32>("a", || Err(Error::Gpu("failed program".into())))
                .is_err()
        );
        assert_eq!(*cache.get_or_create("a", || Ok(31_u32)).unwrap(), 31);
    }

    #[test]
    fn eviction_is_bounded_lru_without_invalidating_live_programs() {
        let cache = ProgramCache::new(2);
        let a = cache.get_or_create("a", || Ok(1_u32)).unwrap();
        let b = cache.get_or_create("b", || Ok(2_u32)).unwrap();
        let weak_b = Arc::downgrade(&b);
        drop(b);
        assert_eq!(
            *cache
                .get_or_create::<u32>("a", || panic!("a recompiled"))
                .unwrap(),
            1
        );
        cache.get_or_create("c", || Ok(3_u32)).unwrap();
        assert!(weak_b.upgrade().is_none());
        assert_eq!(*a, 1);
        assert_eq!(*cache.get_or_create("b", || Ok(4_u32)).unwrap(), 4);
        assert_eq!(cache.state.lock().unwrap().entries.len(), 2);
        assert_eq!(*a, 1);
    }

    #[test]
    fn simultaneous_initialization_compiles_once() {
        let cache = Arc::new(ProgramCache::default());
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let builds = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (cache, barrier, builds) = (cache.clone(), barrier.clone(), builds.clone());
                std::thread::spawn(move || {
                    barrier.wait();
                    cache
                        .get_or_create("same", || {
                            builds.fetch_add(1, Ordering::SeqCst);
                            Ok(37_u32)
                        })
                        .unwrap()
                })
            })
            .collect();
        let programs: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert!(
            programs
                .iter()
                .all(|program| Arc::ptr_eq(program, &programs[0]))
        );
    }
}
