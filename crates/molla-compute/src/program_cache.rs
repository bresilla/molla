use molla_core::{Error, Result};
use std::any::{Any, TypeId};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

struct Entry {
    kind: TypeId,
    variant: String,
    program: Arc<dyn Any + Send + Sync>,
}

pub(crate) struct ProgramCache {
    capacity: usize,
    entries: Mutex<VecDeque<Entry>>,
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
            entries: Mutex::new(VecDeque::new()),
        }
    }

    pub(crate) fn get_or_create<T: Any + Send + Sync>(
        &self,
        variant: &str,
        create: impl FnOnce() -> Result<T>,
    ) -> Result<Arc<T>> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(index) = entries
            .iter()
            .position(|entry| entry.kind == TypeId::of::<T>() && entry.variant == variant)
        {
            let entry = entries.remove(index).expect("matched program entry");
            let program = Arc::clone(&entry.program)
                .downcast::<T>()
                .map_err(|_| Error::Gpu("GPU program cache type mismatch".into()))?;
            entries.push_back(entry);
            return Ok(program);
        }
        let program = Arc::new(create()?);
        entries.push_back(Entry {
            kind: TypeId::of::<T>(),
            variant: variant.to_owned(),
            program: program.clone(),
        });
        if entries.len() > self.capacity {
            entries.pop_front();
        }
        Ok(program)
    }

    pub(crate) fn clear(&self) {
        self.entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
    }
}

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
        assert_eq!(cache.entries.lock().unwrap().len(), 2);
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
