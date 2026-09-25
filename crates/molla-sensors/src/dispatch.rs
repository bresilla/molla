/// IMU/contact/frame work counters, excluding construction, readback and driver allocations.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SensorDispatchStats {
    pub submissions: u64,
    pub kernel_dispatches: u64,
    /// One uniform buffer and three bind groups per preparation (four for IMU).
    pub binding_preparations: u64,
    pub history_copies: u64,
    /// Contact output-clear batches, each clearing four channels.
    pub output_clears: u64,
}

pub(crate) struct BindingsCache<T> {
    entries: Vec<T>,
    capacity: usize,
}

impl<T> BindingsCache<T> {
    pub(crate) fn new() -> Self {
        Self::with_capacity(2)
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub(crate) fn get_or_prepare(
        &mut self,
        matches: impl Fn(&T) -> bool,
        prepare: impl FnOnce() -> T,
    ) -> (&T, bool) {
        self.try_get_or_prepare(matches, || Ok::<T, std::convert::Infallible>(prepare()))
            .unwrap_or_else(|error| match error {})
    }

    pub(crate) fn try_get_or_prepare<E>(
        &mut self,
        matches: impl Fn(&T) -> bool,
        prepare: impl FnOnce() -> Result<T, E>,
    ) -> Result<(&T, bool), E> {
        if let Some(index) = self.entries.iter().position(matches) {
            let last = self.entries.len() - 1;
            self.entries[index..].rotate_left(1);
            return Ok((&self.entries[last], false));
        }
        let prepared = prepare()?;
        if self.entries.len() == self.capacity {
            self.entries.remove(0);
        }
        self.entries.push(prepared);
        Ok((self.entries.last().expect("prepared binding"), true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recent_access_preserves_relative_age_of_other_cached_bindings() {
        let mut cache = BindingsCache::with_capacity(4);
        for value in 1..=4 {
            cache.get_or_prepare(|entry| *entry == value, || value);
        }
        let storage = cache.entries.as_ptr();
        cache.get_or_prepare(|entry| *entry == 2, || panic!("cached"));
        assert_eq!(cache.entries, [1, 3, 4, 2]);
        cache.get_or_prepare(|entry| *entry == 5, || 5);
        assert_eq!(cache.entries, [3, 4, 2, 5]);
        cache.get_or_prepare(|entry| *entry == 3, || panic!("cached"));
        cache.get_or_prepare(|entry| *entry == 6, || 6);
        assert_eq!(cache.entries, [2, 5, 3, 6]);
        assert_eq!(cache.entries.as_ptr(), storage);
    }

    #[test]
    fn larger_fixed_cache_reuses_all_warmed_variants() {
        let mut cache = BindingsCache::with_capacity(128);
        let storage = cache.entries.as_ptr();
        for index in 0..128 {
            assert!(cache.get_or_prepare(|value| *value == index, || index).1);
        }
        for index in (0..128).rev().cycle().take(512) {
            assert!(
                !cache
                    .get_or_prepare(|value| *value == index, || panic!("cache miss"))
                    .1
            );
        }
        assert_eq!(cache.entries.as_ptr(), storage);
        assert_eq!(cache.entries.capacity(), 128);
    }

    #[test]
    fn two_variants_reuse_storage_and_evict_least_recent() {
        let mut cache = BindingsCache::new();
        let storage = cache.entries.as_ptr();
        for value in [1, 2] {
            assert!(cache.get_or_prepare(|entry| *entry == value, || value).1);
        }
        for value in [1, 2, 1, 2, 1] {
            assert!(
                !cache
                    .get_or_prepare(|entry| *entry == value, || panic!("cache miss"))
                    .1
            );
        }
        assert!(cache.get_or_prepare(|entry| *entry == 3, || 3).1);
        assert_eq!(cache.entries, [1, 3]);
        assert_eq!(cache.entries.as_ptr(), storage);
        assert_eq!(cache.entries.capacity(), 2);
        assert_eq!(
            cache.try_get_or_prepare(|entry| *entry == 4, || Err("failed")),
            Err("failed")
        );
        assert_eq!(cache.entries, [1, 3]);
    }
}
