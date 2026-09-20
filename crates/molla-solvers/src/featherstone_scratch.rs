use std::ops::{Index, IndexMut};

pub(super) struct JointScratch<T, const LOCAL: bool> {
    values: Vec<T>,
    start: usize,
    zero: T,
}

impl<T: Copy, const LOCAL: bool> JointScratch<T, LOCAL> {
    pub(super) fn new(capacity: usize, zero: T) -> Self {
        Self {
            values: if LOCAL {
                Vec::with_capacity(capacity)
            } else {
                vec![zero; capacity]
            },
            start: 0,
            zero,
        }
    }

    pub(super) fn reset(&mut self, start: usize, end: usize) {
        if LOCAL {
            self.values.clear();
            self.values.resize(end - start, self.zero);
            self.start = start;
        }
    }
}

impl<T, const LOCAL: bool> Index<usize> for JointScratch<T, LOCAL> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.values[if LOCAL { index - self.start } else { index }]
    }
}

impl<T, const LOCAL: bool> IndexMut<usize> for JointScratch<T, LOCAL> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[if LOCAL { index - self.start } else { index }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_indices_reuse_reset_storage_for_different_articulation_sizes() {
        let mut scratch = JointScratch::<_, true>::new(8, 0.0);
        let pointer = scratch.values.as_ptr();
        for (start, size) in [(0, 8), (71, 2), (99, 6), (105, 0), (107, 8)] {
            scratch.reset(start, start + size);
            assert_eq!(scratch.values.len(), size);
            assert_eq!(scratch.values.as_ptr(), pointer);
            for j in start..start + size {
                assert_eq!(scratch[j], 0.0);
                scratch[j] = j as f64 + 1.0;
            }
            for j in start..start + size {
                assert_eq!(scratch[j], j as f64 + 1.0);
            }
        }
    }

    #[test]
    #[should_panic]
    fn spare_capacity_is_not_another_articulations_state() {
        let mut scratch = JointScratch::<_, true>::new(8, 0.0);
        scratch.reset(10, 12);
        let _ = scratch[12];
    }

    #[test]
    fn flat_storage_retains_global_indices() {
        let mut scratch = JointScratch::<_, false>::new(8, 0.0);
        scratch[1] = 9.0;
        scratch.reset(5, 7);
        assert_eq!(scratch.values.len(), 8);
        assert_eq!(scratch[1], 9.0);
        assert_eq!(scratch[6], 0.0);
    }
}
