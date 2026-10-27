use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub allocations: u64,
    pub reallocations: u64,
    pub requested_bytes: u64,
}

thread_local! {
    static ACTIVE: Cell<Option<Counts>> = const { Cell::new(None) };
}

pub struct CountingAllocator;

fn record(bytes: usize, reallocation: bool) {
    let _ = ACTIVE.try_with(|cell| {
        if let Some(mut counts) = cell.get() {
            counts.allocations += u64::from(!reallocation);
            counts.reallocations += u64::from(reallocation);
            counts.requested_bytes += bytes as u64;
            cell.set(Some(counts));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size(), false);
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record(layout.size(), false);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, size) };
        if !pointer.is_null() {
            record(size, true);
        }
        pointer
    }
}

struct Recording;
impl Drop for Recording {
    fn drop(&mut self) {
        ACTIVE.with(|cell| cell.set(None));
    }
}

pub fn measure<T>(f: impl FnOnce() -> T) -> (T, Counts) {
    ACTIVE.with(|cell| {
        assert!(cell.get().is_none(), "nested allocation measurement");
        cell.set(Some(Counts::default()));
    });
    let guard = Recording;
    let value = f();
    let counts = ACTIVE.with(|cell| cell.get().unwrap());
    drop(guard);
    (value, counts)
}

pub fn quantile(values: &[f64], percentile: usize) -> f64 {
    assert!(!values.is_empty() && (1..=100).contains(&percentile));
    assert!(values.iter().all(|v| v.is_finite()));
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(values.len() * percentile).div_ceil(100) - 1]
}

pub fn bootstrap_p95(values: &[f64]) -> (f64, f64) {
    let mut state = 0x853c49e6748fea9bu64;
    let mut resample = vec![0.0; values.len()];
    let mut estimates: Vec<_> = (0..1000)
        .map(|_| {
            for value in &mut resample {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *value = values[state as usize % values.len()];
            }
            quantile(&resample, 95)
        })
        .collect();
    estimates.sort_by(f64::total_cmp);
    (estimates[24], estimates[974])
}
