use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::spin_loop,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

static LOCK: AtomicBool = AtomicBool::new(false);
static ACTIVE: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicU64 = AtomicU64::new(0);
static LIVE_BLOCKS: AtomicU64 = AtomicU64::new(0);
static INITIAL_BLOCKS: AtomicU64 = AtomicU64::new(0);
static PEAK_BLOCKS: AtomicU64 = AtomicU64::new(0);
static INITIAL: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static REALLOCS: AtomicU64 = AtomicU64::new(0);
static REQUESTED: AtomicU64 = AtomicU64::new(0);
static OVERFLOW: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Default)]
pub struct Counts {
    pub allocations: u64,
    pub reallocations: u64,
    pub requested_bytes: u64,
    pub peak_live_bytes: u64,
    pub peak_extra_bytes: u64,
    pub peak_live_allocations: u64,
    pub peak_extra_allocations: u64,
    pub initial_live_bytes: u64,
    pub overflowed: bool,
}

struct Guard;
impl Guard {
    fn lock() -> Self {
        while LOCK
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spin_loop();
        }
        Self
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        LOCK.store(false, Ordering::Release);
    }
}

fn add(counter: &AtomicU64, value: u64) {
    let previous = counter.load(Ordering::Relaxed);
    if let Some(next) = previous.checked_add(value) {
        counter.store(next, Ordering::Relaxed);
    } else {
        counter.store(u64::MAX, Ordering::Relaxed);
        OVERFLOW.store(true, Ordering::Relaxed);
    }
}

fn change(old: usize, new: usize, allocation: bool, reallocation: bool) {
    let _guard = Guard::lock();
    let live = LIVE.load(Ordering::Relaxed);
    if let Some(next) = live
        .checked_sub(old as u64)
        .and_then(|v| v.checked_add(new as u64))
    {
        LIVE.store(next, Ordering::Relaxed);
    } else {
        OVERFLOW.store(true, Ordering::Relaxed);
    }
    if allocation {
        add(&LIVE_BLOCKS, 1);
    } else if !reallocation {
        let blocks = LIVE_BLOCKS.load(Ordering::Relaxed);
        if let Some(next) = blocks.checked_sub(1) {
            LIVE_BLOCKS.store(next, Ordering::Relaxed);
        } else {
            OVERFLOW.store(true, Ordering::Relaxed);
        }
    }
    if ACTIVE.load(Ordering::Relaxed) {
        PEAK.fetch_max(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
        PEAK_BLOCKS.fetch_max(LIVE_BLOCKS.load(Ordering::Relaxed), Ordering::Relaxed);
        if allocation {
            add(&ALLOCS, 1);
        }
        if reallocation {
            add(&REALLOCS, 1);
        }
        if allocation || reallocation {
            add(&REQUESTED, new as u64);
        }
    }
}

pub struct CountingAllocator;
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            change(0, layout.size(), true, false);
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            change(0, layout.size(), true, false);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe {
            System.dealloc(pointer, layout);
        }
        change(layout.size(), 0, false, false);
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, size) };
        if !pointer.is_null() {
            change(layout.size(), size, false, true);
        }
        pointer
    }
}

pub struct Scope {
    finished: bool,
}
impl Scope {
    pub fn start() -> Result<Self, &'static str> {
        let _guard = Guard::lock();
        if ACTIVE.load(Ordering::Relaxed) {
            return Err("nested host allocation scope");
        }
        let live = LIVE.load(Ordering::Relaxed);
        INITIAL.store(live, Ordering::Relaxed);
        PEAK.store(live, Ordering::Relaxed);
        let blocks = LIVE_BLOCKS.load(Ordering::Relaxed);
        INITIAL_BLOCKS.store(blocks, Ordering::Relaxed);
        PEAK_BLOCKS.store(blocks, Ordering::Relaxed);
        ALLOCS.store(0, Ordering::Relaxed);
        REALLOCS.store(0, Ordering::Relaxed);
        REQUESTED.store(0, Ordering::Relaxed);
        ACTIVE.store(true, Ordering::Relaxed);
        Ok(Self { finished: false })
    }
    pub fn finish(mut self) -> Counts {
        let _guard = Guard::lock();
        ACTIVE.store(false, Ordering::Relaxed);
        self.finished = true;
        let initial = INITIAL.load(Ordering::Relaxed);
        let peak = PEAK.load(Ordering::Relaxed);
        Counts {
            allocations: ALLOCS.load(Ordering::Relaxed),
            reallocations: REALLOCS.load(Ordering::Relaxed),
            requested_bytes: REQUESTED.load(Ordering::Relaxed),
            peak_live_bytes: peak,
            peak_extra_bytes: peak.saturating_sub(initial),
            peak_live_allocations: PEAK_BLOCKS.load(Ordering::Relaxed),
            peak_extra_allocations: PEAK_BLOCKS
                .load(Ordering::Relaxed)
                .saturating_sub(INITIAL_BLOCKS.load(Ordering::Relaxed)),
            initial_live_bytes: initial,
            overflowed: OVERFLOW.load(Ordering::Relaxed),
        }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        if !self.finished {
            let _guard = Guard::lock();
            ACTIVE.store(false, Ordering::Relaxed);
        }
    }
}
