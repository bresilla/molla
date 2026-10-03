//! Scoped dispatch, logical transfer and observed allocation counters on the host thread.

#[cfg(feature = "gpu-metrics")]
mod enabled {
    use crate::{Error, Result};
    use std::{cell::Cell, marker::PhantomData, rc::Rc};

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DispatchCounts {
        pub encoded_dispatches: u64,
        pub overflowed: bool,
    }

    thread_local! {
        static COUNTS: Cell<Option<DispatchCounts>> = const { Cell::new(None) };
        static TRANSFERS: Cell<Option<TransferCounts>> = const { Cell::new(None) };
        static DEVICES: Cell<Option<DeviceCounts>> = const { Cell::new(None) };
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DeviceCounts {
        pub observations: u64,
        pub peak_buffer_bytes: u64,
        pub peak_buffers: u64,
        pub peak_native_allocations: Option<u64>,
        pub invalid: bool,
    }

    /// Observed device totals at forwarded buffer allocations and explicit boundaries.
    pub struct DeviceScope(PhantomData<Rc<()>>);

    impl DeviceScope {
        pub fn start() -> Result<Self> {
            DEVICES.with(|counts| {
                if counts.get().is_some() {
                    return Err(Error::Gpu("nested GPU device metrics scope".into()));
                }
                counts.set(Some(DeviceCounts::default()));
                Ok(Self(PhantomData))
            })
        }

        pub fn finish(self) -> DeviceCounts {
            DEVICES.with(|counts| counts.take().expect("active GPU device metrics scope"))
        }
    }

    impl Drop for DeviceScope {
        fn drop(&mut self) {
            DEVICES.with(|counts| counts.set(None));
        }
    }

    pub fn device_scope_active() -> bool {
        DEVICES.with(|counts| counts.get().is_some())
    }

    pub fn record_device(buffer_bytes: isize, buffers: isize, native_allocations: isize) {
        DEVICES.with(|counts| {
            if let Some(mut value) = counts.get() {
                if let Some(next) = value.observations.checked_add(1) {
                    value.observations = next;
                } else {
                    value.invalid = true;
                }
                match u64::try_from(buffer_bytes) {
                    Ok(bytes) => value.peak_buffer_bytes = value.peak_buffer_bytes.max(bytes),
                    Err(_) => value.invalid = true,
                }
                match u64::try_from(buffers) {
                    Ok(buffers) => value.peak_buffers = value.peak_buffers.max(buffers),
                    Err(_) => value.invalid = true,
                }
                if native_allocations > 0 {
                    value.peak_native_allocations = Some(
                        value
                            .peak_native_allocations
                            .unwrap_or(0)
                            .max(native_allocations as u64),
                    );
                } else if native_allocations < 0 {
                    value.invalid = true;
                }
                counts.set(Some(value));
            }
        });
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TransferCounts {
        pub upload_bytes: u64,
        pub readback_bytes: u64,
        pub overflowed: bool,
    }

    /// Logical queue-write and mapped-read bytes on this orchestration thread.
    pub struct TransferScope(PhantomData<Rc<()>>);

    impl TransferScope {
        pub fn start() -> Result<Self> {
            TRANSFERS.with(|counts| {
                if counts.get().is_some() {
                    return Err(Error::Gpu("nested GPU transfer metrics scope".into()));
                }
                counts.set(Some(TransferCounts::default()));
                Ok(Self(PhantomData))
            })
        }

        pub fn finish(self) -> TransferCounts {
            TRANSFERS.with(|counts| counts.take().expect("active GPU transfer metrics scope"))
        }
    }

    impl Drop for TransferScope {
        fn drop(&mut self) {
            TRANSFERS.with(|counts| counts.set(None));
        }
    }

    fn record_transfer(bytes: u64, upload: bool) {
        TRANSFERS.with(|counts| {
            if let Some(mut value) = counts.get() {
                let total = if upload {
                    &mut value.upload_bytes
                } else {
                    &mut value.readback_bytes
                };
                if let Some(next) = total.checked_add(bytes) {
                    *total = next;
                } else {
                    value.overflowed = true;
                }
                counts.set(Some(value));
            }
        });
    }

    pub fn record_upload(bytes: u64) {
        record_transfer(bytes, true);
    }
    pub fn record_readback(bytes: u64) {
        record_transfer(bytes, false);
    }

    /// Counts instrumented command encodings, not shader invocations or completed GPU work.
    pub struct DispatchScope(PhantomData<Rc<()>>);

    impl DispatchScope {
        pub fn start() -> Result<Self> {
            COUNTS.with(|counts| {
                if counts.get().is_some() {
                    return Err(Error::Gpu("nested GPU dispatch metrics scope".into()));
                }
                counts.set(Some(DispatchCounts::default()));
                Ok(Self(PhantomData))
            })
        }

        pub fn finish(self) -> DispatchCounts {
            COUNTS.with(|counts| counts.take().expect("active GPU dispatch metrics scope"))
        }
    }

    impl Drop for DispatchScope {
        fn drop(&mut self) {
            COUNTS.with(|counts| counts.set(None));
        }
    }

    pub fn record_dispatch() {
        COUNTS.with(|counts| {
            if let Some(mut value) = counts.get() {
                if let Some(next) = value.encoded_dispatches.checked_add(1) {
                    value.encoded_dispatches = next;
                } else {
                    value.overflowed = true;
                }
                counts.set(Some(value));
            }
        });
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn scopes_count_only_encodings_inside_their_lifetime() {
            record_dispatch();
            let scope = DispatchScope::start().unwrap();
            for _ in 0..3 {
                record_dispatch();
            }
            assert_eq!(
                scope.finish(),
                DispatchCounts {
                    encoded_dispatches: 3,
                    overflowed: false
                }
            );
            record_dispatch();
            assert_eq!(
                DispatchScope::start().unwrap().finish(),
                DispatchCounts::default()
            );
        }

        #[test]
        fn nested_scopes_fail_without_resetting_the_outer_count() {
            let scope = DispatchScope::start().unwrap();
            record_dispatch();
            assert!(DispatchScope::start().is_err());
            record_dispatch();
            assert_eq!(scope.finish().encoded_dispatches, 2);
        }

        #[test]
        fn dropping_during_unwind_clears_the_scope() {
            assert!(
                std::panic::catch_unwind(|| {
                    let _scope = DispatchScope::start().unwrap();
                    record_dispatch();
                    panic!("metrics unwind fixture");
                })
                .is_err()
            );
            assert_eq!(
                DispatchScope::start().unwrap().finish(),
                DispatchCounts::default()
            );
        }

        #[test]
        fn threads_keep_independent_counts() {
            let scope = DispatchScope::start().unwrap();
            record_dispatch();
            let other = std::thread::spawn(|| {
                record_dispatch();
                let scope = DispatchScope::start().unwrap();
                record_dispatch();
                record_dispatch();
                scope.finish()
            })
            .join()
            .unwrap();
            assert_eq!(other.encoded_dispatches, 2);
            assert_eq!(scope.finish().encoded_dispatches, 1);
        }

        #[test]
        fn counter_overflow_is_reported_without_wrapping() {
            let scope = DispatchScope::start().unwrap();
            COUNTS.with(|counts| {
                counts.set(Some(DispatchCounts {
                    encoded_dispatches: u64::MAX,
                    overflowed: false,
                }))
            });
            record_dispatch();
            assert_eq!(
                scope.finish(),
                DispatchCounts {
                    encoded_dispatches: u64::MAX,
                    overflowed: true
                }
            );
        }
    }
}

#[cfg(feature = "gpu-metrics")]
pub use enabled::{
    DeviceCounts, DeviceScope, DispatchCounts, DispatchScope, TransferCounts, TransferScope,
    device_scope_active, record_device, record_dispatch, record_readback, record_upload,
};

#[cfg(not(feature = "gpu-metrics"))]
#[inline(always)]
pub fn device_scope_active() -> bool {
    false
}

#[cfg(not(feature = "gpu-metrics"))]
#[inline(always)]
pub fn record_device(_buffer_bytes: isize, _buffers: isize, _native_allocations: isize) {}

#[cfg(not(feature = "gpu-metrics"))]
#[inline(always)]
pub fn record_dispatch() {}

#[cfg(not(feature = "gpu-metrics"))]
#[inline(always)]
pub fn record_upload(_bytes: u64) {}

#[cfg(not(feature = "gpu-metrics"))]
#[inline(always)]
pub fn record_readback(_bytes: u64) {}

#[inline(always)]
pub fn upload_slice(bytes: &[u8]) -> &[u8] {
    record_upload(bytes.len() as u64);
    bytes
}

#[inline(always)]
pub fn upload_value<T>(value: T, length: impl FnOnce(&T) -> usize) -> T {
    record_upload(length(&value) as u64);
    value
}

#[inline(always)]
pub fn device_allocation<D, T>(
    device: D,
    allocate: impl FnOnce(&D) -> T,
    sample: impl FnOnce(&D) -> (isize, isize, isize),
) -> T {
    let buffer = allocate(&device);
    if device_scope_active() {
        let (bytes, buffers, native_allocations) = sample(&device);
        record_device(bytes, buffers, native_allocations);
    }
    buffer
}

/// Device buffer allocation with allocation-event counter observation.
#[macro_export]
macro_rules! gpu_create_buffer {
    ($device:expr, $descriptor:expr $(,)?) => {{
        $crate::gpu_metrics::device_allocation(
            &($device),
            |device| device.create_buffer($descriptor),
            |device| {
                let counters = device.get_internal_counters().hal;
                (
                    counters.buffer_memory.read(),
                    counters.buffers.read(),
                    counters.memory_allocations.read(),
                )
            },
        )
    }};
}

/// Initialized device buffer with logical initial-content byte accounting.
#[macro_export]
macro_rules! gpu_buffer_init {
    ($device:expr, $descriptor:expr $(,)?) => {{
        $crate::gpu_metrics::device_allocation(
            &($device),
            |device| {
                device.create_buffer_init($crate::gpu_metrics::upload_value($descriptor, |d| {
                    d.contents.len()
                }))
            },
            |device| {
                let counters = device.get_internal_counters().hal;
                (
                    counters.buffer_memory.read(),
                    counters.buffers.read(),
                    counters.memory_allocations.read(),
                )
            },
        )
    }};
}

/// Queue buffer upload with logical byte accounting.
#[macro_export]
macro_rules! gpu_write_buffer {
    ($queue:expr, $buffer:expr, $offset:expr, $bytes:expr $(,)?) => {{ ($queue).write_buffer($buffer, $offset, $crate::gpu_metrics::upload_slice($bytes)) }};
}

/// Read-only mapped buffer view with logical byte accounting.
#[macro_export]
macro_rules! gpu_readback {
    ($slice:expr) => {{
        let view = ($slice).get_mapped_range();
        $crate::gpu_metrics::record_readback(view.len() as u64);
        view
    }};
}

/// Mutable mapped buffer view with logical host-write footprint accounting.
#[macro_export]
macro_rules! gpu_mapped_upload {
    ($slice:expr) => {{
        let view = ($slice).get_mapped_range_mut();
        $crate::gpu_metrics::record_upload(view.len() as u64);
        view
    }};
}
