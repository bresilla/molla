#[path = "../benches/support/metrics.rs"]
mod metrics;

#[global_allocator]
static ALLOCATOR: metrics::CountingAllocator = metrics::CountingAllocator;

#[test]
fn nearest_rank_quantiles_and_bootstrap_are_deterministic() {
    let values: Vec<_> = (1..=100).rev().map(f64::from).collect();
    assert_eq!(metrics::quantile(&values, 50), 50.0);
    assert_eq!(metrics::quantile(&values, 95), 95.0);
    assert_eq!(metrics::quantile(&[3.0], 95), 3.0);
    assert_eq!(metrics::bootstrap_p95(&[4.0; 32]), (4.0, 4.0));
    let bounds = metrics::bootstrap_p95(&values);
    assert!(bounds.0 <= 95.0 && bounds.1 >= 95.0);
    assert_eq!(bounds, metrics::bootstrap_p95(&values));
}

#[test]
fn benchmark_allocator_counts_and_recovers_from_unwind() {
    let (_, counts) = metrics::measure(|| unsafe {
        let layout = std::alloc::Layout::from_size_align(128, 8).unwrap();
        let pointer = std::alloc::alloc_zeroed(layout);
        assert!(!pointer.is_null());
        let pointer = std::alloc::realloc(pointer, layout, 256);
        assert!(!pointer.is_null());
        std::alloc::dealloc(
            pointer,
            std::alloc::Layout::from_size_align(256, 8).unwrap(),
        );
    });
    assert_eq!(
        counts,
        metrics::Counts {
            allocations: 1,
            reallocations: 1,
            requested_bytes: 384
        }
    );
    let _ = std::panic::catch_unwind(|| metrics::measure(|| panic!("counter cleanup")));
    assert_eq!(metrics::measure(|| 3).1, metrics::Counts::default());
}
