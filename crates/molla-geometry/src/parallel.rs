//! The thread pool collision work runs on: `MOLLA_THREADS` workers, by default
//! a quarter of the cores and at most four; one keeps all of it on the caller.

use std::sync::OnceLock;

use rayon::prelude::*;

/// Fewest items worth splitting across the pool.
const MIN_PARALLEL: usize = 64;

fn pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::env::var("MOLLA_THREADS")
            .ok()
            .and_then(|threads| threads.parse::<usize>().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| (n.get() / 4).clamp(1, 4)));
        (threads > 1)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("molla-{i}"))
                    .build()
                    .ok()
            })
            .flatten()
    })
    .as_ref()
}

/// `f` over consecutive chunks of `items`, each with the index it starts at;
/// on the pool when there are enough of them. Results keep the chunks' order.
pub(crate) fn map_chunks<T: Sync, R: Send>(items: &[T], f: impl Fn(usize, &[T]) -> R + Sync) -> Vec<R> {
    match pool() {
        Some(pool) if items.len() >= MIN_PARALLEL => {
            let size = items.len().div_ceil(pool.current_num_threads() * 4).max(16);
            pool.install(|| items.par_chunks(size).enumerate().map(|(n, chunk)| f(n * size, chunk)).collect())
        }
        _ => vec![f(0, items)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_every_item_once_in_order() {
        for len in [0, 5, 63, 64, 1000, 4099] {
            let items: Vec<usize> = (0..len).collect();
            let parts = map_chunks(&items, |start, chunk| {
                assert!(chunk.iter().enumerate().all(|(k, &item)| item == start + k));
                chunk.to_vec()
            });
            assert_eq!(parts.concat(), items, "{len}");
        }
    }
}
