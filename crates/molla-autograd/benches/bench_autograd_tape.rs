//! Tape forward + reverse pass on a synthetic 100-op program.
//!
//! ```bash
//! cargo bench -p molla --bench bench_autograd_tape
//! ```

use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use molla_autograd::Tape;

const N_OPS: usize = 100;

fn bench_tape(c: &mut Criterion) {
    let mut group = c.benchmark_group("autograd_tape_cpu");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(8));
    group.bench_function("forward_reverse_100ops", |bencher| {
        bencher.iter(|| {
            let mut tape = Tape::start();
            let a = tape.alloc(vec![1.0; 3]);
            let b = tape.alloc(vec![0.5; 3]);
            // Build a chain of mixed ops.
            let mut acc = a;
            for i in 0..N_OPS {
                if i & 1 == 0 {
                    acc = tape.add(acc, b);
                } else {
                    acc = tape.mul_scalar(acc, 0.999);
                }
            }
            let n = tape.norm(acc);
            tape.stop();
            let grads = tape.backward(black_box(n), &[1.0]);
            black_box(grads.len())
        });
    });
    group.finish();
}

criterion_group!(benches, bench_tape);
criterion_main!(benches);
