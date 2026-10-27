//! CPU serialization and import of already collected point ranges.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_math::Vec3;
use molla_sensors::{
    CloudPoint, CloudReadLimits,
    cloud_export::{read_cloud_ply, write_cloud_ply, write_cloud_xyz_rgb},
};
use std::{io::Cursor, time::Duration};

fn bench_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("sensor_cloud_export");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));
    for n in [1024usize, 65_536, 1_000_000] {
        let points: Vec<_> = (0..n)
            .map(|i| {
                let mut p =
                    CloudPoint::new(Vec3::new(i as f64 * 0.125, -0.5, 1.25), u64::MAX, i as u64);
                p.extra = [i as u32, 0x80604020, 3, 0];
                p
            })
            .collect();
        let mut output = Vec::with_capacity(n * 112 + 4096);
        group.bench_function(BenchmarkId::new("cpu_ply_write", n), |b| {
            b.iter(|| {
                output.clear();
                write_cloud_ply(black_box(&points), &mut output).unwrap();
                black_box(&output);
            })
        });
        group.bench_function(BenchmarkId::new("cpu_xyz_write", n), |b| {
            b.iter(|| {
                output.clear();
                write_cloud_xyz_rgb(black_box(&points), &mut output).unwrap();
                black_box(&output);
            })
        });
        let mut input = Vec::with_capacity(n * 112 + 4096);
        write_cloud_ply(&points, &mut input).unwrap();
        let limits = CloudReadLimits {
            max_points: n,
            max_file_bytes: input.len() as u64,
        };
        group.bench_function(BenchmarkId::new("cpu_ply_read", n), |b| {
            b.iter(|| {
                black_box(read_cloud_ply(Cursor::new(black_box(&input)), limits).unwrap());
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_export);
criterion_main!(benches);
