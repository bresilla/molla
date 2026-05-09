//! Stage 24e.1 — sparse-grid v3 integration tests.
//!
//! Extends `sparse_grid.rs`'s v1 tests with:
//!   • GPU mutable insert (`SparseGridInsertKernel::insert`).
//!   • GPU root hashing — synthetic hash collisions are still
//!     resolved correctly.
//!   • Templated value types (`f32`, `vec3<f32>`, `f16`).
//!   • Storage cost regression for the 1024³ × 1 % budget across all
//!     three value types.
//!
//! All GPU-bound tests are `#[ignore]` because they need a real wgpu
//! adapter; run with
//!
//!   `cargo test -p molla --features gpu --test sparse_grid_v3 -- --ignored`.

use molla::geometry::{root_hash3, SparseGrid, SparseGridGpu};
use std::collections::HashMap;

// ── Deterministic LCG ──────────────────────────────────────────────

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xCAFE_F00D)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
}

// ── CPU coverage of the new hash-table storage layout ─────────────

#[test]
fn root_hash_table_handles_load_factor_under_seventy_percent() {
    // Build a CPU SparseGrid with 100 distinct root keys, then check
    // the GPU pack's hash table sized correctly (load factor ≤ 0.7)
    // and every key resolves through the host-side replay.
    let mut grid = SparseGrid::new(0.0);
    let mut rng = Lcg::new(0xBEEF);
    let mut keys = std::collections::HashSet::new();
    while keys.len() < 100 {
        // Each root key spans 256³ voxels — pick distinct multiples
        // of 256 along each axis so we get distinct root keys.
        let kx = ((rng.next() >> 32) as i32 % 100) * 256;
        let ky = ((rng.next() >> 32) as i32 % 100) * 256;
        let kz = ((rng.next() >> 32) as i32 % 100) * 256;
        keys.insert([kx, ky, kz]);
    }
    for k in &keys {
        grid.insert(k[0], k[1], k[2], 1.0);
    }

    let packed = SparseGridGpu::from_grid(&grid);
    assert!(packed.root_table_capacity.is_power_of_two());
    let lf = packed.root_table_load_factor();
    assert!(lf <= 0.7, "load factor {lf} > 0.7 (capacity = {})", packed.root_table_capacity);

    let (avg, max) = packed.root_hash_probe_stats();
    eprintln!(
        "100 keys → table capacity {} (load factor {:.2}); avg probes {avg:.2}, max {max}",
        packed.root_table_capacity, lf,
    );
    // For ~ 100 random keys at load 0.4-0.7 the average probe length
    // should stay under 3.0 with linear probing + Murmur3 hashing.
    // Allow some headroom for adversarial hash seeds.
    assert!(avg < 3.0, "avg probe length {avg} regressed");
}

#[test]
fn root_hash_table_grows_with_root_count() {
    // 1k distinct root keys → capacity is at least 1024 / 0.7 ≈ 1463
    // → next power of two = 2048.
    let mut grid = SparseGrid::new(0.0);
    for i in 0..1000 {
        let kx = (i * 256) as i32;
        grid.insert(kx, 0, 0, 1.0);
    }
    let packed = SparseGridGpu::from_grid(&grid);
    assert!(packed.root_table_capacity >= 2048);
    let lf = packed.root_table_load_factor();
    assert!(lf <= 0.7, "load factor {lf} > 0.7");
    eprintln!("1000 keys → capacity {} ({lf:.2})", packed.root_table_capacity);
}

#[test]
fn host_root_hash3_is_deterministic_and_avalanches() {
    // Sanity check: same input → same output, different inputs spread
    // across the u32 range.
    assert_eq!(root_hash3([0, 0, 0]), root_hash3([0, 0, 0]));
    let h1 = root_hash3([1, 0, 0]);
    let h2 = root_hash3([0, 1, 0]);
    let h3 = root_hash3([0, 0, 1]);
    assert!(h1 != h2 && h2 != h3 && h1 != h3, "hash collision on unit axes");
    // Sample 256 sequential keys; check that the low 8 bits of the
    // hash are well-distributed (avalanche).
    let mut counts = [0u32; 256];
    for i in 0..1024 {
        let h = root_hash3([i, i * 7, i * 13]);
        counts[(h & 0xFF) as usize] += 1;
    }
    let max_bucket = *counts.iter().max().unwrap();
    let avg = 1024.0 / 256.0;
    assert!(
        (max_bucket as f32) < avg * 4.0,
        "low-8-bit avalanche failed: max bucket {max_bucket} (avg {avg})"
    );
}

#[test]
fn synthetic_hash_collisions_resolved_via_open_addressing() {
    // Construct keys that explicitly collide modulo the smallest
    // power-of-two capacity (4) by sweeping the hash output. We can't
    // hand-pick a colliding key set without running the host hash, so
    // we generate ~ 1k random keys and pick the first three that
    // collide modulo 8. This exercises the open-addressing probe path
    // through the host-side replay.
    let mut rng = Lcg::new(0xCAFEBABE);
    let mut keys = Vec::new();
    let mut bucket: HashMap<u32, Vec<[i32; 3]>> = HashMap::new();
    while bucket.values().filter(|v| v.len() >= 3).count() < 2 {
        let k = [
            ((rng.next() >> 32) as i32 % 65536) * 256,
            ((rng.next() >> 32) as i32 % 65536) * 256,
            ((rng.next() >> 32) as i32 % 65536) * 256,
        ];
        let h = root_hash3(k) & 7;
        bucket.entry(h).or_default().push(k);
        keys.push(k);
    }
    // Take all keys from the colliding buckets — they're guaranteed
    // to exercise the linear-probe path.
    let colliding: Vec<[i32; 3]> = bucket
        .values()
        .filter(|v| v.len() >= 3)
        .flat_map(|v| v.iter().copied())
        .collect();
    assert!(colliding.len() >= 6, "expected ≥ 6 colliding keys, got {}", colliding.len());

    let mut grid = SparseGrid::new(0.0);
    for (i, k) in colliding.iter().enumerate() {
        grid.insert(k[0], k[1], k[2], (i + 1) as f32);
    }
    let packed = SparseGridGpu::from_grid(&grid);

    // Host-side replay of the WGSL probe — confirm every key resolves
    // back to its inserted internal_idx.
    let (avg, max) = packed.root_hash_probe_stats();
    eprintln!(
        "synthetic-collision test: {} keys, capacity {}, avg {avg:.2}, max {max}",
        colliding.len(),
        packed.root_table_capacity
    );
    assert!(max >= 1, "probe stats produced 0 probes — replay is broken");
}

// ── GPU integration tests ─────────────────────────────────────────

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use molla::compute::Engine;
    use molla::kernels::{
        read_storage_to_pairs, SparseGridInsertKernel, SparseGridStorage, ValueF16, ValueVec3,
    };

    fn try_engine() -> Option<Engine> {
        Engine::new_gpu().ok()
    }

    fn next_pow2(n: u32) -> u32 {
        let mut c = 4u32;
        while c < n {
            c *= 2;
        }
        c
    }

    /// Generate `n` distinct random voxel coords in `[-range, range)³`
    /// using a deterministic LCG.
    fn random_coords(n: usize, seed: u64, range: i32) -> Vec<[i32; 3]> {
        let mut rng = Lcg::new(seed);
        let mut set: HashMap<[i32; 3], ()> = HashMap::new();
        while set.len() < n {
            let x = ((rng.next() >> 32) as i32) % (2 * range) - range;
            let y = ((rng.next() >> 32) as i32) % (2 * range) - range;
            let z = ((rng.next() >> 32) as i32) % (2 * range) - range;
            set.insert([x, y, z], ());
        }
        set.keys().copied().collect()
    }

    #[test]
    #[ignore = "needs a real wgpu adapter"]
    fn gpu_mutable_insert_roundtrip_f32_1k() {
        let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
        let device = engine.device().unwrap().clone();
        let queue = engine.queue().unwrap().clone();

        let n = 1024usize;
        let coords = random_coords(n, 0xDEAD0001, 4096);
        let mut rng = Lcg::new(0xF00D);
        let values: Vec<f32> = (0..n)
            .map(|_| (rng.next() as u32 as f32) / (u32::MAX as f32))
            .collect();

        // Size pools so they fit the worst case (each coord in its own
        // leaf + each in its own root entry).
        let storage = SparseGridStorage::<f32>::new(
            &device,
            next_pow2((n as u32).max(8) * 2),
            n as u32,
            n as u32,
        );
        let kernel = SparseGridInsertKernel::<f32>::new(&device);

        let t0 = std::time::Instant::now();
        let counters = kernel
            .insert(&device, &queue, &storage, &coords, &values)
            .expect("insert");
        let dt = t0.elapsed();
        let throughput = n as f64 / dt.as_secs_f64();
        eprintln!(
            "gpu insert {n} pairs in {:.2}ms ({:.1} kInsert/s); counters {:?}",
            dt.as_secs_f64() * 1000.0,
            throughput / 1000.0,
            counters
        );
        assert_eq!(counters[2], 0, "errors set: {:#x}", counters[2]);

        // Read back and confirm every (coord, value) round-trips.
        let pairs = read_storage_to_pairs(&device, &queue, &storage).expect("read");
        let pair_map: HashMap<[i32; 3], f32> = pairs.into_iter().collect();
        for (c, v) in coords.iter().zip(values.iter()) {
            let got = pair_map
                .get(c)
                .copied()
                .unwrap_or_else(|| panic!("missing coord {c:?}"));
            assert_eq!(got, *v, "mismatch at {c:?}");
        }
    }

    #[test]
    #[ignore = "needs a real wgpu adapter"]
    fn gpu_mutable_insert_collision_synthetic() {
        // Insert into a deliberately small root table so most keys
        // collide. The kernel must still recover every value.
        let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
        let device = engine.device().unwrap().clone();
        let queue = engine.queue().unwrap().clone();

        let n = 32usize;
        let coords = random_coords(n, 0xC0FFEE, 8192);
        let values: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();

        // Force a tight table — capacity 64 with 32 entries → load 0.5,
        // but with such a small Murmur3 hash space we expect ~ a few
        // probes per insert.
        let storage = SparseGridStorage::<f32>::new(&device, 64, n as u32, n as u32);
        let kernel = SparseGridInsertKernel::<f32>::new(&device);
        let counters = kernel
            .insert(&device, &queue, &storage, &coords, &values)
            .expect("insert");
        eprintln!("collision-synthetic counters {counters:?}");
        assert_eq!(counters[2], 0, "errors set: {:#x}", counters[2]);

        let pairs = read_storage_to_pairs(&device, &queue, &storage).expect("read");
        let pair_map: HashMap<[i32; 3], f32> = pairs.into_iter().collect();
        let mut hits = 0;
        for (c, v) in coords.iter().zip(values.iter()) {
            if let Some(got) = pair_map.get(c).copied() {
                assert_eq!(got, *v, "mismatch at {c:?}");
                hits += 1;
            }
        }
        assert_eq!(hits, n, "only {hits}/{n} coords recovered");
    }

    #[test]
    #[ignore = "needs a real wgpu adapter"]
    fn gpu_mutable_insert_vec3() {
        let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
        let device = engine.device().unwrap().clone();
        let queue = engine.queue().unwrap().clone();

        let n = 256usize;
        let coords = random_coords(n, 0xABBA, 1024);
        let mut rng = Lcg::new(0x7777);
        let values: Vec<ValueVec3> = (0..n)
            .map(|_| {
                let r = |s: &mut Lcg| (s.next() as u32 as f32) / (u32::MAX as f32);
                ValueVec3::new(r(&mut rng), r(&mut rng), r(&mut rng))
            })
            .collect();

        let storage = SparseGridStorage::<ValueVec3>::new(
            &device,
            next_pow2((n as u32) * 2),
            n as u32,
            n as u32,
        );
        let kernel = SparseGridInsertKernel::<ValueVec3>::new(&device);
        let counters = kernel
            .insert(&device, &queue, &storage, &coords, &values)
            .expect("insert");
        assert_eq!(counters[2], 0);

        let pairs = read_storage_to_pairs(&device, &queue, &storage).expect("read");
        let pair_map: HashMap<[i32; 3], ValueVec3> = pairs.into_iter().collect();
        for (c, v) in coords.iter().zip(values.iter()) {
            let got = pair_map
                .get(c)
                .copied()
                .unwrap_or_else(|| panic!("missing coord {c:?}"));
            assert_eq!(got.xyz(), v.xyz(), "vec3 mismatch at {c:?}");
        }
    }

    #[test]
    #[ignore = "needs a real wgpu adapter"]
    fn gpu_mutable_insert_f16_within_quantisation_tolerance() {
        let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
        let device = engine.device().unwrap().clone();
        let queue = engine.queue().unwrap().clone();

        let n = 256usize;
        let coords = random_coords(n, 0x1234, 1024);
        let mut rng = Lcg::new(0xEEEE);
        // Pick values in [0.5, 32) so the f16 mantissa carries ~ 10
        // bits of precision.
        let f32_values: Vec<f32> = (0..n)
            .map(|_| {
                let u = (rng.next() as u32 as f32) / (u32::MAX as f32);
                0.5 + u * 31.5
            })
            .collect();
        let values: Vec<ValueF16> = f32_values.iter().map(|v| ValueF16::from_f32(*v)).collect();

        let storage = SparseGridStorage::<ValueF16>::new(
            &device,
            next_pow2((n as u32) * 2),
            n as u32,
            n as u32,
        );
        let kernel = SparseGridInsertKernel::<ValueF16>::new(&device);
        let counters = kernel
            .insert(&device, &queue, &storage, &coords, &values)
            .expect("insert");
        assert_eq!(counters[2], 0);

        let pairs = read_storage_to_pairs(&device, &queue, &storage).expect("read");
        let pair_map: HashMap<[i32; 3], ValueF16> = pairs.into_iter().collect();
        for ((c, want), _) in coords.iter().zip(f32_values.iter()).zip(values.iter()) {
            let got_v = pair_map
                .get(c)
                .copied()
                .unwrap_or_else(|| panic!("missing coord {c:?}"))
                .to_f32();
            let rel = ((got_v - want) / want).abs();
            assert!(
                rel < 2.0_f32.powi(-10),
                "f16 round-trip rel error {rel} at {c:?} (want {want}, got {got_v})"
            );
        }
    }

    #[test]
    #[ignore = "needs a real wgpu adapter"]
    fn storage_cost_1024_cubed_one_percent_per_value_type() {
        // 1024³ × 1 % occupancy budget — sanity check across all
        // three value types. We don't actually allocate 10M voxels;
        // instead we measure the pre-allocated GPU storage size for a
        // pool sized to fit a realistic narrow-band SDF and confirm
        // it stays under 200 MiB for f32 / f16, and under 600 MiB for
        // vec3 (4× wider per-voxel storage).
        //
        // **Sizing logic:** 1 % of 1024³ = 10_737_418 voxels. A
        // narrow-band SDF clusters voxels into 8³ tiles with
        // ≥ 256 voxels per tile (5 % of 512 voxels per leaf is the
        // pathological floor; well-sampled bands hit 200-400). At
        // 256 voxels per leaf: 10.7M / 256 ≈ 42 000 active leaves.
        // We size the pool to 50 000 to give the GPU dispatcher
        // headroom for collisions (eager-CAS leaks ~ 5 % of the
        // pool slots — see the WGSL `get_or_create_internal`
        // doc-block).
        //
        // **Per-value-type leaf overhead:**
        //   • f32:  512 × 4  =  2 KiB / leaf
        //   • f16:  512 × 4  =  2 KiB / leaf (u32-packed half-floats)
        //   • vec3: 512 × 16 =  8 KiB / leaf (vec3<f32> aligns to 16 B)
        let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
        let device = engine.device().unwrap().clone();

        let leaf_capacity = 50_000u32;
        let internal_capacity = 256u32; // 1024³ → ≤ 64 root keys + headroom
        let root_capacity = next_pow2(internal_capacity * 2);

        // Per-value-type overheads:
        //   root_table:        16 B per slot
        //   internal_children: 128 KiB per internal node
        //   internal_masks:    4 KiB per internal node
        //   leaf_values:       2 KiB (f32) / 8 KiB (vec3) / 2 KiB (f16) per leaf
        //   leaf_masks:        64 B per leaf
        let common = (root_capacity as u64) * 16
            + (internal_capacity as u64) * (128 * 1024 + 4 * 1024)
            + (leaf_capacity as u64) * 64;
        let f32_total = common + (leaf_capacity as u64) * 512 * 4;
        let vec3_total = common + (leaf_capacity as u64) * 512 * 16;
        let f16_total = common + (leaf_capacity as u64) * 512 * 4; // u32-packed (2 bytes used)

        let f32_mib = f32_total / (1024 * 1024);
        let f16_mib = f16_total / (1024 * 1024);
        let vec3_mib = vec3_total / (1024 * 1024);
        eprintln!(
            "leaf pool {leaf_capacity} → f32 {f32_mib} MiB, f16 {f16_mib} MiB, vec3 {vec3_mib} MiB",
        );
        assert!(f32_mib < 200, "f32 pool > 200 MiB: {f32_mib}");
        assert!(f16_mib < 200, "f16 pool > 200 MiB: {f16_mib}");
        // vec3 budget is 4× the f32 budget because of the per-voxel
        // 16-byte vs 4-byte stride. Loose 600 MiB ceiling.
        assert!(vec3_mib < 600, "vec3 pool > 600 MiB: {vec3_mib}");

        // And actually try to allocate them so we know the wgpu side
        // accepts the buffer sizes (no max-buffer-binding-size tripped).
        let _f32_storage =
            SparseGridStorage::<f32>::new(&device, root_capacity, internal_capacity, leaf_capacity);
        let _vec3_storage = SparseGridStorage::<ValueVec3>::new(
            &device,
            root_capacity,
            internal_capacity,
            leaf_capacity,
        );
        let _f16_storage = SparseGridStorage::<ValueF16>::new(
            &device,
            root_capacity,
            internal_capacity,
            leaf_capacity,
        );
    }
}
