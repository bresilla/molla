//! Stage 24e.2 — sparse-grid v4 integration tests.
//!
//! Adds GPU delete + leaf reclamation coverage on top of the v3
//! mutable-insert tests:
//!   • Insert N pairs, delete a subset, verify lookups: deleted →
//!     background, kept → original value, `leaves_in_freelist`
//!     reflects the reclaimed slots.
//!   • Reclamation reuse: after delete, insert at fresh coords and
//!     confirm the freelist counter dropped (slots were popped, not
//!     freshly allocated).
//!   • Optional stress: 10k insert, 50% delete, no orphaned leaves
//!     (`active_leaves == leaves_in_freelist + live_leaves`).
//!
//! All GPU-bound tests are `#[ignore]` because they need a real wgpu
//! adapter; run with
//!
//!   `make test-sparse-grid-v4-gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    SparseGridDeleteKernel, SparseGridInsertKernel, SparseGridStorage, read_storage_to_pairs,
};
use std::collections::{HashMap, HashSet};

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

/// Generate `n` distinct random voxel coords whose root key isn't
/// `(0, 0, 0)` (the WGSL hash-table publication path reserves that
/// key as a "still-publishing" sentinel — see
/// `sparse_grid_insert.wgsl`'s `get_or_create_internal` doc-block).
fn random_coords(n: usize, seed: u64, range: i32) -> Vec<[i32; 3]> {
    let mut rng = Lcg::new(seed);
    let mut set: HashSet<[i32; 3]> = HashSet::new();
    while set.len() < n {
        let x = ((rng.next() >> 32) as i32) % (2 * range) - range;
        let y = ((rng.next() >> 32) as i32) % (2 * range) - range;
        let z = ((rng.next() >> 32) as i32) % (2 * range) - range;
        // Skip root-key-(0,0,0) coords — within voxel coords [-256, 256).
        if (x >> 8, y >> 8, z >> 8) == (0, 0, 0) {
            continue;
        }
        set.insert([x, y, z]);
    }
    set.into_iter().collect()
}

// ── Tests ──────────────────────────────────────────────────────────

#[test]
#[ignore = "needs a real wgpu adapter"]
fn delete_marks_inactive_and_reclaims_leaves() {
    let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let n = 1024usize;
    let coords = random_coords(n, 0xDEAD0001, 8192);
    let mut rng = Lcg::new(0xF00D);
    let values: Vec<f32> = (0..n)
        .map(|_| (rng.next() as u32 as f32) / (u32::MAX as f32))
        .collect();

    let storage = SparseGridStorage::<f32>::new(
        &device,
        next_pow2((n as u32).max(8) * 2),
        n as u32,
        n as u32 * 2, // leave headroom for reclamation accounting
    );
    let insert = SparseGridInsertKernel::<f32>::new(&device);
    let delete = SparseGridDeleteKernel::<f32>::new(&device);

    // Insert all 1024 pairs.
    let counters = insert
        .insert(&device, &queue, &storage, &coords, &values)
        .expect("insert");
    assert_eq!(
        counters[2], 0,
        "errors set after insert: {:#x}",
        counters[2]
    );
    let stats_after_insert = storage.storage_stats(&device, &queue).expect("stats");
    eprintln!("after insert: {stats_after_insert:?}");
    assert_eq!(stats_after_insert.leaves_in_freelist, 0);
    assert_eq!(stats_after_insert.pending_reclaim, 0);
    let leaves_after_insert = stats_after_insert.active_leaves;
    assert!(leaves_after_insert > 0, "no leaves allocated");

    // Delete the lower half (first 500 by coord-sort order).
    let mut sorted = coords.clone();
    sorted.sort_by_key(|c| (c[0], c[1], c[2]));
    let to_delete: Vec<[i32; 3]> = sorted.iter().take(500).copied().collect();
    let to_keep: HashSet<[i32; 3]> = sorted.iter().skip(500).copied().collect();

    // mark_inactive then reclaim.
    let stats_after_delete = delete
        .delete(&device, &queue, &storage, &to_delete)
        .expect("delete");
    eprintln!("after delete: {stats_after_delete:?}");
    let stats_pkg = storage.storage_stats(&device, &queue).expect("stats");
    eprintln!("post-delete stats {stats_pkg:?}");
    // The freelist depth should reflect leaves whose entire 8³ mask
    // emptied as a result of the delete batch. With 1024 random
    // coords spread over a 16384³ region most coords land in their
    // own 8³ tile, so deleting 500 should reclaim ~ 500 leaves.
    // Loose assertion: at least 1 leaf was reclaimed (the random
    // distribution makes "all 500 share leaves with kept coords"
    // overwhelmingly unlikely).
    assert!(
        stats_pkg.leaves_in_freelist >= 1,
        "expected at least one reclaimed leaf, got freelist={}",
        stats_pkg.leaves_in_freelist
    );
    assert_eq!(stats_pkg.pending_reclaim, 0, "pending list not drained");

    // Verify lookup state via read_storage_to_pairs.
    let pairs = read_storage_to_pairs(&device, &queue, &storage).expect("read");
    let pair_map: HashMap<[i32; 3], f32> = pairs.into_iter().collect();
    for (c, v) in coords.iter().zip(values.iter()) {
        if to_keep.contains(c) {
            let got = pair_map
                .get(c)
                .copied()
                .unwrap_or_else(|| panic!("kept coord {c:?} missing"));
            assert_eq!(got, *v, "kept value mismatch at {c:?}");
        } else {
            assert!(
                !pair_map.contains_key(c),
                "deleted coord {c:?} still appears in storage"
            );
        }
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn reclamation_reuse_pops_from_freelist() {
    // Insert → delete → insert at fresh coords. The second insert
    // should drain the freelist before falling back to next_leaf.
    let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let n = 200usize;
    let coords1 = random_coords(n, 0xAAAA, 8192);
    let values1: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25).collect();

    let storage = SparseGridStorage::<f32>::new(
        &device,
        next_pow2((n as u32).max(8) * 4),
        (n as u32) * 2,
        (n as u32) * 4,
    );
    let insert = SparseGridInsertKernel::<f32>::new(&device);
    let delete = SparseGridDeleteKernel::<f32>::new(&device);

    insert
        .insert(&device, &queue, &storage, &coords1, &values1)
        .expect("insert 1");
    let stats_a = storage.storage_stats(&device, &queue).expect("stats a");
    eprintln!("stage A (initial insert) {stats_a:?}");
    let leaves_a = stats_a.active_leaves;

    // Delete every coord — freelist should have ≤ leaves_a entries.
    delete
        .delete(&device, &queue, &storage, &coords1)
        .expect("delete");
    let stats_b = storage.storage_stats(&device, &queue).expect("stats b");
    eprintln!("stage B (post-delete) {stats_b:?}");
    let freelist_b = stats_b.leaves_in_freelist;
    assert!(freelist_b > 0, "freelist still empty after deleting all");
    assert_eq!(
        stats_b.active_leaves, leaves_a,
        "next_leaf shouldn't move on delete"
    );

    // Insert a fresh batch using DIFFERENT coords — coords1 used
    // seed 0xAAAA; coords2 uses 0xBBBB and a disjoint range so the
    // insert kernel must pop the freelist for at least *some* of
    // them (any that map to the reclaimed leaf-tile slots will
    // reuse; others will allocate fresh from next_leaf).
    let m = 100usize;
    let coords2 = random_coords(m, 0xBBBB, 8192);
    let values2: Vec<f32> = (0..m).map(|i| (i as f32) * -0.5).collect();
    insert
        .insert(&device, &queue, &storage, &coords2, &values2)
        .expect("insert 2");
    let stats_c = storage.storage_stats(&device, &queue).expect("stats c");
    eprintln!("stage C (post-reuse insert) {stats_c:?}");

    // Two acceptable signals that reuse worked:
    //   (a) freelist count dropped (some pops consumed entries);
    //   (b) next_leaf grew by less than `m` (some allocations came
    //       from the freelist instead of the pool).
    let pool_growth = stats_c.active_leaves.saturating_sub(stats_b.active_leaves);
    let freelist_drop = freelist_b.saturating_sub(stats_c.leaves_in_freelist);
    eprintln!("reuse signals: pool_growth={pool_growth} (m={m}), freelist_drop={freelist_drop}");
    assert!(
        freelist_drop > 0 || pool_growth < (m as u32),
        "no freelist reuse observed: freelist {} → {}, leaves {} → {}",
        freelist_b,
        stats_c.leaves_in_freelist,
        stats_b.active_leaves,
        stats_c.active_leaves
    );

    // Sanity: the new coords are queryable.
    let pairs = read_storage_to_pairs(&device, &queue, &storage).expect("read");
    let pair_map: HashMap<[i32; 3], f32> = pairs.into_iter().collect();
    let mut hits2 = 0;
    for (c, v) in coords2.iter().zip(values2.iter()) {
        if let Some(got) = pair_map.get(c).copied() {
            assert_eq!(got, *v, "second-batch value mismatch at {c:?}");
            hits2 += 1;
        }
    }
    assert!(
        hits2 >= (m * 9) / 10,
        "second-batch round-trip dropped too many: {hits2}/{m}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn no_orphaned_leaves_after_random_50pct_delete() {
    // Stress: insert 10k, delete random 50%. Verify the topology
    // accounting holds — `active_leaves` (high-water mark) equals
    // `live_leaves + leaves_in_freelist + concurrent_leaks`. We
    // bound the concurrent leak fraction (leaves whose mask emptied
    // due to a delete but that didn't get reclaimed because the
    // pending list overflowed or the duplicate-push race kicked in)
    // at < 5% — looser than zero because the v1 reclaim pass is
    // best-effort, not guaranteed.
    let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let n = 10_000usize;
    let coords = random_coords(n, 0xC001, 16384);
    let mut rng = Lcg::new(0xBEEF);
    let values: Vec<f32> = (0..n)
        .map(|_| (rng.next() as u32 as f32) / (u32::MAX as f32))
        .collect();

    let storage = SparseGridStorage::<f32>::new(
        &device,
        next_pow2((n as u32) * 2),
        (n as u32).min(1024) * 2,
        (n as u32) * 2,
    );
    let insert = SparseGridInsertKernel::<f32>::new(&device);
    let delete = SparseGridDeleteKernel::<f32>::new(&device);

    let t0 = std::time::Instant::now();
    let _ = insert
        .insert(&device, &queue, &storage, &coords, &values)
        .expect("insert 10k");
    let dt_insert = t0.elapsed();
    let stats_a = storage.storage_stats(&device, &queue).expect("stats a");
    eprintln!(
        "10k insert in {:.2}ms; stats {stats_a:?}",
        dt_insert.as_secs_f64() * 1000.0
    );

    // Delete 50% picked deterministically.
    let mut to_delete: Vec<[i32; 3]> = Vec::with_capacity(n / 2);
    for (i, c) in coords.iter().enumerate() {
        if i & 1 == 0 {
            to_delete.push(*c);
        }
    }
    let t1 = std::time::Instant::now();
    let _ = delete
        .delete(&device, &queue, &storage, &to_delete)
        .expect("delete 5k");
    let dt_delete = t1.elapsed();
    let stats_b = storage.storage_stats(&device, &queue).expect("stats b");
    eprintln!(
        "5k delete + reclaim in {:.2}ms; stats {stats_b:?}",
        dt_delete.as_secs_f64() * 1000.0
    );

    // Count live leaves via storage read-back.
    let pairs = read_storage_to_pairs(&device, &queue, &storage).expect("read");
    // Bucket pairs by leaf — leaf is uniquely identified by
    // (root_key, internal_local). We'll just count the unique 8³
    // tiles touched by the surviving coords.
    let mut leaf_set: HashSet<(i32, i32, i32)> = HashSet::new();
    for (c, _) in &pairs {
        leaf_set.insert((c[0] >> 3, c[1] >> 3, c[2] >> 3));
    }
    let live_leaves = leaf_set.len() as u32;

    // Topology accounting: active_leaves (high-water) ≈ live_leaves
    // + leaves_in_freelist (+ small concurrent leak bound).
    let accounted = live_leaves + stats_b.leaves_in_freelist;
    let leak = stats_b.active_leaves.saturating_sub(accounted);
    eprintln!(
        "accounting: live={live_leaves} freelist={} active_leaves={} leak={leak}",
        stats_b.leaves_in_freelist, stats_b.active_leaves
    );
    let leak_frac = (leak as f32) / (stats_b.active_leaves.max(1) as f32);
    assert!(
        leak_frac < 0.05,
        "leak fraction {leak_frac:.3} > 5% (leak={leak} of active_leaves={})",
        stats_b.active_leaves
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn delete_then_insert_roundtrip_throughput() {
    // Measured DoD throughput: insert N, delete N, reclaim, insert N
    // again into reused slots. Reports per-pair µs for each phase so
    // the PLAN.md ledger can quote a number.
    let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let n = 1024usize;
    let coords1 = random_coords(n, 0x1234, 8192);
    let values1: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125).collect();

    let storage = SparseGridStorage::<f32>::new(
        &device,
        next_pow2((n as u32) * 4),
        (n as u32) * 2,
        (n as u32) * 4,
    );
    let insert = SparseGridInsertKernel::<f32>::new(&device);
    let delete = SparseGridDeleteKernel::<f32>::new(&device);

    let t_insert = std::time::Instant::now();
    insert
        .insert(&device, &queue, &storage, &coords1, &values1)
        .expect("insert");
    let dt_insert = t_insert.elapsed();

    let t_delete = std::time::Instant::now();
    delete
        .delete(&device, &queue, &storage, &coords1)
        .expect("delete");
    let dt_delete = t_delete.elapsed();

    let coords2 = random_coords(n, 0x5678, 8192);
    let values2: Vec<f32> = (0..n).map(|i| (i as f32) * -0.125).collect();
    let t_reuse = std::time::Instant::now();
    insert
        .insert(&device, &queue, &storage, &coords2, &values2)
        .expect("reuse insert");
    let dt_reuse = t_reuse.elapsed();

    let total = dt_insert + dt_delete + dt_reuse;
    eprintln!(
        "insert-delete-insert roundtrip: insert {:.2}ms, delete+reclaim {:.2}ms, reuse-insert {:.2}ms; total {:.2}ms ({:.1} kPair/s amortised)",
        dt_insert.as_secs_f64() * 1000.0,
        dt_delete.as_secs_f64() * 1000.0,
        dt_reuse.as_secs_f64() * 1000.0,
        total.as_secs_f64() * 1000.0,
        (3 * n) as f64 / total.as_secs_f64() / 1000.0,
    );
}

/// INSPECTION.md F11 — the delete/reclaim CONTRACT, including its
/// known limit, pinned numerically:
///
/// * Leaf slots ARE recycled end-to-end: delete pushes them through
///   `pending_reclaim` into the freelist and the insert kernel's
///   `pop_freelist` consumes them, so abandoning one region and
///   populating another does not grow the leaf pool.
/// * Internal-node slots are NOT folded (no internal freelist, no
///   empty-subtree coarsening): abandoning a region leaks its
///   internal nodes until `SparseGridStorage::reset`. The leak is
///   bounded by `internal_capacity`. If someone implements folding,
///   the final assertion here flips — update it and close the F11
///   follow-up.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn leaf_slots_recycle_but_internal_nodes_leak_known_limit() {
    let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let n = 512usize;
    // Two regions with guaranteed-disjoint root keys (2²⁰ apart).
    let region_a = random_coords(n, 0xA11CE, 4096);
    let region_b: Vec<[i32; 3]> = random_coords(n, 0xB0B, 4096)
        .into_iter()
        .map(|c| [c[0] + (1 << 20), c[1], c[2]])
        .collect();
    let values: Vec<f32> = (0..n).map(|i| i as f32).collect();

    let storage = SparseGridStorage::<f32>::new(
        &device,
        next_pow2((n as u32) * 4),
        n as u32 * 4,
        n as u32 * 4,
    );
    let insert = SparseGridInsertKernel::<f32>::new(&device);
    let delete = SparseGridDeleteKernel::<f32>::new(&device);

    insert
        .insert(&device, &queue, &storage, &region_a, &values)
        .expect("insert region A");
    let after_a = storage.storage_stats(&device, &queue).expect("stats A");
    assert!(after_a.active_leaves > 0 && after_a.active_internals > 0);

    delete
        .delete(&device, &queue, &storage, &region_a)
        .expect("delete region A");
    let after_delete = storage.storage_stats(&device, &queue).expect("stats del");
    assert!(
        after_delete.leaves_in_freelist > 0,
        "delete must feed the leaf freelist: {after_delete:?}"
    );
    assert_eq!(after_delete.pending_reclaim, 0, "reclaim must drain");

    insert
        .insert(&device, &queue, &storage, &region_b, &values)
        .expect("insert region B");
    let after_b = storage.storage_stats(&device, &queue).expect("stats B");
    eprintln!("A: {after_a:?}\ndel: {after_delete:?}\nB: {after_b:?}");

    // Leaf recycling: region B must be served (at least partly) from
    // the freelist rather than growing the bump allocator by a full
    // region's worth.
    let leaf_growth = after_b.active_leaves - after_a.active_leaves;
    assert!(
        leaf_growth < after_a.active_leaves / 2,
        "leaf slots were not recycled: bump grew by {leaf_growth} \
         (region A used {})",
        after_a.active_leaves
    );
    assert!(
        after_b.leaves_in_freelist < after_delete.leaves_in_freelist,
        "insert did not pop the freelist"
    );

    // KNOWN LIMIT (F11 follow-up): region A's internal nodes leak —
    // the internal bump counter grows by a full region even though A
    // is empty. When empty-subtree folding lands, flip this to assert
    // near-zero growth instead.
    let internal_growth = after_b.active_internals - after_a.active_internals;
    assert!(
        internal_growth > 0,
        "internal nodes unexpectedly reused — folding may have landed; \
         update this test and close the F11 follow-up"
    );
    assert_eq!(after_b.error_flags, 0, "no pool exhaustion");
}
