use molla_fem::{KrylovRestartCache, solve_transient_pcg, solve_transient_pcg_restarted};
use molla_math::sparse::BsrMatrix;

fn diag(vals: &[f64]) -> BsrMatrix {
    let triplets: Vec<_> = vals.iter().enumerate().map(|(i, v)| (i, i, *v)).collect();
    BsrMatrix::from_scalar_triplets(vals.len(), vals.len(), &triplets).unwrap()
}

#[test]
fn transient_pcg_restart_cache_recycles_previous_corrections() {
    let m = diag(&[1.0, 1.0, 1.0, 1.0]);
    let k = diag(&[2.0, 4.0, 8.0, 16.0]);
    let dt = 0.1;
    let bcs = [];
    let mut cache = KrylovRestartCache::new(3);
    assert!(cache.is_empty());

    let u0 = vec![0.0; 4];
    let f0 = vec![1.0, 0.5, 0.25, 0.125];
    let first =
        solve_transient_pcg_restarted(&m, &k, &f0, &u0, dt, &bcs, &u0, &mut cache, 100, 1.0e-12);
    assert!(first.residual < 1.0e-10);
    assert_eq!(cache.len(), 1);

    let f1 = vec![1.1, 0.55, 0.275, 0.1375];
    let restarted = solve_transient_pcg_restarted(
        &m, &k, &f1, &first.x, dt, &bcs, &first.x, &mut cache, 100, 1.0e-12,
    );
    let plain = solve_transient_pcg(&m, &k, &f1, &first.x, dt, &bcs, &first.x, 100, 1.0e-12);

    assert!(restarted.residual < 1.0e-10);
    assert_eq!(cache.len(), 2);
    assert_eq!(restarted.x.len(), plain.x.len());
    for (a, b) in restarted.x.iter().zip(plain.x.iter()) {
        assert!((a - b).abs() < 1.0e-10);
    }
    assert!(
        restarted.iters <= plain.iters,
        "restarted PCG should not need more iterations on this repeated diagonal system"
    );
}
