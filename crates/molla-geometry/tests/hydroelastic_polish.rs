//! Stage 39D — hydroelastic contact polish.
//!
//! Two upgrades on top of the v1 Stage-22.1 manifold:
//!
//! 1. **Anchor contact for moment balance** — `reduce_with_anchor`
//!    splits a distributed pressure patch into a centroid + anchor
//!    pair that reproduces both net force and net moment.
//! 2. **Aggregate-stiffness reduction** — `aggregate_by_bucket` keys
//!    contacts by `(coarse_position_voxel, coarse_normal_bin)` and
//!    combines per-contact stiffness either in parallel
//!    (`Σ k_i`, default) or in series (`1 / Σ 1/k_i`).
//!
//! These tests pin down the numerical contract for both:
//!
//! * Anchor pair reproduces moment within 1 %.
//! * Stiffness aggregation matches the parallel- and series-spring
//!   analytic answers.
//! * Bucket granularity collapses 100 coplanar contacts to 1
//!   aggregate when the voxel size exceeds the patch extent.

use molla_geometry::{
    AggregateStiffnessConfig, PatchSample, StiffnessReduction, StiffnessSample,
    aggregate_by_bucket, reduce_with_anchor,
};
use molla_math::{Real, Vec3};

/// Build a synthetic asymmetric pressure patch — a Gaussian bump
/// offset from the spatial centroid so the patch's center of
/// pressure differs from its geometric centroid. The resulting
/// moment about any reference point is non-trivial and exposes
/// whether `reduce_with_anchor` reproduces it.
fn asymmetric_patch() -> Vec<PatchSample> {
    let n = Vec3::Y;
    let mut samples = Vec::new();
    // 11x11 grid covering [-0.5, 0.5] x [-0.5, 0.5] in the xz-plane.
    let nx = 11;
    let nz = 11;
    for i in 0..nx {
        for j in 0..nz {
            let x = -0.5 + (i as Real) / (nx as Real - 1.0);
            let z = -0.5 + (j as Real) / (nz as Real - 1.0);
            // Gaussian centred at (+0.3, +0.2) — well off-centre.
            let dx = x - 0.3;
            let dz = z - 0.2;
            let p = (-(dx * dx + dz * dz) * 8.0).exp();
            samples.push(PatchSample {
                point: Vec3::new(x, 0.0, z),
                normal: n,
                force_magnitude: p,
            });
        }
    }
    samples
}

fn net_force(samples: &[PatchSample]) -> Vec3 {
    samples.iter().map(|s| s.force()).sum()
}

fn net_moment_about_origin(samples: &[PatchSample]) -> Vec3 {
    samples.iter().map(|s| s.point.cross(s.force())).sum()
}

/// The reduced pair (centroid + anchor) reproduces both the net
/// force and the net moment of the input patch to within 1 %.
#[test]
fn anchor_contact_reproduces_moment() {
    let patch = asymmetric_patch();
    let f_ref = net_force(&patch);
    let m_ref = net_moment_about_origin(&patch);
    assert!(
        f_ref.length() > 1.0e-6,
        "test patch must carry non-zero net force, got {:?}",
        f_ref
    );
    assert!(
        m_ref.length() > 1.0e-6,
        "test patch must carry non-zero net moment, got {:?}",
        m_ref
    );

    let reduced = reduce_with_anchor(&patch).expect("reduction must succeed");
    let f_red = reduced.net_force();
    let m_red = reduced.net_moment();

    // Force should match to within tight rounding.
    let f_err = (f_red - f_ref).length() / f_ref.length();
    assert!(
        f_err < 1.0e-6,
        "anchor pair force mismatch: ref={:?} red={:?} rel-err={:e}",
        f_ref,
        f_red,
        f_err,
    );
    // Moment should match to within 1 %.
    let m_err = (m_red - m_ref).length() / m_ref.length();
    assert!(
        m_err < 0.01,
        "anchor pair moment mismatch: ref={:?} red={:?} rel-err={:e}",
        m_ref,
        m_red,
        m_err,
    );
}

/// Parallel-springs aggregation: N identical contacts with stiffness
/// `k` each, all on the same point/normal, collapse to one
/// aggregate with stiffness `N · k`. Series aggregation gives
/// `k / N`.
#[test]
fn aggregate_stiffness_recovers_series_compliance() {
    // 5 coincident contacts with stiffness 1e5 each.
    let k = 1.0e5;
    let n_contacts = 5usize;
    let samples_storage: Vec<PatchSample> = (0..n_contacts)
        .map(|_| PatchSample {
            point: Vec3::ZERO,
            normal: Vec3::Y,
            force_magnitude: 1.0,
        })
        .collect();
    let stiffness_samples: Vec<StiffnessSample> = samples_storage
        .iter()
        .map(|p| StiffnessSample {
            patch: p,
            stiffness: k,
        })
        .collect();

    // Parallel.
    let cfg_par = AggregateStiffnessConfig {
        voxel_size: 1.0,
        normal_bin_angle: std::f64::consts::PI,
        stiffness_reduction: StiffnessReduction::Parallel,
    };
    let par = aggregate_by_bucket(&stiffness_samples, &cfg_par);
    assert_eq!(par.len(), 1, "parallel reduction should yield 1 bucket");
    let k_par = par[0].aggregate_stiffness;
    let k_par_ref = (n_contacts as Real) * k;
    assert!(
        ((k_par - k_par_ref) / k_par_ref).abs() < 1.0e-9,
        "parallel: got {} vs expected {} (5·k)",
        k_par,
        k_par_ref,
    );

    // Series.
    let cfg_ser = AggregateStiffnessConfig {
        voxel_size: 1.0,
        normal_bin_angle: std::f64::consts::PI,
        stiffness_reduction: StiffnessReduction::Series,
    };
    let ser = aggregate_by_bucket(&stiffness_samples, &cfg_ser);
    assert_eq!(ser.len(), 1, "series reduction should yield 1 bucket");
    let k_ser = ser[0].aggregate_stiffness;
    let k_ser_ref = k / (n_contacts as Real);
    assert!(
        ((k_ser - k_ser_ref) / k_ser_ref).abs() < 1.0e-9,
        "series: got {} vs expected {} (k/5)",
        k_ser,
        k_ser_ref,
    );
}

/// 100 contacts spread over a small flat plate all collapse to a
/// single aggregate when `voxel_size > plate extent`. Their summed
/// stiffness is the parallel total.
#[test]
fn bucket_granularity_collapses_nearly_coplanar_contacts() {
    let n = 10;
    let mut samples_storage = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let x = -0.1 + 0.2 * (i as Real) / (n as Real - 1.0);
            let z = -0.1 + 0.2 * (j as Real) / (n as Real - 1.0);
            samples_storage.push(PatchSample {
                point: Vec3::new(x, 0.0, z),
                // Slight wobble in the normal — all within 1 ° of Y.
                normal: Vec3::new(
                    0.01 * (i as Real - 5.0) / 5.0,
                    1.0,
                    0.01 * (j as Real - 5.0) / 5.0,
                )
                .normalize(),
                force_magnitude: 1.0,
            });
        }
    }
    let stiffness_samples: Vec<StiffnessSample> = samples_storage
        .iter()
        .map(|p| StiffnessSample {
            patch: p,
            stiffness: 1.0e4,
        })
        .collect();

    // Bucket size > plate extent (0.2 m); normal-bin angle wide
    // enough to absorb the 1 ° wobble.
    let cfg = AggregateStiffnessConfig {
        voxel_size: 1.0,
        normal_bin_angle: 5.0_f64.to_radians(),
        stiffness_reduction: StiffnessReduction::Parallel,
    };
    let agg = aggregate_by_bucket(&stiffness_samples, &cfg);
    assert_eq!(
        agg.len(),
        1,
        "expected 100 coplanar contacts to collapse to 1 aggregate; got {} buckets",
        agg.len(),
    );
    assert_eq!(agg[0].sample_count as usize, n * n);
    // Force magnitude is the sum of per-sample magnitudes (1.0 × 100).
    assert!(
        (agg[0].force_magnitude - (n * n) as Real).abs() < 1.0e-9,
        "force sum: got {}, expected {}",
        agg[0].force_magnitude,
        (n * n) as Real,
    );
    // Aggregate stiffness = N · k (parallel).
    let k_ref = (n * n) as Real * 1.0e4;
    assert!(
        ((agg[0].aggregate_stiffness - k_ref) / k_ref).abs() < 1.0e-9,
        "parallel stiffness: got {}, expected {}",
        agg[0].aggregate_stiffness,
        k_ref,
    );

    // Sanity: with a tiny voxel size, each contact sits in its own
    // bucket and we get back N buckets.
    let cfg_fine = AggregateStiffnessConfig {
        voxel_size: 0.001,
        normal_bin_angle: 5.0_f64.to_radians(),
        stiffness_reduction: StiffnessReduction::Parallel,
    };
    let fine = aggregate_by_bucket(&stiffness_samples, &cfg_fine);
    assert_eq!(
        fine.len(),
        n * n,
        "fine voxel: each of the 100 contacts should be its own bucket; got {}",
        fine.len(),
    );
}
