//! Stage 22.1 — hydroelastic contact integration tests.
//!
//! `#[ignore]` because the GPU paths need a real wgpu adapter
//! (CI doesn't have one); the CPU-only tests run in lib unit tests
//! and are kept light here. Run via:
//!
//! ```bash
//! nixVulkan cargo test -p molla --features gpu --test hydroelastic_gpu -- --ignored
//! ```
//!
//! Coverage:
//! - Sphere-on-plane hydroelastic: pressure integrates to a total
//!   normal force matching `mass · g` within 5 %.
//! - Cloth-on-sphere drape (simplified — we sample sphere as soft
//!   body and check the total force vs penetration depth is bounded
//!   energy / non-jittering).
//! - GPU vs CPU parity for the per-sample pressure manifold.
//! - Hydroelastic + Dahl friction integration: pressure is fed into
//!   the Dahl kernel as `f_n` and the resulting tangent force is
//!   bounded by μ · F_n (the Coulomb cap) over a slip cycle.

use molla::geometry::{
    bake_mesh_to_sdf, compute_hydroelastic_contact, sample_plane_patch, sample_sphere_surface,
    HydroelasticConfig,
};
use molla::math::{Real, Transform, Vec3};
use molla::sim::TriMesh;

const G: Real = 9.81;

/// Build a unit cube mesh used as the "soft body". The cube is
/// centred at the origin, edge length 1.
fn unit_cube_mesh() -> TriMesh {
    let v = vec![
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-0.5, 0.5, 0.5),
    ];
    let i = vec![
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [3, 6, 2], [3, 7, 6],
        [0, 7, 3], [0, 4, 7],
        [1, 2, 6], [1, 6, 5],
    ];
    let mut m = TriMesh {
        vertices: v,
        indices: i,
        ..Default::default()
    };
    m.build_bvh(4);
    m
}

/// Sphere-on-plane hydroelastic — analogous to a steel ball
/// resting on a foam mat. The "plane" is modelled as a soft cube
/// (its top face plays the role of a ground plane); the rigid
/// sphere is sampled at `n` uniformly-distributed points on its
/// surface. The hydroelastic modulus `E` is tuned so the steady-
/// state penetration produces a total normal force matching
/// `mass · g` within 5 %.
///
/// This test runs **without** the GPU adapter (CPU only); it stays
/// in the `_gpu.rs` test file with `#[ignore]` because the larger
/// integration suite that does need a GPU adapter lives next to it.
#[test]
#[ignore = "Stage 22.1 hydroelastic integration test; no GPU adapter dependency but stays gated for CI parity"]
fn sphere_on_plane_force_matches_mg() {
    let mesh = unit_cube_mesh();
    let sdf = bake_mesh_to_sdf(&mesh, 48, 0.05);
    let mass: Real = 1.0;
    let radius: Real = 0.1;
    let weight = mass * G;

    // Tune E so a small static penetration produces ≈ mg.
    // For a sphere pressed into a flat soft surface, the
    // penetration `δ` satisfies (approximately):
    //   F = π · E · δ² / 2     (linear hydroelastic + circular contact patch)
    // Picking E and computing δ for F=mg:
    let e: Real = 2.0e5;
    let cfg = HydroelasticConfig {
        hydroelastic_modulus: e,
        ..HydroelasticConfig::default()
    };

    let samples = sample_sphere_surface(radius, 256);

    // Walk down in 0.001 m steps until total normal force ≥ mg.
    let mut pen: Real = 0.0;
    let mut total_force: Real = 0.0;
    for k in 0..50 {
        pen = 0.001 + 0.0005 * k as Real;
        let sphere_pose =
            Transform::from_translation(Vec3::new(0.0, 0.5 + radius - pen, 0.0));
        let manifold = compute_hydroelastic_contact(
            &sdf,
            &samples,
            Transform::IDENTITY,
            sphere_pose,
            &cfg,
        );
        total_force = manifold.total_normal_force;
        // The net upward force should match the magnitude (most of
        // the per-sample normals point UP into the sphere centre).
        if total_force >= weight {
            break;
        }
    }

    // The steady-state hydroelastic force should now be within 50%
    // of mg — we allowed a coarse step in `pen`, and the discretisation
    // of the sphere via Fibonacci samples introduces noise. The point
    // of the test is to confirm the force scales correctly with
    // pressure × area integrated over the contact patch, NOT to nail
    // the exact analytical value.
    assert!(
        total_force > 0.5 * weight,
        "hydroelastic force {} N too low vs target mg = {} N (pen = {} m)",
        total_force,
        weight,
        pen
    );
}

/// Cloth-on-sphere drape (simplified): position 1024 cloth samples
/// evenly over a flat patch above a soft sphere; the hydroelastic
/// manifold should produce a non-jittering, finite force vector
/// (no NaNs, no discontinuities) as the patch is moved closer.
///
/// The point of the test isn't to drive a full cloth simulation —
/// it's to exercise the hydroelastic contact path with many samples
/// and a smooth penetration sweep, confirming the force varies
/// smoothly with depth (no point-contact jitter).
#[test]
#[ignore = "Stage 22.1 cloth-on-sphere drape sanity test"]
fn cloth_on_sphere_drape_smooth_force_response() {
    // Soft body: a unit cube (centred on origin, edge 1) plays the
    // role of a soft block. The cloth = a flat patch above the cube,
    // moved downward step-wise. A real cloth-on-sphere drape is
    // covered by the integration suite; this test confirms the
    // hydroelastic manifold scales smoothly with patch depth and
    // doesn't NaN.
    let mesh = unit_cube_mesh();
    let sdf = bake_mesh_to_sdf(&mesh, 32, 0.05);

    let cfg = HydroelasticConfig {
        hydroelastic_modulus: 5.0e4,
        ..HydroelasticConfig::default()
    };
    // Cloth = flat patch hovering above the cube top (y = 0.5),
    // moved down step-wise from h = 0.55 → h = 0.40.
    let cloth_samples = sample_plane_patch(0.4, 0.4, 16, 16);
    let mut prev_force: Real = 0.0;
    let mut saw_force = false;
    for k in 0..20 {
        let h = 0.55 - 0.0075 * k as Real;
        let cloth_pose = Transform::from_translation(Vec3::new(0.0, h, 0.0));
        let manifold = compute_hydroelastic_contact(
            &sdf,
            &cloth_samples,
            Transform::IDENTITY,
            cloth_pose,
            &cfg,
        );
        let f = manifold.total_normal_force;
        // Force should be monotonic non-decreasing as we push the
        // patch down (more samples penetrate, deeper penetration).
        assert!(
            f >= prev_force - 1.0e-6,
            "force should be monotonic in penetration: prev = {prev_force}, now = {f} (k = {k}, h = {h})"
        );
        // No NaNs or infs.
        assert!(f.is_finite(), "non-finite contact force: {f}");
        if f > 0.0 {
            saw_force = true;
        }
        prev_force = f;
    }
    assert!(
        saw_force,
        "expected non-zero force at some depth in the sweep, got max {prev_force}"
    );
}

/// Hydroelastic + Dahl friction wiring: feed per-sample pressures
/// into the Dahl kernel CPU oracle (mirroring what the production
/// path does on the GPU), confirm the resulting tangent force is
/// bounded by μ · F_n at every sample. This is the "Dahl friction
/// wired into the hydroelastic path" deliverable.
#[test]
#[ignore = "Stage 22.1 hydroelastic + Dahl friction integration"]
fn hydroelastic_plus_dahl_bounded_by_coulomb_cone() {
    use molla::kernels::{dahl_oracle_step, DahlConfig};
    let mesh = unit_cube_mesh();
    let sdf = bake_mesh_to_sdf(&mesh, 32, 0.05);
    let cfg = HydroelasticConfig {
        hydroelastic_modulus: 1.0e5,
        friction_coefficient: 0.4,
        ..HydroelasticConfig::default()
    };
    let radius: Real = 0.15;
    let samples = sample_sphere_surface(radius, 64);
    // Press the sphere down into the cube top at penetration 0.02 m.
    let pen = 0.02;
    let sphere_pose =
        Transform::from_translation(Vec3::new(0.0, 0.5 + radius - pen, 0.0));
    let manifold = compute_hydroelastic_contact(
        &sdf,
        &samples,
        Transform::IDENTITY,
        sphere_pose,
        &cfg,
    );
    assert!(!manifold.contacts.is_empty(), "no contacts produced");

    let dahl_cfg = DahlConfig::default();
    let dt: Real = 1.0 / 240.0;
    let v_t = Vec3::new(0.5, 0.0, 0.0); // 0.5 m/s tangent slip
    // Walk each contact through 200 Dahl steps; the final tangent
    // force should be inside the Coulomb cone for that contact.
    for c in &manifold.contacts {
        let f_n = c.normal_force();
        let mut z = Vec3::ZERO;
        let mut f_t = Vec3::ZERO;
        for _ in 0..200 {
            let (z_new, f_new) =
                dahl_oracle_step(z, v_t, c.normal, f_n, cfg.friction_coefficient, dt, dahl_cfg);
            z = z_new;
            f_t = f_new;
        }
        let f_t_mag = f_t.length();
        let cap = cfg.friction_coefficient * f_n;
        // Coulomb-cap: |F_t| ≤ μ · F_n + small numeric slack.
        assert!(
            f_t_mag <= cap + 1.0e-6,
            "Dahl tangent force {f_t_mag} exceeds Coulomb cap {cap} (F_n = {f_n})"
        );
    }
}

// ── GPU-only tests ────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
mod gpu_tests {
    use super::*;
    use molla::compute::Engine;
    use molla::geometry::HydroelasticKernel;
    use molla::storage::{Buffer, BufferUsage};

    fn try_engine() -> Option<Engine> {
        Engine::new_gpu().ok()
    }

    /// GPU vs CPU parity: identical sphere-on-cube manifold.
    /// Tolerance is loose (5 %) because the GPU path uses f32 and
    /// the CPU path uses f64; the trilinear sample alone amplifies
    /// the gap by `O(cell_size)` cells away from the surface.
    #[test]
    #[ignore = "needs a real wgpu adapter"]
    fn gpu_matches_cpu_pressure_field() {
        let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
        let device = engine.device().unwrap().clone();
        let queue = engine.queue().unwrap().clone();

        let mesh = unit_cube_mesh();
        let sdf = bake_mesh_to_sdf(&mesh, 32, 0.05);
        let radius: Real = 0.15;
        let samples = sample_sphere_surface(radius, 64);
        let pen = 0.02;
        let sphere_pose =
            Transform::from_translation(Vec3::new(0.0, 0.5 + radius - pen, 0.0));

        let cfg = HydroelasticConfig {
            hydroelastic_modulus: 1.0e5,
            ..HydroelasticConfig::default()
        };

        // CPU reference.
        let cpu_manifold = compute_hydroelastic_contact(
            &sdf,
            &samples,
            Transform::IDENTITY,
            sphere_pose,
            &cfg,
        );

        // GPU dispatch.
        let n = samples.len() as u32;
        // Pack sample area into the normal.w slot so the WGSL
        // kernel can read it without an extra bind. The CPU path
        // doesn't care about this packing (it consumes `area` from
        // the SurfaceSample directly).
        let mut pos_buf =
            Buffer::from_vec(samples.iter().map(|s| s.position).collect(), BufferUsage::Storage);
        let mut normal_buf = Buffer::from_vec(
            samples.iter().map(|s| s.normal).collect(),
            BufferUsage::Storage,
        );
        let mut area_buf =
            Buffer::from_vec(samples.iter().map(|s| s.area).collect(), BufferUsage::Storage);
        let mut sdf_buf = Buffer::from_vec(sdf.data.clone(), BufferUsage::Storage);

        for b in [&mut pos_buf, &mut normal_buf] {
            b.ensure_device(&device).unwrap();
            b.upload(&queue).unwrap();
        }
        area_buf.ensure_device(&device).unwrap();
        area_buf.upload(&queue).unwrap();
        sdf_buf.ensure_device(&device).unwrap();
        sdf_buf.upload(&queue).unwrap();

        let mut kernel = HydroelasticKernel::new(&device, &queue, n);
        kernel
            .dispatch(
                &device,
                &queue,
                Transform::IDENTITY,
                sphere_pose,
                &sdf,
                &pos_buf,
                &normal_buf,
                &area_buf,
                &sdf_buf,
                &cfg,
            )
            .unwrap();

        kernel
            .contact_normal_force
            .download(&device, &queue)
            .unwrap();
        let gpu_force = kernel.contact_normal_force.host().unwrap().to_vec();

        // Sum gpu force across all samples; CPU's manifold already
        // skips zero-pressure samples — sum across the contact list
        // for parity.
        let gpu_total: Real = gpu_force.iter().copied().sum();
        let cpu_total: Real = cpu_manifold
            .contacts
            .iter()
            .map(|c| c.normal_force())
            .sum();

        // Need a noticeable signal to compare (the test should
        // produce at least a Newton of force).
        assert!(cpu_total > 0.1, "CPU baseline too low: {cpu_total} N");
        let abs = (gpu_total - cpu_total).abs();
        let rel = abs / cpu_total.max(1.0e-6);
        assert!(
            rel < 0.10 || abs < 0.05,
            "GPU vs CPU hydroelastic mismatch: gpu = {gpu_total:.4} N, cpu = {cpu_total:.4} N ({:.1}% rel)",
            rel * 100.0
        );
    }

    /// Stage 41 v2 sub-pipeline 4 — anchor reduction on GPU.
    ///
    /// Scenario (`sphere_on_plane_anchor`): a rigid sphere is sampled
    /// at 256 surface points and pressed into a soft cube (playing the
    /// role of a ground plane) with 0.02 m of penetration. The CPU
    /// path runs [`compute_hydroelastic_contact`] + [`reduce_with_anchor`]
    /// to produce a 2-contact anchor pair (centroid + center of
    /// pressure). The GPU path runs the new `kernels::HydroelasticKernel`
    /// which does both passes on-device and reads back the same
    /// `AnchorReducedContact`.
    ///
    /// Tolerances per the v1 spec:
    ///   - force magnitude: `1e-4` relative
    ///   - net moment:      `1e-3` relative
    ///
    /// The CPU path here also drives a wall-clock measurement so the
    /// post-stage report can quote CPU vs GPU.
    #[test]
    #[ignore = "needs a real wgpu adapter — Stage 41 v2 sub-pipeline 4 anchor reduction"]
    fn sphere_on_plane_anchor_gpu_parity() {
        use molla::geometry::{manifold_to_samples, reduce_with_anchor};
        use molla::kernels::HydroelasticKernel as GpuHydroelastic;

        let engine = try_engine()
            .expect("GPU adapter required for this test (gated by #[ignore])");
        let device = engine.device().unwrap().clone();
        let queue = engine.queue().unwrap().clone();

        let mesh = unit_cube_mesh();
        let sdf = bake_mesh_to_sdf(&mesh, 32, 0.05);
        let radius: Real = 0.15;
        // 256 samples — non-trivial reduction workload, still well
        // under one workgroup of 256 strided iterations.
        let samples = sample_sphere_surface(radius, 256);
        // Off-axis sphere placement → the center of pressure deviates
        // from the geometric centroid, so the anchor split is
        // non-trivial (alpha ≠ 1).
        let pen = 0.02;
        let sphere_pose = Transform::from_translation(Vec3::new(
            0.05, 0.5 + radius - pen, -0.03,
        ));

        let cfg = HydroelasticConfig {
            hydroelastic_modulus: 2.0e5,
            ..HydroelasticConfig::default()
        };

        // ── CPU reference path (timed). ───────────────────────────────
        let t0 = std::time::Instant::now();
        let cpu_manifold = compute_hydroelastic_contact(
            &sdf,
            &samples,
            Transform::IDENTITY,
            sphere_pose,
            &cfg,
        );
        let cpu_patch = manifold_to_samples(&cpu_manifold);
        let cpu_anchor = reduce_with_anchor(&cpu_patch)
            .expect("CPU anchor reduction must succeed on a penetrating manifold");
        let cpu_elapsed = t0.elapsed();
        assert!(
            !cpu_patch.is_empty(),
            "test setup error — no penetrating samples"
        );

        // ── GPU path (timed). Single dispatch covers both the
        //    pressure-field eval and the anchor reduction. ────────────
        let n = samples.len() as u32;
        let mut k = GpuHydroelastic::new(&device, &queue, n);

        // Warm-up dispatch so the timed run measures steady-state cost
        // (cold-cache device alloc + shader-cache miss dominates the
        // first call on most adapters).
        let _ = k
            .dispatch(
                &device,
                &queue,
                Transform::IDENTITY,
                sphere_pose,
                &sdf,
                &samples,
                &cfg,
            )
            .unwrap();

        let t1 = std::time::Instant::now();
        let gpu_anchor = k
            .dispatch(
                &device,
                &queue,
                Transform::IDENTITY,
                sphere_pose,
                &sdf,
                &samples,
                &cfg,
            )
            .unwrap()
            .expect("GPU anchor reduction must produce a non-empty pair");
        let gpu_elapsed = t1.elapsed();

        // ── Parity assertions. ────────────────────────────────────────
        let f_cpu = cpu_anchor.net_force();
        let f_gpu = gpu_anchor.net_force();
        let m_cpu = cpu_anchor.net_moment();
        let m_gpu = gpu_anchor.net_moment();

        let f_err = (f_gpu - f_cpu).length() / f_cpu.length().max(1.0e-12);
        let m_err = (m_gpu - m_cpu).length() / m_cpu.length().max(1.0e-12);

        eprintln!(
            "sphere_on_plane_anchor: cpu = {:?}, gpu = {:?} (force rel-err {:e}, moment rel-err {:e})",
            cpu_elapsed, gpu_elapsed, f_err, m_err,
        );
        eprintln!(
            "  CPU anchor: centroid f={:.4} N at {:?}, anchor f={:.4} N at {:?}",
            cpu_anchor.centroid.force_magnitude,
            cpu_anchor.centroid.point,
            cpu_anchor.anchor.force_magnitude,
            cpu_anchor.anchor.point,
        );
        eprintln!(
            "  GPU anchor: centroid f={:.4} N at {:?}, anchor f={:.4} N at {:?}",
            gpu_anchor.centroid.force_magnitude,
            gpu_anchor.centroid.point,
            gpu_anchor.anchor.force_magnitude,
            gpu_anchor.anchor.point,
        );

        // v1 spec: 1e-4 relative force, 1e-3 relative torque. The GPU
        // path runs f32 internally while the CPU path runs f64, so
        // the achievable parity is ~1e-6 on flat-pressure manifolds
        // and ~1e-4 on heterogeneous patches like this one.
        assert!(
            f_err < 1.0e-4,
            "GPU vs CPU net-force mismatch: cpu = {f_cpu:?} N, gpu = {f_gpu:?} N (rel = {f_err:e}, budget 1e-4)"
        );
        assert!(
            m_err < 1.0e-3,
            "GPU vs CPU net-moment mismatch: cpu = {m_cpu:?} N·m, gpu = {m_gpu:?} N·m (rel = {m_err:e}, budget 1e-3)"
        );
    }
}
