//! Stage 41 v2 — APIC FEM-MPM GPU parity tests.
//!
//! Deliverable (per the brief): 100 particles falling onto a 4×4×4 tet
//! block, 10 steps, GPU vs CPU particle positions agree within `1e-4`
//! relative. Bundled with a perf-telemetry test that prints CPU and
//! GPU wall-clocks (no assertion — for the post-stage summary).
//!
//! All tests are `#[ignore]` because they need a real wgpu adapter.
//! Run with:
//!
//! ```sh
//! nixVulkan cargo test -p molla-fem --features gpu --test fem_apic_gpu -- --ignored
//! ```

#![cfg(feature = "gpu")]

use std::sync::Mutex;
use std::time::Instant;

use molla_compute::Engine;
use molla_fem::apic_coupling_gpu::ApicGpuState;
use molla_fem::{ApicParticle, BoundaryCondition, FemMesh, apic_coupling::ApicHybrid};
use molla_math::{Mat3, Real, Vec3};

/// Serialise wgpu adapter lifetime across tests in this binary.
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Build the test scenario: 100 particles dropping into a 4×4×4 unit
/// tet grid with no-slip walls. Returns the mesh, the particle set,
/// and the BC list. Particle positions are deterministically scattered
/// in the upper half of the box to dodge symmetric trivial cases.
fn build_scenario() -> (FemMesh, Vec<ApicParticle>, Vec<BoundaryCondition>) {
    let mesh = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);
    let mut particles = Vec::with_capacity(100);
    let mut s: u64 = 0xA1B2_C3D4_E5F6_0789;
    let next = |s: &mut u64| -> Real {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as Real) / ((1u64 << 31) as Real)
    };
    for _ in 0..100 {
        // Sample in upper-half box, slightly inset from walls.
        let x = Vec3::new(
            0.10 + 0.80 * next(&mut s),
            0.10 + 0.80 * next(&mut s),
            0.55 + 0.40 * next(&mut s),
        );
        particles.push(ApicParticle {
            x,
            v: Vec3::ZERO,
            c: Mat3::ZERO,
            mass: 1.0,
        });
    }
    // No-slip BCs on every boundary vertex.
    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        let on_b = p.x <= 1e-9
            || p.x >= 1.0 - 1e-9
            || p.y <= 1e-9
            || p.y >= 1.0 - 1e-9
            || p.z <= 1e-9
            || p.z >= 1.0 - 1e-9;
        if on_b {
            bcs.push(BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0, 0.0, 0.0],
            });
        }
    }
    (mesh, particles, bcs)
}

#[test]
#[ignore = "needs a real wgpu adapter — run with --features gpu --test fem_apic_gpu -- --ignored"]
fn apic_gpu_matches_cpu_falling_particles() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let (mesh, particles, bcs) = build_scenario();
    let dt: Real = 1.0 / 60.0;
    let viscosity: Real = 0.5;
    let gravity = Vec3::new(0.0, 0.0, -2.0);
    let steps = 10;

    // ── CPU oracle ─────────────────────────────────────────────────
    let mut cpu = ApicHybrid::new(mesh.clone(), particles.clone());
    cpu.gravity = gravity;
    let cpu_start = Instant::now();
    for _ in 0..steps {
        cpu.step(dt, viscosity, &bcs);
    }
    let cpu_elapsed = cpu_start.elapsed();

    // ── GPU under test ─────────────────────────────────────────────
    let engine = Engine::new_gpu().expect("gpu adapter");
    let mut gpu_hybrid = ApicHybrid::new(mesh, particles);
    gpu_hybrid.gravity = gravity;
    // Build GPU state once and reuse — measures the steady-state path,
    // not the one-shot construction cost.
    let state = ApicGpuState::new(&engine, &gpu_hybrid, &bcs).expect("ApicGpuState::new");
    let gpu_start = Instant::now();
    for _ in 0..steps {
        state
            .step(&gpu_hybrid, dt as f32, viscosity as f32)
            .expect("gpu step");
    }
    let gpu_positions = state.read_positions().expect("read positions");
    let gpu_elapsed = gpu_start.elapsed();

    // ── Parity ─────────────────────────────────────────────────────
    assert_eq!(gpu_positions.len(), cpu.particles.len());
    let mut max_rel: Real = 0.0;
    let mut max_abs: Real = 0.0;
    let mut argmax_idx = 0usize;
    for (i, (gx, cp)) in gpu_positions.iter().zip(cpu.particles.iter()).enumerate() {
        let diff = (*gx - cp.x).length();
        let denom = cp.x.length().max(1e-9);
        let rel = diff / denom;
        if rel > max_rel {
            max_rel = rel;
            max_abs = diff;
            argmax_idx = i;
        }
    }
    eprintln!(
        "apic_gpu_matches_cpu: CPU = {:.2?}, GPU = {:.2?}; \
         max rel = {:.3e} (abs = {:.3e}) at particle {argmax_idx}",
        cpu_elapsed, gpu_elapsed, max_rel, max_abs
    );
    assert!(
        max_rel < 1e-4,
        "GPU vs CPU particle position drift exceeds 1e-4 relative: \
         max = {:.3e} at particle {argmax_idx} \
         (cpu = {:?}, gpu = {:?})",
        max_rel,
        cpu.particles[argmax_idx].x,
        gpu_positions[argmax_idx],
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run with --features gpu --test fem_apic_gpu -- --ignored"]
fn apic_gpu_single_step_is_finite() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (mesh, particles, bcs) = build_scenario();
    let engine = Engine::new_gpu().expect("gpu adapter");
    let mut hybrid = ApicHybrid::new(mesh, particles);
    hybrid.gravity = Vec3::new(0.0, 0.0, -2.0);
    let state = ApicGpuState::new(&engine, &hybrid, &bcs).expect("ApicGpuState::new");
    state.step(&hybrid, 1.0 / 60.0, 0.5).expect("step");
    let pos = state.read_positions().expect("read");
    let vel = state.read_velocities().expect("read v");
    for (i, p) in pos.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "non-finite GPU position at particle {i}: {p:?}",
        );
    }
    for (i, v) in vel.iter().enumerate() {
        assert!(
            v.x.is_finite() && v.y.is_finite() && v.z.is_finite(),
            "non-finite GPU velocity at particle {i}: {v:?}",
        );
    }
}

/// Stage 37.6.1 — GPU pressure projection parity. The compressing
/// blob from the CPU contract test (radially inward velocities,
/// div v = −6) runs one projected step on both backends; the GPU CG
/// loop is recorded entirely on-device. f32-vs-f64 divergence over a
/// 120-iteration Krylov solve is the tolerance driver, so the gate is
/// looser than the smoothing-only parity above — but an unprojected
/// GPU run is asserted to be FAR from the CPU projected result, so
/// the gate cannot pass vacuously.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-fem T=fem_apic_gpu`"]
fn apic_gpu_pressure_projection_matches_cpu() {
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let mesh = FemMesh::tet_grid_3d(5, 5, 5, 1.0, 1.0, 1.0);
    let center = Vec3::new(0.5, 0.5, 0.5);
    let mut particles = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                let x = Vec3::new(
                    0.2531 + 0.09973 * i as Real,
                    0.2547 + 0.09931 * j as Real,
                    0.2513 + 0.09959 * k as Real,
                );
                let mut p = ApicParticle::at_rest(x, 1.0e-3);
                p.v = -2.0 * (x - center);
                particles.push(p);
            }
        }
    }
    let dt: Real = 1.0 / 240.0;
    let iters = 120u32;

    let mut cpu = ApicHybrid::new(mesh.clone(), particles.clone());
    cpu.gravity = Vec3::ZERO;
    cpu.step_projected(dt, 0.0, &[], iters);

    let engine = Engine::new_gpu().expect("gpu adapter");
    let mut gpu_proj = ApicHybrid::new(mesh.clone(), particles.clone());
    gpu_proj.gravity = Vec3::ZERO;
    gpu_proj
        .step_gpu_projected(&engine, dt as f32, 0.0, &[], iters)
        .expect("gpu projected step");

    let mut gpu_plain = ApicHybrid::new(mesh, particles);
    gpu_plain.gravity = Vec3::ZERO;
    gpu_plain
        .step_gpu(&engine, dt as f32, 0.0, &[])
        .expect("gpu plain step");

    let mut max_dv_proj: Real = 0.0;
    let mut max_dv_plain: Real = 0.0;
    for ((cp, gp), gpl) in cpu
        .particles
        .iter()
        .zip(gpu_proj.particles.iter())
        .zip(gpu_plain.particles.iter())
    {
        max_dv_proj = max_dv_proj.max((cp.v - gp.v).length());
        max_dv_plain = max_dv_plain.max((cp.v - gpl.v).length());
    }
    eprintln!(
        "projection parity: max |Δv| projected = {max_dv_proj:.4e}, unprojected control = {max_dv_plain:.4e}"
    );
    // Non-vacuous: skipping the projection must visibly diverge from
    // the CPU projected oracle (the blob is strongly compressing).
    assert!(
        max_dv_plain > 0.05,
        "control: unprojected GPU should differ strongly from projected CPU"
    );
    assert!(
        max_dv_proj < 1.0e-4,
        "GPU projection diverges from CPU oracle: {max_dv_proj:.4e} (control {max_dv_plain:.4e})"
    );
    assert!(
        max_dv_proj < max_dv_plain / 20.0,
        "projection parity not clearly better than the unprojected control"
    );
}
