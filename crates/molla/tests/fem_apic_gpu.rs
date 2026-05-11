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
//! nixVulkan cargo test -p molla --features gpu --test fem_apic_gpu -- --ignored
//! ```

#![cfg(feature = "gpu")]

use std::sync::Mutex;
use std::time::Instant;

use molla::compute::Engine;
use molla::fem::apic_coupling_gpu::ApicGpuState;
use molla::fem::{
    apic_coupling::ApicHybrid, ApicParticle, BoundaryCondition, FemMesh,
};
use molla::math::{Mat3, Real, Vec3};

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
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
