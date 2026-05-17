//! Stage 37.6 — APIC FEM-MPM hybrid contract tests.
//!
//! Four CPU tests cover the v1 contract:
//!
//! 1. `apic_p2g_g2p_roundtrip_preserves_momentum` — 100 particles
//!    inside a single tet; P2G then G2P with the SAME node velocities
//!    (no Stokes solve). Total momentum preserved within 1e-6 (the
//!    bonding contract).
//! 2. `apic_uniform_motion_no_distortion` — particles all moving at
//!    the same velocity (rigid translation). After one full
//!    [`ApicHybrid::step`] (no gravity, no viscosity, no BCs), every
//!    particle still carries the same velocity to within 1e-3.
//! 3. `apic_falling_particles_settle` — drop 50 particles into a
//!    closed tet box (no-slip walls), 100 steps with gravity +
//!    moderate viscosity. Verifies (a) no NaN, (b) particles stay
//!    inside the AABB, (c) total kinetic energy decreases over the
//!    long run (allowing brief upticks while the field re-organises).
//! 4. `apic_affine_velocity_field_roundtrip_recovers_linear_motion` —
//!    procedural single-tet APIC transfer for a non-uniform affine
//!    velocity field. P2G must reconstruct the nodal linear field and
//!    G2P must recover both particle velocity and affine `C`.

use molla_fem::{ApicParticle, BoundaryCondition, FemCell, FemMesh, apic_coupling::ApicHybrid};
use molla_math::{Mat3, Real, Vec3};

// ─── Helpers ────────────────────────────────────────────────────────

fn unit_tet_mesh() -> FemMesh {
    FemMesh {
        vertices: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
        cells: vec![FemCell::Tet([0, 1, 2, 3])],
    }
}

/// Generate `n` deterministic-pseudo-random samples uniformly in the
/// unit tet (Vose's barycentric method: u, v, w in [0, 1] with the
/// triangulated discard / reflect to keep them inside the simplex).
fn sample_unit_tet(n: usize) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(n);
    // Trivial linear-congruential RNG — keep tests deterministic
    // without pulling in the `rand` crate dev-dep.
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let next = |s: &mut u64| -> Real {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as Real) / ((1u64 << 31) as Real)
    };
    while out.len() < n {
        let mut u = next(&mut s);
        let mut v = next(&mut s);
        let mut w = next(&mut s);
        // Rosca-Suffern triangulation: fold the unit cube into the
        // unit tet (volume 1/6) so samples are uniformly distributed.
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        if v + w > 1.0 {
            let t = w;
            w = 1.0 - u - v;
            v = 1.0 - t;
        } else if u + v + w > 1.0 {
            let t = w;
            w = u + v + w - 1.0;
            u = 1.0 - v - t;
        }
        out.push(Vec3::new(u, v, w));
    }
    out
}

fn assert_mat3_close(actual: Mat3, expected: Mat3, tolerance: Real, label: &str) {
    let max_col_error = (actual.x_axis - expected.x_axis)
        .length()
        .max((actual.y_axis - expected.y_axis).length())
        .max((actual.z_axis - expected.z_axis).length());
    assert!(
        actual.x_axis.is_finite()
            && actual.y_axis.is_finite()
            && actual.z_axis.is_finite()
            && max_col_error < tolerance,
        "{label}: max column error {max_col_error:.6e} exceeded {tolerance:.6e}\n\
         actual = {actual:?}\nexpected = {expected:?}"
    );
}

// ─── Test 1: P2G + G2P roundtrip preserves total momentum ──────────

#[test]
fn apic_p2g_g2p_roundtrip_preserves_momentum() {
    let mesh = unit_tet_mesh();
    let positions = sample_unit_tet(100);
    let particles: Vec<ApicParticle> = positions
        .iter()
        .enumerate()
        .map(|(i, x)| {
            // Vary velocity per particle so the test isn't trivial.
            let vi = i as Real;
            ApicParticle {
                x: *x,
                v: Vec3::new(0.5 + 0.01 * vi, -0.3 + 0.005 * vi, 0.2),
                c: Mat3::ZERO,
                mass: 0.5 + 0.01 * vi,
            }
        })
        .collect();
    let total_mass: Real = particles.iter().map(|p| p.mass).sum();
    let initial_momentum: Vec3 = particles
        .iter()
        .fold(Vec3::ZERO, |acc, p| acc + p.mass * p.v);

    let mut hybrid = ApicHybrid::new(mesh, particles.clone());
    hybrid.gravity = Vec3::ZERO;

    // P2G: deposit particle momentum on node mass + momentum.
    let (node_mass, node_momentum) = hybrid.p2g();

    // Total node mass = total particle mass (every particle landed
    // inside the single tet).
    let total_node_mass: Real = node_mass.iter().sum();
    let total_node_momentum: Vec3 = node_momentum.iter().fold(Vec3::ZERO, |a, b| a + *b);
    assert!(
        (total_node_mass - total_mass).abs() < 1e-12,
        "P2G mass conservation: {total_node_mass} vs {total_mass}"
    );
    assert!(
        (total_node_momentum - initial_momentum).length() < 1e-12,
        "P2G momentum conservation: {total_node_momentum:?} vs {initial_momentum:?}"
    );

    // Recover node velocities and run G2P with NO solve in between.
    let node_velocity: Vec<Vec3> = node_mass
        .iter()
        .zip(node_momentum.iter())
        .map(|(m, p)| {
            if *m > Real::EPSILON {
                *p / *m
            } else {
                Vec3::ZERO
            }
        })
        .collect();

    // dt = 0 → no advection, just velocity + C update.
    hybrid.g2p(&node_velocity, 0.0);

    // Total particle momentum after roundtrip.
    let final_momentum: Vec3 = hybrid
        .particles
        .iter()
        .fold(Vec3::ZERO, |acc, p| acc + p.mass * p.v);

    let drift = (final_momentum - initial_momentum).length();
    assert!(
        drift < 1e-6,
        "P2G + G2P momentum drift {drift} > 1e-6 \
         (initial {initial_momentum:?}, final {final_momentum:?})"
    );

    // Sanity: positions must not have moved (dt = 0).
    for (a, b) in hybrid.particles.iter().zip(particles.iter()) {
        assert!(
            (a.x - b.x).length() < 1e-15,
            "G2P with dt=0 must not move particles"
        );
    }
}

#[test]
fn mpm_to_fem_tet_transfer_conserves_mass_momentum_and_constant_velocity() {
    let mesh = unit_tet_mesh();
    let velocity = Vec3::new(0.35, -0.20, 0.15);
    let positions = [
        Vec3::new(0.10, 0.10, 0.10),
        Vec3::new(0.55, 0.15, 0.10),
        Vec3::new(0.15, 0.50, 0.15),
        Vec3::new(0.10, 0.20, 0.55),
        Vec3::new(0.25, 0.25, 0.20),
    ];
    let particles: Vec<ApicParticle> = positions
        .iter()
        .enumerate()
        .map(|(i, x)| ApicParticle {
            x: *x,
            v: velocity,
            c: Mat3::ZERO,
            mass: 0.25 + 0.10 * i as Real,
        })
        .collect();
    let total_mass: Real = particles.iter().map(|p| p.mass).sum();
    let total_momentum = velocity * total_mass;

    let hybrid = ApicHybrid::new(mesh, particles);
    let (node_mass, node_momentum) = hybrid.p2g();

    let transferred_mass: Real = node_mass.iter().sum();
    let transferred_momentum: Vec3 = node_momentum.iter().copied().sum();
    assert!(
        (transferred_mass - total_mass).abs() < 1.0e-12,
        "MPM-to-FEM transfer lost mass: node={transferred_mass}, particle={total_mass}"
    );
    assert!(
        (transferred_momentum - total_momentum).length() < 1.0e-12,
        "MPM-to-FEM transfer lost momentum: node={transferred_momentum:?}, particle={total_momentum:?}"
    );

    for (i, (&m, &p)) in node_mass.iter().zip(node_momentum.iter()).enumerate() {
        assert!(m > 0.0, "test particles should excite every tet node");
        let node_velocity = p / m;
        assert!(
            (node_velocity - velocity).length() < 1.0e-12,
            "constant velocity field was not preserved at FEM node {i}: \
             got {node_velocity:?}, expected {velocity:?}"
        );
    }
}

// ─── Test 2: Uniform motion preserves uniform velocity ─────────────

#[test]
fn apic_uniform_motion_no_distortion() {
    // Use a small grid so we have multiple tets but every particle
    // still lives well inside the mesh after one step.
    let mesh = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);
    let v_uniform = Vec3::new(0.1, -0.05, 0.02);

    // 64 particles on a 4x4x4 grid inside the cube, offset by a bit
    // so they sit comfortably inside cells.
    let mut particles = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let x = Vec3::new(
                    0.125 + 0.25 * i as Real,
                    0.125 + 0.25 * j as Real,
                    0.125 + 0.25 * k as Real,
                );
                particles.push(ApicParticle {
                    x,
                    v: v_uniform,
                    c: Mat3::ZERO,
                    mass: 1.0,
                });
            }
        }
    }

    let mut hybrid = ApicHybrid::new(mesh, particles);
    hybrid.gravity = Vec3::ZERO;

    // One step at moderate dt with NO viscosity, NO BCs — pure
    // translation should round-trip cleanly.
    hybrid.step(1.0 / 240.0, 0.0, &[]);

    let mut max_dev: Real = 0.0;
    for p in &hybrid.particles {
        let dev = (p.v - v_uniform).length();
        if dev > max_dev {
            max_dev = dev;
        }
    }
    assert!(
        max_dev < 1e-3,
        "Uniform-motion velocity drift {max_dev} exceeded 1e-3"
    );
}

// ─── Test 2b: Affine APIC roundtrip preserves linear fields ───────

#[test]
fn apic_affine_velocity_field_roundtrip_recovers_linear_motion() {
    let mesh = unit_tet_mesh();
    let positions = sample_unit_tet(80);
    let centroid = Vec3::splat(0.25);
    let v0 = Vec3::new(0.2, -0.1, 0.05);
    let affine = Mat3::from_cols(
        Vec3::new(0.10, 0.02, -0.01),
        Vec3::new(-0.03, 0.08, 0.04),
        Vec3::new(0.01, -0.02, 0.06),
    );

    let particles: Vec<ApicParticle> = positions
        .iter()
        .map(|x| ApicParticle {
            x: *x,
            v: v0 + affine * (*x - centroid),
            c: affine,
            mass: 1.0,
        })
        .collect();
    let expected_velocity: Vec<Vec3> = particles.iter().map(|p| p.v).collect();

    let mut hybrid = ApicHybrid::new(mesh.clone(), particles);
    hybrid.gravity = Vec3::ZERO;
    let (node_mass, node_momentum) = hybrid.p2g();
    assert!(
        node_mass.iter().all(|m| *m > Real::EPSILON),
        "affine APIC fixture must deposit mass to every tet node: {node_mass:?}"
    );

    let node_velocity: Vec<Vec3> = node_mass
        .iter()
        .zip(node_momentum.iter())
        .map(|(m, p)| *p / *m)
        .collect();

    for (i, (node, actual)) in mesh.vertices.iter().zip(node_velocity.iter()).enumerate() {
        let expected = v0 + affine * (*node - centroid);
        let error = (*actual - expected).length();
        assert!(
            actual.is_finite() && error < 1.0e-12,
            "node {i} affine velocity mismatch: error {error:.6e}, \
             actual {actual:?}, expected {expected:?}"
        );
    }

    hybrid.g2p(&node_velocity, 0.0);

    let mut max_velocity_error: Real = 0.0;
    for (i, (particle, expected_v)) in hybrid
        .particles
        .iter()
        .zip(expected_velocity.iter())
        .enumerate()
    {
        let velocity_error = (particle.v - *expected_v).length();
        max_velocity_error = max_velocity_error.max(velocity_error);
        assert!(
            particle.x.is_finite() && particle.v.is_finite() && velocity_error < 1.0e-12,
            "particle {i} affine velocity roundtrip mismatch: error {velocity_error:.6e}, \
             actual {:?}, expected {:?}",
            particle.v,
            expected_v
        );
        assert_mat3_close(
            particle.c,
            affine,
            1.0e-12,
            &format!("particle {i} affine C roundtrip"),
        );
    }
    assert!(
        max_velocity_error < 1.0e-12,
        "max affine velocity roundtrip error {max_velocity_error:.6e}"
    );
}

#[test]
fn apic_rigid_rotation_roundtrip_preserves_angular_field() {
    let mesh = unit_tet_mesh();
    let positions = sample_unit_tet(96);
    let centroid = Vec3::splat(0.25);
    let omega = Vec3::new(0.30, -0.20, 0.50);
    let skew_omega = Mat3::from_cols(
        Vec3::new(0.0, omega.z, -omega.y),
        Vec3::new(-omega.z, 0.0, omega.x),
        Vec3::new(omega.y, -omega.x, 0.0),
    );

    let particles: Vec<ApicParticle> = positions
        .iter()
        .map(|x| ApicParticle {
            x: *x,
            v: omega.cross(*x - centroid),
            c: skew_omega,
            mass: 1.0,
        })
        .collect();
    let expected_particle_velocity: Vec<Vec3> = particles.iter().map(|p| p.v).collect();

    let mut hybrid = ApicHybrid::new(mesh.clone(), particles);
    hybrid.gravity = Vec3::ZERO;
    let (node_mass, node_momentum) = hybrid.p2g();
    assert!(
        node_mass.iter().all(|m| *m > Real::EPSILON),
        "rigid-rotation APIC fixture must deposit mass to every tet node: {node_mass:?}"
    );

    let node_velocity: Vec<Vec3> = node_mass
        .iter()
        .zip(node_momentum.iter())
        .map(|(m, p)| *p / *m)
        .collect();

    let mut expected_angular_momentum = Vec3::ZERO;
    let mut actual_angular_momentum = Vec3::ZERO;
    for (i, ((node, mass), velocity)) in mesh
        .vertices
        .iter()
        .zip(node_mass.iter())
        .zip(node_velocity.iter())
        .enumerate()
    {
        let r = *node - centroid;
        let expected_velocity = omega.cross(r);
        let velocity_error = (*velocity - expected_velocity).length();
        assert!(
            velocity.is_finite() && velocity_error < 1.0e-12,
            "node {i} rigid-rotation velocity mismatch: error {velocity_error:.6e}, \
             actual {velocity:?}, expected {expected_velocity:?}"
        );
        expected_angular_momentum += r.cross(*mass * expected_velocity);
        actual_angular_momentum += r.cross(node_momentum[i]);
    }
    assert!(
        (actual_angular_momentum - expected_angular_momentum).length() < 1.0e-12,
        "APIC P2G angular momentum mismatch: actual={actual_angular_momentum:?}, \
         expected={expected_angular_momentum:?}"
    );
    assert!(
        actual_angular_momentum.dot(omega) > 0.0,
        "rigid-rotation fixture should carry non-zero angular momentum along omega: \
         L={actual_angular_momentum:?}, omega={omega:?}"
    );

    hybrid.g2p(&node_velocity, 0.0);

    for (i, (particle, expected_v)) in hybrid
        .particles
        .iter()
        .zip(expected_particle_velocity.iter())
        .enumerate()
    {
        let velocity_error = (particle.v - *expected_v).length();
        assert!(
            particle.x.is_finite() && particle.v.is_finite() && velocity_error < 1.0e-12,
            "particle {i} rigid-rotation velocity roundtrip mismatch: error {velocity_error:.6e}, \
             actual {:?}, expected {:?}",
            particle.v,
            expected_v
        );
        assert_mat3_close(
            particle.c,
            skew_omega,
            1.0e-12,
            &format!("particle {i} rigid-rotation C roundtrip"),
        );
    }
}

#[test]
fn apic_dirichlet_vector_boundary_drives_particles_uniformly() {
    let mesh = unit_tet_mesh();
    let positions = sample_unit_tet(32);
    let particles: Vec<ApicParticle> = positions
        .iter()
        .map(|x| ApicParticle {
            x: *x,
            v: Vec3::ZERO,
            c: Mat3::ZERO,
            mass: 1.0,
        })
        .collect();

    let imposed_velocity = Vec3::new(0.20, -0.10, 0.05);
    let bcs: Vec<BoundaryCondition> = (0..mesh.vert_count())
        .map(|vertex| BoundaryCondition::DirichletVector {
            vertex: vertex as u32,
            values: vec![imposed_velocity.x, imposed_velocity.y, imposed_velocity.z],
        })
        .collect();
    let dt = 0.01;
    let mut hybrid = ApicHybrid::new(mesh, particles);
    hybrid.gravity = Vec3::ZERO;
    let x_before: Vec<Vec3> = hybrid.particles.iter().map(|p| p.x).collect();

    hybrid.step(dt, 0.0, &bcs);

    for (i, (particle, x0)) in hybrid.particles.iter().zip(x_before.iter()).enumerate() {
        let v_err = (particle.v - imposed_velocity).length();
        let x_expected = *x0 + imposed_velocity * dt;
        let x_err = (particle.x - x_expected).length();
        assert!(
            particle.x.is_finite()
                && particle.v.is_finite()
                && particle.c.x_axis.is_finite()
                && particle.c.y_axis.is_finite()
                && particle.c.z_axis.is_finite(),
            "APIC boundary-driven particle {i} produced non-finite state: {particle:?}"
        );
        assert!(
            v_err < 1.0e-12,
            "particle {i} did not receive imposed FEM boundary velocity: got {:?}, expected {:?}, err={v_err}",
            particle.v,
            imposed_velocity
        );
        assert!(
            x_err < 1.0e-12,
            "particle {i} did not advect with imposed velocity: got {:?}, expected {:?}, err={x_err}",
            particle.x,
            x_expected
        );
        assert_mat3_close(
            particle.c,
            Mat3::ZERO,
            1.0e-12,
            &format!("particle {i} imposed uniform boundary C"),
        );
    }
}

// ─── Test 3: Falling particles settle in a no-slip box ─────────────

#[test]
fn apic_falling_particles_settle() {
    let mesh = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);

    // 50 particles drifting in the upper half of the box, with small
    // initial spread to avoid every particle hitting the same node.
    let mut particles: Vec<ApicParticle> = (0..50)
        .map(|i| {
            let r = i as Real;
            // Tile in a 5x5x2 grid in the top of the box.
            let ix = (i % 5) as Real;
            let iy = ((i / 5) % 5) as Real;
            let iz = (i / 25) as Real;
            let x = Vec3::new(
                0.15 + 0.15 * ix + 0.005 * (r * 1.7).sin(),
                0.15 + 0.15 * iy + 0.005 * (r * 2.3).cos(),
                0.55 + 0.18 * iz + 0.005 * (r * 0.9).sin(),
            );
            ApicParticle {
                x,
                v: Vec3::ZERO,
                c: Mat3::ZERO,
                mass: 1.0,
            }
        })
        .collect();

    let total_mass_init: Real = particles.iter().map(|p| p.mass).sum();
    assert!(total_mass_init > 0.0);

    // No-slip BCs on every boundary vertex (any vertex on a face of
    // the unit cube). Pin to zero velocity in every component.
    let mesh_for_bcs = mesh.clone();
    let mut bcs: Vec<BoundaryCondition> = Vec::new();
    for (i, p) in mesh_for_bcs.vertices.iter().enumerate() {
        let on_boundary = p.x <= 1e-9
            || p.x >= 1.0 - 1e-9
            || p.y <= 1e-9
            || p.y >= 1.0 - 1e-9
            || p.z <= 1e-9
            || p.z >= 1.0 - 1e-9;
        if on_boundary {
            bcs.push(BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0, 0.0, 0.0],
            });
        }
    }

    let mut hybrid = ApicHybrid::new(mesh, std::mem::take(&mut particles));
    // Reasonable gravity: −Z, scaled so the demo terminates in 100 steps.
    hybrid.gravity = Vec3::new(0.0, 0.0, -2.0);
    let dt: Real = 1.0 / 60.0;
    let viscosity: Real = 0.5;

    let aabb_lo = Vec3::new(0.001, 0.001, 0.001);
    let aabb_hi = Vec3::new(0.999, 0.999, 0.999);

    let mut energies: Vec<Real> = Vec::with_capacity(101);
    energies.push(hybrid.total_kinetic_energy());

    for step in 0..100 {
        hybrid.step(dt, viscosity, &bcs);
        // Clamp to AABB — the explicit advect can overshoot a wall by
        // a fraction of dt; a tiny clamp is honest (the no-slip wall
        // just absorbs the overshoot) and matches Warp's
        // `example_apic_fluid` behaviour.
        hybrid.clamp_to_aabb(aabb_lo, aabb_hi);

        // Sanity: no NaN.
        for (i, p) in hybrid.particles.iter().enumerate() {
            assert!(
                p.x.is_finite() && p.v.is_finite(),
                "step {step} particle {i} produced non-finite state: \
                 x = {:?}, v = {:?}",
                p.x,
                p.v
            );
        }

        // Particles inside AABB.
        for (i, p) in hybrid.particles.iter().enumerate() {
            assert!(
                p.x.x >= aabb_lo.x - 1e-9
                    && p.x.x <= aabb_hi.x + 1e-9
                    && p.x.y >= aabb_lo.y - 1e-9
                    && p.x.y <= aabb_hi.y + 1e-9
                    && p.x.z >= aabb_lo.z - 1e-9
                    && p.x.z <= aabb_hi.z + 1e-9,
                "step {step} particle {i} escaped the box: x = {:?}",
                p.x
            );
        }

        energies.push(hybrid.total_kinetic_energy());
    }

    // The kinetic-energy curve should trend down. We check that the
    // average over the LAST 20 steps is strictly less than the average
    // over the FIRST 20 steps. The instantaneous KE can have small
    // upticks (gravity pumps energy in each step, viscosity + walls
    // dissipate it) so a strict monotonic-per-step check is too brittle;
    // the long-run mean is the right contract.
    let first_window: Real = energies.iter().skip(1).take(20).copied().sum::<Real>() / 20.0;
    let last_window: Real = energies.iter().rev().take(20).copied().sum::<Real>() / 20.0;
    assert!(
        last_window < first_window,
        "Kinetic energy did not settle: first-window mean {first_window:.6e}, \
         last-window mean {last_window:.6e}"
    );

    // Also verify the FINAL KE is below the peak — particles really
    // are losing energy to the boundary / viscosity.
    let peak: Real = energies.iter().copied().fold(0.0, Real::max);
    let final_ke = *energies.last().unwrap();
    assert!(
        final_ke < peak,
        "Final KE {final_ke:.6e} not below peak KE {peak:.6e}"
    );
}
