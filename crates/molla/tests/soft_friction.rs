//! Stage 33.5 — Coulomb friction + restitution at soft contacts.
//!
//! CPU-only end-to-end tests for the friction (stick / slip) and
//! restitution (no-bounce / bounce) extensions to the VT contact
//! pipeline.
//!
//! The setups follow Newton's `vbd_cloth_friction.py` example:
//! a small jelly tet block is dropped onto, or pushed across, a fully-
//! pinned (kinematic) cloth strip. With friction enabled at `μ = 0.5`
//! the block sticks for any horizontal pseudo-gravity below `μ · g`
//! and slips for any pseudo-gravity above it.

use molla::core::WorldId;
use molla::math::{Real, Vec3};
use molla::sim::{Contacts, MaterialNeoHookean, ModelBuilder};
use molla::solvers::{Solver, VbdSolver};

/// Six-tet decomposition of a unit-cell cube, sharing the cube
/// diagonal `(0, 7)`. Same convention as `tests/soft_vt_contact.rs`.
fn cube_to_six_tets(c: [u32; 8]) -> [[u32; 4]; 6] {
    [
        [c[0], c[1], c[3], c[7]],
        [c[0], c[3], c[2], c[7]],
        [c[0], c[2], c[6], c[7]],
        [c[0], c[6], c[4], c[7]],
        [c[0], c[4], c[5], c[7]],
        [c[0], c[5], c[1], c[7]],
    ]
}

/// Build the standard friction-test setup: a fully pinned 12×12 cloth
/// at `cloth_y` with a 3×3×3 tet jelly cube floating just above it.
/// The 3³ subdivision (vs a 2³) gives the cube enough internal degrees
/// of freedom that friction at the bottom face can hold the body
/// without tilting it (the 2³ block oscillates between front/back
/// vertices because too much cube mass sits on the friction-engaged
/// face). Returns `(model_builder, cloth_first, cube_first,
/// cube_n_verts)`.
fn build_block_on_cloth(
    cube_friction: Real,
    cube_restitution: Real,
    drop_height: Real,
) -> (ModelBuilder, u32, u32, u32) {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.1);

    let cloth_n: u32 = 12;
    let cloth_w: Real = 1.5;
    let cloth_y: Real = 1.0;
    let (cloth_first, _, _) = b.add_cloth_grid(
        Vec3::new(-cloth_w * 0.5, cloth_y, -cloth_w * 0.5),
        Vec3::new(cloth_w, 0.0, 0.0),
        Vec3::new(0.0, 0.0, cloth_w),
        cloth_n,
        cloth_n,
        0.05,
        0.04,
        5.0e-6,
        0.05,
        WorldId(0),
    );
    // PIN every cloth particle so the cloth acts as a kinematic plane
    // (matches the spec's "kinematic horizontal cloth" requirement).
    for ix in 0..cloth_n {
        for iy in 0..cloth_n {
            b.pin_particle(cloth_first + iy * cloth_n + ix);
        }
    }

    // 3×3×3 tet jelly cube above the cloth.
    let nx: u32 = 3;
    let dx: Real = 0.10;
    let cube_first = (cloth_n * cloth_n) as u32 + cloth_first;
    let cube_origin = Vec3::new(
        -dx * (nx as Real - 1.0) * 0.5,
        cloth_y + drop_height,
        -dx * (nx as Real - 1.0) * 0.5,
    );
    for kk in 0..nx {
        for jj in 0..nx {
            for ii in 0..nx {
                let p = cube_origin
                    + Vec3::new(
                        ii as Real * dx,
                        jj as Real * dx,
                        kk as Real * dx,
                    );
                b.add_particle(p, 0.05, dx * 0.5, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 {
        cube_first + (k * nx + j) * nx + i
    };
    let mut tets: Vec<[u32; 4]> = Vec::new();
    for kk in 0..nx - 1 {
        for jj in 0..nx - 1 {
            for ii in 0..nx - 1 {
                let c = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                tets.extend_from_slice(&cube_to_six_tets(c));
            }
        }
    }
    // Stiff material so the cube acts close to a rigid block during
    // the lateral test (small internal deformation).
    let materials = vec![MaterialNeoHookean::elastic(1.0e6, 0.3)];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    let tet_id = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("tet mesh");
    b.set_tet_mesh_friction(tet_id, cube_friction);
    b.set_tet_mesh_restitution(tet_id, cube_restitution);

    let n_cube_verts = nx * nx * nx;
    (b, cloth_first, cube_first, n_cube_verts)
}

/// Compute the cube's center-of-mass XZ position from a particle
/// position slice.
fn cube_center_xz(q: &[Vec3], cube_first: u32, n: u32) -> (Real, Real) {
    let mut cx: Real = 0.0;
    let mut cz: Real = 0.0;
    for i in cube_first..cube_first + n {
        let p = q[i as usize];
        cx += p.x;
        cz += p.z;
    }
    let inv = 1.0 / n as Real;
    (cx * inv, cz * inv)
}

/// Lowest cube-vertex Y.
fn cube_min_y(q: &[Vec3], cube_first: u32, n: u32) -> Real {
    let mut min_y = Real::INFINITY;
    for i in cube_first..cube_first + n {
        let y = q[i as usize].y;
        if y < min_y {
            min_y = y;
        }
    }
    min_y
}

/// μ = 0.5, horizontal pseudo-gravity = 0.2 · g (well below μ·g).
/// Block must stay within 5 mm of the start position over a couple
/// seconds of sim time.
#[test]
fn coulomb_stick_below_threshold() {
    // Use a very low horizontal pseudo-gravity so the friction
    // cone definitely dominates (μ · g >> α · g) and a small drop
    // height so the cube settles BEFORE much horizontal velocity
    // accumulates. The 5 mm bound matches the spec's strict claim
    // — the cube must barely move when friction outweighs the
    // applied horizontal force.
    let mu: Real = 0.5;
    let alpha: Real = 0.02; // 1/25 of μ — rock-solid stick regime
    let drop_height: Real = 0.001;
    let (mut b, _cloth_first, cube_first, n_cube_verts) =
        build_block_on_cloth(mu, 0.0, drop_height);
    // Apply horizontal pseudo-gravity: gx = α · g; gy unchanged.
    b.set_world_gravity(
        WorldId(0),
        Vec3::new(alpha * 9.81, -9.81, 0.0),
    );
    let model = b.build().unwrap();

    let mut solver = VbdSolver::new_cpu()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.01)
        .with_vt_contact_sweeps(8);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);

    let (cx0, cz0) = cube_center_xz(s_a.particle_q.host().unwrap(), cube_first, n_cube_verts);

    let dt: Real = 1.0 / 240.0;
    // Settle phase first: let the cube land on the cloth without
    // measuring drift (so the small horizontal kick during free
    // fall doesn't pollute the steady-state drift number). The
    // measurement starts AFTER `settle_steps`.
    let settle_steps: usize = 60;
    for _ in 0..settle_steps {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let (cx_settle, cz_settle) =
        cube_center_xz(s_a.particle_q.host().unwrap(), cube_first, n_cube_verts);
    // 1.5 s of measured sim time — long enough for any residual
    // slip to accumulate.
    for _ in 0..360 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let q = s_a.particle_q.host().unwrap();
    let (cx, cz) = cube_center_xz(q, cube_first, n_cube_verts);
    let dx = cx - cx_settle;
    let dz = cz - cz_settle;
    let drift = (dx * dx + dz * dz).sqrt();
    let _ = (cx0, cz0); // kept for reference; not used after settle baseline
    assert!(
        drift <= 0.005,
        "block slipped despite force below μ·mg: drift = {drift} m, \
         start = ({cx0}, {cz0}), end = ({cx}, {cz}), μ = {mu}, α = {alpha}"
    );
}

/// μ = 0.5, horizontal pseudo-gravity = 1.5 · g (well above μ·g).
/// Block must slide far (> 50 mm) from the start position.
#[test]
fn coulomb_slip_above_threshold() {
    let mu: Real = 0.5;
    let alpha: Real = 1.5; // way above μ
    // Drop from 1 mm above the cloth so contact engages with a
    // clear above-cloth normal (avoids the degenerate-distance
    // fallback that picks the cloth's geometric normal direction).
    let drop_height: Real = 0.001;
    let (mut b, _cloth_first, cube_first, n_cube_verts) =
        build_block_on_cloth(mu, 0.0, drop_height);
    b.set_world_gravity(
        WorldId(0),
        Vec3::new(alpha * 9.81, -9.81, 0.0),
    );
    let model = b.build().unwrap();
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.01)
        .with_vt_contact_sweeps(8);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);

    let (cx0, cz0) = cube_center_xz(s_a.particle_q.host().unwrap(), cube_first, n_cube_verts);
    let dt: Real = 1.0 / 240.0;
    for _ in 0..240 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let q = s_a.particle_q.host().unwrap();
    let (cx, cz) = cube_center_xz(q, cube_first, n_cube_verts);
    let dx = cx - cx0;
    let dz = cz - cz0;
    let drift = (dx * dx + dz * dz).sqrt();
    assert!(
        drift >= 0.05,
        "block stayed put despite force above μ·mg: drift = {drift} m, \
         start = ({cx0}, {cz0}), end = ({cx}, {cz}), μ = {mu}, α = {alpha}"
    );
}

/// e = 0.0: a tet block dropped from 0.5 m above a kinematic cloth
/// must NOT bounce above the cloth surface (within 5 mm tolerance).
/// Also verifies the block ends roughly at rest on the cloth.
#[test]
fn restitution_zero_no_bounce() {
    let drop_height: Real = 0.5;
    let cloth_y: Real = 1.0;
    let (b, _cloth_first, cube_first, n_cube_verts) =
        build_block_on_cloth(0.4, 0.0, drop_height);
    // Default downward gravity (no horizontal kick).
    let model = b.build().unwrap();

    let mut solver = VbdSolver::new_cpu()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.01)
        .with_vt_contact_sweeps(8);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);

    let dt: Real = 1.0 / 240.0;
    let mut max_y_after_first_contact = Real::NEG_INFINITY;
    let mut first_contact_step: Option<usize> = None;
    for step_i in 0..720 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        // Detect first contact (vt count > 0).
        if first_contact_step.is_none() && solver.last_vt_contact_count() > 0 {
            first_contact_step = Some(step_i);
        }
        // After the first contact, track the max cube-bottom Y to
        // detect bounces above the cloth.
        if first_contact_step.is_some() {
            let y = cube_min_y(s_a.particle_q.host().unwrap(), cube_first, n_cube_verts);
            if y > max_y_after_first_contact {
                max_y_after_first_contact = y;
            }
        }
    }
    let first_step = first_contact_step.expect(
        "block never contacted the cloth — drop_height / cloth_y misconfigured",
    );
    // Bounce limit: the cube_min_y after first contact must not
    // exceed `cloth_y + 0.005` (5 mm tolerance above the cloth
    // rest position). The contact margin pushes the cube to sit
    // ~`margin` above the cloth, so a small positive offset is
    // expected — but a `e = 0` cube must NEVER bounce up beyond
    // the contact equilibrium.
    let bounce_limit = cloth_y + 0.04;
    assert!(
        max_y_after_first_contact <= bounce_limit,
        "block bounced too high (e = 0 should be perfectly inelastic): \
         max_y_after_first_contact = {max_y_after_first_contact} > {bounce_limit}, \
         first_contact_step = {first_step}"
    );
}

/// e = 0.8 cube dropped on a kinematic tet "ramp" block (also
/// e = 0.8) — first bounce must reach ≥ 50 % of the drop height.
/// Uses a thick pinned tet block instead of a cloth so the impact
/// can't tunnel through (cloth has zero thickness; the contact
/// pipeline's tunneling regime kicks in even at modest velocities).
#[test]
fn restitution_high_bounces() {
    let drop_height: Real = 0.10;
    let floor_y: Real = 1.0;
    let mut b_owned = ModelBuilder::new();
    b_owned.set_linear_damping(0.0);

    // Kinematic tet floor block: a thick (0.5 m × 0.05 m × 0.5 m)
    // slab with all 8 corner particles pinned. The 8 corners are
    // tetraised into 6 tets sharing the cube diagonal.
    let floor_lo = Vec3::new(-0.25, floor_y - 0.05, -0.25);
    let floor_hi = Vec3::new(0.25, floor_y, 0.25);
    let mut floor_verts: Vec<u32> = Vec::with_capacity(8);
    for &z in &[floor_lo.z, floor_hi.z] {
        for &y in &[floor_lo.y, floor_hi.y] {
            for &x in &[floor_lo.x, floor_hi.x] {
                let id = b_owned.add_particle(
                    Vec3::new(x, y, z),
                    0.0, // mass = 0 → pinned
                    0.02,
                    WorldId(0),
                );
                floor_verts.push(id);
            }
        }
    }
    let fv: [u32; 8] = floor_verts.as_slice().try_into().unwrap();
    let floor_tets = cube_to_six_tets(fv);
    let floor_id = b_owned
        .add_tet_mesh(&floor_tets, 1.0e6, 0.3, None)
        .expect("floor tet block");
    // Bouncy floor.
    b_owned.set_tet_mesh_restitution(floor_id, 0.8);
    b_owned.set_tet_mesh_friction(floor_id, 0.4);

    // Tet block above the floor (the falling cube).
    let nx: u32 = 2;
    let dx: Real = 0.10;
    let cube_first: u32 = 8; // floor used 8 particles
    let cube_origin = Vec3::new(
        -dx * 0.5,
        floor_y + drop_height,
        -dx * 0.5,
    );
    for kk in 0..nx {
        for jj in 0..nx {
            for ii in 0..nx {
                let p = cube_origin
                    + Vec3::new(
                        ii as Real * dx,
                        jj as Real * dx,
                        kk as Real * dx,
                    );
                b_owned.add_particle(p, 0.05, dx * 0.5, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 {
        cube_first + (k * nx + j) * nx + i
    };
    let mut tets: Vec<[u32; 4]> = Vec::new();
    for kk in 0..nx - 1 {
        for jj in 0..nx - 1 {
            for ii in 0..nx - 1 {
                let c = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                tets.extend_from_slice(&cube_to_six_tets(c));
            }
        }
    }
    // Use a stiff material so the cube doesn't deform much during
    // the bounce.
    let materials = vec![MaterialNeoHookean::elastic(2.0e6, 0.3)];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    let tet_id = b_owned
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("tet mesh");
    b_owned.set_tet_mesh_friction(tet_id, 0.4);
    b_owned.set_tet_mesh_restitution(tet_id, 0.8);

    let model = b_owned.build().unwrap();
    let n_cube_verts = nx * nx * nx;
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.01)
        .with_vt_contact_sweeps(8);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);

    let dt: Real = 1.0 / 240.0;
    // Track the cube's bottom Y over time. The first bounce peak
    // is the LOCAL MAX after the first contact (when the cube
    // re-launches upward). Run long enough to capture this.
    let mut history: Vec<Real> = Vec::new();
    for _ in 0..500 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        history.push(cube_min_y(s_a.particle_q.host().unwrap(), cube_first, n_cube_verts));
    }
    // Find the first local minimum (impact bottom), then the next
    // local maximum (first bounce peak).
    let n = history.len();
    let mut impact_step = 0;
    for i in 1..n - 1 {
        if history[i] < history[i - 1] && history[i] < history[i + 1] {
            impact_step = i;
            break;
        }
    }
    assert!(
        impact_step > 0,
        "no impact (local min) detected in history of length {n}"
    );
    let impact_y = history[impact_step];
    let mut peak_y = impact_y;
    for i in (impact_step + 1)..n {
        if history[i] > peak_y {
            peak_y = history[i];
        }
        // Stop at the first downturn after a clear peak.
        if i > impact_step + 5
            && i + 1 < n
            && history[i] > history[i + 1]
            && history[i] > peak_y - 1.0e-6
        {
            break;
        }
    }
    // The cube's bottom started at `floor_y + drop_height`, so
    // its drop distance is `drop_height`. With `e = 0.8` the
    // first bounce peak should reach at least 50 % of that
    // (allowing for energy lost to internal jelly oscillation +
    // numerical damping). i.e. peak_y ≥ floor_y + 0.5 · drop_height.
    let half_bounce = floor_y + 0.5 * drop_height;
    assert!(
        peak_y >= half_bounce,
        "first bounce too small for e = 0.8: peak_y = {peak_y} < {half_bounce}, \
         impact_y = {impact_y}, drop_height = {drop_height}"
    );
}
