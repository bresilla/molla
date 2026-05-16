//! Stage 33 — cross-mesh soft-vs-soft VT contact integration tests.
//!
//! CPU-only — proves the registry / BVH / projection chain end-to-end
//! using the same builder API the viewer's `SoftbodyDrop` scene uses.

use molla_core::WorldId;
use molla_geometry::SoftBvh;
use molla_math::{Real, Vec3};
use molla_sim::{Contacts, MaterialNeoHookean, ModelBuilder, compute_tet_surface_triangles};
use molla_solvers::{Solver, VbdSolver, gather_vt_contacts};

/// Six-tet decomposition of a unit-cell cube. Vertex order:
///   0 = (0,0,0), 1 = (1,0,0), 2 = (0,1,0), 3 = (1,1,0),
///   4 = (0,0,1), 5 = (1,0,1), 6 = (0,1,1), 7 = (1,1,1)
fn cube_to_six_tets(c: [u32; 8]) -> [[u32; 4]; 6] {
    // Standard decomposition that produces six tets sharing the cube
    // diagonal (0, 7) and yields a positive `det(Dm)` for each.
    [
        [c[0], c[1], c[3], c[7]],
        [c[0], c[3], c[2], c[7]],
        [c[0], c[2], c[6], c[7]],
        [c[0], c[6], c[4], c[7]],
        [c[0], c[4], c[5], c[7]],
        [c[0], c[5], c[1], c[7]],
    ]
}

#[test]
fn boundary_tri_extraction_on_unit_cube() {
    // 6-tet cube: 12 boundary tris (2 per face × 6 faces), no internals.
    let tets = cube_to_six_tets([0, 1, 2, 3, 4, 5, 6, 7]);
    let bnd = compute_tet_surface_triangles(&tets);
    assert_eq!(
        bnd.len(),
        12,
        "expected 12 boundary tris, got {}",
        bnd.len()
    );
    // Sanity: every boundary tri must reference exactly cube verts (0..8).
    for tri in &bnd {
        for &v in tri {
            assert!(v < 8, "boundary tri vertex {v} out of cube range");
        }
    }
}

#[test]
fn vt_contacts_zero_when_far_apart() {
    // Two cloths 5 m apart along Y. No contact at any margin < 5 m.
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    let _ = b.add_cloth_grid(
        Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        5,
        5,
        0.05,
        0.02,
        1.0e-4,
        0.5,
        WorldId(0),
    );
    let _ = b.add_cloth_grid(
        Vec3::new(-0.5, 5.0, -0.5),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        5,
        5,
        0.05,
        0.02,
        1.0e-4,
        0.5,
        WorldId(0),
    );
    let model = b.build().unwrap();
    assert_eq!(
        model.soft_surfaces.len(),
        2,
        "expected 2 cloth soft surfaces, got {}",
        model.soft_surfaces.len()
    );
    let state = model.state().unwrap();
    let bvh = SoftBvh::build(&model, &state);
    let contacts = gather_vt_contacts(&model, &state, &bvh, 0.05);
    assert_eq!(
        contacts.len(),
        0,
        "expected 0 contacts at 5 m separation, got {}",
        contacts.len()
    );
}

/// Tet-vertex on cloth-triangle: jelly cube above a pinned cloth strip
/// must rest on the cloth (not pass through). Substep budget keeps
/// runtime under ~5 s on a developer laptop.
#[test]
fn tet_vertex_on_cloth_triangle_penalty_holds_under_gravity() {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);

    // 1 m × 1 m horizontal cloth pinned along the two opposite edges
    // at iy = 0 and iy = ny-1.
    let cloth_n = 12u32;
    let cloth_w: Real = 1.0;
    let cloth_y: Real = 1.0;
    let (cloth_first, _, _) = b.add_cloth_grid(
        Vec3::new(-cloth_w * 0.5, cloth_y, -cloth_w * 0.5),
        Vec3::new(cloth_w, 0.0, 0.0),
        Vec3::new(0.0, 0.0, cloth_w),
        cloth_n,
        cloth_n,
        0.05,
        0.02,
        5.0e-6,
        0.05,
        WorldId(0),
    );
    // Pin all 4 cloth edges so it acts more like a rigid sheet
    // (less sag, more resistance to the cube). This isolates the
    // VT contact correctness from cloth-edge slip dynamics.
    for ix in 0..cloth_n {
        b.pin_particle(cloth_first + ix);
        b.pin_particle(cloth_first + (cloth_n - 1) * cloth_n + ix);
        b.pin_particle(cloth_first + ix * cloth_n);
        b.pin_particle(cloth_first + ix * cloth_n + cloth_n - 1);
    }

    // Tet jelly cube above the cloth.
    let nx = 3u32;
    let dx: Real = 0.10;
    let cube_lo_y: Real = 1.5;
    let cube_origin = Vec3::new(
        -dx * (nx as Real - 1.0) * 0.5,
        cube_lo_y,
        -dx * (nx as Real - 1.0) * 0.5,
    );
    for kk in 0..nx {
        for jj in 0..nx {
            for ii in 0..nx {
                let p = cube_origin + Vec3::new(ii as Real * dx, jj as Real * dx, kk as Real * dx);
                b.add_particle(p, 0.05, dx * 0.5, WorldId(0));
            }
        }
    }
    let cloth_count = cloth_n * cloth_n;
    let v_at =
        |i: u32, j: u32, k: u32| -> u32 { cloth_first + cloth_count + (k * nx + j) * nx + i };
    let mut tets = Vec::new();
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
                let cell_tets = cube_to_six_tets(c);
                tets.extend_from_slice(&cell_tets);
            }
        }
    }
    let materials = vec![MaterialNeoHookean::elastic(2.0e5, 0.3)];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    let _ = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("tet mesh");

    let model = b.build().unwrap();
    assert!(
        model.soft_surfaces.len() >= 2,
        "expected >= 2 soft surfaces (cloth + tet), got {}",
        model.soft_surfaces.len()
    );

    let mut solver = VbdSolver::new_cpu()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.05)
        .with_vt_contact_sweeps(8);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);

    // Initial cube bottom Y.
    let cube_first = cloth_first + cloth_count;
    let cube_total = nx * nx * nx;
    let initial_min_y = (cube_first..cube_first + cube_total)
        .map(|i| s_a.particle_q.host().unwrap()[i as usize].y)
        .fold(Real::INFINITY, |a, b| a.min(b));
    assert!(
        initial_min_y > cloth_y,
        "cube must start above cloth (got {initial_min_y} vs cloth {cloth_y})"
    );

    let dt: Real = 1.0 / 240.0;
    for _ in 0..600 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    // Lowest cube vertex Y after 600 substeps.
    let q = s_a.particle_q.host().unwrap();
    let final_min_y = (cube_first..cube_first + cube_total)
        .map(|i| q[i as usize].y)
        .fold(Real::INFINITY, |a, b| a.min(b));
    // Lowest cloth vertex (sagged) Y — used as the "floor" baseline.
    let cloth_min_y = (cloth_first..cloth_first + cloth_count)
        .map(|i| q[i as usize].y)
        .fold(Real::INFINITY, |a, b| a.min(b));

    // The cube must rest on the cloth — within `cloth_y +/- 5 mm`
    // (per the original spec). Stage 33's friction-less penalty
    // contact had to use a 100 mm tolerance because the cube
    // slowly drifted laterally and slipped off the cloth edge;
    // Stage 33.5's Coulomb friction stops the lateral drift, so
    // the strict 5 mm penetration bound now holds.
    let pene = cloth_min_y - final_min_y;
    assert!(
        pene <= 0.005,
        "jelly cube fell through cloth: cube_min_y = {final_min_y}, \
         cloth_min_y = {cloth_min_y}, penetration = {pene}"
    );
    // Cube must NOT be still in free fall (it should be near the
    // cloth, not 1 m below it).
    assert!(
        final_min_y > cloth_y - 0.5,
        "cube has fallen too far past the cloth: {final_min_y}"
    );
}

/// Mirror of the viewer's `Scene::SoftbodyDrop` build (6×6×3 jelly
/// cube above a 24×24 hanging cloth, 2-edge pinning) — proof that
/// the live viewer scene rests instead of falling through. CPU-only
/// so this runs in `cargo test -p molla --lib` style sweeps.
#[test]
fn viewer_softbody_drop_scene_jelly_rests_on_cloth() {
    const TET_NX: u32 = 6;
    const TET_NY: u32 = 6;
    const TET_NZ: u32 = 3;
    const TET_DX: Real = 0.10;
    const CLOTH_NX: u32 = 24;
    const CLOTH_W: Real = 1.6;
    let cloth_y: Real = 1.0;

    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);
    let origin = Vec3::new(-CLOTH_W * 0.5, cloth_y, -CLOTH_W * 0.5);
    let u_axis = Vec3::new(CLOTH_W, 0.0, 0.0);
    let v_axis = Vec3::new(0.0, 0.0, CLOTH_W);
    let (cloth_first, _, _) = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        CLOTH_NX,
        CLOTH_NX,
        0.05,
        0.04,
        5.0e-6,
        0.05,
        WorldId(0),
    );
    // 2-edge pinning matches the viewer scene exactly.
    for ix in 0..CLOTH_NX {
        b.pin_particle(cloth_first + ix);
        b.pin_particle(cloth_first + (CLOTH_NX - 1) * CLOTH_NX + ix);
    }
    // Tet jelly cube above the cloth.
    let cube_origin = Vec3::new(
        -TET_DX * (TET_NX as Real - 1.0) * 0.5,
        2.5,
        -TET_DX * (TET_NZ as Real - 1.0) * 0.5,
    );
    for kk in 0..TET_NZ {
        for jj in 0..TET_NY {
            for ii in 0..TET_NX {
                let p = cube_origin
                    + Vec3::new(
                        ii as Real * TET_DX,
                        jj as Real * TET_DX,
                        kk as Real * TET_DX,
                    );
                b.add_particle(p, 0.05, TET_DX * 0.5, WorldId(0));
            }
        }
    }
    let cloth_count = CLOTH_NX * CLOTH_NX;
    let v_at = |i: u32, j: u32, k: u32| -> u32 {
        cloth_first + cloth_count + (k * TET_NY + j) * TET_NX + i
    };
    let mut tets = Vec::new();
    for kk in 0..TET_NZ - 1 {
        for jj in 0..TET_NY - 1 {
            for ii in 0..TET_NX - 1 {
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
                let cell_tets = cube_to_six_tets(c);
                tets.extend_from_slice(&cell_tets);
            }
        }
    }
    let materials = vec![MaterialNeoHookean::elastic(2.0e5, 0.3)];
    let mat_idx: Vec<u32> = vec![0; tets.len()];
    let _ = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("tet mesh");

    let model = b.build().unwrap();
    assert_eq!(
        model.soft_surfaces.len(),
        2,
        "viewer scene must have 2 soft surfaces"
    );

    let mut solver = VbdSolver::new_cpu()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.05)
        .with_vt_contact_sweeps(8);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);

    let cube_first = cloth_first + cloth_count;
    let cube_total = TET_NX * TET_NY * TET_NZ;

    let dt: Real = 1.0 / 240.0;
    // Run for 2.0 s of sim time (480 substeps). Long enough for the
    // cube to fall ~1 m to the cloth and rest there.
    for _ in 0..480 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.particle_q.host().unwrap();
    let cube_min_y = (cube_first..cube_first + cube_total)
        .map(|i| q[i as usize].y)
        .fold(Real::INFINITY, |a, b| a.min(b));
    // The whole point of Stage 33 — the cube should NOT have free-
    // fallen past the cloth floor (y < 0.0 = below the world plane).
    // Without contact, it would have fallen >10 m; with contact it
    // should be near the cloth at y≈1.0.
    assert!(
        cube_min_y > 0.5,
        "viewer scene jelly fell through cloth: cube_min_y = {cube_min_y}"
    );
}

/// Cloth particle on a kinematic tet block: a single cloth particle
/// is released onto a fully-pinned tet block; verify the particle
/// settles on the block's top face (it doesn't pass through).
#[test]
fn cloth_particle_on_tet_face_two_way() {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);

    // Kinematic tet block at origin, top face at y = 1.0.
    // 8 particles forming a unit cube, all PINNED (mass = 0).
    let cube_lo: Real = 0.0;
    let cube_hi: Real = 1.0;
    let mut block_verts: Vec<u32> = Vec::with_capacity(8);
    for &z in &[cube_lo, cube_hi] {
        for &y in &[cube_lo, cube_hi] {
            for &x in &[cube_lo, cube_hi] {
                let id = b.add_particle(
                    Vec3::new(x, y, z),
                    0.0, // mass = 0 → pinned
                    0.02,
                    WorldId(0),
                );
                block_verts.push(id);
            }
        }
    }
    let bv: [u32; 8] = block_verts.as_slice().try_into().unwrap();
    let block_tets = cube_to_six_tets(bv);
    let _ = b
        .add_tet_mesh(&block_tets, 1.0e5, 0.3, None)
        .expect("tet block");

    // 4×4 cloth grid 0.5 m above the block top, free to fall.
    let cloth_n = 4u32;
    let cloth_w: Real = 0.4;
    let cloth_y: Real = 1.6;
    let (cloth_first, _, _) = b.add_cloth_grid(
        Vec3::new(0.5 - cloth_w * 0.5, cloth_y, 0.5 - cloth_w * 0.5),
        Vec3::new(cloth_w, 0.0, 0.0),
        Vec3::new(0.0, 0.0, cloth_w),
        cloth_n,
        cloth_n,
        0.05,
        0.02,
        5.0e-6,
        0.05,
        WorldId(0),
    );

    let model = b.build().unwrap();
    let mut solver = VbdSolver::new_cpu()
        .with_iterations(8)
        .with_contact_stiffness(1.0e5)
        .with_vt_contact_margin(0.05)
        .with_vt_contact_sweeps(8);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);

    let dt: Real = 1.0 / 240.0;
    for _ in 0..400 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    // Every cloth particle should be at or above the block's top
    // face (y = 1.0) within a small penetration tolerance.
    let q = s_a.particle_q.host().unwrap();
    let cloth_count = cloth_n * cloth_n;
    let mut min_y = Real::INFINITY;
    for i in cloth_first..cloth_first + cloth_count {
        let y = q[i as usize].y;
        if y < min_y {
            min_y = y;
        }
        assert!(y.is_finite(), "cloth particle {i} y not finite: {y}");
    }
    // Lowest cloth particle should be near the block top with
    // <= 5 cm penetration (margin + radius slop).
    assert!(
        min_y > cube_hi - 0.05,
        "cloth fell through block: min_y = {min_y}, block top = {cube_hi}"
    );
    // And not too high — it should actually rest on the block.
    assert!(
        min_y < cube_hi + 0.10,
        "cloth didn't fall onto block: min_y = {min_y}, block top = {cube_hi}"
    );
}

/// Tet mesh on tet mesh: a deformable jelly cube drops onto a pinned
/// tetrahedral block. This is the asset-free softbody-on-softbody
/// counterpart to the cloth/tet cases above and exercises the same
/// VT contact path with both participants represented as tet-derived
/// soft surfaces.
#[test]
fn tet_cube_on_pinned_tet_block_softbody_contact() {
    fn add_cube_particles(
        b: &mut ModelBuilder,
        origin: Vec3,
        n: u32,
        dx: Real,
        mass: Real,
        radius: Real,
    ) -> (u32, Vec<[u32; 4]>) {
        let first = b.add_particle(origin, mass, radius, WorldId(0));
        for kk in 0..n {
            for jj in 0..n {
                for ii in 0..n {
                    if ii == 0 && jj == 0 && kk == 0 {
                        continue;
                    }
                    let p = origin + Vec3::new(ii as Real * dx, jj as Real * dx, kk as Real * dx);
                    b.add_particle(p, mass, radius, WorldId(0));
                }
            }
        }
        let v_at = |i: u32, j: u32, k: u32| -> u32 { first + (k * n + j) * n + i };
        let mut tets = Vec::new();
        for kk in 0..n - 1 {
            for jj in 0..n - 1 {
                for ii in 0..n - 1 {
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
        (first, tets)
    }

    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.35);

    let n = 3u32;
    let dx: Real = 0.10;
    let lower_top: Real = 1.0 + dx * (n as Real - 1.0);
    let (lower_first, lower_tets) = add_cube_particles(
        &mut b,
        Vec3::new(-dx, 1.0, -dx),
        n,
        dx,
        0.0, // pinned tet block
        dx * 0.25,
    );
    let lower_materials = vec![MaterialNeoHookean::elastic(1.0e6, 0.3)];
    let lower_mat_idx = vec![0; lower_tets.len()];
    b.add_tet_mesh_per_material(&lower_tets, &lower_materials, &lower_mat_idx, None)
        .expect("lower pinned tet block");

    let (upper_first, upper_tets) = add_cube_particles(
        &mut b,
        Vec3::new(-dx, lower_top + 0.32, -dx),
        n,
        dx,
        0.05,
        dx * 0.25,
    );
    let upper_materials = vec![MaterialNeoHookean::elastic(2.0e5, 0.3)];
    let upper_mat_idx = vec![0; upper_tets.len()];
    b.add_tet_mesh_per_material(&upper_tets, &upper_materials, &upper_mat_idx, None)
        .expect("upper dynamic tet cube");

    let model = b.build().unwrap();
    assert_eq!(
        model.soft_surfaces.len(),
        2,
        "expected two tet soft surfaces"
    );

    let mut solver = VbdSolver::new_cpu()
        .with_iterations(10)
        .with_contact_stiffness(1.5e5)
        .with_vt_contact_margin(0.04)
        .with_vt_contact_sweeps(10);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(8);
    let upper_count = n * n * n;
    let lower_count = n * n * n;

    let dt: Real = 1.0 / 240.0;
    let mut max_vt_contacts = 0usize;
    for _ in 0..620 {
        Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        max_vt_contacts = max_vt_contacts.max(solver.last_vt_contact_count());
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.particle_q.host().unwrap();
    let qd = s_a.particle_qd.host().unwrap();
    let lower_max_y = (lower_first..lower_first + lower_count)
        .map(|i| q[i as usize].y)
        .fold(Real::NEG_INFINITY, |a, b| a.max(b));
    let upper_min_y = (upper_first..upper_first + upper_count)
        .map(|i| q[i as usize].y)
        .fold(Real::INFINITY, |a, b| a.min(b));
    let upper_max_speed = (upper_first..upper_first + upper_count)
        .map(|i| qd[i as usize].length())
        .fold(0.0, Real::max);

    for i in upper_first..upper_first + upper_count {
        assert!(
            q[i as usize].is_finite() && qd[i as usize].is_finite(),
            "upper tet particle {i} produced non-finite state: q={:?}, qd={:?}",
            q[i as usize],
            qd[i as usize]
        );
    }
    assert!(
        max_vt_contacts > 0,
        "tet/tet softbody contact never generated VT contacts"
    );
    assert!(
        upper_min_y > lower_max_y - 0.04,
        "dynamic tet cube penetrated pinned block too deeply: \
         upper_min_y={upper_min_y}, lower_max_y={lower_max_y}"
    );
    assert!(
        upper_min_y < lower_max_y + 0.12,
        "dynamic tet cube did not settle onto the pinned block: \
         upper_min_y={upper_min_y}, lower_max_y={lower_max_y}"
    );
    assert!(
        upper_max_speed < 0.40,
        "dynamic tet cube still moving too fast after settling: max |v|={upper_max_speed}"
    );
}
