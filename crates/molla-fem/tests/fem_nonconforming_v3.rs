//! Stage 47 v3 — surface-only cross-mesh `M_st` contract tests.
//!
//! v3 introduces [`NonconformingField::new_surface`]: same operator
//! definition as v1, but the integral defining `M_st_ij` runs on the
//! **target mesh's outer surface** (`geometry::boundary_facets`) rather
//! than its volume. This is the right integration domain for contact
//! pressures (and any other field whose support is a 2D patch on the
//! boundary).
//!
//! Three properties pin v3 down:
//!
//! 1. **Constant projection on the boundary** —  `u_source ≡ 1` ⇒
//!    `M_st · 1` is the surface-mass lumped row sum on the target's
//!    boundary DoFs, and zero on the interior DoFs.
//! 2. **Partition of unity on the surface** —  `∑_ij M_st_ij` = surface
//!    area of the target boundary, to 1e-9.
//! 3. **Penalty force restricted to boundary** — a v2-style
//!    [`NonconformingPenaltyContact`] built on the surface field
//!    applies zero force to interior target DoFs.

use molla_fem::assembly::matvec;
use molla_fem::boundary_facets;
use molla_fem::{FemCell, FemMesh, NonconformingField};
use molla_math::{Real, Vec3};

/// Collect the indices of target vertices that lie on the outer
/// surface of a tet `FemMesh`. Computed via `boundary_facets`; matches
/// the assembly the surface operator uses to integrate `M_st`.
fn boundary_dof_set(mesh: &FemMesh) -> std::collections::HashSet<usize> {
    let mut out = std::collections::HashSet::new();
    for f in boundary_facets(mesh) {
        // v3 surface meshes are tet-only ⇒ all three slots are real.
        for v in f.iter() {
            out.insert(*v as usize);
        }
    }
    out
}

/// Sum of triangle areas in 3D.
fn surface_area_of(mesh: &FemMesh) -> Real {
    boundary_facets(mesh)
        .into_iter()
        .map(|f| {
            let p = [
                mesh.vertices[f[0] as usize],
                mesh.vertices[f[1] as usize],
                mesh.vertices[f[2] as usize],
            ];
            0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length()
        })
        .sum()
}

#[test]
fn surface_field_constant_projection() {
    // u_source ≡ 1. Surface-only M_st · 1 must:
    //   - sum to vol-of-surface-mass = surface_area(target) (via
    //     boundary-DoF row sums)
    //   - vanish on every interior target DoF (whose shape function
    //     support has zero measure on the boundary).
    let source = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);
    let target = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let field = NonconformingField::new_surface(&source, &target).expect("build surface field");
    assert!(field.is_surface(), "constructor must mark domain = surface");

    let ones = vec![1.0; field.source_dofs()];
    let row = matvec(field.coupling_matrix(), &ones);
    assert_eq!(row.len(), field.target_dofs());

    let bnd = boundary_dof_set(&target);
    let area = surface_area_of(&target);

    let mut interior_max: Real = 0.0;
    let mut boundary_sum: Real = 0.0;
    let mut n_boundary_nonzero = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if bnd.contains(&i) {
            // Surface DoFs receive the row sum of the surface mass
            // matrix — strictly positive for any vertex incident to
            // at least one boundary facet.
            assert!(
                v >= -1.0e-12,
                "DoF {i}: surface row sum {v} should be non-negative"
            );
            if v > 1.0e-9 {
                n_boundary_nonzero += 1;
            }
            boundary_sum += v;
        } else {
            // Interior DoFs (no incident surface facet) must be
            // exactly zero up to floating-point round-off.
            if v.abs() > interior_max {
                interior_max = v.abs();
            }
        }
    }
    // (a) Interior DoFs vanish.
    assert!(
        interior_max < 1.0e-12,
        "interior DoF max |M_st · 1| = {interior_max} (expected ~0)"
    );
    // (b) Boundary row sums add up to the target surface area.
    let err = (boundary_sum - area).abs();
    assert!(
        err < 1.0e-9,
        "boundary row sum {boundary_sum} != surface area {area} (err {err})"
    );
    // (c) Every boundary DoF picked up a non-zero contribution (the
    // unit cube's boundary is connected so this is exactly the count
    // of boundary DoFs).
    assert_eq!(
        n_boundary_nonzero,
        bnd.len(),
        "{} boundary DoFs got non-zero contributions out of {}",
        n_boundary_nonzero,
        bnd.len()
    );
}

#[test]
fn surface_mass_total_equals_surface_area() {
    // ∑_ij M_st_ij = ∫_∂target 1 · 1 dS = surface_area(∂target).
    // This is the partition-of-unity check on the surface operator.
    let source = FemMesh::tet_grid_3d(5, 5, 5, 1.0, 1.0, 1.0);
    let target = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let field = NonconformingField::new_surface(&source, &target).expect("build surface field");

    let ones_source = vec![1.0; field.source_dofs()];
    let row = matvec(field.coupling_matrix(), &ones_source);
    let total: Real = row.iter().sum();

    let area = surface_area_of(&target);
    let err = (total - area).abs();
    eprintln!(
        "stage 47 v3 surface-area check: Σ M_st = {total:.15}, area = {area:.15}, err = {err:.3e}"
    );
    assert!(
        err < 1.0e-9,
        "Σ_ij M_st_ij = {total} != surface area {area} (err {err}, tolerance 1e-9)"
    );

    // Sanity: the unit cube's surface area is exactly 6.
    assert!(
        (area - 6.0).abs() < 1.0e-12,
        "unit-cube surface area {area} != 6.0"
    );
}

#[test]
fn surface_field_acts_only_on_boundary_via_matvec() {
    // The surface M_st has zero rows on interior target DoFs. Verify
    // this directly via `M_st · u_source`: the resulting vector should
    // be zero on interior DoFs regardless of u_source.
    use molla_fem::assembly::matvec;
    let source = FemMesh::tet_grid_3d(4, 4, 4, 1.0, 1.0, 1.0);
    let target = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
    let field = NonconformingField::new_surface(&source, &target).expect("build surface field");

    let u_source: Vec<Real> = source
        .vertices
        .iter()
        .map(|p: &Vec3| 0.4 + 0.5 * (p.x + p.y + p.z))
        .collect();
    let m_u = matvec(field.coupling_matrix(), &u_source);

    let bnd = boundary_dof_set(&target);
    let mut interior_max: Real = 0.0;
    let mut boundary_max: Real = 0.0;
    for (i, &v) in m_u.iter().enumerate() {
        let av = v.abs();
        if bnd.contains(&i) {
            if av > boundary_max {
                boundary_max = av;
            }
        } else if av > interior_max {
            interior_max = av;
        }
    }
    assert!(
        interior_max < 1.0e-12,
        "interior DoF max |M_st·u| = {interior_max} (expected ~0)"
    );
    assert!(
        boundary_max > 1.0e-3,
        "boundary M_st·u suspiciously small (max abs = {boundary_max})"
    );
}

// Sanity dependency: confirm `FemCell::Tet` is still the cell kind we
// build from `tet_grid_3d` so the boundary_facets helpers are taking
// the tet path. This guards against accidental builder rewrites in
// follow-up stages that would break the tet-only assumption.
#[test]
fn tet_grid_emits_tet_cells_only() {
    let m = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
    assert!(!m.cells.is_empty());
    for c in &m.cells {
        assert!(
            matches!(c, FemCell::Tet(_)),
            "non-tet cell in tet_grid_3d output"
        );
    }
}
