//! Reduced FEM surface-contact coupling checks.
//!
//! The full collision/contact solvers live in `molla-solvers`; this
//! test is the FEM-side CPU oracle for transferring a contact patch
//! pressure into nodal forces before an elasticity solve. It verifies
//! force/moment conservation for the patch load and that the resulting
//! elastic displacement follows the contact normal.

use molla_fem::{
    BoundaryCondition, FemMesh, FunctionSpace, assemble_global,
    compute_element_elasticity_stiffness, solve_steady_state_vector,
};
use molla_math::{Real, Vec3};

const DIM: usize = 3;

fn lame_params(youngs: Real, poisson: Real) -> (Real, Real) {
    let lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu = youngs / (2.0 * (1.0 + poisson));
    (lam, mu)
}

fn nodal_contact_patch_load(
    vertices: &[Vec3],
    patch: &[usize],
    total_force: Vec3,
) -> (Vec<Real>, Vec3, Vec3) {
    let mut f = vec![0.0; vertices.len() * DIM];
    let per_node = total_force / patch.len() as Real;
    let mut centroid = Vec3::ZERO;
    let mut net_force = Vec3::ZERO;
    let mut net_moment = Vec3::ZERO;
    for &i in patch {
        centroid += vertices[i];
        f[i * DIM] += per_node.x;
        f[i * DIM + 1] += per_node.y;
        f[i * DIM + 2] += per_node.z;
        net_force += per_node;
        net_moment += vertices[i].cross(per_node);
    }
    centroid /= patch.len() as Real;
    (f, net_force, net_moment - centroid.cross(total_force))
}

fn find_vertex(vertices: &[Vec3], target: Vec3) -> usize {
    vertices
        .iter()
        .position(|p| (*p - target).length() < 1.0e-9)
        .unwrap_or_else(|| panic!("missing vertex at {target:?}"))
}

fn barycentric_soft_point_load(
    vertices: &[Vec3],
    tri: [usize; 3],
    weights: [Real; 3],
    force: Vec3,
) -> (Vec<Real>, Vec3, Vec3, Vec3) {
    let mut f = vec![0.0; vertices.len() * DIM];
    let mut point = Vec3::ZERO;
    let mut net_force = Vec3::ZERO;
    let mut net_moment = Vec3::ZERO;
    for (&i, &w) in tri.iter().zip(weights.iter()) {
        let node_force = force * w;
        point += vertices[i] * w;
        f[i * DIM] += node_force.x;
        f[i * DIM + 1] += node_force.y;
        f[i * DIM + 2] += node_force.z;
        net_force += node_force;
        net_moment += vertices[i].cross(node_force);
    }
    (f, net_force, net_moment - point.cross(force), point)
}

#[test]
fn surface_contact_patch_loads_fem_and_preserves_resultant() {
    let mesh = FemMesh::tet_grid_3d(4, 2, 2, 2.0, 0.8, 0.6);
    let (lam, mu) = lame_params(2.0e5, 0.25);
    let k = assemble_global(
        &mesh,
        FunctionSpace::P1Vector { dim: DIM },
        |cell, verts| compute_element_elasticity_stiffness(cell, verts, lam, mu),
    )
    .expect("assemble elasticity stiffness");

    let x_max = mesh
        .vertices
        .iter()
        .map(|p| p.x)
        .fold(Real::NEG_INFINITY, Real::max);
    let patch: Vec<usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .filter_map(|(i, p)| ((p.x - x_max).abs() < 1.0e-9).then_some(i))
        .collect();
    assert!(patch.len() >= 4, "expected a tip-face contact patch");

    let contact_force = Vec3::new(0.0, -3.0, 0.0);
    let (f_ext, net_force, moment_residual) =
        nodal_contact_patch_load(&mesh.vertices, &patch, contact_force);
    assert!(
        (net_force - contact_force).length() < 1.0e-12,
        "contact load transfer lost net force: net={net_force:?}, expected={contact_force:?}"
    );
    assert!(
        moment_residual.length() < 1.0e-12,
        "contact load transfer lost net moment about patch centroid: residual={moment_residual:?}"
    );

    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1.0e-9 {
            bcs.push(BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0; DIM],
            });
        }
    }

    let u = solve_steady_state_vector(&k, &f_ext, &bcs, &mesh.vertices, DIM, 20_000, 1.0e-10);
    assert!(
        u.iter().all(|v| v.is_finite()),
        "surface-contact FEM solve produced non-finite displacement"
    );

    let mut avg_tip = Vec3::ZERO;
    for &i in &patch {
        avg_tip += Vec3::new(u[i * DIM], u[i * DIM + 1], u[i * DIM + 2]);
    }
    avg_tip /= patch.len() as Real;
    eprintln!(
        "FEM contact patch: nodes={}, avg_tip_disp={avg_tip:?}, net_force={net_force:?}",
        patch.len()
    );

    assert!(
        avg_tip.y < -1.0e-5,
        "tip should move along the downward contact load, got {avg_tip:?}"
    );
    assert!(
        avg_tip.x.abs() < avg_tip.y.abs() * 0.35 && avg_tip.z.abs() < avg_tip.y.abs() * 0.35,
        "surface contact produced excessive transverse displacement: {avg_tip:?}"
    );
}

#[test]
fn soft_point_contact_barycentric_load_preserves_force_moment_and_moves_fem() {
    let mesh = FemMesh::tet_grid_3d(4, 2, 2, 2.0, 0.8, 0.6);
    let (lam, mu) = lame_params(2.0e5, 0.25);
    let k = assemble_global(
        &mesh,
        FunctionSpace::P1Vector { dim: DIM },
        |cell, verts| compute_element_elasticity_stiffness(cell, verts, lam, mu),
    )
    .expect("assemble elasticity stiffness");

    let x_max = mesh
        .vertices
        .iter()
        .map(|p| p.x)
        .fold(Real::NEG_INFINITY, Real::max);
    let tri = [
        find_vertex(&mesh.vertices, Vec3::new(x_max, 0.0, 0.0)),
        find_vertex(&mesh.vertices, Vec3::new(x_max, 0.4, 0.0)),
        find_vertex(&mesh.vertices, Vec3::new(x_max, 0.0, 0.3)),
    ];
    let weights = [0.20, 0.35, 0.45];
    let soft_contact_force = Vec3::new(0.0, -1.8, 0.4);
    let (f_ext, net_force, moment_residual, contact_point) =
        barycentric_soft_point_load(&mesh.vertices, tri, weights, soft_contact_force);

    assert!(
        (net_force - soft_contact_force).length() < 1.0e-12,
        "soft-to-FEM barycentric transfer lost force: net={net_force:?}, expected={soft_contact_force:?}"
    );
    assert!(
        moment_residual.length() < 1.0e-12,
        "soft-to-FEM barycentric transfer lost moment about contact point: residual={moment_residual:?}"
    );

    let mut bcs = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        if p.x.abs() < 1.0e-9 {
            bcs.push(BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0; DIM],
            });
        }
    }

    let u = solve_steady_state_vector(&k, &f_ext, &bcs, &mesh.vertices, DIM, 20_000, 1.0e-10);
    assert!(
        u.iter().all(|v| v.is_finite()),
        "soft-point FEM contact solve produced non-finite displacement"
    );

    let mut contact_disp = Vec3::ZERO;
    for (&i, &w) in tri.iter().zip(weights.iter()) {
        contact_disp += Vec3::new(u[i * DIM], u[i * DIM + 1], u[i * DIM + 2]) * w;
    }
    eprintln!(
        "soft-to-FEM contact point={contact_point:?}, disp={contact_disp:?}, net_force={net_force:?}"
    );

    assert!(
        contact_disp.dot(soft_contact_force) > 1.0e-5,
        "FEM contact-point displacement should follow the transferred soft contact force: disp={contact_disp:?}, force={soft_contact_force:?}"
    );
    assert!(
        contact_disp.y < 0.0 && contact_disp.z > 0.0,
        "contact displacement should reflect force signs, got {contact_disp:?}"
    );
}
