// Per-cell curl recovery B = ∇×A from lowest-order Nédélec edge DoFs.
// Mirrors the CPU `compute_curl_per_cell` in
// `crates/molla-fem/src/magneto.rs`: B = Σ_e sign_e · A[dof_e] · (2 ∇φ_a×∇φ_b),
// over the 6 tet edges. The per-cell edge DoF indices and orientation
// signs are precomputed host-side. One thread per tet.

struct Uniforms {
    counts: vec4<u32>,   // (cell_count, _, _, _)
};

@group(0) @binding(0) var<storage, read> verts:      array<u32>;      // 4 per cell
@group(0) @binding(1) var<storage, read> positions:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> a_global:   array<f32>;
@group(0) @binding(3) var<storage, read> edge_dofs:  array<u32>;      // 6 per cell
@group(0) @binding(4) var<storage, read> edge_signs: array<f32>;      // 6 per cell

@group(1) @binding(0) var<storage, read_write> out_b: array<f32>;     // 3 per cell

@group(2) @binding(0) var<uniform> u: Uniforms;

const EA = array<u32, 6>(0u, 0u, 0u, 1u, 1u, 2u);
const EB = array<u32, 6>(1u, 2u, 3u, 2u, 3u, 3u);

@compute @workgroup_size(64)
fn curl(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= u.counts.x) { return; }

    let p0 = positions[verts[4u * cell + 0u]].xyz;
    let p1 = positions[verts[4u * cell + 1u]].xyz;
    let p2 = positions[verts[4u * cell + 2u]].xyz;
    let p3 = positions[verts[4u * cell + 3u]].xyz;

    // Nodal shape gradients (match CPU tet_shape_grads / D^-T).
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let e3 = p3 - p0;
    let inv_det = 1.0 / dot(e1, cross(e2, e3));
    var g = array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0),
        cross(e2, e3) * inv_det,
        cross(e3, e1) * inv_det,
        cross(e1, e2) * inv_det,
    );
    g[0] = -(g[1] + g[2] + g[3]);

    var b = vec3<f32>(0.0, 0.0, 0.0);
    for (var e = 0u; e < 6u; e = e + 1u) {
        let curl_e = 2.0 * cross(g[EA[e]], g[EB[e]]);
        let dof = edge_dofs[6u * cell + e];
        let sign = edge_signs[6u * cell + e];
        b = b + sign * a_global[dof] * curl_e;
    }

    out_b[3u * cell + 0u] = b.x;
    out_b[3u * cell + 1u] = b.y;
    out_b[3u * cell + 2u] = b.z;
}
