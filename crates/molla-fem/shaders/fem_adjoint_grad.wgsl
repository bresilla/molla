// GPU per-cell adjoint gradient for linear-elasticity Young's modulus.
// Mirrors the CPU `compute_grad_youngs` in
// `crates/molla-fem/src/adjoint.rs`: each tet contributes
// `λ_localᵀ · K_local · u_local`, which equals `vol · (Bλ)ᵀ D (Bu)` for
// the constant-strain B. The host sums the per-cell scalars and divides
// by −E. Tet meshes only (dim = 3).

struct Uniforms {
    counts: vec4<u32>,   // (cell_count, _, _, _)
    mat: vec4<f32>,      // (lame_lambda, mu, _, _)
};

@group(0) @binding(0) var<storage, read> verts: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tets:  array<u32>;
@group(0) @binding(2) var<storage, read> u_field:   array<f32>;
@group(0) @binding(3) var<storage, read> lam_field: array<f32>;

@group(1) @binding(0) var<storage, read_write> out_energy: array<f32>;

@group(2) @binding(0) var<uniform> uni: Uniforms;

// Voigt strain B·f for a tet: (exx, eyy, ezz, gyz, gxz, gxy) packed as
// two vec3 (axial, shear). g0..g3 are the nodal shape gradients; f0..f3
// are the nodal 3-vectors of the field.
fn strain_axial(g0: vec3<f32>, g1: vec3<f32>, g2: vec3<f32>, g3: vec3<f32>,
                f0: vec3<f32>, f1: vec3<f32>, f2: vec3<f32>, f3: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        g0.x * f0.x + g1.x * f1.x + g2.x * f2.x + g3.x * f3.x,
        g0.y * f0.y + g1.y * f1.y + g2.y * f2.y + g3.y * f3.y,
        g0.z * f0.z + g1.z * f1.z + g2.z * f2.z + g3.z * f3.z,
    );
}

fn strain_shear(g0: vec3<f32>, g1: vec3<f32>, g2: vec3<f32>, g3: vec3<f32>,
                f0: vec3<f32>, f1: vec3<f32>, f2: vec3<f32>, f3: vec3<f32>) -> vec3<f32> {
    // (gyz, gxz, gxy)
    let gyz = g0.z * f0.y + g0.y * f0.z + g1.z * f1.y + g1.y * f1.z
            + g2.z * f2.y + g2.y * f2.z + g3.z * f3.y + g3.y * f3.z;
    let gxz = g0.z * f0.x + g0.x * f0.z + g1.z * f1.x + g1.x * f1.z
            + g2.z * f2.x + g2.x * f2.z + g3.z * f3.x + g3.x * f3.z;
    let gxy = g0.y * f0.x + g0.x * f0.y + g1.y * f1.x + g1.x * f1.y
            + g2.y * f2.x + g2.x * f2.y + g3.y * f3.x + g3.x * f3.y;
    return vec3<f32>(gyz, gxz, gxy);
}

@compute @workgroup_size(64)
fn adjoint_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= uni.counts.x) { return; }

    let i0 = tets[4u * cell + 0u];
    let i1 = tets[4u * cell + 1u];
    let i2 = tets[4u * cell + 2u];
    let i3 = tets[4u * cell + 3u];
    let p0 = verts[i0].xyz;
    let p1 = verts[i1].xyz;
    let p2 = verts[i2].xyz;
    let p3 = verts[i3].xyz;

    // Shape gradients (match CPU D^-T): g1=(e2×e3)/det, etc.
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let e3 = p3 - p0;
    let det = dot(e1, cross(e2, e3));
    let inv_det = 1.0 / det;
    let g1 = cross(e2, e3) * inv_det;
    let g2 = cross(e3, e1) * inv_det;
    let g3 = cross(e1, e2) * inv_det;
    let g0 = -(g1 + g2 + g3);
    let vol = abs(det) / 6.0;

    // Gather nodal u and λ (interleaved x,y,z per vertex).
    let u0 = vec3<f32>(u_field[3u * i0], u_field[3u * i0 + 1u], u_field[3u * i0 + 2u]);
    let u1 = vec3<f32>(u_field[3u * i1], u_field[3u * i1 + 1u], u_field[3u * i1 + 2u]);
    let u2 = vec3<f32>(u_field[3u * i2], u_field[3u * i2 + 1u], u_field[3u * i2 + 2u]);
    let u3 = vec3<f32>(u_field[3u * i3], u_field[3u * i3 + 1u], u_field[3u * i3 + 2u]);
    let l0 = vec3<f32>(lam_field[3u * i0], lam_field[3u * i0 + 1u], lam_field[3u * i0 + 2u]);
    let l1 = vec3<f32>(lam_field[3u * i1], lam_field[3u * i1 + 1u], lam_field[3u * i1 + 2u]);
    let l2 = vec3<f32>(lam_field[3u * i2], lam_field[3u * i2 + 1u], lam_field[3u * i2 + 2u]);
    let l3 = vec3<f32>(lam_field[3u * i3], lam_field[3u * i3 + 1u], lam_field[3u * i3 + 2u]);

    let eu_a = strain_axial(g0, g1, g2, g3, u0, u1, u2, u3);
    let eu_s = strain_shear(g0, g1, g2, g3, u0, u1, u2, u3);
    let el_a = strain_axial(g0, g1, g2, g3, l0, l1, l2, l3);
    let el_s = strain_shear(g0, g1, g2, g3, l0, l1, l2, l3);

    let d11 = uni.mat.x + 2.0 * uni.mat.y;
    let d12 = uni.mat.x;
    let mu = uni.mat.y;

    // D · strain_u (axial block couples via d11/d12; shear is μ·γ).
    let s0 = d11 * eu_a.x + d12 * eu_a.y + d12 * eu_a.z;
    let s1 = d12 * eu_a.x + d11 * eu_a.y + d12 * eu_a.z;
    let s2 = d12 * eu_a.x + d12 * eu_a.y + d11 * eu_a.z;
    let ds = mu * eu_s;

    let energy = vol * (el_a.x * s0 + el_a.y * s1 + el_a.z * s2 + dot(el_s, ds));
    out_energy[cell] = energy;
}
