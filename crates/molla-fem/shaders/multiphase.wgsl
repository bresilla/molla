// Level-set curvature + CSF surface-tension force on a periodic 2D grid.
// Mirrors the CPU `level_set_curvature_2d` / `csf_surface_tension_force_2d`
// in `crates/molla-fem/src/multiphase.rs`. Three per-cell passes:
//   normals   : n = ∇φ/|∇φ|
//   curvature : κ = ∇·n
//   csf       : f = σ κ δ_ε(φ) ∇φ

struct U {
    dims: vec4<u32>,   // (nx, ny, count, _)
    params: vec4<f32>, // (dx, sigma, eps, _)
};

@group(0) @binding(0) var<storage, read> phi:              array<f32>;
@group(0) @binding(1) var<storage, read_write> nfield:     array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> kappa:      array<f32>;
@group(0) @binding(3) var<storage, read_write> out_f:      array<vec2<f32>>;

@group(1) @binding(0) var<uniform> u: U;

fn widx(i: i32, j: i32, nx: i32, ny: i32) -> u32 {
    let ii = ((i % nx) + nx) % nx;
    let jj = ((j % ny) + ny) % ny;
    return u32(jj * nx + ii);
}

@compute @workgroup_size(64)
fn normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= u.dims.z) { return; }
    let nx = i32(u.dims.x);
    let ny = i32(u.dims.y);
    let i = i32(cell % u.dims.x);
    let j = i32(cell / u.dims.x);
    let dx = u.params.x;
    let gx = (phi[widx(i + 1, j, nx, ny)] - phi[widx(i - 1, j, nx, ny)]) / (2.0 * dx);
    let gy = (phi[widx(i, j + 1, nx, ny)] - phi[widx(i, j - 1, nx, ny)]) / (2.0 * dx);
    let len = max(sqrt(gx * gx + gy * gy), 1.0e-12);
    nfield[cell] = vec2<f32>(gx / len, gy / len);
}

@compute @workgroup_size(64)
fn curvature(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= u.dims.z) { return; }
    let nx = i32(u.dims.x);
    let ny = i32(u.dims.y);
    let i = i32(cell % u.dims.x);
    let j = i32(cell / u.dims.x);
    let dx = u.params.x;
    let div_nx = (nfield[widx(i + 1, j, nx, ny)].x - nfield[widx(i - 1, j, nx, ny)].x) / (2.0 * dx);
    let div_ny = (nfield[widx(i, j + 1, nx, ny)].y - nfield[widx(i, j - 1, nx, ny)].y) / (2.0 * dx);
    kappa[cell] = div_nx + div_ny;
}

@compute @workgroup_size(64)
fn csf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= u.dims.z) { return; }
    let nx = i32(u.dims.x);
    let ny = i32(u.dims.y);
    let i = i32(cell % u.dims.x);
    let j = i32(cell / u.dims.x);
    let dx = u.params.x;
    let gx = (phi[widx(i + 1, j, nx, ny)] - phi[widx(i - 1, j, nx, ny)]) / (2.0 * dx);
    let gy = (phi[widx(i, j + 1, nx, ny)] - phi[widx(i, j - 1, nx, ny)]) / (2.0 * dx);
    let eps = u.params.z;
    let p = phi[cell];
    var delta = 0.0;
    if (eps > 0.0 && abs(p) < eps) {
        delta = 0.5 / eps * (1.0 + cos(3.141592653589793 * p / eps));
    }
    let scale = u.params.y * kappa[cell] * delta;
    out_f[cell] = vec2<f32>(scale * gx, scale * gy);
}
