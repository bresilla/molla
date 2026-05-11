// Stage 40 — Implicit MPM colored Gauss-Seidel sweep over the
// **full 27-point Newton Hessian** on a regular grid.
//
// Each thread updates one grid cell in the current color group with
// the GS rule
//
//   v_i ← (M_i · v_pred_i − dt · Σ_{j ≠ i} K_{ij} · v_j) /
//         (M_i + dt · K_{ii})
//
// where `K_{i,i+δ}` (δ ∈ {-1,0,1}³) is stored in
// `k_stencil[ci * 27 + s(δ)]` with `s(δ) = (δz+1)·9 + (δy+1)·3 + (δx+1)`.
// `STENCIL_DIAG = 13` indexes the on-diagonal entry.
//
// The host runs all bookkeeping (P2G assembly, color-group construction,
// floor enforcement, G2P) on CPU; this kernel only owns the inner
// colored-GS sweep so the Stage 40 GPU parity gate can compare a
// CPU-only timestep against an identical timestep where the inner solve
// runs on the GPU.
//
// Inputs (read-only):
//   g_mass            [g_total]            f32        per-cell mass
//   k_stencil         [g_total * 27]       f32        27-stencil Hessian entries
//   v_pred            [g_total]            vec3<f32>  predicted velocity
//   color_cells       [color_n]            u32        cell indices for this color
// I/O (read-write):
//   v_grid            [g_total]            vec3<f32>  velocity (updated in place)
//
// Uniforms:
//   grid_size, color_n, dt, use_off_diag

const WG_SIZE: u32 = 64u;
const STENCIL_DIAG: u32 = 13u;

struct Uniforms {
    grid_size: u32,
    color_n: u32,
    use_off_diag: u32,
    _pad: u32,
    dt: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<storage, read> g_mass: array<f32>;
@group(0) @binding(1) var<storage, read> k_stencil: array<f32>;
@group(0) @binding(2) var<storage, read> v_pred: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> color_cells: array<u32>;
@group(0) @binding(4) var<storage, read_write> v_grid: array<vec3<f32>>;
@group(1) @binding(0) var<uniform> U: Uniforms;

fn cell_flat(ix: i32, iy: i32, iz: i32, g: i32) -> i32 {
    if (ix < 0 || iy < 0 || iz < 0 || ix >= g || iy >= g || iz >= g) {
        return -1;
    }
    return ix + iy * g + iz * g * g;
}

fn stencil_idx(dx: i32, dy: i32, dz: i32) -> u32 {
    return u32((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1));
}

@compute @workgroup_size(WG_SIZE)
fn gs_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= U.color_n) {
        return;
    }
    let ci = color_cells[i];
    let g = i32(U.grid_size);
    let g2 = g * g;
    let m = g_mass[ci];
    let k_diag = k_stencil[ci * 27u + STENCIL_DIAG];
    let denom = m + U.dt * k_diag;
    if (denom <= 0.0) {
        return;
    }

    let ix = i32(ci) % g;
    let iy = (i32(ci) / g) % g;
    let iz = i32(ci) / g2;

    var off = vec3<f32>(0.0, 0.0, 0.0);
    if (U.use_off_diag != 0u) {
        for (var dz = -1; dz <= 1; dz = dz + 1) {
            for (var dy = -1; dy <= 1; dy = dy + 1) {
                for (var dx = -1; dx <= 1; dx = dx + 1) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    let cj = cell_flat(ix + dx, iy + dy, iz + dz, g);
                    if (cj < 0) {
                        continue;
                    }
                    let s = stencil_idx(dx, dy, dz);
                    let kij = k_stencil[ci * 27u + s];
                    if (kij != 0.0) {
                        off = off + kij * v_grid[cj];
                    }
                }
            }
        }
    }

    let rhs = m * v_pred[ci] - U.dt * off;
    let v_new = rhs / denom;
    v_grid[ci] = v_new;
}
