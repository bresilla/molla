// Stage 37.8 — workgroup-tiled GEMM `C = A · B`.
//
// Layout: row-major `A` (M × K), `B` (K × N), `C` (M × N). Each
// workgroup cooperatively computes a 16×16 block of `C`. Inputs are
// staged into shared memory in 16×16 chunks along the K axis to
// amortise global-memory traffic; per-thread accumulation runs in
// registers and writes `C` once at the end. Single entry point.

const TILE: u32 = 16u;

struct Uniforms {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> a_buf: array<f32>;
@group(0) @binding(1) var<storage, read> b_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> c_buf: array<f32>;
@group(1) @binding(0) var<uniform> u: Uniforms;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn tile_matmul(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let lr = lid.y;
    let lc = lid.x;

    let m = u.m;
    let n = u.n;
    let k = u.k;

    var acc: f32 = 0.0;

    let num_tiles = (k + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + lc;
        let b_row = t * TILE + lr;

        // Cooperative load of A tile (row, a_col).
        if (row < m && a_col < k) {
            tile_a[lr][lc] = a_buf[row * k + a_col];
        } else {
            tile_a[lr][lc] = 0.0;
        }
        // Cooperative load of B tile (b_row, col).
        if (b_row < k && col < n) {
            tile_b[lr][lc] = b_buf[b_row * n + col];
        } else {
            tile_b[lr][lc] = 0.0;
        }

        workgroupBarrier();

        // Accumulate into register.
        for (var i: u32 = 0u; i < TILE; i = i + 1u) {
            acc = acc + tile_a[lr][i] * tile_b[i][lc];
        }

        workgroupBarrier();
    }

    if (row < m && col < n) {
        c_buf[row * n + col] = acc;
    }
}
