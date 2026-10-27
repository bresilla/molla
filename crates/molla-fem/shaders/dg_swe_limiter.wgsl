// Stage 43 v5 — DG-P1 SWE Barth-Jespersen limiters.
//
// State layout:
//   state[cell * 9 + node * 3 + comp], node in 0..3, comp = h/hu/hv.
//
// The kernels read from state_in and write to state_out to preserve the
// CPU algorithm's "all cell averages from the pre-limit snapshot"
// contract while still keeping the limiter resident on GPU.

struct CellMeta {
    neighbor0: u32,
    neighbor1: u32,
    neighbor2: u32,
    _pad0: u32,
    nx: f32,
    ny: f32,
    _pad1: f32,
    _pad2: f32,
};

struct Uniforms {
    n_cells: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    g: f32,
    physical_h_min: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<storage, read>       state_in  : array<f32>;
@group(0) @binding(1) var<storage, read>       cells     : array<CellMeta>;
@group(1) @binding(0) var<storage, read_write> state_out : array<f32>;
@group(1) @binding(1) var<uniform>             u         : Uniforms;

const WG_SIZE: u32 = 64u;
const NODES: u32 = 3u;
const COMPONENTS: u32 = 3u;
const INVALID: u32 = 0xffffffffu;
const EPS: f32 = 1.0e-30;
const H_FLOOR: f32 = 1.0e-8;

fn idx(cell: u32, node: u32, comp: u32) -> u32 {
    return cell * NODES * COMPONENTS + node * COMPONENTS + comp;
}

fn load_avg(cell: u32) -> vec3<f32> {
    var avg = vec3<f32>(0.0);
    for (var n: u32 = 0u; n < NODES; n = n + 1u) {
        avg.x = avg.x + state_in[idx(cell, n, 0u)];
        avg.y = avg.y + state_in[idx(cell, n, 1u)];
        avg.z = avg.z + state_in[idx(cell, n, 2u)];
    }
    return avg / 3.0;
}

fn neighbor_at(cell_meta: CellMeta, slot: u32) -> u32 {
    if (slot == 0u) { return cell_meta.neighbor0; }
    if (slot == 1u) { return cell_meta.neighbor1; }
    return cell_meta.neighbor2;
}

fn bj_phi(center: f32, v0: f32, v1: f32, v2: f32, lo: f32, hi: f32) -> f32 {
    var phi = 1.0;
    var vals = array<f32, 3>(v0, v1, v2);
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        let d = vals[i] - center;
        var yi = 1.0;
        if (d > 1.0e-30) {
            yi = min((hi - center) / d, 1.0);
        } else if (d < -1.0e-30) {
            yi = min((lo - center) / d, 1.0);
        }
        phi = min(phi, clamp(yi, 0.0, 1.0));
    }
    return phi;
}

fn mat_vec(m0: vec3<f32>, m1: vec3<f32>, m2: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(m0, v), dot(m1, v), dot(m2, v));
}

fn rotate_to_nt(v: vec3<f32>, nx: f32, ny: f32) -> vec3<f32> {
    let tx = -ny;
    let ty = nx;
    return vec3<f32>(v.x, v.y * nx + v.z * ny, v.y * tx + v.z * ty);
}

fn rotate_to_xy(v: vec3<f32>, nx: f32, ny: f32) -> vec3<f32> {
    let tx = -ny;
    let ty = nx;
    return vec3<f32>(v.x, v.y * nx + v.z * tx, v.y * ny + v.z * ty);
}

@compute @workgroup_size(64)
fn limit_swe_scalar_componentwise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= u.n_cells) { return; }

    let cell_meta = cells[cell];
    let avg = load_avg(cell);

    var lo = avg;
    var hi = avg;
    for (var s: u32 = 0u; s < 3u; s = s + 1u) {
        let nb = neighbor_at(cell_meta, s);
        if (nb != INVALID && nb < u.n_cells) {
            let a = load_avg(nb);
            lo = min(lo, a);
            hi = max(hi, a);
        }
    }
    lo.x = max(lo.x, u.physical_h_min);

    var phi = vec3<f32>(1.0);
    for (var c: u32 = 0u; c < COMPONENTS; c = c + 1u) {
        let center = avg[c];
        let min_c = lo[c];
        let max_c = hi[c];
        phi[c] = bj_phi(
            center,
            state_in[idx(cell, 0u, c)],
            state_in[idx(cell, 1u, c)],
            state_in[idx(cell, 2u, c)],
            min_c,
            max_c,
        );
    }

    for (var n: u32 = 0u; n < NODES; n = n + 1u) {
        var limited = vec3<f32>(
            avg.x + phi.x * (state_in[idx(cell, n, 0u)] - avg.x),
            avg.y + phi.y * (state_in[idx(cell, n, 1u)] - avg.y),
            avg.z + phi.z * (state_in[idx(cell, n, 2u)] - avg.z),
        );
        limited.x = max(limited.x, u.physical_h_min);
        state_out[idx(cell, n, 0u)] = limited.x;
        state_out[idx(cell, n, 1u)] = limited.y;
        state_out[idx(cell, n, 2u)] = limited.z;
    }
}

@compute @workgroup_size(64)
fn limit_swe_characteristic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= u.n_cells) { return; }

    let cell_meta = cells[cell];
    let avg = load_avg(cell);

    var nb_avg = vec3<f32>(0.0);
    var nb_count = 0.0;
    for (var s: u32 = 0u; s < 3u; s = s + 1u) {
        let nb = neighbor_at(cell_meta, s);
        if (nb != INVALID && nb < u.n_cells) {
            nb_avg = nb_avg + load_avg(nb);
            nb_count = nb_count + 1.0;
        }
    }
    if (nb_count > 0.0) {
        nb_avg = nb_avg / nb_count;
    } else {
        nb_avg = avg;
    }

    var nx = cell_meta.nx;
    var ny = cell_meta.ny;
    let n_len = sqrt(nx * nx + ny * ny);
    if (n_len > 1.0e-12) {
        nx = nx / n_len;
        ny = ny / n_len;
    } else {
        nx = 1.0;
        ny = 0.0;
    }

    let h_l = max(avg.x, H_FLOOR);
    let h_r = max(nb_avg.x, H_FLOOR);
    let sqrt_h_l = sqrt(h_l);
    let sqrt_h_r = sqrt(h_r);
    let denom = sqrt_h_l + sqrt_h_r;
    let tx = -ny;
    let ty = nx;

    var hu_l = avg.y;
    var hv_l = avg.z;
    var hu_r = nb_avg.y;
    var hv_r = nb_avg.z;
    if (avg.x < H_FLOOR) {
        hu_l = 0.0;
        hv_l = 0.0;
    }
    if (nb_avg.x < H_FLOOR) {
        hu_r = 0.0;
        hv_r = 0.0;
    }
    let un_l = (hu_l * nx + hv_l * ny) / h_l;
    let ut_l = (hu_l * tx + hv_l * ty) / h_l;
    let un_r = (hu_r * nx + hv_r * ny) / h_r;
    let ut_r = (hu_r * tx + hv_r * ty) / h_r;

    var un_hat = 0.5 * (un_l + un_r);
    var ut_hat = 0.5 * (ut_l + ut_r);
    if (denom > EPS) {
        un_hat = (un_l * sqrt_h_l + un_r * sqrt_h_r) / denom;
        ut_hat = (ut_l * sqrt_h_l + ut_r * sqrt_h_r) / denom;
    }
    let h_hat = 0.5 * (h_l + h_r);
    let c_hat = sqrt(u.g * h_hat);
    var inv_c = 0.0;
    if (c_hat > EPS) {
        inv_c = 1.0 / c_hat;
    }

    let l0 = vec3<f32>(0.5 * (1.0 + un_hat * inv_c), -0.5 * inv_c, 0.0);
    let l1 = vec3<f32>(-ut_hat, 0.0, 1.0);
    let l2 = vec3<f32>(0.5 * (1.0 - un_hat * inv_c), 0.5 * inv_c, 0.0);
    let r0 = vec3<f32>(1.0, 0.0, 1.0);
    let r1 = vec3<f32>(un_hat - c_hat, 0.0, un_hat + c_hat);
    let r2 = vec3<f32>(ut_hat, 1.0, ut_hat);

    let avg_nt = rotate_to_nt(avg, nx, ny);
    let w_bar = mat_vec(l0, l1, l2, avg_nt);

    var w_min = w_bar;
    var w_max = w_bar;
    for (var s2: u32 = 0u; s2 < 3u; s2 = s2 + 1u) {
        let nb2 = neighbor_at(cell_meta, s2);
        if (nb2 != INVALID && nb2 < u.n_cells) {
            let nb_nt = rotate_to_nt(load_avg(nb2), nx, ny);
            let w_nb = mat_vec(l0, l1, l2, nb_nt);
            w_min = min(w_min, w_nb);
            w_max = max(w_max, w_nb);
        }
    }

    var w0 = vec3<f32>(0.0);
    var w1 = vec3<f32>(0.0);
    var w2 = vec3<f32>(0.0);
    for (var n: u32 = 0u; n < NODES; n = n + 1u) {
        let nodal = vec3<f32>(
            state_in[idx(cell, n, 0u)],
            state_in[idx(cell, n, 1u)],
            state_in[idx(cell, n, 2u)],
        );
        let w = mat_vec(l0, l1, l2, rotate_to_nt(nodal, nx, ny));
        if (n == 0u) {
            w0 = w;
        } else if (n == 1u) {
            w1 = w;
        } else {
            w2 = w;
        }
    }

    let phi = vec3<f32>(
        bj_phi(w_bar.x, w0.x, w1.x, w2.x, w_min.x, w_max.x),
        bj_phi(w_bar.y, w0.y, w1.y, w2.y, w_min.y, w_max.y),
        bj_phi(w_bar.z, w0.z, w1.z, w2.z, w_min.z, w_max.z),
    );

    for (var n2: u32 = 0u; n2 < NODES; n2 = n2 + 1u) {
        var w = w2;
        if (n2 == 0u) {
            w = w0;
        } else if (n2 == 1u) {
            w = w1;
        }
        let w_limited = w_bar + phi * (w - w_bar);
        let limited_nt = vec3<f32>(
            dot(vec3<f32>(r0.x, r0.y, r0.z), w_limited),
            dot(vec3<f32>(r1.x, r1.y, r1.z), w_limited),
            dot(vec3<f32>(r2.x, r2.y, r2.z), w_limited),
        );
        var limited = rotate_to_xy(limited_nt, nx, ny);
        limited.x = max(limited.x, u.physical_h_min);
        state_out[idx(cell, n2, 0u)] = limited.x;
        state_out[idx(cell, n2, 1u)] = limited.y;
        state_out[idx(cell, n2, 2u)] = limited.z;
    }
}
