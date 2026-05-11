// Stage 44 v4 — reverse adjoint for the band-3 SH-aware 3DGS raytrace.
//
// Identical to splat_raytrace_sh_backward.wgsl except SH_COUNT is 16
// instead of 9, and the basis evaluation includes the seven band-3
// terms (cubic polynomials of the unit view direction). All other
// geometric chain-rule terms (ρ, α, m², scale, rotation) match v3
// exactly.

const MAX_HITS: u32 = 16u;
const NO_HIT_T: f32 = -1.0;
const MIN_TRANSMITTANCE: f32 = 0.003;
const EPS: f32 = 1e-12;

const SH_C0:  f32 =  0.28209479177387814;
const SH_C1:  f32 =  0.4886025119029199;
const SH_C2A: f32 =  1.0925484305920792;
const SH_C2B: f32 = -1.0925484305920792;
const SH_C2C: f32 =  0.31539156525252005;
const SH_C2D: f32 = -1.0925484305920792;
const SH_C2E: f32 =  0.5462742152960396;
const SH_C3A: f32 = -0.5900435899266435;
const SH_C3B: f32 =  2.890611442640554;
const SH_C3C: f32 = -0.4570457994644658;
const SH_C3D: f32 =  0.3731763325901154;
const SH_C3E: f32 = -0.4570457994644658;
const SH_C3F: f32 =  1.445305721320277;
const SH_C3G: f32 = -0.5900435899266435;

const SH_COUNT: u32 = 16u;

struct Uniforms {
    pixel_count:  u32,
    splat_count:  u32,
    max_distance: f32,
    min_alpha:    f32,
};

@group(0) @binding(0) var<storage, read>  splat_pos:     array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>  splat_scale:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>  splat_rot:     array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>  splat_sh:      array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>  splat_opacity: array<f32>;

@group(1) @binding(0) var<storage, read>  ray_origin:     array<vec4<f32>>;
@group(1) @binding(1) var<storage, read>  ray_direction:  array<vec4<f32>>;
@group(1) @binding(2) var<storage, read>  d_loss_d_rgb:   array<vec4<f32>>;

@group(2) @binding(0) var<storage, read_write> grad_pos:     array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> grad_scale:   array<atomic<u32>>;
@group(2) @binding(2) var<storage, read_write> grad_rot:     array<atomic<u32>>;
// 16 vec4 per splat → stride 64 u32 slots per splat. `.w` of each
// coefficient is unused.
@group(2) @binding(3) var<storage, read_write> grad_sh:      array<atomic<u32>>;
@group(2) @binding(4) var<storage, read_write> grad_opacity: array<atomic<u32>>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

fn atomic_add_pos(slot: u32, v: f32) {
    if (v == 0.0) { return; }
    var old = atomicLoad(&grad_pos[slot]);
    loop {
        let o = bitcast<f32>(old);
        let new_bits = bitcast<u32>(o + v);
        let r = atomicCompareExchangeWeak(&grad_pos[slot], old, new_bits);
        if (r.exchanged) { return; }
        old = r.old_value;
    }
}
fn atomic_add_scale(slot: u32, v: f32) {
    if (v == 0.0) { return; }
    var old = atomicLoad(&grad_scale[slot]);
    loop {
        let o = bitcast<f32>(old);
        let new_bits = bitcast<u32>(o + v);
        let r = atomicCompareExchangeWeak(&grad_scale[slot], old, new_bits);
        if (r.exchanged) { return; }
        old = r.old_value;
    }
}
fn atomic_add_rot(slot: u32, v: f32) {
    if (v == 0.0) { return; }
    var old = atomicLoad(&grad_rot[slot]);
    loop {
        let o = bitcast<f32>(old);
        let new_bits = bitcast<u32>(o + v);
        let r = atomicCompareExchangeWeak(&grad_rot[slot], old, new_bits);
        if (r.exchanged) { return; }
        old = r.old_value;
    }
}
fn atomic_add_sh(slot: u32, v: f32) {
    if (v == 0.0) { return; }
    var old = atomicLoad(&grad_sh[slot]);
    loop {
        let o = bitcast<f32>(old);
        let new_bits = bitcast<u32>(o + v);
        let r = atomicCompareExchangeWeak(&grad_sh[slot], old, new_bits);
        if (r.exchanged) { return; }
        old = r.old_value;
    }
}
fn atomic_add_opacity(slot: u32, v: f32) {
    if (v == 0.0) { return; }
    var old = atomicLoad(&grad_opacity[slot]);
    loop {
        let o = bitcast<f32>(old);
        let new_bits = bitcast<u32>(o + v);
        let r = atomicCompareExchangeWeak(&grad_opacity[slot], old, new_bits);
        if (r.exchanged) { return; }
        old = r.old_value;
    }
}

fn sh_basis_band3(dir: vec3<f32>) -> array<f32, 16> {
    var basis: array<f32, 16>;
    let len_sq = dot(dir, dir);
    if (len_sq <= 0.0) {
        basis[0] = SH_C0;
        for (var i: u32 = 1u; i < SH_COUNT; i = i + 1u) {
            basis[i] = 0.0;
        }
        return basis;
    }
    let inv_len = inverseSqrt(len_sq);
    let x = dir.x * inv_len;
    let y = dir.y * inv_len;
    let z = dir.z * inv_len;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    basis[0] = SH_C0;
    basis[1] = -SH_C1 * y;
    basis[2] =  SH_C1 * z;
    basis[3] = -SH_C1 * x;
    basis[4] =  SH_C2A * (x * y);
    basis[5] =  SH_C2B * (y * z);
    basis[6] =  SH_C2C * (3.0 * zz - 1.0);
    basis[7] =  SH_C2D * (x * z);
    basis[8] =  SH_C2E * (xx - yy);
    basis[9]  = SH_C3A * y * (3.0 * xx - yy);
    basis[10] = SH_C3B * (x * y * z);
    basis[11] = SH_C3C * y * (5.0 * zz - 1.0);
    basis[12] = SH_C3D * z * (5.0 * zz - 3.0);
    basis[13] = SH_C3E * x * (5.0 * zz - 1.0);
    basis[14] = SH_C3F * z * (xx - yy);
    basis[15] = SH_C3G * x * (xx - 3.0 * yy);
    return basis;
}

fn eval_sh_color(splat_idx: u32, basis: array<f32, 16>) -> vec3<f32> {
    let base = splat_idx * SH_COUNT;
    var rgb = vec3<f32>(0.0);
    for (var k: u32 = 0u; k < SH_COUNT; k = k + 1u) {
        rgb = rgb + splat_sh[base + k].xyz * basis[k];
    }
    return rgb;
}

fn ray_splat_response(
    splat_idx: u32,
    ro_world: vec3<f32>,
    rd_world: vec3<f32>,
    max_distance: f32,
    min_alpha: f32,
) -> vec2<f32> {
    let centre = splat_pos[splat_idx].xyz;
    let scale  = splat_scale[splat_idx].xyz;
    let rot    = splat_rot[splat_idx];
    let opacity = splat_opacity[splat_idx];

    let inv_rot = quat_conj(rot);
    let centered = ro_world - centre;
    let o_rot = quat_rotate(inv_rot, centered);
    let d_rot = quat_rotate(inv_rot, rd_world);
    let inv_s = vec3<f32>(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z);
    let o_local = o_rot * inv_s;
    let d_local = d_rot * inv_s;

    let d2 = dot(d_local, d_local);
    if (d2 < EPS) { return vec2<f32>(0.0, NO_HIT_T); }
    let crd = cross(d_local, o_local);
    let m2 = dot(crd, crd) / d2;
    let rho = exp(-0.5 * m2);
    let alpha = clamp(opacity * rho, 0.0, 1.0);
    if (alpha < min_alpha) { return vec2<f32>(0.0, NO_HIT_T); }

    let to_center = centre - ro_world;
    let rd_len2 = dot(rd_world, rd_world);
    if (rd_len2 < EPS) { return vec2<f32>(0.0, NO_HIT_T); }
    let t = dot(to_center, rd_world) / rd_len2;
    if (t <= 0.0 || t > max_distance) { return vec2<f32>(0.0, NO_HIT_T); }
    return vec2<f32>(alpha, t);
}

// Backprop one hit. `dL_dc` is the per-channel gradient on the SH-
// composed colour at this hit; `dL_dalpha` is the back-to-front 3DGS
// recurrence term. `basis` is the 16 SH basis values for the pixel's
// view direction (constant per pixel — cached at top of `backward`).
fn backprop_one_hit(
    splat_idx: u32,
    ro_world: vec3<f32>,
    rd_world: vec3<f32>,
    basis: array<f32, 16>,
    dL_dc: vec3<f32>,
    dL_dalpha: f32,
) {
    let centre = splat_pos[splat_idx].xyz;
    let scale  = splat_scale[splat_idx].xyz;
    let rot    = splat_rot[splat_idx];
    let opacity = splat_opacity[splat_idx];

    let inv_rot = quat_conj(rot);
    let centered = ro_world - centre;
    let o_rot = quat_rotate(inv_rot, centered);
    let d_rot = quat_rotate(inv_rot, rd_world);
    let inv_s = vec3<f32>(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z);
    let o_local = o_rot * inv_s;
    let d_local = d_rot * inv_s;
    let d2 = dot(d_local, d_local);
    let crd = cross(d_local, o_local);
    let m2 = dot(crd, crd) / d2;
    let rho = exp(-0.5 * m2);
    let alpha_raw = opacity * rho;
    let alpha = clamp(alpha_raw, 0.0, 1.0);

    // ── SH colour gradient ────────────────────────────────────────
    // Each splat's SH region in `grad_sh` is 16 vec4 = 64 u32 slots.
    let sh_base = splat_idx * SH_COUNT * 4u;
    for (var k: u32 = 0u; k < SH_COUNT; k = k + 1u) {
        let b = basis[k];
        let slot = sh_base + k * 4u;
        atomic_add_sh(slot + 0u, dL_dc.x * b);
        atomic_add_sh(slot + 1u, dL_dc.y * b);
        atomic_add_sh(slot + 2u, dL_dc.z * b);
    }

    if (alpha_raw <= 0.0 || alpha_raw >= 1.0) {
        return;
    }

    let dL_da_raw = dL_dalpha;
    let dL_dopacity = rho * dL_da_raw;
    let dL_drho     = opacity * dL_da_raw;
    atomic_add_opacity(splat_idx, dL_dopacity);

    let dL_dm2 = -0.5 * rho * dL_drho;
    let inv_d2 = 1.0 / d2;
    let dL_dcrd = (2.0 * inv_d2 * dL_dm2) * crd;
    let dL_dd2  = -m2 * inv_d2 * dL_dm2;

    var dL_ddl = cross(o_local, dL_dcrd);
    let dL_dol = cross(dL_dcrd, d_local);
    dL_ddl = dL_ddl + (2.0 * dL_dd2) * d_local;

    let dL_doR = dL_dol * inv_s;
    let dL_ddR = dL_ddl * inv_s;

    let dL_dinv_s = vec3<f32>(
        dL_dol.x * o_rot.x + dL_ddl.x * d_rot.x,
        dL_dol.y * o_rot.y + dL_ddl.y * d_rot.y,
        dL_dol.z * o_rot.z + dL_ddl.z * d_rot.z,
    );
    let inv_s2 = inv_s * inv_s;
    let dL_dscale = vec3<f32>(
        -dL_dinv_s.x * inv_s2.x,
        -dL_dinv_s.y * inv_s2.y,
        -dL_dinv_s.z * inv_s2.z,
    );
    let sbase = splat_idx * 4u;
    atomic_add_scale(sbase + 0u, dL_dscale.x);
    atomic_add_scale(sbase + 1u, dL_dscale.y);
    atomic_add_scale(sbase + 2u, dL_dscale.z);

    let dL_dcentered = quat_rotate(rot, dL_doR);
    let dL_dpos = -dL_dcentered;
    let pbase = splat_idx * 4u;
    atomic_add_pos(pbase + 0u, dL_dpos.x);
    atomic_add_pos(pbase + 1u, dL_dpos.y);
    atomic_add_pos(pbase + 2u, dL_dpos.z);

    let c = centered;
    let r = rd_world;
    let dM00 = dL_doR.x * c.x + dL_ddR.x * r.x;
    let dM01 = dL_doR.x * c.y + dL_ddR.x * r.y;
    let dM02 = dL_doR.x * c.z + dL_ddR.x * r.z;
    let dM10 = dL_doR.y * c.x + dL_ddR.y * r.x;
    let dM11 = dL_doR.y * c.y + dL_ddR.y * r.y;
    let dM12 = dL_doR.y * c.z + dL_ddR.y * r.z;
    let dM20 = dL_doR.z * c.x + dL_ddR.z * r.x;
    let dM21 = dL_doR.z * c.y + dL_ddR.z * r.y;
    let dM22 = dL_doR.z * c.z + dL_ddR.z * r.z;

    let x = rot.x;
    let y = rot.y;
    let z = rot.z;
    let w = rot.w;

    var dq = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    dq.y += dM00 * (-4.0 * y);
    dq.z += dM00 * (-4.0 * z);
    dq.x += dM01 * (2.0 * y);
    dq.y += dM01 * (2.0 * x);
    dq.z += dM01 * (2.0 * w);
    dq.w += dM01 * (2.0 * z);
    dq.x += dM02 * (2.0 * z);
    dq.y += dM02 * (-2.0 * w);
    dq.z += dM02 * (2.0 * x);
    dq.w += dM02 * (-2.0 * y);
    dq.x += dM10 * (2.0 * y);
    dq.y += dM10 * (2.0 * x);
    dq.z += dM10 * (-2.0 * w);
    dq.w += dM10 * (-2.0 * z);
    dq.x += dM11 * (-4.0 * x);
    dq.z += dM11 * (-4.0 * z);
    dq.x += dM12 * (2.0 * w);
    dq.y += dM12 * (2.0 * z);
    dq.z += dM12 * (2.0 * y);
    dq.w += dM12 * (2.0 * x);
    dq.x += dM20 * (2.0 * z);
    dq.y += dM20 * (2.0 * w);
    dq.z += dM20 * (2.0 * x);
    dq.w += dM20 * (2.0 * y);
    dq.x += dM21 * (-2.0 * w);
    dq.y += dM21 * (2.0 * z);
    dq.z += dM21 * (2.0 * y);
    dq.w += dM21 * (-2.0 * x);
    dq.x += dM22 * (-4.0 * x);
    dq.y += dM22 * (-4.0 * y);

    let qbase = splat_idx * 4u;
    atomic_add_rot(qbase + 0u, dq.x);
    atomic_add_rot(qbase + 1u, dq.y);
    atomic_add_rot(qbase + 2u, dq.z);
    atomic_add_rot(qbase + 3u, dq.w);
}

@compute @workgroup_size(64)
fn backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.pixel_count) { return; }

    let ro = ray_origin[tid].xyz;
    let rd = ray_direction[tid].xyz;
    let dLdC = d_loss_d_rgb[tid].xyz;
    let basis = sh_basis_band3(rd);

    var hit_t:      array<f32, 16>;
    var hit_alpha:  array<f32, 16>;
    var hit_color:  array<vec3<f32>, 16>;
    var hit_index:  array<u32, 16>;
    for (var k: u32 = 0u; k < MAX_HITS; k = k + 1u) {
        hit_t[k]     = uniforms.max_distance + 1.0;
        hit_alpha[k] = 0.0;
        hit_color[k] = vec3<f32>(0.0);
        hit_index[k] = 0u;
    }
    var num_hits: u32 = 0u;

    for (var i: u32 = 0u; i < uniforms.splat_count; i = i + 1u) {
        let resp = ray_splat_response(i, ro, rd, uniforms.max_distance, uniforms.min_alpha);
        let alpha = resp.x;
        let t = resp.y;
        if (t < 0.0) { continue; }

        var insert_idx: u32 = MAX_HITS;
        for (var k: u32 = 0u; k < MAX_HITS; k = k + 1u) {
            if (t < hit_t[k]) { insert_idx = k; break; }
        }
        if (insert_idx >= MAX_HITS) { continue; }
        if (num_hits < MAX_HITS) { num_hits = num_hits + 1u; }
        var k: u32 = num_hits;
        while (k > insert_idx + 1u) {
            let prev = k - 1u;
            hit_t[prev]     = hit_t[prev - 1u];
            hit_alpha[prev] = hit_alpha[prev - 1u];
            hit_color[prev] = hit_color[prev - 1u];
            hit_index[prev] = hit_index[prev - 1u];
            k = k - 1u;
        }
        hit_t[insert_idx]     = t;
        hit_alpha[insert_idx] = alpha;
        hit_color[insert_idx] = eval_sh_color(i, basis);
        hit_index[insert_idx] = i;
    }

    var T_at: array<f32, 16>;
    var cutoff: u32 = num_hits;
    var trans: f32 = 1.0;
    for (var k: u32 = 0u; k < num_hits; k = k + 1u) {
        T_at[k] = trans;
        trans = trans * (1.0 - hit_alpha[k]);
        if (trans < MIN_TRANSMITTANCE) {
            cutoff = k + 1u;
            break;
        }
    }
    if (cutoff == 0u) { return; }

    var S = vec3<f32>(0.0);
    var i_signed: i32 = i32(cutoff) - 1;
    loop {
        if (i_signed < 0) { break; }
        let k = u32(i_signed);
        let a = hit_alpha[k];
        let cv = hit_color[k];
        let T = T_at[k];
        let aT = a * T;
        let dL_dc = vec3<f32>(aT * dLdC.x, aT * dLdC.y, aT * dLdC.z);
        let one_minus_a = max(1.0 - a, EPS);
        let term1 = (cv.x * dLdC.x + cv.y * dLdC.y + cv.z * dLdC.z) * T;
        let term2 = (S.x  * dLdC.x + S.y  * dLdC.y + S.z  * dLdC.z) / one_minus_a;
        let dL_dalpha = term1 - term2;

        backprop_one_hit(hit_index[k], ro, rd, basis, dL_dc, dL_dalpha);

        S = S + cv * (a * T);

        i_signed = i_signed - 1;
    }
}
