// Stage 44 v2 — reverse adjoint for the 3D-Gaussian splat raytrace.
//
// One thread per pixel. Each thread re-runs the same deterministic
// top-K insertion sort as the forward kernel (the front-to-back order
// is uniquely determined by the splat array + ray), then walks the
// hit list back-to-front to accumulate per-splat gradients via the
// standard 3DGS α-composite recurrence:
//
//   For each hit i (back-to-front, i.e. last → first):
//     dL/dc_i  = α_i · T_i · dL/dC_out
//     dL/dα_i  = T_i · c_i · dL/dC_out − S_after · dL/dC_out / (1−α_i)
//   where T_i = ∏_{j<i}(1−α_j) is the front-prefix transmittance and
//   S_after = Σ_{j>i} c_j α_j T_j is the rgb that piles up behind i.
//
// From dL/dα each splat contributes (through ρ = exp(−½ m²),
// α = opacity · ρ) to gradients on opacity, ρ, m², then through
// the canonical-frame chain (crd, d_local, o_local) → (scale,
// rotation, position). Per-splat gradients are scattered via the
// standard `atomicCompareExchangeWeak` CAS loop (WGSL has no native
// `atomic<f32>`; same idiom as `fem_assembly.wgsl`).
//
// Layout matches the forward kernel; outputs additionally carry the
// per-splat gradient buffers and per-pixel `dL/dC_out`. The forward
// kernel writes `alpha_out = 1 − T_final`; we ignore the gradient
// flowing back through alpha_out for v2 (the pose-recovery loss only
// uses rgb), so `dL/dα_out` doesn't enter here. Extending to alpha-
// channel gradients is a one-line addition: add a `T_i · T_final /
// ((1-α_i)·T_i)` extra term to dL/dα_i.

const MAX_HITS: u32 = 16u;
const NO_HIT_T: f32 = -1.0;
const MIN_TRANSMITTANCE: f32 = 0.003;
const EPS: f32 = 1e-12;

struct Uniforms {
    pixel_count:  u32,
    splat_count:  u32,
    max_distance: f32,
    min_alpha:    f32,
};

// ── Splat inputs (read-only) ─────────────────────────────────────
@group(0) @binding(0) var<storage, read>  splat_pos:     array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>  splat_scale:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>  splat_rot:     array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>  splat_color:   array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>  splat_opacity: array<f32>;

// ── Rays + per-pixel upstream gradient (read-only) ────────────────
@group(1) @binding(0) var<storage, read>  ray_origin:     array<vec4<f32>>;
@group(1) @binding(1) var<storage, read>  ray_direction:  array<vec4<f32>>;
@group(1) @binding(2) var<storage, read>  d_loss_d_rgb:   array<vec4<f32>>;

// ── Per-splat gradient outputs (atomic-add scatter) ───────────────
// Padded vec3 lives in slots 0..3 of each splat's stride-4 region.
@group(2) @binding(0) var<storage, read_write> grad_pos:     array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> grad_scale:   array<atomic<u32>>;
@group(2) @binding(2) var<storage, read_write> grad_rot:     array<atomic<u32>>;
@group(2) @binding(3) var<storage, read_write> grad_color:   array<atomic<u32>>;
@group(2) @binding(4) var<storage, read_write> grad_opacity: array<atomic<u32>>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

// ── Quaternion helpers (mirror forward) ──────────────────────────
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

// ── Atomic-add helpers (CAS loop on bit-cast f32). ────────────────
// One per output array — WGSL needs the atomic var distinct per call
// site, but we only have 5 outputs so duplication is acceptable.

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
fn atomic_add_color(slot: u32, v: f32) {
    if (v == 0.0) { return; }
    var old = atomicLoad(&grad_color[slot]);
    loop {
        let o = bitcast<f32>(old);
        let new_bits = bitcast<u32>(o + v);
        let r = atomicCompareExchangeWeak(&grad_color[slot], old, new_bits);
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

// ── Forward response (identical to splat_raytrace.wgsl) ──────────
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

// ── Backprop one hit. Scatters gradients onto the per-splat buffers
// via atomic adds. `dL_dalpha` already includes the back-to-front
// 3DGS recurrence term; `dL_dc` is the per-channel rgb gradient.
fn backprop_one_hit(
    splat_idx: u32,
    ro_world: vec3<f32>,
    rd_world: vec3<f32>,
    dL_dc: vec3<f32>,
    dL_dalpha: f32,
) {
    let centre = splat_pos[splat_idx].xyz;
    let scale  = splat_scale[splat_idx].xyz;
    let rot    = splat_rot[splat_idx];
    let opacity = splat_opacity[splat_idx];

    // --- Re-run the forward to recover all intermediates. ----------
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

    // Scatter rgb gradient. Caller has already multiplied by α·T·dLdC,
    // so we just add it onto the color slot (per channel independent).
    let cbase = splat_idx * 4u; // stride-4 padded vec4 for color
    atomic_add_color(cbase + 0u, dL_dc.x);
    atomic_add_color(cbase + 1u, dL_dc.y);
    atomic_add_color(cbase + 2u, dL_dc.z);

    // If the clamp saturated, the geometric chain has zero gradient
    // (α_raw is outside (0,1)). Skip the rest in that case.
    if (alpha_raw <= 0.0 || alpha_raw >= 1.0) {
        return;
    }

    // dL/d α_raw = dL/dα.
    let dL_da_raw = dL_dalpha;
    // α_raw = opacity · ρ.
    let dL_dopacity = rho * dL_da_raw;
    let dL_drho     = opacity * dL_da_raw;
    atomic_add_opacity(splat_idx, dL_dopacity);

    // ρ = exp(-½ m²) → dρ/dm² = -½ ρ.
    let dL_dm2 = -0.5 * rho * dL_drho;

    // m² = |crd|² / d². d m²/d crd = 2 crd / d²; d m²/d d² = -m²/d².
    let inv_d2 = 1.0 / d2;
    let dL_dcrd = (2.0 * inv_d2 * dL_dm2) * crd;
    let dL_dd2  = -m2 * inv_d2 * dL_dm2;

    // crd = d_l × o_l → d crd / d d_l = -[o_l]_×, d crd / d o_l = [d_l]_×.
    // Equivalently: dL/dd_l = o_l × dL_dcrd; dL/do_l = dL_dcrd × d_l.
    var dL_ddl = cross(o_local, dL_dcrd);
    let dL_dol = cross(dL_dcrd, d_local);

    // d² = dot(d_l, d_l) → add 2 d_l · dL_dd² to dL_ddl.
    dL_ddl = dL_ddl + (2.0 * dL_dd2) * d_local;

    // o_l = oR * inv_s, d_l = dR * inv_s (component-wise).
    let dL_doR = dL_dol * inv_s;
    let dL_ddR = dL_ddl * inv_s;

    // dL/dinv_s_k = (dL/do_l_k) * oR_k + (dL/dd_l_k) * dR_k.
    let dL_dinv_s = vec3<f32>(
        dL_dol.x * o_rot.x + dL_ddl.x * d_rot.x,
        dL_dol.y * o_rot.y + dL_ddl.y * d_rot.y,
        dL_dol.z * o_rot.z + dL_ddl.z * d_rot.z,
    );
    // inv_s = 1/scale → d inv_s / d scale = -1/scale².
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

    // (oR, dR) = M · (centered, rd_world) where M = R(q⁻¹).
    // dL/dcentered = M^T · dL_doR = R(q) · dL_doR = quat_rotate(q, dL_doR).
    // Position: μ = ro - centered (well, centered = ro - μ → dμ = -d centered).
    let dL_dcentered = quat_rotate(rot, dL_doR);
    let dL_dpos = -dL_dcentered;
    let pbase = splat_idx * 4u;
    atomic_add_pos(pbase + 0u, dL_dpos.x);
    atomic_add_pos(pbase + 1u, dL_dpos.y);
    atomic_add_pos(pbase + 2u, dL_dpos.z);

    // ── Rotation gradient. M = R(q⁻¹) entries are quadratic in q.
    // Build dL/dM_{ij} = (dL_doR)_i · centered_j + (dL_ddR)_i · rd_j.
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

    // M = R(q⁻¹) with q⁻¹ = (-qx, -qy, -qz, qw):
    //   M00 = 1 - 2(y²+z²)
    //   M01 = 2(xy + zw)
    //   M02 = 2(xz - yw)
    //   M10 = 2(xy - zw)
    //   M11 = 1 - 2(x²+z²)
    //   M12 = 2(yz + xw)
    //   M20 = 2(xz + yw)
    //   M21 = 2(yz - xw)
    //   M22 = 1 - 2(x²+y²)
    let x = rot.x;
    let y = rot.y;
    let z = rot.z;
    let w = rot.w;

    // ∂M_ij/∂qα — derived above. We accumulate dL/dq directly.
    var dq = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    // dM00: ∂/∂y = -4y, ∂/∂z = -4z.
    dq.y += dM00 * (-4.0 * y);
    dq.z += dM00 * (-4.0 * z);
    // dM01: ∂x=2y, ∂y=2x, ∂z=2w, ∂w=2z.
    dq.x += dM01 * (2.0 * y);
    dq.y += dM01 * (2.0 * x);
    dq.z += dM01 * (2.0 * w);
    dq.w += dM01 * (2.0 * z);
    // dM02: ∂x=2z, ∂y=-2w, ∂z=2x, ∂w=-2y.
    dq.x += dM02 * (2.0 * z);
    dq.y += dM02 * (-2.0 * w);
    dq.z += dM02 * (2.0 * x);
    dq.w += dM02 * (-2.0 * y);
    // dM10: ∂x=2y, ∂y=2x, ∂z=-2w, ∂w=-2z.
    dq.x += dM10 * (2.0 * y);
    dq.y += dM10 * (2.0 * x);
    dq.z += dM10 * (-2.0 * w);
    dq.w += dM10 * (-2.0 * z);
    // dM11: ∂x=-4x, ∂z=-4z.
    dq.x += dM11 * (-4.0 * x);
    dq.z += dM11 * (-4.0 * z);
    // dM12: ∂x=2w, ∂y=2z, ∂z=2y, ∂w=2x.
    dq.x += dM12 * (2.0 * w);
    dq.y += dM12 * (2.0 * z);
    dq.z += dM12 * (2.0 * y);
    dq.w += dM12 * (2.0 * x);
    // dM20: ∂x=2z, ∂y=2w, ∂z=2x, ∂w=2y.
    dq.x += dM20 * (2.0 * z);
    dq.y += dM20 * (2.0 * w);
    dq.z += dM20 * (2.0 * x);
    dq.w += dM20 * (2.0 * y);
    // dM21: ∂x=-2w, ∂y=2z, ∂z=2y, ∂w=-2x.
    dq.x += dM21 * (-2.0 * w);
    dq.y += dM21 * (2.0 * z);
    dq.z += dM21 * (2.0 * y);
    dq.w += dM21 * (-2.0 * x);
    // dM22: ∂x=-4x, ∂y=-4y.
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

    // ── Re-run the same insertion sort as forward. ────────────────
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
        hit_color[insert_idx] = splat_color[i].xyz;
        hit_index[insert_idx] = i;
    }

    // ── First pass: replay forward to recover T_i for each hit and
    // detect the early-exit cutoff (transmittance < MIN_TRANSMITTANCE).
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

    // ── Second pass: back-to-front. ───────────────────────────────
    // Maintain S = Σ_{j>i} c_j α_j T_j  (the rgb that piles up after
    // hit i in front-to-back order).
    var S = vec3<f32>(0.0);
    var i_signed: i32 = i32(cutoff) - 1;
    loop {
        if (i_signed < 0) { break; }
        let k = u32(i_signed);
        let a = hit_alpha[k];
        let cv = hit_color[k];
        let T = T_at[k];
        // dL/dc_i.k = α_i · T_i · dL/dC_out.k (each rgb channel
        // independent). We hand `dL_dc` to backprop_one_hit which
        // scatters into the per-splat color slots component-wise.
        let aT = a * T;
        let dL_dc = vec3<f32>(aT * dLdC.x, aT * dLdC.y, aT * dLdC.z);
        // dL/dα_i = (Σ_k c_i.k · dL/dC_out.k) · T_i − (Σ_k S.k · dL/dC_out.k) / (1−α_i)
        let one_minus_a = max(1.0 - a, EPS);
        let term1 = (cv.x * dLdC.x + cv.y * dLdC.y + cv.z * dLdC.z) * T;
        let term2 = (S.x  * dLdC.x + S.y  * dLdC.y + S.z  * dLdC.z) / one_minus_a;
        let dL_dalpha = term1 - term2;

        backprop_one_hit(hit_index[k], ro, rd, dL_dc, dL_dalpha);

        // Prepend this hit to S for the next (more-front) splat.
        S = S + cv * (a * T);

        i_signed = i_signed - 1;
    }
}
