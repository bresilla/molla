// Stage 44 v5 — differentiable 3D-Gaussian splat raytrace with a
// diffuse + view-dependent spherical-harmonic colour split, FORWARD
// pass.
//
// Geometry and compositing are identical to splat_raytrace_sh_band3,
// but colour is stored as:
//
//   splat_diffuse[i]        = direction-invariant base RGB
//   splat_sh_view[i*15 + k] = band-1..3 residual coefficient k
//
// Forward colour:
//
//   rgb(d) = diffuse + Σ_{k=0..14} sh_view[k] * Y_{k+1}(d)
//
// i.e. the SH band-0 term is intentionally not multiplied by Y_00.

const NO_HIT_T: f32 = -1.0;
const MAX_HITS: u32 = 16u;
const MIN_TRANSMITTANCE: f32 = 0.003;
const EPS: f32 = 1e-12;

// SH basis constants (must match Rust SH_C0/.../SH_C3G).
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
const SH_VIEW_DEP_COUNT: u32 = 15u;

struct Uniforms {
    pixel_count:  u32,
    splat_count:  u32,
    max_distance: f32,
    min_alpha:    f32,
};

@group(0) @binding(0) var<storage, read>  splat_pos:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>  splat_scale:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>  splat_rot:      array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>  splat_diffuse:  array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>  splat_sh_view:  array<vec4<f32>>;
@group(0) @binding(5) var<storage, read>  splat_opacity:  array<f32>;

@group(1) @binding(0) var<storage, read>  ray_origin:     array<vec4<f32>>;
@group(1) @binding(1) var<storage, read>  ray_direction:  array<vec4<f32>>;

@group(2) @binding(0) var<storage, read_write> out_rgba:   array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> out_depth:  array<f32>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
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

fn eval_diffuse_view_color(splat_idx: u32, basis: array<f32, 16>) -> vec3<f32> {
    var rgb = splat_diffuse[splat_idx].xyz;
    let base = splat_idx * SH_VIEW_DEP_COUNT;
    for (var k: u32 = 0u; k < SH_VIEW_DEP_COUNT; k = k + 1u) {
        rgb = rgb + splat_sh_view[base + k].xyz * basis[k + 1u];
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

@compute @workgroup_size(64)
fn render(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.pixel_count) { return; }

    let ro = ray_origin[tid].xyz;
    let rd = ray_direction[tid].xyz;
    let basis = sh_basis_band3(rd);

    var hit_t:     array<f32, 16>;
    var hit_alpha: array<f32, 16>;
    var hit_color: array<vec3<f32>, 16>;
    for (var k: u32 = 0u; k < MAX_HITS; k = k + 1u) {
        hit_t[k]     = uniforms.max_distance + 1.0;
        hit_alpha[k] = 0.0;
        hit_color[k] = vec3<f32>(0.0);
    }
    var num_hits: u32 = 0u;

    for (var i: u32 = 0u; i < uniforms.splat_count; i = i + 1u) {
        let r = ray_splat_response(i, ro, rd, uniforms.max_distance, uniforms.min_alpha);
        let alpha = r.x;
        let t = r.y;
        if (t < 0.0) { continue; }

        var insert_idx: u32 = MAX_HITS;
        for (var k: u32 = 0u; k < MAX_HITS; k = k + 1u) {
            if (t < hit_t[k]) {
                insert_idx = k;
                break;
            }
        }
        if (insert_idx >= MAX_HITS) {
            continue;
        }

        if (num_hits < MAX_HITS) {
            num_hits = num_hits + 1u;
        }
        var k: u32 = num_hits;
        while (k > insert_idx + 1u) {
            let prev = k - 1u;
            hit_t[prev]     = hit_t[prev - 1u];
            hit_alpha[prev] = hit_alpha[prev - 1u];
            hit_color[prev] = hit_color[prev - 1u];
            k = k - 1u;
        }

        let c = eval_diffuse_view_color(i, basis);
        hit_t[insert_idx]     = t;
        hit_alpha[insert_idx] = alpha;
        hit_color[insert_idx] = c;
    }

    var rgb = vec3<f32>(0.0);
    var transmittance: f32 = 1.0;
    var front_t: f32 = NO_HIT_T;
    for (var k: u32 = 0u; k < num_hits; k = k + 1u) {
        let a = hit_alpha[k];
        rgb = rgb + hit_color[k] * a * transmittance;
        if (front_t < 0.0) {
            front_t = hit_t[k];
        }
        transmittance = transmittance * (1.0 - a);
        if (transmittance < MIN_TRANSMITTANCE) {
            break;
        }
    }

    out_rgba[tid] = vec4<f32>(rgb, 1.0 - transmittance);
    out_depth[tid] = front_t;
}
