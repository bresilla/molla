// Stage 37.8 — separable 2D convolution with a 5-tap kernel.
//
// Two entry points: `conv_horiz` (input → temp, convolve along x) and
// `conv_vert` (temp → output, convolve along y). Boundary mode is
// clamp-to-edge ("replicate"), matching scipy's `mode='nearest'`.
//
// Input + output buffers are row-major `H × W` `f32`. The 5-tap weight
// vector is uploaded as a separate uniform.

const RADIUS: i32 = 2;

struct Uniforms {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
};

struct KernelUniform {
    coeffs: vec4<f32>,
    coeff4: f32,  // 5th tap (vec4 holds the first four).
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> in_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<f32>;
@group(1) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(1) var<uniform> k: KernelUniform;

fn tap(i: i32) -> f32 {
    if (i == 0) { return k.coeffs.x; }
    if (i == 1) { return k.coeffs.y; }
    if (i == 2) { return k.coeffs.z; }
    if (i == 3) { return k.coeffs.w; }
    return k.coeff4;
}

fn clamp_x(x: i32, w: i32) -> i32 {
    return clamp(x, 0, w - 1);
}
fn clamp_y(y: i32, h: i32) -> i32 {
    return clamp(y, 0, h - 1);
}

@compute @workgroup_size(16, 16, 1)
fn conv_horiz(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(u.width);
    let h = i32(u.height);
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) {
        return;
    }
    var acc: f32 = 0.0;
    for (var t: i32 = -RADIUS; t <= RADIUS; t = t + 1) {
        let xi = clamp_x(x + t, w);
        acc = acc + in_buf[u32(y * w + xi)] * tap(t + RADIUS);
    }
    out_buf[u32(y * w + x)] = acc;
}

@compute @workgroup_size(16, 16, 1)
fn conv_vert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(u.width);
    let h = i32(u.height);
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) {
        return;
    }
    var acc: f32 = 0.0;
    for (var t: i32 = -RADIUS; t <= RADIUS; t = t + 1) {
        let yi = clamp_y(y + t, h);
        acc = acc + in_buf[u32(yi * w + x)] * tap(t + RADIUS);
    }
    out_buf[u32(y * w + x)] = acc;
}
