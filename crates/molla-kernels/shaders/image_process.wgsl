struct Params {
    start: u32,
    count: u32,
    operation: u32,
    slot: u32,
    a: f32,
    b: f32,
    partial_count: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> partials: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> gains: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> values: array<vec3<f32>, 256>;

fn unpack(p: u32) -> vec3<f32> {
    return vec3<f32>(f32(p & 255u), f32((p >> 8u) & 255u), f32((p >> 16u) & 255u)) / 255.0;
}

fn quantize(v: f32) -> u32 {
    let scaled = clamp(v, 0.0, 1.0) * 255.0;
    let base = floor(scaled);
    return u32(base) + select(0u, 1u, scaled - base >= 0.5);
}

fn encode_srgb(v: f32) -> f32 {
    if v <= 0.0031308 { return 12.92 * v; }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

fn decode_srgb(v: f32) -> f32 {
    if v <= 0.04045 { return v / 12.92; }
    return pow((v + 0.055) / 1.055, 2.4);
}

fn gamma(v: f32) -> f32 {
    if v == 0.0 || v == 1.0 { return v; }
    return pow(v, params.a);
}

@compute @workgroup_size(256)
fn apply(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.count { return; }
    let index = params.start + id.x;
    let packed = pixels[index];
    let c = unpack(packed);
    var out = c;
    switch params.operation {
        case 0u: { out = c * params.a; }
        case 1u: { out = vec3<f32>(gamma(c.x), gamma(c.y), gamma(c.z)); }
        case 2u: { out = vec3<f32>(encode_srgb(c.x), encode_srgb(c.y), encode_srgb(c.z)); }
        case 3u: { out = vec3<f32>(decode_srgb(c.x), decode_srgb(c.y), decode_srgb(c.z)); }
        case 4u: { out = c / (vec3<f32>(1.0) + c); }
        case 5u: { out = c * (vec3<f32>(1.0) + c * params.a) / (vec3<f32>(1.0) + c); }
        case 6u: { out = (c - vec3<f32>(0.5)) * params.b + vec3<f32>(0.5) + vec3<f32>(params.a); }
        case 7u: {
            let l = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
            out = vec3<f32>(l) + (c - vec3<f32>(l)) * params.a;
        }
        case 8u, 9u: { out = c * gains[params.slot].xyz; }
        default: {}
    }
    pixels[index] = (packed & 0xff000000u) | quantize(out.x) | (quantize(out.y) << 8u) | (quantize(out.z) << 16u);
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) local: u32,
          @builtin(workgroup_id) group: vec3<u32>) {
    var value = vec3<f32>(0.0);
    if id.x < params.count { value = unpack(pixels[params.start + id.x]); }
    values[local] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if local < stride {
            if params.operation == 9u { values[local] = max(values[local], values[local + stride]); }
            else { values[local] += values[local + stride]; }
        }
        workgroupBarrier();
    }
    if local == 0u { partials[group.x] = vec4<f32>(values[0], 0.0); }
}

@compute @workgroup_size(256)
fn finish_reduction(@builtin(local_invocation_index) local: u32) {
    var value = vec3<f32>(0.0);
    for (var i = local; i < params.partial_count; i += 256u) {
        if params.operation == 9u { value = max(value, partials[i].xyz); }
        else { value += partials[i].xyz; }
    }
    values[local] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if local < stride {
            if params.operation == 9u { values[local] = max(values[local], values[local + stride]); }
            else { values[local] += values[local + stride]; }
        }
        workgroupBarrier();
    }
    if local == 0u {
        var scale = vec3<f32>(1.0);
        if params.operation == 9u {
            scale = min(vec3<f32>(1.0) / max(values[0], vec3<f32>(1e-9)), vec3<f32>(1e6));
        } else if params.count != 0u {
            let mean = values[0] / f32(params.count);
            let gray = (mean.x + mean.y + mean.z) / 3.0;
            scale = vec3<f32>(gray) / max(mean, vec3<f32>(1e-9));
        }
        gains[params.slot] = vec4<f32>(scale, 0.0);
    }
}
