struct Parameters {
    width: u32,
    height: u32,
    camera_count: u32,
    inv_samples: f32,
};

@group(0) @binding(0) var<storage, read_write> color: array<u32>;
@group(0) @binding(1) var<storage, read_write> sums: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Parameters;

@compute @workgroup_size(8, 8, 1)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height || gid.z >= params.camera_count { return; }
    let index = (gid.z * params.height + gid.y) * params.width + gid.x;
    let packed = color[index];
    sums[index] += vec4<f32>(f32(packed & 255u), f32((packed >> 8u) & 255u),
        f32((packed >> 16u) & 255u), f32(packed >> 24u));
}

@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height || gid.z >= params.camera_count { return; }
    let index = (gid.z * params.height + gid.y) * params.width + gid.x;
    let mean = clamp(sums[index] * params.inv_samples, vec4<f32>(0.0), vec4<f32>(255.0));
    let base = floor(mean);
    let rounded = vec4<u32>(base + select(vec4<f32>(0.0), vec4<f32>(1.0), mean - base >= vec4<f32>(0.5)));
    color[index] = rounded.x | (rounded.y << 8u) | (rounded.z << 16u) | (rounded.w << 24u);
}
