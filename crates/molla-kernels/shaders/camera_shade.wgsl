struct Parameters {
    width: u32, height: u32, camera_count: u32, shape_count: u32,
    max_distance: f32, channels: u32, _pad0: f32, _pad1: f32,
    ambient: vec4<f32>, light_dir: vec4<f32>, light_color: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> distance: array<f32>;
@group(0) @binding(1) var<storage, read> shape: array<i32>;
@group(0) @binding(2) var<storage, read> normal: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> shape_color: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> out_depth: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_color: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_normal: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out_albedo: array<u32>;
@group(0) @binding(8) var<storage, read_write> out_shape: array<u32>;
@group(0) @binding(9) var<uniform> params: Parameters;

fn pack_rgba8(c: vec3<f32>) -> u32 {
    let r = u32(clamp(c.r, 0.0, 1.0) * 255.0 + 0.5);
    let g = u32(clamp(c.g, 0.0, 1.0) * 255.0 + 0.5);
    let b = u32(clamp(c.b, 0.0, 1.0) * 255.0 + 0.5);
    return (255u << 24u) | (b << 16u) | (g << 8u) | r;
}

@compute @workgroup_size(8, 8, 1)
fn shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height || gid.z >= params.camera_count { return; }
    let i = (gid.z * params.height + gid.y) * params.width + gid.x;
    let id = shape[i];
    let t = distance[i];
    let mask = params.channels;
    if id < 0 || u32(id) >= params.shape_count || !(t > 0.0 && t < params.max_distance) {
        if (mask & 1u) != 0u { out_color[i] = 0u; }
        if (mask & 2u) != 0u { out_depth[i] = select(-1.0, -2.0, id == -2); }
        if (mask & 4u) != 0u { out_normal[i] = vec4<f32>(0.0); }
        if (mask & 8u) != 0u { out_albedo[i] = 0u; }
        if (mask & 16u) != 0u { out_shape[i] = select(0xffffffffu, 0xfffffffeu, id == -2); }
        return;
    }
    let n = normal[i].xyz;
    let albedo = shape_color[u32(id)].xyz;
    if (mask & 2u) != 0u { out_depth[i] = t; }
    if (mask & 4u) != 0u { out_normal[i] = vec4<f32>(n, 0.0); }
    if (mask & 8u) != 0u { out_albedo[i] = pack_rgba8(albedo); }
    if (mask & 16u) != 0u { out_shape[i] = u32(id); }
    if (mask & 1u) != 0u {
        let lit = params.ambient.xyz + params.light_color.xyz * max(0.0, dot(n, -params.light_dir.xyz));
        out_color[i] = pack_rgba8(albedo * lit);
    }
}
