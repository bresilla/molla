struct Params {
    ray_count: u32,
    shape_id: i32,
    max_distance: f32,
    initialize: u32,
}

@group(0) @binding(0) var<storage, read> origins: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> directions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> mesh_t: array<f32>;
@group(0) @binding(3) var<storage, read> mesh_tri: array<i32>;
@group(0) @binding(4) var<storage, read> mesh_normal: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> hit_t: array<f32>;
@group(0) @binding(6) var<storage, read_write> hit_shape: array<i32>;
@group(0) @binding(7) var<storage, read_write> hit_point: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> hit_normal: array<vec4<f32>>;
@group(0) @binding(9) var<uniform> params: Params;

@compute @workgroup_size(64)
fn merge_hits(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.ray_count) { return; }
    if (params.initialize != 0u) {
        hit_t[i] = -1.0;
        hit_shape[i] = -1;
        hit_point[i] = vec4<f32>(0.0);
        hit_normal[i] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
        return;
    }
    let t = mesh_t[i];
    if (mesh_tri[i] < 0 || !(t > 0.0 && t < params.max_distance)) { return; }
    if (hit_shape[i] >= 0 && hit_t[i] > 0.0 && !(t < hit_t[i])) { return; }
    let direction = directions[i].xyz;
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    let geometric = mesh_normal[i].xyz;
    let length_squared = dot(geometric, geometric);
    if (length_squared > 1e-24) {
        normal = geometric * inverseSqrt(length_squared);
        if (dot(normal, direction) > 0.0) { normal = -normal; }
    }
    hit_t[i] = t;
    hit_shape[i] = params.shape_id;
    hit_point[i] = vec4<f32>(origins[i].xyz + direction * t, 0.0);
    hit_normal[i] = vec4<f32>(normal, 0.0);
}
