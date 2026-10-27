// Multi-channel camera shading with shared primitive intersections.

const CH_COLOR:       u32 = 1u;
const CH_DEPTH:       u32 = 2u;
const CH_NORMAL:      u32 = 4u;
const CH_ALBEDO:      u32 = 8u;
const CH_SHAPE_INDEX: u32 = 16u;

struct Uniforms {
    width:        u32,
    height:       u32,
    camera_count: u32,
    shape_count:  u32,
    max_distance: f32,
    channels:     u32,
    _pad0:        f32,
    _pad1:        f32,
    ambient:      vec4<f32>,
    light_dir:    vec4<f32>,    // world-space, expected unit length
    light_color:  vec4<f32>,
};

@group(0) @binding(0) var<storage, read> shape_geom_kind:  array<u32>;
@group(0) @binding(1) var<storage, read> shape_geom_data:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> shape_body:       array<i32>;
@group(0) @binding(3) var<storage, read> shape_transform:  array<Transform>;
@group(0) @binding(4) var<storage, read> body_q:           array<Transform>;
@group(0) @binding(5) var<storage, read> shape_color:      array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> shape_metallic:   array<f32>;
@group(0) @binding(7) var<storage, read> shape_roughness:  array<f32>;

@group(1) @binding(0) var<storage, read> ray_origin:    array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> ray_direction: array<vec4<f32>>;

@group(2) @binding(0) var<storage, read_write> out_depth:       array<f32>;
@group(2) @binding(1) var<storage, read_write> out_color:       array<u32>;
@group(2) @binding(2) var<storage, read_write> out_normal:      array<vec4<f32>>;
@group(2) @binding(3) var<storage, read_write> out_albedo:      array<u32>;
@group(2) @binding(4) var<storage, read_write> out_shape_index: array<u32>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

fn pack_rgba8(c: vec3<f32>) -> u32 {
    let r = u32(clamp(c.r, 0.0, 1.0) * 255.0 + 0.5);
    let g = u32(clamp(c.g, 0.0, 1.0) * 255.0 + 0.5);
    let b = u32(clamp(c.b, 0.0, 1.0) * 255.0 + 0.5);
    return (255u << 24u) | (b << 16u) | (g << 8u) | r;
}

@compute @workgroup_size(8, 8, 1)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let cam = gid.z;
    if (px >= uniforms.width || py >= uniforms.height || cam >= uniforms.camera_count) {
        return;
    }

    let pixels_per_cam = uniforms.width * uniforms.height;
    let idx = cam * pixels_per_cam + py * uniforms.width + px;

    let ro = ray_origin[idx].xyz;
    let rd = ray_scaled_direction(ray_direction[idx].xyz);
    let rd_len = length(rd);
    let channels = uniforms.channels;

    if (rd_len < 1e-12) {
        if ((channels & CH_DEPTH) != 0u)       { out_depth[idx] = NO_HIT_T; }
        if ((channels & CH_SHAPE_INDEX) != 0u) { out_shape_index[idx] = 0xffffffffu; }
        if ((channels & CH_NORMAL) != 0u)      { out_normal[idx] = vec4<f32>(0.0); }
        if ((channels & CH_ALBEDO) != 0u)      { out_albedo[idx] = 0u; }
        if ((channels & CH_COLOR) != 0u)       { out_color[idx] = 0u; }
        return;
    }

    var best_t: f32 = uniforms.max_distance;
    var best_shape: i32 = -1;
    var best_point_world: vec3<f32> = vec3<f32>(0.0);
    var best_normal_world: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

    for (var i: u32 = 0u; i < uniforms.shape_count; i = i + 1u) {
        let kind = shape_geom_kind[i];
        if (kind == KIND_SKIP) { continue; }

        let body = shape_body[i];
        var world_xform: Transform;
        if (body >= 0) {
            world_xform = transform_compose(body_q[u32(body)], shape_transform[i]);
        } else {
            world_xform = shape_transform[i];
        }

        let data = shape_geom_data[i].xyz;
        let local_hit = ray_shape(kind, data, world_xform, ro, rd, uniforms.max_distance / rd_len);
        let t = local_hit.t * rd_len;

        if (t > 0.0 && t < best_t) {
            best_t = t;
            best_shape = i32(i);
            let p_local = local_hit.point;
            best_point_world  = transform_point(world_xform, p_local);
            let n_local = shape_normal(kind, data, p_local);
            best_normal_world = normalize(quat_rotate(world_xform.rotation, n_local));
        }
    }

    if (best_shape < 0) {
        if ((channels & CH_DEPTH) != 0u)       { out_depth[idx] = NO_HIT_T; }
        if ((channels & CH_SHAPE_INDEX) != 0u) { out_shape_index[idx] = 0xffffffffu; }
        if ((channels & CH_NORMAL) != 0u)      { out_normal[idx] = vec4<f32>(0.0); }
        if ((channels & CH_ALBEDO) != 0u)      { out_albedo[idx] = 0u; }
        if ((channels & CH_COLOR) != 0u)       { out_color[idx] = 0u; }
        return;
    }

    if ((channels & CH_DEPTH) != 0u)       { out_depth[idx] = best_t; }
    if ((channels & CH_SHAPE_INDEX) != 0u) { out_shape_index[idx] = u32(best_shape); }
    if ((channels & CH_NORMAL) != 0u) {
        out_normal[idx] = vec4<f32>(best_normal_world, 0.0);
    }

    let albedo = shape_color[u32(best_shape)].xyz;
    if ((channels & CH_ALBEDO) != 0u) {
        out_albedo[idx] = pack_rgba8(albedo);
    }

    if ((channels & CH_COLOR) != 0u) {
        // Lambertian shading: albedo * (ambient + light · max(0, n · -light_dir)).
        // Light shines "from light_dir" → surface gets light from -light_dir.
        let l = -uniforms.light_dir.xyz;
        let ndotl = max(0.0, dot(best_normal_world, l));
        let lit = uniforms.ambient.xyz + uniforms.light_color.xyz * ndotl;
        let shaded = albedo * lit;
        out_color[idx] = pack_rgba8(shaded);
    }
}
