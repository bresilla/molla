// Closest-hit primitive queries using shared rigid-shape intersections.

struct Uniforms {
    ray_count:    u32,
    shape_count:  u32,
    max_distance: f32,
    _pad:         f32,
};

@group(0) @binding(0) var<storage, read>  shape_geom_kind:  array<u32>;
@group(0) @binding(1) var<storage, read>  shape_geom_data:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>  shape_body:       array<i32>;
@group(0) @binding(3) var<storage, read>  shape_transform:  array<Transform>;
@group(0) @binding(4) var<storage, read>  body_q:           array<Transform>;

@group(1) @binding(0) var<storage, read>  ray_origin:    array<vec4<f32>>;
@group(1) @binding(1) var<storage, read>  ray_direction: array<vec4<f32>>;

@group(2) @binding(0) var<storage, read_write> hit_t:      array<f32>;
@group(2) @binding(1) var<storage, read_write> hit_shape:  array<i32>;
@group(2) @binding(2) var<storage, read_write> hit_point:  array<vec4<f32>>;
@group(2) @binding(3) var<storage, read_write> hit_normal: array<vec4<f32>>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn cast_rays(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.ray_count) { return; }

    let ro = ray_origin[tid].xyz;
    let rd_raw = ray_direction[tid].xyz;
    let rd_len = length(rd_raw);
    if (rd_len < 1e-12) {
        hit_t[tid] = NO_HIT_T;
        hit_shape[tid] = -1;
        hit_point[tid] = vec4<f32>(0.0);
        hit_normal[tid] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
        return;
    }
    let rd = rd_raw / rd_len;

    var best_t: f32 = uniforms.max_distance;
    var best_shape: i32 = -1;
    var best_point: vec3<f32> = vec3<f32>(0.0);
    var best_normal: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

    for (var i: u32 = 0u; i < uniforms.shape_count; i = i + 1u) {
        let kind = shape_geom_kind[i];
        if (kind == KIND_SKIP) { continue; }

        // World transform = body × shape-local.
        let body = shape_body[i];
        var world_xform: Transform;
        if (body >= 0) {
            world_xform = transform_compose(body_q[u32(body)], shape_transform[i]);
        } else {
            world_xform = shape_transform[i];
        }

        // Ray in shape-local frame.
        let inv = transform_inverse(world_xform);
        let lo = transform_point(inv, ro);
        let ld = quat_rotate(inv.rotation, rd);

        let data = shape_geom_data[i].xyz;
        var t: f32 = NO_HIT_T;
        if      (kind == KIND_SPHERE)    { t = ray_sphere_local(lo, ld, data.x); }
        else if (kind == KIND_PLANE)     { t = ray_plane_local(lo, ld, data); }
        else if (kind == KIND_BOX)       { t = ray_box_local(lo, ld, data); }
        else if (kind == KIND_CAPSULE)   { t = ray_capsule_local(lo, ld, data.x, data.y); }
        else if (kind == KIND_CYLINDER)  { t = ray_cylinder_local(lo, ld, data.x, data.y); }
        else if (kind == KIND_CONE)      { t = ray_cone_local(lo, ld, data.x, data.y); }
        else if (kind == KIND_ELLIPSOID) { t = ray_ellipsoid_local(lo, ld, data); }

        if (t > 0.0 && t < best_t) {
            best_t = t;
            best_shape = i32(i);
            let p_local = lo + ld * t;
            best_point  = transform_point(world_xform, p_local);
            let n_local = shape_normal(kind, data, p_local);
            best_normal = normalize(quat_rotate(world_xform.rotation, n_local));
        }
    }

    if (best_shape < 0) {
        hit_t[tid] = NO_HIT_T;
        hit_shape[tid] = -1;
        hit_point[tid] = vec4<f32>(0.0);
        hit_normal[tid] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    } else {
        hit_t[tid] = best_t;
        hit_shape[tid] = best_shape;
        hit_point[tid] = vec4<f32>(best_point, 0.0);
        hit_normal[tid] = vec4<f32>(best_normal, 0.0);
    }
}
