// H10 — Physical-camera ray generation: pinhole by default, optional
// thin-lens depth-of-field when `lens_radius > 0`. One thread per
// pixel × camera, writing per-pixel `ray_origin` / `ray_direction`
// arrays sized `(camera_count × height × width)`, row-major in
// pixel order, camera-major outermost.
//
// Optical model:
//   * `fov_y`           — vertical full FOV (rad). When the host
//                          fills it from `focal_length / sensor_width`
//                          the rendered FOV matches the photographic
//                          intrinsics.
//   * `lens_radius`     — aperture half-diameter (m). `0` (default)
//                          = pinhole; `> 0` enables DOF.
//   * `focal_distance`  — distance along the optical axis to the
//                          focal plane (m).
//   * `aperture_offset` — `(u, v)` in `[-1, 1]²` (caller-supplied
//                          disc sample, e.g. concentric or Halton),
//                          unused when `lens_radius == 0`.
//
// Convention: camera looks down its local -Z axis. +X = right, +Y =
// up. The pose's rotation transforms local camera space to world;
// the pose's position is the pinhole.
//
// Output linear index = `c * (width * height) + y * width + x`.

struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,
};

struct Uniforms {
    width:           u32,
    height:          u32,
    camera_count:    u32,
    fov_y:           f32,

    aspect:          f32,
    near:            f32,
    lens_radius:     f32,
    focal_distance:  f32,

    aperture_offset: vec2<f32>,
    _pad0:           f32,
    _pad1:           f32,
};

@group(0) @binding(0) var<storage, read>       camera_pose:   array<Transform>;
@group(0) @binding(1) var<storage, read_write> ray_origin:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> ray_direction: array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             uniforms:      Uniforms;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

@compute @workgroup_size(8, 8, 1)
fn generate_rays(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let cam = gid.z;
    if (px >= uniforms.width || py >= uniforms.height || cam >= uniforms.camera_count) {
        return;
    }

    let pose = camera_pose[cam];
    let pixels_per_cam = uniforms.width * uniforms.height;
    let idx = cam * pixels_per_cam + py * uniforms.width + px;

    // Map pixel to NDC in [-1, 1]. Centre of pixel = (px+0.5)/W,
    // y-flip so row 0 is the TOP of the image (positive +Y in
    // camera frame).
    let u = (2.0 * (f32(px) + 0.5) / f32(uniforms.width)) - 1.0;
    let v = 1.0 - (2.0 * (f32(py) + 0.5) / f32(uniforms.height));

    let tan_half_v = tan(0.5 * uniforms.fov_y);
    let tan_half_h = tan_half_v * uniforms.aspect;

    // Camera-local pinhole ray direction (camera looks at -Z).
    let dir_local = normalize(vec3<f32>(u * tan_half_h, v * tan_half_v, -1.0));

    if (uniforms.lens_radius > 0.0 && uniforms.focal_distance > 0.0) {
        // Thin-lens depth-of-field: focal point along the pinhole ray
        // at `focal_distance` (measured along the optical axis). The
        // jittered origin is `lens_radius · aperture_offset` on the
        // lens plane (camera-local +X / +Y); the ray then aims at
        // the same focal point so objects at the focal plane stay
        // sharp while everything else blurs.
        let t_focal = uniforms.focal_distance / max(-dir_local.z, 1.0e-6);
        let focal_point_local = dir_local * t_focal;
        let lens_offset_local = vec3<f32>(
            uniforms.aperture_offset.x * uniforms.lens_radius,
            uniforms.aperture_offset.y * uniforms.lens_radius,
            0.0,
        );
        let jittered_dir_local = normalize(focal_point_local - lens_offset_local);
        let jittered_dir_world = quat_rotate(pose.rotation, jittered_dir_local);
        let lens_offset_world = quat_rotate(pose.rotation, lens_offset_local);
        ray_origin[idx]    = vec4<f32>(pose.position.xyz + lens_offset_world, 0.0);
        ray_direction[idx] = vec4<f32>(jittered_dir_world, 0.0);
    } else {
        // Pinhole — original path, bit-identical to pre-H10 output.
        let dir_world = quat_rotate(pose.rotation, dir_local);
        ray_origin[idx]    = vec4<f32>(pose.position.xyz, 0.0);
        ray_direction[idx] = vec4<f32>(dir_world, 0.0);
    }
}
