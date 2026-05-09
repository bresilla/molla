// Pinhole-camera ray generation kernel — one thread per pixel of
// every camera in the batch. Writes per-pixel `ray_origin` /
// `ray_direction` arrays sized `(camera_count × height × width)`,
// row-major in pixel order, camera-major outermost.
//
// Intrinsics: `fov_y` is the vertical full FOV in radians; `aspect`
// is `width / height`; `near` is unused by the trace pass but kept in
// the uniform for symmetry with classical pipelines (intersection is
// done in object space, not screen space).
//
// Convention: camera looks down its local -Z axis. +X = right, +Y =
// up. The pose's rotation transforms local camera space to world
// space; the pose's position is the pinhole.
//
// One thread per pixel; `(x, y)` in `[0, width) × [0, height)`. The
// outer dimension is camera index `c ∈ [0, camera_count)`. Output
// linear index = `c * (width * height) + y * width + x`.

struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,
};

struct Uniforms {
    width:        u32,
    height:       u32,
    camera_count: u32,
    fov_y:        f32,
    aspect:       f32,
    near:         f32,
    _pad0:        f32,
    _pad1:        f32,
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

    // Camera-local ray direction. Camera looks at -Z, +X right, +Y up.
    let dir_local = normalize(vec3<f32>(u * tan_half_h, v * tan_half_v, -1.0));
    let dir_world = quat_rotate(pose.rotation, dir_local);

    ray_origin[idx]    = vec4<f32>(pose.position.xyz, 0.0);
    ray_direction[idx] = vec4<f32>(dir_world, 0.0);
}
