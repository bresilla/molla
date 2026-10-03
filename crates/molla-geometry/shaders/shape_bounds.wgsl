struct Transform { position: vec4<f32>, rotation: vec4<f32> };
struct ShapeBounds { lo: vec4<f32>, hi: vec4<f32>, data: vec4<f32>, mode: vec4<u32> };
struct Aabb { lo: vec4<f32>, hi: vec4<f32> };
struct Params { count: u32, margin: f32, pad0: u32, pad1: u32 };

@group(0) @binding(0) var<storage, read> local_bounds: array<ShapeBounds>;
@group(0) @binding(1) var<storage, read> shape_body: array<i32>;
@group(0) @binding(2) var<storage, read> shape_pose: array<Transform>;
@group(0) @binding(3) var<storage, read> body_pose: array<Transform>;
@group(1) @binding(0) var<storage, read_write> bounds: array<Aabb>;
@group(1) @binding(1) var<storage, read_write> status: array<atomic<u32>>;
@group(1) @binding(2) var<uniform> params: Params;

fn finite4(v: vec4<f32>) -> bool {
    return all(v == v) && all(abs(v) <= vec4<f32>(3.4028234e38));
}
fn rotate(q: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, p);
    return p + q.w * t + cross(q.xyz, t);
}
fn multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
fn norm(v: vec3<f32>) -> f32 {
    let s = max(abs(v.x), max(abs(v.y), abs(v.z)));
    if (s == 0.0) { return 0.0; }
    return s * length(v / s);
}

@compute @workgroup_size(64)
fn shape_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    bounds[i] = Aabb(vec4<f32>(0.0), vec4<f32>(0.0));
    let body = shape_body[i];
    if (body < -1 || (body >= 0 && u32(body) >= arrayLength(&body_pose))) {
        atomicOr(&status[0], 2u);
        return;
    }
    var pose = shape_pose[i];
    var roundoff_scale = abs(pose.position.xyz);
    if (!finite4(pose.position) || !finite4(pose.rotation)) {
        atomicOr(&status[0], 16u);
        return;
    }
    if (body >= 0) {
        let parent = body_pose[u32(body)];
        if (!finite4(parent.position) || !finite4(parent.rotation)) {
            atomicOr(&status[0], 16u);
            return;
        }
        roundoff_scale = abs(parent.position.xyz) + vec3<f32>(norm(pose.position.xyz));
        pose.position = vec4<f32>(parent.position.xyz + rotate(parent.rotation, pose.position.xyz), 0.0);
        pose.rotation = multiply(parent.rotation, pose.rotation);
    }
    if (!finite4(pose.position) || !finite4(pose.rotation)) {
        atomicOr(&status[0], 16u);
        return;
    }
    let shape = local_bounds[i];
    if (shape.mode.x == 1u) {
        let inf = bitcast<f32>(0x7f800000u);
        bounds[i] = Aabb(vec4<f32>(vec3<f32>(-inf), 0.0), vec4<f32>(vec3<f32>(inf), 0.0));
        return;
    }
    let x = rotate(pose.rotation, vec3<f32>(1.0, 0.0, 0.0));
    let y = rotate(pose.rotation, vec3<f32>(0.0, 1.0, 0.0));
    let z = rotate(pose.rotation, vec3<f32>(0.0, 0.0, 1.0));
    var center = pose.position.xyz;
    var extent: vec3<f32>;
    if (shape.mode.x == 2u) {
        extent = vec3<f32>(shape.data.x);
    } else if (shape.mode.x == 3u) {
        extent = abs(y) * shape.data.x + vec3<f32>(shape.data.y);
    } else if (shape.mode.x == 4u) {
        extent = abs(y) * shape.data.x + sqrt(max(vec3<f32>(0.0), vec3<f32>(1.0) - y * y)) * shape.data.y
            + vec3<f32>(shape.data.z);
    } else if (shape.mode.x == 5u) {
        extent = vec3<f32>(norm(vec3<f32>(x.x, y.x, z.x) * shape.data.xyz),
            norm(vec3<f32>(x.y, y.y, z.y) * shape.data.xyz), norm(vec3<f32>(x.z, y.z, z.z) * shape.data.xyz));
    } else {
        let local_center = shape.hi.xyz * 0.5 + shape.lo.xyz * 0.5;
        let half = shape.hi.xyz * 0.5 - shape.lo.xyz * 0.5;
        center += rotate(pose.rotation, local_center);
        roundoff_scale += vec3<f32>(norm(local_center));
        extent = abs(x) * half.x + abs(y) * half.y + abs(z) * half.z;
    }
    let padding = vec3<f32>(params.margin) + 1.9073486e-6 * (roundoff_scale + abs(center) + extent);
    let lo = vec4<f32>(center - extent - padding, 0.0);
    let hi = vec4<f32>(center + extent + padding, 0.0);
    if (!finite4(lo) || !finite4(hi)) {
        atomicOr(&status[0], 16u);
        return;
    }
    bounds[i] = Aabb(lo, hi);
}
