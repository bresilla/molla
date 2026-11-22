// Sinless hash attribution and MIT notice: ../data/HASH_NOISE_LICENSE.md.
struct Camera {
    eye: vec4<f32>, right: vec4<f32>, up: vec4<f32>, forward: vec4<f32>, projection: vec4<f32>
}
@group(0) @binding(0) var<uniform> camera: Camera;
struct Vertex {
    @location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) seed: f32
}
struct Fragment {
    @builtin(position) position: vec4<f32>, @location(0) world: vec3<f32>,
    @location(1) uv: vec2<f32>, @location(2) seed: f32
}
@vertex fn vertex(v: Vertex) -> Fragment {
    let delta = v.position-camera.eye.xyz;
    let depth = -dot(delta,camera.forward.xyz);
    var out: Fragment;
    out.position = vec4<f32>(dot(delta,camera.right.xyz)*camera.projection.x,
        dot(delta,camera.up.xyz)*2.5,(100.0*depth-5.0)/99.95,depth);
    out.world = v.position; out.uv = v.uv; out.seed = v.seed;
    return out;
}
fn hash(p: vec3<f32>) -> f32 {
    let q = fract(p*vec3<f32>(0.1031,0.11369,0.13787));
    let r = q+dot(q,q.yzx+19.19);
    return fract((r.x+r.y)*r.z);
}
fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p); let f = fract(p); let u = f*f*(3.0-2.0*f);
    return mix(mix(mix(hash(i),hash(i+vec3<f32>(1,0,0)),u.x),
        mix(hash(i+vec3<f32>(0,1,0)),hash(i+vec3<f32>(1,1,0)),u.x),u.y),
        mix(mix(hash(i+vec3<f32>(0,0,1)),hash(i+vec3<f32>(1,0,1)),u.x),
        mix(hash(i+vec3<f32>(0,1,1)),hash(i+vec3<f32>(1,1,1)),u.x),u.y),u.z);
}
@fragment fn fragment(v: Fragment) -> @location(0) vec4<f32> {
    let t = camera.projection.y;
    let p = vec3<f32>(v.uv.x*1.7+v.seed,v.uv.y*6.0+t*0.6,v.seed*0.31);
    let density = 0.65*value_noise(p)+0.35*value_noise(p*2.13+vec3<f32>(3.1,-t*0.25,1.7));
    let edge = exp(-5.0*v.uv.x*v.uv.x)*(1.0-smoothstep(0.65,1.0,abs(v.uv.x)));
    let tips = smoothstep(0.0,0.2,v.uv.y)*(1.0-smoothstep(0.75,1.0,v.uv.y));
    let ground = smoothstep(0.04,0.35,v.world.y);
    let alpha = 0.030*edge*tips*ground*smoothstep(0.22,0.78,density);
    return vec4<f32>(vec3<f32>(0.62,0.71,0.75)*alpha,alpha);
}
