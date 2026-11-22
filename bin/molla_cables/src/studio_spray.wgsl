struct Camera { eye: vec4<f32>, right: vec4<f32>, up: vec4<f32>, forward: vec4<f32>, projection: vec4<f32> }
@group(0) @binding(0) var<uniform> camera: Camera;
struct Vertex { @location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) opacity: f32 }
struct Fragment { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) opacity: f32 }
@vertex fn vertex(v: Vertex) -> Fragment {
    let delta = v.position-camera.eye.xyz;
    let depth = -dot(delta,camera.forward.xyz);
    var out: Fragment;
    out.position = vec4<f32>(dot(delta,camera.right.xyz)*camera.projection.x,
        dot(delta,camera.up.xyz)*2.5,(100.0*depth-5.0)/99.95,depth);
    out.uv = v.uv; out.opacity = v.opacity;
    return out;
}
@fragment fn fragment(v: Fragment) -> @location(0) vec4<f32> {
    let r2 = dot(v.uv,v.uv);
    let alpha = v.opacity * exp(-3.0*r2) * (1.0-smoothstep(0.6,1.0,r2));
    return vec4<f32>(vec3<f32>(1.0)*alpha,alpha);
}
