struct Damping { linear: f32, angular: f32, articulation: u32, enabled: u32 }
struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Spatial { linear: vec4<f32>, angular: vec4<f32> }
struct Inertia { a: vec4<f32>, b: vec4<f32>, c: vec4<f32> }
@group(0) @binding(0) var<storage, read> damping: array<Damping>;
@group(0) @binding(1) var<storage, read> masses: array<f32>;
@group(0) @binding(2) var<storage, read> inertias: array<Inertia>;
@group(0) @binding(3) var<storage, read> poses: array<Transform>;
@group(0) @binding(4) var<storage, read> velocities: array<Spatial>;
@group(0) @binding(5) var<storage, read_write> forces: array<Spatial>;
@group(0) @binding(6) var<storage, read> frame_forces: array<Spatial>;
@group(0) @binding(7) var<storage, read> active_articulations: array<u32>;
@group(0) @binding(8) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(9) var<uniform> dimensions: vec4<u32>;
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn finite(v: vec4<f32>) -> bool { return all(abs(v)<=vec4<f32>(3.402823e38)); }
@compute @workgroup_size(64)
fn apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let body=id.x;
    if body>=dimensions.x || atomicLoad(&status)!=0u { return; }
    var force=frame_forces[body];
    if dimensions.y!=0u { force=forces[body]; }
    let info=damping[body];
    if info.enabled!=0u && active_articulations[info.articulation]!=0u {
        let velocity=velocities[body];
        let rotation=poses[body].rotation;
        let inertia=inertias[body];
        let local=rotate(vec4<f32>(-rotation.xyz,rotation.w),velocity.angular.xyz);
        let momentum=inertia.a.xyz*local.x + inertia.b.xyz*local.y + inertia.c.xyz*local.z;
        force.linear=vec4<f32>(force.linear.xyz-velocity.linear.xyz*(masses[body]*info.linear),0.0);
        force.angular=vec4<f32>(force.angular.xyz-rotate(rotation,momentum)*info.angular,0.0);
    }
    if !finite(force.linear) || !finite(force.angular) { atomicOr(&status,32u); return; }
    forces[body]=force;
}
