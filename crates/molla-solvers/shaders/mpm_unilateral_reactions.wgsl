struct Query { position_mass:vec4<f32>, velocity_dt:vec4<f32>, body_boundary:vec4<u32> }
struct Params { origin_h:vec4<f32>, size:u32, total:u32, bodies:u32, capacity:u32, dt:f32, margin:f32, width:u32, groups:u32 }
@group(0) @binding(0) var<storage,read> centers:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> queries:array<Query>;
@group(0) @binding(2) var<storage,read> impulses:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> wrenches:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> status:atomic<u32>;
@group(0) @binding(5) var<storage,read_write> indirect:array<u32>;
@group(0) @binding(6) var<uniform> params:Params;
@group(0) @binding(7) var<uniform> particles:vec4<u32>;
var<workgroup> linear:array<vec4<f32>,64>;
var<workgroup> angular:array<vec4<f32>,64>;
@compute @workgroup_size(64)
fn reactions(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let body=group.x; var force=vec3<f32>(0.0); var torque=vec3<f32>(0.0);
    if atomicLoad(&status)==0u {
        for (var c=lane;c<params.capacity;c+=64u) {
            let query=queries[c];
            if query.position_mass.w==0.0 || query.body_boundary.x!=body { continue; }
            let impulse=-impulses[c].xyz;
            force+=impulse; torque+=cross(query.position_mass.xyz-centers[body].xyz,impulse);
        }
    }
    linear[lane]=vec4<f32>(force,0.0); angular[lane]=vec4<f32>(torque,0.0); workgroupBarrier();
    for (var stride=32u;stride>0u;stride/=2u) {
        if lane<stride { linear[lane]+=linear[lane+stride]; angular[lane]+=angular[lane+stride]; }
        workgroupBarrier();
    }
    if lane==0u && atomicLoad(&status)==0u {
        let f=linear[0]/params.dt; let t=angular[0]/params.dt;
        if !all(abs(f)<=vec4<f32>(3.402823e38)) || !all(abs(t)<=vec4<f32>(3.402823e38)) { atomicOr(&status,4u); return; }
        wrenches[2u*body]=f; wrenches[2u*body+1u]=t;
    }
}
@compute @workgroup_size(1)
fn finish() {
    indirect[0]=select(0u,particles.x,atomicLoad(&status)==0u); indirect[1]=1u; indirect[2]=1u;
}
