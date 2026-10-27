struct Query { position_mass:vec4<f32>, velocity_dt:vec4<f32>, body_boundary:vec4<u32> }
struct Sample { soil_weight:vec4<f32>, slip:vec4<f32>, metadata:vec4<u32> }
struct Params { width:u32, bodies:u32, contacts:u32, pad:u32 }
@group(0) @binding(0) var<storage,read> queries:array<Query>;
@group(0) @binding(1) var<storage,read> geometry:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> velocities:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> impulses:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> jacobians:array<f32>;
@group(0) @binding(5) var<storage,read> rates:array<f32>;
@group(0) @binding(6) var<storage,read> centers:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read> status:array<u32>;
@group(0) @binding(8) var<storage,read_write> samples:array<Sample>;
@group(0) @binding(9) var<uniform> params:Params;
var<workgroup> linear:vec3<f32>;
var<workgroup> angular:vec3<f32>;
var<workgroup> fault:atomic<u32>;
var<workgroup> supporting_nodes:atomic<u32>;
var<workgroup> soil_sum:array<vec4<f32>,64>;
var<workgroup> slip_sum:array<vec4<f32>,64>;
fn finite3(v:vec3<f32>)->bool { return all(abs(v)<=vec3<f32>(3.402823e38)); }
@compute @workgroup_size(64)
fn sample(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let b=group.x;
    if lane==0u {
        atomicStore(&fault,status[0]);
        atomicStore(&supporting_nodes,0u);
        linear=vec3<f32>(0.0); angular=vec3<f32>(0.0);
        for (var axis=0u;axis<3u;axis++) {
            for (var i=0u;i<params.width;i++) {
                linear[axis]+=jacobians[(6u*b+axis)*params.width+i]*rates[i];
                angular[axis]+=jacobians[(6u*b+axis+3u)*params.width+i]*rates[i];
            }
        }
        if !finite3(linear) || !finite3(angular) || !finite3(centers[b].xyz) { atomicOr(&fault,0x80000000u); }
    }
    workgroupBarrier();
    var soil=vec4<f32>(0.0); var slip=vec4<f32>(0.0); var count=0u;
    for (var c=lane;c<params.contacts;c+=64u) {
        let q=queries[c];
        if q.position_mass.w<=0.0 || q.body_boundary.x!=b { continue; }
        let weight=impulses[c].w;
        if !(abs(weight)<=3.402823e38) || weight<0.0 { atomicOr(&fault,0x80000000u); continue; }
        if weight==0.0 { continue; }
        let velocity=velocities[c].xyz; let normal=geometry[c].xyz;
        let point=linear+cross(angular,q.position_mass.xyz-centers[b].xyz);
        let relative=point-velocity;
        let tangent=relative-normal*dot(relative,normal);
        if !finite3(velocity) || !finite3(normal) || !finite3(tangent) || abs(dot(normal,normal)-1.0)>1e-4 {
            atomicOr(&fault,0x80000000u); continue;
        }
        soil+=vec4<f32>(weight*velocity,weight);
        slip+=vec4<f32>(weight*tangent,weight*length(tangent));
        count++;
    }
    if count>0u { atomicAdd(&supporting_nodes,count); }
    soil_sum[lane]=soil; slip_sum[lane]=slip;
    workgroupBarrier();
    for (var stride=32u;stride>0u;stride/=2u) {
        if lane<stride {
            soil_sum[lane]+=soil_sum[lane+stride];
            slip_sum[lane]+=slip_sum[lane+stride];
        }
        workgroupBarrier();
    }
    if lane==0u {
        let total=soil_sum[0]; let sliding=slip_sum[0];
        var mean=vec3<f32>(0.0); var motion=vec3<f32>(0.0); var speed=0.0;
        if total.w>0.0 { mean=total.xyz/total.w; motion=sliding.xyz/total.w; speed=sliding.w/total.w; }
        if !finite3(mean) || !finite3(motion) || !(total.w<=3.402823e38) || !(speed<=3.402823e38) { atomicOr(&fault,0x80000000u); }
        samples[b]=Sample(vec4<f32>(mean,total.w),vec4<f32>(motion,speed),
            vec4<u32>(atomicLoad(&fault),atomicLoad(&supporting_nodes),0u,0u));
    }
}
