struct Query { position_mass:vec4<f32>, velocity_dt:vec4<f32>, body_boundary:vec4<u32> }
struct Node { normal_mass:vec4<f32>, velocity_gap:vec4<f32>, boundary:vec4<u32> }
struct Params { width:u32, bodies:u32, queries:u32, pad:u32 }
struct Status { fault:atomic<u32>, iterations:u32, backtracks:u32, events:u32 }
@group(0) @binding(0) var<storage,read> centers:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> shapes:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> jacobians:array<f32>;
@group(0) @binding(3) var<storage,read> queries:array<Query>;
@group(0) @binding(4) var<storage,read_write> nodes:array<Node>;
@group(0) @binding(5) var<storage,read_write> rows:array<f32>;
@group(0) @binding(6) var<storage,read_write> geometry:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> status:Status;
@group(0) @binding(8) var<uniform> params:Params;
// ANALYTIC_GEOMETRY
@compute @workgroup_size(64)
fn generate(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;
    if c>=params.queries || atomicLoad(&status.fault)!=0u { return; }
    let query=queries[c]; let mass=query.position_mass.w;
    nodes[c]=Node(vec4<f32>(0.0),vec4<f32>(0.0),vec4<u32>(0u)); geometry[c]=vec4<f32>(0.0);
    for (var i=0u;i<params.width;i++) { rows[c*params.width+i]=0.0; }
    if !all(abs(query.position_mass)<=vec4<f32>(3.402823e38)) || mass<0.0 {
        atomicOr(&status.fault,128u); return;
    }
    if mass==0.0 { return; }
    let b=query.body_boundary.x; let mask=query.body_boundary.y;
    if b>=params.bodies || mask>=64u || any(query.body_boundary.zw!=vec2<u32>(0u))
        || !all(abs(query.velocity_dt)<=vec4<f32>(3.402823e38)) || query.velocity_dt.w<=0.0 {
        atomicOr(&status.fault,128u); return;
    }
    for (var axis=0u;axis<3u;axis++) {
        let flags=(mask>>(2u*axis))&3u;
        if flags==3u || (flags==1u && query.velocity_dt[axis]<0.0) || (flags==2u && query.velocity_dt[axis]>0.0) {
            atomicOr(&status.fault,128u); return;
        }
    }
    let shape=shapes[4u*b+3u]; let rotation=shapes[4u*b+2u]; let norm=dot(rotation,rotation);
    if !all(abs(centers[b])<=vec4<f32>(3.402823e38)) || !all(abs(shape)<=vec4<f32>(3.402823e38))
        || any(shape.xyz<=vec3<f32>(0.0)) || !(shape.w==1.0 || shape.w==2.0 || shape.w==3.0)
        || !all(abs(rotation)<=vec4<f32>(3.402823e38)) || !finite(norm) || abs(norm-1.0)>1e-4 {
        atomicOr(&status.fault,128u); return;
    }
    let orientation=rotation/sqrt(norm);
    let arm=query.position_mass.xyz-centers[b].xyz;
    let local=rotate(vec4<f32>(-orientation.xyz,orientation.w),arm);
    let sample=surface(local,shape);
    let normal=rotate(orientation,sample.xyz);
    let gap=max(0.0,sample.w)/query.velocity_dt.w;
    let moment=cross(arm,normal);
    if !all(abs(sample)<=vec4<f32>(3.402823e38)) || !all(abs(normal)<=vec3<f32>(3.402823e38))
        || !all(abs(moment)<=vec3<f32>(3.402823e38)) || !finite(gap) || abs(dot(normal,normal)-1.0)>1e-5 {
        atomicOr(&status.fault,4u); return;
    }
    for (var i=0u;i<params.width;i++) {
        var value=0.0;
        for (var axis=0u;axis<3u;axis++) {
            let linear=jacobians[(6u*b+axis)*params.width+i];
            let angular=jacobians[(6u*b+axis+3u)*params.width+i];
            if !finite(linear) || !finite(angular) { atomicOr(&status.fault,128u); return; }
            value+=normal[axis]*linear+moment[axis]*angular;
        }
        if !finite(value) { atomicOr(&status.fault,4u); return; }
        rows[c*params.width+i]=value;
    }
    nodes[c]=Node(vec4<f32>(normal,mass),vec4<f32>(query.velocity_dt.xyz,gap),vec4<u32>(mask,0u,0u,0u));
    geometry[c]=vec4<f32>(normal,sample.w);
}
