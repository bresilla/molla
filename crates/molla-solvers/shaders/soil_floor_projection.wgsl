struct Node { normal_mass:vec4<f32>, velocity_gap:vec4<f32>, boundary:vec4<u32> }
struct Params { width:u32, nodes:u32, pad0:u32, pad1:u32 }
struct Status { fault:atomic<u32>, iterations:u32, backtracks:u32, pad:u32 }
@group(0) @binding(0) var<storage,read> nodes:array<Node>;
@group(0) @binding(1) var<storage,read> rows:array<f32>;
@group(0) @binding(2) var<storage,read_write> terms:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> term_rows:array<f32>;
@group(0) @binding(4) var<storage,read> term_impulses:array<f32>;
@group(0) @binding(5) var<storage,read_write> velocities:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> contact_impulses:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> floor_impulses:array<vec4<f32>>;
@group(0) @binding(8) var<storage,read_write> status:Status;
@group(0) @binding(9) var<uniform> params:Params;
fn finite(x:f32)->bool { return abs(x)<=3.402823e38; }

@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;
    if c>=params.nodes || atomicLoad(&status.fault)!=0u { return; }
    let node=nodes[c]; let normal=node.normal_mass.xyz; let mass=node.normal_mass.w;
    let velocity=node.velocity_gap.xyz; let mask=node.boundary.x;
    for (var term=0u;term<4u;term++) {
        terms[4u*c+term]=vec4<f32>(0.0);
        for (var i=0u;i<params.width;i++) {
            let value=rows[c*params.width+i];
            if !finite(value) { atomicOr(&status.fault,32u); return; }
            term_rows[(4u*c+term)*params.width+i]=value;
        }
    }
    if !all(abs(node.normal_mass)<=vec4<f32>(3.402823e38)) || !all(abs(node.velocity_gap)<=vec4<f32>(3.402823e38))
        || mass<0.0 || node.velocity_gap.w<0.0 || mask>=64u || any(node.boundary.yzw!=vec3<u32>(0u)) {
        atomicOr(&status.fault,32u); return;
    }
    for (var axis=0u;axis<3u;axis++) {
        if ((mask>>(2u*axis))&3u)==3u { atomicOr(&status.fault,32u); return; }
    }
    if mass==0.0 { return; }
    var beta=dot(normal,normal);
    if abs(beta-1.0)>1e-5 { atomicOr(&status.fault,32u); return; }
    var events:array<vec2<f32>,3>; var count=0u;
    for (var axis=0u;axis<3u;axis++) {
        let flags=(mask>>(2u*axis))&3u;
        if flags==0u { continue; }
        let sign=select(1.0,-1.0,flags==2u);
        if sign*velocity[axis]<0.0 { atomicOr(&status.fault,32u); return; }
        if sign*normal[axis]<0.0 {
            let tau=-velocity[axis]/normal[axis];
            if !finite(tau) { atomicOr(&status.fault,4u); return; }
            events[count]=vec2<f32>(tau,f32(axis)); count+=1u;
        }
    }
    for (var i=0u;i<count;i++) {
        for (var j=i+1u;j<count;j++) {
            if events[j].x<events[i].x { let temporary=events[i]; events[i]=events[j]; events[j]=temporary; }
        }
    }
    let free_normal=dot(normal,velocity);
    let weight=mass/beta;
    if !finite(free_normal) || !finite(weight) { atomicOr(&status.fault,4u); return; }
    terms[4u*c]=vec4<f32>(weight,free_normal,node.velocity_gap.w,0.0);
    var tau=0.0; var threshold=0.0; var blocked=0u;
    for (var event=0u;event<count;event++) {
        threshold+=beta*(events[event].x-tau); tau=events[event].x;
        let axis=u32(events[event].y); blocked|=1u<<axis;
        var next_beta=0.0;
        for (var k=0u;k<3u;k++) { if (blocked&(1u<<k))==0u { next_beta+=normal[k]*normal[k]; } }
        if next_beta<=0.0 {
            terms[4u*c+event+1u]=vec4<f32>(0.0,0.0,node.velocity_gap.w,1.0);
            return;
        }
        let increment=mass*normal[axis]*normal[axis]/(beta*next_beta);
        let knot=free_normal+threshold;
        if !finite(increment) || !finite(knot) { atomicOr(&status.fault,4u); return; }
        terms[4u*c+event+1u]=vec4<f32>(increment,knot,node.velocity_gap.w,0.0);
        beta=next_beta;
    }
}

@compute @workgroup_size(64)
fn recover(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;
    if c>=params.nodes || atomicLoad(&status.fault)!=0u { return; }
    let node=nodes[c]; let mass=node.normal_mass.w; let normal=node.normal_mass.xyz;
    var impulse=0.0;
    for (var term=0u;term<4u;term++) { impulse+=term_impulses[4u*c+term]; }
    var raw=node.velocity_gap.xyz;
    if mass>0.0 { raw+=(impulse/mass)*normal; }
    if !finite(impulse) || !all(abs(raw)<=vec3<f32>(3.402823e38)) { atomicOr(&status.fault,4u); return; }
    var velocity=raw;
    if mass>0.0 {
        for (var axis=0u;axis<3u;axis++) {
            let flags=(node.boundary.x>>(2u*axis))&3u;
            if flags==1u { velocity[axis]=max(velocity[axis],0.0); }
            if flags==2u { velocity[axis]=min(velocity[axis],0.0); }
        }
    }
    let contact=impulse*normal; let boundary=mass*(velocity-raw);
    if !all(abs(contact)<=vec3<f32>(3.402823e38)) || !all(abs(boundary)<=vec3<f32>(3.402823e38)) {
        atomicOr(&status.fault,4u); return;
    }
    velocities[c]=vec4<f32>(velocity,0.0);
    contact_impulses[c]=vec4<f32>(contact,impulse);
    floor_impulses[c]=vec4<f32>(boundary,0.0);
}
