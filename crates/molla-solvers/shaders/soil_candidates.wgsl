struct Query { position_mass:vec4<f32>, velocity_dt:vec4<f32>, body_boundary:vec4<u32> }
struct Status { fault:atomic<u32>, iterations:u32, backtracks:u32, events:u32 }
struct CandidateCount { value:u32, claims:atomic<u32>, failure:array<u32,15> }
struct Params { origin_h:vec4<f32>, size:u32, total:u32, bodies:u32, capacity:u32, dt:f32, margin:f32, width:u32, groups:u32, grid_window_cells:vec4<i32> }
@group(0) @binding(0) var<storage,read_write> grid:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> centers:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> shapes:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> jacobians:array<f32>;
@group(0) @binding(4) var<storage,read> rates:array<f32>;
@group(0) @binding(5) var<storage,read> velocities:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> queries:array<Query>;
@group(0) @binding(7) var<storage,read_write> node_ids:array<u32>;
@group(0) @binding(8) var<storage,read_write> mapping:array<u32>;
@group(0) @binding(9) var<storage,read_write> offsets:array<u32>;
@group(0) @binding(10) var<storage,read_write> count:CandidateCount;
@group(0) @binding(11) var<storage,read_write> status:Status;
@group(0) @binding(12) var<uniform> params:Params;
var<workgroup> prefix:array<u32,256>;
// ANALYTIC_GEOMETRY
fn world_id(index:u32)->u32 { return index; }
fn physical_id(index:u32)->u32 { return index; }
fn cell(index:u32)->vec3<u32> { let world=world_id(index);return vec3<u32>(world%params.size,(world/params.size)%params.size,world/(params.size*params.size)); }
fn point(index:u32)->vec3<f32> { return bitcast<vec3<f32>>(vec3<u32>(mapping[5u*index+2u],mapping[5u*index+3u],mapping[5u*index+4u])); }
fn boundary(index:u32)->u32 {
    let xyz=cell(index); var mask=0u;
    for (var axis=0u;axis<3u;axis++) {
        if xyz[axis]<2u { mask|=1u<<(2u*axis); }
        if xyz[axis]>=params.size-2u { mask|=1u<<(2u*axis+1u); }
    }
    return mask;
}
fn feasible_wall(v:vec3<f32>,mask:u32)->bool {
    for (var axis=0u;axis<3u;axis++) {
        let flags=(mask>>(2u*axis))&3u;
        if (flags==1u && v[axis]<0.0) || (flags==2u && v[axis]>0.0) { return false; }
    }
    return true;
}
fn world_surface(body:u32,p:vec3<f32>)->vec4<f32> {
    let rotation=shapes[4u*body+2u]; let q=rotation/sqrt(dot(rotation,rotation));
    let local=rotate(vec4<f32>(-q.xyz,q.w),p-centers[body].xyz);
    let value=surface(local,shapes[4u*body+3u]);
    return vec4<f32>(rotate(q,value.xyz),value.w);
}
fn coverage_failure(index:u32,body:u32,reason:u32,values:vec4<f32>) {
    if atomicAdd(&count.claims,1u)!=0u { return; }
    count.failure[0]=reason; count.failure[1]=world_id(index); count.failure[2]=body;
    count.failure[3]=mapping[5u*index]; count.failure[4]=mapping[5u*index+1u];
    count.failure[5]=boundary(index); count.failure[6]=bitcast<u32>(params.dt);
    let location=vec4<f32>(point(index),grid[physical_id(index)].w);
    for (var axis=0u;axis<4u;axis++) {
        count.failure[7u+axis]=bitcast<u32>(values[axis]);
        count.failure[11u+axis]=bitcast<u32>(location[axis]);
    }
}
@compute @workgroup_size(256)
fn reset(@builtin(global_invocation_id) id:vec3<u32>) {
    if atomicLoad(&status.fault)!=0u { return; }
    if id.x<params.capacity {
        queries[id.x]=Query(vec4<f32>(0.0),vec4<f32>(0.0),vec4<u32>(0u));
        node_ids[id.x]=0xffffffffu;
    }
    if id.x==0u { count.value=0u; atomicStore(&count.claims,0u); }
    if id.x<params.total && world_id(id.x)!=0xffffffffu {
        let p=bitcast<vec3<u32>>(params.origin_h.xyz+vec3<f32>(vec3<i32>(cell(id.x))+params.grid_window_cells.xyz)*params.origin_h.w);
        mapping[5u*id.x+2u]=p.x; mapping[5u*id.x+3u]=p.y; mapping[5u*id.x+4u]=p.z;
    }
}
@compute @workgroup_size(64)
fn validate(@builtin(global_invocation_id) id:vec3<u32>) {
    let b=id.x; if b>=params.bodies || atomicLoad(&status.fault)!=0u { return; }
    let s=shapes[4u*b+3u]; let q=shapes[4u*b+2u]; let norm=dot(q,q);
    if !all(abs(centers[b])<=vec4<f32>(3.402823e38)) || !all(abs(s)<=vec4<f32>(3.402823e38))
        || any(s.xyz<=vec3<f32>(0.0)) || !(s.w==1.0 || s.w==2.0 || s.w==3.0)
        || !all(abs(q)<=vec4<f32>(3.402823e38)) || !finite(norm) || abs(norm-1.0)>1e-4 {
        atomicOr(&status.fault,256u); return;
    }
    for (var i=0u;i<6u*params.width;i++) { if !finite(jacobians[b*6u*params.width+i]) { atomicOr(&status.fault,256u); return; } }
}
@compute @workgroup_size(256)
fn classify(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>) {
    let index=id.x; var owner=0xffffffffu;
    if index<params.total && world_id(index)!=0xffffffffu && atomicLoad(&status.fault)==0u {
        let value=grid[physical_id(index)];
        if !all(abs(value)<=vec4<f32>(3.402823e38)) || value.w<0.0 { atomicOr(&status.fault,256u); }
        else if value.w>0.0 {
            if !feasible_wall(value.xyz,boundary(index)) { atomicOr(&status.fault,256u); }
            for (var b=0u;b<params.bodies;b++) {
                let sample=world_surface(b,point(index));
                if !all(abs(sample)<=vec4<f32>(3.402823e38)) { atomicOr(&status.fault,4u); break; }
                if sample.w>params.margin { continue; }
                if owner!=0xffffffffu { atomicOr(&status.fault,1024u); break; }
                owner=b;
            }
        }
    }
    let selected=select(0u,1u,owner!=0xffffffffu);
    prefix[lane]=selected; workgroupBarrier();
    for (var stride=1u;stride<256u;stride*=2u) {
        var previous=0u; if lane>=stride { previous=prefix[lane-stride]; }
        workgroupBarrier(); prefix[lane]+=previous; workgroupBarrier();
    }
    if index<params.total { mapping[5u*index]=owner; mapping[5u*index+1u]=prefix[lane]-selected; }
    if lane==255u { offsets[group.x]=prefix[255]; }
}
@compute @workgroup_size(1)
fn scan_blocks() {
    if atomicLoad(&status.fault)!=0u { return; }
    var total=0u;
    for (var i=0u;i<params.groups;i++) { let value=offsets[i]; offsets[i]=total; total+=value; }
    count.value=total;
    if total>params.capacity { atomicOr(&status.fault,512u); }
}
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) id:vec3<u32>) {
    let index=id.x;
    if index>=params.total || world_id(index)==0xffffffffu || atomicLoad(&status.fault)!=0u { return; }
    let info=vec2<u32>(mapping[5u*index],mapping[5u*index+1u]); if info.x==0xffffffffu { return; }
    let slot=offsets[index/256u]+info.y;
    if slot>=params.capacity { atomicOr(&status.fault,512u); return; }
    let value=grid[physical_id(index)];
    queries[slot]=Query(vec4<f32>(point(index),value.w),vec4<f32>(value.xyz,params.dt),vec4<u32>(info.x,boundary(index),0u,0u));
    node_ids[slot]=world_id(index); mapping[5u*index+1u]=slot;
}
@compute @workgroup_size(256)
fn certify(@builtin(global_invocation_id) id:vec3<u32>) {
    let index=id.x;
    if index>=params.total || atomicLoad(&status.fault)!=0u { return; }
    if index==0u { for (var i=0u;i<params.width;i++) { if !finite(rates[i]) { atomicOr(&status.fault,4u); return; } } }
    if world_id(index)==0xffffffffu || grid[physical_id(index)].w==0.0 { return; }
    var velocity=grid[physical_id(index)].xyz; let info=vec2<u32>(mapping[5u*index],mapping[5u*index+1u]);
    if info.x!=0xffffffffu { velocity=velocities[info.y].xyz; }
    if !all(abs(velocity)<=vec3<f32>(3.402823e38)) || !feasible_wall(velocity,boundary(index)) {
        coverage_failure(index,0xffffffffu,select(2u,1u,!all(abs(velocity)<=vec3<f32>(3.402823e38))),vec4<f32>(velocity,0.0));
        atomicOr(&status.fault,2048u); return;
    }
    let p=point(index);
    for (var b=0u;b<params.bodies;b++) {
        let sample=world_surface(b,p); let normal=sample.xyz; let moment=cross(p-centers[b].xyz,normal);
        var machine=0.0;
        for (var i=0u;i<params.width;i++) {
            var row=0.0;
            for (var axis=0u;axis<3u;axis++) { row+=normal[axis]*jacobians[(6u*b+axis)*params.width+i]+moment[axis]*jacobians[(6u*b+axis+3u)*params.width+i]; }
            machine+=row*rates[i];
        }
        let soil=dot(normal,velocity); let gap=max(0.0,sample.w)/params.dt;
        let slack=(soil-machine)+gap;
        if !finite(slack) || !finite(machine) || !finite(soil) || !finite(gap) { atomicOr(&status.fault,4u); return; }
        if slack < -1e-6*max(1.0,max(abs(soil),max(abs(machine),gap))) {
            coverage_failure(index,b,3u,vec4<f32>(soil,machine,gap,slack));
            atomicOr(&status.fault,2048u); return;
        }
    }
}
@compute @workgroup_size(256)
fn publish_grid(@builtin(global_invocation_id) id:vec3<u32>) {
    let index=id.x;
    if index>=params.total || world_id(index)==0xffffffffu || atomicLoad(&status.fault)!=0u { return; }
    let info=vec2<u32>(mapping[5u*index],mapping[5u*index+1u]);
    if info.x!=0xffffffffu { grid[physical_id(index)]=vec4<f32>(velocities[info.y].xyz,grid[physical_id(index)].w); }
}
