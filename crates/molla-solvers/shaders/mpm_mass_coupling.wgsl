struct Node { arm_mass: vec4<f32>, momentum: vec4<f32> }
struct Span { start: u32, count: atomic<u32> }
struct Params { origin_h: vec4<f32>, size: u32, bodies: u32, capacity: u32, total: u32,
    dt: f32, particles: u32, groups: u32, pad: u32, grid_window_cells: vec4<i32> }
@group(0) @binding(0) var<storage, read_write> grid: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> centers: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> shapes: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(4) var<storage, read_write> spans: array<Span>;
@group(0) @binding(5) var<storage, read_write> owners: array<u32>;
@group(0) @binding(6) var<storage, read> motion: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> impulses: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> wrenches: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(10) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(11) var<uniform> params: Params;
@group(0) @binding(12) var<storage, read_write> metadata: array<atomic<u32>>;
override ACTIVE_GRID: bool = false;
override SPARSE_GRID: bool = false;
override METADATA_WORD: u32;
override TILE_IDS_WORD: u32;
fn world_cell(id: vec3<u32>) -> u32 {
    if !ACTIVE_GRID { return id.x; }
    let slot=id.x/64u+id.y*atomicLoad(&metadata[METADATA_WORD+1u]);
    let missing=params.size*params.size*params.size;
    if slot>=atomicLoad(&metadata[METADATA_WORD]) { return missing; }
    let tile=atomicLoad(&metadata[TILE_IDS_WORD+slot]);
    let width=(params.size+3u)/4u;let local=id.x%64u;
    let xyz=vec3<u32>(tile%width,(tile/width)%width,tile/(width*width))*4u
        +vec3<u32>(local%4u,(local/4u)%4u,local/16u);
    if any(xyz>=vec3<u32>(params.size)) { return missing; }
    return xyz.x+params.size*(xyz.y+params.size*xyz.z);
}
fn physical_cell(id: vec3<u32>, world: u32) -> u32 {
    if !SPARSE_GRID { return world; }
    return (id.x/64u+id.y*atomicLoad(&metadata[METADATA_WORD+1u]))*64u+id.x%64u;
}
fn finite3(v: vec3<f32>) -> bool { return all(abs(v)<=vec3<f32>(3.402823e38)); }
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> { return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v); }
fn cell(id: u32) -> vec3<u32> { return vec3<u32>(id%params.size,(id/params.size)%params.size,id/(params.size*params.size)); }
fn point(id: u32) -> vec3<f32> { return params.origin_h.xyz+vec3<f32>(vec3<i32>(cell(id))+params.grid_window_cells.xyz)*params.origin_h.w; }
fn contains(body: u32, arm: vec3<f32>) -> bool {
    let q=shapes[4u*body+2u]; let s=shapes[4u*body+3u];
    let local=rotate(vec4<f32>(-q.xyz,q.w),arm);
    if s.w==1.0 { return length(local)<s.x; }
    if s.w==2.0 { return all(abs(local)<s.xyz); }
    return length(local.xz)<s.x && abs(local.y)<s.y;
}
@compute @workgroup_size(64)
fn reset(@builtin(global_invocation_id) id: vec3<u32>) {
    if !ACTIVE_GRID && id.x<params.total { owners[id.x]=0xffffffffu; }
    if id.x<params.bodies {
        spans[id.x].start=id.x*params.capacity; atomicStore(&spans[id.x].count,0u);
        let s=shapes[4u*id.x+3u]; let q=shapes[4u*id.x+2u];
        if !finite3(centers[id.x].xyz) || !finite3(s.xyz) || any(s.xyz<=vec3<f32>(0.0))
            || !all(abs(q)<=vec4<f32>(3.402823e38)) || abs(dot(q,q)-1.0)>1e-3
            || !(s.w==1.0 || s.w==2.0 || s.w==3.0) { atomicOr(&status,1u); }
    }
    if id.x==0u { indirect[0]=0u; indirect[1]=1u; indirect[2]=1u; }
}
@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) id: vec3<u32>) {
    if atomicLoad(&status)!=0u { return; }
    let world=world_cell(id);
    if world>=params.size*params.size*params.size { return; }
    let index=physical_cell(id,world);
    if ACTIVE_GRID { owners[index]=0xffffffffu; }
    let value=grid[index];
    if value.w<=1e-9 { return; }
    if !finite3(value.xyz) || !(value.w<=3.402823e38) { atomicOr(&status,4u); return; }
    for (var b=0u;b<params.bodies;b++) {
        let arm=point(world)-centers[b].xyz;
        if !contains(b,arm) { continue; }
        let xyz=cell(world);
        var boundary_mask=0u;
        for (var axis=0u;axis<3u;axis++) {
            if xyz[axis]<2u { boundary_mask|=1u<<(2u*axis); }
            if xyz[axis]>=params.size-2u { boundary_mask|=1u<<(2u*axis+1u); }
        }
        let slot=atomicAdd(&spans[b].count,1u);
        if slot>=params.capacity { atomicOr(&status,16u); return; }
        nodes[b*params.capacity+slot]=Node(vec4<f32>(arm,value.w),vec4<f32>(value.w*value.xyz,f32(boundary_mask)));
        owners[index]=b;
        return;
    }
}
@compute @workgroup_size(64)
fn apply(@builtin(global_invocation_id) id: vec3<u32>) {
    if atomicLoad(&status)!=0u { return; }
    let world=world_cell(id);
    if world<params.size*params.size*params.size {
        let index=physical_cell(id,world);
        let b=owners[index];
        if b!=0xffffffffu {
            let velocity=motion[2u*b].xyz+cross(motion[2u*b+1u].xyz,point(world)-centers[b].xyz);
            if !finite3(velocity) { atomicOr(&status,4u); return; }
            grid[index]=vec4<f32>(velocity,grid[index].w);
        }
    }
    if !ACTIVE_GRID { publish_body(id.x); }
}
fn publish_body(index: u32) {
    if index<params.bodies && atomicLoad(&status)==0u {
        let force=impulses[2u*index]/params.dt; let torque=impulses[2u*index+1u]/params.dt;
        if !finite3(force.xyz) || !finite3(torque.xyz) { atomicOr(&status,4u); return; }
        wrenches[2u*index]=force; wrenches[2u*index+1u]=torque;
    }
}
@compute @workgroup_size(64)
fn publish(@builtin(global_invocation_id) id: vec3<u32>) {
    publish_body(id.x);
}
@compute @workgroup_size(1)
fn finish() {
    indirect[0]=select(0u,params.groups,atomicLoad(&status)==0u);
}
