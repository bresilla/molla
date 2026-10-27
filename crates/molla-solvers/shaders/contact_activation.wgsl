struct Row {
    point: vec4<f32>, normal: vec4<f32>, coefficients: vec4<f32>, endpoints: vec4<u32>,
    relative: vec4<f32>, surface: vec4<f32>, force: vec4<f32>, penalty: vec4<f32>,
}
struct Spatial { linear: vec4<f32>, angular: vec4<f32> }
struct Params { groups: u32, rows: u32, bodies: u32, edges: u32, watches: u32, identities: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> awake: array<u32>;
@group(0) @binding(1) var<storage, read_write> roots: array<u32>;
@group(0) @binding(2) var<storage, read_write> component_awake: array<u32>;
@group(0) @binding(3) var<storage, read> dependencies: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> watches: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> rows: array<Row>;
@group(0) @binding(6) var<storage, read_write> impacts: array<Row>;
@group(0) @binding(7) var<storage, read_write> onsets: array<u32>;
@group(0) @binding(8) var<storage, read> body_groups: array<u32>;
@group(0) @binding(9) var<storage, read> velocities: array<Spatial>;
@group(0) @binding(10) var<storage, read> accelerations: array<Spatial>;
@group(0) @binding(11) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(12) var<storage, read> identities: array<u32>;
@group(0) @binding(13) var<uniform> u: Params;
const NONE: u32 = 0xffffffffu;
const INPUT: u32 = 1048576u;

fn root(group: u32) -> u32 {
    var result = group;
    for (var i=0u; i<u.groups; i++) {
        let parent=roots[result];
        if parent==result { break; }
        roots[result]=roots[parent];
        result=parent;
    }
    return result;
}
fn unite(a: u32, b: u32) {
    if a==NONE || b==NONE { return; }
    let left=root(a); let right=root(b);
    roots[max(left,right)]=min(left,right);
}
fn group(body: u32) -> u32 {
    if body==NONE { return NONE; }
    if body>=u.bodies { atomicOr(&status,INPUT); return NONE; }
    return body_groups[body];
}
fn moving(body: u32) -> bool {
    if body==NONE { return false; }
    let v=velocities[body]; let a=accelerations[body];
    let finite=all(abs(v.linear)<=vec4<f32>(3.402823e38)) && all(abs(v.angular)<=vec4<f32>(3.402823e38))
        && all(abs(a.linear)<=vec4<f32>(3.402823e38)) && all(abs(a.angular)<=vec4<f32>(3.402823e38));
    if !finite { atomicOr(&status,INPUT); return false; }
    return any(v.linear.xyz!=vec3<f32>(0.0)) || any(v.angular.xyz!=vec3<f32>(0.0))
        || any(a.linear.xyz!=vec3<f32>(0.0)) || any(a.angular.xyz!=vec3<f32>(0.0));
}
fn shape_key(shape: u32) -> u32 {
    if u.identities==0u { return shape; }
    if shape>=u.identities { atomicOr(&status,INPUT); return NONE; }
    return identities[shape];
}

@compute @workgroup_size(1)
fn propagate() {
    if atomicLoad(&status)!=0u { return; }
    for (var i=0u; i<u.groups; i++) { roots[i]=i; component_awake[i]=0u; }
    for (var i=0u; i<u.edges; i++) { unite(dependencies[i].x,dependencies[i].y); }
    for (var i=0u; i<u.rows; i++) {
        let row=rows[i];
        if row.coefficients.w==0.0 { continue; }
        if row.coefficients.w!=1.0 { atomicOr(&status,INPUT); return; }
        let a=group(row.endpoints.z); let b=group(row.endpoints.w);
        if atomicLoad(&status)!=0u { return; }
        if a==NONE && b==NONE { atomicOr(&status,INPUT); return; }
        unite(a,b);
        let belt=any(row.surface.xyz!=vec3<f32>(0.0));
        if a!=NONE && b==NONE && (moving(row.endpoints.w) || belt) { awake[a]=1u; }
        if b!=NONE && a==NONE && (moving(row.endpoints.z) || belt) { awake[b]=1u; }
    }
    for (var i=0u; i<u.watches; i++) {
        let watch=watches[i];
        var found=false;
        for (var j=0u; j<u.rows; j++) {
            let row=rows[j];
            if row.coefficients.w!=1.0 { continue; }
            let a=shape_key(row.endpoints.x); let b=shape_key(row.endpoints.y);
            if (a==watch.y && b==watch.z) || (a==watch.z && b==watch.y) { found=true; break; }
        }
        if !found { awake[watch.x]=1u; }
    }
    if atomicLoad(&status)!=0u { return; }
    for (var i=0u; i<u.groups; i++) { component_awake[root(i)]|=awake[i]; }
    for (var i=0u; i<u.groups; i++) { awake[i]=component_awake[root(i)]; }
}

@compute @workgroup_size(64)
fn retire(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x;
    if i>=u.rows || atomicLoad(&status)!=0u { return; }
    let row=rows[i];
    if row.coefficients.w==0.0 { return; }
    let a=group(row.endpoints.z); let b=group(row.endpoints.w);
    if (a!=NONE && awake[a]!=0u) || (b!=NONE && awake[b]!=0u) { return; }
    rows[i].coefficients.w=0.0;
    rows[i].normal.w=0.0;
    rows[i].force=vec4<f32>(0.0,0.0,0.0,-1.0);
    impacts[i].coefficients.w=0.0;
    impacts[i].normal.w=0.0;
    impacts[i].force=vec4<f32>(0.0);
    onsets[i]=0u;
}
