struct Node { normal_mass:vec4<f32>, velocity_gap:vec4<f32>, boundary:vec4<u32> }
struct Query { position_mass:vec4<f32>, velocity_dt:vec4<f32>, body_boundary:vec4<u32> }
struct Params { width:u32, bodies:u32, contacts:u32, pad:u32 }
struct Status { fault:atomic<u32>, iterations:u32, backtracks:u32, pivots:u32 }
@group(0) @binding(0) var<storage,read> nodes:array<Node>;
@group(0) @binding(1) var<storage,read> queries:array<Query>;
@group(0) @binding(2) var<storage,read> centers:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> jacobians:array<f32>;
@group(0) @binding(4) var<storage,read> coefficients:array<f32>;
@group(0) @binding(5) var<storage,read_write> contacts:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> normal_data:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> tangent_rows:array<vec2<f32>>;
@group(0) @binding(8) var<storage,read> tangent_solution:array<vec4<f32>>;
@group(0) @binding(9) var<storage,read> normal_solution:array<vec4<f32>>;
@group(0) @binding(10) var<storage,read_write> velocities:array<vec4<f32>>;
@group(0) @binding(11) var<storage,read_write> impulses:array<vec4<f32>>;
@group(0) @binding(12) var<storage,read_write> status:Status;
@group(0) @binding(13) var<uniform> params:Params;
@group(0) @binding(14) var<storage,read_write> normal_rows:array<f32>;
fn frame(n:vec3<f32>)->mat3x3<f32> {
    let normal=normalize(n); let a=abs(normal);
    var axis=vec3<f32>(1.0,0.0,0.0);
    if a.y<a.x && a.y<=a.z { axis=vec3<f32>(0.0,1.0,0.0); }
    else if a.z<a.x && a.z<a.y { axis=vec3<f32>(0.0,0.0,1.0); }
    let first=normalize(cross(axis,normal));
    return mat3x3<f32>(first,cross(normal,first),normal);
}
@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;
    if c>=params.contacts || atomicLoad(&status.fault)!=0u { return; }
    contacts[c]=vec4<f32>(0.0); normal_data[c]=vec4<f32>(0.0);
    for (var i=0u;i<params.width;i++) { tangent_rows[c*params.width+i]=vec2<f32>(0.0); normal_rows[c*params.width+i]=0.0; }
    let node=nodes[c]; let mass=node.normal_mass.w;
    if mass==0.0 { return; }
    if node.boundary.x!=0u { atomicOr(&status.fault,64u); return; }
    let query=queries[c]; let body=query.body_boundary.x;
    if body>=params.bodies { atomicOr(&status.fault,128u); return; }
    let mu=coefficients[body];
    if !(mu>=0.0 && mu<=3.402823e38) { atomicOr(&status.fault,128u); return; }
    let basis=frame(node.normal_mass.xyz);
    let arm=query.position_mass.xyz-centers[body].xyz;
    let free=transpose(basis)*node.velocity_gap.xyz;
    contacts[c]=vec4<f32>(mass,mu,free.xy);
    normal_data[c]=vec4<f32>(free.z,node.velocity_gap.w,0.0,0.0);
    for (var i=0u;i<params.width;i++) {
        var linear=vec3<f32>(0.0); var angular=vec3<f32>(0.0);
        for (var axis=0u;axis<3u;axis++) {
            linear[axis]=jacobians[(6u*body+axis)*params.width+i];
            angular[axis]=jacobians[(6u*body+axis+3u)*params.width+i];
        }
        let point=linear+cross(angular,arm);
        let row=transpose(basis)*point;
        tangent_rows[c*params.width+i]=row.xy;
        normal_rows[c*params.width+i]=row.z;
    }
}
@compute @workgroup_size(64)
fn recover(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;
    if c>=params.contacts || atomicLoad(&status.fault)!=0u { return; }
    velocities[c]=vec4<f32>(0.0); impulses[c]=vec4<f32>(0.0);
    if nodes[c].normal_mass.w==0.0 { return; }
    let basis=frame(nodes[c].normal_mass.xyz);
    let tangent=tangent_solution[c]; let normal=normal_solution[c];
    let velocity=basis*vec3<f32>(tangent.zw,normal.y);
    let impulse=basis*vec3<f32>(-tangent.xy,normal.x);
    if !all(abs(velocity)<=vec3<f32>(3.402823e38)) || !all(abs(impulse)<=vec3<f32>(3.402823e38)) { atomicOr(&status.fault,4u); return; }
    velocities[c]=vec4<f32>(velocity,0.0); impulses[c]=vec4<f32>(impulse,normal.x);
}
