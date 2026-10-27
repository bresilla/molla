struct Node { normal_mass:vec4<f32>, velocity_gap:vec4<f32>, boundary:vec4<u32> }
struct Query { position_mass:vec4<f32>, velocity_dt:vec4<f32>, body_boundary:vec4<u32> }
struct Contact { soil_mass:vec4<f32>, normal_gap:vec4<f32>, friction:vec4<f32>, boundary:vec4<u32> }
struct Output { machine_impulse_normal:vec4<f32>, soil_velocity_energy:vec4<f32>, machine_velocity_residual:vec4<f32>, wall_impulse:vec4<f32>, status:vec4<u32> }
struct Params { width:u32, bodies:u32, contacts:u32, pad:u32 }
struct Status { fault:atomic<u32> }
@group(0) @binding(0) var<storage,read> nodes:array<Node>;
@group(0) @binding(1) var<storage,read> queries:array<Query>;
@group(0) @binding(2) var<storage,read> centers:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> jacobians:array<f32>;
@group(0) @binding(4) var<storage,read> coefficients:array<f32>;
@group(0) @binding(5) var<storage,read_write> contacts:array<Contact>;
@group(0) @binding(6) var<storage,read_write> point_rows:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read> solution:array<Output>;
@group(0) @binding(8) var<storage,read_write> velocities:array<vec4<f32>>;
@group(0) @binding(9) var<storage,read_write> impulses:array<vec4<f32>>;
@group(0) @binding(10) var<storage,read_write> wall_impulses:array<vec4<f32>>;
@group(0) @binding(11) var<storage,read_write> status:Status;
@group(0) @binding(12) var<uniform> params:Params;
@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;
    if c>=params.contacts || atomicLoad(&status.fault)!=0u { return; }
    contacts[c]=Contact();
    for (var i=0u;i<params.width;i++) { point_rows[c*params.width+i]=vec4<f32>(0.0); }
    let node=nodes[c]; if node.normal_mass.w==0.0 { return; }
    let query=queries[c]; let body=query.body_boundary.x;
    if body>=params.bodies { atomicOr(&status.fault,128u); return; }
    let mu=coefficients[body];
    if !(mu>=0.0 && mu<=3.402823e38) { atomicOr(&status.fault,128u); return; }
    let arm=query.position_mass.xyz-centers[body].xyz;
    contacts[c]=Contact(vec4<f32>(node.velocity_gap.xyz,node.normal_mass.w),
        vec4<f32>(node.normal_mass.xyz,node.velocity_gap.w),vec4<f32>(mu,0.0,0.0,0.0),node.boundary);
    for (var i=0u;i<params.width;i++) {
        var linear=vec3<f32>(0.0); var angular=vec3<f32>(0.0);
        for (var axis=0u;axis<3u;axis++) {
            linear[axis]=jacobians[(6u*body+axis)*params.width+i];
            angular[axis]=jacobians[(6u*body+axis+3u)*params.width+i];
        }
        let point=linear+cross(angular,arm);
        if !all(abs(point)<=vec3<f32>(3.402823e38)) { atomicOr(&status.fault,4u); return; }
        point_rows[c*params.width+i]=vec4<f32>(point,0.0);
    }
}
@compute @workgroup_size(64)
fn recover(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;
    if c>=params.contacts || atomicLoad(&status.fault)!=0u { return; }
    velocities[c]=vec4<f32>(0.0); impulses[c]=vec4<f32>(0.0); wall_impulses[c]=vec4<f32>(0.0);
    if nodes[c].normal_mass.w==0.0 { return; }
    let result=solution[c];
    if result.status.x!=0u { atomicOr(&status.fault,64u); return; }
    velocities[c]=vec4<f32>(result.soil_velocity_energy.xyz,0.0);
    impulses[c]=vec4<f32>(-result.machine_impulse_normal.xyz,result.machine_impulse_normal.w);
    wall_impulses[c]=vec4<f32>(result.wall_impulse.xyz,0.0);
}
