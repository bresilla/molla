@group(0) @binding(13) var<storage,read> newton_status:array<u32>;
@group(0) @binding(14) var<storage,read> preparation_status:array<u32>;
@group(0) @binding(15) var<storage,read> publication_status:array<u32>;
@compute @workgroup_size(64)
fn recover(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;let p=preparation_status[0];let n=newton_status[0];let certificate=publication_status[0];
    let failure=select(0u,4096u,p!=0u)|select(0u,8192u,n!=0u)|select(0u,16384u,certificate!=0u);
    if c==0u {
        let previous=atomicOr(&status.fault,failure);
        if (previous&28672u)==0u {
            status.preparation=p;status.newton=n;status.publication=certificate;
            status.steps=newton_status[1];status.linear=newton_status[2];status.trials=newton_status[3];
        }
    }
    if c>=max(params.contacts,1u) {return;}
    velocities[c]=vec4<f32>(0.0);impulses[c]=vec4<f32>(0.0);wall_impulses[c]=vec4<f32>(0.0);
    if c>=params.contacts || failure!=0u || atomicLoad(&status.fault)!=0u || nodes[c].normal_mass.w==0.0 {return;}
    let result=solution[c];
    if result.status.x!=0u {atomicOr(&status.fault,64u);return;}
    velocities[c]=vec4<f32>(result.soil_velocity_energy.xyz,0.0);
    impulses[c]=vec4<f32>(-result.machine_impulse_normal.xyz,result.machine_impulse_normal.w);
    wall_impulses[c]=vec4<f32>(result.wall_impulse.xyz,0.0);
}
