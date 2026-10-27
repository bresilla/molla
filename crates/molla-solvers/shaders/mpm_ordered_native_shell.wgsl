var<workgroup> shell_force:array<vec4<f32>,64>;
var<workgroup> shell_torque:array<vec4<f32>,64>;

@compute @workgroup_size(64)
fn ordered_native_shell(@builtin(workgroup_id) group:vec3<u32>, @builtin(num_workgroups) groups:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let r=group.x+group.y*groups.x;
    if r>=u.rigid_count || !grid_valid() {return;}
    if rigid_vel[4u*r+3u].w!=0.0 {return;}
    let cr=rigid_center[r];let center=cr.xyz;let radius=cr.w;
    let edge=vec3<f32>(f32(u.grid_size-1u));
    let extent=radius+u.cell_size;
    let lower=vec3<f32>(vec3<i32>(floor((center-vec3<f32>(extent)-u.grid_origin)*u.inv_cell_size))-u.grid_window_cells.xyz-vec3<i32>(1));
    let upper=vec3<f32>(vec3<i32>(ceil((center+vec3<f32>(extent)-u.grid_origin)*u.inv_cell_size))-u.grid_window_cells.xyz+vec3<i32>(1));
    if any(upper<vec3<f32>(0.0)) || any(lower>edge) {return;}
    let lo=vec3<u32>(clamp(lower,vec3<f32>(0.0),edge));
    let hi=vec3<u32>(clamp(upper,vec3<f32>(0.0),edge));
    let size=hi-lo+vec3<u32>(1u);let count=size.x*size.y*size.z;
    var force_sum=vec3<f32>(0.0);var torque_sum=vec3<f32>(0.0);
    for(var batch=0u;batch<(count-1u)/64u+1u;batch++) {
        let flat=batch*64u+lane;if flat>=count {continue;}
        let cell=lo+vec3<u32>(flat%size.x,(flat/size.x)%size.y,flat/(size.x*size.y));
        let world=cell.x+u.grid_size*(cell.y+u.grid_size*cell.z);
        let cidx=grid_storage_index(world);if cidx==0xffffffffu {continue;}
        let m=bitcast<f32>(atomicLoad(&g_mass[cidx]));
        if m<=1e-9 || !(m==m) || g_rigid_mask[cidx]!=0u {continue;}
        let node_pos=u.grid_origin+vec3<f32>(vec3<i32>(cell)+u.grid_window_cells.xyz)*u.cell_size;
        let d=distance(node_pos,center);if d>radius+u.cell_size {continue;}
        var w=1.0;
        if d>radius {let t=(d-radius)/u.cell_size;w=max(0.0,1.0-t);}
        let dv=rigid_vel[4u*r].xyz-g_velocity[cidx].xyz;
        let force=-(m*w/max(u.dt,1e-6))*dv;
        let torque=cross(node_pos-center,force);
        force_sum+=force;torque_sum+=torque;
    }
    shell_force[lane]=vec4<f32>(force_sum,0.0);shell_torque[lane]=vec4<f32>(torque_sum,0.0);
    workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u) {
        if lane<stride {shell_force[lane]+=shell_force[lane+stride];shell_torque[lane]+=shell_torque[lane+stride];}
        workgroupBarrier();
    }
    if lane==0u {
        for(var axis=0u;axis<3u;axis++) {
            let f=r*8u+axis;let t=f+4u;
            atomicStore(&rigid_wrench[f],atomicLoad(&rigid_wrench[f])+shell_force[0][axis]);
            atomicStore(&rigid_wrench[t],atomicLoad(&rigid_wrench[t])+shell_torque[0][axis]);
        }
    }
}
