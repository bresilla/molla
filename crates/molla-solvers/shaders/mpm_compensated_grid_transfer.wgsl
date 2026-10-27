@group(2) @binding(6) var<storage, read_write> transfer_residual: array<atomic<u32>>;
var<workgroup> transfer_local_residual: array<atomic<u32>, 320>;
var<workgroup> transfer_rounding: array<atomic<u32>, 64>;
var<private> transfer_lane:u32;

fn transfer_round(value:f32)->f32 {
    atomicStore(&transfer_rounding[transfer_lane],bitcast<u32>(value));
    return bitcast<f32>(atomicAdd(&transfer_rounding[transfer_lane],0u));
}

fn transfer_error(a:f32,b:f32,sum:f32)->f32 {
    let z=transfer_round(sum-a);
    let left=transfer_round(a-transfer_round(sum-z));
    let right=transfer_round(b-z);
    return transfer_round(left+right);
}

fn transfer_global_tail(index:u32,value:f32) {
    if value==0.0 {return;}
    var old=atomicLoad(&transfer_residual[index]);
    loop {
        let result=atomicCompareExchangeWeak(&transfer_residual[index],old,bitcast<u32>(bitcast<f32>(old)+value));
        if result.exchanged {break;}
        old=result.old_value;
    }
}

fn transfer_local_tail(index:u32,value:f32) {
    if value==0.0 {return;}
    var old=atomicLoad(&transfer_local_residual[index]);
    loop {
        let result=atomicCompareExchangeWeak(&transfer_local_residual[index],old,bitcast<u32>(bitcast<f32>(old)+value));
        if result.exchanged {break;}
        old=result.old_value;
    }
}

@compute @workgroup_size(64)
fn clear_stiffness(@builtin(global_invocation_id) gid:vec3<u32>) {
    let world=grid_cell_id(gid);
    if world>=u.grid_total || !grid_valid() {return;}
    let index=dispatched_grid_index(world,gid);
    for(var field=0u;field<5u;field++) {atomicStore(&transfer_residual[5u*index+field],0u);}
}

@compute @workgroup_size(64)
fn resolve_stiffness(@builtin(global_invocation_id) gid:vec3<u32>) {
    let world=grid_cell_id(gid);
    if world>=u.grid_total || !grid_valid() {return;}
    let index=dispatched_grid_index(world,gid);
    atomicStore(&g_mass[index],bitcast<u32>(bitcast<f32>(atomicLoad(&g_mass[index]))+bitcast<f32>(atomicLoad(&transfer_residual[5u*index]))));
    for(var k=0u;k<3u;k++) {
        atomicStore(&g_momentum[3u*index+k],bitcast<u32>(bitcast<f32>(atomicLoad(&g_momentum[3u*index+k]))+bitcast<f32>(atomicLoad(&transfer_residual[5u*index+1u+k]))));
    }
    atomicStore(&g_kdiag[index],bitcast<u32>(bitcast<f32>(atomicLoad(&g_kdiag[index]))+bitcast<f32>(atomicLoad(&transfer_residual[5u*index+4u]))));
}
