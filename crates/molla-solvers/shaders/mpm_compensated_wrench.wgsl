@group(3) @binding(3) var<storage,read_write> wrench_tail:array<atomic<u32>>;
var<workgroup> wrench_rounding:array<atomic<u32>,64>;
var<private> wrench_lane:u32;

fn wrench_round(value:f32)->f32 {
    atomicStore(&wrench_rounding[wrench_lane],bitcast<u32>(value));
    return bitcast<f32>(atomicAdd(&wrench_rounding[wrench_lane],0u));
}

fn wrench_error(a:f32,b:f32,sum:f32)->f32 {
    let z=wrench_round(sum-a);
    let left=wrench_round(a-wrench_round(sum-z));
    let right=wrench_round(b-z);
    return wrench_round(left+right);
}

fn wrench_add_tail(index:u32,value:f32) {
    if value==0.0 {return;}
    var old=atomicLoad(&wrench_tail[index]);
    loop {
        let result=atomicCompareExchangeWeak(&wrench_tail[index],old,bitcast<u32>(bitcast<f32>(old)+value));
        if result.exchanged {break;}
        old=result.old_value;
    }
}

@compute @workgroup_size(64)
fn wrench_resolve(@builtin(global_invocation_id) gid:vec3<u32>) {
    if gid.x>=u.rigid_count || !grid_valid() {return;}
    for(var k=0u;k<8u;k++) {
        let index=8u*gid.x+k;
        atomicStore(&rigid_wrench[index],bitcast<u32>(bitcast<f32>(atomicLoad(&rigid_wrench[index]))+bitcast<f32>(atomicLoad(&wrench_tail[index]))));
    }
}
