var<workgroup> refinement_norm:f32;
fn base_direction_pair(c:u32,i:u32)->vec2<f32> {
    let stride=max(params.contacts,1u);
    return vec2<f32>(component(direction[stride+c],i),component(direction[2u*stride+c],i));
}
fn solve(id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {control.stop=atomicLoad(&status.fault);}
    if workgroupUniformLoad(&control.stop)!=0u {return;}
    let stride=max(params.contacts,1u);
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var rhs:array<f32,6>;
        direction[stride+c]=Six();direction[2u*stride+c]=Six();
        for(var i=0u;i<6u;i++) {rhs[i]=-blocks[c].residual[i];}
        vector[c]=packed(rhs);scratch[c]=packed(rhs);
    }
    storageBarrier();let initial=scratch_norm();
    if lane==0u {
        refinement_norm=initial;status.beta=initial;
        if initial<0.0 || !finite(initial) {atomicOr(&status.fault,2u);}
        control.stop=atomicLoad(&status.fault);
    }
    if workgroupUniformLoad(&control.stop)!=0u || initial==0.0 {return;}
    var used=0u;
    loop {
        krylov_cycle(id,params.iterations-used);
        used+=workgroupUniformLoad(&control.count);
        if lane==0u {
            status.iterations=used;status.beta=refinement_norm;
            control.stop=atomicLoad(&status.fault)&~8u;
        }
        if workgroupUniformLoad(&control.stop)!=0u {return;}
        if workgroupUniformLoad(&control.relative)<=params.tolerance {return;}
        if used>=params.iterations || workgroupUniformLoad(&control.count)==0u {
            if lane==0u {atomicOr(&status.fault,8u);}return;
        }
        for(var slot=lane;slot<status.active_count;slot+=64u) {
            let c=active_indices[slot];direction[stride+c]=direction[c];
            direction[2u*stride+c]=basis[params.iterations*stride+c].z;
            vector[c]=scaled(scratch[c],-1.0);
        }
        if lane==0u {atomicStore(&status.fault,0u);}
        storageBarrier();
    }
}
