fn cycle_finish(checkpoint:f32) {
    if lane==0u && status.relative>params.tolerance {atomicOr(&status.fault,8u);}
    let count=workgroupUniformLoad(&control.count);
    if lane==0u {
        let used=bitcast<u32>(condensed[2183u].x)+count;
        condensed[2183u].x=bitcast<f32>(used);
        control.scalar=bitcast<f32>(used);
        status.iterations=used;status.beta=refinement_norm;
        control.stop=atomicLoad(&status.fault)&~8u;
    }
    let used=bitcast<u32>(workgroupUniformLoad(&control.scalar));
    if workgroupUniformLoad(&control.stop)!=0u || workgroupUniformLoad(&control.relative)<=params.tolerance {
        cycle_save(checkpoint);
        if lane==0u {condensed[2182u].x=bitcast<f32>(1u);}
        return;
    }
    if used>=params.iterations || count==0u {
        if lane==0u {atomicOr(&status.fault,8u);}
        cycle_save(checkpoint);
        if lane==0u {condensed[2182u].x=bitcast<f32>(1u);}
        return;
    }
    let stride=max(params.contacts,1u);
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];direction[stride+c]=direction[c];
        direction[2u*stride+c]=basis[params.iterations*stride+c].z;
        vector[c]=scaled(scratch[c],-1.0);
    }
    if lane==0u {atomicStore(&status.fault,0u);}
    storageBarrier();cycle_save(checkpoint);
    if lane==0u {
        condensed[2182u].x=bitcast<f32>(2u);
        condensed[2183u].y=bitcast<f32>(bitcast<u32>(condensed[2183u].y)+1u);
    }
}
