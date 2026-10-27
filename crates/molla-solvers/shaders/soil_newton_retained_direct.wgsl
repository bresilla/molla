fn retained_direct_recover() {
    let width=params.width;let n=conditioned_core_width;let stride=n+4u;
    let retained=workgroupUniformLoad(&conditioned_retained);
    for(var slot=lane;slot<status.active_count;slot+=256u) {
        let c=active_indices[slot];let offset=4160u+54u*c;
        var point:array<vec2<f32>,3>;
        if c!=retained {
            for(var a=0u;a<3u;a++) {
                var sum=vec2<f32>(0.0);
                for(var i=0u;i<width;i++) {sum=rate_add(sum,rate_mul(rows[c*width+i][a],condensed[i*stride+n]));}
                point[a]=sum;
            }
        }
        var high:array<f32,6>;var low:array<f32,6>;
        for(var i=0u;i<6u;i++) {
            var sum=vec2<f32>(0.0);
            if c==retained {sum=condensed[(width+i)*stride+n];}
            else {
                sum=condensed[offset+36u+i];
                for(var a=0u;a<3u;a++) {sum=rate_add(sum,-fischer_product(condensed[offset+i*3u+a],point[a]));}
            }
            high[i]=sum.x;low[i]=sum.y;
            if !all(abs(sum)<=vec2<f32>(3.402823e38)) {atomicStore(&condensed_failed,1u);}
        }
        direction[c]=packed(high);basis[params.iterations*max(params.contacts,1u)+c].z=packed(low);
    }
    storageBarrier();
}

fn retained_direct_solve()->bool {
    conditioned_cache();
    if lane==0u {
        control.count=0u;
        for(var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];
            if condensed[4160u+54u*c+53u].x!=1.0 {control.count++;}
        }
    }
    let invalid=workgroupUniformLoad(&control.count);
    if invalid==0u {
        direct_core();if !condensed_ok() {return false;}
        if !condensed_factor(params.width) {return false;}
        direct_recover();
    } else {
        if invalid!=1u {return false;}
        if !conditioned_prepare(0xffffffffu) {return false;}
        retained_direct_recover();
    }
    return condensed_ok();
}
