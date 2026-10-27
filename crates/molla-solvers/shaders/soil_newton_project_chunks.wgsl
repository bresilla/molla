fn project_rates(kind:u32) {
    let chunks=64u/params.width;let dof=lane%params.width;let chunk=lane/params.width;
    var sum=vec2<f32>(0.0);
    if chunk<chunks {
        for (var slot=chunk;slot<status.active_count;slot+=chunks) {
            let c=active_indices[slot];
            for (var axis=0u;axis<3u;axis++) {
                var point=vec2<f32>(0.0);var row=0.0;
                if kind==2u {point=vec2<f32>(scratch[c].k[axis],scratch[c].w[axis]);row=rows[c*params.width+dof][axis];}
                else {
                    row= -response[c*params.width+dof][axis];
                    if kind==1u {point=vec2<f32>(states[c].high.k[axis],states[c].low.k[axis]);}
                    else {point=vec2<f32>(vector[c].k[axis],0.0);}
                }
                sum=rate_add(sum,rate_mul(row,point));
            }
        }
    }
    parts[lane]=sum;workgroupBarrier();
    if lane<params.width {
        var total=vec2<f32>(0.0);if kind==1u {total=free_rates[lane];}
        for (var i=0u;i<chunks;i++) {total=rate_add(total,parts[i*params.width+lane]);}
        rates[lane]=total;
        if !all(abs(total)<=vec2<f32>(3.402823e38)) {atomicOr(&status.fault,2u);}
    }
    workgroupBarrier();
}
