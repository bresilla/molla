fn direct_core() {
    let n=params.width;let stride=n+1u;
    for(var cell=lane;cell<n*stride;cell+=256u) {
        let row_index=cell/stride;let column=cell%stride;
        var sum=vec2<f32>(select(0.0,1.0,column==row_index),0.0);
        for(var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];let offset=4160u+54u*c;let r=response[c*n+row_index];
            if column==n {
                for(var a=0u;a<3u;a++) {sum=rate_add(sum,rate_mul(r[a],condensed[offset+36u+a]));}
            } else {
                var projected:array<vec2<f32>,3>;
                for(var a=0u;a<3u;a++) {
                    var value=vec2<f32>(0.0);
                    for(var b=0u;b<3u;b++) {value=rate_add(value,rate_mul(-r[b],condensed[offset+b*3u+a]));}
                    projected[a]=value;
                }
                let row=rows[c*n+column];
                for(var a=0u;a<3u;a++) {sum=rate_add(sum,rate_mul(row[a],projected[a]));}
            }
        }
        condensed[cell]=sum;
        if !all(abs(sum)<=vec2<f32>(3.402823e38)) {atomicStore(&condensed_failed,1u);}
    }
    storageBarrier();
}
