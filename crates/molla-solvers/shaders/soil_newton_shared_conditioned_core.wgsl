var<private> conditioned_shared_ready:bool=false;

fn conditioned_shared_capacity()->bool {
    return params.width*(params.width+1u)<=params.iterations*max(params.contacts,1u);
}

fn conditioned_shared_accumulate(initial:vec2<f32>,c:u32,row_index:u32,column:u32,sign:f32)->vec2<f32> {
    let width=params.width;let offset=4160u+54u*c;let r=response[c*width+row_index];
    var sum=initial;
    if column==width {
        for(var a=0u;a<3u;a++) {sum=rate_add(sum,sign*rate_mul(-r[a],condensed[offset+36u+a]));}
    } else {
        var projected:array<vec2<f32>,3>;
        for(var a=0u;a<3u;a++) {
            var value=vec2<f32>(0.0);
            for(var b=0u;b<3u;b++) {value=rate_add(value,rate_mul(-r[b],condensed[offset+b*3u+a]));}
            projected[a]=value;
        }
        let row=rows[c*width+column];
        for(var a=0u;a<3u;a++) {sum=rate_add(sum,sign*rate_mul(row[a],projected[a]));}
    }
    return sum;
}

fn conditioned_shared_build() {
    if conditioned_shared_ready || !conditioned_shared_capacity() {return;}
    let width=params.width;let stride=width+1u;
    for(var cell=lane;cell<width*stride;cell+=256u) {
        let row_index=cell/stride;let column=cell%stride;
        var sum=vec2<f32>(select(0.0,1.0,column==row_index),0.0);
        for(var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];
            if condensed[4160u+54u*c+53u].x!=1.0 {continue;}
            sum=conditioned_shared_accumulate(sum,c,row_index,column,1.0);
        }
        basis[cell].v=Six(vec4<f32>(sum,0.0,0.0),vec4<f32>(0.0));
        if !all(abs(sum)<=vec2<f32>(3.402823e38)) {atomicStore(&condensed_failed,1u);}
    }
    storageBarrier();conditioned_shared_ready=condensed_ok();
    if lane==0u {atomicStore(&condensed_failed,0u);}
    workgroupBarrier();
}
