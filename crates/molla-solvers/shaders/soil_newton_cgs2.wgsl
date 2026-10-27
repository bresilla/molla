fn cgs2_orthogonalize(column:u32) {
    let count=column+1u;
    var width=256u;
    loop {if width*count<=256u {break;}width/=2u;}
    let group=lane/width;let member=lane%width;
    for(var sweep=0u;sweep<2u;sweep++) {
        var sum=vec2<f32>(0.0);
        if group<count {
            for(var slot=member;slot<status.active_count;slot+=width) {
                let c=active_indices[slot];let x=basis[group*max(params.contacts,1u)+c].v;let y=scratch[c];
                for(var i=0u;i<6u;i++) {sum=rate_add(sum,rate_mul(component(x,i),vec2<f32>(component(y,i),0.0)));}
            }
        }
        parts[lane]=sum;workgroupBarrier();
        for(var stride=width/2u;stride>0u;stride/=2u) {
            if member<stride {parts[lane]=rate_add(parts[lane],parts[lane+stride]);}
            workgroupBarrier();
        }
        if member==0u && group<count {
            let projection=value(parts[lane]);coefficients[group]=projection;h[group*64u+column]+=projection;
        }
        workgroupBarrier();
        for(var slot=lane;slot<status.active_count;slot+=64u) {
            let c=active_indices[slot];var result=array<f32,6>();
            for(var k=0u;k<6u;k++) {
                var v=component(scratch[c],k);
                for(var i=0u;i<count;i++) {
                    let base=basis[i*max(params.contacts,1u)+c].v;
                    v=value(rate_add(vec2<f32>(v,0.0),rate_mul(-coefficients[i],vec2<f32>(component(base,k),0.0))));
                }
                result[k]=v;
            }
            scratch[c]=packed(result);
        }
        storageBarrier();workgroupBarrier();
    }
}
