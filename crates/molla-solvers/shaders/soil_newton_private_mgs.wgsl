
fn private_mgs_product(local:Six,index:u32)->f32 {
    var sum=vec2<f32>(0.0);
    if lane<status.active_count {
        let c=active_indices[lane];let x=basis[index*max(params.contacts,1u)+c].v;
        for (var i=0u;i<6u;i++) {sum=rate_add(sum,rate_mul(component(x,i),vec2<f32>(component(local,i),0.0)));}
    }
    parts[lane]=sum;let total=reduce_parts();
    if lane==0u {control.scalar=value(total);}
    return workgroupUniformLoad(&control.scalar);
}
fn private_mgs_orthogonalize(j:u32) {
    if status.active_count>256u {
__FALLBACK__
        return;
    }
    var local=Six();
    if lane<status.active_count {local=scratch[active_indices[lane]];}
    for (var orthogonal_pass=0u;orthogonal_pass<2u;orthogonal_pass++) {for (var i=0u;i<=j;i++) {
        let projection=private_mgs_product(local,i);
        if lane==0u {h[i*64u+j]+=projection;}
        if lane<status.active_count {
            let c=active_indices[lane];let base=basis[i*max(params.contacts,1u)+c].v;var result:array<f32,6>;
            for (var k=0u;k<6u;k++) {result[k]=value(rate_add(vec2<f32>(component(local,k),0.0),rate_mul(-projection,vec2<f32>(component(base,k),0.0))));}
            local=packed(result);
        }
    }}
    if lane<status.active_count {scratch[active_indices[lane]]=local;}
    storageBarrier();
}
