var<workgroup> factor_stop:u32;

fn factor_sum(i:u32,j:u32)->f32 {
    let n=params.width;
    var sum=vec2<f32>(mass[i*n+j],0.0);
    for(var k=0u;k<j;k++) {sum=add_nonzero_term(sum,-lower[i*n+k],lower[j*n+k]);}
    return value(sum);
}

@compute @workgroup_size(32)
fn factor(@builtin(local_invocation_index) lane:u32) {
    rate_lane=lane;let n=params.width;
    for(var i=lane;i<n*n;i+=32u) {lower[i]=0.0;}
    storageBarrier();
    if lane==0u {
        factor_stop=atomicLoad(&status.fault);
        if factor_stop==0u {
            if upstream[0]==0u {failure[0]=0u;}
            atomicStore(&status.contact,0u);atomicStore(&status.active_count,0u);atomicStore(&status.max_residual,0u);
            if upstream[0]!=0u {fault(1u,0u);factor_stop=1u;}
        }
    }
    if workgroupUniformLoad(&factor_stop)!=0u {return;}
    if lane<n {
        for(var j=0u;j<n;j++) {
            let m=mass[lane*n+j];if !finite(m) || m!=mass[j*n+lane] {fault(2u,0u);}
        }
    }
    storageBarrier();
    if lane==0u {factor_stop=atomicLoad(&status.fault);}
    if workgroupUniformLoad(&factor_stop)!=0u {return;}
    for(var j=0u;j<n;j++) {
        if lane==j {
            let d=factor_sum(j,j);
            if !finite(d) || d<=0.0 {fault(2u,0u);}
            else {
                let l=sqrt(d);
                if !finite(l) {fault(2u,0u);} else {lower[j*n+j]=l;}
            }
        }
        storageBarrier();
        if lane==0u {factor_stop=atomicLoad(&status.fault);}
        if workgroupUniformLoad(&factor_stop)!=0u {return;}
        if lane<n && lane>j {
            let d=factor_sum(lane,j);
            if !finite(d) {fault(2u,0u);}
            else {
                let l=d/lower[j*n+j];
                if !finite(l) {fault(2u,0u);} else {lower[lane*n+j]=l;}
            }
        }
        storageBarrier();
        if lane==0u {factor_stop=atomicLoad(&status.fault);}
        if workgroupUniformLoad(&factor_stop)!=0u {return;}
    }
}
