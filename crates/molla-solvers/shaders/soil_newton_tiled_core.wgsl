fn tiled_core(chosen:u32,conditioned:bool) {
    let n=params.width;let stride=n+select(1u,4u,conditioned);
    var sums:array<vec2<f32>,5>;
    for(var cell=lane;cell<n*stride;cell+=256u) {
        let i=cell/stride;let j=cell%stride;
        var sum=vec2<f32>(select(0.0,1.0,i==j),0.0);
        if j>n {sum=vec2<f32>(response[chosen*n+i][j-n-1u],0.0);}
        sums[cell/256u]=sum;
    }
    for(var first=0u;first<status.active_count;first+=16u) {
        let count=min(16u,status.active_count-first);
        for(var item=lane;item<count*n*3u;item+=256u) {
            let local=item/(n*3u);let row_index=(item/3u)%n;let a=item%3u;
            let c=active_indices[first+local];let offset=4160u+54u*c;
            var value=vec2<f32>(0.0);
            if !conditioned || c!=chosen {
                let r=response[c*n+row_index];
                for(var b=0u;b<3u;b++) {value=rate_add(value,rate_mul(-r[b],condensed[offset+b*3u+a]));}
            }
            condensed[2304u+item]=value;
        }
        storageBarrier();
        for(var cell=lane;cell<n*stride;cell+=256u) {
            let row_index=cell/stride;let column=cell%stride;
            var sum=sums[cell/256u];
            if column<=n {
                for(var local=0u;local<count;local++) {
                    let c=active_indices[first+local];
                    if conditioned && c==chosen {continue;}
                    if column==n {
                        let offset=4160u+54u*c;let r=response[c*n+row_index];
                        for(var a=0u;a<3u;a++) {
                            sum=rate_add(sum,rate_mul(select(r[a],-r[a],conditioned),condensed[offset+36u+a]));
                        }
                    } else {
                        let row=rows[c*n+column];let base=2304u+(local*n+row_index)*3u;
                        for(var a=0u;a<3u;a++) {sum=rate_add(sum,rate_mul(row[a],condensed[base+a]));}
                    }
                }
            }
            sums[cell/256u]=sum;
        }
        storageBarrier();
    }
    for(var cell=lane;cell<n*stride;cell+=256u) {
        let sum=sums[cell/256u];condensed[cell]=sum;
        if conditioned {condensed[1152u+cell]=sum;}
        if !all(abs(sum)<=vec2<f32>(3.402823e38)) {atomicStore(&condensed_failed,1u);}
    }
    storageBarrier();
}
