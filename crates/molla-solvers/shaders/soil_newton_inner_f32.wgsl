fn apply_inner_f32() {
    project_rates(0u);
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var point:array<f32,3>;
        for(var axis=0u;axis<3u;axis++) {
            var sum=0.0;
            for(var i=0u;i<params.width;i++) {sum=fma(rows[c*params.width+i][axis],value(rates[i]),sum);}
            point[axis]=sum;
        }
        var result:array<f32,6>;
        for(var i=0u;i<6u;i++) {
            var sum=0.0;
            for(var j=0u;j<6u;j++) {sum=fma(value(blocks[c].direct[6u*i+j]),component(vector[c],j),sum);}
            for(var j=0u;j<3u;j++) {sum=fma(value(blocks[c].machine[3u*i+j]),point[j],sum);}
            result[i]=sum;
            if !finite(sum) {atomicOr(&status.fault,2u);}
        }
        scratch[c]=packed(result);
    }
    storageBarrier();
}
