fn apply_transpose() {
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var point=Six();
        for (var axis=0u;axis<3u;axis++) {
            var sum=vec2<f32>(0.0);
            for (var i=0u;i<6u;i++) {sum=rate_add(sum,rate_mul(blocks[c].machine[i*3u+axis],vec2<f32>(component(vector[c],i),0.0)));}
            point.k[axis]=sum.x;point.w[axis]=sum.y;
        }
        scratch[c]=point;
    }
    storageBarrier();
    if lane<params.width {
        var sum=vec2<f32>(0.0);
        for (var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];let point=scratch[c];
            for (var axis=0u;axis<3u;axis++) {sum=rate_add(sum,rate_mul(rows[c*params.width+lane][axis],vec2<f32>(point.k[axis],point.w[axis])));}
        }
        rates[lane]=sum;
    }
    storageBarrier();workgroupBarrier();
    for (var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var result:array<f32,6>;
        for (var i=0u;i<6u;i++) {
            var sum=vec2<f32>(0.0);
            for (var j=0u;j<6u;j++) {sum=rate_add(sum,rate_mul(blocks[c].direct[j*6u+i],vec2<f32>(component(vector[c],j),0.0)));}
            if i<3u {for (var j=0u;j<params.width;j++) {sum=rate_add(sum,rate_mul(-response[c*params.width+j][i],rates[j]));}}
            result[i]=value(sum);
        }
        scratch[c]=packed(result);
    }
    storageBarrier();
}
