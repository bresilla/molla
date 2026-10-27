fn direction_pair(c:u32,i:u32)->vec2<f32> {
    return vec2<f32>(component(direction[c],i),component(basis[params.iterations*max(params.contacts,1u)+c].z,i));
}
fn apply_direction() {
    project_rates(3u);
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var point:array<vec2<f32>,3>;
        for(var axis=0u;axis<3u;axis++) {
            var sum=vec2<f32>(0.0);
            for(var i=0u;i<params.width;i++) {sum=rate_add(sum,rate_mul(rows[c*params.width+i][axis],rates[i]));}
            point[axis]=sum;
        }
        var result:array<f32,6>;
        for(var i=0u;i<6u;i++) {
            var sum=vec2<f32>(0.0);
            for(var j=0u;j<6u;j++) {sum=rate_add(sum,rate_mul(blocks[c].direct[6u*i+j],direction_pair(c,j)));}
            for(var j=0u;j<3u;j++) {sum=rate_add(sum,rate_mul(blocks[c].machine[3u*i+j],point[j]));}
            result[i]=value(sum);
        }
        scratch[c]=packed(result);
    }
    storageBarrier();
}
