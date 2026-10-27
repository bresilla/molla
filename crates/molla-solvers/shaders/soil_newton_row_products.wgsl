fn ROW_PRODUCT() {
    project_rates(PROJECTION_MODE);
    if lane==0u {row_product_count=status.active_count;}
    let count=workgroupUniformLoad(&row_product_count);
    for(var base=0u;base<count;base+=256u) {
        let size=min(256u,count-base);
        if lane<size {
            let c=active_indices[base+lane];
            for(var axis=0u;axis<3u;axis++) {
                var sum=vec2<f32>(0.0);
                for(var i=0u;i<params.width;i++) {sum=rate_add(sum,rate_mul(rows[c*params.width+i][axis],rates[i]));}
                row_product_points[3u*lane+axis]=sum;
            }
        }
        workgroupBarrier();
        for(var cell=lane;cell<6u*size;cell+=256u) {
            let local=cell/6u;let i=cell%6u;let c=active_indices[base+local];
            var sum=vec2<f32>(0.0);
            for(var j=0u;j<6u;j++) {sum=rate_add(sum,fischer_product(blocks[c].direct[6u*i+j],DIRECT_ARGUMENT));}
            for(var j=0u;j<3u;j++) {sum=rate_add(sum,fischer_product(blocks[c].machine[3u*i+j],row_product_points[3u*local+j]));}
            if i<3u {scratch[c].k[i]=value(sum);}else {scratch[c].w[i-3u]=value(sum);}
            if i==0u {scratch[c].k.w=0.0;}else if i==3u {scratch[c].w.w=0.0;}
        }
        workgroupBarrier();
    }
    storageBarrier();
}
