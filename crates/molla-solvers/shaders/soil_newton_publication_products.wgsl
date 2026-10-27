    if !invalid_inputs {
        var sum=vec2<f32>(0.0);
        if lane<params.width {sum=free_rates[lane];}
        for(var tile=0u;tile<status.active_count;tile+=16u) {
            let count=min(16u,status.active_count-tile);
            for(var item=lane;item<count*params.width*3u;item+=256u) {
                let local=item/(params.width*3u);let i=(item/3u)%params.width;let axis=item%3u;
                let c=active_indices[tile+local];let k=-publication_outputs[c].machine_impulse_normal.xyz;
                let product=rate_mul(-response[c*params.width+i][axis],vec2<f32>(k[axis],0.0));
                h[2u*item]=product.x;h[2u*item+1u]=product.y;
            }
            workgroupBarrier();
            if lane<params.width {
                for(var local=0u;local<count;local++) {
                    for(var axis=0u;axis<3u;axis++) {
                        let item=(local*params.width+lane)*3u+axis;
                        sum=rate_add(sum,vec2<f32>(h[2u*item],h[2u*item+1u]));
                    }
                }
            }
            workgroupBarrier();
        }
        if lane<params.width {
            publication_q[lane]=sum;
            if !all(abs(sum)<=vec2<f32>(3.402823e38)) || !finite(value(sum)) {publication_fault(2u,0u);}
        }
    }
