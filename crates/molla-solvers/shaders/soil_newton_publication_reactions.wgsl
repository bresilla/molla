    if valid {
        var momentum=vec2<f32>(0.0);var momentum_scale=1.0;
        if lane<params.width {
            let i=lane;
            for(var j=0u;j<params.width;j++) {
                let m=mass[i*params.width+j];let dj=rate_add(publication_q[j],-free_rates[j]);
                momentum=rate_add(momentum,rate_mul(m,dj));momentum_scale+=abs(m)*abs(value(dj));
            }
        }
        for(var tile=0u;tile<status.active_count;tile+=16u) {
            let count=min(16u,status.active_count-tile);
            for(var item=lane;item<count*params.width;item+=256u) {
                let local=item/params.width;let i=item%params.width;
                let c=active_indices[tile+local];let k=-publication_outputs[c].machine_impulse_normal.xyz;
                var reaction=vec2<f32>(0.0);
                for(var axis=0u;axis<3u;axis++) {reaction=rate_add(reaction,rate_mul(rows[c*params.width+i][axis],vec2<f32>(k[axis],0.0)));}
                h[2u*item]=reaction.x;h[2u*item+1u]=reaction.y;
            }
            workgroupBarrier();
            if lane<params.width {
                for(var local=0u;local<count;local++) {
                    let item=local*params.width+lane;
                    let reaction=vec2<f32>(h[2u*item],h[2u*item+1u]);
                    momentum=rate_add(momentum,reaction);momentum_scale+=abs(value(reaction));
                }
            }
            workgroupBarrier();
        }
        if lane<params.width {
            if !finite(momentum_scale) || !finite(value(momentum)) {publication_fault(2u,0u);}
            else if abs(value(momentum))>1e-6*momentum_scale {publication_fault(4u,0u);}
        }
    }
