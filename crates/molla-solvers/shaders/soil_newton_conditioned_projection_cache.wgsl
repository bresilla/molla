    if width<=params.iterations {
        for(var cell=lane;cell<status.active_count*width;cell+=256u) {
            let row_index=cell%width;let c=active_indices[cell/width];
            if c==chosen || c==retained {continue;}
            let offset=4160u+54u*c;let r=response[c*width+row_index];
            var high=vec4<f32>(0.0);var low=vec4<f32>(0.0);
            for(var a=0u;a<3u;a++) {
                var value=vec2<f32>(0.0);
                for(var b=0u;b<3u;b++) {value=rate_add(value,rate_mul(-r[b],condensed[offset+b*3u+a]));}
                high[a]=value.x;low[a]=value.y;
            }
            basis[row_index*max(params.contacts,1u)+c].v=Six(high,low);
        }
        storageBarrier();
    }
