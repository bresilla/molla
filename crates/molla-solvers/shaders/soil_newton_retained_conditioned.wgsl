var<workgroup> conditioned_retained:u32;
var<private> conditioned_core_width:u32=0u;

fn conditioned_prepare(chosen:u32)->bool {
    conditioned_delta_valid=false;
    let width=params.width;
    if lane==0u {
        atomicStore(&condensed_failed,0u);conditioned_retained=0xffffffffu;
        for(var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];
            if c!=chosen && condensed[4160u+54u*c+53u].x!=1.0 {
                if conditioned_retained!=0xffffffffu {atomicStore(&condensed_failed,1u);}
                conditioned_retained=c;
            }
        }
    }
    if !condensed_ok() {return false;}
    let retained=workgroupUniformLoad(&conditioned_retained);
    let n=width+select(0u,6u,retained!=0xffffffffu);let stride=n+4u;
    conditioned_core_width=n;
    for(var cell=lane;cell<n*stride;cell+=256u) {
        let row_index=cell/stride;let column=cell%stride;var sum=vec2<f32>(0.0);
        if row_index<width {
            if column<width || column==n {
                sum=vec2<f32>(select(0.0,1.0,column==row_index),0.0);
                for(var slot=0u;slot<status.active_count;slot++) {
                    let c=active_indices[slot];if c==chosen || c==retained {continue;}
                    let offset=4160u+54u*c;let r=response[c*width+row_index];
                    if column==n {
                        for(var a=0u;a<3u;a++) {sum=rate_add(sum,rate_mul(-r[a],condensed[offset+36u+a]));}
                    } else {
                        var projected:array<vec2<f32>,3>;
                        for(var a=0u;a<3u;a++) {
                            var value=vec2<f32>(0.0);
                            for(var b=0u;b<3u;b++) {value=rate_add(value,rate_mul(-r[b],condensed[offset+b*3u+a]));}
                            projected[a]=value;
                        }
                        let row=rows[c*width+column];
                        for(var a=0u;a<3u;a++) {sum=rate_add(sum,rate_mul(row[a],projected[a]));}
                    }
                }
            } else if column<n {
                let axis=column-width;
                if axis<3u {sum=vec2<f32>(response[retained*width+row_index][axis],0.0);}
            } else {
                sum=vec2<f32>(response[chosen*width+row_index][column-n-1u],0.0);
            }
        } else {
            let local=row_index-width;
            if column<width {
                for(var axis=0u;axis<3u;axis++) {
                    sum=rate_add(sum,rate_mul(rows[retained*width+column][axis],blocks[retained].machine[3u*local+axis]));
                }
            } else if column<n {
                sum=blocks[retained].direct[6u*local+column-width];
            } else if column==n {
                sum=vec2<f32>(-blocks[retained].residual[local],0.0);
            }
        }
        condensed[cell]=sum;condensed[1600u+cell]=sum;
        if !all(abs(sum)<=vec2<f32>(3.402823e38)) {atomicStore(&condensed_failed,1u);}
    }
    if lane<3u {condensed[4090u+lane]=vec2<f32>(states[chosen].high.k[lane],states[chosen].low.k[lane]);}
    storageBarrier();if !condensed_ok() {return false;}
    return conditioned_factor();
}

fn retained_conditioned_state(c:u32,next:Six)->State {
    let old=states[c];var high:array<f32,6>;var low:array<f32,6>;
    for(var i=0u;i<6u;i++) {
        var x=rate_add(vec2<f32>(component(old.high,i),component(old.low,i)),conditioned_delta(next,params.width+i));
        if i>=3u && flags(c)[i-3u]==0.0 {x=vec2<f32>(0.0);}
        high[i]=x.x;low[i]=x.y;
    }
    return State(packed(high),packed(low));
}
