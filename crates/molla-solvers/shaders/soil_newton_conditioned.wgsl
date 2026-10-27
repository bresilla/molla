var<private> conditioned_model:bool=false;
var<workgroup> conditioned_best:u32;

fn conditioned_cache() {
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];let offset=4160u+54u*c;
        var valid=direct_contact(c);
        if valid {
            for(var column=0u;column<4u;column++) {
                var error=0.0;var scale=0.0;
                for(var i=0u;i<6u;i++) {
                    var rhs=vec2<f32>(-blocks[c].residual[i],0.0);
                    if column<3u {rhs=blocks[c].machine[i*3u+column];}
                    var sum=-rhs;
                    for(var j=0u;j<6u;j++) {
                        var x=condensed[offset+36u+j];
                        if column<3u {x=condensed[offset+j*3u+column];}
                        sum=rate_add(sum,fischer_product(blocks[c].direct[i*6u+j],x));
                    }
                    error+=value(sum)*value(sum);scale+=value(rhs)*value(rhs);
                }
                let relative=sqrt(error)/max(sqrt(scale),1e-30);
                valid=valid && finite(relative) && relative<=params.tolerance;
            }
        }
        condensed[offset+53u]=vec2<f32>(select(0.0,1.0,valid),0.0);
    }
    storageBarrier();
}

fn conditioned_factor()->bool {
    let n=params.width;let stride=n+4u;
    for(var k=0u;k<n;k++) {
        if lane==0u {
            var pivot=k;var largest=abs(value(condensed[k*stride+k]));
            for(var i=k+1u;i<n;i++) {
                let size=abs(value(condensed[i*stride+k]));
                if size>largest {largest=size;pivot=i;}
            }
            control.count=pivot;
            if !finite(largest) || largest<1e-20 {atomicStore(&condensed_failed,1u);}
        }
        if !condensed_ok() {return false;}
        let pivot=workgroupUniformLoad(&control.count);
        if pivot!=k && lane<stride {
            let old=condensed[k*stride+lane];condensed[k*stride+lane]=condensed[pivot*stride+lane];condensed[pivot*stride+lane]=old;
        }
        storageBarrier();
        if lane>k && lane<n {
            let ratio=fischer_div(condensed[lane*stride+k],condensed[k*stride+k]);
            for(var j=k+1u;j<stride;j++) {condensed[lane*stride+j]=rate_add(condensed[lane*stride+j],-fischer_product(ratio,condensed[k*stride+j]));}
            condensed[lane*stride+k]=vec2<f32>(0.0);
        }
        storageBarrier();
    }
    if lane<4u {
        for(var row=n;row>0u;row--) {
            let i=row-1u;var sum=condensed[i*stride+n+lane];
            for(var j=i+1u;j<n;j++) {sum=rate_add(sum,-fischer_product(condensed[i*stride+j],condensed[j*stride+n+lane]));}
            let x=fischer_div(sum,condensed[i*stride+i]);condensed[i*stride+n+lane]=x;
            if !all(abs(x)<=vec2<f32>(3.402823e38)) {atomicStore(&condensed_failed,1u);}
        }
    }
    storageBarrier();
    if lane<4u {
        var error=0.0;var scale=0.0;
        for(var i=0u;i<n;i++) {
            let rhs=condensed[1152u+i*stride+n+lane];var sum=-rhs;
            for(var j=0u;j<n;j++) {sum=rate_add(sum,fischer_product(condensed[1152u+i*stride+j],condensed[j*stride+n+lane]));}
            error+=value(sum)*value(sum);scale+=value(rhs)*value(rhs);
        }
        let relative=sqrt(error)/max(sqrt(scale),1e-30);
        if !finite(relative) || relative>params.tolerance {atomicStore(&condensed_failed,1u);}
    }
    return condensed_ok();
}

fn conditioned_prepare(chosen:u32)->bool {
    let n=params.width;let stride=n+4u;
    if lane==0u {
        atomicStore(&condensed_failed,0u);
        for(var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];
            if c!=chosen && condensed[4160u+54u*c+53u].x!=1.0 {atomicStore(&condensed_failed,1u);}
        }
    }
    if !condensed_ok() {return false;}
    if lane<n {
        var sums:array<vec2<f32>,36>;
        for(var j=0u;j<stride;j++) {sums[j]=vec2<f32>(select(0.0,1.0,j==lane),0.0);}
        for(var a=0u;a<3u;a++) {sums[n+1u+a]=vec2<f32>(response[chosen*n+lane][a],0.0);}
        for(var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];if c==chosen {continue;}
            let offset=4160u+54u*c;let r=response[c*n+lane];var projected:array<vec2<f32>,3>;
            for(var a=0u;a<3u;a++) {
                var sum=vec2<f32>(0.0);
                for(var b=0u;b<3u;b++) {sum=rate_add(sum,rate_mul(-r[b],condensed[offset+b*3u+a]));}
                projected[a]=sum;sums[n]=rate_add(sums[n],rate_mul(-r[a],condensed[offset+36u+a]));
            }
            for(var j=0u;j<n;j++) {for(var a=0u;a<3u;a++) {sums[j]=rate_add(sums[j],rate_mul(rows[c*n+j][a],projected[a]));}}
        }
        for(var j=0u;j<stride;j++) {
            condensed[lane*stride+j]=sums[j];condensed[1152u+lane*stride+j]=sums[j];
            if !all(abs(sums[j])<=vec2<f32>(3.402823e38)) {atomicStore(&condensed_failed,1u);}
        }
    }
    if lane<3u {condensed[4090u+lane]=vec2<f32>(states[chosen].high.k[lane],states[chosen].low.k[lane]);}
    storageBarrier();if !condensed_ok() {return false;}
    return conditioned_factor();
}

fn conditioned_delta(next:Six,i:u32)->vec2<f32> {
    let n=params.width;let offset=i*(n+4u)+n;var dq=condensed[offset];
    for(var a=0u;a<3u;a++) {
        let dk=rate_add(vec2<f32>(next.k[a],0.0),-condensed[4090u+a]);
        dq=rate_add(dq,-fischer_product(condensed[offset+1u+a],dk));
    }
    return dq;
}

fn conditioned_state(c:u32,chosen:u32,next:Six)->State {
    if c==chosen {return State(next,Six());}
    let n=params.width;let offset=4160u+54u*c;var point:array<vec2<f32>,3>;
    for(var i=0u;i<n;i++) {
        let dq=conditioned_delta(next,i);
        for(var a=0u;a<3u;a++) {point[a]=rate_add(point[a],rate_mul(rows[c*n+i][a],dq));}
    }
    let old=states[c];var high:array<f32,6>;var low:array<f32,6>;
    for(var j=0u;j<6u;j++) {
        var change=condensed[offset+36u+j];
        for(var a=0u;a<3u;a++) {change=rate_add(change,-fischer_product(condensed[offset+j*3u+a],point[a]));}
        var x=rate_add(vec2<f32>(component(old.high,j),component(old.low,j)),change);
        if j>=3u && flags(c)[j-3u]==0.0 {x=vec2<f32>(0.0);}
        high[j]=x.x;low[j]=x.y;
    }
    return State(packed(high),packed(low));
}

fn conditioned_release()->bool {
    if workgroupUniformLoad(&outer.releases)>=8u {return false;}
    conditioned_cache();
    if lane==0u {
        var scores:array<f32,8>;
        for(var i=0u;i<8u;i++) {release_queue[i]=0xffffffffu;}
        for(var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];var visited=false;
            for(var i=0u;i<outer.releases;i++) {visited=visited || release_visited[i]==c;}
            if visited {continue;}
            var score=0.0;for(var i=0u;i<6u;i++) {score+=blocks[c].residual[i]*blocks[c].residual[i];}
            for(var i=0u;i<8u;i++) {if score>scores[i] {var j=7u;loop {if j==i {break;}scores[j]=scores[j-1u];release_queue[j]=release_queue[j-1u];j--;}scores[i]=score;release_queue[i]=c;break;}}
        }
        release_best=ReleaseChoice(Six(),3.402823e38,0u);release_contact=0xffffffffu;conditioned_best=0u;
    }
    workgroupBarrier();
    for(var rank=0u;rank<8u;rank++) {
        let c=workgroupUniformLoad(&release_queue[rank]);if c==0xffffffffu {continue;}
        conditioned_model=conditioned_prepare(c);
        if lane<8u {release_choices[lane]=release_local(c,lane);}
        workgroupBarrier();
        if lane==0u {for(var i=0u;i<8u;i++) {
            let candidate=release_choices[i];
            if candidate.valid!=0u && candidate.score<release_best.score {
                release_best=candidate;release_contact=c;conditioned_best=select(0u,1u,conditioned_model);
            }
        }}
        workgroupBarrier();
    }
    let chosen=workgroupUniformLoad(&release_contact);if chosen==0xffffffffu {return false;}
    if workgroupUniformLoad(&conditioned_best)!=0u {
        if !conditioned_prepare(chosen) {return false;}
        for(var slot=lane;slot<status.active_count;slot+=64u) {
            let c=active_indices[slot];states[c]=conditioned_state(c,chosen,release_best.state);
        }
    } else if lane==0u {states[chosen]=State(release_best.state,Six());}
    if lane==0u {release_visited[outer.releases]=chosen;outer.releases++;}
    storageBarrier();return true;
}
