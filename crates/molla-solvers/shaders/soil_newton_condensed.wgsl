struct CondensedInverse { a:array<vec2<f32>,36>, valid:u32 }
@group(0) @binding(21) var<storage,read_write> condensed:array<vec2<f32>>;
var<workgroup> condensed_failed:atomic<u32>;
var<workgroup> condensed_stop:u32;
fn prepare_condensed() {
    if lane==0u {
        atomicStore(&status.fault,0u);atomicStore(&status.fallbacks,0u);
        status.iterations=0u;status.relative=0.0;status.beta=0.0;status.residual=0.0;status.pad=0.0;
    }
    storageBarrier();
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];direction[c]=Six();
        direction[max(params.contacts,1u)+c]=Six();direction[2u*max(params.contacts,1u)+c]=Six();
        basis[params.iterations*max(params.contacts,1u)+c].z=Six();vector[c]=Six();scratch[c]=Six();
    }
    storageBarrier();
}
fn condensed_inverse(c:u32,damping:f32,transposed:bool)->CondensedInverse {
    var lower:array<vec2<f32>,36>;var result=CondensedInverse();
    for(var i=0u;i<6u;i++) {for(var j=0u;j<=i;j++) {
        var sum=vec2<f32>(select(0.0,damping,i==j),0.0);
        for(var k=0u;k<6u;k++) {
            let a=blocks[c].direct[select(k*6u+i,i*6u+k,transposed)];
            let b=blocks[c].direct[select(k*6u+j,j*6u+k,transposed)];
            sum=rate_add(sum,fischer_product(a,b));
        }
        for(var k=0u;k<j;k++) {sum=rate_add(sum,-fischer_product(lower[i*6u+k],lower[j*6u+k]));}
        if !all(abs(sum)<=vec2<f32>(3.402823e38)) {return result;}
        if i==j {
            if value(sum)<=0.0 {return result;}
            lower[i*6u+j]=fischer_sqrt(sum);
        } else {lower[i*6u+j]=fischer_div(sum,lower[j*6u+j]);}
    }}
    for(var column=0u;column<6u;column++) {
        var x:array<vec2<f32>,6>;
        for(var i=0u;i<6u;i++) {
            var sum=vec2<f32>(select(0.0,1.0,i==column),0.0);
            for(var j=0u;j<i;j++) {sum=rate_add(sum,-fischer_product(lower[i*6u+j],x[j]));}
            x[i]=fischer_div(sum,lower[i*6u+i]);
        }
        for(var row=6u;row>0u;row--) {
            let i=row-1u;var sum=x[i];
            for(var j=i+1u;j<6u;j++) {sum=rate_add(sum,-fischer_product(lower[j*6u+i],x[j]));}
            x[i]=fischer_div(sum,lower[i*6u+i]);
            if !all(abs(x[i])<=vec2<f32>(3.402823e38)) {return result;}
            result.a[i*6u+column]=x[i];
        }
    }
    result.valid=1u;return result;
}
fn condensed_cache(c:u32,damping:f32)->bool {
    let e=condensed_inverse(c,damping,false);
    let f=condensed_inverse(c,damping,true);
    if e.valid==0u || f.valid==0u {return false;}
    let offset=4160u+54u*c;
    var dtb:array<vec2<f32>,18>;var fim:array<vec2<f32>,18>;
    for(var i=0u;i<6u;i++) {for(var a=0u;a<3u;a++) {
        var d=vec2<f32>(0.0);var b=vec2<f32>(0.0);
        for(var k=0u;k<6u;k++) {
            d=rate_add(d,fischer_product(blocks[c].direct[k*6u+i],blocks[c].machine[k*3u+a]));
            b=rate_add(b,fischer_product(f.a[i*6u+k],blocks[c].machine[k*3u+a]));
        }
        dtb[i*3u+a]=d;fim[i*3u+a]=b;
    }}
    var base:array<vec2<f32>,6>;
    for(var i=0u;i<6u;i++) {
        var sum=vec2<f32>(0.0);
        for(var k=0u;k<6u;k++) {sum=rate_add(sum,rate_mul(-component(scratch[c],k),e.a[i*6u+k]));}
        base[i]=sum;condensed[offset+36u+i]=sum;
        for(var a=0u;a<3u;a++) {
            var g=vec2<f32>(0.0);
            for(var k=0u;k<6u;k++) {g=rate_add(g,fischer_product(e.a[i*6u+k],dtb[k*3u+a]));}
            condensed[offset+i*3u+a]=g;
            condensed[offset+18u+i*3u+a]=e.a[i*6u+a];
        }
    }
    for(var a=0u;a<3u;a++) {
        var rhs=vec2<f32>(0.0);
        for(var k=0u;k<6u;k++) {rhs=rate_add(rhs,fischer_product(dtb[k*3u+a],base[k]));}
        condensed[offset+51u+a]=rhs;
        for(var b=0u;b<3u;b++) {
            var sum=vec2<f32>(0.0);
            for(var k=0u;k<6u;k++) {sum=rate_add(sum,fischer_product(blocks[c].machine[k*3u+a],fim[k*3u+b]));}
            condensed[offset+42u+a*3u+b]=rate_mul(-damping,sum);
        }
    }
    for(var i=0u;i<54u;i++) {if !all(abs(condensed[offset+i])<=vec2<f32>(3.402823e38)) {return false;}}
    return true;
}
fn condensed_core() {
    let n=params.width;let width=2u*n;let stride=width+1u;
    for(var index=lane;index<width*stride;index+=64u) {
        let i=index/stride;let j=index%stride;
        var sum=vec2<f32>(select(0.0,1.0,(i<n && j==n+i) || (i>=n && j==i-n)),0.0);
        for(var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];let offset=4160u+54u*c;
            if j==width {
                if i<n {
                    let row=rows[c*n+i];
                    for(var a=0u;a<3u;a++) {sum=rate_add(sum,rate_mul(row[a],condensed[offset+51u+a]));}
                } else {
                    let r=response[c*n+i-n];
                    for(var a=0u;a<3u;a++) {sum=rate_add(sum,rate_mul(-r[a],condensed[offset+36u+a]));}
                }
            } else if i<n && j<n {
                let row=rows[c*n+i];let col=rows[c*n+j];
                for(var a=0u;a<3u;a++) {for(var b=0u;b<3u;b++) {
                    sum=rate_add(sum,rate_mul(row[a],rate_mul(col[b],condensed[offset+42u+a*3u+b])));
                }}
            } else if i>=n && j>=n {
                let row=response[c*n+i-n];let col=response[c*n+j-n];
                for(var a=0u;a<3u;a++) {for(var b=0u;b<3u;b++) {
                    sum=rate_add(sum,rate_mul(row[a],rate_mul(col[b],condensed[offset+18u+a*3u+b])));
                }}
            } else {
                let row=rows[c*n+select(j,i,i<n)];
                let col=response[c*n+select(i-n,j-n,i<n)];
                for(var a=0u;a<3u;a++) {for(var b=0u;b<3u;b++) {
                    sum=rate_add(sum,rate_mul(-row[a],rate_mul(col[b],condensed[offset+b*3u+a])));
                }}
            }
        }
        condensed[index]=sum;
        if !all(abs(sum)<=vec2<f32>(3.402823e38)) {atomicStore(&condensed_failed,1u);}
    }
    storageBarrier();
}
fn condensed_ok()->bool {
    workgroupBarrier();
    if lane==0u {condensed_stop=atomicLoad(&condensed_failed);}
    return workgroupUniformLoad(&condensed_stop)==0u;
}
fn condensed_factor(width:u32)->bool {
    let stride=width+1u;
    for(var k=0u;k<width;k++) {
        if lane==0u {
            var pivot=k;var largest=abs(value(condensed[k*stride+k]));
            for(var i=k+1u;i<width;i++) {
                let size=abs(value(condensed[i*stride+k]));
                if size>largest {largest=size;pivot=i;}
            }
            control.count=pivot;
            if !finite(largest) || largest<1e-20 {atomicStore(&condensed_failed,1u);}
        }
        if !condensed_ok() {return false;}
        let pivot=workgroupUniformLoad(&control.count);
        if pivot!=k {for(var j=lane;j<stride;j+=64u) {
            let old=condensed[k*stride+j];condensed[k*stride+j]=condensed[pivot*stride+j];condensed[pivot*stride+j]=old;
        }}
        storageBarrier();
        if lane>k && lane<width {
            let ratio=fischer_div(condensed[lane*stride+k],condensed[k*stride+k]);
            for(var j=k+1u;j<stride;j++) {
                condensed[lane*stride+j]=rate_add(condensed[lane*stride+j],-fischer_product(ratio,condensed[k*stride+j]));
            }
            condensed[lane*stride+k]=vec2<f32>(0.0);
        }
        storageBarrier();
    }
    if lane==0u {
        for(var row=width;row>0u;row--) {
            let i=row-1u;var sum=condensed[i*stride+width];
            for(var j=i+1u;j<width;j++) {sum=rate_add(sum,-fischer_product(condensed[i*stride+j],condensed[j*stride+width]));}
            let x=fischer_div(sum,condensed[i*stride+i]);condensed[i*stride+width]=x;
            if !all(abs(x)<=vec2<f32>(3.402823e38)) {atomicStore(&condensed_failed,1u);}
        }
    }
    storageBarrier();return condensed_ok();
}
fn condensed_recover() {
    let n=params.width;let width=2u*n;let stride=width+1u;
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];let offset=4160u+54u*c;
        var jeta:array<vec2<f32>,3>;var reta:array<vec2<f32>,3>;
        for(var a=0u;a<3u;a++) {
            var j=vec2<f32>(0.0);var r=vec2<f32>(0.0);
            for(var i=0u;i<n;i++) {
                j=rate_add(j,rate_mul(rows[c*n+i][a],condensed[i*stride+width]));
                r=rate_add(r,rate_mul(response[c*n+i][a],condensed[(n+i)*stride+width]));
            }
            jeta[a]=j;reta[a]=r;
        }
        var high:array<f32,6>;var low:array<f32,6>;
        for(var i=0u;i<6u;i++) {
            var sum=condensed[offset+36u+i];
            for(var a=0u;a<3u;a++) {
                sum=rate_add(sum,-fischer_product(condensed[offset+i*3u+a],jeta[a]));
                sum=rate_add(sum,fischer_product(condensed[offset+18u+i*3u+a],reta[a]));
            }
            if !all(abs(sum)<=vec2<f32>(3.402823e38)) {atomicStore(&condensed_failed,1u);}
            high[i]=sum.x;low[i]=sum.y;
        }
        direction[c]=packed(high);basis[params.iterations*max(params.contacts,1u)+c].z=packed(low);
    }
    storageBarrier();
}
fn condensed_direction(damping:f32)->bool {
    if lane==0u {
        atomicStore(&condensed_failed,select(1u,0u,finite(damping) && damping>0.0));
    }
    if !condensed_ok() {return false;}
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];vector[c]=packed(blocks[c].residual);
    }
    storageBarrier();apply_transpose();let scale=scratch_norm();
    if lane==0u {
        control.stop=select(1u,0u,finite(scale) && scale>0.0);
        if scale==0.0 {status.relative=0.0;status.residual=0.0;}
    }
    if workgroupUniformLoad(&control.stop)!=0u {return scale==0.0;}
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        if !condensed_cache(active_indices[slot],damping) {atomicStore(&condensed_failed,1u);}
    }
    storageBarrier();if !condensed_ok() {return false;}
    condensed_core();if !condensed_ok() {return false;}
    if !condensed_factor(2u*params.width) {return false;}
    condensed_recover();if !condensed_ok() {return false;}
    let error=regularized_direction_error(sqrt(damping),scale);
    if lane==0u {status.iterations=1u;}
    return error<=params.tolerance;
}

