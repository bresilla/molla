fn direct_contact(c:u32)->bool {
    var matrix:array<array<vec2<f32>,10>,6>;
    for(var i=0u;i<6u;i++) {
        for(var j=0u;j<6u;j++) {matrix[i][j]=blocks[c].direct[i*6u+j];}
        for(var a=0u;a<3u;a++) {matrix[i][6u+a]=blocks[c].machine[i*3u+a];}
        matrix[i][9u]=vec2<f32>(-blocks[c].residual[i],0.0);
    }
    for(var k=0u;k<6u;k++) {
        var pivot=k;var largest=abs(value(matrix[k][k]));
        for(var i=k+1u;i<6u;i++) {
            let size=abs(value(matrix[i][k]));
            if size>largest {largest=size;pivot=i;}
        }
        if !finite(largest) || largest<1e-20 {return false;}
        if pivot!=k {let row=matrix[k];matrix[k]=matrix[pivot];matrix[pivot]=row;}
        for(var i=k+1u;i<6u;i++) {
            let ratio=fischer_div(matrix[i][k],matrix[k][k]);
            for(var j=k+1u;j<10u;j++) {matrix[i][j]=rate_add(matrix[i][j],-fischer_product(ratio,matrix[k][j]));}
        }
    }
    let offset=4160u+54u*c;
    for(var column=6u;column<10u;column++) {
        for(var row=6u;row>0u;row--) {
            let i=row-1u;var sum=matrix[i][column];
            for(var j=i+1u;j<6u;j++) {sum=rate_add(sum,-fischer_product(matrix[i][j],matrix[j][column]));}
            let x=fischer_div(sum,matrix[i][i]);matrix[i][column]=x;
            if !all(abs(x)<=vec2<f32>(3.402823e38)) {return false;}
            if column==9u {condensed[offset+36u+i]=x;} else {condensed[offset+i*3u+column-6u]=x;}
        }
    }
    return true;
}
fn direct_core() {
    let n=params.width;let stride=n+1u;
    if lane<n {
        var sums:array<vec2<f32>,33>;
        for(var j=0u;j<stride;j++) {sums[j]=vec2<f32>(select(0.0,1.0,j==lane),0.0);}
        for(var slot=0u;slot<status.active_count;slot++) {
            let c=active_indices[slot];let offset=4160u+54u*c;let r=response[c*n+lane];
            var projected:array<vec2<f32>,3>;
            for(var a=0u;a<3u;a++) {
                var sum=vec2<f32>(0.0);
                for(var b=0u;b<3u;b++) {sum=rate_add(sum,rate_mul(-r[b],condensed[offset+b*3u+a]));}
                projected[a]=sum;
                sums[n]=rate_add(sums[n],rate_mul(r[a],condensed[offset+36u+a]));
            }
            for(var j=0u;j<n;j++) {
                let row=rows[c*n+j];
                for(var a=0u;a<3u;a++) {sums[j]=rate_add(sums[j],rate_mul(row[a],projected[a]));}
            }
        }
        for(var j=0u;j<stride;j++) {
            condensed[lane*stride+j]=sums[j];
            if !all(abs(sums[j])<=vec2<f32>(3.402823e38)) {atomicStore(&condensed_failed,1u);}
        }
    }
    storageBarrier();
}
fn direct_recover() {
    let n=params.width;let stride=n+1u;
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];let offset=4160u+54u*c;
        var point:array<vec2<f32>,3>;
        for(var a=0u;a<3u;a++) {
            var sum=vec2<f32>(0.0);
            for(var i=0u;i<n;i++) {sum=rate_add(sum,rate_mul(rows[c*n+i][a],condensed[i*stride+n]));}
            point[a]=sum;
        }
        var high:array<f32,6>;var low:array<f32,6>;
        for(var i=0u;i<6u;i++) {
            var sum=condensed[offset+36u+i];
            for(var a=0u;a<3u;a++) {sum=rate_add(sum,fischer_product(condensed[offset+i*3u+a],point[a]));}
            high[i]=sum.x;low[i]=sum.y;
            if !all(abs(sum)<=vec2<f32>(3.402823e38)) {atomicStore(&condensed_failed,1u);}
        }
        direction[c]=packed(high);basis[params.iterations*max(params.contacts,1u)+c].z=packed(low);
    }
    storageBarrier();
}
fn direct_direction()->bool {
    if lane==0u {atomicStore(&condensed_failed,0u);}
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];scratch[c]=packed(blocks[c].residual);
    }
    storageBarrier();let initial=scratch_norm();
    if lane==0u {
        refinement_norm=initial;control.beta=initial;status.beta=initial;
        control.stop=select(1u,0u,finite(initial) && initial>0.0);
        if initial==0.0 {status.relative=0.0;status.residual=0.0;}
    }
    if workgroupUniformLoad(&control.stop)!=0u {return initial==0.0;}
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        if !direct_contact(active_indices[slot]) {atomicStore(&condensed_failed,1u);}
    }
    storageBarrier();if !condensed_ok() {return false;}
    direct_core();if !condensed_ok() {return false;}
    if !condensed_factor(params.width) {return false;}
    direct_recover();if !condensed_ok() {return false;}
    let error=true_direction_error();
    if lane==0u {status.iterations=1u;}
    return error<=params.tolerance;
}

