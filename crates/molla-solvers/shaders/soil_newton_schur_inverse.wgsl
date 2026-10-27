fn schur_build_inverse(c:u32)->bool {
    var mobility:array<vec3<f32>,3>;
    for (var axis=0u;axis<3u;axis++) { for (var j=0u;j<3u;j++) {
        var sum=vec2<f32>(0.0);
        for (var i=0u;i<params.width;i++) { sum=rate_add(sum,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][j],0.0))); }
        mobility[axis][j]=value(sum);
    } }
    var upper:array<f32,18>;var lower:array<f32,3>;var diagonal:array<f32,3>;
    for (var i=0u;i<6u;i++) { for (var j=0u;j<6u;j++) {
        var sum=blocks[c].direct[i*6u+j];
        if j<3u { for (var axis=0u;axis<3u;axis++) { sum=rate_add(sum,fischer_product(-blocks[c].machine[i*3u+axis],vec2<f32>(mobility[axis][j],0.0))); } }
        let x=value(sum);
        if !finite(x) {atomicOr(&status.fault,2u);return false;}
        if i<3u {upper[i*6u+j]=x;}
        else if j==i {diagonal[i-3u]=x;}
        else if j==i-3u {lower[i-3u]=x;}
        else if x!=0.0 {return false;}
    } }
    var ratio:array<f32,3>;
    for(var i=0u;i<3u;i++) {
        if lower[i]!=0.0 {return false;}
        if abs(diagonal[i])<1e-20 {return false;}
        ratio[i]=lower[i]/diagonal[i];
        if !finite(ratio[i]) {return false;}
    }
    var matrix:array<array<f32,9>,3>;
    for(var i=0u;i<3u;i++) {for(var j=0u;j<3u;j++) {
        let b=upper[6u*i+3u+j];
        matrix[i][j]=value(rate_add(vec2<f32>(upper[6u*i+j],0.0),rate_mul(-b,vec2<f32>(ratio[j],0.0))));
        matrix[i][3u+j]=select(0.0,1.0,i==j);
        matrix[i][6u+j]=-b/diagonal[j];
    }}
    for(var k=0u;k<3u;k++) {
        var pivot=k;var largest=abs(matrix[k][k]);
        for(var i=k+1u;i<3u;i++) {if abs(matrix[i][k])>largest {largest=abs(matrix[i][k]);pivot=i;}}
        if !finite(largest) || largest<1e-20 {return false;}
        if pivot!=k {let row=matrix[k];matrix[k]=matrix[pivot];matrix[pivot]=row;}
        let divisor=matrix[k][k];
        for(var j=0u;j<9u;j++) {matrix[k][j]/=divisor;}
        for(var i=0u;i<3u;i++) {if i!=k {
            let r=matrix[i][k];
            for(var j=0u;j<9u;j++) {matrix[i][j]=value(rate_add(vec2<f32>(matrix[i][j],0.0),rate_mul(-r,vec2<f32>(matrix[k][j],0.0))));}
        }}
    }
    for(var i=0u;i<3u;i++) {for(var j=0u;j<6u;j++) {
        let k=matrix[i][3u+j];
        let rhs=select(0.0,1.0,j==i+3u)/diagonal[i];
        let w=value(rate_add(vec2<f32>(rhs,0.0),rate_mul(-ratio[i],vec2<f32>(k,0.0))));
        if !finite(k) || !finite(w) {return false;}
        inverses[c].a[6u*i+j]=k;inverses[c].a[6u*(i+3u)+j]=w;
    }}
    return true;
}

fn build_inverse(c:u32)->bool {
    if schur_build_inverse(c) {return true;}
    return dense_build_inverse(c);
}
