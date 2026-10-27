fn adjugate_inverse(c:u32,a:array<array<f32,9>,3>,ratio:array<f32,3>,diagonal:array<f32,3>)->bool {
    var scale=0.0;
    for(var i=0u;i<3u;i++) {for(var j=0u;j<3u;j++) {scale=max(scale,abs(a[i][j]));}}
    if !finite(scale) || scale==0.0 {return false;}
    var normalized:array<array<f32,3>,3>;var inverse:array<array<f32,3>,3>;
    for(var i=0u;i<3u;i++) {for(var j=0u;j<3u;j++) {normalized[i][j]=a[i][j]/scale;}}
    for(var i=0u;i<3u;i++) {for(var j=0u;j<3u;j++) {
        let r0=(j+1u)%3u;let r1=(j+2u)%3u;let c0=(i+1u)%3u;let c1=(i+2u)%3u;
        let p=rate_mul(normalized[r0][c0],vec2<f32>(normalized[r1][c1],0.0));
        let q=rate_mul(-normalized[r0][c1],vec2<f32>(normalized[r1][c0],0.0));
        inverse[i][j]=value(rate_add(p,q));
    }}
    var determinant=vec2<f32>(0.0);
    for(var i=0u;i<3u;i++) {determinant=rate_add(determinant,rate_mul(normalized[0][i],vec2<f32>(inverse[i][0],0.0)));}
    let det=value(determinant);
    if !finite(det) || abs(det)<1e-5 {return false;}
    for(var i=0u;i<3u;i++) {for(var j=0u;j<3u;j++) {
        inverse[i][j]=(inverse[i][j]/det)/scale;
        if !finite(inverse[i][j]) {return false;}
    }}
    for(var i=0u;i<3u;i++) {for(var j=0u;j<3u;j++) {
        let expected=select(0.0,1.0,i==j);var residual=vec2<f32>(-expected,0.0);var magnitude=abs(expected);
        for(var k=0u;k<3u;k++) {
            residual=rate_add(residual,rate_mul(a[i][k],vec2<f32>(inverse[k][j],0.0)));
            magnitude+=abs(a[i][k])*abs(inverse[k][j]);
        }
        let error=abs(value(residual))/max(1.0,magnitude);
        if !finite(error) || !finite(magnitude) || error>1e-6 {return false;}
    }}
    var result:array<f32,36>;
    for(var i=0u;i<3u;i++) {for(var j=0u;j<6u;j++) {
        var k=0.0;
        if j<3u {k=inverse[i][j];} else {
            var sum=vec2<f32>(0.0);
            for(var q=0u;q<3u;q++) {sum=rate_add(sum,rate_mul(inverse[i][q],vec2<f32>(a[q][j+3u],0.0)));}
            k=value(sum);
        }
        let rhs=select(0.0,1.0,j==i+3u)/diagonal[i];
        let w=value(rate_add(vec2<f32>(rhs,0.0),rate_mul(-ratio[i],vec2<f32>(k,0.0))));
        if !finite(k) || !finite(w) {return false;}
        result[6u*i+j]=k;result[6u*(i+3u)+j]=w;
    }}
    for(var i=0u;i<36u;i++) {inverses[c].a[i]=result[i];}
    return true;
}
