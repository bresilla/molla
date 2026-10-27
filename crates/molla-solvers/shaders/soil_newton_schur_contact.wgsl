fn schur_contact(c:u32)->bool {
    var ratios:array<vec2<f32>,3>;
    var wall_rhs:array<vec2<f32>,3>;
    for(var i=0u;i<3u;i++) {
        let row=3u+i;let diagonal=blocks[c].direct[row*6u+row];
        if !finite(value(diagonal)) || abs(value(diagonal))<1e-20 {return false;}
        for(var j=0u;j<6u;j++) {
            if j!=i && j!=row && any(blocks[c].direct[row*6u+j]!=vec2<f32>(0.0)) {return false;}
        }
        for(var a=0u;a<3u;a++) {if any(blocks[c].machine[row*3u+a]!=vec2<f32>(0.0)) {return false;}}
        ratios[i]=fischer_div(blocks[c].direct[row*6u+i],diagonal);
        wall_rhs[i]=fischer_div(vec2<f32>(-blocks[c].residual[row],0.0),diagonal);
    }
    var matrix:array<array<vec2<f32>,7>,3>;
    for(var i=0u;i<3u;i++) {
        var rhs=vec2<f32>(-blocks[c].residual[i],0.0);
        for(var j=0u;j<3u;j++) {
            let b=blocks[c].direct[i*6u+3u+j];
            matrix[i][j]=rate_add(blocks[c].direct[i*6u+j],-fischer_product(b,ratios[j]));
            rhs=rate_add(rhs,-fischer_product(b,wall_rhs[j]));
            matrix[i][3u+j]=blocks[c].machine[i*3u+j];
        }
        matrix[i][6u]=rhs;
    }
    for(var k=0u;k<3u;k++) {
        var pivot=k;var largest=abs(value(matrix[k][k]));
        for(var i=k+1u;i<3u;i++) {
            let size=abs(value(matrix[i][k]));
            if size>largest {largest=size;pivot=i;}
        }
        if !finite(largest) || largest<1e-20 {return false;}
        if pivot!=k {let row=matrix[k];matrix[k]=matrix[pivot];matrix[pivot]=row;}
        for(var i=k+1u;i<3u;i++) {
            let ratio=fischer_div(matrix[i][k],matrix[k][k]);
            for(var j=k+1u;j<7u;j++) {matrix[i][j]=rate_add(matrix[i][j],-fischer_product(ratio,matrix[k][j]));}
        }
    }
    let offset=4160u+54u*c;
    for(var column=3u;column<7u;column++) {
        for(var row=3u;row>0u;row--) {
            let i=row-1u;var sum=matrix[i][column];
            for(var j=i+1u;j<3u;j++) {sum=rate_add(sum,-fischer_product(matrix[i][j],matrix[j][column]));}
            let x=fischer_div(sum,matrix[i][i]);matrix[i][column]=x;
            if !all(abs(x)<=vec2<f32>(3.402823e38)) {return false;}
        }
        for(var i=0u;i<3u;i++) {
            let x=matrix[i][column];var rhs=vec2<f32>(0.0);
            if column==6u {rhs=wall_rhs[i];}
            let w=rate_add(rhs,-fischer_product(ratios[i],x));
            if !all(abs(w)<=vec2<f32>(3.402823e38)) {return false;}
            if column==6u {condensed[offset+36u+i]=x;condensed[offset+39u+i]=w;}
            else {condensed[offset+i*3u+column-3u]=x;condensed[offset+(i+3u)*3u+column-3u]=w;}
        }
    }
    return true;
}

fn direct_contact(c:u32)->bool {
    if schur_contact(c) {return true;}
    return dense_direct_contact(c);
}
