fn precondition(c:u32,x:Six)->Six {
    var result:array<f32,6>;
    for(var i=0u;i<6u;i++) {
        var sum=0.0;
        for(var j=0u;j<6u;j++) {sum=fma(inverses[c].a[i*6u+j],component(x,j),sum);}
        result[i]=sum;
        if !finite(sum) {atomicOr(&status.fault,2u);}
    }
    return packed(result);
}
