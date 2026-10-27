fn impulse_pair(axis:u32,first:u32,last:u32,x:array<f32,6>,tail:array<f32,6>)->vec2<f32> {
    if first==3u && last==6u
        && all(abs(vec3<f32>(x[3],x[4],x[5]))<=vec3<f32>(3.402823e38))
        && all(abs(vec3<f32>(tail[3],tail[4],tail[5]))<=vec3<f32>(3.402823e38)) {
        let sign=axes[3u+axis][axis];
        if sign==0.0 { return vec2<f32>(0.0); }
        return two_sum(rounded(0.0+sign*x[3u+axis]),rounded(0.0+sign*tail[3u+axis]));
    }
    var value=vec2<f32>(0.0);
    for (var j=first;j<last;j++) { value=add_product(value,axes[j][axis],x[j],tail[j]); }
    return value;
}
