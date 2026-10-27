fn scalar_product_admitted(a:vec2<f32>,b:f32)->bool {
    let bits=bitcast<vec3<u32>>(vec3<f32>(a,b)) & vec3<u32>(0x7fffffffu);
    return all(bits<=vec3<u32>(0x4e800000u))
        && (bits.x==0u || bits.x>=0x2c000000u)
        && (bits.y==0u || bits.y>=0x17800000u)
        && (bits.z==0u || bits.z>=0x2c000000u);
}
fn guarded_scalar_product(a:vec2<f32>,b:f32)->vec2<f32> {
    if scalar_product_admitted(a,b) {return rate_mul(b,a);}
    return fischer_product(a,vec2<f32>(b,0.0));
}
fn operator_scalar_product(a:vec2<f32>,b:f32)->vec2<f32> {
    if params.pad2==0u {return fischer_product(a,vec2<f32>(b,0.0));}
    return guarded_scalar_product(a,b);
}
