var<workgroup> rate_scratch:array<atomic<u32>,__LANES__>;
var<private> rate_lane:u32;
fn rate_round(v:f32)->f32 {
    atomicStore(&rate_scratch[rate_lane],bitcast<u32>(v));
    return bitcast<f32>(atomicAdd(&rate_scratch[rate_lane],0u));
}
fn rate_sum(a:f32,b:f32)->vec2<f32> {
    let s=f32(f64(a)+f64(b));
    var large=f64(a);var small=f64(b);
    if abs(b)>abs(a) {large=f64(b);small=f64(a);}
    return vec2<f32>(s,f32((large-f64(s))+small));
}
fn rate_add(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {
    let high=rate_sum(a.x,b.x);let low=rate_sum(a.y,b.y);
    let merged=rate_sum(high.x,f32(f64(high.y)+f64(low.x)));
    return rate_sum(merged.x,f32(f64(merged.y)+f64(low.y)));
}
fn rate_mul(a:f32,b:vec2<f32>)->vec2<f32> {
    let product=f64(a)*f64(b.x);let high=f32(product);
    let error=f32(product-f64(high));let low=f32(f64(a)*f64(b.y));
    return rate_sum(high,f32(f64(error)+f64(low)));
}
