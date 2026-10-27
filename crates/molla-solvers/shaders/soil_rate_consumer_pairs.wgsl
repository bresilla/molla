var<workgroup> rate_scratch:array<atomic<u32>,__LANES__>;
var<private> rate_lane:u32;
fn rate_round(v:f32)->f32 {
    atomicStore(&rate_scratch[rate_lane],bitcast<u32>(v));
    return bitcast<f32>(atomicAdd(&rate_scratch[rate_lane],0u));
}
fn rate_sum(a:f32,b:f32)->vec2<f32> {
    let s=rate_round(a+b); let vb=rate_round(s-a); let va=rate_round(s-vb);
    return vec2<f32>(s,rate_round(rate_round(a-va)+rate_round(b-vb)));
}
fn rate_add(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {
    let high=rate_sum(a.x,b.x); let low=rate_sum(a.y,b.y);
    let merged=rate_sum(high.x,rate_round(high.y+low.x));
    return rate_sum(merged.x,rate_round(merged.y+low.y));
}
fn rate_mul(a:f32,b:vec2<f32>)->vec2<f32> {
    let high=rate_round(a*b.x);
    return rate_sum(high,rate_round(fma(a,b.x,-high)+rate_round(a*b.y)));
}
