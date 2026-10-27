@group(0) @binding(0) var<storage,read> projected:array<vec2<f32>>;
@group(0) @binding(1) var<storage,read> soil_status:array<u32>;
@group(0) @binding(2) var<storage,read_write> rates:array<vec2<f32>>;
@group(0) @binding(3) var<storage,read_write> delta:array<vec2<f32>>;
@group(0) @binding(4) var<storage,read_write> machine_status:atomic<u32>;
@group(0) @binding(5) var<uniform> params:vec4<u32>;
var<workgroup> next_rates:array<vec2<f32>,32>;
var<workgroup> next_delta:array<vec2<f32>,32>;
var<workgroup> rounding:atomic<u32>;
fn rounded(v:f32)->f32 {
    atomicStore(&rounding,bitcast<u32>(v));
    return bitcast<f32>(atomicAdd(&rounding,0u));
}
fn two_sum(a:f32,b:f32)->vec2<f32> {
    let s=rounded(a+b); let vb=rounded(s-a); let va=rounded(s-vb);
    return vec2<f32>(s,rounded(rounded(a-va)+rounded(b-vb)));
}
fn add_pair(a:vec2<f32>,b:vec2<f32>)->vec2<f32> {
    let high=two_sum(a.x,b.x); let low=two_sum(a.y,b.y);
    let joined=two_sum(high.x,rounded(high.y+low.x));
    return two_sum(joined.x,rounded(joined.y+low.y));
}
fn finite_pair(v:vec2<f32>)->bool {
    return all(abs(v)<=vec2<f32>(3.402823e38)) && abs(v.x+v.y)<=3.402823e38;
}
@compute @workgroup_size(1)
fn publish() {
    if soil_status[0]!=0u { atomicOr(&machine_status,32u); }
    if atomicLoad(&machine_status)!=0u { return; }
    for (var i=0u;i<params.x;i++) {
        if !finite_pair(projected[i]) || !finite_pair(rates[i]) || !finite_pair(delta[i]) {
            atomicOr(&machine_status,32u); return;
        }
        let supplied=two_sum(projected[i].x,projected[i].y);
        var change=supplied;
        if params.y!=0u {
            next_rates[i]=add_pair(rates[i],supplied);
            next_delta[i]=supplied;
        } else {
            next_rates[i]=supplied;
            change=add_pair(next_rates[i],-rates[i]);
            next_delta[i]=add_pair(delta[i],change);
        }
        if !finite_pair(next_rates[i]) || !finite_pair(change) || !finite_pair(next_delta[i]) {
            atomicOr(&machine_status,32u); return;
        }
    }
    for (var i=0u;i<params.x;i++) { rates[i]=next_rates[i]; delta[i]=next_delta[i]; }
}
