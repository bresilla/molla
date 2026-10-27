var<private> nonlinear_cache_valid:bool=false;
var<private> nonlinear_cache_contact:u32;
var<private> nonlinear_cache_length:vec2<f32>;
var<private> nonlinear_cache_trace:vec2<f32>;
var<private> nonlinear_cache_normal:array<vec2<f32>,3>;
var<private> nonlinear_cache_rho:vec2<f32>;

fn nonlinear_constants(c:u32) {
    if nonlinear_cache_valid && nonlinear_cache_contact==c {return;}
    let node=contacts[c];let normal_length=fischer_length(node.normal_gap.xyz);var trace=vec2<f32>(0.0);
    for(var axis=0u;axis<3u;axis++) {
        nonlinear_cache_normal[axis]=fischer_div(vec2<f32>(node.normal_gap[axis],0.0),normal_length);
        for(var i=0u;i<params.width;i++) {
            trace=rate_add(trace,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][axis],0.0)));
        }
    }
    nonlinear_cache_length=normal_length;nonlinear_cache_trace=trace;
    let scale=1.0/node.soil_mass.w+value(trace);
    if finite(scale) && scale>0.0 {
        nonlinear_cache_rho=fischer_div(vec2<f32>(1.0,0.0),rate_add(fischer_div(vec2<f32>(1.0,0.0),vec2<f32>(node.soil_mass.w,0.0)),trace));
    }
    nonlinear_cache_contact=c;nonlinear_cache_valid=true;
}
