const D: u32 = __DOFS__u;
const STEP_BITS: u32 = __STEP__u;
const DAMPING_BITS: u32 = __DAMPING__u;
@group(0) @binding(0) var<storage, read> input_rates: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> acceleration: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> upstream_status: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_rates: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> scalar_rates: array<f32>;
@group(0) @binding(5) var<storage, read_write> paired_delta: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> scalar_delta: array<f32>;
@group(0) @binding(7) var<storage, read_write> status: atomic<u32>;
var<private> predicted: array<vec2<f32>, D>;
var<private> increments: array<vec2<f32>, D>;

fn finite(value: f32) -> bool { return (bitcast<u32>(value)&0x7f800000u)!=0x7f800000u; }
fn value(pair: vec2<f32>) -> f64 { return f64(pair.x)+f64(pair.y); }
fn split(v: f64) -> vec2<f32> { let high=f32(v); return vec2<f32>(high,f32(v-f64(high))); }

@compute @workgroup_size(1)
fn publish() {
    if atomicLoad(&status)!=0u { return; }
    if upstream_status[0]!=0u { atomicOr(&status,32u); return; }
    let dt=bitcast<f32>(STEP_BITS);
    let damp=max(1.0-bitcast<f32>(DAMPING_BITS)*dt,0.0);
    for (var i=0u; i<D; i++) {
        let previous=input_rates[i];
        if !finite(previous.x) || !finite(previous.y) || previous.x+previous.y!=previous.x {
            atomicOr(&status,32u); return;
        }
        predicted[i]=split((value(previous)+value(acceleration[i])*f64(dt))*f64(damp));
        increments[i]=split(value(predicted[i])-value(previous));
        if !finite(predicted[i].x) || !finite(predicted[i].y)
            || !finite(increments[i].x) || !finite(increments[i].y)
            || predicted[i].x+predicted[i].y!=predicted[i].x {
            atomicOr(&status,32u); return;
        }
    }
    for (var i=0u; i<D; i++) {
        output_rates[i]=predicted[i];
        scalar_rates[i]=predicted[i].x;
        paired_delta[i]=increments[i];
        scalar_delta[i]=increments[i].x;
    }
}
