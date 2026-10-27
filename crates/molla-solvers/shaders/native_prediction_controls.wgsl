const D: u32=__DOFS__u;
const B: u32=__BODIES__u;
struct Load { linear: vec4<f32>, angular: vec4<f32> }
@group(0) @binding(0) var<storage,read> mode: array<u32>;
@group(0) @binding(1) var<storage,read> effort_target: array<f32>;
@group(0) @binding(2) var<storage,read> motor: array<u32>;
@group(0) @binding(3) var<storage,read> motor_cap: array<f32>;
@group(0) @binding(4) var<storage,read> force: array<f32>;
@group(0) @binding(5) var<storage,read> brake: array<f32>;
@group(0) @binding(6) var<storage,read> effort_cap: array<f32>;
@group(0) @binding(7) var<storage,read> state_load: array<Load>;
@group(0) @binding(8) var<storage,read> control_load: array<Load>;
@group(0) @binding(9) var<storage,read_write> status: atomic<u32>;
@group(0) @binding(10) var<storage,read> activation_mask: array<u32>;
@group(0) @binding(11) var<storage,read_write> drive: array<vec2<f32>>;
@group(0) @binding(12) var<storage,read_write> loads: array<Load>;
var<private> applied: array<f32,D>;
var<private> combined: array<Load,B>;
fn finite(v: f32) -> bool { return (bitcast<u32>(v)&0x7f800000u)!=0x7f800000u; }
fn finite3(v: vec3<f32>) -> bool { return finite(v.x)&&finite(v.y)&&finite(v.z); }
@compute @workgroup_size(1)
fn evaluate() {
    if atomicLoad(&status)!=0u || activation_mask[0]==0u { return; }
    if activation_mask[0]!=1u { atomicOr(&status,32u); return; }
    for (var d=0u; d<D; d++) {
        if (mode[d]!=0u && mode[d]!=4u) || motor[d]!=0u || brake[d]!=0.0 || !finite(force[d])
            || motor_cap[d]!=motor_cap[d] || motor_cap[d]<0.0
            || (effort_cap[d]==effort_cap[d] && effort_cap[d]<0.0) {
            atomicOr(&status,32u); return;
        }
        var direct=0.0;
        if mode[d]==4u {
            if !finite(effort_target[d]) { atomicOr(&status,32u); return; }
            direct=clamp(effort_target[d],-motor_cap[d],motor_cap[d]);
        }
        var effort=force[d]+direct;
        if effort_cap[d]==effort_cap[d] && effort_cap[d]<=1e30 {
            effort=clamp(effort,-effort_cap[d],effort_cap[d]);
        }
        if !finite(effort) { atomicOr(&status,32u); return; }
        applied[d]=effort;
    }
    for (var b=0u; b<B; b++) {
        let s=state_load[b]; let c=control_load[b];
        let linear=s.linear.xyz+c.linear.xyz; let angular=s.angular.xyz+c.angular.xyz;
        if !finite3(s.linear.xyz) || !finite3(s.angular.xyz) || !finite3(c.linear.xyz)
            || !finite3(c.angular.xyz) || !finite3(linear) || !finite3(angular) { atomicOr(&status,32u); return; }
        combined[b]=Load(vec4<f32>(linear,0.0),vec4<f32>(angular,0.0));
    }
    for (var d=0u; d<D; d++) { drive[d]=vec2<f32>(applied[d],0.0); }
    for (var b=0u; b<B; b++) { loads[b]=combined[b]; }
}
