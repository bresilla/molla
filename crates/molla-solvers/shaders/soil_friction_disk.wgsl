struct Input { metric_cap:vec4<f32>, velocity_pad:vec4<f32> }
struct Output { impulse_velocity:vec4<f32>, diagnostics:vec4<f32>, status:vec4<u32> }
struct Params { count:u32, iterations:u32, pad0:u32, pad1:u32 }
@group(0) @binding(0) var<storage,read> inputs:array<Input>;
@group(0) @binding(1) var<storage,read_write> outputs:array<Output>;
@group(0) @binding(2) var<uniform> params:Params;
fn finite(v:f32)->bool { return abs(v)<=3.402823e38; }
fn norm(v:vec2<f32>)->f32 {
    let scale=max(abs(v.x),abs(v.y));
    if scale==0.0 { return 0.0; }
    return scale*length(v/scale);
}
fn product(w:vec3<f32>,p:vec2<f32>)->vec2<f32> {
    return vec2<f32>(dot(w.xy,p),dot(w.yz,p));
}
fn shifted(w:vec3<f32>,v:vec2<f32>,shift:f32)->vec2<f32> {
    let x=w.x+shift; let z=w.z+shift;
    let determinant=x*z-w.y*w.y;
    return vec2<f32>(-z*v.x+w.y*v.y,w.y*v.x-x*v.y)/determinant;
}
@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=id.x;
    if c>=params.count { return; }
    outputs[c]=Output(vec4<f32>(0.0),vec4<f32>(0.0),vec4<u32>(1u,0u,0u,0u));
    let input=inputs[c]; let w=input.metric_cap.xyz; let cap=input.metric_cap.w;
    let v=input.velocity_pad.xy;
    if !all(abs(input.metric_cap)<=vec4<f32>(3.402823e38))
        || !all(abs(input.velocity_pad)<=vec4<f32>(3.402823e38))
        || any(input.velocity_pad.zw!=vec2<f32>(0.0)) || cap<0.0 { return; }
    outputs[c].status.x=2u;
    let scale=max(w.x,w.z);
    if w.x<=0.0 || w.z<=0.0 { return; }
    let normalized=w/scale;
    if normalized.x*normalized.z-normalized.y*normalized.y<=0.0 { return; }
    if cap==0.0 || all(v==vec2<f32>(0.0)) {
        outputs[c]=Output(vec4<f32>(0.0,0.0,v),vec4<f32>(0.0),vec4<u32>(0u,0u,select(1u,0u,cap==0.0),0u));
        return;
    }
    outputs[c].status.x=4u;
    let velocity_scale=max(max(abs(v.x),abs(v.y)),cap*scale);
    if !finite(velocity_scale) || velocity_scale<=0.0 { return; }
    let a=normalized*((cap*scale)/velocity_scale); let b=v/velocity_scale;
    var x=vec2<f32>(0.0); var alpha=0.0; var mode=1u; var iterations=0u;
    var feasible=false;
    if a.x*a.z-a.y*a.y>0.0 {
        let free=shifted(a,b,0.0);
        if all(abs(free)<=vec2<f32>(3.402823e38)) && norm(free)<=1.0 { x=free; feasible=true; }
    }
    if !feasible {
        mode=2u;
        var low=0.0; var high=norm(b);
        for (var iteration=0u;iteration<params.iterations;iteration++) {
            let middle=0.5*low+0.5*high;
            let trial=shifted(a,b,middle);
            let radius=norm(trial);
            iterations=iteration+1u;
            if !finite(radius) || radius>1.0 { low=middle; }
            else { high=middle; }
            x=shifted(a,b,high); alpha=high;
            let r=norm(x);
            if finite(r) && r<=1.0 && 1.0-r<=1e-6 { feasible=true; break; }
        }
    }
    if !feasible { outputs[c].status=vec4<u32>(8u,iterations,mode,0u); return; }
    let p=cap*x; let change=product(w,p); let after=v+change;
    let energy=dot(p,v+0.5*change);
    let residual=norm(product(a,x)+b+alpha*x);
    let radius=norm(x);
    let scaled_energy=dot(x,b+0.5*product(a,x));
    if !all(abs(p)<=vec2<f32>(3.402823e38)) || !all(abs(after)<=vec2<f32>(3.402823e38))
        || !finite(energy) || !finite(residual) || !finite(radius) { return; }
    if radius>1.0+1e-6 || residual>1e-6*max(1.0,norm(b)) || scaled_energy>1e-6 {
        outputs[c].status=vec4<u32>(16u,iterations,mode,0u); return;
    }
    outputs[c]=Output(vec4<f32>(p,after),vec4<f32>(energy,alpha,residual,radius),vec4<u32>(0u,iterations,mode,0u));
}
