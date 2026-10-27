fn finite(x:f32)->bool { return abs(x)<=3.402823e38; }
fn rotate(q:vec4<f32>,v:vec3<f32>)->vec3<f32> { return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v); }
fn side(x:f32)->f32 { return select(1.0,-1.0,x<0.0); }
fn surface(p:vec3<f32>,shape:vec4<f32>)->vec4<f32> {
    if shape.w==1.0 {
        let radius=length(p);
        if radius==0.0 { return vec4<f32>(1.0,0.0,0.0,-shape.x); }
        return vec4<f32>(p/radius,radius-shape.x);
    }
    if shape.w==2.0 {
        let delta=abs(p)-shape.xyz;
        let outside=max(delta,vec3<f32>(0.0)); let distance=length(outside);
        if distance>0.0 { return vec4<f32>((outside/distance)*vec3<f32>(side(p.x),side(p.y),side(p.z)),distance); }
        var axis=0u;
        if delta.y>delta[axis] { axis=1u; }
        if delta.z>delta[axis] { axis=2u; }
        var normal=vec3<f32>(0.0); normal[axis]=side(p[axis]);
        return vec4<f32>(normal,delta[axis]);
    }
    let radius=length(p.xz);
    var radial=vec3<f32>(1.0,0.0,0.0);
    if radius>0.0 { radial=vec3<f32>(p.x/radius,0.0,p.z/radius); }
    let delta=vec2<f32>(radius-shape.x,abs(p.y)-shape.y);
    let outside=max(delta,vec2<f32>(0.0)); let distance=length(outside);
    let cap=vec3<f32>(0.0,side(p.y),0.0);
    if distance>0.0 { return vec4<f32>((outside.x*radial+outside.y*cap)/distance,distance); }
    if delta.x>=delta.y { return vec4<f32>(radial,delta.x); }
    return vec4<f32>(cap,delta.y);
}
