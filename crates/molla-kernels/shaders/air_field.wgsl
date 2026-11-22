// MAC air field sampling shared by the air solver and coupled kernels.
// Users declare `cfg: Settings` and `velocity: array<vec4<f32>>` bindings.
struct Settings {
    cells: vec4<u32>,
    origin_h: vec4<f32>,
    physics: vec4<f32>,
    ambient: vec4<f32>,
    gusts: vec4<f32>,
    // Uniform velocity increment of the whole air mass during this step.
    drive: vec4<f32>,
}
// Fixed-point unit of reactions deposited by other kernels, in newtons (2^-24).
const REACTION_QUANTUM: f32 = 5.9604645e-8;

fn index(p: vec3<u32>) -> u32 {
    return (p.z * (cfg.cells.y + 1u) + p.y) * (cfg.cells.x + 1u) + p.x;
}
fn valid(p: vec3<u32>, axis: u32) -> bool {
    var n = cfg.cells.xyz;
    n[axis] += 1u;
    return all(p < n);
}
fn offset(axis: u32) -> vec3<f32> {
    var o = vec3<f32>(0.5);
    o[axis] = 0.0;
    return o;
}
fn outside_weight(grid_point: vec3<f32>) -> f32 {
    let beyond = max(grid_point - vec3<f32>(cfg.cells.xyz), vec3<f32>(-grid_point.x, 0.0, -grid_point.z));
    return clamp(max(beyond.x, max(beyond.y, beyond.z)), 0.0, 1.0);
}
fn background(point: vec3<f32>) -> vec3<f32> {
    if cfg.gusts.x==0.0 {return cfg.ambient.xyz;}
    let phase=0.9*point.x-0.55*cfg.gusts.y;
    return cfg.ambient.xyz+cfg.gusts.x*vec3<f32>(
        0.55*sin(0.7*point.z-0.4*cfg.gusts.y)+0.54*cos(1.2*point.y)*sin(phase),
        -0.405*sin(1.2*point.y)*cos(phase),
        0.45*sin(0.8*point.x+0.31*cfg.gusts.y));
}
fn sample_component(grid_point: vec3<f32>, axis: u32) -> f32 {
    var limit = vec3<f32>(cfg.cells.xyz) - vec3<f32>(1.0);
    limit[axis] += 1.0;
    let s = clamp(grid_point - offset(axis), vec3<f32>(0.0), limit);
    let base = min(floor(s), limit - vec3<f32>(1.0));
    let f = s - base;
    let i = index(vec3<u32>(base));
    let sy = cfg.cells.x + 1u;
    let sz = sy * (cfg.cells.y + 1u);
    let lo = mix(mix(velocity[i][axis], velocity[i + 1u][axis], f.x),
        mix(velocity[i + sy][axis], velocity[i + sy + 1u][axis], f.x), f.y);
    let hi = mix(mix(velocity[i + sz][axis], velocity[i + sz + 1u][axis], f.x),
        mix(velocity[i + sz + sy][axis], velocity[i + sz + sy + 1u][axis], f.x), f.y);
    return mix(lo, hi, f.z);
}
fn sample_air(point: vec3<f32>) -> vec3<f32> {
    let p = (point - cfg.origin_h.xyz) / cfg.origin_h.w;
    let outside = outside_weight(p);
    if outside >= 1.0 { return background(point); }
    let value = vec3<f32>(sample_component(p, 0u), sample_component(p, 1u), sample_component(p, 2u));
    return mix(value, background(point), outside);
}
fn sample_axis(point: vec3<f32>, axis: u32) -> f32 {
    let p = (point - cfg.origin_h.xyz) / cfg.origin_h.w;
    let outside = outside_weight(p);
    if outside >= 1.0 { return background(point)[axis]; }
    return mix(sample_component(p, axis), background(point)[axis], outside);
}
fn sample_exterior(point: vec3<f32>) -> vec3<f32> {
    let edge = clamp(point, cfg.origin_h.xyz, cfg.origin_h.xyz + vec3<f32>(cfg.cells.xyz) * cfg.origin_h.w);
    if cfg.gusts.z == 0.0 || all(edge == point) { return sample_air(point); }
    let delta = (point - edge) / cfg.gusts.z;
    return background(point) + (sample_air(edge) - background(edge)) * exp(-dot(delta, delta));
}
fn turbulence(point: vec3<f32>) -> vec3<f32> {
    if cfg.gusts.w == 0.0 { return vec3<f32>(0.0); }
    let waves = array<vec3<f32>,6>(vec3<f32>(0.7,1.7,-1.1),vec3<f32>(-1.9,3.9,1.3),
        vec3<f32>(3.7,7.0,2.9),vec3<f32>(-7.1,13.0,-4.3),vec3<f32>(12.9,22.0,-9.7),vec3<f32>(-18.3,34.0,21.1));
    let axes = array<vec3<f32>,6>(vec3<f32>(0.8,0.6,-0.7),vec3<f32>(-0.6,0.9,0.8),
        vec3<f32>(0.9,-0.7,0.5),vec3<f32>(0.7,0.8,-0.9),vec3<f32>(-0.8,0.6,0.9),vec3<f32>(0.6,-0.9,-0.7));
    let signed_height = point.y-cfg.origin_h.y;
    let height = abs(signed_height);
    let envelope = exp(-height/1.4);
    var velocity = vec3<f32>(0.0);
    for (var i=0u;i<6u;i++) {
        let k=waves[i]; let a=axes[i]; let index=f32(i);
        let phase=k.x*point.x+k.z*point.z-(0.6+0.73*index)*cfg.gusts.y
            +1.71*index+0.6*sin(0.17*cfg.gusts.y+index);
        let s=sin(phase); let c=cos(phase); let sy=sin(k.y*height); let cy=cos(k.y*height);
        let dy=k.y*cy-sy/1.4;
        let scale=envelope*0.9/((1.0+0.4*index)*length(k));
        velocity+=vec3<f32>(a.z*c*dy+a.y*k.z*s*cy,sy*(a.x*k.z*c+a.z*k.x*s),
            -a.y*k.x*s*cy-a.x*s*dy)*scale;
    }
    velocity.y*=select(1.0,-1.0,signed_height<0.0);
    return velocity*cfg.gusts.w;
}
