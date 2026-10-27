fn quad_dihedral_angle(x0: vec3<f32>, x1: vec3<f32>, x2: vec3<f32>, x3: vec3<f32>) -> f32 {
    let edge = x2 - x0;
    let a = cross(x1 - x0, edge);
    let b = cross(edge, x3 - x0);
    if dot(a,a) <= 1e-14 || dot(b,b) <= 1e-14 || dot(edge,edge) <= 1e-14 { return 0.0; }
    let n0 = normalize(a);
    let n1 = normalize(b);
    return abs(atan2(dot(cross(n0,n1),normalize(edge)), dot(n0,n1)));
}

fn quad_dihedral_gradients(x0: vec3<f32>, x1: vec3<f32>, x2: vec3<f32>, x3: vec3<f32>) -> array<vec3<f32>,4> {
    let u = x1-x0;
    let v = x2-x0;
    let w = x3-x0;
    let a = cross(u,v);
    let b = cross(v,w);
    let aa = dot(a,a);
    let bb = dot(b,b);
    let vv = dot(v,v);
    var g:array<vec3<f32>,4>;
    if aa <= 1e-14 || bb <= 1e-14 || vv <= 1e-14 { return g; }
    let n0 = a*inverseSqrt(aa);
    let n1 = b*inverseSqrt(bb);
    let e = v*inverseSqrt(vv);
    let orientation = sign(dot(cross(n0,n1),e));
    let ga = orientation*cross(n0,e)*inverseSqrt(aa);
    let gb = orientation*cross(e,n1)*inverseSqrt(bb);
    g[1] = cross(v,ga);
    g[2] = cross(ga,u)+cross(w,gb);
    g[3] = cross(gb,v);
    g[0] = -g[1]-g[2]-g[3];
    for(var i=0u;i<4u;i++) { g[i] = clamp(g[i],vec3<f32>(-1e4),vec3<f32>(1e4)); }
    return g;
}
