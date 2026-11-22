fn strain_effective_maximum(strain: f32, shrink: vec2<f32>) -> f32 {
    if (!material_finite(strain) || !(strain > 0.0) || !material_finite(shrink.x) || !material_finite(shrink.y) || any(shrink <= vec2<f32>(0.0))) { return 0.0; }
    let maximum = (1.0+strain)/min(shrink.x,shrink.y);
    if (!material_finite(maximum) || !(maximum > 1.0)) { return 0.0; }
    return maximum;
}

fn strain_largest_stretch(f: mat3x3<f32>) -> f32 {
    var scale = 0.0;
    for (var c = 0u; c < 2u; c++) {
        for (var r = 0u; r < 3u; r++) {
            if (!material_finite(f[c][r])) { return -1.0; }
            scale = max(scale,abs(f[c][r]));
        }
    }
    if (scale == 0.0) { return 0.0; }
    let a = f[0]/scale; let b = f[1]/scale;
    let aa = dot(a,a); let bb = dot(b,b); let ab = dot(a,b);
    let largest = sqrt(0.5*(aa+bb+sqrt((aa-bb)*(aa-bb)+4.0*ab*ab)))*scale;
    if (!material_finite(largest)) { return -1.0; }
    return largest;
}

fn strain_inside(f: mat3x3<f32>, maximum: f32) -> bool {
    let largest = strain_largest_stretch(f);
    return largest >= 0.0 && largest < maximum;
}

fn strain_safe_maximum(maximum: f32) -> f32 {
    return maximum*(1.0-1.9073486328125e-6);
}

fn strain_step_bound(f: mat3x3<f32>, df: mat3x3<f32>, maximum: f32, max_t: f32) -> f32 {
    let limit = strain_safe_maximum(maximum);
    if (!strain_inside(f,limit)) { return 0.0; }
    if (strain_inside(f+df*max_t,limit)) { return max_t; }
    var lower = 0.0; var upper = max_t;
    for (var i = 0u; i < 32u; i++) {
        let middle = 0.5*(lower+upper);
        if (middle == lower || middle == upper) { break; }
        if (strain_inside(f+df*middle,limit)) { lower = middle; }
        else { upper = middle; }
    }
    return 0.9*lower;
}

fn strain_limit_energy(f: mat3x3<f32>, strain: f32, maximum: f32) -> MaterialEnergy {
    var result: MaterialEnergy;
    let scale = material_scale(mat3x3<f32>(f[0],f[1],vec3<f32>(0.0)),2u);
    if (scale == 0.0 || !(strain > 0.0) || !material_finite(strain) || !material_finite(maximum) || !(maximum > 1.0) || !strain_inside(f,maximum)) { return result; }
    let largest = strain_largest_stretch(f);
    let smallest = length(cross(f[0]/scale,f[1]/scale))/(largest/scale)*scale;
    let a = max(vec2<f32>(largest,smallest)-vec2<f32>(1.0),vec2<f32>(0.0));
    result.energy = (2.0/3.0)*(a.x*a.x*a.x+a.y*a.y*a.y)/strain;
    result.valid = select(0u,1u,material_finite(result.energy));
    return result;
}

fn strain_limit_evaluate(deformation: mat3x3<f32>, strain: f32, maximum: f32) -> MaterialResponse {
    var result: MaterialResponse;
    let f = mat3x3<f32>(deformation[0],deformation[1],vec3<f32>(0.0));
    let scale = material_scale(f,2u);
    if (scale == 0.0 || !(strain > 0.0) || !material_finite(strain) || !material_finite(maximum) || !(maximum > 1.0) || !strain_inside(f,maximum)) { return result; }
    let svd = material_svd(f,scale,2u);
    if (svd.valid == 0u) { return result; }
    let s = svd.sigma.xy;
    let a = max(s-vec2<f32>(1.0),vec2<f32>(0.0));
    let g = 2.0*a*a/strain;
    let normal = 4.0*a/strain;
    var symmetric = normal.x;
    if (all(a > vec2<f32>(0.0))) { symmetric = 2.0*(a.x+a.y)/strain; }
    else if (s.x != s.y) { symmetric = (g.x-g.y)/(s.x-s.y); }
    let skew = (g.x+g.y)/(s.x+s.y);
    result.energy = (2.0/3.0)*(a.x*a.x*a.x+a.y*a.y*a.y)/strain;
    result.gradient = mat3x3<f32>(svd.u[0]*g.x,svd.u[1]*g.y,vec3<f32>(0.0))*transpose(svd.v);
    result.minimum_stretch = min(s.x,s.y);
    for (var basis = 0u; basis < 6u; basis++) {
        var d: mat3x3<f32>; d[basis/3u][basis%3u] = 1.0;
        let q = transpose(svd.u)*d*svd.v;
        let sum = 0.5*symmetric*(q[1][0]+q[0][1]);
        let diff = 0.5*skew*(q[1][0]-q[0][1]);
        let z = mat3x3<f32>(
            vec3<f32>(normal.x*q[0][0],sum-diff,g.x/s.x*q[0][2]),
            vec3<f32>(sum+diff,normal.y*q[1][1],g.y/s.y*q[1][2]),
            vec3<f32>(0.0));
        let column = svd.u*z*transpose(svd.v);
        for (var r = 0u; r < 6u; r++) { result.hessian[9u*r+basis] = column[r/3u][r%3u]; }
    }
    result.valid = 1u;
    if (!material_finite(result.energy)) { result.valid = 0u; }
    for (var c = 0u; c < 2u; c++) { for (var r = 0u; r < 3u; r++) { if (!material_finite(result.gradient[c][r])) { result.valid = 0u; } } }
    for (var i = 0u; i < 81u; i++) { if (!material_finite(result.hessian[i])) { result.valid = 0u; } }
    return result;
}
