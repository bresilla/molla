struct MaterialResponse {
    energy: f32,
    gradient: mat3x3<f32>,
    hessian: array<f32, 81>,
    valid: u32,
}

struct MaterialProjection { h: array<f32, 81>, valid: u32 }

fn material_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn material_psd(input: array<f32, 81>, n: u32) -> MaterialProjection {
    var result: MaterialProjection;
    var a = input;
    var scale = 0.0;
    for (var i = 0u; i < 81u; i++) {
        if (!material_finite(a[i])) { return result; }
        scale = max(scale, abs(a[i]));
    }
    result.valid = 1u;
    if (scale == 0.0) { return result; }
    var v: array<f32, 81>;
    for (var i = 0u; i < n; i++) {
        v[9u*i+i] = 1.0;
        for (var k = 0u; k < n; k++) { a[9u*i+k] /= scale; }
    }
    for (var sweep = 0u; sweep < 32u; sweep++) {
        var largest = 0.0;
        for (var p = 0u; p < n; p++) {
            for (var q = p+1u; q < n; q++) {
                let off = a[9u*p+q];
                largest = max(largest, abs(off));
                if (abs(off) <= 1.0e-7) { continue; }
                let tau = (a[9u*q+q] - a[9u*p+p]) / (2.0*off);
                var t = 1.0;
                if (tau != 0.0) { t = sign(tau) / (abs(tau) + sqrt(1.0+tau*tau)); }
                let c = inverseSqrt(1.0+t*t);
                let s = t*c;
                a[9u*p+p] -= t*off;
                a[9u*q+q] += t*off;
                a[9u*p+q] = 0.0;
                a[9u*q+p] = 0.0;
                for (var k = 0u; k < n; k++) {
                    if (k != p && k != q) {
                        let ap = a[9u*k+p]; let aq = a[9u*k+q];
                        a[9u*k+p] = c*ap-s*aq; a[9u*p+k] = a[9u*k+p];
                        a[9u*k+q] = s*ap+c*aq; a[9u*q+k] = a[9u*k+q];
                    }
                    let vp = v[9u*k+p]; let vq = v[9u*k+q];
                    v[9u*k+p] = c*vp-s*vq; v[9u*k+q] = s*vp+c*vq;
                }
            }
        }
        if (largest <= 1.0e-7) { break; }
    }
    for (var i = 0u; i < n; i++) {
        for (var k = 0u; k < n; k++) {
            if (i != k && abs(a[9u*i+k]) > 2.0e-5) { result.valid = 0u; }
            for (var j = 0u; j < n; j++) { result.h[9u*i+k] += v[9u*i+j] * max(a[9u*j+j], 0.0) * v[9u*k+j] * scale; }
        }
    }
    return result;
}

fn material_baraff(f: mat3x3<f32>, mu: f32, lambda: f32, project: bool) -> MaterialResponse {
    var result: MaterialResponse;
    let a = f[0]; let b = f[1];
    let lengths = vec2<f32>(length(a), length(b));
    let shear = dot(a,b);
    result.energy = 0.5*mu*dot(lengths-vec2<f32>(1.0), lengths-vec2<f32>(1.0)) + 0.5*lambda*shear*shear;
    result.gradient[0] = a*(mu*(1.0-1.0/lengths[0])) + b*(lambda*shear);
    result.gradient[1] = b*(mu*(1.0-1.0/lengths[1])) + a*(lambda*shear);
    var shear_h: array<f32,81>;
    for (var c = 0u; c < 2u; c++) {
        let dir = f[c]/lengths[c];
        var t = 1.0-1.0/lengths[c];
        if (project) { t = max(t,0.0); }
        for (var i = 0u; i < 3u; i++) {
            for (var k = 0u; k < 3u; k++) {
                let identity = select(0.0,1.0,i==k);
                result.hessian[9u*(3u*c+i)+3u*c+k] = mu*(t*identity + (1.0-t)*dir[i]*dir[k]);
                shear_h[9u*(3u*c+i)+3u*c+k] = lambda*f[1u-c][i]*f[1u-c][k];
                shear_h[9u*i+3u+k] = lambda*(b[i]*a[k]+shear*identity);
                shear_h[9u*(3u+k)+i] = shear_h[9u*i+3u+k];
            }
        }
    }
    if (project) {
        let projected = material_psd(shear_h,6u);
        if (projected.valid == 0u) { return result; }
        shear_h = projected.h;
    }
    for (var i = 0u; i < 81u; i++) { result.hessian[i] += shear_h[i]; }
    result.valid = 1u;
    return result;
}

fn material_spectral(f: mat3x3<f32>, scale: f32, n: u32, model: u32, mu: f32, lambda: f32, project: bool) -> MaterialResponse {
    var result: MaterialResponse;
    var b = mat3x3<f32>(f[0]/scale,f[1]/scale,f[2]/scale);
    var v = mat3x3<f32>(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0);
    for (var sweep = 0u; sweep < 32u; sweep++) {
        var changed = false;
        for (var p = 0u; p < n; p++) {
            for (var q = p+1u; q < n; q++) {
                let bp = b[p]; let bq = b[q];
                let app = dot(bp,bp); let aqq = dot(bq,bq); let apq = dot(bp,bq);
                if (abs(apq) <= 1.0e-7*sqrt(app*aqq)) { continue; }
                let tau = (aqq-app)/(2.0*apq);
                var t = 1.0;
                if (tau != 0.0) { t = sign(tau)/(abs(tau)+sqrt(1.0+tau*tau)); }
                let c = inverseSqrt(1.0+t*t); let s = t*c;
                b[p] = c*bp-s*bq; b[q] = s*bp+c*bq;
                let vp = v[p]; let vq = v[q];
                v[p] = c*vp-s*vq; v[q] = s*vp+c*vq;
                changed = true;
            }
        }
        if (!changed) { break; }
    }
    var u: mat3x3<f32>;
    var sigma: vec3<f32>;
    for (var i = 0u; i < n; i++) {
        let length_b = length(b[i]);
        if (!(length_b > 0.0)) { return result; }
        u[i] = b[i]/length_b;
        sigma[i] = length_b*scale;
    }
    if (n == 2u) { u[2] = normalize(cross(u[0],u[1])); }
    for (var i = 0u; i < n; i++) {
        for (var k = i+1u; k < n; k++) { if (abs(dot(u[i],u[k])) > 2.0e-5) { return result; } }
    }
    var jacobian = 1.0;
    var trace = 0.0;
    for (var i = 0u; i < n; i++) { jacobian *= sigma[i]; trace += sigma[i]-1.0; }
    result.energy = 0.5*lambda*(jacobian-1.0)*(jacobian-1.0);
    if (model == 0u) { result.energy = 0.5*lambda*trace*trace; }
    var g: vec3<f32>;
    var normal: array<f32,81>;
    for (var i = 0u; i < n; i++) {
        var p = 1.0;
        for (var k = 0u; k < n; k++) { if (k != i) { p *= sigma[k]; } }
        let a = sigma[i]-1.0;
        if (model == 0u) {
            result.energy += mu*a*a;
            g[i] = 2.0*mu*a+lambda*trace;
            normal[9u*i+i] = 2.0*mu;
        } else if (model == 1u) {
            result.energy += 0.25*mu*a*a*a*a;
            g[i] = mu*a*a*a+lambda*(jacobian-1.0)*p;
            normal[9u*i+i] = 3.0*mu*a*a;
        } else {
            result.energy += 0.5*mu*(sigma[i]*sigma[i]-1.0);
            g[i] = mu*sigma[i]+(lambda*(jacobian-1.0)-mu)*p;
            normal[9u*i+i] = mu;
        }
        for (var k = 0u; k < n; k++) {
            var pk = 1.0;
            for (var l = 0u; l < n; l++) { if (l != k) { pk *= sigma[l]; } }
            normal[9u*i+k] += select(lambda*p*pk,lambda,model==0u);
            if (i != k && model != 0u) {
                var r = 1.0;
                for (var l = 0u; l < n; l++) { if (l != i && l != k) { r *= sigma[l]; } }
                normal[9u*i+k] += (lambda*(jacobian-1.0)-select(0.0,mu,model==2u))*r;
            }
        }
    }
    if (model == 2u) { result.energy -= mu*(jacobian-1.0); }
    if (project) {
        let projected = material_psd(normal,n);
        if (projected.valid == 0u) { return result; }
        normal = projected.h;
    }
    result.gradient = mat3x3<f32>(u[0]*g[0],u[1]*g[1],u[2]*g[2])*transpose(v);
    for (var basis = 0u; basis < 3u*n; basis++) {
        var d: mat3x3<f32>;
        d[basis/3u][basis%3u] = 1.0;
        let q = transpose(u)*d*v;
        var z: mat3x3<f32>;
        for (var i = 0u; i < n; i++) {
            for (var k = 0u; k < n; k++) { z[i][i] += normal[9u*i+k]*q[k][k]; }
            for (var k = i+1u; k < n; k++) {
                var r = 1.0;
                for (var l = 0u; l < n; l++) { if (l != i && l != k) { r *= sigma[l]; } }
                var symmetric = 2.0*mu;
                if (model == 1u) {
                    let a = sigma[i]-1.0; let c = sigma[k]-1.0;
                    symmetric = mu*(a*a+a*c+c*c)-lambda*(jacobian-1.0)*r;
                } else if (model == 2u) { symmetric = mu-(lambda*(jacobian-1.0)-mu)*r; }
                var skew = (g[i]+g[k])/(sigma[i]+sigma[k]);
                if (project) { symmetric = max(symmetric,0.0); skew = max(skew,0.0); }
                let sum = 0.5*symmetric*(q[k][i]+q[i][k]);
                let diff = 0.5*skew*(q[k][i]-q[i][k]);
                z[k][i] = sum+diff; z[i][k] = sum-diff;
            }
            if (n == 2u) {
                var out_of_plane = g[i]/sigma[i];
                if (project) { out_of_plane = max(out_of_plane,0.0); }
                z[i][2] = out_of_plane*q[i][2];
            }
        }
        let hd = u*z*transpose(v);
        for (var i = 0u; i < 3u*n; i++) { result.hessian[9u*i+basis] = hd[i/3u][i%3u]; }
    }
    result.valid = 1u;
    return result;
}

fn material_evaluate(deformation: mat3x3<f32>, n: u32, model: u32, mu: f32, lambda: f32, project: bool) -> MaterialResponse {
    var result: MaterialResponse;
    if ((n != 2u && n != 3u) || model > 3u || (model == 3u && n != 2u) || !material_finite(mu) || mu < 0.0 || !material_finite(lambda)) { return result; }
    var f = deformation;
    if (n == 2u) { f[2] = vec3<f32>(0.0); }
    var scale = 0.0;
    for (var c = 0u; c < n; c++) {
        for (var r = 0u; r < 3u; r++) {
            if (!material_finite(f[c][r])) { return result; }
            scale = max(scale,abs(f[c][r]));
        }
    }
    if (scale == 0.0) { return result; }
    let normalized = mat3x3<f32>(f[0]/scale,f[1]/scale,f[2]/scale);
    if ((n == 3u && determinant(normalized) <= 0.0) || (n == 2u && dot(cross(normalized[0],normalized[1]),cross(normalized[0],normalized[1])) == 0.0)) { return result; }
    if (model == 3u) { result = material_baraff(f,mu,lambda,project); }
    else { result = material_spectral(f,scale,n,model,mu,lambda,project); }
    if (!material_finite(result.energy)) { result.valid = 0u; }
    for (var c = 0u; c < n; c++) { for (var r = 0u; r < 3u; r++) { if (!material_finite(result.gradient[c][r])) { result.valid = 0u; } } }
    for (var i = 0u; i < 81u; i++) { if (!material_finite(result.hessian[i])) { result.valid = 0u; } }
    return result;
}
