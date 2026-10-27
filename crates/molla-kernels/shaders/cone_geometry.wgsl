struct ConeRequest {
    rotation: vec4<f32>, primary: vec4<f32>, transverse0: vec4<f32>, transverse1: vec4<f32>,
    angles: vec2<f32>, enabled: vec2<u32>,
}
struct ConeGeometry { swing: vec4<f32>, rows: array<vec4<f32>, 3>, info: vec4<u32> }

fn cone_finite(v: f32) -> bool { return abs(v) <= 3.402823e38; }
fn cone_geometry(request: ConeRequest) -> ConeGeometry {
    var result: ConeGeometry;
    result.info.z = 2u;
    let norm = dot(request.rotation, request.rotation);
    if !cone_finite(norm) || norm <= 1e-12 { return result; }
    let p = request.primary.xyz;
    let b0 = request.transverse0.xyz;
    let b1 = request.transverse1.xyz;
    if !cone_finite(dot(p,p)) || !cone_finite(dot(b0,b0)) || !cone_finite(dot(b1,b1)) { return result; }
    if abs(dot(p,p)-1.0) > 1e-4 || abs(dot(b0,b0)-1.0) > 1e-4 || abs(dot(b1,b1)-1.0) > 1e-4 ||
        length(cross(p,b0)-b1) > 1e-4 || abs(dot(p,b0)) > 1e-4 { return result; }
    var unrestricted = true;
    for (var i = 0u; i < 2u; i++) {
        if request.enabled[i] > 3u || request.enabled[i] == 2u || (request.enabled[i] != 0u &&
            (!cone_finite(request.angles[i]) || request.angles[i] < 0.0 || request.angles[i] > 3.141592654)) { return result; }
        if request.enabled[i] == 3u && request.angles[i] != 3.141592654 { return result; }
        unrestricted = unrestricted && (request.enabled[i] == 0u || request.enabled[i] == 3u);
    }
    let rotation = request.rotation*inverseSqrt(norm);
    let n = vec3<f32>(rotation.w*rotation.w+rotation.x*rotation.x-rotation.y*rotation.y-rotation.z*rotation.z,
        2.0*(rotation.x*rotation.y+rotation.w*rotation.z),
        2.0*(rotation.x*rotation.z-rotation.w*rotation.y));
    let c = clamp(n.x, -1.0, 1.0);
    let t = vec3<f32>(0.0,n.y,n.z);
    let s = length(t);
    let angle = atan2(s,c);
    result.swing.z = angle;
    if c < 0.0 && s <= 1e-6 {
        if unrestricted { result.swing.x = angle; result.info.z = 0u; return result; }
        var radius=request.angles;
        for (var i=0u;i<2u;i++) {
            if request.enabled[i]==0u { radius[i]=3.141592654; }
        }
        let selected=select(0u,1u,radius.y>radius.x);
        result.swing[selected]=angle;
        var count=0u;
        for (var i=0u;i<2u;i++) {
            if i!=selected && request.enabled[i]!=0u && request.angles[i]==0.0 {
                var transverse=vec3<f32>(0.0);transverse[1u+i]=1.0;
                let component=clamp(dot(n,transverse),-1.0,1.0);
                let error=asin(component);
                result.rows[count]=vec4<f32>(cross(n,transverse)*inverseSqrt(1.0-component*component),error);
                result.info.y|=1u<<count;
                count++;
                result.swing.w=max(result.swing.w,abs(error));
            }
        }
        if radius[selected]<3.141592654 {
            var transverse=vec3<f32>(0.0);transverse[1u+selected]=1.0;
            let radial=cross(vec3<f32>(1.0,0.0,0.0),transverse);
            let angular=normalize(radial-n*dot(radial,n));
            let error=angle-radius[selected];
            result.rows[count]=vec4<f32>(angular,error);
            if radius[selected]==0.0 { result.info.y|=1u<<count; }
            count++;
            result.swing.w=max(result.swing.w,max(error,0.0));
        }
        for (var i=0u;i<count;i++) {
            let v=result.rows[i];
            result.rows[i]=vec4<f32>(p*v.x+b0*v.y+b1*v.z,v.w);
        }
        result.info.x=count;
        result.info.z=0u;
        return result;
    }
    var k = 1.0;
    var q = -2.0/3.0;
    if angle < 1e-3 {
        let a2 = angle*angle;
        k += a2/6.0+7.0*a2*a2/360.0;
        q -= a2/5.0;
    } else { k = angle/s; q = (c*s-angle)/(s*s*s); }
    let swing = k*t.yz;
    result.swing = vec4<f32>(swing,angle,0.0);
    if unrestricted { result.info.z = 0u; return result; }
    var derivatives = array<vec3<f32>,2>(
        cross(n,k*vec3<f32>(0.0,1.0,0.0)+t.y*(q*t-vec3<f32>(1.0,0.0,0.0))),
        cross(n,k*vec3<f32>(0.0,0.0,1.0)+t.z*(q*t-vec3<f32>(1.0,0.0,0.0))));
    if angle >= 1e-3 {
        let e = t/s;
        let radial = vec3<f32>(0.0,-e.z,e.y);
        let azimuthal = cross(n,radial);
        derivatives[0] = e.y*radial-k*e.z*azimuthal;
        derivatives[1] = e.z*radial+k*e.y*azimuthal;
    }
    var count = 0u;
    var scale = 4.0;
    for (var i = 0u; i < 2u; i++) {
        if request.enabled[i] != 0u {
            if request.angles[i] == 0.0 {
                result.rows[count] = vec4<f32>(derivatives[i],swing[i]);
                result.info.y |= 1u << count;
                count++;
                result.swing.w = max(result.swing.w,abs(swing[i]));
            } else { scale = min(scale,request.angles[i]); }
        }
    }
    if scale < 4.0 {
        if scale*scale < 1.17549435e-38 { return result; }
        var row = vec4<f32>(0.0,0.0,0.0,-scale*scale);
        var squared = 0.0;
        var weights = vec2<f32>(0.0);
        for (var i = 0u; i < 2u; i++) {
            if request.enabled[i] != 0u && request.angles[i] > 0.0 {
                let ratio = scale/request.angles[i];
                let weight = ratio*ratio;
                if weight < 1.17549435e-38 { return result; }
                weights[i] = weight;
                row += vec4<f32>(2.0*weight*swing[i]*derivatives[i],weight*swing[i]*swing[i]);
                let value = swing[i]/request.angles[i];
                squared += value*value;
            }
        }
        if angle >= 1e-3 {
            let e = t/s;
            let radial = vec3<f32>(0.0,-e.z,e.y);
            row = vec4<f32>(2.0*angle*dot(weights,e.yz*e.yz)*radial +
                2.0*angle*k*(weights.y-weights.x)*e.y*e.z*cross(n,radial),row.w);
        }
        if squared > 1.0 { result.swing.w = max(result.swing.w,angle*(1.0-inverseSqrt(squared))); }
        result.rows[count] = row;
        count++;
    }
    let pole_admissible=request.enabled.x==0u || request.enabled.x==3u || request.enabled.y==0u || request.enabled.y==3u;
    if pole_admissible && 3.141592654-angle<result.swing.w {
        let error=3.141592654-angle;
        result.rows[0]=vec4<f32>(-cross(vec3<f32>(1.0,0.0,0.0),t/s),error);
        result.rows[1]=vec4<f32>(0.0);
        result.rows[2]=vec4<f32>(0.0);
        result.info.y=0u;
        result.swing.w=error;
        count=1u;
    }
    for (var i = 0u; i < count; i++) {
        let v = result.rows[i];
        result.rows[i] = vec4<f32>(p*v.x+b0*v.y+b1*v.z,v.w);
        for (var j = 0u; j < 4u; j++) { if !cone_finite(result.rows[i][j]) { return result; } }
    }
    result.info.x = count;
    result.info.z = 0u;
    return result;
}
