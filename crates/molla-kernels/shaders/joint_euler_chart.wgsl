struct SingularXYZ { angles: vec3<f32>, phase: vec4<f32> }

fn xyz_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w*b.xyz+b.w*a.xyz+cross(a.xyz,b.xyz),a.w*b.w-dot(a.xyz,b.xyz));
}
fn xyz_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t=2.0*cross(q.xyz,v); return v+q.w*t+cross(q.xyz,t);
}
struct XYZPairFrames { left: vec4<f32>, right: vec4<f32> }
fn xyz_pair_frames(lower: vec3<f32>, free: u32) -> XYZPairFrames {
    var values=lower; values[free]=0.0;
    let rx=vec4<f32>(sin(values.x*0.5),0.0,0.0,cos(values.x*0.5));
    let ry=vec4<f32>(0.0,sin(values.y*0.5),0.0,cos(values.y*0.5));
    let rz=vec4<f32>(0.0,0.0,sin(values.z*0.5),cos(values.z*0.5));
    var left=vec4<f32>(0.0,0.0,0.0,1.0);
    var right=left;
    if free==0u { right=xyz_mul(ry,rz); }
    else if free==1u { left=rx; right=rz; }
    else { left=xyz_mul(rx,ry); }
    return XYZPairFrames(left,right);
}
fn xyz_pair_alignment(q: vec4<f32>, lower: vec3<f32>, free: u32) -> ConeGeometry {
    let frames=xyz_pair_frames(lower,free); let left=frames.left; let right=frames.right;
    let relative=xyz_mul(xyz_mul(vec4<f32>(-left.xyz,left.w),normalize(q)),vec4<f32>(-right.xyz,right.w));
    return xyz_hinge_alignment(relative,left,free);
}
fn xyz_hinge_alignment(relative: vec4<f32>, left: vec4<f32>, free: u32) -> ConeGeometry {
    var p=vec4<f32>(0.0); p[free]=1.0;
    var b0=vec4<f32>(0.0); b0[(free+1u)%3u]=1.0;
    var b1=vec4<f32>(0.0); b1[(free+2u)%3u]=1.0;
    let canonical=vec4<f32>(dot(relative.xyz,p.xyz),dot(relative.xyz,b0.xyz),dot(relative.xyz,b1.xyz),relative.w);
    var geometry=cone_geometry(ConeRequest(canonical,p,b0,b1,vec2<f32>(0.0),vec2<u32>(1u)));
    for (var row=0u; row<geometry.info.x; row++) { geometry.rows[row]=vec4<f32>(xyz_rotate(left,geometry.rows[row].xyz),geometry.rows[row].w); }
    return geometry;
}

fn xyz_pair_coordinate(q: vec4<f32>, lower: vec3<f32>, upper: vec3<f32>, free: u32) -> vec4<f32> {
    let frames=xyz_pair_frames(lower,free);
    let relative=normalize(xyz_mul(xyz_mul(vec4<f32>(-frames.left.xyz,frames.left.w),normalize(q)),vec4<f32>(-frames.right.xyz,frames.right.w)));
    return xyz_twist_coordinate(relative,frames.left,lower,upper,free);
}
fn xyz_twist_projection(relative: vec4<f32>, free: u32) -> vec2<f32> {
    if relative.w*relative.w+relative[free]*relative[free]<=1e-12 {
        return vec2<f32>(relative[(free+1u)%3u],relative[(free+2u)%3u]);
    }
    return vec2<f32>(relative[free],relative.w);
}
fn xyz_twist_coordinate(relative: vec4<f32>, left: vec4<f32>, lower: vec3<f32>, upper: vec3<f32>, free: u32) -> vec4<f32> {
    var primary=vec3<f32>(0.0); primary[free]=1.0;
    let component=relative[free];
    let denominator=relative.w*relative.w+component*component;
    var gradient=vec3<f32>(0.0);
    if denominator>1e-12 {
        gradient=((primary*relative.w+cross(relative.xyz,primary))*relative.w+relative.xyz*component)/denominator;
    } else {
        var b0=vec3<f32>(0.0); b0[(free+1u)%3u]=1.0;
        var b1=vec3<f32>(0.0); b1[(free+2u)%3u]=1.0;
        let sine=dot(relative.xyz,b0); let cosine=dot(relative.xyz,b1);
        gradient=(cosine*(b0*relative.w+cross(relative.xyz,b0))-sine*(b1*relative.w+cross(relative.xyz,b1)))/(sine*sine+cosine*cosine);
    }
    let projection=xyz_twist_projection(relative,free);
    let angle=xyz_interval_angle(2.0*atan2(projection.x,projection.y),lower[free],upper[free]);
    return vec4<f32>(xyz_rotate(left,gradient),angle);
}

fn xyz_point_plane(rotation: vec4<f32>, axis: u32, bound: f32) -> vec4<f32> {
    let q=normalize(rotation);
    var local=vec3<f32>(0.0,0.0,1.0);
    var normal=vec3<f32>(0.0,cos(bound),sin(bound));
    var reference=vec3<f32>(1.0,0.0,0.0);
    if axis==2u { local=vec3<f32>(sin(bound),cos(bound),0.0); normal=vec3<f32>(1.0,0.0,0.0); reference=vec3<f32>(0.0,1.0,0.0); }
    let t=2.0*cross(q.xyz,local);
    let direction=local+q.w*t+cross(q.xyz,t);
    return xyz_plane_geometry(direction,normal,reference);
}
fn xyz_plane_geometry(direction: vec3<f32>, normal: vec3<f32>, reference: vec3<f32>) -> vec4<f32> {
    let component=clamp(dot(direction,normal),-1.0,1.0);
    let c=cross(direction,normal);
    let magnitude=length(c);
    var gradient=-sign(component)*normalize(cross(direction,reference));
    if magnitude>1e-6 { gradient=c/magnitude; }
    return vec4<f32>(gradient,asin(component));
}
fn xyz_chart_finite(value: f32) -> bool { return abs(value)<=3.4028234e38; }
fn xyz_interval_angle(angle: f32, lower: f32, upper: f32) -> f32 {
    let tau=6.283185307179586;
    var turns=0.0;
    if xyz_chart_finite(lower) && xyz_chart_finite(upper) {
        turns=floor(((0.5*lower+0.5*upper)-angle)/tau+0.5);
    } else if xyz_chart_finite(lower) {
        turns=max(ceil((lower-angle)/tau),0.0);
    } else if xyz_chart_finite(upper) {
        turns=min(floor((upper-angle)/tau),0.0);
    }
    return angle+tau*turns;
}
fn xyz_regular_chart(angles: vec3<f32>, authored_lower: vec3<f32>, authored_upper: vec3<f32>) -> vec4<f32> {
    var lower=authored_lower;
    var upper=authored_upper;
    var primary=angles;
    var alternate=vec3<f32>(angles.x+3.141592653589793,3.141592653589793-angles.y,angles.z+3.141592653589793);
    for (var axis=0u;axis<3u;axis++) {
        if lower[axis]!=lower[axis] { lower[axis]=bitcast<f32>(0xff800000u); }
        if upper[axis]!=upper[axis] { upper[axis]=bitcast<f32>(0x7f800000u); }
        primary[axis]=xyz_interval_angle(primary[axis],lower[axis],upper[axis]);
        alternate[axis]=xyz_interval_angle(alternate[axis],lower[axis],upper[axis]);
    }
    let primary_gap=max(max(lower-primary,primary-upper),vec3<f32>(0.0));
    let alternate_gap=max(max(lower-alternate,alternate-upper),vec3<f32>(0.0));
    if max(max(alternate_gap.x,alternate_gap.y),alternate_gap.z)<max(max(primary_gap.x,primary_gap.y),primary_gap.z) {
        return vec4<f32>(alternate,-1.0);
    }
    return vec4<f32>(primary,1.0);
}
fn xyz_singular_chart(value: f32, y: f32, authored: vec4<f32>) -> SingularXYZ {
    var bounds=authored;
    for (var axis=0u;axis<2u;axis++) {
        if bounds[2u*axis]!=bounds[2u*axis] { bounds[2u*axis]=bitcast<f32>(0xff800000u); }
        if bounds[2u*axis+1u]!=bounds[2u*axis+1u] { bounds[2u*axis+1u]=bitcast<f32>(0x7f800000u); }
    }
    let sign=select(-1.0,1.0,y>=0.0);
    let zlo=select(-bounds.w,bounds.z,sign>0.0);
    let zhi=select(-bounds.z,bounds.w,sign>0.0);
    let lo=bounds.x+zlo;
    let hi=bounds.y+zhi;
    let tau=6.283185307179586;
    var phase=value;
    if xyz_chart_finite(lo) && xyz_chart_finite(hi) {
        phase+=tau*floor(((0.5*lo+0.5*hi)-phase)/tau+0.5);
    } else if xyz_chart_finite(lo) {
        phase+=tau*ceil((lo-phase)/tau);
    } else if xyz_chart_finite(hi) {
        phase+=tau*floor((hi-phase)/tau);
    }
    let bounded_phase=clamp(phase,lo,hi);
    let contribution=min(max(0.0,max(zlo,bounded_phase-bounds.y)),min(zhi,bounded_phase-bounds.x));
    return SingularXYZ(vec3<f32>(phase-contribution,y,sign*contribution),
        vec4<f32>(phase,lo,hi,select(0.0,1.0,xyz_chart_finite(lo) && xyz_chart_finite(hi) && hi-lo<tau)));
}
