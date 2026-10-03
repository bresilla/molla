const MAX_MIXED_STOP_ROWS: u32 = 1u;
struct MixedStopRow { joint: u32, gradient: array<f32, 6> }
alias MixedStopRows = array<MixedStopRow, MAX_MIXED_STOP_ROWS>;
alias MixedStopBounds = array<f32, MAX_MIXED_STOP_ROWS>;

fn mixed_dot(row: MixedStopRow, values: ptr<function, StopScalars>) -> f32 {
    var value=0.0;
    for (var d=0u; d<6u; d++) { value+=row.gradient[d]*(*values)[row.joint][d]; }
    return value;
}
fn mixed_accumulate(row: MixedStopRow, value: f32, values: ptr<function, StopScalars>) {
    for (var d=0u; d<6u; d++) { (*values)[row.joint][d]+=row.gradient[d]*value; }
}
fn mixed_dot_pair(begin: u32, row: MixedStopRow, values: ptr<function, StopScalars>) -> vec2<f32> {
    let slot=joint_coord_offset[begin+row.joint];
    let saved=atomicLoad(&coordinate_error[slot]);
    var value=vec2<f32>(0.0);
    for (var d=0u; d<6u; d++) {
        let product=rounded(slot,row.gradient[d]*(*values)[row.joint][d]);
        let tail=fma(row.gradient[d],(*values)[row.joint][d],-product);
        let sum=two_sum(slot,value.x,product);
        value=two_sum(slot,sum.x,sum.y+value.y+tail);
    }
    atomicStore(&coordinate_error[slot],saved);
    return value;
}
fn mixed_residual(begin: u32, row: MixedStopRow, velocity: f32, bound: f32,
    response: ptr<function, StopScalars>, response_low: ptr<function, StopScalars>) -> f32 {
    let projection=mixed_dot_pair(begin,row,response);
    let projection_low=mixed_dot_pair(begin,row,response_low);
    let slot=joint_coord_offset[begin+row.joint];
    let saved=atomicLoad(&coordinate_error[slot]);
    let sum=two_sum(slot,projection.x,velocity);
    let gap=two_sum(slot,bound,-sum.x);
    let result=rounded(slot,gap.x+(gap.y-sum.y-projection.y-projection_low.x-projection_low.y));
    atomicStore(&coordinate_error[slot],saved);
    return result;
}
struct MixedPairGeometry { relative: QuaternionPair, left: QuaternionPair }
fn mixed_angular_offset(j: u32) -> u32 { return select(3u,0u,joint_type[j]==JOINT_BALL); }
fn mixed_point_axis(j: u32, axis: u32) -> bool { return motion_axis_point(j,mixed_angular_offset(j)+axis); }
fn mixed_angular_lock_count(j: u32) -> u32 {
    var count=0u;
    for (var axis=0u; axis<3u; axis++) { count+=select(0u,1u,mixed_point_axis(j,axis)); }
    return count;
}
fn mixed_joint_rotation(j: u32) -> QuaternionPair {
    let coord=joint_coord_offset[j]+mixed_angular_offset(j);
    var q=QuaternionPair(vec4<f32>(joint_q[coord],joint_q[coord+1u],joint_q[coord+2u],joint_q[coord+3u]),vec4<f32>(0.0));
    if uniforms.compensated_coordinates!=0u {
        q.low=vec4<f32>(bitcast<f32>(atomicLoad(&coordinate_error[coord])),bitcast<f32>(atomicLoad(&coordinate_error[coord+1u])),
            bitcast<f32>(atomicLoad(&coordinate_error[coord+2u])),bitcast<f32>(atomicLoad(&coordinate_error[coord+3u])));
    }
    return q;
}
fn mixed_predicted_rotation(j: u32, rate: vec3<f32>, advance: f32) -> QuaternionPair {
    let slot=joint_coord_offset[j];
    let saved=atomicLoad(&coordinate_error[slot]);
    let half_rate=rate*0.5;
    var delta=QuaternionPair(vec4<f32>(half_rate*advance,1.0),vec4<f32>(0.0));
    delta.low=vec4<f32>(fma(half_rate,vec3<f32>(advance),-delta.high.xyz),0.0);
    let rotation=normalize_pair(slot,multiply(slot,delta,mixed_joint_rotation(j)));
    atomicStore(&coordinate_error[slot],saved);
    return rotation;
}
fn mixed_euler_geometry(j: u32, rotation: QuaternionPair, lower: vec3<f32>, upper: vec3<f32>) -> EulerBounds {
    let slot=joint_coord_offset[j];
    let saved=atomicLoad(&coordinate_error[slot]);
    let result=euler_bounds_pair(slot,rotation,vec4<f32>(lower.x,upper.x,lower.z,upper.z),
        vec2<f32>(lower.y,upper.y),mixed_point_axis(j,1u));
    atomicStore(&coordinate_error[slot],saved);
    return result;
}
fn mixed_euler_gap(j: u32, axis: u32, geometry: EulerBounds, upper: bool) -> f32 {
    let slot=joint_coord_offset[j];
    let saved=atomicLoad(&coordinate_error[slot]);
    let reference=joint_limit_references[joint_dof_offset[j]+mixed_angular_offset(j)+axis];
    let initial=(select(reference.bounds.x,reference.bounds.z,upper)-geometry.angles[axis])+select(reference.bounds.y,reference.bounds.w,upper);
    let result=euler_reference_gap(slot,initial,vec2<f32>(geometry.sine[axis],geometry.cosine[axis]),
        vec2<f32>(geometry.sine_low[axis],geometry.cosine_low[axis]),
        select(reference.full.xy,reference.full.zw,upper),select(reference.full_low.xy,reference.full_low.zw,upper),1.0);
    atomicStore(&coordinate_error[slot],saved);
    return result;
}
fn mixed_reference_rotation(j: u32, axis: u32) -> QuaternionPair {
    return coordinate_axis_reference(joint_dof_offset[j]+mixed_angular_offset(j)+axis,axis);
}
fn mixed_relative_geometry(j: u32, rotation: QuaternionPair, left: QuaternionPair, right: QuaternionPair) -> QuaternionPair {
    let coord=joint_coord_offset[j];
    let saved=atomicLoad(&coordinate_error[coord]);
    let q=normalize_pair(coord,rotation);
    let inverse_left=QuaternionPair(vec4<f32>(-left.high.xyz,left.high.w),vec4<f32>(-left.low.xyz,left.low.w));
    let inverse_right=QuaternionPair(vec4<f32>(-right.high.xyz,right.high.w),vec4<f32>(-right.low.xyz,right.low.w));
    let relative=normalize_pair(coord,multiply(coord,multiply(coord,inverse_left,q),inverse_right));
    atomicStore(&coordinate_error[coord],saved);
    return relative;
}
fn mixed_pitch_alignment(j: u32, predicted: QuaternionPair) -> ConeGeometry {
    let rotation=mixed_reference_rotation(j,1u);
    let identity=QuaternionPair(vec4<f32>(0.0,0.0,0.0,1.0),vec4<f32>(0.0));
    let relative=mixed_relative_geometry(j,predicted,rotation,identity);
    let q=relative.high+relative.low;
    var geometry=cone_geometry(ConeRequest(vec4<f32>(q.z,q.x,q.y,q.w),
        vec4<f32>(0.0,0.0,1.0,0.0),vec4<f32>(1.0,0.0,0.0,0.0),vec4<f32>(0.0,1.0,0.0,0.0),vec2<f32>(0.0),vec2<u32>(1u)));
    for (var row=0u; row<geometry.info.x; row++) {
        geometry.rows[row]=vec4<f32>(xyz_rotate(normalize(rotation.high+rotation.low),geometry.rows[row].xyz),geometry.rows[row].w);
    }
    return geometry;
}
fn mixed_point_plane(j: u32, rotation: QuaternionPair, axis: u32) -> vec4<f32> {
    let coord=joint_coord_offset[j];
    let saved=atomicLoad(&coordinate_error[coord]);
    let identity=QuaternionPair(vec4<f32>(0.0,0.0,0.0,1.0),vec4<f32>(0.0));
    var left=identity; var right=identity;
    var local=vec4<f32>(0.0,0.0,1.0,0.0);
    var normal=vec3<f32>(0.0,1.0,0.0); var reference=vec3<f32>(1.0,0.0,0.0);
    if axis==0u { left=mixed_reference_rotation(j,axis); }
    else { right=mixed_reference_rotation(j,axis); local=vec4<f32>(0.0,1.0,0.0,0.0); normal=vec3<f32>(1.0,0.0,0.0); reference=vec3<f32>(0.0,1.0,0.0); }
    let relative=mixed_relative_geometry(j,rotation,left,right);
    let inverse=QuaternionPair(vec4<f32>(-relative.high.xyz,relative.high.w),vec4<f32>(-relative.low.xyz,relative.low.w));
    let direction=multiply(coord,multiply(coord,relative,QuaternionPair(local,vec4<f32>(0.0))),inverse);
    let plane=xyz_plane_geometry(direction.high.xyz+direction.low.xyz,normal,reference);
    atomicStore(&coordinate_error[coord],saved);
    return vec4<f32>(xyz_rotate(normalize(left.high+left.low),plane.xyz),plane.w);
}
struct MixedPhaseReference { bound: vec2<f32>, sine: vec2<f32>, cosine: vec2<f32> }
fn mixed_phase_reference(j: u32, positive: bool, upper: bool) -> MixedPhaseReference {
    let slot=joint_coord_offset[j];
    let saved=atomicLoad(&coordinate_error[slot]);
    let dof=joint_dof_offset[j]+mixed_angular_offset(j);
    let x=joint_limit_references[dof]; let z=joint_limit_references[dof+2u];
    let z_upper=select(!upper,upper,positive);
    let sign=select(-1.0,1.0,positive);
    let x_bound=select(x.bounds.xy,x.bounds.zw,upper); let z_bound=select(z.bounds.xy,z.bounds.zw,z_upper);
    let sum=two_sum(slot,x_bound.x,sign*z_bound.x);
    let bound=two_sum(slot,sum.x,sum.y+x_bound.y+sign*z_bound.y);
    let xr=select(x.full.xy,x.full.zw,upper); let zr=select(z.full.xy,z.full.zw,z_upper);
    let xl=select(x.full_low.xy,x.full_low.zw,upper); let zl=select(z.full_low.xy,z.full_low.zw,z_upper);
    let sine=product_sum(slot,vec4<f32>(xr.x,sign*xr.y,0.0,0.0),vec4<f32>(zr.y,zr.x,0.0,0.0));
    let cosine=product_sum(slot,vec4<f32>(xr.y,-sign*xr.x,0.0,0.0),vec4<f32>(zr.y,zr.x,0.0,0.0));
    let sine_pair=two_sum(slot,sine.x,sine.y+xr.x*zl.y+xl.x*zr.y+sign*(xr.y*zl.x+xl.y*zr.x));
    let cosine_pair=two_sum(slot,cosine.x,cosine.y+xr.y*zl.y+xl.y*zr.y-sign*(xr.x*zl.x+xl.x*zr.x));
    atomicStore(&coordinate_error[slot],saved);
    return MixedPhaseReference(bound,sine_pair,cosine_pair);
}
fn mixed_phase_active(j: u32, lower: MixedPhaseReference, upper: MixedPhaseReference) -> bool {
    if !xyz_chart_finite(lower.bound.x) || !xyz_chart_finite(upper.bound.x) { return false; }
    let slot=joint_coord_offset[j];
    let saved=atomicLoad(&coordinate_error[slot]);
    let width=two_sum(slot,upper.bound.x,-lower.bound.x);
    let gap=two_sum(slot,width.x,-6.2831854820251465);
    let residual=gap.x+(gap.y+width.y+upper.bound.y-lower.bound.y+1.7484556025237907e-7);
    atomicStore(&coordinate_error[slot],saved);
    return residual<0.0;
}
fn mixed_phase_gap(j: u32, geometry: EulerBounds, reference: MixedPhaseReference) -> f32 {
    let slot=joint_coord_offset[j];
    let saved=atomicLoad(&coordinate_error[slot]);
    let gap=euler_reference_gap(slot,(reference.bound.x-geometry.phase.x)+reference.bound.y,
        vec2<f32>(geometry.sine.x,geometry.cosine.x),vec2<f32>(geometry.sine_low.x,geometry.cosine_low.x),
        vec2<f32>(reference.sine.x,reference.cosine.x),vec2<f32>(reference.sine.y,reference.cosine.y),1.0);
    atomicStore(&coordinate_error[slot],saved);
    return gap;
}
fn mixed_pair_geometry(j: u32, rotation: QuaternionPair, free: u32) -> MixedPairGeometry {
    let dof=joint_dof_offset[j]; let coord=joint_coord_offset[j];
    let saved=atomicLoad(&coordinate_error[coord]);
    let identity=QuaternionPair(vec4<f32>(0.0,0.0,0.0,1.0),vec4<f32>(0.0));
    var rotations=array<QuaternionPair,3>(identity,identity,identity);
    for (var axis=0u;axis<3u;axis++) {
        if axis==free { continue; }
        rotations[axis]=mixed_reference_rotation(j,axis);
    }
    var left=identity; var right=identity;
    if free==0u { right=multiply(coord,rotations[1],rotations[2]); }
    else if free==1u { left=rotations[0]; right=rotations[2]; }
    else { left=multiply(coord,rotations[0],rotations[1]); }
    atomicStore(&coordinate_error[coord],saved);
    let relative=mixed_relative_geometry(j,rotation,left,right);
    atomicStore(&coordinate_error[coord],saved);
    return MixedPairGeometry(relative,left);
}
fn mixed_twist_gap(j: u32, free: u32, relative: QuaternionPair, angle: f32, upper: bool) -> f32 {
    let dof=joint_dof_offset[j]; let slot=joint_coord_offset[j];
    let saved=atomicLoad(&coordinate_error[slot]);
    let reference=joint_limit_references[dof+mixed_angular_offset(j)+free];
    var gap=select(reference.bounds.x,reference.bounds.z,upper)-angle+select(reference.bounds.y,reference.bounds.w,upper);
    if abs(gap)<=0.25 {
        var projection=vec2<f32>(relative.high[free],relative.high.w);
        var projection_low=vec2<f32>(relative.low[free],relative.low.w);
        let combined=relative.high+relative.low;
        if combined.w*combined.w+combined[free]*combined[free]<=1e-12 {
            projection=vec2<f32>(relative.high[(free+1u)%3u],relative.high[(free+2u)%3u]);
            projection_low=vec2<f32>(relative.low[(free+1u)%3u],relative.low[(free+2u)%3u]);
        }
        let bound_ref=select(reference.half.xy,reference.half.zw,upper);
        let bound_ref_low=select(reference.half_low.xy,reference.half_low.zw,upper);
        let sine=product_sum(slot,vec4<f32>(bound_ref.x,-bound_ref.y,0.0,0.0),vec4<f32>(projection.y,projection.x,0.0,0.0));
        let cosine=product_sum(slot,vec4<f32>(bound_ref.y,bound_ref.x,0.0,0.0),vec4<f32>(projection.y,projection.x,0.0,0.0));
        let s=sine.x+(sine.y+bound_ref.x*projection_low.y-bound_ref.y*projection_low.x+bound_ref_low.x*projection.y-bound_ref_low.y*projection.x);
        let c=cosine.x+(cosine.y+bound_ref.y*projection_low.y+bound_ref.x*projection_low.x+bound_ref_low.y*projection.y+bound_ref_low.x*projection.x);
        let sign=select(1.0,-1.0,c<0.0);
        gap=2.0*atan2(sign*s,sign*c);
    }
    atomicStore(&coordinate_error[slot],saved);
    return gap;
}
fn mixed_distance_radius(j: u32) -> vec2<f32> {
    let coord=joint_coord_offset[j];
    let saved=atomicLoad(&coordinate_error[coord]);
    let high=vec4<f32>(joint_q[coord],joint_q[coord+1u],joint_q[coord+2u],0.0);
    var low=vec4<f32>(0.0);
    if uniforms.compensated_coordinates!=0u {
        for (var axis=0u; axis<3u; axis++) { low[axis]=bitcast<f32>(atomicLoad(&coordinate_error[coord+axis])); }
    }
    let radius=length_pair(coord,high,low);
    atomicStore(&coordinate_error[coord],saved);
    return radius;
}
fn mixed_distance_gap(j: u32, radius: vec2<f32>, upper: bool) -> f32 {
    let reference=joint_limit_references[joint_dof_offset[j]];
    let bound=select(reference.bounds.x,reference.bounds.z,upper);
    if !xyz_chart_finite(bound) { return bound; }
    let coord=joint_coord_offset[j];
    let saved=atomicLoad(&coordinate_error[coord]);
    let difference=two_sum(coord,bound,-radius.x);
    let gap=rounded(coord,difference.x+(difference.y+select(reference.bounds.y,reference.bounds.w,upper)-radius.y));
    atomicStore(&coordinate_error[coord],saved);
    return gap;
}
fn mixed_append(k: u32, offset: u32, gradient: vec3<f32>, lower: f32, upper: f32,
    stops: ptr<function, MixedStopRows>, lo: ptr<function, MixedStopBounds>, hi: ptr<function, MixedStopBounds>, rows: ptr<function, u32>) -> bool {
    if !xyz_chart_finite(lower) && !xyz_chart_finite(upper) { return true; }
    if *rows>=MAX_MIXED_STOP_ROWS || lower!=lower || upper!=upper || lower>upper { return false; }
    if offset!=0u && offset!=3u { return false; }
    var row: MixedStopRow; row.joint=k;
    row.gradient=array<f32,6>(gradient.x,gradient.y,gradient.z,0.0,0.0,0.0);
    if offset==3u { row.gradient=array<f32,6>(0.0,0.0,0.0,gradient.x,gradient.y,gradient.z); }
    (*stops)[*rows]=row; (*lo)[*rows]=lower; (*hi)[*rows]=upper; *rows+=1u;
    return true;
}
fn mixed_angular_rows(k: u32, j: u32, rotation: QuaternionPair, dof: u32, advance: f32,
    stops: ptr<function, MixedStopRows>, lo: ptr<function, MixedStopBounds>, hi: ptr<function, MixedStopBounds>, rows: ptr<function, u32>) -> bool {
    if mixed_angular_lock_count(j)==3u { return true; }
    let infinity=bitcast<f32>(0x7f800000u);
    let angular=mixed_angular_offset(j);
    var lower=vec3<f32>(joint_limit_lower[dof+angular],joint_limit_lower[dof+angular+1u],joint_limit_lower[dof+angular+2u]);
    var upper=vec3<f32>(joint_limit_upper[dof+angular],joint_limit_upper[dof+angular+1u],joint_limit_upper[dof+angular+2u]);
    for (var d=0u; d<3u; d++) {
        if lower[d]!=lower[d] { lower[d]=-infinity; }
        if upper[d]!=upper[d] { upper[d]=infinity; }
    }
    let q=normalize(rotation.high+rotation.low);
    let matrix=mat3x3<f32>(quat_rotate(q,vec3<f32>(1.0,0.0,0.0)),
        quat_rotate(q,vec3<f32>(0.0,1.0,0.0)),quat_rotate(q,vec3<f32>(0.0,0.0,1.0)));
    let cy=length(vec2<f32>(matrix[0].x,matrix[1].x));
    let pair=mixed_angular_lock_count(j)==2u;
    var free_axis=0u;
    if pair {
        for (var axis=0u; axis<3u; axis++) { if !mixed_point_axis(j,axis) { free_axis=axis; } }
        let pair_geometry=mixed_pair_geometry(j,rotation,free_axis);
        let relative=pair_geometry.relative;
        let left=normalize(pair_geometry.left.high+pair_geometry.left.low);
        let geometry=xyz_hinge_alignment(relative.high+relative.low,left,free_axis);
        if geometry.info.z!=0u { return false; }
        for (var row=0u; row<geometry.info.x; row++) {
            let bound=-geometry.rows[row].w/advance;
            if !mixed_append(k,angular,geometry.rows[row].xyz,bound,bound,stops,lo,hi,rows) { return false; }
        }
        if !xyz_chart_finite(lower[free_axis]) && !xyz_chart_finite(upper[free_axis]) { return true; }
        let coordinate=xyz_twist_coordinate(relative.high+relative.low,left,lower,upper,free_axis);
        var lower_gap=-infinity; var upper_gap=infinity;
        if xyz_chart_finite(lower[free_axis]) { lower_gap=mixed_twist_gap(j,free_axis,relative,coordinate.w,false); }
        if xyz_chart_finite(upper[free_axis]) { upper_gap=mixed_twist_gap(j,free_axis,relative,coordinate.w,true); }
        return mixed_append(k,angular,coordinate.xyz,lower_gap/advance,upper_gap/advance,stops,lo,hi,rows);
    }
    let y=atan2(matrix[2].x,cy);
    let pitch_pole=mixed_point_axis(j,1u) && abs(cos(lower.y))<=1.9073486328125e-6;
    if pitch_pole && !pair {
        let geometry=mixed_pitch_alignment(j,rotation);
        if geometry.info.z!=0u { return false; }
        for (var row=0u; row<geometry.info.x; row++) {
            let bound=-geometry.rows[row].w/advance;
            if !mixed_append(k,angular,geometry.rows[row].xyz,bound,bound,stops,lo,hi,rows) { return false; }
        }
    }
    if cy<=1.9073486328125e-6 || pitch_pole {
        let geometry=mixed_euler_geometry(j,rotation,lower,upper);
        if !pitch_pole {
            for (var axis=0u; axis<3u; axis+=2u) {
                if !pair && mixed_point_axis(j,axis) {
                    let plane=mixed_point_plane(j,rotation,axis);
                    if !mixed_append(k,angular,plane.xyz,-plane.w/advance,-plane.w/advance,stops,lo,hi,rows) { return false; }
                }
            }
            var lower_gap=-infinity; var upper_gap=infinity;
            if xyz_chart_finite(lower.y) { lower_gap=mixed_euler_gap(j,1u,geometry,false); }
            if xyz_chart_finite(upper.y) { upper_gap=mixed_euler_gap(j,1u,geometry,true); }
            if !mixed_append(k,angular,geometry.gradients[1],lower_gap/advance,upper_gap/advance,stops,lo,hi,rows) { return false; }
        }
        let lower_phase=mixed_phase_reference(j,geometry.phase.w>0.0,false);
        let upper_phase=mixed_phase_reference(j,geometry.phase.w>0.0,true);
        if mixed_phase_active(j,lower_phase,upper_phase) {
            if !mixed_append(k,angular,vec3<f32>(1.0,0.0,0.0),mixed_phase_gap(j,geometry,lower_phase)/advance,
                mixed_phase_gap(j,geometry,upper_phase)/advance,stops,lo,hi,rows) { return false; }
        }
        return true;
    }
    let geometry=mixed_euler_geometry(j,rotation,lower,upper);
    for (var axis=0u; axis<3u; axis++) {
        var lower_gap=-infinity; var upper_gap=infinity;
        if xyz_chart_finite(lower[axis]) { lower_gap=mixed_euler_gap(j,axis,geometry,false); }
        if xyz_chart_finite(upper[axis]) { upper_gap=mixed_euler_gap(j,axis,geometry,true); }
        if !mixed_append(k,angular,geometry.gradients[axis],lower_gap*geometry.scales[axis]/advance,
            upper_gap*geometry.scales[axis]/advance,stops,lo,hi,rows) { return false; }
    }
    return true;
}
fn mixed_build_rows(begin: u32, n: u32, advance: f32, predicted: ptr<function, StopScalars>,
    stops: ptr<function, MixedStopRows>, lo: ptr<function, MixedStopBounds>, hi: ptr<function, MixedStopBounds>, rows: ptr<function, u32>) -> bool {
    let infinity=bitcast<f32>(0x7f800000u);
    let damp=max(1.0-uniforms.joint_damping*uniforms.dt,0.0);
    for (var k=0u; k<n; k++) {
        let j=begin+k;
        if joint_enabled[j]==0u || (body_flags[u32(joint_child[j])] & 3u)!=0u { continue; }
        let kind=joint_type[j];
        let dof=joint_dof_offset[j];
        let coordinate=joint_coord_offset[j];
        var axes=0u;
        if kind==JOINT_D6 { axes=3u; } else if kind<=1u { axes=1u; }
        for (var d=0u; d<axes; d++) {
            var lower=joint_limit_lower[dof+d]; var upper=joint_limit_upper[dof+d];
            if lower!=lower { lower=-infinity; } if upper!=upper { upper=infinity; }
            if motion_axis_point(j,d) { continue; }
            var gradient=vec3<f32>(0.0); gradient[d]=1.0;
            var lower_gap=coordinate_limit_gap(coordinate+d,dof+d,false);
            var upper_gap=coordinate_limit_gap(coordinate+d,dof+d,true);
            if lower_gap!=lower_gap { lower_gap=-infinity; } if upper_gap!=upper_gap { upper_gap=infinity; }
            if !mixed_append(k,0u,gradient,lower_gap/advance,upper_gap/advance,stops,lo,hi,rows) { return false; }
        }
        if kind==JOINT_D6 || kind==JOINT_BALL {
            let angular=mixed_angular_offset(j);
            let rate=vec3<f32>((*predicted)[k][angular],(*predicted)[k][angular+1u],(*predicted)[k][angular+2u]);
            let nonlinear=mixed_angular_lock_count(j)==0u;
            var rotation=mixed_joint_rotation(j);
            if nonlinear { rotation=mixed_predicted_rotation(j,rate,advance); }
            let first=*rows;
            if !mixed_angular_rows(k,j,rotation,dof,advance,stops,lo,hi,rows) { return false; }
            let half_delta=rate*(0.5*advance);
            for (var r=first; r<*rows; r++) {
                if !nonlinear { continue; }
                let gradient=vec3<f32>((*stops)[r].gradient[angular],(*stops)[r].gradient[angular+1u],(*stops)[r].gradient[angular+2u]);
                let transformed=(gradient-cross(half_delta,gradient))/(1.0+dot(half_delta,half_delta));
                for (var d=0u; d<3u; d++) { (*stops)[r].gradient[angular+d]=transformed[d]; }
                let shift=dot(transformed,rate);
                (*lo)[r]+=shift; (*hi)[r]+=shift;
            }
        }
        if kind==JOINT_DISTANCE {
            let position=vec3<f32>(joint_q[coordinate],joint_q[coordinate+1u],joint_q[coordinate+2u]);
            let radius=mixed_distance_radius(j);
            var lower=joint_limit_lower[dof]; var upper=joint_limit_upper[dof];
            if lower!=lower { lower=-infinity; } if upper!=upper { upper=infinity; }
            if upper==0.0 {
                for (var axis=0u; axis<3u; axis++) {
                    var gradient=vec3<f32>(0.0); gradient[axis]=1.0;
                    let bound=-position[axis]/advance;
                    if !mixed_append(k,0u,gradient,bound,bound,stops,lo,hi,rows) { return false; }
                }
            } else {
                var gradient=vec3<f32>(1.0,0.0,0.0); if radius.x>1e-12 { gradient=position/(radius.x+radius.y); }
                if !mixed_append(k,0u,gradient,mixed_distance_gap(j,radius,false)/advance,
                    mixed_distance_gap(j,radius,true)/advance,stops,lo,hi,rows) { return false; }
            }
        }
        for (var d=0u; d<joint_dof_n(kind); d++) {
            let lower=joint_limit_lower[dof+d]; let upper=joint_limit_upper[dof+d];
            let locked=motion_axis_point(j,d);
            if locked && (kind<=1u || (kind==JOINT_BALL && mixed_angular_lock_count(j)==3u)
                || (kind==JOINT_D6 && (d<3u || d6_angular_lock_count(j)==3u))) { continue; }
            let cap=joint_velocity_limit[dof+d];
            if !xyz_chart_finite(cap) { continue; }
            var gradient=vec3<f32>(0.0); gradient[d%3u]=1.0;
            if !mixed_append(k,select(0u,3u,d>=3u),gradient,-cap/damp,cap/damp,stops,lo,hi,rows) { return false; }
        }
    }
    return true;
}

fn mixed_projected_impulses(begin: u32, rows: u32,
    gram: ptr<function,array<array<f32,MAX_MIXED_STOP_ROWS>,MAX_MIXED_STOP_ROWS>>,
    velocity: ptr<function,MixedStopBounds>, lo: ptr<function,MixedStopBounds>, hi: ptr<function,MixedStopBounds>,
    impulses: ptr<function,MixedStopBounds>) -> bool {
    let slot=joint_coord_offset[begin];
    let saved=atomicLoad(&coordinate_error[slot]);
    var converged=false;
    for (var sweep=0u;sweep<32768u;sweep++) {
        for (var r=0u;r<rows;r++) {
            var sum=vec2<f32>((*velocity)[r],0.0);
            for (var c=0u;c<rows;c++) {
                if c==r { continue; }
                let product=rounded(slot,(*gram)[r][c]*(*impulses)[c]);
                let tail=fma((*gram)[r][c],(*impulses)[c],-product);
                let high=two_sum(slot,sum.x,product);
                sum=two_sum(slot,high.x,high.y+sum.y+tail);
            }
            let free=sum.x+sum.y;
            let bounded=clamp(free,(*lo)[r],(*hi)[r]);
            let gap=two_sum(slot,bounded,-sum.x);
            (*impulses)[r]=(gap.x+(gap.y-sum.y))/(*gram)[r][r];
        }
        converged=true;
        for (var r=0u;r<rows;r++) {
            var sum=vec2<f32>((*velocity)[r],0.0);
            for (var c=0u;c<rows;c++) {
                let product=rounded(slot,(*gram)[r][c]*(*impulses)[c]);
                let tail=fma((*gram)[r][c],(*impulses)[c],-product);
                let high=two_sum(slot,sum.x,product);
                sum=two_sum(slot,high.x,high.y+sum.y+tail);
            }
            let value=sum.x+sum.y;
            var residual=max((*lo)[r]-value,value-(*hi)[r]);
            if (*impulses)[r]>0.0 { residual=abs((*lo)[r]-value); }
            if (*impulses)[r]<0.0 { residual=abs((*hi)[r]-value); }
            var tolerance=1e-7*(1.0+abs((*velocity)[r]));
            if (*lo)[r]==0.0 && (*hi)[r]==0.0 { tolerance=min(tolerance,1e-10); }
            converged=converged && xyz_chart_finite(value) && xyz_chart_finite((*impulses)[r]) && residual<=tolerance;
        }
        if converged { break; }
    }
    atomicStore(&coordinate_error[slot],saved);
    return converged;
}

fn react_mixed_stops(
    begin: u32, n: u32,
    counts: ptr<function, StopCounts>,
    sl: ptr<function, StopVectors>, sa: ptr<function, StopVectors>,
    ul: ptr<function, StopVectors>, ua: ptr<function, StopVectors>,
    inverse: ptr<function, StopBlocks>,
    acceleration: ptr<function, StopScalars>,
) -> bool {
    for (var iteration=0u; iteration<16u; iteration++) {
        var converged=false;
        if !react_mixed_stops_iteration(begin,n,counts,sl,sa,ul,ua,inverse,acceleration,&converged) { return false; }
        if converged { return true; }
    }
    return false;
}

fn react_mixed_stops_iteration(
    begin: u32, n: u32,
    counts: ptr<function, StopCounts>,
    sl: ptr<function, StopVectors>, sa: ptr<function, StopVectors>,
    ul: ptr<function, StopVectors>, ua: ptr<function, StopVectors>,
    inverse: ptr<function, StopBlocks>,
    acceleration: ptr<function, StopScalars>,
    converged_prediction: ptr<function, bool>,
) -> bool {
    let advance = uniforms.dt * max(1.0 - uniforms.joint_damping * uniforms.dt, 0.0);
    if advance == 0.0 { *converged_prediction=true; return true; }
    var parents: StopParents;
    var stops: MixedStopRows;
    var lo: MixedStopBounds;
    var hi: MixedStopBounds;
    var velocity: MixedStopBounds;
    var predicted: StopScalars;
    var rows=0u;
    var violated=false;
    for (var k=0u; k<n; k++) {
        let j=begin+k;
        parents[k]=-1;
        for (var p=0u; p<k; p++) { if joint_child[begin+p]==joint_parent[j] { parents[k]=i32(p); } }
        for (var d=0u; d<(*counts)[k]; d++) {
            predicted[k][d]=joint_qd[joint_dof_offset[j]+d]+uniforms.dt*(*acceleration)[k][d];
        }
    }
    if !mixed_build_rows(begin,n,advance,&predicted,&stops,&lo,&hi,&rows) { return false; }
    for (var r=0u; r<rows; r++) {
        velocity[r]=mixed_dot(stops[r],&predicted);
        if !xyz_chart_finite(velocity[r]) { return false; }
        var tolerance=1e-7*(1.0+abs(velocity[r]));
        if lo[r]==0.0 && hi[r]==0.0 { tolerance=min(tolerance,1e-10); }
        violated=violated || max(lo[r]-velocity[r],velocity[r]-hi[r])>tolerance;
    }
    if !violated { *converged_prediction=true; return true; }
    var gram: array<array<f32, MAX_MIXED_STOP_ROWS>, MAX_MIXED_STOP_ROWS>;
    var unit: StopScalars;
    for (var c = 0u; c < rows; c += 1u) {
        if c > 0u { for (var d=0u;d<6u;d++) { unit[stops[c-1u].joint][d]=0.0; } }
        for (var d=0u;d<6u;d++) { unit[stops[c].joint][d]=stops[c].gradient[d]; }
        var response = stop_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &unit);
        for (var r = 0u; r < rows; r += 1u) { gram[r][c] = mixed_dot(stops[r],&response); }
    }
    for (var r = 0u; r < rows; r += 1u) {
        if !(gram[r][r] > 0.0) { return false; }
        for (var c = 0u; c < r; c += 1u) {
            let value = 0.5 * (gram[r][c] + gram[c][r]);
            gram[r][c] = value;
            gram[c][r] = value;
        }
    }

    var signs: array<f32, MAX_MIXED_STOP_ROWS>;
    var impulses: array<f32, MAX_MIXED_STOP_ROWS>;
    var active_rows: array<u32, MAX_MIXED_STOP_ROWS>;
    var active_count = 0u;
    for (var r=0u;r<rows;r++) {
        if lo[r]==hi[r] {
            signs[r]=1.0;
            active_rows[active_count]=r;
            active_count++;
        }
    }
    var factor: array<array<f32, MAX_MIXED_STOP_ROWS>, MAX_MIXED_STOP_ROWS>;
    var converged = false;
    var dependent=false;
    for (var iteration = 0u; iteration < 4u*rows*rows+1u; iteration += 1u) {
        var entering = rows;
        var entering_sign = 0.0;
        var worst = 0.0;
        for (var r = 0u; r < rows; r += 1u) {
            if signs[r] != 0.0 { continue; }
            var v = velocity[r];
            for (var c = 0u; c < rows; c += 1u) { v += gram[r][c] * signs[c] * impulses[c]; }
            let residual = max(lo[r]-v, v-hi[r]);
            let tolerance = 1e-7 * (1.0 + abs(velocity[r]));
            if residual > tolerance && residual / sqrt(gram[r][r]) > worst {
                entering = r;
                entering_sign = select(-1.0, 1.0, v < lo[r]);
                worst = residual / sqrt(gram[r][r]);
            }
        }
        if entering == rows && !(iteration==0u && active_count>0u) { converged = true; break; }
        if entering<rows {
            signs[entering] = entering_sign;
            active_rows[active_count] = entering;
            active_count += 1u;
        }
        var solved = false;
        for (var retry = 0u; retry <= rows; retry += 1u) {
            var solution: array<f32, MAX_MIXED_STOP_ROWS>;
            for (var r = 0u; r < active_count; r += 1u) {
                let row = active_rows[r];
                for (var c = 0u; c <= r; c += 1u) {
                    let col = active_rows[c];
                    var value = gram[row][col] * signs[row] * signs[col];
                    for (var d = 0u; d < c; d += 1u) { value -= factor[r][d] * factor[c][d]; }
                    if r == c {
                        if !(value > 1e-6*gram[row][row]) { dependent=true; break; }
                        factor[r][c] = sqrt(value);
                    } else { factor[r][c] = value / factor[c][c]; }
                }
                if dependent { break; }
                let bound = select(hi[row], lo[row], signs[row] > 0.0);
                var rhs = signs[row] * (bound - velocity[row]);
                for (var c = 0u; c < r; c += 1u) { rhs -= factor[r][c] * solution[c]; }
                solution[r] = rhs / factor[r][r];
            }
            if dependent { break; }
            var trial: array<f32, MAX_MIXED_STOP_ROWS>;
            for (var r = 0u; r < rows; r += 1u) { trial[r] = 0.0; }
            var positive = true;
            for (var back = 0u; back < active_count; back += 1u) {
                let r = active_count - 1u - back;
                var value = solution[r];
                for (var c = r+1u; c < active_count; c += 1u) { value -= factor[c][r] * solution[c]; }
                solution[r] = value / factor[r][r];
                trial[active_rows[r]] = solution[r];
                let row=active_rows[r];
                positive = positive && (lo[row]==hi[row] || solution[r]>0.0);
            }
            if positive { impulses = trial; solved = true; break; }
            var alpha = 1.0;
            for (var r = 0u; r < active_count; r += 1u) {
                let row = active_rows[r];
                if lo[row]!=hi[row] && trial[row] <= 0.0 {
                    if impulses[row] == 0.0 { alpha = 0.0; }
                    else { alpha = min(alpha, impulses[row] / (impulses[row]-trial[row])); }
                }
            }
            var retained = 0u;
            for (var r = 0u; r < active_count; r += 1u) {
                let row = active_rows[r];
                impulses[row] += alpha * (trial[row]-impulses[row]);
                if lo[row]==hi[row] || impulses[row] > 0.0 { active_rows[retained] = row; retained += 1u; }
                else { impulses[row] = 0.0; signs[row] = 0.0; }
            }
            active_count = retained;
        }
        if !solved { break; }
    }
    if !converged {
        var signed: MixedStopBounds;
        for (var r=0u;r<rows;r++) { signed[r]=signs[r]*impulses[r]; }
        if !mixed_projected_impulses(begin,rows,&gram,&velocity,&lo,&hi,&signed) { return false; }
        for (var r=0u;r<rows;r++) {
            signs[r]=select(-1.0,1.0,signed[r]>=0.0);
            impulses[r]=abs(signed[r]);
        }
        active_count=0u;
    }
    var force: StopScalars;
    for (var r = 0u; r < rows; r += 1u) { mixed_accumulate(stops[r],signs[r]*impulses[r],&force); }
    var response = stop_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &force);
    var response_low: StopScalars;
    for (var refinement = 0u; refinement < 4u; refinement += 1u) {
        var residual: array<f32, MAX_MIXED_STOP_ROWS>;
        var needs_refinement = false;
        for (var r = 0u; r < active_count; r += 1u) {
            let row = active_rows[r];
            let bound = select(hi[row], lo[row], signs[row] > 0.0);
            let error = mixed_residual(begin,stops[row],velocity[row],bound,&response,&response_low);
            if !xyz_chart_finite(error) { return false; }
            needs_refinement = needs_refinement || abs(error) > 1e-7 * (1.0 + abs(velocity[row]));
            var value = signs[row] * error;
            for (var c = 0u; c < r; c += 1u) { value -= factor[r][c] * residual[c]; }
            residual[r] = value / factor[r][r];
        }
        if !needs_refinement { break; }
        var correction: StopScalars;
        for (var back = 0u; back < active_count; back += 1u) {
            let r = active_count - 1u - back;
            let row = active_rows[r];
            var value = residual[r];
            for (var c = r+1u; c < active_count; c += 1u) { value -= factor[c][r] * residual[c]; }
            residual[r] = value / factor[r][r];
            if lo[row]!=hi[row] && impulses[row] + residual[r] < 0.0 { return false; }
            impulses[row] += residual[r];
            mixed_accumulate(stops[row],signs[row]*residual[r],&correction);
        }
        let delta = stop_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &correction);
        for (var k = 0u; k < n; k += 1u) {
            let slot=joint_coord_offset[begin+k];
            let saved=atomicLoad(&coordinate_error[slot]);
            for (var d = 0u; d < (*counts)[k]; d += 1u) {
                let sum=two_sum(slot,response[k][d],delta[k][d]);
                let pair=two_sum(slot,sum.x,sum.y+response_low[k][d]);
                response[k][d]=pair.x; response_low[k][d]=pair.y;
            }
            atomicStore(&coordinate_error[slot],saved);
        }
    }
    for (var r = 0u; r < rows; r += 1u) {
        let tolerance = 2e-5 * (1.0 + abs(velocity[r]));
        if xyz_chart_finite(lo[r]) {
            let error=mixed_residual(begin,stops[r],velocity[r],lo[r],&response,&response_low);
            if !xyz_chart_finite(error) || error>tolerance { return false; }
        }
        if xyz_chart_finite(hi[r]) {
            let error=mixed_residual(begin,stops[r],velocity[r],hi[r],&response,&response_low);
            if !xyz_chart_finite(error) || error<-tolerance { return false; }
        }
        if impulses[r] > 0.0 || lo[r]==hi[r] {
            let bound = select(hi[r], lo[r], signs[r] > 0.0);
            let error=mixed_residual(begin,stops[r],velocity[r],bound,&response,&response_low);
            if !xyz_chart_finite(error) || abs(error)>tolerance { return false; }
        }
    }
    for (var k = 0u; k < n; k += 1u) {
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            (*acceleration)[k][d] += (response[k][d]+response_low[k][d]) / uniforms.dt;
        }
    }
    for (var k=0u;k<n;k++) {
        let j=begin+k;
        if joint_enabled[j]==0u || (body_flags[u32(joint_child[j])] & 3u)!=0u { continue; }
        let dof=joint_dof_offset[j];
        for (var d=0u;d<(*counts)[k];d++) {
            if joint_velocity_limit[dof+d]!=0.0 { continue; }
            let rate=joint_qd[dof+d]+uniforms.dt*(*acceleration)[k][d];
            let roundoff=64.0*1.1920928955078125e-7*max(abs(joint_qd[dof+d]),abs(predicted[k][d]));
            if !xyz_chart_finite(rate) || abs(rate)>1e-10+roundoff { return false; }
            (*acceleration)[k][d]=-joint_qd[dof+d]/uniforms.dt;
        }
    }
    return true;
}
