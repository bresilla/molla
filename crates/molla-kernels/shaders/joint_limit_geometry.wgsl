fn euler_bounds(slot: u32, coordinate: u32, bounds: vec4<f32>, pitch_bounds: vec2<f32>, pitch_point: bool) -> EulerBounds {
    var pair=QuaternionPair(vec4<f32>(coordinates[coordinate],coordinates[coordinate+1u],
        coordinates[coordinate+2u],coordinates[coordinate+3u]),vec4<f32>(0.0));
    if params.retained_coordinates!=0u {
        pair.low=vec4<f32>(coordinate_residues[coordinate],coordinate_residues[coordinate+1u],
            coordinate_residues[coordinate+2u],coordinate_residues[coordinate+3u]);
    }
    return euler_bounds_pair(slot,pair,bounds,pitch_bounds,pitch_point);
}

fn euler_pitch_alignment(slot: u32, coordinate: u32, left: QuaternionPair, right: QuaternionPair) -> ConeGeometry {
    let rotation=normalize(left.high+left.low);
    let relative=euler_pair_relative(slot,coordinate,left,right);
    let q=relative.high+relative.low;
    let p=vec4<f32>(0.0,0.0,1.0,0.0);
    let b0=vec4<f32>(1.0,0.0,0.0,0.0);
    let b1=vec4<f32>(0.0,1.0,0.0,0.0);
    var geometry=cone_geometry(ConeRequest(vec4<f32>(q.z,q.x,q.y,q.w),p,b0,b1,vec2<f32>(0.0),vec2<u32>(1u)));
    for (var row=0u; row<geometry.info.x; row++) {
        let angular=geometry.rows[row].xyz;
        let t=2.0*cross(rotation.xyz,angular);
        geometry.rows[row]=vec4<f32>(angular+rotation.w*t+cross(rotation.xyz,t),geometry.rows[row].w);
    }
    return geometry;
}

fn euler_point_plane(slot: u32, coordinate: u32, axis: u32, left: QuaternionPair, right: QuaternionPair) -> vec4<f32> {
    let relative=euler_pair_relative(slot,coordinate,left,right);
    var local=vec4<f32>(0.0,0.0,1.0,0.0);
    var normal=vec3<f32>(0.0,1.0,0.0);
    var reference=vec3<f32>(1.0,0.0,0.0);
    if axis==2u { local=vec4<f32>(0.0,1.0,0.0,0.0); normal=vec3<f32>(1.0,0.0,0.0); reference=vec3<f32>(0.0,1.0,0.0); }
    let inverse=QuaternionPair(vec4<f32>(-relative.high.xyz,relative.high.w),vec4<f32>(-relative.low.xyz,relative.low.w));
    let direction=multiply(slot,multiply(slot,relative,QuaternionPair(local,vec4<f32>(0.0))),inverse);
    let geometry=xyz_plane_geometry(direction.high.xyz+direction.low.xyz,normal,reference);
    return vec4<f32>(xyz_rotate(normalize(left.high+left.low),geometry.xyz),geometry.w);
}

fn euler_pair_relative(slot: u32, coordinate: u32, left: QuaternionPair, right: QuaternionPair) -> QuaternionPair {
    var pair=QuaternionPair(vec4<f32>(coordinates[coordinate],coordinates[coordinate+1u],coordinates[coordinate+2u],coordinates[coordinate+3u]),vec4<f32>(0.0));
    if params.retained_coordinates!=0u { pair.low=vec4<f32>(coordinate_residues[coordinate],coordinate_residues[coordinate+1u],coordinate_residues[coordinate+2u],coordinate_residues[coordinate+3u]); }
    pair=normalize_pair(slot,pair);
    let inverse_left=QuaternionPair(vec4<f32>(-left.high.xyz,left.high.w),vec4<f32>(-left.low.xyz,left.low.w));
    let inverse_right=QuaternionPair(vec4<f32>(-right.high.xyz,right.high.w),vec4<f32>(-right.low.xyz,right.low.w));
    return normalize_pair(slot,multiply(slot,multiply(slot,inverse_left,pair),inverse_right));
}

fn euler_pair_alignment(slot: u32, coordinate: u32, left: QuaternionPair, right: QuaternionPair, free: u32) -> ConeGeometry {
    let relative=euler_pair_relative(slot,coordinate,left,right);
    return xyz_hinge_alignment(relative.high+relative.low,normalize(left.high+left.low),free);
}

fn euler_pair_bound(slot: u32, coordinate: u32, lower: vec3<f32>, upper: vec3<f32>, free: u32,
    left: QuaternionPair, right: QuaternionPair, bound: vec2<f32>, reference: vec2<f32>, reference_low: vec2<f32>) -> vec4<f32> {
    let relative=euler_pair_relative(slot,coordinate,left,right);
    let value=xyz_twist_coordinate(relative.high+relative.low,normalize(left.high+left.low),lower,upper,free);
    var projection=vec2<f32>(relative.high[free],relative.high.w);
    var projection_low=vec2<f32>(relative.low[free],relative.low.w);
    let combined=relative.high+relative.low;
    if combined.w*combined.w+combined[free]*combined[free]<=1e-12 {
        projection=vec2<f32>(relative.high[(free+1u)%3u],relative.high[(free+2u)%3u]);
        projection_low=vec2<f32>(relative.low[(free+1u)%3u],relative.low[(free+2u)%3u]);
    }
    let gap=euler_reference_gap(slot,(bound.x-value.w)+bound.y,
        projection,projection_low,reference,reference_low,2.0);
    return vec4<f32>(value.xyz,gap);
}
