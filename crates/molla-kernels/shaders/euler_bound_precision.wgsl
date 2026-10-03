struct EulerBounds {
    angles: vec3<f32>, gradients: mat3x3<f32>, scales: vec3<f32>, phase: vec4<f32>,
    sine: vec3<f32>, cosine: vec3<f32>, sine_low: vec3<f32>, cosine_low: vec3<f32>,
}
fn euler_pitch_cosine(slot: u32, high: vec2<f32>, low: vec2<f32>) -> vec2<f32> {
    let scale=max(max(abs(high.x),abs(high.y)),max(abs(low.x),abs(low.y)));
    if scale==0.0 { return vec2<f32>(0.0); }
    let quotient=high/scale;
    let tail=vec2<f32>(fma(-quotient.x,scale,high.x),fma(-quotient.y,scale,high.y))/scale+low/scale;
    let x=two_sum(slot,quotient.x,tail.x); let y=two_sum(slot,quotient.y,tail.y);
    let scaled_high=vec2<f32>(x.x,y.x); let scaled_low=vec2<f32>(x.y,y.y);
    let projection=vec4<f32>(scaled_high,0.0,0.0);
    let square=product_sum(slot,projection,projection);
    let square_low=square.y+2.0*dot(scaled_high,scaled_low)+dot(scaled_low,scaled_low);
    let root=rounded(slot,sqrt(square.x));
    let root_square=rounded(slot,root*root);
    let high_residual=rounded(slot,square.x-root_square);
    let low_residual=rounded(slot,square_low-fma(root,root,-root_square));
    let root_low=rounded(slot,rounded(slot,high_residual+low_residual)/(2.0*root));
    let value=rounded(slot,root*scale);
    return two_sum(slot,value,fma(root,scale,-value)+root_low*scale);
}
fn euler_bounds_pair(slot: u32, initial: QuaternionPair, bounds: vec4<f32>, pitch_bounds: vec2<f32>, pitch_point: bool) -> EulerBounds {
    var pair=initial;
    pair=normalize_pair(slot,pair);
    let inverse=QuaternionPair(vec4<f32>(-pair.high.xyz,pair.high.w),vec4<f32>(-pair.low.xyz,pair.low.w));
    var matrix=mat3x3<f32>();
    var matrix_high=mat3x3<f32>();
    var matrix_low=mat3x3<f32>();
    for (var column=0u; column<3u; column++) {
        var basis=vec4<f32>(0.0); basis[column]=1.0;
        let rotated=multiply(slot,multiply(slot,pair,QuaternionPair(basis,vec4<f32>(0.0))),inverse);
        matrix[column]=rotated.high.xyz+rotated.low.xyz;
        matrix_high[column]=rotated.high.xyz;
        matrix_low[column]=rotated.low.xyz;
    }
    let cy=length(vec2<f32>(matrix[0].x,matrix[1].x));
    let y=atan2(matrix[2].x,cy);
    let pitch_cosine=euler_pitch_cosine(slot,vec2<f32>(matrix_high[0].x,matrix_high[1].x),
        vec2<f32>(matrix_low[0].x,matrix_low[1].x));
    let pitch_pole=pitch_point && abs(cos(pitch_bounds.x))<=1.9073486328125e-6;
    if cy<=1.9073486328125e-6 || pitch_pole {
        let pole_pitch=select(-1.5707963267948966,1.5707963267948966,sin(pitch_bounds.x)>=0.0);
        let chart_pitch=select(y,pole_pitch,pitch_pole);
        let chart=xyz_singular_chart(atan2(matrix[1].z,matrix[1].y),chart_pitch,bounds);
        let x=chart.angles.x;
        var angles=chart.angles;
        angles.y=xyz_interval_angle(y,pitch_bounds.x,pitch_bounds.y);
        var pitch_direction=1.0;
        var pitch_gradient=vec3<f32>(0.0,cos(x),sin(x));
        if !pitch_pole && cy>1e-12 {
            let roll=atan2(-matrix[2].y,matrix[2].z);
            let yaw=atan2(-matrix[1].x,matrix[0].x);
            let regular=xyz_regular_chart(vec3<f32>(roll,y,yaw),
                vec3<f32>(bounds.x,pitch_bounds.x,bounds.z),vec3<f32>(bounds.y,pitch_bounds.y,bounds.w));
            angles.y=regular.y; pitch_direction=regular.w;
            pitch_gradient=vec3<f32>(0.0,cos(roll),sin(roll))*pitch_direction;
        }
        return EulerBounds(angles,mat3x3<f32>(
            vec3<f32>(0.0),pitch_gradient,vec3<f32>(0.0)),
            vec3<f32>(0.0,1.0,0.0),vec4<f32>(chart.phase.xyz,select(-1.0,1.0,chart_pitch>=0.0)),
            vec3<f32>(matrix_high[1].z,matrix_high[2].x,0.0),vec3<f32>(matrix_high[1].y,pitch_cosine.x*pitch_direction,0.0),
            vec3<f32>(matrix_low[1].z,matrix_low[2].x,0.0),vec3<f32>(matrix_low[1].y,pitch_cosine.y*pitch_direction,0.0));
    }
    let x=atan2(-matrix[2].y,matrix[2].z);
    let z=atan2(-matrix[1].x,matrix[0].x);
    let sine=-matrix[2].y/cy;
    let cosine=matrix[2].z/cy;
    let chart=xyz_regular_chart(vec3<f32>(x,y,z),vec3<f32>(bounds.x,pitch_bounds.x,bounds.z),vec3<f32>(bounds.y,pitch_bounds.y,bounds.w));
    return EulerBounds(chart.xyz,mat3x3<f32>(
        vec3<f32>(cy,sine*matrix[2].x,-cosine*matrix[2].x),
        vec3<f32>(0.0,cosine,sine)*chart.w,vec3<f32>(0.0,-sine,cosine)),vec3<f32>(cy,1.0,cy),vec4<f32>(0.0),
        vec3<f32>(-matrix_high[2].y*chart.w,matrix_high[2].x,-matrix_high[1].x*chart.w),
        vec3<f32>(matrix_high[2].z,pitch_cosine.x,matrix_high[0].x)*chart.w,
        vec3<f32>(-matrix_low[2].y*chart.w,matrix_low[2].x,-matrix_low[1].x*chart.w),
        vec3<f32>(matrix_low[2].z,pitch_cosine.y,matrix_low[0].x)*chart.w);
}

fn euler_reference_gap(slot: u32, initial_gap: f32, value: vec2<f32>, value_low: vec2<f32>,
    reference: vec2<f32>, reference_low: vec2<f32>, factor: f32) -> f32 {
    var gap=initial_gap;
    if abs(gap)<=0.25 {
        let projection=vec4<f32>(value.y,value.x,0.0,0.0);
        let sine=product_sum(slot,vec4<f32>(reference.x,-reference.y,0.0,0.0),projection);
        let cosine=product_sum(slot,vec4<f32>(reference.y,reference.x,0.0,0.0),projection);
        let sine_value=sine.x+(sine.y+reference.x*value_low.y-reference.y*value_low.x
            +reference_low.x*value.y-reference_low.y*value.x);
        let cosine_value=cosine.x+(cosine.y+reference.y*value_low.y+reference.x*value_low.x
            +reference_low.y*value.y+reference_low.x*value.x);
        let canonical=select(1.0,-1.0,cosine_value<0.0 && factor==2.0);
        gap=factor*atan2(canonical*sine_value,canonical*cosine_value);
    }
    return gap;
}
