var<private> rate_rounding_child: u32;

fn rate_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let high=position_sum(rate_rounding_child,a.x,b.x);
    return position_sum(rate_rounding_child,high.x,high.y+a.y+b.y);
}
fn rate_product(a: vec2<f32>, b: f32) -> vec2<f32> {
    let high=position_round(rate_rounding_child,a.x*b);
    return position_sum(rate_rounding_child,high,fma(a.x,b,-high)+a.y*b);
}
fn rate_integrate(j: u32, k: u32, d: u32, acceleration: f32, damp: f32) -> f32 {
    if uniforms.paired_rates==0u || uniforms.drift_only==2u {
        return mixed_integrated_rate(j,k,d,acceleration)*damp;
    }
    let index=joint_dof_offset[j]+d;
    if uniforms.drift_only!=0u { return paired_rates[index].x; }
    var predicted=rate_add(paired_rates[index],rate_product(vec2<f32>(acceleration,0.0),uniforms.dt));
    if mixed_velocity_enabled { predicted=vec2<f32>(mixed_velocity[k][d],mixed_velocity_low[k][d]); }
    predicted=rate_product(predicted,damp);
    paired_rates[index]=predicted;
    return predicted.x;
}
fn rate_finish(index: u32, previous: vec2<f32>, reset: bool) {
    if uniforms.paired_rates==0u || uniforms.drift_only==2u { return; }
    if reset || joint_qd[index]!=paired_rates[index].x {
        paired_rates[index]=vec2<f32>(joint_qd[index],0.0);
    }
    if uniforms.drift_only==0u {
        paired_rate_delta[index]=rate_add(paired_rates[index],-previous);
        velocity_delta[index]=paired_rate_delta[index].x;
    }
}
fn rate_cap(index: u32, high: f32) -> f32 {
    let limit=joint_velocity_limit[index];
    let capped=clamp_or_passthrough(high,limit);
    if uniforms.paired_rates!=0u && uniforms.drift_only!=2u && limit==limit && limit<=1e30 {
        let value=paired_rates[index];
        if capped!=high || (high==limit && value.y>0.0) || (high== -limit && value.y<0.0) {
            paired_rates[index]=vec2<f32>(capped,0.0);
        }
    }
    return capped;
}
fn rate_step(index: u32, high: f32) -> vec2<f32> {
    var value=vec2<f32>(high,0.0);
    if uniforms.paired_rates!=0u && uniforms.drift_only!=2u && high==paired_rates[index].x {
        value=paired_rates[index];
    }
    return rate_product(value,uniforms.dt);
}
fn rate_coordinate_advance(coord: u32, index: u32, high: f32) -> f32 {
    if uniforms.paired_rates==0u || uniforms.drift_only==2u {
        return coordinate_advance(coord,high*uniforms.dt);
    }
    let step=rate_step(index,high);
    let low=bitcast<f32>(atomicLoad(&coordinate_error[coord]));
    let sum=coordinate_sum(coord,joint_q[coord],step.x);
    let next=coordinate_sum(coord,sum.x,sum.y+low+step.y);
    atomicStore(&coordinate_error[coord],bitcast<u32>(next.y));
    return next.x;
}
fn rate_quaternion_advance(coord: u32, index: u32, omega: vec3<f32>) -> vec4<f32> {
    if uniforms.paired_rates==0u || uniforms.drift_only==2u {
        return coordinate_quaternion_advance(coord,omega);
    }
    var old=QuaternionPair(vec4<f32>(0.0),vec4<f32>(0.0));
    for (var lane=0u; lane<4u; lane++) {
        old.high[lane]=joint_q[coord+lane];
        old.low[lane]=bitcast<f32>(atomicLoad(&coordinate_error[coord+lane]));
    }
    var step=QuaternionPair(vec4<f32>(0.0),vec4<f32>(0.0));
    for (var lane=0u; lane<3u; lane++) {
        let half=rate_product(rate_step(index+lane,omega[lane]),0.5);
        step.high[lane]=half.x; step.low[lane]=half.y;
    }
    let increment=multiply(coord,step,old);
    var advanced=QuaternionPair(vec4<f32>(0.0),vec4<f32>(0.0));
    for (var lane=0u; lane<4u; lane++) {
        let sum=coordinate_sum(coord,old.high[lane],increment.high[lane]);
        let normalized=coordinate_sum(coord,sum.x,sum.y+old.low[lane]+increment.low[lane]);
        advanced.high[lane]=normalized.x; advanced.low[lane]=normalized.y;
    }
    let result=normalize_pair(coord,advanced);
    for (var lane=0u; lane<4u; lane++) {
        atomicStore(&coordinate_error[coord+lane],bitcast<u32>(result.low[lane]));
    }
    return result.high;
}
