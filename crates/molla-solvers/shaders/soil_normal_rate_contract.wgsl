fn normal_rate_add(value:vec2<f32>,normal:vec3<f32>,column:vec3<f32>,rate:vec2<f32>)->vec2<f32> {
    return rate_add(value,rate_mul(rate_round(dot(normal,column)),rate));
}
fn normal_rate_residual(velocity:vec3<f32>,normal:vec3<f32>,gap:f32,machine:vec2<f32>)->vec2<f32> {
    let soil=rate_round(dot(normal,velocity));
    let residual=rate_add(rate_add(vec2<f32>(soil,0.0),-machine),vec2<f32>(gap,0.0));
    let speed=rate_round(machine.x+machine.y);
    return vec2<f32>(rate_round(residual.x+residual.y),max(1.0,max(abs(soil),max(abs(speed),gap))));
}
