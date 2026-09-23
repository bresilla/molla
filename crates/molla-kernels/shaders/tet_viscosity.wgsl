struct TetViscousBlock {
    force: vec3<f32>,
    tangent: mat3x3<f32>,
    geometric: f32,
    potential: f32,
}

fn tet_viscous_increment_block(previous: mat3x3<f32>, increment: mat3x3<f32>, gradient: vec3<f32>,
    shear: f32, bulk: f32, inverse_dt: f32, volume: f32) -> TetViscousBlock {
    let identity = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    let f = previous + increment;
    let strain = (transpose(previous) * increment + transpose(increment) * previous + transpose(increment) * increment) * 0.5;
    let tr = strain[0][0] + strain[1][1] + strain[2][2];
    let deviator = strain - identity * (tr / 3.0);
    let stress = deviator * (2.0 * shear) + identity * (bulk * tr);
    let scale = volume * inverse_dt;
    let a = f * gradient;
    let aa = mat3x3<f32>(a * a.x, a * a.y, a * a.z);
    let tangent = (f * transpose(f) * (shear * dot(gradient, gradient)) + aa * (bulk + shear / 3.0)) * scale;
    let norm_squared = dot(deviator[0], deviator[0]) + dot(deviator[1], deviator[1]) + dot(deviator[2], deviator[2]);
    return TetViscousBlock(-(f * stress * gradient) * scale, tangent,
        dot(gradient, stress * gradient) * scale,
        (shear * norm_squared + 0.5 * bulk * tr * tr) * scale);
}
