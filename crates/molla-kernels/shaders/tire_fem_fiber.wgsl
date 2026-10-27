struct FiberResponse { stress: mat3x3<f32>, tangent: f32 }

fn add_fiber_stress(P_in: mat3x3<f32>, F: mat3x3<f32>, a0: vec3<f32>, k1: f32, k2: f32, gradient: vec3<f32>) -> FiberResponse {
    var P = P_in;
    if (k1 <= 0.0) {
        return FiberResponse(P, 0.0);
    }
    let Fa0 = F * a0;
    let i4 = dot(Fa0, Fa0);
    if (i4 <= 1.0) {
        return FiberResponse(P, 0.0);
    }
    let stretch = i4 - 1.0;
    var exp_factor: f32 = 1.0;
    if (k2 > 0.0) {
        exp_factor = exp(min(k2 * stretch * stretch, 30.0));
    }
    let coeff = 2.0 * k1 * stretch * exp_factor;
    P[0][0] = P[0][0] + coeff * Fa0.x * a0.x;
    P[1][0] = P[1][0] + coeff * Fa0.x * a0.y;
    P[2][0] = P[2][0] + coeff * Fa0.x * a0.z;
    P[0][1] = P[0][1] + coeff * Fa0.y * a0.x;
    P[1][1] = P[1][1] + coeff * Fa0.y * a0.y;
    P[2][1] = P[2][1] + coeff * Fa0.y * a0.z;
    P[0][2] = P[0][2] + coeff * Fa0.z * a0.x;
    P[1][2] = P[1][2] + coeff * Fa0.z * a0.y;
    P[2][2] = P[2][2] + coeff * Fa0.z * a0.z;
    let projected = dot(gradient, a0);
    var slope = 1.0;
    if (k2 > 0.0 && k2 * stretch * stretch < 30.0) {
        slope = 1.0 + 2.0 * k2 * stretch * stretch;
    }
    let tangent = 2.0 * k1 * exp_factor * (stretch + 2.0 * i4 * slope) * projected * projected;
    return FiberResponse(P, tangent);
}
