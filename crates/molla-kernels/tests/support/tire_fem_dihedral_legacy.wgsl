fn quad_dihedral_angle(x0: vec3<f32>, x1: vec3<f32>, x2: vec3<f32>, x3: vec3<f32>) -> f32 {
    let n0 = safe_normalize(cross(x1 - x0, x2 - x0), vec3<f32>(1.0, 0.0, 0.0));
    let n1 = safe_normalize(cross(x2 - x0, x3 - x0), vec3<f32>(1.0, 0.0, 0.0));
    return acos(clamp(dot(n0, n1), -1.0, 1.0));
}

fn quad_dihedral_grad(local: u32, axis: u32, x0: vec3<f32>, x1: vec3<f32>, x2: vec3<f32>, x3: vec3<f32>) -> f32 {
    let eps = 1.0e-5;
    var p0 = x0;
    var p1 = x1;
    var p2 = x2;
    var p3 = x3;
    if (local == 0u) {
        if (axis == 0u) { p0.x = p0.x + eps; }
        if (axis == 1u) { p0.y = p0.y + eps; }
        if (axis == 2u) { p0.z = p0.z + eps; }
    }
    if (local == 1u) {
        if (axis == 0u) { p1.x = p1.x + eps; }
        if (axis == 1u) { p1.y = p1.y + eps; }
        if (axis == 2u) { p1.z = p1.z + eps; }
    }
    if (local == 2u) {
        if (axis == 0u) { p2.x = p2.x + eps; }
        if (axis == 1u) { p2.y = p2.y + eps; }
        if (axis == 2u) { p2.z = p2.z + eps; }
    }
    if (local == 3u) {
        if (axis == 0u) { p3.x = p3.x + eps; }
        if (axis == 1u) { p3.y = p3.y + eps; }
        if (axis == 2u) { p3.z = p3.z + eps; }
    }
    let plus = quad_dihedral_angle(p0, p1, p2, p3);
    p0 = x0;
    p1 = x1;
    p2 = x2;
    p3 = x3;
    if (local == 0u) {
        if (axis == 0u) { p0.x = p0.x - eps; }
        if (axis == 1u) { p0.y = p0.y - eps; }
        if (axis == 2u) { p0.z = p0.z - eps; }
    }
    if (local == 1u) {
        if (axis == 0u) { p1.x = p1.x - eps; }
        if (axis == 1u) { p1.y = p1.y - eps; }
        if (axis == 2u) { p1.z = p1.z - eps; }
    }
    if (local == 2u) {
        if (axis == 0u) { p2.x = p2.x - eps; }
        if (axis == 1u) { p2.y = p2.y - eps; }
        if (axis == 2u) { p2.z = p2.z - eps; }
    }
    if (local == 3u) {
        if (axis == 0u) { p3.x = p3.x - eps; }
        if (axis == 1u) { p3.y = p3.y - eps; }
        if (axis == 2u) { p3.z = p3.z - eps; }
    }
    let minus = quad_dihedral_angle(p0, p1, p2, p3);
    return clamp((plus - minus) / (2.0 * eps), -1.0e4, 1.0e4);
}

fn quad_dihedral_grad_vec(local: u32, x0: vec3<f32>, x1: vec3<f32>, x2: vec3<f32>, x3: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        quad_dihedral_grad(local, 0u, x0, x1, x2, x3),
        quad_dihedral_grad(local, 1u, x0, x1, x2, x3),
        quad_dihedral_grad(local, 2u, x0, x1, x2, x3),
    );
}
