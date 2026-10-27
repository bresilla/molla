struct MeshTriangleHit { t: f32, normal: vec3<f32> }

fn mesh_triangle_hit(origin: vec3<f32>, direction: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> MeshTriangleHit {
    let miss = MeshTriangleHit(-1.0, vec3<f32>(0.0));
    let first = b - a;
    let second = c - a;
    let extent = max(abs(first), abs(second));
    let magnitude = max(extent.x, max(extent.y, extent.z));
    if (magnitude == 0.0) { return miss; }
    let exponent = clamp((bitcast<u32>(magnitude) >> 23u) & 255u, 1u, 253u);
    let scale = bitcast<f32>((254u - exponent) << 23u);
    let e1 = first * scale;
    let e2 = second * scale;
    let h = cross(direction, e2);
    let determinant = dot(e1, h);
    if (determinant == 0.0) { return miss; }
    let relative = (origin - a) * scale;
    let u = dot(relative, h) / determinant;
    if (u < 0.0 || u > 1.0) { return miss; }
    let q = cross(relative, e1);
    let v = dot(direction, q) / determinant;
    if (v < 0.0 || u + v > 1.0) { return miss; }
    let normal = cross(e1, e2);
    let normal_scale = max(abs(normal.x), max(abs(normal.y), abs(normal.z)));
    if (normal_scale == 0.0) { return miss; }
    return MeshTriangleHit(dot(e2, q) / determinant / scale, normal / normal_scale);
}
