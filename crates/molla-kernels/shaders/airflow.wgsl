@group(0) @binding(0) var<uniform> cfg: Settings;
@group(0) @binding(1) var<storage, read_write> velocity: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> predicted: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(4) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(5) var<storage, read> forces: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> samples: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> reactions: array<atomic<i32>>;

fn advect_component(p: vec3<u32>, i: u32, axis: u32) -> f32 {
    if !valid(p, axis) || (axis == 1u && p.y == 0u) { return 0.0; }
    let h = cfg.origin_h.w;
    let dt = cfg.physics.z;
    let x = cfg.origin_h.xyz + (vec3<f32>(p) + offset(axis)) * h;
    let mid = x - sample_air(x) * (0.5 * dt);
    let back = x - sample_air(mid) * dt;
    var laplacian = 0.0;
    for (var a = 0u; a < 3u; a++) {
        let bound = i32(cfg.cells[a]) - select(1, 0, a == axis);
        for (var delta = -1; delta <= 1; delta += 2) {
            var q = p;
            q[a] = u32(clamp(i32(p[a]) + delta, 0, bound));
            laplacian += velocity[index(q)][axis] - velocity[i][axis];
        }
    }
    return sample_axis(back, axis) + cfg.drive[axis] + cfg.physics.y * dt / (h * h) * laplacian
        + dt * (forces[i][axis] + f32(atomicLoad(&reactions[3u * i + axis])) * REACTION_QUANTUM)
            / (cfg.physics.x * h * h * h);
}
@compute @workgroup_size(8,4,4)
fn advect(@builtin(global_invocation_id) p: vec3<u32>) {
    if any(p > cfg.cells.xyz) { return; }
    let i = index(p);
    predicted[i] = vec4<f32>(advect_component(p, i, 0u), advect_component(p, i, 1u), advect_component(p, i, 2u), 0.0);
}

@compute @workgroup_size(8,4,4)
fn divergence(@builtin(global_invocation_id) tid: vec3<u32>) {
    let p = tid;
    if any(p > cfg.cells.xyz) { return; }
    let i = index(p);
    pressure[i] = 0.0;
    rhs[i] = 0.0;
    if any(p >= cfg.cells.xyz) { return; }
    var div = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        var q = p;
        q[axis] += 1u;
        div += predicted[index(q)][axis] - predicted[i][axis];
    }
    rhs[i] = -cfg.physics.x * cfg.origin_h.w / cfg.physics.z * div;
}

fn relax(tid: vec3<u32>, color: u32) {
    let p = vec3<u32>(2u * tid.x + ((color + tid.y + tid.z) & 1u), tid.y, tid.z);
    if any(p >= cfg.cells.xyz) { return; }
    let i = index(p);
    var sum = rhs[i];
    var diagonal = 6.0;
    let sy = cfg.cells.x + 1u;
    let sz = sy * (cfg.cells.y + 1u);
    if p.x > 0u { sum += pressure[i - 1u]; } else { diagonal += 1.0; }
    if p.x + 1u < cfg.cells.x { sum += pressure[i + 1u]; } else { diagonal += 1.0; }
    if p.y > 0u { sum += pressure[i - sy]; } else { diagonal -= 1.0; }
    if p.y + 1u < cfg.cells.y { sum += pressure[i + sy]; } else { diagonal += 1.0; }
    if p.z > 0u { sum += pressure[i - sz]; } else { diagonal += 1.0; }
    if p.z + 1u < cfg.cells.z { sum += pressure[i + sz]; } else { diagonal += 1.0; }
    pressure[i] += cfg.physics.w * (sum / diagonal - pressure[i]);
}
@compute @workgroup_size(8,4,4)
fn red(@builtin(global_invocation_id) tid: vec3<u32>) { relax(tid, 0u); }
@compute @workgroup_size(8,4,4)
fn black(@builtin(global_invocation_id) tid: vec3<u32>) { relax(tid, 1u); }

@compute @workgroup_size(8,4,4)
fn project(@builtin(global_invocation_id) tid: vec3<u32>) {
    let p = tid;
    if any(p > cfg.cells.xyz) { return; }
    let i = index(p);
    var result = vec4<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis++) {
        if !valid(p, axis) || (axis == 1u && p.y == 0u) { continue; }
        var left = p;
        var right = p;
        left[axis] = u32(max(i32(p[axis]) - 1, 0));
        right[axis] = min(p[axis], cfg.cells[axis] - 1u);
        let l = pressure[index(left)] * select(1.0, -1.0, p[axis] == 0u);
        let r = pressure[index(right)] * select(1.0, -1.0, p[axis] == cfg.cells[axis]);
        result[axis] = predicted[i][axis] - cfg.physics.z / (cfg.physics.x * cfg.origin_h.w) * (r - l);
    }
    velocity[i] = result;
}

@compute @workgroup_size(128)
fn sample_points(@builtin(global_invocation_id) tid: vec3<u32>) {
    if tid.x >= arrayLength(&points) { return; }
    samples[tid.x] = vec4<f32>(sample_exterior(points[tid.x].xyz) + turbulence(points[tid.x].xyz), 0.0);
}
