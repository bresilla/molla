struct Scan {
    counts: vec4<u32>,
    grid: vec4<u32>,
    origin: vec4<f32>,
    forward: vec4<f32>,
    up: vec4<f32>,
    right: vec4<f32>,
    angles: vec4<f32>,
    extent: vec4<f32>,
}
@group(0) @binding(0) var<storage, read> scans: array<Scan>;
@group(0) @binding(1) var<uniform> sizes: vec4<u32>;
@group(1) @binding(0) var<storage, read_write> origins: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> directions: array<vec4<f32>>;
@group(2) @binding(0) var<storage, read_write> distance: array<f32>;
@group(2) @binding(1) var<storage, read_write> shape: array<i32>;
@group(2) @binding(2) var<storage, read_write> point: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read_write> normal: array<vec4<f32>>;
@group(3) @binding(0) var<storage, read_write> first: array<i32>;
@group(3) @binding(1) var<storage, read_write> last: array<i32>;
@group(3) @binding(2) var<storage, read_write> hit_count: array<u32>;

fn descriptor(index: u32, pulse: bool) -> u32 {
    var low = 0u;
    var high = sizes.x;
    while (low + 1u < high) {
        let middle = (low + high) / 2u;
        let offset = select(scans[middle].counts.x, scans[middle].counts.y, pulse);
        if (index < offset) { high = middle; } else { low = middle; }
    }
    return low;
}

fn sample_axis(a: f32, b: f32, n: u32, i: u32) -> f32 {
    if (n <= 1u) { return a; }
    return a + (b - a) * (f32(i) / f32(n - 1u));
}

fn halton(index: u32, base: u32) -> f32 {
    var i = index;
    var weight = 1.0;
    var value = 0.0;
    while (i != 0u) {
        weight /= f32(base);
        value += weight * f32(i % base);
        i /= base;
    }
    return value;
}

fn cone(axis: vec3<f32>, angle: f32, k: u32, i: u32) -> vec3<f32> {
    if (i == 0u || k == 1u || angle <= 0.0) { return axis; }
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(axis.y) < 0.9);
    let e1 = normalize(cross(axis, helper));
    let e2 = normalize(cross(axis, e1));
    let alpha = angle * sqrt(f32(i) / f32(k - 1u));
    // Reduced golden-angle azimuth with split f32 constants.
    let turns = floor(f32(i) * 0.3819660112501051);
    let cycle = turns * 6.2831854820251465;
    let beta = fma(f32(i), 2.399963140487671, -cycle)
        - fma(turns, 6.2831854820251465, -cycle)
        + f32(i) * 8.9240982e-8 + turns * 1.7484556e-7;
    return normalize(cos(alpha) * axis + sin(alpha) * (cos(beta) * e1 + sin(beta) * e2));
}

@compute @workgroup_size(64)
fn generate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sizes.y) { return; }
    let s = scans[descriptor(i, false)];
    let local = i - s.counts.x;
    let pulse = local / s.counts.w;
    let subray = local % s.counts.w;
    var origin = s.origin.xyz;
    var central: vec3<f32>;
    if (s.grid.z == 0u) {
        let theta = sample_axis(s.angles.x, s.angles.y, s.grid.y, pulse / s.grid.x);
        let phi = sample_axis(s.angles.z, s.angles.w, s.grid.x, pulse % s.grid.x);
        central = normalize(s.forward.xyz * sin(theta) * cos(phi)
            + s.up.xyz * cos(theta) + s.right.xyz * sin(theta) * sin(phi));
    } else {
        origin += vec3<f32>(s.extent.x * halton(pulse + 1u, 2u), 0.0, s.extent.y * halton(pulse + 1u, 3u));
        let radius = s.up.w * sqrt(halton(pulse + 1u, 5u));
        let angle = 6.283185307179586 * halton(pulse + 1u, 7u);
        central = normalize(vec3<f32>(sin(radius) * cos(angle), -cos(radius), sin(radius) * sin(angle)));
    }
    origins[i] = vec4<f32>(origin, 0.0);
    directions[i] = vec4<f32>(cone(central, s.forward.w, s.counts.w, subray), 0.0);
}

@compute @workgroup_size(64)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pulse = gid.x;
    if (pulse >= sizes.z) { return; }
    let s = scans[descriptor(pulse, true)];
    let begin = s.counts.x + (pulse - s.counts.y) * s.counts.w;
    var nearest = -1;
    var farthest = -1;
    var minimum = s.origin.w;
    var maximum = 0.0;
    var count = 0u;
    for (var k = 0u; k < s.counts.w; k += 1u) {
        let i = begin + k;
        let t = distance[i];
        if (shape[i] >= 0 && t > 0.0 && t < s.origin.w) {
            count += 1u;
            if (t < minimum) { minimum = t; nearest = i32(i); }
            if (t >= maximum) { maximum = t; farthest = i32(i); }
        } else {
            distance[i] = -1.0;
            shape[i] = -1;
            point[i] = vec4<f32>(0.0);
            normal[i] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
        }
    }
    first[pulse] = nearest;
    last[pulse] = farthest;
    hit_count[pulse] = count;
}
