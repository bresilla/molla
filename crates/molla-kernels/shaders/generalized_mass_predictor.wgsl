const N: u32 = __WIDTH__u;
@group(0) @binding(0) var<storage, read> mass: array<f32>;
@group(0) @binding(1) var<storage, read> drive: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> acceleration: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> status: array<u32>;
var<private> factor: array<f64, N * N>;
var<private> driven: array<f64, N>;
var<private> biased: array<f64, N>;
var<private> published: array<vec2<f32>, N>;

fn finite(value: f64) -> bool {
    return value == value && abs(value) <= f64(1.7976931348623157e308);
}

fn pair_value(value: vec2<f32>) -> f64 {
    return f64(value.x) + f64(value.y);
}

@compute @workgroup_size(1)
fn predict() {
    if status[0] != 0u { return; }
    for (var i = 0u; i < N; i++) {
        driven[i] = pair_value(drive[i]);
        biased[i] = pair_value(bias[i]);
        if !finite(driven[i]) || !finite(biased[i]) { status[0] = 1u; return; }
        for (var j = 0u; j < N; j++) {
            let value = mass[i * N + j];
            if !finite(f64(value)) || value != mass[j * N + i] {
                status[0] = 1u; return;
            }
        }
    }
    for (var i = 0u; i < N; i++) {
        for (var j = 0u; j <= i; j++) {
            var value = f64(mass[i * N + j]);
            for (var k = 0u; k < j; k++) {
                value -= factor[i * N + k] * factor[j * N + k];
            }
            if !finite(value) { status[0] = 2u; return; }
            if i == j {
                if value <= f64(0.0) { status[0] = 2u; return; }
                factor[i * N + j] = sqrt(value);
            } else {
                factor[i * N + j] = value / factor[j * N + j];
            }
        }
        for (var j = 0u; j < i; j++) {
            driven[i] -= factor[i * N + j] * driven[j];
            biased[i] -= factor[i * N + j] * biased[j];
        }
        driven[i] /= factor[i * N + i];
        biased[i] /= factor[i * N + i];
    }
    for (var reverse = 0u; reverse < N; reverse++) {
        let i = N - 1u - reverse;
        for (var j = i + 1u; j < N; j++) {
            driven[i] -= factor[j * N + i] * driven[j];
            biased[i] -= factor[j * N + i] * biased[j];
        }
        driven[i] /= factor[i * N + i];
        biased[i] /= factor[i * N + i];
        let value = driven[i] - biased[i];
        let high = f32(value);
        let low = f32(value - f64(high));
        if !finite(driven[i]) || !finite(biased[i]) || !finite(f64(high)) || !finite(f64(low)) {
            status[0] = 3u; return;
        }
        published[i] = vec2<f32>(high, low);
    }
    var scale = vec3<f64>(f64(1.0));
    var error = vec3<f64>(f64(0.0));
    for (var i = 0u; i < N; i++) {
        let rhs = vec3<f64>(pair_value(drive[i]), pair_value(bias[i]),
            pair_value(drive[i]) - pair_value(bias[i]));
        var applied = vec3<f64>(f64(0.0));
        for (var j = 0u; j < N; j++) {
            applied += f64(mass[i * N + j]) *
                vec3<f64>(driven[j], biased[j], pair_value(published[j]));
        }
        let residual = abs(applied - rhs);
        if !finite(residual.x) || !finite(residual.y) || !finite(residual.z) {
            status[0] = 3u; return;
        }
        error = max(error, residual);
        scale = max(scale, vec3<f64>(f64(1.0)) + abs(rhs));
    }
    if any(error > f64(1e-6) * scale) { status[0] = 3u; return; }
    for (var i = 0u; i < N; i++) { acceleration[i] = published[i]; }
}
