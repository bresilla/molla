struct Settings { iterations: u32, blocked: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> settings: Settings;
var<private> lambda: array<f32, 9>;

fn velocity(i: u32) -> f32 {
    var value = input[81u+i];
    for (var j = 0u; j < 9u; j += 1u) {
        value += input[i*9u+j] * (lambda[j]-input[90u+j]);
    }
    return value;
}

fn normal_block() -> bool {
    var rhs = vec3<f32>(0.0);
    for (var i = 0u; i < 3u; i += 1u) {
        rhs[i] = input[99u+i] - velocity(3u*i);
        for (var j = 0u; j < 3u; j += 1u) {
            rhs[i] += input[27u*i+3u*j] * lambda[3u*j];
        }
    }
    for (var mask = 0u; mask < 7u; mask += 1u) {
        var candidate = vec3<f32>(0.0);
        let count = countOneBits(mask);
        let x = select(select(2u, 1u, (mask & 2u) != 0u), 0u, (mask & 1u) != 0u);
        let y = select(1u, 2u, (mask & 4u) != 0u);
        let ids = vec2<u32>(x, y);
        if count == 1u {
            candidate[ids.x] = rhs[ids.x] / input[30u*ids.x];
        }
        if count == 2u {
            let a = input[30u*ids.x];
            let b = input[27u*ids.x+3u*ids.y];
            let c = input[27u*ids.y+3u*ids.x];
            let d = input[30u*ids.y];
            let determinant = a*d-b*c;
            if determinant <= 1e-6*a*d { continue; }
            candidate[ids.x] = (d*rhs[ids.x]-b*rhs[ids.y])/determinant;
            candidate[ids.y] = (a*rhs[ids.y]-c*rhs[ids.x])/determinant;
        }
        if any(candidate < vec3<f32>(0.0)) { continue; }
        var residual = 0.0;
        for (var i = 0u; i < 3u; i += 1u) {
            var slack = -rhs[i];
            for (var j = 0u; j < 3u; j += 1u) { slack += input[27u*i+3u*j]*candidate[j]; }
            var error = max(0.0, -slack);
            if candidate[i] > 0.0 { error = abs(slack); }
            residual = max(residual, error);
        }
        if residual <= 1e-5 {
            for (var i = 0u; i < 3u; i += 1u) { lambda[3u*i] = candidate[i]; }
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(1)
fn solve() {
    for (var i = 0u; i < 9u; i += 1u) { lambda[i] = input[90u+i]; }
    var block_count = 0u;
    for (var iteration = 0u; iteration < settings.iterations; iteration += 1u) {
        var blocked = false;
        if settings.blocked != 0u { blocked = normal_block(); }
        if blocked { block_count += 1u; }
        for (var r = 0u; r < 3u; r += 1u) {
            let n = 3u*r;
            if !blocked { lambda[n] = max(0.0, lambda[n]+(input[99u+r]-velocity(n))/input[n*9u+n]); }
            var tangent = vec2<f32>(lambda[n+1u], lambda[n+2u]);
            tangent -= vec2<f32>(velocity(n+1u), velocity(n+2u))/input[105u+r];
            let limit = input[102u+r]*lambda[n];
            let magnitude = length(tangent);
            if magnitude > limit { tangent *= limit/magnitude; }
            lambda[n+1u] = tangent.x;
            lambda[n+2u] = tangent.y;
        }
    }
    for (var i = 0u; i < 9u; i += 1u) { output[i] = lambda[i]; output[9u+i] = velocity(i); }
    output[18] = f32(block_count);
}
