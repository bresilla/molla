struct ScalarRow { endpoints: vec4<u32>, velocity: vec4<f32> }
struct Solution { impulse: vec2<f32>, slip: vec2<f32>, cap: f32, regime: u32, iterations: u32, residual: f32 }
struct Sample { force: vec4<f32>, tag: vec4<u32> }
struct Params { dt: f32, weight: f32, substep: u32, count: u32, dofs: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> rows: array<ScalarRow>;
@group(0) @binding(1) var<storage, read> directions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> solutions: array<Solution>;
@group(0) @binding(3) var<storage, read_write> samples: array<Sample>;
@group(0) @binding(4) var<storage, read_write> forces: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> status: atomic<u32>;

fn finite(x: f32) -> bool { return x == x && abs(x) <= 3.4028234e38; }

@compute @workgroup_size(64)
fn publish(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&rows) || atomicLoad(&status) != 0u) { return; }
    if (params.substep == 0u) { samples[i] = Sample(vec4<f32>(0.0),vec4<u32>(0u)); }
    if (rows[i].endpoints.w == 0u) { return; }
    let solution = solutions[i]; let kind = u32(directions[i].w);
    if (!finite(params.dt) || params.dt <= 0.0 || !finite(params.weight) || params.weight <= 0.0
        || solution.regime != rows[i].endpoints.z + 3u || !finite(solution.impulse.x)
        || !finite(solution.residual) || !finite(solution.impulse.y) || kind < 1u || kind > 24u
        || (solution.regime == 5u && (solution.impulse.x < 0.0
            || (solution.impulse.x == 0.0 && solution.impulse.y < 0.0)))) {
        atomicOr(&status,512u); return;
    }
    let force = directions[i].xyz * ((solution.impulse.x+solution.impulse.y) / params.dt);
    var sample = samples[i];
    let sum = sample.force.xyz + force * params.weight;
    if (!finite(force.x) || !finite(force.y) || !finite(force.z) || !finite(sum.x) || !finite(sum.y) || !finite(sum.z)) {
        atomicOr(&status,512u); return;
    }
    forces[i] = vec4<f32>(force,0.0);
    sample.force = vec4<f32>(sum,0.0);
    if (solution.regime == 4u) { sample.tag = vec4<u32>(3u,params.substep,0u,0u); }
    else if (solution.impulse.x != 0.0) { sample.tag = vec4<u32>(kind,params.substep,0u,0u); }
    samples[i] = sample;
}
