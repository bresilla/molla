struct Scalar { impulse: vec2<f32>, slip: vec2<f32>, cap: f32, regime: u32, iterations: u32, residual: f32 }
@group(0) @binding(0) var<storage, read> impact_impulses: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> friction_impulses: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> impact_delta: array<f32>;
@group(0) @binding(3) var<storage, read> friction_delta: array<f32>;
@group(0) @binding(4) var<storage, read> impact_scalars: array<Scalar>;
@group(0) @binding(5) var<storage, read> friction_scalars: array<Scalar>;
@group(0) @binding(6) var<storage, read> impact_slots: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> friction_slots: array<vec2<u32>>;
@group(0) @binding(8) var<storage, read_write> impulses: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> delta: array<f32>;
@group(0) @binding(10) var<storage, read_write> scalars: array<Scalar>;
@group(0) @binding(11) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(12) var<uniform> dimensions: vec4<u32>;
const NONE: u32 = 0xffffffffu;
const INPUT: u32 = 4194304u;
const NONFINITE: u32 = 16777216u;
fn finite(v: vec4<f32>) -> bool { return all(v == v) && all(abs(v) <= vec4<f32>(3.4028234e38)); }
fn valid_slot(slot: vec2<u32>) -> bool { return (slot.x == NONE) == (slot.y == NONE); }
fn valid_scalar(s: Scalar, enabled: bool) -> bool {
    if !finite(vec4<f32>(s.impulse, s.slip)) || !finite(vec4<f32>(s.cap, s.residual, 0.0, 0.0)) {
        atomicOr(&status, NONFINITE); return false;
    }
    if enabled { return s.regime == 4u || s.regime == 5u || s.regime == 6u; }
    return s.regime == 0u && all(s.impulse == vec2<f32>(0.0)) && all(s.slip == vec2<f32>(0.0))
        && s.cap == 0.0 && s.residual == 0.0 && s.iterations == 0u;
}

@compute @workgroup_size(64)
fn validate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if atomicLoad(&status) != 0u { return; }
    if i < dimensions.x {
        let a = impact_slots[i]; let b = friction_slots[i];
        let x = impact_impulses[i]; let y = friction_impulses[i];
        if !finite(x) || !finite(y) { atomicOr(&status, NONFINITE); return; }
        if !valid_slot(a) || !valid_slot(b) || (a.x != NONE && b.x != NONE)
            || (a.x == NONE && any(x != vec4<f32>(0.0)))
            || (b.x == NONE && any(y != vec4<f32>(0.0))) { atomicOr(&status, INPUT); return; }
    }
    if i < dimensions.y {
        let a = impact_slots[dimensions.x+i]; let b = friction_slots[dimensions.x+i];
        if !valid_slot(a) || !valid_slot(b) || (a.x != NONE && b.x != NONE)
            || !valid_scalar(impact_scalars[i], a.x != NONE)
            || !valid_scalar(friction_scalars[i], b.x != NONE) { atomicOr(&status, INPUT); return; }
    }
    if i < dimensions.z {
        let a = impact_delta[i]; let b = friction_delta[i];
        if !finite(vec4<f32>(a,b,0.0,0.0)) { atomicOr(&status, NONFINITE); return; }
        if a != 0.0 && b != 0.0 { atomicOr(&status, INPUT); return; }
    }
}

@compute @workgroup_size(64)
fn merge(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if atomicLoad(&status) != 0u { return; }
    if i < dimensions.x {
        impulses[i] = select(friction_impulses[i], impact_impulses[i], impact_slots[i].x != NONE);
    }
    if i < dimensions.y {
        if impact_slots[dimensions.x+i].x != NONE { scalars[i] = impact_scalars[i]; }
        else { scalars[i] = friction_scalars[i]; }
    }
    if i < dimensions.z { delta[i] = impact_delta[i] + friction_delta[i]; }
}
