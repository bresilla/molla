// GPU layered tire transient relaxation.
//
// One invocation owns one wheel. Inputs are already reduced to raw slip
// commands (`raw_kappa`, `raw_alpha`) and distance travelled `ds`; the kernel
// updates tread/belt/carcass lag states and emits the effective slip state
// consumed by the brush/FEM contact kernels.

struct TransientInput {
    // x=raw_kappa, y=raw_alpha_rad, z=ds_m, w=radius_m
    command: vec4<f32>,
    // x=tread_long_m, y=tread_lat_m, z=tread_yaw_m, w=length_scale
    tread_lengths: vec4<f32>,
    // x=belt_long_m, y=belt_lat_m, z=belt_yaw_m, w=reserved
    belt_lengths: vec4<f32>,
    // x=carcass_long_m, y=carcass_lat_m, z=carcass_yaw_m, w=reserved
    carcass_lengths: vec4<f32>,
};

struct TransientState {
    // x=kappa_eff, y=alpha_eff_rad, z=yaw_rad, w=reserved
    tread: vec4<f32>,
    belt: vec4<f32>,
    carcass: vec4<f32>,
};

struct TransientOutput {
    // x=kappa_eff, y=alpha_eff_rad, z=tread_yaw_rad, w=finite flag
    effective: vec4<f32>,
    // x=longitudinal_deflection_m, y=lateral_deflection_m, zw=reserved
    deflection: vec4<f32>,
};

struct TransientUniforms {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> inputs: array<TransientInput>;
@group(0) @binding(1) var<storage, read> states_in: array<TransientState>;
@group(0) @binding(2) var<storage, read_write> states_out: array<TransientState>;
@group(0) @binding(3) var<storage, read_write> outputs: array<TransientOutput>;
@group(0) @binding(4) var<uniform> uniforms: TransientUniforms;

fn blend(ds: f32, length: f32) -> f32 {
    return clamp(1.0 - exp(-max(ds, 0.0) / max(length, 1.0e-4)), 0.0, 1.0);
}

fn advance_layer(layer: vec4<f32>, raw_kappa: f32, raw_alpha: f32, ds: f32, lengths: vec3<f32>) -> vec4<f32> {
    let bx = blend(ds, lengths.x);
    let by = blend(ds, lengths.y);
    let bt = blend(ds, lengths.z);
    var out = layer;
    out.x = out.x + (raw_kappa - out.x) * bx;
    out.y = out.y + (raw_alpha - out.y) * by;
    out.z = out.z + (raw_alpha - out.z) * bt;
    out.w = 0.0;
    return out;
}

@compute @workgroup_size(64)
fn transient_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.count) {
        return;
    }
    let input = inputs[i];
    let state = states_in[i];
    let raw_kappa = clamp(input.command.x, -8.0, 8.0);
    let raw_alpha = clamp(input.command.y, -1.2, 1.2);
    let ds = max(input.command.z, 0.0);
    let radius = max(input.command.w, 1.0e-6);
    let length_scale = clamp(max(input.tread_lengths.w, 1.0e-4), 0.05, 20.0);

    let tread = advance_layer(
        state.tread,
        raw_kappa,
        raw_alpha,
        ds,
        vec3<f32>(input.tread_lengths.x, input.tread_lengths.y, input.tread_lengths.z) * (length_scale * 0.85),
    );
    let belt = advance_layer(
        state.belt,
        raw_kappa,
        raw_alpha,
        ds,
        vec3<f32>(input.belt_lengths.x, input.belt_lengths.y, input.belt_lengths.z) * length_scale,
    );
    let carcass = advance_layer(
        state.carcass,
        raw_kappa,
        raw_alpha,
        ds,
        vec3<f32>(input.carcass_lengths.x, input.carcass_lengths.y, input.carcass_lengths.z) * (length_scale * 1.15),
    );

    let kappa_eff = 0.58 * tread.x + 0.30 * belt.x + 0.12 * carcass.x;
    let alpha_eff = 0.55 * tread.y + 0.32 * belt.y + 0.13 * carcass.y;
    let yaw_eff = 0.45 * tread.z + 0.35 * belt.z + 0.20 * carcass.z;
    let long_defl = radius * kappa_eff;
    let lat_defl = radius * clamp(tan(alpha_eff), -4.0, 4.0);

    states_out[i].tread = tread;
    states_out[i].belt = belt;
    states_out[i].carcass = carcass;
    outputs[i].effective = vec4<f32>(kappa_eff, alpha_eff, yaw_eff, 1.0);
    outputs[i].deflection = vec4<f32>(long_defl, lat_defl, 0.0, 0.0);
}
