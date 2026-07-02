// GPU batch step for the four-node tire thermal model.
//
// One invocation owns one tire/wheel. This mirrors the reusable
// TireThermalLayers update: sliding heats the surface/tread, rolling and
// casing flex heat the tread/carcass, brake heat enters the rim, and
// conductances exchange/cool the four thermal nodes.

struct ThermalInput {
    // x=sliding_power_w, y=carcass_flex_power_w, z=rolling_power_w, w=brake_rim_power_w
    powers: vec4<f32>,
    // x=road_c, y=dt_s, zw=reserved
    environment: vec4<f32>,
};

struct ThermalState {
    // x=surface_c, y=tread_c, z=carcass_c, w=rim_c
    temperatures: vec4<f32>,
    // x=depth_m, y=max_depth_m, z=wear_rate_m3_per_j, w=ref_patch_area_m2
    wear: vec4<f32>,
    // x=surface_j_per_k, y=tread_j_per_k, z=carcass_j_per_k, w=rim_j_per_k
    capacities: vec4<f32>,
    // x=ambient_c, y=road_w_per_k, z=air_w_per_k, w=surface_tread_w_per_k
    coeffs0: vec4<f32>,
    // x=tread_carcass_w_per_k, y=carcass_rim_w_per_k, zw=reserved
    coeffs1: vec4<f32>,
};

struct ThermalOutput {
    // x=surface_c, y=tread_c, z=carcass_c, w=rim_c
    temperatures: vec4<f32>,
    // x=depth_m, y=max_depth_m, z=wear_ratio, w=mu_wear_factor
    wear: vec4<f32>,
    // x=rubber_peak_c, y=mu_factor, z=carcass_pressure_factor, w=wear_temperature_factor
    diagnostics: vec4<f32>,
};

struct ThermalUniforms {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> inputs: array<ThermalInput>;
@group(0) @binding(1) var<storage, read> states_in: array<ThermalState>;
@group(0) @binding(2) var<storage, read_write> states_out: array<ThermalState>;
@group(0) @binding(3) var<storage, read_write> outputs: array<ThermalOutput>;
@group(0) @binding(4) var<uniform> uniforms: ThermalUniforms;

fn finite_nonnegative(v: f32) -> f32 {
    if (!(v == v) || abs(v) > 3.0e38) {
        return 0.0;
    }
    return max(v, 0.0);
}

fn finite_or(v: f32, fallback: f32) -> f32 {
    if (!(v == v) || abs(v) > 3.0e38) {
        return fallback;
    }
    return v;
}

fn finite_positive_or(v: f32, fallback: f32, floor_v: f32) -> f32 {
    if (!(v == v) || abs(v) > 3.0e38 || v < floor_v) {
        return max(fallback, floor_v);
    }
    return v;
}

fn wear_temperature_factor(surface_c: f32) -> f32 {
    let t = finite_or(surface_c, 20.0);
    if (t < 70.0) {
        return clamp(0.65 + max(t + 20.0, 0.0) / 90.0 * 0.35, 0.45, 1.0);
    }
    let hot = (t - 70.0) / 45.0;
    return clamp(1.0 + hot * hot, 1.0, 6.0);
}

fn step_wear(wear: vec4<f32>, power_w: f32, surface_c: f32, dt: f32) -> vec4<f32> {
    let depth_m = finite_nonnegative(wear.x);
    let max_depth_m = max(finite_positive_or(wear.y, 0.008, 1.0e-6), 1.0e-6);
    let wear_rate_m3_per_j = finite_nonnegative(wear.z);
    let ref_patch_area_m2 = finite_positive_or(wear.w, 0.035, 1.0e-6);
    let factor = wear_temperature_factor(surface_c);
    let volume_removed = wear_rate_m3_per_j * finite_nonnegative(power_w) * factor * dt;
    let depth_step = volume_removed / ref_patch_area_m2;
    return vec4<f32>(
        min(depth_m + depth_step, max_depth_m),
        max_depth_m,
        wear_rate_m3_per_j,
        ref_patch_area_m2,
    );
}

fn wear_mu_factor(wear: vec4<f32>) -> f32 {
    let ratio = clamp(finite_nonnegative(wear.x) / max(wear.y, 1.0e-9), 0.0, 1.0);
    return 1.0 - 0.30 * ratio;
}

fn mu_factor(temperatures: vec4<f32>, ambient_c: f32) -> f32 {
    let surface_c = finite_or(temperatures.x, ambient_c);
    let tread_c = finite_or(temperatures.y, ambient_c);
    let carcass_c = finite_or(temperatures.z, ambient_c);
    let surface_peak = 1.08 - abs(surface_c - 80.0) * 0.006;
    let bulk_hot_penalty = clamp(max(0.5 * (tread_c + carcass_c) - 115.0, 0.0) * 0.0025, 0.0, 0.25);
    return clamp(surface_peak - bulk_hot_penalty, 0.45, 1.08);
}

fn carcass_pressure_factor(carcass_c: f32, ambient_c: f32) -> f32 {
    return clamp(1.0 + (finite_or(carcass_c, ambient_c) - ambient_c) * 0.0015, 0.85, 1.20);
}

@compute @workgroup_size(64)
fn thermal_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.count) {
        return;
    }

    let input = inputs[i];
    var state = states_in[i];
    let sliding_power_w = finite_nonnegative(input.powers.x);
    let carcass_flex_power_w = finite_nonnegative(input.powers.y);
    let rolling_power_w = finite_nonnegative(input.powers.z);
    let brake_rim_power_w = finite_nonnegative(input.powers.w);
    let total_dt = min(finite_nonnegative(input.environment.y), 10.0);
    let ambient_c = finite_or(state.coeffs0.x, 20.0);
    let road_c = finite_or(input.environment.x, ambient_c);

    let c_surface = finite_positive_or(state.capacities.x, 800.0, 1.0);
    let c_tread = finite_positive_or(state.capacities.y, 8000.0, 1.0);
    let c_carcass = finite_positive_or(state.capacities.z, 18000.0, 1.0);
    let c_rim = finite_positive_or(state.capacities.w, 30000.0, 1.0);
    let g_road = finite_nonnegative(state.coeffs0.y);
    let g_air = finite_nonnegative(state.coeffs0.z);
    let g_st = finite_nonnegative(state.coeffs0.w);
    let g_tc = finite_nonnegative(state.coeffs1.x);
    let g_cr = finite_nonnegative(state.coeffs1.y);

    var temperatures = state.temperatures;
    var remaining = total_dt;
    loop {
        if (remaining <= 1.0e-9) {
            break;
        }
        let h = min(remaining, 0.05);
        remaining = remaining - h;

        let surface_c = finite_or(temperatures.x, ambient_c);
        let tread_c = finite_or(temperatures.y, ambient_c);
        let carcass_c = finite_or(temperatures.z, ambient_c);
        let rim_c = finite_or(temperatures.w, ambient_c);

        let q_surface_in = 0.70 * sliding_power_w + 0.15 * rolling_power_w;
        let q_tread_in =
            0.30 * sliding_power_w + 0.35 * rolling_power_w + 0.20 * carcass_flex_power_w;
        let q_carcass_in =
            0.80 * carcass_flex_power_w + 0.50 * rolling_power_w + 0.15 * brake_rim_power_w;
        let q_rim_in = 0.85 * brake_rim_power_w;

        let q_st = g_st * (surface_c - tread_c);
        let q_tc = g_tc * (tread_c - carcass_c);
        let q_cr = g_cr * (carcass_c - rim_c);
        let q_surface_air = g_air * (surface_c - ambient_c);
        let q_surface_road = g_road * (surface_c - road_c);
        let q_carcass_air = 0.35 * g_air * (carcass_c - ambient_c);
        let q_rim_air = 0.25 * g_air * (rim_c - ambient_c);

        temperatures.x = surface_c + (q_surface_in - q_st - q_surface_air - q_surface_road) / c_surface * h;
        temperatures.y = tread_c + (q_tread_in + q_st - q_tc) / c_tread * h;
        temperatures.z = carcass_c + (q_carcass_in + q_tc - q_cr - q_carcass_air) / c_carcass * h;
        temperatures.w = rim_c + (q_rim_in + q_cr - q_rim_air) / c_rim * h;
    }

    temperatures.x = clamp(finite_or(temperatures.x, ambient_c), -80.0, 350.0);
    temperatures.y = clamp(finite_or(temperatures.y, ambient_c), -80.0, 350.0);
    temperatures.z = clamp(finite_or(temperatures.z, ambient_c), -80.0, 350.0);
    temperatures.w = clamp(finite_or(temperatures.w, ambient_c), -80.0, 350.0);
    let wear = step_wear(state.wear, sliding_power_w, temperatures.x, total_dt);

    state.temperatures = temperatures;
    state.wear = wear;
    states_out[i] = state;
    let rubber_peak = max(max(temperatures.x, temperatures.y), temperatures.z);
    let wear_ratio = clamp(wear.x / max(wear.y, 1.0e-9), 0.0, 1.0);
    outputs[i].temperatures = temperatures;
    outputs[i].wear = vec4<f32>(wear.x, wear.y, wear_ratio, wear_mu_factor(wear));
    outputs[i].diagnostics = vec4<f32>(
        rubber_peak,
        mu_factor(temperatures, ambient_c),
        carcass_pressure_factor(temperatures.z, ambient_c),
        wear_temperature_factor(temperatures.x),
    );
}
