struct Camera {
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    projection: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,       // [-1,1] across jet (x) and along jet (y)
    @location(2) params: vec4<f32>,   // x: opacity, y: age [0..1], z: seed, w: kind (0=RCS, 1=CryoAir)
}

struct Fragment {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) params: vec4<f32>,
    @location(2) world: vec3<f32>,
}

@vertex fn vertex(v: Vertex) -> Fragment {
    let delta = v.position - camera.eye.xyz;
    let depth = -dot(delta, camera.forward.xyz);
    var out: Fragment;
    out.position = vec4<f32>(
        dot(delta, camera.right.xyz) * camera.projection.x,
        dot(delta, camera.up.xyz) * 2.5,
        (100.0 * depth - 5.0) / 99.95,
        depth,
    );
    out.uv = v.uv;
    out.params = v.params;
    out.world = v.position;
    return out;
}

fn hash2(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(127.1, 311.7));
    let r = q + dot(q, q.yx + 34.23);
    return fract(r.x * r.y);
}

fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var amp = 0.55;
    var q = p;
    for (var i = 0; i < 3; i++) {
        v += amp * noise2(q);
        q = q * 2.07 + vec2<f32>(13.7, 7.3);
        amp *= 0.48;
    }
    return v;
}

@fragment fn fragment(v: Fragment) -> @location(0) vec4<f32> {
    let opacity = v.params.x;
    let age = v.params.y;
    let seed = v.params.z;
    let kind = v.params.w;

    // Radial falloff with subtle axial Mach-cone tapering.
    let r2 = dot(v.uv, v.uv);
    if r2 >= 1.0 {
        discard;
    }
    let radial = exp(-2.8 * r2) * (1.0 - smoothstep(0.45, 1.0, r2));

    // Advected multi-octave wispy turbulence + fine cryo ice-crystal grain!
    let flow_uv = vec2<f32>(v.uv.x * 2.4 + seed, v.uv.y * 1.5 - age * 2.8 + seed * 0.61);
    let wisp = fbm(flow_uv);
    let fine = noise2(flow_uv * 3.8 + vec2<f32>(seed * 1.7, -age * 4.0));
    let structure = mix(0.55, 1.15, smoothstep(0.18, 0.85, wisp)) * mix(0.75, 1.18, fine);

    // Subtle directional sunlight Mie forward-scattering across the gas cloud.
    let sun = normalize(vec3<f32>(-0.45, 0.8, 0.5));
    let view = normalize(v.world - camera.eye.xyz);
    let mie = 0.82 + 0.36 * pow(max(dot(view, sun), 0.0), 3.0);

    // Distinct physical spectral profiles:
    //   kind < 0.5:  Draco RCS hypergolic/cold-gas (Mach-blue/white throat -> delicate violet-silver mist)
    //   0.5..1.5:    Adiabatic 213 K cryo-vented air (crisp ice-white core -> backlit silver-cyan crystals)
    //   kind >= 1.5: Full-Flow Methalox Raptor exhaust (violet-blue CH* core + incandescent Mach shock diamonds!)
    let core = exp(-7.0 * r2) * (1.0 - smoothstep(0.0, 0.38, age));
    let rcs_col = mix(
        vec3<f32>(0.58, 0.78, 0.98),
        vec3<f32>(0.96, 0.99, 1.00),
        clamp(core * 1.6 + (1.0 - age) * 0.45, 0.0, 1.0),
    );
    let air_col = mix(
        vec3<f32>(0.70, 0.82, 0.92),
        vec3<f32>(0.95, 0.98, 1.00),
        clamp(core * 1.2 + fine * 0.35, 0.0, 1.0),
    );
    let is_raptor = step(1.5, kind);
    let is_air = step(0.5, kind) * (1.0 - is_raptor);
    // Prandtl-Pack shock-diamond modulation along the supersonic exhaust core!
    let diamond = pow(max(cos(v.uv.y * 8.0 + seed * 1.5), 0.0), 3.0) * exp(-5.0 * v.uv.x * v.uv.x) * (1.0 - smoothstep(0.0, 0.55, age));
    let methalox_outer = mix(vec3<f32>(0.98, 0.48, 0.14), vec3<f32>(0.52, 0.44, 0.40), smoothstep(0.32, 0.92, age));
    let methalox_core = mix(vec3<f32>(0.58, 0.66, 1.00), vec3<f32>(1.00, 0.96, 0.86), clamp(diamond * 1.4 + core * 1.2, 0.0, 1.0));
    let raptor_col = mix(methalox_outer, methalox_core, clamp(exp(-3.6 * r2) * (1.0 - age * 0.72) + diamond * 0.65, 0.0, 1.0));

    let base_col = (rcs_col * (1.0 - is_air - is_raptor) + air_col * is_air) * mie + raptor_col * is_raptor * (1.15 + 0.55 * diamond);

    // Gentle sparkle of sunlit frozen moisture crystals in adiabatic cryo air vents!
    let crystal = is_air * smoothstep(0.84, 0.98, fine) * (1.0 - age) * 0.42;

    let max_alpha = mix(0.62, 0.84, is_raptor);
    let alpha = clamp(opacity * radial * structure, 0.0, max_alpha);
    let rgb = (base_col + vec3<f32>(crystal)) * alpha;
    return vec4<f32>(rgb, alpha);
}
