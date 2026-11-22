struct Camera { eye: vec4<f32>, right: vec4<f32>, up: vec4<f32>, forward: vec4<f32>, projection: vec4<f32> }
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var shadows: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(0) @binding(3) var albedo: texture_2d_array<f32>;
@group(0) @binding(4) var material_sampler: sampler;
@group(0) @binding(5) var leaf_albedo: texture_2d_array<f32>;
@group(0) @binding(6) var leaf_normals: texture_2d_array<f32>;
@group(0) @binding(7) var bark_albedo: texture_2d_array<f32>;
@group(0) @binding(8) var bark_normals: texture_2d_array<f32>;
@group(0) @binding(9) var leaf_sampler: sampler;
@group(0) @binding(10) var bark_sampler: sampler;
struct Vertex { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) color: vec4<f32>, @location(3) uv: vec2<f32> }
struct Surface { @builtin(position) position: vec4<f32>, @location(0) world: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) color: vec4<f32>, @location(3) shadow: vec3<f32>, @location(4) uv: vec2<f32>, @location(5) @interpolate(flat) material: f32 }
fn light_position(p: vec3<f32>) -> vec3<f32> {
    let forward = normalize(vec3<f32>(-0.45,0.8,0.5));
    let right = normalize(cross(vec3<f32>(0.0,1.0,0.0),forward)); let up = cross(forward,right);
    let relative = p-vec3<f32>(0.0,1.0,0.0);
    let extent = max(3.5,camera.projection.w);
    let depth = max(8.0,2.0*extent);
    return vec3<f32>(dot(relative,right)/extent,dot(relative,up)/extent,(depth-dot(relative,forward))/(2.0*depth));
}
@vertex fn shadow_vertex(v: Vertex) -> @builtin(position) vec4<f32> { return vec4<f32>(light_position(v.position),1.0); }
fn project(world: vec3<f32>, normal: vec3<f32>, color: vec4<f32>, uv: vec2<f32>) -> Surface {
    let delta = world-camera.eye.xyz; let depth = -dot(delta,camera.forward.xyz);
    var out: Surface;
    out.position = vec4<f32>(dot(delta,camera.right.xyz)*camera.projection.x,dot(delta,camera.up.xyz)*2.5,
        (100.0*depth-5.0)/99.95,depth);
    out.world = world; out.normal = normal; out.color = color; out.shadow = light_position(world);
    out.uv = uv; out.material = color.a;
    return out;
}
@vertex fn vertex(v: Vertex) -> Surface { return project(v.position,v.normal,v.color,v.uv); }
// Canonical leaf vertex: blade fraction t, lateral offset x (blade lengths), face ±1,
// kind 0 blade / 1–3 petiole corners. Instances carry one leaf's pose and look.
struct LeafVertex { @location(0) shape: vec4<f32> }
struct LeafInstance {
    @location(1) center_length: vec4<f32>,
    @location(2) direction_code: vec4<f32>,
    @location(3) normal_flags: vec4<f32>,
    @location(4) anchor_bulge: vec4<f32>,
    @location(5) tint_span: vec4<f32>,
}
const LEAF_MIRROR: u32 = 1u;
const LEAF_DROOP: u32 = 2u;
const LEAF_CAMBER: u32 = 4u;
const LEAF_PLATE: u32 = 8u;
const LEAF_PETIOLE: u32 = 16u;
// World position and unit normal (towards the instance normal) of a canonical leaf vertex.
fn leaf_geometry(v: LeafVertex, i: LeafInstance) -> array<vec3<f32>,2> {
    let d = i.direction_code.xyz; let n = i.normal_flags.xyz; let s = normalize(cross(d,n));
    let length = i.center_length.w; let center = i.center_length.xyz;
    let flags = u32(i.normal_flags.w+0.5);
    let t = v.shape.x; let x = v.shape.y; let kind = u32(v.shape.w+0.5);
    let base = center-d*(0.5*length);
    if kind > 0u {
        var corner = base;
        if (flags & LEAF_PETIOLE) != 0u && kind < 3u {
            corner = i.anchor_bulge.xyz+s*select(0.0007,-0.0007,kind == 1u);
        }
        return array<vec3<f32>,2>(corner,n*v.shape.z);
    }
    var world: vec3<f32>; var along: vec3<f32>; var across: vec3<f32>;
    if (flags & LEAF_PLATE) != 0u {
        let width = i.tint_span.w; let bulge = i.anchor_bulge.w;
        let u = t-0.5; let z = x*length/width;
        world = center+d*(u*length)+s*(x*length)+n*(bulge*(4.0*u*u+4.0*z*z-0.5));
        along = d*length+n*(8.0*bulge*u);
        across = s*length+n*(8.0*bulge*z*length/width);
    } else {
        let droop = (flags & LEAF_DROOP) != 0u; let camber = (flags & LEAF_CAMBER) != 0u;
        let height = select(0.0,0.48*x*x,camber)-select(0.0,0.12*t*t,droop);
        world = base+d*(t*length)+s*(x*length)+n*(height*length);
        along = d*length-n*(select(0.0,0.24*t,droop)*length);
        across = s*length+n*(select(0.0,0.96*x,camber)*length);
    }
    var normal = normalize(cross(across,along));
    if dot(normal,n) < 0.0 { normal = -normal; }
    return array<vec3<f32>,2>(world,normal*v.shape.z);
}
@vertex fn leaf_vertex(v: LeafVertex, i: LeafInstance) -> Surface {
    let geometry = leaf_geometry(v,i);
    let kind = u32(v.shape.w+0.5);
    let flags = u32(i.normal_flags.w+0.5);
    var color = vec4<f32>(113.0/255.0,127.0/255.0,48.0/255.0,6.0);
    var uv = vec2<f32>(0.0);
    if kind == 0u {
        let lateral = select(v.shape.y,-v.shape.y,(flags & LEAF_MIRROR) != 0u);
        uv = vec2<f32>(0.5+0.5*lateral/0.33,1.0-v.shape.x);
        color = vec4<f32>(i.tint_span.xyz,select(5.0,4.0,v.shape.z > 0.0)+i.direction_code.w);
    }
    return project(geometry[0],geometry[1],color,uv);
}
@vertex fn leaf_shadow_vertex(v: LeafVertex, i: LeafInstance) -> @builtin(position) vec4<f32> {
    return vec4<f32>(light_position(leaf_geometry(v,i)[0]),1.0);
}
fn shadow_visibility(p: vec3<f32>,normal: vec3<f32>,bias_scale: f32) -> f32 {
    let uv = p.xy*vec2<f32>(0.5,-0.5)+vec2<f32>(0.5);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) { return 1.0; }
    let bias = bias_scale*(0.00025+0.0005*(1.0-abs(normal.y))); var visibility = 0.0;
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            visibility += textureSampleCompareLevel(shadows,shadow_sampler,uv+vec2<f32>(f32(x),f32(y))/2048.0,p.z-bias);
        }
    }
    return visibility/9.0;
}
fn hash(p: vec3<f32>) -> f32 { return fract(sin(dot(p,vec3<f32>(127.1,311.7,74.7)))*43758.5453); }
fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p); let f = fract(p); let u = f*f*(3.0-2.0*f);
    let e = vec2<f32>(1.0,0.0);
    let near = mix(mix(hash(i),hash(i+e.xyy),u.x),mix(hash(i+e.yxy),hash(i+e.xxy),u.x),u.y);
    let far = mix(mix(hash(i+e.yyx),hash(i+e.xyx),u.x),mix(hash(i+e.yxx),hash(i+e.xxx),u.x),u.y);
    return mix(near,far,u.z);
}
fn perturb(n: vec3<f32>,dpx: vec3<f32>,dpy: vec3<f32>,duvx: vec2<f32>,duvy: vec2<f32>,map: vec3<f32>) -> vec3<f32> {
    let px = cross(n,dpx); let py = cross(dpy,n);
    let t = py*duvx.x+px*duvy.x; let b = py*duvx.y+px*duvy.y;
    let scale = inverseSqrt(max(max(dot(t,t),dot(b,b)),1e-24));
    return normalize(t*(scale*map.x)-b*(scale*map.y)+n*map.z);
}
fn bark_layer(uv: vec2<f32>,duvx: vec2<f32>,duvy: vec2<f32>,layer: i32) -> array<vec4<f32>,2> {
    let stretch = vec2<f32>(1.0,select(1.0,2.0,layer == 2));
    let albedo = textureSampleGrad(bark_albedo,bark_sampler,uv*stretch,layer,duvx*stretch,duvy*stretch);
    let relief = textureSampleGrad(bark_normals,bark_sampler,uv*stretch,layer,duvx*stretch,duvy*stretch);
    let warm = select(vec3<f32>(1.0),vec3<f32>(1.85,0.95,0.75),layer == 2);
    return array<vec4<f32>,2>(vec4<f32>(albedo.rgb*warm,albedo.a),relief);
}
@fragment fn fragment(v: Surface) -> @location(0) vec4<f32> {
    let n0 = normalize(v.normal); var n = n0; let light = normalize(vec3<f32>(-0.45,0.8,0.5));
    let view = normalize(camera.eye.xyz-v.world); let halfway = normalize(light+view);
    var color = pow(v.color.rgb,vec3<f32>(2.2));
    let material = v.material;
    let duvx = dpdx(v.uv); let duvy = dpdy(v.uv);
    let dpx = dpdx(v.world); let dpy = dpdy(v.world);
    let texel = textureSample(albedo, material_sampler, v.uv, clamp(i32(material - 8.0), 0, 1));
    if material >= 8.0 && material < 10.0 {
        if texel.a < 0.05 { discard; }
        color = texel.rgb;
    }
    let ring_phase = length(v.world.yz-vec2<f32>(1.85,0.0))*650.0;
    let rings = sin(ring_phase)*(1.0-smoothstep(1.0,3.141593,fwidth(ring_phase)));
    var gloss = 0.12;
    if material > 0.5 && material < 1.5 {
        let thin = clamp((material-1.0)/0.45,0.0,1.0);
        let blend = clamp(thin*2.0+(value_noise(v.world*vec3<f32>(9.0,3.0,9.0))-0.5)*0.8,0.0,2.0);
        let low = min(i32(blend),1); let f = smoothstep(0.15,0.85,blend-f32(low));
        let a = bark_layer(v.uv,duvx,duvy,low); let b = bark_layer(v.uv,duvx,duvy,low+1);
        let detail = mix(a[1].xyz,b[1].xyz,f)*2.0-1.0;
        color = mix(a[0].rgb,b[0].rgb,f)*(2.0*v.color.rgb)*mix(0.72,1.0,smoothstep(0.55,0.95,detail.z));
        n = perturb(n0,dpx,dpy,duvx,duvy,detail);
        gloss = 0.03;
    }
    if material >= 4.0 && material < 6.0 {
        let code = u32(round(fract(material)*4096.0));
        let layer = i32(code%64u); let palette = i32(code/64u);
        let leaf = textureSampleGrad(leaf_albedo,leaf_sampler,v.uv,layer,duvx,duvy);
        if leaf.a < 0.5 { discard; }
        let last = f32(textureNumLevels(leaf_albedo)-1u);
        let own = textureSampleLevel(leaf_albedo,leaf_sampler,vec2<f32>(0.5),layer,last).rgb;
        let other = textureSampleLevel(leaf_albedo,leaf_sampler,vec2<f32>(0.5),palette,last).rgb;
        let transfer = mix(vec3<f32>(1.0),clamp(other/max(own,vec3<f32>(0.001)),vec3<f32>(0.6),vec3<f32>(1.6)),0.6);
        color = leaf.rgb*transfer*(2.0*v.color.rgb)*vec3<f32>(0.50,0.74,0.52);
        if material >= 5.0 {
            let grey = dot(color,vec3<f32>(0.30,0.59,0.11));
            color = mix(color,vec3<f32>(grey)*vec3<f32>(0.92,1.04,0.86),0.35)*1.45;
        }
        let detail = textureSampleGrad(leaf_normals,leaf_sampler,v.uv,layer,duvx,duvy).xyz*2.0-1.0;
        n = perturb(n0,dpx,dpy,duvx,duvy,vec3<f32>(detail.xy*0.6,detail.z));
    }
    if material > 1.5 && material < 2.5 {
        color *= 0.94+0.06*rings;
    }
    if material > 2.5 && material < 3.5 {
        let radius = length(v.world.xz);
        color *= 0.65+0.35*exp(-radius*radius*0.06);
    }
    let shadow = shadow_visibility(v.shadow,n0,select(1.0,2.0,material >= 10.0));
    let space = camera.projection.y > 0.5;
    if space && material > 14.5 && material < 15.5 {
        // Body-locked crinkled gold Kapton MLI thermal foil (UV space -> zero sliding!).
        let p_foil = vec3<f32>(v.uv * 16.0, 0.37);
        let p_fine = vec3<f32>(v.uv * 38.0, 4.19);
        let gu = (value_noise(p_foil + vec3<f32>(0.22, 0.0, 0.0)) - value_noise(p_foil - vec3<f32>(0.22, 0.0, 0.0)))
               + 0.45 * (value_noise(p_fine + vec3<f32>(0.22, 0.0, 0.0)) - value_noise(p_fine - vec3<f32>(0.22, 0.0, 0.0)));
        let gv = (value_noise(p_foil + vec3<f32>(0.0, 0.22, 0.0)) - value_noise(p_foil - vec3<f32>(0.0, 0.22, 0.0)))
               + 0.45 * (value_noise(p_fine + vec3<f32>(0.0, 0.22, 0.0)) - value_noise(p_fine - vec3<f32>(0.0, 0.22, 0.0)));
        n = perturb(n0, dpx, dpy, duvx, duvy, normalize(vec3<f32>(gu * 2.1, gv * 2.1, 1.0)));
    }
    if space && material > 10.5 && material < 11.5 {
        // Body-locked aerospace Whipple-shield hull panel seams on white MLI.
        let seam = smoothstep(0.015, 0.045, abs(fract(v.uv.x * 1.35) - 0.5))
                 * smoothstep(0.015, 0.045, abs(fract(v.uv.y * 1.35) - 0.5));
        color *= mix(0.89, 1.0, seam);
    }
    if space && material > 12.5 && material < 13.5 {
        // Body-locked photovoltaic solar cell wafer grid & subtle cell tint variation.
        let gu = smoothstep(0.03, 0.075, abs(fract(v.uv.x * 9.0) - 0.5));
        let gv = smoothstep(0.03, 0.075, abs(fract(v.uv.y * 9.0) - 0.5));
        let cell_var = 0.90 + 0.20 * hash(vec3<f32>(floor(v.uv * 9.0), 1.0));
        color = mix(vec3<f32>(0.38, 0.44, 0.54), color * cell_var, gu * gv);
    }
    let diffuse = max(dot(n,light),0.0)*shadow;
    let rim = pow(1.0-max(dot(n,view),0.0),3.0)*max(dot(n,normalize(vec3<f32>(0.5,0.3,-1.0))),0.0);
    let specular = pow(max(dot(n,halfway),0.0),select(36.0,110.0,material < 0.5))*shadow;
    let earthshine = vec3<f32>(0.07,0.16,0.30)*max(-n.y,0.0) + vec3<f32>(0.028,0.036,0.055);
    let studio_ambient = vec3<f32>(0.28,0.33,0.42)*(0.65+0.35*max(n.y,0.0));
    let sun_color = select(vec3<f32>(1.5,1.35,1.12), vec3<f32>(2.35,2.28,2.16), space);
    let lighting = select(studio_ambient, earthshine, space) + sun_color*diffuse;
    var result = color*lighting+vec3<f32>(gloss)*specular+vec3<f32>(0.13,0.18,0.24)*rim;
    if material > 12.5 && material < 13.5 {
        let facing = max(dot(n,view),0.0);
        let fresnel = 0.04+0.96*pow(1.0-facing,5.0);
        let glint = pow(max(dot(n,halfway),0.0),240.0)*shadow;
        let sky = select(vec3<f32>(0.42,0.50,0.60)*(0.6+0.4*max(n.y,0.0)), vec3<f32>(0.16,0.36,0.68)*(0.25+0.75*max(-n.y,0.0)), space);
        result = color*lighting*(0.35+0.25*facing)+vec3<f32>(3.4)*glint+sky*fresnel;
    }
    if (material >= 10.0 && material < 12.5) || (material > 14.5 && material < 15.5) {
        let metal = material >= 11.5;
        let foil = material > 14.5;
        let sheen = pow(max(dot(n,halfway),0.0),select(24.0,select(95.0,140.0,foil),metal))*shadow;
        let reflectance = select(vec3<f32>(0.018),mix(vec3<f32>(0.10),color,select(0.55,0.85,foil)),metal);
        result = color*lighting+reflectance*sheen+vec3<f32>(0.035,0.045,0.055)*rim;
    }
    if material > 3.5 && material < 7.5 {
        let transmission = max(dot(-n,light),0.0)*0.55;
        result = color*(vec3<f32>(0.44,0.50,0.36)+vec3<f32>(1.35,1.38,0.95)*(diffuse+transmission))
            +vec3<f32>(0.06)*specular;
    }
    if material >= 13.8 && material < 14.5 {
        // Emissive RCS plume / glowing docking beacon with soft edge falloff.
        let edge = pow(max(abs(dot(n, view)), 0.05), 0.75);
        result = color * (1.2 + 2.4 * edge);
    }
    result = result/(vec3<f32>(1.0)+result);
    if !space {
        let fog_scale = 3.5/max(3.5,camera.projection.w);
        let fog = 1.0-exp(-0.008*fog_scale*fog_scale*dot(v.world.xz,v.world.xz));
        result = mix(result,pow(vec3<f32>(15.0,21.0,31.0)/255.0,vec3<f32>(2.2)),fog);
    }
    return vec4<f32>(pow(result,vec3<f32>(1.0/2.2)),1.0);
}
