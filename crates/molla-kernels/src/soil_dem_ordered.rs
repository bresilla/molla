use super::*;

pub(super) fn source() -> String {
    let original = include_str!("../shaders/tire_dem.wgsl");
    let start = original.find("fn step_dem_lbvh(").unwrap();
    let end = start + original[start..].find("\n@compute").unwrap();
    let mut body = original[start..end].to_string();
    let begin = body
        .find("                        if AUDIT_CONTACTS {")
        .unwrap();
    let finish = begin + body[begin..].find("\n                    }\n").unwrap();
    body.replace_range(
        begin..finish,
        r#"
                        let distance = length(p0.xyz - pos_in[j].xyz);
                        if distance >= 1e-9 && distance < grain_radius * 2.0 {
                            let hit = vec2<u32>(ordered_key(pos_in[j].xyz), j);
                            if !has_after || ordered_less(after, hit) {
                                var at = used;
                                for (var h = 0u; h < used; h++) {
                                    if ordered_less(hit, hits[h]) { at = h; break; }
                                }
                                if at < 32u {
                                    var h = min(used, 31u);
                                    while h > at { hits[h] = hits[h - 1u]; h--; }
                                    hits[at] = hit;
                                    used = min(used + 1u, 32u);
                                }
                            }
                        }
"#,
    );
    let marker = "    if (u.dims.z > 0u) {";
    assert_eq!(body.matches(marker).count(), 1);
    body = body.replace(
        marker,
        &format!(
            r#"
    var after = vec2<u32>(0u);
    var has_after = false;
    loop {{
        var hits: array<vec2<u32>, 32>;
        var used = 0u;
{marker}"#
        ),
    );
    body = body.replace(
        "    if AUDIT_CONTACTS { return; }",
        r#"
        for (var h = 0u; h < used; h++) {
            let j = hits[h].y;
            force += pair_force(p0.xyz, v0.xyz, pos_in[j].xyz, vel_in[j].xyz,
                grain_radius * 2.0, normal_stiffness, damping, friction);
        }
        if used < 32u { break; }
        after = hits[used - 1u];
        has_after = true;
    }
"#,
    );
    format!(
        r#"{}{}{}
fn ordered_less(a:vec2<u32>, b:vec2<u32>)->bool {{ return a.x < b.x || (a.x == b.x && a.y < b.y); }}
fn ordered_key(p:vec3<f32>)->u32 {{
    let scene_min = u.grid_origin_cell_size.xyz;
    let extent = max(bitcast<vec3<f32>>(u.grid_shape.xyz), vec3<f32>(1.0e-9));
    let normalized = clamp((p - scene_min) / extent, vec3<f32>(0.0), vec3<f32>(1.0));
    let q = vec3<u32>(round(normalized * 1023.0));
    return morton3_10(q.x, q.y, q.z);
}}
"#,
        &original[..start],
        body,
        &original[end..]
    )
}

pub(super) fn pipeline(device: &wgpu::Device, kernel: &TireDemGpuKernel) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("DEM ordered refit"),
        source: wgpu::ShaderSource::Wgsl(source().into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("DEM ordered refit"),
        bind_group_layouts: &[
            Some(&kernel.layout_lbvh_io),
            Some(&kernel.layout_step_uniforms),
        ],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("DEM ordered refit"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("step_dem_lbvh"),
        compilation_options: Default::default(),
        cache: None,
    })
}
