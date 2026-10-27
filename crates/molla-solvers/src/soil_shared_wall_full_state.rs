pub(super) fn selected() -> bool {
    depth() != 0
}

pub(super) fn depth() -> u64 {
    match mode() {
        3 => 2,
        depth => depth,
    }
}

fn mode() -> u64 {
    #[cfg(test)]
    match std::env::var("MOLLA_SHARED_FULL_STATE_DIAGNOSTIC").as_deref() {
        Ok("0") => return 0,
        Ok("1") => return 1,
        Ok("2") => return 2,
        Ok("3") => return 3,
        Ok(_) => panic!("unknown full-state diagnostic history depth"),
        Err(_) => {}
    }
    0
}

pub(super) fn source(source: &str) -> String {
    let mut source = source.to_owned();
    let begin = source.find("struct SharedHistory {").unwrap();
    let end = begin + source[begin..].find('}').unwrap();
    let extra = if depth() == 2 {
        ", full_older_fp:vec4<f32>, full_older_fw:vec4<f32>, full_older_p:vec4<f32>, full_older_w:vec4<f32> "
    } else {
        ""
    };
    source.insert_str(end, &format!(", full_start_p:vec4<f32>, full_start_w:vec4<f32>, full_previous_p:vec4<f32>, full_previous_w:vec4<f32>{extra} "));
    let prepare = "    shared_outputs[c]=Output();";
    assert_eq!(source.matches(prepare).count(), 1);
    source=source.replace(prepare,&format!("{prepare}\n    shared_history[c].previous_k=vec4<f32>(0.0);\n    shared_history[c].previous_w=vec4<f32>(0.0);\n    shared_history[c].full_previous_p=vec4<f32>(0.0);\n    shared_history[c].full_previous_w=vec4<f32>(0.0);"));
    let start = "let old_k=-shared_outputs[c].machine_impulse_normal.xyz;";
    assert!(source.contains(start));
    source=source.replace(start,&format!("{start}\n            shared_history[c].full_start_p=shared_outputs[c].machine_impulse_normal;\n            shared_history[c].full_start_w=shared_outputs[c].wall_impulse;"));
    if depth() == 2 {
        return depth_two(source);
    }
    let terms = "    let top=numerator.x+numerator.y; let bottom=denominator.x+denominator.y;";
    assert_eq!(source.matches(terms).count(), 2);
    source.replace(
        terms,
        &format!(
            "{}\n{terms}",
            include_str!("../shaders/soil_shared_wall_full_state.wgsl")
        ),
    )
}

fn depth_two(mut source: String) -> String {
    let status = "control:vec4<u32> }";
    assert_eq!(source.matches(status).count(), 1);
    source = source.replace(status, "control:vec4<u32>, full_older_q:array<f32,32> }");
    let begin = source.find("fn coop_coefficient(").unwrap();
    let end = source[begin..].find("fn coop_local_sweep(").unwrap() + begin;
    source.replace_range(
        begin..end,
        include_str!("../shaders/soil_shared_wall_full_state_two.wgsl"),
    );
    let coefficient = "if coop_lane==0u { coop_trial=coop_coefficient(coop_start,has_previous); }";
    assert_eq!(source.matches(coefficient).count(), 1);
    source = source.replace(
        coefficient,
        &format!("{coefficient}\n    storageBarrier();"),
    );
    source = source.replace(
        "if coefficient!=0.0 {",
        "if coefficient!=0.0 || bitcast<f32>(shared_status.control.w)!=0.0 {",
    );
    for (kind, old) in [("p", "k"), ("w", "w")] {
        let trial =
            format!("{kind}.xyz+coefficient*({kind}.xyz-shared_history[c].previous_{old}.xyz)");
        assert!(source.contains(&trial));
        source=source.replace(&trial,&format!("{trial}+bitcast<f32>(shared_status.control.w)*(shared_history[c].previous_{old}.xyz-shared_history[c].full_older_{kind}.xyz)"));
        let save = format!("shared_history[c].previous_{old}=shared_history[c].baseline_{old};");
        source=source.replace(&save,&format!("shared_history[c].full_older_{kind}=shared_history[c].previous_{old};\n        {save}"));
    }
    let clear = "    shared_outputs[c]=Output();";
    source=source.replace(clear,&format!("{clear}\n    shared_history[c].full_older_p=vec4<f32>(0.0);\n    shared_history[c].full_older_w=vec4<f32>(0.0);\n    shared_history[c].full_older_fp=vec4<f32>(0.0);\n    shared_history[c].full_older_fw=vec4<f32>(0.0);"));
    if mode() == 3 {
        portfolio(source)
    } else {
        source
    }
}

fn portfolio(mut source: String) -> String {
    let start = source.find("fn coop_coefficient(").unwrap();
    let end = start + source[start..].find("fn coop_local_sweep(").unwrap();
    let mut body = source[start..end].to_owned();
    let contact = "    for (var slot=0u;slot<shared_status.control.y;slot++) {";
    assert_eq!(body.matches(contact).count(), 1);
    body=body.replace(contact,&format!("    var rate_coefficient=0.0;\n    let rate_top=r1.x+r1.y; let rate_bottom=a.x+a.y;\n    if has_previous && finite(rate_top) && finite(rate_bottom) && rate_bottom>1e-30 {{ rate_coefficient=clamp(-rate_top/rate_bottom,-2.0,2.0); }}\n    if !finite(rate_coefficient) {{ rate_coefficient=0.0; }}\n    shared_status.control.z=bitcast<u32>(rate_coefficient);\n{contact}"));
    source.replace_range(start..end, &body);
    let coefficient = "    let coefficient=workgroupUniformLoad(&coop_trial);";
    assert_eq!(source.matches(coefficient).count(), 1);
    source=source.replace(coefficient,&format!("{coefficient}\n    if coop_lane==0u {{ coop_terms[0]=bitcast<f32>(shared_status.control.z); }}\n    let rate_coefficient=workgroupUniformLoad(&coop_terms[0]);"));
    source=source.replace("if coefficient!=0.0 || bitcast<f32>(shared_status.control.w)!=0.0 {","if coefficient!=0.0 || bitcast<f32>(shared_status.control.w)!=0.0 || bitcast<f32>(shared_status.control.z)!=0.0 {");
    let candidate = "        let candidate=coop_natural_merit();";
    assert_eq!(source.matches(candidate).count(), 1);
    source.replace(
        candidate,
        include_str!("../shaders/soil_shared_wall_full_state_portfolio.wgsl"),
    )
}
