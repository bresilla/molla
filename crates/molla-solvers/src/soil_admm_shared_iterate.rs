use super::*;

pub(super) fn slots(dofs: u32) -> Result<u32> {
    if !(1..=32).contains(&dofs) {
        return Err(invalid());
    }
    let available = (38400u32 - 5408 - 1024 * dofs.max(4)) / 72;
    Ok((1 << available.ilog2()).min(256))
}

pub(super) fn source(dofs: u32) -> Result<String> {
    let capacity = slots(dofs)?;
    let mut source = shader_source(dofs)?;
    let stores = [
        (
            "scratch[c].local=Six(vec4<f32>(local.a*rk+local.b*rw,0.0),vec4<f32>(local.b*rk+local.a*rw,0.0));",
            "store_local(c,slot,Six(vec4<f32>(local.a*rk+local.b*rw,0.0),vec4<f32>(local.b*rk+local.a*rw,0.0)));",
        ),
        (
            "scratch[c].z=Six(vec4<f32>(zk,0.0),vec4<f32>(zw,0.0));",
            "store_z(c,slot,Six(vec4<f32>(zk,0.0),vec4<f32>(zw,0.0)));",
        ),
        (
            "scratch[c].dual=Six(vec4<f32>(uk,0.0),vec4<f32>(uw,0.0));",
            "store_dual(c,slot,Six(vec4<f32>(uk,0.0),vec4<f32>(uw,0.0)));",
        ),
    ];
    for (old, new) in stores {
        assert_eq!(source.matches(old).count(), 1);
        source = source.replace(old, new);
    }
    for field in ["z", "dual", "local"] {
        source = source.replace(
            &format!("scratch[c].{field}"),
            &format!("load_{field}(c,slot)"),
        );
    }
    let start = source.find("    for(var iteration=0u;").unwrap();
    let end = source[start..]
        .find("    for(var slot=lane;slot<selected;slot+=64u) {let c=active_contact(slot);states")
        .unwrap()
        + start;
    assert_eq!(source[start..end].matches("storageBarrier();").count(), 2);
    let iteration = source[start..end].replace(
        "storageBarrier();",
        "if shared_iteration {workgroupBarrier();}else {storageBarrier();}",
    );
    source.replace_range(start..end, &iteration);
    let marker = "    let selected=select(params.contacts,workgroupUniformLoad(&active_count),params.pad!=0u);";
    assert_eq!(source.matches(marker).count(), 1);
    source = source.replace(marker, &format!("{marker}\n    let shared_iteration=params.pad==3u && selected<=STATE_CONTACTS;\n    if shared_iteration {{\n        for(var slot=lane;slot<selected;slot+=64u) {{for(var axis=0u;axis<18u;axis++) {{iteration_state[axis*STATE_CONTACTS+slot]=0.0;}}}}\n    }}\n    workgroupBarrier();"));
    let mut helpers = format!(
        "const STATE_CONTACTS:u32={capacity}u;\nvar<workgroup> iteration_state:array<f32,{}>;\n",
        18 * capacity
    );
    helpers.push_str(
        "fn state_cached()->bool {return params.pad==3u && active_count<=STATE_CONTACTS;}\n",
    );
    for (field, offset) in [("z", 0), ("dual", 6), ("local", 12)] {
        helpers.push_str(&format!("fn load_{field}(c:u32,slot:u32)->Six {{\n    if !state_cached() {{return scratch[c].{field};}}\n    var value=Six();\n    for(var axis=0u;axis<3u;axis++) {{value.k[axis]=iteration_state[({offset}u+axis)*STATE_CONTACTS+slot];value.w[axis]=iteration_state[({offset}u+3u+axis)*STATE_CONTACTS+slot];}}\n    return value;\n}}\nfn store_{field}(c:u32,slot:u32,value:Six) {{\n    if !state_cached() {{scratch[c].{field}=value;return;}}\n    for(var axis=0u;axis<3u;axis++) {{iteration_state[({offset}u+axis)*STATE_CONTACTS+slot]=value.k[axis];iteration_state[({offset}u+3u+axis)*STATE_CONTACTS+slot]=value.w[axis];}}\n}}\n"));
    }
    Ok(format!("{helpers}{source}"))
}

#[test]
fn admm_shared_iterate_layout_and_ownership() {
    for n in [0, 33, u32::MAX] {
        assert!(source(n).is_err());
    }
    assert_eq!(slots(10).unwrap(), 256);
    assert_eq!(slots(32).unwrap(), 2);
    for n in 1..=32 {
        let source = source(n).unwrap();
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let mut layout = naga::proc::Layouter::default();
        layout.update(module.to_ctx()).unwrap();
        let bytes: u32 = module
            .global_variables
            .iter()
            .filter(|(_, v)| v.space == naga::AddressSpace::WorkGroup)
            .map(|(_, v)| layout[v.ty].size.div_ceil(16) * 16)
            .sum();
        assert!(bytes <= 38400, "width={n} bytes={bytes}");
        let capacity = slots(n).unwrap();
        for count in [0, 1, capacity - 1, capacity, capacity + 1, 16384] {
            if count > capacity {
                continue;
            }
            let mut visits = vec![0; (18 * capacity) as usize];
            for lane in 0..64 {
                for slot in (lane..count).step_by(64) {
                    for axis in 0..18 {
                        visits[(axis * capacity + slot) as usize] += 1;
                    }
                }
            }
            assert!(
                visits
                    .iter()
                    .enumerate()
                    .all(|(i, &v)| v == u32::from((i as u32) % capacity < count))
            );
        }
        assert!(source.contains("iteration<params.iterations"));
        assert_eq!(
            source
                .matches("if shared_iteration {workgroupBarrier();}else {storageBarrier();}")
                .count(),
            2
        );
        if n == 10 {
            if let Some(path) = std::env::var_os("MOLLA_ADMM_SHARED_ITERATE_ARTIFACT") {
                std::fs::write(path, source).unwrap();
            }
        }
    }
}
