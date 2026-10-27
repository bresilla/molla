use super::*;

pub(super) fn slots(dofs: u32) -> Result<u32> {
    if !(1..=32).contains(&dofs) {
        return Err(invalid());
    }
    let fixed_bytes = 128 * dofs + 32 * dofs.div_ceil(4) + 1056;
    let available = (38400u32 - fixed_bytes) / (12 * dofs) - 1;
    Ok(available.min(512) & !1)
}

pub(super) fn source(dofs: u32) -> Result<String> {
    let capacity = slots(dofs)?;
    let size = (3 * (capacity + 1) * dofs).max(3 * 64 * dofs.max(4));
    let mut source = shader_source(dofs)?;
    for (old, new) in [
        (
            "core:array<f32,1024>".to_owned(),
            format!("core:array<f32,{}>", 32 * dofs),
        ),
        (
            "pivots:array<u32,32>".to_owned(),
            format!("pivots:array<u32,{dofs}>"),
        ),
        ("q:array<f32,32>".to_owned(), format!("q:array<f32,{dofs}>")),
    ] {
        assert_eq!(source.matches(&old).count(), 1);
        source = source.replace(&old, &new);
    }
    let array = format!(
        "var<workgroup> staged_responses:array<vec4<f32>,{}>;",
        64 * dofs.max(4)
    );
    assert_eq!(source.matches(&array).count(), 1);
    source = source.replace(
        &array,
        &format!("var<workgroup> response_values:array<f32,{size}>;"),
    );
    let replacements = [
        (
            "staged_responses[axis*64u+lane]=vec4<f32>(0.0);",
            "stage_response(axis*64u+lane,vec4<f32>(0.0));",
        ),
        (
            "staged_responses[axis*64u+lane]=responses[c*n+axis];",
            "stage_response(axis*64u+lane,responses[c*n+axis]);",
        ),
        (
            "staged_responses[axis*64u+lane]=responses[c*n+base+axis];",
            "stage_response(axis*64u+lane,responses[c*n+base+axis]);",
        ),
        (
            "staged_responses[lane*64u+offset].xyz",
            "staged_response(lane*64u+offset)",
        ),
    ];
    for (old, new) in replacements {
        assert!(source.contains(old));
        source = source.replace(old, new);
    }
    assert!(!source.contains("staged_responses"));
    let marker = "fn staged_rates_wide(lane:u32,selected:u32,local:bool) {";
    assert_eq!(source.matches(marker).count(), 1);
    source = source.replace(marker, &format!("{marker}\n    if params.pad==3u && selected<=RESPONSE_CONTACTS {{cached_rates(lane,selected,local);return;}}"));
    let marker = "    let selected=select(params.contacts,workgroupUniformLoad(&active_count),params.pad!=0u);";
    assert_eq!(source.matches(marker).count(), 1);
    source = source.replace(marker, &format!("{marker}\n    if params.pad==3u && selected<=RESPONSE_CONTACTS {{\n        for(var slot=lane;slot<selected;slot+=64u) {{\n            let c=active_contact(slot);\n            for(var axis=0u;axis<n;axis++) {{\n                let value=responses[c*n+axis].xyz;\n                for(var component=0u;component<3u;component++) {{response_values[(3u*axis+component)*(RESPONSE_CONTACTS+1u)+slot]=value[component];}}\n            }}\n        }}\n    }}\n    workgroupBarrier();"));
    let helpers = r#"
fn stage_response(index:u32,value:vec4<f32>) {
    for(var component=0u;component<3u;component++) {response_values[3u*index+component]=value[component];}
}
fn staged_response(index:u32)->vec3<f32> {
    return vec3<f32>(response_values[3u*index],response_values[3u*index+1u],response_values[3u*index+2u]);
}
fn cached_response(slot:u32,axis:u32)->vec3<f32> {
    let stride=RESPONSE_CONTACTS+1u;let base=3u*axis*stride+slot;
    return vec3<f32>(response_values[base],response_values[base+stride],response_values[base+2u*stride]);
}
fn cached_rates(lane:u32,selected:u32,local:bool) {
    var value=0.0;
    if lane<params.width && !local {value=free_rates[lane].x+free_rates[lane].y;}
    for(var first=0u;first<selected;first+=64u) {
        let slot=first+lane;
        staged_points[lane]=vec4<f32>(0.0);
        if slot<selected {
            let c=active_contact(slot);
            if local {staged_points[lane]=scratch[c].local.k;}else {staged_points[lane]=scratch[c].z.k;}
        }
        workgroupBarrier();
        if lane<params.width {
            for(var offset=0u;offset<min(64u,selected-first);offset++) {
                if local {value+=dot(cached_response(first+offset,lane),staged_points[offset].xyz);}
                else {value-=dot(cached_response(first+offset,lane),staged_points[offset].xyz);}
            }
        }
        workgroupBarrier();
    }
    if lane<params.width {q[lane]=value;}
}
"#;
    Ok(format!(
        "const RESPONSE_CONTACTS:u32={capacity}u;\n{helpers}{source}"
    ))
}

#[test]
fn admm_response_cache_layout_and_order() {
    for n in [0, 33, u32::MAX] {
        assert!(source(n).is_err());
    }
    assert_eq!(slots(10).unwrap(), 298);
    assert_eq!(slots(32).unwrap(), 84);
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
        let cap = slots(n).unwrap();
        assert_eq!(cap % 2, 0);
        assert!(cap >= 64);
        for i in 0..n {
            for j in 0..n {
                assert!(i * 32 + j < 32 * n);
            }
        }
        let mut data = vec![u32::MAX; (3 * n * (cap + 1)) as usize];
        for lane in 0..64 {
            for slot in (lane..cap).step_by(64) {
                for axis in 0..n {
                    for component in 0..3 {
                        data[((3 * axis + component) * (cap + 1) + slot) as usize] =
                            (slot * n + axis) * 3 + component;
                    }
                }
            }
        }
        for axis in 0..n {
            for slot in 0..cap {
                for component in 0..3 {
                    assert_eq!(
                        data[((3 * axis + component) * (cap + 1) + slot) as usize],
                        (slot * n + axis) * 3 + component
                    );
                }
            }
        }
        assert!(source.contains("iteration<params.iterations"));
        assert!(
            source.contains("dot(cached_response(first+offset,lane),staged_points[offset].xyz)")
        );
        assert!(source.contains("for(var slot=0u;slot<selected;slot++)"));
        if n == 10 {
            if let Some(path) = std::env::var_os("MOLLA_ADMM_RESPONSE_CACHE_ARTIFACT") {
                std::fs::write(path, source).unwrap();
            }
        }
    }
}
