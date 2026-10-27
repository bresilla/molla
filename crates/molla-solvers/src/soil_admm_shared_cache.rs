use super::*;

pub(super) fn padded_source(dofs: u32) -> Result<String> {
    let source = shader_source(dofs)?;
    if dofs == 32 {
        return Ok(source);
    }
    let width = dofs.max(4);
    let old = format!("staged_responses:array<vec4<f32>,{}>", 64 * width);
    assert_eq!(source.matches(&old).count(), 1);
    assert_eq!(source.matches("axis*64u+lane").count(), 4);
    assert_eq!(source.matches("lane*64u+offset").count(), 4);
    Ok(source
        .replace(
            &old,
            &format!("staged_responses:array<vec4<f32>,{}>", 65 * width),
        )
        .replace("axis*64u+lane", "axis*65u+lane")
        .replace("lane*64u+offset", "lane*65u+offset"))
}

#[test]
fn admm_padded_staging_layout_and_order() {
    for dofs in 1..=32 {
        let source = padded_source(dofs).unwrap();
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
        assert!(bytes <= 38400, "width={dofs} bytes={bytes}");
        let stride = if dofs == 32 { 64 } else { 65 };
        let mut tile = vec![u32::MAX; stride * dofs.max(4) as usize];
        for axis in 0..dofs as usize {
            for lane in 0..64 {
                tile[axis * stride + lane] = (axis * 64 + lane) as u32;
            }
        }
        for lane in 0..dofs as usize {
            for offset in 0..64 {
                assert_eq!(tile[lane * stride + offset], (lane * 64 + offset) as u32);
            }
        }
        let reversed = source
            .replace(
                &format!(
                    "staged_responses:array<vec4<f32>,{}>",
                    stride * dofs.max(4) as usize
                ),
                &format!("staged_responses:array<vec4<f32>,{}>", 64 * dofs.max(4)),
            )
            .replace("axis*65u+lane", "axis*64u+lane")
            .replace("lane*65u+offset", "lane*64u+offset");
        assert_eq!(reversed, shader_source(dofs).unwrap());
        if dofs == 10 {
            if let Some(path) = std::env::var_os("MOLLA_ADMM_PADDED_ARTIFACT") {
                std::fs::write(path, source).unwrap();
            }
        }
    }
    assert_eq!(padded_source(32).unwrap(), shader_source(32).unwrap());
}

pub(super) fn slots(dofs: u32) -> Result<u32> {
    if !(1..=32).contains(&dofs) {
        return Err(invalid());
    }
    let available = (32 - dofs.max(4)) * 16;
    Ok(if available == 0 {
        0
    } else {
        (1 << available.ilog2()).min(256)
    })
}

pub(super) fn source(dofs: u32) -> Result<String> {
    let capacity = slots(dofs)?;
    let mut source = shader_source(dofs)?;
    let replacements = [
        (
            "fn weights(c:u32)->Weights {",
            "fn weights(c:u32,slot:u32)->Weights {\n    if params.pad==3u && slot<ONCHIP_CONTACTS {return Weights(onchip_contacts[slot].a.xyz,onchip_contacts[slot].b.xyz);}",
        ),
        (
            "fn contact_normal(c:u32)->vec3<f32> {",
            "fn contact_normal(c:u32,slot:u32)->vec3<f32> {\n    if params.pad==3u && slot<ONCHIP_CONTACTS {return onchip_contacts[slot].normal_rho.xyz;}",
        ),
        (
            "fn contact_rho(c:u32)->f32 {",
            "fn contact_rho(c:u32,slot:u32)->f32 {\n    if params.pad==3u && slot<ONCHIP_CONTACTS {return onchip_contacts[slot].normal_rho.w;}",
        ),
        (
            "fn contact_free(c:u32,normal:vec3<f32>)->vec3<f32> {",
            "fn contact_free(c:u32,normal:vec3<f32>,slot:u32)->vec3<f32> {\n    if params.pad==3u && slot<ONCHIP_CONTACTS {return onchip_contacts[slot].free.xyz;}",
        ),
        ("weights(c).a", "weights(c,slot).a"),
        ("let local=weights(c);", "let local=weights(c,slot);"),
        ("contact_normal(c)", "contact_normal(c,slot)"),
        ("contact_rho(c)", "contact_rho(c,slot)"),
        ("contact_free(c,normal)", "contact_free(c,normal,slot)"),
    ];
    for (old, new) in replacements {
        assert!(
            source.contains(old),
            "missing shared-cache source contract: {old}"
        );
        source = source.replace(old, new);
    }
    let marker = "    if params.pad==4u {\n        for(var slot=lane;slot<selected;slot+=64u) {";
    assert_eq!(source.matches(marker).count(), 1);
    let fill = "    if params.pad==3u {\n        for(var slot=lane;slot<min(selected,ONCHIP_CONTACTS);slot+=64u) {\n            let c=active_contact(slot);let normal=normalize(contacts[c].normal_gap.xyz);let local=uncached_weights(c);\n            onchip_contacts[slot]=FixedContact(vec4<f32>(normal,1.0/contacts[c].soil_mass.w),vec4<f32>(uncached_free(c,normal),0.0),vec4<f32>(local.a,0.0),vec4<f32>(local.b,0.0));\n        }\n    }\n    workgroupBarrier();\n";
    source = source.replace(marker, &format!("{fill}{marker}"));
    Ok(format!(
        "const ONCHIP_CONTACTS:u32={capacity}u;\nvar<workgroup> onchip_contacts:array<FixedContact,{}>;\n{source}",
        capacity.max(1)
    ))
}

#[test]
fn admm_shared_cache_layout_and_boundaries() {
    for dofs in [0, 33, u32::MAX] {
        assert!(source(dofs).is_err());
    }
    assert_eq!(slots(10).unwrap(), 256);
    assert_eq!(slots(16).unwrap(), 256);
    assert_eq!(slots(17).unwrap(), 128);
    assert_eq!(slots(24).unwrap(), 128);
    assert_eq!(slots(25).unwrap(), 64);
    assert_eq!(slots(31).unwrap(), 16);
    assert_eq!(slots(32).unwrap(), 0);
    for dofs in 1..=32 {
        let source = source(dofs).unwrap();
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
        assert!(bytes <= 38400, "width={dofs} bytes={bytes}");
        for count in [0, 1, 255, 256, 257, 4096, 16384] {
            let cached = slots(dofs).unwrap().min(count);
            let mut visits = vec![0; count as usize];
            for lane in 0..64 {
                for slot in (lane..cached).step_by(64) {
                    visits[slot as usize] += 1;
                }
            }
            assert!(
                visits
                    .iter()
                    .enumerate()
                    .all(|(i, &v)| v == u32::from((i as u32) < cached))
            );
        }
        assert!(source.contains("iteration<params.iterations"));
        assert!(source.contains("return uncached_free(c,normal);"));
        if dofs == 10 {
            if let Some(path) = std::env::var_os("MOLLA_ADMM_ONCHIP_ARTIFACT") {
                std::fs::write(path, source).unwrap();
            }
        }
    }
}
