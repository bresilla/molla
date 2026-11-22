#[test]
fn attachment_blade_shader_validates() {
    let source = concat!(
        include_str!("../shaders/air_field.wgsl"),
        include_str!("../shaders/attachments.wgsl")
    );
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
}
