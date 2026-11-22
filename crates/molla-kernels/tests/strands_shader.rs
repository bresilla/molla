#[test]
fn strand_shader_parses_and_validates() {
    let source = concat!(
        include_str!("../shaders/strands.wgsl"),
        "\n",
        include_str!("../shaders/strands_coupled.wgsl"),
        "\n",
        include_str!("../shaders/strands_contact.wgsl")
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
