//! Parse and validate primitive shader wiring without requesting a GPU device.

use wgpu::naga::{
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let modules = [
        (
            "mass_response",
            concat!(
                include_str!("../../molla-kernels/shaders/crba.wgsl"),
                include_str!("../../molla-kernels/shaders/articulated_response_common.wgsl"),
            ),
        ),
        (
            "jacobian_response",
            concat!(
                include_str!("../../molla-kernels/shaders/jacobian.wgsl"),
                include_str!("../../molla-kernels/shaders/articulated_response_common.wgsl"),
            ),
        ),
        (
            "reaction_response",
            concat!(
                include_str!("../../molla-kernels/shaders/reaction_wrenches.wgsl"),
                include_str!("../../molla-kernels/shaders/articulated_response_common.wgsl"),
            ),
        ),
        (
            "featherstone",
            concat!(
                include_str!("../../molla-kernels/shaders/featherstone.wgsl"),
                include_str!("../../molla-kernels/shaders/featherstone_stops.wgsl"),
                include_str!("../../molla-kernels/shaders/featherstone_feedback.wgsl"),
                include_str!("../../molla-kernels/shaders/featherstone_mass_inverse.wgsl"),
                include_str!("../../molla-kernels/shaders/quaternion_precision.wgsl"),
                include_str!("../../molla-kernels/shaders/joint_euler_chart.wgsl"),
                include_str!("../../molla-kernels/shaders/euler_bound_precision.wgsl"),
                include_str!("../../molla-kernels/shaders/featherstone_mixed_stops.wgsl"),
                include_str!("../../molla-kernels/shaders/cone_geometry.wgsl"),
            ),
        ),
        (
            "runtime_distance",
            concat!(
                include_str!("../../molla-kernels/shaders/cone_geometry.wgsl"),
                include_str!("../../molla-kernels/shaders/quaternion_precision.wgsl"),
                include_str!("../../molla-kernels/shaders/joint_euler_chart.wgsl"),
                include_str!("../../molla-kernels/shaders/euler_bound_precision.wgsl"),
                include_str!("../../molla-kernels/shaders/joint_limit_geometry.wgsl"),
                include_str!("../../molla-kernels/shaders/featherstone_mass_inverse.wgsl"),
                include_str!("../../molla-kernels/shaders/runtime_distance.wgsl"),
            ),
        ),
        (
            "body_damping",
            include_str!("../shaders/rigid_damping.wgsl"),
        ),
        (
            "scalar_publication",
            include_str!("../shaders/scalar_joint_publication.wgsl"),
        ),
        ("coulomb_disk", include_str!("../shaders/coulomb_disk.wgsl")),
        (
            "contact_response",
            include_str!("../shaders/contact_response.wgsl"),
        ),
        (
            "floating_momentum",
            include_str!("../shaders/rigid_momentum.wgsl"),
        ),
    ];
    let count = modules.len();
    for (name, source) in modules {
        let source = source.replace("__ART_WIDTH__", "2");
        let module = wgsl::parse_str(&source)
            .map_err(|e| format!("{name}: {}", e.emit_to_string(&source)))?;
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .map_err(|e| format!("{name}: {}", e.emit_to_string(&source)))?;
        println!(
            "shader_wiring_ok\tmodule={name}\tentry_points={}",
            module.entry_points.len()
        );
    }
    println!("shader_wiring_complete\tmodules={count}\tdevice_execution=false");
    Ok(())
}
