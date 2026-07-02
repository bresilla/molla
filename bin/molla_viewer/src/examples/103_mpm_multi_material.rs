// MPM multi-material viewer example. Since Stage 24d.4 the runtime
// solver carries REAL per-particle material ids: the three x-bands run
// sand / snow / viscous constitutive branches in one simulation
// (`MpmSimulation::set_materials`, wired at reset in `24_mpm.rs`), and
// the colors below label the actual material each particle simulates.
fn mpm_multi_material_particle_layout(
    state: &MpmExampleState,
) -> ([f32; 3], [f32; 3], Option<(f32, f32, f32, f32)>) {
    (
        [-0.45, state.spout_y - 0.12, -0.16],
        [0.45, state.spout_y + 0.12, 0.16],
        None,
    )
}

fn mpm_multi_material_region(p: [f32; 3]) -> u32 {
    if p[0] < -0.15 {
        0 // sand
    } else if p[0] > 0.15 {
        2 // viscous
    } else {
        1 // snow
    }
}

fn mpm_multi_material_color(region: u32) -> Color {
    match region {
        1 => Color::srgb(0.78, 0.90, 1.00), // snow
        2 => Color::srgb(0.35, 0.72, 1.00), // viscous/fluid-like
        _ => Color::srgb(0.85, 0.74, 0.52), // sand
    }
}

