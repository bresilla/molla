// MPM beam twist scene.  It starts from an elongated block and stamps a
// procedural twist into particle positions; colour is derived from the same
// twist angle so the deformation pattern is easy to read in the viewer.
fn mpm_beam_twist_particle_layout(
    state: &MpmExampleState,
) -> ([f32; 3], [f32; 3], Option<(f32, f32, f32, f32)>) {
    (
        [-0.50, state.spout_y - 0.07, -0.07],
        [0.50, state.spout_y + 0.07, 0.07],
        None,
    )
}

const MPM_BEAM_TWIST_RATE: f32 = std::f32::consts::PI * 1.25;

fn mpm_beam_twist_angle(p: [f32; 3]) -> f32 {
    (p[0] * MPM_BEAM_TWIST_RATE).clamp(-std::f32::consts::PI, std::f32::consts::PI)
}

fn mpm_beam_twist_position(mut p: [f32; 3]) -> [f32; 3] {
    let angle = mpm_beam_twist_angle(p);
    let cy = p[1] - 0.70;
    let cz = p[2];
    let (s, c) = angle.sin_cos();
    p[1] = 0.70 + c * cy - s * cz;
    p[2] = s * cy + c * cz;
    p
}

fn mpm_beam_twist_color(p: [f32; 3]) -> Color {
    let t = (mpm_beam_twist_angle(p) / std::f32::consts::PI * 0.5 + 0.5).clamp(0.0, 1.0);
    Color::srgb(0.25 + 0.65 * t, 0.35, 0.95 - 0.55 * t)
}

