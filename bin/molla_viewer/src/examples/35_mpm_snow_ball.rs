// MPM snow ball (Stomakhin) scene.  This example reuses the shared GPU MPM
// stepping path from `24_mpm.rs`, but gives the reset pass its own spherical
// particle layout so the snow rheology can be inspected as a compact ball.
fn mpm_snow_ball_particle_layout(
    state: &MpmExampleState,
) -> ([f32; 3], [f32; 3], Option<(f32, f32, f32, f32)>) {
    let r = 0.20_f32;
    let cy = state.spout_y;
    ([-r, cy - r, -r], [r, cy + r, r], Some((0.0, cy, 0.0, r)))
}

