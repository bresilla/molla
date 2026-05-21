// MPM two-way coupling scene.  The particle layout starts as a broad shallow
// sand bed, and the reset path drops a kinematic sphere into it so the GPU
// MPM solver's rigid-wrench readback is visible immediately.
fn mpm_two_way_particle_layout() -> ([f32; 3], [f32; 3], Option<(f32, f32, f32, f32)>) {
    ([-0.30, 0.05, -0.30], [0.30, 0.20, 0.30], None)
}

fn mpm_two_way_initial_rigid_sphere() -> ([f32; 3], f32, [f32; 3]) {
    ([0.0, 1.0, 0.0], 0.12, [0.0, -2.5, 0.0])
}

