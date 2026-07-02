// Stage-15 GPU narrow-phase demo state — toggle for the GPU pipeline
// and the most recent CPU-narrow-phase call count read from the
// pipeline. The toggle is applied each frame in `step_simulation`.
#[derive(Resource)]
struct GpuNpExampleState {
    /// User-controlled: enable the full GPU narrow phase.
    use_gpu: bool,
    /// Whether the pipeline currently has GPU narrow phase enabled.
    /// Used to detect when the user flips the toggle so we can
    /// rebuild the pipeline.
    gpu_enabled_now: bool,
    /// Last frame's CPU narrow-phase call count.
    last_cpu_calls: usize,
    /// Last frame's contact count.
    last_contacts: u32,
    /// SAP candidate-pair count (per-frame, last seen).
    last_pairs: usize,
}

impl Default for GpuNpExampleState {
    fn default() -> Self {
        Self {
            use_gpu: true,
            gpu_enabled_now: false,
            last_cpu_calls: 0,
            last_contacts: 0,
            last_pairs: 0,
        }
    }
}

// Stage-15 GPU-narrow-phase demo — pile of mixed boxes / cylinders /
// cones / spheres dropped onto a plane. Designed to exercise the
// Sphere/Box/Capsule/Cylinder/Cone branches of the GPU EPA kernel.
// The right-rail pane lets the user toggle full-GPU vs CPU narrow
// phase and watch the per-frame `last_cpu_narrow_phase_calls`
// counter drop to ~0 (the `Plane` pairs always stay on CPU).
fn build_gpu_narrow_phase_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.2);

    let mut visuals: Vec<BodyVisual> = Vec::new();
    // Ground plane.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    // 16 mixed bodies in a 4x4 grid spaced 0.7 m, dropped from y ∈ [3, 6].
    let n_side = 4i32;
    for ix in 0..n_side {
        for iz in 0..n_side {
            let x = (ix as f64 - 1.5) * 0.7;
            let z = (iz as f64 - 1.5) * 0.7;
            let y = 3.0 + ((ix + iz) as f64) * 0.3;
            let kind = (ix * n_side + iz) % 4;
            match kind {
                0 => {
                    let half = MVec3::splat(0.25);
                    let mass = 0.8;
                    let inertia = box_inertia(mass, 0.25);
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia {
                                mass,
                                com: MVec3::ZERO,
                                inertia,
                            })
                            .with_position(MVec3::new(x, y, z)),
                    );
                    b.add_shape_box(body, half, mp::Transform::IDENTITY)
                        .unwrap();
                    visuals.push(BodyVisual::Box(Vec3::splat(0.25)));
                }
                1 => {
                    let r = 0.25;
                    let mass = 1.0;
                    let i_v = (2.0 / 5.0) * mass * r * r;
                    let inertia = molla::math::Mat3::from_diagonal(MVec3::new(i_v, i_v, i_v));
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia {
                                mass,
                                com: MVec3::ZERO,
                                inertia,
                            })
                            .with_position(MVec3::new(x, y, z)),
                    );
                    b.add_shape_sphere(body, r, mp::Transform::IDENTITY)
                        .unwrap();
                    visuals.push(BodyVisual::Sphere(r as f32));
                }
                2 => {
                    // Cylinder collision; render approximation as a capsule
                    // (no Cylinder variant in `BodyVisual`, capsule is
                    // close enough visually).
                    let half_h: f64 = 0.30;
                    let r: f64 = 0.20;
                    let mass: f64 = 0.7;
                    let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
                        mass * (r * r * 0.25 + (2.0 * half_h).powi(2) / 12.0),
                        mass * r * r * 0.5,
                        mass * (r * r * 0.25 + (2.0 * half_h).powi(2) / 12.0),
                    ));
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia {
                                mass,
                                com: MVec3::ZERO,
                                inertia,
                            })
                            .with_position(MVec3::new(x, y, z)),
                    );
                    b.add_shape_cylinder(body, half_h, r, mp::Transform::IDENTITY)
                        .unwrap();
                    visuals.push(BodyVisual::Capsule {
                        half_h: half_h as f32,
                        radius: r as f32,
                    });
                }
                _ => {
                    let half_h: f64 = 0.30;
                    let r: f64 = 0.20;
                    let mass: f64 = 0.7;
                    let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
                        mass * (r * r * 0.25 + (2.0 * half_h).powi(2) / 12.0),
                        mass * r * r * 0.5,
                        mass * (r * r * 0.25 + (2.0 * half_h).powi(2) / 12.0),
                    ));
                    let body = b.add_body(
                        mp::BodyParams::new()
                            .with_spatial_inertia(mp::SpatialInertia {
                                mass,
                                com: MVec3::ZERO,
                                inertia,
                            })
                            .with_position(MVec3::new(x, y, z)),
                    );
                    b.add_shape_cone(body, half_h, r, mp::Transform::IDENTITY)
                        .unwrap();
                    visuals.push(BodyVisual::Cone {
                        half_h: half_h as f32,
                        radius: r as f32,
                    });
                }
            }
        }
    }

    (b.build().expect("gpu narrow phase model"), visuals)
}

fn gpu_np_pane(ui: &mut egui::Ui, state: &mut GpuNpExampleState, accent: egui::Color32) {
    ui.add_space(2.0);
    let mode_label = if state.gpu_enabled_now {
        "GPU (Stage 15)"
    } else {
        "CPU"
    };
    readout(ui, "Narrow phase", mode_label);
    readout(ui, "SAP pairs", &format!("{}", state.last_pairs));
    readout(ui, "Contacts", &format!("{}", state.last_contacts));
    readout(ui, "CPU NP calls", &format!("{}", state.last_cpu_calls));
    let gpu_calls = state.last_pairs.saturating_sub(state.last_cpu_calls);
    readout(ui, "GPU NP calls", &format!("{}", gpu_calls));
    ui.separator();

    ui.horizontal(|ui| {
        let label = if state.use_gpu {
            "Disable GPU narrow phase"
        } else {
            "Enable GPU narrow phase"
        };
        if button(ui, label, accent).clicked() {
            state.use_gpu = !state.use_gpu;
        }
    });
    ui.separator();
    ui.small(
        "Toggles between CPU and full-GPU narrow phase. With GPU \
         on, the Sphere/Box/Capsule/Cylinder/Cone pairs run through \
         WGSL GJK + EPA on the device; only Plane pairs (and the few \
         that hit the WGSL polytope cap) bump the CPU counter.",
    );
}


