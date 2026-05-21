// Stage-19 GPU broad-phase demo state — toggle for the LBVH path
// + per-backend ms/frame counters. The toggle is applied each
// frame in `step_simulation` (mirrors `GpuNpExampleState`).
#[derive(Resource)]
struct GpuBpExampleState {
    /// User-controlled: enable the GPU LBVH broad phase. False ⇒ CPU SAP.
    use_gpu: bool,
    /// Whether the pipeline currently has GPU broad phase enabled.
    /// Tracks a delta between user toggle and pipeline state so we
    /// only rebuild the pipeline when the user flips it.
    gpu_enabled_now: bool,
    /// Last frame's broad-phase pair count.
    last_pairs: usize,
    /// Smoothed ms/frame for the active backend (whichever is on).
    smoothed_ms: f64,
    /// Most recent per-backend ms/frame snapshot — kept across
    /// toggles so the user can compare CPU vs GPU at a glance.
    last_cpu_ms: f64,
    last_gpu_ms: f64,
}

impl Default for GpuBpExampleState {
    fn default() -> Self {
        Self {
            use_gpu: true,
            gpu_enabled_now: false,
            last_pairs: 0,
            smoothed_ms: 0.0,
            last_cpu_ms: 0.0,
            last_gpu_ms: 0.0,
        }
    }
}

// Stage-19 GPU broad-phase scene — a 32×32×4 = 4096-sphere shower
// dropped onto a plane. Sphere radii are deterministically jittered
// so the AABB sizes vary (more representative workload for the
// LBVH pair-finder than uniform-radius spheres). The right-rail
// pane toggles between CPU SAP and GPU LBVH and shows live ms/frame
// + pair count for both.
fn build_broad_phase_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.20);

    let mut visuals: Vec<BodyVisual> = Vec::new();

    // Ground plane.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    // 32 × 32 × 4 = 4096 spheres on a regular grid, dropped from
    // a few metres up. Spacing is wide enough that the initial
    // configuration is mostly disjoint — the broad phase still has
    // O(n) work even when no pairs overlap, so this stresses the
    // build path. As the spheres settle the pair count climbs.
    let nx: i32 = 32;
    let nz: i32 = 32;
    let ny: i32 = 4;
    let spacing: f64 = 0.8;
    let drop_height: f64 = 6.0;

    // Cheap deterministic RNG (Wang hash-style) for radius jitter
    // — keeps the scene reproducible across runs without pulling
    // in a `rand` dep.
    let mut h: u32 = 0x9E37_79B9;
    let mut next = || {
        h = h.wrapping_mul(2654435761).wrapping_add(0x12345678);
        h
    };

    for ix in 0..nx {
        for iz in 0..nz {
            for iy in 0..ny {
                let x = (ix as f64 - (nx as f64 - 1.0) * 0.5) * spacing;
                let z = (iz as f64 - (nz as f64 - 1.0) * 0.5) * spacing;
                let y = drop_height + iy as f64 * (spacing * 0.9);
                // Radius in [0.18, 0.32] — small enough to fit grid
                // spacing, large enough to overlap as the spheres
                // settle and pile up.
                let r = 0.18 + (next() as f64 / u32::MAX as f64) * 0.14;
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
        }
    }

    (b.build().expect("broad phase model"), visuals)
}

fn gpu_bp_pane(ui: &mut egui::Ui, state: &mut GpuBpExampleState, accent: egui::Color32) {
    ui.add_space(2.0);
    let mode_label = if state.gpu_enabled_now {
        // Stage 19.1 — Morton sort moved to GPU (was CPU stop-gap).
        "GPU LBVH (radix v2)"
    } else {
        "CPU SAP"
    };
    readout(ui, "Broad phase", mode_label);
    readout(ui, "Pairs", &format!("{}", state.last_pairs));
    readout(
        ui,
        "Active ms/frame",
        &format!("{:>5.2} ms", state.smoothed_ms),
    );
    ui.separator();

    // Side-by-side per-backend memory of the last seen ms/frame —
    // updated whenever that backend was the active one. The user
    // toggles, waits a few frames, toggles back, and reads off the
    // speedup directly.
    readout(
        ui,
        "Last CPU SAP",
        &format!("{:>5.2} ms", state.last_cpu_ms),
    );
    readout(
        ui,
        "Last GPU LBVH",
        &format!("{:>5.2} ms", state.last_gpu_ms),
    );
    if state.last_cpu_ms > 0.0 && state.last_gpu_ms > 0.0 {
        let speedup = state.last_cpu_ms / state.last_gpu_ms.max(0.001);
        readout(ui, "Speedup", &format!("{:>5.2}x", speedup));
    }
    ui.separator();

    ui.horizontal(|ui| {
        let label = if state.use_gpu {
            "Switch to CPU SAP"
        } else {
            "Switch to GPU LBVH"
        };
        if button(ui, label, accent).clicked() {
            state.use_gpu = !state.use_gpu;
        }
    });
    ui.separator();
    ui.small(
        "Stage 19 — toggle between CPU sweep-and-prune and the GPU \
         LBVH (Karras-style Linear BVH) broad phase. Both produce \
         the same overlapping-AABB pair list; the GPU path runs the \
         build + tree traversal on the device. The pair list flows \
         into the same CPU narrow phase regardless of toggle, so \
         only the 'collide' phase ms-time changes.",
    );
}


