// ── Stage F1-F5 WCSPH dam-break viewer wiring ──────────────────────

// Open-top tank for the WCSPH dam-break scene — the fluid is poured
// into this box and held by static boundary particles. Mirrors the
// geometry of the `sph_boundary_gpu` parity test.
const SPH_TANK_MIN: [f32; 3] = [-0.35, 0.0, -0.20];
const SPH_TANK_MAX: [f32; 3] = [0.35, 0.6, 0.20];

// Per-frame state for the WCSPH dam-break [`Scene::SphDamBreak`].
#[derive(Resource, Default)]
struct SphExampleState {
    /// The GPU WCSPH solver — built lazily on first entry, then reused.
    sph: Option<SphSimulation>,
    /// False until the dam has been (re)poured; drives the scene reset.
    was_active: bool,
}

// Stage F1-F5 — advance the real GPU [`SphSimulation`] for the
// dam-break scene. `step_simulation` skips `Scene::SphDamBreak`; the
// fluid particles are rendered straight from `sim.mpm_particles` by
// `draw_mpm_gizmos`.
fn update_sph_example(
    mut sim: ResMut<Sim>,
    mut state: ResMut<SphExampleState>,
    run: Res<SimRunState>,
) {
    if sim.scene != Scene::SphDamBreak {
        state.was_active = false;
        return;
    }

    // Build the WCSPH solver once against the viewer's wgpu engine.
    if state.sph.is_none() {
        let s = SphSimulation::new(&sim.engine, SphConfig::water()).expect("sph new");
        state.sph = Some(s);
    }

    let device = sim.engine.device().unwrap().clone();
    let queue = sim.engine.queue().unwrap().clone();

    // (Re)pour the dam on first entry / re-entry to the scene.
    if !state.was_active {
        let h = SphConfig::water().h;
        let walls = box_tank_walls(SPH_TANK_MIN, SPH_TANK_MAX, h, 2);
        // A tall fluid column held against the -X wall — it collapses
        // and surges across the tank floor when the sim starts.
        let lo = [-0.32_f32, 0.05, -0.15];
        let hi = [-0.05_f32, 0.50, 0.15];
        let n = |a: f32, b: f32| (((b - a) / h).floor() as i32 + 1).max(1);
        let mut fluid = Vec::new();
        for ix in 0..n(lo[0], hi[0]) {
            for iy in 0..n(lo[1], hi[1]) {
                for iz in 0..n(lo[2], hi[2]) {
                    fluid.push([
                        lo[0] + ix as f32 * h,
                        lo[1] + iy as f32 * h,
                        lo[2] + iz as f32 * h,
                    ]);
                }
            }
        }
        if let Some(s) = state.sph.as_mut() {
            s.set_scene(&queue, &fluid, &walls).expect("sph set_scene");
        }
        sim.mpm_particles = fluid;
        state.was_active = true;
        return;
    }

    if run.paused {
        return;
    }

    // Advance the WCSPH pipeline. dt = 2e-4 keeps the Tait-EOS sound
    // speed inside the CFL limit; 40 substeps per frame is gentle
    // slow-motion and stays cheap on the GPU (one submit + one
    // frame-end readback).
    let positions = if let Some(s) = state.sph.as_mut() {
        s.step_with(&device, &queue, 2.0e-4, 40).expect("sph step");
        s.read_positions_with(&device, &queue).unwrap_or_default()
    } else {
        Vec::new()
    };
    if !positions.is_empty() {
        sim.mpm_particles = positions;
    }
}

