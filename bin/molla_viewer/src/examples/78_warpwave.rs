// ═══ Scene 78: WarpWave (Stage 37.1) ═══════════════════════════════════
//
// 32×32 P1 tri mesh on the unit square; all four edges pinned (Dirichlet
// u = 0); initial Gaussian pulse at (0.3, 0.3); Newmark-β implicit step
// per frame. Renders as height-field via gizmo lines.

const WARP_WAVE_GRID: usize = 32;

struct WarpWaveExampleState {
    mesh: fem::FemMesh,
    mass: molla::math::sparse::BsrMatrix,
    stiff: molla::math::sparse::BsrMatrix,
    bcs: Vec<fem::BoundaryCondition>,
    u: Vec<f64>,
    v: Vec<f64>,
    a: Vec<f64>,
    c2: f64,
    last_c2_baked: f64,
    dt: f64,
    pulse_amp: f64,
    sim_t: f64,
    last_residual: f64,
    last_iters: usize,
    needs_reset: bool,
    user_reset: bool,
    /// GPU stepper — `Some` once the user has clicked "Use GPU"
    /// (built lazily because `Default` doesn't have access to the
    /// `Engine`). Rebuilt whenever the mesh or BCs change.
    gpu_stepper: Option<fem::WaveStepperGpu>,
    use_gpu: bool,
}

impl Default for WarpWaveExampleState {
    fn default() -> Self {
        let mesh = fem::FemMesh::tri_grid_2d(WARP_WAVE_GRID, WARP_WAVE_GRID, 1.0, 1.0);
        let c2 = 1.0_f64;
        let mass = fem::assemble_global(&mesh, fem::FunctionSpace::P1Nodal, |c, v| {
            fem::compute_element_mass(c, v)
        })
        .expect("WarpWave mass");
        let stiff = fem::assemble_global(&mesh, fem::FunctionSpace::P1Nodal, |c, v| {
            fem::compute_element_wave_stiffness(c, v, c2)
        })
        .expect("WarpWave stiffness");
        // Pin all boundary verts.
        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.x <= 1e-9 || p.x >= 1.0 - 1e-9 || p.y <= 1e-9 || p.y >= 1.0 - 1e-9 {
                bcs.push(fem::BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 0.0,
                });
            }
        }
        let n = mesh.vertices.len();
        // Gaussian pulse initial displacement at (0.3, 0.3).
        let mut u = vec![0.0_f64; n];
        let amp = 0.05;
        for (i, p) in mesh.vertices.iter().enumerate() {
            let dx = p.x - 0.3;
            let dy = p.y - 0.3;
            u[i] = amp * (-50.0 * (dx * dx + dy * dy)).exp();
        }
        // Re-pin BC values.
        for bc in &bcs {
            if let fem::BoundaryCondition::Dirichlet { vertex, value } = bc {
                u[*vertex as usize] = *value;
            }
        }
        Self {
            mesh,
            mass,
            stiff,
            bcs,
            u,
            v: vec![0.0; n],
            a: vec![0.0; n],
            c2,
            last_c2_baked: c2,
            dt: 0.005,
            pulse_amp: amp,
            sim_t: 0.0,
            last_residual: 0.0,
            last_iters: 0,
            needs_reset: false,
            user_reset: false,
            gpu_stepper: None,
            use_gpu: false,
        }
    }
}

fn update_warp_wave(sim: Res<Sim>, run: Res<SimRunState>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpWave || run.paused {
        return;
    }
    let st = &mut warp.wave;
    if st.needs_reset || st.user_reset {
        *st = WarpWaveExampleState::default();
        return;
    }
    let stiff_changed = (st.c2 - st.last_c2_baked).abs() > 1e-9;
    if stiff_changed {
        st.stiff = fem::assemble_global(&st.mesh, fem::FunctionSpace::P1Nodal, |c, v| {
            fem::compute_element_wave_stiffness(c, v, st.c2)
        })
        .expect("WarpWave rebuild stiffness");
        st.last_c2_baked = st.c2;
        // Stiffness changed — invalidate the cached GPU stepper.
        st.gpu_stepper = None;
    }
    // Lazily build the GPU stepper if the user has enabled GPU mode
    // and the cache is empty. Slice H — initial state is uploaded
    // once to the persistent GPU buffers.
    if st.use_gpu && st.gpu_stepper.is_none() {
        match fem::WaveStepperGpu::new(
            &sim.engine,
            &st.mesh,
            &st.mass,
            &st.stiff,
            st.dt,
            fem::NEWMARK_BETA_DEFAULT,
            fem::NEWMARK_GAMMA_DEFAULT,
            &st.bcs,
            &st.u,
            &st.v,
            &st.a,
        ) {
            Ok(s) => st.gpu_stepper = Some(s),
            Err(e) => {
                warn!("WarpWave: GPU stepper init failed ({e}); falling back to CPU");
                st.use_gpu = false;
            }
        }
    }
    if st.use_gpu && st.gpu_stepper.is_some() {
        // Slice H GPU path — `step` runs entirely on GPU; `read_state`
        // pulls back `(u, v, a)` so the height-field gizmos can render.
        // Falling back to CPU here would leak the GPU-resident state
        // out of sync with the host; on error we mark the stepper for
        // rebuild on the next frame.
        let stepper = st.gpu_stepper.as_mut().expect("checked above");
        let step_result = stepper.step();
        let read_result = match step_result {
            Ok(stats) => match stepper.read_state() {
                Ok((u, v, a)) => Ok((stats, u, v, a)),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        match read_result {
            Ok((stats, u, v, a)) => {
                st.u = u;
                st.v = v;
                st.a = a;
                st.last_residual = stats.residual;
                st.last_iters = stats.iters;
            }
            Err(e) => {
                warn!("WarpWave: GPU step/read failed ({e}); rebuilding next frame");
                st.gpu_stepper = None;
            }
        }
    } else {
        let step = fem::solve_wave_transient(
            &st.mass,
            &st.stiff,
            &st.u,
            &st.v,
            &st.a,
            st.dt,
            fem::NEWMARK_BETA_DEFAULT,
            fem::NEWMARK_GAMMA_DEFAULT,
            &st.bcs,
        );
        st.u = step.u_new;
        st.v = step.v_new;
        st.a = step.a_new;
        st.last_residual = step.residual;
        st.last_iters = step.iters;
    }
    st.sim_t += st.dt;
}

fn draw_warp_wave_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpWave {
        return;
    }
    let st = &warp.wave;
    let origin = Vec3::new(-0.5, 0.5, -0.5);
    let h_scale = 1.5_f32;
    // Draw tri edges as height-field lines coloured by displacement.
    for cell in &st.mesh.cells {
        if let fem::FemCell::Tri(v) = cell {
            for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
                let pa = st.mesh.vertices[a as usize];
                let pb = st.mesh.vertices[b as usize];
                let ua = st.u[a as usize] as f32;
                let ub = st.u[b as usize] as f32;
                let from = origin + Vec3::new(pa.x as f32, ua * h_scale, pa.y as f32);
                let to = origin + Vec3::new(pb.x as f32, ub * h_scale, pb.y as f32);
                let amp = (0.5 * (ua + ub)).abs() / st.pulse_amp.max(1e-6) as f32;
                let amp = amp.clamp(0.0, 1.0);
                let colour = Color::srgb(0.30 + 0.65 * amp, 0.30 + 0.40 * amp, 0.95 - 0.40 * amp);
                gizmos.line(from, to, colour);
            }
        }
    }
}

fn warp_wave_pane(ui: &mut egui::Ui, state: &mut WarpWaveExampleState) {
    ui.label("Warp wave (port of example_wave.py)");
    ui.separator();
    readout(ui, "Mesh", &format!("{0}×{0} P1 tris", WARP_WAVE_GRID));
    readout(ui, "DoFs", &format!("{}", state.u.len()));
    readout(ui, "Sim time t", &format!("{:.3} s", state.sim_t));
    readout(ui, "CG iters", &format!("{}", state.last_iters));
    readout(ui, "Residual", &format!("{:.3e}", state.last_residual));
    ui.separator();
    ui.label("Wave speed² c²");
    let _ = ui.add(egui::Slider::new(&mut state.c2, 0.1..=10.0).logarithmic(true));
    ui.label("Time step dt");
    let _ = ui.add(egui::Slider::new(&mut state.dt, 0.001..=0.02).logarithmic(true));
    ui.separator();
    if ui.button("Reset (re-emit pulse)").clicked() {
        state.user_reset = true;
    }
    ui.separator();
    let was_gpu = state.use_gpu;
    let _ = ui.checkbox(&mut state.use_gpu, "Use GPU CG (Stage 36.5 Slice E)");
    if was_gpu != state.use_gpu {
        // Toggle changed — invalidate the cached stepper so the
        // next frame either builds it or drops it.
        state.gpu_stepper = None;
    }
    let backend = if state.use_gpu && state.gpu_stepper.is_some() {
        "GPU resident CG"
    } else if state.use_gpu {
        "GPU (init pending / failed)"
    } else {
        "CPU"
    };
    readout(ui, "Backend", backend);
    ui.small("Newmark-β implicit (β=¼, γ=½) — unconditionally stable.");
    ui.small("Initial Gaussian pulse at (0.3, 0.3); 4 edges pinned.");
}

