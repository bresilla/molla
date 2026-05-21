// ── Scene 112: WarpFemStokesTransfer ─────────────────────────────
const STOKES_TRANSFER_GRID_NX: usize = 7;
const STOKES_TRANSFER_GRID_NZ: usize = 7;
const STOKES_TRANSFER_TRACERS_X: usize = 9;
const STOKES_TRANSFER_TRACERS_Z: usize = 7;

struct WarpFemStokesTransferState {
    tracers: Vec<Vec3>,
    grid_velocities: Vec<Vec3>,
    sim_t: f64,
    dt: f32,
    rotation_strength: f32,
    drift: f32,
    use_rotation: bool,
    max_speed: f32,
    user_reset: bool,
}

impl Default for WarpFemStokesTransferState {
    fn default() -> Self {
        let mut me = Self {
            tracers: Vec::new(),
            grid_velocities: Vec::new(),
            sim_t: 0.0,
            dt: 1.0 / 90.0,
            rotation_strength: 0.75,
            drift: 0.12,
            use_rotation: true,
            max_speed: 0.0,
            user_reset: true,
        };
        me.rebuild_grid();
        me
    }
}

fn stokes_transfer_analytic_velocity(p: Vec3, rotation_strength: f32, drift: f32) -> Vec3 {
    let r = Vec3::new(p.x - 0.5, 0.0, p.z - 0.5);
    Vec3::new(
        drift - rotation_strength * r.z,
        0.0,
        rotation_strength * r.x,
    )
}

fn stokes_transfer_build_grid(
    nx: usize,
    nz: usize,
    rotation_strength: f32,
    drift: f32,
) -> Vec<Vec3> {
    let mut velocities = Vec::with_capacity(nx * nz);
    for j in 0..nz {
        for i in 0..nx {
            let p = Vec3::new(
                i as f32 / (nx.saturating_sub(1).max(1)) as f32,
                0.0,
                j as f32 / (nz.saturating_sub(1).max(1)) as f32,
            );
            velocities.push(stokes_transfer_analytic_velocity(
                p,
                rotation_strength,
                drift,
            ));
        }
    }
    velocities
}

fn stokes_transfer_sample_grid(grid: &[Vec3], nx: usize, nz: usize, p: Vec3) -> Vec3 {
    assert_eq!(grid.len(), nx * nz);
    let gx = p.x.clamp(0.0, 1.0) * (nx.saturating_sub(1)) as f32;
    let gz = p.z.clamp(0.0, 1.0) * (nz.saturating_sub(1)) as f32;
    let i0 = gx.floor() as usize;
    let j0 = gz.floor() as usize;
    let i1 = (i0 + 1).min(nx - 1);
    let j1 = (j0 + 1).min(nz - 1);
    let tx = gx - i0 as f32;
    let tz = gz - j0 as f32;
    let idx = |i: usize, j: usize| j * nx + i;
    let v00 = grid[idx(i0, j0)];
    let v10 = grid[idx(i1, j0)];
    let v01 = grid[idx(i0, j1)];
    let v11 = grid[idx(i1, j1)];
    let vx0 = v00 * (1.0 - tx) + v10 * tx;
    let vx1 = v01 * (1.0 - tx) + v11 * tx;
    vx0 * (1.0 - tz) + vx1 * tz
}

fn stokes_transfer_advect_particle(
    p: Vec3,
    grid: &[Vec3],
    nx: usize,
    nz: usize,
    dt: f32,
    wrap_unit_square: bool,
) -> Vec3 {
    let mut next = p + stokes_transfer_sample_grid(grid, nx, nz, p) * dt;
    if wrap_unit_square {
        next.x = next.x.rem_euclid(1.0);
        next.z = next.z.rem_euclid(1.0);
    }
    next
}

impl WarpFemStokesTransferState {
    fn rebuild_grid(&mut self) {
        let rotation = if self.use_rotation {
            self.rotation_strength
        } else {
            0.0
        };
        self.grid_velocities = stokes_transfer_build_grid(
            STOKES_TRANSFER_GRID_NX,
            STOKES_TRANSFER_GRID_NZ,
            rotation,
            self.drift,
        );
        self.max_speed = self
            .grid_velocities
            .iter()
            .map(|v| v.length())
            .fold(0.0_f32, f32::max);
    }

    fn reset_tracers(&mut self) {
        self.tracers.clear();
        for j in 0..STOKES_TRANSFER_TRACERS_Z {
            for i in 0..STOKES_TRANSFER_TRACERS_X {
                let x = (i as f32 + 0.5) / STOKES_TRANSFER_TRACERS_X as f32;
                let z = (j as f32 + 0.5) / STOKES_TRANSFER_TRACERS_Z as f32;
                self.tracers.push(Vec3::new(x, 0.03, z));
            }
        }
        self.sim_t = 0.0;
        self.user_reset = false;
    }

    fn step(&mut self, dt: f32) {
        for p in &mut self.tracers {
            *p = stokes_transfer_advect_particle(
                *p,
                &self.grid_velocities,
                STOKES_TRANSFER_GRID_NX,
                STOKES_TRANSFER_GRID_NZ,
                dt,
                true,
            );
        }
        self.sim_t += dt as f64;
    }
}

fn update_warp_fem_stokes_transfer(
    time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFemStokesTransfer {
        return;
    }
    let st = &mut warp.fem_stokes_transfer;
    if st.user_reset {
        st.rebuild_grid();
        st.reset_tracers();
    }
    if run.paused {
        return;
    }
    let dt = st.dt.min(time.delta_secs().max(1.0e-4));
    st.step(dt);
}

fn draw_warp_fem_stokes_transfer_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemStokesTransfer {
        return;
    }
    let st = &warp.fem_stokes_transfer;
    let origin = Vec3::new(-0.5, 0.06, -0.5);
    let to_world = |p: Vec3| origin + Vec3::new(p.x, p.y, p.z);

    let corners = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    for e in 0..4 {
        gizmos.line(
            to_world(corners[e]),
            to_world(corners[(e + 1) % 4]),
            Color::srgb(0.25, 0.55, 0.95),
        );
    }

    for j in 0..STOKES_TRANSFER_GRID_NZ {
        for i in 0..STOKES_TRANSFER_GRID_NX {
            let p = Vec3::new(
                i as f32 / (STOKES_TRANSFER_GRID_NX - 1) as f32,
                0.0,
                j as f32 / (STOKES_TRANSFER_GRID_NZ - 1) as f32,
            );
            let v = st.grid_velocities[j * STOKES_TRANSFER_GRID_NX + i];
            gizmos.line(
                to_world(p),
                to_world(p + v * 0.18),
                Color::srgb(0.45, 0.95, 0.55),
            );
        }
    }

    for &p in &st.tracers {
        let v = stokes_transfer_sample_grid(
            &st.grid_velocities,
            STOKES_TRANSFER_GRID_NX,
            STOKES_TRANSFER_GRID_NZ,
            p,
        );
        gizmos.sphere(
            bevy::math::Isometry3d::from_translation(to_world(p)),
            0.014,
            Color::srgb(1.0, 0.75, 0.20),
        );
        gizmos.line(
            to_world(p),
            to_world(p + v * 0.10),
            Color::srgb(1.0, 0.90, 0.35),
        );
    }
}

fn warp_fem_stokes_transfer_pane(ui: &mut egui::Ui, st: &mut WarpFemStokesTransferState) {
    ui.label("Warp FEM Stokes transfer — reduced field-to-particle transfer");
    ui.separator();
    readout(
        ui,
        "Grid",
        &format!("{}×{}", STOKES_TRANSFER_GRID_NX, STOKES_TRANSFER_GRID_NZ),
    );
    readout(ui, "Tracers", &format!("{}", st.tracers.len()));
    readout(ui, "Max |u|", &format!("{:.3}", st.max_speed));
    readout(ui, "t", &format!("{:.2} s", st.sim_t));
    let mut changed = false;
    changed |= ui
        .add(egui::Slider::new(&mut st.dt, 1.0 / 240.0..=1.0 / 30.0).text("dt"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut st.drift, -0.25..=0.25).text("constant drift"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut st.rotation_strength, 0.0..=1.5).text("rotation"))
        .changed();
    changed |= ui
        .checkbox(&mut st.use_rotation, "rotational field")
        .changed();
    if changed {
        st.rebuild_grid();
    }
    if ui.button("Reset tracers").clicked() {
        st.user_reset = true;
    }
    ui.small("Bilinear transfer from a procedural Stokes/FEM velocity grid to particle tracers.");
    ui.small("GPU audit names the APIC/transfer lane; full Stokes-transfer GPU scene remains future work.");
}

