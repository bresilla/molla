// ═══ Scene 85: SoftbodyHanging (Newton port) ═══════════════════════════
//
// Four tetrahedral soft bodies hanging from pinned top-left faces, each
// built with a different Young's modulus (1×, 2×, 5×, 10× the base).
// Side-by-side comparison of how stiffness affects deflection under
// gravity. Uses the VBD tet solver via `add_tet_mesh_per_material` +
// `pin_particle` to fix the left face of each block.
//
// Newton ref: `xtra/newton/newton/examples/softbody/example_softbody_hanging.py`.

const HANG_DIM_X: u32 = 10;
const HANG_DIM_Y: u32 = 3;
const HANG_DIM_Z: u32 = 3;
const HANG_CELL: f64 = 0.10;
const HANG_SPACING_Z: f64 = 0.55;
const HANG_BASE_Y: f64 = 1.3;
const HANG_PARTICLE_MASS: f64 = 0.10;
const HANG_BASE_E: f64 = 5.0e4;
const HANG_NU: f64 = 0.40;
const HANG_E_SWEEP: [f64; 4] = [1.0, 2.0, 5.0, 10.0];

struct SoftbodyHangingState {
    /// Last computed mean Y of the right-most slice of each block —
    /// updated each frame and surfaced in the pane so the user can
    /// confirm stiffer blocks sag less.
    tip_y: [f64; 4],
}

impl Default for SoftbodyHangingState {
    fn default() -> Self {
        Self {
            tip_y: [HANG_BASE_Y; 4],
        }
    }
}

fn hang_cube_to_six_tets(c: [u32; 8]) -> [[u32; 4]; 6] {
    // Same diagonal split as the Jelly scene.
    let (v000, v100, v010, v110, v001, v101, v011, v111) =
        (c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
    [
        [v000, v100, v110, v111],
        [v000, v110, v010, v111],
        [v000, v010, v011, v111],
        [v000, v011, v001, v111],
        [v000, v001, v101, v111],
        [v000, v101, v100, v111],
    ]
}

fn build_softbody_hanging_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.6);
    b.set_angular_damping(0.10);

    let nx = HANG_DIM_X + 1;
    let ny = HANG_DIM_Y + 1;
    let nz = HANG_DIM_Z + 1;
    let dx = HANG_CELL;

    // Layout: 4 blocks lined up along Z, each independent but sharing
    // the global particle pool.
    for (blk, &e_scale) in HANG_E_SWEEP.iter().enumerate() {
        let centre_z = (blk as f64 - 1.5) * HANG_SPACING_Z;
        let origin = MVec3::new(
            -(HANG_CELL * HANG_DIM_X as f64) * 0.5,
            HANG_BASE_Y,
            centre_z - (HANG_CELL * HANG_DIM_Z as f64) * 0.5,
        );
        let mut block_first: u32 = u32::MAX;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let p = origin + MVec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                    let id = b.add_particle(p, HANG_PARTICLE_MASS, dx * 0.5, mp::WorldId(0));
                    if block_first == u32::MAX {
                        block_first = id;
                    }
                }
            }
        }
        let v_at = |i: u32, j: u32, k: u32| -> u32 { block_first + (k * ny + j) * nx + i };
        let mut tets = Vec::new();
        for kk in 0..nz - 1 {
            for jj in 0..ny - 1 {
                for ii in 0..nx - 1 {
                    let corners = [
                        v_at(ii, jj, kk),
                        v_at(ii + 1, jj, kk),
                        v_at(ii, jj + 1, kk),
                        v_at(ii + 1, jj + 1, kk),
                        v_at(ii, jj, kk + 1),
                        v_at(ii + 1, jj, kk + 1),
                        v_at(ii, jj + 1, kk + 1),
                        v_at(ii + 1, jj + 1, kk + 1),
                    ];
                    for tet in hang_cube_to_six_tets(corners) {
                        tets.push(tet);
                    }
                }
            }
        }
        let mat_idx: Vec<u32> = vec![0; tets.len()];
        let materials = vec![mp::MaterialNeoHookean::elastic(
            HANG_BASE_E * e_scale,
            HANG_NU,
        )];
        let _ = b
            .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
            .expect("softbody hanging tet mesh");

        // Pin the entire i=0 face (every particle with ix == 0).
        for k in 0..nz {
            for j in 0..ny {
                b.pin_particle(v_at(0, j, k));
            }
        }
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("softbody hanging model"), Vec::new())
}

fn update_softbody_hanging(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::SoftbodyHanging {
        return;
    }
    let Ok(pq) = sim.state_a.particle_q.host() else {
        return;
    };
    let nx = HANG_DIM_X + 1;
    let ny = HANG_DIM_Y + 1;
    let nz = HANG_DIM_Z + 1;
    let per_block = (nx * ny * nz) as usize;
    let st = &mut warp.softbody_hanging;
    for blk in 0..4usize {
        let base = blk * per_block;
        // Mean Y of the right-most face (i = nx-1) gives a clean
        // "tip droop" readout.
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        let ix = nx - 1;
        for k in 0..nz {
            for j in 0..ny {
                let i = base + ((k * ny + j) * nx + ix) as usize;
                if i < pq.len() {
                    sum += pq[i].y;
                    count += 1;
                }
            }
        }
        if count > 0 {
            st.tip_y[blk] = sum / count as f64;
        }
    }
}

fn draw_softbody_hanging_gizmos(sim: Res<Sim>, mut gizmos: Gizmos) {
    if sim.scene != Scene::SoftbodyHanging {
        return;
    }
    let Ok(pq) = sim.state_a.particle_q.host() else {
        return;
    };
    let nx = HANG_DIM_X + 1;
    let ny = HANG_DIM_Y + 1;
    let nz = HANG_DIM_Z + 1;
    let per_block = (nx * ny * nz) as usize;
    // Draw cube-edges only (12 edges per cell) but de-duplicated across
    // shared faces via the grid topology. We use a colour ramp from
    // soft (blue) → stiff (red) so the comparison reads at a glance.
    for blk in 0..4usize {
        let base = blk * per_block;
        // Stiffness 1× → 10×: log-ramp t ∈ [0, 1].
        let s = (HANG_E_SWEEP[blk] / HANG_E_SWEEP[0]).log10()
            / (HANG_E_SWEEP[3] / HANG_E_SWEEP[0]).log10();
        let t = s.clamp(0.0, 1.0) as f32;
        let col = Color::srgb(0.15 + 0.80 * t, 0.40, 0.95 - 0.80 * t);
        let v = |i: u32, j: u32, k: u32| -> Vec3 {
            let p = pq[base + ((k * ny + j) * nx + i) as usize];
            Vec3::new(p.x as f32, p.y as f32, p.z as f32)
        };
        // X-edges
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx - 1 {
                    gizmos.line(v(i, j, k), v(i + 1, j, k), col);
                }
            }
        }
        // Y-edges
        for k in 0..nz {
            for j in 0..ny - 1 {
                for i in 0..nx {
                    gizmos.line(v(i, j, k), v(i, j + 1, k), col);
                }
            }
        }
        // Z-edges
        for k in 0..nz - 1 {
            for j in 0..ny {
                for i in 0..nx {
                    gizmos.line(v(i, j, k), v(i, j, k + 1), col);
                }
            }
        }
    }
}

fn softbody_hanging_pane(ui: &mut egui::Ui, st: &mut SoftbodyHangingState) {
    ui.label("Newton port — 4 hanging tet blocks, E sweep.");
    ui.separator();
    readout(
        ui,
        "Grid",
        &format!("{}×{}×{} cells / block", HANG_DIM_X, HANG_DIM_Y, HANG_DIM_Z),
    );
    readout(ui, "Cell size", &format!("{:.3} m", HANG_CELL));
    readout(ui, "Poisson ν", &format!("{:.2}", HANG_NU));
    readout(ui, "Base E", &format!("{:.2e} Pa", HANG_BASE_E));
    ui.separator();
    ui.label("Per-block Young's modulus & tip droop");
    for blk in 0..4usize {
        let e = HANG_BASE_E * HANG_E_SWEEP[blk];
        let droop = HANG_BASE_Y - st.tip_y[blk];
        readout(
            ui,
            &format!("blk {} (×{:.0})", blk, HANG_E_SWEEP[blk]),
            &format!("E = {:.2e} | droop = {:.3} m", e, droop),
        );
    }
    ui.separator();
    ui.small("Each block's i=0 face is pinned (left face).");
    ui.small("Colour ramp blue → red ≡ soft → stiff.");
}

