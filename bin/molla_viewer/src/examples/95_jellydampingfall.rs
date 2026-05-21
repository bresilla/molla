// ═══ Scene 95: JellyDampingFall (Stage 36/41) ══════════════════════════
//
// Four identical jelly tet cubes dropped from a height onto the ground,
// each colour-mapped to a target damping value (0.1, 0.5, 1.0, 2.0
// 1/s). Same VBD Neo-Hookean material parameters; the global linear
// damping is set to 0.5/s (the second-lowest swept value) — per-block
// damping needs a per-particle slot (Stage 41 v2 follow-up) so this scene is the
// pedagogical "what the sweep should look like" while we wait for
// that slot to land.

const JDF_DIM: u32 = 4;
const JDF_CELL: f64 = 0.10;
const JDF_SPACING_X: f64 = 0.65;
const JDF_DROP_Y: f64 = 1.6;
const JDF_PARTICLE_MASS: f64 = 0.10;
const JDF_E: f64 = 5.0e4;
const JDF_NU: f64 = 0.40;
const JDF_DAMPING_SWEEP: [f64; 4] = [0.10, 0.50, 1.0, 2.0];

struct JellyDampingFallState {
    com_y: [f64; 4],
    mean_speed: [f64; 4],
}

impl Default for JellyDampingFallState {
    fn default() -> Self {
        Self {
            com_y: [JDF_DROP_Y; 4],
            mean_speed: [0.0; 4],
        }
    }
}

fn jdf_cube_to_six_tets(c: [u32; 8]) -> [[u32; 4]; 6] {
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

fn build_jelly_damping_fall_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.5);
    b.set_angular_damping(0.10);

    let nx = JDF_DIM + 1;
    let ny = JDF_DIM + 1;
    let nz = JDF_DIM + 1;
    let dx = JDF_CELL;

    for (blk, &_damp) in JDF_DAMPING_SWEEP.iter().enumerate() {
        let centre_x = (blk as f64 - 1.5) * JDF_SPACING_X;
        let drop = JDF_DROP_Y - 0.05 * blk as f64;
        let origin = MVec3::new(
            centre_x - (JDF_CELL * JDF_DIM as f64) * 0.5,
            drop,
            -(JDF_CELL * JDF_DIM as f64) * 0.5,
        );
        let mut block_first: u32 = u32::MAX;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let p = origin + MVec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                    let id = b.add_particle(p, JDF_PARTICLE_MASS, dx * 0.5, mp::WorldId(0));
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
                    for tet in jdf_cube_to_six_tets(corners) {
                        tets.push(tet);
                    }
                }
            }
        }
        let mat_idx: Vec<u32> = vec![0; tets.len()];
        let materials = vec![mp::MaterialNeoHookean::elastic(JDF_E, JDF_NU)];
        let _ = b
            .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
            .expect("jelly damping fall tet mesh");
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("jelly damping fall model"), Vec::new())
}

fn update_jelly_damping_fall(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::JellyDampingFall {
        return;
    }
    let Ok(pq) = sim.state_a.particle_q.host() else {
        return;
    };
    let Ok(pqd) = sim.state_a.particle_qd.host() else {
        return;
    };
    let nx = JDF_DIM + 1;
    let ny = JDF_DIM + 1;
    let nz = JDF_DIM + 1;
    let per_block = (nx * ny * nz) as usize;
    let st = &mut warp.jelly_damping;
    for blk in 0..4usize {
        let base = blk * per_block;
        let mut sum_y = 0.0_f64;
        let mut sum_v = 0.0_f64;
        let mut count = 0_usize;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = base + ((k * ny + j) * nx + i) as usize;
                    if idx < pq.len() {
                        sum_y += pq[idx].y;
                        sum_v += pqd[idx].length();
                        count += 1;
                    }
                }
            }
        }
        if count > 0 {
            st.com_y[blk] = sum_y / count as f64;
            st.mean_speed[blk] = sum_v / count as f64;
        }
    }
}

fn draw_jelly_damping_fall_gizmos(sim: Res<Sim>, mut gizmos: Gizmos) {
    if sim.scene != Scene::JellyDampingFall {
        return;
    }
    let Ok(pq) = sim.state_a.particle_q.host() else {
        return;
    };
    let nx = JDF_DIM + 1;
    let ny = JDF_DIM + 1;
    let nz = JDF_DIM + 1;
    let per_block = (nx * ny * nz) as usize;
    for blk in 0..4usize {
        let base = blk * per_block;
        let s = JDF_DAMPING_SWEEP[blk] / JDF_DAMPING_SWEEP[3];
        let t = s.clamp(0.0, 1.0) as f32;
        let col = Color::srgb(0.15 + 0.80 * t, 0.40, 0.95 - 0.80 * t);
        let v = |i: u32, j: u32, k: u32| -> Vec3 {
            let p = pq[base + ((k * ny + j) * nx + i) as usize];
            Vec3::new(p.x as f32, p.y as f32, p.z as f32)
        };
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx - 1 {
                    gizmos.line(v(i, j, k), v(i + 1, j, k), col);
                }
            }
        }
        for k in 0..nz {
            for j in 0..ny - 1 {
                for i in 0..nx {
                    gizmos.line(v(i, j, k), v(i, j + 1, k), col);
                }
            }
        }
        for k in 0..nz - 1 {
            for j in 0..ny {
                for i in 0..nx {
                    gizmos.line(v(i, j, k), v(i, j, k + 1), col);
                }
            }
        }
    }
}

fn jelly_damping_fall_pane(ui: &mut egui::Ui, st: &mut JellyDampingFallState) {
    ui.label("Stage 36/41 — 4 jelly tet cubes, damping sweep");
    ui.separator();
    readout(ui, "Grid", &format!("{0}×{0}×{0} cells / block", JDF_DIM));
    readout(ui, "E", &format!("{:.2e} Pa", JDF_E));
    readout(ui, "ν", &format!("{:.2}", JDF_NU));
    ui.separator();
    ui.label("Per-block damping & live readout");
    for blk in 0..4usize {
        readout(
            ui,
            &format!("blk {} (ζ_target = {:.2}/s)", blk, JDF_DAMPING_SWEEP[blk]),
            &format!(
                "y = {:.3} m | |v| = {:.3} m/s",
                st.com_y[blk], st.mean_speed[blk]
            ),
        );
    }
    ui.separator();
    ui.small(
        "Global damping is set to 0.5/s (one value from the sweep); per-block \
         damping would need a per-particle slot (Stage 41 v2). The colour ramp \
         (blue→red) still indicates the target damping value for each \
         block — useful for spotting which one is which while watching \
         the fall.",
    );
}

