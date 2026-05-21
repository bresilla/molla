// ═══ Scene 113: IkCubeStacking — reduced procedural pick/place IK ═══

const IK_STACK_CUBE_COUNT: usize = 4;
const IK_STACK_CUBE_HALF: f64 = 0.055;
const IK_STACK_GRIPPER_CLEARANCE: f64 = 0.15;
const IK_STACK_PHASE_FRAMES: u32 = 36;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum IkCubeStackingPhase {
    ApproachPick,
    Grasp,
    Lift,
    ApproachPlace,
    Place,
    Retreat,
    Done,
}

impl IkCubeStackingPhase {
    fn label(self) -> &'static str {
        match self {
            Self::ApproachPick => "approach pick",
            Self::Grasp => "grasp",
            Self::Lift => "lift",
            Self::ApproachPlace => "approach place",
            Self::Place => "place",
            Self::Retreat => "retreat",
            Self::Done => "done",
        }
    }
}

struct IkCubeStackingState {
    cube_centers: [MVec3; IK_STACK_CUBE_COUNT],
    active_cube: usize,
    placed_count: usize,
    phase: IkCubeStackingPhase,
    phase_frames: u32,
    ee_target: MVec3,
    obstacle_center: MVec3,
    obstacle_radius: f64,
    obstacle_margin: f64,
    last_iters: usize,
    last_residual: f64,
    last_solve_us: f64,
    stack_support_error: f64,
    obstacle_residual: f64,
}

impl Default for IkCubeStackingState {
    fn default() -> Self {
        let cube_centers = ik_cube_stacking_initial_centers();
        let mut me = Self {
            cube_centers,
            active_cube: 0,
            placed_count: 0,
            phase: IkCubeStackingPhase::ApproachPick,
            phase_frames: 0,
            ee_target: MVec3::ZERO,
            obstacle_center: MVec3::new(0.05, 0.38, 0.38),
            obstacle_radius: 0.09,
            obstacle_margin: 0.04,
            last_iters: 0,
            last_residual: f64::INFINITY,
            last_solve_us: 0.0,
            stack_support_error: 0.0,
            obstacle_residual: 0.0,
        };
        me.ee_target = ik_cube_stacking_phase_target(&me);
        me
    }
}

fn ik_cube_stacking_initial_centers() -> [MVec3; IK_STACK_CUBE_COUNT] {
    [
        MVec3::new(-0.30, IK_STACK_CUBE_HALF, 0.38),
        MVec3::new(-0.16, IK_STACK_CUBE_HALF, 0.38),
        MVec3::new(-0.02, IK_STACK_CUBE_HALF, 0.38),
        MVec3::new(0.12, IK_STACK_CUBE_HALF, 0.38),
    ]
}

fn ik_cube_stacking_place_center(level: usize) -> MVec3 {
    MVec3::new(
        0.34,
        IK_STACK_CUBE_HALF + 2.0 * IK_STACK_CUBE_HALF * level as f64,
        0.36,
    )
}

fn ik_cube_stacking_phase_target(st: &IkCubeStackingState) -> MVec3 {
    if st.phase == IkCubeStackingPhase::Done {
        let top = ik_cube_stacking_place_center(st.placed_count.saturating_sub(1));
        return top + MVec3::Y * (IK_STACK_GRIPPER_CLEARANCE + 0.10);
    }
    let cube = st.cube_centers[st.active_cube.min(IK_STACK_CUBE_COUNT - 1)];
    let place = ik_cube_stacking_place_center(st.placed_count.min(IK_STACK_CUBE_COUNT - 1));
    match st.phase {
        IkCubeStackingPhase::ApproachPick => cube + MVec3::Y * IK_STACK_GRIPPER_CLEARANCE,
        IkCubeStackingPhase::Grasp => cube + MVec3::Y * 0.055,
        IkCubeStackingPhase::Lift => cube + MVec3::Y * 0.30,
        IkCubeStackingPhase::ApproachPlace => place + MVec3::Y * 0.30,
        IkCubeStackingPhase::Place => place + MVec3::Y * 0.070,
        IkCubeStackingPhase::Retreat => place + MVec3::Y * 0.28,
        IkCubeStackingPhase::Done => unreachable!(),
    }
}

fn ik_cube_stacking_stack_support_error(cubes: &[MVec3], placed_count: usize) -> f64 {
    if !cubes
        .iter()
        .all(|c| c.x.is_finite() && c.y.is_finite() && c.z.is_finite())
    {
        return f64::INFINITY;
    }
    let mut err = 0.0_f64;
    for (level, cube) in cubes.iter().take(placed_count).enumerate() {
        let expected = ik_cube_stacking_place_center(level);
        err = err.max((cube.x - expected.x).abs());
        err = err.max((cube.y - expected.y).abs());
        err = err.max((cube.z - expected.z).abs());
        if level == 0 {
            err = err.max((cube.y - IK_STACK_CUBE_HALF).abs());
        } else {
            let below = cubes[level - 1];
            let gap = cube.y - below.y - 2.0 * IK_STACK_CUBE_HALF;
            err = err.max(gap.abs());
        }
    }
    err
}

fn ik_cube_stacking_obstacle_residual(
    p: MVec3,
    obstacle_center: MVec3,
    obstacle_radius: f64,
    margin: f64,
) -> f64 {
    (obstacle_radius + margin - (p - obstacle_center).length()).max(0.0)
}

#[cfg(test)]
fn ik_cube_stacking_targets_are_reachable(cubes: &[MVec3; IK_STACK_CUBE_COUNT]) -> bool {
    let base = MVec3::new(0.0, GPU_IK_BASE_HEIGHT, 0.0);
    cubes
        .iter()
        .copied()
        .chain((0..IK_STACK_CUBE_COUNT).map(ik_cube_stacking_place_center))
        .map(|p| p + MVec3::Y * 0.30)
        .all(|target| (target - base).length() < 1.10 && target.y > 0.0)
}

fn ik_cube_stacking_advance_phase(st: &mut IkCubeStackingState) {
    st.phase_frames = 0;
    st.phase = match st.phase {
        IkCubeStackingPhase::ApproachPick => IkCubeStackingPhase::Grasp,
        IkCubeStackingPhase::Grasp => IkCubeStackingPhase::Lift,
        IkCubeStackingPhase::Lift => IkCubeStackingPhase::ApproachPlace,
        IkCubeStackingPhase::ApproachPlace => IkCubeStackingPhase::Place,
        IkCubeStackingPhase::Place => {
            st.cube_centers[st.active_cube] = ik_cube_stacking_place_center(st.placed_count);
            st.placed_count += 1;
            if st.placed_count >= IK_STACK_CUBE_COUNT {
                IkCubeStackingPhase::Done
            } else {
                st.active_cube = st.placed_count;
                IkCubeStackingPhase::Retreat
            }
        }
        IkCubeStackingPhase::Retreat => IkCubeStackingPhase::ApproachPick,
        IkCubeStackingPhase::Done => IkCubeStackingPhase::Done,
    };
    st.ee_target = ik_cube_stacking_phase_target(st);
}

fn update_ik_cube_stacking(mut sim: ResMut<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::IkCubeStacking {
        return;
    }
    let st = &mut warp.ik_cube_stacking;

    if matches!(
        st.phase,
        IkCubeStackingPhase::Lift | IkCubeStackingPhase::ApproachPlace | IkCubeStackingPhase::Place
    ) && st.active_cube < IK_STACK_CUBE_COUNT
        && st.placed_count <= st.active_cube
    {
        st.cube_centers[st.active_cube] = st.ee_target - MVec3::Y * 0.075;
    }
    st.stack_support_error =
        ik_cube_stacking_stack_support_error(&st.cube_centers, st.placed_count);
    st.obstacle_residual = ik_cube_stacking_obstacle_residual(
        st.ee_target,
        st.obstacle_center,
        st.obstacle_radius,
        st.obstacle_margin,
    );

    let Sim {
        model,
        state_a,
        engine,
        ..
    } = &mut *sim;
    let ee_body = model.body_count - 1;
    let q0: Vec<f64> = state_a
        .joint_q
        .host()
        .expect("joint_q")
        .iter()
        .copied()
        .collect();
    let solver = IkSolver::new(vec![IkObjective::Position {
        body: ee_body,
        point_local: MVec3::ZERO,
        target_world: st.ee_target,
        weight: 1.0,
    }])
    .with_config(IkConfig {
        max_iter: 30,
        residual_tol: 6.0e-4,
        ..IkConfig::default()
    });
    let t0 = std::time::Instant::now();
    let result = solver.solve_gpu(engine, model, state_a, &q0);
    st.last_solve_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
    match result {
        Ok(r) => {
            st.last_iters = r.iters;
            st.last_residual = r.residual;
        }
        Err(e) => warn!("ik_cube_stacking: solve failed: {e}"),
    }

    st.phase_frames += 1;
    if st.phase != IkCubeStackingPhase::Done
        && (st.phase_frames >= IK_STACK_PHASE_FRAMES || st.last_residual < 0.010)
    {
        ik_cube_stacking_advance_phase(st);
    }
}

fn draw_wire_cube(gizmos: &mut Gizmos, center: Vec3, half: f32, color: Color) {
    let h = Vec3::splat(half);
    let corners = [
        center + Vec3::new(-h.x, -h.y, -h.z),
        center + Vec3::new(h.x, -h.y, -h.z),
        center + Vec3::new(h.x, h.y, -h.z),
        center + Vec3::new(-h.x, h.y, -h.z),
        center + Vec3::new(-h.x, -h.y, h.z),
        center + Vec3::new(h.x, -h.y, h.z),
        center + Vec3::new(h.x, h.y, h.z),
        center + Vec3::new(-h.x, h.y, h.z),
    ];
    for (a, b) in [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ] {
        gizmos.line(corners[a], corners[b], color);
    }
}

fn draw_ik_cube_stacking_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::IkCubeStacking {
        return;
    }
    let st = &warp.ik_cube_stacking;
    let to_v = |p: MVec3| Vec3::new(p.x as f32, p.y as f32, p.z as f32);
    for (i, cube) in st.cube_centers.iter().enumerate() {
        let color = if i < st.placed_count {
            Color::srgb(0.35, 0.95, 0.50)
        } else if i == st.active_cube && st.phase != IkCubeStackingPhase::Done {
            Color::srgb(1.0, 0.75, 0.25)
        } else {
            Color::srgb(0.55, 0.75, 1.0)
        };
        draw_wire_cube(&mut gizmos, to_v(*cube), IK_STACK_CUBE_HALF as f32, color);
    }

    for level in 0..IK_STACK_CUBE_COUNT {
        let p = ik_cube_stacking_place_center(level);
        draw_wire_cube(
            &mut gizmos,
            to_v(p),
            IK_STACK_CUBE_HALF as f32,
            Color::srgba(0.25, 1.0, 0.45, 0.35),
        );
    }

    let target = to_v(st.ee_target);
    gizmos.sphere(target, 0.035, Color::srgb(1.0, 0.25, 0.85));
    let obs = to_v(st.obstacle_center);
    let obs_color = if st.obstacle_residual > 0.0 {
        Color::srgb(1.0, 0.20, 0.15)
    } else {
        Color::srgb(0.35, 0.55, 1.0)
    };
    gizmos.sphere(obs, st.obstacle_radius as f32, obs_color);
    gizmos.line(
        target,
        obs,
        if st.obstacle_residual > 0.0 {
            Color::srgb(1.0, 0.25, 0.20)
        } else {
            Color::srgb(0.25, 0.45, 0.95)
        },
    );
}

fn ik_cube_stacking_pane(ui: &mut egui::Ui, st: &mut IkCubeStackingState) {
    ui.label("IK cube stacking — reduced procedural pick/place");
    ui.separator();
    readout(ui, "Solver", "GPU LM (Stage 16)");
    readout(ui, "Phase", st.phase.label());
    readout(
        ui,
        "Placed",
        &format!("{}/{}", st.placed_count, IK_STACK_CUBE_COUNT),
    );
    readout(ui, "Iterations", &format!("{}", st.last_iters));
    readout(ui, "Residual", &format!("{:.5}", st.last_residual));
    readout(
        ui,
        "ms / solve",
        &format!("{:.2}", st.last_solve_us / 1000.0),
    );
    readout(
        ui,
        "Support err",
        &format!("{:.2e}", st.stack_support_error),
    );
    readout(
        ui,
        "Obstacle residual",
        &format!("{:.3}", st.obstacle_residual),
    );
    ui.separator();
    ui.label("Obstacle telemetry sphere");
    ui.add(egui::Slider::new(&mut st.obstacle_center.x, -0.4..=0.6).text("cx"));
    ui.add(egui::Slider::new(&mut st.obstacle_center.y, 0.05..=0.9).text("cy"));
    ui.add(egui::Slider::new(&mut st.obstacle_center.z, 0.0..=0.8).text("cz"));
    ui.add(egui::Slider::new(&mut st.obstacle_radius, 0.03..=0.25).text("radius"));
    ui.add(egui::Slider::new(&mut st.obstacle_margin, 0.0..=0.20).text("margin"));
    if ui.button("Reset stacking").clicked() {
        *st = IkCubeStackingState::default();
    }
    ui.small(
        "Cubes are procedural viewer telemetry; gripper/contact dynamics are not coupled yet.",
    );
    ui.small("Obstacle residual is reported honestly but not yet an avoidance objective.");
}

