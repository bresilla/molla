// 124 — differentiable-vehicle system identification
// (`molla_vehicle::sysid`). A ghost car was driven on a hidden
// tyre-ground friction; a live Gauss-Newton loop recovers that
// friction from the ghost's travelled distance alone, one iteration
// per frame, while a candidate car drives on the current estimate so
// the two visibly converge. The μ-iteration trace is drawn as a gizmo
// strip chart with the hidden truth as a reference line.

/// Rollout length used for both the planted target distance and every
/// optimizer rollout. Matches the molla-vehicle sysid test regime
/// (thousands of 1 ms steps of the 3-DOF chassis — cheap on the host).
const SYSID_ROLLOUT_STEPS: usize = 3000;
const SYSID_ROLLOUT_DT: f64 = 1.0e-3;
/// Hidden ground truth the optimizer must recover.
const SYSID_TRUE_MU: f64 = 0.75;
/// Deliberately wrong initial guess (icy estimate for a grippy road).
const SYSID_MU0: f64 = 0.35;
/// Gauss-Newton iterations to run (one per frame, so convergence is
/// watchable; the crate test converges in well under 20).
const SYSID_ITERS: usize = 16;
/// Respawn both demo cars once the candidate passes this x (m).
const SYSID_LOOP_X: f64 = 22.0;

// Per-frame state for [`Scene::VehicleSysid`].
#[derive(Resource, Default)]
struct VehicleSysidState {
    was_active: bool,
    /// Distance the ghost car travelled on the hidden friction.
    target_distance: f64,
    /// Current friction estimate.
    mu: f64,
    /// μ after each completed Gauss-Newton iteration (index 0 = μ₀).
    mu_history: Vec<f64>,
    /// Candidate car driving live on the current μ estimate.
    candidate: Option<molla_vehicle::Vehicle>,
    candidate_powertrain: molla_vehicle::Powertrain,
    /// Ghost car driving live on the hidden true μ.
    ghost: Option<molla_vehicle::Vehicle>,
    ghost_powertrain: molla_vehicle::Powertrain,
}

// Advance one Gauss-Newton identification step per frame and drive the
// candidate + ghost cars live. `step_simulation` skips this scene; the
// candidate car's poses are written straight into the host body buffer
// (chassis + 4 wheels, same rig as `Scene::Vehicle`), the ghost is
// drawn as a wireframe cuboid.
fn update_vehicle_sysid_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<VehicleSysidState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::VehicleSysid {
        state.was_active = false;
        return;
    }
    if sim.model.body_count < 5 {
        return;
    }
    if !state.was_active {
        // Plant the trajectory the optimizer has to explain: drive the
        // standard car on the hidden friction and record its distance.
        state.target_distance = molla_vehicle::drive_distance(
            SYSID_TRUE_MU,
            SYSID_ROLLOUT_STEPS,
            SYSID_ROLLOUT_DT,
        );
        state.mu = SYSID_MU0;
        state.mu_history = vec![SYSID_MU0];
        state.candidate = Some(molla_vehicle::Vehicle::car());
        state.candidate_powertrain = molla_vehicle::Powertrain::default();
        state.ghost = Some(molla_vehicle::Vehicle::car());
        state.ghost_powertrain = molla_vehicle::Powertrain::default();
        state.was_active = true;
    }

    // ── One Gauss-Newton iteration per frame (mirrors
    // `molla_vehicle::identify_friction`, unrolled so the viewer can
    // plot every iterate). Finite-difference sensitivity through the
    // full rollout — the "differentiable simulation" beat.
    if state.mu_history.len() <= SYSID_ITERS {
        let eps = 1.0e-3;
        let d = molla_vehicle::drive_distance(state.mu, SYSID_ROLLOUT_STEPS, SYSID_ROLLOUT_DT);
        let sens = (molla_vehicle::drive_distance(
            state.mu + eps,
            SYSID_ROLLOUT_STEPS,
            SYSID_ROLLOUT_DT,
        ) - molla_vehicle::drive_distance(
            state.mu - eps,
            SYSID_ROLLOUT_STEPS,
            SYSID_ROLLOUT_DT,
        )) / (2.0 * eps);
        if sens.abs() >= 1.0e-9 {
            state.mu = (state.mu - (d - state.target_distance) / sens).clamp(0.1, 1.5);
        }
        let mu = state.mu;
        state.mu_history.push(mu);
    }

    // ── Drive both cars live: ghost on the hidden truth, candidate on
    // the current estimate. Full throttle, straight line, looped.
    let input = molla_vehicle::DriverInput {
        throttle: 1.0,
        brake: 0.0,
        steer: 0.0,
    };
    let frame = (time.delta_secs() as f64).clamp(0.0, 1.0 / 20.0);
    let steps = ((frame / 1.0e-3).round() as u32).clamp(1, 64);
    let candidate_terrain = molla_vehicle::FlatTerrain::new(0.0, state.mu);
    let ghost_terrain = molla_vehicle::FlatTerrain::new(0.0, SYSID_TRUE_MU);
    {
        let mut cpt = state.candidate_powertrain.clone();
        let c = state.candidate.as_mut().unwrap();
        for _ in 0..steps {
            c.apply_driver(&mut cpt, &input);
            c.step(&candidate_terrain, 1.0e-3);
        }
        state.candidate_powertrain = cpt;
    }
    {
        let mut gpt = state.ghost_powertrain.clone();
        let g = state.ghost.as_mut().unwrap();
        for _ in 0..steps {
            g.apply_driver(&mut gpt, &input);
            g.step(&ghost_terrain, 1.0e-3);
        }
        state.ghost_powertrain = gpt;
    }
    if state.candidate.as_ref().unwrap().pos[0] > SYSID_LOOP_X
        || state.ghost.as_ref().unwrap().pos[0] > SYSID_LOOP_X
    {
        state.candidate = Some(molla_vehicle::Vehicle::car());
        state.candidate_powertrain = molla_vehicle::Powertrain::default();
        state.ghost = Some(molla_vehicle::Vehicle::car());
        state.ghost_powertrain = molla_vehicle::Powertrain::default();
    }

    // ── Candidate car → host body buffer (chassis + wheels rig).
    {
        let v = state.candidate.as_ref().unwrap();
        let chassis_pos = MVec3::new(v.pos[0], v.pos[1], v.pos[2]);
        let chassis_rot = MQuat::from_rotation_y(v.yaw);
        let wheels: Vec<(MVec3, MQuat)> = (0..4)
            .map(|i| {
                let p = v.corner_world_pos(i);
                (
                    MVec3::new(p[0], v.corners[i].wheel_radius, p[2]),
                    MQuat::from_rotation_y(v.yaw + v.corners[i].steer),
                )
            })
            .collect();
        if let Ok(body_q) = sim.state_a.body_q.host_mut()
            && body_q.len() >= 5 {
                body_q[0].position = chassis_pos;
                body_q[0].rotation = chassis_rot;
                for (i, w) in wheels.iter().enumerate() {
                    body_q[1 + i].position = w.0;
                    body_q[1 + i].rotation = w.1;
                }
            }
    }

    // ── Ghost car: wireframe box offset one lane over (drawn, not
    // simulated by the engine). It drives dead straight (steer 0), so
    // an axis-aligned box is faithful.
    {
        let g = state.ghost.as_ref().unwrap();
        let ghost_color = Color::srgb(0.35, 0.9, 0.5);
        let center = Vec3::new(g.pos[0] as f32, g.pos[1] as f32, g.pos[2] as f32 + 3.0);
        let half = Vec3::new(1.5, 0.35, 0.85);
        draw_aabb_edges(&mut gizmos, center - half, center + half, ghost_color);
        // Distance-gap bar between the two cars' noses.
        let c = state.candidate.as_ref().unwrap();
        gizmos.line(
            Vec3::new(c.pos[0] as f32, 0.05, c.pos[2] as f32),
            Vec3::new(g.pos[0] as f32, 0.05, g.pos[2] as f32 + 3.0),
            Color::srgb(0.95, 0.4, 0.3),
        );
    }

    // ── μ-convergence strip chart: iterates in orange, hidden truth as
    // a green reference line, initial guess in grey.
    {
        let base = Vec3::new(-6.0, 2.2, -5.0);
        let dx = 0.55_f32;
        let scale_y = 2.0_f32;
        let truth_y = base.y + (SYSID_TRUE_MU as f32) * scale_y;
        gizmos.line(
            Vec3::new(base.x, truth_y, base.z),
            Vec3::new(base.x + dx * (SYSID_ITERS as f32), truth_y, base.z),
            Color::srgb(0.35, 0.9, 0.5),
        );
        let trace = Color::srgb(0.98, 0.62, 0.20);
        for (i, pair) in state.mu_history.windows(2).enumerate() {
            let a = Vec3::new(
                base.x + dx * i as f32,
                base.y + (pair[0] as f32) * scale_y,
                base.z,
            );
            let b = Vec3::new(
                base.x + dx * (i + 1) as f32,
                base.y + (pair[1] as f32) * scale_y,
                base.z,
            );
            gizmos.line(a, b, trace);
            gizmos.line(b + Vec3::Y * 0.03, b - Vec3::Y * 0.03, trace);
        }
    }
}
