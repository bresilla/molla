//! Newton oracle harness — shells out to a Python `newton-physics`
//! install, runs a declarative scene, and reads back a trajectory.
//!
//! Scenes are described as a Rust struct ([`Scene`]) and serialised to
//! JSON for the Python side. The Python script (embedded as a string in
//! [`PYTHON_RUNNER`]) loads the scene into Newton's `ModelBuilder`, picks
//! the named solver, runs N steps, and prints a JSON trajectory.

use std::path::PathBuf;
use std::process::Command;

use molla::math::{Real, SpatialVector, Transform, Vec3};

/// Resolve the Python interpreter to use. Returns `None` if the user
/// hasn't pointed us at one — the test should be skipped in that case.
pub fn resolve_python() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MOLLA_NEWTON_PYTHON") {
        return Some(PathBuf::from(p));
    }
    let candidates = [
        "../../xtra/newton/.venv/bin/python",
        "xtra/newton/.venv/bin/python",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// One body in a scene description.
#[derive(Debug, Clone)]
pub struct OracleBody {
    pub mass: Real,
    pub radius: Real,
    pub position: Vec3,
    pub linear_velocity: Vec3,
}

/// A trivial scene — bodies under gravity, no contacts. Stage 1 only
/// needs this; richer scenes follow.
#[derive(Debug, Clone)]
pub struct Scene {
    pub gravity: Vec3,
    pub bodies: Vec<OracleBody>,
    pub solver: &'static str, // "semi_implicit"
    pub dt: Real,
    pub steps: usize,
}

/// Recorded trajectory. `body_q[step][body]` is the pose at end of step.
#[derive(Debug, Default)]
pub struct Trajectory {
    pub body_q: Vec<Vec<Transform>>,
    pub body_qd: Vec<Vec<SpatialVector>>,
}

/// Run the scene through Newton (Python) and parse the trajectory back.
pub fn run_with_newton(scene: &Scene) -> Result<Trajectory, String> {
    let python = resolve_python().ok_or_else(|| {
        "no python with newton-physics — set MOLLA_NEWTON_PYTHON".to_string()
    })?;

    let scene_json = scene_to_json(scene);
    let out = Command::new(python)
        .arg("-c")
        .arg(PYTHON_RUNNER)
        .env("MOLLA_SCENE", &scene_json)
        .output()
        .map_err(|e| format!("failed to spawn python: {e}"))?;

    if !out.status.success() {
        return Err(format!(
            "python exited {}; stderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }

    parse_trajectory(&String::from_utf8_lossy(&out.stdout))
}

fn scene_to_json(scene: &Scene) -> String {
    let bodies: Vec<String> = scene
        .bodies
        .iter()
        .map(|b| {
            format!(
                r#"{{"mass":{},"radius":{},"pos":[{},{},{}],"vel":[{},{},{}]}}"#,
                b.mass,
                b.radius,
                b.position.x,
                b.position.y,
                b.position.z,
                b.linear_velocity.x,
                b.linear_velocity.y,
                b.linear_velocity.z,
            )
        })
        .collect();
    format!(
        r#"{{"gravity":[{},{},{}],"bodies":[{}],"solver":"{}","dt":{},"steps":{}}}"#,
        scene.gravity.x,
        scene.gravity.y,
        scene.gravity.z,
        bodies.join(","),
        scene.solver,
        scene.dt,
        scene.steps,
    )
}

fn parse_trajectory(s: &str) -> Result<Trajectory, String> {
    // The Python runner prints lines of the form:
    //   STEP <i> Q <j> <x> <y> <z> <qx> <qy> <qz> <qw>
    //   STEP <i> QD <j> <vx> <vy> <vz> <wx> <wy> <wz>
    // Plus a single line `META <body_count> <steps>`.
    let mut body_count = 0usize;
    let mut step_count = 0usize;
    for line in s.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("META ") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(format!("bad META line: {line}"));
            }
            body_count = parts[0]
                .parse()
                .map_err(|e| format!("bad META body_count: {e}"))?;
            step_count = parts[1]
                .parse()
                .map_err(|e| format!("bad META steps: {e}"))?;
        }
    }
    if body_count == 0 || step_count == 0 {
        return Err(format!("no META line in python output:\n{s}"));
    }

    let mut traj = Trajectory {
        body_q: vec![vec![Transform::IDENTITY; body_count]; step_count],
        body_qd: vec![vec![SpatialVector::ZERO; body_count]; step_count],
    };
    for line in s.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 || parts[0] != "STEP" {
            continue;
        }
        let step: usize = parts[1].parse().map_err(|_| "bad step idx")?;
        let kind = parts[2];
        let body: usize = parts[3].parse().map_err(|_| "bad body idx")?;
        let nums: Vec<Real> = parts[4..]
            .iter()
            .map(|p| p.parse::<Real>())
            .collect::<Result<_, _>>()
            .map_err(|e| format!("bad number: {e}"))?;
        match kind {
            "Q" if nums.len() == 7 => {
                traj.body_q[step][body] = Transform::new(
                    Vec3::new(nums[0], nums[1], nums[2]),
                    glam::DQuat::from_xyzw(nums[3], nums[4], nums[5], nums[6]),
                );
            }
            "QD" if nums.len() == 6 => {
                traj.body_qd[step][body] = SpatialVector::new(
                    Vec3::new(nums[0], nums[1], nums[2]),
                    Vec3::new(nums[3], nums[4], nums[5]),
                );
            }
            _ => {}
        }
    }
    Ok(traj)
}

/// Embedded Python script that drives Newton for a given scene.
///
/// The script reads the scene JSON from `$MOLLA_SCENE`, builds a
/// `newton.ModelBuilder`, runs the named solver, and prints the trajectory
/// in the line-oriented format that [`parse_trajectory`] consumes.
pub const PYTHON_RUNNER: &str = r#"
import json, os, sys
try:
    import warp as wp
    import newton
    from newton.solvers import SemiImplicitSolver
except Exception as e:
    print(f"IMPORT_ERROR {e}", file=sys.stderr)
    sys.exit(2)

scene = json.loads(os.environ["MOLLA_SCENE"])
gx, gy, gz = scene["gravity"]
dt = float(scene["dt"])
n_steps = int(scene["steps"])

builder = newton.ModelBuilder()
builder.gravity = [gx, gy, gz]

for b in scene["bodies"]:
    m = float(b["mass"])
    r = float(b["radius"])
    px, py, pz = b["pos"]
    vx, vy, vz = b["vel"]
    body = builder.add_body()
    builder.add_shape_sphere(body=body, radius=r, density=0.0)
    builder.body_mass[body] = m
    builder.body_inv_mass[body] = 1.0 / m if m > 0 else 0.0
    # Sphere inertia
    inertia = 0.4 * m * r * r
    import numpy as np
    I = np.diag([inertia, inertia, inertia])
    builder.body_inertia[body] = I
    builder.body_inv_inertia[body] = np.linalg.inv(I) if m > 0 else np.zeros((3,3))
    builder.body_q[body] = wp.transform([px, py, pz], wp.quat_identity())
    builder.body_qd[body] = wp.spatial_vector(vx, vy, vz, 0.0, 0.0, 0.0)

model = builder.finalize()
state_a = model.state()
state_b = model.state()
control = model.control()
contacts = newton.Contacts()

solver_name = scene["solver"]
if solver_name == "semi_implicit":
    solver = SemiImplicitSolver(model, angular_damping=0.0)
else:
    print(f"UNKNOWN_SOLVER {solver_name}", file=sys.stderr)
    sys.exit(3)

print(f"META {model.body_count} {n_steps}")
for i in range(n_steps):
    state_a.clear_forces()
    solver.step(state_a, state_b, control, contacts, dt)
    state_a, state_b = state_b, state_a
    q = state_a.body_q.numpy()
    qd = state_a.body_qd.numpy()
    for j in range(model.body_count):
        x, y, z = q[j][0], q[j][1], q[j][2]
        rx, ry, rz, rw = q[j][3], q[j][4], q[j][5], q[j][6]
        print(f"STEP {i} Q {j} {x} {y} {z} {rx} {ry} {rz} {rw}")
        # Newton's spatial layout: top = linear, bottom = angular.
        vx_, vy_, vz_, wx_, wy_, wz_ = qd[j]
        print(f"STEP {i} QD {j} {vx_} {vy_} {vz_} {wx_} {wy_} {wz_}")
"#;
