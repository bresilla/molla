#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::{BodyId, WorldId};
use molla_kernels::soft_rigid_contact::SoftRigidShapeGpu;
use molla_math::{Mat3, Quat, Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, JointTargetMode, ModelBuilder, eval_fk};
use molla_solvers::fem_rigid::FemRigidConfig;
use molla_solvers::fem_rigid_gpu::{FemRigidGpuScene, FemRigidGpuSystem};
use std::sync::Arc;

#[path = "common/reinforced_track.rs"]
mod reinforced_track;

#[path = "common/fem_readback.rs"]
mod fem_readback;
#[path = "common/gpu_readback.rs"]
mod gpu_readback;
use fem_readback::body_poses;
use gpu_readback::read;

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn installed_reinforced_track_engages_a_toothed_sprocket() {
    track_loop(0.02, 1.0 / 4800.0, 4, false, 1.0);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn installed_lightly_tensioned_track_engages_a_toothed_sprocket() {
    track_loop(0.005, 1.0 / 4800.0, 4, false, 1.0);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn installed_refined_track_engages_a_toothed_sprocket() {
    track_loop(0.005, 1.0 / 9600.0, 4, false, 1.0);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn installed_reversed_refined_track_engages_a_toothed_sprocket() {
    track_loop(0.005, 1.0 / 9600.0, 4, false, -1.0);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn installed_reversed_fine_track_engages_a_toothed_sprocket() {
    track_loop(0.005, 1.0 / 19200.0, 4, false, -1.0);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn installed_width_refined_track_engages_a_toothed_sprocket() {
    track_loop(0.005, 1.0 / 4800.0, 8, false, 1.0);
}

#[test]
fn installed_track_nodes_clear_rigid_shapes() {
    track_loop(0.005, 1.0 / 4800.0, 4, true, 1.0);
}

#[test]
fn coarse_track_surface_sampling_detects_missed_intersections() {
    let report = track_run_resolution(0.005, 1.0 / 19200.0, 4, true, 1.0, None, true, 112, false);
    assert!(report.surface_penetration > 0.0005);
}

#[test]
fn longitudinally_refined_track_surface_clears_rigid_shapes() {
    for segments in [224, 448] {
        let report = track_run_resolution(
            0.005,
            1.0 / 19200.0,
            4,
            true,
            1.0,
            None,
            true,
            segments,
            false,
        );
        assert!(
            report.surface_penetration < 0.0005,
            "{segments} segments: {} m penetration",
            report.surface_penetration
        );
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn rubber_track_propels_a_free_carrier_against_ground() {
    track_case(0.005, 1.0 / 19200.0, 4, false, 1.0, Some(0.8));
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn frictionless_ground_cannot_supply_horizontal_track_impulse() {
    track_case(0.005, 1.0 / 19200.0, 4, false, 1.0, Some(0.0));
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn sampled_surface_track_circulates_on_frictionless_ground() {
    track_run_resolution(
        0.005,
        1.0 / 19200.0,
        4,
        false,
        1.0,
        Some(0.0),
        true,
        224,
        true,
    );
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn motor_driven_ground_travel_exceeds_passive_relaxation() {
    let passive = track_run(0.005, 1.0 / 19200.0, 4, false, 1.0, Some(0.8), false);
    let driven = track_run(0.005, 1.0 / 19200.0, 4, false, 1.0, Some(0.8), true);
    let additional = passive.travel.y - driven.travel.y;
    eprintln!(
        "motor causality: passive={:?}, driven={:?}, additional forward travel={additional}",
        passive.travel, driven.travel
    );
    assert!(
        additional > driven.pitch,
        "motor did not add one pitch of chassis travel beyond passive relaxation"
    );
}

#[derive(Default)]
struct TrackReport {
    travel: Vec3,
    pitch: Real,
    surface_penetration: Real,
}

fn track_loop(
    prestrain: Real,
    max_substep: Real,
    width_cells: usize,
    geometry_only: bool,
    direction: Real,
) {
    track_case(
        prestrain,
        max_substep,
        width_cells,
        geometry_only,
        direction,
        None,
    );
}

fn track_case(
    prestrain: Real,
    max_substep: Real,
    width_cells: usize,
    geometry_only: bool,
    direction: Real,
    ground_friction: Option<Real>,
) {
    track_run(
        prestrain,
        max_substep,
        width_cells,
        geometry_only,
        direction,
        ground_friction,
        true,
    );
}

fn track_run(
    prestrain: Real,
    max_substep: Real,
    width_cells: usize,
    geometry_only: bool,
    direction: Real,
    ground_friction: Option<Real>,
    powered: bool,
) -> TrackReport {
    track_run_resolution(
        prestrain,
        max_substep,
        width_cells,
        geometry_only,
        direction,
        ground_friction,
        powered,
        224,
        false,
    )
}

fn track_run_resolution(
    prestrain: Real,
    max_substep: Real,
    width_cells: usize,
    geometry_only: bool,
    direction: Real,
    ground_friction: Option<Real>,
    powered: bool,
    segments: usize,
    sample_surface: bool,
) -> TrackReport {
    eprintln!(
        "track case powered={powered}, ground friction={ground_friction:?}, sampled surface={sample_surface}"
    );
    let floating = ground_friction.is_some();
    let gravity = if floating {
        Vec3::new(0.0, 0.0, -9.81)
    } else {
        Vec3::ZERO
    };
    let mut soft_builder = ModelBuilder::new();
    soft_builder.set_world_gravity(WorldId(0), gravity);
    soft_builder.set_linear_damping(0.0);
    let mut builder = ModelBuilder::new();
    builder.begin_articulation();
    builder.set_world_gravity(WorldId(0), gravity);
    let carrier = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 500.0,
        com: Vec3::ZERO,
        inertia: Mat3::from_diagonal(Vec3::splat(100.0)),
    }));
    if floating {
        builder
            .add_joint_free(
                BodyId::NONE,
                carrier,
                Transform::IDENTITY,
                Transform::IDENTITY,
            )
            .unwrap();
    } else {
        builder
            .add_joint_fixed(
                BodyId::NONE,
                carrier,
                Transform::IDENTITY,
                Transform::IDENTITY,
            )
            .unwrap();
    }
    let reinforced_track::TrackRig {
        mesh,
        path,
        params,
        pulleys,
        lugs,
        nodes,
        drive_joint,
        mut shapes,
        installed,
        center,
    } = reinforced_track::build_track(
        &mut soft_builder,
        &mut builder,
        carrier,
        Vec3::ZERO,
        segments,
        width_cells,
        prestrain,
    );
    assert_eq!(center, Vec3::ZERO);
    let soft_model = soft_builder.build().unwrap();
    let soft_masses = soft_model.particle_mass.host().unwrap().to_vec();
    let mut soft_state = soft_model.state().unwrap();
    for (&node, &position) in nodes.iter().zip(&installed) {
        soft_state.particle_q.host_mut().unwrap()[node as usize] = position;
    }
    if let Some(friction) = ground_friction {
        shapes.push(SoftRigidShapeGpu {
            position: [0.0, 0.0, -0.353, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            data: [5.0, 5.0, 0.1, friction as f32],
            ids: [1, u32::MAX, 0, 0],
        });
    }
    let mut rigid_model = builder.build().unwrap();
    let drive_dof = rigid_model.joint_dof_offset.host().unwrap()[drive_joint.index()] as usize;
    let dof_count = rigid_model.joint_dof_total;
    let coord_count = rigid_model.joint_coord_total;
    let rigid_masses = rigid_model.body_mass.host().unwrap().to_vec();
    rigid_model.joint_effort_limit.host_mut().unwrap()[drive_dof] = 40.0;
    let mut rigid_state = rigid_model.state().unwrap();
    eval_fk(&rigid_model, &mut rigid_state).unwrap();
    let mut deepest = (0.0_f64, 0, 0);
    let triangles = molla_sim::compute_tet_surface_triangles(&mesh.tets);
    let mut edges = std::collections::BTreeSet::new();
    let mut samples = Vec::new();
    for [a, b, c] in triangles {
        let [a, b, c] = [a as usize, b as usize, c as usize];
        samples.push((installed[a] + installed[b] + installed[c]) / 3.0);
        for (i, j) in [(a, b), (b, c), (c, a)] {
            edges.insert((i.min(j), i.max(j)));
        }
    }
    for (a, b) in edges {
        samples.push((installed[a] + installed[b]) * 0.5);
    }
    let mut surface_penetration = 0.0_f64;
    for (s, shape) in shapes.iter().enumerate() {
        let pose = if shape.ids[1] == u32::MAX {
            Transform::IDENTITY
        } else {
            rigid_state.body_q.host().unwrap()[shape.ids[1] as usize]
        };
        for (node, q) in installed.iter().chain(samples.iter()).enumerate() {
            let d = reinforced_track::primitive_gap(shape, pose, *q);
            surface_penetration = surface_penetration.max(-d);
            if node < installed.len() && -d > deepest.0 {
                deepest = (-d, s, node);
            }
        }
    }
    eprintln!("initial penetration (depth, shape, node)={deepest:?}");
    eprintln!("sampled surface penetration: segments={segments}, depth={surface_penetration}");
    eprintln!(
        "deepest initial point={:?}, shape={:?}",
        installed[deepest.2], shapes[deepest.1]
    );
    assert!(
        deepest.0 < 0.0005,
        "installed geometry already interpenetrates: {deepest:?}"
    );
    if geometry_only {
        return TrackReport {
            surface_penetration,
            ..Default::default()
        };
    }
    assert!(
        surface_penetration < 0.0005,
        "installed surface samples already interpenetrate: {surface_penetration}"
    );
    let initial_poses = rigid_state.body_q.host().unwrap().to_vec();
    let evidence_shapes = shapes.clone();
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let mut control = rigid_model.control();
    control.joint_target_mode.host_mut().unwrap()[drive_dof] = JointTargetMode::Velocity as u32;
    control.joint_target_kd.host_mut().unwrap()[drive_dof] = 10.0;
    let mut config = FemRigidConfig {
        max_substep,
        ..Default::default()
    };
    if sample_surface {
        config.contact.iterations = 64;
    }
    let constructor = if sample_surface {
        FemRigidGpuSystem::new_surface_sampled
    } else {
        FemRigidGpuSystem::new
    };
    let mut system = constructor(
        engine.clone(),
        FemRigidGpuScene {
            soft_model,
            soft_state,
            rigid_model,
            rigid_state,
            control,
            shapes,
        },
        config,
    )
    .unwrap();
    if std::env::var_os("MOLLA_FEM_MOMENTUM_DIAGNOSTICS").is_some() {
        system.enable_momentum_diagnostics();
    }
    system.step(if floating { 0.10 } else { 0.02 }).unwrap();
    let settled_poses = body_poses(&engine, &system, rigid_masses.len());
    let settled_pose = settled_poses[0];
    let settled: Vec<Vec3> = read::<[f32; 4]>(
        &engine,
        system.soft_state().particle_q.device_buffer().unwrap(),
        mesh.positions.len(),
    )
    .iter()
    .map(|p| Vec3::new(p[0] as Real, p[1] as Real, p[2] as Real))
    .collect();
    eprintln!(
        "settled carrier={settled_pose:?}, material minJ={}",
        mesh.minimum_volume_ratio(&settled).unwrap()
    );
    assert_eq!(
        read::<u32>(&engine, system.status_buffer(), 1),
        [0],
        "settling failed"
    );
    if powered {
        system
            .set_drive_velocity(drive_dof, direction * 4.0)
            .unwrap();
    } else {
        system.set_drive_effort(drive_dof, 0.0).unwrap();
    }
    let drive_intervals = if floating { 5 } else { 2 };
    for interval in 0..drive_intervals {
        system.step(0.12).unwrap();
        assert_eq!(
            read::<u32>(&engine, system.status_buffer(), 1),
            [0],
            "drive interval {interval} failed"
        );
        eprintln!(
            "completed drive interval {}/{drive_intervals}",
            interval + 1
        );
        if let Some(buffer) = system.momentum_diagnostics_buffer() {
            let values = read::<[f32; 4]>(&engine, buffer, 20);
            for (phase, name) in ["input", "FEM", "rigid", "contact", "drift"]
                .iter()
                .enumerate()
            {
                eprintln!(
                    "momentum {name}: final soft={:?}, rigid={:?}, accumulated soft delta={:?}, rigid delta={:?}",
                    values[phase * 2],
                    values[phase * 2 + 1],
                    values[10 + phase * 2],
                    values[11 + phase * 2]
                );
            }
        }
    }
    assert!(system.soft_state().particle_q.host().is_err());
    let volume = read::<u32>(&engine, system.volume_diagnostics_buffer(), 2);
    let lifetime_min_j = f32::from_bits(volume[0]);
    eprintln!(
        "every-substep minimum J={lifetime_min_j}, invalid tet={}",
        volume[1]
    );
    assert_eq!(read::<u32>(&engine, system.status_buffer(), 1), [0]);
    let q: Vec<Vec3> = read::<[f32; 4]>(
        &engine,
        system.soft_state().particle_q.device_buffer().unwrap(),
        mesh.positions.len(),
    )
    .iter()
    .map(|p| Vec3::new(p[0] as Real, p[1] as Real, p[2] as Real))
    .collect();
    let final_poses = body_poses(&engine, &system, rigid_masses.len());
    let final_pose = final_poses[0];
    let rigid_velocities = read::<[f32; 8]>(
        &engine,
        system.rigid_state().body_qd.device_buffer().unwrap(),
        rigid_masses.len(),
    );
    let carrier_linear = Vec3::new(
        rigid_velocities[0][0] as Real,
        rigid_velocities[0][1] as Real,
        rigid_velocities[0][2] as Real,
    );
    let carrier_angular = Vec3::new(
        rigid_velocities[0][4] as Real,
        rigid_velocities[0][5] as Real,
        rigid_velocities[0][6] as Real,
    );
    let omega = read::<f32>(
        &engine,
        system.rigid_state().joint_qd.device_buffer().unwrap(),
        dof_count,
    );
    let joint_angles = read::<f32>(
        &engine,
        system.rigid_state().joint_q.device_buffer().unwrap(),
        coord_count,
    );
    let velocities = read::<[f32; 4]>(
        &engine,
        system.soft_state().particle_qd.device_buffer().unwrap(),
        mesh.positions.len(),
    );
    let ratio = mesh.minimum_volume_ratio(&q).unwrap();
    let worst = mesh
        .tets
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let volume = |p: &[Vec3]| {
                Mat3::from_cols(
                    p[t[1] as usize] - p[t[0] as usize],
                    p[t[2] as usize] - p[t[0] as usize],
                    p[t[3] as usize] - p[t[0] as usize],
                )
                .determinant()
            };
            (i, volume(&q) / volume(&mesh.positions))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    eprintln!(
        "prestrain={prestrain}, h={max_substep}, settled minJ={}, worst tet={worst:?}, nodes={:?}",
        mesh.minimum_volume_ratio(&settled).unwrap(),
        mesh.tets[worst.0]
    );
    let travel: Real = q
        .iter()
        .take(params.segments * (params.width_cells + 1) * (params.thickness_cells + 1))
        .zip(&installed)
        .map(|(a, b)| (*a - *b).length())
        .sum::<Real>()
        / (params.segments * (params.width_cells + 1) * (params.thickness_cells + 1)) as Real;
    eprintln!(
        "installed loop: nodes={}, tets={}, driveR={}, perimeter={}, minJ={ratio}, mean movement={travel}, wheel rates={omega:?}",
        mesh.positions.len(),
        mesh.tets.len(),
        pulleys[2].radius,
        path.length()
    );
    let mut circulation = Vec::new();
    let mut separation: Real = 0.0;
    let mut material_speed = 0.0;
    for s in 0..params.segments {
        let node = (s * (params.width_cells + 1) + params.width_cells / 2)
            * (params.thickness_cells + 1)
            + params.thickness_cells / 2;
        let local_before =
            settled_pose.rotation.conjugate() * (settled[node] - settled_pose.position);
        let local_after = final_pose.rotation.conjugate() * (q[node] - final_pose.position);
        let (before, _) = path.project(local_before).unwrap();
        let (after, d) = path.project(local_after).unwrap();
        let world_velocity = Vec3::new(
            velocities[node][0] as Real,
            velocities[node][1] as Real,
            velocities[node][2] as Real,
        );
        let local_velocity = final_pose.rotation.conjugate()
            * (world_velocity
                - carrier_linear
                - carrier_angular.cross(q[node] - final_pose.position));
        material_speed += path.sample(after).unwrap().tangent.dot(local_velocity);
        let delta =
            (after - before + path.length() * 0.5).rem_euclid(path.length()) - path.length() * 0.5;
        circulation.push(direction * delta);
        separation = separation.max(d);
    }
    let minimum = circulation.iter().copied().fold(Real::INFINITY, Real::min);
    let maximum = circulation
        .iter()
        .copied()
        .fold(Real::NEG_INFINITY, Real::max);
    let mean = circulation.iter().sum::<Real>() / circulation.len() as Real;
    let pitch = path.length() / lugs as Real;
    material_speed /= params.segments as Real;
    eprintln!(
        "material circulation: pitch={pitch}, range={minimum}..{maximum}, mean={mean}, maximum separation={separation}, mean speed={material_speed}, angles={joint_angles:?}"
    );
    if let Some(dir) = std::env::var_os("MOLLA_BELT_EVIDENCE_DIR") {
        let mut dir = std::path::PathBuf::from(dir);
        if !powered {
            dir.push("unpowered");
        }
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("drive-mode.txt"),
            if powered { "powered\n" } else { "unpowered\n" },
        )
        .unwrap();
        let shape_csv = evidence_shapes
            .iter()
            .map(|shape| {
                format!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    shape.ids[0],
                    shape.ids[1],
                    shape.position[0],
                    shape.position[1],
                    shape.position[2],
                    shape.rotation[0],
                    shape.rotation[1],
                    shape.rotation[2],
                    shape.rotation[3],
                    shape.data[0],
                    shape.data[1],
                    shape.data[2],
                    shape.data[3]
                )
            })
            .collect::<String>();
        std::fs::write(dir.join("shapes.csv"), shape_csv).unwrap();
        for (name, poses) in [
            ("installed", &initial_poses),
            ("settled", &settled_poses),
            ("driven", &final_poses),
        ] {
            let pose_csv = poses
                .iter()
                .map(|pose| {
                    format!(
                        "{},{},{},{},{},{},{}\n",
                        pose.position.x,
                        pose.position.y,
                        pose.position.z,
                        pose.rotation.x,
                        pose.rotation.y,
                        pose.rotation.z,
                        pose.rotation.w
                    )
                })
                .collect::<String>();
            std::fs::write(dir.join(format!("{name}-rigid.csv")), pose_csv).unwrap();
        }
        let faces = molla_sim::compute_tet_surface_triangles(&mesh.tets);
        for (name, positions) in [
            ("installed", &installed),
            ("settled", &settled),
            ("driven", &q),
        ] {
            let mut obj = positions
                .iter()
                .map(|v| format!("v {} {} {}\n", v.x, v.y, v.z))
                .collect::<String>();
            for face in &faces {
                obj.push_str(&format!(
                    "f {} {} {}\n",
                    face[0] + 1,
                    face[1] + 1,
                    face[2] + 1
                ));
            }
            std::fs::write(dir.join(format!("{name}.obj")), obj).unwrap();
        }
    }
    if let Some(friction) = ground_friction {
        let momentum: Vec3 = velocities
            .iter()
            .zip(&soft_masses)
            .map(|(v, m)| Vec3::new(v[0] as Real, v[1] as Real, v[2] as Real) * *m)
            .chain(
                rigid_velocities
                    .iter()
                    .zip(&rigid_masses)
                    .map(|(v, m)| Vec3::new(v[0] as Real, v[1] as Real, v[2] as Real) * *m),
            )
            .sum();
        let chassis_travel = final_pose.position - settled_pose.position;
        let penetration = q
            .iter()
            .map(|p| (-0.253 - p.z).max(0.0))
            .fold(0.0_f64, Real::max);
        eprintln!(
            "free carrier: friction={friction}, settled={settled_pose:?}, final={final_pose:?}, travel={chassis_travel:?}, total P={momentum:?}, ground penetration={penetration}"
        );
        assert!(penetration < 0.001, "ground penetration {penetration}");
        assert!(
            final_pose.position.z.abs() < 0.03,
            "carrier lost ground support"
        );
        assert!(
            final_pose.rotation.angle_between(Quat::IDENTITY) < 0.1,
            "carrier tipped"
        );
        if friction > 0.0 && powered {
            assert!(
                -direction * chassis_travel.y > 0.001,
                "rubber did not propel the carrier"
            );
            assert!(
                -direction * momentum.y > 0.1,
                "ground supplied no traction impulse"
            );
        } else if friction == 0.0 {
            assert!(
                momentum.x.hypot(momentum.y) < 0.02,
                "frictionless ground produced horizontal momentum {momentum:?}"
            );
        }
    }
    assert!(
        lifetime_min_j > 0.5,
        "transient material collapse: {lifetime_min_j}"
    );
    assert_eq!(volume[1], u32::MAX);
    let settled_ratio = mesh.minimum_volume_ratio(&settled).unwrap();
    assert!(
        settled_ratio > 0.5,
        "settled loop inverted/collapsed: {settled_ratio}"
    );
    assert!(ratio > 0.5, "installed loop inverted/collapsed: {ratio}");
    if powered {
        assert!(mean > pitch, "belt has not circulated one material pitch");
        assert!(minimum > pitch * 0.5, "part of the belt is not advancing");
    }
    assert!(
        maximum - minimum < pitch * 0.5,
        "nonuniform transport or excess stretch"
    );
    assert!(separation < 0.02, "belt escaped its pulley envelope");
    if powered {
        assert!(
            direction * omega[drive_dof] as Real > 0.1,
            "sprocket stalled before engagement"
        );
        assert!(
            omega
                .iter()
                .enumerate()
                .any(|(i, w)| i >= if floating { 6 } else { 0 }
                    && i != drive_dof
                    && w.abs() > 0.05),
            "no passive roller response"
        );
    }
    TrackReport {
        travel: final_pose.position - settled_pose.position,
        pitch,
        surface_penetration,
    }
}
