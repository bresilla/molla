#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::{BodyId, WorldId};
use molla_kernels::soft_rigid_contact::SoftRigidShapeGpu;
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, JointTargetMode, ModelBuilder, eval_fk};
use molla_solvers::fem_rigid::FemRigidConfig;
use molla_solvers::fem_rigid_gpu::{FemRigidGpuScene, FemRigidGpuSystem};
use std::sync::Arc;

#[path = "common/gpu_readback.rs"]
mod gpu_readback;
#[path = "common/reinforced_track.rs"]
mod reinforced_track;
use gpu_readback::{body_poses, read};
use reinforced_track::{TrackRig, build_track, primitive_gap};

fn vehicle_scene(
    ground_friction: Option<Real>,
    pivot: bool,
) -> (FemRigidGpuScene, [TrackRig; 2], [usize; 2]) {
    let gravity = if ground_friction.is_some() {
        -Vec3::Z * 9.81
    } else {
        Vec3::ZERO
    };
    let mut soft = ModelBuilder::new();
    soft.set_world_gravity(WorldId(0), gravity);
    soft.set_linear_damping(0.0);
    let mut rigid = ModelBuilder::new();
    rigid.begin_articulation();
    rigid.set_world_gravity(WorldId(0), gravity);
    let chassis = rigid.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1000.0,
        com: Vec3::ZERO,
        inertia: Mat3::from_diagonal(Vec3::splat(200.0)),
    }));
    rigid
        .add_joint_free(
            BodyId::NONE,
            chassis,
            Transform::IDENTITY,
            Transform::IDENTITY,
        )
        .unwrap();
    let tracks = [-0.41, 0.41]
        .map(|x| build_track(&mut soft, &mut rigid, chassis, Vec3::X * x, 224, 4, 0.005));
    let soft_model = soft.build().unwrap();
    let mut soft_state = soft_model.state().unwrap();
    for track in &tracks {
        for (&node, &position) in track.nodes.iter().zip(&track.installed) {
            soft_state.particle_q.host_mut().unwrap()[node as usize] = position;
        }
    }
    let mut rigid_model = rigid.build().unwrap();
    let drives = tracks.each_ref().map(|track| {
        rigid_model.joint_dof_offset.host().unwrap()[track.drive_joint.index()] as usize
    });
    let mut rigid_state = rigid_model.state().unwrap();
    eval_fk(&rigid_model, &mut rigid_state).unwrap();
    let mut control = rigid_model.control();
    for drive in drives {
        rigid_model.joint_effort_limit.host_mut().unwrap()[drive] =
            if pivot { 300.0 } else { 40.0 };
        control.joint_target_mode.host_mut().unwrap()[drive] = JointTargetMode::Velocity as u32;
        control.joint_target_kd.host_mut().unwrap()[drive] = if pivot { 100.0 } else { 10.0 };
    }
    let mut shapes: Vec<_> = tracks
        .iter()
        .flat_map(|track| track.shapes.iter().copied())
        .collect();
    if let Some(friction) = ground_friction {
        shapes.push(SoftRigidShapeGpu {
            position: [0.0, 0.0, -0.353, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            data: [5.0, 5.0, 0.1, friction as f32],
            ids: [1, u32::MAX, 0, 0],
        });
    }
    (
        FemRigidGpuScene {
            soft_model,
            soft_state,
            rigid_model,
            rigid_state,
            control,
            shapes,
        },
        tracks,
        drives,
    )
}

#[test]
fn two_track_topology_and_installed_surfaces_are_consistent() {
    let (scene, tracks, drives) = vehicle_scene(Some(0.8), false);
    assert_eq!(scene.rigid_model.articulation_count, 1);
    assert_eq!(scene.rigid_model.body_count, 13);
    assert_eq!(scene.rigid_model.joint_dof_total, 18);
    assert_eq!(scene.soft_model.particle_count, 8736);
    assert_eq!(scene.soft_model.tet_count, 26880);
    assert_eq!(drives, [8, 14]);
    assert_eq!(scene.shapes.len(), 57);
    let controls = scene.control.joint_target_mode.host().unwrap();
    assert_eq!(
        controls
            .iter()
            .filter(|&&m| m == JointTargetMode::Velocity as u32)
            .count(),
        2
    );
    assert_eq!(
        scene
            .soft_model
            .particle_mass
            .host()
            .unwrap()
            .iter()
            .take(4368)
            .sum::<Real>(),
        scene
            .soft_model
            .particle_mass
            .host()
            .unwrap()
            .iter()
            .skip(4368)
            .sum::<Real>()
    );
    for (side, track) in tracks.iter().enumerate() {
        assert_eq!(track.nodes.len(), 4368);
        assert_eq!(track.nodes[0], (side * 4368) as u32);
        assert_eq!(track.nodes[4367], ((side + 1) * 4368 - 1) as u32);
        assert!(track.mesh.minimum_volume_ratio(&track.installed).unwrap() > 0.5);
        let mut points = track.installed.clone();
        for [a, b, c] in molla_sim::compute_tet_surface_triangles(&track.mesh.tets) {
            let [a, b, c] = [
                track.installed[a as usize],
                track.installed[b as usize],
                track.installed[c as usize],
            ];
            points.extend([
                (a + b) * 0.5,
                (b + c) * 0.5,
                (c + a) * 0.5,
                (a + b + c) / 3.0,
            ]);
        }
        let mut penetration = 0.0_f64;
        for shape in &scene.shapes {
            let pose = if shape.ids[1] == u32::MAX {
                Transform::IDENTITY
            } else {
                scene.rigid_state.body_q.host().unwrap()[shape.ids[1] as usize]
            };
            for &point in &points {
                penetration = penetration.max(-primitive_gap(shape, pose, point));
            }
        }
        eprintln!("two-track initial clearance: side={side}, penetration={penetration}");
        assert!(penetration < 0.0005);
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn two_tracks_drive_forward() {
    vehicle_run("forward", [1.0, 1.0], Some(0.8), false);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn two_track_gpu_scene_finishes_one_submission_window() {
    let (scene, _, _) = vehicle_scene(Some(0.8), false);
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let h = 1.0 / 19200.0;
    let mut system = FemRigidGpuSystem::new(
        engine.clone(),
        scene,
        FemRigidConfig {
            max_substep: h,
            ..Default::default()
        },
    )
    .unwrap();
    eprintln!(
        "two-track submission window starts, pid={}",
        std::process::id()
    );
    system.step(32.0 * h).unwrap();
    assert_eq!(system.queued_substeps(), 0);
    assert_eq!(read::<u32>(&engine, system.status_buffer(), 1), [0]);
    let volume = read::<u32>(&engine, system.volume_diagnostics_buffer(), 2);
    assert!(f32::from_bits(volume[0]) > 0.5);
    assert_eq!(volume[1], u32::MAX);
    assert!(system.soft_state().particle_q.host().is_err());
    eprintln!(
        "two-track submission window finished, lifetimeJ={}",
        f32::from_bits(volume[0])
    );
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn two_tracks_drive_reverse() {
    vehicle_run("reverse", [-1.0, -1.0], Some(0.8), false);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn two_tracks_pivot_positive() {
    vehicle_run("pivot-positive", [1.0, -1.0], Some(0.8), false);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn two_tracks_pivot_negative() {
    vehicle_run("pivot-negative", [-1.0, 1.0], Some(0.8), false);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn two_tracks_preserve_airborne_momentum() {
    vehicle_run("airborne", [1.0, -1.0], None, false);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn two_tracks_circulate_without_ground_traction() {
    vehicle_run("frictionless", [1.0, 1.0], Some(0.0), false);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn two_tracks_sampled_surface_drive_forward() {
    vehicle_run("sampled-forward", [1.0, 1.0], Some(0.8), true);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn two_track_motors_exceed_passive_relaxation() {
    let passive = vehicle_run("passive", [0.0, 0.0], Some(0.8), false);
    let driven = vehicle_run("powered", [1.0, 1.0], Some(0.8), false);
    assert!(
        passive.y - driven.y > 0.05645178659660792,
        "passive={passive:?}, driven={driven:?}"
    );
}

fn positions(engine: &Engine, system: &FemRigidGpuSystem, count: usize) -> Vec<Vec3> {
    read::<[f32; 4]>(
        engine,
        system.soft_state().particle_q.device_buffer().unwrap(),
        count,
    )
    .iter()
    .map(|p| Vec3::new(p[0] as Real, p[1] as Real, p[2] as Real))
    .collect()
}

fn save_state(
    dir: &std::path::Path,
    name: &str,
    q: &[Vec3],
    faces: &[[u32; 3]],
    poses: &[Transform],
) {
    let mut obj = q
        .iter()
        .map(|p| format!("v {} {} {}\n", p.x, p.y, p.z))
        .collect::<String>();
    for face in faces {
        obj.push_str(&format!(
            "f {} {} {}\n",
            face[0] + 1,
            face[1] + 1,
            face[2] + 1
        ));
    }
    std::fs::write(dir.join(format!("{name}.obj")), obj).unwrap();
    let csv = poses
        .iter()
        .map(|p| {
            format!(
                "{},{},{},{},{},{},{}\n",
                p.position.x,
                p.position.y,
                p.position.z,
                p.rotation.x,
                p.rotation.y,
                p.rotation.z,
                p.rotation.w
            )
        })
        .collect::<String>();
    std::fs::write(dir.join(format!("{name}-rigid.csv")), csv).unwrap();
}

fn vehicle_run(name: &str, directions: [Real; 2], friction: Option<Real>, sampled: bool) -> Vec3 {
    let pivot = directions[0] * directions[1] < 0.0;
    let (scene, tracks, drives) = vehicle_scene(friction, pivot);
    let masses = scene.soft_model.particle_mass.host().unwrap().to_vec();
    let rigid_masses = scene.rigid_model.body_mass.host().unwrap().to_vec();
    let inertias = scene.rigid_model.body_inertia.host().unwrap().to_vec();
    let initial = scene.soft_state.particle_q.host().unwrap().to_vec();
    let initial_poses = scene.rigid_state.body_q.host().unwrap().to_vec();
    let tets: Vec<_> = scene
        .soft_model
        .tet_indices
        .host()
        .unwrap()
        .chunks_exact(4)
        .map(|t| [t[0], t[1], t[2], t[3]])
        .collect();
    let faces = molla_sim::compute_tet_surface_triangles(&tets);
    let evidence = std::env::var_os("MOLLA_BELT_EVIDENCE_DIR")
        .map(|dir| std::path::PathBuf::from(dir).join(name));
    if let Some(dir) = &evidence {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("marker-nodes.txt"),
            format!("{}\n{}\n", tracks[0].nodes[7], tracks[1].nodes[7]),
        )
        .unwrap();
        let csv = scene
            .shapes
            .iter()
            .map(|s| {
                format!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    s.ids[0],
                    s.ids[1],
                    s.position[0],
                    s.position[1],
                    s.position[2],
                    s.rotation[0],
                    s.rotation[1],
                    s.rotation[2],
                    s.rotation[3],
                    s.data[0],
                    s.data[1],
                    s.data[2],
                    s.data[3]
                )
            })
            .collect::<String>();
        std::fs::write(dir.join("shapes.csv"), csv).unwrap();
        std::fs::write(
            dir.join("drive-mode.txt"),
            if directions == [0.0, 0.0] {
                "unpowered\n"
            } else {
                "powered\n"
            },
        )
        .unwrap();
        save_state(dir, "installed", &initial, &faces, &initial_poses);
    }
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let config = FemRigidConfig {
        max_substep: 1.0 / 19200.0,
        contact: molla_solvers::soft_rigid_contact::SoftRigidContactConfig {
            iterations: if sampled { 64 } else { 12 },
            ..Default::default()
        },
        ..Default::default()
    };
    eprintln!(
        "vehicle={name}, directions={directions:?}, friction={friction:?}, sampled={sampled}, GPU storage limit={}",
        engine
            .device()
            .unwrap()
            .limits()
            .max_storage_buffer_binding_size
    );
    let constructor = if sampled {
        FemRigidGpuSystem::new_surface_sampled
    } else {
        FemRigidGpuSystem::new
    };
    let mut system = constructor(engine.clone(), scene, config).unwrap();
    system
        .step(if friction.is_some() { 0.10 } else { 0.02 })
        .unwrap();
    assert_eq!(read::<u32>(&engine, system.status_buffer(), 1), [0]);
    let settled = positions(&engine, &system, masses.len());
    let settled_poses = body_poses(&engine, &system, rigid_masses.len());
    if let Some(dir) = &evidence {
        save_state(dir, "settled", &settled, &faces, &settled_poses);
    }
    let mut final_q = settled.clone();
    let mut final_poses = settled_poses.clone();
    for interval in 0..5 {
        for (drive, direction) in drives.into_iter().zip(directions) {
            if direction == 0.0 {
                system.set_drive_effort(drive, 0.0).unwrap();
            } else {
                system.set_drive_velocity(drive, direction * 4.0).unwrap();
            }
        }
        system.step(0.12).unwrap();
        assert_eq!(
            read::<u32>(&engine, system.status_buffer(), 1),
            [0],
            "interval {interval}"
        );
        let volume = read::<u32>(&engine, system.volume_diagnostics_buffer(), 2);
        final_q = positions(&engine, &system, masses.len());
        final_poses = body_poses(&engine, &system, rigid_masses.len());
        let omega = read::<f32>(
            &engine,
            system.rigid_state().joint_qd.device_buffer().unwrap(),
            18,
        );
        eprintln!(
            "vehicle={name}, interval={}, lifetimeJ={}, chassis={:?}, rotor rates={:?}",
            interval + 1,
            f32::from_bits(volume[0]),
            final_poses[0],
            drives.map(|d| omega[d])
        );
        if let Some(dir) = &evidence {
            save_state(
                dir,
                &format!("frame-{}", interval + 1),
                &final_q,
                &faces,
                &final_poses,
            );
        }
        assert!(f32::from_bits(volume[0]) > 0.5, "volume={volume:?}");
        assert_eq!(volume[1], u32::MAX);
    }
    if let Some(dir) = &evidence {
        save_state(dir, "driven", &final_q, &faces, &final_poses);
    }
    assert!(system.soft_state().particle_q.host().is_err());
    let travel = final_poses[0].position - settled_poses[0].position;
    let relative_rotation = settled_poses[0].rotation.conjugate() * final_poses[0].rotation;
    let forward = relative_rotation * -Vec3::Y;
    let yaw = forward.x.atan2(-forward.y);
    let pitch = tracks[0].path.length() / tracks[0].lugs as Real;
    for (side, track) in tracks.iter().enumerate() {
        let local_nodes: Vec<_> = track.nodes.iter().map(|&n| final_q[n as usize]).collect();
        assert!(track.mesh.minimum_volume_ratio(&local_nodes).unwrap() > 0.5);
        let mut circulation = Vec::new();
        let mut separation = 0.0_f64;
        for segment in 0..track.params.segments {
            let index = (segment * (track.params.width_cells + 1) + track.params.width_cells / 2)
                * (track.params.thickness_cells + 1)
                + track.params.thickness_cells / 2;
            let node = track.nodes[index] as usize;
            let before = settled_poses[0].rotation.conjugate()
                * (settled[node] - settled_poses[0].position)
                - track.center;
            let after = final_poses[0].rotation.conjugate()
                * (final_q[node] - final_poses[0].position)
                - track.center;
            let (s0, _) = track.path.project(before).unwrap();
            let (s1, distance) = track.path.project(after).unwrap();
            let delta = (s1 - s0 + track.path.length() * 0.5).rem_euclid(track.path.length())
                - track.path.length() * 0.5;
            circulation.push(delta * directions[side]);
            separation = separation.max(distance);
        }
        let mean = circulation.iter().sum::<Real>() / circulation.len() as Real;
        let min = circulation.iter().copied().fold(Real::INFINITY, Real::min);
        let max = circulation
            .iter()
            .copied()
            .fold(Real::NEG_INFINITY, Real::max);
        eprintln!(
            "vehicle={name}, side={side}, pitch={pitch}, circulation={min}..{max}, mean={mean}, envelope={separation}, drive radius={}",
            track.pulleys[2].radius
        );
        assert!(separation < 0.02);
        if directions[side] != 0.0 {
            assert!(
                mean > pitch && min > pitch * 0.5 && max - min < pitch * 0.5,
                "side={side}, circulation failed"
            );
        }
    }
    let soft_v = read::<[f32; 4]>(
        &engine,
        system.soft_state().particle_qd.device_buffer().unwrap(),
        masses.len(),
    );
    let rigid_v = read::<[f32; 8]>(
        &engine,
        system.rigid_state().body_qd.device_buffer().unwrap(),
        rigid_masses.len(),
    );
    let mut momentum = Vec3::ZERO;
    let mut angular_momentum = Vec3::ZERO;
    for ((p, v), mass) in final_q.iter().zip(&soft_v).zip(&masses) {
        let mv = Vec3::new(v[0] as Real, v[1] as Real, v[2] as Real) * *mass;
        momentum += mv;
        angular_momentum += p.cross(mv);
    }
    for (i, v) in rigid_v.iter().enumerate() {
        let mv = Vec3::new(v[0] as Real, v[1] as Real, v[2] as Real) * rigid_masses[i];
        let w = Vec3::new(v[4] as Real, v[5] as Real, v[6] as Real);
        let rotation = final_poses[i].rotation;
        momentum += mv;
        angular_momentum += final_poses[i].position.cross(mv)
            + rotation * (inertias[i] * (rotation.conjugate() * w));
    }
    eprintln!(
        "vehicle={name}, travel={travel:?}, yaw={yaw}, P={momentum:?}, L={angular_momentum:?}"
    );
    if let Some(friction) = friction {
        let penetration = final_q
            .iter()
            .map(|p| -0.253 - p.z)
            .fold(0.0_f64, Real::max);
        assert!(penetration < 0.001);
        assert!(travel.z.abs() < 0.03);
        let up = relative_rotation * Vec3::Z;
        assert!(up.z > 0.1_f64.cos(), "chassis tipped: {up:?}");
        if friction == 0.0 {
            assert!(momentum.x.hypot(momentum.y) < 0.02);
        } else if pivot {
            assert!(
                directions[0] * yaw > 0.03,
                "pivot did not turn in the commanded direction"
            );
            assert!(
                travel.x.hypot(travel.y) < 0.05,
                "pivot translated excessively"
            );
        } else if directions[0] != 0.0 {
            assert!(
                -directions[0] * travel.y > pitch,
                "insufficient chassis travel"
            );
            assert!(
                travel.x.abs() < 0.02 && yaw.abs() < 0.05,
                "straight drive drifted"
            );
        }
    } else {
        assert!(
            momentum.length() < 0.02 && angular_momentum.length() < 0.02,
            "airborne conservation failed"
        );
    }
    travel
}
