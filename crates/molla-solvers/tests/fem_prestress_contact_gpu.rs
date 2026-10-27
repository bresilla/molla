#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::{BodyId, WorldId};
use molla_kernels::soft_rigid_contact::SoftRigidShapeGpu;
use molla_math::{Mat3, Quat, SpatialInertia, Transform, Vec3};
use molla_sim::soft_belt::{RubberBeltMesh, RubberBeltParams, RubberCordParams};
use molla_sim::{BodyParams, ModelBuilder};
use molla_solvers::fem_rigid::FemRigidConfig;
use molla_solvers::fem_rigid_gpu::{FemRigidGpuScene, FemRigidGpuSystem};
use std::sync::Arc;

fn read<T: bytemuck::Pod>(engine: &Engine, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
    let device = engine.device().unwrap();
    let size = (std::mem::size_of::<T>() * count) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prestress acceptance"),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec()
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn prestressed_rubber_stays_outside_its_support() {
    let penetrations = prestress_support(false);
    assert!(
        penetrations.iter().all(|p| p.0 < 0.001),
        "prestressed rubber penetrates support: {penetrations:?}"
    );
    assert!(penetrations.iter().all(|p| p.1 > 0.0005));
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn prestressed_surface_samples_stay_outside_support() {
    let penetrations = prestress_support(true);
    assert!(
        penetrations.iter().all(|p| p.0 < 0.00005 && p.1 < 0.00005),
        "prestressed surface samples penetrate support: {penetrations:?}"
    );
}

fn prestress_support(sampled: bool) -> Vec<(f32, f32)> {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let mut penetrations = Vec::new();
    for (max_substep, recovery_speed) in [
        (1.0 / 2400.0, 0.5),
        (1.0 / 4800.0, 0.5),
        (1.0 / 9600.0, 0.5),
        (1.0 / 4800.0, 100.0),
    ] {
        let mesh = RubberBeltMesh::circular(RubberBeltParams {
            radius: 0.132,
            segments: 32,
            width_cells: 2,
            thickness_cells: 2,
            ..Default::default()
        })
        .unwrap();
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        builder.set_linear_damping(0.0);
        mesh.append_reinforced_to(
            &mut builder,
            WorldId(0),
            Vec3::ZERO,
            RubberCordParams {
                prestrain: 0.02,
                ..Default::default()
            },
        )
        .unwrap();
        let soft_model = builder.build().unwrap();
        let soft_state = soft_model.state().unwrap();
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        builder.begin_articulation();
        let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 10.0,
            com: Vec3::ZERO,
            inertia: Mat3::from_diagonal(Vec3::splat(0.1)),
        }));
        builder
            .add_joint_revolute(
                BodyId::NONE,
                body,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::X,
            )
            .unwrap();
        let rigid_model = builder.build().unwrap();
        let rigid_state = rigid_model.state().unwrap();
        let control = rigid_model.control();
        let shapes = vec![SoftRigidShapeGpu {
            position: [0.0; 4],
            rotation: Quat::from_rotation_z(std::f64::consts::FRAC_PI_2)
                .to_array()
                .map(|v| v as f32),
            data: [0.122, 0.2, 0.0, 0.0],
            ids: [2, body.index() as u32, 0, 0],
        }];
        let mut config = FemRigidConfig {
            max_substep,
            ..Default::default()
        };
        config.contact.max_recovery_speed = recovery_speed;
        if sampled {
            config.contact.iterations = 64;
        }
        let constructor = if sampled {
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
        system.step(0.02).unwrap();
        assert!(system.soft_state().particle_q.host().is_err());
        assert!(system.soft_state().particle_qd.host().is_err());
        let volume = read::<u32>(&engine, system.volume_diagnostics_buffer(), 2);
        assert!(
            f32::from_bits(volume[0]) > 0.5,
            "substep volume diagnostics {volume:?}"
        );
        assert_eq!(volume[1], u32::MAX);
        let poses = read::<[f32; 8]>(
            &engine,
            system.rigid_state().body_q.device_buffer().unwrap(),
            1,
        );
        let q = read::<[f32; 4]>(
            &engine,
            system.soft_state().particle_q.device_buffer().unwrap(),
            mesh.positions.len(),
        );
        let penetration = q
            .iter()
            .map(|p| 0.122 - (p[1] * p[1] + p[2] * p[2]).sqrt())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "prestress support: h={max_substep}, recovery cap={recovery_speed}, maximum penetration={penetration}, poses={poses:?}"
        );
        let positions: Vec<_> = q
            .iter()
            .map(|p| Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
            .collect();
        let mut surface_penetration = 0.0_f32;
        for [a, b, c] in molla_sim::compute_tet_surface_triangles(&mesh.tets) {
            let [a, b, c] = [
                positions[a as usize],
                positions[b as usize],
                positions[c as usize],
            ];
            for p in [
                (a + b) * 0.5,
                (b + c) * 0.5,
                (c + a) * 0.5,
                (a + b + c) / 3.0,
            ] {
                surface_penetration = surface_penetration.max((0.122 - p.y.hypot(p.z)) as f32);
            }
        }
        eprintln!(
            "sampled={sampled}, vertex penetration={penetration}, surface penetration={surface_penetration}"
        );
        assert!(mesh.minimum_volume_ratio(&positions).unwrap() > 0.5);
        assert_eq!(read::<u32>(&engine, system.status_buffer(), 1), [0]);
        penetrations.push((penetration, surface_penetration));
    }
    penetrations
}
