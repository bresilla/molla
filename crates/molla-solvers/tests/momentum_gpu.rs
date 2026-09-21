#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::momentum::{GeneralizedMomentumBindings, MomentumBindings, MomentumMonitor};
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_momentum_diagnostics_preserve_phase_deltas() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let particles = 130;
    let bodies = 67;
    let pm: Vec<f32> = (0..particles).map(|i| (i % 7 + 1) as f32 * 0.125).collect();
    let bm: Vec<f32> = (0..bodies).map(|i| (i % 5 + 10) as f32).collect();
    let buffer = |name, bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    };
    let particle_mass = buffer("particle mass", bytemuck::cast_slice(&pm));
    let body_mass = buffer("body mass", bytemuck::cast_slice(&bm));
    let particle_velocity = buffer(
        "particle velocity",
        bytemuck::cast_slice(&vec![[0.0_f32; 4]; particles]),
    );
    let body_velocity = buffer(
        "body velocity",
        bytemuck::cast_slice(&vec![[0.0_f32; 8]; bodies]),
    );
    let monitor = MomentumMonitor::new(device);
    let joint_velocity = buffer("generalized velocity", bytemuck::cast_slice(&[1.0_f32; 3]));
    let jacobians = buffer(
        "COM Jacobians",
        bytemuck::cast_slice(&vec![0.0_f32; bodies * 18]),
    );
    let mut expected = [[0.0_f32; 4]; 20];
    for step in 0..2 {
        for phase in 0..5 {
            let pv: Vec<[f32; 4]> = (0..particles)
                .map(|i| {
                    [
                        (i % 3) as f32 + phase as f32 * 0.25,
                        (step * 4 + phase) as f32 * -0.125,
                        (i % 5) as f32 * 0.5,
                        0.0,
                    ]
                })
                .collect();
            let bv: Vec<[f32; 8]> = (0..bodies)
                .map(|i| {
                    [
                        (i % 4) as f32 - phase as f32 * 0.125,
                        (step * 3 + phase) as f32 * 0.25,
                        -0.5,
                        0.0,
                        9.0,
                        8.0,
                        7.0,
                        0.0,
                    ]
                })
                .collect();
            queue.write_buffer(&particle_velocity, 0, bytemuck::cast_slice(&pv));
            let mut uploaded_velocity = bv.clone();
            if phase == 2 {
                let mut j = vec![0.0_f32; bodies * 18];
                for i in 0..bodies {
                    for k in 0..3 {
                        j[i * 18 + k * 3 + k] = bv[i][k];
                    }
                    uploaded_velocity[i][..3].fill(999.0);
                }
                queue.write_buffer(&jacobians, 0, bytemuck::cast_slice(&j));
            }
            queue.write_buffer(&body_velocity, 0, bytemuck::cast_slice(&uploaded_velocity));
            let mut totals = [[0.0; 4]; 2];
            for k in 0..3 {
                totals[0][k] = pv.iter().zip(&pm).map(|(v, m)| v[k] * m).sum();
                totals[1][k] = bv.iter().zip(&bm).map(|(v, m)| v[k] * m).sum();
            }
            for component in 0..2 {
                let slot = phase * 2 + component;
                if phase > 0 {
                    for k in 0..3 {
                        expected[10 + slot][k] += totals[component][k] - expected[slot - 2][k];
                    }
                }
                expected[slot] = totals[component];
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            monitor
                .encode(
                    device,
                    &mut encoder,
                    MomentumBindings {
                        particle_mass: &particle_mass,
                        particle_velocity: &particle_velocity,
                        body_mass: &body_mass,
                        body_velocity: &body_velocity,
                        particles: particles as u32,
                        bodies: bodies as u32,
                        generalized: (phase == 2).then_some(GeneralizedMomentumBindings {
                            joint_velocity: &joint_velocity,
                            jacobians: &jacobians,
                            dofs: 3,
                        }),
                    },
                    phase as u32,
                )
                .unwrap();
            queue.submit([encoder.finish()]);
        }
    }
    let size = 20 * 16;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("momentum acceptance"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(monitor.diagnostics(), 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range();
    assert_eq!(bytemuck::cast_slice::<u8, [f32; 4]>(&mapped), &expected);
}
