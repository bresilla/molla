#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{ConeGeometryKernel, ConeGpuGeometry, ConeGpuRequest};
use molla_math::{Quat, Vec3};
use molla_sim::{
    joint_control::SphericalConeLimits,
    spherical_cone::{basis, evaluate},
};
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

fn wait<T>(future: impl Future<Output = T>) -> T {
    struct Thread(std::thread::Thread);
    impl Wake for Thread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Thread(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

fn run(requests: &[ConeGpuRequest]) -> Vec<ConeGpuGeometry> {
    let engine = Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let kernel = ConeGeometryKernel::new(&engine).unwrap();
    let bytes = std::mem::size_of_val(requests) as u64;
    assert_eq!(
        std::mem::size_of::<ConeGpuRequest>(),
        std::mem::size_of::<ConeGpuGeometry>()
    );
    let buffer = |usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cone geometry test"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        })
    };
    let inputs = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    let outputs = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let staging = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    queue.write_buffer(&inputs, 0, bytemuck::cast_slice(requests));
    assert!(kernel.prepare(device, &inputs, &outputs, 0).is_err());
    assert!(
        kernel
            .prepare(device, &inputs, &outputs, requests.len() as u32 + 1)
            .is_err()
    );
    let dispatch = kernel
        .prepare(device, &inputs, &outputs, requests.len() as u32)
        .unwrap();
    let mut result = Vec::new();
    for _ in 0..3 {
        let mut encoder = device.create_command_encoder(&Default::default());
        dispatch.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&outputs, 0, &staging, 0, bytes);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = staging.slice(..).get_mapped_range();
        let next = bytemuck::cast_slice::<u8, ConeGpuGeometry>(&mapped).to_vec();
        if !result.is_empty() {
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&result),
                bytemuck::cast_slice::<_, u8>(&next)
            );
        }
        result = next;
        drop(mapped);
        staging.unmap();
    }
    let error = wait(scope.pop());
    assert!(error.is_none(), "{error:?}");
    engine.check_device().unwrap();
    result
}

fn orientation(axis: Vec3, angle: f64, azimuth: f64, twist: f64) -> Quat {
    let [p, b0, b1] = basis(axis).unwrap();
    let direction = b0 * azimuth.cos() + b1 * azimuth.sin();
    Quat::from_axis_angle(p.cross(direction), angle) * Quat::from_axis_angle(p, twist)
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn cone_geometry_matches_cpu_in_retained_batches() {
    let mut requests = Vec::new();
    let mut cases = Vec::new();
    for axis in [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(2.0, -3.0, 4.0),
        -Vec3::X,
    ] {
        for angle in [0.0, 1e-5, 0.3, 0.7, 1.7, 2.6, std::f64::consts::PI - 1e-3] {
            for azimuth in [0.0, 0.7, 2.3] {
                for twist in [0.0, -2.9, 1.7] {
                    for limits in [
                        SphericalConeLimits::default(),
                        SphericalConeLimits {
                            angle0: Some(0.4),
                            angle1: Some(0.4),
                        },
                        SphericalConeLimits {
                            angle0: Some(0.4),
                            angle1: Some(0.9),
                        },
                        SphericalConeLimits {
                            angle0: Some(0.0),
                            angle1: Some(0.9),
                        },
                        SphericalConeLimits {
                            angle0: None,
                            angle1: Some(0.0),
                        },
                        SphericalConeLimits {
                            angle0: Some(0.0),
                            angle1: Some(0.0),
                        },
                        SphericalConeLimits {
                            angle0: Some(0.9),
                            angle1: None,
                        },
                        SphericalConeLimits {
                            angle0: Some(3.0),
                            angle1: Some(3.0),
                        },
                        SphericalConeLimits {
                            angle0: Some(0.0),
                            angle1: Some(std::f64::consts::PI),
                        },
                        SphericalConeLimits {
                            angle0: Some(1e-6),
                            angle1: Some(2.8),
                        },
                    ] {
                        let q = orientation(axis, angle, azimuth, twist);
                        for rotation in [q, -q] {
                            requests.push(ConeGpuRequest::new(limits, axis, rotation).unwrap());
                            cases.push((limits, axis, rotation, angle));
                        }
                    }
                }
            }
        }
    }
    let results = run(&requests);
    let mut peak_swing = 0.0_f64;
    let mut peak_scaled_jacobian = 0.0_f64;
    for (index, (gpu, &(limits, axis, q, angle))) in results.iter().zip(&cases).enumerate() {
        let cpu = evaluate(limits, axis, q).unwrap();
        assert_eq!(gpu.meta[2], 0, "case {index}: {:?}", cases[index]);
        assert_eq!(gpu.meta[0] as usize, cpu.row_count);
        let tolerance = if angle > 3.0 { 1e-3 } else { 2e-5 };
        for (g, c) in
            gpu.swing
                .into_iter()
                .zip([cpu.swing[0], cpu.swing[1], cpu.angle, cpu.violation])
        {
            let error = (g as f64 - c).abs();
            peak_swing = peak_swing.max(error);
            assert!(
                error < tolerance,
                "case {index}: swing {g} != {c}; {:?}",
                cases[index]
            );
        }
        for row in 0..cpu.row_count {
            assert_eq!(gpu.meta[1] & (1 << row) != 0, cpu.rows[row].bilateral);
            let expected = cpu.rows[row].angular.extend(cpu.rows[row].error).to_array();
            for (g, c) in gpu.rows[row].into_iter().zip(expected) {
                let error = (g as f64 - c).abs() / (1.0 + c.abs());
                peak_scaled_jacobian = peak_scaled_jacobian.max(error);
                assert!(
                    error < tolerance,
                    "case {index} row {row}: {g} != {c}, scaled error={error}; {:?}",
                    cases[index]
                );
            }
        }
    }
    eprintln!(
        "cone geometry cases={} swing_peak={peak_swing:e} scaled_jacobian_peak={peak_scaled_jacobian:e}",
        cases.len()
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn cone_geometry_accepts_antipodal_charts_and_reports_invalid_status() {
    let bounded = SphericalConeLimits {
        angle0: Some(0.4),
        angle1: Some(0.9),
    };
    let antipode = Quat::from_rotation_y(std::f64::consts::PI);
    let valid = ConeGpuRequest::new(bounded, Vec3::X, Quat::IDENTITY).unwrap();
    let mut requests = vec![
        ConeGpuRequest::new(bounded, Vec3::X, antipode).unwrap(),
        ConeGpuRequest::new(SphericalConeLimits::default(), Vec3::X, antipode).unwrap(),
        ConeGpuRequest::new(
            SphericalConeLimits {
                angle0: Some(std::f64::consts::PI),
                angle1: None,
            },
            Vec3::X,
            antipode,
        )
        .unwrap(),
        ConeGpuRequest::new(
            SphericalConeLimits {
                angle0: Some(std::f64::consts::PI - 1e-9),
                angle1: None,
            },
            Vec3::X,
            antipode,
        )
        .unwrap(),
    ];
    requests.push(
        ConeGpuRequest::new(
            SphericalConeLimits {
                angle0: Some(0.),
                angle1: Some(0.),
            },
            Vec3::X,
            antipode,
        )
        .unwrap(),
    );
    requests.push(
        ConeGpuRequest::new(
            SphericalConeLimits {
                angle0: None,
                angle1: Some(0.9),
            },
            Vec3::X,
            Quat::from_rotation_y(std::f64::consts::PI - 0.001),
        )
        .unwrap(),
    );
    requests.push(
        ConeGpuRequest::new(
            SphericalConeLimits {
                angle0: Some(0.),
                angle1: None,
            },
            Vec3::X,
            antipode,
        )
        .unwrap(),
    );
    for kind in 0..8 {
        let mut bad = valid;
        match kind {
            0 => bad.rotation = [0.0; 4],
            1 => bad.primary = [0.0; 4],
            2 => bad.transverse0 = bad.primary,
            3 => bad.angles[0] = -1.0,
            4 => bad.angles[0] = f32::NAN,
            5 => bad.enabled[0] = 2,
            6 => bad.enabled[0] = 3,
            _ => bad.angles[0] = 1e-30,
        }
        requests.push(bad);
    }
    let result = run(&requests);
    assert_eq!(
        result.iter().map(|g| g.meta[2]).collect::<Vec<_>>(),
        [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2]
    );
    assert_eq!(result[0].meta[0], 1);
    assert!((f64::from(result[0].swing[3]) - (std::f64::consts::PI - 0.9)).abs() < 1e-6);
    assert!(
        (Vec3::new(
            f64::from(result[0].rows[0][0]),
            f64::from(result[0].rows[0][1]),
            f64::from(result[0].rows[0][2])
        ) + Vec3::Y)
            .length()
            < 1e-6
    );
    assert_eq!(result[4].meta[0], 2);
    assert_eq!(result[4].meta[1], 3);
    assert_eq!(result[5].meta[0], 1);
    assert!((result[5].swing[3] - 0.001).abs() < 1e-6);
    assert_eq!(result[6].meta[0], 1);
    assert_eq!(result[6].meta[1], 1);
    assert!(result[6].swing[3].abs() < 1e-6);
    for tiny in [1e-20, 1e-100, 1e-300] {
        assert!(
            ConeGpuRequest::new(
                SphericalConeLimits {
                    angle0: Some(tiny),
                    angle1: Some(1.0)
                },
                Vec3::X,
                Quat::IDENTITY
            )
            .is_err()
        );
    }
}
