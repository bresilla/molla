#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod gpu_readback;

#[test]
#[ignore = "requires MOLLA_CONTACT_BLOCK_REPLAY and MOLLA_CONTACT_BLOCK_OUTPUT through make test-gpu"]
fn captured_full_contact_block_continuation() {
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_CONTACT_BLOCK_REPLAY").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("MOLLA_CONTACT_BLOCK_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let names = [
        "positions",
        "velocities",
        "inverse_mass",
        "contacts",
        "body_poses",
        "body_com",
        "jacobians",
        "mass_matrix",
        "rigid_velocities",
        "body_velocities",
    ];
    let inputs: Vec<_> = names
        .iter()
        .map(|name| std::fs::read(root.join(format!("{name}.bin"))).unwrap())
        .collect();
    let buffers: Vec<_> = inputs
        .iter()
        .map(|bytes| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        })
        .collect();
    let particles = (buffers[0].size() / 16) as u32;
    let rows = (buffers[3].size() / 80) as u32;
    let bodies = (buffers[4].size() / 32) as u32;
    let dofs = (buffers[8].size() / 4) as u32;
    let settings_bytes = std::fs::read(root.join("settings.bin")).unwrap();
    let settings_values: &[f32] = bytemuck::cast_slice(&settings_bytes);
    let settings = SoftRigidSolveSettings {
        iterations: 1,
        dt: settings_values[0],
        penetration_recovery: settings_values[1],
        max_recovery_speed: settings_values[2],
    };
    let bindings = || SoftRigidSolveBindings {
        positions: &buffers[0],
        velocities: &buffers[1],
        inverse_mass: &buffers[2],
        contacts: &buffers[3],
        body_poses: &buffers[4],
        body_com: &buffers[5],
        jacobians: &buffers[6],
        mass_matrix: &buffers[7],
        rigid_velocities: &buffers[8],
        body_velocities: &buffers[9],
        particle_count: particles,
        body_count: bodies,
    };
    let final_velocity = std::fs::read(root.join("final-velocities.bin")).unwrap();
    let final_rigid = std::fs::read(root.join("final-rigid_velocities.bin")).unwrap();
    let final_rows = std::fs::read(root.join("final-rows.bin")).unwrap();
    let seed = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &final_rows,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage,read> source: array<u32>;
             @group(0) @binding(1) var<storage,read_write> destination: array<u32>;
             @compute @workgroup_size(64) fn seed(@builtin(global_invocation_id) id: vec3<u32>) {
                 if id.x < arrayLength(&source) { destination[id.x] = source[id.x]; }
             }"
            .into(),
        ),
    });
    let copy = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("seed"),
        compilation_options: Default::default(),
        cache: None,
    });
    for (blocked, iterations) in [
        (false, 128),
        (true, 0),
        (true, 1),
        (true, 2),
        (true, 4),
        (true, 8),
        (true, 16),
        (true, 32),
        (true, 64),
        (true, 128),
    ] {
        let folder = if blocked {
            output.clone()
        } else {
            output.join("ordered")
        };
        if !blocked {
            std::fs::create_dir(&folder).unwrap();
        }
        for (buffer, bytes) in buffers.iter().zip(&inputs) {
            queue.write_buffer(buffer, 0, bytes);
        }
        queue.write_buffer(&buffers[1], 0, &final_velocity);
        queue.write_buffer(&buffers[8], 0, &final_rigid);
        let kernel = if blocked {
            SoftRigidSolveKernel::new_blocked_contacts(device, dofs, rows, particles)
        } else {
            SoftRigidSolveKernel::new_separated_static(device, dofs, rows, particles)
        }
        .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        kernel
            .encode(device, &mut encoder, bindings(), settings)
            .unwrap();
        queue.submit([encoder.finish()]);
        queue.write_buffer(&buffers[1], 0, &final_velocity);
        queue.write_buffer(&buffers[8], 0, &final_rigid);
        let mut encoder = device.create_command_encoder(&Default::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &copy.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: seed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernel.row_state().as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&copy);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((rows * 8).div_ceil(64), 1, 1);
        }
        if iterations > 0 {
            kernel
                .encode_continuation(
                    device,
                    &mut encoder,
                    bindings(),
                    SoftRigidSolveSettings {
                        iterations,
                        ..settings
                    },
                )
                .unwrap();
        }
        kernel
            .encode_velocity_recovery(device, &mut encoder, bindings(), settings)
            .unwrap();
        kernel
            .encode_residual_check(device, &mut encoder, bindings(), settings, 1e-4)
            .unwrap();
        queue.submit([encoder.finish()]);
        let status = gpu_readback::read::<u32>(&engine, kernel.status(), 1)[0];
        if blocked && iterations == 0 {
            let words = gpu_readback::read::<u32>(&engine, kernel.active_rows(),
                kernel.active_rows().size() as usize / 4);
            let base = (rows + 1 + rows.div_ceil(64) + 132 + particles + 3 * rows) as usize;
            std::fs::write(folder.join("block-map.bin"),
                bytemuck::cast_slice(&words[base..base + 2 * rows as usize])).unwrap();
        }
        assert!(status == 0 || status == 64, "unexpected GPU fault {status}");
        let row_state = gpu_readback::read::<f32>(&engine, kernel.row_state(), rows as usize * 8);
        let maximum = row_state
            .chunks_exact(8)
            .map(|row| row[3])
            .fold(0.0_f32, f32::max);
        eprintln!(
            "full contact block={blocked} iterations={iterations} status={status} maximum={maximum:e}"
        );
        for (name, values) in [
            ("rows", row_state),
            (
                "velocity",
                gpu_readback::read::<f32>(&engine, &buffers[1], particles as usize * 4),
            ),
            (
                "rigid",
                gpu_readback::read::<f32>(&engine, &buffers[8], dofs as usize),
            ),
        ] {
            assert!(values.iter().all(|v| v.is_finite()));
            std::fs::write(
                folder.join(format!("coupled-{name}-{iterations}.bin")),
                bytemuck::cast_slice(&values),
            )
            .unwrap();
        }
        if blocked && iterations == 128 {
            assert_eq!(status, 0, "block contact continuation did not converge");
        }
    }
}
